#!/usr/bin/env python3
"""
Experiment 15 — Batching / compute-contention timing channel (salting-IMMUNE).

Motivation
----------
Exp 01 proved the prefix-cache timing channel. Exp 04/06 showed that per-session
(distinct) salting collapses that channel toward chance, because the attacker and
victim no longer share a cache key. Everything up through Exp 14 lives or dies by
cache SHARING.

This experiment attacks a resource that salting does NOT protect: the shared GPU
and the continuous-batching scheduler. Every request on this server, salted or not,
runs through the same scheduler on the same device. When a victim submits a large
prefill, the attacker's concurrently-submitted probe is batched with it / queues
behind it, and the attacker's own TTFT rises. The attacker measures ONLY their own
latency and infers victim activity.

Crucially we run with a UNIQUE cache_salt on every single request (attacker and
victim), so the prefix cache is never shared by anyone. If a channel still appears,
it cannot be the cache channel — it is pure compute/scheduler contention, and
per-session salting (the strongest salting defense) does nothing to it.

What it can leak
----------------
Not content. It leaks victim *activity*: presence (is someone else using the box
right now) and workload *size* (how long is their prompt). That is a coarse but
genuinely salting-immune co-tenancy oracle — the kind of signal that undermines the
claim that "salt the cache and the timing problem is solved."

Design
------
- measure_ttft(): stream a chat completion, return seconds-to-first-token. Each call
  gets a fresh UUID salt, so it never hits cache.
- victim_worker(): background threads hammering the server with LARGE prefills
  (also uniquely salted), to create prefill contention.
- Condition IDLE : no victim threads. Collect N probe TTFTs.
- Condition BUSY : K victim threads. Collect N probe TTFTs.
- Stats: median delta, AUC = P(busy_ttft > idle_ttft) (0.5 = no channel), bootstrap
  95% CI on AUC, and KS two-sample test (scipy if available).
- Optional --sweep: vary victim prompt length and show probe TTFT tracks victim size.

Ground truth: the condition label (idle/busy) and the victim prompt length ARE the
ground truth here — no cache counters needed, because we are not measuring the cache.

Usage
-----
    python experiments/15_batching_contention.py                    # idle vs busy
    python experiments/15_batching_contention.py --sweep            # + size sweep
    python experiments/15_batching_contention.py --victims 4 --n 200

Requires only: requests (stdlib otherwise). scipy optional (nicer KS).
"""

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path

import requests

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BASE_URL = os.environ.get("CACHELAB_BASE_URL", "http://localhost:8000")
MODEL = os.environ.get("CACHELAB_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
CHAT_URL = BASE_URL.rstrip("/") + "/v1/chat/completions"

# A short probe: the attacker's own request is tiny, so almost all of its measured
# latency variation comes from contention, not from its own compute.
PROBE_PROMPT = "Reply with the single word: ok."

# A large victim prefill: long prompt, short generation. Prefill is the contended
# phase, so we want a big prompt and max_tokens=1.
_LOREM = (
    "In a distributed serving system many tenants share one accelerator and the "
    "scheduler interleaves their requests across a continuous batch. "
)


def victim_prompt(words: int) -> str:
    base = (_LOREM * ((words // len(_LOREM.split())) + 2)).split()
    random.shuffle(base)  # defeat any accidental cross-request prefix match
    return " ".join(base[:words])


# shuffled lorem tokenizes at ~1.5 tokens/word on the Qwen2.5 tokenizer; plus the
# chat template wrapper. If a victim prompt exceeds the server's max_model_len it
# is rejected (400), the victim generates NO load, and the probe falls back to
# baseline -- a dip that must never be read as a channel limit. Guard against it.
def est_tokens(words: int) -> int:
    return int(words * 1.5) + 32  # +chat template overhead


def within_ctx(words: int, ctx: int) -> bool:
    return est_tokens(words) <= ctx - 64  # leave headroom for the template


# --------------------------------------------------------------------------- #
# Timing primitive: TTFT via streaming, unique salt every call
# --------------------------------------------------------------------------- #
def measure_ttft(prompt: str, max_tokens: int = 1, timeout: float = 60.0) -> float:
    """Stream a chat completion; return seconds from send to first SSE token chunk.
    Uses a unique cache_salt so this request can never share a cache entry."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "cache_salt": uuid.uuid4().hex,  # <-- strongest salting: unique per request
    }
    t0 = time.perf_counter()
    with requests.post(CHAT_URL, json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                # first content-bearing chunk marks first token
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {})
                if delta.get("content") or delta.get("role"):
                    return time.perf_counter() - t0
    return time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Victim background load
# --------------------------------------------------------------------------- #
class Victim:
    def __init__(self, words: int, n_threads: int):
        self.words = words
        self.n_threads = n_threads
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.sent = 0
        self.errors = 0
        self.last_error = ""
        self._lock = threading.Lock()

    def _worker(self):
        while not self._stop.is_set():
            try:
                measure_ttft(victim_prompt(self.words), max_tokens=1, timeout=120)
                with self._lock:
                    self.sent += 1
            except Exception as e:
                with self._lock:
                    self.errors += 1
                    self.last_error = str(e)[:200]
                time.sleep(0.05)

    def load_ok(self) -> bool:
        """True if the victim actually sustained load (vs e.g. prompt > ctx -> all 4xx)."""
        return self.sent > 0 and self.sent >= self.errors

    def __enter__(self):
        for _ in range(self.n_threads):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._threads.append(t)
        # let the pipeline fill before the attacker starts probing
        time.sleep(1.0)
        return self

    def __exit__(self, *exc):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)


# --------------------------------------------------------------------------- #
# Stats (stdlib; scipy optional for KS)
# --------------------------------------------------------------------------- #
def auc_greater(busy: list[float], idle: list[float]) -> float:
    """P(busy > idle) over all pairs. 0.5 = no separation."""
    wins = ties = 0
    for b in busy:
        for i in idle:
            if b > i:
                wins += 1
            elif b == i:
                ties += 1
    total = len(busy) * len(idle)
    return (wins + 0.5 * ties) / total if total else 0.5


def bootstrap_auc_ci(busy, idle, iters=2000, seed=0):
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        b = [rng.choice(busy) for _ in busy]
        i = [rng.choice(idle) for _ in idle]
        vals.append(auc_greater(b, i))
    vals.sort()
    lo = vals[int(0.025 * iters)]
    hi = vals[int(0.975 * iters)]
    return lo, hi


def ks_test(a, b):
    try:
        from scipy import stats  # type: ignore
        s, p = stats.ks_2samp(a, b)
        return float(s), float(p)
    except Exception:
        # stdlib fallback: KS statistic only, no p-value
        allv = sorted(set(a) | set(b))
        na, nb = len(a), len(b)
        d = 0.0
        for x in allv:
            fa = sum(1 for v in a if v <= x) / na
            fb = sum(1 for v in b if v <= x) / nb
            d = max(d, abs(fa - fb))
        return d, None


def summarize(name, xs):
    xs = sorted(xs)
    return {
        "condition": name,
        "n": len(xs),
        "median_ms": statistics.median(xs) * 1e3,
        "mean_ms": statistics.fmean(xs) * 1e3,
        "p90_ms": xs[int(0.9 * (len(xs) - 1))] * 1e3,
        "stdev_ms": (statistics.pstdev(xs) * 1e3) if len(xs) > 1 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
def collect_probes(n, warmup, label):
    # warmup (discarded) to let the server settle
    for _ in range(warmup):
        try:
            measure_ttft(PROBE_PROMPT)
        except Exception:
            pass
    out = []
    for k in range(n):
        try:
            out.append(measure_ttft(PROBE_PROMPT))
        except Exception as e:
            print(f"  [{label}] probe {k} failed: {e}", file=sys.stderr)
        if (k + 1) % 25 == 0:
            print(f"  [{label}] {k + 1}/{n}", file=sys.stderr)
    return out


def run_idle_vs_busy(n, warmup, victim_words, victims):
    print(f"== IDLE (no victim) : {n} probes ==", file=sys.stderr)
    idle = collect_probes(n, warmup, "idle")

    print(f"== BUSY ({victims} victim threads @ {victim_words} words) : {n} probes ==",
          file=sys.stderr)
    with Victim(victim_words, victims) as v:
        busy = collect_probes(n, warmup, "busy")
        sent, errors, ok, err = v.sent, v.errors, v.load_ok(), v.last_error
    print(f"   victim sent ~{sent} large prefills during BUSY ({errors} errors)", file=sys.stderr)
    if not ok:
        print(f"   WARNING: victim load degenerate ({errors} errs: {err}); "
              f"prompt may exceed max_model_len -> result is NOT a channel measurement.",
              file=sys.stderr)

    s_idle = summarize("idle", idle)
    s_busy = summarize("busy", busy)
    auc = auc_greater(busy, idle)
    lo, hi = bootstrap_auc_ci(busy, idle)
    ks, p = ks_test(idle, busy)
    delta = s_busy["median_ms"] - s_idle["median_ms"]

    print("\n--- RESULT: batching / compute-contention channel ---")
    print(f"  idle  median TTFT : {s_idle['median_ms']:.2f} ms  (p90 {s_idle['p90_ms']:.1f})")
    print(f"  busy  median TTFT : {s_busy['median_ms']:.2f} ms  (p90 {s_busy['p90_ms']:.1f})")
    print(f"  delta (busy-idle) : {delta:+.2f} ms")
    print(f"  AUC P(busy>idle)  : {auc:.3f}  (95% CI {lo:.3f}-{hi:.3f})  [0.5 = no channel]")
    print(f"  KS statistic      : {ks:.3f}" + (f"  p={p:.2e}" if p is not None else "  (p: install scipy)"))
    verdict = ("CHANNEL PRESENT (salting-immune)" if lo > 0.5 or (p is not None and p < 1e-3 and auc > 0.5)
               else "no separable channel at this load")
    print(f"  verdict           : {verdict}")

    return {
        "mode": "idle_vs_busy",
        "model": MODEL,
        "victim_words": victim_words,
        "victims": victims,
        "victim_sent": sent,
        "victim_errors": errors,
        "load_ok": ok,
        "idle": s_idle,
        "busy": s_busy,
        "delta_ms": delta,
        "auc_busy_gt_idle": auc,
        "auc_ci95": [lo, hi],
        "ks": ks,
        "ks_p": p,
        "verdict": verdict,
        "idle_samples_ms": [x * 1e3 for x in idle],
        "busy_samples_ms": [x * 1e3 for x in busy],
    }


def run_sweep(n, warmup, victims, sizes, ctx):
    print(f"== SWEEP victim prompt size, {victims} threads (ctx={ctx}) ==", file=sys.stderr)
    rows = []
    # idle baseline
    idle = collect_probes(n, warmup, "idle")
    base = statistics.median(idle) * 1e3
    rows.append({"victim_words": 0, "median_ms": base, "delta_ms": 0.0, "auc": 0.5,
                 "victim_sent": 0, "victim_errors": 0, "load_ok": True})
    print(f"  idle baseline median {base:.2f} ms")
    for w in sizes:
        if not within_ctx(w, ctx):
            print(f"  victim {w:5d} words ~{est_tokens(w)} tok > ctx {ctx}: SKIP "
                  f"(would be rejected -> no load, false dip). Raise --ctx / max_model_len to test.",
                  file=sys.stderr)
            rows.append({"victim_words": w, "median_ms": None, "delta_ms": None, "auc": None,
                         "victim_sent": 0, "victim_errors": 0, "load_ok": False,
                         "note": f"skipped: ~{est_tokens(w)} tok exceeds ctx {ctx}"})
            continue
        with Victim(w, victims) as v:
            busy = collect_probes(n, warmup, f"busy{w}")
            sent, errors, ok, err = v.sent, v.errors, v.load_ok(), v.last_error
        med = statistics.median(busy) * 1e3
        auc = auc_greater(busy, idle)
        rows.append({"victim_words": w, "median_ms": med, "delta_ms": med - base, "auc": auc,
                     "victim_sent": sent, "victim_errors": errors, "load_ok": ok})
        flag = "" if ok else f"  <-- DEGENERATE: victim load failed ({errors} errs: {err})"
        print(f"  victim {w:5d} words -> probe median {med:7.2f} ms  (+{med - base:6.2f})  "
              f"AUC {auc:.3f}  [sent {sent}, err {errors}]{flag}")
    return {"mode": "sweep", "model": MODEL, "victims": victims, "ctx": ctx,
            "baseline_ms": base, "rows": rows}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="probes per condition")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--victim-words", type=int, default=1500)
    ap.add_argument("--victims", type=int, default=3, help="concurrent victim threads")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[200, 500, 1000, 1500, 2000, 2500, 4000])
    ap.add_argument("--ctx", type=int, default=4096,
                    help="server max_model_len; victim sizes over this are skipped (they'd 400)")
    ap.add_argument("--out", default="runs/batching_contention.json")
    args = ap.parse_args()

    # fail fast if server is down
    try:
        requests.get(BASE_URL.rstrip("/") + "/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Server not reachable at {BASE_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    if args.sweep:
        result = run_sweep(args.n, args.warmup, args.victims, args.sizes, args.ctx)
    else:
        result = run_idle_vs_busy(args.n, args.warmup, args.victim_words, args.victims)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nsaved {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

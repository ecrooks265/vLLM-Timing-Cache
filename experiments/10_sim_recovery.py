#!/usr/bin/env python
"""Experiment 10 - calibrated simulator: token-granular EarlyBird recovery.

Reproduces the prefix-recovery attack the ROCm/WSL2 hardware cannot run
(block_size=1), with the timing oracle seeded from the REAL hit/miss TTFTs
measured on the live server. Also shows the block_size=16 default-mitigation:
the per-character signal vanishes.
"""
from __future__ import annotations
import argparse, glob, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cachelab.simulator import load_ttft_distributions, TimingOracle, simulate_recovery
from cachelab.io_utils import new_run_dir, save_json

VICTIM_SECRET = ("The wire transfer authorization code is TX-7788-ALPHA and the "
                 "account PIN is 4921, release funds to vendor 00413 now.")


def find_calib(explicit):
    if explicit:
        return explicit
    cands = sorted(glob.glob("runs/*/requests.jsonl"), key=os.path.getmtime, reverse=True)
    for c in cands:
        h, m = load_ttft_distributions(c)
        if h and m:
            return c
    return None


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--calib", default=None, help="path to a labeled requests.jsonl")
    a.add_argument("--block-sizes", default="1,16")
    a.add_argument("--prefix-fraction", type=float, default=0.5)
    a.add_argument("--max-recover", type=int, default=48)
    a.add_argument("--repeats", type=int, default=5)
    args = a.parse_args()

    calib = find_calib(args.calib)
    if not calib:
        print("[!] no labeled requests.jsonl under runs/ - run the audit (Exp 01) first", file=sys.stderr)
        return 1
    hits, misses = load_ttft_distributions(calib)
    oracle = TimingOracle(hits, misses)
    print(f"[*] calibrated from {calib}")
    print(f"    oracle: {oracle.summary()}")
    print(f"    victim secret ({len(VICTIM_SECRET)} chars): {VICTIM_SECRET!r}")

    block_sizes = [int(x) for x in args.block_sizes.split(",") if x.strip()]
    results = []
    for bs in block_sizes:
        r = simulate_recovery(oracle, VICTIM_SECRET, prefix_fraction=args.prefix_fraction,
                              max_recover=args.max_recover, block_size=bs, repeats=args.repeats)
        results.append(r.to_dict())
        print(f"\n=== block_size = {bs} ===")
        print(f"    known prefix : {r.known_prefix!r}")
        print(f"    target       : {r.target!r}")
        print(f"    recovered    : {r.recovered!r}")
        print(f"    --> {r.n_correct}/{len(r.target)} chars correct  (accuracy {r.accuracy*100:.1f}%)")

    run_dir = new_run_dir("runs")
    save_json({"experiment": "10_sim_recovery", "calibrated_from": calib,
               "oracle": oracle.summary(), "victim_secret": VICTIM_SECRET,
               "prefix_fraction": args.prefix_fraction, "repeats": args.repeats,
               "results": results}, run_dir / "sim_recovery.json")
    print(f"\n[*] wrote {run_dir/'sim_recovery.json'}")
    print("\n    Reading: block_size=1 recovers the secret token-by-token (the attack")
    print("    Gu et al. could not execute); block_size=16 collapses to ~0 - the")
    print("    per-character signal is sealed, the default mitigation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

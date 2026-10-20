#!/usr/bin/env python
"""Experiment 5 - salt-propagation / suffix-collision probe."""
from __future__ import annotations
import argparse, statistics, sys, time, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cachelab.config import Config
from cachelab.client import TimingClient
from cachelab.groundtruth import labeled_request
from cachelab.server import wait_until_healthy
from cachelab.io_utils import new_run_dir, save_json

LONG_BASE = (" ".join(f"token{i}" for i in range(200)))

def probe(cfg, client, seq, user, salt):
    with labeled_request(cfg) as box:
        client.measure(cfg.model.name, seq, max_tokens=1, user=user, cache_salt=salt)
    lbl = box[0]
    return (lbl.hits_delta if lbl else 0.0), (bool(lbl.hit) if lbl and lbl.hit is not None else False)

def main() -> int:
    a = argparse.ArgumentParser(); a.add_argument("--config", default="config/_smoke.yaml")
    a.add_argument("--trials", type=int, default=40)
    args = a.parse_args()
    cfg = Config.load(args.config)
    print("[*] waiting for vLLM server health...")
    if not wait_until_healthy(cfg):
        print("[!] server not healthy", file=sys.stderr); return 1
    client = TimingClient(cfg.server.base_url); delay = cfg.timing.inter_request_delay
    diff_hits, same_hits, nosalt_hits = [], [], []
    diff_r, same_r, nosalt_r = [], [], []
    try:
        for _ in range(args.trials):
            seq = f"{uuid.uuid4().hex} {LONG_BASE}"
            client.measure(cfg.model.name, seq, max_tokens=1, user="victim_user", cache_salt="salt-V"); time.sleep(delay)
            h, hit = probe(cfg, client, seq, "attacker_user", "salt-A"); diff_hits.append(h); diff_r.append(1 if hit else 0); time.sleep(delay)
            seq2 = f"{uuid.uuid4().hex} {LONG_BASE}"
            client.measure(cfg.model.name, seq2, max_tokens=1, user="victim_user", cache_salt="salt-V"); time.sleep(delay)
            h2, hit2 = probe(cfg, client, seq2, "attacker_user", "salt-V"); same_hits.append(h2); same_r.append(1 if hit2 else 0); time.sleep(delay)
            seq3 = f"{uuid.uuid4().hex} {LONG_BASE}"
            client.measure(cfg.model.name, seq3, max_tokens=1, user="victim_user", cache_salt=None); time.sleep(delay)
            h3, hit3 = probe(cfg, client, seq3, "attacker_user", None); nosalt_hits.append(h3); nosalt_r.append(1 if hit3 else 0); time.sleep(delay)
    finally:
        client.close()
    def summ(name, hits, rate):
        print(f"    {name:<10} mean hits_delta={statistics.mean(hits):8.1f}  hit-rate={statistics.mean(rate):.2f}  (n={len(hits)})")
    print("\n=== salt-propagation probe (same tokens, differing salt) ===")
    summ("no-salt", nosalt_hits, nosalt_r); summ("same-salt", same_hits, same_r); summ("diff-salt", diff_hits, diff_r)
    leaked = statistics.mean(diff_hits) > 0.5
    print(f"\n    --> {'SUFFIX LEAK: salt did NOT propagate past block 0' if leaked else 'SECURE: salt propagates; no cross-salt block reuse'}")
    run_dir = new_run_dir(cfg.run.out_dir)
    save_json({"experiment": "05_salt_propagation", "model": cfg.model.name, "trials": args.trials,
               "modes": {
                 "no_salt":   {"mean_hits_delta": statistics.mean(nosalt_hits), "hit_rate": statistics.mean(nosalt_r)},
                 "same_salt": {"mean_hits_delta": statistics.mean(same_hits), "hit_rate": statistics.mean(same_r)},
                 "diff_salt": {"mean_hits_delta": statistics.mean(diff_hits), "hit_rate": statistics.mean(diff_r)}},
               "cross_salt_leak": leaked}, run_dir / "salt_propagation.json")
    print(f"[*] wrote {run_dir/'salt_propagation.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

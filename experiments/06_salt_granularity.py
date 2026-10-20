#!/usr/bin/env python
"""Experiment 6 - defense effectiveness vs salt granularity."""
from __future__ import annotations
import argparse, sys, time, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cachelab.config import Config
from cachelab.client import TimingClient
from cachelab.groundtruth import labeled_request
from cachelab.server import wait_until_healthy
from cachelab.io_utils import new_run_dir, save_json
from cachelab.stats import ks_hit_vs_miss, average_precision

def collect(cfg, client, base, vsalt, asalt):
    n, warmup, delay = cfg.timing.num_samples, cfg.timing.num_warmup, cfg.timing.inter_request_delay
    hit, miss, gt_on_hit = [], [], []
    for i in range(n + warmup):
        warm = i < warmup
        hp = f"{uuid.uuid4().hex} {base}"
        client.measure(cfg.model.name, hp, max_tokens=1, user="victim_user", cache_salt=vsalt); time.sleep(delay)
        with labeled_request(cfg) as box:
            m = client.measure(cfg.model.name, hp, max_tokens=1, user="attacker_user", cache_salt=asalt)
        gt = box[0].hit if box[0] else None; time.sleep(delay)
        if not warm and m.ok and m.client_ttft is not None:
            hit.append(m.client_ttft)
            if gt is not None: gt_on_hit.append(1 if gt else 0)
        m2 = client.measure(cfg.model.name, f"{uuid.uuid4().hex} {base}", max_tokens=1, user="attacker_user", cache_salt=asalt); time.sleep(delay)
        if not warm and m2.ok and m2.client_ttft is not None: miss.append(m2.client_ttft)
    return hit, miss, gt_on_hit

def main() -> int:
    a = argparse.ArgumentParser(); a.add_argument("--config", default="config/_smoke.yaml")
    cfg = Config.load(a.parse_args().config)
    print("[*] waiting for vLLM server health...")
    if not wait_until_healthy(cfg):
        print("[!] server not healthy", file=sys.stderr); return 1
    client = TimingClient(cfg.server.base_url)
    base = [l.strip() for l in Path(cfg.attack.victim_prompt_file).read_text().splitlines()
            if l.strip() and not l.startswith("#")][0]
    _w=base.split(); _i=0
    while len(_w)<cfg.attack.prompt_min_words:
        base+=f" ctx{_i}"; _w=base.split(); _i+=1
    print(f"[*] base prompt: {base[:50]}...")
    conditions = [("no-salt (global cache)", None, None),
                  ("shared salt (per-org / coarse)", "org-salt", "org-salt"),
                  ("distinct salts (per-session)", "victim-salt-A", "attacker-salt-B")]
    rows = []; jrows = []
    run_dir = new_run_dir(cfg.run.out_dir)
    try:
        for label, vs, as_ in conditions:
            hit, miss, gt = collect(cfg, client, base, vs, as_)
            ttfts = list(hit) + list(miss); truth = [True]*len(hit) + [False]*len(miss)
            ap = average_precision(ttfts, truth) if ttfts else 0.0
            ks = ks_hit_vs_miss(hit, miss) if (hit and miss) else None
            gt_rate = (sum(gt)/len(gt)) if gt else float("nan")
            won = bool(ks and ks.significant and ap >= cfg.stats.precision_target)
            rows.append((label, gt_rate, ap, (ks.pvalue if ks else float("nan")), won))
            jrows.append({"condition": label, "gt_hit_rate": gt_rate, "average_precision": ap,
                          "ks_pvalue": (ks.pvalue if ks else None), "ks_statistic": (ks.statistic if ks else None),
                          "n_hit": len(hit), "n_miss": len(miss), "attack_succeeds": won})
        print("\n=== defense effectiveness vs salt granularity ===")
        print(f"    {'condition':<34}{'gt_hit':>8}{'AP':>8}{'KS p':>12}   verdict")
        for label, gt_rate, ap, ksp, won in rows:
            print(f"    {label:<34}{gt_rate:>8.2f}{ap:>8.3f}{ksp:>12.2e}   {'ATTACK SUCCEEDS' if won else 'DEFEATED'}")
        print("\n    Reading: salting defeats the attack ONLY at per-session granularity.")
        print("    A shared/coarse salt (kept for cache efficiency) leaves the channel open.")
        save_json({"experiment": "06_salt_granularity", "model": cfg.model.name,
                   "prompt_min_words": cfg.attack.prompt_min_words, "conditions": jrows},
                  run_dir / "salt_granularity.json")
        print(f"\n[*] wrote {run_dir/'salt_granularity.json'}")
    finally:
        client.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

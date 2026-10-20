#!/usr/bin/env python
"""Experiment 4 - cache-salting defense effectiveness (vLLM cache_salt)."""
from __future__ import annotations
import argparse, sys, time, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cachelab.config import Config
from cachelab.client import TimingClient
from cachelab.groundtruth import labeled_request
from cachelab.server import wait_until_healthy
from cachelab.stats import ks_hit_vs_miss, average_precision
from cachelab.io_utils import new_run_dir, save_json

VICTIM_SALT = "victim-tenant-salt-A"
ATTACKER_SALT = "attacker-tenant-salt-B"

def collect(cfg, client, base, vsalt, asalt):
    n, warmup, delay = cfg.timing.num_samples, cfg.timing.num_warmup, cfg.timing.inter_request_delay
    hit, miss, gt_on_hit = [], [], []
    for i in range(n + warmup):
        warm = i < warmup
        hit_prompt = f"{uuid.uuid4().hex} {base}"
        client.measure(cfg.model.name, hit_prompt, max_tokens=1, user="victim_user", cache_salt=vsalt)
        time.sleep(delay)
        with labeled_request(cfg) as box:
            m = client.measure(cfg.model.name, hit_prompt, max_tokens=1, user="attacker_user", cache_salt=asalt)
        gt = box[0].hit if box[0] else None
        time.sleep(delay)
        if not warm and m.ok and m.client_ttft is not None:
            hit.append(m.client_ttft)
            if gt is not None: gt_on_hit.append(1 if gt else 0)
        miss_prompt = f"{uuid.uuid4().hex} {base}"
        m2 = client.measure(cfg.model.name, miss_prompt, max_tokens=1, user="attacker_user", cache_salt=asalt)
        time.sleep(delay)
        if not warm and m2.ok and m2.client_ttft is not None:
            miss.append(m2.client_ttft)
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
    run_dir = new_run_dir(cfg.run.out_dir)
    jrows = []
    try:
        for label, vs, as_ in [("NO DEFENSE (no salt)", None, None),
                               ("cache_salt DEFENSE (victim salt != attacker salt)", VICTIM_SALT, ATTACKER_SALT)]:
            hit, miss, gt = collect(cfg, client, base, vs, as_)
            ttfts = list(hit) + list(miss); truth = [True]*len(hit) + [False]*len(miss)
            ap = average_precision(ttfts, truth) if ttfts else 0.0
            ks = ks_hit_vs_miss(hit, miss) if (hit and miss) else None
            gt_rate = (sum(gt)/len(gt)) if gt else float("nan")
            ksp = f"{ks.pvalue:.2e}" if ks else "nan"; kss = f"{ks.statistic:.3f}" if ks else "nan"
            won = bool(ks and ks.significant and ap >= cfg.stats.precision_target)
            print(f"\n=== {label} ===")
            print(f"    HIT-attempt real cache-hit rate (ground truth): {gt_rate:.2f}")
            print(f"    AP(intended)={ap:.3f}  KS stat={kss}  KS p={ksp}  n_hit={len(hit)} n_miss={len(miss)}")
            print(f"    --> {'ATTACK SUCCEEDS' if won else 'ATTACK DEFEATED'}")
            jrows.append({"condition": label, "gt_hit_rate": gt_rate, "average_precision": ap,
                          "ks_pvalue": (ks.pvalue if ks else None), "ks_statistic": (ks.statistic if ks else None),
                          "n_hit": len(hit), "n_miss": len(miss), "attack_succeeds": won})
        save_json({"experiment": "04_defense_salting", "model": cfg.model.name,
                   "victim_salt": VICTIM_SALT, "attacker_salt": ATTACKER_SALT,
                   "prompt_min_words": cfg.attack.prompt_min_words, "conditions": jrows},
                  run_dir / "salt_defense.json")
        print(f"\n[*] wrote {run_dir/'salt_defense.json'}")
    finally:
        client.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

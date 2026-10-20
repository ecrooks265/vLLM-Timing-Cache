#!/usr/bin/env python
"""Experiment 1 - prove the side channel at per-user and global cache levels.

Phase 1 of the proposal. Thin wrapper over run_experiment.py --audit so results
land in a clearly named place. Defaults to the replication config; pass another.

    python experiments/01_prove_sidechannel.py                       # replication.yaml
    python experiments/01_prove_sidechannel.py config/rocm_7900xtx.yaml
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cfg = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "config" / "replication.yaml")

if __name__ == "__main__":
    sys.exit(subprocess.call(
        [sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
         "--config", cfg, "--audit"]))

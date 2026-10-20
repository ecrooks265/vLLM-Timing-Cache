#!/usr/bin/env python3
"""Probe whether the ROCm attention kernel on this GPU accepts --block-size 1.

Why this exists
---------------
Token-by-token EarlyBird recovery (experiments/02_recover_prefix.py) wants a
prefix-cache block size of 1 so each cached token is an independently observable
hit. The CUDA path in old vLLM supported that; the ROCm paged-attention path on
RDNA3 (gfx1100), and the modern V1 engine generally, may reject block_size=1 or
silently clamp it. Rather than guess, this starts a short-lived server at
block_size=1 and reports what actually happened.

What it does
------------
1. Writes a temp config with server.block_size=1 (copied from your base config).
2. Brings the server up (Docker by default; --native for a bare-metal vLLM).
3. Polls /health up to --timeout seconds.
4. If healthy: PASS — block_size=1 works, EarlyBird is token-granular here.
   If it dies / times out: scans the server log for a block-size/kernel error
   and prints it, then FAIL — fall back to block_size=16 (block-granular
   recovery). See README "EarlyBird on ROCm".
5. Tears the server back down and removes the temp config.

Run this with the normal server STOPPED (it uses the same port 8000).

Usage:
    python scripts/probe_block_size.py --config config/rocm_7900xtx.yaml
    python scripts/probe_block_size.py --native --config config/default.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

REPO = Path(__file__).resolve().parents[1]
COMPOSE = REPO / "docker" / "docker-compose.rocm.yml"
HEALTH = "http://127.0.0.1:8000/health"
PROBE_CFG_REL = "config/_probe_bs1.yaml"   # inside the repo (mounted at /workspace)

# log lines that indicate the kernel rejected block_size=1
_ERR_HINTS = ("block_size", "block size", "unsupported", "assert", "BLOCK_SIZE",
              "PagedAttention", "paged_attention", "not supported")


def _write_probe_config(base_cfg_path: Path) -> Path:
    cfg = yaml.safe_load(base_cfg_path.read_text())
    cfg["server"]["block_size"] = 1
    out = REPO / PROBE_CFG_REL
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def _server_already_up() -> bool:
    try:
        return requests.get(HEALTH, timeout=2).status_code == 200
    except requests.RequestException:
        return False


def _poll_health(timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(HEALTH, timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(3)
    return False


def _scan_log_for_errors(log_text: str) -> list[str]:
    hits = []
    for line in log_text.splitlines():
        low = line.lower()
        if any(h.lower() in low for h in _ERR_HINTS):
            hits.append(line.strip())
    return hits[-15:]  # last few are the most relevant


def run_docker(timeout: float) -> int:
    container_cfg = f"/workspace/{PROBE_CFG_REL}"
    up = ["docker", "compose", "-f", str(COMPOSE)]
    env_prefix = {"CONFIG": container_cfg}
    print(f"[probe] starting server at block_size=1 (config: {container_cfg})")
    # start detached so we can poll, then collect logs
    subprocess.run(up + ["up", "-d"], check=True,
                   env={**_os_environ(), **env_prefix})
    try:
        ok = _poll_health(timeout)
        logs = subprocess.run(up + ["logs", "--no-color"], capture_output=True,
                              text=True).stdout
        return _verdict(ok, logs)
    finally:
        print("[probe] tearing down...")
        subprocess.run(up + ["down"], check=False)


def run_native(cfg_path: Path, timeout: float) -> int:
    import shlex
    cfg = yaml.safe_load(cfg_path.read_text())
    m, s = cfg["model"], cfg["server"]
    args = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", m["name"], "--dtype", m.get("dtype", "bfloat16"),
        "--max-model-len", str(m.get("max_model_len", 4096)),
        "--host", "127.0.0.1", "--port", str(s.get("port", 8000)),
        "--block-size", "1",
        "--gpu-memory-utilization", str(s.get("gpu_memory_utilization", 0.90)),
    ]
    if s.get("enable_prefix_caching", True):
        args.append("--enable-prefix-caching")
    if s.get("enforce_eager", True):
        args.append("--enforce-eager")
    print("[probe] launching native:", " ".join(shlex.quote(a) for a in args))
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    captured: list[str] = []
    try:
        ok = _poll_health(timeout)
        return _verdict(ok, "")
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=10)
            captured.append(out or "")
        except subprocess.TimeoutExpired:
            proc.kill()
        log = "".join(captured)
        if log:
            errs = _scan_log_for_errors(log)
            if errs:
                print("\n[probe] relevant server log lines:")
                for e in errs:
                    print("   ", e)


def _verdict(ok: bool, logs: str) -> int:
    print()
    if ok:
        print("=" * 68)
        print("PASS — the server came up healthy at block_size=1.")
        print("Your ROCm kernel accepts block_size=1: EarlyBird recovery can be")
        print("token-granular. Set server.block_size: 1 in your config for Exp 02.")
        print("=" * 68)
        return 0
    print("=" * 68)
    print("FAIL — the server did NOT become healthy at block_size=1.")
    print("Your ROCm/gfx1100 kernel most likely rejects block_size=1.")
    print("Keep server.block_size: 16. Exp 01 and the audit are unaffected;")
    print("Exp 02 recovery becomes block-granular (recover one 16-token block at")
    print("a time instead of one token). See README 'EarlyBird on ROCm'.")
    print("=" * 68)
    if logs:
        errs = _scan_log_for_errors(logs)
        if errs:
            print("\n[probe] relevant server log lines:")
            for e in errs:
                print("   ", e)
    return 1


def _os_environ() -> dict:
    import os
    return dict(os.environ)


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe ROCm block_size=1 support.")
    ap.add_argument("--config", default="config/rocm_7900xtx.yaml")
    ap.add_argument("--native", action="store_true",
                    help="launch bare-metal vLLM instead of Docker")
    ap.add_argument("--timeout", type=float, default=420.0,
                    help="seconds to wait for /health (model load can be slow)")
    args = ap.parse_args()

    if _server_already_up():
        print("[probe] a server is already listening on :8000 — stop it first.",
              file=sys.stderr)
        return 2

    cfg_path = (REPO / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    if not cfg_path.exists():
        print(f"[probe] config not found: {cfg_path}", file=sys.stderr)
        return 2

    probe_cfg = _write_probe_config(cfg_path)
    try:
        if args.native:
            return run_native(probe_cfg, args.timeout)
        return run_docker(args.timeout)
    finally:
        probe_cfg.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

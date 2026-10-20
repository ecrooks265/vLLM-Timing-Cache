#!/usr/bin/env bash
# One-shot: wait for the vLLM server, run the Exp-01 audit, make figures.
# Usage:  bash scripts/run_replication.sh [config/replication.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${1:-config/replication.yaml}"
# pick an interpreter that can import the harness deps; install if none can
PY=""
for c in python3.13 python3.12 python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import pydantic, scipy, matplotlib" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  PY="$(command -v python3.13 || command -v python3)"
  echo "[*] installing harness deps into $PY (user site)…"
  "$PY" -m pip install --user -q -r requirements.txt
fi
echo "[*] python: $PY ($($PY --version 2>&1))"
echo "[*] config: $CFG"

echo "[*] waiting for http://127.0.0.1:8000/health (up to 300s)…"
for i in $(seq 1 150); do
  code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health || true)"
  if [ "$code" = "200" ]; then echo "    healthy"; break; fi
  if [ "$i" = "150" ]; then
    echo "[!] server not healthy. Recent logs:"; docker logs --tail 30 cachelab-vllm 2>&1 || true
    exit 1
  fi
  sleep 2
done

echo "[*] running audit…"
"$PY" scripts/run_experiment.py --config "$CFG" --audit

echo "[*] generating figures…"
"$PY" scripts/make_figures.py

echo "[*] done. newest run:"
ls -1dt runs/*/ | head -1

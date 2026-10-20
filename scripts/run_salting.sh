#!/usr/bin/env bash
# Run the salting experiments (05 propagation + 06 granularity) and make figures.
# Usage: bash scripts/run_salting.sh [config/replication.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${1:-config/replication.yaml}"
PY=""
for c in python3.13 python3.12 python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import pydantic, scipy, matplotlib" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -z "$PY" ] && { PY="$(command -v python3.13 || command -v python3)"; "$PY" -m pip install --user -q -r requirements.txt; }
echo "[*] python: $PY | config: $CFG"
for i in $(seq 1 150); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health || true)" = "200" ] && { echo "    healthy"; break; }
  [ "$i" = "150" ] && { echo "[!] server not healthy"; docker logs --tail 20 cachelab-vllm 2>&1||true; exit 1; }
  sleep 2
done
echo "[*] 05 salt-propagation (leak probe)…"; "$PY" experiments/05_suffix_collision.py --config "$CFG"
echo "[*] 06 salt-granularity (defense sweep)…"; "$PY" experiments/06_salt_granularity.py --config "$CFG"
echo "[*] figures…"; "$PY" scripts/make_salt_figures.py
echo "[*] newest runs:"; ls -1dt runs/*/ | head -3

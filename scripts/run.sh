#!/usr/bin/env bash
# Bring up the vLLM server and wait until it is healthy.
# Usage: bash scripts/run.sh [config/<file>.yaml]   (default: replication.yaml)
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-config/replication.yaml}"
COMPOSE="docker/docker-compose.rocm.yml"
echo "[run] starting vLLM (config: $CONFIG)"
CACHELAB_CONFIG="/workspace/${CONFIG}" docker compose -f "$COMPOSE" up -d
echo "[run] waiting for health at http://127.0.0.1:8000/health ..."
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health || true)
  if [ "$code" = "200" ]; then echo "[run] healthy (200) after ~$((i*5))s — server ready"; exit 0; fi
  sleep 5
done
echo "[run] NOT healthy after ~600s. Check: docker compose -f $COMPOSE logs --tail=50" >&2
exit 1

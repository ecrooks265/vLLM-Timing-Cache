#!/usr/bin/env bash
# Stop the vLLM server (keeps the built image cached for next time).
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose -f docker/docker-compose.rocm.yml down
echo "[stop] vLLM stopped. Image cachelab-vllm:rocm-navi kept cached."

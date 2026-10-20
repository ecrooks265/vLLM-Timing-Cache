#!/usr/bin/env bash
# Launch the vLLM server locally.
#
# Default path: Docker on ROCm (AMD Radeon RX 7900 XTX / gfx1100). The container
# builds vLLM for gfx1100 on first run, then serves with prefix caching + metrics.
#
# Usage:
#   bash scripts/setup_local.sh                 # docker compose up (ROCm)
#   bash scripts/setup_local.sh --build         # force image rebuild
#   bash scripts/setup_local.sh --native CONFIG # bare-metal vLLM (CUDA or a
#                                               # natively-installed ROCm vLLM)
set -euo pipefail

cd "$(dirname "$0")/.."
COMPOSE="docker/docker-compose.rocm.yml"

# Load .env (HF_TOKEN, HF_HOME) if present.
if [[ -f .env ]]; then set -a; source .env; set +a; fi

MODE="docker"
BUILD_FLAG=""
CONFIG="config/rocm_7900xtx.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)  BUILD_FLAG="--build"; shift ;;
    --native) MODE="native"; shift ;;
    *.yaml)   CONFIG="$1"; shift ;;
    *)        echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ "$MODE" == "docker" ]]; then
  command -v docker >/dev/null || { echo "docker not found on PATH" >&2; exit 1; }
  echo "[*] bringing up vLLM on ROCm via $COMPOSE (config: $CONFIG)"
  echo "[*] first build compiles vLLM for gfx1100 — this takes a while."
  echo "[*] health: http://127.0.0.1:8000/health   metrics: http://127.0.0.1:8000/metrics"
  CONFIG="/workspace/${CONFIG}" docker compose -f "$COMPOSE" up $BUILD_FLAG
  exit 0
fi

# ── native fallback (no container) ──────────────────────────────────────────
# Uses whatever `vllm`/python is on PATH. On ROCm this requires a vLLM already
# built for gfx1100 (see docker/Dockerfile.rocm for the build recipe).
echo "[*] native launch from $CONFIG (no container)"
python - "$CONFIG" <<'PY' > /tmp/cachelab_launch.sh
import sys, yaml, shlex
cfg = yaml.safe_load(open(sys.argv[1]))
m, s = cfg["model"], cfg["server"]
args = [
  "python","-m","vllm.entrypoints.openai.api_server",
  "--model", m["name"],
  "--dtype", m.get("dtype","bfloat16"),
  "--max-model-len", str(m.get("max_model_len",4096)),
  "--host", s.get("host","127.0.0.1"),
  "--port", str(s.get("port",8000)),
  "--block-size", str(s.get("block_size",16)),
  "--gpu-memory-utilization", str(s.get("gpu_memory_utilization",0.90)),
]
if s.get("enable_prefix_caching", True): args.append("--enable-prefix-caching")
if s.get("enforce_eager", True): args.append("--enforce-eager")
print("exec " + " ".join(shlex.quote(a) for a in args))
PY
echo "[*] metrics at http://127.0.0.1:8000/metrics"
exec bash /tmp/cachelab_launch.sh

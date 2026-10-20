#!/usr/bin/env bash
# Container entrypoint: parse the mounted config YAML and launch the vLLM
# OpenAI-compatible server with prefix caching + metrics. Mirrors the argv built
# by cachelab.server.build_launch_command so the harness and the server agree.
set -euo pipefail

CONFIG="${CONFIG:-/workspace/config/rocm_7900xtx.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[serve.sh] config not found at $CONFIG" >&2
  echo "[serve.sh] mount your repo and set CONFIG=/workspace/config/<file>.yaml" >&2
  exit 1
fi

# Emit the launch command from the YAML without extra deps.
python - "$CONFIG" <<'PY' > /tmp/launch.sh
import sys, yaml, shlex
cfg = yaml.safe_load(open(sys.argv[1]))
m, s = cfg["model"], cfg["server"]
args = [
  "python", "-m", "vllm.entrypoints.openai.api_server",
  "--model", m["name"],
  "--dtype", m.get("dtype", "bfloat16"),
  "--max-model-len", str(m.get("max_model_len", 4096)),
  "--host", "0.0.0.0",                       # bind all interfaces inside container
  "--port", str(s.get("port", 8000)),
  "--block-size", str(s.get("block_size", 16)),
  "--gpu-memory-utilization", str(s.get("gpu_memory_utilization", 0.90)),
]
if s.get("enable_prefix_caching", True): args.append("--enable-prefix-caching")
if s.get("enforce_eager", True): args.append("--enforce-eager")
print("exec " + " ".join(shlex.quote(a) for a in args))
PY

echo "[serve.sh] gfx1100 / ROCm — Triton FlashAttention path"
echo "[serve.sh] metrics will be at http://<host>:$(python -c "import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['server'].get('port',8000))")/metrics"
echo "[serve.sh] launching:"; cat /tmp/launch.sh
bash /tmp/launch.sh

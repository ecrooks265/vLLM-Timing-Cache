#!/usr/bin/env bash
# Diagnose whether the built image can see the GPU on WSL, and inspect vLLM's
# ROCm platform detection. Read-only. Run:  bash scripts/container_gpu_check.sh
set -uo pipefail
IMG="${IMG:-cachelab-vllm:rocm-navi}"
docker run --rm \
  --device /dev/dxg \
  -v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro \
  -v /opt/rocm/lib/libhsa-runtime64.so.1:/opt/rocm/lib/libhsa-runtime64.so.1:ro \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  --group-add 44 --group-add 110 \
  --security-opt seccomp=unconfined --ipc host \
  --entrypoint bash "$IMG" -lc '
echo "===== TORCH ====="
python - <<PY
import torch
print("torch:", torch.__version__)
print("torch.version.hip:", getattr(torch.version,"hip",None))
try:
    print("cuda.is_available:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
    if torch.cuda.device_count()>0:
        print("name:", torch.cuda.get_device_name(0))
        x=torch.randn(8,8,device="cuda"); y=(x@x).sum().item()
        print("matmul-on-gpu OK:", round(y,3))
except Exception as e:
    print("TORCH GPU ERROR:", repr(e))
PY
echo "===== rocminfo (gfx) ====="
rocminfo 2>&1 | grep -iE "Name:.*gfx|gfx1100" | head
echo "===== /dev/dxg in container ====="
ls -l /dev/dxg 2>&1
echo "===== vllm platforms file: patch + rocm fn ====="
F=$(python -c "import importlib.util;print(importlib.util.find_spec(\"vllm.platforms\").origin)")
echo "file: $F"
grep -n "CACHELAB_WSL_PATCH" "$F" && echo "(patch present)" || echo "(NO cachelab patch)"
grep -n "def rocm_platform_plugin\|amdsmi\|def .*_platform_plugin" "$F" | head -20
echo "===== vllm version ====="
python -c "import vllm; print(vllm.__version__)"
'

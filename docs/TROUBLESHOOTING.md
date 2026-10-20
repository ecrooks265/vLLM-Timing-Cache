# Troubleshooting — gfx1100 / ROCm / WSL2 / Docker

## GPU not visible inside the container
Symptom: vLLM logs `No HIP GPUs are available` or falls back to CPU.
- `/dev/dxg` must exist in the distro: `ls -l /dev/dxg`. Missing → ROCm-on-WSL
  isn't installed, or Windows GPU driver is too old. Install AMD's WSL driver +
  ROCm-on-WSL packages inside the distro.
- The compose file passes `/dev/dxg` and mounts `/usr/lib/wsl/lib`. If you built
  before those were set, rebuild.
- Confirm on the host side first: `rocminfo | grep gfx1100` inside the distro
  must succeed before Docker can.

## Must use native docker engine, NOT Docker Desktop
AMD ROCm GPU passthrough into Docker Desktop's managed distro is unsupported and
flaky. Install the Docker **engine inside your Ubuntu WSL2 distro**:
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker,render,video $USER   # re-login after
sudo service docker start
docker context use default
```
Verify `docker info` shows `Operating System: Ubuntu` (the distro), not
`Docker Desktop`.

## Build fails: pulls a CUDA torch / wrong arch
The Dockerfile runs `use_existing_torch.py` to keep the ROCm torch from the base
image. If a vLLM version ignores it and drags in CUDA torch, pin a vLLM tag
known-good for the base image's ROCm (see docs/SETUP.md version table).

## Build fails: FlashAttention / CK errors
gfx1100 has no CK FlashAttention target. Build must use `BUILD_FA=0` (set in the
Dockerfile) and runtime must use the Triton FA path
(`VLLM_USE_TRITON_FLASH_ATTN=1`, set in compose). If you see CK FA compile
errors, the BUILD_FA arg isn't taking effect.

## HIP out of memory during warmup (24GB card)
Drop `gpu_memory_utilization` to 0.85 (or 0.80) in config/rocm_7900xtx.yaml.
Llama-3.1-8B bf16 ≈ 16GB weights; the rest is KV cache.

## block_size=1 rejected
Expected on ROCm V1. Keep block_size=16 and use block-granular recovery
(Exp 09). Probe with: `python scripts/probe_block_size.py --config
config/rocm_7900xtx.yaml` (server stopped). PASS → set 1; FAIL → stay at 16.

## Model download 401 / gated
Request access to meta-llama/Llama-3.1-8B-Instruct on HF, put HF_TOKEN in .env.
The HF cache is mounted so you download once across rebuilds.

## Slow first build
vLLM compiles from source for gfx1100 on first build — expect a long build.
It's cached after; only `--build` or a changed Dockerfile recompiles.

## Server up but harness can't connect
Health is on the host loopback: `curl http://127.0.0.1:8000/health` → 200.
Metrics: `curl http://127.0.0.1:8000/metrics`. The compose maps
`127.0.0.1:8000:8000`; the harness runs on the host (distro) in a venv.

## "RuntimeError: Device string must not be empty" (WSL amdsmi)
vLLM's ROCm platform probe calls `amdsmi`, which fails under WSL (amdsmi
"Error code: 34" — no /dev/kfd). The Dockerfile bakes a guarded patch that
forces `RocmPlatform` when torch reports a HIP build, so this should not appear.
If it does (e.g. after swapping the base image), apply the same patch in the
running container and commit it, or rebuild:
```bash
docker exec -it cachelab-vllm python - <<'PY'
import importlib.util, pathlib
p = pathlib.Path(importlib.util.find_spec("vllm.platforms").origin)
print(p)   # confirm the file, then re-apply the patch block from Dockerfile.rocm
PY
```

## Docker says "Operating System: Docker Desktop" even in Ubuntu
Docker Desktop's WSL integration shadows the native engine. Uncheck this distro
in Docker Desktop → Settings → Resources → WSL integration, `wsl --shutdown`,
reopen Ubuntu, then `bash scripts/wsl_docker_native.sh`. Confirm with
`docker info | grep "Operating System"` → must be Ubuntu.

## Native fallback (no Docker)
Proven on this exact hardware (ROCm 7.2 + WSL2, RX 7900 XTX):
```bash
cd ~/repos/cache-timing-lab            # keep on the Linux side, not /mnt/c
python3 -m venv ~/ai-ws && source ~/ai-ws/bin/activate
# ROCm already installed on host (amdgpu-install 7.2, --usecase=wsl,rocm --no-dkms)
git clone https://github.com/vllm-project/vllm.git && cd vllm
git checkout v0.9.2          # or a tag matching your torch; see vLLM ROCm docs
PYTORCH_ROCM_ARCH=gfx1100 HSA_OVERRIDE_GFX_VERSION=11.0.0 BUILD_FA=0 \
  python use_existing_torch.py && pip install -e . --no-build-isolation
```
If vLLM's platform probe fails on WSL, patch `vllm/platforms/__init__.py`
`rocm_platform_plugin()` to return `RocmPlatform` when `torch.version.hip` is
set and `torch.cuda.is_available()`. Then launch via
`bash scripts/setup_local.sh --native config/rocm_7900xtx.yaml`.

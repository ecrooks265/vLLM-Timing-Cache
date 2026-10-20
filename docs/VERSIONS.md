# Known-good versions — vLLM on RX 7900 XTX via Docker, WSL2

Verified working 2026-10-10: native docker in WSL2 Ubuntu → `/dev/dxg` +
host WSL libhsa → ROCm detected → vLLM serving on the 7900 XTX.

## Host (inside the Ubuntu WSL2 distro)
| Component | Version / value |
|-----------|-----------------|
| Windows GPU | AMD Radeon RX 7900 XTX (gfx1100, RDNA3, 24GB) |
| WSL distro | Ubuntu 22.04.5 LTS |
| Kernel | 6.18.40.1-microsoft-standard-WSL2 |
| ROCm (host) | 7.2.0 — `amdgpu-install_7.2.70200-1`, `--usecase=wsl,rocm --no-dkms` |
| Host libhsa | `/opt/rocm/lib/libhsa-runtime64.so.1.18.70200` |
| Docker engine | docker.io **29.1.3**, native in-distro (NOT Docker Desktop) |
| docker compose | v2 plugin (`docker-compose-v2`) |
| GPU device node | `/dev/dxg` (crw-rw-rw-, world-rw) |
| User groups | `video`(44), `render`(110) |

`rocm-smi` prints nothing under WSL — normal; `rocminfo` seeing gfx1100 is the
real check.

## Container (AMD prebuilt navi image — no source build)
| Component | Version / value |
|-----------|-----------------|
| Base image | `rocm/vllm-dev:rocm6.4.2_navi_ubuntu22.04_py3.10_pytorch_2.7_vllm_0.9.2` |
| ROCm (container) | 6.4.2 |
| PyTorch | 2.7.0a0+git295f2ed (`torch.version.hip` = 6.4.43484-123eb5128) |
| vLLM | 0.9.3.dev2+g8fa211321 |
| Python | 3.10 (conda env `py_3.10`) |

Container ROCm 6.4 + host libhsa 7.2 is an intentional mix — the HSA soname
`.1` ABI is stable across that gap, and it works.

## The non-obvious WSL glue (why it works)
All of this is already wired into `docker/docker-compose.rocm.yml` +
`docker/Dockerfile.rocm`. Documented so you know what matters if anything drifts:

1. **Native docker engine, not Docker Desktop.** Docker Desktop's engine can't
   pass `/dev/dxg`. Install with `scripts/wsl_docker_native.sh`; confirm
   `docker info | grep "Operating System"` says Ubuntu.
2. **`group_add` must be numeric GIDs** (`"44"`, `"110"`), not names — the navi
   image has no `render` group, so names error out.
3. **Mount the host's WSL libhsa over the container's** (the key fix):
   `-/opt/rocm/lib/libhsa-runtime64.so.1:/opt/rocm/lib/libhsa-runtime64.so.1:ro`.
   Without it: `torch.cuda.is_available()` False, `rocminfo` empty — the
   container's bare-metal runtime looks for `/dev/kfd`, absent on WSL.
4. **Mount `/usr/lib/wsl/lib`** (+ it on `LD_LIBRARY_PATH`) for `libdxcore`.
5. **Device + flags:** `--device /dev/dxg`, `ipc: host`,
   `security_opt: seccomp=unconfined`, `cap_add: SYS_PTRACE`.
6. **vLLM amdsmi patch** (baked into `Dockerfile.rocm`): forces `RocmPlatform`
   when torch reports HIP, because `amdsmi` fails on WSL (Error 34). Triggers
   correctly once the libhsa mount makes `cuda.is_available()` True.
7. **Env:** `HSA_OVERRIDE_GFX_VERSION=11.0.0`, `VLLM_USE_TRITON_FLASH_ATTN=1`.

## Models
| Config | Model | Gated? | Use |
|--------|-------|--------|-----|
| `config/_smoke.yaml` | Qwen/Qwen2.5-0.5B-Instruct | no | validate the stack (default now) |
| `config/rocm_7900xtx.yaml` | meta-llama/Llama-3.1-8B-Instruct | **yes** | real experiments (needs HF access — pending) |

Select the config at runtime:
`CACHELAB_CONFIG=/workspace/config/_smoke.yaml docker compose -f docker/docker-compose.rocm.yml up`

## Run / stop
```bash
# start (smoke)
CACHELAB_CONFIG=/workspace/config/_smoke.yaml docker compose -f docker/docker-compose.rocm.yml up -d
curl http://127.0.0.1:8000/health          # -> 200
# stop
docker compose -f docker/docker-compose.rocm.yml down
```
The built image `cachelab-vllm:rocm-navi` stays cached, so restarts don't
rebuild. If the navi base tag ever disappears from Docker Hub, pin it by digest
in the compose `BASE_IMAGE` (get it with
`docker inspect --format='{{index .RepoDigests 0}}' rocm/vllm-dev:rocm6.4.2_navi_ubuntu22.04_py3.10_pytorch_2.7_vllm_0.9.2`).

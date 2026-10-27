# cache-timing-lab

White-box testbed for reproducing and measuring prompt-caching timing side
channels in LLM serving, built around the attack from Gu et al.,
*Auditing Prompt Caching in Language Model APIs* (ICML 2025, arXiv:2502.07776)
and the token-reconstruction extension from Song et al., *The Early Bird
Catches the Leak* (arXiv:2409.20002).

The point of running this **white-box** (we control the vLLM server and can read
its real cache state) is to obtain **ground-truth hit/miss labels**. That lets us
separate two things a black-box audit cannot: how often the cache actually hit,
versus how often the timing signal *detected* the hit. Every precision number this
harness reports is anchored to real cache state, not inferred from timing alone.

## What this does

1. **Stands up a vLLM server** with prefix caching enabled, portable between a
   local GPU and a GCP instance. Pinned to a build that still supports
   `block_size=1` so the EarlyBird token-by-token attack is reproducible.
2. **Measures TTFT** (time-to-first-token) for crafted requests, with both
   client-side and server-side timing.
3. **Labels every request hit/miss from ground truth** by reading vLLM's own
   prefix-cache metrics — not from timing.
4. **Runs the audit protocol** at two cache-sharing levels: per-user and global.
   (Per-org is intentionally out of scope — see the proposal.)
5. **Runs a token-by-token prefix-recovery attack** (EarlyBird-style) and scores
   it against the known victim prompt.

## Layout

```
cache-timing-lab/
├── README.md
├── requirements.txt          # harness-only deps (no vLLM; server runs in Docker)
├── pyproject.toml
├── .env.example
├── config/
│   ├── default.yaml          # GCP / NVIDIA + pinned-0.6.x profile (block_size=1)
│   └── rocm_7900xtx.yaml      # AMD RX 7900 XTX / modern-vLLM profile (block_size=16)
├── docker/
│   ├── Dockerfile.rocm        # builds vLLM for gfx1100 (RDNA3)
│   ├── docker-compose.rocm.yml# GPU passthrough + /metrics on 127.0.0.1:8000
│   └── serve.sh               # config-driven launcher (container entrypoint)
├── scripts/
│   ├── setup_local.sh        # bring up vLLM on ROCm via Docker (default)
│   ├── setup_gcp.sh          # provision + launch vLLM on a GCP GPU VM
│   ├── probe_block_size.py   # test whether this GPU accepts block_size=1
│   └── run_experiment.py     # top-level entry point (end-to-end)
├── src/cachelab/
│   ├── __init__.py
│   ├── config.py             # typed config loader
│   ├── server.py             # vLLM launch + health + cache-metrics reader
│   ├── client.py             # OpenAI-compatible client w/ TTFT measurement
│   ├── timing.py             # TTFT harness, sampling, warmup control
│   ├── groundtruth.py        # reads real cache state -> hit/miss labels
│   ├── audit.py              # per-user / global side-channel audit + KS test
│   ├── attack.py             # EarlyBird token-by-token prefix recovery
│   ├── stats.py              # KS test, precision/recall, calibration curves
│   └── io_utils.py           # run dirs, JSONL logging, result schemas
├── experiments/
│   ├── 01_prove_sidechannel.py
│   ├── 02_recover_prefix.py
│   └── 03_calibration_sweep.py
└── tests/
    ├── test_stats.py
    └── test_groundtruth.py
```

> **Docker setup moved to [docs/SETUP.md](docs/SETUP.md).** The server now uses AMD's
> prebuilt consumer-RDNA3 ("navi") vLLM image — **no source build**. The
> from-source Dockerfile is kept as `docker/Dockerfile.rocm.sourcebuild.bak`.
> See [docs/](docs/) for setup, research context, and troubleshooting.

## Quickstart (AMD RX 7900 XTX / ROCm, via Docker)

The server runs in a container built for gfx1100; the measurement harness runs
on the host in a plain venv and talks to it over HTTP.

```bash
# 0. one-time: HF access to the gated model, then creds
#    https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct  (request access)
cp .env.example .env            # set HF_TOKEN=..., optionally HF_HOME=...

# 1. install the HARNESS only (no vLLM here — it lives in the container)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. build + launch the vLLM server on ROCm (first build compiles for gfx1100)
bash scripts/setup_local.sh                 # docker compose up
#   wait for GET http://127.0.0.1:8000/health -> 200

# 3. run the full pipeline against the running server
python scripts/run_experiment.py --config config/rocm_7900xtx.yaml --all
```

On NVIDIA/GCP instead, use the original profile and server path:

```bash
bash scripts/setup_local.sh --native config/default.yaml   # or scripts/setup_gcp.sh
python scripts/run_experiment.py --config config/default.yaml --all
```

## ROCm bring-up notes (7900 XTX / gfx1100)

- **Host prerequisites:** a working ROCm install (`rocm-smi` lists the card),
  Docker, and your user in the `render`/`video` groups. The container gets the
  GPU via `--device=/dev/kfd --device=/dev/dri` (already set in the compose file).
- **Why a source build:** prebuilt `rocm/vllm` images are compiled for datacenter
  archs (gfx942) and ship no gfx1100 kernels; `HSA_OVERRIDE_GFX_VERSION` can't
  create kernels that weren't compiled. `Dockerfile.rocm` builds vLLM with
  `PYTORCH_ROCM_ARCH=gfx1100` and `BUILD_FA=0` (no CK FlashAttention on RDNA3 —
  the Triton FA path is used via `VLLM_USE_TRITON_FLASH_ATTN=1`).
- **Match the base image to your ROCm.** Set `BASE_IMAGE` / `VLLM_REF` build args
  in `docker/docker-compose.rocm.yml`. `rocm/pytorch` tags:
  https://hub.docker.com/r/rocm/pytorch/tags — pick one whose ROCm ≤ your host's.
- **OOM on a 24GB card:** drop `gpu_memory_utilization` to `0.85` in the config.
- **`enforce_eager: true`** is kept on purpose — HIP graphs add TTFT jitter that
  would contaminate the timing signal.

### Running under WSL2 (Windows host)

ROCm for the 7900 XTX on Windows runs inside a WSL2 Linux distro, and GPU
passthrough differs from bare metal:

- Install **ROCm-on-WSL** inside your WSL2 distro (AMD's WSL ROCm packages) and
  install Docker **inside** that distro (not Docker Desktop's default context).
  Verify the GPU is visible with `rocminfo` / `rocm-smi` from the WSL shell first.
- `docker/docker-compose.rocm.yml` is **already set for WSL2** — it uses
  `/dev/dxg` and mounts `/usr/lib/wsl/lib`. No toggling needed. (For a bare-metal
  Linux host instead, flip the commented `/dev/kfd`+`/dev/dri` lines back on and
  drop the `/dev/dxg` device and the wsl lib mount.)
- Keep the repo on the **Linux side** of WSL2 (e.g. `~/repos/cache-timing-lab`
  inside the distro), not under `/mnt/c/...` — building vLLM across the 9p mount
  from the Windows drive is painfully slow.

## EarlyBird on ROCm — the block_size caveat

The token-by-token prefix recovery (Exp 02) wants `block_size=1` so every cached
token is an independently observable hit. Two things fight that on this hardware:

1. The modern vLLM **V1 engine dropped `block_size=1`**.
2. The **ROCm paged-attention path** on gfx1100 generally wants block size 16.

So the ROCm profile defaults to `block_size=16`. **Exp 01 (prove the side
channel) and the per-user/global audit are unaffected** — they work at any block
size. Only Exp 02's *granularity* changes: you recover one 16-token block at a
time instead of one token at a time.

Before relying on either, probe your actual kernel (server stopped):

```bash
python scripts/probe_block_size.py --config config/rocm_7900xtx.yaml
```

- **PASS** → your kernel accepts `block_size=1`; set it in the config and Exp 02
  is token-granular.
- **FAIL** → stay at 16 and use block-granular recovery. If you need true
  token-granular recovery, that's the one case for the pinned `vllm==0.6.3.post1`
  path (`config/default.yaml`), which is a much harder build on gfx1100.

## Status

Scaffold. Every module has a working skeleton, typed signatures, and TODO
markers where the measurement logic gets fleshed out. `stats.py` and the
ground-truth hit/miss labeling are implemented enough to unit-test.

## Safety / scope

Run only against **your own** vLLM instance, or external APIs **you are
authorized to test**. The external-API path is gated behind an explicit
`--allow-external` flag and is not wired to any provider by default.

## Multimodal cache isolation extension

Ruilin’s [multimodal study and results](docs/MULTIMODAL_RESULTS.md) adds a completed vLLM 0.11.0 image-isolation experiment, reuse-latency measurements, runnable scripts and raw data, with historical text/audio evidence kept separate.

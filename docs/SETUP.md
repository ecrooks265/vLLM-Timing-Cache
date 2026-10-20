# Setup — vLLM on RX 7900 XTX (gfx1100) via Docker, WSL2

Turnkey path to a running vLLM server you can stop/start to pick up testing any
time. Uses AMD's **prebuilt navi image** (no vLLM compile). Everything runs
**inside your Ubuntu WSL2 distro** with the **native docker engine**.

## Verified host (yours)
- WSL2 Ubuntu 22.04, kernel `-microsoft-standard-WSL2`
- `/dev/dxg` present · `rocminfo` shows `gfx1100` (RX 7900 XTX) · ROCm 7.2
- user in `render` + `video`

Re-check anytime: `bash scripts/doctor.sh`

## One-time: native docker engine (not Docker Desktop)
Docker Desktop's engine can't pass `/dev/dxg` to containers reliably.

1. Docker Desktop → Settings → Resources → **WSL integration** → **uncheck
   Ubuntu** → Apply & Restart.
2. PowerShell: `wsl --shutdown`, then reopen Ubuntu.
3. In Ubuntu:
   ```bash
   cd /mnt/c/Users/ecroo/Desktop/School/AI_INFOSEC_FINALPROJ/cache-timing-lab
   bash scripts/wsl_docker_native.sh
   newgrp docker          # or open a new shell
   docker info | grep "Operating System"   # must say Ubuntu, NOT Docker Desktop
   ```

## One-time: credentials
```bash
cp .env.example .env      # if not already
# edit .env: HF_TOKEN=<your hf token with Llama-3.1-8B access>
# optional: HF_HOME=/home/<you>/.cache/huggingface   (fast Linux-side cache)
```
Request model access first: huggingface.co/meta-llama/Llama-3.1-8B-Instruct

## Build + run the server
```bash
docker compose -f docker/docker-compose.rocm.yml up --build
```
- First run pulls the navi image (large) + downloads the model once (cached).
- Ready when, in another shell:
  ```bash
  curl http://127.0.0.1:8000/health      # -> 200
  curl http://127.0.0.1:8000/metrics | grep prefix_cache
  ```
- Stop: Ctrl-C (or `docker compose -f docker/docker-compose.rocm.yml down`).
- Restart later (no rebuild): `docker compose -f docker/docker-compose.rocm.yml up`.

## Run the experiments (harness on the host)
In a second Ubuntu shell:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # harness only, no vLLM
python scripts/run_experiment.py --config config/rocm_7900xtx.yaml --all
```

## Confirm the GPU is actually in use
In the server logs you want ROCm/HIP init, not CPU fallback. Quick check:
```bash
docker exec -it cachelab-vllm python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True  Radeon RX 7900 XTX
```
If `False` or "Device string must not be empty" → see docs/TROUBLESHOOTING.md
(WSL amdsmi patch is baked into the image; if GPU still not seen, try a rocm7
navi base tag in docker-compose.rocm.yml).

## Pick up testing any time (the whole point)
```bash
cd .../cache-timing-lab
docker compose -f docker/docker-compose.rocm.yml up -d    # server in background
source .venv/bin/activate
python scripts/run_experiment.py --config config/rocm_7900xtx.yaml --all
```

## Fallback: native venv (no Docker)
If Docker GPU passthrough won't cooperate, the proven path on this exact
hardware is a native build in `~/repos/cache-timing-lab` (ROCm 7.2 + torch +
vLLM in a venv). See docs/TROUBLESHOOTING.md "native fallback".

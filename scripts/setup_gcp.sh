#!/usr/bin/env bash
# Provision a GCP GPU VM and launch the same vLLM server as setup_local.sh.
# Requires gcloud configured. Uses your project's cloud credits for GPU serving.
#
# This creates an L4 instance by default (cheap, enough for Llama 3.1 8B at
# bf16). Swap to a2-highgpu-1g (A100) for lower-latency timing if credits allow.
set -euo pipefail

ZONE="${ZONE:-us-central1-a}"
INSTANCE="${INSTANCE:-cachelab-vllm}"
MACHINE="${MACHINE:-g2-standard-8}"          # L4-backed
GPU="${GPU:-type=nvidia-l4,count=1}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu124-ubuntu-2204}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

cmd="${1:-create}"

case "$cmd" in
  create)
    echo "[*] creating GPU VM $INSTANCE in $ZONE ($MACHINE, $GPU)..."
    gcloud compute instances create "$INSTANCE" \
      --zone="$ZONE" \
      --machine-type="$MACHINE" \
      --accelerator="$GPU" \
      --image-family="$IMAGE_FAMILY" \
      --image-project="$IMAGE_PROJECT" \
      --maintenance-policy=TERMINATE \
      --boot-disk-size=200GB \
      --metadata="install-nvidia-driver=True"
    echo "[*] VM created. SSH in and run the remote setup below."
    ;;

  setup-remote)
    # Run this ON the VM after SSHing in.
    echo "[*] installing project + vLLM on the VM..."
    sudo apt-get update -y && sudo apt-get install -y python3-venv git
    python3 -m venv .venv && source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "[*] launching vLLM..."
    bash scripts/setup_local.sh
    ;;

  tunnel)
    # Forward the VM's vLLM port to localhost so the harness can run locally.
    echo "[*] forwarding localhost:8000 -> $INSTANCE:8000 ..."
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" -- -N -L 8000:localhost:8000
    ;;

  delete)
    gcloud compute instances delete "$INSTANCE" --zone="$ZONE" --quiet
    ;;

  *)
    echo "usage: $0 {create|setup-remote|tunnel|delete}" >&2
    exit 1
    ;;
esac

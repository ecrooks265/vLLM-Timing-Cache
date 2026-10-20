#!/usr/bin/env bash
# Install + start the NATIVE Docker engine inside this Ubuntu WSL2 distro, so
# containers can get the AMD GPU via /dev/dxg. Docker Desktop's engine runs in
# its own distro and can't pass /dev/dxg reliably for ROCm.
#
# BEFORE running this: in Docker Desktop -> Settings -> Resources -> WSL
# integration, UNCHECK this Ubuntu distro, then Apply & Restart. Otherwise the
# Desktop CLI shadows the native one.
#
# Run:  bash scripts/wsl_docker_native.sh   (it will use sudo)
set -euo pipefail

echo "[1/5] current docker engine:"
docker info 2>/dev/null | grep -E "Operating System|Name:" || echo "  (no docker responding)"

if docker info 2>/dev/null | grep -q "Docker Desktop"; then
  echo "!! docker is still Docker Desktop. Disable WSL integration for this"
  echo "   distro in Docker Desktop settings, 'wsl --shutdown' from PowerShell,"
  echo "   reopen Ubuntu, then re-run this script."
fi

echo "[2/5] installing docker engine (docker.io)…"
sudo apt-get update -y
sudo apt-get install -y docker.io docker-compose-v2 curl

echo "[3/5] groups…"
sudo usermod -aG docker,render,video "$USER" || true

echo "[4/5] starting dockerd (WSL has no systemd by default)…"
sudo service docker start || sudo dockerd >/tmp/dockerd.log 2>&1 &
sleep 3
docker context use default >/dev/null 2>&1 || true

echo "[5/5] verify:"
docker info 2>/dev/null | grep -E "Operating System|Server Version|Name:" || {
  echo "  dockerd not up yet — check /tmp/dockerd.log, or run: sudo service docker start"; exit 1; }
echo
echo "If 'Operating System' above is Ubuntu (NOT Docker Desktop), you're set."
echo "You may need to open a NEW shell for the 'docker' group to take effect"
echo "(or run: newgrp docker)."

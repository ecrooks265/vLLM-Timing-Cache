#!/usr/bin/env bash
# Environment discovery for the cache-timing-lab ROCm/WSL/Docker setup.
# Run inside your Ubuntu WSL2 distro:  bash scripts/doctor.sh
# Read-only. Paste the whole output back.
echo "=================== cache-timing-lab doctor ==================="
echo "--- kernel / WSL ---"
uname -r
grep -qi microsoft /proc/sys/kernel/osrelease && echo "WSL2: yes" || echo "WSL2: no (bare metal?)"
echo "--- distro ---"
. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"
echo "--- /dev/dxg (WSL GPU node) ---"
ls -l /dev/dxg 2>/dev/null || echo "MISSING /dev/dxg"
echo "--- /usr/lib/wsl/lib ---"
ls /usr/lib/wsl/lib 2>/dev/null | head -20 || echo "MISSING /usr/lib/wsl/lib"
echo "--- ROCm install ---"
ls -d /opt/rocm* 2>/dev/null || echo "no /opt/rocm"
cat /opt/rocm/.info/version 2>/dev/null || echo "no rocm version file"
echo "--- rocminfo (gfx target) ---"
command -v rocminfo >/dev/null && rocminfo 2>/dev/null | grep -E "Name:|gfx|Marketing" | head -20 || echo "rocminfo NOT on PATH"
echo "--- rocm-smi ---"
command -v rocm-smi >/dev/null && rocm-smi 2>/dev/null | head -15 || echo "rocm-smi NOT on PATH"
echo "--- user groups ---"
id | tr ',' '\n' | grep -E "render|video" || echo "NOT in render/video groups"
echo "--- docker ---"
command -v docker >/dev/null && docker version 2>/dev/null | grep -E "Version|API" | head -6 || echo "docker NOT on PATH"
echo "docker context:"; docker context ls 2>/dev/null
echo "docker info (engine):"; docker info 2>/dev/null | grep -E "Server Version|Operating System|Name:|Docker Root" | head -8
echo "--- docker compose plugin ---"
docker compose version 2>/dev/null || echo "docker compose plugin MISSING"
echo "--- .env present? ---"
[ -f .env ] && echo ".env exists" || echo ".env MISSING (copy from .env.example, set HF_TOKEN)"
echo "--- repo location ---"
pwd
echo "=================== end ==================="

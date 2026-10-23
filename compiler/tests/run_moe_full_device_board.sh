#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then exec sudo "$0" "$@"; fi
found=0
for f in /etc/profile.d/pynq_venv.sh /etc/profile.d/xrt_setup.sh /opt/xilinx/xrt/setup.sh; do
    if [ -f "$f" ]; then source "$f" >/dev/null; found=1; fi
done
[ "$found" -eq 1 ] || { echo "No PYNQ/XRT environment found." >&2; exit 1; }
python3 -c "import pynq" 2>/dev/null || { echo "Python cannot import pynq." >&2; exit 1; }
cd "$(dirname "$0")"
exec python3 board_moe_full_device.py "$@"

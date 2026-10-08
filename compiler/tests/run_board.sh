#!/usr/bin/env bash
# Runs board_regress.py on the board with the PYNQ environment, on either board:
#   sudo ./run_board.sh [test ...] [--no-profile]
# PYNQ-Z1 images have /etc/profile.d/pynq_venv.sh and xrt_setup.sh; Kria-PYNQ
# on Ubuntu for Kria has pynq_venv.sh, and XRT's setup under /opt/xilinx/xrt.
# Each one that exists is sourced.
set -e
found=0
for f in /etc/profile.d/pynq_venv.sh /etc/profile.d/xrt_setup.sh /opt/xilinx/xrt/setup.sh; do
    if [ -f "$f" ]; then
        # shellcheck disable=SC1090
        source "$f" >/dev/null
        found=1
    fi
done
[ $found = 1 ] || { echo "no PYNQ environment found (/etc/profile.d/pynq_venv.sh): install PYNQ (Kria-PYNQ)" >&2; exit 1; }
python3 -c "import pynq" 2>/dev/null || { echo "python3 cannot import pynq (PYNQ venv not active?)" >&2; exit 1; }
cd "$(dirname "$0")"
exec python3 board_regress.py "$@"

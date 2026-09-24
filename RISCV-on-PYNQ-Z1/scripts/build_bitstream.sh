#!/usr/bin/env bash
# One-click PicoRV32/PYNQ-Z1 bitstream build. Options are passed through to
# build_bitstream.tcl: [-jobs N] [-sa_d 8|16] [-proj_dir DIR] [-allow_unplaced_io]
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Override with VIVADO_SETTINGS=/path/to/Vivado/2024.1/settings64.sh
VIVADO_SETTINGS="${VIVADO_SETTINGS:-/home/jon/Projects/Vivado/Vivado/2024.1/settings64.sh}"
if ! command -v vivado >/dev/null 2>&1 && [ -f "$VIVADO_SETTINGS" ]; then
    # shellcheck disable=SC1090
    source "$VIVADO_SETTINGS"
fi
if ! command -v vivado >/dev/null 2>&1; then
    echo "ERROR: vivado not found. Set VIVADO_SETTINGS or source settings64.sh first." >&2
    exit 1
fi

mkdir -p "$repo_root/build"
cd "$repo_root/build"
exec vivado -mode batch -notrace \
    -log vivado.log -journal vivado.jou \
    -source "$repo_root/scripts/build_bitstream.tcl" -tclargs "$@"

#!/usr/bin/env bash
# One-click KV260 bitstream build (docs/kv260_upgrade_plan.md K1a). Options are
# passed to build_bitstream.tcl: [-jobs N] [-sa_d 8|16] [-sa_mhz MHZ] [-proj_dir DIR] [-bd_only]
set -euo pipefail

kv_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

mkdir -p "$kv_root/build"
cd "$kv_root/build"
exec vivado -mode batch -notrace \
    -log vivado.log -journal vivado.jou \
    -source "$kv_root/scripts/build_bitstream.tcl" -tclargs "$@"

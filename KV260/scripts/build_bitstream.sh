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
cmd=(vivado -mode batch -notrace -log vivado.log -journal vivado.jou
     -source "$kv_root/scripts/build_bitstream.tcl" -tclargs "$@")

# Vivado runs in its own memory-capped systemd scope. The UltraScale+ build
# fills the page cache on top of ~6 GB of processes (main + synthesis); in the
# terminal's own cgroup that pushed the session's memory pressure past
# systemd-oomd's limit, which then killed the whole terminal. MemoryHigh makes
# the kernel reclaim Vivado's own cache first; beyond MemoryMax only this
# scope is killed. KV260_BUILD_MEM / KV260_BUILD_MEM_HIGH override the caps,
# KV260_BUILD_NO_SCOPE=1 runs Vivado directly.
MEM=${KV260_BUILD_MEM:-11G}
MEM_HIGH=${KV260_BUILD_MEM_HIGH:-9G}
if [ -z "${KV260_BUILD_NO_SCOPE:-}" ] && command -v systemd-run >/dev/null 2>&1; then
    echo "vivado in a systemd scope: MemoryHigh=$MEM_HIGH MemoryMax=$MEM (log: $kv_root/build/vivado.log)"
    exec systemd-run --user --scope -q -p MemoryHigh="$MEM_HIGH" -p MemoryMax="$MEM" "${cmd[@]}"
fi
exec "${cmd[@]}"

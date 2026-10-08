#!/usr/bin/env bash
# One-click KV260 bitstream build (docs/kv260_upgrade_plan.md K1a / K1b / K1c). Options are
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
# the configuration's name for the log (build_bitstream.tcl names its outputs the same way)
sa_d=8; sa_mhz=50
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case ${args[i]} in
        -sa_d)   sa_d=${args[i+1]:-8} ;;
        -sa_mhz) sa_mhz=${args[i+1]:-50} ;;
    esac
done
cfg=d${sa_d}_${sa_mhz}mhz
cmd=(vivado -mode batch -notrace -log "vivado_$cfg.log" -journal "vivado_$cfg.jou"
     -source "$kv_root/scripts/build_bitstream.tcl" -tclargs "$@")

# Vivado runs in its own systemd scope with a hard cap (MemoryMax): beyond it
# only this scope is killed, never the terminal (systemd-oomd killed the whole
# terminal when Vivado ran in its cgroup). No MemoryHigh: its throttling
# itself raised the session's memory pressure (88%) until oomd killed the
# build. The build runs in one Vivado process (build_bitstream.tcl) to keep
# memory down. KV260_BUILD_MEM overrides the cap, KV260_BUILD_NO_SCOPE=1 runs
# Vivado directly.
MEM=${KV260_BUILD_MEM:-12G}
if [ -z "${KV260_BUILD_NO_SCOPE:-}" ] && command -v systemd-run >/dev/null 2>&1; then
    echo "vivado in a systemd scope: MemoryMax=$MEM (log: $kv_root/build/vivado_$cfg.log)"
    exec systemd-run --user --scope -q -p MemoryMax="$MEM" "${cmd[@]}"
fi
exec "${cmd[@]}"

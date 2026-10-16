#!/usr/bin/env bash
# Stages the RISC-V backend board test (docs/riscv_backend_design.md §5, R1 / R3):
# six SmolLM2-135M decode dispatches (a 2-layer export compiled for the overlay,
# RVB_SRC) prepared by compiler/tests/rv_dispatch.py, the firmware/rvb images
# per case (fw_bram.bin; fw_ddr.bin + code.bin), board_rvb.py, the overlay.
#   SA_KV260_CONFIG=d16_100mhz_w128_p6_gemv_np4_pcpi compiler/scripts/deploy_rvb.sh
#     -> build/deploy_rvb_<config>; on the board: python3 board_run.py rvb
# The GEMV cases (gemv_run) need an overlay with GEMV over PCPI (CAPS bit 26).
#   RVB_SRC=<sa_sources dir>   (default build/hfgen/smollm2_l2_e2e_qf/sa_sources)
#   RVB_T=<T>                  the attention length of the cases (default 64)
#   RVB_GEN=compiled           rvb_gen calls each export's generated function (plan A, §9:
#                              rv_dispatch.py --code compiled) instead of the traced commands
#                              -> build/deploy_rvb_<config>_compiled (default: trace)
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source "$(dirname "$0")/board_env.sh"
SRC=${RVB_SRC:-$SA_REPO/build/hfgen/smollm2_l2_e2e_qf/sa_sources}
T=${RVB_T:-64}
GEN=${RVB_GEN:-trace}
case $GEN in trace) W=rvb; D=$SA_REPO/build/deploy_rvb_$SA_KV260_CONFIG ;;
             compiled) W=rvc; D=$SA_REPO/build/deploy_rvb_${SA_KV260_CONFIG}_compiled ;;
             *) echo "RVB_GEN: trace or compiled" >&2; exit 2 ;; esac
[ -d "$SRC" ] || { echo "no dispatch sources in $SRC (a 2-layer SmolLM2 export compiled for the overlay)" >&2; exit 1; }
rm -rf "$D"
mkdir -p "$D/cases"
# (dispatch number: RMSNorm reduction, quantization, q projection, q.K^T, p.V, up projection)
for n in 3 6 7 16 19 28; do
  c=smol_$n
  "$SA_PY" -u "$SA_COMPILER/tests/rv_dispatch.py" "$SRC/module_main\$async_dispatch_$n.mlir" --t "$T" \
    --out "$SA_REPO/build/$W/$c" --code "$GEN" | grep -v Warning
  make -s -C "$SA_REPO/firmware/rvb" bundle RVB="$SA_REPO/build/$W/$c" OUT="$D/cases/$c" >/dev/null
  cp "$SA_REPO/build/$W/$c"/{rvb_ddr.bin,rvb_expect.bin,rvb_cmp.txt,rvb_list.json} "$D/cases/$c/"
  rm -f "$D/cases/$c"/*.o
  rm -f "$D/cases/$c"/*.elf
done
cp "$SA_COMPILER/tests/board_rvb.py" "$SA_COMPILER/tests/board_run.py" "$SA_REPO/driver/pynq_matmul.py" "$D/"
stage_overlay "$D"
du -sh "$D"
ls "$D" "$D/cases"

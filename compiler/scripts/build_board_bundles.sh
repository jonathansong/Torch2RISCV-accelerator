#!/usr/bin/env bash
# The board bundles of an overlay's acceptance (2026-10-15: the _pcpi overlay, GEMV over
# PCPI; docs/riscv_backend_design.md §4.1): Gemma at 512, Qwen3 and Llama at 2048
# (deploy_longctx.sh; SmolLM2 at 2048 too) and the RISC-V backend test (deploy_rvb.sh), one after another.
# A model's <model>_rv step adds the RISC-V paths to its bundle: the model compiled again
# with --iree-sa-riscv (S4, deploy_rvembed.sh), plan B (deploy_rvsched.sh) and R5 (deploy_r5.sh).
# Each step runs in its own memory-capped scope (MemoryMax 10G: only the step is
# killed beyond it) with its own log; a failing step does not stop the next ones; a
# rerun skips the finished steps.
#   compiler/scripts/build_board_bundles.sh [step ...]   steps: gemma qwen3 llama smollm2 rvb and
#     gemma_rv qwen3_rv llama_rv smollm2_rv (default: gemma qwen3 llama rvb)
#   REDO=1 compiler/scripts/build_board_bundles.sh qwen3  rebuild a finished step
#   SA_KV260_CONFIG (default d16_100mhz_w128_p6_gemv_np4_pcpi), LC_SUFFIX (default _pcpi)
# Logs and markers: build/bundles<LC_SUFFIX>/<step>.log, .done / .failed; bundles:
# build/deploy_hfgen_<model>_lc<len><LC_SUFFIX>, build/deploy_rvb_<config>.
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd)
export SA_KV260_CONFIG=${SA_KV260_CONFIG:-d16_100mhz_w128_p6_gemv_np4_pcpi}
export LC_SUFFIX=${LC_SUFFIX:-_pcpi}
L=$R/build/bundles$LC_SUFFIX
mkdir -p "$L"
BIT=$R/boards/kv260/build/output/$SA_KV260_CONFIG/picorv32.bit
[ -f "$BIT" ] || { echo "the overlay $BIT is missing (boards/kv260/scripts/build_bitstream.sh ... -tag ${LC_SUFFIX#_})" >&2; exit 1; }
# --rv <model>: a <model>_rv step (run inside the step's memory-capped scope)
if [ "${1:-}" = --rv ]; then
  set -e
  case $2 in gemma) len=512 ;; *) len=2048 ;; esac
  tag=$2_lc$len$LC_SUFFIX D=$R/build/deploy_hfgen_$tag E=$R/build/hfgen/${tag}_rv
  rm -rf "$E"; mkdir -p "$E"
  for f in qllama.mlir qllama.irpa abi.txt compile_flags.txt; do ln -s "$R/build/hfgen/$tag/$f" "$E/$f"; done
  SA_COMPILE_FLAGS=--iree-sa-riscv "$R/compiler/scripts/compile_sa.sh" "$E" 2> "$E/compile.log"
  "$R/compiler/scripts/deploy_rvsched.sh" "$D"
  "$R/compiler/scripts/deploy_rvembed.sh" "$D" "$E"
  "$R/compiler/scripts/deploy_r5.sh" "$D"
  exit 0
fi
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(gemma qwen3 llama rvb)

cmd_of() {
  case $1 in
    gemma|qwen3|llama) echo "$R/compiler/scripts/deploy_longctx.sh $1" ;;
    smollm2)           echo "$R/compiler/scripts/deploy_longctx.sh $1" ;;
    rvb)               echo "$R/compiler/scripts/deploy_rvb.sh" ;;
    *_rv)              echo "$R/compiler/scripts/build_board_bundles.sh --rv ${1%_rv}" ;;
    *) return 1 ;;
  esac
}

for s in "${STEPS[@]}"; do
  c=$(cmd_of "$s") || { echo "unknown step $s (gemma qwen3 llama smollm2 rvb, <model>_rv)" >&2; continue; }
  if [ -f "$L/$s.done" ] && [ -z "${REDO:-}" ]; then
    echo "[$s] done before ($(cat "$L/$s.done")): skipped (REDO=1 to rebuild)"
    continue
  fi
  rm -f "$L/$s.done" "$L/$s.failed"
  echo "[$s] start $(date '+%F %T'): $c  (log $L/$s.log)"
  t0=$SECONDS
  systemd-run --user --scope --quiet -p MemoryMax=10G -p MemorySwapMax=0 \
    bash -c "$c" > "$L/$s.log" 2>&1
  rc=$?
  dt=$(( (SECONDS - t0) / 60 ))
  if [ $rc -eq 0 ]; then
    echo "$(date '+%F %T'), $dt min" > "$L/$s.done"
    echo "[$s] PASS in $dt min"
  else
    echo "$(date '+%F %T'), exit $rc after $dt min" > "$L/$s.failed"
    echo "[$s] FAIL (exit $rc after $dt min), last lines:"
    tail -5 "$L/$s.log" | sed 's/^/    /'
  fi
done
echo "summary:"
for s in "${STEPS[@]}"; do
  st=pending
  [ -f "$L/$s.done" ] && st="done ($(cat "$L/$s.done"))"
  [ -f "$L/$s.failed" ] && st="FAILED ($(cat "$L/$s.failed"))"
  echo "  $s: $st"
done

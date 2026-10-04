# Board selection for the deploy scripts (sourced after env.sh):
#   SA_BOARD=pynq-z1 (default)  armv7 runtime, RISCV-on-PYNQ-Z1/bitstreams/l2
#   SA_BOARD=kv260              aarch64 runtime, KV260/build/output (KV260/scripts/build_bitstream.sh)
#   SA_BIT_DIR=<dir>            another directory with picorv32.bit / .hwh
# The driver and the launcher read the board's addresses and clock from the
# .hwh (driver/pynq_matmul.py overlay_info), so a bundle is the same apart
# from the runtime binaries and the overlay.
#   stage_runtime <dir> [program ...]  build the runtime for the board, copy stripped programs
#                                      (default sa-llm-run) into <dir>
#   stage_overlay <dir>                copy picorv32.bit / .hwh into <dir>
SA_BOARD=${SA_BOARD:-pynq-z1}
case "$SA_BOARD" in
  pynq-z1) SA_RT_ARCH=armv7;   SA_BIT_DIR=${SA_BIT_DIR:-$SA_REPO/RISCV-on-PYNQ-Z1/bitstreams/l2} ;;
  kv260)   SA_RT_ARCH=aarch64; SA_BIT_DIR=${SA_BIT_DIR:-$SA_REPO/KV260/build/output} ;;
  *) echo "SA_BOARD must be pynq-z1 or kv260 (got $SA_BOARD)" >&2; exit 2 ;;
esac
SA_RT_BUILD=$SA_REPO/build/iree/build-sa-$SA_RT_ARCH

stage_runtime() {
  local d=$1
  shift
  local progs=("$@")
  [ ${#progs[@]} -eq 0 ] && progs=(sa-llm-run)
  "$SA_COMPILER/scripts/build_sa_runtime.sh" "$SA_RT_ARCH" | tail -1
  for p in "${progs[@]}"; do
    case $p in
      iree-run-module) llvm-strip-18 -o "$d/$p" "$SA_RT_BUILD/tools/iree-run-module" ;;
      *)               llvm-strip-18 -o "$d/$p" "$SA_RT_BUILD/runtime/plugins/hal/drivers/sa/$p" ;;
    esac
  done
}

stage_overlay() {
  local d=$1
  for f in picorv32.bit picorv32.hwh; do
    [ -f "$SA_BIT_DIR/$f" ] || { echo "$SA_BIT_DIR/$f missing ($SA_BOARD overlay)" >&2; exit 1; }
    cp "$SA_BIT_DIR/$f" "$d/"
  done
}

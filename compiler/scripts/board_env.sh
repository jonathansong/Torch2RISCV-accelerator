# Board selection for the deploy scripts (sourced after env.sh):
#   SA_BOARD=kv260 (default)    aarch64 runtime, boards/kv260/build/output/$SA_KV260_CONFIG
#                               (boards/kv260/scripts/build_bitstream.sh; default d8_100mhz = K1b, d8_50mhz = K1a)
#   (the frozen PYNQ-Z1, armv7: the pynq-z1 branch)
#   SA_BIT_DIR=<dir>            another directory with picorv32.bit / .hwh
#   SA_D                        the overlay's array size, from the configuration's name (d16_100mhz -> 16;
#                               default 8): the compiler's --iree-sa-d (compile_sa.sh), the sim's and the
#                               references' D in the tests; SA_DSUF ("" at D = 8, else _d<D>) keeps the
#                               build directories of other D apart (the golden corpus reads the D = 8 ones)
#   SA_GEMV_PORTS               the overlay's GEMV unit (K2b), from the configuration's name: _gemv -> 1,
#                               _gemv_np2 -> 2, _gemv_np4 -> 4 (default 0); compile_sa.sh passes --iree-sa-gemv-ports, and
#                               SA_DSUF gets _gemv / _gemv2 / _gemv4 (GEMV builds apart from the EX-path ones and
#                               by port count: the descriptors differ)
# The driver and the launcher read the board's addresses and clock from the
# .hwh (driver/pynq_matmul.py overlay_info), so a bundle is the same apart
# from the runtime binaries and the overlay.
#   stage_runtime <dir> [program ...]  build the runtime for the board, copy stripped programs
#                                      (default sa-llm-run) into <dir>
#   stage_overlay <dir>                copy picorv32.bit / .hwh into <dir>, then stage_version
#   stage_version <dir>                write <dir>/VERSION: what the board results trace back to
#                                      (generality plan WP0; board_run.py files the results by it)
SA_BOARD=${SA_BOARD:-kv260}
case "$SA_BOARD" in
  kv260)   SA_RT_ARCH=aarch64; SA_KV260_CONFIG=${SA_KV260_CONFIG:-d8_100mhz}
           SA_BIT_DIR=${SA_BIT_DIR:-$SA_REPO/boards/kv260/build/output/$SA_KV260_CONFIG} ;;
  *) echo "SA_BOARD must be kv260 (got $SA_BOARD; the PYNQ-Z1: pynq-z1 branch)" >&2; exit 2 ;;
esac
SA_RT_BUILD=$SA_REPO/build/iree/build-sa-$SA_RT_ARCH
if [ -z "${SA_D:-}" ]; then
  case "${SA_KV260_CONFIG:-}" in d[0-9]*_*) SA_D=${SA_KV260_CONFIG%%_*}; SA_D=${SA_D#d} ;; *) SA_D=8 ;; esac
fi
export SA_D
if [ -z "${SA_GEMV_PORTS:-}" ]; then
  case "${SA_KV260_CONFIG:-}" in *_gemv_np4*) SA_GEMV_PORTS=4 ;; *_gemv_np2*) SA_GEMV_PORTS=2 ;; *_gemv*) SA_GEMV_PORTS=1 ;; *) SA_GEMV_PORTS=0 ;; esac
fi
export SA_GEMV_PORTS
SA_GSUF=_gemv$( [ "$SA_GEMV_PORTS" -le 1 ] || echo "$SA_GEMV_PORTS" )   # GEMV builds: _gemv, _gemv2
SA_DSUF=$( [ "$SA_D" = 8 ] || echo "_d$SA_D" )$( [ "$SA_GEMV_PORTS" = 0 ] || echo "$SA_GSUF" )

# seed_export <d8 export dir> <dir>: the exports do not depend on D (the attention
# length is dynamic), so another D's build directory starts from a copy of the
# D = 8 export (qllama.mlir / .irpa, prompt, KV scales); compile output differs
seed_export() {
  local src=$1 dst=$2
  [ "$src" = "$dst" ] && return 0
  [ -f "$dst/qllama.mlir" ] && return 0
  [ -f "$src/qllama.mlir" ] || { echo "seed_export: no export in $src" >&2; exit 1; }
  mkdir -p "$dst"
  for f in qllama.mlir qllama.irpa prompt.npy kv_scales.npy; do
    [ -f "$src/$f" ] && cp "$src/$f" "$dst/"
  done
  return 0
}

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
  stage_version "$d"
}

# VERSION: commit (dirty: uncommitted changes under compiler / rtl / firmware / driver),
# overlay (configuration, bitstream hash, its build_info.txt: commit, Vivado, WNS), the
# compiler (IREE revision, D, GEMV ports), the model's build (a test_hf_generic bundle's
# build_info.txt), the run parameters (prompt token IDs, sa_args.txt, board.txt). The
# results directory: results/<commit>/<test>/<overlay configuration>/<mode>/ (board_run.py).
stage_version() {
  local d=$1 bi
  {
    echo "commit $(git -C "$SA_REPO" rev-parse HEAD)$(git -C "$SA_REPO" diff --quiet HEAD -- compiler rtl firmware driver || echo ' (dirty)')"
    echo "staged $(date -Iseconds)"
    echo "test $(basename "$d" | sed 's/^deploy_//')"
    echo "bitstream $(sha256sum "$d/picorv32.bit" | cut -c1-16) (${SA_BIT_DIR#$SA_REPO/})"
    echo "board $SA_BOARD (runtime $SA_RT_ARCH), D = $SA_D, GEMV ports $SA_GEMV_PORTS"
    echo "iree $IREE_REV"
    bi=$SA_BIT_DIR/build_info.txt
    [ -f "$bi" ] && sed 's/^/overlay /' "$bi"
    [ -f "$d/build_info.txt" ] && sed 's/^/model /' "$d/build_info.txt"
    [ -f "$d/prompt.npy" ] && echo "prompt $("$SA_PY" -c 'import numpy, os, sys
p = numpy.load(sys.argv[1] + "/prompt.npy"); e = sys.argv[1] + "/expected_tokens.npy"
print(",".join(map(str, p)) + (f" (generate {len(numpy.load(e)) - len(p)})" if os.path.exists(e) else ""))' "$d")"
    [ -f "$d/sa_args.txt" ] && echo "sa_args $(cat "$d/sa_args.txt")"
    [ -f "$d/board.txt" ] && echo "window $(cat "$d/board.txt")"
  } > "$d/VERSION"
  cp "$SA_COMPILER/tests/board_run.py" "$d/"
}

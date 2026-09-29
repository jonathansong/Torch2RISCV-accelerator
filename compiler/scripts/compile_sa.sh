#!/usr/bin/env bash
# Compiles an exported model (qllama.mlir + qllama.irpa) for the sa device:
# parameters imported, linear weights packed and evaluated at compile time,
# the packed parameters exported (docs/iree_compiler_plan.md §6.3).
#   compiler/scripts/compile_sa.sh <dir with qllama.mlir / .irpa> [extra iree-compile flags]
# -> <dir>/sa.vmfb, <dir>/sa_packed.irpa, <dir>/sa_sources/ (dispatch sources)
# Aggressive dispatch fusion is on (C4: 242 -> 200 dispatches per token).
# SA_COMPILE_FLAGS: extra iree-compile flags.
# SA_HOST_FALLBACK=0: no host fallback (default 1: every dispatch also gets a
# VMVX variant; one the sa backend cannot compile runs on the ARM host, plan
# §8.15; IREE's executable linking is off, it would merge the VMVX variants of
# executables that keep their sa ones).
set -euo pipefail
source "$(dirname "$0")/../env.sh"
dir=$1; shift
mkdir -p "$dir/sa_sources"
"$IREE_BUILD/tools/iree-compile" "$dir/qllama.mlir" --iree-hal-target-device=sa \
  --iree-opt-const-expr-max-size-increase-threshold=0 \
  --iree-parameter-import=model="$dir/qllama.irpa" --iree-parameter-import-maximum-size=4294967295 \
  --iree-parameter-export=model="$dir/sa_packed.irpa" --iree-parameter-export-minimum-size=256 \
  --iree-dispatch-creation-enable-aggressive-fusion \
  $( [ "${SA_HOST_FALLBACK:-1}" = 1 ] && echo --iree-sa-host-fallback --iree-hal-link-executables=false ) \
  --iree-hal-dump-executable-sources-to="$dir/sa_sources" ${SA_COMPILE_FLAGS:-} "$@" -o "$dir/sa.vmfb"

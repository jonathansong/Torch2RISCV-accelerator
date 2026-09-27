#!/usr/bin/env bash
# Compiles an exported model (qllama.mlir + qllama.irpa) for the sa device:
# parameters imported, linear weights packed and evaluated at compile time,
# the packed parameters exported (docs/iree_compiler_plan.md §6.3).
#   compiler/scripts/compile_sa.sh <dir with qllama.mlir / .irpa> [extra iree-compile flags]
# -> <dir>/sa.vmfb, <dir>/sa_packed.irpa, <dir>/sa_sources/ (dispatch sources)
set -euo pipefail
source "$(dirname "$0")/../env.sh"
dir=$1; shift
mkdir -p "$dir/sa_sources"
"$IREE_BUILD/tools/iree-compile" "$dir/qllama.mlir" --iree-hal-target-device=sa \
  --iree-opt-const-expr-max-size-increase-threshold=0 \
  --iree-parameter-import=model="$dir/qllama.irpa" --iree-parameter-import-maximum-size=4294967295 \
  --iree-parameter-export=model="$dir/sa_packed.irpa" --iree-parameter-export-minimum-size=256 \
  --iree-hal-dump-executable-sources-to="$dir/sa_sources" "$@" -o "$dir/sa.vmfb"

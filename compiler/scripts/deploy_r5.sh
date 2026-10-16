#!/usr/bin/env bash
# Adds R5 of the RISC-V backend (docs/riscv_backend_design.md §5, model C) to
# staged hfgen bundles with plan B (deploy_rvsched.sh): r5/, the decode loop's
# block argmax as a kernel IREE's llvm-cpu compiled for the core
# (compiler/tests/r5_kernel.py, n = the bundle's vocab / argmax.vmfb's block),
# rt_fw.bin version 6, the aarch64 sa-llm-run (SA_R5), board_rvsched.py.
#   compiler/scripts/deploy_r5.sh <deploy dir>...
#   board: python3 board_run.py rvsched --r5
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source "$(dirname "$0")/board_env.sh"
make -C "$SA_REPO/firmware/rt" >/dev/null
"$SA_COMPILER/scripts/build_sa_runtime.sh" "$SA_RT_ARCH" | tail -1
for D in "$@"; do
  [ -f "$D/argmax.vmfb" ] || { echo "$D: no argmax.vmfb (deploy_rvsched.sh first)" >&2; exit 1; }
  V=$("$SA_PY" -c "import numpy as np; print(np.load('$D/expected_logits.npy', mmap_mode='r').shape[1])")
  "$SA_PY" "$SA_COMPILER/tests/r5_kernel.py" --vocab "$V" --out "$D/r5"
  rm -rf "$D/r5/build"
  cp "$SA_REPO/firmware/rt/rt_fw.bin" "$SA_COMPILER/tests/board_rvsched.py" "$SA_COMPILER/tests/board_run.py" "$D/"
  llvm-strip-18 -o "$D/sa-llm-run" "$SA_RT_BUILD/runtime/plugins/hal/drivers/sa/sa-llm-run"
  sed -i '/^r5 /d' "$D/VERSION"
  echo "r5 $(git -C "$SA_REPO" rev-parse --short HEAD)$(git -C "$SA_REPO" diff --quiet HEAD -- compiler firmware || echo ' (dirty)'): r5/ ($(cat "$D/r5/r5.txt")), rt_fw.bin v6, sa-llm-run" >> "$D/VERSION"
  echo "to the board: scp -r $D/{r5,rt_fw.bin,sa-llm-run,board_rvsched.py,board_run.py,VERSION} ubuntu@<board>:~/$(basename "$D")/"
done

#!/usr/bin/env bash
# Stages the C2 board test (docs/iree_compiler_plan.md §10) in build/deploy_c2:
# the host test (compile + sim run) writes the compiled cases, plus the board's (aarch64)
# iree-run-module with the sa driver, the C1 launcher and the board's overlay.
#   compiler/scripts/deploy_c2.sh
#   scp -r build/deploy_c2/* xilinx@<board>:/home/xilinx/c2/      (copy the contents)
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source "$(dirname "$0")/board_env.sh"            # SA_BOARD=kv260 (the default; the PYNQ-Z1: pynq-z1 branch)
D=$SA_REPO/build/deploy_c2
rm -rf "$D" && mkdir -p "$D"
"$SA_PY" "$SA_COMPILER/tests/test_c2.py" --board-bundle "$D"
stage_runtime "$D" iree-run-module
cp "$SA_COMPILER/tests/board_c2.py" "$SA_COMPILER/runtime/test/board_launcher.py" "$SA_REPO/driver/pynq_matmul.py" \
   "$SA_REPO/firmware/rt/rt_fw.bin" "$D/"
stage_overlay "$D"
ls -la "$D"

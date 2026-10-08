#!/usr/bin/env bash
# Stages the C3 board test (docs/iree_compiler_plan.md §6.8) in build/deploy_c3:
# the host test (export, compile, per-dispatch check, sim run) writes the module
# and the expected run; plus the board's sa-llm-run (armv7 / aarch64, SA_BOARD), the C1 launcher, the board's overlay.
#   compiler/scripts/deploy_c3.sh
#   scp build/deploy_c3/* xilinx@<board>:/home/xilinx/c3/
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source "$(dirname "$0")/board_env.sh"            # SA_BOARD=kv260 (default) | pynq-z1
D=$SA_REPO/build/deploy_c3
rm -rf "$D" && mkdir -p "$D"
"$SA_PY" "$SA_COMPILER/tests/test_c3.py" --board-bundle "$D" "$@"
stage_runtime "$D"
cp "$SA_COMPILER/tests/board_llm.py" "$SA_COMPILER/tests/board_generate.py" "$SA_COMPILER/runtime/test/board_launcher.py" "$SA_REPO/driver/pynq_matmul.py" \
   "$SA_REPO/firmware/rt/rt_fw.bin" "$D/"
stage_overlay "$D"
ls -la "$D"

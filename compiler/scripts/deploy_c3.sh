#!/usr/bin/env bash
# Stages the C3 board test (docs/iree_compiler_plan.md §6.8) in build/deploy_c3:
# the host test (export, compile, per-dispatch check, sim run) writes the module
# and the expected run; plus the armv7 sa-llm-run, the C1 launcher, the L2 overlay.
#   compiler/scripts/deploy_c3.sh
#   scp build/deploy_c3/* xilinx@<board>:/home/xilinx/c3/
set -euo pipefail
source "$(dirname "$0")/../env.sh"
D=$SA_REPO/build/deploy_c3
rm -rf "$D" && mkdir -p "$D"
"$SA_PY" "$SA_COMPILER/tests/test_c3.py" --board-bundle "$D" "$@"
"$SA_COMPILER/scripts/build_sa_runtime.sh" armv7 | tail -1
llvm-strip-18 -o "$D/sa-llm-run" "$SA_REPO/build/iree/build-sa-armv7/runtime/plugins/hal/drivers/sa/sa-llm-run"
cp "$SA_COMPILER/tests/board_llm.py" "$SA_COMPILER/tests/board_generate.py" "$SA_COMPILER/runtime/test/board_launcher.py" "$SA_REPO/driver/pynq_matmul.py" \
   "$SA_REPO/firmware/rt/rt_fw.bin" "$D/"
cp "$SA_REPO/RISCV-on-PYNQ-Z1/bitstreams/l2/picorv32.bit" "$SA_REPO/RISCV-on-PYNQ-Z1/bitstreams/l2/picorv32.hwh" "$D/"
ls -la "$D"

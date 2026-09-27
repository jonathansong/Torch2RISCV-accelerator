#!/usr/bin/env bash
# Stages the C2 board test (docs/iree_compiler_plan.md §10) in build/deploy_c2:
# the host test (compile + sim run) writes the compiled cases, plus the armv7
# iree-run-module with the sa driver, the C1 launcher and the L2 overlay.
#   compiler/scripts/deploy_c2.sh
#   scp -r build/deploy_c2/* xilinx@<board>:/home/xilinx/c2/      (copy the contents)
set -euo pipefail
source "$(dirname "$0")/../env.sh"
D=$SA_REPO/build/deploy_c2
rm -rf "$D" && mkdir -p "$D"
"$SA_PY" "$SA_COMPILER/tests/test_c2.py" --board-bundle "$D"
"$SA_COMPILER/scripts/build_sa_runtime.sh" armv7 | tail -1
llvm-strip-18 -o "$D/iree-run-module" "$SA_REPO/build/iree/build-sa-armv7/tools/iree-run-module"
cp "$SA_COMPILER/tests/board_c2.py" "$SA_COMPILER/runtime/test/board_launcher.py" "$SA_REPO/driver/pynq_matmul.py" \
   "$SA_REPO/firmware/rt/rt_fw.bin" "$D/"
cp "$SA_REPO/RISCV-on-PYNQ-Z1/bitstreams/l2/picorv32.bit" "$SA_REPO/RISCV-on-PYNQ-Z1/bitstreams/l2/picorv32.hwh" "$D/"
ls -la "$D"

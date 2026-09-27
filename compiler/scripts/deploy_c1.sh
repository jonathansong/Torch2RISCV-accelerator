#!/usr/bin/env bash
# Stages the C1 board test (docs/iree_compiler_plan.md §5.6) in build/deploy_c1:
#   compiler/scripts/deploy_c1.sh
#   scp build/deploy_c1/* xilinx@<board>:/home/xilinx/c1/      (copy the files, not the directory)
#   scp -r build/deploy_c1/test_d8 xilinx@<board>:/home/xilinx/c1/
set -euo pipefail
source "$(dirname "$0")/../env.sh"
"$SA_COMPILER/scripts/build_sa_runtime.sh" armv7 | tail -1
D=$SA_REPO/build/deploy_c1
rm -rf "$D" && mkdir -p "$D"
"$SA_PY" "$SA_COMPILER/runtime/tools/make_test_exec.py" --d 8 --out "$D/test_d8"
B=$SA_REPO/build/iree/build-sa-armv7
llvm-strip-18 -o "$D/sa_hal_test" "$B/runtime/plugins/hal/drivers/sa/sa_hal_test"
llvm-strip-18 -o "$D/iree-run-module" "$B/tools/iree-run-module"
cp "$SA_COMPILER/runtime/test/board_launcher.py" "$SA_REPO/driver/pynq_matmul.py" "$SA_REPO/firmware/rt/rt_fw.bin" "$D/"
cp "$SA_REPO/RISCV-on-PYNQ-Z1/bitstreams/l2/picorv32.bit" "$SA_REPO/RISCV-on-PYNQ-Z1/bitstreams/l2/picorv32.hwh" "$D/"
ls -la "$D" "$D/test_d8"

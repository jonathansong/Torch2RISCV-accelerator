#!/usr/bin/env bash
# Stages the C5.5 board test (docs/iree_compiler_plan.md §8.7) in build/deploy_c55:
# a HuggingFace decoder exported by compiler/frontend/export_hf.py (default
# SmolLM2-135M in build/c55/smollm2); the host test (compile, sim run) writes the
# module and the expected run (the sim's); plus the armv7 sa-llm-run, the C1
# launcher, the L2 overlay.
#   compiler/scripts/deploy_c55.sh [test_c55.py arguments]
#   scp build/deploy_c55/* xilinx@<board>:/home/xilinx/c55/
set -euo pipefail
source "$(dirname "$0")/../env.sh"
D=$SA_REPO/build/deploy_c55
rm -rf "$D" && mkdir -p "$D"
"$SA_PY" "$SA_COMPILER/tests/test_c55.py" --model "$SA_REPO/build/llm_cache/SmolLM2-135M" \
  --out "$SA_REPO/build/c55/smollm2" --board-bundle "$D" "$@"
"$SA_COMPILER/scripts/build_sa_runtime.sh" armv7 | tail -1
llvm-strip-18 -o "$D/sa-llm-run" "$SA_REPO/build/iree/build-sa-armv7/runtime/plugins/hal/drivers/sa/sa-llm-run"
cp "$SA_COMPILER/tests/board_llm.py" "$SA_COMPILER/runtime/test/board_launcher.py" "$SA_REPO/driver/pynq_matmul.py" \
   "$SA_REPO/firmware/rt/rt_fw.bin" "$D/"
cp "$SA_REPO/RISCV-on-PYNQ-Z1/bitstreams/l2/picorv32.bit" "$SA_REPO/RISCV-on-PYNQ-Z1/bitstreams/l2/picorv32.hwh" "$D/"
ls -la "$D"

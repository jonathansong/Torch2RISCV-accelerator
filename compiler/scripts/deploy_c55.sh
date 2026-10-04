#!/usr/bin/env bash
# Stages the C5.5 board test (docs/iree_compiler_plan.md §8.7) in build/deploy_c55:
# a HuggingFace decoder exported by compiler/frontend/export_hf.py (default
# SmolLM2-135M in build/c55/smollm2); the host test (compile, sim run) writes the
# module and the expected run (the sim's); plus the board's sa-llm-run (armv7 / aarch64, SA_BOARD), the C1
# launcher, the board's overlay.
#   compiler/scripts/deploy_c55.sh [test_c55.py arguments]
#   scp build/deploy_c55/* xilinx@<board>:/home/xilinx/c55/
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source "$(dirname "$0")/board_env.sh"            # SA_BOARD=pynq-z1 (default) | kv260
D=$SA_REPO/build/deploy_c55
rm -rf "$D" && mkdir -p "$D"
"$SA_PY" "$SA_COMPILER/tests/test_c55.py" --model "$SA_REPO/build/llm_cache/SmolLM2-135M" \
  --out "$SA_REPO/build/c55/smollm2" --board-bundle "$D" "$@"
stage_runtime "$D"
cp "$SA_COMPILER/tests/board_llm.py" "$SA_COMPILER/tests/board_generate.py" "$SA_COMPILER/runtime/test/board_launcher.py" "$SA_REPO/driver/pynq_matmul.py" \
   "$SA_REPO/firmware/rt/rt_fw.bin" "$D/"
stage_overlay "$D"
ls -la "$D"

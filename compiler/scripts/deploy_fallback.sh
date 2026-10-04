#!/usr/bin/env bash
# Stages the host-fallback board test (docs/iree_compiler_plan.md §8.15):
# stories15M with prefill (M = 8, on the accelerator) and every decode linear
# layer forced to the ARM host (VMVX); board_llm.py compares the logits bit
# for bit with the sim (itself bit-exact with the accelerator-only build) and
# prints the split (descriptor lists / host dispatches).
#   compiler/scripts/deploy_fallback.sh   -> build/deploy_fallback
#   board: python3 board_llm.py
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source "$(dirname "$0")/board_env.sh"            # SA_BOARD=pynq-z1 (default) | kv260
O=$SA_REPO/build/fallback/stories_hostlin
D=$SA_REPO/build/deploy_fallback
rm -rf "$O" "$D" && mkdir -p "$O" "$D"
cp "$SA_REPO/build/c6p/stories_m8/qllama.mlir" "$SA_REPO/build/c6p/stories_m8/qllama.irpa" "$O/"
SA_HOST_FALLBACK=1 "$SA_PY" "$SA_COMPILER/tests/test_c6p.py" --out "$O" --prefill 8 \
  --flags=--iree-sa-host-dispatches=matvec --board-bundle "$D" --board-generate 8
stage_runtime "$D"
cp "$SA_COMPILER/tests/board_llm.py" "$SA_COMPILER/tests/board_generate.py" "$SA_COMPILER/runtime/test/board_launcher.py" \
   "$SA_REPO/driver/pynq_matmul.py" "$SA_REPO/firmware/rt/rt_fw.bin" "$D/"
stage_overlay "$D"
ls -la "$D"

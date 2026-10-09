#!/usr/bin/env bash
# Stages the C6.P board test (docs/iree_compiler_plan.md §8.13): prefill + decode.
#   compiler/scripts/deploy_c6p.sh stories|smollm2|qwen3 [M]      (qwen3: Qwen3-0.6B, K4a; window MB, default 640)
# exports the model with prefill chunks of M (default 8) into build/c6p/<model>_m<M>,
# compiles it, checks it on the sim (test_c6p.py) and stages build/deploy_c6p_<model>_m<M>;
# board: python3 board_llm.py [--decode-only]; python3 board_generate.py
#   scp build/deploy_c6p_<model>_m<M>/* xilinx@<board>:/home/xilinx/c6p/
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source "$(dirname "$0")/board_env.sh"            # SA_BOARD=kv260 (the default; the PYNQ-Z1: pynq-z1 branch)
model=${1:?stories, smollm2 or qwen3}
M=${2:-8}
O=$SA_REPO/build/c6p/${model}_m$M$SA_DSUF
D=$SA_REPO/build/deploy_c6p_${model}_m$M
rm -rf "$D" && mkdir -p "$D"
case $model in
  stories)
    "$SA_PY" "$SA_COMPILER/frontend/export.py" --out "$O" --prefill "$M" --d "$SA_D" | tail -2
    "$SA_PY" "$SA_COMPILER/tests/test_c6p.py" --out "$O" --prefill "$M" --check --board-bundle "$D" --board-generate 60 ;;
  smollm2)
    "$SA_PY" "$SA_COMPILER/frontend/export_hf.py" --model "$SA_REPO/build/llm_cache/SmolLM2-135M" --out "$O" \
      --prefill "$M" --d "$SA_D" | tail -2
    "$SA_PY" "$SA_COMPILER/tests/test_c6p.py" --out "$O" --prefill "$M" --check --mb 144 \
      --model "$SA_REPO/build/llm_cache/SmolLM2-135M" --board-bundle "$D" --board-generate 24 ;;
  qwen3)                      # (the export takes ~15 min: kept when there; rm -rf $O to redo it)
    [ -f "$O/qllama.irpa" ] || "$SA_PY" "$SA_COMPILER/frontend/export_hf.py" --model "$SA_REPO/build/llm_cache/Qwen3-0.6B" \
      --out "$O" --prefill "$M" --d "$SA_D" | tail -2
    "$SA_PY" "$SA_COMPILER/tests/test_c6p.py" --out "$O" --prefill "$M" --check --mb "${MB:-640}" \
      --model "$SA_REPO/build/llm_cache/Qwen3-0.6B" --board-bundle "$D" --board-generate 24 ;;
  *) echo "stories, smollm2 or qwen3"; exit 2 ;;
esac
stage_runtime "$D"
cp "$SA_COMPILER/tests/board_llm.py" "$SA_COMPILER/tests/board_generate.py" "$SA_COMPILER/tests/board_profile.py" "$SA_COMPILER/runtime/test/board_launcher.py" \
   "$SA_REPO/driver/pynq_matmul.py" "$SA_REPO/firmware/rt/rt_fw.bin" "$D/"
stage_overlay "$D"
ls -la "$D"

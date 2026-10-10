#!/usr/bin/env bash
# Stages the generic-frontend board test (docs/iree_compiler_plan.md §8.17 item 1):
# SmolLM2-135M exported from the unmodified HF model (W8A8), prefill (M = D, the
# overlay's array size: the prefill micro-kernel needs M a multiple of D) +
# decode; board_llm.py compares the board's logits bit for bit with the sim's
# decode-only run (itself bit-exact with the sim's prefill + decode run).
#   compiler/scripts/deploy_hfgen.sh [build dir, default build/hfgen/smollm2_p<D>]   -> build/deploy_hfgen
#   HFGEN_MODEL=<HF dir under build/llm_cache> HFGEN_TAG=<tag>: another model (e.g. a
#   compressed-tensors W8A8 checkpoint, SmolLM2-135M-W8A8 / w8a8: build/hfgen/smollm2_<tag>_p<D>,
#   build/deploy_hfgen_<tag>; docs/generality_implementation_plan.md WP3)
#   (a build dir with sa.vmfb is reused as it is: the export and compile of a
#   passing test_hf_generic.py run; otherwise exported and compiled here)
#   board: python3 board_llm.py
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source "$(dirname "$0")/board_env.sh"            # SA_BOARD=kv260 (the default; the PYNQ-Z1: pynq-z1 branch)
MODEL=${HFGEN_MODEL:-SmolLM2-135M}
TAG=${HFGEN_TAG:+_$HFGEN_TAG}
O=${1:-$SA_REPO/build/hfgen/smollm2${TAG}_p$SA_D$( [ "$SA_GEMV_PORTS" = 0 ] || echo "$SA_GSUF" )}   # (GEMV overlays: own build)
# (the export fixes M, so it is not seeded from another D's)
SKIP=$([ -f "$O/sa.vmfb" ] && echo --skip-export || true)
D=$SA_REPO/build/deploy_hfgen$TAG
rm -rf "$D" && mkdir -p "$D"
"$SA_PY" -u "$SA_COMPILER/tests/test_hf_generic.py" --model "$SA_REPO/build/llm_cache/$MODEL" --out "$O" \
  $SKIP --quant --rope --attn --prefill "$SA_D" --mb 400 --board-bundle "$D"
stage_runtime "$D"
cp "$SA_COMPILER/tests/board_llm.py" "$SA_COMPILER/tests/board_generate.py" "$SA_COMPILER/runtime/test/board_launcher.py" \
   "$SA_REPO/driver/pynq_matmul.py" "$SA_REPO/firmware/rt/rt_fw.bin" "$D/"
stage_overlay "$D"
ls -la "$D"

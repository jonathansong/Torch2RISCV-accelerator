#!/usr/bin/env bash
# Stages the generic-frontend board test (docs/iree_compiler_plan.md §8.17 item 1):
# SmolLM2-135M exported from the unmodified HF model (W8A8), prefill (M = 8) +
# decode; board_llm.py compares the board's logits bit for bit with the sim's
# decode-only run (itself bit-exact with the sim's prefill + decode run).
#   compiler/scripts/deploy_hfgen.sh [build dir, default build/hfgen/smollm2_p8]   -> build/deploy_hfgen
#   (the build dir must hold the export and compile of a passing test_hf_generic.py run)
#   board: python3 board_llm.py
set -euo pipefail
source "$(dirname "$0")/../env.sh"
O=${1:-$SA_REPO/build/hfgen/smollm2_p8}
D=$SA_REPO/build/deploy_hfgen
rm -rf "$D" && mkdir -p "$D"
"$SA_PY" -u "$SA_COMPILER/tests/test_hf_generic.py" --model "$SA_REPO/build/llm_cache/SmolLM2-135M" --out "$O" \
  --skip-export --quant --rope --attn --prefill 8 --mb 400 --board-bundle "$D"
"$SA_COMPILER/scripts/build_sa_runtime.sh" armv7 | tail -1
llvm-strip-18 -o "$D/sa-llm-run" "$SA_REPO/build/iree/build-sa-armv7/runtime/plugins/hal/drivers/sa/sa-llm-run"
cp "$SA_COMPILER/tests/board_llm.py" "$SA_COMPILER/tests/board_generate.py" "$SA_COMPILER/runtime/test/board_launcher.py" \
   "$SA_REPO/driver/pynq_matmul.py" "$SA_REPO/firmware/rt/rt_fw.bin" "$D/"
cp "$SA_REPO/RISCV-on-PYNQ-Z1/bitstreams/l2/picorv32.bit" "$SA_REPO/RISCV-on-PYNQ-Z1/bitstreams/l2/picorv32.hwh" "$D/"
ls -la "$D"

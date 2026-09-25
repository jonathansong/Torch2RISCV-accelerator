#!/usr/bin/env bash
# LLM step 1 (measurement): stage everything the board needs in build/deploy_llm/.
#
#   notebooks/llm/prepare_llm.sh            (from the repository root; PYTHON must have NumPy)
#   scp -r build/deploy_llm xilinx@<board>:/home/xilinx/llm
#
# Downloads llama2.c (pinned commit) and the TinyStories checkpoints into
# build/llm_cache/, converts them to Q8_0 (quantize_q80.py, for runq.c), and
# group size 32 for all models (arm_baseline.sh's runq_gs32 build), copies the m5 overlay, the firmware and the board scripts. stories110M is
# staged in Q8_0 only: its fp32 file (438 MB) does not fit the board's RAM.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python3}
L2C_COMMIT=350e04fe35433e6d2941dce5a1f53308f87058eb
L2C_URL=https://raw.githubusercontent.com/karpathy/llama2.c/$L2C_COMMIT
HF_URL=https://huggingface.co/karpathy/tinyllamas/resolve/main
CACHE=$ROOT/build/llm_cache
DEST=$ROOT/build/deploy_llm

mkdir -p "$CACHE" "$DEST/llama2c"
fetch() {   # url file
    [ -s "$2" ] || { echo "download $(basename "$2")"; curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"; }
}
for f in run.c runq.c LICENSE; do
    fetch "$L2C_URL/$f" "$CACHE/$f"
    cp "$CACHE/$f" "$DEST/llama2c/"
done
fetch "$L2C_URL/tokenizer.bin" "$CACHE/tokenizer.bin"
cp "$CACHE/tokenizer.bin" "$DEST/"

for m in stories15M stories42M stories110M; do
    fetch "$HF_URL/$m.bin" "$CACHE/$m.bin"
    [ -s "$CACHE/${m}_q80.bin" ] || "$PYTHON" "$ROOT/notebooks/llm/quantize_q80.py" --gs 32 "$CACHE/$m.bin" "$CACHE/${m}_q80.bin"
    cp "$CACHE/${m}_q80.bin" "$DEST/"
done
cp "$CACHE/stories15M.bin" "$CACHE/stories42M.bin" "$DEST/"

cp "$ROOT/RISCV-on-PYNQ-Z1/bitstreams/m5/picorv32.bit" "$ROOT/RISCV-on-PYNQ-Z1/bitstreams/m5/picorv32.hwh" "$DEST/"
cp "$ROOT/firmware/gemm/gemm_fw.bin" "$ROOT/firmware/desc_run/desc_run_fw.bin" "$DEST/"
cp "$ROOT/driver/pynq_matmul.py" "$ROOT/notebooks/llm/arm_baseline.sh" "$ROOT/notebooks/llm/llm_gemv_bench.py" "$DEST/"
du -sh "$DEST"

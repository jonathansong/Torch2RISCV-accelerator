#!/usr/bin/env bash
# Stages a long-context board test (docs/iree_compiler_plan.md §8.22, C6.5 / C6.5b):
# the whole model exported at --max-len (compiled with --iree-sa-max-dynamic of it),
# the usual board_llm.py bundle (the sim's decode-only run of the 40-token
# prompt), plus the teacher-forcing windows of the context length with their fp32
# reference (board_quality.py prep) for board_longctx.py (a prompt of the window's
# length: prefill vs decode only on the device, PPL by position) and board_teacher.py.
#   compiler/scripts/deploy_longctx.sh qwen3|gemma|llama|smollm2  -> build/deploy_hfgen_<model>_lc<max len>
#   LC_MAX_LEN=<n>: another length (default qwen3 / llama / smollm2 2048, gemma 512: its sliding window;
#   llama at 2048 fills the 1280 MiB pool to 5 MiB, mem_preflight.py: at most 2227)
#   LC_MB=<MB>: the sim's device window (default 1100, llama 1400)
#   LC_SUFFIX=<s>: appended to the directories' names (e.g. _pcpi: a new overlay's bundles
#   beside the earlier ones)
#   LC_LAYERS=<n>: a truncated model (a dry run of the whole flow: <tag>_l<n>, ~40 min)
# Runs for hours (the export, the compile, the sim's 40-token runs: Qwen3 ~2.5 h);
# under systemd-run -p MemoryMax=10G. A bundle compiled for more than 512
# positions is slower at short contexts (§8.22): its own directory, the
# regular bundles stay as they are.
#   board: python3 board_longctx.py; python3 board_teacher.py; python3 board_llm.py
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source "$(dirname "$0")/board_env.sh"
case "${1:-}" in
  qwen3) MODEL=Qwen3-0.6B-W8A8 REF=Qwen3-0.6B LEN=${LC_MAX_LEN:-2048} EXTRA= ;;
  gemma) MODEL=gemma-3-270m REF=gemma-3-270m LEN=${LC_MAX_LEN:-512} EXTRA=--prefill-all ;;
  llama) MODEL=Llama-3.2-1B-Instruct-W8A8 REF=Llama-3.2-1B-Instruct LEN=${LC_MAX_LEN:-2048} EXTRA= MB=1400 ;;   # (no prefill_all: it does not fit; board_teacher.py --decode; the sim's window: 1275 MB at 2048)
  smollm2) MODEL=SmolLM2-135M-W8A8 REF=SmolLM2-135M LEN=${LC_MAX_LEN:-2048} EXTRA= ;;
  *) echo "usage: $0 qwen3|gemma|llama|smollm2" >&2; exit 2 ;;
esac
TAG=$1_lc$LEN${LC_LAYERS:+_l$LC_LAYERS}${LC_SUFFIX:-}
LAYERS=${LC_LAYERS:+--layers $LC_LAYERS}
# the KV scales calibrated on at least the context length (§8.22: on 1024 tokens a 2048 window
# lost most past 1024, KL 0.137 in [1536, 2048) against 0.081 calibrated on 2048)
CALIB=$(( LEN > 1024 ? LEN : 1024 ))
O=$SA_REPO/build/hfgen/$TAG
HF=("$SA_PY" -u "$SA_COMPILER/tests/test_hf_generic.py" --model "$SA_REPO/build/llm_cache/$MODEL" --out "$O")
# the export, the compile and the checks each in its own process: Qwen3's
# compile at 2048 peaks at 7.5 GB, next to the exporting process it hit 10 GB
if [ ! -s "$O/sa.vmfb" ]; then
  "${HF[@]}" --export-only --quant --rope --attn --prefill "$SA_D" $EXTRA --max-len "$LEN" \
    --kv-channel k --kv-calib "$SA_REPO/build/llm_cache/wikitext-2-raw-v1-train.txt:$CALIB" $LAYERS
  T0=$SECONDS
  SA_COMPILE_FLAGS=--iree-sa-codegen-report "$SA_COMPILER/scripts/compile_sa.sh" "$O" 2> "$O/compile.log" \
    || { tail -c 3000 "$O/compile.log"; rm -f "$O/sa.vmfb"; exit 1; }
  echo "compiled in $((SECONDS - T0)) s: $(grep -c ': ok$' "$O/compile.log" || true) executables for the accelerator," \
    "$(grep -c ': on the host' "$O/compile.log" || true) for the host"
fi
# no per-step host copies of the KV caches (host_copies.py; §8.22: the packed V cache's strided write cost 640 ms a step)
"$SA_PY" "$SA_COMPILER/tests/host_copies.py" "$O"
"${HF[@]}" --skip-export --quant --rope --attn --prefill "$SA_D" $EXTRA --max-len "$LEN" --mb "${LC_MB:-${MB:-1100}}" \
  --kv-channel k --kv-calib "$SA_REPO/build/llm_cache/wikitext-2-raw-v1-train.txt:$CALIB" --board-bundle "${O}_bundle" $LAYERS
HFGEN_TAG=$TAG HFGEN_BUNDLE=${O}_bundle "$SA_COMPILER/scripts/deploy_hfgen.sh"
D=$SA_REPO/build/deploy_hfgen_$TAG
# the teacher-forcing windows: two of the context length (WikiText-2 test, from token 0)
"$SA_PY" -u "$SA_COMPILER/tests/board_quality.py" prep --model "$SA_REPO/build/llm_cache/$REF" --bundle "$D" \
  --tokens $((2 * LEN)) --window "$LEN" $LAYERS
if [ "$1" = gemma ]; then                        # (prefill_all's logits chunks: the window of the qs bundle, §12.4)
  read -r -a B < "$D/board.txt"
  [ "${B[-1]}" -ge 640 ] || { B[-1]=640; echo "${B[*]}" > "$D/board.txt"; }
fi
ls -la "$D"

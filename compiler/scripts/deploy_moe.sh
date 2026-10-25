#!/usr/bin/env bash
# Stages the native MoE board test (docs/moe_native_lowering_design.md S7): a W8A8
# Mixtral bundle (build/moe/native/<name>: test_hf_moe.py --export, compile_sa.sh with
# --iree-sa-form-moe, and test_moe_sim.py's passing sim run, whose decode-only logits are
# the reference in <bundle>/s6) with the aarch64 sa-llm-run, rt_fw.bin, the overlay and
# board_llm.py. The board runs prefill + decode and compares its logits bit for bit with
# the sim's decode-only run (itself byte-equal with the sim's prefill run); the Top-2
# runs as the executable's embedded RISC-V code (rv_only), the rest as descriptor lists.
#   compiler/scripts/deploy_moe.sh [bundle, default build/moe/native/tm]   -> build/deploy_moe_<name>
#   board: python3 board_run.py llm [--decode-only]
set -euo pipefail
export SA_KV260_CONFIG=${SA_KV260_CONFIG:-d16_100mhz_w128_p6_gemv_np4_pcpi}
source "$(dirname "$0")/../env.sh"
source "$(dirname "$0")/board_env.sh"
B=$(realpath "${1:-$SA_REPO/build/moe/native/tm}")
for f in sa.vmfb sa_packed.irpa s6/prompt.npy s6/expected_tokens.npy s6/expected_logits.npy; do
  [ -f "$B/$f" ] || { echo "$B/$f missing (compile_sa.sh, then compiler/tests/test_moe_sim.py)" >&2; exit 1; }
done
D=$SA_REPO/build/deploy_moe_$(basename "$B")
if [ -d "$D/results" ]; then                        # (board results copied back stay)
  find "$D" -mindepth 1 -maxdepth 1 ! -name results -exec rm -rf {} +
else
  rm -rf "$D" && mkdir -p "$D"
fi
for f in sa.vmfb sa_packed.irpa; do ln -f "$B/$f" "$D/$f" 2>/dev/null || cp "$B/$f" "$D/$f"; done
cp "$B"/s6/{prompt,expected_tokens,expected_logits}.npy "$D/"
# the window: the sim's device peak (sa-llm-run's report, in MiB) + 32 MiB
peak=$(grep -ho 'peak [0-9.]* MB' "$B"/s6/*.run.log 2>/dev/null | awk '{print $2}' | sort -n | tail -1)
mb=$("$SA_PY" -c "import math, os, sys; p = float(sys.argv[1] or 0) * 2**20 or os.path.getsize(sys.argv[2]) * 1.15
print(math.ceil(p / 2**20) + 32)" "${peak:-}" "$D/sa_packed.irpa")
echo "the functional simulator (decode only) $mb" > "$D/board.txt"
# --prefill only for an export with prefill (test_hf_moe --prefill 0: decode only)
grep -q "@prefill" "$B/qllama.mlir" && pf=" --prefill=$SA_D" || pf=
# the KV cache's length (the export's --iree-sa-max-dynamic; hf_generic's default 256)
vl=$(grep -o -- '--iree-sa-max-dynamic=[0-9]*' "$B/compile_flags.txt" 2>/dev/null | cut -d= -f2)
echo "--abi=hfv --pad=$SA_D --valid_len=${vl:-256}$pf" > "$D/sa_args.txt"
{
  echo "commit $(git -C "$SA_REPO" rev-parse HEAD)$(git -C "$SA_REPO" diff --quiet HEAD -- compiler firmware || echo ' (dirty)')"
  echo "built $(date -Iseconds), SA_D=$SA_D SA_GEMV_PORTS=$SA_GEMV_PORTS, from ${B#$SA_REPO/}"
} > "$D/build_info.txt"
make -C "$SA_REPO/firmware/rt" >/dev/null
stage_runtime "$D"
cp "$SA_COMPILER/tests/board_llm.py" "$SA_COMPILER/tests/board_teacher.py" "$SA_COMPILER/tests/board_profile.py" "$SA_COMPILER/runtime/test/board_launcher.py" "$SA_REPO/driver/pynq_matmul.py" \
   "$SA_REPO/firmware/rt/rt_fw.bin" "$D/"
stage_overlay "$D"
ls -la "$D"
echo "to the board: scp -r $D ubuntu@<board>:~/"

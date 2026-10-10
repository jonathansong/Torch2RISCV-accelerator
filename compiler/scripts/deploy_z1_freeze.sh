#!/usr/bin/env bash
# Stages the board regression against the PYNQ-Z1 freeze baselines
# (docs/kv260_upgrade_plan.md §7.2, K1a / K1b) for the KV260: every board test,
# one directory each (each made by its own deploy script, which runs the host
# test first: export, compile, per-dispatch check, sim run), plus
# board_regress.py and run_board.sh at the top. The aarch64 runtime and the
# overlay of boards/kv260/build/output/$SA_KV260_CONFIG (default d8_100mhz; or
# SA_BIT_DIR) go into build/deploy_kv260_<config>.
#   compiler/scripts/deploy_z1_freeze.sh [--board kv260] [--reuse <config>] [test ...]   (default: all; tests below)
#   scp -r build/deploy_kv260_<config> ubuntu@<kv260>:~/kv260_<config>
#   board: sudo ./run_board.sh [test ...]     (sources the board's PYNQ / XRT environment)
#   back:  scp -r ubuntu@<kv260>:~/kv260_<config>/results build/deploy_kv260_<config>/
#          $SA_PY compiler/tests/compare_z1_baselines.py build/deploy_kv260_<config>/results
# (The Z1's own bundle and make_z1_baselines.py: the pynq-z1 branch.)
#
#   ddr          RISC-V reads / writes DDR (tests/ddr_access)                            (K1a step 1)
#   bwtest       DMA bandwidth (notebooks/m2_bw_test.py, firmware/bwtest) on the board's overlay
#   fwdemo       gemm / vector / desc_run firmware vs NumPy (notebooks/m1, m3, m5 demos) (step 3)
#   c1           the sa HAL driver's C1 test (sa_hal_test through board_launcher.py) (step 4)
#   c3           stories15M decode vs the hand-written L5 path (DeviceModel)        deploy_c3.sh
#   c55          SmolLM2-135M decode, export_hf.py (qhf), vs the sim                deploy_c55.sh
#   c6p_stories  stories15M prefill (M = D, the overlay's array size) + decode       deploy_c6p.sh stories D
#   c6p_smollm2  SmolLM2-135M prefill (M = D) + decode, qhf                         deploy_c6p.sh smollm2 D
#   hfgen        SmolLM2-135M prefill (M = D) + decode, unmodified HF                deploy_hfgen.sh
#   c6p_qwen3    Qwen3-0.6B prefill (M = D) + decode (KV260 only; not in the default set: name it;
#                the board needs a 640 MB udmabuf: the 1280 MiB DT pool loaded at boot,
#                boards/kv260/README.md "Accelerator memory pool")              deploy_c6p.sh qwen3 D
# D = SA_D (board_env.sh: from the overlay's configuration); at D != 8 the host builds go to
# build/<test>/<model>_d<D> (the D = 8 ones feed the golden corpus)
# Each deploy runs in a memory-capped scope (MEM, default 12G).
# --reuse <config>: the LLM tests (c1 ... hfgen) are not rebuilt: their directories come from
# build/deploy_kv260_<config> (same D), with this overlay, rt_fw.bin and the board scripts of
# the tree; minutes instead of hours. Only for a new overlay with the same D (DMA width, clock):
# its host tests and sim do not depend on those. A change to the compiler, runtime or exports
# needs the full staging.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
if [ "${1:-}" = "--board" ]; then
  export SA_BOARD=${2:?--board kv260}
  shift 2
fi
REUSE=
if [ "${1:-}" = "--reuse" ]; then
  REUSE=${2:?--reuse <config>}
  shift 2
fi
source "$(dirname "$0")/board_env.sh"
export SA_BOARD SA_BIT_DIR
Z=$SA_REPO/build/deploy_kv260_$SA_KV260_CONFIG          # (per overlay: results are kept)
if [ -n "$REUSE" ]; then
  R=$SA_REPO/build/deploy_kv260_$REUSE
  [ "$R" != "$Z" ] || { echo "--reuse $REUSE: that is this overlay's own bundle"; exit 2; }
  grep -q "D = $SA_D\$" "$R/VERSION" 2>/dev/null || { echo "--reuse: $R/VERSION is missing or not D = $SA_D"; exit 2; }
fi
MEM=${MEM:-12G}
ALL=(ddr bwtest fwdemo c1 c3 c55 c6p_stories c6p_smollm2 hfgen)
TESTS=("$@")
[ ${#TESTS[@]} -eq 0 ] && TESTS=("${ALL[@]}") && rm -rf "$Z"
mkdir -p "$Z"
L2=$SA_BIT_DIR                 # the board's overlay
capped() { systemd-run --user --scope -p MemoryMax="$MEM" -q "$@"; }
stage() {                      # <test> <deploy dir> <deploy command...>
  local t=$1 d=$2
  shift 2
  echo "== $t: $*"
  capped "$@" > "$Z/$t.deploy.log" 2>&1 || { echo "$t: deploy FAILED (see $Z/$t.deploy.log)"; tail -20 "$Z/$t.deploy.log"; exit 1; }
  rm -rf "$Z/$t" && mv "$d" "$Z/$t"
  cp "$SA_COMPILER/tests/board_profile.py" "$Z/$t/"
}
reuse() {                      # <test>: the test's directory from $R, with this overlay and the tree's scripts
  local t=$1 f
  [ -d "$R/$t" ] || { echo "--reuse: $R/$t is missing"; exit 1; }
  echo "== $t: from ${R#$SA_REPO/}"
  rm -rf "$Z/$t" && cp -r "$R/$t" "$Z/$t"
  rm -rf "$Z/$t/results"
  cp "$L2/picorv32.bit" "$L2/picorv32.hwh" "$Z/$t/"
  for f in "$SA_COMPILER/tests/board_llm.py" "$SA_COMPILER/tests/board_generate.py" "$SA_COMPILER/tests/board_profile.py" \
           "$SA_COMPILER/runtime/test/board_launcher.py" "$SA_REPO/driver/pynq_matmul.py" "$SA_REPO/firmware/rt/rt_fw.bin"; do
    [ -f "$Z/$t/$(basename "$f")" ] && cp "$f" "$Z/$t/"
  done
  [ -f "$R/$t.deploy.log" ] && cp "$R/$t.deploy.log" "$Z/"
  return 0
}
for t in "${TESTS[@]}"; do
  if [ -n "$REUSE" ]; then
    case $t in c1|c3|c55|c6p_stories|c6p_smollm2|hfgen|c6p_qwen3) reuse "$t"; continue ;; esac
  fi
  case $t in
    ddr)
      rm -rf "$Z/ddr" && mkdir -p "$Z/ddr"
      cp "$SA_REPO/tests/ddr_access/ddr_test.py" "$SA_REPO/tests/ddr_access/ddr_test.bin" \
         "$L2/picorv32.bit" "$L2/picorv32.hwh" "$Z/ddr/" ;;
    fwdemo)
      rm -rf "$Z/fwdemo" && mkdir -p "$Z/fwdemo"
      cp "$SA_REPO/notebooks/m1_gemm_demo.py" "$SA_REPO/notebooks/m3_vector_demo.py" "$SA_REPO/notebooks/m5_desc_demo.py" \
         "$SA_REPO/firmware/gemm/gemm_fw.bin" "$SA_REPO/firmware/vector/vector_fw.bin" \
         "$SA_REPO/firmware/desc_run/desc_run_fw.bin" "$SA_REPO/firmware/matmul/matmul_fw.bin" \
         "$SA_REPO/firmware/matmul_insn/matmul_insn_fw.bin" "$SA_REPO/driver/pynq_matmul.py" \
         "$L2/picorv32.bit" "$L2/picorv32.hwh" "$Z/fwdemo/" ;;
    c1)          stage c1 "$SA_REPO/build/deploy_c1" "$SA_COMPILER/scripts/deploy_c1.sh" ;;
    bwtest)
      rm -rf "$Z/bwtest" && mkdir -p "$Z/bwtest"
      cp "$SA_REPO/notebooks/m2_bw_test.py" "$SA_REPO/firmware/bwtest/bwtest_fw.bin" "$SA_REPO/driver/pynq_matmul.py" \
         "$L2/picorv32.bit" "$L2/picorv32.hwh" "$Z/bwtest/" ;;
    c3)          stage c3 "$SA_REPO/build/deploy_c3" "$SA_COMPILER/scripts/deploy_c3.sh" ;;
    c55)         stage c55 "$SA_REPO/build/deploy_c55" "$SA_COMPILER/scripts/deploy_c55.sh" ;;
    c6p_stories) stage c6p_stories "$SA_REPO/build/deploy_c6p_stories_m$SA_D" "$SA_COMPILER/scripts/deploy_c6p.sh" stories "$SA_D" ;;
    c6p_smollm2) stage c6p_smollm2 "$SA_REPO/build/deploy_c6p_smollm2_m$SA_D" "$SA_COMPILER/scripts/deploy_c6p.sh" smollm2 "$SA_D" ;;
    hfgen)       stage hfgen "$SA_REPO/build/deploy_hfgen" "$SA_COMPILER/scripts/deploy_hfgen.sh" ;;
    c6p_qwen3)   stage c6p_qwen3 "$SA_REPO/build/deploy_c6p_qwen3_m$SA_D" "$SA_COMPILER/scripts/deploy_c6p.sh" qwen3 "$SA_D" ;;
    *) echo "unknown test $t (${ALL[*]} c6p_qwen3)"; exit 2 ;;
  esac
done
cp "$SA_COMPILER/tests/board_regress.py" "$SA_COMPILER/tests/run_board.sh" "$Z/"
{
  echo "commit $(git -C "$SA_REPO" rev-parse HEAD)$(git -C "$SA_REPO" diff --quiet HEAD -- compiler rtl firmware driver || echo ' (dirty)')"
  echo "staged $(date -Iseconds)"
  echo "bitstream $(sha256sum "$L2/picorv32.bit" | cut -c1-16) (${L2#$SA_REPO/})"
  echo "board $SA_BOARD (runtime $SA_RT_ARCH), D = $SA_D"
  [ -n "$REUSE" ] && echo "LLM tests reused from ${R#$SA_REPO/} ($(head -1 "$R/VERSION"))"
  [ -f "$L2/build_info.txt" ] && sed 's/^/overlay /' "$L2/build_info.txt"
} > "$Z/VERSION"
cat "$Z/VERSION"
du -sh "$Z"/* | sort -k2

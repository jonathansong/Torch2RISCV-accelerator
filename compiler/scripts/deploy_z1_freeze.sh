#!/usr/bin/env bash
# Stages the PYNQ-Z1 freeze regression (docs/kv260_upgrade_plan.md §7.2) in
# build/deploy_z1: every board test of the current main, one directory each
# (each made by its own deploy script, which runs the host test first: export,
# compile, per-dispatch check, sim run), plus board_regress.py at the top.
#   compiler/scripts/deploy_z1_freeze.sh [--board pynq-z1|kv260] [test ...]   (default: all; tests below)
#   --board kv260: the KV260's aarch64 runtime and overlay (KV260/build/output/$SA_KV260_CONFIG,
#   default d8_50mhz; or SA_BIT_DIR)
#   into build/deploy_kv260_<config> (docs/kv260_upgrade_plan.md K1a / K1b; compare the results
#   with compiler/tests/compare_z1_baselines.py); the default pynq-z1 stages build/deploy_z1
#   scp -r build/deploy_z1 xilinx@<board>:/home/xilinx/z1
#   board: sudo ./run_board.sh [test ...]     (sources the board's PYNQ / XRT environment)
#   back:  scp -r xilinx@<board>:/home/xilinx/z1/results build/deploy_z1/
#          $SA_PY compiler/tests/make_z1_baselines.py          (-> tests/baselines/pynq-z1/)
#
#   ddr          RISC-V reads / writes DDR (RISCV-on-PYNQ-Z1/tests/ddr_access)          (K1a step 1)
#   bwtest       DMA bandwidth (notebooks/m2_bw_test.py, firmware/bwtest) on the board's overlay
#   fwdemo       gemm / vector / desc_run firmware vs NumPy (notebooks/m1, m3, m5 demos) (step 3)
#   c1           the sa HAL driver's C1 test (sa_hal_test through board_launcher.py) (step 4)
#   c3           stories15M decode vs the hand-written L5 path (DeviceModel)        deploy_c3.sh
#   c55          SmolLM2-135M decode, export_hf.py (qhf), vs the sim                deploy_c55.sh
#   c6p_stories  stories15M prefill (M = 8) + decode                                deploy_c6p.sh stories 8
#   c6p_smollm2  SmolLM2-135M prefill (M = 8) + decode, qhf                         deploy_c6p.sh smollm2 8
#   hfgen        SmolLM2-135M prefill (M = 8) + decode, unmodified HF (generic)     deploy_hfgen.sh
# Each deploy runs in a memory-capped scope (MEM, default 12G).
set -euo pipefail
source "$(dirname "$0")/../env.sh"
if [ "${1:-}" = "--board" ]; then
  export SA_BOARD=${2:?--board pynq-z1 or kv260}
  shift 2
fi
source "$(dirname "$0")/board_env.sh"
export SA_BOARD SA_BIT_DIR
case $SA_BOARD in
  pynq-z1) Z=$SA_REPO/build/deploy_z1 ;;
  kv260)   Z=$SA_REPO/build/deploy_kv260_$SA_KV260_CONFIG ;;   # (per overlay: results are kept)
  *)       Z=$SA_REPO/build/deploy_$SA_BOARD ;;
esac
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
for t in "${TESTS[@]}"; do
  case $t in
    ddr)
      rm -rf "$Z/ddr" && mkdir -p "$Z/ddr"
      cp "$SA_REPO/RISCV-on-PYNQ-Z1/tests/ddr_access/ddr_test.py" "$SA_REPO/RISCV-on-PYNQ-Z1/tests/ddr_access/ddr_test.bin" \
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
    c6p_stories) stage c6p_stories "$SA_REPO/build/deploy_c6p_stories_m8" "$SA_COMPILER/scripts/deploy_c6p.sh" stories 8 ;;
    c6p_smollm2) stage c6p_smollm2 "$SA_REPO/build/deploy_c6p_smollm2_m8" "$SA_COMPILER/scripts/deploy_c6p.sh" smollm2 8 ;;
    hfgen)       stage hfgen "$SA_REPO/build/deploy_hfgen" "$SA_COMPILER/scripts/deploy_hfgen.sh" ;;
    *) echo "unknown test $t (${ALL[*]})"; exit 2 ;;
  esac
done
cp "$SA_COMPILER/tests/board_regress.py" "$SA_COMPILER/tests/run_board.sh" "$Z/"
{
  echo "commit $(git -C "$SA_REPO" rev-parse HEAD)$(git -C "$SA_REPO" diff --quiet HEAD -- compiler rtl firmware driver || echo ' (dirty)')"
  echo "staged $(date -Iseconds)"
  echo "bitstream $(sha256sum "$L2/picorv32.bit" | cut -c1-16) (${L2#$SA_REPO/})"
  echo "board $SA_BOARD (runtime $SA_RT_ARCH)"
  [ -f "$L2/build_info.txt" ] && sed 's/^/overlay /' "$L2/build_info.txt"
} > "$Z/VERSION"
cat "$Z/VERSION"
du -sh "$Z"/* | sort -k2

#!/usr/bin/env bash
# Stages the PYNQ-Z1 freeze regression (docs/kv260_upgrade_plan.md §7.2) in
# build/deploy_z1: every board test of the current main, one directory each
# (each made by its own deploy script, which runs the host test first: export,
# compile, per-dispatch check, sim run), plus board_regress.py at the top.
#   compiler/scripts/deploy_z1_freeze.sh [test ...]     (default: all; tests below)
#   scp -r build/deploy_z1 xilinx@<board>:/home/xilinx/z1
#   board: sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh && \
#            cd /home/xilinx/z1 && python3 board_regress.py'
#   back:  scp -r xilinx@<board>:/home/xilinx/z1/results build/deploy_z1/
#          $SA_PY compiler/tests/make_z1_baselines.py          (-> tests/baselines/pynq-z1/)
#
#   bwtest       DMA bandwidth (notebooks/m2_bw_test.py, firmware/bwtest) on the L2 overlay
#   c3           stories15M decode vs the hand-written L5 path (DeviceModel)        deploy_c3.sh
#   c55          SmolLM2-135M decode, export_hf.py (qhf), vs the sim                deploy_c55.sh
#   c6p_stories  stories15M prefill (M = 8) + decode                                deploy_c6p.sh stories 8
#   c6p_smollm2  SmolLM2-135M prefill (M = 8) + decode, qhf                         deploy_c6p.sh smollm2 8
#   hfgen        SmolLM2-135M prefill (M = 8) + decode, unmodified HF (generic)     deploy_hfgen.sh
# Each deploy runs in a memory-capped scope (MEM, default 12G).
set -euo pipefail
source "$(dirname "$0")/../env.sh"
Z=$SA_REPO/build/deploy_z1
MEM=${MEM:-12G}
ALL=(bwtest c3 c55 c6p_stories c6p_smollm2 hfgen)
TESTS=("$@")
[ ${#TESTS[@]} -eq 0 ] && TESTS=("${ALL[@]}") && rm -rf "$Z"
mkdir -p "$Z"
L2=$SA_REPO/RISCV-on-PYNQ-Z1/bitstreams/l2
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
cp "$SA_COMPILER/tests/board_regress.py" "$Z/"
{
  echo "commit $(git -C "$SA_REPO" rev-parse HEAD)$(git -C "$SA_REPO" diff --quiet HEAD -- compiler rtl firmware driver || echo ' (dirty)')"
  echo "staged $(date -Iseconds)"
  echo "bitstream $(sha256sum "$L2/picorv32.bit" | cut -c1-16) (RISCV-on-PYNQ-Z1/bitstreams/l2)"
} > "$Z/VERSION"
cat "$Z/VERSION"
du -sh "$Z"/* | sort -k2

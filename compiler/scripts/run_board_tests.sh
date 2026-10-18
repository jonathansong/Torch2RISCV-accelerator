#!/usr/bin/env bash
# On the board: the acceptance tests of the _pcpi overlay's bundles (compiler/scripts/
# build_board_bundles.sh), one after another through each bundle's board_run.py (results
# under <bundle>/results/<commit>/<test>/<overlay>/<mode>/). Each test gets its own log; a
# failing test does not stop the next ones; a rerun skips the tests that passed (REDO=1:
# run them again). A bundle that is not in the home directory is skipped.
#   sudo bash ~/run_board_tests.sh               all: rvb gemma qwen3 llama
#   sudo bash ~/run_board_tests.sh qwen3 llama   only those bundles
#   sudo REDO=1 bash ~/run_board_tests.sh rvb
#   rvbc: the RVB cases with generated code (deploy_rvb.sh RVB_GEN=compiled); the models' rvcode:
#   every dispatch as RISC-V code (deploy_rvcode.sh); their "rvsched --rv": the core's decode loop
#   calling that code (S3); the "--embedded" variants: the code iree-compile embedded in sa.vmfb
#   (S4, deploy_rvembed.sh); qwen3g / gemmag / llamag: the models compiled with the per-layer copies merged
#   (--iree-flow-enable-executable-generalization; the qwen3 bundle with its own sa.vmfb); "rvsched --r5": the block argmax by a kernel IREE's llvm-cpu compiled
#   for the core (R5, deploy_r5.sh); "rvperf": the performance-counter breakdown of plan A against the lists
#   (board_rvperf.py; then compiler/tests/rv_perf.py on the rvperf_*.csv)
#   (rvsched: the decode loop on the RISC-V core, after compiler/scripts/deploy_rvsched.sh; without
#   REDO only the tests not passed yet run)
# LC_SUFFIX: the model bundles of another build_board_bundles.sh suffix (default _pcpi; logs in
# ~/board_tests<suffix>). smollm2: SmolLM2-135M at 2048, every test of the release set.
# smollm2f / gemmaf / qwen3f / llamaf: the same models with decode's gated FFN in one dispatch
# (SA_FUSE_FFN=1, docs/fusion_design.md; build/fuse/bundles.sh), against the g bundles' speed.
# smollm2f2 / gemmaf2 / qwen3f2 / llamaf2: the same with each chunk's gating overlapping the next
# chunk's weight streaming (fusion_design.md §4.9).
# smollm2a / gemmaa / qwen3a / llamaa: the f2 builds with decode's attention in one dispatch
# per layer too (SA_FUSE_ATTN=1, fusion_design.md §6.4; build/fuse/bundles_attn.sh).
# Logs: ~/board_tests<LC_SUFFIX, default _pcpi>/<bundle>.<mode>.log, markers .done / .failed, summary.txt.
set -u
H=${BUNDLE_HOME:-/home/ubuntu}
CFG=${SA_KV260_CONFIG:-d16_100mhz_w128_p6_gemv_np4_pcpi}
SUF=${LC_SUFFIX:-_pcpi}
L=$H/board_tests$SUF
mkdir -p "$L"
[ "$(id -u)" = 0 ] || { echo "run as root: sudo bash $0 $*" >&2; exit 1; }
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh 2>/dev/null

# <short name> <bundle directory> then its tests (board_run.py arguments, one per item)
declare -A DIR=(
  [rvb]=deploy_rvb_$CFG
  [rvbc]=deploy_rvb_${CFG}_compiled
  [rvbe]=deploy_rvb_${CFG}_embedded
  [gemma]=deploy_hfgen_gemma_lc512$SUF
  [qwen3]=deploy_hfgen_qwen3_lc2048$SUF
  [llama]=deploy_hfgen_llama_lc2048$SUF
  [smollm2]=deploy_hfgen_smollm2_lc2048$SUF
  [qwen3g]=deploy_hfgen_qwen3g_pcpi
  [gemmag]=deploy_hfgen_gemmag_pcpi
  [llamag]=deploy_hfgen_llamag_pcpi
  [smollm2f]=deploy_hfgen_smollm2f_pcpi
  [gemmaf]=deploy_hfgen_gemmaf_pcpi
  [qwen3f]=deploy_hfgen_qwen3f_pcpi
  [llamaf]=deploy_hfgen_llamaf_pcpi
  [smollm2f2]=deploy_hfgen_smollm2f2_pcpi
  [gemmaf2]=deploy_hfgen_gemmaf2_pcpi
  [qwen3f2]=deploy_hfgen_qwen3f2_pcpi
  [llamaf2]=deploy_hfgen_llamaf2_pcpi
  [smollm2a]=deploy_hfgen_smollm2a_pcpi
  [gemmaa]=deploy_hfgen_gemmaa_pcpi
  [qwen3a]=deploy_hfgen_qwen3a_pcpi
  [llamaa]=deploy_hfgen_llamaa_pcpi
)
tests_of() {
  case $1 in
    rvb|rvbc|rvbe) echo "rvb" ;;
    gemma) printf '%s\n' "llm" "longctx" "longctx --sweep 64,128,256,512" "teacher" "rvsched" "rvcode" "rvsched --rv" "rvcode --embedded" "rvsched --rv --embedded" "rvsched --r5" "rvperf" ;;
    qwen3) printf '%s\n' "llm" "longctx" "longctx --sweep 64,256,512,1024,2048" "teacher" "rvsched" "rvcode" "rvsched --rv" "rvcode --embedded" "rvsched --rv --embedded" "rvsched --r5" "rvperf" ;;
    llama) printf '%s\n' "llm" "longctx" "longctx --sweep 64,256,512,1024,2048" "teacher" "rvsched" "rvcode" "rvsched --rv" "rvcode --embedded" "rvsched --rv --embedded" "rvsched --r5" "rvperf" ;;
    gemmag) printf '%s\n' "llm" "rvsched" "rvcode --embedded" "rvsched --rv --embedded" "rvperf" "longctx" "longctx --sweep 64,128,256,512" \
      "teacher" "profile 40 --prompt 32" "profile 40 --prompt 32 --batch" "rvsched --r5" ;;
    qwen3g|llamag) printf '%s\n' "llm" "rvsched" "rvcode --embedded" "rvsched --rv --embedded" "rvperf" "longctx" "longctx --sweep 64,256,512,1024,2048" \
      "teacher" "profile 40 --prompt 32" "profile 40 --prompt 32 --batch" "rvsched --r5" ;;
    smollm2) printf '%s\n' "llm" "teacher" "profile 40 --prompt 32" "profile 40 --prompt 32 --batch" "longctx" "longctx --sweep 64,256,512,1024,2048" \
      "rvsched" "rvcode --embedded" "rvsched --rv --embedded" "rvsched --r5" "rvperf" ;;
    # the SwiGLU fusion (SA_FUSE_FFN=1, docs/fusion_design.md): bit-exact checks, decode speed, the RISC-V paths
    gemmaf|gemmaf2|gemmaa) printf '%s\n' "llm" "profile 40 --prompt 32 --batch" "profile 40 --prompt 32" "longctx" "longctx --sweep 64,128,256,512" \
      "rvsched" "rvcode --embedded" "rvsched --rv --embedded" ;;
    smollm2f|qwen3f|llamaf|smollm2f2|qwen3f2|llamaf2|smollm2a|qwen3a|llamaa) printf '%s\n' "llm" "profile 40 --prompt 32 --batch" "profile 40 --prompt 32" "longctx" \
      "longctx --sweep 64,256,512,1024,2048" "rvsched" "rvcode --embedded" "rvsched --rv --embedded" ;;
  esac
}

SEL=("$@")
[ ${#SEL[@]} -eq 0 ] && SEL=(rvb gemma qwen3 llama)
for b in "${SEL[@]}"; do
  d=$H/${DIR[$b]:-}
  [ -n "${DIR[$b]:-}" ] || { echo "unknown bundle $b (rvb rvbc rvbe gemma qwen3 llama smollm2 qwen3g gemmag llamag smollm2f gemmaf qwen3f llamaf smollm2f2 gemmaf2 qwen3f2 llamaf2 smollm2a gemmaa qwen3a llamaa)"; continue; }
  [ -d "$d" ] || { echo "[$b] $d not on the board: skipped"; continue; }
  while IFS= read -r t; do
    [ -n "$t" ] || continue
    name=$b.$(echo "$t" | tr ' ,' '__' | sed 's/--//g')
    if [ -f "$L/$name.done" ] && [ -z "${REDO:-}" ]; then
      echo "[$name] passed before: skipped"
      continue
    fi
    rm -f "$L/$name.done" "$L/$name.failed"
    echo "[$name] start $(date '+%F %T'): (cd $d && python3 board_run.py $t)"
    t0=$SECONDS
    # shellcheck disable=SC2086
    (cd "$d" && python3 board_run.py $t) > "$L/$name.log" 2>&1 < /dev/null
    rc=$?
    dt=$((SECONDS - t0))
    if [ $rc -eq 0 ]; then
      echo "$(date '+%F %T'), $dt s" > "$L/$name.done"
      echo "[$name] PASS in $dt s"
    else
      echo "$(date '+%F %T'), exit $rc after $dt s" > "$L/$name.failed"
      echo "[$name] FAIL (exit $rc after $dt s), last lines:"
      tail -5 "$L/$name.log" | sed 's/^/    /'
    fi
  done < <(tests_of "$b")
done

{
  echo "summary $(date '+%F %T') ($CFG)"
  for f in "$L"/*.done "$L"/*.failed; do
    [ -e "$f" ] || continue
    n=$(basename "$f"); echo "  ${n%.*}: ${n##*.} ($(cat "$f"))"
  done
} | tee "$L/summary.txt"

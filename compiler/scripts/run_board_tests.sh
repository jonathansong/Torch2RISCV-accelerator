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
#   (S4, deploy_rvembed.sh); qwen3g: Qwen3 compiled with the per-layer copies merged
#   (--iree-flow-enable-executable-generalization; the qwen3 bundle with its own sa.vmfb); "rvsched --r5": the block argmax by a kernel IREE's llvm-cpu compiled
#   for the core (R5, deploy_r5.sh); "rvperf": the performance-counter breakdown of plan A against the lists
#   (board_rvperf.py; then compiler/tests/rv_perf.py on the rvperf_*.csv)
#   (rvsched: the decode loop on the RISC-V core, after compiler/scripts/deploy_rvsched.sh; without
#   REDO only the tests not passed yet run)
# Logs: ~/board_tests_pcpi/<bundle>.<mode>.log, markers .done / .failed, summary.txt.
set -u
H=${BUNDLE_HOME:-/home/ubuntu}
CFG=${SA_KV260_CONFIG:-d16_100mhz_w128_p6_gemv_np4_pcpi}
L=$H/board_tests_pcpi
mkdir -p "$L"
[ "$(id -u)" = 0 ] || { echo "run as root: sudo bash $0 $*" >&2; exit 1; }
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh 2>/dev/null

# <short name> <bundle directory> then its tests (board_run.py arguments, one per item)
declare -A DIR=(
  [rvb]=deploy_rvb_$CFG
  [rvbc]=deploy_rvb_${CFG}_compiled
  [rvbe]=deploy_rvb_${CFG}_embedded
  [gemma]=deploy_hfgen_gemma_lc512_pcpi
  [qwen3]=deploy_hfgen_qwen3_lc2048_pcpi
  [llama]=deploy_hfgen_llama_lc2048_pcpi
  [qwen3g]=deploy_hfgen_qwen3g_pcpi
)
tests_of() {
  case $1 in
    rvb|rvbc|rvbe) echo "rvb" ;;
    gemma) printf '%s\n' "llm" "longctx" "longctx --sweep 64,128,256,512" "teacher" "rvsched" "rvcode" "rvsched --rv" "rvcode --embedded" "rvsched --rv --embedded" "rvsched --r5" "rvperf" ;;
    qwen3) printf '%s\n' "llm" "longctx" "longctx --sweep 64,256,512,1024,2048" "teacher" "rvsched" "rvcode" "rvsched --rv" "rvcode --embedded" "rvsched --rv --embedded" "rvsched --r5" "rvperf" ;;
    llama) printf '%s\n' "llm" "longctx" "longctx --sweep 64,256,512,1024,2048" "teacher" "rvsched" "rvcode" "rvsched --rv" "rvcode --embedded" "rvsched --rv --embedded" "rvsched --r5" "rvperf" ;;
    qwen3g) printf '%s\n' "llm" "rvsched" "rvcode --embedded" "rvsched --rv --embedded" "rvperf" ;;
  esac
}

SEL=("$@")
[ ${#SEL[@]} -eq 0 ] && SEL=(rvb gemma qwen3 llama)
for b in "${SEL[@]}"; do
  d=$H/${DIR[$b]:-}
  [ -n "${DIR[$b]:-}" ] || { echo "unknown bundle $b (rvb rvbc rvbe gemma qwen3 llama qwen3g)"; continue; }
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

#!/usr/bin/env bash
# Runs host tests with a bounded number of jobs, one log per test. For long
# runs from your own terminal (processes started by a Claude Code session end
# with it).
#   compiler/scripts/run_tests.sh [-j JOBS] [-t THREADS] [-o LOGDIR] TEST...
# TEST: c3 | c5 | c5cfg | smollm2 | qwen3l2 | synth | "any command line"
#   -j JOBS     tests at a time (default 1)
#   -t THREADS  threads per test: OpenMP / BLAS, and iree-compile
#               single-threaded when 1 (default 2)
#   -o LOGDIR   logs (default build/c6/logs); the summary line of each test
#               is printed when it ends
# Example: compiler/scripts/run_tests.sh -j 2 smollm2 qwen3l2 c5cfg
set -uo pipefail
source "$(dirname "$0")/../env.sh"
jobs=1 threads=2 logdir="$SA_REPO/build/c6/logs"
while getopts "j:t:o:" o; do
  case $o in
    j) jobs=$OPTARG ;; t) threads=$OPTARG ;; o) logdir=$OPTARG ;;
    *) sed -n 2,13p "$0"; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ $# -gt 0 ] || { sed -n 2,13p "$0"; exit 2; }
mkdir -p "$logdir"
cd "$SA_REPO"
export OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
flags=""
[ "$threads" = 1 ] && flags="--mlir-disable-threading"
T=compiler/tests
declare -A cmd=(
  [c3]="$SA_PY -u $T/test_c3.py --skip-export --out build/c4/fuse"
  [c5]="$SA_PY -u $T/test_c5.py"
  [c5cfg]="$SA_PY -u $T/test_c5.py --configs"
  [smollm2]="$SA_PY -u $T/test_c55.py --model build/llm_cache/SmolLM2-135M --out build/c55/smollm2 --check --steps 6 --generate 2 --flags '$flags'"
  [qwen3l2]="$SA_PY -u $T/test_c55.py --model build/llm_cache/Qwen3-0.6B --out build/c55/qwen3_l2 --check --steps 6 --generate 2 --mb 384 --flags '$flags'"
  [synth]="$SA_PY -u $T/test_c55.py --model build/llm_cache/synth-k16k --out build/c6/k16k --check --mb 64 --flags '$flags'"
)
run() {
  local name=$1 c=${cmd[$1]:-$1}
  local log="$logdir/$(printf '%s' "$name" | tr -c 'A-Za-z0-9_' '_' | cut -c1-40).log"
  local t0=$SECONDS
  echo "[start] $name -> $log"
  bash -c "$c" > "$log" 2>&1 < /dev/null
  local rc=$?
  echo "[$( [ $rc = 0 ] && echo PASS || echo FAIL )] $name ($((SECONDS - t0)) s): $(grep -aE 'PASS|FAIL' "$log" | tail -1)"
  return $rc
}
fail=0
for t in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge "$jobs" ]; do wait -n || fail=1; done
  run "$t" &
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || fail=1; done
exit $fail

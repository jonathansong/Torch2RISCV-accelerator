#!/usr/bin/env bash
# Runs host tests with a bounded number of jobs, one log per test. For long
# runs from your own terminal (processes started by a Claude Code session end
# with it).
#   compiler/scripts/run_tests.sh [-j JOBS] [-t THREADS] [-o LOGDIR] TEST...
# TEST: golden | fallback | c3 | c5 | c5cfg | smollm2 | qwen3l2 | synth | qwen3export | qwen3 | "any command line"
#   -j JOBS     tests at a time (default 1)
#   -t THREADS  threads per test: OpenMP / BLAS, and iree-compile
#               single-threaded when 1 (default 2)
#   -o LOGDIR   logs (default build/c6/logs); the summary line of each test
#               is printed when it ends
#   -m MEM      memory cap per test (systemd scope, default 6G; 0: none)
# Example: compiler/scripts/run_tests.sh -j 2 smollm2 qwen3l2 c5cfg
set -uo pipefail
source "$(dirname "$0")/../env.sh"
jobs=1 threads=2 logdir="$SA_REPO/build/c6/logs" mem=6G
while getopts "j:t:o:m:" o; do
  case $o in
    j) jobs=$OPTARG ;; t) threads=$OPTARG ;; o) logdir=$OPTARG ;; m) mem=$OPTARG ;;
    *) sed -n 2,14p "$0"; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ $# -gt 0 ] || { sed -n 2,14p "$0"; exit 2; }
mkdir -p "$logdir"
cd "$SA_REPO"
export OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
flags=""
[ "$threads" = 1 ] && flags="--mlir-disable-threading"
T=compiler/tests
declare -A cmd=(
  # C8 guard: the pass lit tests, then the golden corpus byte for byte (~30 s)
  [golden]="$IREE_BUILD/llvm-project/bin/llvm-lit -q compiler/plugins/sa/test && $SA_PY -u $T/golden.py"
  # the host fallback (plan §8.15): a small model and stories15M with host linear layers, on sim
  [fallback]="$SA_PY -u $T/test_fallback.py"
  [c3]="$SA_PY -u $T/test_c3.py --skip-export --out build/c4/fuse"
  [c5]="$SA_PY -u $T/test_c5.py"
  [c5cfg]="$SA_PY -u $T/test_c5.py --configs"
  [smollm2]="$SA_PY -u $T/test_c55.py --model build/llm_cache/SmolLM2-135M --out build/c55/smollm2 --check --steps 6 --generate 2 --flags '$flags'"
  [qwen3l2]="$SA_PY -u $T/test_c55.py --model build/llm_cache/Qwen3-0.6B --out build/c55/qwen3_l2 --check --steps 6 --generate 2 --mb 384 --flags '$flags'"
  [synth]="$SA_PY -u $T/test_c55.py --model build/llm_cache/synth-k16k --out build/c6/k16k --check --mb 64 --flags '$flags'"
  # C6.1: Qwen3-0.6B, all 28 layers (export once, then compile / check / sim)
  [qwen3export]="$SA_PY -u compiler/frontend/export_hf.py --model build/llm_cache/Qwen3-0.6B --out build/c6/qwen3 --tokens 8"
  [qwen3]="$SA_PY -u $T/test_c55.py --model build/llm_cache/Qwen3-0.6B --out build/c6/qwen3 --check --steps 6 --generate 2 --mb 768 --flags '$flags'"
)
run() {
  local name=$1 c=${cmd[$1]:-$1}
  local log="$logdir/$(printf '%s' "$name" | tr -c 'A-Za-z0-9_' '_' | cut -c1-40).log"
  local t0=$SECONDS
  echo "[start] $name -> $log"
  # its own scope with a memory cap: a test that grows is OOM-killed alone
  # instead of the whole session's memory pressure (systemd-oomd kills the
  # terminal's scope, and with it every job, above 50% for 20 s)
  if [ "$mem" != 0 ] && command -v systemd-run > /dev/null; then
    systemd-run --user --scope -q -p MemoryMax="$mem" -p MemorySwapMax=0 bash -c "$c" > "$log" 2>&1 < /dev/null
  else
    bash -c "$c" > "$log" 2>&1 < /dev/null
  fi
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

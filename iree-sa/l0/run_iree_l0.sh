#!/usr/bin/env bash
# On the board (no sudo): IREE runtime + an llvm-cpu armv7 module.
#   cd /home/xilinx/iree_l0 && bash run_iree_l0.sh
cd "$(dirname "$0")"
set -u
ok=1
echo "== iree-cpuinfo"; ./iree-cpuinfo || ok=0
for dev in local-sync local-task; do
    echo "== iree-run-module --device=$dev"
    ./iree-run-module --device=$dev --module=model_armv7.vmfb --function=main \
        --input=@input.npy --expected_output=@expected.npy || ok=0
done
if [ -d ops ]; then
    echo "== single-op tests"
    (cd ops && RUN=../iree-run-module bash run_op_tests.sh) | tee ops.txt
    grep -q FAIL ops.txt && ok=0
fi
echo "== iree-benchmark-module --device=local-task"
./iree-benchmark-module --device=local-task --module=model_armv7.vmfb --function=main \
    --input=@input.npy --benchmark_repetitions=3 2>&1 | grep -E "BM_|Run on|CPU Caches" || ok=0
[ $ok = 1 ] && echo "IREE L0 PASS" || echo "IREE L0 FAIL"

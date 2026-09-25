#!/usr/bin/env bash
# On the board: run every <op>.vmfb against its expected output (op_tests.py).
#   cd /home/xilinx/iree_l0/ops && bash run_op_tests.sh
cd "$(dirname "$0")"
RUN=${RUN:-../iree-run-module}
for m in *.vmfb; do
    op=${m%.vmfb}
    if $RUN --device=local-sync --module="$m" --function=main --input=@${op}_in.npy \
            --expected_output=@${op}_exp.npy > "${op}.log" 2>&1; then
        printf "%-12s PASS\n" "$op"
    else
        printf "%-12s FAIL  %s\n" "$op" "$(grep -m1 -oE 'element at index [0-9]+ \([^)]*\) does not match the expected \([^)]*\)|UNAVAILABLE.*|INVALID.*' "${op}.log")"
    fi
done

#!/usr/bin/env bash
# LLM step 1: llama2.c on the PYNQ-Z1's ARM Cortex-A9 (650 MHz, NEON), no accelerator.
#
#   cd /home/xilinx/llm && bash arm_baseline.sh [tokens]      (no sudo needed)
#
# Builds run.c (fp32) and runq.c (Q8_0: int8 weights and activations, fp32
# scales per group), each single-threaded and with OpenMP on both cores, and
# generates greedily (-t 0) from a fixed prompt; prints llama2.c's
# "achieved tok/s". runq_gs32 is runq.c with the group size fixed at 32 at
# compile time: the Cortex-A9 has no integer divide, and stock runq.c
# divides by the run-time GS in its inner loop (__divsi3 ~29 % of the time
# on the board); with a constant the divisions become shifts and the group
# loop can be vectorized. Then a gprof build of each shows how much of the time the
# matmul (GEMV) takes. Results go to arm_baseline.txt.
set -uo pipefail

N=${1:-256}
PROMPT="Once upon a time"
CFLAGS=${CFLAGS:-"-Ofast -march=armv7-a -mfpu=neon -mfloat-abi=hard -mtune=cortex-a9"}
OUT=arm_baseline.txt
cd "$(dirname "$0")"
mkdir -p bin

echo "== build (gcc $(gcc -dumpversion))"
sed -e 's/^int GS = 0;.*/#define GS 32/' \
    -e 's/^\( *\)GS = group_size;.*/\1if (group_size != GS) { fprintf(stderr, "group size %d, runq_gs32 needs 32\\n", group_size); exit(EXIT_FAILURE); }/' \
    llama2c/runq.c > llama2c/runq_gs32.c
for p in run runq runq_gs32; do
    gcc $CFLAGS -o bin/$p llama2c/$p.c -lm
    gcc $CFLAGS -fopenmp -o bin/${p}_omp llama2c/$p.c -lm
    gcc $CFLAGS -pg -fno-inline -o bin/${p}_pg llama2c/$p.c -lm
done

tps() {   # binary model [env] -> tok/s
    env ${3:-} "./bin/$1" "$2" -z tokenizer.bin -t 0 -n "$N" -i "$PROMPT" 2>&1 >/dev/null |
        sed -n 's/achieved tok\/s: //p'
}

{
echo "llama2.c $N tokens, greedy, prompt \"$PROMPT\"; $(nproc) cores; $(date -I)"
printf "%-22s %-10s %12s %12s\n" model build "1 thread" "2 threads"
for m in stories15M.bin stories42M.bin stories15M_q80.bin stories42M_q80.bin stories110M_q80.bin; do
    [ -s "$m" ] || continue
    for p in run runq runq_gs32; do
        case $m:$p in *_q80*:run|*M.bin:runq*) continue ;; esac
        printf "%-22s %-10s %12s %12s\n" "$m" "$p" "$(tps $p "$m")" "$(tps ${p}_omp "$m" OMP_NUM_THREADS=2)"
    done
done

echo
echo "gprof (single thread, -fno-inline): share of time per function"
for mp in stories15M.bin:run stories15M_q80.bin:runq stories15M_q80.bin:runq_gs32 stories42M_q80.bin:runq_gs32; do
    m=${mp%%:*} p=${mp##*:}
    [ -s "$m" ] || continue
    rm -f gmon.out
    "./bin/${p}_pg" "$m" -z tokenizer.bin -t 0 -n "$N" -i "$PROMPT" >/dev/null 2>&1
    echo "-- $m ($p)"
    gprof -b -p "bin/${p}_pg" gmon.out 2>/dev/null | awk 'NR>5 && $1+0>0.5 {printf "   %6.1f%%  %s\n", $1, $NF}' | head -8
done
} | tee "$OUT"
echo "saved $OUT"

#!/bin/bash
# all prepared dispatches through RVB_TEST: $1 = bram | ddr, $2 = CPU DDR latency
cd "$(dirname "$0")"
for d in 3 6 7 16 19 28; do
  r=$(make -s sim RVB=../../build/rvb/smol_$d RVB_CODE=$1 SIM_CPU_DDR_LAT=$2 2>&1 | grep -E "^TB|bytes|ERROR" | tr '\n' ' ')
  echo "smol_$d [$1 lat $2] $(head -1 ../../build/rvb/smol_$d/rvb_args.h | sed 's/.*main\$async_dispatch_[0-9]*_//; s/,.*//') $(sed -n 's/#define RVB_COMMANDS //p' ../../build/rvb/smol_$d/rvb_args.h) cmds :: $r"
done

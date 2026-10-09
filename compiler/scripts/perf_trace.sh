#!/bin/bash
# Command trace of a staged LLM test on the functional simulator, for compiler/tests/perf_model.py:
# 2 prefill chunks (32 tokens) + 2 decode steps, one list per dispatch (SA_PROFILE=1).
#   MODEL=c6p_qwen3 MB=640 OUT=qwen3 compiler/scripts/perf_trace.sh   (DEPLOY: the staged test bundle)
R=$(cd "$(dirname "$0")/../.." && pwd)
D=${DEPLOY:-$R/build/deploy_kv260_d16_100mhz_w128_p6_red}/${MODEL:-c6p_qwen3}
O=${OUTDIR:-$R/build/perfmodel}; mkdir -p $O
P=$R/build/iree/venv/bin/python
S=/tmp/sa_perf_$$.sock
$P $R/compiler/sim/sa_sim_server.py --d 16 --mb ${MB:-640} --socket $S --once --trace $O/${OUT:-qwen3}.trace > $O/${OUT:-qwen3}.server.log 2>&1 &
sleep 5
T=$($P -c "import numpy as np; print(','.join(map(str, np.resize(np.load('$D/prompt.npy'), 32).tolist())))")
time SA_PROFILE=1 SA_SIM_SOCKET=$S $R/build/iree/build-sa-host/runtime/plugins/hal/drivers/sa/sa-llm-run --device=sa \
  --module=$D/sa.vmfb --parameters=model=$D/sa_packed.irpa --tokens=$T --generate=2 --pad=16 --prefill=16
wait

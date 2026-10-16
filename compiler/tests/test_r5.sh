#!/usr/bin/env bash
# R5 of the RISC-V backend on the host (docs/riscv_backend_design.md §5): the
# block argmax kernel compiled by IREE's llvm-cpu (compiler/tests/r5_kernel.py)
# inside plan B's decode loop, the sim's emulation of the core calling its host
# build. A 2-layer SmolLM2, 20 tokens three ways: the ARM's loop, the core's
# loop with its own scan, the core's loop with the R5 kernel (SA_R5); the
# tokens must be identical, and the kernel must have run.
#   compiler/tests/test_r5.sh [<hfgen dir, default build/hfgen/smollm2_l2_e2e_vfix>]
set -euo pipefail
source "$(dirname "$0")/../env.sh"
CHK=${1:-$SA_REPO/build/hfgen/smollm2_l2_e2e_vfix}
RUN=$SA_REPO/build/iree/build-sa-host/runtime/plugins/hal/drivers/sa/sa-llm-run
T=$(mktemp -d /tmp/r5_XXXX)
trap 'rm -rf "$T"' EXIT
V=$("$SA_PY" - "$CHK" <<'PY'
import re, sys
m = re.search(r"func\.func @main\([^{]*-> !torch\.vtensor<\[1,1,(\d+)\],f32>", open(sys.argv[1] + "/qllama.mlir").read(1 << 22))
print(m.group(1) if m else 0)
PY
)
"$SA_PY" "$SA_COMPILER/frontend/argmax_module.py" --vocab "$V" --out "$T/argmax.vmfb" | tail -1
"$SA_PY" "$SA_COMPILER/tests/r5_kernel.py" --vocab "$V" --out "$T/r5"
for m in ref rvs r5; do
  x=; [ $m != ref ] && x=--rvsched
  r5=; [ $m = r5 ] && r5=$T/r5
  "$SA_PY" "$SA_COMPILER/sim/sa_sim_server.py" --d "${SA_D:-16}" --mb 256 --socket "$T/s.sock" --once >/dev/null 2>&1 &
  for i in $(seq 50); do [ -S "$T/s.sock" ] && break; sleep 0.2; done
  SA_SIM_SOCKET=$T/s.sock SA_R5=$r5 "$RUN" --device=sa --module="$T/argmax.vmfb" --module="$CHK/sa.vmfb" \
    --parameters=model="$CHK/sa_packed.irpa" --abi=hfv --pad=16 --valid_len=256 --tokens=1,504,3290,282 \
    --generate=20 $x > "$T/$m.out" 2> "$T/$m.err" || { tail -5 "$T/$m.err"; exit 1; }
  wait || true
  rm -f "$T/s.sock"
done
grep -q "^sa r5: the block argmax by" "$T/r5.err" || { echo "test_r5: the R5 kernel was not loaded"; tail -5 "$T/r5.err"; exit 1; }
for m in rvs r5; do
  if [ "$(head -1 "$T/ref.out")" != "$(head -1 "$T/$m.out")" ]; then
    echo "test_r5: the tokens of $m differ from the ARM loop's"; head -1 "$T/ref.out" "$T/$m.out"; exit 1
  fi
done
echo "test_r5 PASS: $(head -1 "$T/r5.out" | cut -c1-80)... identical in the ARM's loop, the core's, the core's with $(grep -o 'R5 kernel [^ ]*' "$T/r5.err")"

#!/usr/bin/env bash
# Adds plan B of the RISC-V backend (the decode loop on the PicoRV32,
# docs/riscv_backend_design.md §7.1) to staged hfgen bundles, the model as it
# is (no recompile): argmax.vmfb (the logits' block maxima for the core's
# argmax, compiler/frontend/argmax_module.py), rt_fw.bin version 3
# (RT_RUN_SCHED; the earlier entries unchanged), the aarch64 sa-llm-run with
# --rvsched, board_rvsched.py and board_run.py. VERSION gets an "rvsched" line.
#   compiler/scripts/deploy_rvsched.sh build/deploy_hfgen_qwen3_lc2048_pcpi [...]
#   board: python3 board_run.py rvsched   (or run_board_tests.sh: its rvsched tests)
# Before that, the aarch64 build's --rvsched is checked under qemu against the
# functional simulator on a 2-layer export (RVS_CHECK=<export dir>, default
# build/hfgen/smollm2_l2_e2e_vfix; RVS_CHECK= skips it).
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source "$(dirname "$0")/board_env.sh"
[ $# -ge 1 ] || { echo "usage: $0 <deploy dir> [...]" >&2; exit 2; }
make -C "$SA_REPO/firmware/rt" >/dev/null
"$SA_COMPILER/scripts/build_sa_runtime.sh" "$SA_RT_ARCH" | tail -1
A=$SA_RT_BUILD/runtime/plugins/hal/drivers/sa/sa-llm-run
CHK=${RVS_CHECK-$SA_REPO/build/hfgen/smollm2_l2_e2e_vfix}
if [ -n "$CHK" ]; then
  Q=$(command -v qemu-aarch64-static || echo "$SA_REPO/build/qemu/usr/bin/qemu-aarch64-static")
  T=$(mktemp -d /tmp/rvs_chk_XXXX)
  trap 'rm -rf "$T"' EXIT
  V=$("$SA_PY" - "$CHK" <<'PY'
import re, sys
m = re.search(r"func\.func @main\([^{]*-> !torch\.vtensor<\[1,1,(\d+)\],f32>", open(sys.argv[1] + "/qllama.mlir").read(1 << 22))
print(m.group(1) if m else 0)
PY
)
  "$SA_PY" "$SA_COMPILER/frontend/argmax_module.py" --vocab "$V" --out "$T/argmax.vmfb" | tail -1
  for m in ref rvs; do
    x=; [ $m = rvs ] && x=--rvsched
    "$SA_PY" "$SA_COMPILER/sim/sa_sim_server.py" --d "$SA_D" --mb 256 --socket "$T/s.sock" --once >/dev/null 2>&1 &
    sleep 3
    SA_SIM_SOCKET=$T/s.sock "$Q" "$A" --device=sa --module="$T/argmax.vmfb" --module="$CHK/sa.vmfb" \
      --parameters=model="$CHK/sa_packed.irpa" --abi=hfv --pad=16 --valid_len=256 --tokens=1,504,3290,282 \
      --generate=20 $x > "$T/$m.out" 2> "$T/$m.err" || { tail -5 "$T/$m.err"; exit 1; }
    wait || true
  done
  if [ "$(head -1 "$T/ref.out")" != "$(head -1 "$T/rvs.out")" ]; then
    echo "aarch64 --rvsched (qemu, sim): tokens differ"; head -1 "$T/ref.out" "$T/rvs.out"; exit 1
  fi
  echo "aarch64 --rvsched (qemu, sim): tokens identical to the host loop's ($(grep -o 'rvsched: [0-9]* buckets[^;]*' "$T/rvs.err"))"
fi
for D in "$@"; do
  V=$("$SA_PY" -c "import numpy as np; print(np.load('$D/expected_logits.npy', mmap_mode='r').shape[1])")
  "$SA_PY" "$SA_COMPILER/frontend/argmax_module.py" --vocab "$V" --out "$D/argmax.vmfb" | tail -1
  rm -f "$D/argmax.mlir"
  cp "$SA_REPO/firmware/rt/rt_fw.bin" "$SA_COMPILER/tests/board_rvsched.py" "$SA_COMPILER/tests/board_run.py" "$D/"
  llvm-strip-18 -o "$D/sa-llm-run" "$A"
  sed -i '/^rvsched /d' "$D/VERSION"
  echo "rvsched $(git -C "$SA_REPO" rev-parse --short HEAD)$(git -C "$SA_REPO" diff --quiet HEAD -- compiler firmware || echo ' (dirty)'): argmax.vmfb, rt_fw.bin v3, sa-llm-run" >> "$D/VERSION"
  echo "$D: plan B added (argmax.vmfb for $V logits)"
done
echo "to the board (only the changed files):"
for D in "$@"; do
  echo "  scp $D/{argmax.vmfb,rt_fw.bin,sa-llm-run,board_rvsched.py,board_run.py,VERSION} ubuntu@<board>:~/$(basename "$D")/"
done

#!/usr/bin/env bash
# A board bundle (board_llm.py's: sa.vmfb, sa_packed.irpa, prompt.npy,
# expected_logits.npy, sa_args.txt) run by the aarch64 sa-llm-run under qemu
# (user mode) against the x86 functional simulator: its logits must be
# bit-exact with the bundle's reference, as the board's. Dispatches left to the
# host run on the ARM there, as on the board: an ARM / x86 difference in their
# arithmetic shows here, not first on the board (the generic path's prefill had
# one, docs/generality_implementation_plan.md WP3).
#   compiler/scripts/check_bundle_aarch64.sh <bundle> [generate, default 2] [D, default SA_D or 8]
set -euo pipefail
source "$(dirname "$0")/../env.sh"
B=$1; N=${2:-2}; D=${3:-${SA_D:-8}}
A=$SA_REPO/build/iree/build-sa-aarch64/runtime/plugins/hal/drivers/sa/sa-llm-run
Q=$(command -v qemu-aarch64-static || echo "$SA_REPO/build/qemu/usr/bin/qemu-aarch64-static")
[ -x "$Q" ] || { echo "no qemu-aarch64-static (compiler/scripts/test_aarch64_runtime.sh fetches it)"; exit 2; }
[ -x "$A" ] || "$SA_COMPILER/scripts/build_sa_runtime.sh" aarch64 | tail -1
mb=$(awk '{print $NF}' "$B/board.txt" 2>/dev/null || echo 256)
p=$("$SA_PY" -c "import numpy as np;print(','.join(str(int(t)) for t in np.load('$B/prompt.npy')))")
S=/tmp/sa_a64_$$.sock
out=$(mktemp /tmp/sa_a64_XXXX.f32)
trap 'rm -f "$S" "$out"' EXIT
"$SA_PY" "$SA_COMPILER/sim/sa_sim_server.py" --d "$D" --mb "$mb" --socket "$S" --once >/dev/null 2>&1 &
sleep 3
SA_SIM_SOCKET=$S "$Q" "$A" --device=sa --module="$B/sa.vmfb" --parameters=model="$B/sa_packed.irpa" --tokens="$p" \
  --generate="$N" --logits_out="$out" $(cat "$B/sa_args.txt" 2>/dev/null) > /dev/null 2>&1 || true
wait || true
"$SA_PY" - "$B" "$out" <<'PY'
import os, sys
import numpy as np
b, f = sys.argv[1:]
want = np.load(f"{b}/expected_logits.npy")
args = open(f"{b}/sa_args.txt").read() if os.path.exists(f"{b}/sa_args.txt") else ""
if "--prefill=" in args:                         # the prompt's last position, then the generated steps
    want = want[len(np.load(f"{b}/prompt.npy")) - 1:]
got = np.fromfile(f, np.float32)
got = got.reshape(-1, want.shape[1]) if got.size else np.zeros((0, want.shape[1]), np.float32)
ok = sum(got[i].tobytes() == want[i].tobytes() for i in range(min(len(got), len(want))))
print(f"aarch64 (qemu) vs the sim: {ok}/{len(got)} logits rows bit-exact" + ("" if len(got) and ok == len(got) else " FAILED"))
sys.exit(0 if len(got) and ok == len(got) else 1)
PY

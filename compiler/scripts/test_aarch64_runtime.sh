#!/usr/bin/env bash
# Checks the aarch64 runtime (KV260, docs/kv260_upgrade_plan.md K1a) without the
# board: the static aarch64 binaries run under qemu-aarch64 (user mode) with the
# board transport, rt_fw's ring served by the functional simulator
# (compiler/sim/sa_board_emu.py):
#   1. sa_hal_test (C1): dispatches, binding offsets, a device fault, a stress run
#   2. sa-llm-run, stories15M decode (build/deploy_z1/c3): logits bit-exact with the C3 reference
#   3. sa-llm-run, stories15M prefill + decode (build/deploy_z1/c6p_stories): bit-exact with the sim
# The bundles come from compiler/scripts/deploy_z1_freeze.sh. qemu-aarch64-static
# is used from PATH, else fetched without root (apt-get download) into build/qemu.
#   compiler/scripts/test_aarch64_runtime.sh [steps, default 8]
set -euo pipefail
source "$(dirname "$0")/../env.sh"
N=${1:-8}
A=$SA_REPO/build/iree/build-sa-aarch64/runtime/plugins/hal/drivers/sa
Z=$SA_REPO/build/deploy_z1
W=$SA_REPO/build/aarch64_test
mkdir -p "$W"
Q=$(command -v qemu-aarch64-static || true)
if [ -z "$Q" ]; then
  Q=$SA_REPO/build/qemu/usr/bin/qemu-aarch64-static
  if [ ! -x "$Q" ]; then
    (cd "$W" && rm -f qemu-user-static*.deb && apt-get download qemu-user-static >/dev/null)
    dpkg-deb -x "$W"/qemu-user-static*.deb "$SA_REPO/build/qemu"
  fi
fi
[ -x "$A/sa-llm-run" ] || "$SA_COMPILER/scripts/build_sa_runtime.sh" aarch64 | tail -1
emu() { "$SA_PY" "$SA_COMPILER/sim/sa_board_emu.py" --d 8 --file "/dev/shm/sa_devmem_a64_$$" -- "$Q" "$@"; }
trap 'rm -f /dev/shm/sa_devmem_a64_$$' EXIT
fail=0
echo "== 1. sa_hal_test (C1)"
"$SA_PY" "$SA_COMPILER/runtime/tools/make_test_exec.py" --d 8 --out "$W/test_d8" | tail -1
emu "$A/sa_hal_test" "$W/test_d8" 20 | grep -E "PASS|FAIL|differ|wrong" | tail -3 | grep -q "C1 HAL test PASS" \
  && echo "C1 HAL test PASS" || { echo "C1 HAL test FAILED"; fail=1; }
check() {                      # <name> <bundle>: N generated tokens, logits vs expected_logits.npy
  local name=$1 b=$Z/$2
  [ -d "$b" ] || { echo "$b missing (compiler/scripts/deploy_z1_freeze.sh)"; fail=1; return; }
  local p
  p=$("$SA_PY" -c "import numpy as np;print(','.join(str(int(t)) for t in np.load('$b/prompt.npy')))")
  emu "$A/sa-llm-run" --device=sa --module="$b/sa.vmfb" --parameters=model="$b/sa_packed.irpa" --tokens="$p" \
    --generate="$N" --logits_out="$W/$name.f32" $(cat "$b/sa_args.txt" 2>/dev/null) > "$W/$name.log" 2>&1 || true
  "$SA_PY" - "$b" "$W/$name.f32" "$name" <<'EOF' || fail=1
import sys
import numpy as np
b, f, name = sys.argv[1:]
want = np.load(f"{b}/expected_logits.npy")
args = open(f"{b}/sa_args.txt").read() if __import__("os").path.exists(f"{b}/sa_args.txt") else ""
if "--prefill=" in args:                         # the prompt's last position, then the generated steps
    want = want[len(np.load(f"{b}/prompt.npy")) - 1:]
got = np.fromfile(f, np.float32).reshape(-1, want.shape[1]) if __import__("os").path.exists(f) else np.zeros((0, 1))
n = len(got)
ok = sum(got[i].tobytes() == want[i].tobytes() for i in range(min(n, len(want))))
print(f"{name}: {ok}/{n} logits rows bit-exact" + ("" if n and ok == n else " FAILED"))
sys.exit(0 if n and ok == n else 1)
EOF
}
echo "== 2. sa-llm-run, stories15M decode"
check c3 c3
echo "== 3. sa-llm-run, stories15M prefill + decode"
check c6p_stories c6p_stories
[ $fail = 0 ] && echo "aarch64 runtime PASS" || { echo "aarch64 runtime FAILED (logs in $W)"; exit 1; }

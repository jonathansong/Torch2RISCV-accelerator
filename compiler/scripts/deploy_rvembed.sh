#!/usr/bin/env bash
# Adds S4 of the RISC-V backend (docs/riscv_backend_design.md §9.3: the rv32
# code compiled by iree-compile itself and embedded in sa.vmfb, --iree-sa-riscv)
# to a staged hfgen bundle: its sa.vmfb replaced by the same model compiled with
# the code table (every executable's descriptor part checked identical to the
# bundle's, so the other tests are unaffected: without SA_RV=embedded the
# loader ignores the code), rt_fw.bin version 5, the aarch64 sa-llm-run, the
# board scripts. The previous sa.vmfb is kept as sa.vmfb.descriptors.
#   compiler/scripts/deploy_rvembed.sh <deploy dir> <dir compiled with SA_COMPILE_FLAGS=--iree-sa-riscv>
#   board: python3 board_run.py rvcode --embedded; python3 board_run.py rvsched --rv --embedded
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source "$(dirname "$0")/board_env.sh"
D=$1 C=$2
[ -s "$C/sa.vmfb" ] || { echo "no $C/sa.vmfb" >&2; exit 1; }
"$SA_PY" - "$D/sa.vmfb" "$C/sa.vmfb" <<'PY'
import struct, sys
def blobs(p):
    """each executable: its descriptor part (to the extensions' end, the code table word cleared), code bytes"""
    d, out, i = open(p, "rb").read(), [], 0
    while (i := d.find(b"SADESC1\0", i)) >= 0:
        h = struct.unpack_from("<IIIIIIIIIII", d, i + 8)
        n, ext, tab = h[3], h[9], h[10]
        desc = bytearray(d[i:i + ext + 32 * n])
        desc[48:52] = b"\0\0\0\0"
        code = sum(struct.unpack_from("<II", d, i + tab + 8 * k)[1] for k in range(n)) if tab else 0
        out.append((bytes(desc), code, n))
        i += 8
    return out
a, b = blobs(sys.argv[1]), blobs(sys.argv[2])
if sorted(x[0] for x in a) != sorted(x[0] for x in b):
    sys.exit(f"the executables' descriptor parts differ ({len(a)} in the bundle, {len(b)} in the new compile)")
exports = sum(x[2] for x in b)
code = sum(x[1] for x in b)
if not code:
    sys.exit("the new compile has no RISC-V code (SA_COMPILE_FLAGS=--iree-sa-riscv)")
print(f"{len(b)} executables, descriptor parts identical to the bundle's; {exports} exports, "
      f"{code} bytes of embedded rv32 code")
PY
make -C "$SA_REPO/firmware/rt" >/dev/null
"$SA_COMPILER/scripts/build_sa_runtime.sh" "$SA_RT_ARCH" | tail -1
[ -f "$D/sa.vmfb.descriptors" ] || cp "$D/sa.vmfb" "$D/sa.vmfb.descriptors"
cp "$C/sa.vmfb" "$D/sa.vmfb"
cp "$SA_REPO/firmware/rt/rt_fw.bin" "$SA_COMPILER/tests/board_rvcode.py" "$SA_COMPILER/tests/board_rvsched.py" "$SA_COMPILER/tests/board_rvperf.py" \
   "$SA_COMPILER/tests/board_run.py" "$D/"
llvm-strip-18 -o "$D/sa-llm-run" "$SA_RT_BUILD/runtime/plugins/hal/drivers/sa/sa-llm-run"
sed -i '/^rvembed /d' "$D/VERSION"
echo "rvembed $(git -C "$SA_REPO" rev-parse --short HEAD)$(git -C "$SA_REPO" diff --quiet HEAD -- compiler firmware || echo ' (dirty)'): sa.vmfb with embedded rv32 code (--iree-sa-riscv), rt_fw.bin v5, sa-llm-run" >> "$D/VERSION"
echo "$D: S4 added (sa.vmfb with the code table; the previous one in sa.vmfb.descriptors)"
echo "to the board: scp $D/{sa.vmfb,rt_fw.bin,sa-llm-run,board_rvcode.py,board_rvsched.py,board_run.py,VERSION} ubuntu@<board>:~/$(basename "$D")/"

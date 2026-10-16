#!/usr/bin/env bash
# Adds plan A of the RISC-V backend (every dispatch as generated RISC-V code,
# docs/riscv_backend_design.md §9.4) to a staged hfgen bundle: the model's code
# image (rv/rvcode.bin + rvcode.map, compiler/tests/rv_image.py), rt_fw.bin
# version 4 (RT_CALL), the aarch64 sa-llm-run with SA_RV, board_rvcode.py and
# board_run.py. The image must come from the same executables as the bundle's
# sa.vmfb (checked: the sa-desc executables compared).
#   compiler/scripts/deploy_rvcode.sh <deploy dir> <dir compiled with --iree-sa-emit-riscv (sa.vmfb, rv/)>
#   board: python3 board_run.py rvcode
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source "$(dirname "$0")/board_env.sh"
D=$1 C=$2
[ -f "$C/rv/rvcode.bin" ] || { echo "no $C/rv/rvcode.bin (compiler/tests/rv_image.py)" >&2; exit 1; }
"$SA_PY" - "$D/sa.vmfb" "$C/sa.vmfb" <<'PY'
import struct, sys
def blobs(p):
    d, out, i = open(p, "rb").read(), [], 0
    while (i := d.find(b"SADESC1\0", i)) >= 0:
        h = struct.unpack_from("<IIIIIIIIII", d, i + 8)
        out.append(d[i:i + h[9] + 32 * h[3]])
        i += 8
    return sorted(out)
a, b = blobs(sys.argv[1]), blobs(sys.argv[2])
if a != b:
    sys.exit(f"the bundle's executables ({len(a)}) differ from the image's compile ({len(b)})")
print(f"{len(a)} executables: the bundle's and the image's compile identical")
PY
make -C "$SA_REPO/firmware/rt" >/dev/null
"$SA_COMPILER/scripts/build_sa_runtime.sh" "$SA_RT_ARCH" | tail -1
mkdir -p "$D/rv"
cp "$C/rv/rvcode.bin" "$C/rv/rvcode.map" "$D/rv/"
cp "$SA_REPO/firmware/rt/rt_fw.bin" "$SA_COMPILER/tests/board_rvcode.py" "$SA_COMPILER/tests/board_run.py" "$D/"
llvm-strip-18 -o "$D/sa-llm-run" "$SA_RT_BUILD/runtime/plugins/hal/drivers/sa/sa-llm-run"
sed -i '/^rvcode /d' "$D/VERSION"
echo "rvcode $(git -C "$SA_REPO" rev-parse --short HEAD)$(git -C "$SA_REPO" diff --quiet HEAD -- compiler firmware || echo ' (dirty)'): rv/rvcode.bin $(stat -c %s "$D/rv/rvcode.bin") bytes, rt_fw.bin v4, sa-llm-run" >> "$D/VERSION"
echo "$D: plan A added ($(wc -l < "$D/rv/rvcode.map") functions)"
echo "to the board: scp -r $D/{rv,rt_fw.bin,sa-llm-run,board_rvcode.py,board_run.py,VERSION} ubuntu@<board>:~/$(basename "$D")/"

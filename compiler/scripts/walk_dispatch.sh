#!/usr/bin/env bash
# One dispatch of a compiled model through the sa backend, stage by stage
# (docs/gemv_walkthrough.md): its executable source (the export's sa_sources/,
# dumped by compile_sa.sh) recompiled alone with the IR printed after each sa
# pass, its sa-desc binary checked byte for byte against the one in sa.vmfb,
# and the binary disassembled.
#   compiler/scripts/walk_dispatch.sh build/hfgen/qwen3_lc2048_pcpi 7 build/walk
# Writes to <out>: d<N>_exe_sa.mlir (the source, sa variant only), after_<pass>.mlir
# (iree-sa-to-sahl, iree-sahl-tile, iree-sahl-schedule, iree-sahl-expand-kernels,
# iree-sahl-to-sahw, iree-sahw-fuse-ve), d<N>.sadesc (from sa.vmfb), d<N>.dis.txt (sadis.py).
set -euo pipefail
source "$(dirname "$0")/../env.sh"
E=$1 N=$2 O=${3:-$SA_REPO/build/walk}
mkdir -p "$O"
SRC="$E/sa_sources/module_main\$async_dispatch_$N.mlir"
[ -f "$SRC" ] || { echo "no $SRC (compile_sa.sh dumps the sources)" >&2; exit 1; }
"$SA_PY" - "$SRC" "$O/d${N}_exe_sa.mlir" <<'PY'
import sys
t = open(sys.argv[1]).read()
a = t.find("  hal.executable.variant public @vmvx")     # the host fallback's variant, if any
open(sys.argv[2], "w").write(t if a < 0 else t[:a].rstrip() + "\n}\n")
PY
PASSES=(iree-sa-to-sahl iree-sahl-tile iree-sahl-schedule iree-sahl-expand-kernels iree-sahl-to-sahw iree-sahw-fuse-ve)
FLAGS=(--iree-hal-target-device=sa "--iree-sa-d=$SA_D" "--iree-sa-gemv-ports=$SA_GEMV_PORTS")
[ -f "$E/compile_flags.txt" ] && read -r -a CF < "$E/compile_flags.txt" && FLAGS+=("${CF[@]}")
"$SA_REPO/build/iree/build-compiler/tools/iree-compile" --compile-mode=hal-executable "$O/d${N}_exe_sa.mlir" \
  "${FLAGS[@]}" -o "$O/d${N}_alone.sadesc" --mlir-disable-threading \
  $(printf -- '--mlir-print-ir-after=%s ' "${PASSES[@]}") 2> "$O/d${N}_passes.txt"
"$SA_PY" - "$O" "$N" "$E/sa.vmfb" <<'PY'
import re, struct, sys
o, n, vmfb = sys.argv[1:]
t = open(f"{o}/d{n}_passes.txt").read()
parts = re.split(r"// -----// IR Dump After .*?\((iree-[a-z0-9-]+)\) //----- //\n", t)
for name, body in zip(parts[1::2], parts[2::2]):
    open(f"{o}/after_{name}.mlir", "w").write(body)
alone = open(f"{o}/d{n}_alone.sadesc", "rb").read()
name = re.search(rb"main\$async_dispatch_" + n.encode() + rb"_\w+", alone).group(0)
d = open(vmfb, "rb").read()
i = 0
while True:                                        # the executable in sa.vmfb holding this export
    i = d.find(b"SADESC1\0", i)
    if i < 0:
        sys.exit(f"{name.decode()} not in {vmfb}")
    so, sb = struct.unpack_from("<II", d, i + 36)
    if name in d[i + so:i + so + sb]:
        break
    i += 8
blob = d[i:i + len(alone)]
open(f"{o}/d{n}.sadesc", "wb").write(blob)
print(f"{name.decode()}: {len(alone)} bytes, {'identical to' if blob == alone else 'DIFFERENT from'} the one in sa.vmfb")
PY
"$SA_PY" "$SA_COMPILER/runtime/tools/sadis.py" "$O/d$N.sadesc" | tee "$O/d$N.dis.txt"

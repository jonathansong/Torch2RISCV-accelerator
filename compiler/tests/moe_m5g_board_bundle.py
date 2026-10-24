#!/usr/bin/env python3
"""Stage a KV260 test of compiler-generated integer MoE route code."""

import argparse
import hashlib
from pathlib import Path
import shutil
import struct
import subprocess

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "moe_route_top2_export.mlir"
COMPILE = REPO / "build/iree/build-compiler/tools/iree-compile"
M5D = REPO / "build/moe/m5d/board"
OVERLAY = REPO / "boards/kv260/build/output/d16_100mhz_w128_p6_gemv_np4_pcpi"


def stage(out):
    if not (M5D / "VERSION").is_file():
        raise RuntimeError("stage M5d first: compiler/tests/moe_m5d_board_bundle.py")
    out.mkdir(parents=True, exist_ok=True)
    binaries = out / "compiler"
    binaries.mkdir(exist_ok=True)
    subprocess.run([str(COMPILE), str(SOURCE), "--compile-from=executable-configurations",
                    "--iree-sa-riscv", f"--iree-hal-dump-executable-binaries-to={binaries}",
                    "-o", str(out / "route.vmfb")], check=True)
    binary = next(binaries.glob("*.sadesc")).read_bytes()
    extension, table = struct.unpack_from("<II", binary, 44)
    if struct.unpack_from("<I", binary, extension + 20)[0] & 2 != 2:
        raise RuntimeError("route export is not RISC-V-only")
    offset, count = struct.unpack_from("<II", binary, table)
    if not count or offset + count > len(binary):
        raise RuntimeError("missing embedded route code")
    code = binary[offset:offset + count]
    (out / "moe_route_from_top2.bin").write_bytes(code)
    for source in (OVERLAY / "picorv32.bit", OVERLAY / "picorv32.hwh",
                   REPO / "firmware/rt/rt_fw.bin", REPO / "driver/pynq_matmul.py",
                   HERE / "board_moe_route_from_top2.py"):
        shutil.copy2(source, out / source.name)
    for name in ("ascending", "ties", "random"):
        case = out / name
        case.mkdir(exist_ok=True)
        for tensor in ("values_ref.npy", "indices_ref.npy"):
            shutil.copy2(M5D / name / tensor, case / tensor)
    run = out / "run_board.sh"
    run.write_text("""#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then exec sudo "$0" "$@"; fi
for f in /etc/profile.d/pynq_venv.sh /etc/profile.d/xrt_setup.sh /opt/xilinx/xrt/setup.sh; do
  if [ -f "$f" ]; then source "$f" >/dev/null; fi
done
cd "$(dirname "$0")"
exec python3 board_moe_route_from_top2.py "$@"
""")
    run.chmod(0o755)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    (out / "VERSION").write_text(f"commit {revision}\noverlay {OVERLAY.name}\n"
                                 f"compiled route code {count} bytes sha256 {hashlib.sha256(code).hexdigest()}\n")
    print(f"staged {out}: {count} compiler-generated rv32 bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "build/moe/m5g/board")
    stage(parser.parse_args().out)

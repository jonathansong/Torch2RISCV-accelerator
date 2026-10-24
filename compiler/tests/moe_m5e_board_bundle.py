#!/usr/bin/env python3
"""Stage the D=16 native Top-2 to M4 route bridge for KV260."""

import argparse
from pathlib import Path
import shutil
import subprocess

import moe_route_from_top2_kernel

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
M5D = REPO / "build/moe/m5d/board"
OVERLAY = REPO / "boards/kv260/build/output/d16_100mhz_w128_p6_gemv_np4_pcpi"


def stage(out):
    if not (M5D / "VERSION").is_file():
        raise RuntimeError("stage M5d first: compiler/tests/moe_m5d_board_bundle.py")
    out.mkdir(parents=True, exist_ok=True)
    code, _ = moe_route_from_top2_kernel.build(str(out))
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
                                 f"route bridge {len(code)} rv32 bytes\nsource {M5D / 'VERSION'}\n")
    print(f"staged {out} ({len(code)} rv32 bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "build/moe/m5e/board")
    stage(parser.parse_args().out)

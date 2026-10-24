#!/usr/bin/env python3
"""Stage a KV260 chain: compiler Top-2, route bridge, then M4 FFN."""

import argparse
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
M5D = REPO / "build/moe/m5d/board"
M5E = REPO / "build/moe/m5e/board"


def stage(out):
    for source in (M5D / "router.vmfb", M5D / "iree-run-module",
                   M5E / "moe_route_from_top2.bin"):
        if not source.is_file():
            raise RuntimeError(f"missing {source}; stage M5d/M5e and build aarch64 runtime first")
    out.mkdir(parents=True, exist_ok=True)
    sources = [M5D / name for name in (
        "router.vmfb", "iree-run-module", "picorv32.bit", "picorv32.hwh", "rt_fw.bin",
        "pynq_matmul.py", "board_launcher.py")]
    sources += [M5E / "moe_route_from_top2.bin",
                HERE / "board_moe_iree_ffn.py", HERE / "board_moe_ffn.py",
                REPO / "compiler/runtime/tools/moe_m3b.py",
                REPO / "compiler/runtime/tools/sadesc.py",
                REPO / "llm/compile_layer.py"]
    for source in sources:
        shutil.copy2(source, out / source.name)
    for name in ("ascending", "ties", "random"):
        case = out / name
        case.mkdir(exist_ok=True)
        for tensor in ("x.npy", "w.npy", "values_ref.npy", "indices_ref.npy"):
            shutil.copy2(M5D / name / tensor, case / tensor)
    run = out / "run_board.sh"
    run.write_text("""#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then exec sudo "$0" "$@"; fi
for f in /etc/profile.d/pynq_venv.sh /etc/profile.d/xrt_setup.sh /opt/xilinx/xrt/setup.sh; do
  if [ -f "$f" ]; then source "$f" >/dev/null; fi
done
cd "$(dirname "$0")"
exec python3 board_moe_iree_ffn.py "$@"
""")
    run.chmod(0o755)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    (out / "VERSION").write_text(f"commit {revision}\nsource M5d {M5D / 'VERSION'}\n"
                                 f"source M5e {M5E / 'VERSION'}\n")
    print(f"staged {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "build/moe/m5f/board")
    stage(parser.parse_args().out)

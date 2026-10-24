#!/usr/bin/env python3
"""Make a five-output diagnostic entry from the failed M5t model VMFB."""

from pathlib import Path
import shutil
import subprocess

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "build/moe/m5k/real-layer0/moe_model.mlir"
OUT = REPO / "build/moe/m5t/debug"
COMPILE = REPO / "build/iree/build-compiler/tools/iree-compile"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ir = SOURCE.read_text()
    old = "util.func public @main(%arg0: !hal.buffer_view) -> !hal.buffer_view {"
    new = ("util.func public @main(%arg0: !hal.buffer_view) -> "
           "(!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, "
           "!hal.buffer_view, !hal.buffer_view) {")
    if ir.count(old) != 1 or ir.count("util.return %2#4 : !hal.buffer_view") != 1:
        raise ValueError("model HAL layout changed; inspect the debug transform")
    ir = ir.replace(old, new).replace(
        "util.return %2#4 : !hal.buffer_view",
        "util.return %2#0, %2#1, %2#2, %2#3, %2#4 : "
        "!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, "
        "!hal.buffer_view, !hal.buffer_view")
    source = OUT / "moe_model_debug.mlir"
    source.write_text(ir)
    subprocess.run([str(COMPILE), str(source), "--compile-from=hal", "-o",
                    str(OUT / "moe_model_debug.vmfb")], check=True)
    shutil.copy2(Path(__file__).with_name("board_moe_m5t_diagnose.py"), OUT)
    reference = REPO / "build/moe/m4c/tiny-mixtral/layer0/moe_token_acceptance.npz"
    with np.load(reference) as data:
        np.save(OUT / "scores_expected.npy", data["scores"][0])
        np.save(OUT / "ids_expected.npy", data["pairs"][0])
    run = OUT / "run_diagnose.sh"
    run.write_text('''#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then exec sudo "$0" "$@"; fi
for f in /etc/profile.d/pynq_venv.sh /etc/profile.d/xrt_setup.sh /opt/xilinx/xrt/setup.sh; do
  if [ -f "$f" ]; then source "$f" >/dev/null; fi
done
cd "$(dirname "$0")"
exec python3 board_moe_m5t_diagnose.py "$@"
''')
    run.chmod(0o755)
    print(f"diagnostic files staged in {OUT}")


if __name__ == "__main__":
    main()

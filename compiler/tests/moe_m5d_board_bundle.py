#!/usr/bin/env python3
"""Stage the D=16 IREE MoE Top-2 KV260 acceptance bundle."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess

import numpy as np

from test_moe_top2_raw import topk_reference

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OVERLAY = REPO / "boards/kv260/build/output/d16_100mhz_w128_p6_gemv_np4_pcpi"
RUNTIME = REPO / "build/iree/build-sa-aarch64/tools/iree-run-module"
COMPILE = REPO / "build/iree/build-compiler/tools/iree-compile"


def stage(out, runtime):
    out.mkdir(parents=True, exist_ok=True)
    source = HERE / "moe_router_w8_top2_d16.mlir"
    command = [str(COMPILE), str(source), "--iree-hal-target-device=sa", "--iree-sa-form-moe",
               "--iree-sa-riscv", "--iree-sa-d=16", "--iree-sa-gemv-ports=4",
               "--iree-hal-link-executables=false", "-o", str(out / "router.vmfb")]
    subprocess.run(command, check=True)
    for src in (OVERLAY / "picorv32.bit", OVERLAY / "picorv32.hwh", REPO / "firmware/rt/rt_fw.bin",
                REPO / "compiler/runtime/test/board_launcher.py", REPO / "driver/pynq_matmul.py",
                HERE / "board_moe_compiler_top2.py"):
        shutil.copy2(src, out / src.name)
    if runtime.is_file():
        shutil.copy2(runtime, out / "iree-run-module")

    rng = np.random.default_rng(340)
    cases = {}
    for name in ("ascending", "ties", "random"):
        x = np.zeros(32, dtype=np.int8)
        w = np.zeros((1, 32, 16), dtype=np.int8)
        if name == "ascending":
            x[0] = 1
            w[0, 0] = np.arange(-8, 8, dtype=np.int8)
        elif name == "ties":
            x[0] = 1
            w[0, 0] = np.array([5, 5, -2, 5, 1, 0, -1, 2, 3, 4, 5, -3, 2, 1, 0, -4], dtype=np.int8)
        else:
            x[:] = rng.integers(-4, 5, 32, dtype=np.int8)
            w[:] = rng.integers(-4, 5, w.shape, dtype=np.int8)
        scores = (x.astype(np.int32) @ w[0].astype(np.int32)).astype(np.float32)
        values, indices = topk_reference(scores)
        case = out / name
        case.mkdir(exist_ok=True)
        np.save(case / "x.npy", x)
        np.save(case / "w.npy", w)
        np.save(case / "values_ref.npy", values.reshape(1, 2))
        np.save(case / "indices_ref.npy", indices.reshape(1, 2))
        cases[name] = indices.tolist()

    (out / "run_board.sh").write_text("""#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then exec sudo "$0" "$@"; fi
for f in /etc/profile.d/pynq_venv.sh /etc/profile.d/xrt_setup.sh /opt/xilinx/xrt/setup.sh; do
  if [ -f "$f" ]; then source "$f" >/dev/null; fi
done
cd "$(dirname "$0")"
exec python3 board_moe_compiler_top2.py "$@"
""")
    (out / "run_board.sh").chmod(0o755)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    digest = hashlib.sha256((out / "router.vmfb").read_bytes()).hexdigest()
    (out / "VERSION").write_text(f"commit {revision}\nrouter.vmfb sha256 {digest}\n"
                                 f"overlay {OVERLAY.name}\nexpected IDs {cases}\n")
    print(f"staged {out}, runtime {'present' if runtime.is_file() else 'missing'}")
    if not runtime.is_file():
        print("Build with compiler/scripts/build_sa_runtime.sh aarch64, then copy its iree-run-module into the bundle.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "build/moe/m5d/board")
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    args = parser.parse_args()
    stage(args.out, args.runtime)

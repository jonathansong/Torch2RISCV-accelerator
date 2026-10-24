#!/usr/bin/env python3
"""Stage a KV260 bundle for a Torch-exported synthetic MoE layer."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "compiler/frontend"))
import moe  # noqa: E402
from moe_native import NativeMixtralMoe, export  # noqa: E402

COMPILE = REPO / "build/iree/build-compiler/tools/iree-compile"
OVERLAY = REPO / "boards/kv260/build/output/d16_100mhz_w128_p6_gemv_np4_pcpi"


def stage(out):
    config = moe.MoeConfig(32, 64, 4, 2)
    weights = moe.synthetic_weights(config, seed=94)
    out.mkdir(parents=True, exist_ok=True)
    mlir, irpa = export(NativeMixtralMoe(weights), str(out))
    target = out / "moe_layer.vmfb"
    result = subprocess.run([
        str(COMPILE), mlir, "--iree-hal-target-device=sa",
        f"--iree-parameter-import=model={irpa}", "--iree-sa-form-moe",
        "--iree-sa-riscv", "--iree-sa-host-fallback", "--iree-sa-d=16",
        "--iree-sa-gemv-ports=4", "--iree-hal-link-executables=false",
        "--iree-opt-const-expr-max-size-increase-threshold=0",
        "--iree-sa-codegen-report", "-o", str(target)], capture_output=True, text=True)
    if result.returncode or "topk_1x4xf32: ok" not in result.stderr:
        raise RuntimeError(f"native Top-2 compilation failed:\n{result.stderr}")
    for source in (OVERLAY / "picorv32.bit", OVERLAY / "picorv32.hwh",
                   REPO / "firmware/rt/rt_fw.bin", REPO / "driver/pynq_matmul.py",
                   REPO / "compiler/runtime/test/board_launcher.py",
                   HERE / "board_moe_native_layer.py",
                   REPO / "build/iree/build-sa-aarch64/tools/iree-run-module"):
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, out / source.name)
    rng = np.random.default_rng(714)
    vectors = {
        "ramp": np.linspace(-0.625, 0.75, config.dim, dtype=np.float32),
        "zero": np.zeros(config.dim, dtype=np.float32),
        "random": rng.normal(size=config.dim).astype(np.float32),
    }
    for name, x in vectors.items():
        path = out / name
        path.mkdir(exist_ok=True)
        np.save(path / "x.npy", x[None])
        output, scores, ids, routing = moe.moe_forward(x, config, weights)
        for i, value in enumerate((output[None], scores[None], ids[None].astype(np.int64),
                                   routing[None])):
            np.save(path / f"ref_{i}.npy", value)
    script = out / "run_board.sh"
    script.write_text("""#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then exec sudo "$0" "$@"; fi
for f in /etc/profile.d/pynq_venv.sh /etc/profile.d/xrt_setup.sh /opt/xilinx/xrt/setup.sh; do
  if [ -f "$f" ]; then source "$f" >/dev/null; fi
done
cd "$(dirname "$0")"
exec python3 board_moe_native_layer.py "$@"
""")
    script.chmod(0o755)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    (out / "VERSION").write_text(f"commit {revision}\noverlay {OVERLAY.name}\n"
                                 f"vmfb sha256 {hashlib.sha256(target.read_bytes()).hexdigest()}\n"
                                 "Torch-exported synthetic layer; Top-2 SA, other operations may fallback\n")
    print(f"staged {out}: {target.stat().st_size} VMFB bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "build/moe/m5h/board")
    stage(parser.parse_args().out)

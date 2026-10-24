#!/usr/bin/env python3
"""Stage real Mixtral compiled route plus shared FFN for KV260."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OVERLAY = REPO / "boards/kv260/build/output/d16_100mhz_w128_p6_gemv_np4_pcpi"
DEFAULT_BUNDLE = REPO / "build/moe/m5k/real-layer0"
DEFAULT_REFERENCE = REPO / "build/moe/m4c/tiny-mixtral/layer0"


def stage(bundle, reference, out):
    bundle, reference, out = Path(bundle), Path(reference), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sources = [bundle / name for name in (
        "moe_params.bin", "moe_manifest.json", "moe_route.bin", "moe_ffn.sadesc")]
    sources += [reference / name for name in (
        "moe_acceptance.npz", "moe_token_acceptance.npz")]
    sources += [OVERLAY / "picorv32.bit", OVERLAY / "picorv32.hwh",
                REPO / "firmware/rt/rt_fw.bin", REPO / "driver/pynq_matmul.py",
                REPO / "compiler/runtime/tools/sadesc.py",
                HERE / "board_moe_compiled_ffn.py"]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, out / source.name)
    run = out / "run_board.sh"
    run.write_text("""#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then exec sudo "$0" "$@"; fi
for f in /etc/profile.d/pynq_venv.sh /etc/profile.d/xrt_setup.sh /opt/xilinx/xrt/setup.sh; do
  if [ -f "$f" ]; then source "$f" >/dev/null; fi
done
cd "$(dirname "$0")"
exec python3 board_moe_compiled_ffn.py "$@"
""")
    run.chmod(0o755)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    route_hash = hashlib.sha256((out / "moe_route.bin").read_bytes()).hexdigest()
    (out / "VERSION").write_text(f"commit {revision}\nsource {bundle}\n"
                                 f"route sha256 {route_hash}\n")
    print(f"staged {out}: compiler-generated route and reusable FFN, five real-layer cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, default=REPO / "build/moe/m5m/board")
    args = parser.parse_args()
    stage(args.bundle, args.reference, args.out)

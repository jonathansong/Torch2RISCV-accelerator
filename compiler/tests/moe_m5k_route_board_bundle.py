#!/usr/bin/env python3
"""Stage a small KV260 check of the model-derived compiler route export."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OVERLAY = REPO / "boards/kv260/build/output/d16_100mhz_w128_p6_gemv_np4_pcpi"


def stage(bundle, reference, out):
    bundle, reference, out = Path(bundle), Path(reference), Path(out)
    manifest = json.loads((bundle / "compiler_bundle.json").read_text())
    fields = manifest["recognized"]
    if fields["experts"] != 8 or manifest["d"] != 16:
        raise ValueError("this real Mixtral board fixture requires eight experts and D=16")
    vectors = np.load(reference)
    ids = vectors["pairs"][0].astype(np.int32)
    scores = vectors["scores"][0].astype(np.float32)
    weights = vectors["routing"][0].astype(np.float32)
    if ids.shape != (2,) or scores.shape != (8,) or weights.shape != (2,):
        raise ValueError("invalid M4 real-token route reference")
    cases = {
        "real_token": {"values": scores[ids].tolist(), "ids": ids.tolist(),
                       "weights": weights.tolist()},
        "ties": {"values": [5.0, 5.0], "ids": [0, 1], "weights": [0.5, 0.5]},
        "far_expert": {"values": [3.0, 1.0], "ids": [7, 0],
                       "weights": [0.8807970881462097, 0.11920291930437088]},
    }
    out.mkdir(parents=True, exist_ok=True)
    for source in (bundle / "moe_route.bin", OVERLAY / "picorv32.bit",
                   OVERLAY / "picorv32.hwh", REPO / "firmware/rt/rt_fw.bin",
                   REPO / "driver/pynq_matmul.py", HERE / "board_moe_model_route.py"):
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, out / source.name)
    (out / "cases.json").write_text(json.dumps({"experts": fields["experts"],
        "expert_stride": fields["expert_stride"], "cases": cases}, indent=2) + "\n")
    run = out / "run_board.sh"
    run.write_text("""#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then exec sudo "$0" "$@"; fi
for f in /etc/profile.d/pynq_venv.sh /etc/profile.d/xrt_setup.sh /opt/xilinx/xrt/setup.sh; do
  if [ -f "$f" ]; then source "$f" >/dev/null; fi
done
cd "$(dirname "$0")"
exec python3 board_moe_model_route.py "$@"
""")
    run.chmod(0o755)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    (out / "VERSION").write_text(f"commit {revision}\nsource {bundle}\n"
                                 f"model-derived route experts {fields['experts']} "
                                 f"stride {fields['expert_stride']}\n")
    print(f"staged {out}: {len(cases)} cases, stride {fields['expert_stride']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path,
                        default=REPO / "build/moe/m5k/real-layer0")
    parser.add_argument("--reference", type=Path,
                        default=REPO / "build/moe/m4c/tiny-mixtral/layer0/moe_token_acceptance.npz")
    parser.add_argument("--out", type=Path, default=REPO / "build/moe/m5k/board-route")
    args = parser.parse_args()
    stage(args.bundle, args.reference, args.out)

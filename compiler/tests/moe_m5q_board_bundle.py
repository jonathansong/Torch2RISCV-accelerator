#!/usr/bin/env python3
"""Stage the real Mixtral layer-0 router-to-FFN single-VMFB KV260 test."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "compiler/frontend"))
import moe  # noqa: E402

OVERLAY = REPO / "boards/kv260/build/output/d16_100mhz_w128_p6_gemv_np4_pcpi"
RUNTIME = REPO / "build/iree/build-sa-aarch64/tools/iree-run-module"
BUNDLE = REPO / "build/moe/m5k/real-layer0"
REFERENCE = REPO / "build/moe/m4c/tiny-mixtral/layer0/moe_token_acceptance.npz"
WINDOW_MB = 128


def stage(out, bundle=BUNDLE, reference=REFERENCE, runtime=RUNTIME):
    out, bundle = Path(out).resolve(), Path(bundle).resolve()
    manifest = json.loads((bundle / "moe_manifest.json").read_text())
    if manifest["d"] != 16 or manifest["experts"] != 8 or manifest["dim"] != 1024:
        raise ValueError("expected the D=16, eight-expert tiny Mixtral layer")
    if manifest["total_bytes"] + (16 << 20) > (WINDOW_MB << 20):
        raise ValueError("packed parameters leave insufficient room in the 128 MiB window")
    router = np.load(bundle / "moe_router.npy")
    if router.shape != (8, 1024) or router.dtype != np.float32:
        raise ValueError("expected model-derived FP32 router weights")
    with np.load(reference) as data:
        x = np.asarray(data["x"], dtype=np.float32).reshape(1, 1024)
        scores = np.asarray(data["scores"], dtype=np.float32).reshape(1, 8)
        ids = np.asarray(data["pairs"], dtype=np.int32).reshape(1, 2)
        expected = np.asarray(data["output"], dtype=np.float32).reshape(1, 1024)
    values = np.take_along_axis(scores, ids, axis=1)
    router_scores = router @ x.reshape(-1)
    if not np.allclose(router_scores, scores.reshape(-1), rtol=2e-4, atol=2e-5):
        raise ValueError("reference router scores do not match the checkpoint")
    selected, _ = moe.route(scores.reshape(-1), 2)
    if not np.array_equal(selected, ids.reshape(-1)):
        raise ValueError("reference Top-2 IDs do not match router scores")
    out.mkdir(parents=True, exist_ok=True)
    files = [bundle / "moe_chain.vmfb", bundle / "moe_manifest.json",
             bundle / "moe_top2_abi.json",
             OVERLAY / "picorv32.bit", OVERLAY / "picorv32.hwh",
             REPO / "firmware/rt/rt_fw.bin",
             REPO / "compiler/runtime/test/board_launcher.py",
             REPO / "driver/pynq_matmul.py", HERE / "board_moe_iree_chain.py",
             Path(runtime)]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, out / path.name)
    top2_abi = json.loads((bundle / "moe_top2_abi.json").read_text())
    np.save(out / "moe_params.npy", np.fromfile(bundle / "moe_params.bin", dtype=np.uint8))
    shutil.copy2(bundle / "moe_router.npy", out / "router.npy")
    case = out / "token"
    case.mkdir(exist_ok=True)
    np.save(case / "x.npy", x)
    np.save(case / "scores.npy", scores)
    np.save(case / "scores_init.npy", np.zeros((1, 8), dtype=np.float32))
    np.save(case / "values.npy", values)
    np.save(case / "values_init.npy",
            np.full(((top2_abi["values_offset"] + 8) // 4,), -np.inf,
                    dtype=np.float32))
    np.save(case / "ids.npy", ids)
    ids_init = np.zeros(top2_abi["ids_offset"] + 8, dtype=np.uint8)
    at = top2_abi["ids_offset"]
    ids_init[at:at + 8] = np.full(2, np.iinfo(np.int32).max,
                                   dtype=np.int32).view(np.uint8)
    np.save(case / "ids_init.npy", ids_init)
    np.save(case / "route_zero.npy", np.zeros(96, dtype=np.uint8))
    np.save(case / "out_zero.npy", np.zeros((1, 1024), dtype=np.float32))
    np.save(case / "expected.npy", expected)
    run = out / "run_board.sh"
    run.write_text('''#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then exec sudo "$0" "$@"; fi
for f in /etc/profile.d/pynq_venv.sh /etc/profile.d/xrt_setup.sh /opt/xilinx/xrt/setup.sh; do
  if [ -f "$f" ]; then source "$f" >/dev/null; fi
done
cd "$(dirname "$0")"
exec python3 board_moe_iree_chain.py --router --real --cases token --window-mb 128 "$@"
''')
    run.chmod(0o755)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    digest = hashlib.sha256((out / "moe_chain.vmfb").read_bytes()).hexdigest()
    (out / "VERSION").write_text(f"commit {revision}\nVMFB sha256 {digest}\n"
                                 f"packed bytes {manifest['total_bytes']}\nwindow MiB {WINDOW_MB}\n")
    print(f"staged {out}: real layer-0 token, {manifest['total_bytes']} parameter bytes, "
          f"{WINDOW_MB} MiB window")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "build/moe/m5q/board")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    args = parser.parse_args()
    stage(args.out, args.bundle, args.reference, args.runtime)

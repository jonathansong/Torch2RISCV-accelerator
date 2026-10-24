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
import moe_hf  # noqa: E402
import moe  # noqa: E402
from moe_iree_chain import build as build_chain  # noqa: E402

OVERLAY = REPO / "boards/kv260/build/output/d16_100mhz_w128_p6_gemv_np4_pcpi"
RUNTIME = REPO / "build/iree/build-sa-aarch64/tools/iree-run-module"
BUNDLE = REPO / "build/moe/m5k/real-layer0"
MODEL = REPO / "build/llm_cache/TitanML-tiny-mixtral"
REFERENCE = REPO / "build/moe/m4c/tiny-mixtral/layer0/moe_token_acceptance.npz"
WINDOW_MB = 128


def stage(out, bundle=BUNDLE, model=MODEL, reference=REFERENCE, runtime=RUNTIME):
    out, bundle = Path(out).resolve(), Path(bundle).resolve()
    manifest = json.loads((bundle / "moe_manifest.json").read_text())
    if manifest["d"] != 16 or manifest["experts"] != 8 or manifest["dim"] != 1024:
        raise ValueError("expected the D=16, eight-expert tiny Mixtral layer")
    if manifest["total_bytes"] + (16 << 20) > (WINDOW_MB << 20):
        raise ValueError("packed parameters leave insufficient room in the 128 MiB window")
    chain = out.parent / "real-chain"
    build_chain(bundle, chain, include_top2=True, include_router=True)
    _, weights, source = moe_hf.load_layer(str(model), 0, d=16)
    if source["adapter"] != "mixtral":
        raise ValueError("expected Mixtral checkpoint")
    with np.load(reference) as data:
        x = np.asarray(data["x"], dtype=np.float32).reshape(1, 1024)
        scores = np.asarray(data["scores"], dtype=np.float32).reshape(1, 8)
        ids = np.asarray(data["pairs"], dtype=np.int32).reshape(1, 2)
        expected = np.asarray(data["output"], dtype=np.float32).reshape(1, 1024)
    values = np.take_along_axis(scores, ids, axis=1)
    router_scores = weights.router @ x.reshape(-1)
    if not np.allclose(router_scores, scores.reshape(-1), rtol=2e-4, atol=2e-5):
        raise ValueError("reference router scores do not match the checkpoint")
    selected, _ = moe.route(scores.reshape(-1), 2)
    if not np.array_equal(selected, ids.reshape(-1)):
        raise ValueError("reference Top-2 IDs do not match router scores")
    out.mkdir(parents=True, exist_ok=True)
    files = [chain / "moe_chain.vmfb", bundle / "moe_manifest.json",
             OVERLAY / "picorv32.bit", OVERLAY / "picorv32.hwh",
             REPO / "firmware/rt/rt_fw.bin",
             REPO / "compiler/runtime/test/board_launcher.py",
             REPO / "driver/pynq_matmul.py", HERE / "board_moe_iree_chain.py",
             Path(runtime)]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, out / path.name)
    np.save(out / "moe_params.npy", np.fromfile(bundle / "moe_params.bin", dtype=np.uint8))
    np.save(out / "router.npy", np.asarray(weights.router, dtype=np.float32))
    case = out / "token"
    case.mkdir(exist_ok=True)
    np.save(case / "x.npy", x)
    np.save(case / "scores.npy", scores)
    np.save(case / "scores_init.npy", np.zeros((1, 8), dtype=np.float32))
    np.save(case / "values.npy", values)
    np.save(case / "values_init.npy", np.full((1, 2), -np.inf, dtype=np.float32))
    np.save(case / "ids.npy", ids)
    ids_init = np.zeros(72, dtype=np.uint8)
    ids_init[64:72] = np.full(2, np.iinfo(np.int32).max,
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
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    args = parser.parse_args()
    stage(args.out, args.bundle, args.model, args.reference, args.runtime)

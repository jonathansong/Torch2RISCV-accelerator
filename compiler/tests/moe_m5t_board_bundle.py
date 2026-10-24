#!/usr/bin/env python3
"""Stage the completed M5 one-input native Mixtral MoE board acceptance."""

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
HYBRID = REPO / "build/moe/m4c/hybrid-sim/layer0/moe_acceptance.npz"
TOKEN = REPO / "build/moe/m4c/tiny-mixtral/layer0/moe_token_acceptance.npz"
WINDOW_BYTES = 128 << 20


def reference_cases(hybrid=HYBRID, token=TOKEN):
    for source, prefix in ((hybrid, "hybrid"), (token, "token")):
        with np.load(source) as data:
            inputs = data["x"]
            if inputs.ndim == 1:
                inputs = inputs.reshape(1, -1)
            for i, x in enumerate(inputs):
                yield (f"{prefix}{i}", np.asarray(x, dtype=np.float32).reshape(1, -1),
                       np.asarray(data["scores"][i], dtype=np.float32),
                       np.asarray(data["pairs"][i], dtype=np.int32),
                       np.asarray(data["output"][i], dtype=np.float32).reshape(1, -1))


def stage(out, bundle=BUNDLE, runtime=RUNTIME):
    out, bundle = Path(out).resolve(), Path(bundle).resolve()
    manifest = json.loads((bundle / "moe_manifest.json").read_text())
    dim, experts = manifest["dim"], manifest["experts"]
    if (manifest["d"], dim, experts, manifest["top_k"]) != (16, 1024, 8, 2):
        raise ValueError("expected real layer-0 D=16 Mixtral Top-2 bundle")
    abi = json.loads((bundle / "moe_top2_abi.json").read_text())
    # This is a peak upper bound: archive router + packed weights, five scratch
    # buffers, one input, and 2 MiB of command/runtime allocations.
    required = (manifest["total_bytes"] + experts * dim * 4 +
                (experts * 4 + abi["values_offset"] + 8 + abi["ids_offset"] + 8 +
                 96 + dim * 4) + dim * 4 + (2 << 20))
    if required > WINDOW_BYTES:
        raise ValueError(f"MoE buffers need {required} bytes in a {WINDOW_BYTES}-byte window")
    router = np.load(bundle / "moe_router.npy")
    if router.shape != (experts, dim) or router.dtype != np.float32:
        raise ValueError("invalid model router tensor")
    cases = list(reference_cases())
    for name, x, scores, ids, expected in cases:
        if x.shape != (1, dim) or scores.shape != (experts,) or ids.shape != (2,) or expected.shape != (1, dim):
            raise ValueError(f"{name}: invalid reference shape")
        actual_scores = router @ x.reshape(-1)
        if not np.allclose(actual_scores, scores, rtol=2e-4, atol=2e-5):
            raise ValueError(f"{name}: reference router scores disagree with checkpoint")
        selected, _ = moe.route(actual_scores, 2)
        if not np.array_equal(selected, ids):
            raise ValueError(f"{name}: reference Top-2 IDs disagree with checkpoint")
    from iree.runtime import ParameterIndex
    archive = ParameterIndex()
    archive.load(str(bundle / "moe_native.irpa"))
    if {key: entry.length for key, entry in archive.items()} != {
            "router": router.nbytes, "params": manifest["total_bytes"]}:
        raise ValueError("parameter archive disagrees with packed manifest")
    files = [bundle / "moe_model.vmfb", bundle / "moe_native.irpa",
             bundle / "moe_top2_abi.json",
             OVERLAY / "picorv32.bit", OVERLAY / "picorv32.hwh",
             REPO / "firmware/rt/rt_fw.bin", REPO / "driver/pynq_matmul.py",
             REPO / "compiler/runtime/test/board_launcher.py",
             HERE / "board_moe_native_model.py", Path(runtime)]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    out.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.copy2(path, out / path.name)
    for name, x, _scores, _ids, expected in cases:
        case = out / name
        case.mkdir(exist_ok=True)
        np.save(case / "x.npy", x)
        np.save(case / "expected.npy", expected)
    run = out / "run_board.sh"
    run.write_text('''#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then exec sudo "$0" "$@"; fi
for f in /etc/profile.d/pynq_venv.sh /etc/profile.d/xrt_setup.sh /opt/xilinx/xrt/setup.sh; do
  if [ -f "$f" ]; then source "$f" >/dev/null; fi
done
cd "$(dirname "$0")"
exec python3 board_moe_native_model.py "$@"
''')
    run.chmod(0o755)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    digest = hashlib.sha256((out / "moe_model.vmfb").read_bytes()).hexdigest()
    (out / "VERSION").write_text(f"commit {revision}\nVMFB sha256 {digest}\n"
                                 f"packed bytes {manifest['total_bytes']}\n"
                                 f"estimated window bytes {required}\n"
                                 f"window bytes {WINDOW_BYTES}\n")
    print(f"staged {out}: {len(cases)} real-model inputs, "
          f"estimated {required} of {WINDOW_BYTES} window bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "build/moe/m5t/board")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    args = parser.parse_args()
    stage(args.out, args.bundle, args.runtime)

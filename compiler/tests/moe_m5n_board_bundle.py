#!/usr/bin/env python3
"""Stage a small Mixtral-like single-VMFB route-to-FFN KV260 test."""

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
from moe_device_bundle import build as build_device_bundle  # noqa: E402
from moe_iree_chain import build as build_chain  # noqa: E402
import moe  # noqa: E402
from synth_hf import write_safetensors  # noqa: E402

OVERLAY = REPO / "boards/kv260/build/output/d16_100mhz_w128_p6_gemv_np4_pcpi"
RUNTIME = REPO / "build/iree/build-sa-aarch64/tools/iree-run-module"


def checkpoint(path, weights):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({
        "architectures": ["MixtralForCausalLM"], "model_type": "mixtral",
        "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
        "num_local_experts": 4, "num_experts_per_tok": 2,
        "norm_topk_prob": True}) + "\n")
    prefix = "model.layers.0.block_sparse_moe"
    tensors = {f"{prefix}.gate.weight": weights.router}
    for expert in range(4):
        ep = f"{prefix}.experts.{expert}"
        tensors[f"{ep}.w1.weight"] = weights.gate[expert]
        tensors[f"{ep}.w3.weight"] = weights.up[expert]
        tensors[f"{ep}.w2.weight"] = weights.down[expert]
    write_safetensors(str(path / "model.safetensors"), tensors)


def stage(out, runtime=RUNTIME):
    out = Path(out).resolve()
    root = out.parent
    config = moe.MoeConfig(32, 64, 4, 2)
    weights = moe.synthetic_weights(config, seed=94)
    packed = moe.quantize_weights(config, weights)
    model = root / "synthetic-model"
    bundle = root / "synthetic-bundle"
    chain = root / "synthetic-chain"
    checkpoint(model, weights)
    build_device_bundle(model, 0, bundle)
    build_chain(bundle, chain)
    out.mkdir(parents=True, exist_ok=True)
    files = [chain / "moe_chain.vmfb", bundle / "moe_params.bin",
             bundle / "moe_manifest.json", OVERLAY / "picorv32.bit",
             OVERLAY / "picorv32.hwh", REPO / "firmware/rt/rt_fw.bin",
             REPO / "compiler/runtime/test/board_launcher.py",
             REPO / "driver/pynq_matmul.py", HERE / "board_moe_iree_chain.py"]
    if Path(runtime).is_file():
        files.append(Path(runtime))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, out / path.name)
    np.save(out / "moe_params.npy", np.fromfile(out / "moe_params.bin", dtype=np.uint8))
    for name, x in (("ramp", np.linspace(-0.625, 0.75, 32, dtype=np.float32)),
                    ("reverse", np.linspace(0.75, -0.625, 32, dtype=np.float32)),
                    ("random", np.random.default_rng(340).normal(size=32).astype(np.float32))):
        case = out / name
        case.mkdir(exist_ok=True)
        expected, scores, ids, routing = moe.moe_forward(x, config, packed, quantized=True)
        values = scores[ids]
        np.save(case / "values.npy", values.reshape(1, 2).astype(np.float32))
        np.save(case / "ids.npy", ids.reshape(1, 2).astype(np.int32))
        np.save(case / "x.npy", x.reshape(1, 32))
        np.save(case / "route_zero.npy", np.zeros(96, dtype=np.uint8))
        np.save(case / "out_zero.npy", np.zeros((1, 32), dtype=np.float32))
        np.save(case / "expected.npy", expected.reshape(1, 32))
    run = out / "run_board.sh"
    run.write_text('''#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then exec sudo "$0" "$@"; fi
for f in /etc/profile.d/pynq_venv.sh /etc/profile.d/xrt_setup.sh /opt/xilinx/xrt/setup.sh; do
  if [ -f "$f" ]; then source "$f" >/dev/null; fi
done
cd "$(dirname "$0")"
exec python3 board_moe_iree_chain.py "$@"
''')
    run.chmod(0o755)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    digest = hashlib.sha256((out / "moe_chain.vmfb").read_bytes()).hexdigest()
    (out / "VERSION").write_text(f"commit {revision}\nVMFB sha256 {digest}\n")
    print(f"staged {out}: one VMFB, 3 synthetic cases, "
          f"runtime {'present' if Path(runtime).is_file() else 'missing'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "build/moe/m5n/board")
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    args = parser.parse_args()
    stage(args.out, args.runtime)

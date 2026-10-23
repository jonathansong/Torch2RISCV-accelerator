#!/usr/bin/env python3
"""Build a self-contained KV260 bundle for the complete SA Mixtral decoder."""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def stage(dense, experts, reference, hf_reference, out):
    if os.path.exists(out):
        raise FileExistsError(f"bundle already exists: {out}")
    os.makedirs(out)
    sources = {
        "mixtral_dense.w8a8": dense,
        "full_device_sim_logits.npz": reference,
        "teacher_logits.npz": hf_reference,
        "board_moe_full_device.py": os.path.join(HERE, "board_moe_full_device.py"),
        "run_board.sh": os.path.join(HERE, "run_moe_full_device_board.sh"),
        "moe_model.py": os.path.join(REPO, "compiler", "runtime", "tools", "moe_model.py"),
        "moe_m3b.py": os.path.join(REPO, "compiler", "runtime", "tools", "moe_m3b.py"),
        "compile_model.py": os.path.join(REPO, "llm", "compile_model.py"),
        "compile_layer.py": os.path.join(REPO, "llm", "compile_layer.py"),
        "export_w8a8.py": os.path.join(REPO, "llm", "export_w8a8.py"),
        "ref_model.py": os.path.join(REPO, "llm", "ref_model.py"),
        "checkpoint.py": os.path.join(REPO, "llm", "checkpoint.py"),
        "tokenizer.py": os.path.join(REPO, "llm", "tokenizer.py"),
        "pynq_matmul.py": os.path.join(REPO, "driver", "pynq_matmul.py"),
        "sadesc.py": os.path.join(REPO, "compiler", "runtime", "tools", "sadesc.py"),
        "picorv32.bit": os.path.join(experts, "layer0", "picorv32.bit"),
        "picorv32.hwh": os.path.join(experts, "layer0", "picorv32.hwh"),
        "rt_fw.bin": os.path.join(experts, "layer0", "rt_fw.bin"),
        "moe_top2.bin": os.path.join(experts, "layer0", "moe_top2.bin"),
    }
    for name, source in sources.items():
        if not os.path.isfile(source):
            raise FileNotFoundError(source)
        try:
            os.link(source, os.path.join(out, name))
        except OSError:
            shutil.copy2(source, os.path.join(out, name))
    for layer in range(2):
        target = os.path.join(out, "experts", f"layer{layer}")
        os.makedirs(target)
        for name in ("moe_manifest.json", "moe_params.bin"):
            source = os.path.join(experts, f"layer{layer}", name)
            try:
                os.link(source, os.path.join(target, name))
            except OSError:
                shutil.copy2(source, os.path.join(target, name))
    subprocess.run([sys.executable, "-c",
                    "import board_moe_full_device; from export_w8a8 import W8A8; "
                    "from moe_model import MoEModelCompiler; "
                    "m=W8A8('mixtral_dense.w8a8'); c=MoEModelCompiler(m,8); "
                    "print(len(c.build_layer(0)),len(c.build_layer(1)),len(c.build_final()))"],
                   cwd=out, check=True)
    print(f"{out}: ready for board transfer; {sum(os.path.getsize(os.path.join(root, name)) for root, _, files in os.walk(out) for name in files)} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dense", default="build/moe/m4f/mixtral_dense.w8a8")
    parser.add_argument("--experts", default="build/moe/m4c/board")
    parser.add_argument("--reference", default="build/moe/m4f/full_device_sim_logits.npz")
    parser.add_argument("--hf-reference", default="build/moe/m4c/hybrid-sim/teacher_logits.npz")
    parser.add_argument("--out", default="build/moe/m4f/board")
    args = parser.parse_args()
    stage(args.dense, args.experts, args.reference, args.hf_reference, args.out)


if __name__ == "__main__":
    main()

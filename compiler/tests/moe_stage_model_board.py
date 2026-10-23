#!/usr/bin/env python3
"""Stage the two-layer Mixtral board runner using existing M4c expert bundles."""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FRONTEND = os.path.join(REPO, "compiler", "frontend")


def stage(dense, reference, m4c, out):
    if os.path.exists(out):
        raise FileExistsError(f"board bundle already exists: {out}")
    for layer in range(2):
        path = os.path.join(m4c, f"layer{layer}")
        for name in ("moe_params.bin", "moe_manifest.json"):
            if not os.path.isfile(os.path.join(path, name)):
                raise FileNotFoundError(os.path.join(path, name))
    os.makedirs(os.path.join(out, "model_dense"))
    for name in ("config.json", "model.safetensors"):
        src = os.path.join(dense, name)
        dst = os.path.join(out, "model_dense", name)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    shutil.copy2(reference, os.path.join(out, "teacher_logits.npz"))
    for name in ("moe.py", "moe_hf.py", "moe_mixtral_numpy.py"):
        shutil.copy2(os.path.join(FRONTEND, name), os.path.join(out, name))
    for name in ("moe_m3b.py", "compile_layer.py", "sadesc.py", "pynq_matmul.py"):
        shutil.copy2(os.path.join(m4c, "layer0", name), os.path.join(out, name))
    shutil.copy2(os.path.join(HERE, "board_moe_model.py"), os.path.join(out, "board_moe_model.py"))
    shutil.copy2(os.path.join(HERE, "run_moe_model_board.sh"), os.path.join(out, "run_board.sh"))
    subprocess.run([sys.executable, "-c",
                    "import board_moe_model, moe_mixtral_numpy; print('board imports PASS')"],
                   cwd=out, check=True)
    print(f"{out}: host decoder and dense checkpoint ready; uses ../moe-m4c/layer0,layer1 on board")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dense", default="build/moe/m4e/model_dense")
    parser.add_argument("--reference", default="build/moe/m4c/hybrid-sim/teacher_logits.npz")
    parser.add_argument("--m4c", default="build/moe/m4c/board")
    parser.add_argument("--out", default="build/moe/m4e/board")
    args = parser.parse_args()
    stage(args.dense, args.reference, args.m4c, args.out)


if __name__ == "__main__":
    main()

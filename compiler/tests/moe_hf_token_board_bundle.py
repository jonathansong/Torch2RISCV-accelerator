#!/usr/bin/env python3
"""Build per-layer KV260 bundles from a real-token Mixtral trace."""

import argparse
import json
import os
import shutil


STATIC_FILES = ("picorv32.bit", "picorv32.hwh", "rt_fw.bin", "moe_top2.bin",
                "pynq_matmul.py", "moe_m3b.py", "sadesc.py", "run_board.sh")


def bundle(template, trace_root, out):
    if os.path.exists(out):
        raise FileExistsError(f"bundle already exists: {out}")
    with open(os.path.join(trace_root, "report.json")) as f:
        report = json.load(f)
    layers = sorted(int(layer) for layer in report["layers"])
    if not layers:
        raise ValueError("trace report contains no layers")
    for layer in layers:
        source = os.path.join(trace_root, f"layer{layer}")
        with open(os.path.join(source, "moe_manifest.json")) as f:
            manifest = json.load(f)
        if manifest["source"]["layer"] != layer or manifest["source"]["architecture"] != report["architecture"]:
            raise ValueError(f"layer {layer}: trace and packed parameters disagree")
        target = os.path.join(out, f"layer{layer}")
        os.makedirs(target, exist_ok=False)
        for name in STATIC_FILES:
            shutil.copy2(os.path.join(template, name), os.path.join(target, name))
        shutil.copy2(os.path.join(os.path.dirname(__file__), "board_moe_hf.py"),
                     os.path.join(target, "board_moe_hf.py"))
        for name in ("moe_params.bin", "moe_manifest.json"):
            shutil.copy2(os.path.join(source, name), os.path.join(target, name))
        shutil.copy2(os.path.join(source, "moe_token_acceptance.npz"),
                     os.path.join(target, "moe_acceptance.npz"))
        with open(os.path.join(target, "README.txt"), "w") as f:
            f.write(f"M4c Mixtral real-token layer {layer} acceptance.\n"
                    f"Input tokens: {report['tokens']}; HF last-token argmax: {report['last_hf_token']}.\n"
                    "Run ./run_board.sh on KV260. This checks one routed MoE layer\n"
                    "against W8A8 host golden values; it does not run the complete decoder.\n")
    print(f"{out}: {len(layers)} layer bundles")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--template", default="build/moe/m4b/board")
    parser.add_argument("--trace", default="build/moe/m4c/tiny-mixtral")
    parser.add_argument("--out", default="build/moe/m4c/board")
    args = parser.parse_args()
    bundle(args.template, args.trace, args.out)


if __name__ == "__main__":
    main()

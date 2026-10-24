#!/usr/bin/env python3
"""Locate the first invalid stage of the native M5t board program."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import board_launcher as BL  # noqa: E402


def main():
    case = HERE / "token0"
    mm, buf, env = BL.start(str(HERE / "picorv32.bit"), str(HERE / "rt_fw.bin"), mb=128)
    env["SA_RV"] = "embedded"
    names = ("scores", "values", "ids", "route", "out")
    try:
        command = [str(HERE / "iree-run-module"), "--device=sa",
                   f"--module={HERE / 'moe_model_debug.vmfb'}",
                   f"--parameters=moe={HERE / 'moe_native.irpa'}", "--function=main",
                   f"--input=@{case / 'x.npy'}"]
        command += [f"--output=@{case / ('debug_' + name + '.npy')}" for name in names]
        subprocess.run(command, check=True, env=env)
        scores = np.load(case / "debug_scores.npy").reshape(-1)
        abi = json.loads((HERE / "moe_top2_abi.json").read_text())
        values = np.load(case / "debug_values.npy").view("<f4").reshape(-1)
        ids = np.load(case / "debug_ids.npy").view("<i4").reshape(-1)
        route = np.load(case / "debug_route.npy").view("<u4").reshape(-1)
        output = np.load(case / "debug_out.npy").reshape(-1)
        ref_scores = np.load(HERE / "scores_expected.npy").reshape(-1)
        ref_ids = np.load(HERE / "ids_expected.npy").reshape(-1)
        print("scores:", scores.tolist())
        print("reference scores:", ref_scores.tolist())
        print("values:", values[abi["values_offset"] // 4:abi["values_offset"] // 4 + 2].tolist())
        print("selected IDs:", ids[abi["ids_offset"] // 4:abi["ids_offset"] // 4 + 2].tolist())
        print("reference IDs:", ref_ids.tolist())
        print("route record:", route[16:24].tolist())
        print("output first 8:", output[:8].tolist())
        print("finite output:", int(np.isfinite(output).sum()), "/", output.size)
    finally:
        head, beat = BL.stop(mm, buf)
    print(f"firmware entries {head}, heartbeat {beat}")


if __name__ == "__main__":
    raise SystemExit(main())

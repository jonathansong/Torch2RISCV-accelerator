#!/usr/bin/env python3
"""Run the Torch-exported MoE layer through IREE on KV260."""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board_launcher as BL  # noqa: E402


def main():
    runner = os.path.join(HERE, "iree-run-module")
    if not os.path.isfile(runner):
        raise RuntimeError("iree-run-module missing; build and stage the aarch64 SA runtime")
    mm, buffer, env = BL.start(os.path.join(HERE, "picorv32.bit"),
                               os.path.join(HERE, "rt_fw.bin"), mb=32)
    env["SA_RV"] = "embedded"
    cases = []
    try:
        for name in ("ramp", "zero", "random"):
            path = os.path.join(HERE, name)
            outputs = [os.path.join(path, f"out_{i}.npy") for i in range(4)]
            command = [runner, "--device=sa", f"--module={HERE}/moe_layer.vmfb",
                       f"--parameters=model={HERE}/moe_layer.irpa", "--function=main",
                       f"--input=@{path}/x.npy", *(f"--output=@{item}" for item in outputs)]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            if result.returncode:
                print(f"{name}: FAIL\n{result.stdout[-1500:]}{result.stderr[-1500:]}")
                cases.append({"name": name, "pass": False, "error": result.stderr[-1500:]})
                continue
            got = [np.load(item) for item in outputs]
            ref = [np.load(os.path.join(path, f"ref_{i}.npy")) for i in range(4)]
            ids_ok = np.array_equal(got[2], ref[2])
            errors = [float(np.max(np.abs(got[i] - ref[i]))) for i in (0, 1, 3)]
            passed = ids_ok and errors[0] <= 2e-5 and errors[1] <= 2e-5 and errors[2] <= 2e-5
            print(f"{name}: {'PASS' if passed else 'FAIL'} IDs {got[2].reshape(-1).tolist()}, "
                  f"max errors {errors}")
            cases.append({"name": name, "pass": bool(passed),
                          "ids": got[2].reshape(-1).tolist(), "max_errors": errors})
    finally:
        head, beat = BL.stop(mm, buffer)
        print(f"rt_fw completed {head} entries, heartbeat {beat}")
    passed = all(case["pass"] for case in cases) and len(cases) == 3
    with open(os.path.join(HERE, "board_report.json"), "w") as stream:
        json.dump({"pass": passed, "cases": cases, "rt_entries": head,
                   "heartbeat": beat}, stream, indent=2)
        stream.write("\n")
    print("native MoE layer board PASS" if passed else "native MoE layer board FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

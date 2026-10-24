#!/usr/bin/env python3
"""KV260 acceptance for IREE-generated native SA MoE Top-2 exports."""

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
        raise RuntimeError("iree-run-module missing; rebuild and stage the aarch64 SA runtime")
    mm, buf, env = BL.start(os.path.join(HERE, "picorv32.bit"), os.path.join(HERE, "rt_fw.bin"), mb=32)
    env["SA_RV"] = "embedded"
    failures = 0
    cases = []
    try:
        for case in ("ascending", "ties", "random"):
            path = os.path.join(HERE, case)
            values_path, indices_path = os.path.join(path, "values.npy"), os.path.join(path, "indices.npy")
            command = [runner, "--device=sa", f"--module={HERE}/router.vmfb", "--function=quantized_router",
                       f"--input=@{path}/x.npy", f"--input=@{path}/w.npy",
                       f"--output=@{values_path}", f"--output=@{indices_path}"]
            result = subprocess.run(command, capture_output=True, text=True, env=env)
            if result.returncode:
                print(f"{case}: iree-run-module failed\n{result.stdout[-1500:]}{result.stderr[-1500:]}")
                cases.append({"name": case, "pass": False, "error": result.stderr[-1500:]})
                failures += 1
                continue
            values, indices = np.load(values_path), np.load(indices_path)
            expected_values = np.load(os.path.join(path, "values_ref.npy"))
            expected_indices = np.load(os.path.join(path, "indices_ref.npy"))
            same = (values.shape == expected_values.shape and indices.shape == expected_indices.shape and
                    values.tobytes() == expected_values.tobytes() and
                    indices.tobytes() == expected_indices.tobytes())
            print(f"{case}: {'PASS' if same else 'FAIL'} IDs {indices.reshape(-1).tolist()}, "
                  f"values {values.reshape(-1).tolist()}")
            cases.append({"name": case, "pass": bool(same), "indices": indices.reshape(-1).tolist(),
                          "values": values.reshape(-1).tolist()})
            failures += not same
    finally:
        head, beat = BL.stop(mm, buf)
        print(f"rt_fw completed {head} entries, heartbeat {beat}")
    print("IREE MoE Top-2 board PASS" if failures == 0 else f"IREE MoE Top-2 board FAILED ({failures})")
    with open(os.path.join(HERE, "board_report.json"), "w") as stream:
        json.dump({"pass": failures == 0, "cases": cases, "rt_entries": head, "heartbeat": beat}, stream, indent=2)
        stream.write("\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""KV260 acceptance for a single IREE route-to-FFN VMFB."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import board_launcher as BL  # noqa: E402


def main():
    runner = HERE / "iree-run-module"
    if not runner.is_file():
        raise FileNotFoundError(runner)
    manifest = json.loads((HERE / "moe_manifest.json").read_text())
    mm, buf, env = BL.start(str(HERE / "picorv32.bit"), str(HERE / "rt_fw.bin"), mb=32)
    env["SA_RV"] = "embedded"
    cases = []
    try:
        for name in ("ramp", "reverse", "random"):
            case = HERE / name
            command = [str(runner), "--device=sa", f"--module={HERE / 'moe_chain.vmfb'}",
                       "--function=main", f"--input=@{case / 'values.npy'}",
                       f"--input=@{case / 'ids.npy'}", f"--input=@{case / 'x.npy'}",
                       f"--input=@{HERE / 'moe_params.npy'}",
                       f"--input=@{case / 'route_zero.npy'}",
                       f"--input=@{case / 'out_zero.npy'}",
                       f"--output=@{case / 'route.npy'}",
                       f"--output=@{case / 'out.npy'}"]
            run = subprocess.run(command, capture_output=True, text=True, env=env)
            if run.returncode:
                error = (run.stdout + run.stderr)[-2000:]
                print(f"{name}: IREE failed: {error}")
                cases.append({"name": name, "pass": False, "error": error})
                continue
            route = np.load(case / "route.npy").view("<u4")
            got = np.load(case / "out.npy").reshape(-1)
            ids = np.load(case / "ids.npy").reshape(-1)
            expected = np.load(case / "expected.npy").reshape(-1)
            record = route[16:24]
            offsets = ids.astype(np.uint32) * np.uint32(manifest["expert_stride"])
            error = float(np.max(np.abs(got - expected)))
            valid = (route.size == 24 and record[0] == 2
                     and np.array_equal(record[1:3], ids.astype(np.uint32))
                     and np.array_equal(record[3:5], offsets)
                     and np.isfinite(got).all() and error <= 3e-3)
            print(f"{name}: {'PASS' if valid else 'FAIL'} experts {ids.tolist()}, max error {error:.6g}")
            cases.append({"name": name, "pass": bool(valid), "ids": ids.tolist(),
                          "max_error": error})
    finally:
        head, beat = BL.stop(mm, buf)
    result = {"pass": all(case["pass"] for case in cases), "cases": cases,
              "rt_entries": head, "heartbeat": beat}
    (HERE / "board_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print("single-VMFB route+FFN board PASS" if result["pass"]
          else "single-VMFB route+FFN board FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

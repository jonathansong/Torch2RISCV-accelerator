#!/usr/bin/env python3
"""Accept the one-input native Mixtral MoE VMFB on KV260."""

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
    cases = sorted(path for path in HERE.iterdir()
                   if path.is_dir() and (path / "x.npy").is_file())
    if not cases:
        raise RuntimeError("no staged MoE inputs")
    mm, buf, env = BL.start(str(HERE / "picorv32.bit"), str(HERE / "rt_fw.bin"), mb=128)
    env["SA_RV"] = "embedded"
    results = []
    try:
        for case in cases:
            command = [str(runner), "--device=sa", f"--module={HERE / 'moe_model.vmfb'}",
                       f"--parameters=moe={HERE / 'moe_native.irpa'}", "--function=main",
                       f"--input=@{case / 'x.npy'}", f"--output=@{case / 'out.npy'}"]
            run = subprocess.run(command, capture_output=True, text=True, env=env)
            if run.returncode:
                error = (run.stdout + run.stderr)[-3000:]
                print(f"{case.name}: FAIL IREE: {error}", flush=True)
                results.append({"name": case.name, "pass": False, "error": error})
                continue
            output = np.load(case / "out.npy").reshape(-1)
            expected = np.load(case / "expected.npy").reshape(-1)
            if output.shape != expected.shape:
                raise ValueError(f"{case.name}: output shape {output.shape} != {expected.shape}")
            error = float(np.max(np.abs(output - expected)))
            passed = bool(np.isfinite(output).all() and
                          np.allclose(output, expected, rtol=3e-3, atol=3e-3))
            print(f"{case.name}: {'PASS' if passed else 'FAIL'}, "
                  f"max |output| {float(np.max(np.abs(output))):.6g}, "
                  f"max error {error:.6g}", flush=True)
            results.append({"name": case.name, "pass": passed, "max_error": error})
    finally:
        head, heartbeat = BL.stop(mm, buf)
    report = {"pass": len(results) == len(cases) and all(r["pass"] for r in results),
              "cases": results, "window_mb": 128, "rt_entries": head,
              "heartbeat": heartbeat}
    (HERE / "board_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("native model ABI board PASS" if report["pass"] else "native model ABI board FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

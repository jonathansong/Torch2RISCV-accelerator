#!/usr/bin/env python3
"""KV260 acceptance for a single IREE route-to-FFN VMFB."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import board_launcher as BL  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top2", action="store_true", help="run compiled Top-2 before route and FFN")
    parser.add_argument("--router", action="store_true", help="run VMVX router before Top-2, route, and FFN")
    parser.add_argument("--cases", default="ramp,reverse,random", help="comma-separated staged case names")
    parser.add_argument("--window-mb", type=int, default=32)
    parser.add_argument("--real", action="store_true", help="use the real-model router tolerance")
    args = parser.parse_args()
    top2 = args.top2 or args.router
    runner = HERE / "iree-run-module"
    if not runner.is_file():
        raise FileNotFoundError(runner)
    manifest = json.loads((HERE / "moe_manifest.json").read_text())
    mm, buf, env = BL.start(str(HERE / "picorv32.bit"), str(HERE / "rt_fw.bin"),
                            mb=args.window_mb)
    env["SA_RV"] = "embedded"
    cases = []
    try:
        for name in args.cases.split(","):
            case = HERE / name
            command = [str(runner), "--device=sa", f"--module={HERE / 'moe_chain.vmfb'}",
                       "--function=main"]
            if top2:
                command += [f"--input=@{case / ('scores_init.npy' if args.router else 'scores.npy')}",
                            f"--input=@{case / 'values_init.npy'}",
                            f"--input=@{case / 'ids_init.npy'}"]
            else:
                command += [f"--input=@{case / 'values.npy'}",
                            f"--input=@{case / 'ids.npy'}"]
            command.append(f"--input=@{case / 'x.npy'}")
            if args.router:
                command += [f"--input=@{HERE / 'router.npy'}"]
            command += [f"--input=@{HERE / 'moe_params.npy'}",
                       f"--input=@{case / 'route_zero.npy'}",
                       f"--input=@{case / 'out_zero.npy'}"]
            if args.router:
                command += [f"--output=@{case / 'scores_device.npy'}"]
            if top2:
                command += [f"--output=@{case / 'values_device.npy'}",
                            f"--output=@{case / 'ids_device.npy'}"]
            command += [f"--output=@{case / 'route.npy'}",
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
            magnitude = float(np.max(np.abs(got)))
            valid = (route.size == 24 and record[0] == 2
                     and np.array_equal(record[1:3], ids.astype(np.uint32))
                     and np.array_equal(record[3:5], offsets)
                     and np.isfinite(got).all() and error <= 3e-3)
            selected_ids = None
            score_error = None
            if args.router:
                device_scores = np.load(case / "scores_device.npy").reshape(-1)
                reference_scores = np.load(case / "scores.npy").reshape(-1)
                score_error = float(np.max(np.abs(device_scores - reference_scores)))
                valid = valid and bool(np.allclose(device_scores, reference_scores,
                                                   rtol=2e-4 if args.real else 1e-5,
                                                   atol=2e-5 if args.real else 1e-5))
            if top2:
                selected_values = np.load(case / "values_device.npy").reshape(-1)
                selected_ids = np.load(case / "ids_device.npy").view("<i4")[16:18]
                valid = (valid and np.array_equal(selected_ids, ids)
                         and np.allclose(selected_values,
                                         np.load(case / "values.npy").reshape(-1),
                                         rtol=2e-4 if args.real else 1e-5 if args.router else 0,
                                         atol=2e-5 if args.real else 1e-5 if args.router else 0))
            print(f"{name}: {'PASS' if valid else 'FAIL'} experts {ids.tolist()}, "
                  f"router error {score_error if score_error is not None else 'input'}, "
                  f"Top-2 {selected_ids.tolist() if selected_ids is not None else 'input'}, "
                  f"route {record[:5].tolist()}, max |output| {magnitude:.6g}, "
                  f"max error {error:.6g}")
            cases.append({"name": name, "pass": bool(valid), "ids": ids.tolist(),
                          "top2_ids": selected_ids.tolist() if selected_ids is not None else None,
                          "router_error": score_error,
                          "route_status": int(record[0]), "route_ids": record[1:3].tolist(),
                          "max_output": magnitude, "max_error": error})
    finally:
        head, beat = BL.stop(mm, buf)
    result = {"pass": all(case["pass"] for case in cases), "cases": cases,
              "window_mb": args.window_mb,
              "rt_entries": head, "heartbeat": beat}
    (HERE / "board_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print("single-VMFB route+FFN board PASS" if result["pass"]
          else "single-VMFB route+FFN board FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

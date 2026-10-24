#!/usr/bin/env python3
"""Run IREE Top-2, route bridge, and M4 FFN in one KV260 acceptance script."""

import json
import os
import subprocess
import sys

import numpy as np
from pynq import allocate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board_launcher as BL  # noqa: E402
import board_moe_ffn as FFN  # noqa: E402
from moe_m3b import top2_ffn_template  # noqa: E402
from pynq_matmul import Device  # noqa: E402

EXPERTS, STRIDE = 16, 6784
DIM, HIDDEN, D = 32, 64, 16


def run_iree():
    runner = os.path.join(HERE, "iree-run-module")
    mm, buf, env = BL.start(os.path.join(HERE, "picorv32.bit"),
                            os.path.join(HERE, "rt_fw.bin"), mb=32)
    env["SA_RV"] = "embedded"
    try:
        for name in ("ascending", "ties", "random"):
            case = os.path.join(HERE, name)
            command = [runner, "--device=sa", f"--module={HERE}/router.vmfb",
                       "--function=quantized_router", f"--input=@{case}/x.npy",
                       f"--input=@{case}/w.npy", f"--output=@{case}/values.npy",
                       f"--output=@{case}/indices.npy"]
            result = subprocess.run(command, capture_output=True, text=True, env=env)
            if result.returncode:
                raise RuntimeError(f"{name}: IREE Top-2 failed: {result.stderr[-1500:]}")
            for tensor in ("values", "indices"):
                actual = np.load(os.path.join(case, f"{tensor}.npy"))
                expected = np.load(os.path.join(case, f"{tensor}_ref.npy"))
                if actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
                    raise RuntimeError(f"{name}: IREE {tensor} differs from reference")
            print(f"{name}: IREE Top-2 PASS", flush=True)
    finally:
        head, beat = BL.stop(mm, buf)
        print(f"IREE rt_fw completed {head} entries, heartbeat {beat}", flush=True)


def run_bridge_ffn():
    binary = np.fromfile(os.path.join(HERE, "moe_route_from_top2.bin"), np.uint8)
    packed, experts = FFN.make_experts(EXPERTS, STRIDE)
    device = Device(os.path.join(HERE, "picorv32.bit"), os.path.join(HERE, "rt_fw.bin"))
    buffers = []
    cases = []
    try:
        def alloc(shape, dtype):
            value = allocate(shape=shape, dtype=dtype)
            buffers.append(value)
            return value

        code = alloc(binary.shape, np.uint8)
        values, ids, route = alloc((2,), np.float32), alloc((2,), np.uint32), alloc((8,), np.uint32)
        call_args = alloc((18,), np.uint32)
        x, table, output = alloc((DIM,), np.float32), alloc(packed.shape, np.uint8), alloc((DIM,), np.float32)
        code[:] = binary
        code.flush()
        call_args[:] = 0
        call_args[0:3] = [values.physical_address, ids.physical_address, route.physical_address]
        call_args[16:18] = [EXPERTS, STRIDE]
        call_args.flush()
        x[:] = np.linspace(-0.75, 0.875, DIM, dtype=np.float32)
        x.flush()
        table[:] = packed
        table.flush()
        xq, xs = FFN.quantize_activation(np.asarray(x))
        template = top2_ffn_template(DIM, HIDDEN, D, ret=False)
        for name in ("ascending", "ties", "random"):
            case = os.path.join(HERE, name)
            values[:] = np.load(os.path.join(case, "values.npy")).reshape(2)
            ids[:] = np.load(os.path.join(case, "indices.npy")).reshape(2)
            route[:] = 0
            values.flush()
            ids.flush()
            route.flush()
            rc = device.call(code.physical_address, call_args.physical_address, perf=True)
            route.invalidate()
            expected_ids = np.load(os.path.join(case, "indices_ref.npy")).reshape(2)
            route_ok = (rc["status"] == 0 and route[0] == 2 and route[7] == 0 and
                        np.array_equal(route[1:3], expected_ids) and
                        np.array_equal(route[3:5], expected_ids.astype(np.uint32) * STRIDE))
            if not route_ok:
                raise RuntimeError(f"{name}: invalid route record {route.tolist()}")
            weights = route[5:7].view(np.float32).copy()
            output[:] = 0
            output.flush()
            fc = device.wait(device.submit(template, bases=[x.physical_address, route.physical_address,
                                 table.physical_address, output.physical_address], perf=True))
            output.invalidate()
            wanted = weights[0] * FFN.expert_reference(xq, xs, experts[int(route[1])])
            wanted += weights[1] * FFN.expert_reference(xq, xs, experts[int(route[2])])
            error = float(np.max(np.abs(np.asarray(output) - wanted)))
            ok = fc["status"] == 0 and np.allclose(output, wanted, rtol=3e-3, atol=3e-3)
            cases.append({"name": name, "pass": bool(ok), "ids": route[1:3].tolist(),
                          "weights": weights.tolist(), "route_cycles": int(rc["cycles"]),
                          "ffn_cycles": int(fc["cycles"]), "max_error": error})
            print(f"{name}: {'PASS' if ok else 'FAIL'} IDs {route[1:3].tolist()}, "
                  f"route {rc['cycles']} cycles, FFN {fc['cycles']} cycles, max error {error:.6g}", flush=True)
    finally:
        device.close()
        for value in reversed(buffers):
            value.freebuffer()
    return cases


def main():
    run_iree()
    cases = run_bridge_ffn()
    passed = len(cases) == 3 and all(case["pass"] for case in cases)
    with open(os.path.join(HERE, "board_report.json"), "w") as stream:
        json.dump({"pass": passed, "cases": cases}, stream, indent=2)
        stream.write("\n")
    print("IREE Top-2 to M4 FFN board PASS" if passed else "IREE Top-2 to M4 FFN board FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""KV260 check for a model-derived compiler route kernel and stride."""

import json
import os
import sys

import numpy as np
from pynq import allocate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import Device  # noqa: E402


def main():
    with open(os.path.join(HERE, "cases.json")) as stream:
        cases = json.load(stream)
    experts, stride = cases["experts"], cases["expert_stride"]
    binary = np.fromfile(os.path.join(HERE, "moe_route.bin"), np.uint8)
    device = Device(os.path.join(HERE, "picorv32.bit"), os.path.join(HERE, "rt_fw.bin"))
    buffers, results = [], []
    try:
        def alloc(shape, dtype):
            value = allocate(shape=shape, dtype=dtype)
            buffers.append(value)
            return value

        code = alloc(binary.shape, np.uint8)
        values = alloc((2,), np.float32)
        ids = alloc((2,), np.uint32)
        # The compiler export's third binding has a 64-byte subspan offset.
        # Keep that offset inside a real allocation; the record is words 16:24.
        route = alloc((24,), np.uint32)
        args = alloc((18,), np.uint32)
        code[:] = binary
        code.flush()
        args[:] = 0
        args[0:3] = [values.physical_address, ids.physical_address, route.physical_address]
        args[16:18] = [experts, stride]
        args.flush()
        for name, case in cases["cases"].items():
            values[:] = np.asarray(case["values"], np.float32)
            ids[:] = np.asarray(case["ids"], np.uint32)
            route[:] = 0
            values.flush()
            ids.flush()
            route.flush()
            completion = device.call(code.physical_address, args.physical_address, perf=True)
            route.invalidate()
            record = route[16:24]
            expected_offsets = ids.astype(np.uint32) * np.uint32(stride)
            expected_weights = np.asarray(case["weights"], np.float32)
            weights = record[5:7].view(np.float32)
            max_error = float(np.max(np.abs(weights - expected_weights)))
            passed = (completion["status"] == 0 and record[0] == 2 and record[7] == 0 and
                      np.array_equal(record[1:3], ids) and
                      np.array_equal(record[3:5], expected_offsets) and
                      max_error <= 5e-5)
            print(f"{name}: {'PASS' if passed else 'FAIL'} status {completion['status']}, "
                  f"IDs {record[1:3].tolist()}, offsets {record[3:5].tolist()}, "
                  f"max weight error {max_error:.6g}, cycles {completion['cycles']}", flush=True)
            results.append({"name": name, "pass": bool(passed),
                            "status": int(completion["status"]),
                            "ids": record[1:3].tolist(), "offsets": record[3:5].tolist(),
                            "max_weight_error": max_error, "cycles": int(completion["cycles"])})
    finally:
        device.close()
        for buffer in reversed(buffers):
            buffer.freebuffer()
    passed = len(results) == len(cases["cases"]) and all(case["pass"] for case in results)
    with open(os.path.join(HERE, "board_report.json"), "w") as stream:
        json.dump({"pass": passed, "cases": results}, stream, indent=2)
        stream.write("\n")
    print("model-derived route board PASS" if passed else "model-derived route board FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

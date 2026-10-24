#!/usr/bin/env python3
"""KV260 acceptance for the native Top-2 to M4 route-record bridge."""

import json
import os
import sys

import numpy as np
from pynq import allocate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import Device  # noqa: E402

EXPERTS = 16
STRIDE = 6784


def main():
    binary = np.fromfile(os.path.join(HERE, "moe_route_from_top2.bin"), np.uint8)
    device = Device(os.path.join(HERE, "picorv32.bit"), os.path.join(HERE, "rt_fw.bin"))
    buffers = []
    results = []
    try:
        def alloc(shape, dtype):
            value = allocate(shape=shape, dtype=dtype)
            buffers.append(value)
            return value

        code = alloc(binary.shape, np.uint8)
        values = alloc((2,), np.float32)
        ids = alloc((2,), np.uint32)
        route = alloc((8,), np.uint32)
        args = alloc((18,), np.uint32)
        code[:] = binary
        code.flush()
        args[:] = 0
        args[0:3] = [values.physical_address, ids.physical_address, route.physical_address]
        args[16:18] = [EXPERTS, STRIDE]
        args.flush()
        for name in ("ascending", "ties", "random"):
            case = os.path.join(HERE, name)
            source_values = np.load(os.path.join(case, "values_ref.npy")).reshape(2)
            source_ids = np.load(os.path.join(case, "indices_ref.npy")).reshape(2)
            values[:] = source_values
            ids[:] = source_ids
            route[:] = 0
            values.flush()
            ids.flush()
            route.flush()
            completion = device.call(code.physical_address, args.physical_address, perf=True)
            route.invalidate()
            ids_ok = np.array_equal(route[1:3], source_ids.astype(np.uint32))
            offsets_ok = np.array_equal(route[3:5], source_ids.astype(np.uint32) * STRIDE)
            selected = source_values.astype(np.float32)
            reference = np.exp(selected - selected[0]).astype(np.float32)
            reference /= reference.sum(dtype=np.float32)
            weights = route[5:7].view(np.float32)
            weights_ok = np.allclose(weights, reference, rtol=2e-4, atol=2e-4)
            ok = (completion["status"] == 0 and route[0] == 2 and route[7] == 0 and
                  ids_ok and offsets_ok and weights_ok)
            results.append({"name": name, "pass": bool(ok), "ids": route[1:3].tolist(),
                            "offsets": route[3:5].tolist(), "weights": weights.tolist(),
                            "cycles": int(completion["cycles"])})
            print(f"{name}: {'PASS' if ok else 'FAIL'} IDs {route[1:3].tolist()}, "
                  f"weights {weights.tolist()}, cycles {completion['cycles']}")
    finally:
        device.close()
        for value in reversed(buffers):
            value.freebuffer()
    passed = all(case["pass"] for case in results) and len(results) == 3
    with open(os.path.join(HERE, "board_report.json"), "w") as stream:
        json.dump({"pass": passed, "cases": results}, stream, indent=2)
        stream.write("\n")
    print("native Top-2 route bridge board PASS" if passed else "native Top-2 route bridge board FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

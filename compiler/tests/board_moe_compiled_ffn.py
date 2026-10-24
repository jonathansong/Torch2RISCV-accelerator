#!/usr/bin/env python3
"""Run model-derived compiler route and reusable FFN artifacts on KV260."""

import json
import os
import sys

import numpy as np
from pynq import allocate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sadesc  # noqa: E402
from pynq_matmul import DescList, Device  # noqa: E402


def main():
    with open(os.path.join(HERE, "moe_manifest.json")) as stream:
        manifest = json.load(stream)
    dim = int(manifest["dim"])
    experts = int(manifest["experts"])
    stride = int(manifest["expert_stride"])
    params = np.fromfile(os.path.join(HERE, "moe_params.bin"), np.uint8)
    code_bytes = np.fromfile(os.path.join(HERE, "moe_route.bin"), np.uint8)
    descriptor_d, _, exports = sadesc.read(open(os.path.join(HERE, "moe_ffn.sadesc"), "rb").read())
    if descriptor_d != 16 or len(exports) != 1 or exports[0][0] != "moe_top2_ffn" or exports[0][2] != 4:
        raise RuntimeError("invalid compiled FFN descriptor")
    template = DescList()
    template.rows = exports[0][1].tolist()
    cases = []
    for filename in ("moe_acceptance.npz", "moe_token_acceptance.npz"):
        reference = np.load(os.path.join(HERE, filename))
        vectors = reference["x"]
        if vectors.ndim == 1:
            vectors = np.broadcast_to(vectors, (len(reference["pairs"]), dim))
        for i, pair in enumerate(reference["pairs"]):
            cases.append((f"{filename}:{i}", vectors[i], reference["scores"][i],
                          pair, reference["routing"][i], reference["output"][i]))

    device = Device(os.path.join(HERE, "picorv32.bit"), os.path.join(HERE, "rt_fw.bin"))
    buffers, results = [], []
    try:
        def alloc(shape, dtype):
            value = allocate(shape=shape, dtype=dtype)
            buffers.append(value)
            return value

        code = alloc(code_bytes.shape, np.uint8)
        values = alloc((2,), np.float32)
        ids = alloc((2,), np.uint32)
        route = alloc((8,), np.uint32)
        args = alloc((18,), np.uint32)
        x = alloc((dim,), np.float32)
        packed = alloc(params.shape, np.uint8)
        output = alloc((dim,), np.float32)
        code[:] = code_bytes
        code.flush()
        packed[:] = params
        packed.flush()
        args[:] = 0
        args[0:3] = [values.physical_address, ids.physical_address, route.physical_address]
        args[16:18] = [experts, stride]
        args.flush()
        for name, source_x, scores, pair, routing, expected in cases:
            selected = np.asarray(scores)[np.asarray(pair, np.int32)]
            values[:] = selected
            ids[:] = pair
            route[:] = 0
            x[:] = source_x
            output[:] = 0
            for buffer in (values, ids, route, x, output):
                buffer.flush()
            rc = device.call(code.physical_address, args.physical_address, perf=True)
            route.invalidate()
            route_weights = route[5:7].view(np.float32)
            expected_offsets = np.asarray(pair, np.uint32) * np.uint32(stride)
            route_ok = (rc["status"] == 0 and route[0] == 2 and route[7] == 0 and
                        np.array_equal(route[1:3], pair) and
                        np.array_equal(route[3:5], expected_offsets) and
                        np.allclose(route_weights, routing, rtol=2e-4, atol=2e-4))
            if not route_ok:
                print(f"{name}: FAIL invalid route {route.tolist()}", flush=True)
                results.append({"name": name, "pass": False, "route": route.tolist(),
                                "route_cycles": int(rc["cycles"])})
                continue
            fc = device.wait(device.submit(template, bases=[x.physical_address,
                route.physical_address, packed.physical_address + manifest["expert_table_offset"],
                output.physical_address], perf=True))
            output.invalidate()
            error = float(np.max(np.abs(np.asarray(output) - expected)))
            passed = fc["status"] == 0 and np.allclose(output, expected, rtol=3e-3, atol=3e-3)
            print(f"{name}: {'PASS' if passed else 'FAIL'} experts {route[1:3].tolist()}, "
                  f"route {rc['cycles']} cycles, FFN {fc['cycles']} cycles, max error {error:.6g}",
                  flush=True)
            results.append({"name": name, "pass": bool(passed),
                            "ids": route[1:3].tolist(), "route_cycles": int(rc["cycles"]),
                            "ffn_cycles": int(fc["cycles"]), "max_error": error})
    finally:
        device.close()
        for buffer in reversed(buffers):
            buffer.freebuffer()
    passed = len(results) == len(cases) and all(case["pass"] for case in results)
    with open(os.path.join(HERE, "board_report.json"), "w") as stream:
        json.dump({"pass": passed, "cases": results}, stream, indent=2)
        stream.write("\n")
    print("model-derived route+FFN board PASS" if passed else "model-derived route+FFN board FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

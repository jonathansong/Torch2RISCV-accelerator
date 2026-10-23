#!/usr/bin/env python3
"""KV260 M4 acceptance for a host-packed Hugging Face MoE layer."""

import argparse
import json
import os
import sys

import numpy as np
from pynq import allocate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from moe_m3b import top2_ffn_template  # noqa: E402
from pynq_matmul import Device  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", default=os.path.join(HERE, "moe_top2.bin"))
    parser.add_argument("--params", default=os.path.join(HERE, "moe_params.bin"))
    parser.add_argument("--manifest", default=os.path.join(HERE, "moe_manifest.json"))
    parser.add_argument("--reference", default=os.path.join(HERE, "moe_acceptance.npz"))
    parser.add_argument("--bitfile", default=os.path.join(HERE, "picorv32.bit"))
    parser.add_argument("--firmware", default=os.path.join(HERE, "rt_fw.bin"))
    args = parser.parse_args()
    with open(args.manifest) as f:
        manifest = json.load(f)
    dim, hidden = int(manifest["dim"]), int(manifest["hidden"])
    experts, stride, d = int(manifest["experts"]), int(manifest["expert_stride"]), int(manifest["d"])
    if int(manifest["top_k"]) != 2:
        raise ValueError("M4 board acceptance currently requires top_k=2")
    params = np.fromfile(args.params, np.uint8)
    code_bytes = np.fromfile(args.kernel, np.uint8)
    reference = np.load(args.reference)
    expert_table_offset = int(manifest["expert_table_offset"])
    buffers = []
    device = Device(args.bitfile, args.firmware)
    failures = 0
    try:
        def alloc(shape, dtype):
            value = allocate(shape=shape, dtype=dtype)
            buffers.append(value)
            return value

        code = alloc((len(code_bytes),), np.uint8)
        scores, route = alloc((experts,), np.float32), alloc((8,), np.uint32)
        call_args = alloc((18,), np.uint32)
        x, packed, output = alloc((dim,), np.float32), alloc(params.shape, np.uint8), alloc((dim,), np.float32)
        code[:] = code_bytes; code.flush()
        x[:] = reference["x"]; x.flush()
        packed[:] = params; packed.flush()
        call_args[:] = 0
        call_args[0], call_args[1] = scores.physical_address, route.physical_address
        call_args[16], call_args[17] = experts, stride
        call_args.flush()
        dl = top2_ffn_template(dim, hidden, d, int(manifest["alignment"]), ret=False)
        for case, pair in enumerate(reference["pairs"]):
            scores[:] = reference["scores"][case]
            route[:] = 0; scores.flush(); route.flush()
            rc = device.call(code.physical_address, call_args.physical_address, perf=True)
            route.invalidate()
            ids = route[1:3].astype(np.int32)
            routing = route[5:7].view(np.float32)
            route_ok = (rc["status"] == 0 and np.array_equal(ids, pair) and
                        np.allclose(routing, reference["routing"][case], rtol=2e-4, atol=2e-4))
            output[:] = 0; output.flush()
            fc = device.wait(device.submit(dl, bases=[x.physical_address, route.physical_address,
                             packed.physical_address + expert_table_offset, output.physical_address], perf=True))
            output.invalidate()
            wanted = reference["output"][case]
            error = float(np.max(np.abs(np.asarray(output) - wanted)))
            ffn_ok = fc["status"] == 0 and np.allclose(output, wanted, rtol=3e-3, atol=3e-3)
            ok = route_ok and ffn_ok
            failures += not ok
            print(f"experts {int(pair[0])},{int(pair[1])}: {'PASS' if ok else 'FAIL'}, "
                  f"route cycles {rc['cycles']}, full FFN cycles {fc['cycles']}, max error {error:.6g}")
    finally:
        device.close()
        for buffer in reversed(buffers):
            buffer.freebuffer()
    architecture = manifest["source"]["architecture"]
    print(f"{architecture} layer {manifest['source']['layer']} board PASS" if not failures
          else f"{architecture} board FAIL: {failures} cases")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

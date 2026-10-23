#!/usr/bin/env python3
"""KV260 M2 acceptance: PicoRV32 top-1 writes sa_moe_route v1 through RT_CALL.

Copy this script, pynq_matmul.py, rt_fw.bin, the overlay and moe_top1.bin to
the board, then run:
  sudo python3 board_moe_top1.py --experts 4 --expert-stride 6784
"""

import argparse
import os
import sys

import numpy as np
from pynq import allocate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import Device  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", default=os.path.join(HERE, "moe_top1.bin"))
    parser.add_argument("--bitfile", default=os.path.join(HERE, "picorv32.bit"))
    parser.add_argument("--firmware", default=os.path.join(HERE, "rt_fw.bin"))
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--expert-stride", type=int, required=True)
    args = parser.parse_args()
    code_bytes = np.fromfile(args.kernel, np.uint8)
    code = allocate(shape=(len(code_bytes),), dtype=np.uint8)
    scores = allocate(shape=(args.experts,), dtype=np.float32)
    route = allocate(shape=(8,), dtype=np.uint32)
    call_args = allocate(shape=(18,), dtype=np.uint32)  # 16 bindings, then constants
    device = None
    failures = 0
    try:
        code[:] = code_bytes
        code.flush()
        call_args[:] = 0
        call_args[0] = scores.physical_address
        call_args[1] = route.physical_address
        call_args[16] = args.experts
        call_args[17] = args.expert_stride
        call_args.flush()
        device = Device(args.bitfile, args.firmware)
        for expert in range(args.experts):
            scores[:] = -4.0
            scores[expert] = 4.0
            route[:] = 0xDEADBEEF
            scores.flush()
            route.flush()
            completion = device.call(code.physical_address, call_args.physical_address, perf=True)
            route.invalidate()
            expected = [1, expert, 0, expert * args.expert_stride, 0, 0x3F800000, 0, 0]
            ok = completion["status"] == 0 and np.array_equal(route, expected)
            failures += not ok
            print(f"expert {expert}: {'PASS' if ok else 'FAIL'}, route {route.tolist()}, "
                  f"cycles {completion['cycles']}")
    finally:
        if device is not None:
            device.close()
        for buffer in (call_args, route, scores, code):
            buffer.freebuffer()
    print("moe_top1 board PASS" if not failures else f"moe_top1 board FAIL: {failures} cases")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

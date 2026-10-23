#!/usr/bin/env python3
"""KV260 M2 acceptance: PicoRV32 top-1 writes sa_moe_route v1 through RT_CALL.

Copy this script, pynq_matmul.py, rt_fw.bin, the overlay and moe_top1.bin to
the board, then use run_board.sh so the PYNQ and XRT environments are active.
"""

import argparse
import os
import sys

import numpy as np
from pynq import allocate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import DescList, Device, MEM_ACC, MEM_SPAD_A, laddr  # noqa: E402


D, DIM, HIDDEN = 16, 32, 64


def packed_gate(expert):
    """A deterministic W[HIDDEN,DIM] and its Wp[HIDDEN/D,DIM,D] bytes."""
    values = (np.arange(HIDDEN * DIM, dtype=np.int32) + expert * 17) % 255 - 127
    weight = values.astype(np.int8).reshape(HIDDEN, DIM)
    packed = np.ascontiguousarray(weight.reshape(HIDDEN // D, D, DIM).transpose(0, 2, 1)).view(np.uint8).ravel()
    return weight, packed


def selected_gate_list():
    row_bytes = DIM * D
    return (DescList()
            .ld(0, laddr(MEM_SPAD_A, 0), 1, DIM, DIM, base=0)
            .ldparam(12, 6, base=1)  # sa_moe_route v1: expert_offset[0]
            .gemv(0, laddr(MEM_ACC, 0), HIDDEN // D, row_bytes, row_bytes, 0,
                  base=2, dyn=[("ddr", 6, True)])
            .st(0, laddr(MEM_ACC, 0), 1, HIDDEN * 4, HIDDEN * 4, base=3)
            .end(0x5A))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", default=os.path.join(HERE, "moe_top1.bin"))
    parser.add_argument("--bitfile", default=os.path.join(HERE, "picorv32.bit"))
    parser.add_argument("--firmware", default=os.path.join(HERE, "rt_fw.bin"))
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--expert-stride", type=int, required=True)
    args = parser.parse_args()
    code_bytes = np.fromfile(args.kernel, np.uint8)
    code = scores = route = call_args = x = expert_table = output = None
    device = Device(args.bitfile, args.firmware)
    failures = 0
    try:
        code = allocate(shape=(len(code_bytes),), dtype=np.uint8)
        scores = allocate(shape=(args.experts,), dtype=np.float32)
        route = allocate(shape=(8,), dtype=np.uint32)
        call_args = allocate(shape=(18,), dtype=np.uint32)  # 16 bindings, then constants
        x = allocate(shape=(DIM,), dtype=np.int8)
        expert_table = allocate(shape=(args.experts * args.expert_stride,), dtype=np.uint8)
        output = allocate(shape=(HIDDEN,), dtype=np.int32)
        x[:] = np.arange(-DIM // 2, DIM // 2, dtype=np.int8)
        expert_table[:] = 0
        weights = []
        for expert in range(args.experts):
            weight, packed = packed_gate(expert)
            if len(packed) > args.expert_stride:
                raise ValueError(f"expert stride {args.expert_stride} is smaller than gate weight {len(packed)}")
            expert_table[expert * args.expert_stride:expert * args.expert_stride + len(packed)] = packed
            weights.append(weight)
        x.flush()
        expert_table.flush()
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
            route_ok = completion["status"] == 0 and np.array_equal(route, expected)
            output[:] = 0
            output.flush()
            gemv_completion = device.wait(device.submit(
                selected_gate_list(),
                bases=[x.physical_address, route.physical_address, expert_table.physical_address,
                       output.physical_address],
                perf=True))
            output.invalidate()
            wanted_output = weights[expert].astype(np.int32) @ np.asarray(x, dtype=np.int32)
            gemv_ok = gemv_completion["status"] == 0 and np.array_equal(output, wanted_output)
            ok = route_ok and gemv_ok
            failures += not ok
            print(f"expert {expert}: {'PASS' if ok else 'FAIL'}, route {route.tolist()}, "
                  f"route cycles {completion['cycles']}, GEMV cycles {gemv_completion['cycles']}, "
                  f"max |result| {int(np.max(np.abs(wanted_output)))}")
    finally:
        if device is not None:
            device.close()
        for buffer in (output, expert_table, x, call_args, route, scores, code):
            if buffer is not None:
                buffer.freebuffer()
    print("moe_top1 board PASS" if not failures else f"moe_top1 board FAIL: {failures} cases")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

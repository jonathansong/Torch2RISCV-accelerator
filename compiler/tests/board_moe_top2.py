#!/usr/bin/env python3
"""KV260 M3 acceptance: device top-2, two dynamic GEMVs and VE combine."""

import argparse
import os
import sys

import numpy as np
from pynq import allocate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import DescList, Device, MEM_ACC, MEM_SPAD_A, VOPS, laddr  # noqa: E402

D, DIM, HIDDEN = 16, 32, 64
T_IF, T_FF = 2 | 3 << 2, 3 | 3 << 2


def packed_gate(expert):
    values = (np.arange(HIDDEN * DIM, dtype=np.int32) + expert * 17) % 255 - 127
    weight = values.astype(np.int8).reshape(HIDDEN, DIM)
    packed = np.ascontiguousarray(weight.reshape(HIDDEN // D, D, DIM).transpose(0, 2, 1)).view(np.uint8).ravel()
    return weight, packed


def top2_list():
    strips, row_bytes = HIDDEN // D, DIM * D
    y0, y1 = 0, strips + 1
    return (DescList()
            .ld(0, laddr(MEM_SPAD_A, 0), 1, DIM, DIM, base=0)
            .ldparam(12, 6, base=1).ldparam(20, 7, base=1)
            .gemv(0, laddr(MEM_ACC, y0), strips, row_bytes, row_bytes, 0,
                  base=2, dyn=[("ddr", 6, True)])
            .ve(laddr(MEM_ACC, y0), 0, laddr(MEM_ACC, y0), HIDDEN, VOPS["copy"], T_IF,
                fp=True, dyn=[("A", 7)])
            .ldparam(16, 6, base=1).ldparam(24, 7, base=1)
            .gemv(0, laddr(MEM_ACC, y1), strips, row_bytes, row_bytes, 0,
                  base=2, dyn=[("ddr", 6, True)])
            .ve(laddr(MEM_ACC, y1), 0, laddr(MEM_ACC, y1), HIDDEN, VOPS["copy"], T_IF,
                fp=True, dyn=[("A", 7)])
            .ve(laddr(MEM_ACC, y0), laddr(MEM_ACC, y1), laddr(MEM_ACC, y0), HIDDEN,
                VOPS["add"], T_FF, fp=True)
            .st(0, laddr(MEM_ACC, y0), 1, HIDDEN * 4, HIDDEN * 4, base=3)
            .end(0x5A))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", default=os.path.join(HERE, "moe_top2.bin"))
    parser.add_argument("--bitfile", default=os.path.join(HERE, "picorv32.bit"))
    parser.add_argument("--firmware", default=os.path.join(HERE, "rt_fw.bin"))
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--expert-stride", type=int, required=True)
    args = parser.parse_args()
    code_bytes = np.fromfile(args.kernel, np.uint8)
    buffers = []
    device = Device(args.bitfile, args.firmware)
    failures = 0
    try:
        def alloc(shape, dtype):
            value = allocate(shape=shape, dtype=dtype)
            buffers.append(value)
            return value
        code, scores, route = alloc((len(code_bytes),), np.uint8), alloc((args.experts,), np.float32), alloc((8,), np.uint32)
        call_args = alloc((18,), np.uint32)
        x = alloc((DIM,), np.int8)
        expert_table = alloc((args.experts * args.expert_stride,), np.uint8)
        output = alloc((HIDDEN,), np.float32)
        code[:] = code_bytes; code.flush()
        x[:] = np.arange(-DIM // 2, DIM // 2, dtype=np.int8); x.flush()
        expert_table[:] = 0
        weights = []
        for expert in range(args.experts):
            weight, packed = packed_gate(expert)
            if len(packed) > args.expert_stride:
                raise ValueError("expert stride is smaller than the gate weight")
            expert_table[expert * args.expert_stride:expert * args.expert_stride + len(packed)] = packed
            weights.append(weight)
        expert_table.flush()
        call_args[:] = 0
        call_args[0], call_args[1] = scores.physical_address, route.physical_address
        call_args[16], call_args[17] = args.experts, args.expert_stride
        call_args.flush()
        cases = ((0, 1), (1, 3), (2, 0), (3, 2))
        for first, second in cases:
            scores[:] = -4.0; scores[first] = 3.0; scores[second] = 1.0
            route[:] = 0; scores.flush(); route.flush()
            rc = device.call(code.physical_address, call_args.physical_address, perf=True)
            route.invalidate()
            ids = route[1:3].astype(np.int32)
            routing = route[5:7].view(np.float32)
            expected_routing = np.exp(np.array([3.0, 1.0], np.float32) - 3.0)
            expected_routing /= expected_routing.sum()
            route_ok = (rc["status"] == 0 and np.array_equal(ids, [first, second]) and
                        np.allclose(routing, expected_routing, rtol=2e-4, atol=2e-4))
            output[:] = 0; output.flush()
            gc = device.wait(device.submit(top2_list(), bases=[x.physical_address, route.physical_address,
                             expert_table.physical_address, output.physical_address], perf=True))
            output.invalidate()
            raw0 = weights[first].astype(np.int32) @ np.asarray(x, dtype=np.int32)
            raw1 = weights[second].astype(np.int32) @ np.asarray(x, dtype=np.int32)
            wanted = raw0.astype(np.float32) * routing[0] + raw1.astype(np.float32) * routing[1]
            combine_ok = gc["status"] == 0 and np.allclose(output, wanted, rtol=1e-6, atol=2e-3)
            ok = route_ok and combine_ok
            failures += not ok
            print(f"experts {first},{second}: {'PASS' if ok else 'FAIL'}, weights {routing.tolist()}, "
                  f"route cycles {rc['cycles']}, GEMV+combine cycles {gc['cycles']}")
    finally:
        device.close()
        for buffer in reversed(buffers):
            buffer.freebuffer()
    print("moe_top2 board PASS" if not failures else f"moe_top2 board FAIL: {failures} cases")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

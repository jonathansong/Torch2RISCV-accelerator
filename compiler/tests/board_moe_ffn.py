#!/usr/bin/env python3
"""KV260 M3b acceptance: device top-2 full W8A8 gated FFN."""

import argparse
import os
import sys

import numpy as np
from pynq import allocate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from moe_m3b import expert_layout, top2_ffn_template  # noqa: E402
from pynq_matmul import Device  # noqa: E402

D, DIM, HIDDEN = 16, 32, 64


def quantize_rows(weight):
    scale = np.max(np.abs(weight), axis=-1).astype(np.float32) / np.float32(127.0)
    scale = np.where(scale == 0, np.float32(1.0), scale).astype(np.float32)
    return np.clip(np.rint(weight / scale[..., None]), -127, 127).astype(np.int8), scale


def pack_linear(weight):
    n, k = weight.shape
    return np.ascontiguousarray(weight.reshape(n // D, D, k).transpose(0, 2, 1)).view(np.uint8).ravel()


def make_experts(count, stride, seed=47):
    rng = np.random.default_rng(seed)
    layout = expert_layout(DIM, HIDDEN)
    if stride != layout["stride"]:
        raise ValueError(f"expert stride must be {layout['stride']} for DIM={DIM}, HIDDEN={HIDDEN}")
    table = np.zeros(count * stride, np.uint8)
    experts = []
    for expert in range(count):
        gate, gs = quantize_rows(rng.normal(0, 1 / np.sqrt(DIM), (HIDDEN, DIM)).astype(np.float32))
        up, us = quantize_rows(rng.normal(0, 1 / np.sqrt(DIM), (HIDDEN, DIM)).astype(np.float32))
        down, ds = quantize_rows(rng.normal(0, 1 / np.sqrt(HIDDEN), (DIM, HIDDEN)).astype(np.float32))
        base = expert * stride
        for name, data in (("gate_w", pack_linear(gate)), ("gate_s", gs.view(np.uint8)),
                           ("up_w", pack_linear(up)), ("up_s", us.view(np.uint8)),
                           ("down_w", pack_linear(down)), ("down_s", ds.view(np.uint8))):
            table[base + layout[name]:base + layout[name] + data.size] = data
        experts.append((gate, gs, up, us, down, ds))
    return table, experts


def quantize_activation(value):
    scale = np.float32(np.max(np.abs(value)) / np.float32(127.0))
    scale = np.float32(1.0) if scale == 0 else scale
    return np.clip(np.rint(value / scale), -127, 127).astype(np.int8), scale


def expert_reference(xq, xs, weights):
    gate, gs, up, us, down, ds = weights
    g = (gate.astype(np.int32) @ xq.astype(np.int32)).astype(np.float32) * gs * xs
    u = (up.astype(np.int32) @ xq.astype(np.int32)).astype(np.float32) * us * xs
    hidden = g / (np.float32(1.0) + np.exp(-g).astype(np.float32)) * u
    hq, hs = quantize_activation(hidden)
    return (down.astype(np.int32) @ hq.astype(np.int32)).astype(np.float32) * ds * hs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", default=os.path.join(HERE, "moe_top2.bin"))
    parser.add_argument("--bitfile", default=os.path.join(HERE, "picorv32.bit"))
    parser.add_argument("--firmware", default=os.path.join(HERE, "rt_fw.bin"))
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--expert-stride", type=int, default=6784)
    args = parser.parse_args()
    code_bytes = np.fromfile(args.kernel, np.uint8)
    packed, experts = make_experts(args.experts, args.expert_stride)
    buffers = []
    device = Device(args.bitfile, args.firmware)
    failures = 0
    try:
        def alloc(shape, dtype):
            value = allocate(shape=shape, dtype=dtype)
            buffers.append(value)
            return value

        code = alloc((len(code_bytes),), np.uint8)
        scores, route = alloc((args.experts,), np.float32), alloc((8,), np.uint32)
        call_args = alloc((18,), np.uint32)
        x, expert_table, output = alloc((DIM,), np.float32), alloc(packed.shape, np.uint8), alloc((DIM,), np.float32)
        code[:] = code_bytes; code.flush()
        x[:] = np.linspace(-0.75, 0.875, DIM, dtype=np.float32); x.flush()
        expert_table[:] = packed; expert_table.flush()
        call_args[:] = 0
        call_args[0], call_args[1] = scores.physical_address, route.physical_address
        call_args[16], call_args[17] = args.experts, args.expert_stride
        call_args.flush()
        dl = top2_ffn_template(DIM, HIDDEN, D, ret=False)
        xq, xs = quantize_activation(np.asarray(x))
        for first, second in ((0, 1), (1, 3), (2, 0), (3, 2)):
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
            fc = device.wait(device.submit(dl, bases=[x.physical_address, route.physical_address,
                             expert_table.physical_address, output.physical_address], perf=True))
            output.invalidate()
            wanted = routing[0] * expert_reference(xq, xs, experts[first])
            wanted += routing[1] * expert_reference(xq, xs, experts[second])
            ffn_ok = fc["status"] == 0 and np.allclose(output, wanted, rtol=3e-3, atol=3e-3)
            ok = route_ok and ffn_ok
            failures += not ok
            error = float(np.max(np.abs(np.asarray(output) - wanted)))
            print(f"experts {first},{second}: {'PASS' if ok else 'FAIL'}, weights {routing.tolist()}, "
                  f"route cycles {rc['cycles']}, full FFN cycles {fc['cycles']}, max error {error:.6g}")
    finally:
        device.close()
        for buffer in reversed(buffers):
            buffer.freebuffer()
    print("moe_top2_ffn board PASS" if not failures else f"moe_top2_ffn board FAIL: {failures} cases")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

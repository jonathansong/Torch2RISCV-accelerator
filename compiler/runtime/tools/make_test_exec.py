#!/usr/bin/env python3
"""C1 test executable: hand-written sa-desc-v1 exports plus their test data.

Until the compiler generates executables (C2), the runtime is tested with
templates built by the hand-written generator (llm/compile_layer.py):

  export 0 "qlinear": bindings x (fp32[k]), weights (packed W | s_w), y (fp32[n]);
                      y = dequant(quant(x) W^T), as DeviceModel.linear(quant_act(x))
  export 1 "axpb":    bindings x (fp32[m]), y (fp32[m]); constants A, B (fp32 bits);
                      y = x * A + B (push constants -> PARAM0 / PARAM1 -> dynamic fields)
  export 2 "fault":   binding x; an LD past the end of the accumulator memory: the
                      device reports a range error (the driver's error path)

The expected outputs come from the functional simulator running each export
the way the sa HAL driver does (sadesc.dispatch_list: SETREG bases and
params, CALL, END) and are checked against NumPy / DeviceModel.

    python3 compiler/runtime/tools/make_test_exec.py --d 8 --out build/c1/test
writes test.sadesc and x0/w0/y0 (qlinear), x1/y1 (axpb) .bin files plus
test.txt (sizes and constants) for the C test program.
"""
import argparse
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "llm"))
import compile_layer as CL  # noqa: E402
import sadesc  # noqa: E402
from compile_layer import F32, MEM_ACC, T_FF, acc  # noqa: E402
from export_w8a8 import pack_b  # noqa: E402
from pynq_matmul import VOPS, DescList  # noqa: E402
from ref_model import DeviceModel, SfuExact, quantize_rows  # noqa: E402
from sa_funcsim import SaError, SaFuncSim  # noqa: E402

K, N, M = 64, 160, 96


def qlinear_template(d):
    lay = CL.Layout(d)
    x, tmp, y = lay.acc0, lay.acc0 + 64, lay.acc0 + 72
    wbytes = K * N                                          # packed W, then s_w
    dl = DescList().ld(0, acc(x), 1, 4 * K, 4 * K, base=0)
    s_x = CL.quant_act(dl, lay, x, K, tmp)
    CL.linear(dl, lay, K, N, 0, wbytes, s_x, out=y, wbase=1)
    dl.st(0, acc(y), 1, 4 * N, 4 * N, base=2)
    return dl


def axpb_template(d):
    lay = CL.Layout(d)
    x = lay.acc0
    dl = DescList().ld(0, acc(x), 1, 4 * M, 4 * M, base=0)
    dl.ve(acc(x), 0, acc(x), M, VOPS["copy"], T_FF, fp=True, dyn=[("A", 0), ("B", 1)])
    dl.st(0, acc(x), 1, 4 * M, 4 * M, base=1)
    return dl


def fault_template(d):
    depth = SaFuncSim(d, 0x10000000, 0x1000).depth(MEM_ACC)
    return DescList().ld(0, acc(depth - 1), 4, 64, 64, base=0)


def f32bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def run_in_sim(d, blob, export, buffers, constants):
    """Place the executable and the buffers in a simulated DDR, run the dispatch
    list the driver would build; returns the buffers after the run."""
    base, size = 0x10000000, 0x200000
    sim = SaFuncSim(d, base, size)
    _, _, exports = sadesc.read(blob)
    tmpl = base
    sim.ddr_write(tmpl, b"".join(r.tobytes() for _, r, _, _, _ in exports))
    offs, o = [], 0
    for _, r, _, _, _ in exports:
        offs.append(o)
        o += r.nbytes
    phys, a = [], base + 0x10000
    for b in buffers:
        phys.append(a)
        sim.ddr_write(a, b)
        a += -(-len(b) // 4096) * 4096
    dl = sadesc.dispatch_list(tmpl + offs[export], phys, constants)
    sim.ddr_write(base + 0x100000, dl.array().tobytes())
    n = sim.run_list(base + 0x100000)
    return [sim.ddr_read(p, len(b)).tobytes() for p, b in zip(phys, buffers)], n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = args.d
    os.makedirs(args.out, exist_ok=True)
    ex = sadesc.Executable(d, caps=sadesc.CAPS_DESC | sadesc.CAPS_CMDX | sadesc.CAPS_FPVE)
    ex.add("qlinear", qlinear_template(d), bindings=3)
    ex.add("axpb", axpb_template(d), bindings=2, constants=2)
    ex.add("fault", fault_template(d), bindings=1)
    blob = ex.to_bytes()
    with open(os.path.join(args.out, "test.sadesc"), "wb") as f:
        f.write(blob)

    rng = np.random.default_rng(3)
    w = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
    wq, sw = quantize_rows(w)
    x0 = (rng.standard_normal(K) * 2).astype(np.float32)
    wblob = pack_b(wq, d).tobytes() + sw.tobytes()
    (x0o, _, y0), n0 = run_in_sim(d, blob, 0, [x0.tobytes(), wblob, bytes(4 * N)], [])
    dm = DeviceModel.__new__(DeviceModel)                  # only quant_act / linear are used
    dm.d, dm.sfu = d, SfuExact
    want0 = dm.linear(*dm.quant_act(x0), (wq.astype(np.float64), sw)).astype(np.float32)
    ok0 = y0 == want0.tobytes()

    x1 = rng.standard_normal(M).astype(np.float32)
    A, B = 0.5, -2.25
    (_, y1), n1 = run_in_sim(d, blob, 1, [x1.tobytes(), bytes(4 * M)], [f32bits(A), f32bits(B)])
    want1 = ((x1 * np.float32(A)).astype(np.float32) + np.float32(B)).astype(np.float32)
    ok1 = y1 == want1.tobytes()

    try:
        run_in_sim(d, blob, 2, [x0.tobytes()], [])
        ok2 = False
    except SaError as e:
        ok2 = True
        print(f"fault in the simulator: {e}")

    for name, data in (("x0", x0.tobytes()), ("w0", wblob), ("y0", y0), ("x1", x1.tobytes()), ("y1", y1)):
        with open(os.path.join(args.out, name + ".bin"), "wb") as f:
            f.write(data)
    with open(os.path.join(args.out, "test.txt"), "w") as f:
        f.write(f"d {d}\nk {K}\nn {N}\nm {M}\nA {f32bits(A)}\nB {f32bits(B)}\n")
    print(f"{os.path.join(args.out, 'test.sadesc')}: {len(blob)} bytes, D = {d}, exports "
          + ", ".join(f"{e[0]} ({e[1].shape[0]} descriptors)" for e in sadesc.read(blob)[2]))
    print(f"qlinear in the simulator ({n0} descriptors): {'bit-exact with DeviceModel' if ok0 else 'DIFFERENT'}")
    print(f"axpb in the simulator ({n1} descriptors): {'bit-exact with NumPy' if ok1 else 'DIFFERENT'}")
    return 0 if ok0 and ok1 and ok2 else 1


if __name__ == "__main__":
    sys.exit(main())

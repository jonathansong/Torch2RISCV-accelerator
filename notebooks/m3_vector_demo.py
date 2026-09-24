#!/usr/bin/env python3
"""M3 demo: the vector engine, fused after matmul (int8 GEMM) and standalone.

Put these files in one directory on the board:
    picorv32.bit, picorv32.hwh   RISCV-on-PYNQ-Z1/build/output/ (scripts/build_bitstream.sh)
    gemm_fw.bin                  firmware/gemm/ (make)         - GEMM + VE epilogue
    vector_fw.bin                firmware/vector/ (make)       - standalone vector ops
    matmul_insn_fw.bin           firmware/matmul_insn/ (make)  - Phase 4 path (legacy)
    pynq_matmul.py               driver/
    m3_vector_demo.py            this file

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \
                  && cd /home/xilinx/m3 && python3 m3_vector_demo.py'

Acceptance (docs/double_buffer_design.md §12, M3): int8 GEMM with bias + RELU +
requantization computed on the chip, bit-exact with NumPy; every vector op and
type combination bit-exact standalone; the int32 GEMM and the legacy firmware
still pass; cycles measured.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import (MatmulOverlay, Requant, golden, qgemm_golden, regression,  # noqa: E402
                         vector_golden)

QSHAPES = [  # M, N, K, bias, relu, Requant
    (8, 8, 8, True, True, Requant(181, 15, -3)),
    (24, 48, 40, False, False, Requant(-97, 12, 7)),
    (64, 64, 64, True, True, Requant(300, 16, 0)),
    (128, 128, 128, True, False, Requant(51, 16, 10, -100, 100)),
    (256, 256, 256, True, True, Requant(9, 16, -128)),
]

I8, I16, I32 = np.int8, np.int16, np.int32
VCASES = [  # op, in, out, n, y elements (None = n), relu, Requant
    ("add", I8, I8, 4096, None, False, None),
    ("sub", I8, I16, 4096, None, False, None),
    ("mul", I8, I16, 4096, None, False, None),
    ("mul", I16, I32, 4096, None, False, None),
    ("max", I16, I16, 2048, 8, False, None),                        # broadcast
    ("min", I8, I8, 4000, None, False, Requant(1, 0, 0, -50, 50)),  # clamp window
    ("add", I32, I8, 8192, 64, True, Requant(181, 15, -3)),         # bias (period 8) + relu + requant
    ("copy", I32, I32, 8200, None, True, None),                     # 1025 groups: 3 chunks
    ("add", I32, I32, 16384, None, False, None),                    # saturating int32
    ("sub", I8, I32, 12000, 24, False, None),                       # period 3
]


def rand(rng, dtype, n):
    if dtype == I32:   # mostly moderate values, a few near the int32 limits
        v = rng.integers(-2**24, 2**24, n, dtype=np.int64)
        big = rng.random(n) < 0.05
        v[big] = rng.integers(-2**31, 2**31, int(big.sum()), dtype=np.int64)
        return v.astype(I32)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max + 1, n, dtype=np.int64).astype(dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("--dir", default=HERE, help="directory with the firmware .bin files")
    args = ap.parse_args()
    fw = lambda name: os.path.join(args.dir, name)

    mm = MatmulOverlay(args.bit, fw("gemm_fw.bin"))
    rng = np.random.default_rng(0)
    ok = True

    print("== fused: int8 C = requant(relu(A @ B + bias)), epilogue on the vector engine")
    print(f"{'M x N x K':>14} {'bias':>5} {'relu':>5} {'result':>7} {'cycles':>9} {'MAC/cycle':>10}"
          f" {'int32 GEMM':>11} {'MAC/cycle':>10}")
    for m, n, k, use_bias, relu, rq in QSHAPES:
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b = rng.integers(-128, 128, (k, n), dtype=np.int8)
        bias = rng.integers(-2**16, 2**16, n, dtype=np.int32) if use_bias else None
        c, st = mm.gemm(a, b, bias, quant=rq, relu=relu)
        good = np.array_equal(c, qgemm_golden(a, b, bias, rq, relu))
        c32, st32 = mm.gemm(a, b)                      # the same GEMM with int32 output
        good &= np.array_equal(c32, golden(a, b))
        ok &= good
        print(f"{f'{m}x{n}x{k}':>14} {'yes' if use_bias else '-':>5} {'yes' if relu else '-':>5} "
              f"{'PASS' if good else 'FAIL':>7} {st['riscv_cycles']:9d} {st['mac_per_cycle']:10.1f}"
              f" {st32['riscv_cycles']:11d} {st32['mac_per_cycle']:10.1f}")

    print("\n== standalone vector operations (DDR -> SPAD/ACC -> VE -> DDR)")
    print(f"{'op':>5} {'in->out':>10} {'n':>6} {'y':>6} {'post':>13} {'result':>7} {'cycles':>8}"
          f" {'elem/cycle':>11} {'B/cycle':>8}")
    for op, it, ot, n, ny, relu, rq in VCASES:
        x = rand(rng, it, n)
        y = rand(rng, it, ny or n)
        out, st = mm.vector(op, x, y, out_dtype=ot, relu=relu, requant=rq)
        good = np.array_equal(out, vector_golden(op, x, y, ot, relu, rq))
        ok &= good
        moved = x.nbytes + (0 if op == "copy" else y.nbytes) + out.nbytes
        post = "+".join(p for p, on in (("relu", relu), ("requant", rq is not None)) if on) or "-"
        print(f"{op:>5} {f'{np.dtype(it).name[3:]}->{np.dtype(ot).name[3:]}':>10} {n:6d} "
              f"{ny or n:6d} {post:>13} {'PASS' if good else 'FAIL':>7} {st['riscv_cycles']:8d}"
              f" {st['elem_per_cycle']:11.2f} {moved / st['riscv_cycles']:8.2f}")

    print("\n== legacy firmware on the new hardware")
    mm.load_firmware(fw("matmul_insn_fw.bin"))
    passed, total, st = regression(mm, batches=4, batch_size=256, seed=3)
    ok &= passed == total
    print(f"matmul_insn_fw.bin   {passed}/{total} match NumPy, {st['riscv_cycles_per_job']:.1f} cycles/job")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

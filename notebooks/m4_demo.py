#!/usr/bin/env python3
"""M4 demo: the D = 16 array (8 DSP columns + 8 LUT columns, VL = 16) on one HP port.

Put these files in one directory on the board:
    picorv32.bit, picorv32.hwh   RISCV-on-PYNQ-Z1/build/output/ (scripts/build_bitstream.sh)
    gemm_fw.bin                  firmware/gemm/ (make)         - GEMM (+ VE epilogue)
    vector_fw.bin                firmware/vector/ (make)       - standalone vector ops
    bwtest_fw.bin                firmware/bwtest/ (make)       - DMA bandwidth
    matmul_insn_fw.bin           firmware/matmul_insn/ (make)  - Phase 4 path (legacy)
    matmul_fw.bin                firmware/matmul/ (make)       - Phase 3 path (legacy)
    pynq_matmul.py               driver/
    m4_demo.py                   this file

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \
                  && cd /home/xilinx/m4 && python3 m4_demo.py'

The firmware reads D from the CAPS register and the driver from the .hwh, so
the same files also run on the M3 (D = 8) overlay for comparison.

Acceptance (docs/double_buffer_design.md §12, M4): everything of M1-M3 bit-exact
at D = 16; GEMM MAC/cycle and DMA bandwidth measured, to decide whether the
three HP ports are needed (§10.1).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import (MatmulOverlay, Requant, golden, qgemm_golden, regression,  # noqa: E402
                         vector_golden)

I8, I16, I32 = np.int8, np.int16, np.int32
M3_MAC = {(64, 64, 64): 37.5, (128, 128, 128): 49.7, (256, 256, 256): 56.5}   # D = 8 board (M3)


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
    d = mm.d
    rng = np.random.default_rng(0)
    ok = True
    print(f"overlay: D = {d}, {mm.nports} DMA port(s), peak {d * d} MAC/cycle")

    print("\n== int32 GEMM (resident B, double-buffered A strips, one exec per strip)")
    print(f"{'M x N x K':>16} {'bias':>5} {'result':>7} {'cycles':>9} {'us':>9} {'MAC/cycle':>10}"
          f" {'% peak':>7} {'M3 (D=8)':>9}")
    shapes = [(d, d, d, False), (3 * d, 2 * d, 5 * d, True), (64, 64, 64, False),
              (128, 128, 128, False), (32, 256, 64, True), (256, 256, 256, False),
              (512, 256, 256, False), (256, 128, 1024, False)]
    for m, n, k, use_bias in shapes:
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b = rng.integers(-128, 128, (k, n), dtype=np.int8)
        bias = rng.integers(-2**20, 2**20, (m, n), dtype=np.int32) if use_bias else None
        c, st = mm.gemm(a, b, bias)
        good = np.array_equal(c, golden(a, b) + (bias if use_bias else 0))
        ok &= good
        ref = M3_MAC.get((m, n, k))
        print(f"{f'{m}x{n}x{k}':>16} {'yes' if use_bias else '-':>5} {'PASS' if good else 'FAIL':>7} "
              f"{st['riscv_cycles']:9d} {st['us']:9.1f} {st['mac_per_cycle']:10.1f}"
              f" {100 * st['mac_per_cycle'] / (d * d):6.1f}% {f'{ref:.1f}' if ref else '-':>9}")

    print("\n== fused int8 GEMM: requant(relu(A @ B + bias)) on the vector engine")
    print(f"{'M x N x K':>16} {'bias':>5} {'relu':>5} {'result':>7} {'cycles':>9} {'MAC/cycle':>10}")
    for m, n, k, use_bias, relu, rq in [
            (d, d, d, True, True, Requant(181, 15, -3)),
            (3 * d, 3 * d, 5 * d, False, False, Requant(-97, 12, 7)),
            (64, 64, 64, True, True, Requant(300, 16, 0)),
            (128, 128, 128, True, False, Requant(51, 16, 10, -100, 100)),
            (256, 256, 256, True, True, Requant(9, 16, -128))]:
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b = rng.integers(-128, 128, (k, n), dtype=np.int8)
        bias = rng.integers(-2**16, 2**16, n, dtype=np.int32) if use_bias else None
        c, st = mm.gemm(a, b, bias, quant=rq, relu=relu)
        good = np.array_equal(c, qgemm_golden(a, b, bias, rq, relu))
        ok &= good
        print(f"{f'{m}x{n}x{k}':>16} {'yes' if use_bias else '-':>5} {'yes' if relu else '-':>5} "
              f"{'PASS' if good else 'FAIL':>7} {st['riscv_cycles']:9d} {st['mac_per_cycle']:10.1f}")

    print("\n== standalone vector operations (DDR -> SPAD/ACC -> VE -> DDR)")
    print(f"{'op':>5} {'in->out':>10} {'n':>6} {'y':>6} {'post':>13} {'result':>7} {'cycles':>8}"
          f" {'elem/cycle':>11} {'B/cycle':>8}")
    for op, it, ot, n, ny, relu, rq in [
            ("add", I8, I8, 4096, None, False, None),
            ("sub", I8, I16, 4096, None, False, None),
            ("mul", I8, I16, 4096, None, False, None),
            ("mul", I16, I32, 4096, None, False, None),
            ("max", I16, I16, 2048, d, False, None),                       # broadcast
            ("min", I8, I8, 4000 // d * d, None, False, Requant(1, 0, 0, -50, 50)),
            ("add", I32, I8, 8192, 8 * d, True, Requant(181, 15, -3)),     # bias (period 8)
            ("copy", I32, I32, 513 * 16, None, True, None),                # several chunks
            ("add", I32, I32, 16384, None, False, None),                   # saturating int32
            ("sub", I8, I32, 12000 // (3 * d) * 3 * d, 3 * d, False, None)]:  # period 3
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

    print("\n== DMA bandwidth (one HP port, 8 B/cycle peak)")
    mm.load_firmware(fw("bwtest_fw.bin"))
    res, copies_ok = mm.bandwidth()
    ok &= copies_ok
    for name, nbytes, cyc, bpc in res:
        print(f"{name:45} {nbytes:7d} B {cyc:8d} cycles {bpc:6.2f} B/cycle")
    print(f"stored copies {'match' if copies_ok else 'DIFFER'}")

    print("\n== legacy firmware (8x8x8 jobs zero-padded to D x D)")
    for name in ("matmul_insn_fw.bin", "matmul_fw.bin"):
        mm.load_firmware(fw(name))
        passed, total, st = regression(mm, batches=4, batch_size=256, seed=3)
        ok &= passed == total
        print(f"{name:20} {passed}/{total} match NumPy, {st['riscv_cycles_per_job']:.1f} cycles/job")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Descriptor DMA: the same work issued command by command (PCPI) and as one list.

Needs an overlay with the descriptor fetch unit (CAPS bit 21; rtl/sysarray/
sa_cmdfetch.v) and the performance counters. Put these files in one directory
on the board:
    picorv32.bit, picorv32.hwh   RISCV-on-PYNQ-Z1/build/output/ (scripts/build_bitstream.sh)
    gemm_fw.bin, vector_fw.bin   firmware/gemm, firmware/vector (PCPI path)
    desc_run_fw.bin              firmware/desc_run (list path: one mat_submit)
    pynq_matmul.py               driver/
    m5_desc_demo.py              this file

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \\
                  && cd /home/xilinx/m5 && python3 m5_desc_demo.py'

The list path runs the same commands in the same order (build_gemm_list /
build_vector_list mirror the firmware schedules); the ARM builds the list in
DDR and PicoRV32 only submits it. Every result is checked against NumPy.
"front end idle" = scheduler head empty + everything idle (performance
counters): the time the hardware waits for commands.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import (MEM_SPAD_A, MEM_SPAD_B, SPAD_BYTES, VOPS, DescList,  # noqa: E402
                         MatmulOverlay, Requant, allocate, golden, laddr, qgemm_golden,
                         vector_golden)

I8, I16, I32 = np.int8, np.int16, np.int32


def front_idle(st):
    p = st.get("perf")
    return (p["STARVE"] + p["ALL_IDLE"]) / p["CYCLES"] if p else float("nan")


def rand(rng, dtype, n):
    info = np.iinfo(dtype)
    lo, hi = (-2**24, 2**24) if dtype == I32 else (info.min, info.max + 1)
    return rng.integers(lo, hi, n, dtype=np.int64).astype(dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("--dir", default=HERE, help="directory with the firmware .bin files")
    args = ap.parse_args()

    mm = MatmulOverlay(args.bit, os.path.join(args.dir, "gemm_fw.bin"))
    d = mm.d
    rng = np.random.default_rng(5)
    ok = True
    print(f"overlay: D = {d}, peak {d * d} MAC/cycle")

    print("\n== GEMM: PCPI firmware vs one descriptor list")
    print(f"{'GEMM':>22} {'result':>7} {'PCPI cyc':>9} {'list cyc':>9} {'speedup':>8} "
          f"{'desc':>5} {'front idle PCPI':>16} {'list':>6}")
    for m, n, k, int8 in [(d, d, d, False), (64, 64, 64, False), (32, 256, 64, False),
                          (128, 128, 128, False), (256, 256, 256, False),
                          (64, 64, 64, True), (256, 256, 256, True)]:
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b = rng.integers(-128, 128, (k, n), dtype=np.int8)
        if int8:
            rq, bias = Requant(181, 15, -3), rng.integers(-2**16, 2**16, n, dtype=np.int32)
            exp = qgemm_golden(a, b, bias, rq, True)
            c1, s1 = mm.gemm(a, b, bias, quant=rq, relu=True)
            c2, s2 = mm.gemm_list(a, b, bias, quant=rq, relu=True)
        else:
            exp = golden(a, b)
            c1, s1 = mm.gemm(a, b)
            c2, s2 = mm.gemm_list(a, b)
        good = np.array_equal(c1, exp) and np.array_equal(c2, exp) and s2["dl_exec"] == s2["descriptors"]
        ok &= good
        name = f"{m}x{n}x{k}{' int8' if int8 else ''}"
        print(f"{name:>22} {'PASS' if good else 'FAIL':>7} {s1['riscv_cycles']:9d} {s2['riscv_cycles']:9d} "
              f"{s1['riscv_cycles'] / s2['riscv_cycles']:7.2f}x {s2['descriptors']:5d} "
              f"{100 * front_idle(s1):15.1f}% {100 * front_idle(s2):5.1f}%")

    print("\n== vector operations: PCPI firmware vs one descriptor list")
    print(f"{'operation':>30} {'result':>7} {'PCPI cyc':>9} {'list cyc':>9} {'speedup':>8} {'desc':>5}")
    for op, it, ot, n, ny, relu, rq in [
            ("add", I8, I8, 4096, None, False, None),
            ("mul", I16, I32, 4096, None, False, None),
            ("add", I32, I32, 16384, None, False, None),
            ("add", I32, I8, 8192, 8 * d, True, Requant(181, 15, -3)),
            ("max", I16, I16, 2048, d, False, None)]:
        x, y = rand(rng, it, n), rand(rng, it, ny or n)
        exp = vector_golden(op, x, y, ot, relu, rq)
        o1, s1 = mm.vector(op, x, y, out_dtype=ot, relu=relu, requant=rq)
        o2, s2 = mm.vector_list(op, x, y, out_dtype=ot, relu=relu, requant=rq)
        good = np.array_equal(o1, exp) and np.array_equal(o2, exp)
        ok &= good
        name = f"{op} {np.dtype(it).name}->{np.dtype(ot).name} n={n}" + (f" y={ny}" if ny else "")
        print(f"{name:>30} {'PASS' if good else 'FAIL':>7} {s1['riscv_cycles']:9d} {s2['riscv_cycles']:9d} "
              f"{s1['riscv_cycles'] / s2['riscv_cycles']:7.2f}x {s2['descriptors']:5d}")

    print("\n== one list, reused: int8 add of 4096 elements, buffers through BASE0..2 (relocation)")
    n = 4096
    sbank = SPAD_BYTES // d // 2
    dl = DescList()
    dl.ld(0, laddr(MEM_SPAD_A, 0), n // 256, 256, 256, base=0)            # x = BASE0 + 0
    dl.ld(0, laddr(MEM_SPAD_B, 0), n // 256, 256, 256, base=1)            # y = BASE1 + 0
    dl.ve(laddr(MEM_SPAD_A, 0), laddr(MEM_SPAD_B, 0), laddr(MEM_SPAD_B, sbank // 2), n, VOPS["add"], 0)
    dl.st(0, laddr(MEM_SPAD_B, sbank // 2), n // 256, 256, 256, base=2)   # out = BASE2 + 0
    dl.end(0x4E5)
    for run in range(3):
        bufs = [allocate(shape=(n,), dtype=np.int8) for _ in range(3)]
        try:
            x, y = rand(rng, I8, n), rand(rng, I8, n)
            bufs[0][:], bufs[1][:], bufs[2][:] = x, y, 0
            for buf in bufs:
                buf.flush()
            st = mm.run_list(dl, bases=[buf.physical_address for buf in bufs] + [0])
            bufs[2].invalidate()
            good = np.array_equal(np.array(bufs[2]), vector_golden("add", x, y)) and st["dl_status"] == 0x4E5
        finally:
            for buf in bufs:
                buf.freebuffer()
        ok &= good
        print(f"  run {run}: {'PASS' if good else 'FAIL'}, {st['riscv_cycles']} cycles, "
              f"{st['descriptors']} descriptors, END status {st['dl_status']:#x}")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Performance counters: where the cycles go, measured on the board.

Needs an overlay with performance counters (CAPS bit 20; rtl/sysarray/sa_perf.v,
docs/perf_counters_and_desc_dma_plan.md P1-P3). Put these files in one
directory on the board:
    picorv32.bit, picorv32.hwh   RISCV-on-PYNQ-Z1/build/output/ (scripts/build_bitstream.sh)
    gemm_fw.bin, vector_fw.bin   firmware/gemm, firmware/vector (make)
    pynq_matmul.py               driver/
    m4_perf.py                   this file

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \\
                  && cd /home/xilinx/m4 && python3 m4_perf.py'

For every GEMM / vector operation: the result is checked against NumPy, the
counters against exact invariants (tiles, useful steps x D^2 = MACs, bytes
stored, commands), and the cycle breakdown is printed.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import (MatmulOverlay, Requant, format_breakdown, golden,  # noqa: E402
                         perf_breakdown, qgemm_golden, vector_golden)

GEMMS = [  # M, N, K, int8 output
    (64, 64, 64, False), (128, 128, 128, False), (32, 256, 64, False),
    (256, 256, 256, False), (512, 256, 256, False), (256, 128, 1024, False),
    (256, 256, 256, True),
]
VECTORS = [  # op, in, out, n, relu, requant
    ("add", np.int8, np.int8, 4096, False, None),
    ("mul", np.int16, np.int32, 4096, False, None),
    ("add", np.int32, np.int32, 16384, False, None),
    ("copy", np.int32, np.int8, 8192, True, Requant(181, 15, -3)),
]


# The counter window (mat_perf enable .. freeze) also covers the few instructions
# between mat_perf and rdcycle at both ends. On the board PicoRV32 fetches over
# the AXI interconnect (~10 cycles per instruction), so CYCLES exceeds the
# firmware's rdcycle window by a constant ~100-200 cycles.
WINDOW_SLACK = 512


def window(ok_list, pf, st):
    off = pf["CYCLES"] - st["riscv_cycles"]
    print(f"  counter window: CYCLES {pf['CYCLES']} = rdcycle window {st['riscv_cycles']} + {off}")
    check(ok_list, 0 <= off <= WINDOW_SLACK, f"CYCLES - rdcycle window = {off}, outside 0 .. {WINDOW_SLACK}")


def check(ok_list, cond, what):
    if not cond:
        print(f"  COUNTER CHECK FAILED: {what}")
    ok_list.append(bool(cond))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("--dir", default=HERE, help="directory with the firmware .bin files")
    args = ap.parse_args()

    mm = MatmulOverlay(args.bit, os.path.join(args.dir, "gemm_fw.bin"))
    d = mm.d
    rng = np.random.default_rng(3)
    checks = []
    print(f"overlay: D = {d}, peak {d * d} MAC/cycle")

    for m, n, k, int8 in GEMMS:
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b = rng.integers(-128, 128, (k, n), dtype=np.int8)
        if int8:
            rq = Requant(9, 16, -128)
            bias = rng.integers(-2**16, 2**16, n, dtype=np.int32)
            c, st = mm.gemm(a, b, bias, quant=rq, relu=True)
            good = np.array_equal(c, qgemm_golden(a, b, bias, rq, True))
        else:
            c, st = mm.gemm(a, b)
            good = np.array_equal(c, golden(a, b))
        pf = st["perf"]
        if pf is None:
            print("this overlay has no performance counters (CAPS bit 20 clear): build with P1 RTL")
            return 2
        bd = perf_breakdown(pf, d)
        tiles = (m // d) * (n // d)
        print(f"\n{m}x{n}x{k}{' int8' if int8 else ''}: {'PASS' if good else 'FAIL'}, "
              f"{st['riscv_cycles']} cycles, {st['mac_per_cycle']:.1f} MAC/cycle")
        print(format_breakdown(bd))
        check(checks, good, "result differs from NumPy")
        check(checks, pf["EX_TILES"] == tiles, f"EX_TILES {pf['EX_TILES']} != {tiles}")
        check(checks, bd["macs"] == m * n * k, f"EX_USEFUL x D^2 {bd['macs']} != {m * n * k}")
        check(checks, 8 * pf["ST_BEATS"] == (1 if int8 else 4) * m * n, "ST bytes")
        check(checks, pf["CMD_ST"] == m // d, "CMD_ST")
        check(checks, pf["CMD_VE"] == (m // d if int8 else 0), "CMD_VE")
        window(checks, pf, st)

    for op, it, ot, n, relu, rq in VECTORS:
        info = np.iinfo(it)
        x = rng.integers(info.min, info.max + 1, n, dtype=np.int64).astype(it)
        y = rng.integers(info.min, info.max + 1, n, dtype=np.int64).astype(it)
        out, st = mm.vector(op, x, y, out_dtype=ot, relu=relu, requant=rq)
        good = np.array_equal(out, vector_golden(op, x, y, ot, relu, rq))
        pf = st["perf"]
        bd = perf_breakdown(pf, d)
        print(f"\nvector {op} {np.dtype(it).name}->{np.dtype(ot).name} n={n}: {'PASS' if good else 'FAIL'}, "
              f"{st['riscv_cycles']} cycles, {st['elem_per_cycle']:.2f} elem/cycle")
        print(format_breakdown(bd))
        check(checks, good, "result differs from NumPy")
        check(checks, pf["VE_GROUPS"] == n // d, "VE_GROUPS")
        check(checks, pf["CMD_VE"] == st["chunks"], "CMD_VE != chunks")
        check(checks, 8 * pf["ST_BEATS"] == n * np.dtype(ot).itemsize, "ST bytes")
        window(checks, pf, st)

    ok = all(checks)
    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

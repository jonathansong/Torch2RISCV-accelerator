#!/usr/bin/env python3
"""M1 demo: tiled GEMM on the double-buffered accelerator, vs the Phase 3/4 paths.

Put these files in one directory on the board:
    picorv32.bit, picorv32.hwh   RISCV-on-PYNQ-Z1/build/output/ (scripts/build_bitstream.sh)
    gemm_fw.bin                  firmware/gemm/ (make)         - funct7 = 1 ISA
    matmul_insn_fw.bin           firmware/matmul_insn/ (make)  - Phase 4 path (legacy)
    matmul_fw.bin                firmware/matmul/ (make)       - Phase 3 path (legacy)
    pynq_matmul.py               driver/
    m1_gemm_demo.py              this file

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \
                  && cd /home/xilinx/m1 && python3 m1_gemm_demo.py'

Acceptance (docs/double_buffer_design.md §12, M1): NumPy-exact GEMMs of
several shapes (non-square, K >> D, with bias) through the new ISA; the
Phase 3/4 firmware still passes unchanged; cycles measured.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import MatmulOverlay, golden, regression  # noqa: E402

SHAPES = [  # M, N, K, bias
    (8, 8, 8, False),
    (24, 16, 40, True),
    (64, 64, 64, False),
    (128, 128, 128, False),
    (32, 256, 64, True),
    (256, 256, 256, False),
]


def legacy_gemm(mm, a, b):
    """The same GEMM as independent 8x8x8 jobs on the Phase 4 path; partial sums on the ARM."""
    m, k = a.shape
    n = b.shape[1]
    at = a.reshape(m // 8, 8, k // 8, 8).transpose(0, 2, 1, 3)   # [i, kk] -> 8x8
    bt = b.reshape(k // 8, 8, n // 8, 8).transpose(0, 2, 1, 3)   # [kk, j] -> 8x8
    ai, aj = np.meshgrid(np.arange(m // 8), np.arange(n // 8), indexing="ij")
    jobs_a, jobs_b = [], []
    for kk in range(k // 8):
        jobs_a.append(at[ai, kk].reshape(-1, 8, 8))
        jobs_b.append(bt[kk, aj].reshape(-1, 8, 8))
    c_parts, stats = mm.matmul(np.concatenate(jobs_a), np.concatenate(jobs_b), timeout=30.0)
    c = c_parts.reshape(k // 8, m // 8, n // 8, 8, 8).sum(axis=0)
    return c.transpose(0, 2, 1, 3).reshape(m, n), stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("--dir", default=HERE, help="directory with the firmware .bin files")
    args = ap.parse_args()
    fw = lambda name: os.path.join(args.dir, name)

    mm = MatmulOverlay(args.bit, fw("gemm_fw.bin"))
    rng = np.random.default_rng(0)
    ok = True

    print("== new ISA: tiled GEMM (resident B, double-buffered A strips and C tiles)")
    print(f"{'M x N x K':>16} {'bias':>5} {'result':>7} {'cycles':>9} {'us':>9} {'MAC/cycle':>10}")
    for m, n, k, use_bias in SHAPES:
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b = rng.integers(-128, 128, (k, n), dtype=np.int8)
        bias = rng.integers(-2**20, 2**20, (m, n), dtype=np.int32) if use_bias else None
        c, st = mm.gemm(a, b, bias)
        expect = golden(a, b) + (bias if use_bias else 0)
        good = np.array_equal(c, expect)
        ok &= good
        print(f"{f'{m}x{n}x{k}':>16} {'yes' if use_bias else '-':>5} {'PASS' if good else 'FAIL':>7} "
              f"{st['riscv_cycles']:9d} {st['us']:9.1f} {st['mac_per_cycle']:10.1f}")

    print("\n== the same 64x64x64 GEMM as 512 independent 8x8x8 jobs (Phase 4 path)")
    a = rng.integers(-128, 128, (64, 64), dtype=np.int8)
    b = rng.integers(-128, 128, (64, 64), dtype=np.int8)
    c_new, st_new = mm.gemm(a, b)
    mm.load_firmware(fw("matmul_insn_fw.bin"))
    c_old, st_old = legacy_gemm(mm, a, b)
    good = np.array_equal(c_new, golden(a, b)) and np.array_equal(c_old, golden(a, b))
    ok &= good
    print(f"new ISA:     {st_new['riscv_cycles']:8d} cycles ({st_new['mac_per_cycle']:.1f} MAC/cycle)")
    print(f"Phase 4 path:{st_old['riscv_cycles']:8d} cycles ({64**3 / st_old['riscv_cycles']:.1f} MAC/cycle), "
          f"plus the partial sums on the ARM")
    print(f"speedup {st_old['riscv_cycles'] / st_new['riscv_cycles']:.1f}x  {'PASS' if good else 'FAIL'}")

    print("\n== legacy firmware on the new hardware")
    for name in ("matmul_insn_fw.bin", "matmul_fw.bin"):
        mm.load_firmware(fw(name))
        passed, total, st = regression(mm, batches=4, batch_size=256, seed=3)
        ok &= passed == total
        print(f"{name:20} {passed}/{total} match NumPy, {st['riscv_cycles_per_job']:.1f} cycles/job")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Phase 3 demo: ARM -> PicoRV32 -> matmul unit.

Put these files in one directory on the board:
    picorv32.bit, picorv32.hwh   RISCV-on-PYNQ-Z1/build/output/ (scripts/build_bitstream.sh)
    matmul_fw.bin                firmware/matmul/ (make)
    pynq_matmul.py               driver/
    phase3_matmul_demo.py        this file

Run as root inside the PYNQ venv with XRT sourced:
    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \
                  && cd /home/xilinx/phase3 && python3 phase3_matmul_demo.py'

Flow per call: the ARM allocates A/B/C in DDR and passes their physical
addresses to the PicoRV32 through the BRAM mailbox; the PicoRV32 firmware
programs the matmul CSRs for every 8x8x8 job, polls for completion and
reports back; the ARM reads C and compares with NumPy.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import MatmulOverlay, golden, regression  # noqa: E402


def demo(mm):
    """End-to-end check on one batch of 64 random 8x8 tiles."""
    rng = np.random.default_rng(0)
    a = rng.integers(-128, 128, (64, 8, 8), dtype=np.int8)
    b = rng.integers(-128, 128, (64, 8, 8), dtype=np.int8)
    c, stats = mm.matmul(a, b)
    ok = np.array_equal(c, golden(a, b))
    print(f"[demo]       64 tiles: {'PASS' if ok else 'FAIL'}  {stats}")
    return ok


def regression_run(mm, batches, batch_size):
    """Random batches vs NumPy (plan 6.3); per-job cycles are the CSR-path baseline."""
    passed, total, stats = regression(mm, batches=batches, batch_size=batch_size, seed=1)
    print(f"[regression] {passed}/{total} matrices match NumPy ({100 * passed / total:.2f} %)")
    print(f"[latency]    per 8x8x8 job: {stats['riscv_cycles_per_job']:.0f} RISC-V cycles "
          f"({stats['us_per_job']:.2f} us @ 50 MHz), of which accelerator "
          f"{stats['accel_cycles_per_job']:.0f}")
    return passed == total


def extremes(mm):
    """int8 corner cases."""
    ok = True
    for name, av, bv in [("-128 x -128", -128, -128), ("-128 x 127", -128, 127), ("127 x 127", 127, 127)]:
        a = np.full((1, 8, 8), av, np.int8)
        b = np.full((1, 8, 8), bv, np.int8)
        c, _ = mm.matmul(a, b)
        good = np.array_equal(c, golden(a, b))
        ok &= good
        print(f"[extremes]   {name:12} -> C[0,0] = {c[0, 0, 0]:7d}  {'PASS' if good else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("--fw", default=os.path.join(HERE, "matmul_fw.bin"))
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    mm = MatmulOverlay(args.bit, args.fw)          # load overlay + firmware
    results = [demo(mm), regression_run(mm, args.batches, args.batch_size), extremes(mm)]
    print("PASS" if all(results) else "FAIL")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

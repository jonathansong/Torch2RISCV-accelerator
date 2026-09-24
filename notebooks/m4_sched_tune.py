#!/usr/bin/env python3
"""M4 schedule tuning: the GEMM firmware with and without A prefetch / B split.

Same files as m4_demo.py (only picorv32.bit/.hwh, gemm_fw.bin and
pynq_matmul.py are used):

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \
                  && cd /home/xilinx/m4 && python3 m4_sched_tune.py'

Every shape runs with each schedule (MBOX_GEMM_FLAGS); all results are checked
against NumPy. "M4" is the schedule measured in bitstreams/m4, "default" what
the firmware does without flags (prefetch; B split only for B >= 16 KB).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import (GEMM_FORCE_BSPLIT, GEMM_NO_BSPLIT, GEMM_NO_PREFETCH,  # noqa: E402
                         MatmulOverlay, Requant, golden, qgemm_golden)

# "default" = what gemm() does without flags: prefetch, B split only for B >= 16 KB
SCHEDULES = [("M4", GEMM_NO_PREFETCH | GEMM_NO_BSPLIT), ("prefetch", GEMM_NO_BSPLIT),
             ("B split", GEMM_NO_PREFETCH | GEMM_FORCE_BSPLIT), ("both", GEMM_FORCE_BSPLIT),
             ("default", 0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("--dir", default=HERE, help="directory with gemm_fw.bin")
    args = ap.parse_args()

    mm = MatmulOverlay(args.bit, os.path.join(args.dir, "gemm_fw.bin"))
    d = mm.d
    rng = np.random.default_rng(1)
    ok = True
    print(f"overlay: D = {d}, peak {d * d} MAC/cycle; MAC/cycle per schedule (cycles)")
    print(f"{'GEMM':>24} " + " ".join(f"{name:>17}" for name, _ in SCHEDULES) + f" {'gain':>6}")
    cases = [(64, 64, 64, False, None), (128, 128, 128, False, None), (48, 32, 80, True, None),
             (32, 256, 64, True, None), (256, 256, 256, False, None), (512, 256, 256, False, None),
             (256, 128, 1024, False, None), (256, 256, 256, True, Requant(9, 16, -128))]
    for m, n, k, use_bias, rq in cases:
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b = rng.integers(-128, 128, (k, n), dtype=np.int8)
        if rq is not None:
            bias = rng.integers(-2**16, 2**16, n, dtype=np.int32) if use_bias else None
            expect = qgemm_golden(a, b, bias, rq, True)
        else:
            bias = rng.integers(-2**20, 2**20, (m, n), dtype=np.int32) if use_bias else None
            expect = golden(a, b) + (bias if use_bias else 0)
        cells, mpc = [], []
        for _, flags in SCHEDULES:
            c, st = mm.gemm(a, b, bias, quant=rq, relu=rq is not None, flags=flags)
            good = np.array_equal(c, expect)
            ok &= good
            mpc.append(st["mac_per_cycle"])
            cells.append(f"{st['mac_per_cycle']:6.1f} ({st['riscv_cycles']:7d})" + ("" if good else "!"))
        label = f"{m}x{n}x{k}" + (" +bias" if use_bias else "") + (" int8" if rq is not None else "")
        print(f"{label:>24} " + " ".join(f"{x:>17}" for x in cells) + f" {mpc[-1] / mpc[0]:5.2f}x")

    print("PASS" if ok else "FAIL (! = result differs from NumPy)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

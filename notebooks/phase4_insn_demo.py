#!/usr/bin/env python3
"""Phase 4 demo: matmul through the custom instructions (PCPI), vs the CSR path.

Put these files in one directory on the board:
    picorv32.bit, picorv32.hwh   RISCV-on-PYNQ-Z1/build/output/ (scripts/build_bitstream.sh)
    matmul_insn_fw.bin           firmware/matmul_insn/ (make)  - mat_trigger / mat_wait
    matmul_fw.bin                firmware/matmul/ (make)       - CSR path, for comparison
    pynq_matmul.py               driver/
    phase4_insn_demo.py          this file

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \
                  && cd /home/xilinx/phase4 && python3 phase4_insn_demo.py'

Acceptance (plan 7.5): the custom-instruction firmware computes every
matmul correctly without any CSR access (the firmware only issues
mat_trigger/mat_wait/mat_cycles/mat_reset).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import MatmulOverlay, golden, regression  # noqa: E402


def check(mm, name, a, b):
    c, stats = mm.matmul(a, b)
    ok = np.array_equal(c, golden(a, b))
    print(f"[{name:10}] {len(a):4d} tiles: {'PASS' if ok else 'FAIL'}")
    return ok, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("--insn-fw", default=os.path.join(HERE, "matmul_insn_fw.bin"))
    ap.add_argument("--csr-fw", default=os.path.join(HERE, "matmul_fw.bin"))
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    mm = MatmulOverlay(args.bit, args.insn_fw)
    results = []

    # --- correctness through the custom instructions
    rng = np.random.default_rng(0)
    a = rng.integers(-128, 128, (64, 8, 8), dtype=np.int8)
    b = rng.integers(-128, 128, (64, 8, 8), dtype=np.int8)
    results.append(check(mm, "demo", a, b)[0])

    passed, total, _ = regression(mm, batches=args.batches, batch_size=args.batch_size, seed=1)
    print(f"[regression] {passed}/{total} matrices match NumPy ({100 * passed / total:.2f} %)")
    results.append(passed == total)

    for name, av, bv in [("-128x-128", -128, -128), ("-128x127", -128, 127), ("127x127", 127, 127)]:
        results.append(check(mm, name, np.full((1, 8, 8), av, np.int8), np.full((1, 8, 8), bv, np.int8))[0])

    # --- same batch through both paths (Phase 6 preview)
    a = rng.integers(-128, 128, (args.batch_size, 8, 8), dtype=np.int8)
    b = rng.integers(-128, 128, (args.batch_size, 8, 8), dtype=np.int8)
    rows = []
    for label, fw in (("CSR (Phase 3)", args.csr_fw), ("custom insn (Phase 4)", args.insn_fw)):
        mm.load_firmware(fw)
        ok, st = check(mm, label.split()[0], a, b)
        results.append(ok)
        rows.append((label, st))
    print()
    print(f"{'path':24} {'cycles/job':>11} {'us/job':>8} {'accel':>7} {'CPU side':>9}")
    for label, st in rows:
        per, acc = st["riscv_cycles_per_job"], st["accel_cycles_per_job"]
        print(f"{label:24} {per:11.1f} {st['us_per_job']:8.2f} {acc:7.1f} {per - acc:9.1f}")
    speedup = rows[0][1]["riscv_cycles_per_job"] / rows[1][1]["riscv_cycles_per_job"]
    print(f"custom-instruction path is {speedup:.2f}x faster per 8x8x8 job")

    print("PASS" if all(results) else "FAIL")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

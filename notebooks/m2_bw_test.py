#!/usr/bin/env python3
"""M2: DMA bandwidth self-test on the sysarray overlay (runs on the M1 bitstream too).

Files in one directory on the board: picorv32.bit / picorv32.hwh,
bwtest_fw.bin (firmware/bwtest), pynq_matmul.py (driver), this script.

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \\
                  && cd /home/xilinx/m2 && python3 m2_bw_test.py'

Each test moves 64 KB (test 5: 64 KB each way at once) and is timed on the
PicoRV32 (rdcycle around queue + mat_fence, 50 MHz). One 64-bit HP port at
50 MHz peaks at 8 B/cycle per direction (400 MB/s).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import MatmulOverlay, RISCV_HZ  # noqa: E402


def main():
    mm = MatmulOverlay(os.path.join(HERE, "picorv32.bit"), os.path.join(HERE, "bwtest_fw.bin"))
    res, copies_ok = mm.bandwidth()
    print(f"{'test':44} {'cycles':>8} {'B/cycle':>8} {'MB/s':>8} {'% of 8 B/c':>10}")
    for name, nbytes, cyc, bpc in res:
        print(f"{name:44} {cyc:8d} {bpc:8.2f} {bpc * RISCV_HZ / 1e6:8.1f} {100 * bpc / 8:9.1f}%")
    print("stored data matches source" if copies_ok else "STORED DATA MISMATCH")
    print("PASS" if copies_ok else "FAIL")
    return 0 if copies_ok else 1


if __name__ == "__main__":
    sys.exit(main())

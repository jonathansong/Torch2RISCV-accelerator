#!/usr/bin/env python3
"""M2: DMA bandwidth self-test on the sysarray overlay (runs on the M1 bitstream too).

Files in one directory on the board: picorv32.bit / picorv32.hwh,
bwtest_fw.bin (firmware/bwtest), pynq_matmul.py (driver), this script.

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \\
                  && cd /home/xilinx/m2 && python3 m2_bw_test.py'

Each test moves 64 KB (test 5: 64 KB each way at once) and is timed on the
PicoRV32 (rdcycle around queue + mat_fence, at the overlay's clock: 50 MHz on
the PYNQ-Z1 and in the KV260's K1a). One 64-bit HP port peaks at 8 B/cycle
per direction (400 MB/s at 50 MHz); the K2a overlays' 128-bit port at 16.
Rows of 8 bytes (test 4) fill only half of a 128-bit beat.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import MatmulOverlay  # noqa: E402


def main():
    mm = MatmulOverlay(os.path.join(HERE, "picorv32.bit"), os.path.join(HERE, "bwtest_fw.bin"))
    res, copies_ok = mm.bandwidth()
    bw = mm.info.dma_w // 8                            # peak bytes per cycle and direction
    print(f"{mm.info.board} overlay, D = {mm.d}, {mm.riscv_hz / 1e6:.1f} MHz, DMA {mm.info.dma_w} bits")
    print(f"{'test':44} {'cycles':>8} {'B/cycle':>8} {'MB/s':>8} {f'% of {bw} B/c':>10}")
    for name, nbytes, cyc, bpc in res:
        print(f"{name:44} {cyc:8d} {bpc:8.2f} {bpc * mm.riscv_hz / 1e6:8.1f} {100 * bpc / bw:9.1f}%")
    print("stored data matches source" if copies_ok else "STORED DATA MISMATCH")
    print("PASS" if copies_ok else "FAIL")
    return 0 if copies_ok else 1


if __name__ == "__main__":
    sys.exit(main())

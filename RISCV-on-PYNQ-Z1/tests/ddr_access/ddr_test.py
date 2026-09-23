#!/usr/bin/env python3
"""PicoRV32 -> DDR (S_AXI_HP0) board test for the PYNQ-Z1 overlay.

Copy to one directory on the board, then run as root (or `%run` in Jupyter):
    picorv32.bit  picorv32.hwh   (build/output/)
    ddr_test.bin                 (make -C tests/ddr_access)
    ddr_test.py

    sudo -i bash -c 'cd /home/xilinx/ddr_test && \
                     python3 ddr_test.py [--words N] [--bit picorv32.bit] [--fw ddr_test.bin]'

pynq needs both the image's venv (/etc/profile.d/pynq_venv.sh) and the XRT
environment (/etc/profile.d/xrt_setup.sh); plain `sudo python3` has neither
("No module named 'pynq'" / "No Devices Found"). A root login shell
(`sudo -i`) sources both, same as Jupyter.

Flow: hold RISC-V in reset -> load firmware into BRAM -> allocate src/dst
in DDR -> pass physical addresses through the mailbox -> release reset ->
wait for STATUS_DONE -> check dst from the ARM side.
"""
import argparse
import os
import sys
import time

import numpy as np
from pynq import GPIO, MMIO, Overlay, allocate

HERE = os.path.dirname(os.path.abspath(__file__))

# Addresses from scripts/pico_bit.tcl
BRAM_ARM_BASE = 0x40010000   # psBramController (RISC-V sees it at 0xC0000000)
BRAM_BYTES    = 0x2000       # RISC-V-visible part of the BRAM
INTC_BASE     = 0x40020000   # axi_intc, input 0 = PicoRV32 trap
RESET_EMIO    = 0            # PS GPIO EMIO[0] -> riscvReset aux_reset_in (1 = hold)

# Keep in sync with mailbox.h
MBOX_OFFSET    = 0x1F00
MBOX_STATUS    = 0x00
MBOX_SRC_ADDR  = 0x04
MBOX_DST_ADDR  = 0x08
MBOX_N_WORDS   = 0x0C
MBOX_SRC_SUM   = 0x10
MBOX_ERRORS    = 0x14
MBOX_CYCLES    = 0x18
STATUS_RUNNING = 0x00000001
STATUS_DONE    = 0x600D600D
SUB_BYTES      = 64
SUB_HALVES     = 32

GUARD_WORDS = 4
POISON      = 0xDEADBEEF
RISCV_HZ    = 50e6           # subprocessorClk default output


def load_firmware(bram, path):
    fw = open(path, "rb").read()
    if len(fw) > MBOX_OFFSET:
        sys.exit(f"{path}: {len(fw)} bytes overlaps the mailbox at {MBOX_OFFSET:#x}")
    fw += b"\0" * (-len(fw) % 4)
    for off in range(0, BRAM_BYTES, 4):           # clears .bss and mailbox too
        bram.write(off, 0)
    for off, word in enumerate(np.frombuffer(fw, dtype="<u4")):
        bram.write(off * 4, int(word))
    return len(fw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("--fw", default=os.path.join(HERE, "ddr_test.bin"))
    ap.add_argument("--words", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    Overlay(args.bit)
    reset = GPIO(GPIO.get_gpio_pin(RESET_EMIO), "out")
    bram = MMIO(BRAM_ARM_BASE, BRAM_BYTES)
    intc = MMIO(INTC_BASE, 0x1000)

    reset.write(1)
    fw_len = load_firmware(bram, args.fw)
    print(f"firmware: {fw_len} bytes loaded into BRAM")

    n = args.words
    sub_words = (SUB_BYTES + 2 * SUB_HALVES) // 4
    rng = np.random.default_rng(0x5EED)
    src = allocate(shape=(n,), dtype=np.uint32)
    dst = allocate(shape=(n + sub_words + GUARD_WORDS,), dtype=np.uint32)
    src[:] = rng.integers(0, 2**32, size=n, dtype=np.uint32)
    dst[:] = POISON
    src.flush()
    dst.flush()
    print(f"src @ {src.physical_address:#010x}, dst @ {dst.physical_address:#010x}, {n} words")

    bram.write(MBOX_OFFSET + MBOX_SRC_ADDR, src.physical_address)
    bram.write(MBOX_OFFSET + MBOX_DST_ADDR, dst.physical_address)
    bram.write(MBOX_OFFSET + MBOX_N_WORDS, n)

    t0 = time.time()
    reset.write(0)
    status = 0
    while time.time() - t0 < args.timeout:
        status = bram.read(MBOX_OFFSET + MBOX_STATUS)
        if status == STATUS_DONE:
            break
        time.sleep(0.001)
    reset.write(1)   # park the core again (it is halted in trap anyway)

    fails = []
    if status != STATUS_DONE:
        what = "started but never finished" if status == STATUS_RUNNING else "never started"
        fails.append(f"status = {status:#010x} after {args.timeout}s ({what})")

    dst.invalidate()
    src_u64 = src.astype(np.uint64)
    exp_sum = int(src_u64.sum()) & 0xFFFFFFFF
    exp_dst = ((src_u64 * 3 + np.arange(n, dtype=np.uint64)) & 0xFFFFFFFF).astype(np.uint32)
    exp_bytes = (np.arange(SUB_BYTES) ^ 0x5A).astype(np.uint8)
    exp_halves = (0xA000 + 3 * np.arange(SUB_HALVES)).astype(np.uint16)

    got_sum = bram.read(MBOX_OFFSET + MBOX_SRC_SUM)
    fw_errors = bram.read(MBOX_OFFSET + MBOX_ERRORS)
    cycles = bram.read(MBOX_OFFSET + MBOX_CYCLES)
    raw = np.array(dst[n:n + sub_words]).view(np.uint8)
    got_bytes = raw[:SUB_BYTES]
    got_halves = raw[SUB_BYTES:].view("<u2")

    if got_sum != exp_sum:
        fails.append(f"DDR read: src_sum {got_sum:#010x}, expected {exp_sum:#010x}")
    if fw_errors:
        fails.append(f"firmware read-back found {fw_errors} mismatches")
    bad = np.flatnonzero(dst[:n] != exp_dst)
    if bad.size:
        i = bad[0]
        fails.append(f"DDR word write: {bad.size} bad words, first dst[{i}] = "
                     f"{int(dst[i]):#010x}, expected {int(exp_dst[i]):#010x}")
    if not np.array_equal(got_bytes, exp_bytes):
        fails.append("DDR byte write (WSTRB) mismatch: " + got_bytes[:8].tobytes().hex())
    if not np.array_equal(got_halves, exp_halves):
        fails.append("DDR halfword write mismatch: " + got_halves[:4].tobytes().hex())
    if np.any(dst[n + sub_words:] != POISON):
        fails.append("guard words after dst were overwritten")

    trap_irq = intc.read(0x0) & 1   # ISR bit 0 (informational)
    src.freebuffer()
    dst.freebuffer()

    # Timed region: n reads (sum) + n reads + n writes (copy) + n reads
    # (check) = 4n word accesses; the few sub-word accesses are ignored.
    moved = 4 * n * 4
    secs = cycles / RISCV_HZ
    print(f"src_sum = {got_sum:#010x}, firmware errors = {fw_errors}, trap irq = {trap_irq}")
    if cycles:
        print(f"{cycles} cycles ({secs * 1e3:.2f} ms @ 50 MHz), "
              f"~{moved / secs / 1e6:.1f} MB/s over {moved} bytes of DDR accesses")

    if fails:
        print("FAIL")
        for f in fails:
            print("  - " + f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

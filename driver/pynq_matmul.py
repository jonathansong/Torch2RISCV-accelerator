"""PYNQ driver for the overlay: ARM -> PicoRV32 -> matmul_unit.

The ARM allocates A/B/C (and a table of job descriptors) in DDR, hands
their physical addresses to the PicoRV32 through the BRAM mailbox and
releases it from reset. Two firmwares consume the same mailbox:

  matmul_fw.bin       (firmware/matmul)       CSR path: programs the matmul
                                              CSRs per job and polls STATUS
  matmul_insn_fw.bin  (firmware/matmul_insn)  custom-instruction path:
                                              mat_trigger(desc, C) + mat_wait

    from pynq_matmul import MatmulOverlay
    mm = MatmulOverlay("picorv32.bit", "matmul_insn_fw.bin")
    C, stats = mm.matmul(A, B)      # A, B: (n, 8, 8) int8 -> C: (n, 8, 8) int32
    mm.load_firmware("matmul_fw.bin")   # switch path without reloading the overlay

Run as root inside the PYNQ venv with XRT sourced (Jupyter already is).
"""
import os
import time

import numpy as np
from pynq import GPIO, MMIO, Overlay, allocate

HERE = os.path.dirname(os.path.abspath(__file__))

# Overlay addresses (RISCV-on-PYNQ-Z1/scripts/pico_bit.tcl)
BRAM_ARM_BASE = 0x40010000
BRAM_BYTES = 0x2000
RESET_EMIO = 0                 # PS GPIO EMIO[0] -> RISC-V reset (1 = hold)
RISCV_HZ = 50e6                # subprocessorClk; matmul_unit shares it

# firmware/include/mailbox.h
MBOX = 0x1F00
MBOX_STATUS, MBOX_N_JOBS, MBOX_A_BASE, MBOX_B_BASE, MBOX_C_BASE = 0x00, 0x04, 0x08, 0x0C, 0x10
MBOX_JOBS_DONE, MBOX_ERRORS, MBOX_FIRST_ERR = 0x14, 0x18, 0x1C
MBOX_TOTAL_CYCLES, MBOX_ACCEL_CYCLES, MBOX_UNIT_ID = 0x20, 0x24, 0x28
MBOX_DESC_BASE = 0x2C
MBOX_WORDS = 0x100 // 4
STATUS_RUNNING, STATUS_DONE = 0x00000001, 0x600D600D
ERR_NO_UNIT, ERR_TIMEOUT = 0xDEAD0001, 0xDEAD0002
MATMUL_ID = 0x4D4D3038
DIM_888 = 8 | (8 << 10) | (8 << 20)     # descriptor/CSR DIM_M_N_K

MATMUL_ERR_CODES = {1: "bad DIM", 2: "bad address", 3: "read SLVERR/DECERR",
                    4: "write SLVERR/DECERR"}


def _describe(first_err):
    if first_err == ERR_NO_UNIT:
        return "matmul unit not found (ID mismatch)"
    if first_err == ERR_TIMEOUT:
        return "matmul job never finished"
    code = (first_err >> 8) & 0xF
    return f"matmul STATUS {first_err:#x} ({MATMUL_ERR_CODES.get(code, 'unknown')})"


class MatmulOverlay:
    def __init__(self, bitfile=None, firmware=None, download=True):
        bitfile = bitfile or os.path.join(HERE, "picorv32.bit")
        firmware = firmware or os.path.join(HERE, "matmul_fw.bin")
        self.overlay = Overlay(bitfile, download=download)
        self.reset = GPIO(GPIO.get_gpio_pin(RESET_EMIO), "out")
        self.bram = MMIO(BRAM_ARM_BASE, BRAM_BYTES)
        self.reset.write(1)
        self.load_firmware(firmware)

    def load_firmware(self, path):
        """Copy a firmware image into the program BRAM (RISC-V held in reset)."""
        self.reset.write(1)
        self.firmware = os.path.basename(path)
        fw = open(path, "rb").read()
        if len(fw) > MBOX:
            raise ValueError(f"{path}: {len(fw)} bytes overlaps the mailbox at {MBOX:#x}")
        fw += b"\0" * (-len(fw) % 4)
        for off in range(0, BRAM_BYTES, 4):
            self.bram.write(off, 0)
        for i, word in enumerate(np.frombuffer(fw, dtype="<u4")):
            self.bram.write(4 * i, int(word))

    def _mbox(self, off):
        return self.bram.read(MBOX + off)

    def matmul(self, a, b, timeout=5.0):
        """C[i] = A[i] @ B[i] for a batch of 8x8 int8 matrices on the accelerator."""
        a = np.ascontiguousarray(a, dtype=np.int8).reshape(-1, 8, 8)
        b = np.ascontiguousarray(b, dtype=np.int8).reshape(-1, 8, 8)
        if a.shape != b.shape:
            raise ValueError(f"A {a.shape} and B {b.shape} batch sizes differ")
        n = a.shape[0]

        abuf = allocate(shape=a.shape, dtype=np.int8)
        bbuf = allocate(shape=b.shape, dtype=np.int8)
        cbuf = allocate(shape=(n, 8, 8), dtype=np.int32)
        dbuf = allocate(shape=(max(n, 1), 4), dtype=np.uint32)   # {A, B, DIM, 0} per job
        try:
            for buf in (abuf, bbuf, cbuf, dbuf):
                # jobs are 64/256 B slices; a 256 B aligned base keeps every
                # slice inside one 4 KB page, as the unit requires
                if buf.physical_address % 256:
                    raise RuntimeError(f"buffer at {buf.physical_address:#x} is not 256 B aligned")
            abuf[:] = a
            bbuf[:] = b
            cbuf[:] = 0
            jobs = np.arange(n, dtype=np.uint32)
            dbuf[:n, 0] = abuf.physical_address + 64 * jobs
            dbuf[:n, 1] = bbuf.physical_address + 64 * jobs
            dbuf[:n, 2] = DIM_888
            dbuf[:n, 3] = 0
            for buf in (abuf, bbuf, cbuf, dbuf):
                buf.flush()

            self.reset.write(1)
            for w in range(MBOX_WORDS):
                self.bram.write(MBOX + 4 * w, 0)
            self.bram.write(MBOX + MBOX_N_JOBS, n)
            self.bram.write(MBOX + MBOX_A_BASE, abuf.physical_address)
            self.bram.write(MBOX + MBOX_B_BASE, bbuf.physical_address)
            self.bram.write(MBOX + MBOX_C_BASE, cbuf.physical_address)
            self.bram.write(MBOX + MBOX_DESC_BASE, dbuf.physical_address)

            t0 = time.perf_counter()
            self.reset.write(0)
            status = 0
            while time.perf_counter() - t0 < timeout:
                status = self._mbox(MBOX_STATUS)
                if status == STATUS_DONE:
                    break
            wall = time.perf_counter() - t0
            self.reset.write(1)

            if status != STATUS_DONE:
                where = "running" if status == STATUS_RUNNING else "not started"
                raise TimeoutError(f"firmware {where} after {timeout}s, "
                                   f"{self._mbox(MBOX_JOBS_DONE)}/{n} jobs done")
            errors = self._mbox(MBOX_ERRORS)
            if errors:
                raise RuntimeError(f"{errors} job(s) failed: {_describe(self._mbox(MBOX_FIRST_ERR))}")

            cbuf.invalidate()
            c = np.array(cbuf)
        finally:
            for buf in (abuf, bbuf, cbuf, dbuf):
                buf.freebuffer()

        total = self._mbox(MBOX_TOTAL_CYCLES)
        accel = self._mbox(MBOX_ACCEL_CYCLES)
        stats = {
            "firmware": self.firmware,
            "jobs": n,
            "unit_id": hex(self._mbox(MBOX_UNIT_ID)),
            "riscv_cycles": total,
            "accel_cycles": accel,
            "riscv_cycles_per_job": total / n if n else 0,
            "accel_cycles_per_job": accel / n if n else 0,
            "us_per_job": total / n / RISCV_HZ * 1e6 if n else 0,
            "wall_s": wall,
        }
        return c, stats


def golden(a, b):
    return np.asarray(a, np.int32) @ np.asarray(b, np.int32)


def regression(mm, batches=10, batch_size=256, seed=0):
    """Random batches on hardware vs NumPy; returns (passed, total, stats of last batch)."""
    rng = np.random.default_rng(seed)
    passed = total = 0
    stats = None
    for _ in range(batches):
        a = rng.integers(-128, 128, (batch_size, 8, 8), dtype=np.int8)
        b = rng.integers(-128, 128, (batch_size, 8, 8), dtype=np.int8)
        c, stats = mm.matmul(a, b)
        ok = np.all(c == golden(a, b), axis=(1, 2))
        passed += int(ok.sum())
        total += batch_size
    return passed, total, stats


if __name__ == "__main__":
    mm = MatmulOverlay()
    passed, total, stats = regression(mm)
    print(f"{passed}/{total} matrices match NumPy")
    print(stats)
    raise SystemExit(0 if passed == total else 1)

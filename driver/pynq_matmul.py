"""PYNQ driver for the overlay: ARM -> PicoRV32 -> matmul_unit.

The ARM allocates A/B/C (and a table of job descriptors) in DDR, hands
their physical addresses to the PicoRV32 through the BRAM mailbox and
releases it from reset. Two firmwares consume the same mailbox:

  matmul_fw.bin       (firmware/matmul)       CSR path: programs the matmul
                                              CSRs per job and polls STATUS
  matmul_insn_fw.bin  (firmware/matmul_insn)  custom-instruction path:
                                              mat_trigger(desc, C) + mat_wait
  gemm_fw.bin         (firmware/gemm)         double-buffered accelerator (M1+):
                                              whole C = A @ B (+ bias) with the
                                              funct7 = 1 ISA -> MatmulOverlay.gemm();
                                              M3: int8 output through the vector
                                              engine (gemm(..., quant=Requant(...)))
  vector_fw.bin       (firmware/vector)       M3 vector engine alone, funct7 = 2
                                              -> MatmulOverlay.vector()
  bwtest_fw.bin       (firmware/bwtest)       DMA bandwidth -> MatmulOverlay.bandwidth()

    from pynq_matmul import MatmulOverlay
    mm = MatmulOverlay("picorv32.bit", "matmul_insn_fw.bin")
    C, stats = mm.matmul(A, B)      # A, B: (n, 8, 8) int8 -> C: (n, 8, 8) int32
    mm.load_firmware("matmul_fw.bin")   # switch path without reloading the overlay

Run as root inside the PYNQ venv with XRT sourced (Jupyter already is).
"""
import os
import time
from dataclasses import dataclass

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
MBOX_GEMM_M, MBOX_GEMM_N, MBOX_GEMM_K, MBOX_BIAS_BASE, MBOX_EXT_STATUS = 0x30, 0x34, 0x38, 0x3C, 0x40
MBOX_BW_CYCLES = 0x44
MBOX_V_LEN, MBOX_V_OP, MBOX_V_TYPES, MBOX_V_PERIOD = 0x60, 0x64, 0x68, 0x6C
MBOX_V_SCALE, MBOX_V_SHIFT, MBOX_V_ZP, MBOX_V_LO, MBOX_V_HI = 0x70, 0x74, 0x78, 0x7C, 0x80
MBOX_GEMM_Q = 0x84
BW_TESTS = [  # name, bytes moved (bwtest_fw.c)
    ("LD  DDR -> SPAD_A, 64 KB contiguous", 65536),
    ("ST  SPAD_A -> DDR, 64 KB", 65536),
    ("LD  DDR -> ACC, 64 KB contiguous", 65536),
    ("ST  ACC -> DDR, 64 KB", 65536),
    ("LD  8-byte rows, pitch 16 (1-beat bursts)", 65536),
    ("LD SPAD_B + ST SPAD_A concurrently", 131072),
]
SA_D = 8                                 # array size of the overlay build
SPAD_BYTES = 128 * 1024                  # per SPAD
MBOX_WORDS = 0x100 // 4
STATUS_RUNNING, STATUS_DONE = 0x00000001, 0x600D600D
ERR_NO_UNIT, ERR_TIMEOUT = 0xDEAD0001, 0xDEAD0002
MATMUL_ID = 0x4D4D3038
DIM_888 = 8 | (8 << 10) | (8 << 20)     # descriptor/CSR DIM_M_N_K

# vector engine (firmware/include/sysarray_intrinsics.h)
VOPS = {"add": 0, "sub": 1, "mul": 2, "max": 3, "min": 4, "copy": 5}
V_RELU, V_REQUANT = 0x10, 0x20
VTYPES = {np.dtype(np.int8): 0, np.dtype(np.int16): 1, np.dtype(np.int32): 2}
I32_MIN, I32_MAX = -2**31, 2**31 - 1


@dataclass
class Requant:
    """y = clip(((x * scale + 2**(shift-1)) >> shift) + zp, lo, hi), then saturated
    to the output type; scale is a signed 16-bit multiplier, 0 <= shift <= 31."""
    scale: int = 1
    shift: int = 0
    zp: int = 0
    lo: int = I32_MIN
    hi: int = I32_MAX


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

    def _use(self, name):
        """Load firmware `name` (next to this file) unless it is already loaded."""
        if self.firmware != name:
            self.load_firmware(os.path.join(HERE, name))

    def _run(self, params, timeout):
        """Clear the mailbox, write `params` {offset: value}, run the firmware to
        completion; returns the wall time. Raises on timeout or accelerator error."""
        self.reset.write(1)
        for w in range(MBOX_WORDS):
            self.bram.write(MBOX + 4 * w, 0)
        for off, val in params.items():
            self.bram.write(MBOX + off, int(val) & 0xFFFFFFFF)
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
            raise TimeoutError(f"{self.firmware} {'running' if status == STATUS_RUNNING else 'not started'} "
                               f"after {timeout}s")
        xst = self._mbox(MBOX_EXT_STATUS)
        if xst & 2:
            raise RuntimeError(f"accelerator error: code {(xst >> 8) & 0xF}, engine {(xst >> 12) & 0xF} "
                               f"(ext status {xst:#x})")
        return wall

    @staticmethod
    def _requant_params(op, rq):
        rq = rq or Requant()
        if not (-2**15 <= rq.scale < 2**15 and 0 <= rq.shift < 32 and I32_MIN <= rq.zp <= I32_MAX
                and I32_MIN <= rq.lo <= rq.hi <= I32_MAX):
            raise ValueError(f"{rq}: scale is int16, shift 0..31, zp / lo <= hi int32")
        return {MBOX_V_OP: op, MBOX_V_SCALE: rq.scale, MBOX_V_SHIFT: rq.shift, MBOX_V_ZP: rq.zp,
                MBOX_V_LO: rq.lo, MBOX_V_HI: rq.hi}

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


    def gemm(self, a, b, bias=None, quant=None, relu=False, timeout=5.0):
        """C = A @ B (+ bias) with gemm_fw.bin: A (M, K) int8, B (K, N) int8.

        quant None: bias (M, N) int32 or None -> C (M, N) int32.
        quant Requant(...) (M3): C (M, N) int8 = requant(relu(A @ B + bias)) computed
        by the vector engine on the chip; bias is then an int32 vector (N,) or None.
        M, N, K multiples of 8; B must fit in SPAD_B (K * N <= 128 KB), K <= 8192
        (int8 output: K <= 4096, N <= 3640)."""
        self._use("gemm_fw.bin")
        a = np.ascontiguousarray(a, dtype=np.int8)
        b = np.ascontiguousarray(b, dtype=np.int8)
        (m, k), (k2, n) = a.shape, b.shape
        if k != k2 or m % SA_D or n % SA_D or k % SA_D:
            raise ValueError(f"shapes {a.shape} x {b.shape}: need matching K and multiples of {SA_D}")
        if k * n > SPAD_BYTES or k > SPAD_BYTES // 16:
            raise ValueError(f"B ({k} x {n}) does not fit the resident-B schedule")
        if quant is not None and (k > SPAD_BYTES // 32 or n + n // SA_D > SPAD_BYTES // 32):
            raise ValueError(f"int8 output: K {k} or N {n} too large for the epilogue layout")
        bshape = ((n,) if quant is not None else (m, n))
        bufs = [allocate(shape=a.shape, dtype=np.int8), allocate(shape=b.shape, dtype=np.int8),
                allocate(shape=(m, n), dtype=np.int8 if quant is not None else np.int32)]
        if bias is not None:
            bufs.append(allocate(shape=bshape, dtype=np.int32))
        abuf, bbuf, cbuf = bufs[:3]
        try:
            abuf[:] = a
            bbuf[:] = b
            cbuf[:] = 0
            if bias is not None:
                bufs[3][:] = np.asarray(bias, dtype=np.int32).reshape(bshape)
            for buf in bufs:
                buf.flush()
            params = {MBOX_A_BASE: abuf.physical_address, MBOX_B_BASE: bbuf.physical_address,
                      MBOX_C_BASE: cbuf.physical_address, MBOX_GEMM_M: m, MBOX_GEMM_N: n, MBOX_GEMM_K: k,
                      MBOX_BIAS_BASE: bufs[3].physical_address if bias is not None else 0}
            if quant is not None:
                params.update(self._requant_params(V_REQUANT | (V_RELU if relu else 0), quant))
                params[MBOX_GEMM_Q] = 1
            wall = self._run(params, timeout)
            cbuf.invalidate()
            c = np.array(cbuf)
        finally:
            for buf in bufs:
                buf.freebuffer()
        cycles = self._mbox(MBOX_TOTAL_CYCLES)
        return c, {"firmware": self.firmware, "shape": (m, n, k), "tiles": self._mbox(MBOX_JOBS_DONE),
                   "riscv_cycles": cycles, "mac_per_cycle": m * n * k / cycles if cycles else 0,
                   "us": cycles / RISCV_HZ * 1e6, "wall_s": wall}

    def vector(self, op, x, y=None, out_dtype=None, relu=False, requant=None, timeout=5.0):
        """One vector-engine operation with vector_fw.bin (M3):
            out = saturate(clip(requant(relu(op(x, y)))))
        op: "add" "sub" "mul" "max" "min" "copy" (y unused); x: 1-D int8 / int16 / int32
        (mul: int8 / int16 only), length a multiple of 8; y: same dtype, the same
        length or 8 * period elements, period <= 512 (repeated; 8 = broadcast one group);
        out_dtype int8 / int16 / int32 (default x's); requant: Requant or None."""
        self._use("vector_fw.bin")
        x = np.ascontiguousarray(x).ravel()
        it = VTYPES[x.dtype]
        out_dtype = np.dtype(out_dtype or x.dtype)
        ot = VTYPES[out_dtype]
        code = VOPS[op]
        n = x.size
        if n % SA_D or n == 0:
            raise ValueError(f"length {n}: need a positive multiple of {SA_D}")
        if op == "mul" and it == 2:
            raise ValueError("mul takes int8 / int16 inputs")
        if op == "copy":
            y, period = x[:SA_D], 0
        else:
            y = np.ascontiguousarray(y, dtype=x.dtype).ravel()
            if y.size == n:
                period = 0
            elif y.size % SA_D == 0 and 0 < y.size // SA_D <= 512:
                period = y.size // SA_D
            else:
                raise ValueError(f"y: {y.size} elements, need {n} or 8 * period (period <= 512)")
        bufs = [allocate(shape=(n,), dtype=x.dtype), allocate(shape=(y.size,), dtype=x.dtype),
                allocate(shape=(n,), dtype=out_dtype)]
        xbuf, ybuf, obuf = bufs
        try:
            xbuf[:] = x
            ybuf[:] = y
            obuf[:] = 0
            for buf in bufs:
                buf.flush()
            params = {MBOX_A_BASE: xbuf.physical_address, MBOX_B_BASE: ybuf.physical_address,
                      MBOX_C_BASE: obuf.physical_address, MBOX_V_LEN: n, MBOX_V_TYPES: it | (ot << 2),
                      MBOX_V_PERIOD: period}
            params.update(self._requant_params(
                code | (V_RELU if relu else 0) | (V_REQUANT if requant is not None else 0), requant))
            wall = self._run(params, timeout)
            obuf.invalidate()
            out = np.array(obuf)
        finally:
            for buf in bufs:
                buf.freebuffer()
        cycles = self._mbox(MBOX_TOTAL_CYCLES)
        return out, {"firmware": self.firmware, "len": n, "chunks": self._mbox(MBOX_JOBS_DONE),
                     "riscv_cycles": cycles, "elem_per_cycle": n / cycles if cycles else 0,
                     "us": cycles / RISCV_HZ * 1e6, "wall_s": wall}

    def bandwidth(self, timeout=5.0):
        """DMA bandwidth self-test (bwtest_fw.bin). Returns [(name, bytes, cycles, B/cycle)]
        and checks the three stored copies against the source."""
        src = allocate(shape=(65536,), dtype=np.uint8)
        dst = allocate(shape=(3 * 65536,), dtype=np.uint8)
        try:
            src[:] = np.random.default_rng(7).integers(0, 256, 65536, dtype=np.uint8)
            dst[:] = 0xA5
            src.flush()
            dst.flush()
            self.reset.write(1)
            for w in range(MBOX_WORDS):
                self.bram.write(MBOX + 4 * w, 0)
            self.bram.write(MBOX + MBOX_A_BASE, src.physical_address)
            self.bram.write(MBOX + MBOX_C_BASE, dst.physical_address)
            t0 = time.perf_counter()
            self.reset.write(0)
            while time.perf_counter() - t0 < timeout and self._mbox(MBOX_STATUS) != STATUS_DONE:
                pass
            self.reset.write(1)
            if self._mbox(MBOX_STATUS) != STATUS_DONE:
                raise TimeoutError("bandwidth firmware did not finish")
            if self._mbox(MBOX_ERRORS):
                raise RuntimeError(f"DMA error, ext status {self._mbox(MBOX_FIRST_ERR):#x}")
            dst.invalidate()
            copies_ok = all(np.array_equal(dst[65536 * i:65536 * (i + 1)], src) for i in range(3))
        finally:
            src.freebuffer()
            dst.freebuffer()
        res = []
        for i, (name, nbytes) in enumerate(BW_TESTS):
            cyc = self._mbox(MBOX_BW_CYCLES + 4 * i)
            res.append((name, nbytes, cyc, nbytes / cyc if cyc else 0.0))
        return res, copies_ok


def golden(a, b):
    return np.asarray(a, np.int32) @ np.asarray(b, np.int32)


def vector_golden(op, x, y=None, out_dtype=None, relu=False, requant=None):
    """NumPy model of the vector engine (rtl/sysarray/sa_ve.v), same arguments as vector()."""
    x = np.asarray(x).ravel()
    out_dtype = np.dtype(out_dtype or x.dtype)
    a = x.astype(np.int64)
    b = np.resize(np.asarray(y if y is not None else x, dtype=np.int64).ravel(), a.size)
    r = {"add": lambda: np.clip(a + b, I32_MIN, I32_MAX), "sub": lambda: np.clip(a - b, I32_MIN, I32_MAX),
         "mul": lambda: a * b, "max": lambda: np.maximum(a, b), "min": lambda: np.minimum(a, b),
         "copy": lambda: a}[op]()
    if relu:
        r = np.maximum(r, 0)
    lo, hi = I32_MIN, I32_MAX
    if requant is not None:
        rnd = (1 << (requant.shift - 1)) if requant.shift else 0
        r = ((r * requant.scale + rnd) >> requant.shift) + requant.zp
        lo, hi = requant.lo, requant.hi
    info = np.iinfo(out_dtype)
    return np.clip(np.clip(r, lo, hi), info.min, info.max).astype(out_dtype)


def qgemm_golden(a, b, bias, quant, relu=False):
    """int8 C of gemm(a, b, bias, quant, relu): requant(relu(A @ B + bias))."""
    c = golden(a, b).astype(np.int64)
    if bias is not None:
        c = np.clip(c + np.asarray(bias, np.int64).reshape(1, -1), I32_MIN, I32_MAX)
    return vector_golden("copy", c.ravel(), out_dtype=np.int8, relu=relu, requant=quant).reshape(c.shape)


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

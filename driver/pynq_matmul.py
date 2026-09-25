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
  desc_run_fw.bin     (firmware/desc_run)     runs a descriptor list the ARM built
                                              (DescList) with one mat_submit ->
                                              MatmulOverlay.run_list / gemm_list / vector_list
  rt_fw.bin           (firmware/rt)           L1 resident runtime: command ring in DDR,
                                              completion records, notify interrupt -> Device

    from pynq_matmul import MatmulOverlay
    mm = MatmulOverlay("picorv32.bit", "matmul_insn_fw.bin")
    C, stats = mm.matmul(A, B)      # A, B: (n, 8, 8) int8 -> C: (n, 8, 8) int32
    mm.load_firmware("matmul_fw.bin")   # switch path without reloading the overlay

Run as root inside the PYNQ venv with XRT sourced (Jupyter already is).
"""
import os
import re
import time
from dataclasses import dataclass

import numpy as np
try:                                     # off the board only the pure parts (goldens,
    from pynq import GPIO, MMIO, Overlay, allocate   # DescList, list builders) are usable
except ImportError:
    GPIO = MMIO = Overlay = allocate = None

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
MBOX_GEMM_FLAGS = 0x88                   # gemm_fw schedule switches (0 = tuned)
GEMM_NO_PREFETCH, GEMM_NO_BSPLIT, GEMM_FORCE_BSPLIT = 1, 2, 4   # default: B split if B >= 16 KB
MBOX_PERF_COUNT = 0x8C                   # counters copied to the perf area (0 = none)
MBOX_DL_ADDR, MBOX_DL_COUNT, MBOX_DL_BASE0 = 0x90, 0x94, 0x98   # desc_run_fw inputs
MBOX_DL_STATUS, MBOX_DL_EXEC = 0xA8, 0xAC                       #             outputs
ERR_NO_DESC = 0xDEAD0003
# L1 resident runtime (firmware/rt, docs/llm_inference_plan.md §5.1)
MBOX_RING_BASE, MBOX_RING_SIZE, MBOX_RING_TAIL, MBOX_RING_HEAD = 0xB0, 0xB4, 0xB8, 0xBC
MBOX_CPL_BASE, MBOX_FW_STATE, MBOX_FW_VERSION, MBOX_HEARTBEAT = 0xC0, 0xC4, 0xC8, 0xCC
RT_READY = 0x52554E00
RT_RUN_LIST, RT_NOP, RT_RESET, RT_EXIT = 0x01, 0x02, 0x03, 0x04
RT_F_IRQ, RT_F_PERF = 1 << 8, 1 << 9

# Performance counters (rtl/sysarray/sa_perf.v, docs/perf_counters_and_desc_dma_plan.md):
# the firmware copies them to the 256 bytes below the mailbox after its
# measurement window. Order = PC_* in rtl/sysarray/sa_defs.vh.
PERF_AREA = 0x1E00                       # program BRAM offset; firmware must end below it
PERF_NAMES = ["CYCLES", "CMD_LD", "CMD_ST", "CMD_EX", "CMD_VE", "PCPI_QFULL", "PCPI_FENCE",
              "HAZ_LD", "HAZ_ST", "HAZ_EX", "HAZ_VE", "DISP_FULL", "STARVE", "ALL_IDLE",
              "EX_STEP", "EX_USEFUL", "EX_SWAPWAIT", "EX_TILES", "LD_BUSY", "LD_BEATS",
              "LD_ARSTALL", "ST_BUSY", "ST_BEATS", "ST_WSTALL", "VE_ACTIVE", "VE_RDBLOCK",
              "VE_CREDIT", "VE_GROUPS"]  # 28..31 reserved
BW_TESTS = [  # name, bytes moved (bwtest_fw.c)
    ("LD  DDR -> SPAD_A, 64 KB contiguous", 65536),
    ("ST  SPAD_A -> DDR, 64 KB", 65536),
    ("LD  DDR -> ACC, 64 KB contiguous", 65536),
    ("ST  ACC -> DDR, 64 KB", 65536),
    ("LD  8-byte rows, pitch 16 (1-beat bursts)", 65536),
    ("LD SPAD_B + ST SPAD_A concurrently", 131072),
]
SA_D = 8                                 # array size if the .hwh does not say (M1-M3 builds)
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
# L2 fp32 vector engine (docs/llm_inference_plan.md §6)
VT_F32 = 3
VOP_TRANSPOSE = 6
VFUNC = {"none": 0, "exp": 1, "recip": 2, "rsqrt": 3, "abs": 4}
VIDX = {"lin": 0, "mod": 1, "div": 2, "imm": 3}
VRED = {"none": 0, "sum": 1, "max": 2}


def f32bits(x):
    """float -> fp32 bit pattern (int)"""
    return int(np.asarray(x, np.float32).view(np.uint32))
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


# ------------------------------------------------------------------ descriptors
MEM_SPAD_A, MEM_SPAD_B, MEM_ACC = 1, 2, 3
LD_LINEAR, LD_INTERLEAVE = 0, 1
_M32, _M64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF


def laddr(mem, word):
    """Local address of a word in SPAD_A / SPAD_B / ACC."""
    return (mem << 28) | word


class DescList:
    """A list of 64-byte descriptors (docs/double_buffer_design.md §8.6):
    8 little-endian 64-bit words each, w0 = {tag = index, flags, opcode}.
    Every command carries its whole configuration (no mat_cfg state).
    DDR addresses are physical; base=n makes one relative to BASE n (0..3;
    0..15 with the L1 extensions), fence_before waits for all engines first
    (use it, or fence(), between a store and a later load of the same DDR
    bytes: DDR is not tracked).

    L1 command extensions (CAPS bit 24, docs/llm_inference_plan.md §5.3):
    dyn=[(field, param, add)] takes up to two fields from PARAM registers
    (replace, or add when add=True; field names in DYN_FIELDS), and setreg /
    loop_end / call / ret / ldparam / relative jump."""
    LD, ST, EX, VE, FENCE, JUMP, END = 0x01, 0x02, 0x03, 0x04, 0x10, 0x11, 0x12
    LOOP_END, SETREG, CALL, RET, LDPARAM = 0x13, 0x14, 0x15, 0x16, 0x17
    RELOC, FENCE_BEFORE = 1 << 8, 1 << 11
    _DMA = {"ddr": 1, "laddr": 2, "rows": 3, "row_bytes": 4, "pitch": 5}
    DYN_FIELDS = {
        LD: _DMA, ST: _DMA,
        EX: {"a": 1, "b": 2, "c": 3, "kt": 4, "repeat": 5, "bstep": 6, "cstep": 7},
        VE: {"src1": 1, "src2": 2, "dst": 3, "len": 4, "valid": 5, "rowlen": 6, "p1": 7, "period": 8,
             "A": 9, "B": 10, "imm": 11},
        JUMP: {"target": 1}, CALL: {"target": 1}, LOOP_END: {"count": 1},
        SETREG: {"v0": 1, "v1": 2, "v2": 3}, LDPARAM: {"addr": 1},
    }
    REG_BASE, REG_PARAM = 0, 16            # SETREG / mat_cfg register numbers: BASE n = n, PARAM n = 16 + n

    def __init__(self):
        self.rows = []

    def _put(self, op, flags, *words, dyn=()):
        if len(dyn) > 2:
            raise ValueError("at most two dynamic fields per descriptor")
        for i, d in enumerate(dyn):
            name, param, add = (tuple(d) + (False,))[:3]
            flags |= (self.DYN_FIELDS[op][name] | (param & 7) << 4 | (0x80 if add else 0)) << (16 + 8 * i)
        w = [(len(self.rows) << 32) | flags | op] + [x & _M64 for x in words]
        self.rows.append(w + [0] * (8 - len(w)))
        return self

    @classmethod
    def _flags(cls, base, fence_before):
        f = cls.FENCE_BEFORE if fence_before else 0
        if base is not None:
            f |= cls.RELOC | (base & 3) << 9 | (base >> 2 & 3) << 13
        return f

    def ld(self, ddr, la, rows, row_bytes, pitch, mode=LD_LINEAR, base=None, fence_before=False, dyn=()):
        return self._put(self.LD, self._flags(base, fence_before), ddr & _M32,
                         la | rows << 32 | row_bytes << 48, pitch | mode << 32, dyn=dyn)

    def st(self, ddr, la, rows, row_bytes, pitch, base=None, fence_before=False, dyn=()):
        return self._put(self.ST, self._flags(base, fence_before), ddr & _M32,
                         la | rows << 32 | row_bytes << 48, pitch, dyn=dyn)

    def ex(self, a, b, c, kt, acc=False, repeat=1, bstep=0, cstep=0, crow=1, fence_before=False, dyn=()):
        return self._put(self.EX, self._flags(None, fence_before),
                         a | b << 16 | c << 32 | kt << 48 | int(acc) << 60,
                         repeat | bstep << 16 | cstep << 32 | crow << 48, dyn=dyn)

    def ve(self, src1, src2, dst, length, op, types, period=0, scale=1, shift=0, zp=0,
           lo=-2**31, hi=2**31 - 1, fence_before=False, dyn=(), fp=False, func="none", m1="lin", m2="lin",
           reduce="none", swapneg=False, imm=0.0, A=1.0, B=0.0, rowlen=0, valid=0, p1=0, t2=None):
        """Integer VE (fp=False: as M3), or the L2 fp32 pipeline (fp=True):
        src1 (index mode m1 with period p1), [SWAPNEG], op with src2 (m2
        with `period`, or the immediate imm), y * A + B, func, RELU (op
        flag), output type, or a row reduction (rowlen groups per row, the
        first `valid` elements of each row count; 0 = all). t2: type of src2
        when it differs from src1's (0 int8, 2 int32, 3 fp32; fp only)."""
        if t2 is not None:
            types |= {0: 1, 2: 2, 3: 3}[t2] << 4
        w3 = op | types << 8 | period << 16 | (scale & 0xFFFF) << 32 | shift << 48
        w5, w6, w7 = hi & _M32, 0, 0
        if fp:
            flags = 1 | VFUNC[func] << 1 | VIDX[m1] << 4 | VIDX[m2] << 6 | VRED[reduce] << 8 | int(swapneg) << 10
            w3 |= flags << 53
            w5 |= f32bits(imm) << 32
            w6 = f32bits(A) | f32bits(B) << 32
            w7 = rowlen | valid << 16 | p1 << 32
        return self._put(self.VE, self._flags(None, fence_before), src1 | src2 << 32, dst | length << 32,
                         w3, (zp & _M32) | (lo & _M32) << 32, w5, w6, w7, dyn=dyn)

    def transpose(self, src, dst, length, types, stride, fence_before=False, dyn=()):
        """D x D block transpose of an R x `stride`-word matrix (length elements)."""
        return self._put(self.VE, self._flags(None, fence_before), src, dst | length << 32,
                         VOP_TRANSPOSE | types << 8, 0, 0, 0, stride << 48, dyn=dyn)

    def fence(self, mask=0):
        return self._put(self.FENCE, 0, mask)

    def jump(self, addr, rel=False, dyn=()):
        """rel: addr is a signed offset in descriptors from this one."""
        return self._put(self.JUMP, 0, addr & _M32, int(rel), dyn=dyn)

    def end(self, status=0):
        return self._put(self.END, 0, status & _M32)

    # ---- L1 command extensions
    def setreg(self, *regs, dyn=()):
        """regs: up to three (register, value[, add]); register = n for BASE n,
        REG_PARAM + n for PARAM n; add=True adds value to the register."""
        if not 1 <= len(regs) <= 3:
            raise ValueError("SETREG takes 1..3 registers")
        w1, vals, addbits = 0, [0, 0, 0], 0
        for i, r in enumerate(regs):
            reg, val, add = (tuple(r) + (False,))[:3]
            w1 |= (0x40 | reg) << (8 * i)
            vals[i] = val & _M32
            addbits |= int(add) << i
        return self._put(self.SETREG, 0, w1, *vals, addbits, dyn=dyn)

    def loop_end(self, offset, count, k1=0, s1=0, k2=0, s2=0, dyn=()):
        """Close a loop: jump back `offset` descriptors (negative) while fewer
        than `count` iterations ran; each iteration adds s1 to PARAM k1 and s2
        to PARAM k2."""
        return self._put(self.LOOP_END, 0, offset & _M32, count | (k1 & 7) << 16 | (k2 & 7) << 19,
                         s1 & _M32, s2 & _M32, dyn=dyn)

    def call(self, addr, rel=False, dyn=()):
        return self._put(self.CALL, 0, addr & _M32, int(rel), dyn=dyn)

    def ret(self):
        return self._put(self.RET, 0)

    def ldparam(self, addr, param, mul=1, add=0, base=None, fence_before=False, dyn=()):
        """PARAM[param] = mem32[addr] * mul + add (addr 4-byte aligned)."""
        return self._put(self.LDPARAM, self._flags(base, fence_before), addr & _M32, param & 7, mul & 0xFFFF,
                         add & _M32, dyn=dyn)

    def __len__(self):
        return len(self.rows)

    def array(self):
        return np.array(self.rows, dtype=np.uint64)


def _banks(d):
    """(SPAD bank words, ACC bank words) of array size d."""
    return SPAD_BYTES // d // 2, 262144 // (4 * d) // 2


def build_gemm_list(d, m, n, k, a_addr, b_addr, c_addr, bias_addr=None, quant=None, relu=False,
                    prefetch=True, bsplit="auto"):
    """The gemm_fw schedule as a descriptor list (same commands, same order):
    resident B (split over the two SPAD_B banks when B >= 16 KB), A strips
    alternating banks with the next strip prefetched before the current
    store, int32 C (+ bias rows) or int8 C through the VE epilogue (quant,
    bias = vector of n)."""
    sbank, cbank = _banks(d)
    nt, kt = n // d, k // d
    cbytes = n if quant is not None else 4 * n
    vbias, qout = cbank - nt, sbank // 2
    nt0 = (nt + 1) // 2
    nt1 = nt - nt0
    split = nt1 != 0 and nt0 * k <= sbank and (k * n >= 16384 if bsplit == "auto" else bool(bsplit))
    bias_strip = bias_addr is not None and quant is None
    dl = DescList()

    def load_strip(i, ab, cb):
        dl.ld(a_addr + i * k, laddr(MEM_SPAD_A, ab), d, k, k, LD_INTERLEAVE)
        if bias_strip:
            dl.ld(bias_addr + i * 4 * n, laddr(MEM_ACC, cb), d, 4 * n, 4 * n)

    if split:
        dl.ld(b_addr, laddr(MEM_SPAD_B, 0), k, nt0 * d, n, LD_INTERLEAVE)
    else:
        dl.ld(b_addr, laddr(MEM_SPAD_B, 0), k, n, n, LD_INTERLEAVE)
    load_strip(0, 0, 0)
    if split:
        dl.ld(b_addr + nt0 * d, laddr(MEM_SPAD_B, sbank), k, nt1 * d, n, LD_INTERLEAVE)
    if quant is not None and bias_addr is not None:
        for cb in (0, cbank):
            dl.ld(bias_addr, laddr(MEM_ACC, cb + vbias), 1, 4 * n, 4 * n)
    vop = (V_REQUANT | (V_RELU if relu else 0) |
           (VOPS["add"] if bias_addr is not None else VOPS["copy"])) if quant is not None else 0
    for s_, i in enumerate(range(0, m, d)):
        odd = s_ & 1
        ab, cb = (sbank if odd else 0), (cbank if odd else 0)
        if i != 0 and not prefetch:
            load_strip(i, ab, cb)
        if split:
            dl.ex(ab, 0, cb, kt, bias_strip, nt0, k, 1, nt)
            dl.ex(ab, sbank, cb + nt0, kt, bias_strip, nt1, k, 1, nt)
        else:
            dl.ex(ab, 0, cb, kt, bias_strip, nt, k, 1, nt)
        if quant is not None:
            dl.ve(laddr(MEM_ACC, cb), laddr(MEM_ACC, cb + vbias), laddr(MEM_SPAD_A, ab + qout), d * n,
                  vop, 2 | 0 << 2, nt, quant.scale, quant.shift, quant.zp, quant.lo, quant.hi)
        if prefetch and i + d < m:
            load_strip(i + d, sbank if not odd else 0, cbank if not odd else 0)
        if quant is not None:
            dl.st(c_addr + i * n, laddr(MEM_SPAD_A, ab + qout), d, cbytes, cbytes)
        else:
            dl.st(c_addr + i * 4 * n, laddr(MEM_ACC, cb), d, cbytes, cbytes)
    return dl.end(0x600D)


def build_vector_list(d, op, it, ot, n, ny, x_addr, y_addr, o_addr, relu=False, requant=None):
    """The vector_fw schedule as a descriptor list: chunks of up to 512
    groups alternating banks, 256-byte DMA rows plus a tail, a periodic src2
    resident in both banks. it / ot: 0 int8, 1 int16, 2 int32; ny = elements
    of y (n, or d * period)."""
    esize = {0: 1, 1: 2, 2: 4}
    sbank, cbank = _banks(d)
    code = VOPS[op]
    unary = op == "copy"
    period = 0 if unary or ny == n else ny // d
    m1 = MEM_ACC if it == 2 else MEM_SPAD_A
    m2 = MEM_ACC if it == 2 else MEM_SPAD_B
    w2 = cbank // 4 if it == 2 else 0
    md = MEM_ACC if ot == 2 else MEM_SPAD_B
    wd = cbank // 2 if ot == 2 else sbank // 2
    gin, gout = d * esize[it], d * esize[ot]
    bank = lambda mem, b: b * (cbank if mem == MEM_ACC else sbank)
    wbytes = lambda mem: 4 * d if mem == MEM_ACC else d
    cg = (512 // period) * period if period > 1 else 512
    rq = requant or Requant()
    vop = code | (V_RELU if relu else 0) | (V_REQUANT if requant is not None else 0)
    dl = DescList()

    def xfer(store, ddr, mem, word, nbytes):
        rows, tail = divmod(nbytes, 256)
        f = dl.st if store else dl.ld
        if rows:
            f(ddr, laddr(mem, word), rows, 256, 256)
        if tail:
            f(ddr + rows * 256, laddr(mem, word + rows * 256 // wbytes(mem)), 1, tail, tail)

    if not unary and period:
        for b in (0, 1):
            xfer(False, y_addr, m2, bank(m2, b) + w2, period * gin)
    groups = n // d
    for chunk, g in enumerate(range(0, groups, cg)):
        cnt, b = min(cg, groups - g), chunk & 1
        xfer(False, x_addr + g * gin, m1, bank(m1, b), cnt * gin)
        if not unary and not period:
            xfer(False, y_addr + g * gin, m2, bank(m2, b) + w2, cnt * gin)
        dl.ve(laddr(m1, bank(m1, b)), laddr(m2, bank(m2, b) + w2), laddr(md, bank(md, b) + wd), cnt * d,
              vop, it | ot << 2, period, rq.scale, rq.shift, rq.zp, rq.lo, rq.hi)
        xfer(True, o_addr + g * gout, md, bank(md, b) + wd, cnt * gout)
    return dl.end(0x5EC)


def overlay_params(bitfile):
    """D and NPORTS of the sa_unit in the overlay, from the .hwh next to the .bit
    (module-reference parameters); (SA_D, 1) if not found."""
    try:
        hwh = open(os.path.splitext(bitfile)[0] + ".hwh").read()
    except OSError:
        return SA_D, 1
    i = hwh.find('MODTYPE="sa_unit"')
    if i < 0:
        return SA_D, 1
    params = dict(re.findall(r'<PARAMETER NAME="(\w+)" VALUE="([^"]*)"', hwh[i:i + 20000]))
    return int(params.get("D", SA_D)), int(params.get("NPORTS", 1))


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
        self.d, self.nports = overlay_params(bitfile)     # array size, DMA ports
        self.reset = GPIO(GPIO.get_gpio_pin(RESET_EMIO), "out")
        self.bram = MMIO(BRAM_ARM_BASE, BRAM_BYTES)
        self.reset.write(1)
        self.load_firmware(firmware)

    def load_firmware(self, path):
        """Copy a firmware image into the program BRAM (RISC-V held in reset)."""
        self.reset.write(1)
        self.firmware = os.path.basename(path)
        fw = open(path, "rb").read()
        if len(fw) > PERF_AREA:
            raise ValueError(f"{path}: {len(fw)} bytes overlaps the perf area / mailbox at {PERF_AREA:#x}")
        fw += b"\0" * (-len(fw) % 4)
        for off in range(0, BRAM_BYTES, 4):
            self.bram.write(off, 0)
        for i, word in enumerate(np.frombuffer(fw, dtype="<u4")):
            self.bram.write(4 * i, int(word))

    def _mbox(self, off):
        return self.bram.read(MBOX + off)

    def _perf(self):
        """Counters of the last run ({name: value}), or None when the overlay has
        none (CAPS bit 20 clear) or the firmware did not copy them."""
        n = min(self._mbox(MBOX_PERF_COUNT), len(PERF_NAMES))
        if n == 0:
            return None
        return {PERF_NAMES[i]: self.bram.read(PERF_AREA + 4 * i) for i in range(n)}

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


    def gemm(self, a, b, bias=None, quant=None, relu=False, flags=0, timeout=5.0):
        """C = A @ B (+ bias) with gemm_fw.bin: A (M, K) int8, B (K, N) int8.

        quant None: bias (M, N) int32 or None -> C (M, N) int32.
        quant Requant(...) (M3): C (M, N) int8 = requant(relu(A @ B + bias)) computed
        by the vector engine on the chip; bias is then an int32 vector (N,) or None.
        flags: schedule switches for measurements (GEMM_NO_PREFETCH | GEMM_NO_BSPLIT |
        GEMM_FORCE_BSPLIT; 3 = the M4 schedule, 0 = tuned default). M, N, K multiples of D (8 or 16, self.d); B must fit in SPAD_B (K * N <= 128 KB),
        K <= 65536 / D (int8 output: K <= 32768 / D, N + N / D <= 32768 / D)."""
        self._use("gemm_fw.bin")
        a = np.ascontiguousarray(a, dtype=np.int8)
        b = np.ascontiguousarray(b, dtype=np.int8)
        (m, k), (k2, n) = a.shape, b.shape
        d = self.d
        bank = SPAD_BYTES // d // 2                    # SPAD bank words = A strip limit on K
        if k != k2 or m % d or n % d or k % d:
            raise ValueError(f"shapes {a.shape} x {b.shape}: need matching K and multiples of {d}")
        if k * n > SPAD_BYTES or k > bank:
            raise ValueError(f"B ({k} x {n}) does not fit the resident-B schedule")
        # int8 output: A strip and int8 strip share a SPAD bank (halves); C strip
        # + bias vector in one ACC bank (bank / 2 words, D int32 each)
        if quant is not None and (k > bank // 2 or n > bank // 2 or n + n // d > bank // 2):
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
                      MBOX_BIAS_BASE: bufs[3].physical_address if bias is not None else 0,
                      MBOX_GEMM_FLAGS: flags}
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
                   "us": cycles / RISCV_HZ * 1e6, "wall_s": wall, "perf": self._perf()}

    def vector(self, op, x, y=None, out_dtype=None, relu=False, requant=None, timeout=5.0):
        """One vector-engine operation with vector_fw.bin (M3):
            out = saturate(clip(requant(relu(op(x, y)))))
        op: "add" "sub" "mul" "max" "min" "copy" (y unused); x: 1-D int8 / int16 / int32
        (mul: int8 / int16 only), length a multiple of D; y: same dtype, the same
        length or D * period elements, period <= 512 (repeated; D = broadcast one group);
        out_dtype int8 / int16 / int32 (default x's); requant: Requant or None."""
        self._use("vector_fw.bin")
        x = np.ascontiguousarray(x).ravel()
        it = VTYPES[x.dtype]
        out_dtype = np.dtype(out_dtype or x.dtype)
        ot = VTYPES[out_dtype]
        code = VOPS[op]
        n = x.size
        d = self.d
        if n % d or n == 0:
            raise ValueError(f"length {n}: need a positive multiple of {d}")
        if op == "mul" and it == 2:
            raise ValueError("mul takes int8 / int16 inputs")
        if op == "copy":
            y, period = x[:d], 0
        else:
            y = np.ascontiguousarray(y, dtype=x.dtype).ravel()
            if y.size == n:
                period = 0
            elif y.size % d == 0 and 0 < y.size // d <= 512:
                period = y.size // d
            else:
                raise ValueError(f"y: {y.size} elements, need {n} or {d} * period (period <= 512)")
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
                     "us": cycles / RISCV_HZ * 1e6, "wall_s": wall, "perf": self._perf()}

    # ------------------------------------------------------ descriptor lists
    def run_list(self, dl, bases=(0, 0, 0, 0), count=0, timeout=5.0):
        """Run a DescList with desc_run_fw.bin: one mat_submit, then mat_fence.
        bases: BASE0..BASE3 for descriptors built with base=n. Returns stats."""
        self._use("desc_run_fw.bin")
        rows = dl.array()
        buf = allocate(shape=rows.shape, dtype=np.uint64)     # page aligned (>= 64 B)
        try:
            buf[:] = rows
            buf.flush()
            params = {MBOX_DL_ADDR: buf.physical_address, MBOX_DL_COUNT: count}
            for i, b in enumerate(bases):
                params[MBOX_DL_BASE0 + 4 * i] = b
            wall = self._run(params, timeout)
        finally:
            buf.freebuffer()
        if self._mbox(MBOX_ERRORS) and self._mbox(MBOX_FIRST_ERR) == ERR_NO_DESC:
            raise RuntimeError("this overlay has no descriptor fetch unit (CAPS bit 21)")
        cycles = self._mbox(MBOX_TOTAL_CYCLES)
        return {"firmware": self.firmware, "descriptors": len(dl), "riscv_cycles": cycles,
                "us": cycles / RISCV_HZ * 1e6, "wall_s": wall, "dl_status": self._mbox(MBOX_DL_STATUS),
                "dl_exec": self._mbox(MBOX_DL_EXEC), "perf": self._perf()}

    def gemm_list(self, a, b, bias=None, quant=None, relu=False, timeout=5.0):
        """gemm() as one descriptor list built here (build_gemm_list) and run
        with a single mat_submit; same arguments, results and limits."""
        a = np.ascontiguousarray(a, dtype=np.int8)
        b = np.ascontiguousarray(b, dtype=np.int8)
        (m, k), (k2, n) = a.shape, b.shape
        d = self.d
        if k != k2 or m % d or n % d or k % d or k * n > SPAD_BYTES:
            raise ValueError(f"shapes {a.shape} x {b.shape}: need matching K, multiples of {d}, K*N <= 128 KB")
        bshape = (n,) if quant is not None else (m, n)
        bufs = [allocate(shape=a.shape, dtype=np.int8), allocate(shape=b.shape, dtype=np.int8),
                allocate(shape=(m, n), dtype=np.int8 if quant is not None else np.int32)]
        if bias is not None:
            bufs.append(allocate(shape=bshape, dtype=np.int32))
        try:
            bufs[0][:] = a
            bufs[1][:] = b
            bufs[2][:] = 0
            if bias is not None:
                bufs[3][:] = np.asarray(bias, dtype=np.int32).reshape(bshape)
            for buf in bufs:
                buf.flush()
            dl = build_gemm_list(d, m, n, k, bufs[0].physical_address, bufs[1].physical_address,
                                 bufs[2].physical_address, bufs[3].physical_address if bias is not None else None,
                                 quant, relu)
            st = self.run_list(dl, timeout=timeout)
            bufs[2].invalidate()
            c = np.array(bufs[2])
        finally:
            for buf in bufs:
                buf.freebuffer()
        st.update(shape=(m, n, k), mac_per_cycle=m * n * k / st["riscv_cycles"] if st["riscv_cycles"] else 0)
        return c, st

    def vector_list(self, op, x, y=None, out_dtype=None, relu=False, requant=None, timeout=5.0):
        """vector() as one descriptor list (build_vector_list); same arguments."""
        x = np.ascontiguousarray(x).ravel()
        out_dtype = np.dtype(out_dtype or x.dtype)
        it, ot, n, d = VTYPES[x.dtype], VTYPES[out_dtype], x.size, self.d
        if n % d or n == 0:
            raise ValueError(f"length {n}: need a positive multiple of {d}")
        y = x[:d] if op == "copy" else np.ascontiguousarray(y, dtype=x.dtype).ravel()
        if op != "copy" and y.size != n and (y.size % d or not 0 < y.size // d <= 512):
            raise ValueError(f"y: {y.size} elements, need {n} or {d} * period (period <= 512)")
        bufs = [allocate(shape=(n,), dtype=x.dtype), allocate(shape=(y.size,), dtype=x.dtype),
                allocate(shape=(n,), dtype=out_dtype)]
        try:
            bufs[0][:] = x
            bufs[1][:] = y
            bufs[2][:] = 0
            for buf in bufs:
                buf.flush()
            dl = build_vector_list(d, op, it, ot, n, y.size, bufs[0].physical_address,
                                   bufs[1].physical_address, bufs[2].physical_address, relu, requant)
            st = self.run_list(dl, timeout=timeout)
            bufs[2].invalidate()
            out = np.array(bufs[2])
        finally:
            for buf in bufs:
                buf.freebuffer()
        st.update(len=n, elem_per_cycle=n / st["riscv_cycles"] if st["riscv_cycles"] else 0)
        return out, st

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


class Device:
    """L1 host interface (docs/llm_inference_plan.md §5): the resident runtime
    firmware (firmware/rt, rt_fw.bin) serving a submission ring in DDR.

        dev = Device("picorv32.bit")
        seq = dev.submit(dl, bases=[...], params=[...])   # returns at once
        rec = dev.wait(seq)                                # completion record
        dev.close()

    submit() copies a DescList into a DDR buffer (or takes a physical list
    address), writes a 64-byte ring entry (+ an optional PARAM0..7 block),
    flushes and rings the doorbell (mailbox RING_TAIL); up to ring_size
    entries may be outstanding (not yet collected). The firmware completes entries in order and
    writes {seq, status, cycles, descriptors decoded, END value}; wait()
    polls RING_HEAD in the BRAM, or blocks on the notify interrupt
    (matmul_0/notify_irq through the AXI interrupt controller) when
    use_irq=True. Status 0 = success, otherwise the extended status of the
    failed list (the unit has been reset; later entries run normally)."""

    def __init__(self, bitfile=None, firmware=None, ring_size=64, download=True, use_irq=False,
                 timeout=5.0):
        if ring_size & (ring_size - 1):
            raise ValueError("ring_size must be a power of 2")
        self.mm = MatmulOverlay(bitfile, firmware or os.path.join(HERE, "rt_fw.bin"), download)
        self.size, self.tail, self.head = ring_size, 0, 0
        self.ring = allocate(shape=(ring_size, 8), dtype=np.uint64)
        self.cpl = allocate(shape=(ring_size, 8), dtype=np.uint32)
        self.params = allocate(shape=(ring_size, 8), dtype=np.uint32)
        self.ring[:] = 0
        self.cpl[:] = 0
        self.ring.flush()
        self.cpl.flush()
        self.keep = {}                               # seq -> buffers alive until completion
        self.done = {}                               # seq -> completion record
        self.irq = None
        if use_irq:
            from pynq import Interrupt
            self.irq = Interrupt("matmul_0/notify_irq")
        bram = self.mm.bram
        for w in range(MBOX_WORDS):
            bram.write(MBOX + 4 * w, 0)
        bram.write(MBOX + MBOX_RING_BASE, self.ring.physical_address)
        bram.write(MBOX + MBOX_RING_SIZE, ring_size)
        bram.write(MBOX + MBOX_CPL_BASE, self.cpl.physical_address)
        self.mm.reset.write(0)
        t0 = time.perf_counter()
        while self.mm._mbox(MBOX_FW_STATE) != RT_READY:
            if time.perf_counter() - t0 > timeout:
                raise RuntimeError(f"rt_fw not ready (FW_STATE {self.mm._mbox(MBOX_FW_STATE):#x}); "
                                   "the overlay needs CAPS bits 21, 22 and 24")
        self.d = self.mm.d

    def completed(self):
        """Entries completed so far (RING_HEAD)."""
        return self.mm._mbox(MBOX_RING_HEAD)

    def submit(self, dl, bases=(0, 0, 0, 0), params=None, count=0, irq=None, perf=False, kind=RT_RUN_LIST,
               timeout=5.0):
        """Queue one entry; returns its sequence number. dl: DescList or a
        physical list address (None for NOP / RESET / EXIT)."""
        t0 = time.perf_counter()
        # an entry's slot, and its completion record, may be reused only
        # after the ARM has read that record: collect finished records while
        # the ring is full (counting RING_HEAD alone would let the firmware
        # overwrite records not yet read)
        self._collect()
        while self.tail - self.head >= self.size:
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError("submission ring full")
            self._collect()
        seq, slot = self.tail, self.tail % self.size
        keep = []
        addr = 0
        if isinstance(dl, DescList):
            buf = allocate(shape=(len(dl), 8), dtype=np.uint64)
            buf[:] = dl.array()
            buf.flush()
            keep.append(buf)
            addr = buf.physical_address
        elif dl is not None:
            addr = int(dl)
        pb = 0
        if params is not None:
            self.params[slot, :] = 0
            self.params[slot, :len(params)] = [int(p) & _M32 for p in params]
            self.params.flush()
            pb = self.params.physical_address + 32 * slot
        irq = self.irq is not None if irq is None else irq
        flags = kind | (RT_F_IRQ if irq else 0) | (RT_F_PERF if perf else 0)
        b = list(bases) + [0] * (4 - len(bases))
        self.ring[slot, :] = [flags | seq << 32, addr, count, b[0] & _M32, b[1] & _M32, b[2] & _M32,
                              b[3] & _M32, pb]
        self.ring.flush()
        self.keep[seq] = keep
        self.tail += 1
        self.mm.bram.write(MBOX + MBOX_RING_TAIL, self.tail)          # doorbell
        return seq

    def _collect(self):
        head = self.completed()
        if head != self.head:
            self.cpl.invalidate()
        while self.head < head:
            c = self.cpl[self.head % self.size]
            if int(c[0]) != self.head & _M32:
                raise RuntimeError(f"completion record {self.head}: seq {int(c[0])}")
            self.done[self.head] = {"seq": self.head, "status": int(c[1]), "cycles": int(c[2]),
                                    "descriptors": int(c[3]), "end": int(c[4])}
            for buf in self.keep.pop(self.head, []):
                buf.freebuffer()
            self.head += 1

    def wait(self, seq, timeout=5.0):
        """Completion record of `seq` (after all earlier entries)."""
        t0 = time.perf_counter()
        while True:
            self._collect()
            if seq in self.done:
                return self.done.pop(seq)
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError(f"entry {seq} not completed (head {self.completed()}, tail {self.tail})")
            if self.irq is not None:
                import asyncio
                try:
                    asyncio.get_event_loop().run_until_complete(asyncio.wait_for(self.irq.wait(), 0.2))
                except asyncio.TimeoutError:
                    pass

    def run(self, dl, bases=(0, 0, 0, 0), params=None, count=0, timeout=5.0):
        return self.wait(self.submit(dl, bases, params, count), timeout)

    def close(self):
        """EXIT entry (the firmware leaves its loop), then hold the core in reset."""
        try:
            self.wait(self.submit(None, kind=RT_EXIT), 2.0)
        finally:
            self.mm.reset.write(1)
            for keep in self.keep.values():
                for buf in keep:
                    buf.freebuffer()
            for buf in (self.ring, self.cpl, self.params):
                buf.freebuffer()


def perf_breakdown(perf, d):
    """Derived metrics of a counter dict (stats["perf"]) for array size d:
    fractions of CYCLES, bandwidth while busy, MACs. Returns a dict."""
    cyc = perf["CYCLES"] or 1
    frac = lambda k: perf[k] / cyc
    per_busy = lambda beats, busy: 8 * perf[beats] / perf[busy] if perf[busy] else 0.0
    return {
        "cycles": perf["CYCLES"],
        "ex_useful": frac("EX_USEFUL"),                                  # array doing new MACs
        "ex_fill": (perf["EX_STEP"] - perf["EX_USEFUL"]) / cyc,          # skew fill / drain steps
        "ex_swapwait": frac("EX_SWAPWAIT"),                              # waiting for the ACC drain
        "head_blocked": {e: frac("HAZ_" + e) for e in ("LD", "ST", "EX", "VE")},
        "queue_full": frac("DISP_FULL"),
        "starve": frac("STARVE"),                                        # no command at the head
        "all_idle": frac("ALL_IDLE"),
        "cpu_on_full_queue": frac("PCPI_QFULL"),
        "ld_bytes_per_busy_cycle": per_busy("LD_BEATS", "LD_BUSY"),
        "st_bytes_per_busy_cycle": per_busy("ST_BEATS", "ST_BUSY"),
        "ld_busy": frac("LD_BUSY"), "st_busy": frac("ST_BUSY"), "ve_active": frac("VE_ACTIVE"),
        "macs": perf["EX_USEFUL"] * d * d,
        "commands": {e: perf["CMD_" + e] for e in ("LD", "ST", "EX", "VE")},
    }


def format_breakdown(b):
    """One-screen text of perf_breakdown()."""
    pct = lambda x: f"{100 * x:5.1f}%"
    hb = b["head_blocked"]
    return "\n".join([
        f"  EX   useful {pct(b['ex_useful'])}  fill {pct(b['ex_fill'])}  drain-wait {pct(b['ex_swapwait'])}",
        f"  head blocked  LD {pct(hb['LD'])}  ST {pct(hb['ST'])}  EX {pct(hb['EX'])}  VE {pct(hb['VE'])}"
        f"   queue full {pct(b['queue_full'])}",
        f"  front end     starve {pct(b['starve'])}  all idle {pct(b['all_idle'])}"
        f"  CPU stalled on a full queue {pct(b['cpu_on_full_queue'])}",
        f"  DMA           LD busy {pct(b['ld_busy'])} at {b['ld_bytes_per_busy_cycle']:.2f} B/cycle,"
        f"  ST busy {pct(b['st_busy'])} at {b['st_bytes_per_busy_cycle']:.2f} B/cycle,  VE active {pct(b['ve_active'])}",
    ])


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

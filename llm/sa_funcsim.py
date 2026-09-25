"""Functional simulator of rtl/sysarray (docs/llm_inference_plan.md §4.3).

Executes descriptor lists (docs/double_buffer_design.md §8.6) on a NumPy model
of DDR, SPAD_A, SPAD_B and ACC, bit-exact with the RTL: the local memories
are byte arrays with the RTL word layout (SPAD word = D bytes, ACC word = D
int32), commands run one at a time in list order. That is the RTL result for
any list that is correct under the scoreboard rules (local-memory hazards
are ordered by the scoreboard, DDR hazards need a FENCE, which a sequential
model always satisfies).

Checks mirror sa_sched.v / sa_cmdfetch.v: an illegal command raises SaError
with the RTL engine and error code, and the list stops (sticky error).

Covers the M5 hardware: LD / ST (LINEAR, INTERLEAVE), EX (repeat, bstep,
cstep, crow, accumulate), integer VE (ADD SUB MUL MAX MIN COPY, RELU,
REQUANT, clamp, periodic src2), FENCE, JUMP, END, count limit; and the L1
command extensions (docs/llm_inference_plan.md §5.3): BASE0-15, PARAM0-7,
two dynamic field slots per descriptor, SETREG, LOOP_END (2 nested levels),
relative JUMP, CALL / RET (depth 4), LDPARAM. L2 is added here first too.
"""
import numpy as np

import fp32

MEM_SPAD_A, MEM_SPAD_B, MEM_ACC = 1, 2, 3
ENG_LD, ENG_ST, ENG_EX, ENG_VE, ENG_FETCH = 0, 1, 2, 3, 4
XERR_SHAPE, XERR_RANGE, XERR_RRESP, XERR_BRESP = 1, 2, 3, 4
DESC_LD, DESC_ST, DESC_EX, DESC_VE = 0x01, 0x02, 0x03, 0x04
DESC_FENCE, DESC_JUMP, DESC_END = 0x10, 0x11, 0x12
DESC_LOOP_END, DESC_SETREG, DESC_CALL, DESC_RET, DESC_LDPARAM = 0x13, 0x14, 0x15, 0x16, 0x17
NBASE, NPARAM, LOOP_DEPTH, CALL_DEPTH = 16, 8, 2, 4
# dynamic field slots (w0[23:16] slot 0, w0[31:24] slot 1): [3:0] field (0 = none),
# [6:4] PARAM, [7] 0 replace / 1 add. Field -> (word, lsb, width) per opcode.
_F_DMA = {1: (1, 0, 32), 2: (2, 0, 16), 3: (2, 32, 16), 4: (2, 48, 16), 5: (3, 0, 32)}
DYN_FIELDS = {
    0x01: _F_DMA, 0x02: _F_DMA,
    0x03: {1: (1, 0, 16), 2: (1, 16, 16), 3: (1, 32, 16), 4: (1, 48, 12), 5: (2, 0, 12), 6: (2, 16, 16),
           7: (2, 32, 16)},
    0x04: {1: (1, 0, 16), 2: (1, 32, 16), 3: (2, 0, 16), 4: (2, 32, 32), 5: (7, 16, 16), 6: (7, 0, 16),
           7: (7, 32, 16), 8: (3, 16, 16), 9: (6, 0, 32), 10: (6, 32, 32), 11: (5, 32, 32)},
    0x11: {1: (1, 0, 32)}, 0x15: {1: (1, 0, 32)}, 0x13: {1: (2, 0, 16)},
    0x14: {1: (2, 0, 32), 2: (3, 0, 32), 3: (4, 0, 32)}, 0x17: {1: (1, 0, 32)},
    0x10: {}, 0x12: {}, 0x16: {},
}
VOP_ADD, VOP_SUB, VOP_MUL, VOP_MAX, VOP_MIN, VOP_COPY = range(6)
VT_I8, VT_I16, VT_I32, VT_F32 = 0, 1, 2, 3
VOP_TRANSPOSE = 6
# L2 fp32 VE (docs/llm_inference_plan.md §6): flags = w3[63:53]
VF_FP, VF_FUNC, VF_M1, VF_M2, VF_RED, VF_SWAPNEG = 0, 1, 4, 6, 8, 10          # bit offsets in flags
FUNC_NONE, FUNC_EXP, FUNC_RECIP, FUNC_RSQRT, FUNC_ABS = range(5)
IDX_LIN, IDX_MOD, IDX_DIV, IDX_IMM = range(4)
RED_NONE, RED_SUM, RED_MAX = range(3)
MODE_INTERLEAVE = 1
_VT_DTYPE = {VT_I8: "<i1", VT_I16: "<i2", VT_I32: "<i4"}
_VT_RANGE = {VT_I8: (-128, 127), VT_I16: (-32768, 32767), VT_I32: (-2**31, 2**31 - 1)}
_M32 = 0xFFFFFFFF


class SaError(Exception):
    def __init__(self, engine, code, index=None, msg=""):
        super().__init__(f"engine {engine} code {code}" + (f" at descriptor {index}" if index is not None else "")
                         + (f": {msg}" if msg else ""))
        self.engine, self.code, self.index = engine, code, index


def _s32(x):
    x &= _M32
    return x - (1 << 32) if x & 0x80000000 else x


class SaFuncSim:
    def __init__(self, d=8, ddr_base=0, ddr_bytes=1 << 20):
        assert d in (8, 16)
        self.d = d
        self.logd = d.bit_length() - 1
        self.spad_words = 131072 // d
        self.acc_words = 262144 // (4 * d)
        self.mem = {MEM_SPAD_A: np.zeros(self.spad_words * d, np.uint8),
                    MEM_SPAD_B: np.zeros(self.spad_words * d, np.uint8),
                    MEM_ACC: np.zeros(self.acc_words * 4 * d, np.uint8)}
        self.ddr_base = ddr_base
        self.ddr = np.zeros(ddr_bytes, np.uint8)
        self.bases = [0] * NBASE
        self.params = [0] * NPARAM
        self.dl_status = 0
        self.dl_exec = 0
        self.cmds = {"LD": 0, "ST": 0, "EX": 0, "VE": 0}

    # ------------------------------------------------------------ helpers
    def wbytes(self, mem):
        return 4 * self.d if mem == MEM_ACC else self.d

    def depth(self, mem):
        return self.acc_words if mem == MEM_ACC else self.spad_words

    def _ddr(self, addr, n, eng=ENG_LD, code=XERR_RRESP):
        off = addr - self.ddr_base
        if off < 0 or off + n > self.ddr.size:
            raise SaError(eng, code, msg=f"DDR {addr:#x}+{n} outside the model")
        return off

    def ddr_write(self, addr, data):
        data = np.frombuffer(bytes(data), np.uint8) if not isinstance(data, np.ndarray) else data.view(np.uint8).ravel()
        off = self._ddr(addr, data.size)
        self.ddr[off:off + data.size] = data

    def ddr_read(self, addr, n):
        off = self._ddr(addr, n)
        return self.ddr[off:off + n].copy()

    # ----------------------------------------------------------- engines
    def ld(self, ddr, laddr, rows, rb, pitch, mode=0):
        mem, word = (laddr >> 28) & 0xF, laddr & 0xFFFF
        ilv = mode & MODE_INTERLEAVE
        self._check_dma(ENG_LD, ddr, mem, word, rows, rb, pitch, ilv)
        buf, wb = self.mem[mem], self.wbytes(mem)
        wpr = -(-rb // wb)
        for r in range(rows):
            src = self._ddr(ddr + r * pitch, rb)
            row = self.ddr[src:src + rb]
            if ilv:
                for c in range(rb // self.d):
                    w = word + c * rows + r
                    buf[w * wb:w * wb + self.d] = row[c * self.d:(c + 1) * self.d]
            else:
                start = (word + r * wpr) * wb
                buf[start:start + rb] = row
        self.cmds["LD"] += 1

    def st(self, ddr, laddr, rows, rb, pitch):
        mem, word = (laddr >> 28) & 0xF, laddr & 0xFFFF
        self._check_dma(ENG_ST, ddr, mem, word, rows, rb, pitch, 0)
        buf, wb = self.mem[mem], self.wbytes(mem)
        wpr = -(-rb // wb)
        for r in range(rows):
            dst = self._ddr(ddr + r * pitch, rb, ENG_ST, XERR_BRESP)
            start = (word + r * wpr) * wb
            self.ddr[dst:dst + rb] = buf[start:start + rb]
        self.cmds["ST"] += 1

    def _check_dma(self, eng, ddr, mem, word, rows, rb, pitch, ilv):
        if ddr & 7 or rb & 7 or rb == 0 or rows == 0 or pitch & 7 or \
                (ilv and (eng != ENG_LD or mem == MEM_ACC or rb & (self.d - 1))):
            raise SaError(eng, XERR_SHAPE, msg="DMA shape")
        if mem not in (MEM_SPAD_A, MEM_SPAD_B, MEM_ACC):
            raise SaError(eng, XERR_RANGE, msg=f"memory id {mem}")
        span = (rb >> self.logd) * rows if ilv else -(-rb // self.wbytes(mem)) * rows
        if word + span - 1 >= self.depth(mem):
            raise SaError(eng, XERR_RANGE, msg="local range")

    def ex(self, a, b, c, kt, acc=False, repeat=1, bstep=0, cstep=0, crow=1):
        d = self.d
        k = kt * d
        rep = max(repeat, 1)
        crow = crow or 1
        if kt == 0:
            raise SaError(ENG_EX, XERR_SHAPE, msg="Kt = 0")
        if a + k > self.spad_words or b + (rep - 1) * bstep + k - 1 >= self.spad_words or \
                c + (rep - 1) * cstep + (d - 1) * crow >= self.acc_words:
            raise SaError(ENG_EX, XERR_RANGE, msg="EX range")
        sa, sb = self.mem[MEM_SPAD_A].view(np.int8), self.mem[MEM_SPAD_B].view(np.int8)
        amat = sa[a * d:(a + k) * d].reshape(kt, d, d).transpose(1, 0, 2).reshape(d, k).astype(np.int64)
        accw = self.mem[MEM_ACC].view("<i4")
        for r in range(rep):
            b0 = b + r * bstep
            bmat = sb[b0 * d:(b0 + k) * d].reshape(k, d).astype(np.int64)
            cm = amat @ bmat
            for i in range(d):
                w = c + r * cstep + i * crow
                row = cm[i]
                if acc:
                    row = row + accw[w * d:(w + 1) * d]
                accw[w * d:(w + 1) * d] = ((row + 2**31) % 2**32 - 2**31).astype(np.int32)
        self.cmds["EX"] += 1

    def ve(self, src1, src2, dst, groups, op_byte, types, period=0, scale=1, shift=0, zp=0,
           lo=-2**31, hi=2**31 - 1, flags=0, imm=0, a=0, b=0, rowlen=0, valid=0, p1=0, s=0):
        """flags .. s: the L2 fields (flags = w3[63:53], IMM, A, B, ROWLEN,
        VALID, P1, S); with FP = 0 and op != TRANSPOSE they must be 0."""
        if op_byte & 7 == VOP_TRANSPOSE:
            return self.ve_transpose(src1, dst, groups, op_byte, types, flags, imm, a, b, rowlen, valid, p1, s)
        if flags & 1:
            return self.ve_fp(src1, src2, dst, groups, op_byte, types, period, flags, imm, a, b, rowlen, valid, p1)
        if flags or imm or a or b or rowlen or valid or p1 or s:
            raise SaError(ENG_VE, XERR_SHAPE, msg="L2 fields set without FP")
        op, relu, requant = op_byte & 7, (op_byte >> 4) & 1, (op_byte >> 5) & 1
        it, ot = types & 3, (types >> 2) & 3
        m1, w1 = (src1 >> 28) & 0xF, src1 & 0xFFFF
        m2, w2 = (src2 >> 28) & 0xF, src2 & 0xFFFF
        md, wd = (dst >> 28) & 0xF, dst & 0xFFFF
        unary = op == VOP_COPY
        if groups == 0 or it > VT_I32 or ot > VT_I32 or op > VOP_COPY or (op == VOP_MUL and it == VT_I32):
            raise SaError(ENG_VE, XERR_SHAPE, msg="VE shape")
        mem_for = lambda t, m: m == MEM_ACC if t == VT_I32 else m in (MEM_SPAD_A, MEM_SPAD_B)
        wg_in, wg_out = (2 if it == VT_I16 else 1), (2 if ot == VT_I16 else 1)
        g2n = groups if period == 0 else 1 if period == 1 else min(period, groups)
        if not (mem_for(it, m1) and (unary or mem_for(it, m2)) and mem_for(ot, md)) or \
                w1 + groups * wg_in - 1 >= self.depth(m1) or \
                (not unary and w2 + g2n * wg_in - 1 >= self.depth(m2)) or \
                wd + groups * wg_out - 1 >= self.depth(md):
            raise SaError(ENG_VE, XERR_RANGE, msg="VE range")
        d = self.d
        esz = np.dtype(_VT_DTYPE[it]).itemsize
        read = lambda m, w, g: self.mem[m][w * self.wbytes(m):w * self.wbytes(m) + g * d * esz] \
            .view(_VT_DTYPE[it]).astype(np.int64)
        xa = read(m1, w1, groups)
        if unary:
            xb = np.zeros_like(xa)
        else:
            g = np.arange(groups)
            gi = g if period == 0 else np.zeros_like(g) if period == 1 else g % period
            xb = read(m2, w2, g2n).reshape(g2n, d)[gi].ravel()
        sat = lambda x: np.clip(x, -2**31, 2**31 - 1)
        if op == VOP_ADD:
            r = sat(xa + xb)
        elif op == VOP_SUB:
            r = sat(xa - xb)
        elif op == VOP_MUL:
            lo16 = lambda v: ((v & 0xFFFF) ^ 0x8000) - 0x8000
            r = lo16(xa) * lo16(xb)
        elif op == VOP_MAX:
            r = np.maximum(xa, xb)
        elif op == VOP_MIN:
            r = np.minimum(xa, xb)
        else:
            r = xa
        if relu:
            r = np.where(r < 0, 0, r)
        if requant:
            s = _s32(scale & 0xFFFF | (0xFFFF0000 if scale & 0x8000 else 0))
            q = (r * s + ((1 << (shift - 1)) if shift else 0)) >> shift
            q = q + _s32(zp)
        else:
            q = r
        lo_, hi_ = _s32(lo), _s32(hi)
        cval = np.where(q < lo_, lo_, np.where(q > hi_, hi_, q))
        tmin, tmax = _VT_RANGE[ot]
        cval = np.clip(cval, tmin, tmax).astype(_VT_DTYPE[ot])
        out = cval.view(np.uint8)
        start = wd * self.wbytes(md)
        self.mem[md][start:start + out.size] = out
        self.cmds["VE"] += 1

    # ------------------------------------------------ L2: fp32 VE, TRANSPOSE
    def _range_ok(self, mem, word, nwords):
        return word + nwords - 1 < self.depth(mem)

    def _idx(self, mode, P, G):
        """source group of every output group, and the number of groups read"""
        g = np.arange(G)
        if mode == IDX_LIN:
            return g, G
        if P == 0:
            raise SaError(ENG_VE, XERR_SHAPE, msg="MOD / DIV with period 0")
        if mode == IDX_MOD:
            return g % P, min(P, G)
        return g // P, G              # DIV: the range check takes G words (as sa_sched.v, no divider)

    def _read_fp(self, mem, word, t, ngroups):
        """ngroups x D elements of type t at mem/word as fp32 bits"""
        d = self.d
        if t == VT_I8:
            v = self.mem[mem][word * d:(word + ngroups) * d].view(np.int8).astype(np.int64)
            return fp32.from_int(v).reshape(ngroups, d)
        raw = self.mem[mem][word * 4 * d:(word + ngroups) * 4 * d].view("<u4")
        if t == VT_I32:
            return fp32.from_int(raw.view("<i4").astype(np.int64)).reshape(ngroups, d)
        return fp32.ftz(raw).reshape(ngroups, d)

    def ve_fp(self, src1, src2, dst, G, op_byte, types, period, flags, imm, A, B, rowlen, valid, p1):
        import fp32 as F
        import sfu
        d = self.d
        op, relu = op_byte & 7, (op_byte >> 4) & 1
        it, ot = types & 3, (types >> 2) & 3
        t2 = (it, VT_I8, VT_I32, VT_F32)[(types >> 4) & 3]              # src2 type: 0 = as src1
        func, mo1, mo2 = (flags >> VF_FUNC) & 7, (flags >> VF_M1) & 3, (flags >> VF_M2) & 3
        red, swapneg = (flags >> VF_RED) & 3, (flags >> VF_SWAPNEG) & 1
        m1, w1 = (src1 >> 28) & 0xF, src1 & 0xFFFF
        m2, w2 = (src2 >> 28) & 0xF, src2 & 0xFFFF
        md, wd = (dst >> 28) & 0xF, dst & 0xFFFF
        unary = op == VOP_COPY
        spad = (MEM_SPAD_A, MEM_SPAD_B)
        mem_ok = lambda t, m: m in spad if t == VT_I8 else m == MEM_ACC if t in (VT_I32, VT_F32) else False
        requant = (op_byte >> 5) & 1                                  # integer-only post-op
        if G == 0 or op > VOP_COPY or func > FUNC_ABS or mo1 == 3 or red == 3 or requant \
                or it == VT_I16 or ot == VT_I16 or (red and ot != VT_F32) or t2 == VT_I16:
            raise SaError(ENG_VE, XERR_SHAPE, msg="VE fp shape")
        rows = 1
        if rowlen:
            if G % rowlen:
                raise SaError(ENG_VE, XERR_SHAPE, msg="LEN not a multiple of ROWLEN")
            rows = G // rowlen
        else:
            rowlen = G
        i1, n1 = self._idx(mo1, p1, G)
        use2 = not unary and mo2 != IDX_IMM
        if use2:
            i2, n2 = self._idx(mo2, period, G)
        nd = G                        # range check: G words also for REDUCE outputs (as sa_sched.v)
        if not (mem_ok(it, m1) and (not use2 or mem_ok(t2, m2)) and mem_ok(ot, md)) or \
                not self._range_ok(m1, w1, n1) or (use2 and not self._range_ok(m2, w2, n2)) or \
                not self._range_ok(md, wd, nd):
            raise SaError(ENG_VE, XERR_RANGE, msg="VE fp range")
        n1 = int(i1.max()) + 1                                       # words actually read
        a = self._read_fp(m1, w1, it, n1)[i1]                        # G x D
        if swapneg:
            a2 = a.copy()
            a2[:, 0::2], a2[:, 1::2] = F.neg(a[:, 1::2]), a[:, 0::2]
            a = a2
        if unary:
            y = a
        else:
            bv = np.full_like(a, np.uint32(imm)) if mo2 == IDX_IMM else \
                self._read_fp(m2, w2, t2, int(i2.max()) + 1)[i2]
            y = {0: F.add, 1: F.sub, 2: F.mul, 3: F.fmax, 4: F.fmin}[op](a, bv)
        y = F.add(F.mul(y, np.uint32(A)), np.uint32(B))
        y = {FUNC_NONE: lambda v: v, FUNC_EXP: sfu.exp, FUNC_RECIP: sfu.recip, FUNC_RSQRT: sfu.rsqrt,
             FUNC_ABS: sfu.fabs}[func](y)
        if relu:                                                     # negative (incl. -0, -inf) -> +0; NaN stays
            nan = (((y >> 23) & 0xFF) == 255) & ((y & 0x7FFFFF) != 0)
            y = np.where(((y >> 31) == 1) & ~nan, np.uint32(0), y).astype(np.uint32)
        # valid mask: element (g, l) is number (g mod rowlen) * D + l of its row
        e = (np.arange(G) % rowlen)[:, None] * d + np.arange(d)[None, :]
        vmask = np.ones((G, d), bool) if valid == 0 else e < valid
        if red:
            ident = np.uint32(0) if red == RED_SUM else F.NINF
            fn = F.add if red == RED_SUM else F.fmax
            y = np.where(vmask, y, ident).reshape(rows, rowlen, d)
            acc = y[:, 0, :]
            for gi in range(1, rowlen):
                acc = fn(acc, y[:, gi, :])
            while acc.shape[1] > 1:
                acc = fn(acc[:, 0::2], acc[:, 1::2])
            out = np.repeat(acc, d, axis=1).astype(np.uint32)       # rows x D, broadcast
            self.mem[MEM_ACC][wd * 4 * d:(wd + rows) * 4 * d] = out.view(np.uint8).ravel()
        else:
            y = np.where(vmask, y, np.uint32(0)).astype(np.uint32)
            if ot == VT_F32:
                self.mem[MEM_ACC][wd * 4 * d:(wd + G) * 4 * d] = y.view(np.uint8).ravel()
            elif ot == VT_I32:
                v = F.to_int(y, -2**31, 2**31 - 1).astype("<i4")
                self.mem[MEM_ACC][wd * 4 * d:(wd + G) * 4 * d] = v.view(np.uint8).ravel()
            else:
                v = F.to_int(y, -127, 127).astype(np.int8)
                self.mem[md][wd * d:(wd + G) * d] = v.view(np.uint8).ravel()
        self.cmds["VE"] += 1

    def ve_transpose(self, src1, dst, G, op_byte, types, flags, imm, a, b, rowlen, valid, p1, S):
        """D x D block transpose: block k = (kb, ks) of an R x S-word matrix
        (row stride S words) -> D consecutive words at dst + k * D."""
        d = self.d
        it, ot = types & 3, (types >> 2) & 3
        m1, w1 = (src1 >> 28) & 0xF, src1 & 0xFFFF
        md, wd = (dst >> 28) & 0xF, dst & 0xFFFF
        if (op_byte & ~7) or flags or imm or a or b or rowlen or valid or p1 or S == 0 or it != ot or \
                it == VT_I16 or G % d or (G // d) % S:            # (G/D) % S: not checked by the RTL
            raise SaError(ENG_VE, XERR_SHAPE, msg="TRANSPOSE shape")
        nblk = G // d
        esz = 1 if it == VT_I8 else 4
        spad = (MEM_SPAD_A, MEM_SPAD_B)
        if not ((it == VT_I8 and m1 in spad and md in spad) or (it != VT_I8 and m1 == md == MEM_ACC)):
            raise SaError(ENG_VE, XERR_RANGE, msg="TRANSPOSE memories")
        last_src = ((nblk - 1) // S) * d * S + ((nblk - 1) % S) + (d - 1) * S
        if w1 + last_src >= self.depth(m1) or wd + nblk * d - 1 >= self.depth(md):
            raise SaError(ENG_VE, XERR_RANGE, msg="TRANSPOSE range")
        wb = d * esz
        src = self.mem[m1].copy()
        for k in range(nblk):
            base = w1 + (k // S) * d * S + (k % S)
            blk = np.stack([src[(base + j * S) * wb:(base + j * S + 1) * wb].view(f"<u{esz}") for j in range(d)])
            out = blk.T.copy()                                       # word i = element i of every input word
            self.mem[md][(wd + k * d) * wb:(wd + k * d + d) * wb] = out.view(np.uint8).ravel()
        self.cmds["VE"] += 1

    # ---------------------------------------------------- descriptor lists
    def set_reg(self, idx, value):
        """mat_cfg keys 11-26 (idx 0-15: BASE0-15) and 27-34 (idx 16-23: PARAM0-7)."""
        if idx < NBASE:
            self.bases[idx] = value & _M32
        else:
            self.params[idx - NBASE] = value & _M32

    def run_list(self, addr, count=0, bases=None, params=None, max_desc=1_000_000):
        """Execute the list at DDR `addr` (mat_submit). bases / params: initial
        BASE0.. / PARAM0.. values (default: keep). Returns the number of
        descriptors decoded; the END value is in self.dl_status. Raises
        SaError (with .index = descriptor number in the list) on an illegal
        descriptor or command."""
        for i, b in enumerate(bases or []):
            self.bases[i] = b & _M32
        for i, p in enumerate(params or []):
            self.params[i] = p & _M32
        if addr & 63:
            raise SaError(ENG_FETCH, XERR_SHAPE, 0, "list address not 64-byte aligned")
        loops, calls = [], []                    # [[LOOP_END address, remaining]], [return address]
        n = 0
        while n < max_desc:
            w = [int(x) for x in self.ddr[self._ddr(addr, 64, ENG_FETCH):][:64].view("<u8")]
            idx = n
            n += 1
            self.dl_exec = n

            def fail(msg, code=XERR_SHAPE):
                raise SaError(ENG_FETCH, code, idx, msg)

            op, hdr = w[0] & 0xFF, w[0] & _M32
            if op not in DYN_FIELDS or hdr & (1 << 15):
                fail(f"bad header {hdr:#x}")
            for slot in ((hdr >> 16) & 0xFF, (hdr >> 24) & 0xFF):
                f = slot & 0xF
                if f == 0:
                    continue
                if f not in DYN_FIELDS[op]:
                    fail(f"dynamic field {f} undefined for opcode {op:#x}")
                wd, lsb, width = DYN_FIELDS[op][f]
                mask = (1 << width) - 1
                p = self.params[(slot >> 4) & 7]
                old = (w[wd] >> lsb) & mask
                new = ((old + p) if slot & 0x80 else p) & mask
                w[wd] = (w[wd] & ~(mask << lsb)) | (new << lsb)
            limit = count != 0 and n == count
            nxt = addr + 64
            rel = lambda off: addr + 64 * _s32(off)
            if op in (DESC_LD, DESC_ST, DESC_EX, DESC_VE):
                try:
                    self._command(op, hdr, w)
                except SaError as e:
                    e.index = idx
                    raise
            elif op == DESC_FENCE:
                pass
            elif op == DESC_JUMP or op == DESC_CALL:
                target = rel(w[1]) if w[2] & 1 else w[1] & _M32
                if target & 63:
                    fail("jump target not 64-byte aligned")
                if op == DESC_CALL:
                    if len(calls) == CALL_DEPTH:
                        fail("call stack overflow")
                    calls.append(nxt)
                nxt = target
            elif op == DESC_RET:
                if not calls:
                    fail("call stack underflow")
                nxt = calls.pop()
            elif op == DESC_END:
                self.dl_status = w[1] & _M32
                return n
            elif op == DESC_SETREG:
                for i in range(3):
                    e = (w[1] >> (8 * i)) & 0x7F
                    if e & 0x40:
                        r = e & 0x3F
                        if r >= NBASE + NPARAM:
                            fail(f"SETREG register {r}")
                        v = w[2 + i] & _M32
                        cur = self.bases[r] if r < NBASE else self.params[r - NBASE]
                        self.set_reg(r, cur + v if (w[5] >> i) & 1 else v)
            elif op == DESC_LOOP_END:
                cnt = w[2] & 0xFFFF
                if cnt == 0:
                    fail("LOOP_END count 0")
                if loops and loops[-1][0] == addr:
                    rem = loops[-1][1]
                else:
                    if len(loops) == LOOP_DEPTH:
                        fail("loop stack overflow")
                    loops.append([addr, cnt])
                    rem = cnt
                for k, stride in (((w[2] >> 16) & 7, w[3]), ((w[2] >> 19) & 7, w[4])):
                    self.params[k] = (self.params[k] + stride) & _M32
                if rem > 1:
                    loops[-1][1] = rem - 1
                    nxt = rel(w[1])
                else:
                    loops.pop()
            elif op == DESC_LDPARAM:
                a = (w[1] + (self.bases[((hdr >> 13) & 3) << 2 | (hdr >> 9) & 3] if hdr & (1 << 8) else 0)) & _M32
                if a & 3:
                    fail("LDPARAM address not 4-byte aligned")
                try:
                    v = int(self.ddr_read(a, 4).view("<u4")[0])
                except SaError:
                    fail(f"LDPARAM read {a:#x}", XERR_RRESP)
                self.params[w[2] & 7] = (v * (w[3] & 0xFFFF) + w[4]) & _M32
            if limit:
                return n
            addr = nxt
        raise RuntimeError(f"list did not end within {max_desc} descriptors")

    def _command(self, op, hdr, w):
        f = lambda x, lo, n: (x >> lo) & ((1 << n) - 1)
        base = self.bases[f(hdr, 13, 2) << 2 | f(hdr, 9, 2)]
        ddr = (f(w[1], 0, 32) + (base if hdr & (1 << 8) else 0)) & _M32
        if op == DESC_LD:
            self.ld(ddr, f(w[2], 0, 32), f(w[2], 32, 16), f(w[2], 48, 16), f(w[3], 0, 32), f(w[3], 32, 2))
        elif op == DESC_ST:
            self.st(ddr, f(w[2], 0, 32), f(w[2], 32, 16), f(w[2], 48, 16), f(w[3], 0, 32))
        elif op == DESC_EX:
            rep = f(w[2], 0, 12)
            self.ex(f(w[1], 0, 16), f(w[1], 16, 16), f(w[1], 32, 16), f(w[1], 48, 12), bool(f(w[1], 60, 1)),
                    rep or 1, f(w[2], 16, 16), f(w[2], 32, 16), f(w[2], 48, 16))
        else:
            self.ve(f(w[1], 0, 32), f(w[1], 32, 32), f(w[2], 0, 32), f(f(w[2], 32, 32) >> self.logd, 0, 16),
                    f(w[3], 0, 8), f(w[3], 8, 6), f(w[3], 16, 16), f(w[3], 32, 16), f(w[3], 48, 5),
                    f(w[4], 0, 32), f(w[4], 32, 32), f(w[5], 0, 32),
                    flags=f(w[3], 53, 11), imm=f(w[5], 32, 32), a=f(w[6], 0, 32), b=f(w[6], 32, 32),
                    rowlen=f(w[7], 0, 16), valid=f(w[7], 16, 16), p1=f(w[7], 32, 16), s=f(w[7], 48, 16))

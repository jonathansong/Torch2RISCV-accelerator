"""Hand-written descriptor-list generator for the LLM layers (docs/llm_inference_plan.md §7.2, §8).

L3: activation quantization and W8A8 linear layers y = x W^T with the
weights of a .w8a8 file (llm/export_w8a8.py). Each function appends to a
driver DescList; DDR addresses are offsets relocated by a BASE register
(the model file in one, inputs / outputs in another), so a list is built
once and runs wherever the buffers land.

Local memory plan (words; sbank / cbank = one SPAD / ACC bank):
    SPAD_A   [0, k)                   the replicated int8 A strip (k = in)
    SPAD_B   bank p                   weight chunk p (double-buffered)
    ACC      bank p, [0, SCR)         chunk scratch: C rows, s_w chunk, y chunk
    ACC      [SCR, cbank) of bank 0   fp32 activations (the caller's)
             and of bank 1

A linear layer is computed in chunks of NC output tiles (NC * D outputs):
    LD   W chunk j   -> SPAD_B bank j % 2  (one contiguous LD: pre-packed tiles)
    LD   s_w chunk j -> ACC bank j % 2
    EX   C = A strip x W chunk (D identical rows; row 0 is used)
    LD   chunk j + 1 (overlaps the EX)
    VE   y = float(C row 0) * s_w          (I32 x F32 -> F32)
    VE   y = y * s_x                       (s_x broadcast word, DIV)
    [ST  y chunk -> DDR]
which is DeviceModel.linear() step for step, so the results are bit-exact
with it (with SfuExact). Long layers (the classifier) use one LOOP_END over
chunk pairs whose DDR addresses advance through two PARAM registers.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "driver"))
sys.path.insert(0, HERE)
from pynq_matmul import MEM_ACC, MEM_SPAD_A, MEM_SPAD_B, VOPS, DescList, _banks, laddr  # noqa: E402

I8, I32, F32 = 0, 2, 3
T_FF = F32 | F32 << 2
MUL, COPY = VOPS["mul"], VOPS["copy"]
INV127 = 1.0 / 127                                  # rounded to fp32 by the driver, as F(1/127)


class Layout:
    """Local memory plan for array size d."""

    def __init__(self, d):
        self.d = d
        self.sbank, self.cbank = _banks(d)          # words per SPAD / ACC bank
        self.scr = self.cbank // 4                  # chunk scratch at the start of each ACC bank
        self.acc0 = self.scr                        # first activation word (bank 0)
        self.acc1 = self.cbank + self.scr           # bank 1

    def chunk_tiles(self, k, nt):
        """Output tiles per chunk: the largest divisor of nt whose weights fit a
        SPAD_B bank (k words per tile) and whose scratch (C rows, s_w, y) fits."""
        cap = min(self.sbank // k, self.scr // (self.d + 2))
        if cap < 1:
            raise ValueError(f"in = {k}: one weight tile does not fit a SPAD_B bank")
        return max(n for n in range(1, cap + 1) if nt % n == 0)


def acc(w):
    return laddr(MEM_ACC, w)


def quant_act(dl, lay, x, n, tmp, a_strip=0):
    """x: ACC word of an fp32 vector of n elements (n % D == 0) -> the int8 A
    strip (D identical rows, SPAD_A words a_strip .. a_strip + n - 1) and its
    scale. tmp: 3 ACC scratch words (amax, s_x, 1 / s_x). Returns the ACC word
    of s_x (broadcast over the lanes)."""
    d = lay.d
    dl.ve(acc(x), 0, acc(tmp), n, COPY, T_FF, fp=True, func="abs", reduce="max")            # amax
    dl.ve(acc(tmp), 0, acc(tmp + 1), d, COPY, T_FF, fp=True, A=INV127)                     # s_x
    dl.ve(acc(tmp), 0, acc(tmp + 2), d, COPY, T_FF, fp=True, A=INV127, func="recip")       # 1 / s_x
    dl.ve(acc(x), acc(tmp + 2), laddr(MEM_SPAD_A, a_strip), n * d, MUL, F32 | I8 << 2, fp=True,
          m1="div", p1=d, m2="div", period=n)                                              # xq, replicated
    return tmp + 1


def linear(dl, lay, k, n_out, w_off, sw_off, s_x, out=None, out_ddr=None, wbase=0, iobase=1, a_strip=0,
           loop=None, params=(0, 1)):
    """y = dequant(A strip x W^T): k inputs (the A strip at SPAD_A a_strip),
    n_out outputs; w_off / sw_off: offsets of the packed weights and of s_w
    (relocated by BASE wbase); s_x: ACC word of the activation scale.
    Output: ACC words from `out` (fp32, n_out / D words) or DDR offset
    out_ddr (relocated by BASE iobase). loop: use LOOP_END (default: when the
    output goes to DDR and there are more than 4 chunks); PARAM params[0] /
    params[1] are overwritten then. Returns the chunk size in tiles."""
    d, sb, cb = lay.d, lay.sbank, lay.cbank
    if k % d or n_out % d or (out is None) == (out_ddr is None):
        raise ValueError("linear: k, n_out multiples of D; exactly one of out / out_ddr")
    nt = n_out // d
    nc = lay.chunk_tiles(k, nt)
    nch = nt // nc
    wbytes, fbytes = nc * k * d, 4 * nc * d                         # per chunk: weights, fp32 vector
    o_sw, o_y = d * nc, d * nc + nc                                # scratch offsets in the ACC bank
    use_loop = (out_ddr is not None and nch > 4) if loop is None else loop
    if use_loop and out_ddr is None:
        raise ValueError("linear: the looped form writes to DDR")
    p_w, p_f = params

    def load(j, dyn):
        p = j & 1
        dl.ld(w_off + j * wbytes, laddr(MEM_SPAD_B, p * sb), nc, k * d, k * d, base=wbase,
              dyn=[("ddr", p_w, True)] if dyn else ())
        dl.ld(sw_off + j * fbytes, acc(p * cb + o_sw), 1, fbytes, fbytes, base=wbase,
              dyn=[("ddr", p_f, True)] if dyn else ())

    def chunk(j, dyn, prefetch):
        p = j & 1
        c = p * cb
        dl.ex(a_strip, p * sb, c, k // d, repeat=nc, bstep=k, cstep=1, crow=nc)
        if prefetch:
            load(j + 1, dyn)
        y = c + o_y if out is None else out + j * nc
        dl.ve(acc(c), acc(c + o_sw), acc(y), nc * d, MUL, I32 | F32 << 2, fp=True, t2=F32)   # * s_w
        dl.ve(acc(y), acc(s_x), acc(y), nc * d, MUL, T_FF, fp=True, m2="div", period=nc)     # * s_x
        if out_ddr is not None:
            dl.st(out_ddr + j * fbytes, acc(y), 1, fbytes, fbytes, base=iobase,
                  dyn=[("ddr", p_f, True)] if dyn else ())

    load(0, False)
    j = 0
    if use_loop:
        pairs = (nch - 1) // 2                                     # the prefetch of the last pair stays in range
        if pairs:
            dl.setreg((DescList.REG_PARAM + p_w, 0), (DescList.REG_PARAM + p_f, 0))
            start = len(dl)
            chunk(0, True, True)
            chunk(1, True, True)
            dl.loop_end(start - len(dl), pairs, k1=p_w, s1=2 * wbytes, k2=p_f, s2=2 * fbytes)
            j = 2 * pairs
    for jj in range(j, nch):
        chunk(jj, False, jj + 1 < nch)
    return nc


def linear_layer(dl, lay, x, k, n_out, w_off, sw_off, tmp, **kw):
    """quant_act + linear: x (ACC, fp32, k elements) -> y. Returns the chunk size."""
    s_x = quant_act(dl, lay, x, k, tmp)
    return linear(dl, lay, k, n_out, w_off, sw_off, s_x, **kw)

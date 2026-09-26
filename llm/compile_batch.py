"""L5b: batched decode of D sequences with one static list (docs/llm_inference_plan.md §11.1).

The D rows of the matrix engine's A strip carry D different sequences
(instead of D copies of one): every weight byte streamed from DDR serves
D tokens. Each sequence has its own position, KV cache and activations;
its results are bit-exact with a single-sequence run (DeviceModel):

- activations are D x n row-major fp32 matrices in ACC (row b =
  sequence b); element-wise VE steps just run D times longer, per-sequence
  reductions use REDUCE with ROWLEN = n / D (the single-sequence order),
  per-sequence scalars broadcast with DIV, per-channel vectors with MOD;
- the int8 A strip (word kb * D + b = chunk kb of sequence b) is the
  quantized matrix sent through DDR: ST rows, FENCE, LD INTERLEAVE;
- GEMMs write C straight into the output matrices (EX C row stride =
  the matrix row), chunk boundaries aligned to the q / k / v and h1 / h3
  segments, which become separate matrices; dequantization runs once per
  linear over the whole matrix; the classifier writes a D x vocab logits
  matrix to DDR (strided ST);
- attention runs per sequence (unrolled over the sequences, a head loop
  each), with that sequence's PARAMs loaded from its argument-table entry.

Argument table (io area, 32 bytes per sequence, written by the ARM per
step): pos, token, pos_pad, pos_pad / D, pos_pad * head_size,
pos_pad * D, pos + 1, 0. KV region: sequence b's cache at b * kv_bytes(cfg).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compile_layer as CL  # noqa: E402
import compile_model as CM  # noqa: E402
from compile_layer import F32, I8, I32, T_FF, acc  # noqa: E402
from compile_model import ADD, COPY, MUL, NEG0  # noqa: E402
from pynq_matmul import LD_INTERLEAVE, MEM_SPAD_A, MEM_SPAD_B, DescList, laddr  # noqa: E402

A_POS, A_TOKEN, A_PP, A_TILES, A_KTLEN, A_PLEN, A_VALID = range(7)     # argument-table words
ARG_BYTES = 32
IO_XQ = 0x400                                   # xq matrix on its way to the A strip
IO_X = 0x4000                                   # x matrix dump (tests)
IO_LOGITS = 0x10000                             # D x vocab fp32


def arg_table(poses, tokens, d, head_size):
    """The argument table for one step (poses / tokens of the D sequences)."""
    t = np.zeros((len(poses), ARG_BYTES // 4), "<u4")
    for b, (pos, tok) in enumerate(zip(poses, tokens)):
        pp = (pos // d + 1) * d
        t[b, :7] = [pos, tok, pp, pp // d, pp * head_size, pp * d, pos + 1]
    return t.tobytes()


def io_bytes(cfg, d):
    return IO_LOGITS + 4 * d * cfg.vocab


class BatchCompiler(CM.ModelCompiler):
    """D sequences per step; the regions as ModelCompiler (the KV region holds
    D caches, the io area needs io_bytes())."""

    def _alloc(self):
        c, d, lay = self.cfg, self.d, self.lay
        n = lambda elems: elems                                      # a D x elems matrix = elems words
        banks = [[lay.acc0, lay.cbank], [lay.acc1, 2 * lay.cbank]]  # [next free, end] per ACC bank

        def take(words, bank):
            a = banks[bank][0]
            banks[bank][0] += words
            if banks[bank][0] > banks[bank][1]:
                raise ValueError(f"batched ACC plan does not fit bank {bank} (D = {d})")
            return a
        # bank 0: persistent
        self.X, self.XN, self.Y = take(n(c.dim), 0), take(n(c.dim), 0), take(n(c.dim), 0)
        self.G = take(c.dim // d, 0)
        self.SW = take(max(c.dim + 2 * c.kv_dim, 2 * c.hidden, c.dim) // d, 0)
        self.AM, self.SX, self.INV, self.SS, self.R = (take(d, 0) for _ in range(5))
        self.TMP, self.S = take(8, 0), take(8, 0)
        self.rmax = c.seq_len // d
        self.SCF, self.E = take(self.rmax, 0), take(self.rmax, 0)
        self.SC = take(d * self.rmax, 0)
        self.OC = take(d * (c.head_size // d), 0)
        # bank 1: the attention-phase buffers and the FFN-phase buffers overlay each other
        base = banks[1][0]
        self.Q, self.K, self.V = take(n(c.dim), 1), take(n(c.kv_dim), 1), take(n(c.kv_dim), 1)
        self.T1, self.T2 = take(n(c.dim), 1), take(n(c.dim), 1)
        self.COS, self.SIN = take(n(c.dim), 1), take(n(c.dim), 1)
        self.ATT = take(n(c.dim), 1)
        end_a = banks[1][0]
        banks[1][0] = base
        self.H1, self.H3, self.U = take(n(c.hidden), 1), take(n(c.hidden), 1), take(n(c.hidden), 1)
        self.acc_used = (banks[0][0] - lay.acc0) + max(end_a, banks[1][0]) - lay.acc1
        sb = lay.sbank
        # SPAD_A: A strips at 0 (bank 0); bank 1: raw K_h, staging rows (embedding, xq, K / V int8)
        self.KRAW = sb
        self.EMBS = sb + c.seq_len * (c.head_size // d)
        self.XQS = self.EMBS + c.dim
        self.KSTM = self.XQS + max(c.dim, c.hidden)
        self.VSTM = self.KSTM + c.kv_dim
        if self.VSTM + c.kv_dim > 2 * sb or c.hidden > sb:
            raise ValueError("batched SPAD_A plan does not fit")
        self.KT, self.VB = 0, sb

    # ------------------------------------------------------------ helpers
    def arg(self, b, word):
        return self.rio(ARG_BYTES * b + 4 * word)

    def seq_kv(self, b, layer):
        koff, voff = self.kv_offsets(layer)
        base = b * CM.kv_bytes(self.cfg)
        return base + koff, base + voff

    # ------------------------------------------------------------ pieces
    def embed(self, dl):
        c, m, d = self.cfg, self.m, self.d
        for b in range(d):
            row = self.EMBS + b * (c.dim // d)
            dl.ldparam(self.arg(b, A_TOKEN), CM.P_SCR, mul=c.dim, base=self.rio.base)
            dl.ld(self.rm(m.offset("emb_q")), laddr(MEM_SPAD_A, row), 1, c.dim, c.dim, base=self.rm.base,
                  dyn=[("ddr", CM.P_SCR, True)])
            dl.ldparam(self.arg(b, A_TOKEN), CM.P_SCR, mul=4, base=self.rio.base)
            dl.ldparam(self.rm(m.offset("emb_s")), CM.P_SCR, base=self.rm.base, dyn=[("addr", CM.P_SCR, True)])
            dl.ve(laddr(MEM_SPAD_A, row), 0, acc(self.X + b * (c.dim // d)), c.dim, COPY, I8 | F32 << 2, fp=True,
                  B=NEG0, dyn=[("A", CM.P_SCR)])

    def rmsnorm(self, dl, x, g_name, layer, dst):
        c, d = self.cfg, self.d
        rw = c.dim // d
        dl.ld(self.rm(self.m.offset(g_name, layer)), acc(self.G), 1, 4 * c.dim, 4 * c.dim, base=self.rm.base)
        dl.ve(acc(x), acc(x), acc(self.SS), d * c.dim, MUL, T_FF, fp=True, reduce="sum", rowlen=rw, B=NEG0)
        dl.ve(acc(self.SS), 0, acc(self.R), d * d, COPY, T_FF, fp=True, func="rsqrt", A=1.0 / c.dim, B=1e-5)
        dl.ve(acc(x), acc(self.R), acc(dst), d * c.dim, MUL, T_FF, fp=True, m2="div", period=rw, B=NEG0)
        dl.ve(acc(dst), acc(self.G), acc(dst), d * c.dim, MUL, T_FF, fp=True, m2="mod", period=rw, B=NEG0)

    def quant(self, dl, x, k):
        """x: D x k fp32 matrix -> the int8 A strip (SPAD_A 0) of the D sequences;
        s_x of sequence b at SX + b."""
        d = self.d
        rw = k // d
        dl.ve(acc(x), 0, acc(self.AM), d * k, COPY, T_FF, fp=True, func="abs", reduce="max", rowlen=rw)
        dl.ve(acc(self.AM), 0, acc(self.SX), d * d, COPY, T_FF, fp=True, A=CL.INV127)
        dl.ve(acc(self.AM), 0, acc(self.INV), d * d, COPY, T_FF, fp=True, A=CL.INV127, func="recip")
        dl.ve(acc(x), acc(self.INV), laddr(MEM_SPAD_A, self.XQS), d * k, MUL, F32 | I8 << 2, fp=True,
              m2="div", period=rw)
        dl.st(self.rio(IO_XQ), laddr(MEM_SPAD_A, self.XQS), d, k, k, base=self.rio.base)
        dl.fence()
        dl.ld(self.rio(IO_XQ), laddr(MEM_SPAD_A, 0), d, k, k, LD_INTERLEAVE, base=self.rio.base)

    def linear(self, dl, x, k, name, layer, segs=None, **kw):
        """segs: [(tiles, ACC word of the output matrix)] covering the outputs in
        order (default one segment into Y); None + out_ddr: the classifier."""
        c, d, lay, m = self.cfg, self.d, self.lay, self.m
        self.quant(dl, x, k)
        if segs is None:                                            # classifier: chunked, DDR logits
            return CL.linear(dl, lay, k, self._nout(name), self.rm(m.offset(name, layer)),
                             self.rm(m.offset("s_" + name, layer)), self.SX, wbase=self.rm.base,
                             iobase=self.rio.base, params=(CM.P_L0, CM.P_L1), rows=d, **kw)
        n_out = self._nout(name)
        nt = n_out // d
        sb = lay.sbank
        g = 0
        for t, _ in segs:
            g = np.gcd(g, t)
        nc = max(x_ for x_ in range(1, min(sb // k, g) + 1) if g % x_ == 0)
        wbytes = nc * k * d
        w_off = self.rm(m.offset(name, layer))
        dl.ld(self.rm(m.offset("s_" + name, layer)), acc(self.SW), 1, 4 * n_out, 4 * n_out, base=self.rm.base)
        where = []                                                  # per chunk: (C word, row stride)
        t0 = 0
        for tiles, base in segs:
            for j in range(tiles // nc):
                where.append((base + j * nc, tiles))
            t0 += tiles
        assert t0 == nt
        load = lambda j: dl.ld(w_off + j * wbytes, laddr(MEM_SPAD_B, (j & 1) * sb), nc, k * d, k * d,
                               base=self.rm.base)
        load(0)
        for j, (cw, stride) in enumerate(where):
            dl.ex(0, (j & 1) * sb, cw, k // d, repeat=nc, bstep=k, cstep=1, crow=stride)
            if j + 1 < len(where):
                load(j + 1)
        sw = self.SW
        for tiles, base in segs:                                    # dequantize each output matrix
            dl.ve(acc(base), acc(sw), acc(base), d * tiles * d, MUL, I32 | F32 << 2, fp=True, t2=F32,
                  m2="mod", period=tiles)
            dl.ve(acc(base), acc(self.SX), acc(base), d * tiles * d, MUL, T_FF, fp=True, m2="div", period=tiles)
            sw += tiles
        return nc

    def rope(self, dl):
        c, m, d = self.cfg, self.m, self.d
        rb, rw = 4 * c.dim, c.dim // d
        for b in range(d):
            dl.ldparam(self.arg(b, A_POS), CM.P_SCR, mul=rb, base=self.rio.base)
            for tab, dst in (("rope_cos", self.COS), ("rope_sin", self.SIN)):
                dl.ld(self.rm(m.offset(tab)), acc(dst + b * rw), 1, rb, rb, base=self.rm.base,
                      dyn=[("ddr", CM.P_SCR, True)])
        n = d * c.dim
        for v in (self.Q, self.K):
            dl.ve(acc(v), acc(self.COS), acc(self.T1), n, MUL, T_FF, fp=True, B=NEG0)
            dl.ve(acc(v), acc(self.SIN), acc(self.T2), n, MUL, T_FF, fp=True, swapneg=True, B=NEG0)
            dl.ve(acc(self.T1), acc(self.T2), acc(v), n, ADD, T_FF, fp=True, B=NEG0)

    def kv_append(self, dl, layer):
        c, d = self.cfg, self.d
        kvp = self.m.get("kvp", layer)
        rw = c.kv_dim // d
        for src, inv, st in ((self.K, kvp[2], self.KSTM), (self.V, kvp[3], self.VSTM)):
            dl.ve(acc(src), 0, laddr(MEM_SPAD_A, st), d * c.kv_dim, MUL, F32 | I8 << 2, fp=True, m2="imm",
                  imm=float(inv), B=NEG0)
        for b in range(d):
            koff, voff = self.seq_kv(b, layer)
            dl.ldparam(self.arg(b, A_POS), CM.P_SCR, mul=c.kv_dim, base=self.rio.base)
            for st, off in ((self.KSTM, koff), (self.VSTM, voff)):
                dl.st(self.rkv(off), laddr(MEM_SPAD_A, st + b * rw), 1, c.kv_dim, c.kv_dim, base=self.rkv.base,
                      dyn=[("ddr", CM.P_SCR, True)])

    def attention_all(self, dl, layer):
        c, d = self.cfg, self.d
        rw = c.dim // d
        dl.fence()                                                  # the K / V rows of this step are in DDR
        for b in range(d):
            for p, word in ((CM.P_VALID, A_VALID), (CM.P_POSPAD, A_PP), (CM.P_TILES, A_TILES),
                            (CM.P_KTLEN, A_KTLEN), (CM.P_PLEN, A_PLEN)):
                dl.ldparam(self.arg(b, word), p, base=self.rio.base)
            self.attention(dl, layer, q=self.Q + b * rw, att=self.ATT + b * rw, kv=self.seq_kv(b, layer),
                           fence=False)

    def silu_mul(self, dl):
        c, d = self.cfg, self.d
        n = d * c.hidden
        dl.ve(acc(self.H1), 0, acc(self.U), n, COPY, T_FF, fp=True, A=-1.0, func="exp", B=NEG0)
        dl.ve(acc(self.U), 0, acc(self.U), n, COPY, T_FF, fp=True, B=1.0, func="recip")
        dl.ve(acc(self.H1), acc(self.U), acc(self.U), n, MUL, T_FF, fp=True, B=NEG0)
        dl.ve(acc(self.U), acc(self.H3), acc(self.U), n, MUL, T_FF, fp=True, B=NEG0)

    def residual(self, dl):
        dl.ve(acc(self.X), acc(self.Y), acc(self.X), self.d * self.cfg.dim, ADD, T_FF, fp=True, B=NEG0)

    def layer(self, dl, l):
        c, d = self.cfg, self.d
        t = lambda n: n // d
        self.rmsnorm(dl, self.X, "rms_att", l, self.XN)
        self.linear(dl, self.XN, c.dim, "wqkv", l, segs=[(t(c.dim), self.Q), (t(c.kv_dim), self.K),
                                                          (t(c.kv_dim), self.V)])
        self.rope(dl)
        self.kv_append(dl, l)
        self.attention_all(dl, l)
        self.linear(dl, self.ATT, c.dim, "wo", l, segs=[(t(c.dim), self.Y)])
        self.residual(dl)
        self.rmsnorm(dl, self.X, "rms_ffn", l, self.XN)
        self.linear(dl, self.XN, c.dim, "w13", l, segs=[(t(c.hidden), self.H1), (t(c.hidden), self.H3)])
        self.silu_mul(dl)
        self.linear(dl, self.U, c.hidden, "w2", l, segs=[(t(c.dim), self.Y)])
        self.residual(dl)

    def final(self, dl):
        self.rmsnorm(dl, self.X, "rms_final", None, self.XN)
        self.linear(dl, self.XN, self.cfg.dim, "wcls", None, out_ddr=self.rio(IO_LOGITS))

    def build(self, layers=None, logits=True, dump_x=False, prologue=None):
        c = self.cfg
        dl = DescList()
        self.embed(dl)
        for l in (range(c.layers) if layers is None else layers):
            self.layer(dl, l)
        if dump_x:
            dl.st(self.rio(IO_X), acc(self.X), 1, 4 * self.d * c.dim, 4 * self.d * c.dim, base=self.rio.base)
        if logits:
            self.final(dl)
        return dl.end(0x5B)

"""L4: the whole decoder as one static descriptor list (docs/llm_inference_plan.md §8).

build_token_list() emits, for a .w8a8 model: embedding -> layers ->
final RMSNorm -> classifier (logits to DDR). The list is static: per
token, only the ARM's parameter block (PARAM0..7 of the ring entry) and a
two-word argument block in DDR (pos, token) change. Everything else the
device derives itself (LDPARAM: KV row, RoPE row, embedding row / scale).

Numerics: step for step DeviceModel.forward (llm/ref_model.py) with the
bit-exact SFU, so the logits are bit-exact with DeviceModel(sfu=SfuExact).

Regions (DDR address = BASE[index] + add + offset; add is 0 on the board,
the absolute address in co-simulation where the BASE registers stay 0):
    model  the .w8a8 file
    io     argument block (pos, token) at 0, x dump at IO_X, logits at IO_LOGITS
    kv     the int8 KV cache: layer l, K at l * 2 * S * kv_dim, V after it
           (S = seq_len rows of kv_dim bytes; rows > pos must be zero:
           the host clears the cache when a sequence starts)

PARAM (plan §8.5, as implemented):
    P0 pos + 1 (VALID)          P1 pos_pad = ceil((pos + 1) / D) * D
    P2 pos_pad / D              P3 scratch (LDPARAM results)
    P4 pos_pad * head_size      P5 pos_pad * D
    P6, P7 head loop (K / V byte offset, q / att word offset); the
    classifier's chunk loop reuses them
token_params() gives P0..P7 for a position.

Attention per head (a LOOP_END over the heads):
    LD K_h rows [0, pos_pad) -> SPAD_A, TRANSPOSE -> K^T B strips (SPAD_B)
    q_h: amax, s_q, 1 / s_q, int8 A strip (replicated)
    EX scores = Q_rep K^T;  VE * s_q * a_k -> fp32
    softmax (VALID = pos + 1): max, exp(s - m), sum, recip, (e * r) * 127 -> int8 A strip
    LD V_h (INTERLEAVE) -> B strips;  EX P_rep V;  VE * a_v -> att[h]
Per-layer constants (1/s_k, 1/s_v, a_k, a_v, weight offsets) are
immediates: the layers are emitted one after another (a LOOP over the
layers would load them with LDPARAM into dynamic fields; see plan §8.4).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compile_layer as CL  # noqa: E402
from compile_layer import F32, I8, I32, T_FF, acc  # noqa: E402
from pynq_matmul import LD_INTERLEAVE, MEM_SPAD_A, MEM_SPAD_B, VOPS, DescList, laddr  # noqa: E402

ADD, SUB, MUL, COPY = VOPS["add"], VOPS["sub"], VOPS["mul"], VOPS["copy"]
P_VALID, P_POSPAD, P_TILES, P_SCR, P_KTLEN, P_PLEN, P_L0, P_L1 = range(8)
ARG_POS, ARG_TOKEN = 0, 4
IO_X, IO_LOGITS = 0x100, 0x10000
NEG0 = -0.0                     # B = -0: y + (-0) = y for every y (keeps the sign of zero)


def token_params(pos, d, head_size):
    pp = (pos // d + 1) * d
    return [pos + 1, pp, pp // d, 0, pp * head_size, pp * d, 0, 0]


def arg_block(pos, token):
    return np.array([pos, token], "<u4").tobytes()


def kv_bytes(cfg):
    return cfg.layers * 2 * cfg.seq_len * cfg.kv_dim


class Region:
    """(BASE index, add): DDR field = add + offset, relocated by BASE[index]."""

    def __init__(self, base, add=0):
        self.base, self.add = base, add

    def __call__(self, off):
        return self.add + off


class ModelCompiler:
    def __init__(self, m, d=None, model=Region(0), io=Region(1), kv=Region(2), prefetch=True, fuse_silu=True):
        """m: export_w8a8.W8A8. Regions: where the model file, the io area and
        the KV cache are (BASE index + fixed add). prefetch: load the first
        weight chunk of every linear layer while the vector steps before it run
        (right after the previous layer's last EX, into the SPAD_B bank that EX
        does not use; attention then keeps to the other bank). fuse_silu: compute
        W13 in h1 / h3 chunk pairs and the SiLU of each pair right after it, so it
        overlaps the next chunk's EX (the FFN activations live in the ACC bank
        the following chunks' scratch does not use)."""
        self.prefetch, self.fuse_silu = prefetch, fuse_silu
        self._seq, self._li, self._pre_bank = [], 0, None
        self.m, self.cfg = m, m.cfg
        c = self.cfg
        self.d = d or m.d
        if self.d != m.d:
            raise ValueError("the .w8a8 packing D differs")
        if c.kv_dim != c.dim or c.head_size % self.d or c.seq_len % self.d:
            raise ValueError("needs kv_dim == dim and head_size, seq_len multiples of D")
        self.lay = CL.Layout(self.d)
        self.R = Region
        self.rm, self.rio, self.rkv = model, io, kv
        self._alloc()

    # ------------------------------------------------------------ memory plan
    def _alloc(self):
        c, d, lay = self.cfg, self.d, self.lay
        w = lambda n: n // d
        nxt = [lay.acc0]

        def take(words):
            a = nxt[0]
            nxt[0] += words
            return a
        self.X, self.XN, self.Y = take(w(c.dim)), take(w(c.dim)), take(w(c.dim))
        self.QKV = take(w(c.dim + 2 * c.kv_dim))
        self.ATT = take(w(c.dim))
        self.H13 = take(w(2 * c.hidden))
        self.U = take(w(c.hidden))
        self.T1, self.T2 = take(w(2 * c.dim)), take(w(2 * c.dim))
        self.G, self.COS, self.SIN = take(w(c.dim)), take(w(c.dim)), take(w(c.dim))
        self.TMP = take(8)                                          # quant_act: amax, s_x, 1/s_x
        self.S = take(8)                                            # softmax: max, sum, recip; rmsnorm
        self.rmax = c.seq_len // d                                  # score words at most
        self.SCF, self.E = take(self.rmax), take(self.rmax)
        self.SC = take(d * self.rmax)                               # EX scores (D rows, crow = rmax)
        self.OC = take(d * (c.head_size // d))                      # EX att_h (D rows)
        if nxt[0] > lay.cbank:
            raise ValueError(f"ACC plan needs {nxt[0]} words, bank 0 has {lay.cbank}")
        self.acc_used = nxt[0]
        # second copies of the FFN activations in ACC bank 1 (SiLU fused into W13)
        self.H13b, self.Ub = lay.acc1, lay.acc1 + w(2 * c.hidden)
        if self.Ub + w(c.hidden) > 2 * lay.cbank:
            raise ValueError("ACC bank 1 plan does not fit")
        sb = lay.sbank
        # SPAD_A: A strips at 0 (bank 0); raw K_h, embedding row, KV staging in bank 1
        self.KRAW = sb
        self.EMB = sb + c.seq_len * (c.head_size // d)
        self.KST, self.VST = self.EMB + w(c.dim), self.EMB + 2 * w(c.dim)
        if self.VST + w(c.dim) > 2 * sb or c.hidden > sb:
            raise ValueError("SPAD_A plan does not fit")
        # SPAD_B: K^T strips (bank 0), V strips (bank 1); with prefetching both in the
        # bank the prefetched weights do not use (attention_banks())
        self.KT, self.VB = 0, sb
        if 2 * c.seq_len * (c.head_size // d) > sb:
            raise ValueError("K^T and V strips do not fit one SPAD_B bank")

    # ------------------------------------------------------------ pieces
    def rmsnorm(self, dl, x, g_name, layer, dst):
        """dst = (x * rsqrt(sum(x^2) / dim + 1e-5)) * g, as DeviceModel.rmsnorm."""
        c, d = self.cfg, self.d
        dl.ld(self.rm(self.m.offset(g_name, layer)), acc(self.G), 1, 4 * c.dim, 4 * c.dim, base=self.rm.base)
        dl.ve(acc(x), acc(x), acc(self.S), c.dim, MUL, T_FF, fp=True, reduce="sum", B=NEG0)        # sum x^2
        dl.ve(acc(self.S), 0, acc(self.S + 1), d, COPY, T_FF, fp=True, func="rsqrt",
              A=1.0 / c.dim, B=1e-5)                                                            # r
        dl.ve(acc(x), acc(self.S + 1), acc(dst), c.dim, MUL, T_FF, fp=True, m2="div",
              period=c.dim // d, B=NEG0)                                                         # x * r
        dl.ve(acc(dst), acc(self.G), acc(dst), c.dim, MUL, T_FF, fp=True, B=NEG0)                # * g

    def linear(self, dl, x, k, name, layer, **kw):
        s_x = CL.quant_act(dl, self.lay, x, k, self.TMP)
        bank0, pre = (self._pre_bank, True) if self._pre_bank is not None else (0, False)
        self._li += 1
        nxt = self._seq[self._li] if self._li < len(self._seq) else None
        self._pre_bank = None

        def after(free):                                # the next layer's chunk 0 -> the free bank
            if nxt is not None:
                self._prefetch(dl, nxt, free)
        return CL.linear(dl, self.lay, k, self._nout(name), self.rm(self.m.offset(name, layer)),
                         self.rm(self.m.offset("s_" + name, layer)), s_x, wbase=self.rm.base,
                         iobase=self.rio.base, params=(P_L0, P_L1), bank0=bank0, preloaded=pre,
                         after_last_ex=after if self.prefetch else None, **kw)

    def _prefetch(self, dl, lin, bank):
        name, layer, k = lin
        CL.prefetch(dl, self.lay, k, self._nout(name), self.rm(self.m.offset(name, layer)), bank,
                    wbase=self.rm.base)
        self._pre_bank = bank

    def attention_banks(self):
        """(K^T word, V word) in SPAD_B: both in the bank the prefetched weights
        of the next linear layer do not use."""
        if self._pre_bank is None:
            return self.KT, self.VB
        b = (1 - self._pre_bank) * self.lay.sbank
        return b, b + self.cfg.seq_len * (self.cfg.head_size // self.d)

    def _nout(self, name):
        c = self.cfg
        return {"wqkv": c.dim + 2 * c.kv_dim, "wo": c.dim, "w13": 2 * c.hidden, "w2": c.dim,
                "wcls": c.vocab}[name]

    def embed(self, dl):
        """X = float(emb_q[token]) * emb_s[token]."""
        c, m = self.cfg, self.m
        dl.ldparam(self.rio(ARG_TOKEN), P_SCR, mul=c.dim, base=self.rio.base)
        dl.ld(self.rm(m.offset("emb_q")), laddr(MEM_SPAD_A, self.EMB), 1, c.dim, c.dim, base=self.rm.base,
              dyn=[("ddr", P_SCR, True)])
        dl.ldparam(self.rio(ARG_TOKEN), P_SCR, mul=4, base=self.rio.base)
        dl.ldparam(self.rm(m.offset("emb_s")), P_SCR, base=self.rm.base, dyn=[("addr", P_SCR, True)])
        dl.ve(laddr(MEM_SPAD_A, self.EMB), 0, acc(self.X), c.dim, COPY, I8 | F32 << 2, fp=True,
              B=NEG0, dyn=[("A", P_SCR)])

    def rope(self, dl):
        """q | k (in QKV) rotated by the cos / sin rows of pos."""
        c, m, d = self.cfg, self.m, self.d
        rb = 4 * c.dim
        dl.ldparam(self.rio(ARG_POS), P_SCR, mul=rb, base=self.rio.base)
        for tab, dst in (("rope_cos", self.COS), ("rope_sin", self.SIN)):
            dl.ld(self.rm(m.offset(tab)), acc(dst), 1, rb, rb, base=self.rm.base, dyn=[("ddr", P_SCR, True)])
        n, per = 2 * c.dim, c.dim // d
        dl.ve(acc(self.QKV), acc(self.COS), acc(self.T1), n, MUL, T_FF, fp=True, m2="mod", period=per, B=NEG0)
        dl.ve(acc(self.QKV), acc(self.SIN), acc(self.T2), n, MUL, T_FF, fp=True, m2="mod", period=per,
              swapneg=True, B=NEG0)
        dl.ve(acc(self.T1), acc(self.T2), acc(self.QKV), n, ADD, T_FF, fp=True, B=NEG0)

    def kv_offsets(self, layer):
        c = self.cfg
        k = layer * 2 * c.seq_len * c.kv_dim
        return k, k + c.seq_len * c.kv_dim

    def kv_append(self, dl, layer):
        c, d = self.cfg, self.d
        kvp = self.m.get("kvp", layer)
        koff, voff = self.kv_offsets(layer)
        dl.ldparam(self.rio(ARG_POS), P_SCR, mul=c.kv_dim, base=self.rio.base)
        for src, inv, st, off in ((self.QKV + c.dim // d, kvp[2], self.KST, koff),
                                  (self.QKV + (c.dim + c.kv_dim) // d, kvp[3], self.VST, voff)):
            dl.ve(acc(src), 0, laddr(MEM_SPAD_A, st), c.kv_dim, MUL, F32 | I8 << 2, fp=True, m2="imm",
                  imm=float(inv), B=NEG0)
            dl.st(self.rkv(off), laddr(MEM_SPAD_A, st), 1, c.kv_dim, c.kv_dim, base=self.rkv.base,
                  dyn=[("ddr", P_SCR, True)])

    def attention(self, dl, layer, q=None, att=None, kv=None, fence=True):
        """Multi-head attention of one sequence: q (ACC word of its q vector,
        default QKV), att (output word, default ATT), kv ((K, V) offsets in the
        KV region, default this layer's); the PARAMs P0..P5 of its position."""
        c, d = self.cfg, self.d
        hs = c.head_size
        hw = hs // d                                    # words per head row
        kvp = self.m.get("kvp", layer)
        a_k, a_v = float(kvp[4]), float(kvp[5])
        koff, voff = self.kv_offsets(layer) if kv is None else kv
        q = self.QKV if q is None else q
        att = self.ATT if att is None else att
        smax = c.seq_len
        kt, vb = self.attention_banks()
        if fence:
            dl.fence()                                  # this token's K / V rows are in DDR
        dl.setreg((DescList.REG_PARAM + P_L0, 0), (DescList.REG_PARAM + P_L1, 0))
        start = len(dl)
        # scores
        dl.ld(self.rkv(koff), laddr(MEM_SPAD_A, self.KRAW), 1, hs, c.kv_dim, base=self.rkv.base,
              dyn=[("ddr", P_L0, True), ("rows", P_POSPAD)])
        dl.transpose(laddr(MEM_SPAD_A, self.KRAW), laddr(MEM_SPAD_B, kt), d * d, I8 | I8 << 2, hw,
                     dyn=[("len", P_KTLEN)])
        dl.ve(acc(q), 0, acc(self.TMP), hs, COPY, T_FF, fp=True, func="abs", reduce="max",
              dyn=[("src1", P_L1, True)])                                                        # amax
        dl.ve(acc(self.TMP), 0, acc(self.TMP + 1), d, COPY, T_FF, fp=True, A=CL.INV127)            # s_q
        dl.ve(acc(self.TMP), 0, acc(self.TMP + 2), d, COPY, T_FF, fp=True, A=CL.INV127, func="recip")
        dl.ve(acc(q), acc(self.TMP + 2), laddr(MEM_SPAD_A, 0), hs * d, MUL, F32 | I8 << 2, fp=True,
              m1="div", p1=d, m2="div", period=hs, dyn=[("src1", P_L1, True)])                  # q_h strip
        dl.ex(0, kt, self.SC, hw, repeat=1, bstep=hs, cstep=1, crow=self.rmax, dyn=[("repeat", P_TILES)])
        dl.ve(acc(self.SC), acc(self.TMP + 1), acc(self.SCF), d, MUL, I32 | F32 << 2, fp=True, t2=F32,
              m2="div", period=self.rmax, A=a_k, B=NEG0, dyn=[("len", P_POSPAD)])                 # * s_q * a_k
        # softmax over pos + 1 valid scores
        dl.ve(acc(self.SCF), 0, acc(self.S), d, COPY, T_FF, fp=True, reduce="max",
              dyn=[("len", P_POSPAD), ("valid", P_VALID)])                                      # m
        dl.ve(acc(self.SCF), acc(self.S), acc(self.E), d, SUB, T_FF, fp=True, m2="div", period=self.rmax,
              func="exp", B=NEG0, dyn=[("len", P_POSPAD), ("valid", P_VALID)])                  # e
        dl.ve(acc(self.E), 0, acc(self.S + 1), d, COPY, T_FF, fp=True, reduce="sum", B=NEG0,
              dyn=[("len", P_POSPAD)])                                                           # sum
        dl.ve(acc(self.S + 1), 0, acc(self.S + 2), d, COPY, T_FF, fp=True, func="recip", B=NEG0)  # r
        dl.ve(acc(self.E), acc(self.S + 2), laddr(MEM_SPAD_A, 0), d * d, MUL, F32 | I8 << 2, fp=True,
              m1="div", p1=d, m2="div", period=smax, A=127.0, B=NEG0, dyn=[("len", P_PLEN)])     # p strip
        # att_h = P V
        dl.ld(self.rkv(voff), laddr(MEM_SPAD_B, vb), 1, hs, c.kv_dim, LD_INTERLEAVE, base=self.rkv.base,
              dyn=[("ddr", P_L0, True), ("rows", P_POSPAD)])
        dl.ex(0, vb, self.OC, 1, repeat=hw, bstep=d, cstep=1, crow=hw,
              dyn=[("kt", P_TILES), ("bstep", P_POSPAD)])
        dl.ve(acc(self.OC), 0, acc(att), hs, COPY, I32 | F32 << 2, fp=True, A=a_v, B=NEG0,
              dyn=[("dst", P_L1, True)])
        dl.loop_end(start - len(dl), c.heads, k1=P_L0, s1=hs, k2=P_L1, s2=hw)

    def silu_mul(self, dl, h13=None, u=None, g0=0, groups=None):
        """U = (h1 * recip(1 + exp(-h1))) * h3, for groups [g0, g0 + groups) (default all)."""
        c, d = self.cfg, self.d
        h13 = self.H13 if h13 is None else h13
        u = (self.U if u is None else u) + g0
        n = d * (c.hidden // d if groups is None else groups)
        h1, h3 = h13 + g0, h13 + c.hidden // d + g0
        dl.ve(acc(h1), 0, acc(u), n, COPY, T_FF, fp=True, A=-1.0, func="exp", B=NEG0)
        dl.ve(acc(u), 0, acc(u), n, COPY, T_FF, fp=True, B=1.0, func="recip")
        dl.ve(acc(h1), acc(u), acc(u), n, MUL, T_FF, fp=True, B=NEG0)
        dl.ve(acc(u), acc(h3), acc(u), n, MUL, T_FF, fp=True, B=NEG0)

    def ffn_up(self, dl, l):
        """W13 and SiLU; returns the ACC word of U (the input of W2)."""
        c, d, lay = self.cfg, self.d, self.lay
        if not self.fuse_silu:
            self.linear(dl, self.XN, c.dim, "w13", l, out=self.H13)
            self.silu_mul(dl)
            return self.U
        half = c.hidden // d                                        # tiles of h1 (= of h3)
        cap = min(lay.sbank // c.dim, lay.scr // (d + 2))
        nc = max(x for x in range(1, min(cap, half) + 1) if half % x == 0)
        nh = half // nc
        bank0 = self._pre_bank if self._pre_bank is not None else 0  # chunk 0's SPAD_B / ACC scratch bank
        # the SiLU after each pair overlaps the next (even-position) chunk, whose scratch is in ACC bank0:
        # keep the FFN activations in the other bank
        h13, u = (self.H13b, self.Ub) if bank0 == 0 else (self.H13, self.U)
        order = [j for m in range(nh) for j in (m, nh + m)]

        def after(i, j):
            if j >= nh:                                             # h3 chunk m done (h1 chunk m before it)
                self.silu_mul(dl, h13, u, (j - nh) * nc, nc)
        self.linear(dl, self.XN, c.dim, "w13", l, out=h13, order=order, after_chunk=after, nc=nc)
        return u

    def residual(self, dl):
        dl.ve(acc(self.X), acc(self.Y), acc(self.X), self.cfg.dim, ADD, T_FF, fp=True, B=NEG0)

    def layer(self, dl, l):
        c = self.cfg
        self.rmsnorm(dl, self.X, "rms_att", l, self.XN)
        self.linear(dl, self.XN, c.dim, "wqkv", l, out=self.QKV)
        self.rope(dl)
        self.kv_append(dl, l)
        self.attention(dl, l)
        self.linear(dl, self.ATT, c.dim, "wo", l, out=self.Y)
        self.residual(dl)
        self.rmsnorm(dl, self.X, "rms_ffn", l, self.XN)
        u = self.ffn_up(dl, l)
        self.linear(dl, u, c.hidden, "w2", l, out=self.Y)
        self.residual(dl)

    def final(self, dl):
        self.rmsnorm(dl, self.X, "rms_final", None, self.XN)
        self.linear(dl, self.XN, self.cfg.dim, "wcls", None, out_ddr=self.rio(IO_LOGITS))

    def build(self, layers=None, logits=True, dump_x=False, prologue=None):
        """The token list: [prologue (SETREG P0..P5 for co-simulation)] embed,
        layers (default all), [x -> io IO_X], [final + logits], END."""
        c = self.cfg
        dl = DescList()
        if prologue is not None:
            p = prologue
            dl.setreg(*[(DescList.REG_PARAM + i, p[i]) for i in range(3)])
            dl.setreg(*[(DescList.REG_PARAM + i, p[i]) for i in range(3, 6)])
        lays = list(range(c.layers) if layers is None else layers)
        self._seq = [(n, l, c.hidden if n == "w2" else c.dim) for l in lays for n in ("wqkv", "wo", "w13", "w2")]
        if logits:
            self._seq.append(("wcls", None, c.dim))
        self._li, self._pre_bank = 0, None
        if self.prefetch and self._seq:
            self._prefetch(dl, self._seq[0], 0)             # hidden behind the embedding and the first norm
        self.embed(dl)
        for l in lays:
            self.layer(dl, l)
        if dump_x:
            dl.st(self.rio(IO_X), acc(self.X), 1, 4 * c.dim, 4 * c.dim, base=self.rio.base)
        if logits:
            self.final(dl)
        return dl.end(0x14)

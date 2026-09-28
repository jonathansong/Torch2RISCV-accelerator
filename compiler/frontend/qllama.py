"""Stage C0: the quantized llama as a PyTorch module (docs/iree_compiler_plan.md §3).

QLlama computes what llm/ref_model.py DeviceModel.forward computes, step
for step, with the quantization written out explicitly (the compiler never
quantizes by itself):

- int8 weights with one fp32 scale per output channel (Wq|Wk|Wv and W1|W3
  concatenated by rows as in the .w8a8 file); int8 x int8 -> int32 matmuls
  written as int32 matmuls of sign-extended operands; y = float(acc) * s_w,
  then * s_x (two roundings);
- activations quantized per token: amax = max|x|, s_x = amax * (1/127),
  x_q = clamp(round_half_even(x * (1 / s_x)), -127, 127);
- RMSNorm (sum x^2 * (1/dim) + 1e-5, rsqrt), RoPE in the SWAPNEG form,
  softmax as max / exp / sum / reciprocal / multiply, SiLU as
  h1 * (1 / (1 + exp(-h1))), residuals: the DeviceModel operation order;
- an int8 KV cache with static per-layer scales: module buffers kc, vc of
  (layers * seq_len, kv_dim), layer l's rows at l * seq_len, written in place
  at row l * seq_len + pos with one index_copy_ per layer. (Writing through
  a per-layer view instead, kc[l].index_copy_, exports as index_put plus a
  slice write-back that IREE's CPU backend rejects: "write affecting
  operations on global resources are restricted to workgroup distributed
  contexts".)

decode(token, pos, valid) -> logits
  token: int64[1]; pos: int64[1];
  valid: float32[T], T = pos_pad = ceil((pos + 1) / D) * D, 1.0 for positions
  0..pos and 0.0 after them. Attention reads the first T cache rows; T is a
  dynamic dimension carried by `valid` (an input-backed shape, which exports
  cleanly; the device gets it as a push constant). Only its length is used:
  the attention mask is the prefix arange(T) <= pos (a length, as attention
  kernels take; the sa backend maps it to the VE's VALID count).

Reductions use torch's order here; the device (and DeviceModel) use the
lane order of LLM plan §6.4, so eager results differ from DeviceModel by
rounding only. The compiled program on the accelerator must match
DeviceModel bit for bit (docs/iree_compiler_plan.md §3.2).
"""
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "llm"))
from ref_model import quantize_rows, rope_tables  # noqa: E402

F32 = torch.float32


def to_i8(x):
    """fp32 -> int8: round half to even, saturate to [-127, 127], NaN -> 0 (ref_model.to_i8)."""
    return torch.clamp(torch.round(torch.nan_to_num(x, nan=0.0)), -127.0, 127.0).to(torch.int8)


def qlinear(xq, s_x, wq, s_w):
    """dequant(x_q W_q^T): int8 x int8 -> int32, then * s_w (per channel), * s_x."""
    acc = torch.matmul(xq.to(torch.int32), wq.to(torch.int32).t())
    return (acc.to(F32) * s_w) * s_x


def f32c(v):
    """A constant as the fp32 value DeviceModel uses (F(v), round to nearest),
    written as a Python float that fp32 represents exactly: the export and
    torch-mlir then cannot round it differently (1 / 288 came out one ulp low)."""
    return float(np.float32(v))


def quant_act(x):
    """Per-row (last dim) dynamic int8 quantization: (x_q, s_x)."""
    amax = torch.amax(torch.abs(x), dim=-1, keepdim=True)
    s_x = amax * f32c(1.0 / 127.0)
    inv = torch.reciprocal(s_x)
    return to_i8(x * inv), s_x


class QLlama(torch.nn.Module):
    def __init__(self, cfg, w, kv_scales, d=8, n_layers=None, tap=None):
        """n_layers: return the residual stream x after that many layers instead of
        the logits (DeviceModel.forward(..., n_layers); for localizing differences).
        tap (with n_layers): return this intermediate of the last layer instead
        ("q", "k", "att", "x_att": x after the attention block)."""
        super().__init__()
        self.cfg, self.d = cfg, d
        self.n_layers, self.tap = n_layers, tap
        c = cfg
        if c.kv_heads != c.heads:
            raise ValueError("QLlama: multi-head attention only (kv_heads == heads)")
        L = c.layers
        t = lambda a, dt=None: torch.tensor(np.ascontiguousarray(a), dtype=dt)

        def lin(names, l):
            q, s = quantize_rows(np.concatenate([w[n][l] for n in names]))
            return t(q, torch.int8), t(s, F32)
        emb_q, emb_s = quantize_rows(w["tok"])
        self.emb_q = torch.nn.Parameter(t(emb_q, torch.int8), requires_grad=False)
        self.emb_s = torch.nn.Parameter(t(emb_s, F32), requires_grad=False)
        if w["wcls"] is not w["tok"]:
            cq, cs = quantize_rows(w["wcls"])
            self.cls_q = torch.nn.Parameter(t(cq, torch.int8), requires_grad=False)
            self.cls_s = torch.nn.Parameter(t(cs, F32), requires_grad=False)
        else:                                                     # tied classifier (TinyStories)
            self.cls_q, self.cls_s = self.emb_q, self.emb_s
        P = lambda a: torch.nn.Parameter(a, requires_grad=False)
        self.wqkv = torch.nn.ParameterList()
        self.s_wqkv = torch.nn.ParameterList()
        self.wo, self.s_wo = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.w13, self.s_w13 = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.w2, self.s_w2 = torch.nn.ParameterList(), torch.nn.ParameterList()
        for l in range(L):
            for (qs, ss, names) in ((self.wqkv, self.s_wqkv, ("wq", "wk", "wv")), (self.wo, self.s_wo, ("wo",)),
                                    (self.w13, self.s_w13, ("w1", "w3")), (self.w2, self.s_w2, ("w2",))):
                q, s = lin(names, l)
                qs.append(P(q))
                ss.append(P(s))
        self.rms_att = P(t(w["rms_att"], F32))
        self.rms_ffn = P(t(w["rms_ffn"], F32))
        self.rms_final = P(t(w["rms_final"], F32))
        cos, sin = rope_tables(cfg)
        self.rope_cos = P(t(np.repeat(cos, 2, axis=1), F32))       # (seq, dim): a value per rotation pair
        self.rope_sin = P(t(np.repeat(sin, 2, axis=1), F32))
        kv = np.asarray(kv_scales, np.float32)
        s_k, s_v = kv[:, 0], kv[:, 1]
        # per-layer constants computed as DeviceModel does (python float math, rounded to fp32)
        self.inv_sk = [float(np.float32(1.0 / np.float32(x))) for x in s_k]
        self.inv_sv = [float(np.float32(1.0 / np.float32(x))) for x in s_v]
        self.a_k = [float(np.float32(float(x) / np.sqrt(c.head_size))) for x in s_k]
        self.a_v = [float(np.float32(float(x) / 127.0)) for x in s_v]
        self.register_buffer("kc", torch.zeros(L * c.seq_len, c.kv_dim, dtype=torch.int8))
        self.register_buffer("vc", torch.zeros(L * c.seq_len, c.kv_dim, dtype=torch.int8))

    def kv_cache(self):
        """(K, V) as (layers, seq_len, kv_dim) views."""
        c = self.cfg
        return self.kc.view(c.layers, c.seq_len, c.kv_dim), self.vc.view(c.layers, c.seq_len, c.kv_dim)

    # ------------------------------------------------------------ pieces
    def rmsnorm(self, x, g):
        ss = torch.sum(x * x, dim=-1, keepdim=True)
        r = torch.rsqrt(ss * f32c(1.0 / self.cfg.dim) + f32c(1e-5))
        return (x * r) * g

    @staticmethod
    def swapneg(v):
        """(x0, x1) -> (-x1, x0) for every pair."""
        p = v.reshape(v.shape[:-1] + (-1, 2))
        return torch.stack((-p[..., 1], p[..., 0]), dim=-1).reshape(v.shape)

    def rope(self, v, cos, sin):
        return v * cos + self.swapneg(v) * sin

    def attention(self, l, q, T, pos):
        c = self.cfg
        H, hs = c.heads, c.head_size
        r0 = l * c.seq_len
        kh = self.kc[r0:r0 + T].to(torch.int32).reshape(T, H, hs).permute(1, 0, 2)    # (H, T, hs)
        vh = self.vc[r0:r0 + T].to(torch.int32).reshape(T, H, hs).permute(1, 0, 2)
        qq, s_q = quant_act(q.reshape(H, hs))                                    # (H, hs), (H, 1)
        sc = torch.matmul(kh, qq.to(torch.int32).unsqueeze(-1)).squeeze(-1)     # (H, T) int32
        sc = (sc.to(F32) * s_q) * self.a_k[l]
        # a prefix mask given by the length pos + 1 (as attention kernels take
        # sequence lengths); `valid` only carries the padded length T
        mask = torch.arange(T) <= pos                                            # (T,)
        m = torch.amax(torch.where(mask, sc, torch.tensor(float("-inf"))), dim=-1, keepdim=True)
        e = torch.where(mask, torch.exp(sc - m), torch.tensor(0.0))
        r = torch.reciprocal(torch.sum(e, dim=-1, keepdim=True))
        pq = to_i8((e * r) * 127.0).to(torch.int32)                             # (H, T)
        att = torch.matmul(pq.unsqueeze(1), vh).squeeze(1)                      # (H, hs) int32
        return (att.to(F32) * self.a_v[l]).reshape(c.dim)

    def forward(self, token, pos, valid):
        c = self.cfg
        T = valid.shape[0]
        # gathers written with index_select (a tensor index, no data-dependent scalar)
        x = (self.emb_q.index_select(0, token).to(F32) * self.emb_s.index_select(0, token).unsqueeze(-1))[0]
        cos = self.rope_cos.index_select(0, pos)[0]
        sin = self.rope_sin.index_select(0, pos)[0]
        for l in range(c.layers if self.n_layers is None else self.n_layers):
            xq, s_x = quant_act(self.rmsnorm(x, self.rms_att[l]))
            qkv = qlinear(xq, s_x, self.wqkv[l], self.s_wqkv[l])
            q = self.rope(qkv[:c.dim], cos, sin)
            k = self.rope(qkv[c.dim:c.dim + c.kv_dim], cos, sin)
            v = qkv[c.dim + c.kv_dim:]
            row = pos + l * c.seq_len
            self.kc.index_copy_(0, row, to_i8(k * self.inv_sk[l]).unsqueeze(0))
            self.vc.index_copy_(0, row, to_i8(v * self.inv_sv[l]).unsqueeze(0))
            att = self.attention(l, q, T, pos)
            last = self.n_layers is not None and l == self.n_layers - 1
            if last and self.tap in ("q", "k", "att"):
                return {"q": q, "k": k, "att": att}[self.tap]
            aq, s_a = quant_act(att)
            x = x + qlinear(aq, s_a, self.wo[l], self.s_wo[l])
            if last and self.tap == "x_att":
                return x
            xn = self.rmsnorm(x, self.rms_ffn[l])
            if last and self.tap == "xn_ffn":
                return xn
            xq, s_x = quant_act(xn)
            if last and self.tap == "sx_ffn":
                return s_x.reshape(1) * 1.0
            h13 = qlinear(xq, s_x, self.w13[l], self.s_w13[l])
            h1, h3 = h13[:c.hidden], h13[c.hidden:]
            sig = torch.reciprocal(1.0 + torch.exp(h1 * -1.0))
            u = (h1 * sig) * h3
            if last and self.tap in ("h13", "u"):
                return {"h13": h13, "u": u}[self.tap]
            hq, s_h = quant_act(u)
            x = x + qlinear(hq, s_h, self.w2[l], self.s_w2[l])
        if self.n_layers is not None:
            return x
        xq, s_x = quant_act(self.rmsnorm(x, self.rms_final))
        return qlinear(xq, s_x, self.cls_q, self.cls_s)

    # ------------------------------------------------------------ prefill (plan §8.13)
    def attention_rows(self, l, q, T, posv):
        """attention of M rows q (M, dim) at positions posv (M,): row m sees the
        positions <= posv[m]; each row as attention() computes it."""
        c = self.cfg
        H, hs = c.heads, c.head_size
        M = q.shape[0]
        r0 = l * c.seq_len
        kh = self.kc[r0:r0 + T].to(torch.int32).reshape(T, H, hs).permute(1, 2, 0)    # (H, hs, T)
        vh = self.vc[r0:r0 + T].to(torch.int32).reshape(T, H, hs).permute(1, 0, 2)    # (H, T, hs)
        qq, s_q = quant_act(q.reshape(M, H, hs))                                      # (M, H, hs), (M, H, 1)
        sc = torch.matmul(qq.permute(1, 0, 2).to(torch.int32), kh)                    # (H, M, T) int32
        sc = (sc.to(F32) * s_q.permute(1, 0, 2)) * self.a_k[l]
        mask = torch.arange(T)[None, :] <= posv[:, None]                             # (M, T)
        m = torch.amax(torch.where(mask, sc, torch.tensor(float("-inf"))), dim=-1, keepdim=True)
        e = torch.where(mask, torch.exp(sc - m), torch.tensor(0.0))
        r = torch.reciprocal(torch.sum(e, dim=-1, keepdim=True))
        pq = to_i8((e * r) * 127.0).to(torch.int32)                                 # (H, M, T)
        att = torch.matmul(pq, vh)                                                   # (H, M, hs) int32
        return (att.to(F32) * self.a_v[l]).permute(1, 0, 2).reshape(M, c.dim)

    def prefill(self, tokens, start, valid):
        """A chunk of M prompt tokens at positions start .. start + M - 1 (start: [1]):
        their KV rows written, the logits of the last one returned; valid[T] only
        carries the padded attention length. Each row as forward() computes a
        decode step (chunking: compiler/frontend/export.py prefill_starts)."""
        c = self.cfg
        M, T = tokens.shape[0], valid.shape[0]
        posv = start + torch.arange(M)
        x = self.emb_q.index_select(0, tokens).to(F32) * self.emb_s.index_select(0, tokens).unsqueeze(-1)
        cos = self.rope_cos.index_select(0, posv)                                    # (M, dim)
        sin = self.rope_sin.index_select(0, posv)
        for l in range(c.layers):
            xq, s_x = quant_act(self.rmsnorm(x, self.rms_att[l]))
            qkv = qlinear(xq, s_x, self.wqkv[l], self.s_wqkv[l])
            q = self.rope(qkv[:, :c.dim], cos, sin)
            k = self.rope(qkv[:, c.dim:c.dim + c.kv_dim], cos, sin)
            v = qkv[:, c.dim + c.kv_dim:]
            rows = posv + l * c.seq_len
            self.kc.index_copy_(0, rows, to_i8(k * self.inv_sk[l]))
            self.vc.index_copy_(0, rows, to_i8(v * self.inv_sv[l]))
            att = self.attention_rows(l, q, T, posv)
            aq, s_a = quant_act(att)
            x = x + qlinear(aq, s_a, self.wo[l], self.s_wo[l])
            xq, s_x = quant_act(self.rmsnorm(x, self.rms_ffn[l]))
            h13 = qlinear(xq, s_x, self.w13[l], self.s_w13[l])
            h1, h3 = h13[:, :c.hidden], h13[:, c.hidden:]
            u = (h1 * torch.reciprocal(1.0 + torch.exp(h1 * -1.0))) * h3
            hq, s_h = quant_act(u)
            x = x + qlinear(hq, s_h, self.w2[l], self.s_w2[l])
        xq, s_x = quant_act(self.rmsnorm(x[M - 1], self.rms_final))
        return qlinear(xq, s_x, self.cls_q, self.cls_s)


def step_inputs(token, pos, d, static_len=None):
    """(token, pos, valid) tensors of one decode step. static_len: valid has this
    fixed length (attention over a static number of positions, masked by pos;
    the sa backend's form) instead of pos_pad."""
    pp = static_len or (pos // d + 1) * d
    valid = torch.zeros(pp, dtype=F32)
    valid[:pos + 1] = 1.0
    return torch.tensor([token]), torch.tensor([pos]), valid

#!/usr/bin/env python3
"""Reference models of a llama2.c transformer (docs/llm_inference_plan.md §4.2).

Fp32Model    the fp32 forward pass of llama2.c run.c, in NumPy float32.
DeviceModel  the same network with the device numerics of plan §3:
             - W8A8: int8 weights with one fp32 scale per output channel,
               int8 activations with a dynamic per-token scale, exact int32
               accumulation, fp32 dequantization (x s_w, then x s_x);
             - int8 KV cache with static per-layer scales, int8 q with a
               per-head dynamic scale, int8 softmax probabilities (scale 1/127);
             - fp32 everywhere else, one rounding per VE step, reductions in
               the fixed lane order of plan §6.4 (lane-wise sequential over
               groups of D, then a pairwise tree over the D lanes);
             - EXP / RECIP / RSQRT: float64 results rounded to fp32 for now
               (Sfu64); the functional simulator's bit-exact algorithms
               replace them in L2.

    python3 llm/ref_model.py stories15M.bin tokenizer.bin -i "Once upon a time" -n 256 [--device]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkpoint  # noqa: E402
from tokenizer import BOS, Tokenizer  # noqa: E402

F = np.float32


def rope_tables(cfg):
    """(cos, sin) float32 (seq_len, dim // 2) as run.c computes them:
    freq = 1 / 10000^(head_dim / head_size), val = pos * freq, pairs (i, i+1)."""
    hs = cfg.head_size
    hd = (np.arange(0, cfg.dim, 2) % hs).astype(F)
    freq = F(1.0) / np.power(F(10000.0), hd / F(hs))
    val = np.arange(cfg.seq_len, dtype=F)[:, None] * freq[None, :]
    return np.cos(val).astype(F), np.sin(val).astype(F)


# ============================================================== fp32 model
class Fp32Model:
    def __init__(self, cfg, w):
        self.cfg, self.w = cfg, w
        self.cos, self.sin = rope_tables(cfg)
        self.reset()

    def reset(self):
        c = self.cfg
        self.kc = np.zeros((c.layers, c.seq_len, c.kv_dim), F)
        self.vc = np.zeros((c.layers, c.seq_len, c.kv_dim), F)

    @staticmethod
    def rmsnorm(x, g):
        ss = F(np.dot(x, x)) / F(x.size) + F(1e-5)
        return (g * (F(1.0) / np.sqrt(ss) * x)).astype(F)

    def rope(self, v, pos):
        c, s = self.cos[pos, : v.size // 2], self.sin[pos, : v.size // 2]
        v0, v1 = v[0::2].copy(), v[1::2].copy()
        v[0::2] = v0 * c - v1 * s
        v[1::2] = v0 * s + v1 * c

    def forward(self, token, pos):
        c, w = self.cfg, self.w
        hs, rep = c.head_size, c.heads // c.kv_heads
        x = np.array(w["tok"][token], F)
        for l in range(c.layers):
            xb = self.rmsnorm(x, w["rms_att"][l])
            q = w["wq"][l] @ xb
            k = w["wk"][l] @ xb
            v = w["wv"][l] @ xb
            self.rope(q, pos)
            self.rope(k, pos)
            self.kc[l, pos], self.vc[l, pos] = k, v
            att = np.empty(c.dim, F)
            for h in range(c.heads):
                kh = self.kc[l, : pos + 1, (h // rep) * hs:(h // rep + 1) * hs]
                vh = self.vc[l, : pos + 1, (h // rep) * hs:(h // rep + 1) * hs]
                s = (kh @ q[h * hs:(h + 1) * hs]) / F(np.sqrt(F(hs)))
                e = np.exp(s - s.max())
                att[h * hs:(h + 1) * hs] = (e / e.sum()) @ vh
            x = x + w["wo"][l] @ att
            xb = self.rmsnorm(x, w["rms_ffn"][l])
            h1, h3 = w["w1"][l] @ xb, w["w3"][l] @ xb
            x = x + w["w2"][l] @ ((h1 * (F(1.0) / (F(1.0) + np.exp(-h1)))) * h3)
        x = self.rmsnorm(x, w["rms_final"])
        return (w["wcls"] @ x).astype(F)


# ====================================================== device numerics
class Sfu64:
    """EXP / RECIP / RSQRT as float64 rounded to fp32, with the plan §6.5
    range rules (EXP: x < -87 -> 0, x > 88 -> +inf)."""
    @staticmethod
    def exp(x):
        x = np.asarray(x, F)
        y = np.exp(x.astype(np.float64)).astype(F)
        return np.where(x < -87, F(0), np.where(x > 88, F(np.inf), y)).astype(F)

    @staticmethod
    def recip(x):
        with np.errstate(divide="ignore"):
            return (1.0 / np.asarray(x, np.float64)).astype(F)

    @staticmethod
    def rsqrt(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (1.0 / np.sqrt(np.asarray(x, np.float64))).astype(F)


def to_i8(x):
    """fp32 -> int8: round to nearest even, saturate to [-127, 127], NaN -> 0."""
    y = np.rint(np.asarray(x, F))
    return np.clip(np.nan_to_num(y, nan=0.0, posinf=127, neginf=-127), -127, 127).astype(np.int8)


def quantize_rows(wt):
    """Symmetric int8 per row (output channel): (q int8, s float32)."""
    wt = np.asarray(wt, F)
    s = (np.abs(wt).max(axis=-1) / F(127)).astype(F)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = to_i8(wt / s[..., None])
    return q, s


class DeviceModel:
    def __init__(self, cfg, w, kv_scales, d=8, sfu=Sfu64):
        """kv_scales: (layers, 2) float32 static scales of K and V."""
        self.cfg, self.d, self.sfu = cfg, d, sfu
        self.cos, self.sin = rope_tables(cfg)
        self.kv_scales = np.asarray(kv_scales, F)
        L = cfg.layers
        self.emb_q, self.emb_s = quantize_rows(w["tok"])
        self.lin = {}
        for name in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
            qs = [quantize_rows(w[name][l]) for l in range(L)]
            self.lin[name] = [(q.astype(np.float64), s) for q, s in qs]   # float64: exact int dots
        qc, sc = quantize_rows(w["wcls"])
        self.cls = (qc.astype(np.float64), sc)
        self.rms_att, self.rms_ffn = np.asarray(w["rms_att"], F), np.asarray(w["rms_ffn"], F)
        self.rms_final = np.asarray(w["rms_final"], F)
        self.reset()

    def reset(self):
        c = self.cfg
        self.kc = np.zeros((c.layers, c.seq_len, c.kv_dim), np.int8)
        self.vc = np.zeros((c.layers, c.seq_len, c.kv_dim), np.int8)

    # ---- VE primitives (one fp32 rounding each)
    def _pad(self, v, fill):
        n = -v.size % self.d
        return np.concatenate([v, np.full(n, fill, F)]) if n else v

    def red_sum(self, v):
        g = self._pad(np.asarray(v, F), 0.0).reshape(-1, self.d)
        acc = g[0].copy()
        for row in g[1:]:
            acc = (acc + row).astype(F)
        while acc.size > 1:
            acc = (acc[0::2] + acc[1::2]).astype(F)
        return acc[0]

    def red_max(self, v):
        return F(np.max(np.asarray(v, F)))

    def quant_act(self, x):
        amax = self.red_max(np.abs(x))
        s_x = F(amax * F(1 / 127))
        inv = self.sfu.recip(s_x)
        with np.errstate(invalid="ignore"):
            return to_i8((x * inv).astype(F)), s_x

    def linear(self, xq, s_x, wq_s):
        wq, s_w = wq_s
        acc = (wq @ xq.astype(np.float64)).astype(np.int64)          # exact int32 dot products
        y = (acc.astype(F) * s_w).astype(F)                          # int32 -> fp32 (RNE), x s_w
        return (y * s_x).astype(F)

    def rmsnorm(self, x, g):
        ss = self.red_sum((x * x).astype(F))
        r = self.sfu.rsqrt(F(F(ss * F(1.0 / x.size)) + F(1e-5)))
        return ((x * r).astype(F) * g).astype(F)

    def rope(self, v, pos):
        c = np.repeat(self.cos[pos, : v.size // 2], 2)
        s = np.repeat(self.sin[pos, : v.size // 2], 2)
        sw = np.empty_like(v)
        sw[0::2], sw[1::2] = -v[1::2], v[0::2]                       # SWAPNEG
        return ((v * c).astype(F) + (sw * s).astype(F)).astype(F)

    def forward(self, token, pos):
        c, sfu, d = self.cfg, self.sfu, self.d
        hs, rep = c.head_size, c.heads // c.kv_heads
        x = (self.emb_q[token].astype(F) * self.emb_s[token]).astype(F)
        for l in range(c.layers):
            s_k, s_v = self.kv_scales[l]
            xq, s_x = self.quant_act(self.rmsnorm(x, self.rms_att[l]))
            q = self.rope(self.linear(xq, s_x, self.lin["wq"][l]), pos)
            k = self.rope(self.linear(xq, s_x, self.lin["wk"][l]), pos)
            v = self.linear(xq, s_x, self.lin["wv"][l])
            self.kc[l, pos] = to_i8((k * F(1.0 / s_k)).astype(F))
            self.vc[l, pos] = to_i8((v * F(1.0 / s_v)).astype(F))
            a_k = F(float(s_k) / np.sqrt(hs))
            a_v = F(float(s_v) / 127.0)
            att = np.empty(c.dim, F)
            for h in range(c.heads):
                kh = self.kc[l, : pos + 1, (h // rep) * hs:(h // rep + 1) * hs].astype(np.int64)
                vh = self.vc[l, : pos + 1, (h // rep) * hs:(h // rep + 1) * hs].astype(np.int64)
                qq, s_q = self.quant_act(q[h * hs:(h + 1) * hs])
                sc = ((kh @ qq.astype(np.int64)).astype(F) * s_q).astype(F)
                sc = (sc * a_k).astype(F)
                m = self.red_max(sc)
                e = sfu.exp((sc - m).astype(F))
                p = (e * sfu.recip(self.red_sum(e))).astype(F)
                pq = to_i8((p * F(127)).astype(F)).astype(np.int64)
                att[h * hs:(h + 1) * hs] = ((pq @ vh).astype(F) * a_v).astype(F)
            aq, s_a = self.quant_act(att)
            x = (x + self.linear(aq, s_a, self.lin["wo"][l])).astype(F)
            xq, s_x = self.quant_act(self.rmsnorm(x, self.rms_ffn[l]))
            h1 = self.linear(xq, s_x, self.lin["w1"][l])
            h3 = self.linear(xq, s_x, self.lin["w3"][l])
            sig = sfu.recip(F(1) + sfu.exp((h1 * F(-1)).astype(F)))
            hq, s_h = self.quant_act((((h1 * sig).astype(F)) * h3).astype(F))
            x = (x + self.linear(hq, s_h, self.lin["w2"][l])).astype(F)
        xq, s_x = self.quant_act(self.rmsnorm(x, self.rms_final))
        return self.linear(xq, s_x, self.cls)


# ============================================================ generation
def generate(model, prompt_tokens, steps, forced=None, keep_logits=False):
    """run.c generate() with greedy sampling. forced: a token sequence to
    feed instead of sampling (teacher forcing). Returns (tokens, logits)."""
    model.reset()
    steps = min(steps, model.cfg.seq_len)
    tokens, logits = [prompt_tokens[0]], []
    token = prompt_tokens[0]
    for pos in range(steps):
        lg = model.forward(token, pos)
        if keep_logits:
            logits.append(lg)
        if pos < len(prompt_tokens) - 1:
            nxt = prompt_tokens[pos + 1]
        elif forced is not None:
            if pos + 1 >= len(forced):
                break
            nxt = forced[pos + 1]
        else:
            nxt = int(np.argmax(lg))
        if nxt == BOS:
            break
        tokens.append(nxt)
        token = nxt
    return tokens, logits


def default_kv_scales(cfg, w, tok, prompts=("Once upon a time",), steps=64):
    """Absmax static KV scales from a short fp32 calibration run."""
    m = Fp32Model(cfg, w)
    amax = np.zeros((cfg.layers, 2), F)
    for p in prompts:
        toks, _ = generate(m, tok.encode(p), steps)
        n = len(toks)
        amax[:, 0] = np.maximum(amax[:, 0], np.abs(m.kc[:, :n]).max(axis=(1, 2)))
        amax[:, 1] = np.maximum(amax[:, 1], np.abs(m.vc[:, :n]).max(axis=(1, 2)))
    return amax / F(127)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("tokenizer")
    ap.add_argument("-i", "--prompt", default="Once upon a time")
    ap.add_argument("-n", "--steps", type=int, default=256)
    ap.add_argument("--device", action="store_true", help="device numerics (W8A8, int8 KV)")
    ap.add_argument("--kv-scales", help=".npy of (layers, 2) KV scales (default: absmax calibration)")
    args = ap.parse_args()
    cfg, w = checkpoint.load(args.checkpoint)
    tok = Tokenizer(args.tokenizer, cfg.vocab)
    if args.device:
        kv = np.load(args.kv_scales) if args.kv_scales else default_kv_scales(cfg, w, tok)
        model = DeviceModel(cfg, w, kv)
    else:
        model = Fp32Model(cfg, w)
    t0 = time.time()
    toks, _ = generate(model, tok.encode(args.prompt), args.steps)
    print(tok.decode(toks))
    print(f"{len(toks) - 1} tokens in {time.time() - t0:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()

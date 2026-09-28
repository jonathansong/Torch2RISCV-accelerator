"""Stage C5.5: quantized decoders from HuggingFace checkpoints (docs/iree_compiler_plan.md §8.9).

A llama-family decoder (LlamaForCausalLM, Qwen3ForCausalLM) built from a
HuggingFace directory (config.json + *.safetensors), without transformers:

- HFConfig: the structure (GQA, head_dim, QK-norm, RoPE theta, eps, tied
  embeddings);
- load_weights: fp32 weights; wq / wk (and q_norm / k_norm) permuted per head
  from HF's rotate-half RoPE to the interleaved pairs of llama2.c (as
  llama2.c's export does), so RoPE is v * cos + swapneg(v) * sin as in QLlama;
- Fp32Model: the fp32 reference (HF semantics; torch);
- QModel: the quantized decoder in QLlama's form (compiler/frontend/qllama.py):
  int8 weights with a per-channel scale, per-token int8 activations, int8 KV
  cache with static per-layer scales (calibrated with Fp32Model), the same
  decode(token, pos, valid[T]) interface. GQA is written so that the kv head
  is the batch of the attention matmuls and the G = heads / kv_heads query
  heads of a group are the rows:
    scores[kv, g, t] = sum_j q[kv, g, j] K[t, kv, j]
    out[kv, g, j]    = sum_t p[kv, g, t] V[t, kv, j]
  (one pass over K / V per kv head; multi-head attention is G = 1).

There is no hand-written device model for these networks: the compiled
program is checked per dispatch against its IR (dispatch_check.py) and end to
end against the fp32 model within a tolerance.
"""
import json
import os

import numpy as np
import torch

F32 = torch.float32


def f32c(v):
    return float(np.float32(v))


def to_i8(x):
    return torch.clamp(torch.round(torch.nan_to_num(x, nan=0.0)), -127.0, 127.0).to(torch.int8)


def quant_act(x):
    amax = torch.amax(torch.abs(x), dim=-1, keepdim=True)
    s_x = amax * f32c(1.0 / 127.0)
    inv = torch.reciprocal(s_x)
    return to_i8(x * inv), s_x


def qlinear(xq, s_x, wq, s_w):
    acc = torch.matmul(xq.to(torch.int32), wq.to(torch.int32).t())
    return (acc.to(F32) * s_w) * s_x


def quantize_rows(w):
    """int8 rows with one fp32 scale each (ref_model.quantize_rows)."""
    w = np.asarray(w, np.float32)
    amax = np.abs(w).max(axis=1)
    s = (amax / 127.0).astype(np.float32)
    s[s == 0] = 1.0
    q = np.clip(np.round(w / s[:, None]), -127, 127).astype(np.int8)
    return q, s


class HFConfig:
    def __init__(self, path, seq_len=256):
        c = json.load(open(os.path.join(path, "config.json")))
        self.arch = (c.get("architectures") or ["LlamaForCausalLM"])[0]
        self.dim = c["hidden_size"]
        self.hidden = c["intermediate_size"]
        self.layers = c["num_hidden_layers"]
        self.heads = c["num_attention_heads"]
        self.kv_heads = c.get("num_key_value_heads", self.heads)
        self.head_size = c.get("head_dim") or self.dim // self.heads
        self.vocab = c["vocab_size"]
        self.theta = float(c.get("rope_theta", 10000.0))
        self.eps = float(c.get("rms_norm_eps", 1e-5))
        self.tie = bool(c.get("tie_word_embeddings", False))
        self.qk_norm = self.arch.startswith("Qwen3")
        self.seq_len = min(seq_len, c.get("max_position_embeddings", seq_len))
        self.q_dim = self.heads * self.head_size
        self.kv_dim = self.kv_heads * self.head_size
        if c.get("rope_scaling"):
            raise ValueError(f"rope_scaling {c['rope_scaling']} not supported yet")

    def __repr__(self):
        return (f"{self.arch}: dim {self.dim}, hidden {self.hidden}, {self.layers} layers, {self.heads} heads / "
                f"{self.kv_heads} kv heads of {self.head_size}, vocab {self.vocab}, theta {self.theta}, "
                f"eps {self.eps}, tied {self.tie}, qk_norm {self.qk_norm}, seq_len {self.seq_len}")


def _interleave_rows(w, heads, hs):
    """HF rotate-half rows (per head: first half, second half) -> interleaved
    pairs (x0, x1, ...): the inverse of llama2.c export.py's permute."""
    w = np.asarray(w)
    shape = w.shape
    w = w.reshape(heads, 2, hs // 2, *shape[1:])
    return np.swapaxes(w, 1, 2).reshape(shape)


def read_safetensors(path):
    """{name: float32 array} of a .safetensors file (8-byte header length, JSON
    header, raw little-endian data; F32 / F16 / BF16)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        raw = data[a:b]
        dt = info["dtype"]
        if dt == "F32":
            v = np.frombuffer(raw, np.float32)
        elif dt == "F16":
            v = np.frombuffer(raw, np.float16).astype(np.float32)
        elif dt == "BF16":
            v = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            raise ValueError(f"{name}: dtype {dt}")
        out[name] = v.reshape(info["shape"])
    return out


def load_weights(path, cfg):
    t = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            t.update(read_safetensors(os.path.join(path, f)))
    g = lambda n: np.asarray(t[n], np.float32)
    hs = cfg.head_size
    w = {"tok": g("model.embed_tokens.weight"), "rms_final": g("model.norm.weight")}
    w["wcls"] = w["tok"] if cfg.tie or "lm_head.weight" not in t else g("lm_head.weight")
    names = {"wq": "self_attn.q_proj", "wk": "self_attn.k_proj", "wv": "self_attn.v_proj", "wo": "self_attn.o_proj",
             "w1": "mlp.gate_proj", "w3": "mlp.up_proj", "w2": "mlp.down_proj",
             "rms_att": "input_layernorm", "rms_ffn": "post_attention_layernorm"}
    if cfg.qk_norm:
        names.update({"q_norm": "self_attn.q_norm", "k_norm": "self_attn.k_norm"})
    for k, n in names.items():
        w[k] = np.stack([g(f"model.layers.{l}.{n}.weight") for l in range(cfg.layers)])
    w["wq"] = np.stack([_interleave_rows(x, cfg.heads, hs) for x in w["wq"]])
    w["wk"] = np.stack([_interleave_rows(x, cfg.kv_heads, hs) for x in w["wk"]])
    if cfg.qk_norm:
        w["q_norm"] = np.stack([_interleave_rows(x, 1, hs) for x in w["q_norm"]])
        w["k_norm"] = np.stack([_interleave_rows(x, 1, hs) for x in w["k_norm"]])
    return w


class BpeTokenizer:
    """Byte-level BPE from a HuggingFace tokenizer.json (GPT-2 style). Decoding is
    exact; encoding splits on spaces / punctuation before BPE (close to, not
    always equal to, the regex pre-tokenizer): enough for tests and demos."""

    def __init__(self, path):
        t = json.load(open(os.path.join(path, "tokenizer.json")))
        m = t["model"]
        self.vocab = m["vocab"]
        self.inv = {i: s for s, i in self.vocab.items()}
        for a in t.get("added_tokens", []):
            self.inv[a["id"]] = a["content"]
        merges = [tuple(x.split(" ")) if isinstance(x, str) else tuple(x) for x in m["merges"]]
        self.ranks = {p: i for i, p in enumerate(merges)}
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        self.b2u = {b: chr(c) for b, c in zip(bs, cs)}
        self.u2b = {chr(c): b for b, c in zip(bs, cs)}

    def _bpe(self, word):
        parts = list(word)
        while len(parts) > 1:
            pairs = [(self.ranks.get((parts[i], parts[i + 1]), 1 << 30), i) for i in range(len(parts) - 1)]
            r, i = min(pairs)
            if r == 1 << 30:
                break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        return parts

    def encode(self, text):
        import re
        ids = []
        for piece in re.findall(r" ?[A-Za-z]+| ?[0-9]| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+", text):
            u = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            ids += [self.vocab[p] for p in self._bpe(u)]
        return ids

    def decode(self, ids):
        s = "".join(self.inv.get(i, "") for i in ids)
        return bytes(self.u2b.get(ch, 32) for ch in s).decode("utf-8", errors="replace")


def rope_tables(cfg):
    """(cos, sin) (seq_len, head_size // 2): pair i of a head rotates by
    pos * theta^(-2i / head_size)."""
    hs = cfg.head_size
    freq = 1.0 / np.power(cfg.theta, np.arange(0, hs, 2, dtype=np.float64) / hs)
    val = np.arange(cfg.seq_len, dtype=np.float64)[:, None] * freq[None, :]
    return np.cos(val).astype(np.float32), np.sin(val).astype(np.float32)


def swapneg(v):
    p = v.reshape(v.shape[:-1] + (-1, 2))
    return torch.stack((-p[..., 1], p[..., 0]), dim=-1).reshape(v.shape)


class Fp32Model:
    """The fp32 reference (HF semantics), one token at a time."""

    def __init__(self, cfg, w):
        self.cfg = cfg
        self.w = {k: torch.tensor(v) for k, v in w.items()}
        cos, sin = rope_tables(cfg)
        self.cos = torch.tensor(np.repeat(cos, 2, axis=1))
        self.sin = torch.tensor(np.repeat(sin, 2, axis=1))
        self.kc = torch.zeros(cfg.layers, cfg.seq_len, cfg.kv_dim)
        self.vc = torch.zeros(cfg.layers, cfg.seq_len, cfg.kv_dim)
        self.kv_amax = np.zeros((cfg.layers, 2), np.float32)

    def rms(self, x, g):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.cfg.eps) * g

    @torch.no_grad()
    def forward(self, token, pos):
        c, w = self.cfg, self.w
        H, Hk, hs = c.heads, c.kv_heads, c.head_size
        G = H // Hk
        cos, sin = self.cos[pos], self.sin[pos]
        x = w["tok"][token].clone()
        for l in range(c.layers):
            xb = self.rms(x, w["rms_att"][l])
            q = (w["wq"][l] @ xb).reshape(H, hs)
            k = (w["wk"][l] @ xb).reshape(Hk, hs)
            v = w["wv"][l] @ xb
            if c.qk_norm:
                q = self.rms(q, w["q_norm"][l])
                k = self.rms(k, w["k_norm"][l])
            q = q * cos + swapneg(q) * sin
            k = (k * cos + swapneg(k) * sin).reshape(-1)
            self.kc[l, pos], self.vc[l, pos] = k, v
            self.kv_amax[l, 0] = max(self.kv_amax[l, 0], float(k.abs().max()))
            self.kv_amax[l, 1] = max(self.kv_amax[l, 1], float(v.abs().max()))
            K = self.kc[l, :pos + 1].reshape(pos + 1, Hk, hs)
            V = self.vc[l, :pos + 1].reshape(pos + 1, Hk, hs)
            att = torch.empty(H, hs)
            for h in range(H):
                s = (K[:, h // G] @ q[h]) / np.sqrt(hs)
                att[h] = torch.softmax(s, dim=0) @ V[:, h // G]
            x = x + w["wo"][l] @ att.reshape(-1)
            xb = self.rms(x, w["rms_ffn"][l])
            h1, h3 = w["w1"][l] @ xb, w["w3"][l] @ xb
            x = x + w["w2"][l] @ (torch.nn.functional.silu(h1) * h3)
        return w["wcls"] @ self.rms(x, w["rms_final"])


def calibrate_kv(cfg, w, tokens):
    """Static per-layer KV scales: amax / 127 over a calibration run of the fp32 model."""
    m = Fp32Model(cfg, w)
    for pos, t in enumerate(tokens[:cfg.seq_len]):
        m.forward(t, pos)
    return (np.maximum(m.kv_amax, 1e-8) / 127.0).astype(np.float32)


class QModel(torch.nn.Module):
    """The quantized decoder (QLlama's form, generalized): decode(token, pos, valid[T]) -> logits."""

    def __init__(self, cfg, w, kv_scales):
        super().__init__()
        self.cfg = cfg
        c = cfg
        L = c.layers
        P = lambda a: torch.nn.Parameter(a, requires_grad=False)
        t = lambda a, dt=None: torch.tensor(np.ascontiguousarray(a), dtype=dt)

        def lin(names, l):
            q, s = quantize_rows(np.concatenate([w[n][l] for n in names]))
            return P(t(q, torch.int8)), P(t(s, F32))
        emb_q, emb_s = quantize_rows(w["tok"])
        self.emb_q, self.emb_s = P(t(emb_q, torch.int8)), P(t(emb_s, F32))
        if w["wcls"] is not w["tok"]:
            cq, cs = quantize_rows(w["wcls"])
            self.cls_q, self.cls_s = P(t(cq, torch.int8)), P(t(cs, F32))
        else:
            self.cls_q, self.cls_s = self.emb_q, self.emb_s
        self.wqkv, self.s_wqkv = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.wo, self.s_wo = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.w13, self.s_w13 = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.w2, self.s_w2 = torch.nn.ParameterList(), torch.nn.ParameterList()
        for l in range(L):
            for qs, ss, names in ((self.wqkv, self.s_wqkv, ("wq", "wk", "wv")), (self.wo, self.s_wo, ("wo",)),
                                  (self.w13, self.s_w13, ("w1", "w3")), (self.w2, self.s_w2, ("w2",))):
                q, s = lin(names, l)
                qs.append(q)
                ss.append(s)
        self.rms_att, self.rms_ffn = P(t(w["rms_att"], F32)), P(t(w["rms_ffn"], F32))
        self.rms_final = P(t(w["rms_final"], F32))
        if c.qk_norm:
            self.q_norm, self.k_norm = P(t(w["q_norm"], F32)), P(t(w["k_norm"], F32))
        cos, sin = rope_tables(cfg)
        self.rope_cos = P(t(np.repeat(cos, 2, axis=1), F32))       # (seq, head_size)
        self.rope_sin = P(t(np.repeat(sin, 2, axis=1), F32))
        kv = np.asarray(kv_scales, np.float32)
        self.inv_sk = [f32c(1.0 / np.float32(x)) for x in kv[:, 0]]
        self.inv_sv = [f32c(1.0 / np.float32(x)) for x in kv[:, 1]]
        self.a_k = [f32c(float(x) / np.sqrt(c.head_size)) for x in kv[:, 0]]
        self.a_v = [f32c(float(x) / 127.0) for x in kv[:, 1]]
        self.register_buffer("kc", torch.zeros(L * c.seq_len, c.kv_dim, dtype=torch.int8))
        self.register_buffer("vc", torch.zeros(L * c.seq_len, c.kv_dim, dtype=torch.int8))

    def rmsnorm(self, x, g):
        ss = torch.sum(x * x, dim=-1, keepdim=True)
        return (x * torch.rsqrt(ss * f32c(1.0 / x.shape[-1]) + f32c(self.cfg.eps))) * g

    def attention(self, l, q, T, pos):
        c = self.cfg
        Hk, hs, G = c.kv_heads, c.head_size, c.heads // c.kv_heads
        r0 = l * c.seq_len
        kh = self.kc[r0:r0 + T].to(torch.int32).reshape(T, Hk, hs).permute(1, 2, 0)   # (Hk, hs, T)
        vh = self.vc[r0:r0 + T].to(torch.int32).reshape(T, Hk, hs).permute(1, 0, 2)   # (Hk, T, hs)
        qq, s_q = quant_act(q.reshape(Hk, G, hs))                                     # (Hk, G, hs), (Hk, G, 1)
        sc = torch.matmul(qq.to(torch.int32), kh)                                    # (Hk, G, T) int32
        sc = (sc.to(F32) * s_q) * self.a_k[l]
        mask = torch.arange(T) <= pos
        m = torch.amax(torch.where(mask, sc, torch.tensor(float("-inf"))), dim=-1, keepdim=True)
        e = torch.where(mask, torch.exp(sc - m), torch.tensor(0.0))
        r = torch.reciprocal(torch.sum(e, dim=-1, keepdim=True))
        pq = to_i8((e * r) * 127.0).to(torch.int32)                                 # (Hk, G, T)
        att = torch.matmul(pq, vh)                                                   # (Hk, G, hs) int32
        return (att.to(F32) * self.a_v[l]).reshape(c.q_dim)

    def forward(self, token, pos, valid):
        c = self.cfg
        T = valid.shape[0]
        hs = c.head_size
        x = (self.emb_q.index_select(0, token).to(F32) * self.emb_s.index_select(0, token).unsqueeze(-1))[0]
        cos = self.rope_cos.index_select(0, pos)[0]
        sin = self.rope_sin.index_select(0, pos)[0]
        rope = lambda v: v * cos + swapneg(v) * sin
        for l in range(c.layers):
            xq, s_x = quant_act(self.rmsnorm(x, self.rms_att[l]))
            qkv = qlinear(xq, s_x, self.wqkv[l], self.s_wqkv[l])
            q = qkv[:c.q_dim].reshape(c.heads, hs)
            k = qkv[c.q_dim:c.q_dim + c.kv_dim].reshape(c.kv_heads, hs)
            v = qkv[c.q_dim + c.kv_dim:]
            if c.qk_norm:
                q = self.rmsnorm(q, self.q_norm[l])
                k = self.rmsnorm(k, self.k_norm[l])
            q = rope(q).reshape(-1)
            k = rope(k).reshape(-1)
            row = pos + l * c.seq_len
            self.kc.index_copy_(0, row, to_i8(k * self.inv_sk[l]).unsqueeze(0))
            self.vc.index_copy_(0, row, to_i8(v * self.inv_sv[l]).unsqueeze(0))
            att = self.attention(l, q, T, pos)
            aq, s_a = quant_act(att)
            x = x + qlinear(aq, s_a, self.wo[l], self.s_wo[l])
            xq, s_x = quant_act(self.rmsnorm(x, self.rms_ffn[l]))
            h13 = qlinear(xq, s_x, self.w13[l], self.s_w13[l])
            h1, h3 = h13[:c.hidden], h13[c.hidden:]
            u = (h1 * torch.reciprocal(1.0 + torch.exp(h1 * -1.0))) * h3
            hq, s_h = quant_act(u)
            x = x + qlinear(hq, s_h, self.w2[l], self.s_w2[l])
        xq, s_x = quant_act(self.rmsnorm(x, self.rms_final))
        return qlinear(xq, s_x, self.cls_q, self.cls_s)

#!/usr/bin/env python3
"""Generic frontend (docs/iree_compiler_plan.md §8.16): an unmodified
HuggingFace causal LM (transformers' own modeling code) exported for the sa
device, instead of a hand-rewritten model (qhf.py).

- HFDecoder: decode(input_ids[1, 1], position_ids[1, 1]) -> logits[1, 1, V]
  around AutoModelForCausalLM, with a static KV cache (PositionedCache: HF's
  StaticCache written at the given positions, into buffers of the module, so
  torch.export sees buffer mutations and the cache becomes mutable globals);
- export: torch.export (non-strict) -> functional ATen (run_decompositions) ->
  the graph rewrites (GRAPH_PASSES) -> iree-turbine, the weights externalized
  (<out>/qllama.mlir, <out>/qllama.irpa: what compile_sa.sh expects);
- the program runs with sa-llm-run --abi=hfv --pad=D (--attn; --static-len or no --attn: --abi=hf); what the sa backend does not
  compile runs on the host (the VMVX fallback, §8.15).

    python3 compiler/frontend/hf_generic.py --model build/llm_cache/SmolLM2-135M --out build/hfgen/smollm2_raw
"""
import argparse
import os
import time
import types

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import StaticCache


class PositionedCache(StaticCache):
    """HF's static cache, written at the positions the caller gives (an input of
    the exported function) instead of its own counter, into the buffers of the
    owning module (read through its attributes, so torch.export sees buffer
    mutations)."""
    def __init__(self, config, max_cache_len, owner):
        super().__init__(config=config, max_cache_len=max_cache_len)
        self.positions = None
        cache = self

        def make_update(i):
            def update(layer, key_states, value_states, *a, **k):
                keys, values = getattr(owner, f"k{i}"), getattr(owner, f"v{i}")
                keys.index_copy_(2, cache.positions, key_states)
                values.index_copy_(2, cache.positions, value_states)
                return keys, values
            return update
        for i, layer in enumerate(self.layers):
            layer.update = types.MethodType(make_update(i), layer)


# Error attribution (docs/generality_implementation_plan.md WP4; quality_eval.py):
# the device forms to keep in fp32 in eager runs. Empty in exports: the graphs
# do not change. "act": linear layers on fp32 activations (weights int8);
# "kv": the KV cache in fp32; "q": the attention's query not quantized; "probs":
# the attention probabilities not quantized to int8.
SA_FP32 = set()


def _at(layer, pos):
    """The cache layer's position counters set to the input position: a full
    layer's cumulative_length, and a sliding-window layer's python int (HF
    builds the masks' cache positions from layer 0's: Gemma 3 starts with a
    sliding layer). In an export the int stays (the sa attention does not use
    HF's mask)."""
    layer.cumulative_length = pos
    if hasattr(layer, "cumulative_length_int"):
        try:
            layer.cumulative_length_int = int(pos)
        except Exception:                        # (symbolic in torch.export)
            pass


class HFDecoder(torch.nn.Module):
    """decode(input_ids[1, 1], position_ids[1, 1]) -> logits; the KV cache as buffers."""
    def __init__(self, model, max_len):
        super().__init__()
        self.model = model
        cfg = model.config
        self.cache = PositionedCache(cfg, max_len, self)
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.cache.early_initialization(batch_size=1, num_heads=cfg.num_key_value_heads, head_dim=hd,
                                        dtype=torch.float32, device=torch.device("cpu"))
        for i, layer in enumerate(self.cache.layers):
            self.register_buffer(f"k{i}", layer.keys)
            self.register_buffer(f"v{i}", layer.values)

    def prefill(self, input_ids, position_ids, logits=True):
        """A chunk of M prompt tokens at position_ids[1, M] (given by the caller):
        their KV rows written; the last one's logits, (logits="all":
        prefill_all) every row's [1, M, vocab], or (logits=False, the chunks
        before the last: prefill_kv) the last row's hidden state."""
        self.cache.positions = position_ids[0]
        for layer in self.cache.layers:
            _at(layer, position_ids.reshape(-1)[0])
        if logits:
            return self.model(input_ids=input_ids, position_ids=position_ids, past_key_values=self.cache,
                              use_cache=True, logits_to_keep=0 if logits == "all" else 1).logits
        h = self.model.model(input_ids=input_ids, position_ids=position_ids, past_key_values=self.cache,
                             use_cache=True).last_hidden_state
        return h[:, -1]

    def forward(self, input_ids, position_ids):
        self.cache.positions = position_ids[0]
        # HF builds the causal mask from the cache's own length (the positions
        # seen so far): here the input position, set (not mutated) each step
        for layer in self.cache.layers:
            _at(layer, position_ids.reshape(-1)[0])
        return self.model(input_ids=input_ids, position_ids=position_ids, past_key_values=self.cache,
                          use_cache=True).logits


# ------------------------------------------------------------------ KV cache and attention (F4)
class ObservedCache(PositionedCache):
    """Calibration: the running max |k|, |v| per layer and cache channel
    [layers, 2, kv_heads * head_dim] (after RoPE / QK-norm, as they are written
    into the cache)."""
    def __init__(self, config, max_cache_len, owner):
        super().__init__(config, max_cache_len, owner)
        hd = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.amax = np.zeros((len(self.layers), 2, config.num_key_value_heads * hd), np.float32)
        for i, layer in enumerate(self.layers):
            upd = layer.update

            def observed(key_states, value_states, *a, _i=i, _upd=upd, **k):
                for j, x in enumerate((key_states, value_states)):        # [1, kv_heads, M, hd]
                    self.amax[_i, j] = np.maximum(self.amax[_i, j], x[0].abs().amax(1).reshape(-1).numpy())
                return _upd(key_states, value_states, *a, **k)
            layer.update = observed


class SaCache(StaticCache):
    """The sa form of the KV cache (qhf.QModel): int8 rows [layers * T, kv_heads
    * head_dim], layer l's at l * T + position, with a static per-layer scale;
    update() returns the layer's T rows for sa_attention. One K and one V
    buffer for all layers, as qhf: per-layer buffers (60 mutable globals) hit
    an IREE 3.11 issue (compiler/tests/iree_global_merge_repro.py)."""
    def __init__(self, config, max_cache_len, owner, kv_scales):
        super().__init__(config=config, max_cache_len=max_cache_len)
        self.positions = None
        self.max_len = max_cache_len
        # the attention's length (export --dyn-len: valid's, a dynamic dimension;
        # None: the whole cache, max_len)
        self.T = None
        kv = np.asarray(kv_scales, np.float32)
        hd = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        Hk = config.num_key_value_heads
        G = config.num_attention_heads // Hk
        # per layer [L, 2] (one K and one V scale a layer), or per cache channel
        # [L, 2, kv_heads * hd] (calibrate_kv(channel=...)); a layer's row with a
        # single value is a per-layer scale (the same export as [L, 2])
        if kv.ndim == 3:
            ch = [[not np.all(kv[i, j] == kv[i, j, 0]) for j in (0, 1)] for i in range(len(kv))]
            kv1 = kv[:, :, 0]
        else:
            ch, kv1 = [[False, False]] * len(kv), kv
        self.k_ch = [c[0] for c in ch]
        self.v_ch = [c[1] for c in ch]

        def vec(x, shape):
            # (per head: [Hk, G, hd], repeated over the G query heads of a KV
            # head, an operand indexed as the query rows, not broadcast)
            v = torch.from_numpy(np.ascontiguousarray(x, np.float32)).reshape(shape)
            return v.expand(Hk, G, hd).contiguous() if len(shape) == 3 else v
        # K per channel: the scale folded into the query, q * s_k per head
        # channel before its quantization (sum_d q_d s_d k_d = (q * s) . k), the
        # scores then scaled by 1 / sqrt(hd) only (kv_k = 1); V per channel: the
        # P V products scaled per output channel
        self.inv_sk = [vec(np.float32(1.0) / kv[i, 0], (-1,)) if self.k_ch[i] else _f32c(1.0 / np.float32(x))
                       for i, x in enumerate(kv1[:, 0])]
        self.inv_sv = [vec(np.float32(1.0) / kv[i, 1], (-1,)) if self.v_ch[i] else _f32c(1.0 / np.float32(x))
                       for i, x in enumerate(kv1[:, 1])]
        self.sk = [vec(kv[i, 0], (Hk, 1, hd)) if self.k_ch[i] else None for i in range(len(kv1))]
        self.a_k = [_f32c(1.0 / np.sqrt(hd)) if self.k_ch[i] else _f32c(float(x) / np.sqrt(hd))
                    for i, x in enumerate(kv1[:, 0])]
        # the score scale when the attention's own is not 1 / sqrt(head_dim)
        # (Gemma: query_pre_attn_scalar^-1/2); per layer, from sa_attention's scaling
        self.kv_k = [1.0 if self.k_ch[i] else float(x) for i, x in enumerate(kv1[:, 0])]
        # (the dequantization scales: SA_FP32 attribution; per channel [Hk, G, hd])
        self.kv_v = [vec(kv[i, 1], (Hk, 1, hd)) if self.v_ch[i] else float(x) for i, x in enumerate(kv1[:, 1])]
        self.hd = hd
        self.a_v = [vec(kv[i, 1] * np.float32(1.0 / 127.0), (Hk, 1, hd)) if self.v_ch[i] else _f32c(float(x) / 127.0)
                    for i, x in enumerate(kv1[:, 1])]
        cache, T = self, max_cache_len

        def make_update(i):
            def update(layer, key_states, value_states, *a, **k):
                kr = key_states[0].transpose(0, 1).reshape(key_states.shape[2], -1)     # (M, kv_heads * hd)
                vr = value_states[0].transpose(0, 1).reshape(value_states.shape[2], -1)
                if "kv" in SA_FP32:                   # (attribution: an fp32 cache beside the int8 one)
                    kf = owner.__dict__.setdefault("_kf", torch.zeros(owner.kc.shape, dtype=torch.float32))
                    vf = owner.__dict__.setdefault("_vf", torch.zeros(owner.vc.shape, dtype=torch.float32))
                    for m in range(kr.shape[0]):
                        row = cache.positions[m:m + 1] + i * T
                        kf.index_copy_(0, row, kr[m:m + 1].float())
                        vf.index_copy_(0, row, vr[m:m + 1].float())
                    n = T if cache.T is None else cache.T
                    return kf[i * T:i * T + n], vf[i * T:i * T + n]
                kq, vq = _to_i8(kr * cache.inv_sk[i]), _to_i8(vr * cache.inv_sv[i])
                for m in range(kr.shape[0]):          # row by row: a scalar position each (a decode step's write)
                    row = cache.positions[m:m + 1] + i * T
                    owner.kc.index_copy_(0, row, kq[m:m + 1])
                    owner.vc.index_copy_(0, row, vq[m:m + 1])
                n = T if cache.T is None else cache.T    # (the first n rows: the positions attended)
                return owner.kc[i * T:i * T + n], owner.vc[i * T:i * T + n]
            return update
        for i, layer in enumerate(self.layers):
            layer.update = types.MethodType(make_update(i), layer)


def sa_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """The attention of the sa device (qhf.QModel.attention), registered with
    transformers' AttentionInterface as "sa": query [1, H, M, hd] (after RoPE;
    M = 1 in decode, the chunk's rows in prefill), key / value the layer's int8
    cache rows [T, kv_heads * hd] (SaCache); per query row (a decode step's
    form: its scalar position): q per head quantized, int32 scores, the prefix
    mask from the position (HF's mask is not used), the unnormalized softmax
    e = exp(s - max) quantized (eq = i8(e * 127): the row's largest is 127
    whatever the row's spread), int32 eq V normalized by the quantized sum
    (* 127 / sum(eq): the weights sum to 1 exactly; plan §12.4)."""
    cache = module.sa_cache
    l = module.layer_idx
    if SA_FP32 & {"kv", "q", "probs"}:
        return _sa_attention_fp32(module, query, key, value, scaling)
    H, M, hd = query.shape[1], query.shape[2], query.shape[3]
    T = key.shape[0]
    Hk = key.shape[1] // hd
    G = H // Hk
    kh = key.to(torch.int32).reshape(T, Hk, hd).permute(1, 2, 0)                  # (Hk, hd, T)
    vh = value.to(torch.int32).reshape(T, Hk, hd).permute(1, 0, 2)                # (Hk, T, hd)
    # the query rows as [M, H, hd] (HF's transpose undone: each row contiguous)
    qr = query[0].transpose(0, 1)
    outs = []
    for m in range(M):
        q = qr[m].reshape(Hk, G, hd)
        if cache.k_ch[l]:                                                        # (K per channel)
            q = q * cache.sk[l]
        qq, s_q = _quant_act(q)                                                  # (Hk, G, hd), (Hk, G, 1)
        sc = torch.matmul(qq.to(torch.int32), kh)                                # (Hk, G, T) int32
        a_k = cache.a_k[l]
        if scaling is not None and abs(scaling * np.sqrt(cache.hd) - 1.0) > 1e-9:
            a_k = _f32c(cache.kv_k[l] * float(scaling))
        sc = (sc.to(torch.float32) * s_q) * a_k
        # the softmax over [H, T] rows (one per head: the device's row pieces,
        # sahl-tile; Llama's 8 x 4 heads beyond what one piece holds)
        sc = sc.reshape(H, T)
        mask = torch.arange(T) <= cache.positions[m:m + 1]
        mx = torch.amax(torch.where(mask, sc, torch.tensor(float("-inf"))), dim=-1, keepdim=True)
        e = torch.where(mask, torch.exp(sc - mx), torch.tensor(0.0))
        eq = _to_i8(e * 127.0)
        pq = eq.to(torch.int32).reshape(Hk, G, T)
        att = torch.matmul(pq, vh)                                               # (Hk, G, hd) int32
        # sum(eq) in fp32 straight from the int8 (exact: at most 127 T)
        rq = torch.reciprocal(torch.sum(eq.to(torch.float32), dim=-1, keepdim=True)).reshape(Hk, G, 1) * 127.0
        outs.append(((att.to(torch.float32) * rq) * cache.a_v[l]).reshape(1, 1, H, hd))  # (a_v: a scalar or [Hk, G, hd])
    return (outs[0] if M == 1 else torch.cat(outs, dim=1)), None


class _Given(torch.nn.Module):
    """A projection replaced by its precomputed output (_attention_rows)."""
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, x):
        return self.v


def _attention_rows(forward):
    """The attention module's own forward in prefill (M > 1) row by row, each
    exactly as a decode step's (as qhf.prefill): the q / k / v projections of
    the M rows computed together first (the linear layers are what gain from
    the rows), then the module's forward on each row with those projections
    given and the row's cos / sin looked up at its scalar position in the RoPE
    table (sa_rope, as a decode step does: QK-norm, RoPE, the KV row, the sa
    attention at the row's position), and o_proj on the M rows together. In one
    piece the chunk's QK-norm / RoPE / attention inputs are M x heads x head_dim
    tensors that do not fit ACC (left to the host: not bit-exact with decode),
    and their row slices strided views. With the chunk's RoPE rows sliced
    instead of looked up, a model without QK-norm (Llama, SmolLM2) had each
    row's RoPE + int8 quantization fused into the KV scatter, which went to the
    host, whose arithmetic differs between the ARM and the x86 sim (the board
    off by an int8 step). Decode (M = 1): the forward unchanged."""
    names = ("q_proj", "k_proj", "v_proj", "o_proj")

    def rows(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
        M = hidden_states.shape[1]
        if M == 1:
            return forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kw)
        saved = {n: getattr(self, n) for n in names}
        proj = {n: saved[n](hidden_states) for n in names[:3]}
        cache = self.sa_cache
        positions = cache.positions
        cos, sin = position_embeddings
        outs = []
        try:
            setattr(self, "o_proj", _Given(None))
            self.o_proj.forward = lambda x: x
            for m in range(M):
                for n in names[:3]:
                    setattr(self, n, _Given(proj[n][:, m:m + 1]))
                cache.positions = positions[m:m + 1]
                rope = getattr(self, "sa_rope", None)
                if rope is not None:             # (the table lookup of a decode step)
                    ty = getattr(self, "layer_type", None)
                    emb = rope(hidden_states[:, m:m + 1], positions[m:m + 1][None],
                               *([ty] if ty is not None and rope.types != [None] else []))
                else:
                    emb = (cos[:, m:m + 1], sin[:, m:m + 1])
                o, _ = forward(self, hidden_states[:, m:m + 1], emb, None, past_key_values, **kw)
                outs.append(o)
        finally:
            for n, v in saved.items():
                setattr(self, n, v)
            cache.positions = positions
        return saved["o_proj"](torch.cat(outs, dim=1)), None
    return rows


def _sa_attention_fp32(module, query, key, value, scaling):
    """sa_attention with some of its quantizations kept in fp32 (SA_FP32;
    attribution only); what is not switched computes exactly as sa_attention
    (int32 products, the same scale order), so a difference is the switch's."""
    cache, l = module.sa_cache, module.layer_idx
    H, M, hd = query.shape[1], query.shape[2], query.shape[3]
    T = key.shape[0]
    Hk = key.shape[1] // hd
    G = H // Hk
    kv32 = key.is_floating_point()               # ("kv": the fp32 cache)
    a_k = cache.a_k[l]
    if scaling is not None and abs(scaling * np.sqrt(cache.hd) - 1.0) > 1e-9:
        a_k = _f32c(cache.kv_k[l] * float(scaling))
    scale = float(scaling) if scaling is not None else 1.0 / np.sqrt(hd)
    if kv32:
        kh = key.float().reshape(T, Hk, hd).permute(1, 2, 0)
        vh = value.float().reshape(T, Hk, hd).permute(1, 0, 2)
    else:
        kh = key.to(torch.int32).reshape(T, Hk, hd).permute(1, 2, 0)
        vh = value.to(torch.int32).reshape(T, Hk, hd).permute(1, 0, 2)
    qr = query[0].transpose(0, 1)
    outs = []
    for m in range(M):
        q = qr[m].reshape(Hk, G, hd)
        if "q" in SA_FP32 or kv32:
            qf = q if "q" in SA_FP32 else (lambda t: t[0].to(torch.float32) * t[1])(_quant_act(q))
            sk = cache.sk[l][:, :1].transpose(1, 2) if cache.k_ch[l] else _f32c(cache.kv_k[l])     # ([Hk, hd, 1])
            kf = kh if kv32 else kh.to(torch.float32) * sk
            sc = torch.matmul(qf, kf) * scale
        else:
            qq, s_q = _quant_act(q * cache.sk[l] if cache.k_ch[l] else q)
            sc = (torch.matmul(qq.to(torch.int32), kh).to(torch.float32) * s_q) * a_k
        mask = torch.arange(T) <= cache.positions[m:m + 1]
        mx = torch.amax(torch.where(mask, sc, torch.tensor(float("-inf"))), dim=-1, keepdim=True)
        e = torch.where(mask, torch.exp(sc - mx), torch.tensor(0.0))
        if "probs" in SA_FP32 or kv32:
            if "probs" in SA_FP32:
                p = e * torch.reciprocal(torch.sum(e, dim=-1, keepdim=True))
            else:
                eq = _to_i8(e * 127.0).to(torch.float32)
                p = eq * torch.reciprocal(torch.sum(eq, dim=-1, keepdim=True))
            sv = cache.kv_v[l][:, :1] if cache.v_ch[l] else _f32c(cache.kv_v[l])
            vf = vh if kv32 else vh.to(torch.float32) * sv
            outs.append(torch.matmul(p, vf).reshape(1, 1, H, hd))
        else:
            pq = _to_i8(e * 127.0).to(torch.int32)
            rq = torch.reciprocal(torch.sum(pq.to(torch.float32), dim=-1, keepdim=True)) * 127.0
            outs.append(((torch.matmul(pq, vh).to(torch.float32) * rq) * cache.a_v[l]).reshape(1, 1, H, hd))
    return (outs[0] if M == 1 else torch.cat(outs, dim=1)), None


class SaDecoder(HFDecoder):
    """HFDecoder with the sa KV cache and attention (F4)."""
    def __init__(self, model, max_len, kv_scales):
        torch.nn.Module.__init__(self)
        # the sa attention is causal over the whole cache: a sliding window
        # (Gemma 3's local layers) is the same only while the cache fits in it
        win = getattr(model.config, "sliding_window", None)
        if win and "sliding_attention" in (getattr(model.config, "layer_types", None) or []) and max_len > win:
            raise NotImplementedError(f"max_len {max_len} beyond the sliding window {win} (plan C6.5)")
        from transformers import AttentionInterface
        AttentionInterface.register("sa", sa_attention)
        self.model = model
        cfg = model.config
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        kv_dim = cfg.num_key_value_heads * hd
        self.cache = SaCache(cfg, max_len, self, kv_scales)
        self.kv_scales = np.asarray(kv_scales, np.float32)     # (as calibrated: reused by quality_eval)
        # K and V start with different contents (see SaCache; the rows are only
        # read after they are written: the attention masks later positions)
        L = len(self.cache.layers)
        self.register_buffer("kc", torch.zeros(L * max_len, kv_dim, dtype=torch.int8))
        self.register_buffer("vc", torch.ones(L * max_len, kv_dim, dtype=torch.int8))
        for mod in model.modules():
            if hasattr(mod, "layer_idx") and hasattr(mod, "q_proj"):
                mod.sa_cache = self.cache
                rope = next((x for x in model.modules() if isinstance(x, RopeTable)), None)
                if rope is not None:
                    object.__setattr__(mod, "sa_rope", rope)          # (not a submodule: no second copy)
                mod.forward = types.MethodType(_attention_rows(type(mod).forward), mod)
        model.set_attn_implementation("sa")


def kv_calib_arg(v, default_tokens=1024):
    """--kv-calib FILE[:TOKENS] -> (file, tokens) or None"""
    if not v:
        return None
    f, _, n = v.rpartition(":") if ":" in v else (v, "", "")
    return (f, int(n) if n else default_tokens)


def calibrate_kv(model, tokens, max_len, channel=""):
    """Static KV scales: amax / 127 over the fp32 model on the tokens, per layer
    [L, 2] (qhf.calibrate_kv), or with channel "k" / "v" / "kv" those parts per
    cache channel [L, 2, kv_heads * hd] (the others' rows a layer's single value).
    Per channel K: a few channels' amax is 10-100x the median one's (Qwen3: the
    RoPE pairs of low frequency), so one scale a layer leaves most channels a
    handful of int8 steps. Tokens beyond max_len: windows of max_len, each from
    position 0 (a fresh cache), the max over them (per channel scales need
    more text than the 45-token CALIB: each channel's amax is from fewer values)."""
    hd = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads
    a = None
    for w in range(0, max(len(tokens), 1), max_len):
        obs = HFDecoder(model, max_len)
        obs.cache = ObservedCache(model.config, max_len, obs)
        obs.cache.early_initialization(batch_size=1, num_heads=model.config.num_key_value_heads, head_dim=hd,
                                       dtype=torch.float32, device=torch.device("cpu"))
        for pos, t in enumerate(tokens[w:w + max_len]):
            obs(torch.tensor([[t]]), torch.tensor([[pos]]))
        a = obs.cache.amax if a is None else np.maximum(a, obs.cache.amax)  # [L, 2, C]
    if not channel:
        return (np.maximum(a.max(-1), 1e-8) / 127.0).astype(np.float32)
    out = np.repeat(a.max(-1, keepdims=True), a.shape[-1], -1)
    for j, part in enumerate("kv"):
        if part in channel:
            out[:, j] = a[:, j]
    return (np.maximum(out, 1e-8) / 127.0).astype(np.float32)


def hf_model(model_dir, layers=None):
    """The HF model in fp32 (layers: only the first ones, a truncated model for tests).
    A compressed-tensors W8A8 checkpoint (ct_import.py): its quantized layers
    dequantized, their weights carrying the checkpoint's (q, s) for quantize()."""
    from transformers import AutoConfig
    import ct_import
    if ct_import.quant_config(model_dir) is not None:
        m, r = ct_import.load_model(model_dir, layers)
        print(f"  ct_import: {len(r['quantized'])} quantized linear layers from the checkpoint, ignored {r['ignored']}")
        return m
    cfg = AutoConfig.from_pretrained(model_dir)
    if layers:
        cfg.num_hidden_layers = layers
        if getattr(cfg, "layer_types", None):
            cfg.layer_types = cfg.layer_types[:layers]
    return AutoModelForCausalLM.from_pretrained(model_dir, config=cfg, dtype=torch.float32,
                                                attn_implementation="sdpa").eval()


def load(model_dir, max_len=256, quant=False, rope=False, attn=False, kv_scales=None, layers=None, kv_channel="",
         kv_calib=None):
    torch.set_grad_enabled(False)
    m = hf_model(model_dir, layers)
    if rope:
        print("  rope_rewrite:", rope_rewrite(m, max_len))
        print("  norm_rewrite:", norm_rewrite(m))
        n = act_rewrite(m)
        if n:
            print("  act_rewrite:", n)
    kv = None
    if attn and kv_scales is not None:    # given (e.g. qhf's, for a bit-exact comparison): a .npy or an array
        kv = (np.load(kv_scales) if isinstance(kv_scales, str) else np.asarray(kv_scales)).astype(np.float32)
    elif attn:                            # KV scales from the fp32 model (after the RoPE / norm rewrites)
        import export_hf
        from hf_tokenizer import BpeTokenizer
        # kv_calib: (text file, tokens) to calibrate on instead of CALIB
        toks = BpeTokenizer(model_dir).encode(export_hf.CALIB)
        if kv_calib:
            with open(kv_calib[0]) as f:
                toks = BpeTokenizer(model_dir).encode(f.read(16 * kv_calib[1]))[:kv_calib[1]]
        kv = calibrate_kv(m, toks, max_len, kv_channel)
        print(f"  calibrate_kv: {len(kv)} layers on {len(toks)} tokens" +
              (f", per channel: {kv_channel}" if kv_channel else ""))
    if quant:
        print("  quantize:", quantize(m))
    return SaDecoder(m, max_len, kv) if attn else HFDecoder(m, max_len)


# ------------------------------------------------------------------ module rewrites (before the export)
def _f32c(v):
    return float(np.float32(v))


def _to_i8(x):
    return torch.clamp(torch.round(torch.nan_to_num(x, nan=0.0)), -127.0, 127.0).to(torch.int8)


def _quant_act(x):
    """per-token int8: (x / s_x rounded, s_x = max |x| / 127) as qhf.quant_act"""
    amax = torch.amax(torch.abs(x), dim=-1, keepdim=True)
    s_x = amax * _f32c(1.0 / 127.0)
    return _to_i8(x * torch.reciprocal(s_x)), s_x


def _quantize_rows(w):
    """int8 rows with one fp32 scale each (qhf.quantize_rows)"""
    w = w.detach().float()
    amax = w.abs().amax(dim=1)
    s = (amax / 127.0).to(torch.float32)
    s[s == 0] = 1.0
    q = torch.clamp(torch.round(w / s[:, None]), -127, 127).to(torch.int8)
    return q, s


class QLinear(torch.nn.Module):
    """W8A8 linear layer: int8 weight rows with a per-row scale, per-token int8
    activations, an int32 product, dequantized (acc * s_w) * s_x (qhf.qlinear)."""
    def __init__(self, lin, qs=None):
        super().__init__()
        q, s = qs if qs is not None else _quantize_rows(lin.weight)
        self.wq = torch.nn.Parameter(q, requires_grad=False)
        self.s_w = torch.nn.Parameter(s, requires_grad=False)
        self.bias = None if lin.bias is None else torch.nn.Parameter(lin.bias.detach().float(), requires_grad=False)

    def forward(self, x):
        if "act" in SA_FP32:                     # (attribution: weight-only int8)
            y = torch.matmul(x, (self.wq.to(torch.float32) * self.s_w[:, None]).t())
            return y if self.bias is None else y + self.bias
        xq, s_x = _quant_act(x)
        # on rows [tokens, in] (qhf's form): a 3-D int32 matmul is exported as a
        # batch_matmul accumulating in i64, which the sa backend leaves to the host
        rows = xq.reshape(-1, xq.shape[-1])
        acc = torch.matmul(rows.to(torch.int32), self.wq.to(torch.int32).t())
        acc = acc.reshape(tuple(xq.shape[:-1]) + (acc.shape[-1],))
        y = (acc.to(torch.float32) * self.s_w) * s_x
        return y if self.bias is None else y + self.bias


class QEmbedding(torch.nn.Module):
    """int8 embedding rows with a per-row scale: e[t] = q[t] * s[t], times the
    module's embed_scale if it has one (Gemma: sqrt(hidden); not folded into s,
    which a tied classifier shares)."""
    def __init__(self, emb, qs=None):
        super().__init__()
        q, s = qs if qs is not None else _quantize_rows(emb.weight)
        self.q = torch.nn.Parameter(q, requires_grad=False)
        self.s = torch.nn.Parameter(s, requires_grad=False)
        scale = getattr(emb, "embed_scale", None)
        if scale is None and type(emb).forward is not torch.nn.Embedding.forward:
            raise NotImplementedError(f"{type(emb).__name__}: an embedding with its own forward and no embed_scale")
        self.mul = None if scale is None else _f32c(float(scale))

    def forward(self, ids):
        # row by row, each an index_select by one id: the sa backend's row gather
        # with a scalar index (prefill's M tokens: M gathers)
        flat = ids.reshape(-1)
        e = torch.cat([torch.index_select(self.q, 0, flat[m:m + 1]).to(torch.float32) *
                       torch.index_select(self.s, 0, flat[m:m + 1]).unsqueeze(-1) for m in range(flat.shape[0])])
        e = e.reshape(tuple(ids.shape) + (self.q.shape[-1],))
        return e if self.mul is None else e * self.mul


# RoPE (F3): the rotary module's cos / sin as tables, and rotate_half as the
# pair swap of interleaved halves
class RopeTable(torch.nn.Module):
    """A rotary-embedding module replaced by its own output for every position:
    cos / sin tables [max_len, head_dim] computed once by the original module
    (any RoPE variant, frequency scaling included), then looked up by position."""
    def __init__(self, rotary, max_len, head_dim, perm=None):
        super().__init__()
        x = torch.zeros(1, 1, head_dim)
        # a rotary module per layer type (Gemma 3: a local and a global base): a table each
        self.types = sorted(getattr(rotary, "layer_types", None) or []) or [None]
        for ty in self.types:
            cos, sin = rotary(x, torch.arange(max_len)[None], *([ty] if ty else []))
            cos, sin = cos[0].float(), sin[0].float()
            if perm is not None:
                cos, sin = cos[:, perm], sin[:, perm]
            sfx = "" if ty is None else "_" + ty
            setattr(self, "cos_t" + sfx, torch.nn.Parameter(cos.contiguous(), requires_grad=False))
            setattr(self, "sin_t" + sfx, torch.nn.Parameter(sin.contiguous(), requires_grad=False))

    def forward(self, x, position_ids, layer_type=None):
        if self.types != [None]:
            sfx = "_" + layer_type
            return self._rows(getattr(self, "cos_t" + sfx), getattr(self, "sin_t" + sfx), x, position_ids)
        return self._rows(self.cos_t, self.sin_t, x, position_ids)

    def _rows(self, cos_t, sin_t, x, position_ids):
        # row by row, each an index_select by one position: the sa backend's
        # row gather with a scalar index (prefill's M positions: M gathers)
        shape = tuple(position_ids.shape) + (cos_t.shape[-1],)
        pos = position_ids.reshape(-1)
        rows = lambda t: torch.cat([torch.index_select(t, 0, pos[m:m + 1]) for m in range(pos.shape[0])])
        return rows(cos_t).reshape(shape).to(x.dtype), rows(sin_t).reshape(shape).to(x.dtype)


def swapneg(v):
    """(x0, x1) -> (-x1, x0) for every pair (the sa VE's SWAPNEG): rotate_half of
    a head whose halves are interleaved."""
    p = v.reshape(v.shape[:-1] + (-1, 2))
    return torch.stack((-p[..., 1], p[..., 0]), dim=-1).reshape(v.shape)


def interleave_perm(hd):
    """head dims [0 .. h) and [h .. hd) interleaved: new 2j = old j, new 2j + 1 = old j + h"""
    h = hd // 2
    return torch.stack((torch.arange(h), torch.arange(h) + h), dim=1).reshape(-1)


def rope_rewrite(model, max_len):
    """F3 (plan §8.16): the rotary module -> RopeTable; each head's dims of the q
    / k projections (and q / k norms) interleaved (the dot products of q and k
    are unchanged: both permuted alike), the tables' columns alike, and
    rotate_half -> swapneg in every transformers modeling module that defines
    it (in the interleaved layout the two are the same). The last one holds for
    the whole process: an unmodified model of the same family loaded afterwards
    computes RoPE wrongly (compute such references first)."""
    import sys
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    perm = interleave_perm(hd)
    n = {"rotary": 0, "projections": 0, "norms": 0, "rotate_half": 0}
    for mod in list(model.modules()):
        for cname, child in list(mod.named_children()):
            if type(child).__name__.endswith("RotaryEmbedding"):
                setattr(mod, cname, RopeTable(child, max_len, hd, perm))
                n["rotary"] += 1
    for mod in model.modules():
        for pname in ("q_proj", "k_proj"):
            lin = getattr(mod, pname, None)
            if isinstance(lin, torch.nn.Linear):
                w = lin.weight.data
                heads = w.shape[0] // hd
                lin.weight.data = w.reshape(heads, hd, -1)[:, perm].reshape(w.shape).contiguous()
                qs = getattr(lin.weight, "_sa_qs", None)
                if qs is not None:        # an imported checkpoint's int8 rows and scales, alike
                    q, s = qs
                    lin.weight._sa_qs = (q.reshape(heads, hd, -1)[:, perm].reshape(q.shape).contiguous(),
                                         s.reshape(heads, hd)[:, perm].reshape(-1).contiguous())
                if lin.bias is not None:
                    lin.bias.data = lin.bias.data.reshape(heads, hd)[:, perm].reshape(-1).contiguous()
                n["projections"] += 1
        for pname in ("q_norm", "k_norm"):
            norm = getattr(mod, pname, None)
            if norm is not None and getattr(norm, "weight", None) is not None and norm.weight.shape[-1] == hd:
                norm.weight.data = norm.weight.data[perm].contiguous()
                n["norms"] += 1
    for name, m in list(sys.modules.items()):
        # (the module's own dict: getattr on transformers' lazy modules imports more)
        if name.startswith("transformers.models.") and "rotate_half" in getattr(m, "__dict__", {}):
            m.rotate_half = swapneg
            n["rotate_half"] += 1
    return n


class SaRMSNorm(torch.nn.Module):
    """RMSNorm as the sa device computes it (qhf.rmsnorm): sum(x * x) times the
    constant 1 / n instead of mean() (a division), then rsqrt; the same weight."""
    def __init__(self, norm):
        super().__init__()
        eps = getattr(norm, "variance_epsilon", None)
        self.eps = _f32c(eps if eps is not None else norm.eps)
        # the gain the module applies: its weight, or 1 + weight (Gemma), found by
        # running it on ones (any other form is refused, not guessed)
        with torch.no_grad():
            w = norm.weight.detach().float()
            g = norm(torch.ones(1, w.shape[-1])).float()[0] * float(np.sqrt(np.float32(1.0) + np.float32(self.eps)))
        if torch.allclose(g, w, rtol=1e-5, atol=1e-6):
            self.weight = norm.weight
        elif torch.allclose(g, 1.0 + w, rtol=1e-5, atol=1e-6):
            self.weight = torch.nn.Parameter(1.0 + w, requires_grad=False)
        else:
            raise NotImplementedError(f"{type(norm).__name__}: an RMSNorm gain other than w or 1 + w")

    def forward(self, x):
        ss = torch.sum(x * x, dim=-1, keepdim=True)
        return (x * torch.rsqrt(ss * _f32c(1.0 / x.shape[-1]) + self.eps)) * self.weight


def norm_rewrite(model):
    """Modules whose class name ends with RMSNorm -> SaRMSNorm (mean -> sum * 1 / n)."""
    n = 0
    for mod in list(model.modules()):
        for cname, child in list(mod.named_children()):
            if type(child).__name__.endswith("RMSNorm") and getattr(child, "weight", None) is not None:
                setattr(mod, cname, SaRMSNorm(child))
                n += 1
    return n


class SaGELUTanh(torch.nn.Module):
    """GELU's tanh approximation as the sa device computes it: x * 0.5 * (1 +
    tanh(u)) = x * sigmoid(2u) = x * 1 / (1 + exp(z)), z = -2u = x * (a + b * x * x)
    (u = sqrt(2 / pi) * (x + 0.044715 * x^3)); multiplies, adds, EXP and RECIP
    only (the VE has no tanh, VMVX no fpowi). Large negative x: EXP saturates to
    inf, RECIP to 0, the result -0 (torch: -0 as well)."""
    A = _f32c(-2.0 * np.sqrt(2.0 / np.pi))
    B = _f32c(-2.0 * np.sqrt(2.0 / np.pi) * 0.044715)

    def forward(self, x):
        z = x * (self.A + (x * x) * self.B)
        return x * torch.reciprocal(torch.exp(z) + 1.0)


def act_rewrite(model):
    """Activation modules whose class name has GELU -> SaGELUTanh when the module
    is the tanh approximation (found by running it; the exact erf form and any
    other is refused, not approximated)."""
    n = 0
    x = torch.linspace(-8.0, 8.0, 257)
    want = torch.nn.functional.gelu(x, approximate="tanh")
    for mod in list(model.modules()):
        for cname, child in list(mod.named_children()):
            if "GELU" not in type(child).__name__.upper():
                continue
            if not torch.allclose(child(x), want, rtol=1e-5, atol=1e-6):
                raise NotImplementedError(f"{type(child).__name__}: a GELU other than the tanh approximation")
            setattr(mod, cname, SaGELUTanh())
            n += 1
    return n


def quantize(model):
    """Every nn.Linear -> QLinear, nn.Embedding -> QEmbedding (W8A8, qhf's
    scheme). A weight shared by several modules (a classifier tied to the
    embedding) is quantized once and the int8 tensor shared (the packed-gather
    optimization of the sa backend reads it once, plan §8.12 C6.0). A weight
    with `_sa_qs` (an imported checkpoint, ct_import.py) keeps its int8 rows
    and scales as they are."""
    shared, n = {}, {"linear": 0, "embedding": 0, "imported": 0}

    def qs_of(weight):
        key = weight.data_ptr()
        if key not in shared:
            qs = getattr(weight, "_sa_qs", None)
            shared[key] = qs if qs is not None else _quantize_rows(weight)
            n["imported"] += qs is not None
        return shared[key]

    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear):
                q = QLinear(child, qs_of(child.weight))
                n["linear"] += 1
            elif isinstance(child, torch.nn.Embedding):
                qs = qs_of(child.weight)
                if getattr(child.weight, "_sa_qs", None) is not None and bool((qs[0] == -128).any()):
                    # the row gather copies through the VE, whose int8 output saturates to +-127
                    import ct_import
                    raise ct_import.UnsupportedCheckpoint(
                        f"{name or 'model'}.{cname}: the embedding shares the imported int8 table of a quantized "
                        "classifier, which has -128; supported: the classifier in `ignore` (quantized here) or untied")
                q = QEmbedding(child, qs)
                n["embedding"] += 1
            else:
                continue
            setattr(mod, cname, q)
    # tied tensors: one Parameter object for each shared int8 table
    params = {}
    for mod in model.modules():
        for pname in ("wq", "q"):
            p = getattr(mod, pname, None)
            if isinstance(p, torch.nn.Parameter):
                key = id(p.data) if False else p.data_ptr()
                if key in params:
                    setattr(mod, pname, params[key])
                else:
                    params[key] = p
    return n


# ------------------------------------------------------------------ graph rewrites
def square_pow(graph):
    """pow(x, 2) -> x * x (VMVX has no fpowi; on the sa VE a multiply)."""
    n = 0
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target == torch.ops.aten.pow.Tensor_Scalar and node.args[1] == 2:
            with graph.inserting_before(node):
                sq = graph.call_function(torch.ops.aten.mul.Tensor, (node.args[0], node.args[0]))
            sq.meta = dict(node.meta)
            node.replace_all_uses_with(sq)
            graph.erase_node(node)
            n += 1
    return n


GRAPH_PASSES = [square_pow]


class _Fn(torch.nn.Module):
    """One function of the module (decode, prefill, prefill_kv, prefill_all) as a module to export."""
    def __init__(self, w, kind):
        super().__init__()
        self.w, self.kind = w, kind

    def forward(self, input_ids, position_ids, valid=None):
        # --dyn-len: the attention over valid's length (the caller's: the last
        # position + 1 rounded up to the array size) instead of the whole cache
        self.w.cache.T = None if valid is None else valid.shape[0]
        try:
            if self.kind == "main":
                return self.w(input_ids, position_ids)
            return self.w.prefill(input_ids, position_ids,
                                  logits={"prefill": True, "prefill_all": "all"}.get(self.kind, False))
        finally:
            self.w.cache.T = None


def export(w, out, prefill=0, prefill_all=False, dyn_len=None):
    """main (decode); with prefill = M also prefill / prefill_kv of M tokens, the
    three sharing the parameters and the KV cache (plan §8.13, §8.17);
    prefill_all: also prefill_all (the logits of all M positions: teacher
    forcing on the board, generality plan §13.1); dyn_len: each function also
    takes valid[T] (T dynamic, at most the cache's length) and attends to the
    first T cache rows only (sa-llm-run --abi=hfv --pad=D), not the whole cache;
    None (the default): with the sa KV cache (--attn), dyn_len."""
    from iree.turbine import aot
    from iree.turbine.aot import decompositions
    os.makedirs(out, exist_ok=True)
    if dyn_len is None:
        dyn_len = isinstance(w.cache, SaCache)
    t0 = time.time()
    progs = {"main": 1} | ({"prefill": prefill, "prefill_kv": prefill} if prefill else {}) \
        | ({"prefill_all": prefill} if prefill and prefill_all else {})
    eps = {}
    for name, M in progs.items():
        args = {"input_ids": torch.ones(1, M, dtype=torch.int64), "position_ids": torch.arange(M)[None]}
        dyn = None
        if dyn_len:
            if not isinstance(w.cache, SaCache):
                raise ValueError("--dyn-len needs --attn (the sa KV cache)")
            args["valid"] = torch.zeros(max(M, 16), dtype=torch.float32)   # (not 1: no specialization)
            dyn = {"input_ids": None, "position_ids": None,
                   "valid": {0: torch.export.Dim("T", min=1, max=w.cache.max_len)}}
        ep = torch.export.export(_Fn(w, name), (), args, dynamic_shapes=dyn, strict=False)
        # functional ATen (the cache writes as buffer mutations), with iree-turbine's
        # decomposition table (as aot.export(module) does: the frontend's usual op forms)
        ep = ep.run_decompositions(decompositions.current_aot_decompositions())
        for p in GRAPH_PASSES:
            print(f"  {name}: {p.__name__}: {p(ep.graph)}")
        ep.graph.lint()
        eps[name] = ep
    aot.externalize_module_parameters(w, external_scope="model")   # (after: marked tensors break aot_autograd)
    if not prefill:
        exp = aot.export(eps["main"])
    else:
        fx = aot.FxPrograms()
        for name, ep in eps.items():
            fx.programs[name] = ep
        exp = aot.export(fx)
    mlir, irpa = os.path.join(out, "qllama.mlir"), os.path.join(out, "qllama.irpa")
    exp.save_mlir(mlir)
    aot.save_module_parameters(irpa, w)
    with open(os.path.join(out, "abi.txt"), "w") as f:      # (sa-llm-run --abi: test_hf_generic --skip-export)
        f.write("hfv\n" if dyn_len else "hf\n")
    flags = os.path.join(out, "compile_flags.txt")         # (compile_sa.sh: the compiler plans the SPAD for
    if getattr(w.cache, "max_len", 256) != 256:            # dynamic lengths up to max_dynamic, default 256)
        with open(flags, "w") as f:
            f.write(f"--iree-sa-max-dynamic={w.cache.max_len}\n")
    elif os.path.exists(flags):
        os.remove(flags)
    print(f"exported in {time.time() - t0:.0f} s: {mlir} ({os.path.getsize(mlir) / 1e6:.1f} MB), "
          f"{irpa} ({os.path.getsize(irpa) / 1e6:.1f} MB)")
    return mlir, irpa


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="a HuggingFace directory (config.json, *.safetensors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--quant", action="store_true", help="W8A8: linear layers and embeddings (QLinear / QEmbedding)")
    ap.add_argument("--rope", action="store_true", help="RoPE tables and the pair swap (RopeTable, swapneg)")
    ap.add_argument("--attn", action="store_true", help="int8 KV cache and the sa attention (SaCache, sa_attention)")
    ap.add_argument("--kv-scales", help="KV scales (.npy [layers, 2]) instead of calibrating")
    ap.add_argument("--prefill", type=int, default=0, help="also prefill / prefill_kv of this many tokens (needs --attn)")
    ap.add_argument("--prefill-all", action="store_true", help="with --prefill: also prefill_all (every position's logits)")
    ap.add_argument("--layers", type=int, help="only the first layers (a truncated model, for tests)")
    ap.add_argument("--kv-channel", default="", help="k / v / kv: those KV scales per cache channel (calibrate_kv)")
    ap.add_argument("--kv-calib", help="FILE[:TOKENS] text to calibrate the KV scales on (default: export_hf.CALIB)")
    ap.add_argument("--static-len", action="store_true", help="the attention over the whole cache (default with "
                    "--attn: over valid[T], T dynamic; sa-llm-run --abi=hfv --pad=D)")
    ap.add_argument("--dyn-len", action="store_true", help=argparse.SUPPRESS)      # (the default)
    args = ap.parse_args()
    export(load(args.model, args.max_len, args.quant, args.rope, args.attn, args.kv_scales, args.layers, args.kv_channel,
                kv_calib_arg(args.kv_calib)), args.out,
           args.prefill, args.prefill_all, False if args.static_len else None)


if __name__ == "__main__":
    main()

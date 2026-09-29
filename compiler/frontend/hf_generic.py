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
- the program runs with sa-llm-run --abi=hf; what the sa backend does not
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
        their KV rows written; the last one's logits, or (logits=False, the
        chunks before the last: prefill_kv) the last row's hidden state."""
        self.cache.positions = position_ids[0]
        for layer in self.cache.layers:
            layer.cumulative_length = position_ids.reshape(-1)[0]
        if logits:
            return self.model(input_ids=input_ids, position_ids=position_ids, past_key_values=self.cache,
                              use_cache=True, logits_to_keep=1).logits
        h = self.model.model(input_ids=input_ids, position_ids=position_ids, past_key_values=self.cache,
                             use_cache=True).last_hidden_state
        return h[:, -1]

    def forward(self, input_ids, position_ids):
        self.cache.positions = position_ids[0]
        # HF builds the causal mask from the cache's own length (the positions
        # seen so far): here the input position, set (not mutated) each step
        for layer in self.cache.layers:
            layer.cumulative_length = position_ids.reshape(-1)[0]
        return self.model(input_ids=input_ids, position_ids=position_ids, past_key_values=self.cache,
                          use_cache=True).logits


# ------------------------------------------------------------------ KV cache and attention (F4)
class ObservedCache(PositionedCache):
    """Calibration: the running max |k|, |v| per layer (after RoPE / QK-norm, as
    they are written into the cache)."""
    def __init__(self, config, max_cache_len, owner):
        super().__init__(config, max_cache_len, owner)
        self.amax = np.zeros((len(self.layers), 2), np.float32)
        for i, layer in enumerate(self.layers):
            upd = layer.update

            def observed(key_states, value_states, *a, _i=i, _upd=upd, **k):
                self.amax[_i, 0] = max(self.amax[_i, 0], float(key_states.abs().max()))
                self.amax[_i, 1] = max(self.amax[_i, 1], float(value_states.abs().max()))
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
        kv = np.asarray(kv_scales, np.float32)
        hd = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.inv_sk = [_f32c(1.0 / np.float32(x)) for x in kv[:, 0]]
        self.inv_sv = [_f32c(1.0 / np.float32(x)) for x in kv[:, 1]]
        self.a_k = [_f32c(float(x) / np.sqrt(hd)) for x in kv[:, 0]]
        self.a_v = [_f32c(float(x) / 127.0) for x in kv[:, 1]]
        cache, T = self, max_cache_len

        def make_update(i):
            def update(layer, key_states, value_states, *a, **k):
                kr = key_states[0].transpose(0, 1).reshape(key_states.shape[2], -1)     # (M, kv_heads * hd)
                vr = value_states[0].transpose(0, 1).reshape(value_states.shape[2], -1)
                kq, vq = _to_i8(kr * cache.inv_sk[i]), _to_i8(vr * cache.inv_sv[i])
                for m in range(kr.shape[0]):          # row by row: a scalar position each (a decode step's write)
                    row = cache.positions[m:m + 1] + i * T
                    owner.kc.index_copy_(0, row, kq[m:m + 1])
                    owner.vc.index_copy_(0, row, vq[m:m + 1])
                return owner.kc[i * T:(i + 1) * T], owner.vc[i * T:(i + 1) * T]
            return update
        for i, layer in enumerate(self.layers):
            layer.update = types.MethodType(make_update(i), layer)


def sa_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """The attention of the sa device (qhf.QModel.attention), registered with
    transformers' AttentionInterface as "sa": query [1, H, M, hd] (after RoPE;
    M = 1 in decode, the chunk's rows in prefill), key / value the layer's int8
    cache rows [T, kv_heads * hd] (SaCache); per query row (a decode step's
    form: its scalar position): q per head quantized, int32 scores, the prefix
    mask from the position (HF's mask is not used), softmax, int8
    probabilities, int32 P V."""
    cache = module.sa_cache
    l = module.layer_idx
    H, M, hd = query.shape[1], query.shape[2], query.shape[3]
    T = key.shape[0]
    Hk = key.shape[1] // hd
    G = H // Hk
    kh = key.to(torch.int32).reshape(T, Hk, hd).permute(1, 2, 0)                  # (Hk, hd, T)
    vh = value.to(torch.int32).reshape(T, Hk, hd).permute(1, 0, 2)                # (Hk, T, hd)
    outs = []
    for m in range(M):
        qq, s_q = _quant_act(query[0, :, m, :].reshape(Hk, G, hd))               # (Hk, G, hd), (Hk, G, 1)
        sc = torch.matmul(qq.to(torch.int32), kh)                                # (Hk, G, T) int32
        sc = (sc.to(torch.float32) * s_q) * cache.a_k[l]
        mask = torch.arange(T) <= cache.positions[m:m + 1]
        mx = torch.amax(torch.where(mask, sc, torch.tensor(float("-inf"))), dim=-1, keepdim=True)
        e = torch.where(mask, torch.exp(sc - mx), torch.tensor(0.0))
        r = torch.reciprocal(torch.sum(e, dim=-1, keepdim=True))
        pq = _to_i8((e * r) * 127.0).to(torch.int32)                             # (Hk, G, T)
        att = torch.matmul(pq, vh)                                               # (Hk, G, hd) int32
        outs.append((att.to(torch.float32) * cache.a_v[l]).reshape(1, 1, H, hd))
    return (outs[0] if M == 1 else torch.cat(outs, dim=1)), None


class SaDecoder(HFDecoder):
    """HFDecoder with the sa KV cache and attention (F4)."""
    def __init__(self, model, max_len, kv_scales):
        torch.nn.Module.__init__(self)
        from transformers import AttentionInterface
        AttentionInterface.register("sa", sa_attention)
        self.model = model
        cfg = model.config
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        kv_dim = cfg.num_key_value_heads * hd
        self.cache = SaCache(cfg, max_len, self, kv_scales)
        # K and V start with different contents (see SaCache; the rows are only
        # read after they are written: the attention masks later positions)
        L = len(self.cache.layers)
        self.register_buffer("kc", torch.zeros(L * max_len, kv_dim, dtype=torch.int8))
        self.register_buffer("vc", torch.ones(L * max_len, kv_dim, dtype=torch.int8))
        for mod in model.modules():
            if hasattr(mod, "layer_idx") and hasattr(mod, "q_proj"):
                mod.sa_cache = self.cache
        model.set_attn_implementation("sa")


def calibrate_kv(model, tokens, max_len):
    """Static per-layer KV scales: amax / 127 over the fp32 model on the tokens
    (qhf.calibrate_kv)."""
    obs = HFDecoder(model, max_len)
    obs.cache = ObservedCache(model.config, max_len, obs)
    hd = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads
    obs.cache.early_initialization(batch_size=1, num_heads=model.config.num_key_value_heads, head_dim=hd,
                                   dtype=torch.float32, device=torch.device("cpu"))
    for pos, t in enumerate(tokens[:max_len]):
        obs(torch.tensor([[t]]), torch.tensor([[pos]]))
    return (np.maximum(obs.cache.amax, 1e-8) / 127.0).astype(np.float32)


def load(model_dir, max_len=256, quant=False, rope=False, attn=False, kv_scales=None):
    torch.set_grad_enabled(False)
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="sdpa").eval()
    if rope:
        print("  rope_rewrite:", rope_rewrite(m, max_len))
        print("  norm_rewrite:", norm_rewrite(m))
    kv = None
    if attn and kv_scales is not None:    # given (e.g. qhf's, for a bit-exact comparison)
        kv = np.load(kv_scales).astype(np.float32)
    elif attn:                            # KV scales from the fp32 model (after the RoPE / norm rewrites)
        import export_hf
        from hf_tokenizer import BpeTokenizer
        kv = calibrate_kv(m, BpeTokenizer(model_dir).encode(export_hf.CALIB), max_len)
        print(f"  calibrate_kv: {len(kv)} layers")
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
        xq, s_x = _quant_act(x)
        # on rows [tokens, in] (qhf's form): a 3-D int32 matmul is exported as a
        # batch_matmul accumulating in i64, which the sa backend leaves to the host
        rows = xq.reshape(-1, xq.shape[-1])
        acc = torch.matmul(rows.to(torch.int32), self.wq.to(torch.int32).t())
        acc = acc.reshape(tuple(xq.shape[:-1]) + (acc.shape[-1],))
        y = (acc.to(torch.float32) * self.s_w) * s_x
        return y if self.bias is None else y + self.bias


class QEmbedding(torch.nn.Module):
    """int8 embedding rows with a per-row scale: e[t] = q[t] * s[t]."""
    def __init__(self, emb, qs=None):
        super().__init__()
        q, s = qs if qs is not None else _quantize_rows(emb.weight)
        self.q = torch.nn.Parameter(q, requires_grad=False)
        self.s = torch.nn.Parameter(s, requires_grad=False)

    def forward(self, ids):
        # row by row, each an index_select by one id: the sa backend's row gather
        # with a scalar index (prefill's M tokens: M gathers)
        flat = ids.reshape(-1)
        e = torch.cat([torch.index_select(self.q, 0, flat[m:m + 1]).to(torch.float32) *
                       torch.index_select(self.s, 0, flat[m:m + 1]).unsqueeze(-1) for m in range(flat.shape[0])])
        return e.reshape(tuple(ids.shape) + (self.q.shape[-1],))


# RoPE (F3): the rotary module's cos / sin as tables, and rotate_half as the
# pair swap of interleaved halves
class RopeTable(torch.nn.Module):
    """A rotary-embedding module replaced by its own output for every position:
    cos / sin tables [max_len, head_dim] computed once by the original module
    (any RoPE variant, frequency scaling included), then looked up by position."""
    def __init__(self, rotary, max_len, head_dim, perm=None):
        super().__init__()
        x = torch.zeros(1, 1, head_dim)
        cos, sin = rotary(x, torch.arange(max_len)[None])
        cos, sin = cos[0].float(), sin[0].float()
        if perm is not None:
            cos, sin = cos[:, perm], sin[:, perm]
        self.cos_t = torch.nn.Parameter(cos.contiguous(), requires_grad=False)
        self.sin_t = torch.nn.Parameter(sin.contiguous(), requires_grad=False)

    def forward(self, x, position_ids):
        # row by row, each an index_select by one position: the sa backend's
        # row gather with a scalar index (prefill's M positions: M gathers)
        shape = tuple(position_ids.shape) + (self.cos_t.shape[-1],)
        pos = position_ids.reshape(-1)
        rows = lambda t: torch.cat([torch.index_select(t, 0, pos[m:m + 1]) for m in range(pos.shape[0])])
        return rows(self.cos_t).reshape(shape).to(x.dtype), rows(self.sin_t).reshape(shape).to(x.dtype)


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
        self.weight = norm.weight
        eps = getattr(norm, "variance_epsilon", None)
        self.eps = _f32c(eps if eps is not None else norm.eps)

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


def quantize(model):
    """Every nn.Linear -> QLinear, nn.Embedding -> QEmbedding (W8A8, qhf's
    scheme). A weight shared by several modules (a classifier tied to the
    embedding) is quantized once and the int8 tensor shared (the packed-gather
    optimization of the sa backend reads it once, plan §8.12 C6.0)."""
    shared, n = {}, {"linear": 0, "embedding": 0}

    def qs_of(weight):
        key = weight.data_ptr()
        if key not in shared:
            shared[key] = _quantize_rows(weight)
        return shared[key]

    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear):
                q = QLinear(child, qs_of(child.weight))
                n["linear"] += 1
            elif isinstance(child, torch.nn.Embedding):
                q = QEmbedding(child, qs_of(child.weight))
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
    """One function of the module (decode, prefill, prefill_kv) as a module to export."""
    def __init__(self, w, kind):
        super().__init__()
        self.w, self.kind = w, kind

    def forward(self, input_ids, position_ids):
        if self.kind == "main":
            return self.w(input_ids, position_ids)
        return self.w.prefill(input_ids, position_ids, logits=self.kind == "prefill")


def export(w, out, prefill=0):
    """main (decode); with prefill = M also prefill / prefill_kv of M tokens, the
    three sharing the parameters and the KV cache (plan §8.13, §8.17)."""
    from iree.turbine import aot
    from iree.turbine.aot import decompositions
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    progs = {"main": 1} | ({"prefill": prefill, "prefill_kv": prefill} if prefill else {})
    eps = {}
    for name, M in progs.items():
        args = {"input_ids": torch.ones(1, M, dtype=torch.int64), "position_ids": torch.arange(M)[None]}
        ep = torch.export.export(_Fn(w, name), (), args, strict=False)
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
    args = ap.parse_args()
    export(load(args.model, args.max_len, args.quant, args.rope, args.attn, args.kv_scales), args.out, args.prefill)


if __name__ == "__main__":
    main()

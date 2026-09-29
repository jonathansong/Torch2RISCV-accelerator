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

    def forward(self, input_ids, position_ids):
        self.cache.positions = position_ids[0]
        return self.model(input_ids=input_ids, position_ids=position_ids, past_key_values=self.cache,
                          use_cache=True).logits


def load(model_dir, max_len=256, quant=False):
    torch.set_grad_enabled(False)
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="sdpa").eval()
    if quant:
        print("  quantize:", quantize(m))
    return HFDecoder(m, max_len)


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
        acc = torch.matmul(xq.to(torch.int32), self.wq.to(torch.int32).t())
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
        return self.q[ids].to(torch.float32) * self.s[ids].unsqueeze(-1)


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


def export(w, out):
    from iree.turbine import aot
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    args = {"input_ids": torch.tensor([[1]]), "position_ids": torch.tensor([[0]])}
    ep = torch.export.export(w, (), args, strict=False)
    # functional ATen (the cache writes as buffer mutations), with iree-turbine's
    # decomposition table (as aot.export(module) does: the frontend's usual op forms)
    from iree.turbine.aot import decompositions
    ep = ep.run_decompositions(decompositions.current_aot_decompositions())
    for p in GRAPH_PASSES:
        print(f"  {p.__name__}: {p(ep.graph)}")
    ep.graph.lint()
    aot.externalize_module_parameters(w, external_scope="model")   # (after: marked tensors break aot_autograd)
    exp = aot.export(ep)
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
    args = ap.parse_args()
    export(load(args.model, args.max_len, args.quant), args.out)


if __name__ == "__main__":
    main()

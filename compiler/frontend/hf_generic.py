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


def load(model_dir, max_len=256):
    torch.set_grad_enabled(False)
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="sdpa").eval()
    return HFDecoder(m, max_len)


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
    ep = ep.run_decompositions()              # functional ATen (the cache writes as buffer mutations)
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
    args = ap.parse_args()
    export(load(args.model, args.max_len), args.out)


if __name__ == "__main__":
    main()

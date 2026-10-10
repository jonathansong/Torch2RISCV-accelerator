#!/usr/bin/env python3
"""Imports a compressed-tensors W8A8 checkpoint (llm-compressor's output, as
vLLM reads it; docs/generality_implementation_plan.md WP2) for the generic
frontend (hf_generic.py): the checkpoint's int8 weights and per-channel scales
are used as they are, not quantized again.

Accepted (anything else is refused with the layer, the field and what is
supported; nothing falls back silently or is quantized again):
- quant_method compressed-tensors, format int-quantized, status compressed;
- weights: int, 8 bits, static, strategy channel, symmetric (a zero point, if
  stored, all zeros), no groups / blocks, act-order none or static (no g_idx);
- input activations: int, 8 bits, dynamic, strategy token, symmetric (the
  device's per-token quantization: s_x = amax / 127, as vLLM's dynamic int8);
  no output activation or KV cache scheme;
- the scales in fp32 / bf16 / fp16 (made fp32; the int8 values untouched,
  -128 included); `ignore` (e.g. lm_head) stays in the floating-point form
  given and is quantized later by the frontend like an unquantized model.

load_model() builds the transformers model in fp32 from the configuration
without its quantization_config: every quantized linear layer's weight is the
dequantized q * s (for the fp32 references and the KV calibration) and the
Parameter carries `_sa_qs = (q int8 [N, K], s fp32 [N])`, which
hf_generic.quantize() takes instead of quantizing the weight, and
hf_generic.rope_rewrite() permutes with the weight. The safetensors are read
tensor by tensor (mmap); the dequantized fp32 model is the only full copy.

    python3 compiler/frontend/ct_import.py <model dir>      (the scheme and a summary of the layers)
"""
import json
import os
import re
import sys

import torch


class UnsupportedCheckpoint(ValueError):
    pass


def quant_config(model_dir):
    """The quantization_config of config.json, or None (an unquantized model)."""
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f).get("quantization_config")


def _check_args(group, kind, a, want):
    """want: {field: expected value or a predicate}; the first mismatch raises."""
    if a is None:
        raise UnsupportedCheckpoint(f"config group {group}: no {kind} scheme (supported: {_describe(want)})")
    for k, v in want.items():
        got = a.get(k)
        ok = v(got) if callable(v) else got == v
        if not ok:
            raise UnsupportedCheckpoint(f"config group {group}: {kind}.{k} = {got!r}; supported: {_describe(want)}")


def _describe(want):
    return ", ".join(f"{k} {'(unset)' if v is _unset else '(unset or static)' if callable(v) else v}"
                     for k, v in want.items())


_unset = lambda v: v in (None, False, {})          # noqa: E731
# actorder "static" (GPTQ's column order, per channel): only the order GPTQ
# worked in, folded into the weights, no g_idx stored (checked per layer)
_static = lambda v: _unset(v) or v == "static"   # noqa: E731
WEIGHTS = {"type": "int", "num_bits": 8, "dynamic": False, "strategy": "channel", "symmetric": True,
           "group_size": _unset, "block_structure": _unset, "actorder": _static}
INPUTS = {"type": "int", "num_bits": 8, "dynamic": True, "strategy": "token", "symmetric": True,
          "group_size": _unset, "block_structure": _unset}


class Scheme:
    """The checked quantization_config: which Linear modules are quantized."""
    def __init__(self, qc):
        if qc.get("quant_method") != "compressed-tensors":
            raise UnsupportedCheckpoint(f"quant_method = {qc.get('quant_method')!r}; supported: compressed-tensors")
        if qc.get("format") != "int-quantized":
            raise UnsupportedCheckpoint(f"format = {qc.get('format')!r}; supported: int-quantized")
        if qc.get("quantization_status", "compressed") != "compressed":
            raise UnsupportedCheckpoint(f"quantization_status = {qc.get('quantization_status')!r}; supported: compressed")
        if qc.get("kv_cache_scheme"):
            raise UnsupportedCheckpoint("kv_cache_scheme is set; supported: none (the KV scales are calibrated here)")
        self.targets = []
        for name, g in (qc.get("config_groups") or {}).items():
            _check_args(name, "weights", g.get("weights"), WEIGHTS)
            _check_args(name, "input_activations", g.get("input_activations"), INPUTS)
            if g.get("output_activations"):
                raise UnsupportedCheckpoint(f"config group {name}: output_activations is set; supported: none")
            self.targets += g.get("targets") or []
        if not self.targets:
            raise UnsupportedCheckpoint("no config_groups with targets")
        self.ignore = qc.get("ignore") or []

    @staticmethod
    def _match(name, cls, pats):
        for p in pats:
            if p.startswith("re:"):
                if re.fullmatch(p[3:], name):
                    return True
            elif p == name or p == cls:
                return True
        return False

    def quantized(self, name, module):
        return (isinstance(module, torch.nn.Linear) and self._match(name, type(module).__name__, self.targets)
                and not self._match(name, type(module).__name__, self.ignore))


class Tensors:
    """The checkpoint's tensors by name, opened lazily (one or several safetensors)."""
    def __init__(self, model_dir):
        from safetensors import safe_open
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                files = sorted(set(json.load(f)["weight_map"].values()))
        else:
            files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
        if not files:
            raise UnsupportedCheckpoint(f"{model_dir}: no *.safetensors")
        self.where = {}
        self.files = [safe_open(os.path.join(model_dir, f), framework="pt") for f in files]
        for h in self.files:
            for k in h.keys():
                self.where[k] = h

    def __contains__(self, k):
        return k in self.where

    def keys(self):
        return self.where.keys()

    def get(self, k):
        return self.where[k].get_tensor(k)


def _scale(name, s, n):
    if s.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise UnsupportedCheckpoint(f"{name}.weight_scale: dtype {s.dtype}; supported: fp32, bf16, fp16")
    if s.numel() != n or (s.dim() == 2 and s.shape[1] != 1) or s.dim() > 2:
        raise UnsupportedCheckpoint(f"{name}.weight_scale: shape {list(s.shape)}; expected [{n}, 1] (per output channel)")
    return s.reshape(n).to(torch.float32)


def quantized_layer(t, name, shape):
    """(q int8 [N, K], s fp32 [N]) of one quantized linear layer, checked."""
    for k in (name + ".weight", name + ".weight_scale"):
        if k not in t:
            raise UnsupportedCheckpoint(f"{name}: tensor {k} missing")
    q = t.get(name + ".weight")
    if q.dtype != torch.int8:
        raise UnsupportedCheckpoint(f"{name}.weight: dtype {q.dtype}; supported: int8 (format int-quantized)")
    if list(q.shape) != list(shape):
        raise UnsupportedCheckpoint(f"{name}.weight: shape {list(q.shape)}; the model expects {list(shape)}")
    s = _scale(name, t.get(name + ".weight_scale"), shape[0])
    if name + ".weight_zero_point" in t and bool(torch.any(t.get(name + ".weight_zero_point") != 0)):
        raise UnsupportedCheckpoint(f"{name}.weight_zero_point: non-zero; supported: symmetric (zero point 0)")
    for k in ("weight_g_idx", "input_scale", "input_zero_point"):
        if f"{name}.{k}" in t:
            raise UnsupportedCheckpoint(f"{name}.{k} present; supported: no act-order, dynamic activations")
    return q, s


def load_model(model_dir, layers=None):
    """The transformers model in fp32, quantized linear layers dequantized with
    weight._sa_qs = (q, s) (layers: only the first ones). Returns (model, report)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    qc = quant_config(model_dir)
    scheme = Scheme(qc)
    cfg = AutoConfig.from_pretrained(model_dir)
    if hasattr(cfg, "quantization_config"):
        del cfg.quantization_config
    if layers:
        cfg.num_hidden_layers = layers
        if getattr(cfg, "layer_types", None):
            cfg.layer_types = cfg.layer_types[:layers]
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, attn_implementation="sdpa").eval()
    t = Tensors(model_dir)
    mods = dict(model.named_modules())
    params = dict(model.named_parameters(remove_duplicate=False))
    done, report = set(), {"quantized": [], "ignored": [], "other": 0}
    with torch.no_grad():
        for name, mod in mods.items():
            if not isinstance(mod, torch.nn.Linear):
                continue
            if not scheme.quantized(name, mod):
                if name + ".weight" in t:
                    report["ignored"].append(name)
                continue
            q, s = quantized_layer(t, name, mod.weight.shape)
            mod.weight.copy_(q.to(torch.float32) * s[:, None])
            mod.weight._sa_qs = (q, s)
            done |= {name + ".weight", name + ".weight_scale", name + ".weight_zero_point"}
            report["quantized"].append(name)
        for k in t.keys():
            if k in done:
                continue
            if k not in params:
                m = re.match(r"model\.layers\.(\d+)\.", k)
                if layers and m and int(m.group(1)) >= layers:
                    continue
                raise UnsupportedCheckpoint(f"tensor {k}: not a parameter of the model")
            v = t.get(k)
            if _tied(model, k):
                # a tied classifier also stored on its own (RedHatAI's Llama-3.2-1B
                # W8A8): it must be the embedding's table, else loading it into the
                # shared tensor would silently replace one of the two
                emb = next((e for e, p in params.items() if e != k and p is params[k]), None)
                if emb is None or emb not in t or not torch.equal(v, t.get(emb)):
                    raise UnsupportedCheckpoint(f"{k}: stored apart from the tied {emb} and different; supported: "
                                                "the same table (or tie_word_embeddings false)")
                done.add(k)
                continue
            if not v.is_floating_point():
                raise UnsupportedCheckpoint(f"{k}: dtype {v.dtype} in a layer the scheme does not quantize")
            if list(v.shape) != list(params[k].shape):
                raise UnsupportedCheckpoint(f"{k}: shape {list(v.shape)}; the model expects {list(params[k].shape)}")
            params[k].copy_(v.to(torch.float32))
            done.add(k)
            report["other"] += 1
    missing = [k for k in params if k not in done and not _tied(model, k)]
    if missing:
        raise UnsupportedCheckpoint(f"parameters missing from the checkpoint: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    model.tie_weights()
    return model, report


def _tied(model, k):
    """lm_head.weight tied to the embedding (one tensor, loaded under the embedding's name)."""
    return k == "lm_head.weight" and getattr(model.config, "tie_word_embeddings", False)


def main():
    d = sys.argv[1]
    qc = quant_config(d)
    if qc is None:
        print(f"{d}: no quantization_config (an unquantized model)")
        return 0
    Scheme(qc)
    m, r = load_model(d)
    print(f"{d}: compressed-tensors W8A8, {len(r['quantized'])} quantized linear layers, ignored {r['ignored']}, "
          f"{r['other']} other tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())

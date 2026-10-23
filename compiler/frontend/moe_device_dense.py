#!/usr/bin/env python3
"""Pack the dense part of a Mixtral checkpoint for the SA token compiler.

Query and key rows are permuted from Hugging Face's split-half RoPE order to
adjacent pairs. This changes their storage basis only: both the rotated query
and the cached key use the same permutation, so attention dot products agree.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "llm"))

import checkpoint  # noqa: E402
import export_w8a8 as EW  # noqa: E402
import moe_hf  # noqa: E402
from ref_model import quantize_rows  # noqa: E402

LAYER_NAMES = ("rms_att", "rms_ffn", "wqkv", "s_wqkv", "wo", "s_wo",
               "wrouter", "s_wrouter", "kvp")


def permute_rope_rows(weight, heads, head_size):
    """Apply the split-half -> adjacent-pair permutation to projection rows."""
    if weight.shape[0] != heads * head_size or head_size % 2:
        raise ValueError("invalid query/key projection shape for RoPE")
    return np.ascontiguousarray(weight.reshape(heads, 2, head_size // 2, -1)
                                .transpose(0, 2, 1, 3).reshape(weight.shape))


def rope_tables(cfg, theta):
    freq = np.float32(1.0) / np.power(np.float32(theta),
             np.arange(0, cfg.head_size, 2, dtype=np.float32) / np.float32(cfg.head_size))
    angles = np.arange(cfg.seq_len, dtype=np.float32)[:, None] * freq[None, :]
    cos = np.tile(np.repeat(np.cos(angles).astype(np.float32), 2, axis=1), (1, cfg.heads))
    sin = np.tile(np.repeat(np.sin(angles).astype(np.float32), 2, axis=1), (1, cfg.heads))
    return cos, sin


def tensors(model_dir, seq_len, d, kv_scale):
    raw = moe_hf._load_json(os.path.join(model_dir, "config.json"))
    hf = moe_hf.load_config(model_dir)
    if hf.adapter != "mixtral" or hf.top_k != 2:
        raise ValueError("SA dense decoder currently requires Mixtral top-2")
    heads = int(raw["num_attention_heads"])
    kv_heads = int(raw["num_key_value_heads"])
    vocab = int(raw["vocab_size"])
    if hf.experts > d or heads % kv_heads:
        raise ValueError("router must fit one D-wide output tile, and heads must group")
    cfg = checkpoint.Config(hf.dim, hf.hidden, hf.layers, heads, kv_heads, vocab, seq_len)
    if cfg.dim % d or cfg.head_size % d or seq_len % d or vocab % d:
        raise ValueError("model dimensions and context must be multiples of D")
    index = moe_hf.SafeTensorIndex(model_dir)
    f = lambda name: np.ascontiguousarray(index.get(name), dtype=np.float32)

    def linear(weight):
        q, scale = quantize_rows(weight)
        return EW.pack_b(q, d), scale

    emb_q, emb_s = quantize_rows(f("model.embed_tokens.weight"))
    wcls, s_wcls = linear(f("lm_head.weight"))
    cos, sin = rope_tables(cfg, float(raw.get("rope_theta", 10000.0)))
    glob = {"emb_q": emb_q, "emb_s": emb_s,
            "rms_final": f("model.norm.weight"), "wcls": wcls, "s_wcls": s_wcls,
            "rope_cos": cos, "rope_sin": sin}
    layers = []
    for layer in range(cfg.layers):
        root = f"model.layers.{layer}."
        att = root + "self_attn."
        q = permute_rope_rows(f(att + "q_proj.weight"), heads, cfg.head_size)
        k = permute_rope_rows(f(att + "k_proj.weight"), kv_heads, cfg.head_size)
        v = f(att + "v_proj.weight")
        wqkv, s_wqkv = linear(np.concatenate((q, k, v)))
        wo, s_wo = linear(f(att + "o_proj.weight"))
        router = f(root + "block_sparse_moe.gate.weight")
        padded = np.zeros((d, cfg.dim), np.float32)
        padded[:hf.experts] = router
        wrouter, s_wrouter = linear(padded)
        # The route firmware consumes only the first `experts` scores.
        kvp = EW.kv_params(cfg, kv_scale, kv_scale)
        layers.append({"rms_att": f(root + "input_layernorm.weight"),
                       "rms_ffn": f(root + "post_attention_layernorm.weight"),
                       "wqkv": wqkv, "s_wqkv": s_wqkv, "wo": wo, "s_wo": s_wo,
                       "wrouter": wrouter, "s_wrouter": s_wrouter, "kvp": kvp})
    return cfg, glob, layers


def export(model_dir, path, seq_len=16, d=16, kv_scale=0.02):
    cfg, glob, layers = tensors(model_dir, seq_len, d, kv_scale)
    names = list(glob) + list(LAYER_NAMES)
    entries = []
    offset = EW._align(64 + EW.ENT.size * len(names))
    for name, value in glob.items():
        entries.append((name, offset, value.nbytes, 0))
        offset = EW._align(offset + value.nbytes)
    layer_base, relative = offset, 0
    for name in LAYER_NAMES:
        entries.append((name, relative, layers[0][name].nbytes, 1))
        relative = EW._align(relative + layers[0][name].nbytes)
    total = layer_base + relative * cfg.layers
    with open(path, "wb") as output:
        output.truncate(total)
        header = EW.HDR.pack(EW.MAGIC, EW.VERSION, d, cfg.dim, cfg.hidden, cfg.layers,
                             cfg.heads, cfg.kv_heads, cfg.vocab, cfg.seq_len, len(entries),
                             layer_base, relative)
        output.seek(0)
        output.write(header)
        for i, (name, off, size, per_layer) in enumerate(entries):
            output.seek(64 + i * EW.ENT.size)
            output.write(EW.ENT.pack(name.encode().ljust(16, b"\0"), off, size, per_layer, 0))
            if per_layer:
                for layer in range(cfg.layers):
                    value = layers[layer][name]
                    if value.nbytes != size:
                        raise ValueError(f"layer {layer} has inconsistent {name} shape")
                    output.seek(layer_base + layer * relative + off)
                    output.write(value.tobytes())
            else:
                output.seek(off)
                output.write(glob[name].tobytes())
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="build/llm_cache/TitanML-tiny-mixtral")
    parser.add_argument("--out", default="build/moe/m4f/mixtral_dense.w8a8")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--kv-scale", type=float, default=0.02)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    size = export(args.model, args.out, args.seq_len, args.d, args.kv_scale)
    print(f"{args.out}: {size} bytes of dense SA parameters")


if __name__ == "__main__":
    main()

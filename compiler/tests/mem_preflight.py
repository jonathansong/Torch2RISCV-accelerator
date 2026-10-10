#!/usr/bin/env python3
"""Device memory preflight of a model for the generic frontend
(docs/generality_implementation_plan.md §9.3, WP5): from the checkpoint's
safetensors header only (nothing loaded), what the board's accelerator pool
must hold, item by item, against the pool:

  the int8 linear weights (a compressed-tensors checkpoint's own, else
  quantized here: one byte per weight), their fp32 scales (one per output
  row), the embedding table quantized here (int8 + a scale per row; shared with
  a tied classifier: once), the other fp32 tensors (norms, biases), the int8 KV
  cache (layers x T x kv_heads x head_dim, K and V) and a margin for the
  runtime's work buffers.

Calibrated on the board (4-port overlay, 2026-10-10): the runtime's peak is
the packed parameters (sa_packed.irpa) plus the KV cache plus a few MiB
(Qwen3-0.6B 582.4 MiB packed -> 595.4 MiB peak with 14 MiB of KV; SmolLM2
131.1 -> 136.1 / 138.3). The items below sum to 2.0% / 1.5% less than the
packed files (constants such as the RoPE tables and the KV scales, padding):
the estimate takes the items x 1.025.

    python3 compiler/tests/mem_preflight.py build/llm_cache/Llama-3.2-1B-Instruct-W8A8 [--t 256] [--pool-mib 1280]
"""
import argparse
import json
import os
import struct
import sys

SIZE = {"I8": 1, "U8": 1, "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def tensors(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else \
        sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    out = {}
    for f in files:
        out.update(header(os.path.join(model_dir, f)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model")
    ap.add_argument("--t", type=int, default=256, help="the KV cache's length (the export's max_len)")
    ap.add_argument("--pool-mib", type=float, default=1280, help="the accelerator pool (K5-M M1: 1280 MiB)")
    ap.add_argument("--margin-mib", type=float, default=32, help="work buffers and allocator slack")
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    t = tensors(args.model)
    mib = lambda b: b / 2**20                                         # noqa: E731
    lin_q = lin_s = emb = emb_s = other = 0
    tied = cfg.get("tie_word_embeddings", False)
    for k, v in t.items():
        n = 1
        for x in v["shape"]:
            n *= x
        if k.endswith(".weight_scale") or k.endswith(".weight_zero_point"):
            continue
        if k.endswith("embed_tokens.weight"):
            emb += n                                                  # int8 here
            emb_s += 4 * v["shape"][0]
        elif k == "lm_head.weight":
            if not tied:
                lin_q += n
                lin_s += 4 * v["shape"][0]
        elif k.endswith(".weight") and len(v["shape"]) == 2:
            lin_q += n                                                # int8 (imported or quantized here)
            lin_s += 4 * v["shape"][0]
        else:
            other += 4 * n                                            # norms, biases: fp32
    L = cfg["num_hidden_layers"]
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    kv = 2 * L * args.t * cfg["num_key_value_heads"] * hd
    rows = [("int8 linear weights", lin_q), ("their scales (fp32 per row)", lin_s),
            (f"embedding int8{' (shared with the tied classifier)' if tied else ''}", emb), ("its scales", emb_s),
            ("other fp32 (norms, biases)", other)]
    packed = sum(b for _, b in rows)
    rows += [("constants, padding (2.5%)", 0.025 * packed)]
    packed *= 1.025
    rows += [(f"int8 KV cache (T = {args.t})", kv), ("margin (work buffers)", args.margin_mib * 2**20)]
    total = sum(b for _, b in rows)
    print(f"{args.model}: {L} layers, kv_heads {cfg['num_key_value_heads']} x head_dim {hd}, vocab {cfg['vocab_size']}, "
          f"tied {tied}")
    for name, b in rows:
        print(f"  {name:45s} {mib(b):9.1f} MiB")
    print(f"  {'packed parameters (sa_packed.irpa, about)':45s} {mib(packed):9.1f} MiB")
    print(f"  {'total':45s} {mib(total):9.1f} MiB  of the {args.pool_mib:.0f} MiB pool: "
          f"{'FITS' if mib(total) <= args.pool_mib else 'DOES NOT FIT'} ({args.pool_mib - mib(total):+.1f} MiB)")
    tmax = int((args.pool_mib * 2**20 - packed - args.margin_mib * 2**20) // (2 * L * cfg["num_key_value_heads"] * hd))
    print(f"  the longest KV cache that fits: T = {tmax}")
    return 0 if mib(total) <= args.pool_mib else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Capacity preflight for a Hugging Face sparse-MoE checkpoint.

Estimates the packed accelerator footprint from safetensors headers, without
loading tensor values. Dense and expert linear weights are modeled as W8 with
one fp32 scale per output row; embeddings are modeled as int8 with one fp32
scale per vocabulary row; norms and other tensors are modeled as fp32. KV is
int8 with the same packed-V padding used by the generic LLM frontend.
"""

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "compiler", "frontend"))

import moe_hf  # noqa: E402


ALIGNMENT = 64
D = 16
V_PAD = 16


def _align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _linear_sizes(shape):
    if len(shape) != 2:
        return None
    rows, cols = (int(x) for x in shape)
    packed_rows = _align(rows, D)
    weight = packed_rows * cols
    scales = packed_rows * 4
    return weight, scales, _align(weight, ALIGNMENT) + _align(scales, ALIGNMENT)


def estimate(model_dir, *, kv_len=2048, pool_mib=1280, margin_mib=64):
    if kv_len <= 0 or not math.isfinite(pool_mib) or pool_mib <= 0:
        raise ValueError("KV length and pool size must be positive and finite")
    if not math.isfinite(margin_mib) or margin_mib < 0:
        raise ValueError("Runtime margin must be nonnegative and finite")
    config = moe_hf.load_config(model_dir)
    index = moe_hf.SafeTensorIndex(model_dir)
    raw = moe_hf._load_json(os.path.join(model_dir, "config.json"))
    tied = bool(raw.get("tie_word_embeddings", False))
    category_bytes = {
        "dense_linear_weights_w8": 0, "dense_linear_scales_f32": 0,
        "expert_weights_w8": 0, "expert_scales_f32": 0,
        "router_weights_w8": 0, "router_scales_f32": 0,
        "embedding_weights_i8": 0, "embedding_scales_f32": 0,
        "other_fp32": 0,
    }
    padded_parameter_bytes = 0
    expert_weight_total = 0
    expert_scale_total = 0
    seen_embedding = False

    for name, (_, info) in index.entries.items():
        shape = tuple(int(x) for x in info["shape"])
        if info["dtype"] not in ("F32", "F16", "BF16"):
            raise ValueError(f"{name}: expected floating-point checkpoint weights")
        if name.endswith("embed_tokens.weight"):
            if seen_embedding:
                continue
            seen_embedding = True
            vocab, dim = shape
            n = _align(vocab, D) * dim
            s = _align(vocab, D) * 4
            category_bytes["embedding_weights_i8"] += n
            category_bytes["embedding_scales_f32"] += s
            padded_parameter_bytes += _align(n, ALIGNMENT) + _align(s, ALIGNMENT)
            continue
        if name == "lm_head.weight" and tied:
            continue
        linear = _linear_sizes(shape)
        if linear is None:
            category_bytes["other_fp32"] += math.prod(shape) * 4
            padded_parameter_bytes += _align(math.prod(shape) * 4, ALIGNMENT)
            continue
        weight_bytes, scale_bytes, packed_bytes = linear
        if ".experts." in name:
            prefix = "expert"
            expert_weight_total += weight_bytes
            expert_scale_total += scale_bytes
        elif name.endswith(".block_sparse_moe.gate.weight") or ".mlp.gate.weight" in name:
            prefix = "router"
        else:
            prefix = "dense_linear"
        category_bytes[prefix + "_weights_w8"] += weight_bytes
        category_bytes[prefix + "_scales_f32"] += scale_bytes
        padded_parameter_bytes += packed_bytes

    layers = config.layers
    kv_heads = int(raw["num_key_value_heads"])
    head_dim = int(raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"])
    kv_dim = kv_heads * head_dim
    kv_bytes = 2 * layers * kv_len * kv_dim + layers * V_PAD * kv_dim
    margin_bytes = int(margin_mib * 2**20)
    # Reserve an estimated 2.5% packing allowance. This is not a measured
    # full-model allocation or a guarantee about executable/scratch sizes.
    parameter_estimate = math.ceil(padded_parameter_bytes * 1.025)
    required = parameter_estimate + kv_bytes + margin_bytes
    pool_bytes = int(pool_mib * 2**20)
    per_token_kv = 2 * layers * kv_dim
    max_kv = max(0, (pool_bytes - parameter_estimate - margin_bytes - layers * V_PAD * kv_dim) // per_token_kv)
    bytes_per_expert_layer = (expert_weight_total + expert_scale_total) // layers
    active_expert_layer = bytes_per_expert_layer * config.top_k // config.experts
    active_experts_model = active_expert_layer * layers
    return {
        "model": os.path.basename(os.path.abspath(model_dir)),
        "architecture": config.architecture,
        "adapter": config.adapter,
        "layers": layers,
        "dim": config.dim,
        "expert_hidden": config.hidden,
        "experts": config.experts,
        "top_k": config.top_k,
        "head_dim": head_dim,
        "kv_heads": kv_heads,
        "kv_len": kv_len,
        "pool_mib": pool_mib,
        "parameter_estimate_mib": parameter_estimate / 2**20,
        "kv_cache_mib": kv_bytes / 2**20,
        "runtime_margin_mib": margin_mib,
        "required_mib": required / 2**20,
        "remaining_mib": (pool_bytes - required) / 2**20,
        "fits": required <= pool_bytes,
        "max_kv_len": int(max_kv),
        "model_max_position_embeddings": raw.get("max_position_embeddings"),
        "estimate_only": True,
        "assumptions": ["W8 linear weights and per-row fp32 scales",
                        "int8 embeddings and KV cache", "2.5% packing allowance",
                        "runtime scratch covered by user-selected margin",
                        "pool capacity must be checked on the board"],
        "expert_table_per_layer_mib": bytes_per_expert_layer / 2**20,
        "active_expert_bytes_per_token_model_mib": active_experts_model / 2**20,
        "category_mib": {name: size / 2**20 for name, size in category_bytes.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="Hugging Face Mixtral/Qwen2-MoE checkpoint directory")
    parser.add_argument("--kv-len", type=int, default=2048)
    parser.add_argument("--pool-mib", type=float, default=1280)
    parser.add_argument("--margin-mib", type=float, default=64)
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    parser.add_argument("--out", help="write a JSON report")
    args = parser.parse_args()
    try:
        result = estimate(args.model, kv_len=args.kv_len, pool_mib=args.pool_mib, margin_mib=args.margin_mib)
    except ValueError as exc:
        parser.error(str(exc))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['model']}: {result['architecture']}, {result['layers']} layers, "
              f"{result['experts']} experts/top-{result['top_k']}, dim={result['dim']}, "
              f"expert_hidden={result['expert_hidden']}")
        for category, mib in result["category_mib"].items():
            print(f"  {category:32s} {mib:9.1f} MiB")
        print(f"  packed parameter estimate       {result['parameter_estimate_mib']:9.1f} MiB")
        print(f"  int8 KV cache (T={args.kv_len})          {result['kv_cache_mib']:9.1f} MiB")
        print(f"  runtime margin                  {result['runtime_margin_mib']:9.1f} MiB")
        print(f"  total required                  {result['required_mib']:9.1f} MiB")
        print(f"  pool available                  {args.pool_mib:9.1f} MiB")
        print(f"  remaining                       {result['remaining_mib']:+9.1f} MiB")
        print(f"  memory-only KV length limit     {result['max_kv_len']:9d}")
        print(f"  model max positions             {result['model_max_position_embeddings']}")
        print(f"  expert table per layer          {result['expert_table_per_layer_mib']:9.1f} MiB")
        print(f"  active expert bytes/token       {result['active_expert_bytes_per_token_model_mib']:9.1f} MiB")
        print("  estimated capacity: " + ("FITS" if result["fits"] else "DOES NOT FIT"))
    return 0 if result["fits"] else 1


if __name__ == "__main__":
    sys.exit(main())

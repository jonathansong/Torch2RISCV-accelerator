#!/usr/bin/env python3
"""Creates a small Hugging Face-compatible Mixtral checkpoint for M4 tests."""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from synth_hf import write_safetensors  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()
    if args.dim % 16 or args.hidden % 16:
        raise ValueError("dim and hidden must be multiples of 16")
    os.makedirs(args.out, exist_ok=True)
    config = {"architectures": ["MixtralForCausalLM"], "model_type": "mixtral",
              "hidden_size": args.dim, "intermediate_size": args.hidden,
              "num_hidden_layers": args.layers, "num_local_experts": args.experts,
              "num_experts_per_tok": args.top_k, "norm_topk_prob": True}
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    rng = np.random.default_rng(args.seed)
    linear = lambda shape, fan_in: rng.normal(0, 1 / np.sqrt(fan_in), shape).astype(np.float32)
    tensors = {}
    for layer in range(args.layers):
        root = f"model.layers.{layer}.block_sparse_moe"
        tensors[root + ".gate.weight"] = linear((args.experts, args.dim), args.dim)
        for expert in range(args.experts):
            p = f"{root}.experts.{expert}"
            tensors[p + ".w1.weight"] = linear((args.hidden, args.dim), args.dim)
            tensors[p + ".w3.weight"] = linear((args.hidden, args.dim), args.dim)
            tensors[p + ".w2.weight"] = linear((args.dim, args.hidden), args.hidden)
    write_safetensors(os.path.join(args.out, "model.safetensors"), tensors)
    print(f"{args.out}: tiny Mixtral schema, {args.layers} layers, {args.experts} experts/top-{args.top_k}, "
          f"dim {args.dim}, hidden {args.hidden}")


if __name__ == "__main__":
    main()

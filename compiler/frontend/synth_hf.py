#!/usr/bin/env python3
"""A synthetic HuggingFace decoder directory (config.json, model.safetensors,
tokenizer.json) with random weights: shapes the real models do not have yet
on this board (a K beyond one SPAD bank, ...), for export_hf.py / test_c55.py.

    python3 compiler/frontend/synth_hf.py --out build/llm_cache/synth-k16k --tokenizer build/llm_cache/SmolLM2-135M \\
        --dim 256 --hidden 16384 --layers 1 --heads 4 --kv-heads 2
"""
import argparse
import json
import os
import shutil

import numpy as np


def write_safetensors(path, tensors):
    """F32 tensors: 8-byte header length, JSON header, raw little-endian data."""
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        b = np.ascontiguousarray(a, np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(len(h).to_bytes(8, "little"))
        f.write(h)
        for b in blobs:
            f.write(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True, help="a directory with tokenizer.json (its vocabulary size is used)")
    ap.add_argument("--arch", default="LlamaForCausalLM", choices=["LlamaForCausalLM", "Qwen3ForCausalLM"])
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=16384)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    shutil.copy(os.path.join(args.tokenizer, "tokenizer.json"), os.path.join(args.out, "tokenizer.json"))
    tj = json.load(open(os.path.join(args.out, "tokenizer.json")))
    vocab = max(tj["model"]["vocab"].values()) + 1
    vocab = max([vocab] + [t["id"] + 1 for t in tj.get("added_tokens", [])])
    hs = args.head_dim or args.dim // args.heads
    cfg = {"architectures": [args.arch], "hidden_size": args.dim, "intermediate_size": args.hidden,
           "num_hidden_layers": args.layers, "num_attention_heads": args.heads,
           "num_key_value_heads": args.kv_heads, "head_dim": hs, "vocab_size": vocab, "rope_theta": 10000.0,
           "rms_norm_eps": 1e-5, "tie_word_embeddings": True, "max_position_embeddings": 2048}
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    rng = np.random.default_rng(args.seed)
    # scaled like a trained model's init: the residual stream stays O(1)
    lin = lambda n, k: rng.normal(0, 1 / np.sqrt(k), (n, k)).astype(np.float32)
    norm = lambda n: (1 + 0.1 * rng.normal(size=n)).astype(np.float32)
    t = {"model.embed_tokens.weight": rng.normal(0, 1, (vocab, args.dim)).astype(np.float32),
         "model.norm.weight": norm(args.dim)}
    for l in range(args.layers):
        p = f"model.layers.{l}."
        t[p + "self_attn.q_proj.weight"] = lin(args.heads * hs, args.dim)
        t[p + "self_attn.k_proj.weight"] = lin(args.kv_heads * hs, args.dim)
        t[p + "self_attn.v_proj.weight"] = lin(args.kv_heads * hs, args.dim)
        t[p + "self_attn.o_proj.weight"] = lin(args.dim, args.heads * hs)
        t[p + "mlp.gate_proj.weight"] = lin(args.hidden, args.dim)
        t[p + "mlp.up_proj.weight"] = lin(args.hidden, args.dim)
        t[p + "mlp.down_proj.weight"] = lin(args.dim, args.hidden)
        t[p + "input_layernorm.weight"] = norm(args.dim)
        t[p + "post_attention_layernorm.weight"] = norm(args.dim)
        if args.arch.startswith("Qwen3"):
            t[p + "self_attn.q_norm.weight"] = norm(hs)
            t[p + "self_attn.k_norm.weight"] = norm(hs)
    write_safetensors(os.path.join(args.out, "model.safetensors"), t)
    print(f"{args.out}: {cfg}")


if __name__ == "__main__":
    main()

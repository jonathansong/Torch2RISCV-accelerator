#!/usr/bin/env python3
"""A synthetic HuggingFace decoder directory (config.json, model.safetensors,
tokenizer.json) with random weights: shapes the real models do not have yet
on this board (a K beyond one SPAD bank, ...), for export_hf.py / test_c55.py.
With --ct VARIANT a compressed-tensors W8A8 checkpoint of the same weights
(llm-compressor's format, docs/generality_implementation_plan.md WP2): the
linear layers int8 per output channel with a weight_scale, lm_head ignored
(tied to the embedding); the variants the importer accepts (ok, neg128,
bf16-scale) and the ones it must refuse (ct_import.py).

    python3 compiler/frontend/synth_hf.py --out build/llm_cache/synth-k16k --tokenizer build/llm_cache/SmolLM2-135M \\
        --dim 256 --hidden 16384 --layers 1 --heads 4 --kv-heads 2
"""
import argparse
import json
import os
import shutil

import numpy as np


CT_OK = ("ok", "neg128", "bf16-scale", "lm-head-dup")
CT_BAD = ("group", "static-act", "asym", "int4", "fp8", "missing-scale", "bad-shape", "kv-scheme", "lm-head-diff")


class BF16(np.ndarray):
    """fp32 values stored as bfloat16 (round to nearest even)."""


def _bf16_bytes(a):
    u = np.ascontiguousarray(a, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).tobytes()


def write_safetensors(path, tensors):
    """F32 / I8 / BF16 tensors: 8-byte header length, JSON header, raw little-endian data."""
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        if isinstance(a, BF16):
            dt, b = "BF16", _bf16_bytes(np.asarray(a))
        elif a.dtype == np.int8:
            dt, b = "I8", np.ascontiguousarray(a).tobytes()
        else:
            dt, b = "F32", np.ascontiguousarray(a, np.float32).tobytes()
        header[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
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
    ap.add_argument("--ct", choices=CT_OK + CT_BAD, help="a compressed-tensors W8A8 checkpoint (variant)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    shutil.copy(os.path.join(args.tokenizer, "tokenizer.json"), os.path.join(args.out, "tokenizer.json"))
    tj = json.load(open(os.path.join(args.out, "tokenizer.json")))
    vocab = max(tj["model"]["vocab"].values()) + 1
    vocab = max([vocab] + [t["id"] + 1 for t in tj.get("added_tokens", [])])
    hs = args.head_dim or args.dim // args.heads
    cfg = {"architectures": [args.arch], "model_type": {"LlamaForCausalLM": "llama", "Qwen3ForCausalLM": "qwen3"}[args.arch],
           "hidden_size": args.dim, "intermediate_size": args.hidden,
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
    if args.ct:
        t = compress(t, cfg, args.ct, rng)
        json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    write_safetensors(os.path.join(args.out, "model.safetensors"), t)
    print(f"{args.out}: {cfg}")


def compress(t, cfg, variant, rng):
    """The linear layers' weights -> int8 rows + weight_scale (amax / 127.5 and
    [-128, 127], as llm-compressor's minmax observer: -128 occurs), and the
    quantization_config; the variant breaks one thing (CT_BAD) or exercises one."""
    w_args = {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel", "dynamic": False,
              "group_size": None, "block_structure": None, "actorder": None, "observer": "minmax"}
    a_args = {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "token", "dynamic": True,
              "group_size": None, "block_structure": None, "observer": None}
    qc = {"quant_method": "compressed-tensors", "format": "int-quantized", "quantization_status": "compressed",
          "config_groups": {"group_0": {"targets": ["Linear"], "weights": w_args, "input_activations": a_args,
                                        "output_activations": None}},
          "ignore": ["lm_head"], "kv_cache_scheme": None}
    out = {}
    for k, w in t.items():
        if not (k.endswith("_proj.weight") and w.ndim == 2):
            out[k] = w
            continue
        s = np.abs(w).max(axis=1) / 127.5
        q = np.clip(np.rint(w / s[:, None]), -128, 127).astype(np.int8)
        if variant == "neg128":            # every row reaches -128
            q[:, 0] = -128
        name = k[:-len(".weight")]
        out[k] = q
        sc = s[:, None].astype(np.float32)
        out[name + ".weight_scale"] = sc.view(BF16) if variant == "bf16-scale" else sc
    first = next(k for k in out if k.endswith("q_proj.weight_scale"))
    if variant == "group":
        w_args.update(strategy="group", group_size=128)
    elif variant == "static-act":
        a_args.update(dynamic=False, strategy="tensor")
    elif variant == "asym":
        w_args.update(symmetric=False)
    elif variant == "int4":
        w_args.update(num_bits=4)
    elif variant == "fp8":
        w_args.update(type="float")
        qc["format"] = "float-quantized"
    elif variant == "missing-scale":
        del out[first]
    elif variant == "bad-shape":
        out[first] = out[first][:-1]
    elif variant in ("lm-head-dup", "lm-head-diff"):
        # the tied classifier also stored on its own (as RedHatAI's Llama-3.2-1B W8A8)
        e = out["model.embed_tokens.weight"]
        out["lm_head.weight"] = e.copy() if variant == "lm-head-dup" else e + np.float32(0.01)
    elif variant == "kv-scheme":
        qc["kv_cache_scheme"] = {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "tensor"}
    cfg["quantization_config"] = qc
    return out


if __name__ == "__main__":
    main()

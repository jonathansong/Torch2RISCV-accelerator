#!/usr/bin/env python3
"""Extract only the dense Mixtral tensors needed by the host decoder."""

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "compiler", "frontend"))

import moe_hf  # noqa: E402


def required_names(config):
    names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for layer in range(config.layers):
        prefix = f"model.layers.{layer}."
        names.extend(prefix + item for item in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "block_sparse_moe.gate.weight"))
    return names


def pack(model_dir, out_dir):
    if os.path.exists(out_dir):
        raise FileExistsError(f"output already exists: {out_dir}")
    config = moe_hf.load_config(model_dir)
    if config.adapter != "mixtral":
        raise ValueError("dense checkpoint extraction currently supports Mixtral only")
    source = moe_hf.SafeTensorIndex(model_dir)
    selected = required_names(config)
    missing = [name for name in selected if name not in source]
    if missing:
        raise ValueError("missing dense checkpoint tensors: " + ", ".join(missing))
    metadata = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name in selected:
        _, info = source.entries[name]
        begin, end = map(int, info["data_offsets"])
        size = end - begin
        metadata[name] = {"dtype": info["dtype"], "shape": info["shape"],
                          "data_offsets": [offset, offset + size]}
        offset += size
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 8)
    os.makedirs(out_dir)
    shutil.copy2(os.path.join(model_dir, "config.json"), os.path.join(out_dir, "config.json"))
    path = os.path.join(out_dir, "model.safetensors")
    with open(path, "wb") as dst:
        dst.write(len(header).to_bytes(8, "little"))
        dst.write(header)
        for name in selected:
            filename, info = source.entries[name]
            src_path, data_start = source.headers[filename]
            begin, end = map(int, info["data_offsets"])
            remaining = end - begin
            with open(src_path, "rb") as src:
                src.seek(data_start + begin)
                while remaining:
                    chunk = src.read(min(1 << 20, remaining))
                    if not chunk:
                        raise IOError(f"short read in {src_path} for {name}")
                    dst.write(chunk)
                    remaining -= len(chunk)
    packed = moe_hf.SafeTensorIndex(out_dir)
    if set(packed.entries) != set(selected):
        raise RuntimeError("dense checkpoint tensor index mismatch")
    print(f"{out_dir}: {len(selected)} dense tensors, {os.path.getsize(path)} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="build/llm_cache/TitanML-tiny-mixtral")
    parser.add_argument("--out", default="build/moe/m4e/model_dense")
    args = parser.parse_args()
    pack(args.model, args.out)


if __name__ == "__main__":
    main()

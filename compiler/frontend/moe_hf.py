#!/usr/bin/env python3
"""Hugging Face sparse-MoE checkpoint adapter and SA parameter packer.

M4 initially supports the Mixtral and Qwen2-MoE tensor schemas. It reads only
the requested layer from unsharded or sharded safetensors checkpoints, maps the
framework names to the common gate/up/down representation in moe.py, quantizes
the layer to W8, and writes the sa-moe-parameters-v1 bundle used by M0-M3b.
"""

import argparse
from dataclasses import dataclass
import json
import os

import numpy as np

import moe


class UnsupportedMoeCheckpoint(ValueError):
    pass


@dataclass(frozen=True)
class HFMoeConfig:
    architecture: str
    model_type: str
    dim: int
    hidden: int
    experts: int
    top_k: int
    layers: int
    adapter: str
    normalize_topk: bool = True

    def sa_config(self, d=16, alignment=64):
        return moe.MoeConfig(self.dim, self.hidden, self.experts, self.top_k, d, alignment)


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsupportedMoeCheckpoint(f"cannot read {path}: {exc}") from exc


def load_config(model_dir):
    raw = _load_json(os.path.join(model_dir, "config.json"))
    architecture = (raw.get("architectures") or [""])[0]
    model_type = raw.get("model_type", "")
    if model_type == "mixtral" or architecture.startswith("Mixtral"):
        adapter = "mixtral"
        hidden = raw.get("intermediate_size")
        experts = raw.get("num_local_experts")
        top_k = raw.get("num_experts_per_tok")
        normalize = bool(raw.get("norm_topk_prob", True))
    elif model_type == "qwen2_moe" or architecture.startswith("Qwen2Moe"):
        adapter = "qwen2_moe"
        hidden = raw.get("moe_intermediate_size")
        experts = raw.get("num_experts")
        top_k = raw.get("num_experts_per_tok")
        normalize = bool(raw.get("norm_topk_prob", True))
    else:
        raise UnsupportedMoeCheckpoint(
            f"unsupported MoE architecture {architecture!r}, model_type {model_type!r}; "
            "supported: MixtralForCausalLM and Qwen2MoeForCausalLM")
    required = {"hidden_size": raw.get("hidden_size"), "expert hidden size": hidden,
                "expert count": experts, "experts per token": top_k,
                "layer count": raw.get("num_hidden_layers")}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise UnsupportedMoeCheckpoint("missing config fields: " + ", ".join(missing))
    return HFMoeConfig(architecture, model_type, int(required["hidden_size"]), int(hidden), int(experts),
                       int(top_k), int(required["layer count"]), adapter, normalize)


class SafeTensorIndex:
    """Minimal lazy NumPy reader for one or several safetensors files."""

    DTYPES = {"F32": np.dtype("<f4"), "F16": np.dtype("<f2"), "BF16": np.dtype("<u2")}

    def __init__(self, model_dir):
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            weight_map = _load_json(index_path).get("weight_map", {})
            files = sorted(set(weight_map.values()))
        else:
            files = sorted(name for name in os.listdir(model_dir) if name.endswith(".safetensors"))
            weight_map = None
        if not files:
            raise UnsupportedMoeCheckpoint(f"{model_dir}: no safetensors files")
        self.entries = {}
        self.headers = {}
        for filename in files:
            path = os.path.join(model_dir, filename)
            try:
                with open(path, "rb") as f:
                    header_size = int.from_bytes(f.read(8), "little")
                    header = json.loads(f.read(header_size))
            except (OSError, json.JSONDecodeError) as exc:
                raise UnsupportedMoeCheckpoint(f"cannot read {path}: {exc}") from exc
            data_start = 8 + header_size
            self.headers[filename] = (path, data_start)
            for name, info in header.items():
                if name != "__metadata__":
                    self.entries[name] = (filename, info)
        if weight_map is not None:
            for name, filename in weight_map.items():
                if name not in self.entries or self.entries[name][0] != filename:
                    raise UnsupportedMoeCheckpoint(f"safetensors index mismatch for {name}")

    def __contains__(self, name):
        return name in self.entries

    def get(self, name):
        if name not in self.entries:
            raise UnsupportedMoeCheckpoint(f"tensor {name} is missing")
        filename, info = self.entries[name]
        dtype_name = info["dtype"]
        if dtype_name not in self.DTYPES:
            raise UnsupportedMoeCheckpoint(f"{name}: unsupported dtype {dtype_name}; expected F32/F16/BF16")
        shape = tuple(int(x) for x in info["shape"])
        begin, end = (int(x) for x in info["data_offsets"])
        dtype = self.DTYPES[dtype_name]
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if end - begin != expected:
            raise UnsupportedMoeCheckpoint(f"{name}: byte size does not match shape {shape}")
        path, data_start = self.headers[filename]
        value = np.memmap(path, mode="r", dtype=dtype, offset=data_start + begin, shape=shape)
        if dtype_name == "BF16":
            return (np.asarray(value, np.uint16).astype(np.uint32) << 16).view(np.float32)
        return np.asarray(value, dtype=np.float32)


def _names(config, layer, expert=None):
    if not 0 <= layer < config.layers:
        raise UnsupportedMoeCheckpoint(f"layer {layer} is outside [0, {config.layers})")
    if config.adapter == "mixtral":
        root = f"model.layers.{layer}.block_sparse_moe"
        if expert is None:
            return root + ".gate.weight"
        root += f".experts.{expert}"
        return root + ".w1.weight", root + ".w3.weight", root + ".w2.weight"
    root = f"model.layers.{layer}.mlp"
    if expert is None:
        return root + ".gate.weight"
    root += f".experts.{expert}"
    return root + ".gate_proj.weight", root + ".up_proj.weight", root + ".down_proj.weight"


def load_layer(model_dir, layer=0, d=16, alignment=64):
    """Returns (MoeConfig, MoeWeights, source metadata) for one HF MoE layer."""
    hf = load_config(model_dir)
    config = hf.sa_config(d, alignment)
    tensors = SafeTensorIndex(model_dir)
    router = tensors.get(_names(hf, layer))
    gate, up, down = [], [], []
    for expert in range(hf.experts):
        gn, un, dn = _names(hf, layer, expert)
        gate.append(tensors.get(gn))
        up.append(tensors.get(un))
        down.append(tensors.get(dn))
    weights = moe.MoeWeights(np.ascontiguousarray(router, np.float32),
                             np.ascontiguousarray(np.stack(gate), np.float32),
                             np.ascontiguousarray(np.stack(up), np.float32),
                             np.ascontiguousarray(np.stack(down), np.float32))
    try:
        moe.validate_weights(config, weights)
    except ValueError as exc:
        raise UnsupportedMoeCheckpoint(f"layer {layer}: {exc}") from exc
    source = {"architecture": hf.architecture, "model_type": hf.model_type, "adapter": hf.adapter,
              "layer": layer, "layers": hf.layers, "normalize_topk": hf.normalize_topk}
    return config, weights, source


def pack_layer(model_dir, out_dir, layer=0, d=16, alignment=64):
    config, weights, source = load_layer(model_dir, layer, d, alignment)
    quantized = moe.quantize_weights(config, weights)
    blob, manifest = moe.pack_parameters(config, quantized)
    manifest["source"] = source
    total_expert_bytes = config.experts * manifest["expert_stride"]
    active_expert_bytes = config.top_k * manifest["expert_stride"]
    manifest["capacity"] = {
        "packed_layer_bytes": len(blob),
        "all_expert_bytes": total_expert_bytes,
        "active_expert_bytes_per_token": active_expert_bytes,
        "active_fraction": config.top_k / config.experts,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "moe_params.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(out_dir, "moe_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    if config.top_k == 2:
        write_acceptance_vectors(out_dir, config, quantized)
    return config, quantized, manifest


def write_acceptance_vectors(out_dir, config, weights):
    """Writes deterministic top-2 board inputs and W8A8 golden outputs."""
    x = np.linspace(-0.625, 0.75, config.dim, dtype=np.float32)
    pairs = [(i, (i + 1) % config.experts) for i in range(min(config.experts, 4))]
    scores, routing, output = [], [], []
    xq, xs = moe.quantize_activation(x)
    for first, second in pairs:
        score = np.full(config.experts, -4.0, np.float32)
        score[first], score[second] = 3.0, 1.0
        ids, rw = moe.route(score, 2)
        y = sum((weight * moe.w8a8_expert(xq, xs, weights, int(expert))).astype(np.float32)
                for expert, weight in zip(ids, rw))
        scores.append(score)
        routing.append(rw)
        output.append(y)
    np.savez(os.path.join(out_dir, "moe_acceptance.npz"), x=x, pairs=np.asarray(pairs, np.int32),
             scores=np.asarray(scores, np.float32), routing=np.asarray(routing, np.float32),
             output=np.asarray(output, np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Hugging Face checkpoint directory")
    parser.add_argument("--out", required=True, help="output SA MoE bundle directory")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--alignment", type=int, default=64)
    args = parser.parse_args()
    config, _, manifest = pack_layer(args.model, args.out, args.layer, args.d, args.alignment)
    cap = manifest["capacity"]
    print(f"{args.out}: {manifest['source']['architecture']} layer {args.layer}, "
          f"{config.experts} experts/top-{config.top_k}, stride {manifest['expert_stride']} bytes, "
          f"packed {cap['packed_layer_bytes']} bytes, active/token {cap['active_expert_bytes_per_token']} bytes")


if __name__ == "__main__":
    main()

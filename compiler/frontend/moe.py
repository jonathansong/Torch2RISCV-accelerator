"""Reference and parameter-layout utilities for sparse MoE bring-up.

This module is deliberately independent of transformers and IREE. It defines
the numerical and packed-parameter contracts used by the first MoE milestone:
batch-1 routing, W8A8 gated experts, deterministic top-k, and one uniform
expert stride. Later compiler lowering can consume the same manifest without
changing the reference model.
"""

import argparse
from dataclasses import dataclass
import json
import math
import os

import numpy as np


F32 = np.float32


def _align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class MoeConfig:
    dim: int
    hidden: int
    experts: int
    top_k: int = 1
    d: int = 16
    alignment: int = 64

    def __post_init__(self):
        if min(self.dim, self.hidden, self.experts, self.top_k, self.d) <= 0:
            raise ValueError("MoE dimensions, expert count, top-k and D must be positive")
        if self.top_k > self.experts:
            raise ValueError("top-k cannot exceed the expert count")
        if self.dim % self.d or self.hidden % self.d:
            raise ValueError(f"dim and hidden must be multiples of D={self.d}")
        if self.alignment < 16 or self.alignment & (self.alignment - 1):
            raise ValueError("alignment must be a power of two and at least 16")


@dataclass(frozen=True)
class MoeWeights:
    router: np.ndarray        # [E, H]
    gate: np.ndarray          # [E, I, H]
    up: np.ndarray            # [E, I, H]
    down: np.ndarray          # [E, H, I]


@dataclass(frozen=True)
class QuantizedMoeWeights:
    router_q: np.ndarray
    router_scale: np.ndarray
    gate_q: np.ndarray
    gate_scale: np.ndarray
    up_q: np.ndarray
    up_scale: np.ndarray
    down_q: np.ndarray
    down_scale: np.ndarray


def validate_weights(config, weights):
    expected = {
        "router": (config.experts, config.dim),
        "gate": (config.experts, config.hidden, config.dim),
        "up": (config.experts, config.hidden, config.dim),
        "down": (config.experts, config.dim, config.hidden),
    }
    for name, shape in expected.items():
        value = getattr(weights, name)
        if value.shape != shape:
            raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
        if value.dtype != np.float32:
            raise ValueError(f"{name} must be float32, got {value.dtype}")


def synthetic_weights(config, seed=0):
    """Creates deterministic, trained-scale-like fp32 weights."""
    rng = np.random.default_rng(seed)
    linear = lambda shape, fan_in: rng.normal(0, 1 / math.sqrt(fan_in), shape).astype(F32)
    weights = MoeWeights(
        router=linear((config.experts, config.dim), config.dim),
        gate=linear((config.experts, config.hidden, config.dim), config.dim),
        up=linear((config.experts, config.hidden, config.dim), config.dim),
        down=linear((config.experts, config.dim, config.hidden), config.hidden),
    )
    validate_weights(config, weights)
    return weights


def quantize_rows(weight):
    """Symmetric int8 quantization with one fp32 scale per output row."""
    weight = np.asarray(weight, dtype=F32)
    amax = np.max(np.abs(weight), axis=-1)
    scale = (amax / F32(127.0)).astype(F32)
    scale = np.where(scale == 0, F32(1.0), scale).astype(F32)
    q = np.clip(np.rint(weight / scale[..., None]), -127, 127).astype(np.int8)
    return q, scale


def quantize_activation(value):
    """Per-vector dynamic int8 quantization; zero vectors remain finite."""
    value = np.asarray(value, dtype=F32)
    amax = np.max(np.abs(value), axis=-1, keepdims=True)
    scale = (amax / F32(127.0)).astype(F32)
    scale = np.where(scale == 0, F32(1.0), scale).astype(F32)
    q = np.clip(np.rint(value / scale), -127, 127).astype(np.int8)
    return q, scale


def quantize_weights(config, weights):
    validate_weights(config, weights)
    rq, rs = quantize_rows(weights.router)
    gq, gs = quantize_rows(weights.gate)
    uq, us = quantize_rows(weights.up)
    dq, ds = quantize_rows(weights.down)
    return QuantizedMoeWeights(rq, rs, gq, gs, uq, us, dq, ds)


def qlinear(x_q, x_scale, weight_q, weight_scale):
    accum = np.matmul(x_q.astype(np.int32), np.swapaxes(weight_q.astype(np.int32), -1, -2))
    return accum.astype(F32) * np.asarray(x_scale, dtype=F32) * np.asarray(weight_scale, dtype=F32)


def route(scores, top_k, normalize=True):
    """Deterministic top-k: descending score, then ascending expert ID."""
    scores = np.asarray(scores, dtype=F32)
    if scores.ndim != 1:
        raise ValueError(f"batch-1 routing expects a rank-1 score vector, got {scores.shape}")
    if not 0 < top_k <= scores.size:
        raise ValueError("top-k must be in [1, expert_count]")
    clean = np.nan_to_num(scores, nan=-np.inf)
    ids = np.lexsort((np.arange(scores.size), -clean))[:top_k].astype(np.int32)
    selected = clean[ids].astype(F32)
    if normalize:
        selected = selected - np.max(selected)
        weights = np.exp(selected).astype(F32)
        total = np.sum(weights, dtype=F32)
        weights = weights / total if np.isfinite(total) and total != 0 else np.full(top_k, 1 / top_k, F32)
    else:
        weights = selected
    return ids, weights.astype(F32)


def fp32_expert(x, weights, expert_id):
    gate = weights.gate[expert_id] @ x
    up = weights.up[expert_id] @ x
    hidden = gate / (F32(1.0) + np.exp(-gate).astype(F32)) * up
    return (weights.down[expert_id] @ hidden).astype(F32)


def w8a8_expert(x_q, x_scale, weights, expert_id):
    gate = qlinear(x_q, x_scale, weights.gate_q[expert_id], weights.gate_scale[expert_id])
    up = qlinear(x_q, x_scale, weights.up_q[expert_id], weights.up_scale[expert_id])
    hidden = gate / (F32(1.0) + np.exp(-gate).astype(F32)) * up
    hidden_q, hidden_scale = quantize_activation(hidden)
    return qlinear(hidden_q, hidden_scale, weights.down_q[expert_id], weights.down_scale[expert_id]).astype(F32)


def moe_forward(x, config, weights, *, forced_ids=None, forced_weights=None, quantized=False):
    """Runs one token and returns (output, scores, ids, routing_weights)."""
    x = np.asarray(x, dtype=F32)
    if x.shape != (config.dim,):
        raise ValueError(f"input has shape {x.shape}, expected {(config.dim,)}")
    if quantized:
        x_q, x_scale = quantize_activation(x)
        scores = qlinear(x_q, x_scale, weights.router_q, weights.router_scale)
    else:
        scores = weights.router @ x
    if forced_ids is None:
        ids, routing_weights = route(scores, config.top_k)
    else:
        ids = np.asarray(forced_ids, dtype=np.int32)
        if ids.shape != (config.top_k,) or np.any(ids < 0) or np.any(ids >= config.experts):
            raise ValueError("forced IDs must contain top_k valid expert IDs")
        routing_weights = (np.full(config.top_k, F32(1 / config.top_k)) if forced_weights is None
                           else np.asarray(forced_weights, dtype=F32))
        if routing_weights.shape != (config.top_k,):
            raise ValueError("forced weights must contain top_k values")
    output = np.zeros(config.dim, dtype=F32)
    for expert_id, routing_weight in zip(ids, routing_weights):
        if quantized:
            expert_output = w8a8_expert(x_q, x_scale, weights, int(expert_id))
        else:
            expert_output = fp32_expert(x, weights, int(expert_id))
        output += F32(routing_weight) * expert_output
    return output, np.asarray(scores, dtype=F32), ids, routing_weights


def pack_linear(weight_q, d=16):
    """Packs W[N,K] as the backend's Wp[N/D,K,D] byte layout."""
    weight_q = np.asarray(weight_q, dtype=np.int8)
    if weight_q.ndim != 2 or weight_q.shape[0] % d:
        raise ValueError(f"weight must be [N,K] with N divisible by D={d}")
    n, k = weight_q.shape
    return np.ascontiguousarray(weight_q.reshape(n // d, d, k).transpose(0, 2, 1)).tobytes()


def unpack_linear(data, n, k, d=16):
    packed = np.frombuffer(data, np.int8)
    if n % d or packed.size != n * k:
        raise ValueError("packed weight size or shape is invalid")
    return packed.reshape(n // d, k, d).transpose(0, 2, 1).reshape(n, k).copy()


def _pack_linear_padded(weight_q, d):
    """Packs a linear weight after zero-padding its output rows to D."""
    weight_q = np.asarray(weight_q, dtype=np.int8)
    if weight_q.ndim != 2:
        raise ValueError("weight must be rank 2")
    n, k = weight_q.shape
    padded_n = _align(n, d)
    if padded_n != n:
        padded = np.zeros((padded_n, k), dtype=np.int8)
        padded[:n] = weight_q
        weight_q = padded
    return pack_linear(weight_q, d), padded_n


def pack_parameters(config, weights):
    """Returns (blob, JSON-serializable manifest) with a uniform expert stride."""
    sections = []
    blob = bytearray()

    def append(name, data):
        offset = _align(len(blob), config.alignment)
        blob.extend(b"\0" * (offset - len(blob)))
        blob.extend(data)
        section = {"name": name, "offset": offset, "size": len(data)}
        sections.append(section)
        return section

    router_data, router_padded_experts = _pack_linear_padded(weights.router_q, config.d)
    append("router.weight", router_data)
    append("router.scale", np.ascontiguousarray(weights.router_scale, F32).tobytes())
    expert_table_offset = _align(len(blob), config.alignment)
    blob.extend(b"\0" * (expert_table_offset - len(blob)))
    expert_sections = []
    stride = None
    for expert in range(config.experts):
        start = len(blob)
        current = []
        for projection, q, scale in (
                ("gate", weights.gate_q[expert], weights.gate_scale[expert]),
                ("up", weights.up_q[expert], weights.up_scale[expert]),
                ("down", weights.down_q[expert], weights.down_scale[expert])):
            current.append(append(f"experts.{expert}.{projection}.weight", pack_linear(q, config.d)))
            current.append(append(f"experts.{expert}.{projection}.scale",
                                  np.ascontiguousarray(scale, F32).tobytes()))
        end = _align(len(blob), config.alignment)
        blob.extend(b"\0" * (end - len(blob)))
        this_stride = end - start
        if stride is None:
            stride = this_stride
        elif this_stride != stride:
            raise AssertionError("expert records do not have a uniform stride")
        expert_sections.append({"expert": expert, "offset": start, "size": this_stride,
                                "sections": current})
    manifest = {
        "format": "sa-moe-parameters-v1",
        "layout": "iree-sa-packed-w8",
        "alignment": config.alignment,
        "d": config.d,
        "dim": config.dim,
        "hidden": config.hidden,
        "experts": config.experts,
        "router_padded_experts": router_padded_experts,
        "top_k": config.top_k,
        "expert_table_offset": expert_table_offset,
        "expert_stride": stride,
        "sections": sections,
        "expert_records": expert_sections,
        "total_bytes": len(blob),
    }
    return bytes(blob), manifest


def write_synthetic_bundle(path, config, seed=0):
    """Writes a self-contained M0 parameter blob, manifest and golden vectors."""
    os.makedirs(path, exist_ok=True)
    fp = synthetic_weights(config, seed)
    quantized = quantize_weights(config, fp)
    blob, manifest = pack_parameters(config, quantized)
    x = np.random.default_rng(seed + 1).normal(0, 1, config.dim).astype(F32)
    fp_out, fp_scores, fp_ids, fp_routing = moe_forward(x, config, fp)
    q_out, q_scores, q_ids, q_routing = moe_forward(x, config, quantized, quantized=True)
    with open(os.path.join(path, "moe_params.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(path, "moe_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    np.savez(os.path.join(path, "moe_reference.npz"), x=x, fp32_output=fp_out, fp32_scores=fp_scores,
             fp32_ids=fp_ids, fp32_routing=fp_routing, w8a8_output=q_out, w8a8_scores=q_scores,
             w8a8_ids=q_ids, w8a8_routing=q_routing)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Create a synthetic sparse-MoE M0 bundle")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = MoeConfig(args.dim, args.hidden, args.experts, args.top_k, args.d)
    manifest = write_synthetic_bundle(args.out, config, args.seed)
    print(f"{args.out}: {manifest['experts']} experts, top-{manifest['top_k']}, "
          f"stride {manifest['expert_stride']} bytes, total {manifest['total_bytes']} bytes")


if __name__ == "__main__":
    main()

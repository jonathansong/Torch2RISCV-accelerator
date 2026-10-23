"""Small batch-1 Mixtral decoder with a pluggable routed expert FFN.

The NumPy path runs fp32 embedding, RMSNorm, causal grouped-query attention,
residuals, router projection, and the output head. The caller supplies one FFN
result per layer and token, allowing a host reference, SA simulator, or KV260
device to share exactly the same non-expert model code.
"""

import math
import os

import numpy as np

import moe_hf


def rms_norm(x, weight, eps):
    x = np.asarray(x, np.float32)
    variance = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (x * (np.float32(1.0) / np.sqrt(variance + np.float32(eps))) * weight).astype(np.float32)


def rotate_half(x):
    mid = x.shape[-1] // 2
    return np.concatenate((-x[..., mid:], x[..., :mid]), axis=-1)


def softmax(x):
    shifted = np.asarray(x, np.float32) - np.max(x)
    values = np.exp(shifted).astype(np.float32)
    return values / np.sum(values, dtype=np.float32)


def forward(model_dir, tokens, expert_fn):
    """Return (logits[T,V], trace) for a batch-1 prompt.

    expert_fn(layer, position, x[H], router_scores[E]) must return one fp32
    expert mixture output [H]. It may run the device; this function does not
    select experts or perform their projections.
    """
    config = moe_hf.load_config(model_dir)
    if config.adapter != "mixtral":
        raise ValueError("NumPy decoder currently supports Mixtral only")
    raw = moe_hf._load_json(os.path.join(model_dir, "config.json"))
    tokens = np.asarray(tokens, np.int64)
    if tokens.ndim != 1 or not tokens.size or np.any(tokens < 0) or np.any(tokens >= int(raw["vocab_size"])):
        raise ValueError("token IDs must be a nonempty vector within the vocabulary")
    index = moe_hf.SafeTensorIndex(model_dir)
    dim = config.dim
    heads = int(raw["num_attention_heads"])
    kv_heads = int(raw["num_key_value_heads"])
    head_dim = int(raw.get("head_dim") or dim // heads)
    if heads % kv_heads or heads * head_dim != dim or head_dim % 2:
        raise ValueError("unsupported Mixtral attention geometry")
    group = heads // kv_heads
    eps = np.float32(raw.get("rms_norm_eps", 1e-5))
    window = raw.get("sliding_window")
    window = int(window) if window is not None else len(tokens)
    x = index.get("model.embed_tokens.weight")[tokens].copy()
    if x.shape != (len(tokens), dim):
        raise ValueError("embedding shape does not match Mixtral config")

    positions = np.arange(len(tokens), dtype=np.float32)
    inv_freq = np.float32(1.0) / np.power(
        np.float32(raw.get("rope_theta", 10000.0)),
        np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim))
    freqs = positions[:, None] * inv_freq[None, :]
    angles = np.concatenate((freqs, freqs), axis=-1)
    cos, sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
    trace = {}

    for layer in range(config.layers):
        prefix = f"model.layers.{layer}."
        att = prefix + "self_attn."
        xb = rms_norm(x, index.get(prefix + "input_layernorm.weight"), eps)
        q = (xb @ index.get(att + "q_proj.weight").T).reshape(len(tokens), heads, head_dim)
        k = (xb @ index.get(att + "k_proj.weight").T).reshape(len(tokens), kv_heads, head_dim)
        v = (xb @ index.get(att + "v_proj.weight").T).reshape(len(tokens), kv_heads, head_dim)
        q = q * cos[:, None, :] + rotate_half(q) * sin[:, None, :]
        k = k * cos[:, None, :] + rotate_half(k) * sin[:, None, :]
        attention = np.empty((len(tokens), heads, head_dim), np.float32)
        scale = np.float32(1.0 / math.sqrt(head_dim))
        for pos in range(len(tokens)):
            start = max(0, pos + 1 - window)
            for head in range(heads):
                kv_head = head // group
                scores = (k[start:pos + 1, kv_head] @ q[pos, head]) * scale
                attention[pos, head] = softmax(scores) @ v[start:pos + 1, kv_head]
        x = (x + attention.reshape(len(tokens), dim) @ index.get(att + "o_proj.weight").T).astype(np.float32)

        xb = rms_norm(x, index.get(prefix + "post_attention_layernorm.weight"), eps)
        router = index.get(prefix + "block_sparse_moe.gate.weight")
        router_scores = (xb @ router.T).astype(np.float32)
        result = np.stack([np.asarray(expert_fn(layer, pos, xb[pos], router_scores[pos]), np.float32)
                           for pos in range(len(tokens))])
        if result.shape != x.shape:
            raise ValueError(f"layer {layer}: expert output has shape {result.shape}, expected {x.shape}")
        trace[layer] = {"ffn_input": xb.copy(), "router_scores": router_scores.copy(),
                        "ffn_output": result.copy()}
        x = (x + result).astype(np.float32)

    x = rms_norm(x, index.get("model.norm.weight"), eps)
    logits = x @ index.get("lm_head.weight").T
    return np.asarray(logits, np.float32), trace

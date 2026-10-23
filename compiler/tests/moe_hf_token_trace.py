#!/usr/bin/env python3
"""Capture real token driven Mixtral MoE inputs for SA layer acceptance.

The complete Hugging Face decoder supplies attention, residuals, and layer
normalization. For each MoE layer this tool compares its fp32 output with the
repository reference and emits the input, router scores, and W8A8 FFN golden
output expected by the existing simulator and KV260 layer runners.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "compiler", "frontend"))

import moe  # noqa: E402
import moe_hf  # noqa: E402


def relative_l2(got, expected):
    denominator = np.linalg.norm(expected.astype(np.float64))
    return float(np.linalg.norm((got - expected).astype(np.float64)) / max(denominator, 1e-12))


def trace(model_dir, out_dir, tokens, layers=None):
    import torch
    from transformers import AutoModelForCausalLM

    hf = moe_hf.load_config(model_dir)
    if hf.adapter != "mixtral":
        raise ValueError("token trace currently supports Mixtral only")
    raw = moe_hf._load_json(os.path.join(model_dir, "config.json"))
    vocab = int(raw["vocab_size"])
    if not tokens or any(token < 0 or token >= vocab for token in tokens):
        raise ValueError(f"token IDs must be in [0, {vocab})")
    selected = list(range(hf.layers)) if layers is None else list(layers)
    if not selected or len(set(selected)) != len(selected) or any(layer < 0 or layer >= hf.layers for layer in selected):
        raise ValueError(f"layers must be nonempty, distinct IDs in [0, {hf.layers})")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, local_files_only=True,
        attn_implementation="sdpa").eval()
    captured = {layer: {} for layer in selected}
    hooks = []

    def record_input(layer):
        def hook(_module, inputs):
            captured[layer]["x"] = inputs[0].detach().cpu().numpy().copy()
        return hook

    def record_router(layer):
        def hook(_module, _inputs, output):
            logits, weights, ids = output
            captured[layer]["scores"] = logits.detach().cpu().numpy().copy()
            captured[layer]["routing"] = weights.detach().cpu().numpy().copy()
            captured[layer]["ids"] = ids.detach().cpu().numpy().copy()
        return hook

    def record_output(layer):
        def hook(_module, _inputs, output):
            captured[layer]["output"] = output.detach().cpu().numpy().copy()
        return hook

    for layer in selected:
        block = model.model.layers[layer].mlp
        hooks.append(block.register_forward_pre_hook(record_input(layer)))
        hooks.append(block.gate.register_forward_hook(record_router(layer)))
        hooks.append(block.register_forward_hook(record_output(layer)))
    try:
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([tokens], dtype=torch.long),
                           use_cache=False).logits[0, -1].float().cpu().numpy()
    finally:
        for hook in hooks:
            hook.remove()
    del model

    report = {"architecture": hf.architecture, "model": os.path.abspath(model_dir),
              "tokens": tokens, "last_hf_token": int(np.argmax(logits)), "layers": {}}
    for layer in selected:
        observed = captured[layer]
        if set(observed) != {"x", "scores", "routing", "ids", "output"}:
            raise RuntimeError(f"layer {layer}: missing Hugging Face MoE hook values")
        x = np.asarray(observed["x"][0, -1], np.float32)
        scores = np.asarray(observed["scores"][-1], np.float32)
        ids = np.asarray(observed["ids"][-1], np.int32)
        routing = np.asarray(observed["routing"][-1], np.float32)
        wanted = np.asarray(observed["output"][0, -1], np.float32)
        config, fp, _ = moe_hf.load_layer(model_dir, layer)
        fp_y, fp_scores, _, _ = moe.moe_forward(x, config, fp, forced_ids=ids,
                                                 forced_weights=routing)
        if not np.allclose(fp_scores, scores, rtol=2e-4, atol=2e-5):
            raise RuntimeError(f"layer {layer}: checkpoint router does not match Hugging Face")
        if not np.allclose(fp_y, wanted, rtol=2e-3, atol=2e-5):
            raise RuntimeError(f"layer {layer}: fp32 expert output does not match Hugging Face")
        q = moe.quantize_weights(config, fp)
        # The board route kernel consumes the actual HF fp32 router scores.
        # The FFN golden follows that route with W8A8 expert projections.
        route_ids, route_weights = moe.route(scores, config.top_k)
        if not np.array_equal(route_ids, ids):
            raise RuntimeError(f"layer {layer}: SA top-k disagrees with Hugging Face")
        if not np.allclose(route_weights, routing, rtol=2e-4, atol=2e-5):
            raise RuntimeError(f"layer {layer}: SA routing weights disagree with Hugging Face")
        q_y, _, _, _ = moe.moe_forward(x, config, q, forced_ids=route_ids,
                                       forced_weights=route_weights, quantized=True)
        layer_dir = os.path.join(out_dir, f"layer{layer}")
        os.makedirs(layer_dir, exist_ok=True)
        np.savez(os.path.join(layer_dir, "moe_token_acceptance.npz"),
                 x=x, scores=scores[None, :], pairs=route_ids[None, :],
                 routing=route_weights[None, :], output=q_y[None, :],
                 hf_output=wanted, hf_token=np.int32(report["last_hf_token"]))
        metrics = {"hf_ids": ids.tolist(), "sa_ids": route_ids.tolist(),
                   "hf_routing": routing.tolist(),
                   "fp32_vs_hf_relative_l2": relative_l2(fp_y, wanted),
                   "w8a8_vs_hf_relative_l2": relative_l2(q_y, wanted),
                   "w8a8_vs_hf_max_abs": float(np.max(np.abs(q_y - wanted)))}
        report["layers"][str(layer)] = metrics
        print(f"layer {layer}: experts {ids.tolist()}, fp32/HF relative L2 "
              f"{metrics['fp32_vs_hf_relative_l2']:.6g}, W8A8/HF relative L2 "
              f"{metrics['w8a8_vs_hf_relative_l2']:.6g}", flush=True)
        del fp, q
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"HF last-token argmax {report['last_hf_token']}; traces written to {out_dir}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokens", default="1,42,43", help="comma-separated input token IDs")
    parser.add_argument("--layers", help="comma-separated MoE layer IDs (default: all)")
    args = parser.parse_args()
    tokens = [int(token) for token in args.tokens.split(",")]
    layers = None if args.layers is None else [int(layer) for layer in args.layers.split(",")]
    trace(args.model, args.out, tokens, layers)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Numerical validation of an imported Hugging Face MoE layer."""

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


def validate(model_dir, layer=0, samples=8, seed=0):
    config, fp, source = moe_hf.load_layer(model_dir, layer)
    q = moe.quantize_weights(config, fp)
    rng = np.random.default_rng(seed)
    route_matches, output_errors, forced_errors, max_abs = 0, [], [], []
    cases = []
    for sample in range(samples):
        x = rng.normal(0, 1, config.dim).astype(np.float32)
        fp_out, fp_scores, fp_ids, fp_routing = moe.moe_forward(x, config, fp)
        q_out, q_scores, q_ids, q_routing = moe.moe_forward(x, config, q, quantized=True)
        route_matches += int(np.array_equal(fp_ids, q_ids))
        output_errors.append(relative_l2(q_out, fp_out))
        max_abs.append(float(np.max(np.abs(q_out - fp_out))))
        forced_q, _, _, _ = moe.moe_forward(x, config, q, forced_ids=fp_ids,
                                             forced_weights=fp_routing, quantized=True)
        forced_errors.append(relative_l2(forced_q, fp_out))
        cases.append({"sample": sample, "fp_ids": fp_ids.tolist(), "w8a8_ids": q_ids.tolist(),
                      "fp_weights": fp_routing.tolist(), "w8a8_weights": q_routing.tolist(),
                      "relative_l2": output_errors[-1], "forced_route_relative_l2": forced_errors[-1],
                      "max_abs": max_abs[-1],
                      "router_score_relative_l2": relative_l2(q_scores, fp_scores)})
    return {
        "source": source,
        "config": {"dim": config.dim, "hidden": config.hidden, "experts": config.experts,
                   "top_k": config.top_k, "d": config.d},
        "samples": samples,
        "routing_agreement": route_matches / samples,
        "output_relative_l2_mean": float(np.mean(output_errors)),
        "output_relative_l2_max": float(np.max(output_errors)),
        "forced_route_relative_l2_mean": float(np.mean(forced_errors)),
        "forced_route_relative_l2_max": float(np.max(forced_errors)),
        "output_max_abs": float(np.max(max_abs)),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = validate(args.model, args.layer, args.samples, args.seed)
    text = json.dumps(result, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

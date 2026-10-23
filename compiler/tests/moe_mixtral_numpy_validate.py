#!/usr/bin/env python3
"""Compare the portable NumPy Mixtral host path with Hugging Face logits."""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for path in (os.path.join(REPO, "compiler", "frontend"),
             os.path.join(REPO, "compiler", "runtime", "tools"),
             HERE, os.path.join(REPO, "driver"), os.path.join(REPO, "llm")):
    sys.path.insert(0, path)

import moe  # noqa: E402
import moe_hf  # noqa: E402
import moe_mixtral_numpy  # noqa: E402
from moe_hf_hybrid_sim import SimulatedExpertLayer, relative_l2  # noqa: E402


def validate(model_dir, bundles, tokens, reference, backend):
    hf = moe_hf.load_config(model_dir)
    if backend == "fp32":
        weights = [moe_hf.load_layer(model_dir, layer)[1] for layer in range(hf.layers)]

        def expert(layer, _pos, x, scores):
            ids, routing = moe.route(scores, hf.top_k)
            return moe.moe_forward(x, hf.sa_config(), weights[layer],
                                   forced_ids=ids, forced_weights=routing)[0]
    else:
        layers = [SimulatedExpertLayer(os.path.join(bundles, f"layer{layer}"), layer)
                  for layer in range(hf.layers)]

        def expert(layer, _pos, x, scores):
            ids, routing = moe.route(scores, hf.top_k)
            return layers[layer].run(x, ids, routing)

    logits, trace = moe_mixtral_numpy.forward(model_dir, tokens, expert)
    with np.load(reference) as target:
        expected = target["hf" if backend == "fp32" else "hybrid"]
    if logits.shape != expected.shape:
        raise ValueError(f"logits shape {logits.shape} vs reference {expected.shape}")
    positions = []
    for pos in range(len(tokens)):
        got = logits[pos]
        wanted = expected[pos]
        positions.append({"position": pos, "got_top": int(np.argmax(got)),
                          "reference_top": int(np.argmax(wanted)),
                          "relative_l2": relative_l2(got, wanted),
                          "max_abs": float(np.max(np.abs(got - wanted)))})
    result = {"backend": backend, "tokens": tokens, "positions": positions,
              "top_token_agreement": all(p["got_top"] == p["reference_top"] for p in positions),
              "layer_count": len(trace)}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="build/llm_cache/TitanML-tiny-mixtral")
    parser.add_argument("--bundles", default="build/moe/m4c/tiny-mixtral")
    parser.add_argument("--reference", default="build/moe/m4c/hybrid-sim/teacher_logits.npz")
    parser.add_argument("--tokens", default="1,42,43")
    parser.add_argument("--backend", choices=("fp32", "sim"), default="fp32")
    args = parser.parse_args()
    tokens = [int(value) for value in args.tokens.split(",")]
    validate(args.model, args.bundles, tokens, args.reference, args.backend)


if __name__ == "__main__":
    main()

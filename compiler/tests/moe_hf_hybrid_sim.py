#!/usr/bin/env python3
"""Run a Mixtral decoder with both MoE FFNs replaced by SA simulator output.

Hugging Face supplies embedding, attention, residuals, normalization, and the
output head. The SA functional simulator supplies each routed W8A8 expert FFN.
This measures how quantized expert outputs propagate through later layers and
the final logits. It is not a full SA decoder implementation.
"""

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
import moe_m3b  # noqa: E402
import sadesc  # noqa: E402
from sa_funcsim import SaFuncSim  # noqa: E402


def align(value, alignment=4096):
    return (value + alignment - 1) // alignment * alignment


def relative_l2(got, expected):
    denominator = np.linalg.norm(expected.astype(np.float64))
    return float(np.linalg.norm((got - expected).astype(np.float64)) / max(denominator, 1e-12))


class SimulatedExpertLayer:
    def __init__(self, bundle, expected_layer):
        with open(os.path.join(bundle, "moe_manifest.json")) as f:
            self.manifest = json.load(f)
        m = self.manifest
        if m["source"]["layer"] != expected_layer or m["source"]["adapter"] != "mixtral":
            raise ValueError(f"{bundle}: wrong layer or unsupported adapter")
        if int(m["top_k"]) != 2:
            raise ValueError("the descriptor currently requires top-2")
        self.dim = int(m["dim"])
        self.experts = int(m["experts"])
        self.stride = int(m["expert_stride"])
        params = np.fromfile(os.path.join(bundle, "moe_params.bin"), np.uint8)
        _, _, exports = sadesc.read(moe_m3b.make_executable(
            self.dim, int(m["hidden"]), int(m["d"]), int(m["alignment"])))
        rows = exports[0][1]
        base = 0x10000000
        self.xaddr, self.routeaddr, paramaddr = base + 0x10000, base + 0x20000, base + 0x100000
        self.outaddr = align(paramaddr + params.size)
        self.listaddr = align(self.outaddr + self.dim * 4)
        size = align(self.listaddr + (len(rows) + 8) * 64 - base)
        self.sim = SaFuncSim(int(m["d"]), base, size)
        self.sim.ddr_write(base, rows.tobytes())
        self.sim.ddr_write(paramaddr, params)
        bindings = [self.xaddr, self.routeaddr,
                    paramaddr + int(m["expert_table_offset"]), self.outaddr]
        dispatch = sadesc.dispatch_list(base, bindings, [])
        self.sim.ddr_write(self.listaddr, dispatch.array().tobytes())

    def run(self, x, ids, weights):
        x = np.asarray(x, np.float32)
        ids = np.asarray(ids, np.uint32)
        weights = np.asarray(weights, np.float32)
        if x.shape != (self.dim,) or ids.shape != (2,) or weights.shape != (2,):
            raise ValueError("invalid routed FFN input shape")
        offsets = ids * np.uint32(self.stride)
        route = moe.pack_route_record(ids, offsets, weights)
        self.sim.ddr_write(self.xaddr, np.ascontiguousarray(x))
        self.sim.ddr_write(self.routeaddr, route)
        self.sim.run_list(self.listaddr)
        return self.sim.ddr_read(self.outaddr, self.dim * 4).view("<f4").copy()


def run(model_dir, bundle_root, out_dir, tokens):
    import torch
    from transformers import AutoModelForCausalLM

    hf = moe_hf.load_config(model_dir)
    raw = moe_hf._load_json(os.path.join(model_dir, "config.json"))
    if hf.adapter != "mixtral" or not tokens or any(t < 0 or t >= int(raw["vocab_size"]) for t in tokens):
        raise ValueError("expected Mixtral and nonempty token IDs within the vocabulary")
    layers = [SimulatedExpertLayer(os.path.join(bundle_root, f"layer{layer}"), layer)
              for layer in range(hf.layers)]
    if any(layer.dim != hf.dim or layer.experts != hf.experts for layer in layers):
        raise ValueError("packed MoE dimensions do not match the checkpoint")

    torch.set_num_threads(min(2, torch.get_num_threads()))
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, local_files_only=True,
        attn_implementation="sdpa").eval()
    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        reference_logits = model(input_ids=input_ids, use_cache=False).logits[0].float().cpu().numpy().copy()

    gate_results = {}
    per_layer = {}
    acceptance = {}
    hooks = []

    def capture_gate(layer):
        def hook(_module, _inputs, output):
            gate_results[layer] = tuple(t.detach().cpu().numpy().copy() for t in output)
        return hook

    def replace_ffn(layer):
        def hook(_module, inputs, original):
            x = inputs[0].detach().cpu().numpy()[0]
            hf_output = original.detach().cpu().numpy()[0]
            scores, hf_weights, hf_ids = gate_results.pop(layer)
            if scores.shape != (len(tokens), hf.experts):
                raise ValueError(f"layer {layer}: unexpected router score shape {scores.shape}")
            replacement = []
            details = []
            selected_ids, selected_weights = [], []
            for pos, (vector, score, expected, expected_weights) in enumerate(
                    zip(x, scores, hf_output, hf_weights)):
                ids, weights = moe.route(score, hf.top_k)
                if not np.array_equal(ids, hf_ids[pos]) or not np.allclose(
                        weights, expected_weights, rtol=2e-4, atol=2e-5):
                    raise RuntimeError(
                        f"layer {layer}, position {pos}: SA route {ids.tolist()}, {weights.tolist()} "
                        f"vs HF {hf_ids[pos].tolist()}, {expected_weights.tolist()}; "
                        f"scores {score.tolist()}")
                got = layers[layer].run(vector, ids, weights)
                replacement.append(got)
                selected_ids.append(ids)
                selected_weights.append(weights)
                details.append({"position": pos, "ids": ids.tolist(),
                                "relative_l2_vs_hf_ffn": relative_l2(got, expected),
                                "max_abs_vs_hf_ffn": float(np.max(np.abs(got - expected)))})
            per_layer[str(layer)] = details
            acceptance[layer] = {"x": np.asarray(x, np.float32),
                                 "scores": np.asarray(scores, np.float32),
                                 "pairs": np.asarray(selected_ids, np.int32),
                                 "routing": np.asarray(selected_weights, np.float32),
                                 "output": np.asarray(replacement, np.float32)}
            return torch.from_numpy(np.stack(replacement)[None, :, :]).to(original.device)
        return hook

    for layer in range(hf.layers):
        block = model.model.layers[layer].mlp
        hooks.append(block.gate.register_forward_hook(capture_gate(layer)))
        hooks.append(block.register_forward_hook(replace_ffn(layer)))
    try:
        with torch.no_grad():
            hybrid_logits = model(input_ids=input_ids, use_cache=False).logits[0].float().cpu().numpy().copy()
    finally:
        for hook in hooks:
            hook.remove()
    if len(per_layer) != hf.layers:
        raise RuntimeError("not all MoE layers ran through the SA simulator")

    # Give the board runner an independent W8A8 host golden, and check each
    # simulator call against it before writing acceptance vectors.
    for layer, values in acceptance.items():
        config, fp, _ = moe_hf.load_layer(model_dir, layer)
        quantized = moe.quantize_weights(config, fp)
        golden = []
        for pos, (vector, ids, weights) in enumerate(
                zip(values["x"], values["pairs"], values["routing"])):
            wanted, _, _, _ = moe.moe_forward(vector, config, quantized,
                                               forced_ids=ids, forced_weights=weights,
                                               quantized=True)
            got = values["output"][pos]
            error = float(np.max(np.abs(got - wanted)))
            if not np.allclose(got, wanted, rtol=3e-3, atol=3e-3):
                raise RuntimeError(f"layer {layer}, position {pos}: SA/host W8A8 mismatch, max {error}")
            per_layer[str(layer)][pos]["sim_vs_host_w8a8_max_abs"] = error
            golden.append(wanted)
        values["sim_output"] = values["output"]
        values["output"] = np.asarray(golden, np.float32)
        del fp, quantized

    ref_top = np.argmax(reference_logits, axis=1)
    hybrid_top = np.argmax(hybrid_logits, axis=1)
    positions = [{"position": pos, "hf_token": int(ref_top[pos]),
                  "hybrid_token": int(hybrid_top[pos]),
                  "logit_relative_l2": relative_l2(hybrid_logits[pos], reference_logits[pos]),
                  "logit_max_abs": float(np.max(np.abs(hybrid_logits[pos] - reference_logits[pos])))}
                 for pos in range(len(tokens))]
    report = {"model": os.path.abspath(model_dir), "tokens": tokens,
              "sa_moe_layers": hf.layers, "sa_moe_calls": sum(map(len, per_layer.values())),
              "reference_top_token": int(ref_top[-1]), "hybrid_top_token": int(hybrid_top[-1]),
              "top_token_agreement": bool(np.array_equal(ref_top, hybrid_top)),
              "logit_relative_l2": relative_l2(hybrid_logits, reference_logits),
              "logit_max_abs": float(np.max(np.abs(hybrid_logits - reference_logits))),
              "positions": positions, "layers": per_layer,
              "execution_scope": "HF attention/residual/norm/head; SA simulator routed expert FFNs; host top-k"}
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "teacher_logits.npz"),
             hf=reference_logits, hybrid=hybrid_logits)
    for layer, values in acceptance.items():
        layer_dir = os.path.join(out_dir, f"layer{layer}")
        os.makedirs(layer_dir, exist_ok=True)
        np.savez(os.path.join(layer_dir, "moe_acceptance.npz"), **values)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    for layer, details in per_layer.items():
        print(f"layer {layer}: " + ", ".join(
            f"pos {d['position']} experts {d['ids']} relL2={d['relative_l2_vs_hf_ffn']:.5g}"
            for d in details), flush=True)
    for position in positions:
        print(f"position {position['position']}: HF token {position['hf_token']}, "
              f"hybrid token {position['hybrid_token']}, logit relative L2 "
              f"{position['logit_relative_l2']:.6g}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="build/llm_cache/TitanML-tiny-mixtral")
    parser.add_argument("--bundles", default="build/moe/m4c/tiny-mixtral")
    parser.add_argument("--out", default="build/moe/m4c/hybrid-sim")
    parser.add_argument("--tokens", default="1,42,43")
    args = parser.parse_args()
    run(args.model, args.bundles, args.out, [int(t) for t in args.tokens.split(",")])


if __name__ == "__main__":
    main()

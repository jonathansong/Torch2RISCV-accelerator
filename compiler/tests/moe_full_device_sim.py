#!/usr/bin/env python3
"""Run dense Mixtral SA stages and both routed expert FFNs in the simulator.

The functional simulator cannot execute the RISC-V route firmware, so this
runner computes the top-2 record on the host from SA-produced router scores.
It checks the complete SA dense/attention/FFN numerical path before board use.
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
             os.path.join(REPO, "driver"), os.path.join(REPO, "llm")):
    sys.path.insert(0, path)

import compile_model as CM  # noqa: E402
import moe  # noqa: E402
import moe_m3b  # noqa: E402
import moe_model  # noqa: E402
from export_w8a8 import W8A8  # noqa: E402
from sa_funcsim import SaFuncSim  # noqa: E402


def align(n, size=4096):
    return (n + size - 1) // size * size


def relative_l2(got, expected):
    return float(np.linalg.norm((got - expected).astype(np.float64)) /
                 max(np.linalg.norm(expected.astype(np.float64)), 1e-12))


def run(dense, bundles, reference, tokens, logits_out=None):
    model = W8A8(dense)
    c = model.cfg
    if c.layers != 2 or len(tokens) > c.seq_len:
        raise ValueError("this acceptance runner expects two layers and a prompt within the packed context")
    manifests = []
    for layer in range(c.layers):
        with open(os.path.join(bundles, f"layer{layer}", "moe_manifest.json")) as f:
            manifest = json.load(f)
        if int(manifest["dim"]) != c.dim or int(manifest["d"]) != model.d:
            raise ValueError("dense and MoE bundle dimensions differ")
        manifests.append(manifest)
    compiler = moe_model.MoEModelCompiler(model, int(manifests[0]["experts"]))
    stages = [compiler.build_layer(layer) for layer in range(c.layers)]
    final = compiler.build_final()
    expert = moe_m3b.top2_ffn_template(c.dim, c.hidden, model.d,
                                       int(manifests[0]["alignment"]), ret=False)

    base = 0x10000000
    io_off = align(model.raw.size)
    io_size = align(CM.IO_LOGITS + c.vocab * 4)
    kv_off = io_off + io_size
    param_off = align(kv_off + CM.kv_bytes(c))
    max_param = max(os.path.getsize(os.path.join(bundles, f"layer{l}", "moe_params.bin"))
                    for l in range(c.layers))
    list_off = align(param_off + max_param)
    list_size = align(max(len(expert), len(final), *(len(stage) for stage in stages)) * 64 + 4096)
    sim = SaFuncSim(model.d, base, list_off + list_size)
    sim.ddr_write(base, model.raw)
    bases = [base, base + io_off, base + kv_off]
    io = base + io_off
    ffn_addr = io + moe_model.IO_FFN
    route_addr = io + 0x3000
    list_addr = base + list_off
    logits = []
    trace = []
    for pos, token in enumerate(tokens):
        sim.ddr_write(io + CM.ARG_POS, CM.arg_block(pos, int(token)))
        params = CM.token_params(pos, model.d, c.head_size)
        for layer, stage in enumerate(stages):
            sim.ddr_write(list_addr, stage.array().tobytes())
            sim.run_list(list_addr, bases=bases, params=params)
            scores = sim.ddr_read(io + moe_model.IO_SCORES, model.d * 4).view("<f4")
            count = int(manifests[layer]["experts"])
            ids, weights = moe.route(scores[:count], 2)
            offsets = ids.astype(np.uint32) * np.uint32(manifests[layer]["expert_stride"])
            sim.ddr_write(route_addr, moe.pack_route_record(ids, offsets, weights))
            layer_params = np.fromfile(os.path.join(bundles, f"layer{layer}", "moe_params.bin"), np.uint8)
            sim.ddr_write(base + param_off, layer_params)
            sim.ddr_write(list_addr, expert.array().tobytes())
            sim.run_list(list_addr, bases=[ffn_addr, route_addr,
                base + param_off + int(manifests[layer]["expert_table_offset"]), ffn_addr])
            trace.append({"position": pos, "layer": layer, "experts": ids.tolist(),
                          "weights": weights.tolist()})
        sim.ddr_write(list_addr, final.array().tobytes())
        sim.run_list(list_addr, bases=bases, params=params)
        logits.append(sim.ddr_read(io + CM.IO_LOGITS, c.vocab * 4).view("<f4").copy())
        print(f"position {pos}: SA token {int(np.argmax(logits[-1]))}, "
              f"expert pairs {[row['experts'] for row in trace[-c.layers:]]}", flush=True)

    got = np.stack(logits)
    if logits_out is not None:
        np.savez(logits_out, sa=got)
    hf_routes = {}
    for layer in range(c.layers):
        path = os.path.join(os.path.dirname(reference), f"layer{layer}", "moe_acceptance.npz")
        with np.load(path) as data:
            hf_routes[layer] = (data["pairs"][:len(tokens)].copy(),
                                data["routing"][:len(tokens)].copy())
    for record in trace:
        layer, pos = record["layer"], record["position"]
        pairs, weights = hf_routes[layer]
        record["hf_experts"] = pairs[pos].tolist()
        record["route_agreement"] = record["experts"] == record["hf_experts"]
        record["weight_max_abs_vs_hf"] = float(np.max(np.abs(
            np.asarray(record["weights"], np.float32) - weights[pos])))
    with np.load(reference) as data:
        hf = data["hf"][:len(tokens)].copy()
    if hf.shape != got.shape:
        raise ValueError(f"HF reference shape {hf.shape} differs from SA {got.shape}")
    positions = [{"position": i, "sa_top": int(np.argmax(got[i])), "hf_top": int(np.argmax(hf[i])),
                  "relative_l2_vs_hf": relative_l2(got[i], hf[i]),
                  "max_abs_vs_hf": float(np.max(np.abs(got[i] - hf[i])))}
                 for i in range(len(tokens))]
    return {"tokens": tokens, "positions": positions, "routing": trace,
            "dense_parameter_bytes": int(model.raw.size), "expert_parameter_bytes": max_param,
            "simulator_ddr_bytes": list_off + list_size,
            "execution_scope": "SA simulator dense decoder, GQA, RoPE, router projection and expert FFNs; host top-2 record"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dense", default="build/moe/m4f/mixtral_dense.w8a8")
    parser.add_argument("--bundles", default="build/moe/m4c/board")
    parser.add_argument("--reference", default="build/moe/m4c/hybrid-sim/teacher_logits.npz")
    parser.add_argument("--tokens", default="1,42,43")
    parser.add_argument("--out", default="build/moe/m4f/full_device_sim.json")
    parser.add_argument("--logits-out", default="build/moe/m4f/full_device_sim_logits.npz")
    args = parser.parse_args()
    report = run(args.dense, args.bundles, args.reference,
                 [int(value) for value in args.tokens.split(",")], args.logits_out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    for row in report["positions"]:
        print(f"position {row['position']}: SA {row['sa_top']}, HF {row['hf_top']}, "
              f"relative L2 {row['relative_l2_vs_hf']:.6g}")
    return 0 if (all(row["sa_top"] == row["hf_top"] for row in report["positions"]) and
                 all(row["route_agreement"] for row in report["routing"])) else 1


if __name__ == "__main__":
    raise SystemExit(main())

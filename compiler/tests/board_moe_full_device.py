#!/usr/bin/env python3
"""KV260 acceptance for the complete two-layer SA Mixtral decoder."""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import compile_model as CM  # noqa: E402
import moe_m3b  # noqa: E402
import moe_model  # noqa: E402
from export_w8a8 import W8A8  # noqa: E402
from pynq_matmul import Device  # noqa: E402


def cma_kib():
    with open("/proc/meminfo") as f:
        items = dict(line.split(":", 1) for line in f if line.startswith(("CmaTotal:", "CmaFree:")))
    return {key: int(value.strip().split()[0]) for key, value in items.items()}


def relative_l2(got, expected):
    return float(np.linalg.norm((got - expected).astype(np.float64)) /
                 max(np.linalg.norm(expected.astype(np.float64)), 1e-12))


def run(args):
    from pynq import allocate

    model = W8A8(args.dense)
    cfg = model.cfg
    tokens = [int(value) for value in args.tokens.split(",")]
    if cfg.layers != 2 or len(tokens) > cfg.seq_len:
        raise ValueError("expected a two-layer Mixtral model and a prompt within the packed context")
    manifests = []
    for layer in range(cfg.layers):
        with open(os.path.join(args.bundles, f"layer{layer}", "moe_manifest.json")) as f:
            m = json.load(f)
        if int(m["dim"]) != cfg.dim or int(m["d"]) != model.d:
            raise ValueError("dense and expert parameter dimensions disagree")
        manifests.append(m)
    compiler = moe_model.MoEModelCompiler(model, int(manifests[0]["experts"]))
    stages = [compiler.build_layer(layer) for layer in range(cfg.layers)]
    final = compiler.build_final()
    expert = moe_m3b.top2_ffn_template(cfg.dim, cfg.hidden, model.d,
                                       int(manifests[0]["alignment"]), ret=False)
    io_size = (CM.IO_LOGITS + cfg.vocab * 4 + 4095) // 4096 * 4096
    largest = max(os.path.getsize(os.path.join(args.bundles, f"layer{l}", "moe_params.bin"))
                  for l in range(cfg.layers))
    buffers = []
    device = Device(args.bitfile, args.firmware, timeout=30.0)
    before = cma_kib()
    try:
        def alloc(shape, dtype):
            value = allocate(shape=shape, dtype=dtype)
            buffers.append(value)
            return value

        dense = alloc((model.raw.size,), np.uint8)
        io = alloc((io_size,), np.uint8)
        kv = alloc((CM.kv_bytes(cfg),), np.uint8)
        params = alloc((largest,), np.uint8)
        code_data = np.fromfile(args.kernel, np.uint8)
        code = alloc((code_data.size,), np.uint8)
        route = alloc((8,), np.uint32)
        route_args = alloc((18,), np.uint32)
        dense[:] = model.raw
        io[:] = 0
        kv[:] = 0
        params[:] = 0
        code[:] = code_data
        route[:] = 0
        route_args[:] = 0
        route_args[0] = io.physical_address + moe_model.IO_SCORES
        route_args[1] = route.physical_address
        for value in buffers:
            value.flush()
        after = cma_kib()
        memory = {"allocated_buffer_bytes": int(sum(value.nbytes for value in buffers)),
                  "dense_bytes": int(dense.nbytes), "expert_slot_bytes": int(params.nbytes),
                  "io_bytes": int(io.nbytes), "kv_bytes": int(kv.nbytes),
                  "cma_before_kib": before, "cma_after_kib": after}
        print(f"allocated {memory['allocated_buffer_bytes']} bytes; CMA free "
              f"{before.get('CmaFree', -1)} -> {after.get('CmaFree', -1)} KiB", flush=True)

        bases = [dense.physical_address, io.physical_address, kv.physical_address]
        active = None
        records = []
        logits = []
        for pos, token in enumerate(tokens):
            io[:8] = np.frombuffer(CM.arg_block(pos, token), np.uint8)
            io.flush()
            token_params = CM.token_params(pos, model.d, cfg.head_size)
            for layer, stage in enumerate(stages):
                if active != layer:
                    blob = np.memmap(os.path.join(args.bundles, f"layer{layer}", "moe_params.bin"),
                                     dtype=np.uint8, mode="r")
                    params[:blob.size] = blob
                    params.flush()
                    active = layer
                sc = device.wait(device.submit(stage, bases=bases, params=token_params, perf=True), timeout=30.0)
                if sc["status"] != 0:
                    raise RuntimeError(f"layer {layer} SA stage failed: {sc}")
                route[:] = 0
                route.flush()
                route_args[16] = int(manifests[layer]["experts"])
                route_args[17] = int(manifests[layer]["expert_stride"])
                route_args.flush()
                rc = device.call(code.physical_address, route_args.physical_address,
                                 timeout=30.0, perf=True)
                if rc["status"] != 0:
                    raise RuntimeError(f"layer {layer} route failed: {rc}")
                fc = device.wait(device.submit(expert, bases=[
                    io.physical_address + moe_model.IO_FFN, route.physical_address,
                    params.physical_address + int(manifests[layer]["expert_table_offset"]),
                    io.physical_address + moe_model.IO_FFN], perf=True), timeout=30.0)
                if fc["status"] != 0:
                    raise RuntimeError(f"layer {layer} expert FFN failed: {fc}")
                route.invalidate()
                ids = route[1:3].astype(np.int32).tolist()
                records.append({"position": pos, "layer": layer, "experts": ids,
                                "dense_cycles": sc["cycles"], "route_cycles": rc["cycles"],
                                "ffn_cycles": fc["cycles"]})
                print(f"position {pos} layer {layer}: experts {ids}, dense {sc['cycles']}, "
                      f"route {rc['cycles']}, FFN {fc['cycles']} cycles", flush=True)
            last = device.wait(device.submit(final, bases=bases, params=token_params, perf=True), timeout=30.0)
            if last["status"] != 0:
                raise RuntimeError(f"final projection failed: {last}")
            io.invalidate()
            logits.append(np.asarray(io[CM.IO_LOGITS:CM.IO_LOGITS + 4 * cfg.vocab]
                                     .view("<f4"), dtype=np.float32).copy())
        got = np.stack(logits)
    finally:
        device.close()
        for value in reversed(buffers):
            value.freebuffer()

    with np.load(args.reference) as data:
        sim = data["sa"][:len(tokens)].copy()
    with np.load(args.hf_reference) as data:
        hf = data["hf"][:len(tokens)].copy()
    with open(args.sim_report) as f:
        sim_routes = json.load(f)["routing"]
    if got.shape != sim.shape or got.shape != hf.shape:
        raise ValueError("board, SA simulator, and HF logit shapes disagree")
    positions = [{"position": i, "board_top": int(np.argmax(got[i])),
                  "sim_top": int(np.argmax(sim[i])), "hf_top": int(np.argmax(hf[i])),
                  "relative_l2_vs_sim": relative_l2(got[i], sim[i]),
                  "max_abs_vs_sim": float(np.max(np.abs(got[i] - sim[i]))),
                  "relative_l2_vs_hf": relative_l2(got[i], hf[i])}
                 for i in range(len(tokens))]
    route_agreement = len(records) == len(sim_routes) and all(
        row["position"] == expected["position"] and
        row["layer"] == expected["layer"] and
        row["experts"] == expected["experts"]
        for row, expected in zip(records, sim_routes))
    passed = route_agreement and all(row["board_top"] == row["sim_top"] == row["hf_top"] and
                 row["relative_l2_vs_sim"] < 0.01 and row["max_abs_vs_sim"] < 0.01
                 for row in positions)
    report = {"tokens": tokens, "positions": positions, "calls": records,
              "memory": memory, "route_agreement": route_agreement, "full_device_pass": passed,
              "execution_scope": "SA embedding/norm/GQA/attention/router/experts/head; RISC-V top-2; ARM dispatch only"}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    for row in positions:
        print(f"position {row['position']}: board {row['board_top']}, sim {row['sim_top']}, "
              f"HF {row['hf_top']}, relL2 vs sim {row['relative_l2_vs_sim']:.6g}, "
              f"max abs {row['max_abs_vs_sim']:.6g}")
    print("full-device Mixtral board PASS" if passed else "full-device Mixtral board FAIL")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dense", default=os.path.join(HERE, "mixtral_dense.w8a8"))
    parser.add_argument("--bundles", default=os.path.join(HERE, "experts"))
    parser.add_argument("--reference", default=os.path.join(HERE, "full_device_sim_logits.npz"))
    parser.add_argument("--sim-report", default=os.path.join(HERE, "full_device_sim.json"))
    parser.add_argument("--hf-reference", default=os.path.join(HERE, "teacher_logits.npz"))
    parser.add_argument("--bitfile", default=os.path.join(HERE, "picorv32.bit"))
    parser.add_argument("--firmware", default=os.path.join(HERE, "rt_fw.bin"))
    parser.add_argument("--kernel", default=os.path.join(HERE, "moe_top2.bin"))
    parser.add_argument("--tokens", default="1,42,43")
    parser.add_argument("--out", default=os.path.join(HERE, "board_full_device_report.json"))
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

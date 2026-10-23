#!/usr/bin/env python3
"""Run a two-layer Mixtral model with host attention and KV260 expert FFNs."""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import moe  # noqa: E402
import moe_hf  # noqa: E402
import moe_mixtral_numpy  # noqa: E402
from moe_m3b import top2_ffn_template  # noqa: E402
from pynq_matmul import Device  # noqa: E402


def relative_l2(got, wanted):
    denominator = np.linalg.norm(wanted.astype(np.float64))
    return float(np.linalg.norm((got - wanted).astype(np.float64)) / max(denominator, 1e-12))


class BoardExperts:
    """One overlay and one reusable parameter buffer for both layers."""

    def __init__(self, layer_root, bitfile, firmware, kernel):
        from pynq import allocate

        self.layer_root = layer_root
        self.manifests = []
        for layer in range(2):
            with open(os.path.join(layer_root, f"layer{layer}", "moe_manifest.json")) as f:
                manifest = json.load(f)
            if manifest["source"]["layer"] != layer or manifest["source"]["adapter"] != "mixtral":
                raise ValueError(f"layer{layer}: wrong MoE manifest")
            if int(manifest["top_k"]) != 2:
                raise ValueError("board model runner currently requires top-2")
            self.manifests.append(manifest)
        shapes = {(m["dim"], m["hidden"], m["experts"], m["d"], m["alignment"])
                  for m in self.manifests}
        if len(shapes) != 1:
            raise ValueError("the two MoE layers require one shared descriptor geometry")
        self.dim, self.hidden, self.experts, self.d, self.alignment = map(int, shapes.pop())
        self.buffers = []
        self.device = Device(bitfile, firmware)
        self.active_layer = None
        self.calls = []
        try:
            if self.device.d != self.d:
                raise ValueError(f"overlay D={self.device.d} does not match MoE D={self.d}")
            def alloc(shape, dtype):
                value = allocate(shape=shape, dtype=dtype)
                self.buffers.append(value)
                return value

            code_bytes = np.fromfile(kernel, np.uint8)
            self.code = alloc((len(code_bytes),), np.uint8)
            self.code[:] = code_bytes
            self.code.flush()
            self.scores = alloc((self.experts,), np.float32)
            self.route = alloc((8,), np.uint32)
            self.args = alloc((18,), np.uint32)
            self.x = alloc((self.dim,), np.float32)
            self.out = alloc((self.dim,), np.float32)
            largest = max(os.path.getsize(os.path.join(layer_root, f"layer{i}", "moe_params.bin"))
                          for i in range(2))
            self.params = alloc((largest,), np.uint8)
            self.args[:] = 0
            self.args[0] = self.scores.physical_address
            self.args[1] = self.route.physical_address
            self.args.flush()
            self.descriptor = top2_ffn_template(self.dim, self.hidden, self.d,
                                                self.alignment, ret=False)
        except Exception:
            self.close()
            raise

    def load_layer(self, layer):
        if layer == self.active_layer:
            return
        manifest = self.manifests[layer]
        path = os.path.join(self.layer_root, f"layer{layer}", "moe_params.bin")
        data = np.fromfile(path, np.uint8)
        if data.size > self.params.size:
            raise ValueError(f"layer {layer}: parameter buffer too small")
        self.params[:data.size] = data
        self.params.flush()
        self.args[16] = self.experts
        self.args[17] = int(manifest["expert_stride"])
        self.args.flush()
        self.active_layer = layer
        print(f"layer {layer}: loaded {data.size} packed parameter bytes", flush=True)

    def __call__(self, layer, position, x, scores):
        self.load_layer(layer)
        manifest = self.manifests[layer]
        ids, weights = moe.route(scores, 2)
        self.x[:] = x
        self.scores[:] = scores
        self.route[:] = 0
        self.x.flush(); self.scores.flush(); self.route.flush()
        rc = self.device.call(self.code.physical_address, self.args.physical_address, perf=True)
        self.route.invalidate()
        got_ids = self.route[1:3].astype(np.int32)
        got_weights = self.route[5:7].view(np.float32)
        if rc["status"] != 0 or not np.array_equal(got_ids, ids) or not np.allclose(
                got_weights, weights, rtol=2e-4, atol=2e-4):
            raise RuntimeError(f"layer {layer}, pos {position}: route mismatch/status {rc['status']}")
        self.out[:] = 0
        self.out.flush()
        fc = self.device.wait(self.device.submit(
            self.descriptor, bases=[self.x.physical_address, self.route.physical_address,
                                    self.params.physical_address + int(manifest["expert_table_offset"]),
                                    self.out.physical_address], perf=True))
        if fc["status"] != 0:
            raise RuntimeError(f"layer {layer}, pos {position}: FFN status {fc['status']}")
        self.out.invalidate()
        self.calls.append({"layer": layer, "position": position, "ids": ids.tolist(),
                           "route_cycles": rc["cycles"], "ffn_cycles": fc["cycles"]})
        print(f"layer {layer} pos {position}: experts {ids.tolist()}, route {rc['cycles']} cycles, "
              f"FFN {fc['cycles']} cycles", flush=True)
        return np.asarray(self.out, np.float32).copy()

    def close(self):
        try:
            if hasattr(self, "device"):
                self.device.close()
        finally:
            for buffer in reversed(self.buffers):
                buffer.freebuffer()
            self.buffers.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.path.join(HERE, "model_dense"))
    parser.add_argument("--layers", default=os.path.join(HERE, "..", "moe-m4c"))
    parser.add_argument("--reference", default=os.path.join(HERE, "teacher_logits.npz"))
    parser.add_argument("--bitfile", default=os.path.join(HERE, "..", "moe-m4c", "layer0", "picorv32.bit"))
    parser.add_argument("--firmware", default=os.path.join(HERE, "..", "moe-m4c", "layer0", "rt_fw.bin"))
    parser.add_argument("--kernel", default=os.path.join(HERE, "..", "moe-m4c", "layer0", "moe_top2.bin"))
    parser.add_argument("--tokens", default="1,42,43")
    parser.add_argument("--out", default=os.path.join(HERE, "board_model_report.json"))
    args = parser.parse_args()
    tokens = [int(value) for value in args.tokens.split(",")]
    config = moe_hf.load_config(args.model)
    if config.layers != 2:
        raise ValueError("this board runner currently supports the two-layer tiny Mixtral profile")
    with np.load(args.reference) as reference:
        expected_hybrid = reference["hybrid"].copy()
        expected_hf = reference["hf"].copy()
    engine = BoardExperts(args.layers, args.bitfile, args.firmware, args.kernel)
    try:
        logits, _ = moe_mixtral_numpy.forward(args.model, tokens, engine)
    finally:
        engine.close()
    if logits.shape != expected_hybrid.shape or logits.shape != expected_hf.shape:
        raise ValueError("reference logit shape does not match board output")
    positions = []
    for pos in range(len(tokens)):
        got = logits[pos]
        hybrid = expected_hybrid[pos]
        positions.append({"position": pos, "board_top": int(np.argmax(got)),
                          "sim_hybrid_top": int(np.argmax(hybrid)),
                          "hf_top": int(np.argmax(expected_hf[pos])),
                          "relative_l2_vs_sim": relative_l2(got, hybrid),
                          "max_abs_vs_sim": float(np.max(np.abs(got - hybrid)))})
    success = all(p["board_top"] == p["sim_hybrid_top"] and
                  p["relative_l2_vs_sim"] < 0.01 and p["max_abs_vs_sim"] < 0.01
                  for p in positions)
    report = {"tokens": tokens, "positions": positions, "device_calls": engine.calls,
              "board_vs_hybrid_pass": success,
              "board_vs_hf_top_agreement": sum(p["board_top"] == p["hf_top"] for p in positions)}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    for p in positions:
        print(f"position {p['position']}: board {p['board_top']}, sim {p['sim_hybrid_top']}, "
              f"HF {p['hf_top']}, relL2 vs sim {p['relative_l2_vs_sim']:.6g}, "
              f"max abs {p['max_abs_vs_sim']:.6g}")
    print("two-layer board model PASS" if success else "two-layer board model FAIL")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

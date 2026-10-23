#!/usr/bin/env python3
"""M4: Hugging Face Mixtral layer import, packing, and SA execution."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for path in (os.path.join(REPO, "compiler", "frontend"), os.path.join(REPO, "compiler", "runtime", "tools"),
             HERE, os.path.join(REPO, "driver"), os.path.join(REPO, "llm")):
    sys.path.insert(0, path)

import moe  # noqa: E402
import moe_hf  # noqa: E402
import moe_m3b  # noqa: E402
import sadesc  # noqa: E402
from sa_funcsim import SaFuncSim  # noqa: E402
from synth_hf import write_safetensors  # noqa: E402


class MoeM4Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.model = os.path.join(self.temp.name, "tiny-mixtral")
        self.bundle = os.path.join(self.temp.name, "bundle")
        os.makedirs(self.model)
        self.dim, self.hidden, self.experts = 32, 64, 4
        config = {"architectures": ["MixtralForCausalLM"], "model_type": "mixtral",
                  "hidden_size": self.dim, "intermediate_size": self.hidden,
                  "num_hidden_layers": 1, "num_local_experts": self.experts,
                  "num_experts_per_tok": 2, "norm_topk_prob": True}
        with open(os.path.join(self.model, "config.json"), "w") as f:
            json.dump(config, f)
        rng = np.random.default_rng(83)
        linear = lambda shape, fan_in: rng.normal(0, 1 / np.sqrt(fan_in), shape).astype(np.float32)
        root = "model.layers.0.block_sparse_moe"
        first = {root + ".gate.weight": linear((self.experts, self.dim), self.dim)}
        second = {"model.embed_tokens.weight": linear((128, self.dim), self.dim)}
        self.expected = {"router": first[root + ".gate.weight"], "gate": [], "up": [], "down": []}
        for expert in range(self.experts):
            p = f"{root}.experts.{expert}"
            values = {p + ".w1.weight": linear((self.hidden, self.dim), self.dim),
                      p + ".w3.weight": linear((self.hidden, self.dim), self.dim),
                      p + ".w2.weight": linear((self.dim, self.hidden), self.hidden)}
            (first if expert < 2 else second).update(values)
            self.expected["gate"].append(values[p + ".w1.weight"])
            self.expected["up"].append(values[p + ".w3.weight"])
            self.expected["down"].append(values[p + ".w2.weight"])
        write_safetensors(os.path.join(self.model, "model-00001-of-00002.safetensors"), first)
        write_safetensors(os.path.join(self.model, "model-00002-of-00002.safetensors"), second)
        weight_map = {name: ("model-00001-of-00002.safetensors" if name in first
                             else "model-00002-of-00002.safetensors") for name in first | second}
        with open(os.path.join(self.model, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f)

    def tearDown(self):
        self.temp.cleanup()

    def test_mixtral_schema_maps_to_common_expert_representation(self):
        config, weights, source = moe_hf.load_layer(self.model)
        self.assertEqual(config, moe.MoeConfig(32, 64, 4, 2))
        self.assertEqual(source["adapter"], "mixtral")
        np.testing.assert_array_equal(weights.router, self.expected["router"])
        for name in ("gate", "up", "down"):
            np.testing.assert_array_equal(getattr(weights, name), np.stack(self.expected[name]))

    def test_checkpoint_pack_runs_on_m3b_descriptor(self):
        config, quantized, manifest = moe_hf.pack_layer(self.model, self.bundle)
        with open(os.path.join(self.bundle, "moe_params.bin"), "rb") as f:
            params = f.read()
        with open(os.path.join(self.bundle, "moe_manifest.json")) as f:
            saved = json.load(f)
        self.assertEqual(saved, manifest)
        self.assertEqual(manifest["source"]["architecture"], "MixtralForCausalLM")
        self.assertEqual(manifest["capacity"]["active_fraction"], 0.5)
        self.assertEqual(manifest["capacity"]["active_expert_bytes_per_token"],
                         2 * manifest["expert_stride"])
        reference = np.load(os.path.join(self.bundle, "moe_acceptance.npz"))
        self.assertEqual(reference["scores"].shape, (4, config.experts))
        self.assertEqual(reference["output"].shape, (4, config.dim))

        x = np.linspace(-0.625, 0.75, config.dim, dtype=np.float32)
        ids = np.array([3, 1], np.uint32)
        routing = np.array([0.8, 0.2], np.float32)
        offsets = ids * np.uint32(manifest["expert_stride"])
        route = moe.pack_route_record(ids, offsets, routing)
        _, _, exports = sadesc.read(moe_m3b.make_executable(config.dim, config.hidden, config.d))
        rows = exports[0][1]
        base = 0x10000000
        template, xaddr, routeaddr, paramaddr, outaddr, listaddr = [
            base + value for value in (0, 0x10000, 0x11000, 0x12000, 0x30000, 0x31000)]
        sim = SaFuncSim(config.d, base, 0x40000)
        sim.ddr_write(template, rows.tobytes())
        sim.ddr_write(xaddr, x)
        sim.ddr_write(routeaddr, route)
        sim.ddr_write(paramaddr, params)
        bindings = [xaddr, routeaddr, paramaddr + manifest["expert_table_offset"], outaddr]
        dispatch = sadesc.dispatch_list(template, bindings, [])
        sim.ddr_write(listaddr, dispatch.array().tobytes())
        sim.run_list(listaddr)
        got = sim.ddr_read(outaddr, config.dim * 4).view("<f4")
        xq, xs = moe.quantize_activation(x)
        expected = sum((weight * moe.w8a8_expert(xq, xs, quantized, int(expert))).astype(np.float32)
                       for expert, weight in zip(ids, routing))
        np.testing.assert_allclose(got, expected, rtol=3e-3, atol=3e-3)

    def test_unsupported_dense_architecture_is_diagnosed(self):
        with open(os.path.join(self.model, "config.json"), "w") as f:
            json.dump({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}, f)
        with self.assertRaisesRegex(moe_hf.UnsupportedMoeCheckpoint, "unsupported MoE architecture"):
            moe_hf.load_config(self.model)


if __name__ == "__main__":
    unittest.main()

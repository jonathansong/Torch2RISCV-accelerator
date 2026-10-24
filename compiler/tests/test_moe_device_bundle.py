#!/usr/bin/env python3
"""Check model-derived MoE artifacts against the IREE-recognized ABI."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for directory in ("compiler/frontend", "compiler/runtime/tools", "llm"):
    sys.path.insert(0, str(REPO / directory))

import moe  # noqa: E402
from moe_device_bundle import build, recognized_fields, route_source  # noqa: E402
from moe_iree_chain import build as build_chain, COMPILE  # noqa: E402
from synth_hf import write_safetensors  # noqa: E402
import sadesc  # noqa: E402
from sa_funcsim import SaFuncSim  # noqa: E402


class MoeDeviceBundleTest(unittest.TestCase):
    def test_synthetic_checkpoint_builds_reusable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({
                "architectures": ["MixtralForCausalLM"], "model_type": "mixtral",
                "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                "num_local_experts": 4, "num_experts_per_tok": 2,
                "norm_topk_prob": True}) + "\n")
            weights = moe.synthetic_weights(moe.MoeConfig(32, 64, 4, 2), seed=94)
            prefix = "model.layers.0.block_sparse_moe"
            tensors = {f"{prefix}.gate.weight": weights.router}
            for expert in range(4):
                expert_prefix = f"{prefix}.experts.{expert}"
                tensors[f"{expert_prefix}.w1.weight"] = weights.gate[expert]
                tensors[f"{expert_prefix}.w3.weight"] = weights.up[expert]
                tensors[f"{expert_prefix}.w2.weight"] = weights.down[expert]
            write_safetensors(str(model / "model.safetensors"), tensors)
            out = root / "bundle"
            result = build(model, 0, out)
            self.assertEqual(result["recognized"], {"dim": 32, "hidden": 64,
                                                      "experts": 4, "expert_stride": 6784})
            manifest = json.loads((out / "moe_manifest.json").read_text())
            self.assertEqual(manifest["expert_stride"], result["recognized"]["expert_stride"])
            self.assertEqual(recognized_fields(out / "preprocessing.mlir"), result["recognized"])
            coverage = result["native_module_dispatches"]
            self.assertTrue(any("topk" in name for name in coverage["sa"]))
            self.assertTrue(any("batch_matmul" in name for name in coverage["host"]))
            self.assertEqual(coverage, json.loads((out / "compiler_bundle.json").read_text())
                             ["native_module_dispatches"])
            self.assertEqual(len(result["artifacts"]), 6)
            for name, info in result["artifacts"].items():
                self.assertEqual((out / name).stat().st_size, info["bytes"])
                self.assertGreater(info["bytes"], 0)
            self.assertIn("experts = 4 : i64, stride = 6784 : i64",
                          (out / "route.mlir").read_text())
            self.assertGreater((out / "moe_route.bin").stat().st_size, 0)

            # Execute the newly generated reusable FFN export using the same
            # model-derived packed parameters and dynamic route offsets.
            descriptor_d, _, exports = sadesc.read((out / "moe_ffn.sadesc").read_bytes())
            self.assertEqual(descriptor_d, 16)
            self.assertEqual(len(exports), 1)
            x = np.linspace(-0.625, 0.75, 32, dtype=np.float32)
            quantized = moe.quantize_weights(moe.MoeConfig(32, 64, 4, 2), weights)
            _, _, ids, routing = moe.moe_forward(x, moe.MoeConfig(32, 64, 4, 2),
                                                  quantized, quantized=True)
            offsets = ids.astype(np.uint32) * np.uint32(manifest["expert_stride"])
            route = moe.pack_route_record(ids, offsets, routing)
            base = 0x10000000
            template, xaddr, routeaddr, paramaddr, outaddr, listaddr = [
                base + value for value in (0, 0x10000, 0x11000, 0x12000, 0x30000, 0x31000)]
            sim = SaFuncSim(16, base, 0x40000)
            sim.ddr_write(template, exports[0][1].tobytes())
            sim.ddr_write(xaddr, x)
            sim.ddr_write(routeaddr, route)
            sim.ddr_write(paramaddr, (out / "moe_params.bin").read_bytes())
            dispatch = sadesc.dispatch_list(template, [xaddr, routeaddr,
                paramaddr + manifest["expert_table_offset"], outaddr], [])
            sim.ddr_write(listaddr, dispatch.array().tobytes())
            sim.run_list(listaddr)
            got = sim.ddr_read(outaddr, 32 * 4).view("<f4")
            xq, xs = moe.quantize_activation(x)
            expected = sum((weight * moe.w8a8_expert(xq, xs, quantized, int(expert))).astype(np.float32)
                           for expert, weight in zip(ids, routing))
            np.testing.assert_allclose(got, expected, rtol=3e-3, atol=3e-3)

            # A single VMFB carries both existing device exports and schedules
            # the route record producer before its FFN consumer.
            chain = build_chain(out, root / "chain")
            self.assertGreater(chain.stat().st_size, 0)
            lowered = root / "chain_vm.mlir"
            subprocess.run([str(COMPILE), str(root / "chain/moe_chain.mlir"),
                            "--compile-from=hal", "--compile-to=vm", "-o", str(lowered)],
                           check=True, capture_output=True, text=True)
            vm = lowered.read_text()
            self.assertEqual(vm.count("vm.call.variadic @hal.command_buffer.dispatch("), 2)
            self.assertEqual(vm.count("vm.call @hal.command_buffer.execution_barrier("), 1)
            self.assertEqual(vm.count("vm.call @hal.device.queue.execute("), 1)

    def test_rejects_other_device_width(self):
        with self.assertRaisesRegex(ValueError, "D=16"):
            build("missing", 0, "missing", d=8)
        self.assertIn("experts = 8 : i64, stride = 11042816 : i64",
                      route_source(8, 11042816))


if __name__ == "__main__":
    unittest.main()

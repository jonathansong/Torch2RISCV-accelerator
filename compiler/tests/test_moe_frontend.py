#!/usr/bin/env python3
"""Unit tests for the first, framework-independent MoE milestone."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "frontend"))

import moe  # noqa: E402


class MoeFrontendTest(unittest.TestCase):
    def setUp(self):
        self.config = moe.MoeConfig(dim=32, hidden=64, experts=4, top_k=2)
        self.fp = moe.synthetic_weights(self.config, seed=7)
        self.q = moe.quantize_weights(self.config, self.fp)
        self.x = np.linspace(-1, 1, self.config.dim, dtype=np.float32)

    def test_deterministic_topk_and_ties(self):
        ids, weights = moe.route(np.array([2, 2, 1, 2], np.float32), 3)
        np.testing.assert_array_equal(ids, [0, 1, 3])
        np.testing.assert_allclose(weights, np.full(3, 1 / 3, np.float32), rtol=0, atol=1e-7)

    def test_host_routing_record_round_trip(self):
        _, manifest = moe.pack_parameters(self.config, self.q)
        scores = np.array([-2, 4, 1, 3], np.float32)
        record, ids, offsets, weights = moe.host_route_record(scores, manifest, top_k=2)
        self.assertEqual(len(record), moe.ROUTE_RECORD_SIZE)
        got_ids, got_offsets, got_weights = moe.unpack_route_record(record)
        np.testing.assert_array_equal(got_ids, ids)
        np.testing.assert_array_equal(got_offsets, offsets)
        np.testing.assert_array_equal(offsets, ids.astype(np.uint32) * manifest["expert_stride"])
        np.testing.assert_array_equal(got_weights, weights)

    def test_forced_experts_match_manual_fp32_combine(self):
        ids = np.array([3, 1], np.int32)
        routing = np.array([0.25, 0.75], np.float32)
        got, _, got_ids, got_weights = moe.moe_forward(
            self.x, self.config, self.fp, forced_ids=ids, forced_weights=routing)
        expected = (routing[0] * moe.fp32_expert(self.x, self.fp, 3) +
                    routing[1] * moe.fp32_expert(self.x, self.fp, 1))
        np.testing.assert_array_equal(got_ids, ids)
        np.testing.assert_array_equal(got_weights, routing)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_every_expert_is_addressable_in_w8a8_reference(self):
        for expert in range(self.config.experts):
            ids = np.array([expert, expert], np.int32)
            weights = np.array([1, 0], np.float32)
            got, _, _, _ = moe.moe_forward(self.x, self.config, self.q, forced_ids=ids,
                                            forced_weights=weights, quantized=True)
            xq, xs = moe.quantize_activation(self.x)
            expected = moe.w8a8_expert(xq, xs, self.q, expert)
            np.testing.assert_array_equal(got, expected)

    def test_zero_activation_is_finite(self):
        zero = np.zeros(self.config.dim, np.float32)
        got, scores, _, routing = moe.moe_forward(zero, self.config, self.q, quantized=True)
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(np.isfinite(routing)))
        np.testing.assert_array_equal(got, np.zeros_like(got))

    def test_backend_linear_layout_round_trip(self):
        weight = self.q.gate_q[2]
        packed = moe.pack_linear(weight, self.config.d)
        got = moe.unpack_linear(packed, *weight.shape, self.config.d)
        np.testing.assert_array_equal(got, weight)

    def test_parameter_manifest_has_uniform_aligned_experts(self):
        blob, manifest = moe.pack_parameters(self.config, self.q)
        json.dumps(manifest)
        self.assertEqual(len(blob), manifest["total_bytes"])
        self.assertEqual(manifest["format"], "sa-moe-parameters-v1")
        stride = manifest["expert_stride"]
        base = manifest["expert_table_offset"]
        self.assertEqual(base % self.config.alignment, 0)
        self.assertEqual(stride % self.config.alignment, 0)
        for expert, record in enumerate(manifest["expert_records"]):
            self.assertEqual(record["offset"], base + expert * stride)
            self.assertEqual(record["size"], stride)
            for section in record["sections"]:
                self.assertEqual(section["offset"] % self.config.alignment, 0)
                self.assertLessEqual(section["offset"] + section["size"], len(blob))

    def test_synthetic_bundle_is_self_contained(self):
        with tempfile.TemporaryDirectory() as path:
            manifest = moe.write_synthetic_bundle(path, self.config, seed=3)
            with open(os.path.join(path, "moe_manifest.json")) as f:
                saved_manifest = json.load(f)
            with open(os.path.join(path, "moe_params.bin"), "rb") as f:
                blob = f.read()
            vectors = np.load(os.path.join(path, "moe_reference.npz"))
            self.assertEqual(saved_manifest, manifest)
            self.assertEqual(len(blob), manifest["total_bytes"])
            self.assertEqual(vectors["x"].shape, (self.config.dim,))
            self.assertEqual(vectors["w8a8_output"].shape, (self.config.dim,))
            self.assertEqual(vectors["w8a8_ids"].shape, (self.config.top_k,))

    def test_validation_rejects_invalid_shape_and_topk(self):
        with self.assertRaisesRegex(ValueError, "top-k"):
            moe.MoeConfig(dim=32, hidden=64, experts=2, top_k=3)
        bad = moe.MoeWeights(self.fp.router, self.fp.gate[:, :-1], self.fp.up, self.fp.down)
        with self.assertRaisesRegex(ValueError, "gate has shape"):
            moe.validate_weights(self.config, bad)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Check the tensor-level Mixtral frontend against the M4 reference."""

import os
import sys
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "frontend"))

import moe  # noqa: E402
import moe_native  # noqa: E402


class NativeMoeTest(unittest.TestCase):
    def test_synthetic_layer_matches_reference(self):
        config = moe.MoeConfig(32, 64, 4, 2)
        weights = moe.synthetic_weights(config, seed=94)
        layer = moe_native.NativeMixtralMoe(weights)
        inputs = [np.linspace(-0.625, 0.75, config.dim, dtype=np.float32),
                  np.zeros(config.dim, dtype=np.float32)]
        rng = np.random.default_rng(714)
        inputs.extend(rng.normal(size=config.dim).astype(np.float32) for _ in range(16))
        with torch.no_grad():
            for x in inputs:
                got = layer(torch.from_numpy(x[None]))
                expected = moe.moe_forward(x, config, weights)
                np.testing.assert_allclose(got[0].numpy()[0], expected[0], rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(got[1].numpy()[0], expected[1], rtol=2e-6, atol=2e-6)
                np.testing.assert_array_equal(got[2].numpy()[0], expected[2])
                np.testing.assert_allclose(got[3].numpy()[0], expected[3], rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()

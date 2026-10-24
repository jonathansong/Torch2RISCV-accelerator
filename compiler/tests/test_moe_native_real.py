#!/usr/bin/env python3
"""Compare native Torch MoE against the existing real Mixtral token case."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "compiler/frontend"))
import moe_hf  # noqa: E402
from moe_native import NativeMixtralMoe  # noqa: E402


class NativeMoeRealTest(unittest.TestCase):
    def test_cached_mixtral_layer0_token(self):
        checkpoint = REPO / "build/llm_cache/TitanML-tiny-mixtral"
        case = REPO / "build/moe/m4c/tiny-mixtral/layer0/moe_token_acceptance.npz"
        if not checkpoint.is_dir() or not case.is_file():
            self.skipTest("cached Mixtral checkpoint and M4 token case required")
        _, weights, _ = moe_hf.load_layer(str(checkpoint), 0)
        vectors = np.load(case)
        with torch.no_grad():
            result, scores, ids, routing = NativeMixtralMoe(weights)(
                torch.from_numpy(vectors["x"].copy()[None]))
        np.testing.assert_array_equal(ids.numpy(), vectors["pairs"].astype(np.int64))
        np.testing.assert_allclose(scores.numpy(), vectors["scores"], rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(routing.numpy(), vectors["routing"], rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(result.numpy()[0], vectors["hf_output"],
                                   rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()

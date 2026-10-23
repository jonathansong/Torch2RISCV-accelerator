#!/usr/bin/env python3
"""M1: host top-1 record drives a dynamically addressed GEMV in the simulator."""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "compiler", "frontend"))
sys.path.insert(0, os.path.join(REPO, "compiler", "runtime", "tools"))
sys.path.insert(0, os.path.join(REPO, "driver"))
sys.path.insert(0, os.path.join(REPO, "llm"))

import moe  # noqa: E402
import moe_m1  # noqa: E402
import sadesc  # noqa: E402
from pynq_matmul import DescList  # noqa: E402
from sa_funcsim import MODE_GEMV, SaFuncSim  # noqa: E402


D = 16
DDR_BASE = 0x10000000
TEMPLATE = DDR_BASE
X_ADDR = DDR_BASE + 0x10000
ROUTE_ADDR = DDR_BASE + 0x11000
PARAM_ADDR = DDR_BASE + 0x12000
OUT_ADDR = DDR_BASE + 0x20000
LIST_ADDR = DDR_BASE + 0x21000


class MoeM1Test(unittest.TestCase):
    def setUp(self):
        self.config = moe.MoeConfig(dim=32, hidden=64, experts=4, top_k=1, d=D)
        self.fp = moe.synthetic_weights(self.config, seed=19)
        self.q = moe.quantize_weights(self.config, self.fp)
        self.params, self.manifest = moe.pack_parameters(self.config, self.q)
        rng = np.random.default_rng(23)
        self.x = rng.integers(-127, 128, self.config.dim, dtype=np.int8)
        _, _, exports = sadesc.read(moe_m1.make_executable(self.config.dim, self.config.hidden, D))
        self.rows = exports[0][1]

    def run_scores(self, scores):
        route_record, ids, offsets, _ = moe.host_route_record(scores, self.manifest, top_k=1)
        expert = int(ids[0])
        sim = SaFuncSim(D, DDR_BASE, 0x40000)
        sim.trace = []
        sim.ddr_write(TEMPLATE, self.rows.tobytes())
        sim.ddr_write(X_ADDR, self.x)
        sim.ddr_write(ROUTE_ADDR, route_record)
        sim.ddr_write(PARAM_ADDR, self.params)
        sim.ddr_write(OUT_ADDR, bytes(self.config.hidden * 4))
        bindings = [X_ADDR, ROUTE_ADDR, PARAM_ADDR + self.manifest["expert_table_offset"], OUT_ADDR]
        dispatch = sadesc.dispatch_list(TEMPLATE, bindings, [], setup=[])
        sim.ddr_write(LIST_ADDR, dispatch.array().tobytes())
        sim.run_list(LIST_ADDR)
        got = sim.ddr_read(OUT_ADDR, self.config.hidden * 4).view("<i4")
        want = self.q.gate_q[expert].astype(np.int32) @ self.x.astype(np.int32)
        gemv = [entry for entry in sim.trace
                if (entry[0] & 0xFF) == DescList.LD and ((entry[5] >> 32) & 3) == MODE_GEMV]
        self.assertEqual(len(gemv), 1)
        expected_weight_addr = bindings[2] + int(offsets[0])
        self.assertEqual(gemv[0][1], expected_weight_addr)
        np.testing.assert_array_equal(got, want)
        return expert

    def test_all_experts_are_selected_by_dynamic_offset(self):
        for expert in range(self.config.experts):
            with self.subTest(expert=expert):
                scores = np.full(self.config.experts, -10, np.float32)
                scores[expert] = 10
                self.assertEqual(self.run_scores(scores), expert)

    def test_w8a8_router_selects_the_device_expert(self):
        xq, xs = moe.quantize_activation(self.x)
        scores = moe.qlinear(xq, xs, self.q.router_q, self.q.router_scale)
        expected = int(moe.route(scores, 1)[0][0])
        self.assertEqual(self.run_scores(scores), expected)

    def test_template_has_one_shared_gemv(self):
        gemv_descriptors = [row for row in self.rows
                            if (int(row[0]) & 0xFF) == DescList.LD and ((int(row[3]) >> 32) & 3) == MODE_GEMV]
        self.assertEqual(len(gemv_descriptors), 1)
        slot = (int(gemv_descriptors[0][0]) >> 16) & 0xFF
        self.assertEqual(slot & 0xF, DescList.DYN_FIELDS[DescList.LD]["ddr"])
        self.assertEqual((slot >> 4) & 7, 6)
        self.assertTrue(slot & 0x80)


if __name__ == "__main__":
    unittest.main()

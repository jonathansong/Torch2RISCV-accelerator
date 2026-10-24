#!/usr/bin/env python3
"""Native IREE Top-2 output feeds the existing MoE route and FFN ABI."""

import ctypes
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "compiler", "frontend"), HERE]
import moe  # noqa: E402
import moe_route_from_top2_kernel  # noqa: E402
import moe_top2_kernel  # noqa: E402
from test_moe_top2_raw import topk_reference  # noqa: E402


class RouteFromTop2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.keep = {}
        callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)
        cls.callback = callback_type(lambda address: cls.keep[address].ctypes.data)
        cls.bridge_code, bridge_host = moe_route_from_top2_kernel.build(
            os.path.join(cls.temp.name, "bridge"))
        _, old_host = moe_top2_kernel.build(os.path.join(cls.temp.name, "original"))
        cls.bridge = ctypes.CDLL(bridge_host)
        cls.original = ctypes.CDLL(old_host)
        cls.bridge.moe_route_from_top2_host_init(cls.callback)
        cls.original.moe_top2_host_init(cls.callback)
        for lib in (cls.bridge, cls.original):
            lib.r5_entry.argtypes = [ctypes.POINTER(ctypes.c_uint32),
                                     ctypes.POINTER(ctypes.c_uint32)]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_bridge(self, values, ids, count, stride):
        route = np.full(8, 0xBAD0BAD0, np.uint32)
        self.keep.clear()
        self.keep.update({0x1000: np.ascontiguousarray(values, np.float32),
                          0x2000: np.ascontiguousarray(ids, np.uint32),
                          0x3000: route})
        args = (ctypes.c_uint32 * 3)(0x1000, 0x2000, 0x3000)
        constants = (ctypes.c_uint32 * 2)(count, stride)
        self.bridge.r5_entry(args, constants)
        return route

    def run_original(self, scores, stride):
        route = np.zeros(8, np.uint32)
        self.keep.clear()
        self.keep.update({0x1000: np.ascontiguousarray(scores, np.float32),
                          0x3000: route})
        args = (ctypes.c_uint32 * 2)(0x1000, 0x3000)
        constants = (ctypes.c_uint32 * 2)(len(scores), stride)
        self.original.r5_entry(args, constants)
        return route

    def test_finite_scores_match_existing_m4_route(self):
        self.assertGreater(len(self.bridge_code), 0)
        rng = np.random.default_rng(501)
        cases = [np.array([5, 5, 1, 0], np.float32),
                 np.array([-4, -3, -2, -1], np.float32),
                 np.array([7, -20, 6, -30], np.float32)]
        cases.extend(rng.normal(size=n).astype(np.float32)
                     for n in (4, 8, 16) for _ in range(20))
        for scores in cases:
            with self.subTest(scores=scores):
                values, ids = topk_reference(scores)
                got = self.run_bridge(values, ids, len(scores), 6784)
                expected = self.run_original(scores, 6784)
                np.testing.assert_array_equal(got, expected)
                route_ids, offsets, weights = moe.unpack_route_record(got.tobytes())
                np.testing.assert_array_equal(route_ids, ids)
                np.testing.assert_array_equal(offsets, ids.astype(np.uint32) * 6784)
                self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

    def test_invalid_top2_is_rejected(self):
        for values, ids, count, stride in [
            ([1, 0], [0, 0], 4, 6784),
            ([1, 0], [0, 4], 4, 6784),
            ([np.nan, 0], [0, 1], 4, 6784),
            ([np.inf, 0], [0, 1], 4, 6784),
            ([1, 0], [0, 1], 4, 0),
            ([1, 0], [0, 1], 16, 0x40000000),
        ]:
            with self.subTest(values=values, ids=ids, count=count, stride=stride):
                route = self.run_bridge(values, ids, count, stride)
                self.assertEqual(route[0], 0)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Check the call-free rv32 MoE bridge against exact finite Top-2 routing."""

import ctypes
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moe_route_from_top2_int_kernel  # noqa: E402
from test_moe_top2_raw import topk_reference  # noqa: E402


class IntegerRouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.code, host = moe_route_from_top2_int_kernel.build(cls.temp.name)
        cls.lib = ctypes.CDLL(host)
        cls.keep = {}
        callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)
        cls.callback = callback_type(lambda address: cls.keep[address].ctypes.data)
        cls.lib.moe_route_int_host_init(cls.callback)
        cls.lib.r5_entry.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def route(self, values, ids, count, stride=6784):
        record = np.full(8, 0xBAD0BAD0, np.uint32)
        self.keep.clear()
        self.keep.update({0x1000: np.ascontiguousarray(values, np.float32),
                          0x2000: np.ascontiguousarray(ids, np.uint32),
                          0x3000: record})
        args = (ctypes.c_uint32 * 3)(0x1000, 0x2000, 0x3000)
        constants = (ctypes.c_uint32 * 2)(count, stride)
        self.lib.r5_entry(args, constants)
        return record

    def test_finite_router_scores(self):
        self.assertGreater(len(self.code), 0)
        self.assertLess(len(self.code), 1024)
        rng = np.random.default_rng(182)
        cases = [np.array([7, 6, -2, -3], np.float32),
                 np.array([5, 5, 5, 5], np.float32),
                 np.array([54, 40, -8, -9], np.float32),
                 np.array([-400, -401, -402, -403], np.float32)]
        cases += [rng.uniform(-100, 100, n).astype(np.float32)
                  for n in (4, 8, 16) for _ in range(200)]
        max_error = 0.0
        worst = None
        for scores in cases:
            with self.subTest(scores=scores):
                values, ids = topk_reference(scores)
                record = self.route(values, ids, len(scores))
                self.assertEqual(record[0], 2)
                np.testing.assert_array_equal(record[1:3], ids)
                np.testing.assert_array_equal(record[3:5], ids.astype(np.uint32) * 6784)
                self.assertEqual(record[7], 0)
                weights = record[5:7].view(np.float32)
                exact = np.exp((values - values[0]).astype(np.float64))
                exact /= exact.sum()
                error = float(np.max(np.abs(weights - exact)))
                if error > max_error:
                    max_error, worst = error, scores
        self.assertLess(max_error, 5e-5, f"worst scores {worst}")

    def test_rejects_invalid_values_and_offsets(self):
        for values, ids, count, stride in [
            ([np.nan, 0], [0, 1], 4, 6784),
            ([np.inf, 0], [0, 1], 4, 6784),
            ([512, 0], [0, 1], 4, 6784),
            ([0, 1], [0, 1], 4, 6784),
            ([1, 0], [0, 0], 4, 6784),
            ([1, 0], [0, 4], 4, 6784),
            ([1, 0], [0, 1], 4, 0),
            ([1, 0], [0, 1], 16, 0x40000000),
        ]:
            with self.subTest(values=values, ids=ids):
                self.assertEqual(self.route(values, ids, count, stride)[0], 0)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Raw Top-2 firmware preserves finite-score IREE routing values and IDs."""

import ctypes
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moe_top2_raw_kernel  # noqa: E402


class RawTop2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.code, host = moe_top2_raw_kernel.build(cls.temp.name)
        cls.lib = ctypes.CDLL(host)
        cls.keep = {}
        callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)
        cls.callback = callback_type(lambda address: cls.keep[address].ctypes.data)
        cls.lib.moe_top2_raw_host_init(cls.callback)
        cls.lib.r5_entry.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def route(self, scores):
        scores = np.ascontiguousarray(scores, dtype=np.float32)
        values = np.empty(2, dtype=np.float32)
        indices = np.empty(2, dtype=np.uint32)
        self.keep.clear()
        self.keep.update({0x1000: scores, 0x2000: values, 0x3000: indices})
        args = (ctypes.c_uint32 * 3)(0x1000, 0x2000, 0x3000)
        constants = (ctypes.c_uint32 * 1)(len(scores))
        self.lib.r5_entry(args, constants)
        return values, indices

    def test_position_independent_code_and_host_results(self):
        self.assertGreater(len(self.code), 0)
        rng = np.random.default_rng(745)
        cases = [np.array([2.0, 2.0, 1.0, 2.0]),
                 np.array([-0.0, 0.0, -0.0, 0.0]),
                 np.array([3.0, 3.0, 3.0, 3.0]),
                 np.array([-4.0, -3.0, -2.0, -1.0]),
                 np.array([1.0, 0.0, 0.0, 0.0])]
        cases.extend(rng.standard_normal(n).astype(np.float32) for n in (4, 8, 16) for _ in range(20))
        for scores in cases:
            scores = np.ascontiguousarray(scores, dtype=np.float32)
            with self.subTest(scores=scores):
                values, indices = self.route(scores)
                expected_ids = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:2]
                np.testing.assert_array_equal(indices, expected_ids)
                np.testing.assert_array_equal(values.view(np.uint32), scores[expected_ids].view(np.uint32))


if __name__ == "__main__":
    unittest.main()

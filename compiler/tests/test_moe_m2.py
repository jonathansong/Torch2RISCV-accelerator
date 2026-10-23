#!/usr/bin/env python3
"""M2: device top-1 kernel writes the route consumed by the M1 GEMV."""

import ctypes
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "compiler", "frontend"))
sys.path.insert(0, os.path.join(REPO, "compiler", "runtime", "tools"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "driver"))
sys.path.insert(0, os.path.join(REPO, "llm"))

import moe  # noqa: E402
import moe_m1  # noqa: E402
import moe_top1_kernel  # noqa: E402
import sadesc  # noqa: E402
from pynq_matmul import DescList  # noqa: E402
from sa_funcsim import MODE_GEMV, SaFuncSim  # noqa: E402


class MoeM2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.code, host = moe_top1_kernel.build(cls.temp.name)
        cls.lib = ctypes.CDLL(host)
        cls.keep = {}
        ptr_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)
        cls.ptr_callback = ptr_type(lambda address: cls.keep[address].ctypes.data)
        cls.lib.moe_top1_host_init(cls.ptr_callback)
        cls.lib.r5_entry.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.config = moe.MoeConfig(32, 64, 4, 1)
        self.fp = moe.synthetic_weights(self.config, seed=29)
        self.q = moe.quantize_weights(self.config, self.fp)
        self.params, self.manifest = moe.pack_parameters(self.config, self.q)
        self.x = np.arange(-16, 16, dtype=np.int8)
        _, _, exports = sadesc.read(moe_m1.make_executable(32, 64, 16))
        self.rows = exports[0][1]

    def device_route(self, scores):
        record = np.zeros(moe.ROUTE_RECORD_SIZE, np.uint8)
        self.keep.clear()
        self.keep[0x1000] = np.ascontiguousarray(scores, np.float32)
        self.keep[0x2000] = record
        args = (ctypes.c_uint32 * 2)(0x1000, 0x2000)
        constants = (ctypes.c_uint32 * 2)(self.config.experts, self.manifest["expert_stride"])
        self.lib.r5_entry(args, constants)
        return record.tobytes()

    def run_selected_gemv(self, record):
        base = 0x10000000
        template, xaddr, routeaddr = base, base + 0x10000, base + 0x11000
        paramaddr, outaddr, listaddr = base + 0x12000, base + 0x20000, base + 0x21000
        sim = SaFuncSim(16, base, 0x40000)
        sim.trace = []
        sim.ddr_write(template, self.rows.tobytes())
        sim.ddr_write(xaddr, self.x)
        sim.ddr_write(routeaddr, record)
        sim.ddr_write(paramaddr, self.params)
        bindings = [xaddr, routeaddr, paramaddr + self.manifest["expert_table_offset"], outaddr]
        dispatch = sadesc.dispatch_list(template, bindings, [])
        sim.ddr_write(listaddr, dispatch.array().tobytes())
        sim.run_list(listaddr)
        return sim.ddr_read(outaddr, self.config.hidden * 4).view("<i4"), sim.trace, bindings[2]

    def test_kernel_is_position_independent_and_small(self):
        self.assertGreater(len(self.code), 0)
        self.assertLess(len(self.code), 2048)

    def test_device_top1_to_dynamic_gemv_for_every_expert(self):
        for expert in range(self.config.experts):
            with self.subTest(expert=expert):
                scores = np.full(self.config.experts, -4, np.float32)
                scores[expert] = 4
                record = self.device_route(scores)
                ids, offsets, weights = moe.unpack_route_record(record)
                np.testing.assert_array_equal(ids, [expert])
                np.testing.assert_array_equal(offsets, [expert * self.manifest["expert_stride"]])
                np.testing.assert_array_equal(weights, [1.0])
                got, trace, expert_base = self.run_selected_gemv(record)
                want = self.q.gate_q[expert].astype(np.int32) @ self.x.astype(np.int32)
                np.testing.assert_array_equal(got, want)
                gemv = [entry for entry in trace
                        if (entry[0] & 0xFF) == DescList.LD and ((entry[5] >> 32) & 3) == MODE_GEMV]
                self.assertEqual(len(gemv), 1)
                self.assertEqual(gemv[0][1], expert_base + int(offsets[0]))

    def test_tie_and_nan_match_host_routing(self):
        cases = [np.array([2, 2, 1, 2], np.float32),
                 np.array([np.nan, 3, 3, -1], np.float32)]
        for scores in cases:
            expected = moe.route(scores, 1)[0]
            got = moe.unpack_route_record(self.device_route(scores))[0]
            np.testing.assert_array_equal(got, expected)


if __name__ == "__main__":
    unittest.main()

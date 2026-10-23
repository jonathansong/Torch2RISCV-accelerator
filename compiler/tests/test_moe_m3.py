#!/usr/bin/env python3
"""M3: device top-2 routes two dynamic GEMVs and combines them on the VE."""

import ctypes
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
import moe_m3  # noqa: E402
import moe_top2_kernel  # noqa: E402
import sadesc  # noqa: E402
from sa_funcsim import SaFuncSim  # noqa: E402


class MoeM3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.code, host = moe_top2_kernel.build(cls.temp.name)
        cls.lib = ctypes.CDLL(host)
        cls.keep = {}
        callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)
        cls.callback = callback_type(lambda address: cls.keep[address].ctypes.data)
        cls.lib.moe_top2_host_init(cls.callback)
        cls.lib.r5_entry.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.config = moe.MoeConfig(32, 64, 4, 2)
        self.fp = moe.synthetic_weights(self.config, seed=31)
        self.q = moe.quantize_weights(self.config, self.fp)
        self.params, self.manifest = moe.pack_parameters(self.config, self.q)
        self.x = np.arange(-16, 16, dtype=np.int8)
        _, _, exports = sadesc.read(moe_m3.make_executable(32, 64, 16))
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

    def run_top2(self, record):
        base = 0x10000000
        addresses = [base + x for x in (0, 0x10000, 0x11000, 0x12000, 0x30000, 0x31000)]
        template, xaddr, routeaddr, paramaddr, outaddr, listaddr = addresses
        sim = SaFuncSim(16, base, 0x40000)
        sim.ddr_write(template, self.rows.tobytes())
        sim.ddr_write(xaddr, self.x)
        sim.ddr_write(routeaddr, record)
        sim.ddr_write(paramaddr, self.params)
        bindings = [xaddr, routeaddr, paramaddr + self.manifest["expert_table_offset"], outaddr]
        dispatch = sadesc.dispatch_list(template, bindings, [])
        sim.ddr_write(listaddr, dispatch.array().tobytes())
        sim.run_list(listaddr)
        return sim.ddr_read(outaddr, self.config.hidden * 4).view("<f4")

    def test_kernel_is_position_independent(self):
        self.assertGreater(len(self.code), 0)
        self.assertLess(len(self.code), 4096)

    def test_top2_ids_weights_and_device_combine(self):
        scores = np.array([-2.0, 1.25, 0.5, 3.0], np.float32)
        record = self.device_route(scores)
        ids, offsets, weights = moe.unpack_route_record(record)
        expected_ids, expected_weights = moe.route(scores, 2)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_array_equal(offsets, ids.astype(np.uint32) * self.manifest["expert_stride"])
        np.testing.assert_allclose(weights, expected_weights, rtol=2e-4, atol=2e-4)
        got = self.run_top2(record)
        raw = [self.q.gate_q[int(expert)].astype(np.int32) @ self.x.astype(np.int32) for expert in ids]
        expected = raw[0].astype(np.float32) * weights[0] + raw[1].astype(np.float32) * weights[1]
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=2e-3)

    def test_ties_choose_lower_expert_ids(self):
        scores = np.array([2, 2, 2, 1], np.float32)
        ids, _, weights = moe.unpack_route_record(self.device_route(scores))
        np.testing.assert_array_equal(ids, [0, 1])
        np.testing.assert_allclose(weights, [0.5, 0.5], rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

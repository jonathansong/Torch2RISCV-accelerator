#!/usr/bin/env python3
"""M3b: complete top-2 W8A8 gated FFN runs in one accelerator dispatch."""

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
import moe_m3b  # noqa: E402
import moe_top2_kernel  # noqa: E402
import sadesc  # noqa: E402
from sa_funcsim import SaFuncSim  # noqa: E402


class MoeM3bTest(unittest.TestCase):
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
        self.fp = moe.synthetic_weights(self.config, seed=47)
        self.q = moe.quantize_weights(self.config, self.fp)
        self.params, self.manifest = moe.pack_parameters(self.config, self.q)
        self.x = np.linspace(-0.75, 0.875, self.config.dim, dtype=np.float32)
        _, _, exports = sadesc.read(moe_m3b.make_executable(32, 64, 16))
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

    def run_ffn(self, record):
        base = 0x10000000
        template, xaddr, routeaddr, paramaddr, outaddr, listaddr = [
            base + x for x in (0, 0x10000, 0x11000, 0x12000, 0x30000, 0x31000)]
        sim = SaFuncSim(16, base, 0x40000)
        sim.ddr_write(template, self.rows.tobytes())
        sim.ddr_write(xaddr, self.x)
        sim.ddr_write(routeaddr, record)
        sim.ddr_write(paramaddr, self.params)
        bindings = [xaddr, routeaddr, paramaddr + self.manifest["expert_table_offset"], outaddr]
        dispatch = sadesc.dispatch_list(template, bindings, [])
        sim.ddr_write(listaddr, dispatch.array().tobytes())
        sim.run_list(listaddr)
        return sim.ddr_read(outaddr, self.config.dim * 4).view("<f4")

    def test_layout_matches_parameter_manifest(self):
        layout = moe_m3b.expert_layout(self.config.dim, self.config.hidden, self.config.alignment)
        record = self.manifest["expert_records"][0]
        start = record["offset"]
        actual = {section["name"].rsplit(".", 2)[1] + "_" + section["name"].rsplit(".", 1)[1][0]:
                  section["offset"] - start for section in record["sections"]}
        self.assertEqual(layout["gate_w"], actual["gate_w"])
        self.assertEqual(layout["gate_s"], actual["gate_s"])
        self.assertEqual(layout["up_w"], actual["up_w"])
        self.assertEqual(layout["up_s"], actual["up_s"])
        self.assertEqual(layout["down_w"], actual["down_w"])
        self.assertEqual(layout["down_s"], actual["down_s"])
        self.assertEqual(layout["stride"], self.manifest["expert_stride"])

    def test_complete_top2_w8a8_ffn(self):
        scores = np.array([1.25, -0.5, 2.75, 0.25], np.float32)
        record = self.device_route(scores)
        ids, _, weights = moe.unpack_route_record(record)
        got = self.run_ffn(record)
        xq, xs = moe.quantize_activation(self.x)
        expected = sum((weight * moe.w8a8_expert(xq, xs, self.q, int(expert))).astype(np.float32)
                       for expert, weight in zip(ids, weights))
        np.testing.assert_allclose(got, expected, rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    unittest.main()

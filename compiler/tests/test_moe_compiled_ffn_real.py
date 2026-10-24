#!/usr/bin/env python3
"""Execute the model-derived real Mixtral FFN descriptor in SaFuncSim."""

from pathlib import Path
import json
import sys
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for directory in ("compiler/frontend", "compiler/runtime/tools", "llm"):
    sys.path.insert(0, str(REPO / directory))
import moe  # noqa: E402
import sadesc  # noqa: E402
from sa_funcsim import SaFuncSim  # noqa: E402

BUNDLE = REPO / "build/moe/m5m/board"


class CompiledRealFfnTest(unittest.TestCase):
    def test_real_layer_shared_descriptor(self):
        if not (BUNDLE / "moe_ffn.sadesc").is_file():
            self.skipTest("stage compiler/tests/moe_m5m_board_bundle.py first")
        manifest = json.loads((BUNDLE / "moe_manifest.json").read_text())
        dim, stride = manifest["dim"], manifest["expert_stride"]
        params = (BUNDLE / "moe_params.bin").read_bytes()
        d, _, exports = sadesc.read((BUNDLE / "moe_ffn.sadesc").read_bytes())
        self.assertEqual(d, 16)
        self.assertEqual(len(exports), 1)
        self.assertEqual(exports[0][2], 4)
        base = 0x10000000
        template = base
        xaddr = base + 0x10000
        routeaddr = base + 0x11000
        paramaddr = base + 0x20000
        outaddr = (paramaddr + len(params) + 0xFFFF) & ~0xFFFF
        listaddr = outaddr + 0x10000
        sim = SaFuncSim(16, base, 128 * 1024 * 1024)
        sim.ddr_write(template, exports[0][1].tobytes())
        sim.ddr_write(paramaddr, params)
        for filename in ("moe_acceptance.npz", "moe_token_acceptance.npz"):
            reference = np.load(BUNDLE / filename)
            vectors = reference["x"]
            if vectors.ndim == 1:
                vectors = np.broadcast_to(vectors, (len(reference["pairs"]), dim))
            for i, pair in enumerate(reference["pairs"]):
                with self.subTest(source=filename, case=i):
                    ids = np.asarray(pair, np.uint32)
                    routing = np.asarray(reference["routing"][i], np.float32)
                    route = moe.pack_route_record(ids, ids * np.uint32(stride), routing)
                    sim.ddr_write(xaddr, np.ascontiguousarray(vectors[i], np.float32))
                    sim.ddr_write(routeaddr, route)
                    dispatch = sadesc.dispatch_list(template, [xaddr, routeaddr,
                        paramaddr + manifest["expert_table_offset"], outaddr], [])
                    sim.ddr_write(listaddr, dispatch.array().tobytes())
                    sim.run_list(listaddr)
                    got = sim.ddr_read(outaddr, dim * 4).view("<f4")
                    np.testing.assert_allclose(got, reference["output"][i],
                                               rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    unittest.main()

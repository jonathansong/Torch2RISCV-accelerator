#!/usr/bin/env python3
"""Runs a packed real-size Hugging Face MoE layer in the SA functional simulator."""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for path in (os.path.join(REPO, "compiler", "frontend"), os.path.join(REPO, "compiler", "runtime", "tools"),
             HERE, os.path.join(REPO, "driver"), os.path.join(REPO, "llm")):
    sys.path.insert(0, path)

import moe  # noqa: E402
import moe_m3b  # noqa: E402
import sadesc  # noqa: E402
from sa_funcsim import SaFuncSim  # noqa: E402


def align(value, alignment=4096):
    return (value + alignment - 1) // alignment * alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", help="directory containing params, manifest, and acceptance vectors")
    parser.add_argument("--cases", type=int, default=1)
    args = parser.parse_args()
    with open(os.path.join(args.bundle, "moe_manifest.json")) as f:
        manifest = json.load(f)
    params = np.fromfile(os.path.join(args.bundle, "moe_params.bin"), np.uint8)
    reference = np.load(os.path.join(args.bundle, "moe_acceptance.npz"))
    d, dim, hidden = (int(manifest[name]) for name in ("d", "dim", "hidden"))
    _, _, exports = sadesc.read(moe_m3b.make_executable(dim, hidden, d, int(manifest["alignment"])))
    rows = exports[0][1]
    base, template = 0x10000000, 0x10000000
    xaddr, routeaddr, paramaddr = base + 0x10000, base + 0x20000, base + 0x100000
    outaddr = align(paramaddr + params.size)
    listaddr = align(outaddr + dim * 4)
    size = align(listaddr + (len(rows) + 8) * 64 - base)
    sim = SaFuncSim(d, base, size)
    sim.ddr_write(template, rows.tobytes())
    sim.ddr_write(xaddr, reference["x"])
    sim.ddr_write(paramaddr, params)
    bindings = [xaddr, routeaddr, paramaddr + int(manifest["expert_table_offset"]), outaddr]
    dispatch = sadesc.dispatch_list(template, bindings, [])
    sim.ddr_write(listaddr, dispatch.array().tobytes())
    failures = 0
    for case in range(min(args.cases, len(reference["pairs"]))):
        pair = reference["pairs"][case].astype(np.uint32)
        offsets = pair * np.uint32(manifest["expert_stride"])
        route = moe.pack_route_record(pair, offsets, reference["routing"][case])
        sim.ddr_write(routeaddr, route)
        sim.run_list(listaddr)
        got = sim.ddr_read(outaddr, dim * 4).view("<f4")
        wanted = reference["output"][case]
        error = float(np.max(np.abs(got - wanted)))
        ok = np.allclose(got, wanted, rtol=3e-3, atol=3e-3)
        failures += not ok
        print(f"experts {int(pair[0])},{int(pair[1])}: {'PASS' if ok else 'FAIL'}, max error {error:.6g}")
    print("real-size MoE simulator PASS" if not failures else f"real-size MoE simulator FAIL: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

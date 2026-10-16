#!/usr/bin/env python3
"""Code size of plan A's generated RISC-V code against an instruction memory
(docs/riscv_backend_design.md §8): which TCM or I-cache would hold a step.
The exports' rv32 bytes come from an sa.vmfb compiled with --iree-sa-riscv
(S4, the sa-desc code table); which exports one step calls, and how often,
from a profile of the step (SA_PROFILE + SA_PROFILE_PERF=<csv>, the sim is
enough: the calls do not depend on the device).

Per section (decode step, prefill chunk): the step's working set (the code of
every distinct export it calls), the part called once per layer (what an
I-cache must hold to reuse code from one layer to the next), the largest
function, and the code that covers 50 / 90 / 99% of the calls.

    compiler/tests/rv_codesize.py <sa.vmfb with code> <profile.csv> [--layers 28] [--json out.json]
"""
import argparse
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "runtime", "tools"))
import sadesc  # noqa: E402


def code_sizes(path):
    """{export name: rv32 bytes} from every sa-desc executable's code table"""
    d, out, i = open(path, "rb").read(), {}, 0
    while (i := d.find(b"SADESC1\0", i)) >= 0:
        h = struct.unpack_from("<IIIIIIIIIII", d, i + 8)
        n, tab = h[3], h[10]
        names = [e[0] for e in sadesc.read(d[i:])[2]]
        for k in range(n):
            out[names[k]] = struct.unpack_from("<II", d, i + tab + 8 * k)[1] if tab else 0
        i += 8
    return out


def sections(path):
    """[(label, per_step, {export: calls per step})] from an SA_PROFILE_PERF csv"""
    out = []
    for line in open(path):
        line = line.strip()
        if line.startswith("# section"):
            parts = line.split()
            out.append((parts[3], float(parts[5]), {}))
        elif line and not line.startswith("export,") and out:
            f = line.split(",")
            if not f[0].startswith("host:"):
                out[-1][2][f[0]] = int(f[1]) / out[-1][1]
    return out


def kb(x):
    return f"{x / 1024:.1f} KB"


def analyse(sizes, calls, layers):
    missing = [e for e in calls if e not in sizes or not sizes[e]]
    used = {e: sizes[e] for e in calls if sizes.get(e)}
    total_calls = sum(calls[e] for e in used)
    ws = sum(used.values())
    per_layer = {e: s for e, s in used.items() if layers and calls[e] >= layers - 0.01}
    by_calls = sorted(used, key=lambda e: -calls[e] / max(used[e], 1))  # most calls per byte first
    cover = {}
    for q in (0.5, 0.9, 0.99):
        acc = b = 0
        for e in by_calls:
            acc += calls[e]
            b += used[e]
            if acc >= q * total_calls:
                break
        cover[q] = b
    big = sorted(used, key=lambda e: -used[e])[:5]
    sz = sorted(used.values())
    return {"exports": len(used), "no_code": len(missing), "calls": total_calls, "working_set": ws,
            "per_layer_exports": len(per_layer), "per_layer_set": sum(per_layer.values()),
            "per_layer_calls": sum(calls[e] for e in per_layer), "median": sz[len(sz) // 2] if sz else 0,
            "largest": [(e, used[e], calls[e]) for e in big], "cover": cover}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("vmfb")
    ap.add_argument("profile")
    ap.add_argument("--layers", type=int, default=0, help="the model's layers (the per-layer part)")
    ap.add_argument("--json")
    args = ap.parse_args()
    sizes = code_sizes(args.vmfb)
    allcode = sum(sizes.values())
    print(f"{args.vmfb}: {len(sizes)} exports, {sum(1 for s in sizes.values() if s)} with code, {kb(allcode)} of rv32 in all")
    res = {"exports": len(sizes), "all_code": allcode, "sections": {}}
    for label, per_step, calls in sections(args.profile):
        a = analyse(sizes, calls, args.layers)
        res["sections"][label] = a
        print(f"  {label}: {a['calls']:.0f} calls per step of {a['exports']} exports ({a['no_code']} without code); "
              f"working set {kb(a['working_set'])}; per layer ({args.layers}): {a['per_layer_exports']} exports, "
              f"{kb(a['per_layer_set'])}, {a['per_layer_calls'] / max(a['calls'], 1):.0%} of the calls; "
              f"median function {a['median']} B")
        print("    calls covered by the code: " +
              ", ".join(f"{int(q * 100)}% {kb(b)}" for q, b in a["cover"].items()))
        for e, s, c in a["largest"]:
            print(f"    {kb(s):>10}  {c:6.1f} calls  {e}")
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()

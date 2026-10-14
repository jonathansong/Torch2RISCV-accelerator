#!/usr/bin/env python3
"""Cycle model of decode attention by context length (docs/iree_compiler_plan.md
§8.22, the baseline of the long-T plan a-e): the decode dispatches over the
dynamic length T (export names with a D: q . K^T, the masked max / exp, the
sum, p . V), each run on the functional simulator at the lengths --t through
dispatch_check.py (so each result is also checked against the oracle) with the
executed commands captured, and timed by perf_model.run_list (four-port GEMV
overlay, the decode calibration build/perfmodel/qwen3_dyn_cal.json).

A dispatch kind appears once per layer (layer 2's are the same code): the first
of each kind is measured; the total per layer is their sum, per token x --layers.

--max-dynamic N1,N2: the same dispatches also recompiled for those
max_dynamic (golden.compile_one on the source with its attribute replaced),
so one table shows the static block size's cost at short T (a model compiled
for 2048 against one compiled for 256 at the same T).

    compiler/tests/attn_cycles.py build/hfgen/qwen3_l2_t2048lc [--t 16,128,256,512,1024,2048]
        [--max-dynamic 256] [--layers 28] [--json out.json]
"""
import argparse
import collections
import json
import os
import re
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "llm"))
import dispatch_check as DC  # noqa: E402
import golden as G  # noqa: E402
import perf_model as PM  # noqa: E402

ENG = {PM.OP_LD: "LD", PM.OP_ST: "ST", PM.OP_EX: "EX", PM.OP_VE: "VE"}


def arch(cal):
    a = PM.Arch(gemv_ports=4)
    c = json.load(open(cal))
    a.ve_a, a.ve_b, a.f_desc, a.list_ovh = tuple(c["ve_a"]), tuple(c["ve_b"]), c["f_desc"], c["list_ovh"]
    return a


def dynamic_dispatches(build):
    """{kind: (number, export name)}: the decode dispatches over T, the first of each kind"""
    out = {}
    src = os.path.join(build, "sa_sources")
    for f in sorted(os.listdir(src), key=lambda f: int(re.search(r"(\d+)\.mlir$", f).group(1)) if re.search(r"\d+\.mlir$", f) else 0):
        m = re.match(r"module_main\$async_dispatch_(\d+)\.mlir$", f)
        if not m:
            continue
        e = re.search(r"hal\.executable\.export public @([\w$]+)", open(os.path.join(src, f)).read())
        if not e:
            continue
        kind = re.sub(r"^main\$async_dispatch_\d+_", "", e.group(1))
        if re.search(r"(^|x|_)D(x|_|$)", kind) and kind not in out:
            out[kind] = (int(m.group(1)), e.group(1))
    return out


def measure(a, src, blob, t, d):
    """(check result, cycles, busy per engine, commands per engine) of one dispatch at T"""
    cap = []
    orig = DC.SaFuncSim.run_list

    def run_list(self, addr):
        self.trace = []
        try:
            return orig(self, addr)
        finally:
            cap.append(self.trace)
            self.trace = None
    DC.SaFuncSim.run_list = run_list
    try:
        res = DC.check(src, blob, d, t)
    finally:
        DC.SaFuncSim.run_list = orig
    cmds = [PM.decode(a, np.asarray(r, np.uint64)) for tr in cap for r in tr]
    cyc, busy = PM.run_list(a, cmds, gemv=True)
    return res, cyc, dict(zip(("LD", "ST", "EX", "VE"), busy)), collections.Counter(ENG.get(c.op, "FENCE") for c in cmds)


def ok(res):
    return res is True or (isinstance(res, str) and "OK" in res)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("build", help="a test_long_context.py build (sa_sources, sa_bins, compile_flags.txt)")
    ap.add_argument("--t", default="16,128,256,512,1024,2048")
    ap.add_argument("--max-dynamic", default="", help="also recompiled for these max_dynamic (comma-separated)")
    ap.add_argument("--layers", type=int, default=28, help="per token: the model's layers (Qwen3-0.6B 28)")
    ap.add_argument("--cal", default=os.path.join(REPO, "build/perfmodel/qwen3_dyn_cal.json"))
    ap.add_argument("--json")
    args = ap.parse_args()
    a = arch(args.cal)
    flags = open(os.path.join(args.build, "compile_flags.txt")).read()
    own = int(re.search(r"--iree-sa-max-dynamic=(\d+)", flags).group(1))
    d = int(os.environ.get("SA_D", "16"))
    kinds = dynamic_dispatches(args.build)
    if not kinds:
        print(f"no decode dispatch over a dynamic length in {args.build}/sa_sources")
        return 1
    Ts = [int(t) for t in args.t.split(",")]
    variants = [own] + [int(m) for m in args.max_dynamic.split(",") if m and int(m) != own]
    tmp = tempfile.mkdtemp(prefix="attn_cycles_")
    rows, fails = [], 0
    print(f"{args.build}: compiled for {own}; dispatches over T: " + ", ".join(f"#{n} {k}" for k, (n, _) in kinds.items()))
    for md in variants:
        for kind, (n, name) in kinds.items():
            src = os.path.join(args.build, "sa_sources", f"module_main$async_dispatch_{n}.mlir")
            if md == own:
                blob = open(os.path.join(args.build, "sa_bins", f"module_main$async_dispatch_{n}_sa_desc_v1.sadesc"), "rb").read()
            else:
                text = re.sub(r"max_dynamic = \d+", f"max_dynamic = {md}", G.sa_only(open(src).read()))
                src = os.path.join(tmp, f"d{n}_{md}.mlir")
                open(src, "w").write(text)
                blob, err = G.compile_one(src, [f"--iree-sa-max-dynamic={md}"], os.path.join(tmp, f"d{n}_{md}.sadesc"))
                if blob is None:
                    print(f"  #{n} at max_dynamic {md}: compile failed: {err}")
                    fails += 1
                    continue
            for t in Ts:
                if t > md:
                    continue
                res, cyc, busy, cnt = measure(a, src, blob, t, d)
                fails += not ok(res)
                rows.append({"max_dynamic": md, "T": t, "dispatch": n, "kind": kind, "check": "OK" if ok(res) else str(res),
                             "cycles": cyc, "busy": busy, "commands": dict(cnt)})
    # the table: per variant and T, each kind's cycles (busy LD / EX / VE), the layer's total, per token
    for md in variants:
        print(f"\nmax_dynamic {md}  (cycles at {a.mhz:.0f} MHz; busy LD / EX / VE in thousands)")
        names = [k for k in kinds if any(r["kind"] == k and r["max_dynamic"] == md for r in rows)]
        print(f"{'T':>6}  " + "  ".join(f"{k[:30]:>30}" for k in names) + f"  {'layer':>9}  {'ms/token':>9}")
        for t in Ts:
            rs = {r["kind"]: r for r in rows if r["max_dynamic"] == md and r["T"] == t}
            if not rs:
                continue
            cell = lambda r: (f"{r['cycles'] / 1e3:7.1f}k ({r['busy']['LD'] / 1e3:.0f}/{r['busy']['EX'] / 1e3:.0f}/"  # noqa: E731
                              f"{r['busy']['VE'] / 1e3:.0f})" + ("" if r["check"] == "OK" else " BAD"))
            layer = sum(r["cycles"] for r in rs.values())
            print(f"{t:6d}  " + "  ".join(f"{cell(rs[k]) if k in rs else '-':>30}" for k in names)
                  + f"  {layer / 1e3:8.1f}k  {layer * args.layers / (a.mhz * 1e3):9.2f}")
    if args.json:
        json.dump({"build": args.build, "own_max_dynamic": own, "layers": args.layers, "rows": rows}, open(args.json, "w"), indent=1)
    print(f"\n{len(rows)} measurements, {fails} not OK against the oracle" if fails else f"\n{len(rows)} measurements, all OK against the oracle")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""The KV260 decode step's fixed part (docs/generality_implementation_plan.md
§13.4): from a board_profile.py run with SA_PROFILE_PERF (one list per
dispatch, the unit's event counters per export, the decode section), per
dispatch kind the device cycles against the LD floor (bytes at the four-port
GEMV peak, 57.95 B/cycle), split by the counters (z1_profile.py's PC_*):
all-idle (descriptor fetch, fences: the per-list fixed overhead), VE, EX, LD.

    compiler/tests/kv260_fixed.py build/deploy_hfgen_qwen3_qs [more bundles] [--top 25]
"""
import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z1_profile as Z  # noqa: E402

BW = 57.95                     # B/cycle: the four-port GEMV peak (bwtest)
MHZ = 100.0


def group(kind):
    if kind.startswith("matvec_like"):
        return "GEMV"
    if kind.startswith(("batch_matmul", "matmul_like")):
        return "attention matmul"
    if kind.startswith("reduction"):
        return "reduction"
    if kind.startswith("elementwise"):
        return "elementwise"
    if "scatter" in kind:
        return "scatter"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundles", nargs="+")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    for b in args.bundles:
        csv = next((os.path.join(b, f) for f in ("prof3.csv", "prof.csv") if os.path.exists(os.path.join(b, f))), None)
        if not csv:
            print(f"{b}: no profile csv")
            continue
        ps, rows = Z.sections(csv)[-1]                       # the decode section
        agg = collections.defaultdict(lambda: collections.Counter())
        for r in rows:
            k = Z.kind(r["export"])
            a = agg[k]
            a["calls"] += r["calls"] / ps
            a["cyc"] += r["cycles"] / ps
            a["idle"] += r["c13"] / ps
            a["ve"] += r["c24"] / ps
            a["ex"] += r["c14"] / ps
            a["ld"] += r["c18"] / ps
            a["bytes"] += 8 * r["c19"] / ps
            a["st"] += r["c21"] / ps
        tot = sum(a["cyc"] for a in agg.values())
        floor = sum(a["bytes"] for a in agg.values()) / BW
        calls = sum(a["calls"] for a in agg.values())
        idle = sum(a["idle"] for a in agg.values())
        print(f"\n== {b}: decode {tot / MHZ / 1e3:.1f} ms/step device, {calls:.0f} dispatches; LD floor {floor / MHZ / 1e3:.1f} ms; "
              f"above it {(tot - floor) / MHZ / 1e3:.1f} ms; all-idle {idle / MHZ / 1e3:.1f} ms ({idle / calls:.0f} cycles per dispatch)")
        g = collections.defaultdict(collections.Counter)
        for k, a in agg.items():
            g[group(k)].update(a)
        print(f"  {'group':18s} {'calls':>6s} {'ms':>7s} {'floor':>7s} {'excess':>7s} {'idle':>6s} {'VE':>6s} {'EX':>6s} {'cyc/call':>9s}")
        for name, a in sorted(g.items(), key=lambda x: -x[1]["cyc"]):
            f = a["bytes"] / BW
            print(f"  {name:18s} {a['calls']:6.0f} {a['cyc'] / 1e5:7.2f} {f / 1e5:7.2f} {(a['cyc'] - f) / 1e5:7.2f} "
                  f"{a['idle'] / 1e5:6.2f} {a['ve'] / 1e5:6.2f} {a['ex'] / 1e5:6.2f} {a['cyc'] / max(a['calls'], 1):9.0f}")
        print(f"  top kinds by cycles above the floor (ms per step):")
        ex = sorted(agg.items(), key=lambda x: -(x[1]["cyc"] - x[1]["bytes"] / BW))
        for k, a in ex[:args.top]:
            f = a["bytes"] / BW
            print(f"    {(a['cyc'] - f) / 1e5:6.2f}  calls {a['calls']:5.0f}  {a['cyc'] / max(a['calls'], 1):8.0f} cyc/call  "
                  f"idle {a['idle'] / max(a['calls'], 1):6.0f}  VE {a['ve'] / max(a['calls'], 1):7.0f}  {k[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

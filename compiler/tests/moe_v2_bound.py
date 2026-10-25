#!/usr/bin/env python3
"""Upper bound of v2 (docs/moe_native_lowering_design.md §9, §20): both experts'
FFN and down projections of a decode MoE layer in one dispatch, from the
command-level performance model (perf_model.py) on a sim trace
(compiler/scripts/perf_trace.sh with SA_RV / SA_RV_ONLY=rv_only for the Top-2).

Prints the modeled decode step per dispatch (against the board's measured step),
each MoE dispatch's GEMV bytes against the 4-port stream bound, and the MoE
dispatches of a layer separate / joined into one list / joined without any FENCE
(the last not a legal schedule: an upper bound).

    python3 compiler/tests/moe_v2_bound.py --trace build/perfmodel/moe_tm.trace --cal build/perfmodel/qwen3g_cal.json
"""
import argparse
import collections
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perf_model as PM  # noqa: E402

OP_FENCE = 0x10


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", required=True)
    ap.add_argument("--cal", required=True, help="perf_model.py --save-cal of the same overlay")
    ap.add_argument("--gemv-ports", type=int, default=4)
    ap.add_argument("--moe", default=r"matvec_like_(224x16x1024|64x16x3584)_", help="the experts' dispatch kinds")
    ap.add_argument("--prefill-steps", type=int, default=2, help="trace steps before decode")
    args = ap.parse_args()
    arch = PM.Arch(gemv_ports=args.gemv_ports)
    cal = json.load(open(args.cal))
    arch.ve_a, arch.ve_b, arch.f_desc, arch.list_ovh = tuple(cal["ve_a"]), tuple(cal["ve_b"]), cal["f_desc"], cal["list_ovh"]
    segs = PM.segments(PM.decode_trace(args.trace, arch))
    dec = [x for s in segs[args.prefill_steps:] for x in s if not x[0].startswith("prefill")]
    nsteps = len(segs) - args.prefill_steps
    run = lambda cmds: PM.run_list(arch, cmds)[0]   # noqa: E731
    per, calls = collections.defaultdict(float), collections.Counter()
    for name, cmds in dec:
        k = name.split("dispatch_", 1)[1]
        per[k] += run(cmds) / nsteps
        calls[k] += 1
    tot = sum(per.values())
    print(f"modeled decode step: {tot:.0f} cycles ({tot / arch.mhz / 1e3:.2f} ms at {arch.mhz:.0f} MHz)")
    bound = 16.0 * args.gemv_ports
    for k, v in sorted(per.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {k[:44]:44s} {v:9.0f} {100 * v / tot:5.1f}%  ({calls[k] // nsteps} per step)")
    moe = re.compile(args.moe)
    print(f"MoE dispatches, one call, against the GEMV stream ({bound:.0f} B / cycle):")
    seen = set()
    for name, cmds in dec:
        k = name.split("dispatch_", 1)[1]
        if not moe.search(k) or k in seen:
            continue
        seen.add(k)
        gb = sum(c.beats * 16 for c in cmds if c.op == PM.OP_LD and c.gemv)
        cy = run(cmds)
        print(f"  {k[:44]:44s} {cy:8.0f} cycles, {gb / 1e6:.2f} MB of weights, bound {gb / bound:.0f} ({100 * gb / bound / cy:.0f}%)")
    groups, cur = [], []
    for name, cmds in dec:
        if moe.search(name):
            cur.append(cmds)
        elif cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    sep = sum(run(c) for g in groups for c in g) / nsteps
    joined = sum(run([x for c in g for x in c]) for g in groups) / nsteps
    nofence = sum(run([x for c in g for x in c if x.op != OP_FENCE]) for g in groups) / nsteps
    print(f"a layer's {len(groups[0])} MoE dispatches, per step: separate {sep:.0f}, one list {joined:.0f} "
          f"(saves {sep - joined:.0f}), one list without any FENCE {nofence:.0f} (saves {sep - nofence:.0f} = "
          f"{100 * (sep - nofence) / tot:.2f}% of the step)")


if __name__ == "__main__":
    main()

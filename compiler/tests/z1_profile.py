#!/usr/bin/env python3
"""The Z1 time breakdown (docs/kv260_upgrade_plan.md §7.2 z1_profile.md, K2
"读权重以外的部分"): per path and dispatch type, the decode step's device
cycles split by the unit's event counters (rtl/sysarray/sa_defs.vh PC_*),
from the board_profile.py runs of board_regress.py (results/*_perf.csv,
SA_PROFILE_PERF: one list per dispatch, counters over each list).

    $SA_PY compiler/tests/z1_profile.py [results dir] [-o out.md]

Per dispatch type (the export name without its number):
  cycles   rt_fw's window per list (rdcycle around submit + fence)
  LD       LD busy cycles (PC_LD_BUSY); bytes = 8 x PC_LD_BEATS; floor = bytes / 7.94
           (bwtest's measured B/cycle: the weight-read lower bound)
  EX       array enabled (PC_EX_STEP); useful = PC_EX_USEFUL
  VE       VE active (PC_VE_ACTIVE; the SFU is inside the VE)
  ST       ST busy (PC_ST_BUSY)
  idle     all engines idle and no command in the pipeline (PC_ALL_IDLE): the
           descriptor fetch and fences, i.e. fixed overhead; plus rt_fw's own
           (cycles - PC_CYCLES)
  overlap  LD + EX + VE + ST - (PC_CYCLES - idle): engine cycles that ran
           concurrently with another engine (the busy columns do not add up)
"""
import argparse
import collections
import os
import re
import sys

PC = dict(CYCLES=0, HAZ_LD=7, HAZ_ST=8, HAZ_EX=9, HAZ_VE=10, STARVE=12, ALL_IDLE=13, EX_STEP=14, EX_USEFUL=15,
          LD_BUSY=18, LD_BEATS=19, ST_BUSY=21, VE_ACTIVE=24)
BW = 7.94                                     # bwtest: LD DDR -> SPAD, B/cycle (Z1, 50 MHz)
PATHS = [("c6p_stories", "stories15M, hand-written path (export.py)"),
         ("c6p_smollm2", "SmolLM2-135M, hand-written qhf path (export_hf.py)"),
         ("hfgen", "SmolLM2-135M, generic path (unmodified HF)")]


def sections(path):
    """-> [(per_step, [row dict])] per report (prefill chunks, then decode)"""
    out, cur, hdr = [], None, None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("# section"):
            cur = (float(line.split()[-1]), [])
            out.append(cur)
            hdr = None
        elif hdr is None:
            hdr = line.split(",")
        else:
            v = line.split(",")
            name = ",".join(v[:len(v) - len(hdr) + 1])            # (names have no commas; be safe)
            row = dict(zip(hdr[1:], map(int, v[len(v) - len(hdr) + 1:])))
            row["export"] = name
            cur[1].append(row)
    return out


def kind(name):
    t = re.sub(r"^\w+\$async_dispatch_\d+_?", "", name)
    return re.sub(r"\d+", "N", t) or name


def table(rows, per_step, top=None):
    g = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        k = kind(r["export"])
        c = g[k]
        c["n"] += r["calls"]
        c["cycles"] += r["cycles"]
        for nm, i in PC.items():
            c[nm] += r.get(f"c{i}", 0)
    tot = collections.Counter()
    for c in g.values():
        tot.update(c)
    hdr = ("| dispatch type | per step | cycles | % | LD busy | LD bytes | floor | EX | EX useful | VE | ST | idle | "
           "overlap |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [hdr]

    def row(name, c):
        cyc = c["cycles"]
        idle = c["ALL_IDLE"] + max(0, c["cycles"] - c["CYCLES"])
        busy = c["LD_BUSY"] + c["EX_STEP"] + c["VE_ACTIVE"] + c["ST_BUSY"]
        ovl = busy - (c["CYCLES"] - c["ALL_IDLE"])
        b = 8 * c["LD_BEATS"]
        f = lambda x: f"{x / per_step / 1e3:,.0f}k"          # noqa: E731
        return (f"| {name} | {c['n'] / per_step:.0f} | {f(cyc)} | {100 * cyc / max(1, tot['cycles']):.1f} | "
                f"{f(c['LD_BUSY'])} | {b / per_step / 1e6:.2f} MB | {f(b / BW)} | {f(c['EX_STEP'])} | "
                f"{f(c['EX_USEFUL'])} | {f(c['VE_ACTIVE'])} | {f(c['ST_BUSY'])} | {f(idle)} | {f(ovl)} |")
    order = sorted(g.items(), key=lambda kv: -kv[1]["cycles"])
    for k, c in order[:top] if top else order:
        lines.append(row(k, c))
    if top and len(order) > top:
        rest = collections.Counter()
        for _, c in order[top:]:
            rest.update(c)
        lines.append(row(f"({len(order) - top} other types)", rest))
    lines.append(row("**total**", tot))
    return lines, tot


def summary(tot, per_step, unit="step"):
    cyc = tot["cycles"] / per_step
    idle = (tot["ALL_IDLE"] + max(0, tot["cycles"] - tot["CYCLES"])) / per_step
    floor = 8 * tot["LD_BEATS"] / BW / per_step
    busy = {k: tot[k] / per_step for k in ("LD_BUSY", "EX_STEP", "VE_ACTIVE", "ST_BUSY")}
    active = (tot["CYCLES"] - tot["ALL_IDLE"]) / per_step
    ovl = sum(busy.values()) - active
    haz = sum(tot[k] for k in ("HAZ_LD", "HAZ_ST", "HAZ_EX", "HAZ_VE")) / per_step
    p = lambda x: f"{x / 1e6:.2f}M ({100 * x / cyc:.0f}%)"   # noqa: E731
    return [f"- device cycles per {unit}: **{cyc / 1e6:.2f}M** ({cyc / 50e3:.0f} ms at 50 MHz)",
            f"- bytes loaded: {8 * tot['LD_BEATS'] / per_step / 1e6:.2f} MB; at bwtest's {BW} B/cycle that is "
            f"{p(floor)} — the load floor; LD busy {p(busy['LD_BUSY'])}",
            f"- EX {p(busy['EX_STEP'])} (useful {p(tot['EX_USEFUL'] / per_step)}), VE / SFU {p(busy['VE_ACTIVE'])}, "
            f"ST {p(busy['ST_BUSY'])}",
            f"- fixed overhead (all engines idle, plus rt_fw outside the counter window): {p(idle)}",
            f"- overlap: the busy columns add up to {sum(busy.values()) / 1e6:.2f}M against {active / 1e6:.2f}M "
            f"non-idle cycles, so {ovl / 1e6:.2f}M engine cycles ran alongside another engine",
            f"- not hidden behind the load floor: {p(cyc - floor)}; scoreboard waits (HAZ_*) {haz / 1e6:.2f}M"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=os.path.join(os.path.dirname(__file__), "..", "..", "build",
                                                               "deploy_z1", "results"))
    ap.add_argument("-o", "--out")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    out = ["# PYNQ-Z1 time breakdown (decode)", "",
           "Per decode step on the board (L2 bitstream, D = 8, 50 MHz), from the unit's event counters over each",
           "dispatch's list (`board_profile.py` with `SA_PROFILE_PERF`; profiling runs one list per dispatch, so",
           "dispatches do not overlap each other here). Columns: see `compiler/tests/z1_profile.py`. The busy",
           "columns overlap and do not add up; *floor* is the bytes loaded at bwtest's 7.94 B/cycle (the",
           "load-bound minimum), *idle* the cycles with every engine idle (fixed overhead).", ""]
    found = False
    for t, title in PATHS:
        p = os.path.join(a.results, t + "_perf.csv")
        if not os.path.exists(p):
            print(f"{p}: missing", file=sys.stderr)
            continue
        secs = sections(p)
        per_step, rows = secs[-1]                    # the last report: decode after the prefill
        lines, tot = table(rows, per_step, a.top)
        out += [f"## {title}", ""] + summary(tot, per_step) + [""] + lines + [""]
        if len(secs) > 1:
            ps, pr = secs[0]
            lines, ptot = table(pr, ps, 6)
            out += ["Prefill, per chunk of 8 tokens (after the first chunk):", ""] + summary(ptot, ps, "chunk") + [""] + lines + [""]
        found = True
    if not found:
        sys.exit("no *_perf.csv in " + a.results)
    text = "\n".join(out)
    if a.out:
        open(a.out, "w").write(text)
        print(f"-> {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

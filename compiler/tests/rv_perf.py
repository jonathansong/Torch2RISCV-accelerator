#!/usr/bin/env python3
"""Where plan A's time goes (docs/riscv_backend_design.md §9.10): the
performance counters of compiler/tests/board_rvperf.py's runs, the generated
RISC-V code (S4) against the descriptor lists.

The unit's cycles in a window are split by its scheduler's events
(rtl/sysarray/sa_perf.v, docs/perf_counters_and_desc_dma_plan.md §1.2):
ALL_IDLE (nothing queued, no engine busy: waiting for the issuer), STARVE (an
engine busy, no command at the head: issue partly hidden), HAZ (the head waits
on a dependency), DISP_FULL (the engine's queue full: the issuer is ahead) and
the rest (commands moving). The core's side: instructions retired (rt_fw 7),
instructions per command, cycles per instruction (every fetch from DDR), and
the cycles it spent stalled on PCPI (PCPI_QFULL: queue full; PCPI_FENCE).

    compiler/tests/rv_perf.py <dir with rvperf_desc.csv, rvperf_rv.csv, rvperf_sched.csv, rvperf_sched_rv.csv>
        [--section decode] [--top 12] [--mhz 100] [--json out.json]
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

# firmware/include/sysarray_intrinsics.h SA_PC_*
PC = {"CYCLES": 0, "CMD_LD": 1, "CMD_ST": 2, "CMD_EX": 3, "CMD_VE": 4, "PCPI_QFULL": 5, "PCPI_FENCE": 6,
      "HAZ_LD": 7, "HAZ_ST": 8, "HAZ_EX": 9, "HAZ_VE": 10, "DISP_FULL": 11, "STARVE": 12, "ALL_IDLE": 13,
      "EX_STEP": 14, "EX_USEFUL": 15, "LD_BUSY": 18, "LD_BEATS": 19, "ST_BUSY": 21, "VE_ACTIVE": 24}


def read(path):
    """{section label: (per_step, [row dicts])}"""
    out, cur = {}, None
    hdr = None
    for line in open(path):
        line = line.strip()
        if line.startswith("# section"):
            p = line.split()
            cur = p[3]
            out[cur] = (float(p[5]), [])
        elif line.startswith("export,"):
            hdr = line.split(",")
        elif line and cur:
            v = line.split(",")
            r = {"export": v[0]}
            for k, x in zip(hdr[1:], v[1:]):
                r[k] = int(x)
            out[cur][1].append(r)
    return out


def c(r, name):
    return r.get(f"c{PC[name]}", 0)


def agg(rows):
    """sums of a group of rows"""
    s = defaultdict(int)
    for r in rows:
        for k, v in r.items():
            if k != "export":
                s[k] += v
    return s


def split(s):
    """the unit's window cycles by state, the commands, the core's side"""
    cyc = c(s, "CYCLES")
    haz = sum(c(s, k) for k in ("HAZ_LD", "HAZ_ST", "HAZ_EX", "HAZ_VE"))
    idle, starve, full = c(s, "ALL_IDLE"), c(s, "STARVE"), c(s, "DISP_FULL")
    cmds = sum(c(s, k) for k in ("CMD_LD", "CMD_ST", "CMD_EX", "CMD_VE"))
    return {"cycles": cyc, "all_idle": idle, "starve": starve, "haz": haz, "disp_full": full,
            "moving": cyc - idle - starve - haz - full, "cmds": cmds, "core_cycles": s.get("cycles", 0),
            "instret": s.get("instret", 0), "pcpi_qfull": c(s, "PCPI_QFULL"), "pcpi_fence": c(s, "PCPI_FENCE"),
            "calls": s.get("calls", 0)}


def kind(export):
    """an export's kernel kind: the name without the dispatch number"""
    return re.sub(r"^\w+\$async_dispatch_\d+_", "", export)


def ms(x, mhz):
    return x / (mhz * 1e3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir")
    ap.add_argument("--section", default="decode")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--mhz", type=float, default=100.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    f = {k: os.path.join(a.dir, f"rvperf_{k}.csv") for k in ("desc", "rv", "sched", "sched_rv")}
    res = {}
    # 1. per dispatch windows: one decode step, the lists against the code
    if os.path.exists(f["desc"]) and os.path.exists(f["rv"]):
        D, R = read(f["desc"]), read(f["rv"])
        if a.section not in D or a.section not in R:
            sys.exit(f"no section '{a.section}' in both ({list(D)}, {list(R)})")
        (pd, dr), (pr, rr) = D[a.section], R[a.section]
        if not any(f"c{PC['CYCLES']}" in r for r in dr + rr):
            sys.exit("no event counters in the csv (SA_PROFILE_PERF on the board, rt_fw 7)")
        sd = {k: v / pd for k, v in split(agg(dr)).items()}
        sr = {k: v / pr for k, v in split(agg(rr)).items()}
        print(f"1. one {a.section} step, every dispatch in its own window ({sd['calls']:.0f} dispatches; the unit's cycles, ms):")
        print(f"   {'':12}{'window':>9}{'all idle':>10}{'starve':>9}{'hazard':>9}{'q full':>9}{'moving':>9}"
              f"{'commands':>10}{'core instr':>12}{'instr/cmd':>10}{'CPI':>9}")
        for name, s in (("lists", sd), ("S4 code", sr)):
            ipc = s["instret"] / s["cmds"] if s["cmds"] else 0
            cpi = s["core_cycles"] / s["instret"] if s["instret"] else 0
            print(f"   {name:12}" + "".join(f"{ms(s[k], a.mhz):9.2f}" for k in ("cycles", "all_idle", "starve", "haz",
                                                                                 "disp_full", "moving"))
                  + f" {s['cmds']:9.0f}{s['instret']:12.0f}{ipc:10.1f}{cpi:9.1f}")
        dl = {k: sr[k] - sd[k] for k in ("cycles", "all_idle", "starve", "haz", "disp_full", "moving")}
        print("   difference  " + "".join(f"{ms(dl[k], a.mhz):+9.2f}" for k in dl))
        print(f"   S4: the core stalled on PCPI {ms(sr['pcpi_qfull'], a.mhz):.2f} ms (queue full) + "
              f"{ms(sr['pcpi_fence'], a.mhz):.2f} ms (fences) per step")
        res["step"] = {"lists": sd, "s4": sr}
        # 2. by kernel kind: where the difference is
        gd, gr = defaultdict(list), defaultdict(list)
        for r in dr:
            gd[kind(r["export"])].append(r)
        for r in rr:
            gr[kind(r["export"])].append(r)
        rows = []
        for k in set(gd) | set(gr):
            x = {n: v / pd for n, v in split(agg(gd.get(k, []))).items()}
            y = {n: v / pr for n, v in split(agg(gr.get(k, []))).items()}
            rows.append((k, x, y))
        rows.sort(key=lambda t: -(t[2]["cycles"] - t[1]["cycles"]))
        print(f"\n2. by kernel kind, the largest differences first (per step; ms of the unit's window):")
        print(f"   {'kernel':46}{'calls':>6}{'lists':>8}{'S4':>8}{'diff':>8}{'S4 idle+starve':>15}"
              f"{'instr/cmd':>10}{'CPI':>9}")
        for k, x, y in rows[:a.top]:
            ipc = y["instret"] / y["cmds"] if y["cmds"] else 0
            cpi = y["core_cycles"] / y["instret"] if y["instret"] else 0
            print(f"   {k[:46]:46}{y['calls']:6.0f}{ms(x['cycles'], a.mhz):8.2f}{ms(y['cycles'], a.mhz):8.2f}"
                  f"{ms(y['cycles'] - x['cycles'], a.mhz):+8.2f}{ms(y['all_idle'] + y['starve'], a.mhz):15.2f}"
                  f"{ipc:10.1f}{cpi:9.1f}")
        res["kinds"] = [{"kind": k, "lists": x, "s4": y} for k, x, y in rows]
    # 3. plan B's loop: the whole run (lists against code)
    if os.path.exists(f["sched"]) and os.path.exists(f["sched_rv"]):
        print("\n3. plan B's decode loop on the core, counters over the whole run (per step, ms):")
        print(f"   {'':12}{'window':>9}{'all idle':>10}{'starve':>9}{'hazard':>9}{'q full':>9}{'moving':>9}"
              f"{'commands':>10}{'core instr':>12}{'CPI':>9}")
        out = {}
        for name, key in (("lists", "sched"), ("S4 code", "sched_rv")):
            S = read(f[key])
            (ps, rows), = [(p, r) for p, r in S.values()]
            s = {k: v / ps for k, v in split(agg(rows)).items()}
            cpi = s["core_cycles"] / s["instret"] if s["instret"] else 0
            print(f"   {name:12}" + "".join(f"{ms(s[k], a.mhz):9.2f}" for k in ("cycles", "all_idle", "starve", "haz",
                                                                                 "disp_full", "moving"))
                  + f" {s['cmds']:9.0f}{s['instret']:12.0f}{cpi:9.1f}")
            out[key] = s
        d = {k: out["sched_rv"][k] - out["sched"][k] for k in ("cycles", "all_idle", "starve", "haz", "disp_full", "moving")}
        print("   difference  " + "".join(f"{ms(d[k], a.mhz):+9.2f}" for k in d))
        res["sched"] = out
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

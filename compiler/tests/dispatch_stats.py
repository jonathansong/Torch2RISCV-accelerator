#!/usr/bin/env python3
"""WP9 per-dispatch statistics (docs/generality_implementation_plan.md §13.1):
where the work of one step runs (accelerator / host fallback), per dispatch
category, and the coverage (dynamic dispatches, time, MACs).

Inputs:
  --csv   the SA_PROFILE_PERF csv of a run with SA_PROFILE=1 (one list per
          dispatch, so every export is timed): sa-llm-run on the sim or
          board_profile.py on the board (there also the unit's event counters:
          DDR bytes, EX steps). Sections: "prefill" (per chunk after the
          first) and "decode" (per step); host fallbacks are "host:" rows.
  --report the compiler's codegen report (iree-compile --iree-sa-codegen-report
          stderr, as test_hf_generic.py / compile_sa.sh with
          SA_COMPILE_FLAGS=--iree-sa-codegen-report write it): the static
          coverage and why a dispatch runs on the host.

MACs come from the dispatch names (IREE names a contraction by its loop
ranges: matvec_like_128x16x1024 = 128 * 16 * 1024 MACs), so they are the
model's work, not the padded work of the array. A dynamic loop range (D in the
name: the attention's length T with --abi=hfv) takes the length the profiled
run used: sa-llm-run --pad=P passes T = (pos // P + 1) * P at decode position
pos, and (s + M - 1) // P * P + P for a prefill chunk of M rows starting at s;
--prompt / --chunk / --pad describe the run (board_profile.py 40 --prompt 32
with a M = 16 bundle: the defaults), --dyn-len gives T directly.

    compiler/tests/dispatch_stats.py --csv prof.csv [--section decode] [--report compile.err] [--json out.json]
"""
import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from z1_profile import sections  # noqa: E402

PC_CYCLES, PC_EX_USEFUL, PC_LD_BEATS, PC_ST_BEATS = 0, 15, 19, 22   # rtl/sysarray/sa_defs.vh
LANE = 8                                                              # LD/ST_BEATS count 8-byte lanes
CONTRACTIONS = ("matmul", "matvec", "contract", "conv")

# why a dispatch runs on the host -> the plan's classes (§12 acceptance:
# frontend, rewrite, lowering, hardware primitive, memory capacity, runtime)
REASONS = [
    (r"host-dispatches", "forced (testing)"),
    (r"ACC full|SPAD|capacity|does not fit|too large|bank", "memory capacity"),
    (r"exp|div|sqrt|rsqrt|tanh|erf|log|pow|f64|i64|unsupported element|float", "hardware primitive"),
    (r"", "lowering"),
]


def split_name(export):
    """-> (category, dims) from an IREE dispatch name"""
    t = re.sub(r"^.*?\$async_dispatch_\d+_?", "", export.split(":", 1)[-1])
    words, dims = [], []
    for w in t.split("_"):
        if re.fullmatch(r"(\d+|D)(x(\d+|D))*", w):
            dims = [int(x) if x.isdigit() else None for x in w.split("x")] if not dims else dims
            break
        if re.fullmatch(r"[a-z]+", w) and not re.fullmatch(r"[if]\d+", w):
            words.append(w)
        else:
            break
    return "_".join(words) or "other", dims


def macs(cat, dims, dyn=None):
    """a contraction's MACs; a dynamic loop range (None) counts as dyn (0 without it)"""
    if not any(c in cat for c in CONTRACTIONS) or not dims or (None in dims and not dyn):
        return 0
    return math.prod(dyn if d is None else d for d in dims)


def dyn_len(label, per_step, prompt, chunk, pad):
    """the mean attention length T of the profiled run (see the docstring)"""
    n = int(round(per_step))
    if label == "prefill":            # the chunks after the first (the profile is reset after it)
        starts, s0 = [], 0
        while True:
            starts.append(prompt - chunk if s0 + chunk >= prompt else s0)
            if s0 + chunk >= prompt:
                break
            s0 += chunk
        ts = [(s + chunk - 1) // pad * pad + pad for s in starts[1:]][-n:]
    else:                             # decode steps at positions prompt, prompt + 1, ...
        ts = [(pos // pad + 1) * pad for pos in range(prompt, prompt + n)]
    return sum(ts) / len(ts) if ts else None


def reason_class(why):
    for pat, cls in REASONS:
        if re.search(pat, why):
            return cls
    return "lowering"


def read_report(path):
    """-> {export: None (accelerator) | why (host)}"""
    out = {}
    for line in open(path, errors="replace"):
        m = re.match(r"sa codegen: (\S+): (ok|on the host \((.*)\)|failed)", line.rstrip("\n"))
        if not m:
            continue
        if m.group(2) == "ok":
            out[m.group(1)] = None
        elif m.group(2).startswith("on the host"):
            out[m.group(1)] = m.group(3)
    return out


def section_rows(path, section):
    secs = sections(path)
    labels = []
    for line in open(path):
        if line.startswith("# section"):
            f = line.split()
            labels.append(f[3] if len(f) >= 6 else str(len(labels)))
    if section.isdigit():
        i = int(section)
    elif section in labels:
        i = labels.index(section)
    else:
        sys.exit(f"{path}: no section {section!r} (sections: {', '.join(labels)})")
    per_step, rows = secs[i]
    return labels[i], per_step, rows


def stats(rows, per_step, hz, dyn=None):
    if any(r["export"] == "(list of dispatches)" for r in rows):
        sys.exit("batched profile (SA_PROFILE=batch): per-dispatch statistics need SA_PROFILE=1")
    cats, tot = {}, dict(acc_calls=0, host_calls=0, cycles=0, host_ns=0, sub_ns=0, acc_macs=0, host_macs=0,
                         ld_bytes=0, st_bytes=0, ex_useful=0)
    have_perf = any(f"c{PC_LD_BEATS}" in r for r in rows)
    for r in rows:
        host = r["export"].startswith("host:")
        cat, dims = split_name(r["export"])
        m = macs(cat, dims, dyn) * r["calls"]
        c = cats.setdefault(cat, dict(acc_calls=0, host_calls=0, cycles=0, host_ns=0, acc_macs=0, host_macs=0,
                                      ld_bytes=0, st_bytes=0, exports=set()))
        c["exports"].add(r["export"])
        if host:
            for d in (c, tot):
                d["host_calls"] += r["calls"]
                d["host_ns"] += r["host_ns"]
                d["host_macs"] += m
        else:
            for d in (c, tot):
                d["acc_calls"] += r["calls"]
                d["cycles"] += r["cycles"]
                d["acc_macs"] += m
                d["ld_bytes"] += r.get(f"c{PC_LD_BEATS}", 0) * LANE
                d["st_bytes"] += r.get(f"c{PC_ST_BEATS}", 0) * LANE
            tot["sub_ns"] += r["host_ns"]
            tot["ex_useful"] += r.get(f"c{PC_EX_USEFUL}", 0)
    s = 1.0 / per_step
    acc_ms = tot["cycles"] / hz * 1e3 * s
    host_ms = tot["host_ns"] / 1e6 * s
    calls = tot["acc_calls"] + tot["host_calls"]
    allm = tot["acc_macs"] + tot["host_macs"]
    summary = dict(
        per_step=per_step, have_perf=have_perf,
        dispatches=calls * s, acc_dispatches=tot["acc_calls"] * s, host_dispatches=tot["host_calls"] * s,
        acc_cycles=tot["cycles"] * s, acc_ms=acc_ms, host_ms=host_ms, submit_ms=tot["sub_ns"] / 1e6 * s,
        macs=allm * s, acc_macs=tot["acc_macs"] * s,
        cov_dynamic=tot["acc_calls"] / calls if calls else None,
        cov_time=acc_ms / (acc_ms + host_ms) if acc_ms + host_ms and tot["cycles"] else None,
        cov_macs=tot["acc_macs"] / allm if allm else None,
        ld_mb=tot["ld_bytes"] * s / 1e6 if have_perf else None,
        st_mb=tot["st_bytes"] * s / 1e6 if have_perf else None,
        ld_gbs=tot["ld_bytes"] / (tot["cycles"] / hz) / 1e9 if have_perf and tot["cycles"] else None,
    )
    table = []
    for cat, c in cats.items():
        table.append(dict(category=cat, exports=len(c["exports"]), calls=(c["acc_calls"] + c["host_calls"]) * s,
                          host_calls=c["host_calls"] * s, cycles=c["cycles"] * s,
                          cyc_pct=100 * c["cycles"] / tot["cycles"] if tot["cycles"] else 0,
                          acc_ms=c["cycles"] / hz * 1e3 * s, host_ms=c["host_ns"] / 1e6 * s,
                          macs=(c["acc_macs"] + c["host_macs"]) * s,
                          ld_mb=c["ld_bytes"] * s / 1e6 if have_perf else None))
    table.sort(key=lambda t: (-(t["acc_ms"] + t["host_ms"]), t["category"]))
    return summary, table


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", help="SA_PROFILE_PERF csv (SA_PROFILE=1)")
    ap.add_argument("--section", default="decode", help="prefill / decode / an index (default decode)")
    ap.add_argument("--report", help="codegen report (iree-compile --iree-sa-codegen-report stderr)")
    ap.add_argument("--hz", type=float, default=float(os.environ.get("SA_DEVICE_HZ", 100e6)))
    ap.add_argument("--json", help="write the statistics here")
    ap.add_argument("--prompt", type=int, default=32, help="the run's prompt tokens (for the dynamic length T)")
    ap.add_argument("--chunk", type=int, default=16, help="the prefill chunk M")
    ap.add_argument("--pad", type=int, default=16, help="sa-llm-run --pad (T rounded up to it)")
    ap.add_argument("--dyn-len", type=float, help="the dynamic length T itself (instead of --prompt / --chunk / --pad)")
    args = ap.parse_args()
    if not args.csv and not args.report:
        ap.error("--csv and / or --report")
    out = {}
    if args.report:
        rep = read_report(args.report)
        host = {k: v for k, v in rep.items() if v is not None}
        by_class = {}
        for k, why in sorted(host.items()):
            by_class.setdefault(reason_class(why), []).append((k, why))
        print(f"static: {len(rep)} executables, {len(rep) - len(host)} on the accelerator, {len(host)} on the host")
        for cls, items in by_class.items():
            print(f"  {cls}: {len(items)}")
            for k, why in items[:8]:
                print(f"    {k}: {why}")
            if len(items) > 8:
                print(f"    ... {len(items) - 8} more")
        out["static"] = dict(executables=len(rep), host=len(host),
                             host_reasons={cls: [dict(export=k, why=w) for k, w in v] for cls, v in by_class.items()})
    if args.csv:
        label, per_step, rows = section_rows(args.csv, args.section)
        dyn = args.dyn_len or dyn_len(label, per_step, args.prompt, args.chunk, args.pad)
        summary, table = stats(rows, per_step, args.hz, dyn)
        summary["dyn_len"] = dyn
        unit = "chunk" if label == "prefill" else "step"
        S = summary
        print(f"\n{label}, per {unit} ({per_step:g} {unit}s profiled):")
        print(f"  dispatches {S['dispatches']:.1f}: accelerator {S['acc_dispatches']:.1f}, host {S['host_dispatches']:.1f}"
              f"  -> dynamic coverage {pct(S['cov_dynamic'])}")
        print(f"  accelerator {S['acc_ms']:.2f} ms ({S['acc_cycles']:.0f} cycles at {args.hz / 1e6:g} MHz), "
              f"host fallback {S['host_ms']:.2f} ms  -> time coverage {pct(S['cov_time'])}"
              + ("" if S["acc_cycles"] else " (no device cycles: the sim)"))
        print(f"  MACs {S['macs'] / 1e6:.1f}M, on the accelerator {S['acc_macs'] / 1e6:.1f}M  -> MAC coverage {pct(S['cov_macs'])}"
              f" (dynamic length T = {dyn:g})")
        print(f"  submissions (list build, ring, wait) {S['submit_ms']:.2f} ms")
        if S["have_perf"]:
            print(f"  DDR read {S['ld_mb']:.1f} MB, write {S['st_mb']:.2f} MB; read {S['ld_gbs']:.2f} GB/s over the device time")
        print(f"\n  {'category':<28} {'exports':>7} {'calls':>8} {'host':>6} {'cycles':>12} {'cyc %':>6} "
              f"{'acc ms':>8} {'host ms':>8} {'MMAC':>8}" + (f" {'LD MB':>8}" if S["have_perf"] else ""))
        for t in table:
            print(f"  {t['category']:<28.28} {t['exports']:>7} {t['calls']:>8.1f} {t['host_calls']:>6.1f} {t['cycles']:>12.0f} "
                  f"{t['cyc_pct']:>5.1f}% {t['acc_ms']:>8.2f} {t['host_ms']:>8.2f} {t['macs'] / 1e6:>8.2f}"
                  + (f" {t['ld_mb']:>8.2f}" if S["have_perf"] else ""))
        out[label] = dict(summary=summary, categories=table)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

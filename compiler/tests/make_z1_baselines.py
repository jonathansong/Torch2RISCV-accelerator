#!/usr/bin/env python3
"""The PYNQ-Z1 freeze baselines (docs/kv260_upgrade_plan.md §7.2): the reference
the KV260 port is checked against, without the Z1 board.

Run on the host after board_regress.py passed on the board and its results/
were copied back (build/deploy_z1/results, staged by deploy_z1_freeze.sh):

    $SA_PY compiler/tests/make_z1_baselines.py [--deploy build/deploy_z1] [--skip-dcheck]

Writes tests/baselines/pynq-z1/:
  golden_manifest.txt   the golden corpus re-recorded with this compiler (golden.py
                        --record): case, dispatch, key, sha256 of its descriptors
  dispatch_check.json   dispatch_check.py --dirty per model build: each dispatch's result
  stories15m_tokens.txt the board's tokens of the stories15M runs (c3, c6p_stories)
  smollm2_tokens.txt    the board's tokens of the SmolLM2-135M runs (c55, c6p_smollm2, hfgen)
  perf.md               speeds, bandwidth, profiles, resources and timing of the bitstream
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "tests", "baselines", "pynq-z1")
GOLDEN = os.path.join(REPO, "build", "golden")
PY = sys.executable

# model builds checked per dispatch: (name, build dir); binaries in sa_bin or sa_binaries
BUILDS = [
    ("stories15M decode (c3)", "build/c3/stories15M"),
    ("SmolLM2-135M decode, qhf (c55)", "build/c55/smollm2"),
    ("stories15M prefill + decode (c6p_stories)", "build/c6p/stories_m8"),
    ("SmolLM2-135M prefill + decode, qhf (c6p_smollm2)", "build/c6p/smollm2_m8"),
    ("SmolLM2-135M prefill + decode, generic HF (hfgen)", "build/hfgen/smollm2_p8"),
]
TOKENS = {"stories15m_tokens.txt": ["c3", "c6p_stories"], "smollm2_tokens.txt": ["c55", "c6p_smollm2", "hfgen"]}
LINE = re.compile(r"^#\s*(\S+) (.{44}) (.*)$")


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    return r.returncode, r.stdout + r.stderr


def golden_manifest(commit):
    rc, out = sh([PY, os.path.join(REPO, "compiler", "tests", "golden.py"), "--record"])
    print(out.strip().splitlines()[-1])
    if rc:
        sys.exit("golden.py --record failed")
    m = json.load(open(os.path.join(GOLDEN, "manifest.json")))
    lines = [f"# golden corpus at {commit}: case, dispatch source, key, sha256 of the descriptors (- : fails to compile)"]
    for case, c in m["cases"].items():
        for e in c["entries"]:
            p = os.path.join(GOLDEN, "exp", e["key"] + ".sadesc")
            h = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "-"
            lines.append(f"{case} {e['name']} {e['key']} {h}")
    with open(os.path.join(OUT, "golden_manifest.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"golden_manifest.txt: {len(lines) - 1} entries")


def dispatch_checks():
    res = {}
    for name, d in BUILDS:
        src = os.path.join(REPO, d, "sa_sources")
        bins = next((os.path.join(REPO, d, b) for b in ("sa_bin", "sa_binaries") if os.path.isdir(os.path.join(REPO, d, b))), None)
        if not os.path.isdir(src) or not bins:
            print(f"{name}: no sources / binaries in {d}, skipped")
            continue
        rc, out = sh(["systemd-run", "--user", "--scope", "-p", "MemoryMax=12G", "-q", PY,
                      os.path.join(REPO, "compiler", "tests", "dispatch_check.py"), "--dirty", src, bins])
        per = {}
        for l in out.splitlines():
            mm = LINE.match(l)
            if mm:
                per[mm.group(1)] = {"dispatch": mm.group(2).strip(), "result": mm.group(3).strip()}
        counts = out.strip().splitlines()[-1] if out.strip() else ""
        res[name] = {"build": d, "summary": counts, "dispatches": per}
        print(f"{name}: {counts}")
    with open(os.path.join(OUT, "dispatch_check.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)


def tokens(deploy, results):
    for fname, runs in TOKENS.items():
        lines = []
        for t in runs:
            p = os.path.join(results, t + "_tokens.txt")
            if not os.path.exists(p):
                print(f"{fname}: no board tokens of {t}")
                continue
            got = [int(x) for x in open(p).read().split()]
            want = [int(x) for x in np.load(os.path.join(deploy, t, "expected_tokens.npy"))]
            prompt = [int(x) for x in np.load(os.path.join(deploy, t, "prompt.npy"))]
            same = got[:len(want)] == want
            if not same:
                sys.exit(f"{t}: the board's tokens differ from the host reference")
            args = open(os.path.join(deploy, t, "sa_args.txt")).read().strip() if os.path.exists(
                os.path.join(deploy, t, "sa_args.txt")) else ""
            lines += [f"# {t}: {len(prompt)} prompt tokens, {len(got) - len(prompt)} generated; sa-llm-run {args}".rstrip(),
                      "prompt " + " ".join(map(str, prompt)), "tokens " + " ".join(map(str, got)), ""]
        with open(os.path.join(OUT, fname), "w") as f:
            f.write("\n".join(lines))
        print(f"{fname}: {sum(1 for l in lines if l.startswith('tokens'))} runs")


def perf(results, commit):
    def log(n):
        p = os.path.join(results, n + ".log")
        return open(p).read() if os.path.exists(p) else ""
    out = [f"# PYNQ-Z1 baseline: performance ({commit})", ""]
    summ = open(os.path.join(results, "summary.txt")).read().strip()
    out += ["## Board regression", "", "```", summ, "```", ""]
    out += ["## Speed (board_llm.py: sa-llm-run's timing line)", "", "| test | timing |", "|---|---|"]
    for t in ("c3", "c55", "c6p_stories", "c6p_smollm2", "hfgen"):
        for l in log(t).splitlines():
            if "ms per step" in l or l.startswith("prefill:"):
                out.append(f"| {t} | {l.strip()} |")
    out += ["", "## DMA bandwidth (m2_bw_test.py)", "", "```", log("bwtest").split("--- stderr ---")[0].strip(), "```", ""]
    for t in ("c6p_stories", "c6p_smollm2", "hfgen"):
        p = log(t + "_profile").split("--- stderr ---")[0].strip()
        if p:
            out += [f"## Profile: {t} (board_profile.py)", "", "```", p, "```", ""]
    l2 = os.path.join(REPO, "RISCV-on-PYNQ-Z1", "bitstreams", "l2")
    rd = open(os.path.join(l2, "README.md")).read().splitlines()
    res = [l for l in rd if "LUT" in l and "WNS" in l]
    out += ["## Bitstream (RISCV-on-PYNQ-Z1/bitstreams/l2)", ""] + res + [
        "", "Reports: `utilization.rpt`, `timing_summary.rpt` in the same directory.", ""]
    with open(os.path.join(OUT, "perf.md"), "w") as f:
        f.write("\n".join(out))
    print("perf.md written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", default=os.path.join(REPO, "build", "deploy_z1"))
    ap.add_argument("--skip-dcheck", action="store_true")
    ap.add_argument("--skip-golden", action="store_true")
    args = ap.parse_args()
    results = os.path.join(args.deploy, "results")
    summ = os.path.join(results, "summary.txt")
    if not os.path.exists(summ) or "REGRESSION PASS" not in open(summ).read():
        sys.exit(f"{summ}: no passing board regression (run board_regress.py on the board, copy results/ back)")
    commit = sh(["git", "rev-parse", "--short", "HEAD"])[1].strip()
    staged = open(os.path.join(args.deploy, "VERSION")).read().split()[1][:len(commit)]
    if staged != commit:
        print(f"warning: the bundles were staged at {staged}, HEAD is {commit}")
    os.makedirs(OUT, exist_ok=True)
    tokens(args.deploy, results)
    perf(results, commit)
    if not args.skip_golden:
        golden_manifest(commit)
    if not args.skip_dcheck:
        dispatch_checks()
    print(f"-> {os.path.relpath(OUT, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compares a board regression's results with the PYNQ-Z1 freeze baselines
(tests/baselines/pynq-z1/; docs/kv260_upgrade_plan.md K1a: the token
sequences must be identical) and puts the speeds and device cycles side by
side.

    $SA_PY compiler/tests/compare_z1_baselines.py [results dir, default build/deploy_kv260_d8_100mhz/results]

Exit 0 when every run with a baseline has identical tokens.
"""
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE = os.path.join(REPO, "tests", "baselines", "pynq-z1")
Z1_RESULTS = os.path.join(REPO, "build", "deploy_z1", "results")    # (speeds; optional)


def baseline_tokens():
    """{run: [tokens]} from the baseline token files ('# <run>: ...' then 'tokens ...')."""
    out = {}
    for f in ("stories15m_tokens.txt", "smollm2_tokens.txt"):
        run = None
        for line in open(os.path.join(BASE, f)):
            if line.startswith("# "):
                run = line[2:].split(":")[0].strip()
            elif line.startswith("tokens ") and run:
                out[run] = [int(t) for t in line.split()[1:]]
    return out


def timing(results, run):
    """sa-llm-run's timing line, and the profile's decode cycles per step."""
    t = c = ""
    p = os.path.join(results, run + ".log")
    if os.path.exists(p):
        for line in open(p):
            if "ms per step" in line:
                m = re.findall(r"([\d.]+) ms per step[^(]*\(([\d.]+) tokens/s\)", line)
                if m:
                    t = f"{m[-1][0]} ms/step ({m[-1][1]} tok/s)"
    p = os.path.join(results, run + "_profile.log")
    if os.path.exists(p):
        for line in open(p):
            m = re.search(r"per step \(decode.*?\): [\d.]+ dispatches, device [\d.]+ ms \((\d+) cycles", line)
            if m:
                c = m.group(1)
    return t, c


def main():
    results = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "build", "deploy_kv260_d8_100mhz", "results")
    base = baseline_tokens()
    ok, rows = True, []
    for run, want in base.items():
        p = os.path.join(results, run + "_tokens.txt")
        if not os.path.exists(p):
            rows.append((run, "MISSING", "", "", "", ""))
            ok = False
            continue
        got = [int(t) for t in open(p).read().split()]
        same = got == want
        ok &= same
        t, c = timing(results, run)
        zt, zc = timing(Z1_RESULTS, run) if os.path.isdir(Z1_RESULTS) else ("", "")
        status = f"identical ({len(got)} tokens)" if same else \
            f"DIFFERENT at {next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))}"
        rows.append((run, status, zt, t, zc, c))
    print(f"{'run':14s} {'tokens vs the Z1 baseline':28s} {'Z1':28s} {'this board':28s} {'Z1 cycles':>10s} {'cycles':>10s}")
    for r in rows:
        print(f"{r[0]:14s} {r[1]:28s} {r[2]:28s} {r[3]:28s} {r[4]:>10s} {r[5]:>10s}")
    print("TOKENS IDENTICAL" if ok else "TOKENS DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

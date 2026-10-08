#!/usr/bin/env python3
"""Golden corpus of the sa code generator (docs/iree_compiler_plan.md §8.14, C8 R0).

The C8 refactor must not change what the compiler emits: every dispatch of the
corpus compiles to the same sa-desc bytes as when the corpus was recorded.

A dispatch source (iree-compile --iree-hal-dump-executable-sources-to) is
self-contained (its #hal.executable.target carries the target config), and
`iree-compile --compile-mode=hal-executable` on it gives the same bytes as the
whole-model compile (--iree-hal-dump-executable-binaries-to). So the corpus is
the sources alone, copied into build/golden/src (the models' build directories
change when they are re-exported), deduplicated by content and flags.

    python3 compiler/tests/golden.py --record [-j 6]   # (re)record from the cases below
    python3 compiler/tests/golden.py [-j 6] [--case stories_m8]   # compare
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
GOLDEN = os.path.join(REPO, "build", "golden")
IREE_BUILD = os.environ.get("IREE_BUILD", os.path.join(REPO, "build", "iree", "build-compiler"))
COMPILE = os.path.join(IREE_BUILD, "tools", "iree-compile")

# (case, the model's sa_sources, extra iree-compile flags)
CASES = [
    ("stories_m8", "build/c6p/stories_m8/sa_sources", []),                  # decode + prefill (M = 8)
    ("stories_m8_none", "build/c6p/stories_m8/sa_sources", ["--iree-sa-ukernels=none"]),
    ("stories_m16", "build/c6p/stories_m16/sa_sources", []),
    ("stories_m16_none", "build/c6p/stories_m16/sa_sources", ["--iree-sa-ukernels=none"]),
    ("smollm2_m8", "build/c6p/smollm2_m8/sa_sources", []),                  # GQA, tied embedding
    ("smollm2_m8_none", "build/c6p/smollm2_m8/sa_sources", ["--iree-sa-ukernels=none"]),
    ("smollm2_m16", "build/c6p/smollm2_m16/sa_sources", []),
    ("qwen3", "build/c6/qwen3/sa_sources", []),                             # QK-norm, 28 layers
    ("k16k", "build/c6/k16k/sa_sources", []),                               # K blocks
    ("hf_smollm2_q", "build/hfgen/smollm2_q/sa_sources", []),               # generic frontend (unmodified HF, W8A8)
    ("cfg_d16", "build/c6/cfg/d16/sa_sources", []),                         # other targets
    ("cfg_big", "build/c6/cfg/big/sa_sources", []),
    ("smollm2_d16", "build/c55/smollm2_d16/sa_sources", []),               # D = 16, GQA (K1c: per-row broadcasts of 9 rows)
]


def sa_only(text):
    """The source without the host fallback's other variants (plan §8.15): the
    corpus is the sa code generator's."""
    lines, out, skip = text.split("\n"), [], None
    for l in lines:
        if skip is None and l.lstrip().startswith("hal.executable.variant") and '<"sa"' not in l:
            skip = len(l) - len(l.lstrip())           # its indentation: the variant ends at the same one
            continue
        if skip is not None:
            if l.strip() == "}" and len(l) - len(l.lstrip()) == skip:
                skip = None
            continue
        out.append(l)
    return "\n".join(out)


def canonical(data):
    """The disassembly with each BASE register named by its setup (binding and
    offset) instead of its number: two executables equal here differ only in
    the numbering of the BASE registers (the order the lowering met the DDR
    views), not in what they do."""
    import re
    sys.path.insert(0, os.path.join(REPO, "compiler", "runtime", "tools"))
    import sadesc
    import sadis
    out = []
    _, _, exps = sadesc.read(data)
    for (name, rows, nb, nc, cyc, setup), ext in zip(exps, sadesc.read_ext(data)):
        names = {s[1]: f"<b{s[2]}+c{s[3]}*{s[4]}/{s[5]}+{s[6]}>" for s in setup if s[0] == 0}
        pnames = {s[1]: f"<c{s[3]}*{s[4]}/{s[5]}+{s[6]}>" for s in setup if s[0] != 0}   # setup PARAMs
        ext = dict(ext)
        if "prefix_reg" in ext:                      # (the BASE register the prefix was generated for; unused without one)
            ext["prefix_reg"] = names.get(ext["prefix_reg"], ext["prefix_reg"]) if ext.get("prefix") else None
        out.append(f"{name} {rows.shape[0]} {nb} {nc} {ext}")
        out += sorted(f"setup {names[s[1]]}" if s[0] == 0 else f"setup PARAM {pnames[s[1]]}" for s in setup)
        for i, w in enumerate(rows):
            t = sadis.line(i, w)
            t = re.sub(r"base=(\d+)", lambda m: "base=" + names.get(int(m.group(1)), m.group(1)), t)
            t = re.sub(r"\bBASE(\d+)\b", lambda m: "BASE" + names.get(int(m.group(1)), m.group(1)), t)
            t = re.sub(r"dyn\(f(\d+),p(\d+)", lambda m: f"dyn(f{m.group(1)},p" + pnames.get(int(m.group(2)), m.group(2)), t)
            t = re.sub(r"param=(\d+)", lambda m: "param=" + pnames.get(int(m.group(1)), m.group(1)), t)
            out.append(t)
    return "\n".join(out)


def key_of(text, flags):
    return hashlib.sha256((" ".join(flags) + "\n" + text).encode()).hexdigest()[:20]


def compile_one(src, flags, out):
    """sa-desc bytes, or None and the error's last lines."""
    r = subprocess.run([COMPILE, src, "--compile-mode=hal-executable", "--iree-hal-target-device=sa", *flags,
                        "-o", out], capture_output=True, text=True, timeout=600)
    if r.returncode:
        return None, "\n".join(r.stderr.strip().splitlines()[-3:])
    with open(out, "rb") as f:
        return f.read(), ""


def run_all(items, jobs, tmp):
    """items: {key: (src, flags)} -> {key: (bytes or None, err)}"""
    res = {}
    with cf.ThreadPoolExecutor(jobs) as ex:
        futs = {ex.submit(compile_one, src, flags, os.path.join(tmp, k + ".sadesc")): k
                for k, (src, flags) in items.items()}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            res[futs[fu]] = fu.result()
            if i % 200 == 0:
                print(f"  {i}/{len(items)}", flush=True)
    return res


def record(args):
    manifest, items = {}, {}
    for case, d, flags in CASES:
        d = os.path.join(REPO, d)
        if not os.path.isdir(d):
            print(f"{case}: {d} missing, skipped")
            continue
        entries = []
        for name in sorted(os.listdir(d)):
            text = sa_only(open(os.path.join(d, name)).read())
            k = key_of(text, flags)
            entries.append({"name": name, "key": k})
            items.setdefault(k, (name, text, flags))
        manifest[case] = {"flags": flags, "from": os.path.relpath(d, REPO), "entries": entries}
    shutil.rmtree(GOLDEN, ignore_errors=True)
    for sub in ("src", "exp"):
        os.makedirs(os.path.join(GOLDEN, sub))
    todo = {}
    for k, (name, text, flags) in items.items():
        p = os.path.join(GOLDEN, "src", k + ".mlir")
        with open(p, "w") as f:
            f.write(text)
        todo[k] = (p, flags)
    t0 = time.time()
    res = run_all(todo, args.jobs, os.path.join(GOLDEN, "exp"))
    errors = {}
    for k, (data, err) in res.items():
        if data is None:
            errors[k] = err
    with open(os.path.join(GOLDEN, "manifest.json"), "w") as f:
        json.dump({"cases": manifest, "errors": errors}, f, indent=1)
    n = sum(len(c["entries"]) for c in manifest.values())
    print(f"recorded {len(todo)} unique dispatches ({n} in {len(manifest)} cases, {len(errors)} fail to compile) "
          f"in {time.time() - t0:.0f} s -> {os.path.relpath(GOLDEN, REPO)}")
    return 0


def compare(args):
    mf = json.load(open(os.path.join(GOLDEN, "manifest.json")))
    cases = {c: v for c, v in mf["cases"].items() if not args.case or c in args.case}
    keys = sorted({e["key"] for v in cases.values() for e in v["entries"]})
    flags = {e["key"]: v["flags"] for v in cases.values() for e in v["entries"]}
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="sa_golden_") as tmp:
        res = run_all({k: (os.path.join(GOLDEN, "src", k + ".mlir"), flags[k]) for k in keys}, args.jobs, tmp)
    bad, renamed = {}, set()
    for k in keys:
        data, err = res[k]
        if k in mf["errors"]:
            if data is not None:
                bad[k] = "compiles now (failed when recorded)"
            continue
        if data is None:
            bad[k] = "fails to compile: " + err
            continue
        with open(os.path.join(GOLDEN, "exp", k + ".sadesc"), "rb") as f:
            want = f.read()
        if data != want:
            if canonical(data) == canonical(want):
                renamed.add(k)
                continue
            at = next((i for i, (a, b) in enumerate(zip(data, want)) if a != b), min(len(data), len(want)))
            bad[k] = f"differs ({len(data)} vs {len(want)} bytes, first at byte {at})"
    for c, v in cases.items():
        ks = [e for e in v["entries"] if e["key"] in bad]
        nr = sum(e["key"] in renamed for e in v["entries"])
        print(f"{c}: {len(v['entries']) - len(ks)}/{len(v['entries'])} identical"
              + (f" ({nr} up to the numbering of BASE / PARAM registers)" if nr else ""))
        for e in ks[:args.show]:
            print(f"  {e['name']} [{e['key']}]: {bad[e['key']]}")
        if len(ks) > args.show:
            print(f"  ... {len(ks) - args.show} more")
    print(f"{len(keys)} unique dispatches in {time.time() - t0:.0f} s: "
          + ("golden corpus IDENTICAL" if not bad else f"{len(bad)} DIFFER")
          + (f" ({len(renamed)} up to the numbering of BASE / PARAM registers)" if renamed else ""))
    print("golden PASS" if not bad else "golden FAIL")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", action="store_true", help="record the corpus from CASES (replaces build/golden)")
    ap.add_argument("--case", action="append", help="compare only these cases")
    ap.add_argument("-j", "--jobs", type=int, default=6)
    ap.add_argument("--show", type=int, default=10, help="differing dispatches listed per case")
    args = ap.parse_args()
    return record(args) if args.record else compare(args)


if __name__ == "__main__":
    sys.exit(main())

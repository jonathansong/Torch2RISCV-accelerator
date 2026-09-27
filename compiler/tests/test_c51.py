#!/usr/bin/env python3
"""C5.1 (docs/iree_compiler_plan.md §8.7): dispatches through the C5 pipeline
(bufferize -> sa-to-sahl -> sahl-to-sahw -> sahw-fuse-ve), the others through
the C3/C4 generator.

  1. compile (default --iree-sa-new-codegen=on) with the per-dispatch report:
     which dispatches the C5 pipeline takes, and why not the others;
  2. every dispatch: dispatch_check.py (oracle vs functional simulator);
  3. descriptor and VE counts of the C5 dispatches vs the C3/C4 generator
     (--iree-sa-new-codegen=off), a first look at the cost (the board profile
     is the measure).

    python3 compiler/tests/test_c51.py [--model build/c4/fuse]
"""
import argparse
import collections
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
COMPILER = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(COMPILER, ".."))
sys.path.insert(0, os.path.join(COMPILER, "runtime", "tools"))
import sadesc  # noqa: E402


def compile_to(model, d, flags):
    os.makedirs(d)
    for f in ("qllama.mlir", "qllama.irpa"):
        os.symlink(os.path.join(os.path.abspath(model), f), os.path.join(d, f))
    r = subprocess.run([os.path.join(COMPILER, "scripts", "compile_sa.sh"), d,
                        f"--iree-hal-dump-executable-binaries-to={os.path.join(d, 'bin')}"],
                       capture_output=True, text=True, env=dict(os.environ, SA_COMPILE_FLAGS=flags))
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    return r.stderr


def counts(path):
    _, _, exps = sadesc.read(open(path, "rb").read())
    ops = [int(r[0]) & 0xFF for r in exps[0][1]]
    return len(ops), ops.count(4)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=os.path.join(REPO, "build", "c4", "fuse"))
    args = ap.parse_args()
    work = tempfile.mkdtemp(prefix="sa_c51_")
    ok = True
    try:
        log = compile_to(args.model, os.path.join(work, "new"), "--iree-sa-codegen-report --mlir-disable-threading")
        compile_to(args.model, os.path.join(work, "old"), "--iree-sa-new-codegen=off")
        paths = {}
        reasons = collections.Counter()
        for line in log.splitlines():
            m = re.match(r"sa codegen: (\S+): (sahl|legacy)(?: \((.*)\))?", line)
            if m:
                paths[m.group(1)] = m.group(2)
                if m.group(2) == "legacy":
                    reasons[m.group(3)] += 1
        n_new = sum(1 for v in paths.values() if v == "sahl")
        print(f"C5 pipeline: {n_new} of {len(paths)} executables; the others (C3/C4 generator) because:")
        for why, c in reasons.most_common():
            print(f"  {c:3d}  {why}")
        new = os.path.join(work, "new")
        r = subprocess.run([sys.executable, os.path.join(HERE, "dispatch_check.py"), os.path.join(new, "sa_sources"),
                            os.path.join(new, "bin")], capture_output=True, text=True)
        print(f"per-dispatch differential check: {r.stdout.strip().splitlines()[-1]}")
        if r.returncode:
            print("\n".join(l for l in r.stdout.splitlines() if not l.endswith(" OK")))
            ok = False
        more = []
        for f in sorted(glob.glob(os.path.join(new, "bin", "*.sadesc"))):
            name = os.path.basename(f)
            o = os.path.join(work, "old", "bin", name)
            a, b = counts(f), counts(o)
            if a != b:
                more.append(f"  {name[:-len('_sa_desc_v1.sadesc')]}: {a[0]} descriptors / {a[1]} VE "
                            f"(C3/C4: {b[0]} / {b[1]})")
        print(f"descriptor / VE counts different from the C3/C4 generator: {len(more)}")
        print("\n".join(more))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("C5.1 PASS" if ok else "C5.1 FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

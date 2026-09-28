#!/usr/bin/env python3
"""C5 (docs/iree_compiler_plan.md §8): the C5 code generator.

  1. lit tests of the sa plugin (compiler/plugins/sa/test);
  2. an exported model (default build/c4/fuse, stories15M) compiled with the
     micro-kernels (--iree-sa-ukernels=all) and without (=none, the generic
     lowering only; dispatches it cannot compile are reported and compiled
     as faulting exports);
  3. every dispatch of both: dispatch_check.py (oracle vs functional simulator);
  4. descriptor / VE counts where the two differ (the micro-kernels' effect;
     the board profile is the measure).

    python3 compiler/tests/test_c5.py [--model build/c4/fuse] [--only all|none]
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
    return len(ops), ops.count(4), ops.count(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=os.path.join(REPO, "build", "c4", "fuse"))
    ap.add_argument("--only", choices=["all", "none"])
    args = ap.parse_args()
    build = os.environ.get("IREE_BUILD", os.path.join(REPO, "build", "iree", "build-compiler"))
    ok = True
    r = subprocess.run([os.path.join(build, "llvm-project", "bin", "llvm-lit"), "-q",
                        os.path.join(COMPILER, "plugins", "sa", "test")], capture_output=True, text=True,
                       env=dict(os.environ, IREE_BUILD=build))
    print(f"lit tests (compiler/plugins/sa/test): {'PASS' if r.returncode == 0 else 'FAIL'}")
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-2000:])
        ok = False
    work = tempfile.mkdtemp(prefix="sa_c5_")
    try:
        modes = [args.only] if args.only else ["all", "none"]
        for mode in modes:
            log = compile_to(args.model, os.path.join(work, mode),
                             f"--iree-sa-ukernels={mode} --iree-sa-codegen-report --iree-sa-allow-unsupported "
                             "--mlir-disable-threading")
            failed = collections.Counter()
            n = 0
            for line in log.splitlines():
                m = re.match(r"sa codegen: (\S+): (ok|failed)(?: \((.*)\))?", line)
                if m:
                    n += 1
                    if m.group(2) == "failed":
                        failed[m.group(3)] += 1
            print(f"--iree-sa-ukernels={mode}: {n - sum(failed.values())} of {n} executables compiled")
            for why, c in failed.most_common():
                print(f"  {c:3d} not compiled: {why}")
            ok &= not failed or mode == "none"
            d = os.path.join(work, mode)
            r = subprocess.run([sys.executable, os.path.join(HERE, "dispatch_check.py"), os.path.join(d, "sa_sources"),
                                os.path.join(d, "bin")], capture_output=True, text=True)
            last = r.stdout.strip().splitlines()[-1]
            print(f"  per-dispatch differential check: {last}")
            bad = [l for l in r.stdout.splitlines() if not l.endswith(" OK") and "UNSUPPORTED" not in l and
                   not re.match(r"^(OK|FAIL|UNSUPPORTED) ", l)]
            if [l for l in bad if "FAIL" in l or "differ" in l or "error" in l]:
                print("\n".join(bad))
                ok = False
        if len(modes) == 2:
            diff = []
            for f in sorted(glob.glob(os.path.join(work, "all", "bin", "*.sadesc"))):
                name = os.path.basename(f)
                o = os.path.join(work, "none", "bin", name)
                a, b = counts(f), counts(o)
                if a != b:
                    diff.append(f"  {name[:-len('_sa_desc_v1.sadesc')]}: all {a[0]} descriptors / {a[1]} VE / "
                                f"{a[2]} EX, none {b[0]} / {b[1]} / {b[2]}")
            print(f"executables the micro-kernels change: {len(diff)}")
            print("\n".join(diff))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("C5 PASS" if ok else "C5 FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

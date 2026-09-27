#!/usr/bin/env python3
"""C5.0 (docs/iree_compiler_plan.md §8.7): the sahw dialect path gives the same
executables as the C3/C4 path.

  1. lit tests of the sa plugin (compiler/plugins/sa/test: the sahw dialect
     parses / prints, sahw-split-head, sahw-assign-registers);
  2. an exported model (default: build/c4/fuse, stories15M) compiled twice,
     --iree-sa-codegen=templates and =dialect: every executable (sa-desc)
     byte for byte equal.

    python3 compiler/tests/test_c50.py [--model build/c4/fuse]
"""
import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
COMPILER = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(COMPILER, ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=os.path.join(REPO, "build", "c4", "fuse"),
                    help="directory with qllama.mlir / qllama.irpa (compiler/frontend/export.py)")
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
    work = tempfile.mkdtemp(prefix="sa_c50_")
    try:
        bins = {}
        for mode in ("templates", "dialect"):
            d = os.path.join(work, mode)
            os.makedirs(d)
            for f in ("qllama.mlir", "qllama.irpa"):
                os.symlink(os.path.join(os.path.abspath(args.model), f), os.path.join(d, f))
            bins[mode] = os.path.join(d, "bin")
            r = subprocess.run([os.path.join(COMPILER, "scripts", "compile_sa.sh"), d,
                                f"--iree-hal-dump-executable-binaries-to={bins[mode]}"],
                               capture_output=True, text=True,
                               env=dict(os.environ, SA_COMPILE_FLAGS=f"--iree-sa-codegen={mode}"))
            if r.returncode:
                print(f"compile ({mode}) failed:\n{r.stderr[-3000:]}")
                return 1
        names = sorted(os.listdir(bins["templates"]))
        same = [n for n in names if os.path.exists(os.path.join(bins["dialect"], n))
                and filecmp.cmp(os.path.join(bins["templates"], n), os.path.join(bins["dialect"], n), shallow=False)]
        extra = sorted(set(os.listdir(bins["dialect"])) - set(names))
        print(f"executables: {len(same)}/{len(names)} byte-identical (templates vs dialect)"
              + (f"; only in dialect: {extra}" if extra else ""))
        for n in names:
            if n not in same:
                print(f"  DIFFERENT: {n}")
        ok &= len(same) == len(names) and not extra and len(names) > 0
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("C5.0 PASS" if ok else "C5.0 FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

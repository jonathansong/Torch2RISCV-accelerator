#!/usr/bin/env python3
"""Single-op IREE modules for the armv7 board (bisects llvm-cpu codegen problems).

Each op is exported with iree-turbine, compiled for armv7 exactly like
export_and_compile.py (same flags, libm shim, import check), checked on the
host (x86 build) against torch, and written as <op>.vmfb / <op>_in.npy /
<op>_exp.npy into --out with run_op_tests.sh.

    build/iree/venv/bin/python iree-sa/l0/op_tests.py --out build/deploy_iree_l0/ops
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_and_compile import ARMV7_FLAGS, HERE, HOST_FLAGS, build_shim, check_no_imports  # noqa: E402

F = torch.nn.functional
OPS = {
    "add": lambda x: x + 1.5,
    "mul": lambda x: x * x,
    "linear": None,                                  # nn.Linear 64 -> 32 (set below)
    "exp": torch.exp,
    "amax": lambda x: x.amax(-1),
    "sum": lambda x: x.sum(-1),
    "maximum": lambda x: torch.maximum(x, torch.zeros_like(x)),
    "relu": F.relu,
    "rsqrt": lambda x: torch.rsqrt(x * x + 1.0),
    "div": lambda x: x / (x * x + 1.0),
    "sigmoid": torch.sigmoid,
    "silu": F.silu,
    "softmax": lambda x: torch.softmax(x, -1),
    "sub_max": lambda x: x - x.amax(-1, keepdim=True),
    "exp_sub_max": lambda x: torch.exp(x - x.amax(-1, keepdim=True)),
}


class Op(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="build/deploy_iree_l0/ops")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(1)
    build_shim()
    os.environ["IREE_LLD"] = os.path.join(os.path.dirname(__import__("iree.compiler").compiler.__file__),
                                          "_mlir_libs", "iree-lld")
    bindir = os.path.dirname(sys.executable)
    from iree.turbine import aot
    lin = torch.nn.Linear(64, 32)
    names = []
    for name, fn in OPS.items():
        model = Op(lin if name == "linear" else fn).eval()
        x = torch.randn(4, 64) * 2
        with torch.no_grad():
            exp = model(x).numpy()
        mlir = os.path.join(args.out, f"{name}.mlir")
        aot.export(model, x).save_mlir(mlir)
        dump = os.path.join(args.out, f"{name}_bin")
        shutil.rmtree(dump, ignore_errors=True)
        subprocess.run([os.path.join(bindir, "iree-compile"), mlir, *ARMV7_FLAGS,
                        f"--iree-hal-dump-executable-binaries-to={dump}", "-o",
                        os.path.join(args.out, f"{name}.vmfb")], check=True)
        check_no_imports(dump)
        shutil.rmtree(dump)
        os.remove(mlir)
        np.save(os.path.join(args.out, f"{name}_in.npy"), x.numpy())
        np.save(os.path.join(args.out, f"{name}_exp.npy"), exp)
        names.append(name)
    shutil.copy(os.path.join(HERE, "run_op_tests.sh"), args.out)
    print(f"{len(names)} op tests in {args.out}: {' '.join(names)}")


if __name__ == "__main__":
    main()

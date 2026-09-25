#!/usr/bin/env python3
"""L0 (docs/llm_inference_plan.md §4.4): IREE toolchain check for the PYNQ-Z1 ARM.

Exports a small torch model (2-layer MLP with RMSNorm and softmax) with
iree-turbine, compiles it with iree-compile for
  - the host (x86-64 llvm-cpu), run here with the IREE Python runtime, and
  - the board (llvm-cpu, armv7a-linux-gnueabihf, Cortex-A9 + NEON),
and writes the board bundle: model_armv7.vmfb, input.npy, expected.npy
(torch fp32 output; the host IREE result is checked against it too).

    build/iree/venv/bin/python iree-sa/l0/export_and_compile.py --out build/iree/l0
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# ARMv7 has no IEEE maxNum/minNum instruction, so LLVM emits fmaxf/fminf calls,
# which IREE's embedded ELF executables lack: embedded_ld.sh links a small
# freestanding shim (armv7_libm_shim.c) into every armv7 executable.
ARMV7_FLAGS = ["--iree-hal-target-device=local", "--iree-hal-local-target-device-backends=llvm-cpu",
               "--iree-llvmcpu-target-triple=armv7a-unknown-linux-gnueabihf",
               "--iree-llvmcpu-target-cpu=cortex-a9", "--iree-llvmcpu-target-cpu-features=+neon,+vfp3",
               f"--iree-llvmcpu-embedded-linker-path={HERE}/embedded_ld.sh"]


def build_shim():
    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    subprocess.run(["clang-18", "--target=armv7a-linux-gnueabihf", "-mcpu=cortex-a9", "-mfpu=neon",
                    "-mfloat-abi=hard", "-fPIC", "-O2", "-ffreestanding", "-fno-builtin", "-fno-exceptions",
                    "-fno-unwind-tables", "-fno-asynchronous-unwind-tables", "-c",
                    os.path.join(HERE, "armv7_libm_shim.c"), "-o", os.path.join(HERE, "build", "armv7_libm_shim.o")],
                   check=True)


HOST_FLAGS = ["--iree-hal-target-device=local", "--iree-hal-local-target-device-backends=llvm-cpu",
              "--iree-llvmcpu-target-cpu=host"]


def check_no_imports(dump):
    """Fail here (not on the board) if a linked armv7 executable imports a
    symbol (IREE's platform-agnostic ELF loader rejects any import) or still
    contains an empty libm function (the bundled musl fmaf)."""
    bad = []
    for f in sorted(os.listdir(dump)):
        if not f.endswith(".so"):
            continue
        out = subprocess.run(["llvm-readelf-18", "--dyn-syms", "-r", os.path.join(dump, f)],
                             capture_output=True, text=True, check=True).stdout
        und = sorted({ln.split()[-1] for ln in out.splitlines() if " UND " in ln and ln.split()[-1] != "UND"})
        if und:
            bad.append(f"{f}: {', '.join(und)}")
        syms = subprocess.run(["llvm-readelf-18", "-s", os.path.join(dump, f)], capture_output=True, text=True,
                              check=True).stdout
        empty = sorted({ln.split()[-1] for ln in syms.splitlines()
                        if " FUNC " in ln and ln.split()[2] == "0" and ln.split()[-1] in ("fmaf", "fmaxf", "fminf")})
        if empty:
            bad.append(f"{f}: empty libm function(s) {', '.join(empty)} (shim not linked)")
    if bad:
        sys.exit("armv7 executables import symbols:\n  " + "\n  ".join(bad))
    print(f"armv7 executables: no imports, libm shim linked ({dump})")


class Block(torch.nn.Module):
    def __init__(self, d=64, hidden=128, out=32):
        super().__init__()
        self.g = torch.nn.Parameter(torch.rand(d) + 0.5)
        self.w1 = torch.nn.Linear(d, hidden)
        self.w2 = torch.nn.Linear(hidden, out)

    def forward(self, x):
        x = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * self.g      # RMSNorm
        h = torch.nn.functional.silu(self.w1(x))
        return torch.softmax(self.w2(h), dim=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="build/iree/l0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    model = Block().eval()
    x = torch.randn(4, 64)
    with torch.no_grad():
        expected = model(x).numpy()

    from iree.turbine import aot
    exported = aot.export(model, x)
    mlir = os.path.join(args.out, "model.mlir")
    exported.save_mlir(mlir)
    bindir = os.path.dirname(sys.executable)
    build_shim()
    os.environ["IREE_LLD"] = os.path.join(os.path.dirname(__import__("iree.compiler").compiler.__file__),
                                          "_mlir_libs", "iree-lld")
    dump = os.path.join(args.out, "armv7_binaries")
    for name, flags in (("host", HOST_FLAGS), ("armv7", ARMV7_FLAGS)):
        vmfb = os.path.join(args.out, f"model_{name}.vmfb")
        extra = [f"--iree-hal-dump-executable-binaries-to={dump}"] if name == "armv7" else []
        subprocess.run([os.path.join(bindir, "iree-compile"), mlir, *flags, *extra, "-o", vmfb], check=True)
        print(f"compiled {vmfb} ({os.path.getsize(vmfb)} bytes)")
    check_no_imports(dump)

    import iree.runtime as rt
    config = rt.Config("local-task")
    with open(os.path.join(args.out, "model_host.vmfb"), "rb") as f:
        vm = rt.VmModule.copy_buffer(config.vm_instance, f.read())
    ctx = rt.SystemContext(config=config)
    ctx.add_vm_module(vm)
    got = np.asarray(ctx.modules.module["main"](x.numpy()))
    err = float(np.abs(got - expected).max())
    print(f"host IREE vs torch: max abs error {err:.2e}")
    np.save(os.path.join(args.out, "input.npy"), x.numpy())
    np.save(os.path.join(args.out, "expected.npy"), expected)
    if err > 1e-5:
        sys.exit("host result does not match torch")


if __name__ == "__main__":
    main()

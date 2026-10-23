#!/usr/bin/env python3
"""Builds the position-independent PicoRV32 MoE top-1 kernel and host twin."""

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
IREE_SRC = os.path.join(REPO, "build", "iree", "src")
BUILTINS = os.path.join(IREE_SRC, "third_party", "llvm-project", "compiler-rt", "lib", "builtins")
SOURCE = os.path.join(REPO, "firmware", "moe_top1", "moe_top1.c")
CLANG = os.environ.get("CLANG", "clang-18")
LLD = os.environ.get("LLD", "ld.lld-18")
RV = ["--target=riscv32-unknown-elf", "-march=rv32imc", "-mabi=ilp32", "-mcmodel=medany", "-mno-relax",
      "-O2", "-ffreestanding", "-fno-builtin", "-ffunction-sections", "-fdata-sections", "-fno-jump-tables",
      "-fno-asynchronous-unwind-tables", "-fno-unwind-tables"]


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"failed: {' '.join(command)}\n{result.stdout[-2000:]}{result.stderr[-2000:]}")
    return result.stdout


def build(out):
    out = os.path.abspath(out)
    build_dir = os.path.join(out, "build")
    os.makedirs(build_dir, exist_ok=True)
    kernel = os.path.join(build_dir, "moe_top1.o")
    compare = os.path.join(build_dir, "comparesf2.o")
    run([CLANG] + RV + ["-c", SOURCE, "-o", kernel])
    run([CLANG] + RV + ["-w", f"-I{BUILTINS}", "-c", os.path.join(BUILTINS, "comparesf2.c"), "-o", compare])

    def link(base, tag):
        script = os.path.join(build_dir, f"moe_top1_{tag}.ld")
        with open(script, "w") as f:
            f.write(f"""ENTRY(r5_entry)
SECTIONS {{
  . = {base:#x};
  .text : {{ *(.text.r5_entry) *(.text .text.*) *(.rodata .rodata.* .srodata .srodata.*) }}
  .data : {{ *(.data .data.* .sdata .sdata.* .bss .bss.* .sbss .sbss.* COMMON) }}
  /DISCARD/ : {{ *(.comment) *(.riscv.attributes) *(.note*) *(.eh_frame*) }}
}}
""")
        elf = os.path.join(build_dir, f"moe_top1_{tag}.elf")
        run([LLD, "-m", "elf32lriscv", "--no-relax", "--gc-sections", "-T", script,
             "-Map", os.path.join(build_dir, f"moe_top1_{tag}.map"), "-o", elf, kernel, compare])
        symbols = run(["llvm-nm-18", "-S", "--size-sort", elf])
        if not re.search(rf"^{base:08x}(?: [0-9a-f]+)? T r5_entry$", symbols, re.M):
            raise RuntimeError("r5_entry is not at the start of the blob")
        binary = os.path.join(build_dir, f"moe_top1_{tag}.bin")
        run(["llvm-objcopy-18", "-O", "binary", "--only-section=.text", elf, binary])
        with open(binary, "rb") as f:
            return f.read(), symbols

    code, symbols = link(0, "zero")
    moved, _ = link(0x12340, "moved")
    if moved != code:
        raise RuntimeError("MoE top-1 kernel is not position independent")
    with open(os.path.join(out, "moe_top1.bin"), "wb") as f:
        f.write(code)
    with open(os.path.join(out, "moe_top1.map"), "w") as f:
        f.write(symbols)
    host = os.path.join(out, "moe_top1_host.so")
    run([CLANG, "-O2", "-fPIC", "-shared", "-DMOE_TOP1_HOST", SOURCE, "-o", host])
    with open(os.path.join(out, "moe_top1.txt"), "w") as f:
        f.write(f"r5_entry {len(code)} sa-moe-route-v1\n")
    return code, host


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    code, _ = build(args.out)
    print(f"moe top-1: {len(code)} bytes of position-independent rv32 code and host twin -> {args.out}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        sys.exit(str(error))

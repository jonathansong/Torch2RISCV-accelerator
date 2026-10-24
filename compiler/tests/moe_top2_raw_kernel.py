#!/usr/bin/env python3
"""Build the position-independent raw Top-2 PicoRV32 kernel and its host twin."""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import moe_top1_kernel as common  # noqa: E402

SOURCE = os.path.join(REPO, "firmware", "moe_top2_raw", "moe_top2_raw.c")


def build(out):
    out = os.path.abspath(out)
    obj_dir = os.path.join(out, "build")
    os.makedirs(obj_dir, exist_ok=True)
    kernel = os.path.join(obj_dir, "moe_top2_raw.o")
    compare = os.path.join(obj_dir, "comparesf2.o")
    common.run([common.CLANG] + common.RV + ["-c", SOURCE, "-o", kernel])
    common.run([common.CLANG] + common.RV + ["-w", f"-I{common.BUILTINS}", "-c",
                os.path.join(common.BUILTINS, "comparesf2.c"), "-o", compare])

    def link(base, tag):
        script = os.path.join(obj_dir, f"moe_top2_raw_{tag}.ld")
        with open(script, "w") as f:
            f.write(f"""ENTRY(r5_entry)
SECTIONS {{
  . = {base:#x};
  .text : {{ *(.text.r5_entry) *(.text .text.*) *(.rodata .rodata.* .srodata .srodata.*) }}
  .data : {{ *(.data .data.* .sdata .sdata.* .bss .bss.* .sbss .sbss.* COMMON) }}
  /DISCARD/ : {{ *(.comment) *(.riscv.attributes) *(.note*) *(.eh_frame*) }}
}}
""")
        elf = os.path.join(obj_dir, f"moe_top2_raw_{tag}.elf")
        common.run([common.LLD, "-m", "elf32lriscv", "--no-relax", "--gc-sections", "-T", script,
                    "-o", elf, kernel, compare])
        symbols = common.run(["llvm-nm-18", "-S", "--size-sort", elf])
        if not re.search(rf"^{base:08x}(?: [0-9a-f]+)? T r5_entry$", symbols, re.M):
            raise RuntimeError("r5_entry is not at the start of the blob")
        binary = os.path.join(obj_dir, f"moe_top2_raw_{tag}.bin")
        common.run(["llvm-objcopy-18", "-O", "binary", "--only-section=.text", elf, binary])
        with open(binary, "rb") as f:
            return f.read()

    code = link(0, "zero")
    if link(0x12340, "moved") != code:
        raise RuntimeError("raw Top-2 kernel is not position independent")
    with open(os.path.join(out, "moe_top2_raw.bin"), "wb") as f:
        f.write(code)
    host = os.path.join(out, "moe_top2_raw_host.so")
    common.run([common.CLANG, "-O2", "-fPIC", "-shared", "-DMOE_TOP2_RAW_HOST", SOURCE, "-o", host])
    return code, host


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    code, _ = build(args.out)
    print(f"raw top-2: {len(code)} bytes of position-independent rv32 code -> {args.out}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        sys.exit(str(error))

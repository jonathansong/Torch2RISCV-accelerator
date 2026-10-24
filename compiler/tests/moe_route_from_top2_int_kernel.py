#!/usr/bin/env python3
"""Build an integer-only position-independent MoE route bridge and host twin."""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import moe_top1_kernel as common  # noqa: E402

SOURCE = os.path.join(REPO, "firmware", "moe_route_from_top2_int", "moe_route_from_top2_int.c")


def build(out):
    out = os.path.abspath(out)
    obj_dir = os.path.join(out, "build")
    os.makedirs(obj_dir, exist_ok=True)
    kernel = os.path.join(obj_dir, "moe_route_int.o")
    common.run([common.CLANG] + common.RV + ["-c", SOURCE, "-o", kernel])

    def link(base, tag):
        script = os.path.join(obj_dir, f"moe_route_int_{tag}.ld")
        with open(script, "w") as f:
            f.write(f"""ENTRY(r5_entry)
SECTIONS {{
  . = {base:#x};
  .text : {{ *(.text.r5_entry) *(.text .text.*) *(.rodata .rodata.* .srodata .srodata.*) }}
  .data : {{ *(.data .data.* .sdata .sdata.* .bss .bss.* .sbss .sbss.* COMMON) }}
  /DISCARD/ : {{ *(.comment) *(.riscv.attributes) *(.note*) *(.eh_frame*) }}
}}
""")
        elf = os.path.join(obj_dir, f"moe_route_int_{tag}.elf")
        common.run([common.LLD, "-m", "elf32lriscv", "--no-relax", "--gc-sections", "-T",
                    script, "-o", elf, kernel])
        symbols = common.run(["llvm-nm-18", "-S", "--size-sort", elf])
        if not re.search(rf"^{base:08x}(?: [0-9a-f]+)? T r5_entry$", symbols, re.M):
            raise RuntimeError("r5_entry is not at the start of the blob")
        binary = os.path.join(obj_dir, f"moe_route_int_{tag}.bin")
        common.run(["llvm-objcopy-18", "-O", "binary", "--only-section=.text", elf, binary])
        with open(binary, "rb") as f:
            return f.read()

    code = link(0, "zero")
    if link(0x12340, "moved") != code:
        raise RuntimeError("integer route bridge is not position independent")
    with open(os.path.join(out, "moe_route_from_top2_int.bin"), "wb") as f:
        f.write(code)
    host = os.path.join(out, "moe_route_from_top2_int_host.so")
    common.run([common.CLANG, "-O2", "-fPIC", "-shared", "-DMOE_ROUTE_INT_HOST",
                SOURCE, "-o", host])
    return code, host


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        code, _ = build(args.out)
        print(f"integer route bridge: {len(code)} position-independent rv32 bytes -> {args.out}")
    except RuntimeError as error:
        sys.exit(str(error))

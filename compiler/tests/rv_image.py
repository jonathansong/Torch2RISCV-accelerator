#!/usr/bin/env python3
"""A model's generated RISC-V code as one image (docs/riscv_backend_design.md
§9.4, plan A S2): every export's LLVM IR (iree-compile --iree-sa-emit-riscv)
and the runtime (firmware/rv/sa_rv_rt.c) compiled for rv32imc position
independent (medany: PC-relative addressing, no jump tables, no linker
relaxation) and linked at 0 into a flat binary the driver can load anywhere
in the device window; plus the same code for the host's sim transport (the
commands submitted to the functional simulator one by one, SA_RV_SIM).

    compiler/tests/rv_image.py <dir of .ll> --out <dir> [--jobs 4]
writes <out>/rvcode.bin, rvcode.map ("offset symbol" per export), rvcode_host.so.
"""
import argparse
import concurrent.futures as cf
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FW = os.path.join(REPO, "firmware")
CLANG = os.environ.get("CLANG", "clang-18")
RV = ["--target=riscv32-unknown-elf", "-march=rv32imc", "-mabi=ilp32", "-O2", "-ffreestanding", "-fno-builtin",
      "-mcmodel=medany", "-fno-jump-tables", "-mno-relax", "-fno-asynchronous-unwind-tables", "-fno-unwind-tables", "-Wno-override-module", f"-I{FW}/include"]
HOST = ["-O1", "-fPIC", "-Wno-override-module", "-DSA_RV_SIM", f"-I{FW}/include"]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{' '.join(cmd[:4])} ...: {r.stderr[-1500:]}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lldir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    lls = sorted(glob.glob(os.path.join(args.lldir, "*.ll")))
    if not lls:
        raise SystemExit(f"no .ll in {args.lldir}")
    obj = os.path.join(args.out, "obj")
    os.makedirs(obj, exist_ok=True)
    rt = os.path.join(FW, "rv", "sa_rv_rt.c")
    jobs = []
    for src in lls + [rt]:
        b = os.path.splitext(os.path.basename(src))[0]
        jobs.append([CLANG, *RV, "-c", src, "-o", os.path.join(obj, b + ".rv.o")])
        jobs.append([CLANG, *HOST, "-c", src, "-o", os.path.join(obj, b + ".host.o")])
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(run, jobs))
    rvobjs = sorted(glob.glob(os.path.join(obj, "*.rv.o")))
    # position independence: no absolute address relocations
    bad = [l for o in rvobjs for l in run(["llvm-readelf-18", "-r", o]).splitlines()
           if " R_RISCV_HI20 " in l or " R_RISCV_LO12" in l or " R_RISCV_32 " in l]
    if bad:
        raise SystemExit(f"absolute relocations in the RISC-V code: {bad[:3]}")
    elf = os.path.join(args.out, "rvcode.elf")
    lds = os.path.join(obj, "rvcode.ld")
    open(lds, "w").write("SECTIONS { . = 0; .text : { *(.text .text.*) } .rodata : { *(.rodata .rodata.* .srodata "
                         ".srodata.*) } .data : { *(.data .data.* .sdata .sdata.*) } .bss : { *(.bss .bss.* .sbss "
                         ".sbss.* COMMON) } /DISCARD/ : { *(.eh_frame .eh_frame_hdr .comment) } }\n")
    run(["ld.lld-18", "--no-relax", "-T", lds, "-e", "0", "-o", elf, *rvobjs])
    if run(["llvm-readelf-18", "-S", elf]).count(".bss") and \
            int(run(["llvm-size-18", "-A", elf]).split(".bss")[1].split()[0]) > 0:
        raise SystemExit("the RISC-V code has zero-initialized data (.bss): not in the flat image")
    binf = os.path.join(args.out, "rvcode.bin")
    run(["llvm-objcopy-18", "-O", "binary", elf, binf])
    names = {os.path.splitext(os.path.basename(x))[0] for x in lls}
    with open(os.path.join(args.out, "rvcode.map"), "w") as m:
        for line in run(["llvm-nm-18", "--defined-only", elf]).splitlines():
            a, t, sym = line.split()
            if t in "Tt" and sym in names:
                m.write(f"{int(a, 16):x} {sym}\n")
    run([CLANG, "-shared", "-o", os.path.join(args.out, "rvcode_host.so"),
         *sorted(glob.glob(os.path.join(obj, "*.host.o")))])
    print(f"{len(lls)} exports: rvcode.bin {os.path.getsize(binf) / 1024:.1f} KB (rv32imc, position independent), "
          f"rvcode_host.so -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

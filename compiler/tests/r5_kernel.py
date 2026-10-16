#!/usr/bin/env python3
"""R5 of the RISC-V backend (docs/riscv_backend_design.md §5, model C): a
scalar kernel for the PicoRV32 compiled by IREE's llvm-cpu backend. The kernel
is the decode loop's block argmax (plan B, firmware/include/rvsched.h): the
index of the first largest of n f32 block maxima, written as linalg, compiled
by iree-compile for riscv32 (rv32imc, ilp32, soft float, a static library
object), linked with the ABI shim (firmware/r5/r5_shim.c) and the soft-float
routines it calls (compiler-rt builtins) into one position-independent blob
with the entry r5_entry at offset 0 (checked: linked at two addresses, the
same bytes). The same kernel compiled by llvm-cpu for the host, with the shim,
is the sim's build (r5_host.so); it is checked against numpy here.

    compiler/tests/r5_kernel.py --vocab 151936 --out <dir>      (n: argmax_module.py's default block)
    compiler/tests/r5_kernel.py --n 1187 --out <dir>
Writes <dir>/r5.bin, r5.txt ("<symbol> <n> <bytes>"), r5.map, r5_host.so;
the runtime takes the directory as SA_R5 (sa-llm-run --rvsched, sa_rvsched.c).
"""
import argparse
import ctypes
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "compiler", "frontend"))
import argmax_module  # noqa: E402

IREE_SRC = os.path.join(REPO, "build", "iree", "src")
IREE_COMPILE = os.path.join(REPO, "build", "iree", "build-compiler", "tools", "iree-compile")
BUILTINS = os.path.join(IREE_SRC, "third_party", "llvm-project", "compiler-rt", "lib", "builtins")
CLANG = os.environ.get("CLANG", "clang-18")
LLD = os.environ.get("LLD", "ld.lld-18")
RV = ["--target=riscv32-unknown-elf", "-march=rv32imc", "-mabi=ilp32", "-mcmodel=medany", "-mno-relax", "-O2",
      "-ffreestanding", "-fno-builtin", "-ffunction-sections", "-fdata-sections", "-fno-jump-tables",
      "-fno-asynchronous-unwind-tables", "-fno-unwind-tables"]
# the soft-float routines an f32 kernel may call (the linker keeps the ones it does)
SOFT_FLOAT = ["comparesf2", "addsf3", "subsf3", "mulsf3", "divsf3", "negsf2", "floatsisf", "floatunsisf", "fixsfsi",
              "fixunssfsi", "extendsfdf2", "truncdfsf2", "comparedf2", "adddf3", "subdf3", "muldf3"]
LLVMCPU = ["--iree-hal-target-device=local", "--iree-hal-local-target-device-backends=llvm-cpu",
           "--iree-llvmcpu-link-embedded=false", "--iree-llvmcpu-link-static", "--iree-llvmcpu-debug-symbols=false"]
RV_TARGET = ["--iree-llvmcpu-target-triple=riscv32-unknown-elf", "--iree-llvmcpu-target-cpu=generic-rv32",
             "--iree-llvmcpu-target-cpu-features=+m,+c", "--iree-llvmcpu-target-abi=ilp32"]
HOST_TARGET = ["--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu", "--iree-llvmcpu-target-cpu=generic"]


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        sys.exit(f"failed: {' '.join(cmd)}\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    return r.stdout


def mlir(n):
    """the index of the first largest of n f32 (a NaN-free input: ogt keeps the first of equal values)"""
    return f"""func.func @r5_argmax(%x: tensor<{n}xf32>) -> tensor<i32> {{
  %ninf = arith.constant 0xFF800000 : f32
  %z = arith.constant 0 : i32
  %v0 = tensor.empty() : tensor<f32>
  %i0 = tensor.empty() : tensor<i32>
  %v = linalg.fill ins(%ninf : f32) outs(%v0 : tensor<f32>) -> tensor<f32>
  %i = linalg.fill ins(%z : i32) outs(%i0 : tensor<i32>) -> tensor<i32>
  %r:2 = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}} ins(%x : tensor<{n}xf32>) outs(%v, %i : tensor<f32>, tensor<i32>) {{
  ^bb0(%a: f32, %m: f32, %k: i32):
    %idx = linalg.index 0 : index
    %ii = arith.index_cast %idx : index to i32
    %gt = arith.cmpf ogt, %a, %m : f32
    %nm = arith.select %gt, %a, %m : f32
    %nk = arith.select %gt, %ii, %k : i32
    linalg.yield %nm, %nk : f32, i32
  }} -> (tensor<f32>, tensor<i32>)
  return %r#1 : tensor<i32>
}}
"""


def export_info(src):
    """the one export of the compiled executable: its function, bindings, push constants, workgroup count and size"""
    ir = run([IREE_COMPILE, src] + LLVMCPU + RV_TARGET + ["--compile-to=executable-targets"])
    exports = re.findall(r"hal\.executable\.export public @(\w+) ordinal\(\d+\) layout\((#hal\.pipeline\.layout<.*?>)\)"
                         r" count\([^)]*\) -> \(index, index, index\) \{(.*?)\} attributes \{workgroup_size = \[(.*?)\]",
                         ir, re.S)
    if len(exports) != 1:
        sys.exit(f"expected one export, found {len(exports)} (the kernel must fuse into one dispatch)")
    name, layout, body, size = exports[0]
    consts = {m.group(1): int(m.group(2)) for m in re.finditer(r"(%c\w*) = arith\.constant (\d+) : index", body)}
    ret = re.search(r"hal\.return (.*?) :", body).group(1).split(", ")
    if not all(v in consts for v in ret):
        sys.exit(f"{name}: a dynamic workgroup count ({body.strip()})")
    m = re.search(r"constants = (\d+)", layout)
    return {"symbol": name, "bindings": layout.count("#hal.pipeline.binding<"), "constants": int(m.group(1)) if m else 0,
            "count": [consts[v] for v in ret], "size": [int(x.split(":")[0]) for x in size.split(",")]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vocab", type=int, help="the logits' size: n = vocab / argmax_module.py's default block")
    ap.add_argument("--n", type=int, help="the number of block maxima")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = args.n or (args.vocab // argmax_module.blocks(args.vocab)[0] if args.vocab else 0)
    if not n:
        sys.exit("--vocab or --n")
    out = os.path.abspath(args.out)
    b = os.path.join(out, "build")
    os.makedirs(b, exist_ok=True)
    src = os.path.join(b, "r5_argmax.mlir")
    open(src, "w").write(mlir(n))
    # 1. the kernel by IREE's llvm-cpu: riscv32 and the host
    info = export_info(src)
    for tag, target in (("rv32", RV_TARGET), ("host", HOST_TARGET)):
        run([IREE_COMPILE, src] + LLVMCPU + target + [f"--iree-llvmcpu-static-library-output-path={b}/kernel_{tag}.o",
                                                      "-o", f"{b}/kernel_{tag}.vmfb"])
        # (the export's function is local in the static library; the library query's tables hold absolute
        # pointers, so the shim calls the function itself)
        run(["llvm-objcopy-18", f"--globalize-symbol={info['symbol']}", f"{b}/kernel_{tag}.o"])
    # 2. the shim, the soft-float routines
    inc = os.path.join(b, "inc")
    os.makedirs(inc, exist_ok=True)
    open(os.path.join(inc, "assert.h"), "w").write("#define static_assert _Static_assert\n")
    defs = [f"-DR5_KERNEL={info['symbol']}", f"-DR5_BINDINGS={info['bindings']}", f"-DR5_CONSTANTS={info['constants']}"]
    defs += [f"-DR5_COUNT_{a}={v}" for a, v in zip("XYZ", info["count"])]
    defs += [f"-DR5_SIZE_{a}={v}" for a, v in zip("XYZ", info["size"])]
    shim = os.path.join(REPO, "firmware", "r5", "r5_shim.c")
    incs = [f"-I{inc}", f"-I{os.path.join(IREE_SRC, 'runtime', 'src')}"]
    run([CLANG] + RV + incs + defs + ["-c", shim, "-o", f"{b}/shim_rv32.o"])
    objs = [f"{b}/shim_rv32.o", f"{b}/kernel_rv32.o"]
    for f in SOFT_FLOAT:
        o = f"{b}/{f}.o"
        run([CLANG] + RV + ["-w", f"-I{BUILTINS}", "-c", os.path.join(BUILTINS, f + ".c"), "-o", o])
        objs.append(o)
    # 3. one blob: r5_entry first, code and read-only data only, position independent
    def link(base, tag):
        ld = f"{b}/r5_{tag}.ld"
        open(ld, "w").write(f"""ENTRY(r5_entry)
SECTIONS {{
  . = {base:#x};
  .text : {{ *(.text.r5_entry) *(.text .text.*) *(.rodata .rodata.* .srodata .srodata.*) }}
  .data : {{ *(.data .data.* .sdata .sdata.* .bss .bss.* .sbss .sbss.* COMMON) }}
  /DISCARD/ : {{ *(.comment) *(.riscv.attributes) *(.note*) *(.eh_frame*) }}
}}
""")
        elf = f"{b}/r5_{tag}.elf"
        run([LLD, "-m", "elf32lriscv", "--no-relax", "--gc-sections", "-T", ld, "-Map", f"{b}/r5_{tag}.map", "-o", elf]
            + objs)
        heads = run(["llvm-readelf-18", "-S", elf])
        m = re.search(r"\.data\s+\w+\s+\w+\s+\w+\s+(\w+)", heads)
        if m and int(m.group(1), 16):
            sys.exit(f"{elf}: writable data ({int(m.group(1), 16)} bytes): the blob must be code and constants only")
        syms = run(["llvm-nm-18", elf])
        if not re.search(rf"^{base:08x} T r5_entry$", syms, re.M):
            sys.exit(f"{elf}: r5_entry is not at the start")
        binf = f"{b}/r5_{tag}.bin"
        run(["llvm-objcopy-18", "-O", "binary", "--only-section=.text", elf, binf])
        return open(binf, "rb").read()
    code = link(0, "0")
    if link(0x12340, "moved") != code:
        sys.exit("the blob is not position independent (different bytes when linked at another address)")
    open(os.path.join(out, "r5.bin"), "wb").write(code)
    syms = run(["llvm-nm-18", "-S", "--size-sort", f"{b}/r5_0.elf"])
    open(os.path.join(out, "r5.map"), "w").write(syms)
    open(os.path.join(out, "r5.txt"), "w").write(f"{info['symbol']} {n} {len(code)}\n")
    # 4. the host build (the sim), checked against numpy
    so = os.path.join(out, "r5_host.so")
    run([CLANG, "-O1", "-fPIC", "-shared", "-DR5_HOST"] + incs + defs + [shim, f"{b}/kernel_host.o", "-o", so])
    lib = ctypes.CDLL(so)
    keep = {}
    PTR = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)
    ptr = PTR(lambda a: keep[a].ctypes.data)
    lib.r5_host_init(ptr)
    rng = np.random.default_rng(0)
    cases = [rng.standard_normal(n).astype(np.float32) for _ in range(20)]
    tie = rng.standard_normal(n).astype(np.float32)
    tie[n // 3] = tie[2 * n // 3] = tie.max() + 1
    cases += [tie, np.full(n, -3.0, np.float32), -np.abs(rng.standard_normal(n)).astype(np.float32)]
    for x in cases:
        res = np.array([-1], np.int32)
        keep.clear()
        keep[0x1000] = x
        keep[0x2000] = res
        argb = (ctypes.c_uint32 * 64)(0x1000, 0x2000)
        lib.r5_entry(argb, ctypes.byref(argb, 64))
        if res[0] != int(np.argmax(x)):
            sys.exit(f"host build: {res[0]}, numpy {int(np.argmax(x))}")
    calls = sorted(set(re.findall(r" [Tt] (__\w+)$", syms, re.M)))      # (the soft-float routines kept)
    print(f"r5: {info['symbol']} (n = {n}, {info['bindings']} bindings, workgroups {info['count']}) by llvm-cpu: "
          f"{len(code)} bytes of rv32 with the shim and soft float ({', '.join(calls) or 'none'}), position independent; "
          f"the host build = numpy on {len(cases)} inputs -> {out}")


if __name__ == "__main__":
    main()

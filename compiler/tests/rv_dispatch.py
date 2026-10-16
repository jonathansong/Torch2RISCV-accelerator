#!/usr/bin/env python3
"""One compiled dispatch as a PicoRV32 program (docs/riscv_backend_design.md
§5, R2 / R3 first form): the dispatch is laid out in the system testbench's
DDR window as dispatch_check.py lays it out for the functional simulator
(the export's template, its bindings with generated inputs, the driver's
list: SETREG of the registers, CALL, END), run on the functional simulator
(checked against the oracle, the executed commands traced), and written as

  rvb_ddr.hex     the DDR image before the run ($readmemh, @word addresses)
  rvb_expect.hex  the DDR image after the run (the bindings' pages)
  rvb_cmp.txt     the byte ranges to compare: one "offset length" a line
  rvb_args.h      the list's offset in the window (the descriptor path: mat_submit)
  rvb_gen.c       the same commands issued over PCPI (model A): one
                  sysarray_intrinsics.h call per command, the configuration
                  keys written only when their value changes; DDR addresses
                  relative to the window's base W (an argument: the board's
                  buffer is wherever it is allocated)
  rvb_list.json   the list's contents for the board (rebuilt there at the
                  buffer's address: the registers, BASE values relative to W)
  rvb_ddr.bin / rvb_expect.bin   the images as raw bytes (the board)

This first form is a straight-line program from the trace (the dynamic
fields, loops and register setup resolved for the given T): bit-identical
by construction of the commands, and the measure of the issue cost per
command; the structural emitter (loops, registers kept) is step R2.

    compiler/tests/rv_dispatch.py <dispatch source .mlir> --t 16 --out build/rvb/<name>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "llm"))
sys.path.insert(0, os.path.join(REPO, "compiler", "runtime", "tools"))
import dispatch_check as DC  # noqa: E402
import golden as G  # noqa: E402
import sadesc  # noqa: E402
from sa_funcsim import SaFuncSim  # noqa: E402

DDR_BASE = 0x18000000                 # firmware/sim/tb_system.v
OP_LD, OP_ST, OP_EX, OP_VE, OP_FENCE = 0x01, 0x02, 0x03, 0x04, 0x10
MODE_GEMV = 2


def f(x, lo, n):
    return (x >> lo) & ((1 << n) - 1)


def layout(src, blob, d, t, seed=1):
    """dispatch_check.check's layout at DDR_BASE: (sim, phys, bufs, regions, list address, size)"""
    keep = DC.O.load_dispatch(src)
    func = keep[2]
    _, _, exps = sadesc.read(blob)
    name, rows, nb, nc, cyc, setup = exps[0]
    ext = sadesc.read_ext(blob)[0]
    consts = DC.push_constants(func, t)
    regions = DC.binding_regions(func, DC.O.Interp({}, consts, d))
    rng = np.random.default_rng(seed)
    bufs = {}
    for b, regs in regions.items():
        size = max(o + n for o, n, _, _ in regs)
        buf = bytearray(-(-size // 64) * 64 + 64)
        for o, n, et, w in regs:
            buf[o:o + n] = DC.fill(rng, et, n // DC.ELEM_BYTES[et], t)
        bufs[b] = buf
    want = {b: bytearray(v) for b, v in bufs.items()}
    DC.O.run_dispatch(func, want, consts, d)
    tsize = -(-max(rows.nbytes, 0x1000) // 0x1000) * 0x1000
    total = tsize + sum(-(-len(b) // 4096) * 4096 for b in bufs.values()) + 0x10000
    size = -(-total // 0x10000) * 0x10000
    cfg = DC.target_config(src)
    sim = SaFuncSim(d, DDR_BASE, size, spad_bytes=cfg.get("spad_bytes", 131072), acc_bytes=cfg.get("acc_bytes", 262144))
    sim.ddr_write(DDR_BASE, rows.tobytes())
    phys, a = [0] * max(nb, max(bufs, default=0) + 1), DDR_BASE + tsize
    for b in sorted(bufs):
        phys[b] = a
        sim.ddr_write(phys[b], bytes(bufs[b]))
        a += -(-len(bufs[b]) // 4096) * 4096
    lst = sadesc.dispatch_list(DDR_BASE, phys, consts, setup, ext)
    lst_addr = DDR_BASE + size - 0x10000
    sim.ddr_write(lst_addr, lst.array().tobytes())
    regs = sadesc.setup_values(setup, phys, consts)
    if ext.get("prefix", 0):
        regs.append((15, dict(regs)[ext["prefix_reg"]]))
    lst_json = {"list_off": lst_addr - DDR_BASE, "end": 0x5A,
                # (a BASE in the window relative to it, flagged; anything else as it is)
                "regs": [[int(i), int(v) - DDR_BASE, True] if int(i) < 16 and DDR_BASE <= int(v) < DDR_BASE + size
                         else [int(i), int(v), False] for i, v in regs],
                "calls": ([0] if ext.get("prefix", 0) else []) + ([64 * ext.get("prefix", 0)] if ext.get("head", 0) else [])
                         + [64 * (ext.get("prefix", 0) + ext.get("head", 0))]}
    return sim, phys, bufs, want, regions, lst_addr, size, name, lst_json


class Emitter:
    """C statements for the traced commands, the PCPI configuration tracked"""

    def __init__(self, base):
        self.base = base
        self.lines, self.cfg, self.vcfg, self.gcfg = [], {}, {}, {}
        self.counts = {"LD": 0, "GEMV": 0, "ST": 0, "EX": 0, "VE": 0, "FENCE": 0}
        self.insns = 0

    def _set(self, table, fn, key, name, v):
        if table.get(key) != v:
            table[key] = v
            self.lines.append(f"    {fn}({name}, 0x{v:x}u);")
            self.insns += 1

    def a(self, ddr):
        return f"W + 0x{ddr - self.base:x}u"

    def cmd(self, r):
        hdr, ddr = int(r[0]), int(r[1])
        w = [0] + [int(x) for x in r[3:10]]
        op = hdr & 0xFF
        if op != OP_FENCE and (hdr >> 11) & 1:                 # FENCE_BEFORE
            self.lines.append("    mat_fence(0);")
            self.insns += 1
            self.counts["FENCE"] += 1
        if op == OP_LD and f(w[3], 32, 2) == MODE_GEMV:
            self.counts["GEMV"] += 1
            for k, n, v in ((0, "SA_GCFG_KBYTES", f(w[2], 48, 16)), (1, "SA_GCFG_PITCH", f(w[3], 0, 32)),
                            (2, "SA_GCFG_XWORD", f(w[3], 34, 16)), (3, "SA_GCFG_CSTEP", f(w[3], 50, 8))):
                self._set(self.gcfg, "gemv_cfg", k, n, v)
            laddr = f(w[2], 0, 32)
            assert laddr >> 28 == 3, "GEMV result not in ACC"
            rs2 = f(w[3], 58, 1) << 28 | (f(w[2], 32, 16) - 1) << 16 | (laddr & 0xFFFF)
            self.lines.append(f"    gemv_run({self.a(ddr)}, 0x{rs2 & 0xFFFF:x}u, {f(w[2], 32, 16)}u, {f(w[3], 58, 1)}u);")
        elif op == OP_LD:
            self.counts["LD"] += 1
            for k, n, v in ((0, "SA_CFG_LD_ROWS", f(w[2], 32, 16)), (1, "SA_CFG_LD_ROW_BYTES", f(w[2], 48, 16)),
                            (2, "SA_CFG_LD_PITCH", f(w[3], 0, 32)), (3, "SA_CFG_LD_MODE", f(w[3], 32, 2))):
                self._set(self.cfg, "mat_cfg", k, n, v)
            self.lines.append(f"    mat_load({self.a(ddr)}, 0x{f(w[2], 0, 32):x}u);")
        elif op == OP_ST:
            self.counts["ST"] += 1
            for k, n, v in ((4, "SA_CFG_ST_ROWS", f(w[2], 32, 16)), (5, "SA_CFG_ST_ROW_BYTES", f(w[2], 48, 16)),
                            (6, "SA_CFG_ST_PITCH", f(w[3], 0, 32))):
                self._set(self.cfg, "mat_cfg", k, n, v)
            self.lines.append(f"    mat_store({self.a(ddr)}, 0x{f(w[2], 0, 32):x}u);")
        elif op == OP_EX:
            self.counts["EX"] += 1
            for k, n, v in ((7, "SA_CFG_EX_REPEAT", f(w[2], 0, 12)), (8, "SA_CFG_EX_B_STEP", f(w[2], 16, 16)),
                            (9, "SA_CFG_EX_C_STEP", f(w[2], 32, 16)), (10, "SA_CFG_EX_C_ROW", f(w[2], 48, 16))):
                self._set(self.cfg, "mat_cfg", k, n, v)
            self.lines.append(f"    mat_exec(0x{f(w[1], 0, 16):x}u, 0x{f(w[1], 16, 16):x}u, 0x{f(w[1], 32, 16):x}u, "
                              f"{f(w[1], 48, 12)}u, {f(w[1], 60, 1)}u);")
        elif op == OP_VE:
            self.counts["VE"] += 1
            for k, n, v in ((0, "SA_VCFG_OP", f(w[3], 0, 8)), (1, "SA_VCFG_LEN", f(w[2], 32, 32)),
                            (2, "SA_VCFG_DST", f(w[2], 0, 32)), (3, "SA_VCFG_TYPES", f(w[3], 8, 6)),
                            (4, "SA_VCFG_SRC2_MOD", f(w[3], 16, 16)), (5, "SA_VCFG_SCALE", f(w[3], 32, 16)),
                            (6, "SA_VCFG_SHIFT", f(w[3], 48, 5)), (7, "SA_VCFG_ZP", f(w[4], 0, 32)),
                            (8, "SA_VCFG_CLAMP_LO", f(w[4], 32, 32)), (9, "SA_VCFG_CLAMP_HI", f(w[5], 0, 32)),
                            (10, "SA_VCFG_FLAGS", f(w[3], 53, 11)), (11, "SA_VCFG_IMM", f(w[5], 32, 32)),
                            (12, "SA_VCFG_A", f(w[6], 0, 32)), (13, "SA_VCFG_B", f(w[6], 32, 32)),
                            (14, "SA_VCFG_ROW", f(w[7], 0, 32)), (15, "SA_VCFG_P1S", f(w[7], 32, 32))):
                self._set(self.vcfg, "vec_cfg", k, n, v)
            self.lines.append(f"    vec_run(0x{f(w[1], 0, 32):x}u, 0x{f(w[1], 32, 32):x}u);")
        elif op == OP_FENCE:
            self.counts["FENCE"] += 1
            self.lines.append(f"    mat_fence({f(w[1], 0, 4)}u);")
        else:
            raise SystemExit(f"command {op:#x} in the trace")
        self.insns += 1


def hexfile(path, base, data):
    """$readmemh of 32-bit words: @word-address lines (relative to DDR_BASE / 4)"""
    with open(path, "w") as fh:
        a = np.frombuffer(bytes(data) + b"\0" * (-len(data) % 4), "<u4")
        fh.write(f"@{(base - DDR_BASE) // 4:x}\n")
        for i in range(0, len(a), 8):
            fh.write(" ".join(f"{x:08x}" for x in a[i:i + 8]) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source")
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--code", choices=["trace", "compiled", "embedded"], default="trace",
                    help="rvb_gen: the traced commands as straight-line C (R2's first form), or a call of the "
                         "export's compiled function (sahw -> LLVM -> riscv32, riscv_backend_design.md §9: "
                         "rvb_gen.ll, linked with firmware/rv/sa_rv_rt.c), or the rv32 code iree-compile "
                         "embedded in the sa-desc file (--iree-sa-riscv, S4: rvb_code.bin, included as it is)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    text = G.sa_only(open(args.source).read())
    src = os.path.join(args.out, "dispatch.mlir")
    open(src, "w").write(text)
    for old_ll in glob.glob(os.path.join(args.out, "*.ll")):
        os.remove(old_ll)
    emit = ([f"--iree-sa-emit-riscv={args.out}"] if args.code == "compiled" else
            ["--iree-sa-riscv"] if args.code == "embedded" else [])
    for stale in ("rvb_code.bin", "rvb_embed.S"):
        if os.path.exists(os.path.join(args.out, stale)):
            os.remove(os.path.join(args.out, stale))
    blob, err = G.compile_one(src, emit, os.path.join(args.out, "dispatch.sadesc"))
    if blob is None:
        raise SystemExit(err)
    d = DC.target_config(src).get("d", 16)
    sim, phys, bufs, want, regions, lst_addr, size, name, lst_json = layout(src, blob, d, args.t)
    before = sim.ddr_read(DDR_BASE, size).tobytes()
    sim.trace = []
    sim.run_list(lst_addr)
    trace, sim.trace = sim.trace, None
    after = sim.ddr_read(DDR_BASE, size).tobytes()
    # the functional simulator against the oracle (as dispatch_check.check)
    for b, regs in regions.items():
        got = after[phys[b] - DDR_BASE:phys[b] - DDR_BASE + len(bufs[b])]
        for o, n, et, w in regs:
            if got[o:o + n] != bytes(want[b][o:o + n]):
                raise SystemExit(f"{name}: the functional simulator differs from the oracle (binding {b})")
    hexfile(os.path.join(args.out, "rvb_ddr.hex"), DDR_BASE, before)
    hexfile(os.path.join(args.out, "rvb_expect.hex"), DDR_BASE, after)
    open(os.path.join(args.out, "rvb_ddr.bin"), "wb").write(before)
    open(os.path.join(args.out, "rvb_expect.bin"), "wb").write(after)
    lst_json.update(name=name, t=args.t, window=size)
    json.dump(lst_json, open(os.path.join(args.out, "rvb_list.json"), "w"), indent=1)
    with open(os.path.join(args.out, "rvb_cmp.txt"), "w") as fh:
        for b in sorted(bufs):
            fh.write(f"{phys[b] - DDR_BASE} {-(-len(bufs[b]) // 4096) * 4096}\n")
    em = Emitter(DDR_BASE)
    for r in trace:
        em.cmd(r)
    with open(os.path.join(args.out, "rvb_args.h"), "w") as fh:
        fh.write(f"/* {name}, T = {args.t} (compiler/tests/rv_dispatch.py) */\n")
        fh.write(f"#define RVB_LIST_OFF 0x{lst_addr - DDR_BASE:x}u\n#define RVB_DDR_BYTES {size}\n")
        fh.write(f"#define RVB_COMMANDS {sum(v for k, v in em.counts.items() if k != 'FENCE')}\n")
    if args.code == "embedded":
        # the export's rv32 code from the sa-desc code table (header word 48), as iree-compile made it
        import struct
        tab = struct.unpack_from("<I", blob, 48)[0]
        if not tab:
            raise SystemExit(f"{name}: no RISC-V code table (iree-compile --iree-sa-riscv)")
        off, n = struct.unpack_from("<II", blob, tab)
        open(os.path.join(args.out, "rvb_code.bin"), "wb").write(blob[off:off + n])
        with open(os.path.join(args.out, "rvb_embed.S"), "w") as fh:
            fh.write('/* the export\'s code from the sa-desc file (iree-compile --iree-sa-riscv), as it is */\n'
                     '    .section .rvb_gen, "ax"\n    .balign 4\n    .globl rvb_code\nrvb_code:\n'
                     f'    .incbin "{os.path.join(os.path.abspath(args.out), "rvb_code.bin")}"\n')
        keep = DC.O.load_dispatch(src)           # (the tuple holds the MLIR context alive)
        consts = [int(c) & 0xFFFFFFFF for c in DC.push_constants(keep[2], args.t)]
        with open(os.path.join(args.out, "rvb_gen.c"), "w") as fh:
            fh.write(f"/* {name}, T = {args.t}: the export's rv32 code embedded by iree-compile (rvb_code.bin, {n} bytes)\n"
                     f" * with the dispatch's bindings and push constants (rv_dispatch.py --code embedded) */\n"
                     f"#include <stdint.h>\n\nvoid rvb_code(const uint32_t *bindings, const uint32_t *push_constants);\n\n"
                     f'__attribute__((section(".rvb_gen"), noinline)) void rvb_gen(uint32_t W)\n{{\n'
                     f"    uint32_t b[{max(len(phys), 1)}], pc[{max(len(consts), 1)}];\n")
            for i, a in enumerate(phys):
                fh.write(f"    b[{i}] = W + 0x{int(a) - DDR_BASE:x}u;\n" if a else f"    b[{i}] = 0;\n")
            for i, c in enumerate(consts):
                fh.write(f"    pc[{i}] = 0x{c:x}u;\n")
            fh.write("    rvb_code(b, pc);\n}\n")
        if os.path.exists(os.path.join(args.out, "rvb_gen.ll")):
            os.remove(os.path.join(args.out, "rvb_gen.ll"))
    elif args.code == "compiled":
        # the export's generated function, called with the dispatch's bindings (relative to the
        # window W) and push constants
        lls = [x for x in glob.glob(os.path.join(args.out, "*.ll")) if not x.endswith("rvb_gen.ll")]
        if len(lls) != 1:
            raise SystemExit(f"{name}: {len(lls)} generated .ll files")
        sym = os.path.basename(lls[0])[:-3]
        os.replace(lls[0], os.path.join(args.out, "rvb_gen.ll"))
        keep = DC.O.load_dispatch(src)           # (the tuple holds the MLIR context alive)
        consts = [int(c) & 0xFFFFFFFF for c in DC.push_constants(keep[2], args.t)]
        nb = len(phys)
        with open(os.path.join(args.out, "rvb_gen.c"), "w") as fh:
            fh.write(f"/* {name}, T = {args.t}: the export's compiled function (rvb_gen.ll, sahw -> LLVM ->\n"
                     f" * riscv32) with the dispatch's bindings and push constants (rv_dispatch.py --code compiled) */\n"
                     f"#include <stdint.h>\n\nvoid {sym}(const uint32_t *bindings, const uint32_t *push_constants);\n\n"
                     f'__attribute__((section(".rvb_gen"), noinline)) void rvb_gen(uint32_t W)\n{{\n'
                     f"    uint32_t b[{max(nb, 1)}], pc[{max(len(consts), 1)}];\n")
            for i, a in enumerate(phys):
                fh.write(f"    b[{i}] = W + 0x{int(a) - DDR_BASE:x}u;\n" if a else f"    b[{i}] = 0;\n")
            for i, c in enumerate(consts):
                fh.write(f"    pc[{i}] = 0x{c:x}u;\n")
            fh.write(f"    {sym}(b, pc);\n}}\n")
    else:
        with open(os.path.join(args.out, "rvb_gen.c"), "w") as fh:
            fh.write(f'/* {name}, T = {args.t}: the traced commands over PCPI (rv_dispatch.py) */\n'
                     f'#include "sysarray_intrinsics.h"\n\n'
                     f'__attribute__((section(".rvb_gen"), noinline)) void rvb_gen(uint32_t W)\n{{\n')
            fh.write("\n".join(em.lines) + "\n    mat_fence(0);\n}\n")
        if os.path.exists(os.path.join(args.out, "rvb_gen.ll")):
            os.remove(os.path.join(args.out, "rvb_gen.ll"))
    print(f"{name}: {len(trace)} commands traced {em.counts}, {em.insns} PCPI instructions, "
          f"DDR window {size // 1024} KB, list at {lst_addr:#x} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Per-dispatch differential test of the sa backend (docs/iree_compiler_plan.md §6.8 H).

For every dispatch of a compiled module:
  - the source (iree-compile --iree-hal-dump-executable-sources-to) gives the
    semantics: compiler/tests/oracle.py executes it;
  - the executable (--iree-hal-dump-executable-binaries-to, sa-desc) gives the
    code: its template runs in the functional simulator (llm/sa_funcsim.py)
    with the dispatch list the driver builds (register setup included);
both on the same binding contents, and every byte the dispatch may write
(the regions of its stores) must be equal.

Inputs: push constants that are binding offsets take the first call site's
values (util.assume.int); dynamic lengths (workload ordinals) take --t;
fp32 data is normal(0, 1), int8 / int32 data int8-valued, i64 scalars
(token, position, row) --t - 3 (rows partly masked).

    python3 compiler/tests/dispatch_check.py <sources dir> <binaries dir> [--d 8] [--t 16] [numbers...]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMPILER = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(COMPILER, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(COMPILER, "runtime", "tools"))
sys.path.insert(0, os.path.join(REPO, "llm"))
import oracle as O  # noqa: E402
import sadesc  # noqa: E402
from sa_funcsim import SaError, SaFuncSim  # noqa: E402

from iree.compiler import ir  # noqa: E402

ELEM_BYTES = {"f32": 4, "i8": 1, "i32": 4, "i64": 8, "i16": 2, "i1": 1}


def ops_of(func):
    return [o.operation for o in func.regions[0].blocks[0].operations]


def push_constants(func, t, site=0):
    """Values of the push constants: binding offsets from util.assume.int (first
    call site), 64-bit halves (the high word 0), the rest t (lengths)."""
    loads = {}
    n = 0
    for op in ops_of(func):
        if op.name == "hal.interface.constant.load":
            ordinal = ir.IntegerAttr(op.attributes["ordinal"]).value
            loads[op.results[0]] = ordinal
            n = max(n, ordinal + 1)
    vals = [t] * n
    # a constant shifted left (the high word of a 64-bit value) is 0
    for op in ops_of(func):
        if op.name == "arith.shli":
            src = op.operands[0].owner
            if src.name == "arith.extui" and src.operands[0] in loads:
                vals[loads[src.operands[0]]] = 0
    # values fixed by util.assume.int (first call site)
    for op in ops_of(func):
        if op.name != "util.assume.int":
            continue
        text = str(op)
        for i, operand in enumerate(op.operands):
            src = operand.owner
            while src.name in ("arith.index_castui", "arith.index_cast", "arith.extui") and \
                    src.operands[0] not in loads:
                src = src.operands[0].owner
            key = src.operands[0] if src.name in ("arith.index_castui", "arith.index_cast", "arith.extui") else None
            if key not in loads:
                continue
            # the i-th operand's assumption list: '%x[<umin = A, umax = A, ...>, ...]' or '%x<...>'
            m = re.findall(r"<umin = (\d+), umax = (\d+)", text.split("\n")[i + 1] if "\n" in text else text)
            if len(m) > site and m[site][0] == m[site][1]:
                vals[loads[key]] = int(m[site][0])
    return vals


def binding_regions(func, interp):
    """{binding: [(offset, nbytes, element type, is_written)]} from the subspans
    and the stores, evaluated with the interpreter (offsets, dynamic sizes)."""
    regions = {}
    written = set()
    for op in ops_of(func):
        if op.name == "iree_tensor_ext.dispatch.tensor.store":
            written.add(op.operands[1])
    for op in ops_of(func):
        interp.run_op(op) if op.name not in ("iree_tensor_ext.dispatch.tensor.load",
                                              "iree_tensor_ext.dispatch.tensor.store", "linalg.generic",
                                              "linalg.fill", "linalg.batch_matmul", "tensor.empty",
                                              "iree_linalg_ext.scatter", "func.return") else None
        if op.name == "hal.interface.binding.subspan":
            _, b, off, dyn = interp.val(op.results[0])
            dims, et = interp._dispatch_type(op.results[0])
            dyn = list(dyn)
            dims = [dyn.pop(0) if x is None else x for x in dims]
            n = int(np.prod(dims)) if dims else 1
            regions.setdefault(b, []).append((off, n * ELEM_BYTES[et], et, op.results[0] in written))
    return regions


def fill(rng, et, n, t):
    if et == "f32":
        return rng.standard_normal(n).astype(np.float32).tobytes()
    if et in ("i8", "i32", "i16"):
        return rng.integers(-127, 128, n).astype(O.NP[et]).tobytes()
    if et == "i64":
        return np.full(n, max(0, t - 3), np.int64).tobytes()     # pos + 1 < T: partly masked rows
    return bytes(n)


def call_sites(func):
    """Number of call sites described by the util.assume.int lists."""
    n = 1
    for op in ops_of(func):
        if op.name == "util.assume.int":
            for line in str(op).split("\n")[1:]:
                n = max(n, len(re.findall(r"<umin = ", line)))
    return n


def check(src, blob, d, t, seed=1, verbose=False, site=0):
    keep = O.load_dispatch(src)
    func = keep[2]
    dd, _, exps = sadesc.read(blob)
    if len(exps) != 1:
        return f"{len(exps)} exports"
    name, rows, nb, nc, cyc, setup = exps[0]
    ext = sadesc.read_ext(blob)[0]
    if cyc == 0xFFFFFFFF:
        return "UNSUPPORTED"
    consts = push_constants(func, t, site)
    probe = O.Interp({}, consts, d)
    regions = binding_regions(func, probe)
    rng = np.random.default_rng(seed)
    bufs = {}
    for b, regs in regions.items():
        size = max(o + n for o, n, _, _ in regs)
        buf = bytearray(-(-size // 64) * 64 + 64)
        for o, n, et, w in regs:
            buf[o:o + n] = fill(rng, et, n // ELEM_BYTES[et], t)
        bufs[b] = buf
    want = {b: bytearray(v) for b, v in bufs.items()}
    O.run_dispatch(func, want, consts, d)
    # the template in the simulator
    base = 0x10000000
    tsize = -(-max(rows.nbytes, 0x10000) // 0x10000) * 0x10000          # the template, whole 64 KB pages
    size = max(0x800000, -(-(tsize + sum(-(-len(b) // 4096) * 4096 for b in bufs.values())) // 0x100000) * 0x100000
               + 0x800000)
    sim = SaFuncSim(d, base, size)
    sim.ddr_write(base, rows.tobytes())
    phys, a = [0] * max(nb, max(bufs, default=0) + 1), base + tsize
    for b in sorted(bufs):
        phys[b] = a
        sim.ddr_write(a, bytes(bufs[b]))
        a += -(-len(bufs[b]) // 4096) * 4096
    lst = sadesc.dispatch_list(base, phys, consts, setup, ext)
    lst_addr = base + size - 0x10000
    sim.ddr_write(lst_addr, lst.array().tobytes())
    try:
        sim.run_list(lst_addr)
    except SaError as e:
        return f"device error: {e}"
    bad = []
    for b, regs in regions.items():
        got = sim.ddr_read(phys[b], len(bufs[b])).tobytes()
        for o, n, et, w in regs:
            g, x = got[o:o + n], bytes(want[b][o:o + n])
            if g != x:
                if et == "f32":
                    gv, xv = np.frombuffer(g, np.float32), np.frombuffer(x, np.float32)
                    k = int(np.argmax(gv != xv))
                    bad.append(f"b{b}@{o} f32[{n // 4}]: {int(np.sum(gv.view(np.uint32) != xv.view(np.uint32)))} "
                               f"differ, first [{k}] {gv[k]!r} vs {xv[k]!r}")
                else:
                    gv, xv = np.frombuffer(g, O.NP[et]), np.frombuffer(x, O.NP[et])
                    k = int(np.argmax(gv != xv))
                    bad.append(f"b{b}@{o} {et}[{len(gv)}]: {int(np.sum(gv != xv))} differ, first [{k}] {gv[k]} vs {xv[k]}")
    return "; ".join(bad) if bad else "OK"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sources")
    ap.add_argument("binaries")
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("numbers", nargs="*")
    args = ap.parse_args()
    srcs = sorted(glob.glob(os.path.join(args.sources, "*.mlir")),
                  key=lambda p: int(re.findall(r"_(\d+)\.mlir$", p)[0]))
    counts = {}
    for p in srcs:
        num = re.findall(r"_(\d+)\.mlir$", p)[0]
        if args.numbers and num not in args.numbers:
            continue
        bins = glob.glob(os.path.join(args.binaries, f"*dispatch_{num}_*.sadesc")) + \
            glob.glob(os.path.join(args.binaries, f"*dispatch_{num}.sadesc"))
        if not bins:
            res = "no binary"
        else:
            try:
                keep = O.load_dispatch(p)
                blob = open(bins[0], "rb").read()
                res = "OK"
                for site in range(call_sites(keep[2])):      # every call site's offsets
                    r = check(p, blob, args.d, args.t, site=site)
                    if r != "OK":
                        res = f"call site {site}: {r}"
                        break
            except NotImplementedError as e:
                res = f"oracle: {e}"
        key = "OK" if res == "OK" else "UNSUPPORTED" if res == "UNSUPPORTED" else "FAIL"
        counts[key] = counts.get(key, 0) + 1
        keep = O.load_dispatch(p)
        f = keep[2]
        print(f"#{num:>4} {str(f.attributes['sym_name']).strip(chr(34)).split('_', 3)[-1][:44]:44s} {res}")
    print(" ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 0 if set(counts) <= {"OK"} else 1


if __name__ == "__main__":
    sys.exit(main())

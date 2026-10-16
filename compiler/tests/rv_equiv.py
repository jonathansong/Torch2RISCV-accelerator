#!/usr/bin/env python3
"""Command-stream equivalence of the sahw -> LLVM -> riscv32 backend
(docs/riscv_backend_design.md §9.5, S1): for each dispatch, the export's
generated function (iree-compile --iree-sa-emit-riscv) compiled for the host
with the runtime's PCPI operations recorded (firmware/rv/sa_rv_rt.c,
SA_RV_HOST; compiler/tests/rv_host_main.c) must issue exactly the PCPI
operations that the descriptor template issues when the functional simulator
runs it with the same bindings and push constants (dispatch_check's layout,
the executed commands traced, mapped to PCPI by the same rules: every key of
the command, then the command). Each .ll is also compiled for rv32imc (code
size).

    compiler/tests/rv_equiv.py <sa_sources dir or .mlir files> [--flags "..."] [--t 16,304] [--jobs 4]
    compiler/tests/rv_equiv.py --golden [--t 16,304]      every case of the golden corpus (golden.py)
Prints one line per failing dispatch and a summary; exit 1 on a mismatch.
"""
import argparse
import concurrent.futures as cf
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

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

DDR_BASE = 0x18000000
CLANG = os.environ.get("CLANG", "clang-18")
FW = os.path.join(REPO, "firmware")


def f(x, lo, n):
    return (x >> lo) & ((1 << n) - 1)


def pcpi_lines(r):
    """The PCPI operations of one traced command ([hdr, ddr, n, w1..w7]), as
    sa_rv_cmd issues them (every key, then the command)"""
    hdr, ddr = int(r[0]), int(r[1])
    w = [0] + [int(x) for x in r[3:10]]
    op, out = hdr & 0xFF, []
    if op != 0x10 and (hdr >> 11) & 1:
        out.append("mat_fence 0")
    if op == 0x01 and f(w[3], 32, 2) == 2:
        out += [f"gemv_cfg 0 {f(w[2], 48, 16):x}", f"gemv_cfg 1 {f(w[3], 0, 32):x}", f"gemv_cfg 2 {f(w[3], 34, 16):x}",
                f"gemv_cfg 3 {f(w[3], 50, 8):x}",
                f"gemv_run {ddr:x} {f(w[2], 0, 32) & 0xFFFF:x} {f(w[2], 32, 16):x} {f(w[3], 58, 1):x}"]
    elif op == 0x01:
        out += [f"mat_cfg 0 {f(w[2], 32, 16):x}", f"mat_cfg 1 {f(w[2], 48, 16):x}", f"mat_cfg 2 {f(w[3], 0, 32):x}",
                f"mat_cfg 3 {f(w[3], 32, 2):x}", f"mat_load {ddr:x} {f(w[2], 0, 32):x}"]
    elif op == 0x02:
        out += [f"mat_cfg 4 {f(w[2], 32, 16):x}", f"mat_cfg 5 {f(w[2], 48, 16):x}", f"mat_cfg 6 {f(w[3], 0, 32):x}",
                f"mat_store {ddr:x} {f(w[2], 0, 32):x}"]
    elif op == 0x03:
        out += [f"mat_cfg 7 {f(w[2], 0, 12):x}", f"mat_cfg 8 {f(w[2], 16, 16):x}", f"mat_cfg 9 {f(w[2], 32, 16):x}",
                f"mat_cfg a {f(w[2], 48, 16):x}",
                f"mat_exec {f(w[1], 0, 16):x} {f(w[1], 16, 16):x} {f(w[1], 32, 16):x} {f(w[1], 48, 12):x} {f(w[1], 60, 1):x}"]
    elif op == 0x04:
        keys = [f(w[3], 0, 8), f(w[2], 32, 32), f(w[2], 0, 32), f(w[3], 8, 6), f(w[3], 16, 16), f(w[3], 32, 16),
                f(w[3], 48, 5), f(w[4], 0, 32), f(w[4], 32, 32), f(w[5], 0, 32), f(w[3], 53, 11), f(w[5], 32, 32),
                f(w[6], 0, 32), f(w[6], 32, 32), f(w[7], 0, 32), f(w[7], 32, 32)]
        out += [f"vec_cfg {k:x} {v:x}" for k, v in enumerate(keys)]
        out.append(f"vec_run {f(w[1], 0, 32):x} {f(w[1], 32, 32):x}")
    elif op == 0x10:
        out.append(f"mat_fence {f(w[1], 0, 4):x}")
    else:
        out.append(f"? {op:#x}")
    return out


def layout(src, blob, t, seed=1):
    """rv_dispatch.layout without the oracle: (sim, phys, constants, list address, window bytes)"""
    keep = DC.O.load_dispatch(src)
    func = keep[2]
    _, _, exps = sadesc.read(blob)
    name, rows, nb, nc, cyc, setup = exps[0]
    ext = sadesc.read_ext(blob)[0]
    d = DC.target_config(src).get("d", 16)
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
    return sim, phys[:nb], [int(c) for c in consts], lst_addr, size, name


def one(item):
    """(source, flags, ts) -> (name, status, detail, rv32 text bytes)"""
    path, flags, ts = item
    tmp = tempfile.mkdtemp(prefix="rveq_")
    try:
        src = os.path.join(tmp, "dispatch.mlir")
        open(src, "w").write(G.sa_only(open(path).read()))
        blob, err = G.compile_one(src, flags + [f"--iree-sa-emit-riscv={tmp}"], os.path.join(tmp, "d.sadesc"))
        if blob is None:                         # (no template at all: the corpus's expected failures)
            plain, _ = G.compile_one(src, flags, os.path.join(tmp, "p.sadesc"))
            return os.path.basename(path), "compile" if plain else "no template", err, 0
        lls = glob.glob(os.path.join(tmp, "*.ll"))
        if len(lls) != 1:
            return os.path.basename(path), "ll", f"{len(lls)} .ll files", 0
        ll = lls[0]
        sym = os.path.basename(ll)[:-3]
        exe = os.path.join(tmp, "rv_host")
        r = subprocess.run([CLANG, "-O1", "-w", "-DSA_RV_HOST", f"-DSA_RV_ENTRY={sym}", f"-I{FW}/include", ll,
                            os.path.join(HERE, "rv_host_main.c"), os.path.join(FW, "rv", "sa_rv_rt.c"), "-o", exe],
                           capture_output=True, text=True)
        if r.returncode:
            return sym, "host build", r.stderr[-800:], 0
        obj = os.path.join(tmp, "rv.o")
        r = subprocess.run([CLANG, "--target=riscv32-unknown-elf", "-march=rv32imc", "-mabi=ilp32", "-O2", "-w", "-c",
                            ll, "-o", obj], capture_output=True, text=True)
        if r.returncode:
            return sym, "rv32 build", r.stderr[-800:], 0
        sz = subprocess.run(["llvm-size-18", obj], capture_output=True, text=True).stdout.split("\n")[1].split()
        rvbytes = int(sz[0]) + int(sz[1])
        for t in ts:
            sim, phys, consts, lst_addr, size, name = layout(src, blob, t)
            img = os.path.join(tmp, "ddr.bin")
            sim.ddr[:size].tofile(img)               # (no copy: the lm_head windows are 160 MB)
            wantf = os.path.join(tmp, "want.txt")
            reads, rd = [], sim.ddr_read

            class Sink:                            # the trace as PCPI lines, straight to a file
                def __init__(self, fh):
                    self.fh, self.n = fh, 0

                def append(self, row):
                    for l in pcpi_lines(row):
                        self.fh.write(l + "\n")
                        self.n += 1

            def logged(a, n, *rest):               # (LDPARAM reads 4 bytes through ddr_read)
                v = rd(a, n, *rest)
                if n == 4:
                    reads.append((int(a), int(np.asarray(v).view("<u4")[0])))
                return v
            with open(wantf, "w") as fh:
                sim.trace = Sink(fh)
                sim.ddr_read = logged
                sim.run_list(lst_addr)
                sim.ddr_read = rd
                nwant = sim.trace.n
            sim.trace = None
            del sim
            rdf = os.path.join(tmp, "reads.txt")
            open(rdf, "w").write("".join(f"{a:x} {v:x}\n" for a, v in reads))
            gotf = os.path.join(tmp, "got.txt")
            with open(gotf, "w") as fo:
                r = subprocess.run([exe, img, hex(DDR_BASE), str(len(phys)), *map(hex, phys), str(len(consts)),
                                    *map(hex, [c & 0xFFFFFFFF for c in consts]), rdf], stdout=fo,
                                   stderr=subprocess.PIPE, text=True)
            if r.returncode:
                return sym, "host run", r.stderr[-500:], rvbytes
            # compare line by line, the "#" notes aside (their LDPARAM addresses against the reads)
            gaddr, k, ngot, diff = [], 0, 0, None
            with open(gotf) as fg, open(wantf) as fw:
                for l in fg:
                    if l.startswith("#"):
                        if l.startswith("# ldparam "):
                            gaddr.append(int(l.split()[2], 16))
                        continue
                    ngot += 1
                    w = fw.readline()
                    if diff is None and l != w:
                        diff = (k, l.strip(), w.strip() or "-")
                    k += 1
                if diff is None and fw.readline():
                    diff = (k, "-", "(more)")
            if gaddr != [a for a, _ in reads]:
                return sym, "MISMATCH", f"T = {t}: LDPARAM addresses {[hex(a) for a in gaddr[:4]]} vs " \
                                        f"{[hex(a) for a, _ in reads[:4]]} ({len(gaddr)} vs {len(reads)})", rvbytes
            if diff is not None or ngot != nwant:
                k, g, w = diff or (ngot, "-", "-")
                return sym, "MISMATCH", (f"T = {t}: {ngot} vs {nwant} PCPI operations, first difference at "
                                         f"{k}: got {g} want {w}"), rvbytes
            want = range(nwant)
        return sym, "ok", f"{len(want)} PCPI operations", rvbytes
    except Exception as e:                       # (a dispatch the harness cannot lay out)
        return os.path.basename(path), "harness", f"{type(e).__name__}: {str(e)[:300]}", 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sources", nargs="*")
    ap.add_argument("--golden", action="store_true", help="the golden corpus's cases (golden.py CASES)")
    ap.add_argument("--flags", default="")
    ap.add_argument("--t", default="16,304", help="sequence lengths to run each dispatch at")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress", action="store_true", help="one line per dispatch on stderr")
    args = ap.parse_args()
    ts = [int(x) for x in args.t.split(",")]
    items, seen = [], set()
    groups = [(c, os.path.join(REPO, s), fl) for c, s, fl in G.CASES] if args.golden else \
             [("", s, args.flags.split()) for s in args.sources]
    for case, s, fl in groups:
        paths = sorted(glob.glob(os.path.join(s, "*.mlir"))) if os.path.isdir(s) else [s]
        for p in paths:
            text = G.sa_only(open(p).read())
            k = G.key_of(text, fl)               # the corpus's unique dispatches
            if k in seen:
                continue
            seen.add(k)
            items.append((p, fl, ts))
    if args.limit:
        items = items[:args.limit]
    counts, rv = {}, []
    # a fresh worker per dispatch: the MLIR contexts of loading a dispatch are not freed (memory grew
    # past 8 GB over ~130 dispatches with long-lived workers)
    import multiprocessing as mp
    with cf.ProcessPoolExecutor(args.jobs, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
        for i, (name, status, detail, rvb) in enumerate(ex.map(one, items)):
            if args.progress:
                print(f"[{i + 1}/{len(items)}] {status} {name}", file=sys.stderr, flush=True)
            counts[status] = counts.get(status, 0) + 1
            if status not in ("ok", "no template"):
                print(f"{status:10s} {name}: {detail}", flush=True)
            elif rvb:
                rv.append(rvb)
    n = len(items)
    print(f"{n} unique dispatches, T = {ts}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if rv:
        print(f"rv32imc code (the generated functions, without the runtime): {sum(rv) / 1024:.1f} KB in all, "
              f"median {int(np.median(rv))} B, max {max(rv)} B")
    bad = sum(v for k, v in counts.items() if k not in ("ok", "no template"))
    print("rv_equiv " + ("PASS" if bad == 0 else "FAIL"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

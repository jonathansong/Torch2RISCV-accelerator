#!/usr/bin/env python3
"""Schedule prototype of the fused decode attention (docs/fusion_design.md §6.2.2,
§6.6 step 1): what stage 1 (the q path, the attention core and the output
quantization in one dispatch per layer) gains, and what the per-bank
scoreboard costs it, before any compiler work.

The commands are the real ones: each unfused decode dispatch of the chain is
compiled by the current compiler, run on the functional simulator at T
through dispatch_check.py (so also checked against the oracle) with the
executed commands captured, as attn_cycles.py does. Then, timed by
perf_model.run_list:

  unfused     the dispatches one after the other, a FENCE between them (the
              batched runtime's list)
  in order    one list: the LDs and STs of intermediates dropped (those on
              transient bindings, except the chain's input, q_proj's output,
              and its outputs, the int8 row and its scale), the GEMV reading
              p where the probabilities wrote it, the commands in the same
              order (what fusion_whatif.py's bound assumes)
  scheduled   the same commands per unit of query rows (the probabilities'
              row pieces: Qwen3 4 rows = 2 KV heads): the q path, the A
              tiles of all heads, then per unit u its scores, followed by
              unit u - 1's probabilities, sum and p . V; unit u's ACC work in
              bank u mod 2; p (the GEMV's x) in SPAD_A bank 0 (with the A
              tiles) or bank 1 (with the raw K blocks)
  interleaved as scheduled, unit u - 1's commands spread over unit u's
              K blocks, a share after each block's EX (a T loop carrying the
              previous unit's work)
  blocked T loop  as interleaved, but unit u - 1's row-long VEs (exp, the
              scaling to int8, the sums) and GEMVs (p . V over T) cut into one
              piece per K block of unit u, each piece paying its own fixed
              cost: the form a T loop body can take (the max before it; Σp,
              p . V accumulated over the blocks, integers)
  k sub-loops the T loop of unit u split into k loops over its blocks, a
              k-th of unit u - 1's commands (whole) after each; k = 2 with
              two KV heads per unit puts them between the heads' T loops:
              valid for any T, the implementable form
  tiles alternating  the q . K^T loop's tiles in SPAD_B banks 0 / 1 and the
              next block's transpose ahead of the current block's copies
              (double buffering; applied to the unfused list too)

The segmentation follows the kernels' structure (the q . K^T kernel of §8.22
steps a-c, the row pieces of C6.5b, p . V on the GEMV unit) and the Qwen3
dispatch numbering of the 2-layer build; --dispatches gives another model's.
The scheduled list repeats the per-row Σp broadcast (the sum's transpose and
p . V's prologue) per unit: conservative.

    compiler/tests/attn_schedule.py build/hfgen/qwen3_l2_t2048lc [--t 48,512,2048] [--layers 28]
"""
import argparse
import json
import os
import re
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "llm"))
import dispatch_check as DC  # noqa: E402
import fusion_whatif as F  # noqa: E402
import golden as G  # noqa: E402
import perf_model as P  # noqa: E402

# Qwen3 numbering (build/hfgen/qwen3_l2_t2048lc): q path, q . K^T, probabilities, Σp, p . V, amax, int8
QWEN3 = dict(qpath=(8, 9, 13, 17, 18, 19), scores=20, probs=21, psum=22, pv=23, out=(24, 25))
A0, A1, B0, B1, C0, C1 = 1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5


def alternate_tiles(cmds):
    """the q . K^T loop's transposed tiles alternating SPAD_B banks 0 / 1 (the loop body unrolled by
    two): the j-th transpose into SPAD_B and the EX after it use bank j mod 2; an EX before any
    transpose (iteration 0's, in the rotated loop) bank 1. And the body reordered: block i + 1's
    transpose right after block i's EX, ahead of block i's copies (which wait for that EX), so the
    VE transposes while the EX runs"""
    cmds = list(cmds)
    i = 0
    while i < len(cmds):                               # [EX, copies..., transpose] -> [EX, transpose, copies...]
        c = cmds[i][0]
        if c.op == P.OP_VE and c.kind == 1 and c.w & (B0 | B1):
            k = i - 1
            while k >= 0 and cmds[k][0].op == P.OP_VE and cmds[k][0].kind == 0:
                k -= 1
            if k >= 0 and k < i - 1 and cmds[k][0].op == P.OP_EX:
                cmds.insert(k + 1, cmds.pop(i))
        i += 1
    out, j, ex_bank = [], 0, B1
    for c, b in cmds:
        c = clone(c)
        if c.op == P.OP_VE and c.kind == 1 and c.w & (B0 | B1):
            bank = B0 if j % 2 == 0 else B1
            c.w = (c.w & ~(B0 | B1)) | bank
            ex_bank, j = bank, j + 1
        elif c.op == P.OP_EX and c.r & (B0 | B1):
            c.r = (c.r & ~(B0 | B1)) | ex_bank
        out.append((c, b))
    return out


def arch(cal):
    a = P.Arch(gemv_ports=4)
    c = json.load(open(cal))
    a.ve_a, a.ve_b, a.f_desc, a.list_ovh = tuple(c["ve_a"]), tuple(c["ve_b"]), c["f_desc"], c["list_ovh"]
    return a


def indirect(src):
    """per binding: transient (Indirect) or not (globals, parameters)"""
    m = re.search(r"bindings = \[(.*?)\]", open(src).read())
    return [("Indirect" in b) for b in m.group(1).split("#hal.pipeline.binding")[1:]]


def capture(a, src, blob, t, d):
    """(check result, [(Cmd, binding of its DDR range or None)]) of one dispatch at T"""
    rows, phys = [], []
    orig_run, orig_list = DC.SaFuncSim.run_list, DC.sadesc.dispatch_list

    def run_list(self, addr):
        self.trace = []
        try:
            return orig_run(self, addr)
        finally:
            rows.extend(self.trace)
            self.trace = None

    def dispatch_list(base, ph, *rest):
        phys.extend(ph)
        return orig_list(base, ph, *rest)
    DC.SaFuncSim.run_list, DC.sadesc.dispatch_list = run_list, dispatch_list
    try:
        res = DC.check(src, blob, d, t)
    finally:
        DC.SaFuncSim.run_list, DC.sadesc.dispatch_list = orig_run, orig_list
    starts = sorted((p, b) for b, p in enumerate(phys) if p)
    out = []
    for r in rows:
        r = np.asarray(r, np.uint64)
        c, rg, bind = P.decode(a, r), F.ddr_range(r), None
        if rg:
            for p, b in starts:
                if p <= rg[0]:
                    bind = b
        out.append((c, bind))
    return res, out


def clone(c, swap_acc=False):
    c2 = P.Cmd()
    for f in P.Cmd.__slots__:
        setattr(c2, f, getattr(c, f))
    c2.rng = list(c.rng)
    if swap_acc:
        for f in ("r", "w"):
            m = getattr(c2, f)
            setattr(c2, f, (m & ~(C0 | C1)) | (C1 if m & C0 else 0) | (C0 if m & C1 else 0))
    return c2


def set_a(c, field, bank):
    m = getattr(c, field)
    if m & (A0 | A1):
        setattr(c, field, (m & ~(A0 | A1)) | (A0 if bank == 0 else A1))


def concat(parts, fences):
    """one list from command lists, renumbered (the front end fetches in this order), FENCEs between parts"""
    out = []
    for i, cmds in enumerate(parts):
        if fences and i and out:
            out.append(F.fence_cmd(len(out)))
        for c in cmds:
            c2 = clone(c)
            c2.n = len(out)
            out.append(c2)
    return out


def pieces_of(a, c, m):
    """c as m commands of 1 / m of its work each (a VE over a row's groups, a GEMV over its K), each with
    its own fixed cost; other commands whole"""
    if m <= 1 or not ((c.op == P.OP_VE and c.groups >= m and c.kind != 1) or c.gemv):
        return [c]
    out = []
    for j in range(m):
        c2 = clone(c)
        if c.gemv:
            c2.bytes, c2.k = c.bytes // m, max(c.k // m, a.d)
            c2.beats, c2.bursts = -(-c.beats // m), -(-c.bursts // m)
        else:
            c2.groups = c.groups // m + (c.groups % m if j == m - 1 else 0)
        out.append(c2)
    return out


def starts_at(cmds, pred):
    return [i for i, c in enumerate(cmds) if pred(i, c)]


def split(cmds, idx):
    """[prologue, seg 0, seg 1, ...] at the indices"""
    idx = list(idx)
    return [cmds[:idx[0]]] + [cmds[i:j] for i, j in zip(idx, idx[1:] + [len(cmds)])]


def build(disp, num, pbank, a):
    """the three lists from the captured dispatches {number: [(Cmd, binding)]} and their transient flags"""
    first, outs = num["qpath"][0], num["out"]

    def kept(n):
        res = []
        for c, b in disp[n]["cmds"]:
            if c.op in (P.OP_LD, P.OP_ST) and not c.gemv and b is not None and disp[n]["indirect"][b]:
                if not ((n == first and c.op == P.OP_LD) or (n in outs and c.op == P.OP_ST)):
                    continue
            res.append(clone(c))
        return res
    order = list(num["qpath"]) + [num["scores"], num["probs"], num["psum"], num["pv"]] + list(num["out"])
    unfused = concat([[c for c, _ in disp[n]["cmds"]] for n in order], fences=True)

    # p: written by the probabilities' VEs into SPAD_A, read by the sum's VEs and the GEMV
    K = {n: kept(n) for n in order}
    for c in K[num["probs"]]:
        if c.op == P.OP_VE and c.w & (A0 | A1):
            set_a(c, "w", pbank)
    for c in K[num["psum"]]:
        if c.op == P.OP_VE and c.r & (A0 | A1):
            set_a(c, "r", pbank)
    for c in K[num["pv"]]:
        if c.gemv:
            set_a(c, "r", pbank)
    inorder = concat([K[n] for n in order], fences=False)

    # segmentation (from the full lists: the dropped LDs mark the boundaries)
    def segs(n, marker):
        full = disp[n]["cmds"]
        idx = starts_at([c for c, _ in full], marker)
        kk = iter(K[n])
        cur = next(kk, None)
        # map each full command to its kept clone (same order; dropped ones -> None)
        mapped = []
        for c, b in full:
            drop = (c.op in (P.OP_LD, P.OP_ST) and not c.gemv and b is not None and disp[n]["indirect"][b]
                    and not ((n == first and c.op == P.OP_LD) or (n in outs and c.op == P.OP_ST)))
            if drop:
                mapped.append(None)
            else:
                mapped.append(cur)
                cur = next(kk, None)
        parts = split(mapped, idx)
        return [[c for c in p if c is not None] for p in parts]

    sc = disp[num["scores"]]["cmds"]
    qrow = max(c.bytes for c, _ in sc if c.op == P.OP_LD and c.w & (C0 | C1) and c.bytes < 4096)
    heads = segs(num["scores"], lambda i, c: c.op == P.OP_LD and c.bytes == qrow and c.w & (C0 | C1))
    s_pro, heads = heads[0], heads[1:]
    tiles, scores = [], []
    for h in heads:                                   # the A tile (VE into SPAD_A) up to the first K block LD
        k = next((i for i, c in enumerate(h) if c.op == P.OP_LD), len(h))
        tiles.append(h[:k])
        scores.append(h[k:])
    pc = [c for c, _ in disp[num["probs"]]["cmds"]]
    pieces = segs(num["probs"], lambda i, c: c.op == P.OP_LD and (i == 0 or pc[i - 1].op != P.OP_LD))[1:]
    U = len(pieces)
    rows_per = len([c for c in K[num["pv"]] if c.gemv]) // U
    hpu = len(heads) // U
    ps = K[num["psum"]]
    red = [c for c in ps if c.op == P.OP_VE and c.kind == 4]
    ps_tail = [c for c in ps if not (c.op == P.OP_VE and c.kind == 4)]
    pvc = [c for c, _ in disp[num["pv"]]["cmds"]]
    pv = segs(num["pv"], lambda i, c: c.op == P.OP_LD and i + 1 < len(pvc) and pvc[i + 1].gemv)
    pv_pro, pv_rows = pv[0], pv[1:]
    assert len(heads) == U * hpu and len(pv_rows) == U * rows_per and len(red) == U * rows_per, \
        (len(heads), U, hpu, len(pv_rows), rows_per, len(red))

    def post(u):
        sw = bool(u & 1)
        cmds = [clone(c, sw) for c in pieces[u]]
        cmds += [clone(c, sw) for c in red[u * rows_per:(u + 1) * rows_per]]
        cmds += [clone(c, sw) for c in ps_tail + pv_pro]              # Σp per row: once per unit (conservative)
        for r in pv_rows[u * rows_per:(u + 1) * rows_per]:
            cmds += [clone(c, sw) for c in r]
        return cmds
    def assemble(interleave, blocked=False, nsub=0):
        part = [c for n in num["qpath"] for c in K[n]] + s_pro + [c for t_ in tiles for c in t_]
        for u in range(U):
            sc_u = [clone(c, bool(u & 1)) for h in range(u * hpu, (u + 1) * hpu) for c in scores[h]]
            prev = post(u - 1) if u else []
            if nsub and prev:
                # the T loop as k sub-loops, a k-th of unit u - 1's commands (whole) after each
                exs = [i for i, c in enumerate(sc_u) if c.op == P.OP_EX]
                nsub = min(nsub, len(exs))
                at = {exs[min(len(exs) - 1, (j + 1) * len(exs) // nsub - 1)]: j for j in range(nsub)}
                share = -(-len(prev) // nsub)
                out = []
                for i, c in enumerate(sc_u):
                    out.append(c)
                    if i in at:
                        j = at[i]
                        out += prev[j * share:(j + 1) * share]
                part += out + prev[nsub * share:]
            elif blocked and prev:
                # the T loop carrying unit u - 1's work: its row-long VEs and GEMVs cut into one piece per
                # K block of unit u, block j's pieces after the EX following block j's K load (one loop body)
                kl = [i for i, c in enumerate(sc_u) if c.op == P.OP_LD and c.w & (A0 | A1)]
                lds = [next((x for x in range(i, len(sc_u)) if sc_u[x].op == P.OP_EX), i) for i in kl]
                m = max(len(lds), 1)
                cut = [pieces_of(a, c, m) for c in prev]
                slots = [[x[j] for x in cut if len(x) == m] for j in range(m)]
                whole = [x[0] for x in cut if len(x) != m]
                out, j = [], 0
                for i, c in enumerate(sc_u):
                    out.append(c)
                    if i in lds:
                        if j == 0:
                            out += whole                # (the small ones: the max's broadcast, Σp per row)
                        out += slots[j]
                        j += 1
                part += out
            elif interleave and prev:
                # unit u - 1's commands spread over unit u's K blocks, one share after each block's EX
                exs = [i for i, c in enumerate(sc_u) if c.op == P.OP_EX]
                share = -(-len(prev) // max(len(exs), 1))
                out, k = [], 0
                for i, c in enumerate(sc_u):
                    out.append(c)
                    if c.op == P.OP_EX:
                        out += prev[k:k + share]
                        k += share
                part += out + prev[k:]
            else:
                part += sc_u + prev
        part += post(U - 1)
        part += [c for n in num["out"] for c in K[n]]
        return concat([part], fences=False)
    subs = {k: assemble(False, False, k) for k in (2, 4, 8)}
    return unfused, inorder, assemble(False), assemble(True), assemble(True, True), subs, dict(units=U, heads_per_unit=hpu, rows_per_unit=rows_per)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("build", help="a 2-layer build with sa_sources (test_long_context.py / test_hf_generic.py)")
    ap.add_argument("--t", default="48,512,2048")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--cal", default=os.path.join(REPO, "build/perfmodel/qwen3g_cal.json"))
    ap.add_argument("--dispatches", help="json {qpath: [...], scores: n, probs: n, psum: n, pv: n, out: [...]} (default Qwen3's)")
    ap.add_argument("--json")
    args = ap.parse_args()
    a = arch(args.cal)
    num = json.loads(args.dispatches) if args.dispatches else QWEN3
    d = int(os.environ.get("SA_D", "16"))
    flags = open(os.path.join(args.build, "compile_flags.txt")).read().split() \
        if os.path.exists(os.path.join(args.build, "compile_flags.txt")) else []
    order = list(num["qpath"]) + [num["scores"], num["probs"], num["psum"], num["pv"]] + list(num["out"])
    tmp = tempfile.mkdtemp(prefix="attn_schedule_")
    blobs = {}
    for n in order:
        src = os.path.join(args.build, "sa_sources", f"module_main$async_dispatch_{n}.mlir")
        one = os.path.join(tmp, f"d{n}.mlir")
        open(one, "w").write(G.sa_only(open(src).read()))
        blob, err = G.compile_one(one, flags, os.path.join(tmp, f"d{n}.sadesc"))
        if blob is None:
            print(f"#{n}: compile failed: {err}")
            return 1
        blobs[n] = (one, blob, indirect(src))
    res = []
    for t in [int(x) for x in args.t.split(",")]:
        disp, bad = {}, []
        for n in order:
            one, blob, ind = blobs[n]
            r, cmds = capture(a, one, blob, t, d)
            if not (r is True or (isinstance(r, str) and "OK" in r)):
                bad.append(f"#{n}: {r}")
            disp[n] = dict(cmds=cmds, indirect=ind)
        row = dict(T=t, check="OK" if not bad else "; ".join(bad))
        print(f"\n== T = {t}  (dispatch checks: {row['check']})")
        print(f"   {'list':34s} {'cycles':>9s} {'vs unfused':>10s}   {'LD':>7s} {'ST':>7s} {'EX':>7s} {'VE':>7s}  (busy, k)")
        base = None
        alt = dict(disp)
        alt[num["scores"]] = dict(disp[num["scores"]], cmds=alternate_tiles(disp[num["scores"]]["cmds"]))
        unf_a, ino_a, _, _, blk_a, subs_a, _ = build(alt, num, 1, a)
        for pb in (0, 1):
            unf, ino, sch, itl, blk, subs, info = build(disp, num, pb, a)
            lists = [("unfused", unf), ("in order (the bound's form)", ino)] if pb == 0 else []
            lists.append((f"scheduled, p in SPAD_A bank {pb}", sch))
            lists.append((f"interleaved, p in SPAD_A bank {pb}", itl))
            lists.append((f"blocked T loop, p in SPAD_A bank {pb}", blk))
            for k_, l_ in subs.items():
                lists.append((f"{k_} sub-loops, p in SPAD_A bank {pb}", l_))
            if pb == 1:
                lists += [("tiles alternating: unfused", unf_a), ("tiles alternating: in order", ino_a),
                          ("tiles alternating: 2 sub-loops", subs_a[2]), ("tiles alternating: blocked", blk_a)]
            for name, cmds in lists:
                cy, busy = P.run_list(a, cmds, gemv=True)
                base = base or cy
                print(f"   {name:34s} {cy / 1e3:8.1f}k {100 * (cy / base - 1):+9.1f}%   "
                      + " ".join(f"{b / 1e3:6.1f}k" for b in busy))
                row[name] = dict(cycles=cy, busy=busy)
        floor = max(row["unfused"]["busy"][0], row["unfused"]["busy"][3], row["unfused"]["busy"][2])
        best = min(row[f"{k}, p in SPAD_A bank {b}"]["cycles"] for k in ("scheduled", "interleaved", "blocked T loop", "2 sub-loops", "4 sub-loops", "8 sub-loops")
                   for b in (0, 1))
        impl = row["2 sub-loops, p in SPAD_A bank 1"]["cycles"]
        print(f"   implementable (2 sub-loops, p in bank 1): saved {(base - impl) / 1e3:.1f}k per layer, "
              f"{(base - impl) * args.layers / (a.mhz * 1e3):.1f} ms per token; with alternating tiles "
              f"{(base - row['tiles alternating: 2 sub-loops']['cycles']) / 1e3:.1f}k, "
              f"{(base - row['tiles alternating: 2 sub-loops']['cycles']) * args.layers / (a.mhz * 1e3):.1f} ms")
        saved = base - best
        print(f"   units {info['units']} ({info['heads_per_unit']} KV heads, {info['rows_per_unit']} query rows each); "
              f"busiest engine of the unfused commands {floor / 1e3:.1f}k")
        print(f"   saved per layer: in order {(base - row['in order (the bound' + chr(39) + 's form)']['cycles']) / 1e3:.1f}k, "
              f"scheduled {saved / 1e3:.1f}k cycles; x {args.layers} layers = {saved * args.layers / (a.mhz * 1e3):.1f} ms per token")
        res.append(row)
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

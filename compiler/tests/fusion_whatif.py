#!/usr/bin/env python3
"""Upper bound of cross-operator fusion (dispatches merged, intermediates kept
on chip) from the command-level performance model (perf_model.py).

A step's dispatches (one list each in the SA_PROFILE=1 trace) are joined into
one list as the batched runtime runs them, with a FENCE before a dispatch that
touches DDR an earlier dispatch since the last FENCE wrote, or writes what it
read (the barriers IREE places; their count is printed against the board's).
A dispatch's own FENCEs stay. Then, in the joined step:

  intermediate  a ST whose bytes a later dispatch of the same step reads back
                (LD), and that no later step reads before writing it again
                (transient buffers; the KV cache, the logits and the
                parameters are not intermediates)

Scenarios, cycles per prefill chunk (16 tokens) / per decode step:
  joined        the batched list as is (validates against the board's
                profile --batch device cycles)
  no barriers   the FENCEs between dispatches dropped, DDR traffic kept: what
                the barriers alone cost (not a legal schedule)
  on chip       intermediates' ST and the LDs that read only them dropped,
                barriers recomputed from the traffic left: fusion with
                unlimited local memory
  on chip, no barriers   both: every dispatch of a step fused (the ideal)
Every scenario keeps the dispatches' commands and engine timing otherwise, so
the gaps are upper bounds (no capacity limit, no extra local copies). --only
REGEX fuses one chain: only intermediates whose producer and consumer kinds
both match (e.g. the FFN's gate / up projections, the reduction and the
quantization of h).

    compiler/scripts/perf_trace.sh                      (DEPLOY / MODEL / OUT: the trace)
    compiler/tests/fusion_whatif.py --trace build/perfmodel/qwen3g.trace \\
        --board <bundle>/results/<commit>/<test>/<overlay>/profile/perf.csv [--cal cal.json]
"""
import argparse
import bisect
import dataclasses
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import perf_model as P  # noqa: E402


def ddr_range(r):
    """[lo, hi) of an LD / ST row of the trace (None for other commands)"""
    op = int(r[0]) & 0xFF
    if op not in (P.OP_LD, P.OP_ST):
        return None
    ddr, w2, w3 = int(r[1]), int(r[4]), int(r[5])
    rows, rb, pitch = P.field(w2, 32, 16), P.field(w2, 48, 16), P.field(w3, 0, 32)
    if rows == 0 or rb == 0:
        return None
    mode = P.field(w3, 32, 2) if op == P.OP_LD else 0
    if mode == 2 or pitch == rb:
        return ddr, ddr + rows * rb
    return ddr, ddr + (rows - 1) * pitch + rb


def load(path, arch):
    """-> steps: [[(export, [Cmd], [range or None])]]"""
    raw = np.fromfile(path, np.uint64).reshape(-1, 10)
    lists, k = [], 0
    for _, nrow, name in (l.split() for l in open(path + ".idx")):
        nrow = int(nrow)
        rows = raw[k:k + nrow]
        k += nrow
        lists.append((name, [P.decode(arch, r) for r in rows], [ddr_range(r) for r in rows]))
    steps, cur = [], []
    for item in lists:
        if re.search(r"\$async_dispatch_0_", item[0]) and cur:
            steps.append(cur)
            cur = []
        cur.append(item)
    if cur:
        steps.append(cur)
    return steps


class Intervals:
    """intervals [lo, hi) with an overlap query"""
    def __init__(self):
        self.lo, self.iv, self.maxlen = [], [], 0

    def add(self, lo, hi, tag=None):
        i = bisect.bisect(self.lo, lo)
        self.lo.insert(i, lo)
        self.iv.insert(i, (lo, hi, tag))
        self.maxlen = max(self.maxlen, hi - lo)

    def remove_within(self, lo, hi):
        """drop the intervals inside [lo, hi)"""
        i = bisect.bisect_left(self.lo, lo)
        j = bisect.bisect_left(self.lo, hi)
        keep = [x for x in self.iv[i:j] if x[1] > hi]
        self.iv[i:j] = keep
        self.lo[i:j] = [x[0] for x in keep]

    def overlapping(self, lo, hi):
        i = bisect.bisect_left(self.lo, lo - self.maxlen)
        j = bisect.bisect_left(self.lo, hi)
        return [x for x in self.iv[i:j] if x[0] < hi and lo < x[1]]

    def covered(self, lo, hi):
        """[lo, hi) covered by the union of the intervals"""
        xs = sorted(self.overlapping(lo, hi))
        at = lo
        for a, b, _ in xs:
            if a > at:
                return False
            at = max(at, b)
            if at >= hi:
                return True
        return at >= hi


def intermediates(steps, s, pairs=None):
    """indices (dispatch, command) of the intermediate STs of step s, and of the LDs reading only them;
    pairs: each dropped command -> (producer dispatch, consumer dispatch)"""
    if pairs is None:
        pairs = {}
    step = steps[s]
    st_at = {}
    read_back = set()
    readers = []                                      # (LD tag, range, the writers it read)
    live = Intervals()                                # the latest writer of each range so far
    for di, (_, cmds, rngs) in enumerate(step):
        for ci, (c, rg) in enumerate(zip(cmds, rngs)):
            if rg is None or c.gemv:
                continue
            if c.op == P.OP_LD:
                ws = [t for _, _, t in live.overlapping(*rg)]
                readers.append(((di, ci), rg, ws))
                read_back.update(t for t in ws if t[0] < di)
            elif c.op == P.OP_ST:
                live.remove_within(*rg)                   # (IREE reuses transient buffers: rewritten)
                live.add(rg[0], rg[1], (di, ci))
                st_at[(di, ci)] = rg
    # persistent: a later step reads the bytes before it writes them
    persistent = set()
    cand = Intervals()
    for tag in read_back:
        cand.add(*st_at[tag], tag)
    for t in range(s + 1, len(steps)):
        written = Intervals()
        for _, cmds, rngs in steps[t]:
            for c, rg in zip(cmds, rngs):
                if rg is None or c.gemv:
                    continue
                if c.op == P.OP_ST:
                    written.add(*rg)
                elif not written.covered(*rg):
                    persistent.update(tag for _, _, tag in cand.overlapping(*rg))
    inter = read_back - persistent
    drop = set(inter)
    pairs.clear()
    first_reader = {}
    for tag, _, ws in readers:
        for t in ws:
            if t in inter and t[0] < tag[0]:
                first_reader.setdefault(t, tag[0])
    for t, r in first_reader.items():
        pairs[t] = (t[0], r)
    for tag, (lo, hi), ws in readers:
        # an LD that reads only intermediates written by earlier dispatches, all of its bytes
        if ws and all(t in inter and t[0] < tag[0] for t in ws):
            iv = Intervals()
            for t in ws:
                iv.add(*st_at[t])
            if iv.covered(lo, hi):
                drop.add(tag)
                pairs[tag] = (max(t[0] for t in ws), tag[0])
    return drop


def never_written(steps):
    """the DDR ranges no ST of the trace writes: an Intervals of every ST (query: no overlap)"""
    iv = Intervals()
    for st in steps:
        for _, cmds, rngs in st:
            for c, rg in zip(cmds, rngs):
                if c.op == P.OP_ST and rg:
                    iv.add(*rg)
    return iv


def merge_consts(arch, step, written, max_bytes=4096):
    """the constant tail inputs batched: in each dispatch, LDs into ACC from DDR that nothing writes
    (the parameters: per-channel scales), small, following one another in DDR, become one LD at
    the first one's place. Returns (the new step, LDs removed)."""
    out, removed = [], 0
    for name, cmds, rngs in step:
        keep = [True] * len(cmds)
        chains = {}                                    # DDR end of a chain -> index of its head
        for i, (c, rg) in enumerate(zip(cmds, rngs)):
            if (c.op != P.OP_LD or c.gemv or not rg or rg[1] - rg[0] > max_bytes or not c.rng
                    or c.rng[0][0] != P.MEM_ACC or written.overlapping(*rg)):
                continue
            h = chains.pop(rg[0], None)
            if h is None:
                chains[rg[1]] = i
                continue
            head = cmds[h]
            head.bytes += c.bytes                      # (a fresh copy, below)
            head.rng = head.rng + c.rng
            head.w |= c.w
            keep[i] = False
            removed += 1
            chains[rg[1]] = h
        new_cmds, new_rngs = [], []
        for i, (c, rg) in enumerate(zip(cmds, rngs)):
            if keep[i]:
                new_cmds.append(c)
                new_rngs.append(rg)
        for c in new_cmds:                             # the merged heads: one transfer of all their bytes
            if c.op == P.OP_LD and not c.gemv and c.rng and len(c.rng) > 1:
                c.beats, c.bursts = P.dma_shape(arch, 0, 1, c.bytes, c.bytes, P.MEM_ACC)
        out.append((name, new_cmds, new_rngs))
    return out, removed


def copy_step(step):
    out = []
    for name, cmds, rngs in step:
        cc = []
        for c in cmds:
            c2 = P.Cmd()
            for f in P.Cmd.__slots__:
                setattr(c2, f, getattr(c, f))
            c2.rng = list(c.rng)
            cc.append(c2)
        out.append((name, cc, list(rngs)))
    return out


def fence_cmd(n):
    c = P.Cmd()
    c.op, c.eng, c.n, c.fence_before, c.mask = P.OP_FENCE, -1, n, 0, 0xF
    c.r = c.w = 0
    c.rng = []
    c.gemv = False
    return c


def join(step, drop=frozenset(), barriers=True):
    """one list for the step: FENCEs between dependent dispatches; dropped commands left out"""
    out, n = [], 0
    reads, writes = Intervals(), Intervals()          # DDR touched since the last FENCE
    fences = 0
    for di, (_, cmds, rngs) in enumerate(step):
        keep = [(c, rg) for ci, (c, rg) in enumerate(zip(cmds, rngs)) if (di, ci) not in drop]
        dep = False
        for c, rg in keep:
            if rg is None:
                continue
            if writes.overlapping(*rg) or (c.op == P.OP_ST and reads.overlapping(*rg)):
                dep = True
                break
        if dep and barriers and out:
            out.append(fence_cmd(n))
            n += 1
            fences += 1
            reads, writes = Intervals(), Intervals()
        for c, rg in keep:
            c2 = P.Cmd()
            for f in P.Cmd.__slots__:
                setattr(c2, f, getattr(c, f))
            c2.n = n
            n += 1
            out.append(c2)
            if c.op == P.OP_FENCE:
                reads, writes = Intervals(), Intervals()
            elif rg is not None and not c.gemv:        # (GEMV streams weights: never written)
                (writes if c.op == P.OP_ST else reads).add(*rg)
    return out, fences


def kind(name):
    """an export's kind: its name without the dispatch number (main$async_dispatch_12_matvec_like_... -> matvec_like_...)"""
    k = re.sub(r"^.*?\$async_dispatch_\d+_", "", name)
    return re.sub(r"_(i8|i32|i64|f32)(x(i8|i32|i64|f32))*$", "", k)


def by_pair(arch, step, drop, pairs, top):
    """the dropped commands' standalone cycles per (producer kind -> consumer kind)"""
    agg = {}
    for (di, ci) in drop:
        c = step[di][1][ci]
        p, q = pairs[(di, ci)]
        key = (kind(step[p][0]), kind(step[q][0]))
        n, cy = agg.get(key, (0, 0.0))
        agg[key] = (n + 1, cy + P.cost(arch, c))
    tot = sum(cy for _, cy in agg.values()) or 1
    print(f"   dropped commands by producer -> consumer (standalone cycles; {tot / 1e6:.2f}M in all):")
    for (a, b), (n, cy) in sorted(agg.items(), key=lambda x: -x[1][1])[:top]:
        print(f"     {100 * cy / tot:5.1f}%  {n:6d}  {a[:44]:44s} -> {b[:44]}")


def bytes_of(step, drop, op):
    return sum(c.bytes for di, (_, cmds, _) in enumerate(step) for ci, c in enumerate(cmds)
               if c.op == op and (di, ci) in drop)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", required=True)
    ap.add_argument("--board", required=True, help="board_profile.py csv (per dispatch): calibration")
    ap.add_argument("--cal", help="calibration json (perf_model.py --save-cal) instead of fitting")
    ap.add_argument("--save-cal")
    ap.add_argument("--prefill-seg", type=int, default=1)
    ap.add_argument("--gemv-ports", type=int, default=4)
    ap.add_argument("--logits", default=r"matvec_like_\d{4,}x", help="the vocabulary projection (left out of the prefill chunk)")
    ap.add_argument("--top", type=int, default=12, help="producer -> consumer pairs shown")
    ap.add_argument("--only", help="regex of dispatch kinds: fuse only the chains among them (e.g. one FFN)")
    ap.add_argument("--merge-consts", action="store_true",
                    help="also: no fusion, the constant tail inputs (per-channel scales) of each dispatch in one LD")
    ap.add_argument("--json")
    args = ap.parse_args()
    arch = P.Arch(gemv_ports=args.gemv_ports)
    steps = load(args.trace, arch)
    print("steps in trace:", [(len(s), s[0][0][:40]) for s in steps])
    segs = [[(n, c) for n, c, _ in s] for s in steps]
    pre = segs[args.prefill_seg]
    dec = segs[args.prefill_seg + 1]
    if args.cal:
        cal = json.load(open(args.cal))
        arch.ve_a, arch.ve_b, arch.f_desc, arch.list_ovh = tuple(cal["ve_a"]), tuple(cal["ve_b"]), cal["f_desc"], cal["list_ovh"]
    else:
        bp, _ = P.load_board(args.board, 0)
        bd, nd = P.load_board(args.board, 1)
        P.calibrate(arch, [(pre, bp), ([x for s in segs[args.prefill_seg + 1:] for x in s if not x[0].startswith("prefill")], bd)])
    if args.save_cal:
        json.dump(dict(ve_a=list(map(float, arch.ve_a)), ve_b=list(map(float, arch.ve_b)), f_desc=arch.f_desc,
                       list_ovh=arch.list_ovh), open(args.save_cal, "w"), indent=1)
    res = {}
    written = never_written(steps) if args.merge_consts else None
    for label, s in (("prefill chunk", args.prefill_seg), ("decode step", args.prefill_seg + 1)):
        step = steps[s]
        if label.startswith("prefill"):
            step = [x for x in step if not re.search(args.logits, x[0])]
        sub = [x for x in steps]
        sub[s] = step
        pairs = {}
        drop = intermediates(sub, s, pairs)
        if args.only:                                  # one chain: producer and consumer both of these kinds
            rx = re.compile(args.only)
            drop = {t for t in drop if rx.search(kind(step[pairs[t][0]][0])) and rx.search(kind(step[pairs[t][1]][0]))}
        per_list = sum(P.run_list(arch, cmds, gemv=False)[0] for _, cmds, _ in step)
        rows = []
        for name, dr, bar in (("joined", frozenset(), True), ("no barriers", frozenset(), False),
                              ("on chip", drop, True), ("on chip, no barriers", drop, False)):
            cmds, fences = join(step, dr, bar)
            cy, busy = P.run_list(arch, cmds)
            rows.append((name, cy, fences, busy))
        if args.merge_consts:                          # no fusion: the constant tail inputs in one LD each
            merged, nrem = merge_consts(arch, copy_step(step), written)
            cmds, fences = join(merged)
            cy, busy = P.run_list(arch, cmds)
            rows.append((f"constants batched (-{nrem} LD)", cy, fences, busy))
        ld_all = sum(c.bytes for _, cmds, _ in step for c in cmds if c.op == P.OP_LD)
        st_all = sum(c.bytes for _, cmds, _ in step for c in cmds if c.op == P.OP_ST)
        st_d, ld_d = bytes_of(step, drop, P.OP_ST), bytes_of(step, drop, P.OP_LD)
        nd_ = sum(1 for _ in step)
        n_ld = sum(1 for _, cmds, _ in step for c in cmds if c.op == P.OP_LD)
        n_st = sum(1 for _, cmds, _ in step for c in cmds if c.op == P.OP_ST)
        k_st = sum(1 for di, ci in drop if step[di][1][ci].op == P.OP_ST)
        k_ld = len(drop) - k_st
        print(f"\n== {label}: {nd_} dispatches, LD {ld_all / 1e6:.2f} MB ({n_ld} commands), ST {st_all / 1e6:.2f} MB "
              f"({n_st}); intermediates: ST {st_d / 1e6:.2f} MB, {k_st} commands ({100 * st_d / max(st_all, 1):.0f}% "
              f"of ST bytes), LD reading them back {ld_d / 1e6:.2f} MB, {k_ld} commands ({100 * ld_d / max(ld_all, 1):.1f}% "
              f"of LD bytes)")
        print(f"   one list per dispatch (calibration form): {per_list / 1e6:.2f}M cycles")
        base = rows[0][1]
        print(f"   {'scenario':30s} {'cycles':>9s} {'ms':>8s} {'vs joined':>9s} {'FENCEs':>7s}  {'LD busy':>8s} {'EX busy':>8s} {'VE busy':>8s}")
        for name, cy, fences, busy in rows:
            print(f"   {name:30s} {cy / 1e6:8.2f}M {cy / arch.mhz / 1e3:8.1f} {100 * (cy / base - 1):+8.1f}% {fences:7d}  "
                  f"{busy[0] / cy * 100:7.1f}% {busy[2] / cy * 100:7.1f}% {busy[3] / cy * 100:7.1f}%")
        by_pair(arch, step, drop, pairs, args.top)
        res[label] = dict(dispatches=nd_, ld=ld_all, st=st_all, st_inter=st_d, ld_inter=ld_d, per_list=per_list,
                          scenarios={n: dict(cycles=cy, fences=f) for n, cy, f, _ in rows})
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()

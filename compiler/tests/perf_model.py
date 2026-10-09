#!/usr/bin/env python3
"""Command-level performance model of the sa unit (docs/kv260_upgrade_plan.md
K2b step 1: choose the DMA / local-memory / array architecture before RTL).

Input: a command trace of a real model run on the functional simulator
(compiler/sim/sa_sim_server.py --trace, one list per dispatch with
SA_PROFILE=1: every LD / ST / EX / VE / FENCE as executed, dynamic fields and
BASE applied, with the export name of its list), and the board's per-export
event counters (board_profile.py, SA_PROFILE_PERF csv) for calibration.

Timing rules follow the RTL (rtl/sysarray):
  front end   descriptors fetched at f_desc cycles each (sa_cmdfetch: 64-byte
              descriptors, a few bursts in flight), FENCE / FENCE_BEFORE wait
              for the engines; per list a fixed overhead (rt_fw, prefix / head
              calls, END, completion record)
  scheduler   program order; a command waits while another engine has an
              in-flight command with a conflicting bank access (6 banks:
              SPAD_A / SPAD_B / ACC halves; EX and VE (and LD and ST) share ports, so any
              common bank); engine queues of 4, in-flight masks of 8 (sa_sched)
  LD / ST     one command at a time; fixed latency + max(beats, bursts x
              c_burst); bursts <= 16 beats, split at 4 KB; a 128-bit beat whose
              two lanes do not land in one word costs 2 write cycles (sa_ld)
  EX          repeat x (K + 2(D-1)) streaming steps; the drain overlaps the
              next command (sa_ex)
  VE          cycles = a_class x groups + b_class per command, classes: integer,
              transpose, fp elementwise, fp function (batched SFU), fp reduce,
              fp reduce of a function; a, b fitted to the board's VE_ACTIVE

    MODEL=c6p_qwen3 MB=640 OUT=qwen3 compiler/scripts/perf_trace.sh          (the trace, ~5 min)
    board: board_profile.py 40 --prompt 32 SA_PROFILE_PERF=...csv            (section 0 prefill chunk 2, 1 decode)
    $SA_PY compiler/tests/perf_model.py --trace build/perfmodel/qwen3.trace --board c6p_qwen3_dec_perf.csv \\
        [--save-cal cal.json | --cal cal.json]
Prints the calibration, model vs board per export, and the what-if table (SCENARIOS).
The board's profile table keeps 512 exports, so some attention exports have no board row;
the what-if totals use the whole trace.
"""
import argparse
import collections
import dataclasses
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from z1_profile import sections  # noqa: E402

OP_LD, OP_ST, OP_EX, OP_VE, OP_FENCE = 0x01, 0x02, 0x03, 0x04, 0x10
ENG = {OP_LD: 0, OP_ST: 1, OP_EX: 2, OP_VE: 3}
MEM_SPAD_A, MEM_SPAD_B, MEM_ACC = 1, 2, 3
VE_CLASSES = ["int", "transpose", "fp", "fp func", "fp reduce", "fp reduce func"]


@dataclasses.dataclass
class Arch:
    """The hardware being modelled (defaults: d16_100mhz_w128_p6_red, the calibrated baseline)."""
    d: int = 16
    mhz: float = 100.0
    ld_bpc: float = 16.0        # LD bytes per cycle into the local memories (DMA_W / 8 x ports, banks)
    st_bpc: float = 16.0
    ld_lat: float = 65.0        # per LD command: issue + DDR latency + last write (bwtest: 4161 cycles for 64 KB)
    st_lat: float = 70.0
    c_burst: float = 2.63       # cycles per burst issue (bwtest: 8-byte rows, 1-beat bursts)
    ld_overlap: bool = False    # LD commands overlap (the next one's bursts issue under the latency)
    f_desc: float = 4.0         # front end: cycles per descriptor
    list_ovh: float = 1500.0    # per list (rt_fw)
    spad_words: int = 8192
    acc_words: int = 4096
    ve_a: tuple = (1, 2, 2, 32, 4, 32)     # per class: cycles per group, cycles per command
    ve_b: tuple = (8, 8, 8, 40, 40, 60)
    ex_scale: float = 1.0       # EX steps scale (D = 32 for the same work: about 1 / 2)
    gemv_bpc: float = 0.0       # H4: decode GEMV unit, weight bytes per cycle (0: none)
    ooo: bool = False           # H2: a blocked command does not block later independent ones (per-engine windows)
    gemv_ports: int = 2         # LD mode GEMV (K2b streaming unit): 16 B / cycle per read port


def field(x, lo, n):
    return (int(x) >> lo) & ((1 << n) - 1)


class Cmd:
    __slots__ = ("op", "eng", "n", "fence_before", "mask", "r", "w", "kind", "groups", "bytes", "beats", "bursts",
                 "steps", "wbytes", "gemv", "rng", "xw", "k")


def banks(arch, mem, first, last):
    if mem == MEM_SPAD_A:
        s, h = 0, arch.spad_words // 2
    elif mem == MEM_SPAD_B:
        s, h = 2, arch.spad_words // 2
    elif mem == MEM_ACC:
        s, h = 4, arch.acc_words // 2
    else:
        return 0
    m = 0
    if first < h:
        m |= 1 << s
    if last >= h:
        m |= 2 << s
    return m


def dma_shape(arch, ddr, rows, rb, pitch, mem):
    """(beats incl. split-lane extra cycles, bursts) of an LD / ST at 128 bits"""
    if rows == 0 or rb == 0:
        return 0, 0
    if pitch == rb:                                   # contiguous rows: one long transfer
        rows, rb = 1, rows * rb
    a = ddr & 15
    beats = (a + rb + 15) // 16
    # bursts: <= 16 beats (256 B), split at 4 KB; for a row of rb bytes from ddr
    first = ddr & ~15
    last = (ddr + rb - 1) & ~15
    pages = (last >> 12) - (first >> 12) + 1
    bursts = (beats + 15) // 16 + pages - 1
    wpb = 4 * arch.d if mem == MEM_ACC else arch.d     # bytes per local word
    split = 1 if (a & 8 or wpb < 16) else 0          # beats whose lanes go one per cycle
    extra = beats * split
    if pitch % 16:
        extra = beats // 2 if not split else extra    # every other row misaligned
    return rows * (beats + extra), rows * bursts


def decode_trace(path, arch, names=None):
    """-> [(export name, [Cmd])] per list"""
    raw = np.fromfile(path, np.uint64).reshape(-1, 10)
    idx = [l.split() for l in open(path + ".idx")]
    out, k = [], 0
    for run, nrow, name in idx:
        nrow = int(nrow)
        cmds = [decode(arch, raw[k + i]) for i in range(nrow)]
        k += nrow
        out.append((name, cmds))
    return out


def decode(arch, r):
    hdr, ddr, n = int(r[0]), int(r[1]), int(r[2])
    w = [0, int(r[3]), int(r[4]), int(r[5]), int(r[6]), int(r[7]), int(r[8]), int(r[9])]
    c = Cmd()
    c.op = hdr & 0xFF
    c.eng = ENG.get(c.op, -1)
    c.n = n
    c.fence_before = (hdr >> 11) & 1
    c.mask = w[1] & 0xF if c.op == OP_FENCE else 0
    c.r = c.w = 0
    c.kind = -1
    c.groups = c.bytes = c.beats = c.bursts = c.steps = c.wbytes = 0
    c.gemv = False
    c.rng = []                                         # (mem, first word, last word, write): H2's fine-grained check
    c.xw = c.k = 0
    d = arch.d
    if c.op in (OP_LD, OP_ST):
        laddr = field(w[2], 0, 32)
        mem, word = (laddr >> 28) & 0xF, laddr & 0xFFFF
        rows, rb, pitch = field(w[2], 32, 16), field(w[2], 48, 16), field(w[3], 0, 32)
        mode = field(w[3], 32, 2) if c.op == OP_LD else 0
        if mode == 2:                                  # GEMV: x in SPAD_A, one ACC word per strip
            c.gemv, c.xw, c.k = True, field(w[3], 34, 16), rb // d
            cstep = field(w[3], 50, 8)
            c.r = banks(arch, MEM_SPAD_A, c.xw, c.xw + c.k // d - 1)
            c.w = banks(arch, MEM_ACC, word, word + (rows - 1) * cstep)
            c.rng = [(MEM_SPAD_A, c.xw, c.xw + c.k // d - 1, False), (MEM_ACC, word, word + (rows - 1) * cstep, True)]
            c.bytes, c.steps = rows * rb, rows
            c.beats, c.bursts = dma_shape(arch, ddr, 1, rows * rb, rows * rb, MEM_SPAD_B)
            return c
        wb = 4 * d if mem == MEM_ACC else d
        words = (rb // d) * rows if mode else ((rb + wb - 1) // wb) * rows
        m = banks(arch, mem, word, word + max(words, 1) - 1)
        c.rng.append((mem, word, word + max(words, 1) - 1, c.op == OP_LD))
        if c.op == OP_LD:
            c.w = m
        else:
            c.r = m
        c.bytes = rows * rb
        c.beats, c.bursts = dma_shape(arch, ddr, rows, rb, pitch, mem)
    elif c.op == OP_EX:
        a, b, cc, kt, acc = field(w[1], 0, 16), field(w[1], 16, 16), field(w[1], 32, 16), field(w[1], 48, 12), field(w[1], 60, 1)
        rep = field(w[2], 0, 12) or 1
        bstep, cstep, crow = field(w[2], 16, 16), field(w[2], 32, 16), field(w[2], 48, 16) or 1
        c.r = banks(arch, MEM_SPAD_A, a, a + kt * d - 1) | banks(arch, MEM_SPAD_B, b, b + kt * d - 1 + (rep - 1) * bstep)
        c.w = banks(arch, MEM_ACC, cc, cc + (rep - 1) * cstep + (d - 1) * crow)
        c.rng += [(MEM_SPAD_A, a, a + kt * d - 1, False), (MEM_SPAD_B, b, b + kt * d - 1 + (rep - 1) * bstep, False),
                  (MEM_ACC, cc, cc + (rep - 1) * cstep + (d - 1) * crow, True)]
        if acc:
            c.r |= c.w
        c.steps = rep * (kt * d + 2 * (d - 1))
        c.wbytes = rep * kt * d * d                    # B bytes streamed (the weights)
    elif c.op == OP_VE:
        src1, src2, dst = field(w[1], 0, 32), field(w[1], 32, 32), field(w[2], 0, 32)
        g = field(w[2], 32, 32) >> (d.bit_length() - 1)
        op_byte, types, period = field(w[3], 0, 8), field(w[3], 8, 6), field(w[3], 16, 16)
        flags = field(w[3], 53, 11)
        rowlen = field(w[7], 0, 16)
        fp, func, mo1, mo2, red = flags & 1, (flags >> 1) & 7, (flags >> 4) & 3, (flags >> 6) & 3, (flags >> 8) & 3
        op = op_byte & 7
        if op == 6:
            c.kind = 1
        elif not fp:
            c.kind = 0
        elif red:
            c.kind = 5 if func in (1, 2, 3) else 4
        else:
            c.kind = 3 if func in (1, 2, 3) else 2
        c.groups = max(g, 1)

        def span(addr, mo):
            m, wd = (addr >> 28) & 0xF, addr & 0xFFFF
            if mo == 3:
                return 0
            n_ = period if (mo == 1 and period) else (c.groups // max(period, 1) + 1 if mo == 2 else c.groups)
            c.rng.append((m, wd, wd + max(n_, 1) - 1, False))
            return banks(arch, m, wd, wd + max(n_, 1) - 1)
        unary = op == 5
        c.r = span(src1, mo1 if fp else 0) | (0 if unary else span(src2, mo2 if fp else (1 if period else 0)))
        nd = (c.groups // rowlen if rowlen else 1) if red else c.groups
        c.w = banks(arch, (dst >> 28) & 0xF, dst & 0xFFFF, (dst & 0xFFFF) + max(nd, 1) - 1)
        c.rng.append(((dst >> 28) & 0xF, dst & 0xFFFF, (dst & 0xFFFF) + max(nd, 1) - 1, True))
    return c


def conflict(a, b):
    """address-range conflict (RAW, WAR, WAW) between two commands"""
    for (m1, l1, h1, w1) in a.rng:
        for (m2, l2, h2, w2) in b.rng:
            if m1 == m2 and (w1 or w2) and l1 <= h2 and l2 <= h1:
                return True
    return False


def cost(arch, c, gemv=False):
    if c.op == OP_LD and c.gemv:                       # streamed into the unit; strips written back
        return arch.ld_lat + max(c.beats / max(arch.gemv_ports, 1), arch.c_burst * c.bursts / max(arch.gemv_ports, 1)) \
            + 2 * c.steps
    if c.op == OP_LD:
        bpc = arch.ld_bpc / 16.0                       # (ports: bursts dealt round-robin, issue per port)
        return arch.ld_lat + max(c.beats / bpc, arch.c_burst * c.bursts / bpc)
    if c.op == OP_ST:
        bpc = arch.st_bpc / 16.0
        return arch.st_lat + max(c.beats / bpc, arch.c_burst * c.bursts / bpc)
    if c.op == OP_EX:
        if gemv and arch.gemv_bpc:
            return c.wbytes / arch.gemv_bpc + 16
        return c.steps * arch.ex_scale
    if c.op == OP_VE:
        return arch.ve_a[c.kind] * c.groups + arch.ve_b[c.kind]
    return 0


def xload(c, xtag, d):
    """GEMV: x read from SPAD_A (K / D words) unless resident; any write to SPAD_A drops it"""
    if c.op == OP_LD and c.gemv:
        if xtag[0] == (c.xw, c.k):
            return 0
        xtag[0] = (c.xw, c.k)
        return c.k // d
    if c.w & 3:
        xtag[0] = None
    return 0


def run_list(arch, cmds, gemv=False):
    """-> (cycles, busy per engine [LD, ST, EX, VE]) of one list"""
    eng_free = [0.0] * 4                    # engine idle from
    xtag = [None]                           # the GEMV unit's resident x (word, K)
    hist = [[] for _ in range(4)]           # per engine: (dispatch, start, done, r, w)
    busy = [0.0] * 4
    t_disp = 0.0                            # last dispatch
    disp_hist = []
    last_done = 0.0
    for i, c in enumerate(cmds):
        tf = c.n * arch.f_desc
        if len(disp_hist) >= 10:            # input FIFO (8) + d1 + d2
            tf = max(tf, disp_hist[-10])
        if c.op == OP_FENCE or c.fence_before:
            tf = max(tf, last_done)
        if c.op == OP_FENCE:
            disp_hist.append(max(tf, t_disp))
            continue
        e = c.eng
        t = max(tf, t_disp + 1)
        h = hist[e]
        if len(h) >= 4:
            t = max(t, h[-4][1])             # engine queue of 4: the 4th previous has started
        if len(h) >= 8:
            t = max(t, h[-8][2])             # in-flight mask queue of 8
        while True:                          # bank hazards against other engines' in-flight commands
            t0 = t
            for o in range(4):
                if o == e:
                    continue
                share = {e, o} in ({2, 3}, {0, 1})
                for (_, _, dn, r, w) in reversed(hist[o][-8:]):
                    if dn <= t:
                        continue
                    if (c.r & w) or (c.w & r) or (c.w & w) or (share and ((c.r | c.w) & (r | w))):
                        t = max(t, dn)
            if t == t0:
                break
        t_disp = t
        disp_hist.append(t)
        cy = cost(arch, c, gemv) + xload(c, xtag, arch.d)
        start = max(t + 1, eng_free[e])
        if c.op == OP_LD and arch.ld_overlap and h:
            start = max(t + 1, eng_free[e] - arch.ld_lat)
        done = start + cy
        eng_free[e] = done
        if c.op == OP_EX:                    # the drain overlaps the next command's streaming
            done += 2 * arch.d
        busy[e] += cy
        h.append((t, start, done, c.r, c.w))
        last_done = max(last_done, done)
    return last_done + arch.list_ovh, busy


def run_list_ooo(arch, cmds, gemv=False, window=16):
    """H2: per-engine issue in order, but a command waits only for earlier commands (any engine,
    dispatched or not) whose address ranges conflict; EX / VE (LD / ST) still share their ports.
    The scheduler looks at most `window` commands past the oldest undispatched one."""
    eng_free = [0.0] * 4
    xtag = [None]
    last_start = [0.0] * 4
    busy = [0.0] * 4
    done_t = []
    disp_t = []
    last_done = 0.0
    fence_t = 0.0
    live = []                                # indices of earlier commands (bounded look-back)
    for i, c in enumerate(cmds):
        tf = c.n * arch.f_desc
        if len(disp_t) >= window:
            tf = max(tf, sorted(disp_t[-window:])[0])
        if c.op == OP_FENCE or c.fence_before:
            fence_t = max(fence_t, last_done)
        tf = max(tf, fence_t)
        if c.op == OP_FENCE:
            disp_t.append(tf)
            done_t.append(tf)
            live.append(None)
            continue
        e = c.eng
        t = max(tf, last_start[e])
        for j in range(max(0, i - 64), i):
            p = cmds[j]
            if p.op == OP_FENCE or p.eng == e or done_t[j] <= t:
                continue
            share = {e, p.eng} in ({2, 3}, {0, 1})
            if conflict(c, p) or (share and ((c.r | c.w) & (p.r | p.w))):
                t = max(t, done_t[j])
        cy = cost(arch, c, gemv) + xload(c, xtag, arch.d)
        start = max(t + 1, eng_free[e])
        if c.op == OP_LD and arch.ld_overlap:
            start = max(t + 1, eng_free[e] - arch.ld_lat)
        done = start + cy
        eng_free[e] = done
        last_start[e] = start
        if c.op == OP_EX:
            done += 2 * arch.d
        busy[e] += cy
        disp_t.append(t)
        done_t.append(done)
        last_done = max(last_done, done)
    return last_done + arch.list_ovh, busy


def segments(lists):
    """split the trace into steps: a step starts at an export *_async_dispatch_0_*"""
    segs, cur = [], []
    for name, cmds in lists:
        if re.search(r"\$async_dispatch_0_", name) and cur:
            segs.append(cur)
            cur = []
        cur.append((name, cmds))
    if cur:
        segs.append(cur)
    return segs


def board_rows(path, section=0):
    per_step, rows = sections(path)[section]
    return {r["export"]: r for r in rows}, per_step


def fit_ve(arch, pairs):
    """least squares of a_class, b_class to the board's VE_ACTIVE per export, over (seg, board) pairs"""
    feats, ys = [], []
    for seg, board in pairs:
        agg = collections.defaultdict(lambda: np.zeros(12))
        calls = collections.Counter()
        for name, cmds in seg:
            calls[name] += 1
            for c in cmds:
                if c.op == OP_VE:
                    agg[name][c.kind] += c.groups
                    agg[name][6 + c.kind] += 1
        for name, f in agg.items():
            if name in board:
                b = board[name]
                w = np.sqrt(b["calls"] / calls[name] * calls[name])      # weight ~ sqrt(calls)
                feats.append(f / calls[name] * w)
                ys.append(b["c24"] / b["calls"] * w)
    A, y = np.array(feats), np.array(ys)
    x = nnls(A, y)
    return x, y, A @ x


def nnls(A, y, iters=3000):
    """non-negative least squares (projected gradient; small problems)"""
    scale = np.maximum(A.max(axis=0), 1e-9)
    As = A / scale
    x = np.linalg.lstsq(As, y, rcond=None)[0].clip(0)
    L = np.linalg.norm(As, 2) ** 2
    for _ in range(iters):
        x = (x - As.T @ (As @ x - y) / L).clip(0)
    return x / scale


def model_seg(arch, seg, gemv_names=()):
    """-> {export: (calls, [cycles, LD, ST, EX, VE busy] summed)}"""
    tot = collections.defaultdict(lambda: np.zeros(5))
    calls = collections.Counter()
    for name, cmds in seg:
        cy, busy = (run_list_ooo if arch.ooo else run_list)(arch, cmds, gemv=name in gemv_names)
        calls[name] += 1
        tot[name] += np.array([cy, *busy])
    return {k: (calls[k], v) for k, v in tot.items()}


def fit_frontend(arch, pairs, grid=(0, 1, 2, 3, 4, 6, 8, 12, 16, 24)):
    """f_desc (grid) and list_ovh (least squares per call) against the board's cycles per export"""
    best = None
    for f in grid:
        a = dataclasses.replace(arch, f_desc=f, list_ovh=0.0)
        res, wts, rows = [], [], []
        for seg, board in pairs:
            for name, (k, v) in model_seg(a, seg).items():
                b = board.get(name)
                if b:
                    res.append(b["cycles"] / b["calls"] - v[0] / k)
                    wts.append(b["calls"])
                    rows.append((b["cycles"] / b["calls"], v[0] / k, b["calls"]))
        ovh = max(0.0, float(np.average(res, weights=wts)))
        err = sum(n * abs(mc + ovh - bc) for bc, mc, n in rows) / sum(n * bc for bc, mc, n in rows)
        if best is None or err < best[0]:
            best = (err, f, ovh)
    return best


def compare(arch, seg, board, label, per_step=1.0, gemv_names=(), top=25):
    m = model_seg(arch, seg, gemv_names)
    print(f"== {label}")
    print(f"{'export':58s} {'calls':>5s} {'board':>9s} {'model':>9s} {'err':>6s}  {'LDbusy b/m':>17s} {'EX b/m':>15s} {'VE b/m':>15s}")
    sb = sm = 0
    rows = []
    for name, (k, v) in m.items():
        b = board.get(name)
        if not b:
            continue
        n = b["calls"]
        bc, mc = b["cycles"] / n, v[0] / k
        sb += b["cycles"]
        sm += mc * n
        rows.append((b["cycles"], name, n, bc, mc, b["c18"] / n, v[1] / k, b["c14"] / n, v[3] / k, b["c24"] / n, v[4] / k))
    for r in sorted(rows, reverse=True)[:top]:
        _, name, n, bc, mc, lb, lm, eb, em, vb, vm = r
        print(f"{name[:58]:58s} {n:5d} {bc:9.0f} {mc:9.0f} {100 * (mc / bc - 1):+5.0f}%  {lb:8.0f}/{lm:<8.0f} {eb:7.0f}/{em:<7.0f} {vb:7.0f}/{vm:<7.0f}")
    print(f"total per step: board {sb / per_step / 1e6:.2f}M, model {sm / per_step / 1e6:.2f}M ({100 * (sm / sb - 1):+.1f}%)")
    return sb, sm


def load_board(path, section):
    rows, per_step = board_rows(path, section)
    for b in rows.values():
        b["calls"] = int(round(b["calls"]))
    return rows, per_step


def calibrate(arch, pairs):
    x, y, pred = fit_ve(arch, pairs)
    arch.ve_a, arch.ve_b = tuple(x[:6]), tuple(x[6:])
    print("VE fit:", ", ".join(f"{k}: {a:.2f}/g + {b:.0f}" for k, a, b in zip(VE_CLASSES, x[:6], x[6:])),
          f"r = {np.corrcoef(y, pred)[0, 1]:.4f}")
    err, f, ovh = fit_frontend(arch, pairs)
    arch.f_desc, arch.list_ovh = f, ovh
    print(f"front end fit: f_desc {f} cycles / descriptor, list overhead {ovh:.0f} cycles, weighted |err| {100 * err:.1f}%")


SCENARIOS = [
    # label, Arch changes, GEMV unit used in decode
    ("baseline (d16_100mhz_w128_p6_red)", {}, False),
    ("LD 32 B/cycle (2 ports + 2 sub-banks)", dict(ld_bpc=32, st_bpc=32), False),
    ("H4 GEMV 32 B/cycle, LD 16", dict(gemv_bpc=32), True),
    ("LD 32 + H4 GEMV 32", dict(ld_bpc=32, st_bpc=32, gemv_bpc=32), True),
    ("LD 48 + H4 GEMV 48", dict(ld_bpc=48, st_bpc=48, gemv_bpc=48), True),
    ("LD 64 + H4 GEMV 64", dict(ld_bpc=64, st_bpc=64, gemv_bpc=64), True),
    ("LD 32 + D = 32 array (EX steps / 2)", dict(ld_bpc=32, st_bpc=32, ex_scale=0.5), False),
    ("LD 32 + GEMV 32 + LD commands pipelined", dict(ld_bpc=32, st_bpc=32, gemv_bpc=32, ld_overlap=True), True),
    ("LD 32 + GEMV 32 + descriptor fetch 4 cyc", dict(ld_bpc=32, st_bpc=32, gemv_bpc=32, f_desc=4.0), True),
    ("H2 out-of-order + range checks (baseline)", dict(ooo=True), False),
    ("LD 32 + GEMV 32 + H2", dict(ld_bpc=32, st_bpc=32, gemv_bpc=32, ooo=True), True),
    ("LD 64 + GEMV 64 + pipelined LD + fetch 4 + H2", dict(ld_bpc=64, st_bpc=64, gemv_bpc=64, ld_overlap=True,
                                                          f_desc=4.0, ooo=True), True),
]


def total(arch, seg, gemv_names=()):
    m = model_seg(arch, seg, gemv_names)
    return sum(v for _, v in m.values())


def whatif(arch, pre, dec, ndec):
    dec_names = {n for n, _ in dec}
    base = None
    print("== what-if (100 MHz; cycles per prefill chunk of 16 tokens / per decode step)")
    print(f"{'scenario':48s} {'prefill':>9s} {'tok/s':>7s} {'decode':>9s} {'tok/s':>6s}  {'dec LD':>7s} {'dec EX':>7s} {'dec VE':>7s}")
    for label, ch, gemv in SCENARIOS:
        a = dataclasses.replace(arch, **ch)
        tp = total(a, pre)
        td = total(a, dec, dec_names if gemv else ()) / ndec
        if base is None:
            base = (tp[0], td[0])
        print(f"{label:48s} {tp[0] / 1e6:8.2f}M {16 * a.mhz * 1e6 / tp[0]:7.2f} {td[0] / 1e6:8.2f}M {a.mhz * 1e6 / td[0]:6.2f}  "
              f"{td[1] / 1e6:6.2f}M {td[3] / 1e6:6.2f}M {td[4] / 1e6:6.2f}M   x{base[0] / tp[0]:.2f} / x{base[1] / td[0]:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", required=True, help="sim trace: prefill chunks, then decode steps")
    ap.add_argument("--board", required=True, help="board_profile.py csv: section 0 one prefill chunk, section 1 decode")
    ap.add_argument("--prefill-seg", type=int, default=1, help="trace step that matches section 0")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--logits", default=r"matvec_like_\d{4,}x", help="the vocabulary projection (left out of the prefill chunk)")
    ap.add_argument("--cal", help="load the calibration (VE, front end) from this json instead of fitting")
    ap.add_argument("--save-cal", help="write the fitted calibration here")
    args = ap.parse_args()
    arch = Arch()
    segs = segments(decode_trace(args.trace, arch))
    print("steps in trace:", [(len(s), s[0][0][:40]) for s in segs])
    pre = segs[args.prefill_seg]
    pre_chunk = [x for x in pre if not re.search(args.logits, x[0])]     # per chunk: without the logits
    dec = [x for s in segs[args.prefill_seg + 1:] for x in s if not x[0].startswith("prefill")]
    bp, _ = load_board(args.board, 0)
    bd, nd = load_board(args.board, 1)
    if args.cal:
        import json
        cal = json.load(open(args.cal))
        arch.ve_a, arch.ve_b, arch.f_desc, arch.list_ovh = tuple(cal["ve_a"]), tuple(cal["ve_b"]), cal["f_desc"], cal["list_ovh"]
        print(f"calibration from {args.cal}")
    else:
        calibrate(arch, [(pre, bp), (dec, bd)])
    if args.save_cal:
        import json
        json.dump(dict(ve_a=list(map(float, arch.ve_a)), ve_b=list(map(float, arch.ve_b)), f_desc=arch.f_desc,
                       list_ovh=arch.list_ovh), open(args.save_cal, "w"), indent=1)
    ndec = len(segs) - args.prefill_seg - 1
    compare(arch, pre, bp, "prefill chunk", top=args.top)
    compare(arch, dec, bd, "decode step", per_step=nd, top=args.top)
    whatif(arch, pre_chunk, dec, ndec)


if __name__ == "__main__":
    main()

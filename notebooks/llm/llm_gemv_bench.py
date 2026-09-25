#!/usr/bin/env python3
"""LLM step 1: the decode GEMVs of the TinyStories models on the accelerator, as is.

Decode = per token, every weight matrix times one activation vector (GEMV,
int8 here). Mapped onto the unchanged m5 overlay as C = W @ X with the
weights W (out x in) streamed as A strips and X (in x D) resident in SPAD_B:
column 0 is the token's activation, the other D - 1 columns are free, so the
same run also shows batch-D decode (D sequences share one pass over W).

For every distinct shape of stories15M / 42M / 110M: result vs NumPy, cycles
with gemm_fw (PCPI) and as one descriptor list, weight bytes per cycle, and
the counter breakdown (is the LD DMA at ~8 B/cycle the limit?). Then per
model: GEMV time per token, the resulting upper bound on tokens/s (GEMVs
only: no attention, norms, softmax or ARM-side work), and the bounds at 8
and 16 weight bytes per cycle. If arm_baseline.txt (arm_baseline.sh) is in
the directory, llama2.c's measured ARM tokens/s are printed next to it.

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \\
                  && cd /home/xilinx/llm && python3 llm_gemv_bench.py'

Files next to this script: picorv32.bit/.hwh (m5), gemm_fw.bin,
desc_run_fw.bin, pynq_matmul.py (staged by prepare_llm.sh).
"""
import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import RISCV_HZ, MatmulOverlay, format_breakdown, perf_breakdown  # noqa: E402

# name: dim, hidden_dim, layers, vocab (llama2.c TinyStories checkpoints; n_kv_heads = n_heads)
MODELS = {"stories15M": (288, 768, 6, 32000), "stories42M": (512, 1376, 8, 32000),
          "stories110M": (768, 2048, 12, 32000)}


def gemvs(dim, hidden, layers, vocab):
    """[(out, in, count per token, what)] of one decode step."""
    return [(dim, dim, 4 * layers, "wq wk wv wo"), (hidden, dim, 2 * layers, "w1 w3"),
            (dim, hidden, layers, "w2"), (vocab, dim, 1, "classifier")]


def check(w, x, c):
    """c == w @ x in int32, in row chunks (the classifier is 32000 rows)."""
    xi = x.astype(np.int32)
    return all(np.array_equal(c[r:r + 2048], w[r:r + 2048].astype(np.int32) @ xi)
               for r in range(0, w.shape[0], 2048))


def arm_baseline(path):
    """{model file: (1 thread, 2 threads) tok/s} from arm_baseline.txt (the best
    build per model: runq or runq_gs32 for Q8_0), or {}."""
    res = {}
    try:
        for line in open(path):
            f = line.split()
            if len(f) == 4 and f[0].endswith(".bin") and re.match(r"[\d.]+$", f[2]):
                res[f[0]] = max(res.get(f[0], (0.0, 0.0)), (float(f[2]), float(f[3])), key=lambda v: v[1])
    except OSError:
        pass
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated subset of " + ",".join(MODELS))
    ap.add_argument("--breakdown", action="store_true", help="print the full counter breakdown per shape")
    args = ap.parse_args()

    mm = MatmulOverlay(args.bit, os.path.join(HERE, "gemm_fw.bin"))
    d = mm.d
    rng = np.random.default_rng(1)
    ok = True
    print(f"overlay: D = {d}, {mm.nports} DMA port(s), {RISCV_HZ / 1e6:.0f} MHz; "
          f"EX takes at most {d} weight bytes per cycle")

    models = [m for m in args.models.split(",") if m]
    shapes = sorted({(o, i) for m in models for o, i, _, _ in gemvs(*MODELS[m])})
    best = {}
    print(f"\n{'GEMV out x in':>14} {'result':>6} {'PCPI cyc':>9} {'list cyc':>9} {'W B/cyc':>8} "
          f"{'EX useful':>9} {'LD busy':>8} {'LD B/busy':>9} {'ST busy':>8} {'front idle':>10}")
    for out, inp in shapes:
        w = rng.integers(-127, 128, (out, inp), dtype=np.int8)
        x = rng.integers(-128, 128, (inp, d), dtype=np.int8)
        c1, s1 = mm.gemm(w, x)
        c2, s2 = mm.gemm_list(w, x)
        good = check(w, x, c1) and np.array_equal(c1, c2)
        ok &= good
        st = min((s1, s2), key=lambda s: s["riscv_cycles"])
        cyc = st["riscv_cycles"]
        best[(out, inp)] = cyc
        p = st.get("perf")
        b = perf_breakdown(p, d) if p else None
        pct = lambda v: f"{100 * v:5.1f}%" if b else "  n/a"
        print(f"{f'{out} x {inp}':>14} {'PASS' if good else 'FAIL':>6} {s1['riscv_cycles']:9d} {s2['riscv_cycles']:9d} "
              f"{out * inp / cyc:8.2f} {pct(b and b['ex_useful']):>9} {pct(b and b['ld_busy']):>8} "
              f"{(b['ld_bytes_per_busy_cycle'] if b else 0):9.2f} {pct(b and b['st_busy']):>8} "
              f"{pct(b and (b['starve'] + b['all_idle'])):>10}")
        if args.breakdown and b:
            print(format_breakdown(b))

    arm = arm_baseline(os.path.join(HERE, "arm_baseline.txt"))
    print(f"\nper token, GEMVs only (attention, norms, softmax, SiLU, RoPE and the ARM side not included)")
    print(f"{'model':>12} {'weights':>9} {'GEMV ms':>8} {'tok/s':>7} {'batch-' + str(d) + ' tok/s':>14} "
          f"{'@8 B/cyc':>9} {'@16 B/cyc':>9}   ARM llama2.c tok/s (fp32 | int8, 1 / 2 threads)")
    for m in models:
        g = gemvs(*MODELS[m])
        wbytes = sum(o * i * n for o, i, n, _ in g)
        cyc = sum(best[(o, i)] * n for o, i, n, _ in g)
        t = cyc / RISCV_HZ
        bound = lambda r: RISCV_HZ * r / wbytes
        a32, a8 = arm.get(m + ".bin"), arm.get(m + "_q80.bin")
        fmt = lambda v: f"{v[0]:.1f} / {v[1]:.1f}" if v else "-"
        print(f"{m:>12} {wbytes / 2**20:7.1f}MB {1e3 * t:8.2f} {1 / t:7.1f} {d / t:14.1f} "
              f"{bound(8):9.1f} {bound(16):9.1f}   {fmt(a32)} | {fmt(a8)}")
        for o, i, n, what in g:
            print(f"{'':>14}{what:>12}: {n:3d} x {o} x {i}  {100 * best[(o, i)] * n / cyc:5.1f}% of the GEMV time")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

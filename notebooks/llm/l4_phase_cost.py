#!/usr/bin/env python3
"""Where the cycles of one stories15M token go: each phase of the token list timed alone on the board.

Every phase of compile_model.ModelCompiler (no prefetching, so the phases
do not overlap each other) is built into its own small list and run
through the rt_fw ring at a given position; the device cycles of the
completion record are the phase's cost. The data is whatever the model
buffer and the previous phases left in the local memories (the timing of
the engines does not depend on the values). Printed: cycles per phase,
per layer and per token (the layer phases x layers + the classifier), and
the share of the linear layers' weight streaming (7.9 B / cycle) in it.

Files: as l4_prefetch_ab.py.

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \\
                  && cd /home/xilinx/l4ab && python3 l4_phase_cost.py [--pos 35]'
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compile_layer as CL  # noqa: E402
import compile_model as CM  # noqa: E402
from export_w8a8 import W8A8  # noqa: E402
from pynq_matmul import Device, DescList, allocate  # noqa: E402

IO_BYTES = 0x40000


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("--model", default=os.path.join(HERE, "stories15M_d8.w8a8"))
    ap.add_argument("--pos", type=int, default=35)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    dev = Device(args.bit, os.path.join(HERE, "rt_fw.bin"), ring_size=16)
    m = W8A8(args.model)
    c, d = m.cfg, dev.d
    io_off = -(-m.raw.size // 4096) * 4096
    kv_off = io_off + IO_BYTES
    kvb = CM.kv_bytes(c)
    lst_off = kv_off + -(-kvb // 4096) * 4096
    buf = allocate(shape=(lst_off + 64 * 4096,), dtype=np.uint8)
    buf[:m.raw.size] = m.raw
    buf[io_off:lst_off] = 0
    buf[io_off:io_off + 8] = np.frombuffer(CM.arg_block(args.pos, 1), np.uint8)
    base = buf.physical_address
    bases = [base, base + io_off, base + kv_off]
    prm = CM.token_params(args.pos, d, c.head_size)

    def timed(name, emit):
        mc = CM.ModelCompiler(m, prefetch=False)
        dl = DescList()
        emit(mc, dl)
        dl.end(1)
        rows = dl.array()
        buf[lst_off:lst_off + rows.nbytes] = np.frombuffer(rows.tobytes(), np.uint8)
        buf.flush()
        cyc = []
        for _ in range(args.reps):
            rec = dev.wait(dev.submit(base + lst_off, bases=bases, params=prm), timeout=10.0)
            if rec["status"]:
                raise RuntimeError(f"{name}: status {rec['status']:#x}")
            cyc.append(rec["cycles"])
        return int(np.median(cyc)), len(dl) - 1

    L = 0
    layer_phases = [
        ("RMSNorm (att)", lambda mc, dl: mc.rmsnorm(dl, mc.X, "rms_att", L, mc.XN)),
        ("quant + Wqkv 288->864", lambda mc, dl: mc.linear(dl, mc.XN, c.dim, "wqkv", L, out=mc.QKV)),
        ("RoPE (q, k)", lambda mc, dl: mc.rope(dl)),
        ("KV append", lambda mc, dl: mc.kv_append(dl, L)),
        ("attention (6 heads)", lambda mc, dl: mc.attention(dl, L)),
        ("quant + Wo 288->288", lambda mc, dl: mc.linear(dl, mc.ATT, c.dim, "wo", L, out=mc.Y)),
        ("residual", lambda mc, dl: mc.residual(dl)),
        ("RMSNorm (ffn)", lambda mc, dl: mc.rmsnorm(dl, mc.X, "rms_ffn", L, mc.XN)),
        ("quant + W13 288->1536", lambda mc, dl: mc.linear(dl, mc.XN, c.dim, "w13", L, out=mc.H13)),
        ("SiLU * h3", lambda mc, dl: mc.silu_mul(dl)),
        ("quant + W2 768->288", lambda mc, dl: mc.linear(dl, mc.U, c.hidden, "w2", L, out=mc.Y)),
        ("residual", lambda mc, dl: mc.residual(dl)),
    ]
    wbytes = {"Wqkv": c.dim * (c.dim + 2 * c.kv_dim), "Wo": c.dim * c.dim, "W13": c.dim * 2 * c.hidden,
              "W2": c.hidden * c.dim}
    print(f"D = {d}, pos {args.pos}; cycles per phase (median of {args.reps}), no prefetching")
    print(f"{'phase':>24} {'desc':>5} {'cycles':>9} {'weights at 7.9 B/cyc':>21}")
    per_layer = 0
    lin_min = 0
    for name, emit in layer_phases:
        cyc, n = timed(name, emit)
        per_layer += cyc
        w = next((wbytes[k] for k in wbytes if name.split()[2:3] == [k]), 0)
        lin_min += w / 7.9
        print(f"{name:>24} {n:5d} {cyc:9d} {w / 7.9:21.0f}" if w else f"{name:>24} {n:5d} {cyc:9d}")
    emb, _ = timed("embedding", lambda mc, dl: mc.embed(dl))
    fin, n = timed("final norm + classifier", lambda mc, dl: mc.final(dl))
    cls_w = c.dim * c.vocab / 7.9
    total = emb + c.layers * per_layer + fin
    print(f"{'embedding':>24} {'':>5} {emb:9d}")
    print(f"{'final norm + classifier':>24} {n:5d} {fin:9d} {cls_w:21.0f}")
    print(f"per layer: {per_layer} cycles, of which weight streaming at least {lin_min:.0f} "
          f"({100 * lin_min / per_layer:.0f}%), everything else {per_layer - lin_min:.0f}")
    print(f"per token (sum, phases not overlapped): {total} cycles = {50e6 / total:.1f} tok/s; "
          f"weights {100 * (c.layers * lin_min + cls_w) / total:.0f}% of it")
    buf.freebuffer()
    dev.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

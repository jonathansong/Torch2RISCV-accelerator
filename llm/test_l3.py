#!/usr/bin/env python3
"""L3 host test: W8A8 linear layers as descriptor lists (docs/llm_inference_plan.md §7).

Real stories15M weights (.w8a8 from llm/export_w8a8.py) and real layer
inputs: a DeviceModel (bit-exact SFU) forward pass over a prompt records
the vectors that enter each linear layer at the last position. For
layers 0 and L-1 (Wqkv, Wo, W13, W2) and the classifier, the list of
compile_layer.linear_layer runs in the functional simulator and must be

- bit-exact with DeviceModel.linear(quant_act(x)) (the model the
  device numerics are defined by),
- close to the fp32 reference x W^T (relative L2 error reported, gate 5%),
- identical between the looped (LOOP_END + PARAM) and unrolled forms.

    python3 llm/test_l3.py build/llm_cache/stories15M.bin build/llm_cache/stories15M_d8.w8a8 \\
        --kv build/llm_cache/kv/stories15M_kv_p99.99.npy
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import checkpoint  # noqa: E402
import compile_layer as CL  # noqa: E402
from export_w8a8 import W8A8, unpack_b  # noqa: E402
from ref_model import DeviceModel, SfuExact  # noqa: E402
from sa_funcsim import SaFuncSim  # noqa: E402
from pynq_matmul import DescList  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

F = np.float32
DDR_BASE = 0x10000000


def capture_inputs(dm, tokens):
    """Run tokens through dm; return [(layer, name, x)] of the last position,
    the fp32 vectors quantized for Wqkv, Wo, W13, W2 (per layer) and the classifier."""
    seen = []
    orig = dm.quant_act

    def rec(x):
        if x.size in (dm.cfg.dim, dm.cfg.hidden):
            seen.append(np.array(x, F))
        return orig(x)
    dm.quant_act = rec
    for pos, t in enumerate(tokens):
        seen.clear()
        dm.forward(t, pos)
    dm.quant_act = orig
    names = ["wqkv", "wo", "w13", "w2"]
    out = [(i // 4, names[i % 4], x) for i, x in enumerate(seen[:-1])]
    return out + [(None, "wcls", seen[-1])]


class Rig:
    """A funcsim DDR image: the .w8a8 file at BASE0, an io area at BASE1."""

    def __init__(self, d, model_bytes, io_bytes=1 << 20):
        self.d = d
        self.io_off = -(-len(model_bytes) // 4096) * 4096
        self.size = self.io_off + io_bytes
        self.model = model_bytes
        self.bases = [DDR_BASE, DDR_BASE + self.io_off]

    def run(self, dl, inputs, out_off, out_bytes):
        """inputs: [(io offset, bytes)]; the list goes at the end of the io area."""
        sim = SaFuncSim(self.d, DDR_BASE, self.size)
        sim.ddr_write(DDR_BASE, self.model)
        for off, data in inputs:
            sim.ddr_write(self.bases[1] + off, data)
        la = DDR_BASE + self.size - 64 * (len(dl) + 1)             # the list at the end of the io area
        sim.ddr_write(la, dl.array().tobytes())
        n = sim.run_list(la, bases=self.bases)
        return sim.ddr_read(self.bases[1] + out_off, out_bytes).tobytes(), n, sim


def linear_list(lay, m, layer, name, loop=None):
    """LD x -> quant -> linear -> ST y; x at io 0, y at io 0x10000."""
    c = m.cfg
    k = c.hidden if name == "w2" else c.dim
    n_out = {"wqkv": c.dim + 2 * c.kv_dim, "wo": c.dim, "w13": 2 * c.hidden, "w2": c.dim, "wcls": c.vocab}[name]
    x, tmp, out = lay.acc0, lay.acc0 + 256, lay.acc0 + 264
    dl = DescList().ld(0, CL.acc(x), 1, 4 * k, 4 * k, base=1)
    s_x = CL.quant_act(dl, lay, x, k, tmp)
    kw = dict(out_ddr=0x10000) if name == "wcls" else dict(out=out)
    nc = CL.linear(dl, lay, k, n_out, m.offset(name, layer), m.offset("s_" + name, layer), s_x, loop=loop, **kw)
    if name != "wcls":
        dl.st(0x10000, CL.acc(out), 1, 4 * n_out, 4 * n_out, base=1)
    return dl.end(0x13), n_out, nc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("w8a8")
    ap.add_argument("--kv", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("-i", "--prompt", default="Once upon a time, there was a little")
    ap.add_argument("--save", help="write the cases (x, DeviceModel golden, fp32 reference) to this .npz "
                                   "for the board test notebooks/llm/l3_linear_demo.py")
    args = ap.parse_args()
    cfg, w = checkpoint.load(args.checkpoint)
    m = W8A8(args.w8a8)
    d = m.d
    lay = CL.Layout(d)
    kv = np.load(args.kv)
    dm = DeviceModel(cfg, w, kv, d=d, sfu=SfuExact)
    tok = Tokenizer(args.tokenizer or os.path.join(os.path.dirname(args.checkpoint), "tokenizer.bin"), cfg.vocab)
    tokens = tok.encode(args.prompt)
    t0 = time.time()
    caps = capture_inputs(dm, tokens)
    print(f"D = {d}; captured {len(caps)} layer inputs at position {len(tokens) - 1} ({time.time() - t0:.1f} s)")
    rig = Rig(d, m.raw.tobytes())
    ok = True
    saved = {}
    for layer, name, x in caps:
        if layer not in (None, 0, cfg.layers - 1):
            continue
        dl, n_out, nc = linear_list(lay, m, layer, name)
        t0 = time.time()
        got, n, _ = rig.run(dl, [(0, x.tobytes())], 0x10000, 4 * n_out)
        y = np.frombuffer(got, F)
        # golden: DeviceModel.linear on the same packed weights
        wq = unpack_b(m.get(name, layer)).astype(np.float64)
        want = dm.linear(*dm.quant_act(x), (wq, m.get("s_" + name, layer)))
        # fp32 reference
        wf = {"wqkv": ("wq", "wk", "wv"), "wo": ("wo",), "w13": ("w1", "w3"), "w2": ("w2",), "wcls": ("wcls",)}[name]
        wfull = np.concatenate([w[p] if layer is None else w[p][layer] for p in wf]).astype(np.float64)
        ref = wfull @ x.astype(np.float64)
        rel = np.linalg.norm(y - ref) / np.linalg.norm(ref)
        exact = got == want.astype(F).tobytes()
        good = exact and rel < 0.05
        extra = ""
        if name == "wcls":                                           # looped (default) vs unrolled
            got2, n2, _ = rig.run(linear_list(lay, m, layer, name, loop=False)[0], [(0, x.tobytes())],
                                  0x10000, 4 * n_out)
            same = got2 == got
            good &= same
            extra = f"; unrolled form ({n2} descriptors) {'identical' if same else 'DIFFERENT'}"
        ok &= good
        key = f"{'cls' if layer is None else layer}_{name}"
        saved[key + "_x"], saved[key + "_want"], saved[key + "_ref"] = x, want.astype(F), ref.astype(F)
        lname = "cls" if layer is None else f"L{layer}"
        print(f"  {'PASS' if good else 'FAIL'}  {lname:>3} {name:5s} {len(x):4d} -> {n_out:5d}: "
              f"{n:4d} descriptors ({len(dl)} in the list), chunk {nc} tiles, "
              f"{'bit-exact' if exact else 'NOT bit-exact'} with DeviceModel, "
              f"rel. error vs fp32 {100 * rel:.2f}%{extra} ({time.time() - t0:.1f} s)")
    if args.save:
        np.savez(args.save, prompt=args.prompt, **saved)
        print(f"saved {len(saved) // 3} cases to {args.save}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

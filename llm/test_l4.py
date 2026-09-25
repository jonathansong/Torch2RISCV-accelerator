#!/usr/bin/env python3
"""L4 host test: the whole decoder as one static descriptor list (docs/llm_inference_plan.md §8).

The list of compile_model.ModelCompiler runs token after token in the
functional simulator; per token only the parameter block and the
argument block (pos, token) change. Checked against
DeviceModel(sfu=SfuExact), bit for bit:

- the logits of every token,
- the int8 KV cache (all layers, rows 0..pos),
- with --layers N: the residual stream after N layers (logits off).

Models: a random tiny model (dim 32, hidden 64, 2 heads, 2 layers,
vocab 64, seq 16: the co-simulation model of plan §8.6) and, given a
checkpoint, a real one (stories15M) for the first tokens of a prompt.

    python3 llm/test_l4.py [--d 8] [--tiny-tokens 16]
    python3 llm/test_l4.py --checkpoint build/llm_cache/stories15M.bin \\
        --kv build/llm_cache/kv/stories15M_kv_p99.99.npy --tokens 12
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import checkpoint  # noqa: E402
import compile_model as CM  # noqa: E402
from export_w8a8 import W8A8, export  # noqa: E402
from ref_model import DeviceModel, Fp32Model, SfuExact  # noqa: E402
from sa_funcsim import SaFuncSim  # noqa: E402

F = np.float32
DDR = 0x10000000


def tiny_model(seed=3, dim=32, hidden=64, heads=2, layers=2, vocab=64, seq=16):
    """(cfg, w, kv_scales) of a random llama-style model; KV scales = absmax / 127
    over an fp32 run of random tokens."""
    rng = np.random.default_rng(seed)
    cfg = checkpoint.Config(dim, hidden, layers, heads, heads, vocab, seq)
    g = lambda *s: (rng.standard_normal(s) / np.sqrt(s[-1])).astype(F)
    w = {"tok": g(vocab, dim) * 4, "rms_att": 1 + 0.1 * g(layers, dim), "wq": g(layers, dim, dim),
         "wk": g(layers, dim, dim), "wv": g(layers, dim, dim), "wo": g(layers, dim, dim),
         "rms_ffn": 1 + 0.1 * g(layers, dim), "w1": g(layers, hidden, dim), "w2": g(layers, dim, hidden),
         "w3": g(layers, hidden, dim), "rms_final": 1 + 0.1 * g(dim), "wcls": g(vocab, dim)}
    fm = Fp32Model(cfg, w)
    for pos, t in enumerate(rng.integers(0, vocab, seq)):
        fm.forward(int(t), pos)
    kv = np.stack([np.abs(fm.kc).max(axis=(1, 2)), np.abs(fm.vc).max(axis=(1, 2))], 1) / F(127)
    return cfg, w, kv.astype(F)


class Rig:
    """Functional simulator DDR: model file, io area, KV cache (BASE0..2), the list."""

    def __init__(self, m, list_rows):
        self.m = m
        self.io = -(-m.raw.size // 4096) * 4096
        self.kv = self.io + 0x40000
        self.lst = self.kv + -(-CM.kv_bytes(m.cfg) // 4096) * 4096
        self.size = self.lst + 64 * (list_rows + 1)
        self.sim = SaFuncSim(m.d, DDR, self.size)
        self.sim.ddr_write(DDR, m.raw.tobytes())
        self.bases = [DDR, DDR + self.io, DDR + self.kv]

    def run(self, dl, pos, token):
        c, s = self.m.cfg, self.sim
        s.ddr_write(DDR + self.lst, dl.array().tobytes())
        s.ddr_write(DDR + self.io, CM.arg_block(pos, token))
        return s.run_list(DDR + self.lst, bases=self.bases, params=CM.token_params(pos, self.m.d, c.head_size))

    def logits(self):
        return self.sim.ddr_read(DDR + self.io + CM.IO_LOGITS, 4 * self.m.cfg.vocab).view(F)

    def x(self):
        return self.sim.ddr_read(DDR + self.io + CM.IO_X, 4 * self.m.cfg.dim).view(F)

    def kv_cache(self):
        c = self.m.cfg
        raw = self.sim.ddr_read(DDR + self.kv, CM.kv_bytes(c)).view(np.int8)
        return raw.reshape(c.layers, 2, c.seq_len, c.kv_dim)


def run_model(name, cfg, w, kv, m, tokens, n_layers=None):
    d = m.d
    dm = DeviceModel(cfg, w, kv, d=d, sfu=SfuExact)
    mc = CM.ModelCompiler(m)
    layers = None if n_layers is None else range(n_layers)
    dl = mc.build(layers=layers, logits=n_layers is None, dump_x=n_layers is not None)
    rig = Rig(m, len(dl))
    ok = True
    t0 = time.time()
    print(f"{name}: D = {d}, {len(dl)} descriptors in the list, ACC words used {mc.acc_used}")
    for pos, t in enumerate(tokens):
        n = rig.run(dl, pos, t)
        want = dm.forward(t, pos, n_layers=n_layers)
        got = rig.x() if n_layers is not None else rig.logits()
        exact = got.tobytes() == want.astype(F).tobytes()
        kc = rig.kv_cache()
        nl = cfg.layers if n_layers is None else n_layers
        kv_ok = all(np.array_equal(kc[l, 0, :pos + 1], dm.kc[l, :pos + 1]) and
                    np.array_equal(kc[l, 1, :pos + 1], dm.vc[l, :pos + 1]) for l in range(nl))
        good = exact and kv_ok
        ok &= good
        extra = ""
        if not exact:
            diff = np.flatnonzero(got.view(np.uint32) != want.astype(F).view(np.uint32))
            extra = f"; {diff.size} words differ, first [{diff[0]}] {got[diff[0]]!r} vs {want[diff[0]]!r}"
        top = f", argmax {int(np.argmax(got))}" if n_layers is None else ""
        print(f"  pos {pos:3d} token {t:5d}: {n:5d} descriptors executed, "
              f"{'bit-exact' if exact else 'DIFFERENT'}{top}, KV {'ok' if kv_ok else 'DIFFERENT'}{extra}")
    print(f"  {name}: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.1f} s)")
    return ok


def save_golden(cfg, w, kv, d, tok, path, gen, prompt="Once upon a time, there was a little girl named Lily. She"):
    """DeviceModel (bit-exact SFU) run of the prompt plus `gen` greedy tokens: the
    token fed at each position and the logits it produced."""
    dm = DeviceModel(cfg, w, kv, d=d, sfu=SfuExact)
    toks = tok.encode(prompt)
    n_prompt = len(toks)
    fed, logits = [], []
    t = toks[0]
    for pos in range(n_prompt + gen):
        lg = dm.forward(t, pos)
        fed.append(t)
        logits.append(lg.astype(F))
        t = toks[pos + 1] if pos + 1 < n_prompt else int(np.argmax(lg))
    np.savez(path, tokens=np.array(fed), logits=np.stack(logits), n_prompt=n_prompt, prompt=prompt)
    print(f"saved {len(fed)} tokens ({n_prompt} prompt + {gen} greedy) to {path}: "
          f"{tok.decode(fed + [int(np.argmax(logits[-1]))])!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--tiny-tokens", type=int, default=16)
    ap.add_argument("--checkpoint")
    ap.add_argument("--kv")
    ap.add_argument("--tokens", type=int, default=10)
    ap.add_argument("--layers", type=int, help="only this many layers; compare x instead of logits")
    ap.add_argument("--save", help="with --checkpoint: write the board test's golden run to this .npz "
                                   "(prompt, then --gen greedy tokens; DeviceModel logits of every token)")
    ap.add_argument("--gen", type=int, default=20)
    args = ap.parse_args()
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        cfg, w, kv = tiny_model()
        path = os.path.join(tmp, "tiny.w8a8")
        export(cfg, w, kv, args.d, path)
        rng = np.random.default_rng(7)
        toks = [int(t) for t in rng.integers(0, cfg.vocab, args.tiny_tokens)]
        ok &= run_model("tiny", cfg, w, kv, W8A8(path), toks, args.layers)
        if args.checkpoint:
            cfg, w = checkpoint.load(args.checkpoint)
            kv = np.load(args.kv)
            path = os.path.join(tmp, "model.w8a8")
            export(cfg, w, kv, args.d, path)
            from tokenizer import Tokenizer
            tok = Tokenizer(os.path.join(os.path.dirname(args.checkpoint), "tokenizer.bin"), cfg.vocab)
            toks = tok.encode("Once upon a time, there was a little girl named Lily. She")[:args.tokens]
            ok &= run_model(os.path.basename(args.checkpoint), cfg, w, kv, W8A8(path), toks, args.layers)
            if args.save:
                save_golden(cfg, w, kv, args.d, tok, args.save, args.gen)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

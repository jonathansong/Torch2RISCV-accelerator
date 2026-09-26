#!/usr/bin/env python3
"""L5b host test: D sequences per step with one static list (docs/llm_inference_plan.md §11.1).

The batched list (compile_batch.BatchCompiler) runs step after step in
the functional simulator; sequence b starts at step b (the positions of
the sequences differ inside a step; a slot that has not started runs a
dummy token at pos 0, whose KV row is overwritten when it starts). Every
sequence's logits and KV cache are compared bit for bit with its own
single-sequence DeviceModel(sfu=SfuExact).

    python3 llm/test_l5b.py [--steps 20]
    python3 llm/test_l5b.py --checkpoint build/llm_cache/stories15M.bin \\
        --kv build/llm_cache/kv/stories15M_kv_p99.99.npy --steps 12
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
import compile_batch as CB  # noqa: E402
import compile_model as CM  # noqa: E402
from export_w8a8 import W8A8, export  # noqa: E402
from ref_model import DeviceModel, SfuExact  # noqa: E402
from sa_funcsim import SaFuncSim  # noqa: E402
from test_l4 import tiny_model  # noqa: E402

F = np.float32
DDR = 0x10000000
DUMMY = 1


class BatchRig:
    def __init__(self, m, list_rows):
        c, d = m.cfg, m.d
        self.m, self.d = m, d
        self.io = -(-m.raw.size // 4096) * 4096
        self.kv = self.io + -(-CB.io_bytes(c, d) // 4096) * 4096
        self.kvb = CM.kv_bytes(c)
        self.lst = self.kv + d * self.kvb
        self.size = self.lst + 64 * (list_rows + 1)
        self.sim = SaFuncSim(d, DDR, self.size)
        self.sim.ddr_write(DDR, m.raw.tobytes())
        self.bases = [DDR, DDR + self.io, DDR + self.kv]

    def step(self, dl, poses, tokens):
        s = self.sim
        s.ddr_write(DDR + self.lst, dl.array().tobytes())
        s.ddr_write(DDR + self.io, CB.arg_table(poses, tokens, self.d, self.m.cfg.head_size))
        return s.run_list(DDR + self.lst, bases=self.bases)

    def logits(self):
        v = self.m.cfg.vocab
        return self.sim.ddr_read(DDR + self.io + CB.IO_LOGITS, 4 * self.d * v).view(F).reshape(self.d, v)

    def kv_cache(self, b):
        c = self.m.cfg
        raw = self.sim.ddr_read(DDR + self.kv + b * self.kvb, self.kvb).view(np.int8)
        return raw.reshape(c.layers, 2, c.seq_len, c.kv_dim)


def run_batch(name, cfg, w, kv, m, seqs, steps):
    """seqs: D token sequences (teacher-forced inputs); sequence b starts at step b."""
    d = m.d
    bc = CB.BatchCompiler(m)
    dl = bc.build()
    rig = BatchRig(m, len(dl))
    dms = [DeviceModel(cfg, w, kv, d=d, sfu=SfuExact) for _ in range(d)]
    print(f"{name}: D = {d} sequences per step, {len(dl)} descriptors in the list, ACC words {bc.acc_used}")
    ok = True
    t0 = time.time()
    for step in range(steps):
        poses = [max(step - b, 0) for b in range(d)]
        live = [step >= b for b in range(d)]
        toks = [seqs[b][poses[b]] if live[b] else DUMMY for b in range(d)]
        n = rig.step(dl, poses, toks)
        lg = rig.logits()
        bad = []
        for b in range(d):
            if not live[b]:
                continue
            want = dms[b].forward(toks[b], poses[b])
            kc = rig.kv_cache(b)
            p = poses[b]
            if lg[b].tobytes() != want.astype(F).tobytes() or not all(
                    np.array_equal(kc[l, 0, :p + 1], dms[b].kc[l, :p + 1]) and
                    np.array_equal(kc[l, 1, :p + 1], dms[b].vc[l, :p + 1]) for l in range(cfg.layers)):
                bad.append(b)
        ok &= not bad
        print(f"  step {step:3d}: positions {poses}, {n} descriptors executed: "
              f"{'all live sequences bit-exact (logits + KV)' if not bad else f'sequences {bad} DIFFERENT'}")
    print(f"  {name}: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.1f} s)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--checkpoint")
    ap.add_argument("--kv")
    args = ap.parse_args()
    ok = True
    d = 8
    with tempfile.TemporaryDirectory() as tmp:
        cfg, w, kv = tiny_model()
        export(cfg, w, kv, d, os.path.join(tmp, "t.w8a8"))
        rng = np.random.default_rng(11)
        seqs = [[int(t) for t in rng.integers(0, cfg.vocab, cfg.seq_len)] for _ in range(d)]
        ok &= run_batch("tiny", cfg, w, kv, W8A8(os.path.join(tmp, "t.w8a8")), seqs,
                        min(args.steps, cfg.seq_len))
        if args.checkpoint:
            cfg, w = checkpoint.load(args.checkpoint)
            kv = np.load(args.kv)
            export(cfg, w, kv, d, os.path.join(tmp, "m.w8a8"))
            from eval_quant import EVAL
            from tokenizer import Tokenizer
            tok = Tokenizer(os.path.join(os.path.dirname(args.checkpoint), "tokenizer.bin"), cfg.vocab)
            seqs = [tok.encode(p) + [0] * cfg.seq_len for p in EVAL]
            ok &= run_batch(os.path.basename(args.checkpoint), cfg, w, kv, W8A8(os.path.join(tmp, "m.w8a8")),
                            seqs, args.steps)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

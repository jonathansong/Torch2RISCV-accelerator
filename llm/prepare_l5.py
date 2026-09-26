#!/usr/bin/env python3
"""Reference data for the L5 board acceptance (notebooks/llm/l5_generate.py, plan §9.2).

1. greedy: prompt + N greedy tokens of DeviceModel(sfu=SfuExact), the model
   the device is bit-exact with: the tokens fed at each position and an
   MD5 of each position's logits (the board compares them bit for bit).
2. accuracy: the L0 evaluation (llm/eval_quant.py EVAL prompts, 256
   steps): the fp32 model's greedy token sequences, its top-5 at every
   position, and DeviceModel(SfuExact)'s argmax when teacher-forced on
   them (the board's argmax must equal it; top-1 vs fp32 >= 95%).

    python3 llm/prepare_l5.py build/llm_cache/stories15M.bin \\
        --kv build/llm_cache/kv/stories15M_kv_p99.99.npy -o l5_ref.npz
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import checkpoint  # noqa: E402
from eval_quant import EVAL  # noqa: E402
from ref_model import F, DeviceModel, Fp32Model, SfuExact, generate  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402


def md5(logits):
    return hashlib.md5(np.asarray(logits, F).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--kv", required=True)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--greedy", type=int, default=128)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    cfg, w = checkpoint.load(args.checkpoint)
    tok = Tokenizer(os.path.join(os.path.dirname(args.checkpoint), "tokenizer.bin"), cfg.vocab)
    dm = DeviceModel(cfg, w, np.load(args.kv), d=args.d, sfu=SfuExact)
    out = {"prompt": args.prompt}

    t0 = time.time()
    dm.reset()
    ptoks = tok.encode(args.prompt)
    fed, hashes = [], []
    t = ptoks[0]
    for pos in range(len(ptoks) - 1 + args.greedy):
        lg = dm.forward(t, pos)
        fed.append(t)
        hashes.append(md5(lg))
        t = ptoks[pos + 1] if pos + 1 < len(ptoks) else int(np.argmax(lg))
    out["greedy_tokens"], out["greedy_md5"] = np.array(fed), np.array(hashes)
    out["greedy_text"] = tok.decode(fed[:1] + fed[1:] + [t])
    print(f"greedy: {len(fed)} positions ({time.time() - t0:.0f} s): {out['greedy_text']!r}")

    t0 = time.time()
    fp = Fp32Model(cfg, w)
    toks_all, top5_all, dev_all = [], [], []
    top1 = n = 0
    for p in EVAL:
        toks, lg32 = generate(fp, tok.encode(p), args.steps, keep_logits=True)
        _, lgd = generate(dm, toks, args.steps, forced=toks, keep_logits=True)
        k = len(lgd)
        top5 = np.argsort(np.stack(lg32[:k]), axis=1)[:, -5:][:, ::-1]
        dev = np.stack(lgd).argmax(1)
        top1 += int((dev == top5[:, 0]).sum())
        n += k
        toks_all.append(np.array(toks[:k]))
        top5_all.append(top5)
        dev_all.append(dev)
    out["eval_prompts"] = np.array(EVAL)
    for i in range(len(EVAL)):
        out[f"eval{i}_tokens"], out[f"eval{i}_fp32_top5"], out[f"eval{i}_dev_argmax"] = \
            toks_all[i], top5_all[i], dev_all[i]
    print(f"accuracy: DeviceModel(SfuExact) teacher-forced top-1 vs fp32 {100 * top1 / n:.1f}% "
          f"over {n} positions ({time.time() - t0:.0f} s)")
    np.savez(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

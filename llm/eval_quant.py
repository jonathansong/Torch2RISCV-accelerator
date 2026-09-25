#!/usr/bin/env python3
"""Accuracy of the device numerics against fp32 (docs/llm_inference_plan.md §4.2).

For every model:
  1. calibration: fp32 runs over CALIB prompts give the per-layer static KV
     scales, absmax / 127 or the 99.99th percentile of |K|, |V| / 127;
  2. evaluation over EVAL prompts (disjoint from CALIB): the fp32 model
     generates greedily; the device model is teacher-forced on the same
     tokens. Per position: top-1 agreement, top-1 of the device within the
     fp32 top-5, KL(p_fp32 || p_device). Free running: the position where
     the device's own greedy text first differs from fp32.

    python3 llm/eval_quant.py build/llm_cache/stories15M.bin build/llm_cache/tokenizer.bin [...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkpoint  # noqa: E402
from ref_model import F, DeviceModel, Fp32Model, generate  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

CALIB = ["Tom had a red kite", "The little bird wanted to fly", "Sara and Ben went to the beach",
         "Once there was a big bear who"]
EVAL = ["Once upon a time", "One day, a little girl named Lily", "The cat sat on the mat and",
        "Tom and his dog went to the park", "There was a big red ball", "Mom said to Timmy",
        "In a small town, there lived", "The sun was shining and the birds"]


def kv_calibration(model, tok, steps):
    """{method: (layers, 2) scales} from fp32 runs over CALIB."""
    ks, vs = [], []
    for p in CALIB:
        toks, _ = generate(model, tok.encode(p), steps)
        n = len(toks)
        ks.append(np.abs(model.kc[:, :n]).reshape(model.cfg.layers, -1))
        vs.append(np.abs(model.vc[:, :n]).reshape(model.cfg.layers, -1))
    k, v = np.concatenate(ks, axis=1), np.concatenate(vs, axis=1)
    return {"absmax": np.stack([k.max(1), v.max(1)], 1).astype(F) / F(127),
            "p99.99": np.stack([np.percentile(k, 99.99, 1), np.percentile(v, 99.99, 1)], 1).astype(F) / F(127)}


def log_softmax(x):
    x = np.asarray(x, np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def evaluate(cfg, w, kv, refs, steps):
    dev = DeviceModel(cfg, w, kv)
    top1 = top5 = n = 0
    kls = []
    for toks, lg32 in refs:
        _, lgd = generate(dev, toks, steps, forced=toks, keep_logits=True)
        a, b = np.stack(lg32[:len(lgd)]), np.stack(lgd)
        pa, pb = log_softmax(a), log_softmax(b)
        kls.append((np.exp(pa) * (pa - pb)).sum(-1))
        ta, tb = a.argmax(-1), b.argmax(-1)
        top1 += int((ta == tb).sum())
        top5 += int(sum(tb[i] in np.argsort(a[i])[-5:] for i in range(len(tb))))
        n += len(tb)
    kl = np.concatenate(kls)
    return {"top1": top1 / n, "top5": top5 / n, "kl_mean": float(kl.mean()), "kl_p99": float(np.percentile(kl, 99)),
            "positions": n}


def first_divergence(cfg, w, tok, kv, prompt, ref_tokens, steps):
    dev = DeviceModel(cfg, w, kv)
    own, _ = generate(dev, tok.encode(prompt), steps)
    for i, (a, b) in enumerate(zip(ref_tokens, own)):
        if a != b:
            return i
    return min(len(own), len(ref_tokens))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoints", nargs="+", help="llama2.c .bin files, then the tokenizer.bin last")
    ap.add_argument("-n", "--steps", type=int, default=256)
    ap.add_argument("--out", help="write the results as JSON")
    ap.add_argument("--save-kv", help="directory for <model>_kv_<method>.npy")
    args = ap.parse_args()
    *models, tok_path = args.checkpoints
    results = {}
    for path in models:
        t0 = time.time()
        name = os.path.splitext(os.path.basename(path))[0]
        cfg, w = checkpoint.load(path)
        tok = Tokenizer(tok_path, cfg.vocab)
        fp = Fp32Model(cfg, w)
        scales = kv_calibration(fp, tok, args.steps)
        refs = [generate(fp, tok.encode(p), args.steps, keep_logits=True) for p in EVAL]
        results[name] = {}
        for method, kv in scales.items():
            r = evaluate(cfg, w, kv, refs, args.steps)
            r["first_divergence"] = [first_divergence(cfg, w, tok, kv, p, refs[i][0], args.steps)
                                     for i, p in enumerate(EVAL)]
            results[name][method] = r
            if args.save_kv:
                os.makedirs(args.save_kv, exist_ok=True)
                np.save(os.path.join(args.save_kv, f"{name}_kv_{method}.npy"), kv)
            print(f"{name:>12} KV {method:>7}: top-1 {100 * r['top1']:5.1f}%  in fp32 top-5 {100 * r['top5']:5.1f}%  "
                  f"KL mean {r['kl_mean']:.4f} p99 {r['kl_p99']:.4f}  ({r['positions']} positions)  "
                  f"greedy text diverges at token {r['first_divergence']}", flush=True)
        print(f"{name:>12} done in {time.time() - t0:.0f} s", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

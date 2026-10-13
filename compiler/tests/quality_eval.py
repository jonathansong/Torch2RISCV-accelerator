#!/usr/bin/env python3
"""Quantization quality of the generic frontend's models on the host
(docs/generality_implementation_plan.md §3.2, WP3 / WP4): teacher-forced on
WikiText-2 (raw) test, windows of --window tokens from a fixed start, every
variant against the original model in fp32:
top-1 agreement, mean correlation of the logits, mean KL(reference || variant)
and each one's perplexity.

Variants (names on the command line, --variants):
  fp32      the original HF model (the reference)
  w8a8      self-quantized W8A8 linear layers (hf_generic.quantize), fp32 attention
  sa        w8a8 + RoPE tables, int8 KV cache and the sa attention, with the
            device's SFU (exp / recip / rsqrt): what the device computes (the
            sim is bit-exact with it up to reduction order)
  ct-w8a8   the compressed-tensors checkpoint (--ct DIR): its int8 weights and
            scales, per-token int8 activations, fp32 attention (the
            checkpoint's own semantics, as vLLM runs it)
  ct-sa     the checkpoint on the sa form (as sa above)
  sa:X+Y / ct-sa:X+Y   the sa form with some of its parts kept in fp32, for the
            error's attribution (hf_generic.SA_FP32): act (linear layers on
            fp32 activations), kv (an fp32 KV cache), q (the attention's query),
            probs (the attention probabilities), sfu (exact exp / recip / rsqrt
            instead of the device's); the parts not named compute exactly as sa;
            kch / vch: the KV cache's K / V scales per channel (hf_generic.calibrate_kv)

    python3 compiler/tests/quality_eval.py --model build/llm_cache/SmolLM2-135M \\
        --ct build/llm_cache/SmolLM2-135M-W8A8 [--tokens 1024 --window 256] [--out results.json]
"""
import argparse
import contextlib
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "frontend"))
sys.path.insert(0, HERE)
import hf_generic as H  # noqa: E402


TEXT = os.path.join(HERE, "..", "..", "build", "llm_cache", "wikitext-2-raw-v1-test.txt")


def wikitext_tokens(model_dir, n, skip):
    """The test split's text joined (as the usual PPL evaluation), tokenized with
    the model's tokenizer; n tokens from token `skip` on. The text is a file
    (the export environment has no `datasets`), written once with
    build/llmc_venv/bin/python -c "from datasets import load_dataset;
    open(TEXT, 'w').write('\\n\\n'.join(load_dataset('Salesforce/wikitext',
    'wikitext-2-raw-v1', split='test')['text']))"."""
    from transformers import AutoTokenizer
    with open(TEXT) as f:
        text = f.read()
    ids = AutoTokenizer.from_pretrained(model_dir)(text, add_special_tokens=False)["input_ids"]
    if skip + n > len(ids):
        raise SystemExit(f"only {len(ids)} tokens")
    return ids[skip:skip + n]


def logits_full(model, ids):
    """One forward over the window (fp32 attention variants): [L, V]."""
    return model(torch.tensor([ids])).logits[0].float()


def logits_steps(dec, ids):
    """Token by token through a decoder (the sa form: KV cache rows): [L, V]."""
    return torch.stack([dec(torch.tensor([[t]]), torch.tensor([[p]]))[0, -1] for p, t in enumerate(ids)]).float()


def build(name, args):
    """(callable ids -> logits [L, V], context manager) of a variant."""
    import test_hf_generic as TH
    if name == "fp32":
        m = H.hf_model(args.model)
        return lambda ids: logits_full(m, ids), contextlib.nullcontext
    if name in ("w8a8", "ct-w8a8"):
        m = H.hf_model(args.model if name == "w8a8" else args.ct)
        H.quantize(m)
        return lambda ids: logits_full(m, ids), contextlib.nullcontext
    base, _, opts = name.partition(":")
    if base in ("sa", "ct-sa"):
        src = args.model if base == "sa" else args.ct
        keep = set(o for o in opts.split("+") if o)
        kv_channel = "".join(p for p in "kv" if f"{p}ch" in keep)
        keep -= {"kch", "vch"}
        bad = keep - {"act", "kv", "q", "probs", "sfu"}
        if bad:
            raise SystemExit(f"unknown fp32 parts {bad}")

        kv = []                          # (the KV scales calibrated on the first window's load, then reused)

        def run(ids):
            H.SA_FP32.clear()
            H.SA_FP32.update(keep - {"sfu"})
            try:
                dec = H.load(src, max_len=args.window, quant=True, rope=True, attn=True,     # (a fresh cache per window)
                             kv_scales=kv[0] if kv else None, kv_channel=kv_channel,
                             kv_calib=H.kv_calib_arg(args.kv_calib))
                kv[:] = [dec.kv_scales]
                out = logits_steps(dec, ids)
                del dec
                return out
            finally:
                H.SA_FP32.clear()
                gc.collect()             # (HF modules hold reference cycles: two Llama-1B decoders exceed 10 GB)
        return run, (contextlib.nullcontext if "sfu" in keep else TH.device_sfu)
    raise SystemExit(f"unknown variant {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="the original HF model (the reference)")
    ap.add_argument("--ct", help="a compressed-tensors W8A8 checkpoint of it")
    ap.add_argument("--variants", default=None, help="comma-separated (default: fp32,w8a8,sa and with --ct ct-w8a8,ct-sa)")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--skip", type=int, default=0, help="first token of the evaluated span")
    ap.add_argument("--out", help="results (JSON)")
    ap.add_argument("--ref-cache", help="the fp32 reference's logits (.npy): loaded if present, else saved, so that "
                    "each variant can run in a process of its own (Llama-1B: fp32 and an sa form do not fit in 10 GB together)")
    ap.add_argument("--kv-calib", help="FILE[:TOKENS] text for the sa variants' KV scales (default: export_hf.CALIB)")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    names = (args.variants or "fp32,w8a8,sa" + (",ct-w8a8,ct-sa" if args.ct else "")).split(",")
    if names[0] != "fp32":
        names = ["fp32"] + [n for n in names if n != "fp32"]
    ids = wikitext_tokens(args.model, args.tokens, args.skip)
    wins = [ids[i:i + args.window] for i in range(0, len(ids) - args.window + 1, args.window)]
    # the reference first (its logits kept), then each variant window by window.
    # sa's RoPE rewrite changes rotate_half in transformers for the whole
    # process, so the fp32 / w8a8 forms run before any sa one
    order = sorted(names, key=lambda n: "sa" in n.partition(":")[0].split("-"))
    ref, res = None, {}
    if args.ref_cache and os.path.exists(args.ref_cache):
        a = np.load(args.ref_cache, mmap_mode="r")
        assert a.shape[:2] == (len(wins), args.window), (a.shape, len(wins), args.window)
        ref = [torch.from_numpy(np.array(a[w])) for w in range(len(wins))]
        order = [n for n in order if n != "fp32"]
    for name in order:
        t0 = time.time()
        f, ctx = build(name, args)
        nll, agree, corr, kl, n = 0.0, 0, 0.0, 0.0, 0
        refs = [] if ref is None else ref
        with ctx():
            for w, win in enumerate(wins):
                lg = f(win)
                lp = torch.log_softmax(lg, -1)
                tgt = torch.tensor(win[1:])
                nll += float(-lp[:-1].gather(1, tgt[:, None]).sum())
                if ref is None:
                    refs.append(lg)
                    continue
                r = ref[w]
                rlp = torch.log_softmax(r, -1)
                agree += int((lg.argmax(-1) == r.argmax(-1)).sum())
                kl += float((rlp.exp() * (rlp - lp)).sum())
                corr += sum(float(np.corrcoef(lg[i].numpy(), r[i].numpy())[0, 1]) for i in range(len(win)))
                n += len(win)
        if ref is None:
            ref = refs
            if args.ref_cache:
                np.save(args.ref_cache, torch.stack(ref).numpy())
        f = ctx = None                   # (the model freed before the next one loads: one at a time in memory)
        gc.collect()
        P = len(wins) * (args.window - 1)
        res[name] = {"ppl": math.exp(nll / P), "seconds": round(time.time() - t0, 1)}
        if n:
            res[name].update(top1=agree / n, corr=corr / n, kl=kl / n)
        print(f"{name:8s} PPL {res[name]['ppl']:8.3f}" +
              (f"  top-1 vs fp32 {agree}/{n} ({100 * agree / n:.1f}%)  corr {corr / n:.4f}  KL {kl / n:.4f}" if n else "") +
              f"  ({res[name]['seconds']} s)", flush=True)
    summary = {"model": args.model, "ct": args.ct, "tokens": len(wins) * args.window, "window": args.window,
               "skip": args.skip, "dataset": "Salesforce/wikitext wikitext-2-raw-v1 test", "results": res}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

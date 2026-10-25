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
--moe-routing (an MoE model, hf_generic.QMixtralMoe): per variant, the experts it
selects against the fp32 model's at every layer and position (the share of the k
experts in common, and how often the sets are equal), and the router alone: its
W8A8 Top-k against an fp32 router on the same (the variant's own) hidden state.
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


ROUTES = None                       # --moe-routing: {"ids": {layer: [k ids per position]}, "own": ...}


def instrument(model):
    """--moe-routing: record each MoE layer's selected experts (an fp32 model's
    routers by a hook; a quantized one's QMixtralMoe.route, with an fp32 router's
    Top-k on the same input), layers in model order, positions in call order."""
    if ROUTES is None:
        return
    layer = 0
    for mod in model.modules():
        if isinstance(mod, H.QMixtralMoe):
            def route(x, route=mod.route, mod=mod, l=layer):
                r = route(x)
                ROUTES["ids"].setdefault(l, []).append(r[1][0].numpy().copy())
                own = torch.topk((x @ mod.router_fp32.t())[0], mod.k).indices
                ROUTES["own"].setdefault(l, []).append(own.numpy().copy())
                return r
            mod.route = route
            layer += 1
        elif type(mod).__name__ in H.MOE_ROUTERS:
            def hook(m, i, o, l=layer):
                idx = next(t for t in o if t.dtype == torch.int64)
                ROUTES["ids"].setdefault(l, []).extend(r.copy() for r in idx.numpy())
            mod.register_forward_hook(hook)
            layer += 1


def overlap(a, b):
    """[N, k] id sets: (mean share in common, share of equal sets)"""
    common = np.array([len(set(x) & set(y)) for x, y in zip(a, b)])
    return common.mean() / a.shape[1], float(np.mean(common == a.shape[1]))


def build(name, args):
    """(callable ids -> logits [L, V], context manager) of a variant."""
    import test_hf_generic as TH
    if name == "fp32":
        m = H.hf_model(args.model)
        instrument(m)
        return lambda ids: logits_full(m, ids), contextlib.nullcontext
    if name in ("w8a8", "ct-w8a8"):
        m = H.hf_model(args.model if name == "w8a8" else args.ct)
        H.quantize(m)
        instrument(m)
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
                instrument(dec.model)
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
    ap.add_argument("--moe-routing", action="store_true", help="also the MoE layers' expert selection against fp32's")
    ap.add_argument("--ranges", help="also per range of positions within the window, e.g. 0,256,512,1024,2048 "
                    "(the long-context loss by position, iree_compiler_plan.md §8.22)")
    args = ap.parse_args()
    global ROUTES
    if args.moe_routing:
        ROUTES = {"ids": {}, "own": {}}
        args.ref_cache = None            # (the fp32 run gives the reference routing too)
    edges = [int(x) for x in args.ranges.split(",")] if args.ranges else []
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
    ref_routes = []                      # --moe-routing: per window {layer: [L, k]} of fp32
    if args.ref_cache and os.path.exists(args.ref_cache):
        a = np.load(args.ref_cache, mmap_mode="r")
        assert a.shape[:2] == (len(wins), args.window), (a.shape, len(wins), args.window)
        ref = [torch.from_numpy(np.array(a[w])) for w in range(len(wins))]
        order = [n for n in order if n != "fp32"]
    for name in order:
        t0 = time.time()
        f, ctx = build(name, args)
        nll, agree, corr, kl, n = 0.0, 0, 0.0, 0.0, 0
        pos = {"nll": np.zeros(args.window), "agree": np.zeros(args.window), "kl": np.zeros(args.window)}
        refs = [] if ref is None else ref
        with ctx():
            routes = []                  # per window (layer -> [L, k] ids, [L, k] fp32-router ids)
            for w, win in enumerate(wins):
                if ROUTES is not None:
                    ROUTES["ids"].clear()
                    ROUTES["own"].clear()
                lg = f(win)
                if ROUTES is not None:
                    routes.append(({l: np.stack(v) for l, v in ROUTES["ids"].items()},
                                   {l: np.stack(v) for l, v in ROUTES["own"].items()}))
                tgt = torch.tensor(win[1:] + [0])
                r = None if ref is None else ref[w]
                for c in range(0, len(win), 256):        # (row blocks: a 2048-token window's [L, V] fp32 is 1.2 GB)
                    lp = torch.log_softmax(lg[c:c + 256], -1)
                    nll_i = -lp.gather(1, tgt[c:c + 256, None])[:, 0]
                    if c + 256 >= len(win):
                        nll_i = nll_i[:-1]                # (the last position has no target)
                    nll += float(nll_i.sum())
                    pos["nll"][c:c + len(nll_i)] += nll_i.numpy()
                    if r is None:
                        continue
                    rc = r[c:c + 256]
                    rlp = torch.log_softmax(rc, -1)
                    agree_i = (lg[c:c + 256].argmax(-1) == rc.argmax(-1))
                    kl_i = (rlp.exp() * (rlp - lp)).sum(-1)
                    agree += int(agree_i.sum())
                    kl += float(kl_i.sum())
                    pos["agree"][c:c + len(agree_i)] += agree_i.numpy()
                    pos["kl"][c:c + len(kl_i)] += kl_i.numpy()
                    del lp, rlp
                if r is None:
                    refs.append(lg)
                    continue
                corr += sum(float(np.corrcoef(lg[i].numpy(), r[i].numpy())[0, 1]) for i in range(len(win)))
                n += len(win)
        if ref is None:
            ref = refs
            if args.ref_cache:
                np.save(args.ref_cache, np.stack([x.numpy() for x in ref]))
        f = ctx = None                   # (the model freed before the next one loads: one at a time in memory)
        gc.collect()
        P = len(wins) * (args.window - 1)
        res[name] = {"ppl": math.exp(nll / P), "seconds": round(time.time() - t0, 1)}
        if ROUTES is not None and name == "fp32":
            ref_routes = [r[0] for r in routes]
        elif ROUTES is not None and routes and routes[0][0]:
            layers = sorted(ref_routes[0])
            per_layer = []
            for l in layers:
                a = np.concatenate([r[0][l] for r in routes])
                b = np.concatenate([rr[l] for rr in ref_routes])
                o = np.concatenate([r[1][l] for r in routes]) if routes[0][1] else None
                per_layer.append((overlap(a, b), overlap(a, o) if o is not None else (float("nan"),) * 2))
            share = float(np.mean([p[0][0] for p in per_layer]))
            exact = float(np.mean([p[0][1] for p in per_layer]))
            own = float(np.mean([p[1][0] for p in per_layer]))
            own_exact = float(np.mean([p[1][1] for p in per_layer]))
            worst = int(np.argmin([p[0][0] for p in per_layer]))
            res[name]["routing"] = {"share_vs_fp32": share, "same_set_vs_fp32": exact,
                                    "router_share_vs_fp32_router": own, "router_same_set": own_exact,
                                    "per_layer_share": [p[0][0] for p in per_layer]}
            print(f"    routing vs fp32: {100 * share:.1f}% of the experts in common, the same set at "
                  f"{100 * exact:.1f}% of positions (worst layer {layers[worst]}: {100 * per_layer[worst][0][0]:.1f}%); "
                  f"the W8A8 router vs an fp32 router on the same input: {100 * own:.1f}% ({100 * own_exact:.1f}% same set)",
                  flush=True)
        if n:
            res[name].update(top1=agree / n, corr=corr / n, kl=kl / n)
        if edges:                        # (positions [a, b) of every window; the PPL over the targets of [a, b))
            res[name]["ranges"] = {}
            for a, b in zip(edges, edges[1:]):
                b = min(b, args.window)
                if a >= b:
                    continue
                r = {"ppl": math.exp(pos["nll"][a:min(b, args.window - 1)].sum() / (len(wins) * (min(b, args.window - 1) - a)))}
                if n:
                    r.update(top1=pos["agree"][a:b].sum() / (len(wins) * (b - a)),
                             kl=pos["kl"][a:b].sum() / (len(wins) * (b - a)))
                res[name]["ranges"][f"{a}-{b}"] = r
                print(f"    [{a}, {b}) PPL {r['ppl']:8.3f}" +
                      (f"  top-1 {100 * r['top1']:.1f}%  KL {r['kl']:.4f}" if n else ""), flush=True)
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

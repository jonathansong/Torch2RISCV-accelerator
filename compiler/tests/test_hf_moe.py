#!/usr/bin/env python3
"""S1 of docs/moe_native_lowering_design.md: hf_generic's W8A8 Mixtral MoE block
(QMixtralMoe) against transformers' fp32 model.

  synthetic (default): a random 2-layer Mixtral (E=4, H=64, I=128):
    - block: Top-2 ids equal to HF's where the routing is clear (2nd vs 3rd
      logit apart), output within the W8A8 error, rows (prefill) = row by row;
    - export (hf_generic.export, decode + a 16-token prefill): no softmax op,
      the Top-2 over the padded [1, 16] scores (no sliced scores);
  --model DIR: a Mixtral-style checkpoint (build/llm_cache/TitanML-tiny-mixtral;
    Granite MoE: --layers 2 of build/llm_cache/granite-3.1-1b-a400m-base):
    decode of --tokens on the host, the W8A8 model (SaDecoder: int8 KV cache,
    sa attention, QLinear, QMixtralMoe) against HF fp32: the top token, and
    every layer's expert ids where HF's routing is clear (check_model).

  --model DIR --export OUT: also export that W8A8 model (decode + a 16-token
    prefill) to OUT for compile_sa.sh, the KV scales calibrated on --tokens.

    python3 compiler/tests/test_hf_moe.py [--model DIR [--layers N]] [--export DIR]
"""
import argparse
import os
import re
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "frontend"))
import hf_generic as hg  # noqa: E402


def synthetic_model(seed=0):
    from transformers import MixtralConfig, MixtralForCausalLM
    torch.manual_seed(seed)
    cfg = MixtralConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, num_local_experts=4,
                        num_experts_per_tok=2, max_position_embeddings=256, tie_word_embeddings=False)
    m = MixtralForCausalLM(cfg).eval()
    with torch.no_grad():                    # routing spread out: the default init's logits are ~0
        for layer in m.model.layers:
            layer.mlp.gate.weight.normal_(0.0, 0.5)
    return m


def sa_decoder(m, max_len, calib_tokens):
    """hf_generic.load(quant, rope, attn) on a model in memory, the KV scales
    calibrated on token ids (the checkpoints here have no tokenizer)."""
    hg.rope_rewrite(m, max_len)
    hg.norm_rewrite(m)
    hg.act_rewrite(m)
    kv = hg.calibrate_kv(m, calib_tokens, max_len)
    n = hg.quantize(m)
    print(f"  quantize: {n}")
    return hg.SaDecoder(m, max_len, kv)


def moe_block(layer):
    """A decoder layer's sparse MoE block (Mixtral: mlp, Granite MoE: block_sparse_moe)."""
    return layer.mlp if hasattr(layer, "mlp") else layer.block_sparse_moe


def router_of(block):
    return block.gate if hasattr(getattr(block, "gate", None), "weight") else block.router


def router_out(router, out):
    """A router's forward output -> (logits, ids): Mixtral's (logits, weights, ids),
    Granite MoE's (ids, weights, logits)."""
    return (out[2], out[0]) if type(router).__name__ == "GraniteMoeTopKRouter" else (out[0], out[2])


def hf_router_ids(block, x):
    r = router_of(block)
    return router_out(r, r(x.reshape(-1, x.shape[-1])))[1]


def check_block():
    """QMixtralMoe against MixtralSparseMoeBlock on random rows."""
    m = synthetic_model()
    block = m.model.layers[0].mlp
    q = hg.QMixtralMoe(block)
    torch.manual_seed(1)
    x = torch.randn(64, 1, block.gate.weight.shape[1])
    clear = agree = 0
    errs = []
    with torch.no_grad():
        for r in range(x.shape[0]):
            xr = x[r]
            logits = xr @ block.gate.weight.t()
            top = torch.sort(logits[0], descending=True).values
            _, ids, w = q.route(xr)
            want = hf_router_ids(block, xr)[0]
            assert abs(float(w.sum()) - 1.0) < 1e-6
            if float(top[1] - top[2]) > 0.05 * float(top[0].abs() + 1):
                clear += 1
                agree += sorted(ids[0].tolist()) == sorted(want.tolist())
                y, ref = q(xr[None]), block(xr[None])
                errs.append(float((y - ref).norm() / ref.norm()))
        rows = x[:4].reshape(1, 4, -1)
        by_row = torch.cat([q(x[r][None]) for r in range(4)], dim=1)
        assert torch.equal(q(rows), by_row), "rows not computed one at a time"
    print(f"block: ids agree with HF on {agree}/{clear} clearly routed rows, "
          f"output rel L2 vs HF fp32 mean {np.mean(errs):.4f} max {np.max(errs):.4f}")
    assert clear >= 32 and agree == clear, "Top-2 ids differ from HF's"
    assert np.max(errs) < 0.08, "W8A8 output error beyond 8%"


def check_export(out):
    """Export the synthetic model; the MoE graph has the forms the SA compiler
    takes (§11.1): no softmax op, Top-2 over [1, 16] scores."""
    m = synthetic_model()
    w = sa_decoder(m, 64, list(range(1, 33)))
    mlir, _ = hg.export(w, out, prefill=16)
    text = open(mlir).read()
    softmax = len(re.findall(r"aten\._?softmax", text))
    sorts = re.findall(r"torch\.aten\.(?:sort|topk)\b[^\n]*", text)
    # the input's type: the first after the colon
    shapes = sorted({re.search(r": !torch\.vtensor<\[([0-9,]+)\],f32>", line).group(1) for line in sorts})
    print(f"export: {mlir}: softmax ops {softmax}, Top-2 ops {len(sorts)} over f32 {shapes}")
    assert softmax == 0, "a softmax op left (linalg.softmax: host)"
    # decode: 2 layers x 1 row; prefill / prefill_kv: 2 layers x 16 rows each
    assert len(sorts) == 2 * (1 + 16 + 16), f"expected one Top-2 per layer and row, got {len(sorts)}"
    assert shapes == [f"1,{hg.ROUTER_PAD}"], f"Top-2 not over the padded scores: {shapes}"


MARGIN = 0.05     # HF's 2nd vs 3rd router logit: closer, a W8A8 model may pick either (reported, not checked)


def check_model(model_dir, tokens, max_len=64, layers=None):
    """Top tokens equal to HF fp32's; expert ids equal where HF's routing is
    clear (MARGIN); and on the W8A8 model's own hidden states, its W8A8 router
    picks the ids an fp32 router would (the router's quantization alone). HF
    runs first: hf_generic's rewrites patch shared transformers functions."""
    torch.set_grad_enabled(False)
    ref = hg.hf_model(model_dir, layers)
    ref_logits_l = []          # [layer] -> router logits [T, E] (one call over all the tokens)
    hooks = [router_of(moe_block(layer)).register_forward_hook(
                 lambda mod, i, o: ref_logits_l.append(router_out(mod, o)[0].clone()))
             for layer in ref.model.layers]
    ref_logits = ref(torch.tensor([tokens])).logits[0]
    for h in hooks:
        h.remove()
    del ref

    m = hg.hf_model(model_dir, layers)
    fp32_router = [router_of(moe_block(layer)).weight.detach().clone() for layer in m.model.layers]
    k = router_of(moe_block(m.model.layers[0])).top_k
    w = sa_decoder(m, max_len, tokens)
    steps = []                 # per call of a block: (layer, W8A8 ids, fp32-router ids on the same x)
    for L, layer in enumerate(m.model.layers):
        mod = moe_block(layer)
        def route(x, route=mod.route, L=L):
            r = route(x)
            v, f = torch.topk((x @ fp32_router[L].t())[0], k + 1)
            steps.append((L, r[1][0].tolist(), f[:k].tolist(), float(v[k - 1] - v[k])))
            return r
        mod.route = route
    L = len(m.model.layers)
    tok_ok = ids_ok = router_ok = True
    near = []
    for p, t in enumerate(tokens):
        lg = w(torch.tensor([[t]]), torch.tensor([[p]]))[0, -1]
        r = ref_logits[p]
        tok_ok &= int(lg.argmax()) == int(r.argmax())
        line = []
        for l, got, fp, fgap in steps[-L:]:
            v, i = torch.sort(ref_logits_l[l][p], descending=True)
            want, gap = i[:k].tolist(), float(v[k - 1] - v[k])
            same = sorted(got) == sorted(want)
            if fgap > MARGIN:                  # (the fp32 router's own near ties: either way)
                router_ok &= sorted(got) == sorted(fp)
            if gap > MARGIN:
                ids_ok &= same
            elif not same:
                near.append(f"L{l} pos {p}: {got} vs HF {want} (HF gap {gap:.4f})")
            line.append(f"L{l} {got}{'' if same else ' HF ' + str(want)}")
        print(f"  pos {p} token {t}: top {int(lg.argmax())} (HF {int(r.argmax())}), "
              f"logit rel L2 {float((lg - r).norm() / r.norm()):.4f}, experts: {'; '.join(line)}")
    for n in near:
        print(f"  near tie, differs: {n}")
    # (a truncated model's logits are nearly flat: its top token is reported, not checked)
    assert tok_ok or layers, "top tokens differ from HF"
    assert ids_ok, f"expert ids differ from HF where its routing is clear (gap > {MARGIN})"
    assert router_ok, "the W8A8 router picks other ids than an fp32 router on the same hidden state (clear gap)"
    print(f"model: {model_dir}: {len(tokens)} steps: top tokens {'equal to HF fp32' if tok_ok else 'reported'}; "
          f"expert ids equal where "
          f"HF's gap > {MARGIN}; W8A8 router = fp32 router on the same hidden states where its gap > {MARGIN}")


def export_model(model_dir, tokens, out, max_len=64, prefill=16, layers=None):
    m = hg.hf_model(model_dir, layers)
    hg.export(sa_decoder(m, max_len, tokens), out, prefill=prefill)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="a Mixtral checkpoint directory")
    ap.add_argument("--tokens", default="1,42,43", help="token ids for --model")
    ap.add_argument("--layers", type=int, help="--model: only its first N layers")
    ap.add_argument("--export", help="the export's directory (synthetic: default a temporary one)")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    if args.model:
        tokens = [int(t) for t in args.tokens.split(",")]
        check_model(args.model, tokens, layers=args.layers)
        if args.export:
            export_model(args.model, tokens, args.export, layers=args.layers)
        return
    check_block()
    if args.export:
        check_export(args.export)
    else:
        with tempfile.TemporaryDirectory() as d:
            check_export(d)
    print("PASS")


if __name__ == "__main__":
    main()

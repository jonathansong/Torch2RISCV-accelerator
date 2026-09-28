#!/usr/bin/env python3
"""Stage C5.5: export a HuggingFace decoder (compiler/frontend/qhf.py) like export.py does QLlama.

1. HFConfig + weights (config.json, *.safetensors) -> KV scales calibrated with
   the fp32 model -> QModel -> iree-turbine export: decode(token, pos, valid[T])
   with T dynamic, the KV cache as mutable globals, the weights externalized
   (<out>/qllama.mlir, <out>/qllama.irpa: the names compile_sa.sh expects);
2. llvm-cpu on the host: the IREE module against eager QModel (the export),
   and QModel against the fp32 model (the quantization: top-1 agreement);
3. <out>/prompt.npy, <out>/kv_scales.npy for the device tests.

    python3 compiler/frontend/export_hf.py --model build/llm_cache/SmolLM2-135M --out build/c55/smollm2
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import export as E  # noqa: E402
import qhf  # noqa: E402

PROMPT = "Once upon a time, there was a little girl named Lily. She"
CALIB = ("Once upon a time, there was a little girl named Lily. She lived in a big house with her family. "
         "The quick brown fox jumps over the lazy dog. In 1492, Columbus sailed across the Atlantic Ocean.")


def build(model_dir, seq_len, layers=None):
    """(cfg, weights, QModel, tokenizer); layers: keep only the first ones (sim tests of big models)."""
    cfg = qhf.HFConfig(model_dir, seq_len)
    w = qhf.load_weights(model_dir, cfg)
    if layers:
        cfg.layers = layers
        for k, v in w.items():
            if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[0] > layers and k not in ("tok", "wcls"):
                w[k] = v[:layers]
    tok = qhf.BpeTokenizer(model_dir)
    kv = qhf.calibrate_kv(cfg, w, tok.encode(CALIB))
    return cfg, w, kv, tok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="HuggingFace directory (config.json, *.safetensors, tokenizer.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--layers", type=int, help="keep only the first layers")
    ap.add_argument("--tokens", type=int, default=12, help="decode steps compared on the host")
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--prefill", type=int, default=0, help="also export prefill with chunks of this many tokens")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg, w, kv, tok = build(args.model, args.seq_len, args.layers)
    print(cfg)
    m = qhf.QModel(cfg, w, kv).eval()
    mlir, irpa = E.export(m, cfg, args.d, args.out, prefill=args.prefill)
    vmfb, _ = E.compile_host(mlir, args.out)
    im = E.IreeModel(vmfb, irpa)
    prompt = tok.encode(PROMPT)
    np.save(os.path.join(args.out, "prompt.npy"), np.array(prompt, np.int64))
    np.save(os.path.join(args.out, "kv_scales.npy"), kv)
    f = qhf.Fp32Model(cfg, w)
    worst = 0.0
    agree = same = 0
    corr = 1.0
    t0 = time.time()
    steps = prompt[:args.tokens]
    with torch.no_grad():
        for pos, t in enumerate(steps):
            a = qhf_inputs(t, pos, args.d)
            got = im(*a)
            e = m(*a).numpy()
            worst = max(worst, float(np.abs(got - e).max() / np.abs(e).max()))
            corr = min(corr, float(np.corrcoef(got, e)[0, 1]))
            same += int(got.argmax() == e.argmax())
            agree += int(e.argmax() == f.forward(t, pos).numpy().argmax())
    # fp32 rounding differs between IREE and eager torch (exp, sums); an int8
    # rounding flip then grows through the layers and can flip a close argmax:
    # judged by correlation (the sa device is compared with eager's device SFU)
    ok = corr > 0.98
    print(f"{len(steps)} steps in {time.time() - t0:.1f} s: IREE (llvm-cpu) vs eager QModel: argmax {same}/{len(steps)}, "
          f"min correlation {corr:.5f}, max |diff| / max|logit| {worst:.2e}; QModel vs fp32 top-1 {agree}/{len(steps)}")
    if args.prefill:
        ok &= E.check_prefill(vmfb, irpa, qhf.QModel(cfg, w, kv).eval(), steps, args.prefill, args.d, qhf_inputs)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def qhf_inputs(token, pos, d):
    pp = (pos // d + 1) * d
    valid = torch.zeros(pp, dtype=torch.float32)
    valid[:pos + 1] = 1.0
    return torch.tensor([token]), torch.tensor([pos]), valid


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""The compressed-tensors importer (compiler/frontend/ct_import.py,
docs/generality_implementation_plan.md WP2) on synthetic checkpoints
(synth_hf.py --ct): the variants it must refuse fail with a message naming the
field; for the accepted ones, after the generic frontend's load (quantize,
RoPE rewrite), every linear layer's int8 rows and fp32 scales are the
checkpoint's, byte for byte (the q / k rows permuted as the rewrite does),
-128 kept, the dequantized fp32 weights q * s, and the embedding quantized once
and shared with the tied classifier. With --e2e also end to end: Llama-shaped
checkpoints (neg128, bf16-scale; dim 256, head_dim 64) through
test_hf_generic.py --quant --rope --attn (export, compile, the sim against the
torch W8A8 model step by step and teacher-forced), with the overlay's
configuration (SA_D, SA_GEMV_PORTS from board_env.sh). Without --prefill: on
random-weight models prefill + decode is not always bit-exact with decode only,
self-quantized models alike (seeds 1-3: 0 / 4 / 4 of 5 rows), an open issue of
the generic path that the import does not change.

    python3 compiler/tests/test_ct_import.py [--out build/ct_synth] [--e2e]
"""
import argparse
import os
import resource
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(HERE, "..", "frontend")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, FRONTEND)
import ct_import  # noqa: E402
import hf_generic as H  # noqa: E402
import synth_hf  # noqa: E402

# what each refused variant's message must name
EXPECT = {"group": "strategy", "static-act": "dynamic", "asym": "symmetric", "int4": "num_bits",
          "fp8": "format", "missing-scale": "missing", "bad-shape": "shape", "kv-scheme": "kv_cache_scheme"}


def synth(out, variant, arch="Qwen3ForCausalLM", dim=64, hidden=256):
    d = os.path.join(out, variant if dim == 64 else f"{variant}_{arch[:5].lower()}{dim}")
    subprocess.run([sys.executable, os.path.join(FRONTEND, "synth_hf.py"), "--out", d, "--tokenizer",
                    os.path.join(REPO, "build", "llm_cache", "SmolLM2-135M"), "--arch", arch, "--dim", str(dim),
                    "--hidden", str(hidden), "--layers", "2", "--heads", "4", "--kv-heads", "2", "--ct", variant],
                   check=True, stdout=subprocess.DEVNULL)
    return d


def e2e(out, variant):
    """export, compile, sim (test_hf_generic.py); the generic frontend's Qwen3
    form still leaves dispatches to the host, so a Llama-shaped model."""
    d = synth(out, variant, "LlamaForCausalLM", 256, 768)
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "test_hf_generic.py"), "--model", d,
                        "--out", d + "_out", "--quant", "--rope", "--attn", "--mb", "200"], capture_output=True, text=True)
    keep = ("compiled in", "steps vs", "teacher-forced", "prefill M", "hf_generic")
    for l in r.stdout.splitlines():
        if any(k in l for k in keep):
            print("    " + l)
    if r.returncode:
        print(r.stderr[-2000:])
    return r.returncode == 0


def check_accepted(d, variant):
    t = ct_import.Tensors(d)
    m = H.load(d, quant=True, rope=True)
    hd = m.model.config.head_dim
    perm = H.interleave_perm(hd)
    bad, n, neg = [], 0, 0
    for name, mod in m.model.named_modules():
        if not isinstance(mod, H.QLinear) or name == "lm_head":
            continue
        q = t.get(name + ".weight")
        s = t.get(name + ".weight_scale").to(torch.float32).reshape(-1)
        if name.endswith(("q_proj", "k_proj")):          # the RoPE rewrite's row permutation
            heads = q.shape[0] // hd
            q = q.reshape(heads, hd, -1)[:, perm].reshape(q.shape)
            s = s.reshape(heads, hd)[:, perm].reshape(-1)
        if mod.wq.dtype != torch.int8 or mod.wq.numpy().tobytes() != q.numpy().tobytes():
            bad.append(name + " (int8)")
        if mod.s_w.dtype != torch.float32 or mod.s_w.numpy().tobytes() != s.numpy().tobytes():
            bad.append(name + " (scale)")
        n += 1
        neg += int((mod.wq == -128).sum())
    emb = m.model.model.embed_tokens
    head = m.model.lm_head
    shared = isinstance(emb, H.QEmbedding) and isinstance(head, H.QLinear) and emb.q is head.wq
    # the dequantized fp32 weights (the references' model) are q * s
    raw = H.hf_model(d)
    deq = all(torch.equal(mod.weight, mod.weight._sa_qs[0].float() * mod.weight._sa_qs[1][:, None])
              for mod in raw.modules() if isinstance(mod, torch.nn.Linear) and hasattr(mod.weight, "_sa_qs"))
    ok = not bad and n == 14 and shared and deq and (neg > 0) and (variant != "neg128" or neg >= 14 * 64)
    print(f"  {variant}: {n} linear layers byte-identical with the checkpoint{'' if not bad else ' EXCEPT ' + str(bad)}, "
          f"{neg} weights = -128, embedding / classifier {'one shared table' if shared else 'NOT SHARED'}, "
          f"dequantized weights {'q * s' if deq else 'WRONG'}: {'ok' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(REPO, "build", "ct_synth"))
    ap.add_argument("--e2e", action="store_true", help="also export, compile and run on the sim (minutes)")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    ok = True
    print("refused:")
    for v in synth_hf.CT_BAD:
        d = synth(args.out, v)
        try:
            H.hf_model(d)
            print(f"  {v}: ACCEPTED (must be refused)")
            ok = False
        except ct_import.UnsupportedCheckpoint as e:
            good = EXPECT[v] in str(e)
            ok &= good
            print(f"  {v}: {e}{'' if good else '  (expected the message to name ' + EXPECT[v] + ')'}")
    print("accepted:")
    for v in synth_hf.CT_OK:
        ok &= check_accepted(synth(args.out, v), v)
    if args.e2e:
        print(f"end to end (SA_D {os.environ.get('SA_D', '8')}, GEMV ports {os.environ.get('SA_GEMV_PORTS', '0')}):")
        for v in ("neg128", "bf16-scale"):
            print(f"  {v}:")
            ok &= e2e(args.out, v)
    print(f"peak host memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    print("ct_import PASS" if ok else "ct_import FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

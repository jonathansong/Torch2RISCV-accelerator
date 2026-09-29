#!/usr/bin/env python3
"""The generic frontend (docs/iree_compiler_plan.md §8.16): an unmodified
HuggingFace decoder exported (hf_generic.py), compiled for the sa device with
the host fallback, run on the sim (sa-llm-run --abi=hf) token by token against
transformers' eager fp32 model: correlation, max |diff|, argmax per step, and
the split between the accelerator and the host (SA_STATS).

    python3 compiler/tests/test_hf_generic.py --model build/llm_cache/SmolLM2-135M --out build/hfgen/smollm2_raw [--skip-export] [--mb 1100]
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
COMPILER = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(COMPILER, "frontend"))
import hf_generic as H  # noqa: E402
import test_c6p as T  # noqa: E402


class device_sfu:
    """with device_sfu(): torch's exp / reciprocal / rsqrt / sigmoid / silu compute as
    the sa device's SFU does (llm/ref_model.py SfuExact; as qhf.device_sfu): the
    reference for a quantized model, whose int8 roundings flip with 1-ulp
    differences (compare with the device up to reduction order)."""
    def __enter__(self):
        import torch.nn.functional as F
        sys.path.insert(0, os.path.join(COMPILER, "..", "llm"))
        from ref_model import SfuExact
        wrap = lambda f: (lambda x, *a, **k: torch.from_numpy(np.ascontiguousarray(
            f(x.detach().numpy().astype(np.float32)))).reshape(x.shape))
        self.saved = (torch.exp, torch.reciprocal, torch.rsqrt, torch.sigmoid, F.silu)
        exp, recip, rsqrt = wrap(SfuExact.exp), wrap(SfuExact.recip), wrap(SfuExact.rsqrt)
        torch.exp, torch.reciprocal, torch.rsqrt = exp, recip, rsqrt
        torch.sigmoid = lambda x: recip(1.0 + exp(x * -1.0))
        F.silu = lambda x, inplace=False: x * recip(1.0 + exp(x * -1.0))
        return self

    def __exit__(self, *a):
        import torch.nn.functional as F
        torch.exp, torch.reciprocal, torch.rsqrt, torch.sigmoid, F.silu = self.saved


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--mb", type=int, default=1100, help="sim window (the fp32 weights: ~4 bytes per parameter)")
    ap.add_argument("--prompt", default="1,504,3144,17,1812")
    ap.add_argument("--generate", type=int, default=4)
    ap.add_argument("--quant", action="store_true", help="W8A8 (QLinear / QEmbedding); the reference is then "
                    "the quantized model in torch (and the fp32 model's argmax for information)")
    ap.add_argument("--rope", action="store_true", help="RoPE tables and the pair swap (F3)")
    ap.add_argument("--attn", action="store_true", help="int8 KV cache and the sa attention (F4)")
    ap.add_argument("--kv-scales", help="KV scales (.npy) instead of calibrating (F4)")
    ap.add_argument("--prefill", type=int, default=0, help="also export prefill of M tokens (needs --attn); the "
                    "sim run with prefill + decode must be bit-exact with decode only (plan §8.17)")
    ap.add_argument("--min-corr", type=float, default=0.999)
    args = ap.parse_args()
    # the wrapper (static cache at the input positions) vs plain HF on the whole prompt
    from transformers import AutoModelForCausalLM
    pr = [int(t) for t in args.prompt.split(",")]
    plain = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa").eval()
    full = plain(torch.tensor([pr])).logits[0]
    wr = H.HFDecoder(plain, 256)
    dmax = max(float((wr(torch.tensor([[t]]), torch.tensor([[i]]))[0, -1] - full[i]).abs().max()) for i, t in enumerate(pr))
    print(f"HFDecoder vs plain HF (the whole prompt at once, fp32): max |diff| {dmax:.3g}")
    # the teacher-forced reference (plain HF fp32 on a fixed text), before any
    # rewrite: rope_rewrite replaces rotate_half in the transformers modeling
    # modules for the whole process
    import export_hf
    from hf_tokenizer import BpeTokenizer
    tf_toks = BpeTokenizer(args.model).encode(export_hf.CALIB)[:40]
    tf_ref = plain(torch.tensor([tf_toks])).logits[0].detach().numpy()
    if dmax > 1e-3:
        print("hf_generic FAIL (the wrapper)")
        return 1
    del plain, wr
    if args.rope:                        # the rewrite keeps the fp32 model's logits (checked here)
        base = H.load(args.model)
        pr = [int(t) for t in args.prompt.split(",")]
        a = [base(torch.tensor([[t]]), torch.tensor([[i]]))[0, -1] for i, t in enumerate(pr)]
        rw = H.load(args.model, rope=True)
        b = [rw(torch.tensor([[t]]), torch.tensor([[i]]))[0, -1] for i, t in enumerate(pr)]
        d = max(float((x - y).abs().max()) for x, y in zip(a, b))
        print(f"rope rewrite vs the original model (fp32, torch): max |diff| {d:.3g}")
        del base, rw
    w = H.load(args.model, quant=args.quant, rope=args.rope, attn=args.attn, kv_scales=args.kv_scales)
    if not args.skip_export:
        H.export(w, args.out, args.prefill)
        t0 = time.time()
        log = subprocess.run([os.path.join(COMPILER, "scripts", "compile_sa.sh"), args.out],
                             capture_output=True, text=True, env=dict(os.environ, SA_COMPILE_FLAGS="--iree-sa-codegen-report"))
        if log.returncode:
            print(log.stderr[-3000:])
            return 1
        ok_n = log.stderr.count(": ok\n")
        host_n = log.stderr.count(": on the host")
        print(f"compiled in {time.time() - t0:.0f} s: {ok_n} executables for the accelerator, {host_n} for the host")
    prompt = [int(t) for t in args.prompt.split(",")]
    ref, tok = [], None
    import contextlib
    with device_sfu() if args.quant else contextlib.nullcontext():
        w = H.load(args.model, quant=args.quant, rope=args.rope, attn=args.attn, kv_scales=args.kv_scales)   # (fresh state)
        for pos in range(len(prompt) + args.generate):
            tok = prompt[pos] if pos < len(prompt) else tok
            lg = w(torch.tensor([[tok]]), torch.tensor([[pos]]))[0, -1].numpy()
            ref.append(lg)
            tok = int(np.argmax(lg))
    ref = np.stack(ref)
    sock = f"/tmp/sa_hf_{os.getpid()}.sock"
    srv = subprocess.Popen([T.PY, os.path.join(COMPILER, "sim", "sa_sim_server.py"), "--d", "8", "--mb", str(args.mb),
                            "--socket", sock, "--once"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    lp = os.path.join(args.out, "logits_hf.f32")
    r = subprocess.run([T.RUN, "--device=sa", f"--module={args.out}/sa.vmfb",
                        f"--parameters=model={args.out}/sa_packed.irpa", "--abi=hf", "--tokens=" + args.prompt,
                        f"--generate={args.generate}", f"--logits_out={lp}"],
                       capture_output=True, text=True, env=dict(os.environ, SA_SIM_SOCKET=sock, SA_STATS="1"))
    srv.wait(timeout=120)
    if r.returncode:
        print(r.stderr[-3000:])
        return 1
    got = np.fromfile(lp, np.float32).reshape(-1, ref.shape[1])
    corr = [float(np.corrcoef(got[i], ref[i])[0, 1]) for i in range(len(got))]
    same = [int(np.argmax(got[i])) == int(np.argmax(ref[i])) for i in range(len(got))]
    stats = [l for l in r.stderr.splitlines() if l.startswith("sa:")]
    print(f"{len(got)} steps vs the torch model{' (W8A8)' if args.quant else ' (fp32)'}: min corr {min(corr):.6f}, max |diff| "
          f"{float(np.max(np.abs(got - ref))):.3g}, argmax equal in {sum(same)}/{len(same)}; {stats[-1] if stats else ''}")
    ok = len(got) == len(ref) and min(corr) > args.min_corr and all(same)
    if args.quant:
        # a quantized model: int8 roundings flip with the order of operations (the
        # device's vs torch's), and the flips grow over the steps. The criterion
        # is the quality: teacher-forced top-1 agreement with plain HF fp32 on a
        # fixed text (the hand-written qhf path scores 36 / 40 on SmolLM2)
        toks, fp = tf_toks, tf_ref
        srv = subprocess.Popen([T.PY, os.path.join(COMPILER, "sim", "sa_sim_server.py"), "--d", "8", "--mb",
                                str(args.mb), "--socket", sock, "--once"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        time.sleep(3)
        tq = os.path.join(args.out, "logits_tf.f32")
        subprocess.run([T.RUN, "--device=sa", f"--module={args.out}/sa.vmfb",
                        f"--parameters=model={args.out}/sa_packed.irpa", "--abi=hf",
                        "--tokens=" + ",".join(map(str, toks)), "--generate=0", f"--logits_out={tq}"],
                       capture_output=True, text=True, env=dict(os.environ, SA_SIM_SOCKET=sock))
        srv.wait(timeout=120)
        tf = np.fromfile(tq, np.float32).reshape(-1, fp.shape[1])
        n = min(len(tf), len(fp))
        agree = int(np.sum(tf[:n].argmax(1) == fp[:n].argmax(1)))
        mcorr = float(np.mean([np.corrcoef(tf[i], fp[i])[0, 1] for i in range(n)]))
        print(f"teacher-forced, {n} positions: top-1 agreement with plain HF fp32 {agree}/{n}, mean corr {mcorr:.4f}")
        ok = n == len(toks) and agree >= int(0.85 * n) and mcorr > 0.95
    if args.prefill:
        # prefill + decode vs decode only: the prompt's last position, then the generated steps
        M = args.prefill
        pp = prompt if len(prompt) >= 2 * M else [int(v) for v in np.resize(np.array(prompt), 2 * M + M // 2)]
        def run(pre):
            srv = subprocess.Popen([T.PY, os.path.join(COMPILER, "sim", "sa_sim_server.py"), "--d", "8", "--mb",
                                    str(args.mb), "--socket", sock, "--once"], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
            time.sleep(3)
            lp = os.path.join(args.out, f"logits_p{pre}.f32")
            r2 = subprocess.run([T.RUN, "--device=sa", f"--module={args.out}/sa.vmfb",
                                 f"--parameters=model={args.out}/sa_packed.irpa", "--abi=hf",
                                 "--tokens=" + ",".join(map(str, pp)), f"--generate={args.generate}",
                                 f"--logits_out={lp}"] + ([f"--prefill={pre}"] if pre else []),
                                capture_output=True, text=True, env=dict(os.environ, SA_SIM_SOCKET=sock))
            srv.wait(timeout=120)
            if r2.returncode:
                print(r2.stderr[-2000:])
            lg = np.fromfile(lp, np.float32)
            return r2.stdout.strip().splitlines()[0], lg.reshape(-1, ref.shape[1]), r2.stdout
        dt, dl, _ = run(0)
        pt, pl, pout = run(M)
        rows = dl[len(pp) - 1:]
        exact = [pl[i].tobytes() == rows[i].tobytes() for i in range(min(len(pl), len(rows)))]
        same = dt.split(":")[1].split() == pt.split(":")[1].split()
        pline = [l for l in pout.splitlines() if l.startswith("prefill")]
        print(f"prefill M = {M}, a {len(pp)}-token prompt: logits bit-exact with decode only in {sum(exact)}/"
              f"{len(exact)} rows, tokens {'identical' if same else 'DIFFERENT'}; {pline[0] if pline else ''}")
        ok &= len(exact) == args.generate + 1 and all(exact) and same
    print("hf_generic PASS" if ok else "hf_generic FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

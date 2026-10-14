#!/usr/bin/env python3
"""The generic frontend (docs/iree_compiler_plan.md §8.16): an unmodified
HuggingFace decoder exported (hf_generic.py), compiled for the sa device with
the host fallback, run on the sim (sa-llm-run --abi=hf) token by token against
transformers' eager fp32 model: correlation, max |diff|, argmax per step, and
the split between the accelerator and the host (SA_STATS).

    python3 compiler/tests/test_hf_generic.py --model build/llm_cache/SmolLM2-135M --out build/hfgen/smollm2_raw [--skip-export] [--mb 1100]
"""
import argparse
import gc
import math
import os
import subprocess
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def overlay_env():
    """SA_D / SA_GEMV_PORTS (compile_sa.sh's --iree-sa-d / --iree-sa-gemv-ports) from
    SA_KV260_CONFIG as compiler/scripts/board_env.sh derives them; set
    explicitly they must agree with it (a D = 8 compile for the D = 16 overlay
    passes in the sim and fails only on the device)."""
    cfg = os.environ.get("SA_KV260_CONFIG", "")
    m = re.match(r"d(\d+)_", cfg)
    if not m:
        return
    want = {"SA_D": m.group(1),
            "SA_GEMV_PORTS": "4" if "_gemv_np4" in cfg else "2" if "_gemv_np2" in cfg else "1" if "_gemv" in cfg else "0"}
    for k, v in want.items():
        if os.environ.get(k, v) != v:
            sys.exit(f"{k}={os.environ[k]} but SA_KV260_CONFIG={cfg} means {k}={v}")
        os.environ[k] = v


overlay_env()
D = int(os.environ.get("SA_D", "8"))   # the array size (the overlay's; compile_sa.sh passes --iree-sa-d)
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
    ap.add_argument("--kv-channel", default="", help="k / v / kv: those KV scales per cache channel (F4)")
    ap.add_argument("--kv-calib", help="FILE[:TOKENS] text to calibrate the KV scales on (F4)")
    ap.add_argument("--prefill", type=int, default=0, help="also export prefill of M tokens (needs --attn); the "
                    "sim run with prefill + decode must be bit-exact with decode only (plan §8.17)")
    ap.add_argument("--prefill-all", action="store_true", help="with --prefill: also export prefill_all and check "
                    "its logits of every prompt position (sa-llm-run --prefill_all_out) against decode only")
    ap.add_argument("--min-corr", type=float, default=0.999)
    ap.add_argument("--layers", type=int, help="only the first layers (a truncated model)")
    ap.add_argument("--board-bundle", help="with --prefill: stage the board test here (board_llm.py)")
    ap.add_argument("--board-generate", type=int, default=16)
    ap.add_argument("--static-len", action="store_true", help="export the attention over the whole cache (default "
                    "with --attn: over valid[T], T dynamic; sa-llm-run --abi=hfv --pad=D)")
    ap.add_argument("--dyn-len", action="store_true", help=argparse.SUPPRESS)      # (the default)
    ap.add_argument("--max-len", type=int, default=256, help="the KV cache's length (the longest context); "
                    "sa-llm-run then needs --valid_len of it (written to sa_args.txt)")
    args = ap.parse_args()
    global ABI
    args.dyn_len = args.attn and not args.static_len
    abi_file = os.path.join(args.out, "abi.txt")
    if args.skip_export:                 # (the export's: a build before --dyn-len became the default has none: hf)
        args.dyn_len = os.path.exists(abi_file) and open(abi_file).read().strip() == "hfv"
    ABI = ["--abi=hfv", f"--pad={D}"] if args.dyn_len else ["--abi=hf"]
    if args.max_len != 256:              # (sa-llm-run's default valid_len)
        ABI.append(f"--valid_len={args.max_len}")
    torch.set_grad_enabled(False)        # (an autograd graph would keep each fp32 model alive past its del)
    # the wrapper (static cache at the input positions) vs plain HF on the whole prompt
    pr = [int(t) for t in args.prompt.split(",")]
    plain = H.hf_model(args.model, args.layers)
    full = plain(torch.tensor([pr])).logits[0]
    wr = H.HFDecoder(plain, max(256, args.max_len))
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
    gc.collect()                         # (HF modules hold reference cycles: freed only by the collector)
    if args.rope:                        # the rewrite keeps the fp32 model's logits (checked here)
        base = H.load(args.model, args.max_len, layers=args.layers)
        pr = [int(t) for t in args.prompt.split(",")]
        a = [base(torch.tensor([[t]]), torch.tensor([[i]]))[0, -1] for i, t in enumerate(pr)]
        del base                         # (one fp32 model at a time: Llama-1B's is ~5 GB)
        gc.collect()
        rw = H.load(args.model, args.max_len, rope=True, layers=args.layers)
        b = [rw(torch.tensor([[t]]), torch.tensor([[i]]))[0, -1] for i, t in enumerate(pr)]
        del rw
        gc.collect()
        d = max(float((x - y).abs().max()) for x, y in zip(a, b))
        print(f"rope rewrite vs the original model (fp32, torch): max |diff| {d:.3g}")
    if not args.skip_export:
        w = H.load(args.model, args.max_len, quant=args.quant, rope=args.rope, attn=args.attn, kv_scales=args.kv_scales, layers=args.layers,
                   kv_channel=args.kv_channel, kv_calib=H.kv_calib_arg(args.kv_calib))
        H.export(w, args.out, args.prefill, args.prefill_all, args.dyn_len)
        del w                            # (freed before the compile and the reload below)
        gc.collect()
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
        w = H.load(args.model, args.max_len, quant=args.quant, rope=args.rope, attn=args.attn, kv_scales=args.kv_scales, layers=args.layers,
                   kv_channel=args.kv_channel, kv_calib=H.kv_calib_arg(args.kv_calib))   # (fresh state)
        for pos in range(len(prompt) + args.generate):
            tok = prompt[pos] if pos < len(prompt) else tok
            lg = w(torch.tensor([[tok]]), torch.tensor([[pos]]))[0, -1].numpy()
            ref.append(lg)
            tok = int(np.argmax(lg))
        # the torch model on the teacher-forced text (its rows are rewritten in
        # order; the attention reads only positions <= the current one)
        eager_tf = np.stack([w(torch.tensor([[t]]), torch.tensor([[p]]))[0, -1].numpy()
                             for p, t in enumerate(tf_toks)]) if args.quant else None
    ref = np.stack(ref)
    sock = f"/tmp/sa_hf_{os.getpid()}.sock"
    srv = subprocess.Popen([T.PY, os.path.join(COMPILER, "sim", "sa_sim_server.py"), "--d", str(D), "--mb", str(args.mb),
                            "--socket", sock, "--once"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    lp = os.path.join(args.out, "logits_hf.f32")
    r = subprocess.run([T.RUN, "--device=sa", f"--module={args.out}/sa.vmfb",
                        f"--parameters=model={args.out}/sa_packed.irpa", *ABI, "--tokens=" + args.prompt,
                        f"--generate={args.generate}", f"--logits_out={lp}"],
                       capture_output=True, text=True, env=dict(os.environ, SA_SIM_SOCKET=sock, SA_STATS="1"))
    if r.returncode:
        srv.kill()             # (it waits for a connection the run may never have made)
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
        # device's vs torch's), and the flips grow over the steps. The criterion:
        # teacher-forced on a fixed text, the device's top-1 agreement with plain
        # HF fp32 as good as the torch W8A8 model's (-2 of 40), mean corr alike
        # (-0.01): the compiler keeps the quantized model's quality. The quality
        # of the quantization itself is printed (SmolLM2: 34-36 / 40; Qwen3-0.6B
        # 25 / 40, plan C6.2), not judged here.
        toks, fp = tf_toks, tf_ref
        srv = subprocess.Popen([T.PY, os.path.join(COMPILER, "sim", "sa_sim_server.py"), "--d", str(D), "--mb",
                                str(args.mb), "--socket", sock, "--once"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        time.sleep(3)
        tq = os.path.join(args.out, "logits_tf.f32")
        subprocess.run([T.RUN, "--device=sa", f"--module={args.out}/sa.vmfb",
                        f"--parameters=model={args.out}/sa_packed.irpa", *ABI,
                        "--tokens=" + ",".join(map(str, toks)), "--generate=0", f"--logits_out={tq}"],
                       capture_output=True, text=True, env=dict(os.environ, SA_SIM_SOCKET=sock))
        srv.wait(timeout=120)
        tf = np.fromfile(tq, np.float32).reshape(-1, fp.shape[1])
        n = min(len(tf), len(fp))
        agree = int(np.sum(tf[:n].argmax(1) == fp[:n].argmax(1)))
        mcorr = float(np.mean([np.corrcoef(tf[i], fp[i])[0, 1] for i in range(n)]))
        e_agree = int(np.sum(eager_tf.argmax(1) == fp.argmax(1)))
        e_mcorr = float(np.mean([np.corrcoef(eager_tf[i], fp[i])[0, 1] for i in range(len(fp))]))
        # the device vs the torch model directly: where their argmax differ (d
        # positions, rounding flips grown over the layers) each side wins about
        # half; the agreement's spread is then ~sqrt(d), so the tolerance is
        # max(2, 2 sqrt(d)) (a 40-position gate; the quality itself:
        # quality_eval.py), and their logits must stay close (mean corr >= 0.98)
        dd = int(np.sum(tf[:n].argmax(1) != eager_tf[:n].argmax(1)))
        dcorr = float(np.mean([np.corrcoef(tf[i], eager_tf[i])[0, 1] for i in range(n)]))
        tol = max(2, int(np.ceil(2 * np.sqrt(dd))))
        print(f"teacher-forced, {n} positions: top-1 agreement with plain HF fp32 {agree}/{n}, mean corr {mcorr:.4f} "
              f"(the torch W8A8 model: {e_agree}/{len(fp)}, {e_mcorr:.4f}); device vs torch model: argmax differs at "
              f"{dd}, mean corr {dcorr:.4f} (tolerance {tol})")
        ok = n == len(toks) and agree >= e_agree - tol and mcorr >= e_mcorr - 0.01 and dcorr >= 0.98
    if args.prefill:
        # prefill + decode vs decode only: the prompt's last position, then the generated steps
        M = args.prefill
        pp = prompt if len(prompt) >= 2 * M else [int(v) for v in np.resize(np.array(prompt), 2 * M + M // 2)]
        def run(pre, generate=args.generate, extra=()):
            srv = subprocess.Popen([T.PY, os.path.join(COMPILER, "sim", "sa_sim_server.py"), "--d", str(D), "--mb",
                                    str(args.mb), "--socket", sock, "--once"], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
            time.sleep(3)
            lp = os.path.join(args.out, f"logits_p{pre}.f32")
            r2 = subprocess.run([T.RUN, "--device=sa", f"--module={args.out}/sa.vmfb",
                                 f"--parameters=model={args.out}/sa_packed.irpa", *ABI,
                                 "--tokens=" + ",".join(map(str, pp)), f"--generate={generate}",
                                 f"--logits_out={lp}"] + ([f"--prefill={pre}"] if pre else []) + list(extra),
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
        if args.prefill_all:
            # every prompt position's logits (prefill_all) vs decode only's rows
            ap_ = os.path.join(args.out, "logits_all.f32")
            _, al, _ = run(M, 0, [f"--prefill_all_out={ap_}"])
            rows = np.fromfile(ap_, np.float32).reshape(-1, ref.shape[1])
            ex = [rows[i].tobytes() == dl[i].tobytes() for i in range(min(len(rows), len(pp)))]
            print(f"prefill_all: {len(rows)} prompt positions, bit-exact with decode only in {sum(ex)}/{len(ex)}; "
                  f"the last one's row as prefill's: {al[0].tobytes() == pl[0].tobytes()}")
            ok &= len(rows) == len(pp) and all(ex) and al[0].tobytes() == pl[0].tobytes()
        if args.board_bundle and ok:
            stage(args, pp, run)
    print("hf_generic PASS" if ok else "hf_generic FAIL")
    return 0 if ok else 1


def stage(args, prompt, run):
    """The board bundle (board_llm.py): module, packed parameters, the sim's
    decode-only run over the prompt and --board-generate tokens (the board runs
    prefill + decode: compared with the rows from the prompt's last position),
    the tokenizer, sa_args.txt (--abi=hf[v] [--pad=D] --prefill=M), board.txt (the
    reference, the window in MB)."""
    import shutil
    dst = args.board_bundle
    os.makedirs(dst, exist_ok=True)
    for f in ("sa.vmfb", "sa_packed.irpa"):
        shutil.copy(os.path.join(args.out, f), os.path.join(dst, f))
    line, logits, _ = run(0, args.board_generate)
    tokens = [int(t) for t in line.split(":")[1].split()]
    np.save(os.path.join(dst, "prompt.npy"), np.array(prompt, np.int64))
    np.save(os.path.join(dst, "expected_tokens.npy"), np.array(tokens[:len(logits)], np.int64))
    np.save(os.path.join(dst, "expected_logits.npy"), logits.astype(np.float32))
    shutil.copy(os.path.join(args.model, "tokenizer.json"), os.path.join(dst, "tokenizer.json"))
    shutil.copy(os.path.join(COMPILER, "frontend", "hf_tokenizer.py"), os.path.join(dst, "hf_tokenizer.py"))
    with open(os.path.join(dst, "build_info.txt"), "w") as f:     # (stage_version's "model" lines: WP0)
        git = lambda *a: subprocess.run(["git", "-C", COMPILER, *a], capture_output=True, text=True)   # noqa: E731
        dirty = git("diff", "--quiet", "HEAD", "--", ".").returncode != 0
        f.write(f"commit {git('rev-parse', 'HEAD').stdout.strip()}{' (dirty)' if dirty else ''}\n"
                f"built {time.strftime('%Y-%m-%dT%H:%M:%S')}, SA_D={D} SA_GEMV_PORTS={os.environ.get('SA_GEMV_PORTS', '0')}\n"
                f"command {' '.join(sys.argv[1:])}\n")
    with open(os.path.join(dst, "sa_args.txt"), "w") as f:
        f.write(" ".join(ABI) + f" --prefill={args.prefill}\n")
    # the window: the runtime's peak is the packed parameters + the int8 KV
    # cache + a few MiB (calibrated on the board, compiler/tests/mem_preflight.py);
    # the 4-port overlay's pool is 1280 MiB (K5-M M1)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    cfg = getattr(cfg, "text_config", cfg)
    if args.layers:
        cfg.num_hidden_layers = args.layers
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kv = 2 * cfg.num_hidden_layers * args.max_len * cfg.num_key_value_heads * hd
    mb = math.ceil((os.path.getsize(os.path.join(args.out, "sa_packed.irpa")) + kv) / 2**20) + 16
    if mb > 1280:
        raise SystemExit(f"board bundle: window {mb} MB beyond the 1280 MiB pool")
    with open(os.path.join(dst, "board.txt"), "w") as f:
        f.write(f"the functional simulator (decode only) {mb}\n")
    print(f"board bundle: {dst} ({len(prompt)} prompt tokens, {args.board_generate} generated, window {mb} MB)")


if __name__ == "__main__":
    sys.exit(main())

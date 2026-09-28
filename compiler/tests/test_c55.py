#!/usr/bin/env python3
"""C5.5 (docs/iree_compiler_plan.md §8.7, §8.9): a HuggingFace decoder on the sa device.

The model is exported by compiler/frontend/export_hf.py (QModel of qhf.py:
<out>/qllama.mlir, qllama.irpa, prompt.npy, kv_scales.npy). There is no
hand-written device model for it, so:
  1. compile (compile_sa.sh; --flags for extra iree-compile flags; --skip-compile);
  2. --check: every dispatch, dispatch_check.py (the IR run by the oracle vs
     the functional simulator) at T = 16, 80, 256: bit-exact;
  3. end to end on the sim device: sa-llm-run (prompt, then greedy tokens),
     each step's logits against the eager QModel on the same tokens with the
     device's EXP / RECIP / RSQRT (qhf.device_sfu): the first step equal up to
     reduction order, then the same argmax and a correlation > 0.99 (a sum
     order difference flips an int8 rounding, which grows through the layers),
     and the generated text;
  4. --board-bundle DIR: the whole prompt and --board-generate tokens on the
     sim, staged as the board test's reference (board_llm.py: bit-exact).

    python3 compiler/tests/test_c55.py --model build/llm_cache/SmolLM2-135M --out build/c55/smollm2 \\
        [--check] [--steps 8] [--generate 8] [--mb 144] [--board-bundle build/deploy_c55]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
COMPILER = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(COMPILER, ".."))
sys.path.insert(0, os.path.join(COMPILER, "frontend"))
PY = sys.executable
RUN = os.path.join(REPO, "build", "iree", "build-sa-host", "runtime", "plugins", "hal", "drivers", "sa", "sa-llm-run")


def sh(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd[:3])} ...: exit {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    return r.stdout


def reference(model, out, tokens, d=8):
    """Eager QModel logits (device SFU) on the given tokens (teacher forced)."""
    import export_hf as X
    import qhf
    kv = np.load(os.path.join(out, "kv_scales.npy"))
    cfg, w, _, tok = X.build(model, 256, layers=len(kv))       # export_hf.py --layers: the first layers
    m = qhf.QModel(cfg, w, kv).eval()
    rows = []
    with torch.no_grad(), qhf.device_sfu():
        for pos, t in enumerate(tokens):
            rows.append(m(*X.qhf_inputs(t, pos, d)).numpy())
    return np.stack(rows), tok


def compare(got, ref):
    agree = sum(int(g.argmax() == r.argmax()) for g, r in zip(got, ref))
    corr = min(float(np.corrcoef(g, r)[0, 1]) for g, r in zip(got, ref))
    first = float(np.abs(got[0] - ref[0]).max() / np.abs(ref[0]).max())
    return agree, corr, first


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="the export_hf.py directory")
    ap.add_argument("--flags", default="", help="extra iree-compile flags")
    ap.add_argument("--skip-compile", action="store_true", help="use the compiled sa.vmfb / sa_packed.irpa")
    ap.add_argument("--check", action="store_true", help="dispatch_check.py at T = 16, 80, 256")
    ap.add_argument("--steps", type=int, default=8, help="prompt tokens on the sim")
    ap.add_argument("--generate", type=int, default=8, help="then greedy tokens on the sim")
    ap.add_argument("--mb", type=int, default=144, help="device memory window (MB): SmolLM2-135M peaks at 133.6")
    ap.add_argument("--board-bundle", help="stage the board test here")
    ap.add_argument("--board-generate", type=int, default=24)
    args = ap.parse_args()
    out = args.out
    ok = True
    t0 = time.time()
    if not args.skip_compile:
        compile_sa(out, args.flags)
        print(f"compiled for sa in {time.time() - t0:.1f} s: {len(os.listdir(os.path.join(out, 'sa_sources')))} "
              f"dispatches, sa.vmfb {os.path.getsize(os.path.join(out, 'sa.vmfb')) / 1e6:.1f} MB, "
              f"sa_packed.irpa {os.path.getsize(os.path.join(out, 'sa_packed.irpa')) / 1e6:.1f} MB")
    if args.check:
        for t in (16, 80, 256):
            r = subprocess.run([PY, os.path.join(HERE, "dispatch_check.py"), os.path.join(out, "sa_sources"),
                                os.path.join(out, "sa_bin"), "--t", str(t)], capture_output=True, text=True)
            print(f"per-dispatch differential check, T = {t}: {r.stdout.strip().splitlines()[-1]}")
            if r.returncode:
                print("\n".join(l for l in r.stdout.splitlines() if not l.endswith(" OK")))
                ok = False
    prompt = [int(t) for t in np.load(os.path.join(out, "prompt.npy"))]
    t0 = time.time()
    sim_tokens, got, info = run_sim(out, prompt[:args.steps], args.generate, args.mb)
    ref, tok = reference(args.model, out, sim_tokens[:len(got)])
    agree, corr, first = compare(got, ref)
    good = agree == len(got) and corr > 0.99 and first < 1e-4
    print(f"sim device ({info}), {len(got)} steps in {time.time() - t0:.0f} s: first step max |diff| / max|logit| "
          f"{first:.1e}; argmax {agree}/{len(got)} as eager QModel (device SFU), min correlation {corr:.5f} "
          f"({'OK' if good else 'DIFFERENT'})")
    print(f"  text: {tok.decode(sim_tokens)!r}")
    ok &= good
    if args.board_bundle:
        stage(args, out, prompt, tok)
    print("C5.5 PASS" if ok else "C5.5 FAILED")
    return 0 if ok else 1


def compile_sa(out, flags):
    """compile_sa.sh -> sa.vmfb, sa_packed.irpa, sa_sources/, sa_bin/ (for dispatch_check)."""
    shutil.rmtree(os.path.join(out, "sa_bin"), ignore_errors=True)
    shutil.rmtree(os.path.join(out, "sa_sources"), ignore_errors=True)
    sh([os.path.join(COMPILER, "scripts", "compile_sa.sh"), out,
        f"--iree-hal-dump-executable-binaries-to={os.path.join(out, 'sa_bin')}"],
       env=dict(os.environ, SA_COMPILE_FLAGS=flags))


def run_sim(out, tokens, generate, mb):
    """sa-llm-run on the functional simulator: (all tokens, logits per step, its stats line)."""
    sock = f"/tmp/sa_c55_{os.getpid()}.sock"
    srv = subprocess.Popen([PY, os.path.join(COMPILER, "sim", "sa_sim_server.py"), "--d", "8", "--mb", str(mb),
                            "--socket", sock, "--once"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    logits_path = os.path.join(out, "logits_sim.f32")
    try:
        res = sh([RUN, "--device=sa", f"--module={os.path.join(out, 'sa.vmfb')}",
                  f"--parameters=model={os.path.join(out, 'sa_packed.irpa')}", "--tokens=" + ",".join(map(str, tokens)),
                  f"--generate={generate}", f"--logits_out={logits_path}", "--pad=8"],
                 env=dict(os.environ, SA_SIM_SOCKET=sock))
    finally:
        srv.wait(timeout=60)
    lines = res.strip().splitlines()
    all_tokens = [int(t) for t in lines[0].split(":")[1].split()]
    got = np.fromfile(logits_path, np.float32)
    return all_tokens, got.reshape(len(tokens) + generate, -1), \
        "; ".join(lines[1:])


def stage(args, out, prompt, tok):
    """Board bundle: module, packed parameters, the sim's greedy run over the whole
    prompt (tokens and logits: the board must match bit for bit), the tokenizer,
    board.txt (reference name, window MB)."""
    dst = args.board_bundle
    os.makedirs(dst, exist_ok=True)
    for f in ("sa.vmfb", "sa_packed.irpa", "prompt.npy"):
        shutil.copy(os.path.join(out, f), os.path.join(dst, f))
    t0 = time.time()
    tokens, logits, _ = run_sim(out, prompt, args.board_generate, args.mb)
    # the tokens fed to the steps (board_llm.py runs len - len(prompt) generated steps)
    np.save(os.path.join(dst, "expected_tokens.npy"), np.array(tokens[:len(logits)], np.int64))
    np.save(os.path.join(dst, "expected_logits.npy"), logits.astype(np.float32))
    shutil.copy(os.path.join(args.model, "tokenizer.json"), os.path.join(dst, "tokenizer.json"))
    shutil.copy(os.path.join(COMPILER, "frontend", "hf_tokenizer.py"), os.path.join(dst, "hf_tokenizer.py"))
    mc = json.load(open(os.path.join(args.model, "config.json")))
    eos = mc.get("eos_token_id")
    with open(os.path.join(dst, "model.json"), "w") as f:
        json.dump({"tokenizer": "hf", "bos": None, "stop": eos[0] if isinstance(eos, list) else eos,
                   "context": 256}, f)
    with open(os.path.join(dst, "sa_args.txt"), "w") as f:
        f.write("--pad=8\n")                             # as the sim run: the attention length follows pos
    with open(os.path.join(dst, "board.txt"), "w") as f:
        f.write(f"the functional simulator {args.mb}\n")
    print(f"board bundle: {dst} ({len(tokens)} expected tokens from the sim in {time.time() - t0:.0f} s): "
          f"{tok.decode(tokens)!r}")


if __name__ == "__main__":
    sys.exit(main())

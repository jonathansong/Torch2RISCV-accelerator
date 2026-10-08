#!/usr/bin/env python3
"""C6.P (docs/iree_compiler_plan.md §8.13): prefill + decode on the sa device.

The model is exported with its prefill function (compiler/frontend/export.py
or export_hf.py --prefill M, into --out); then:
  1. compile for sa (compile_sa.sh; --skip-compile to reuse);
  2. --check: every dispatch of both functions against the oracle at
     T = 16, 80, 256 (dispatch_check.py);
  3. the sim device twice with the same prompt: decode only, and prefill (the
     prompt in chunks of M) + decode. The prefill run's logits (the prompt's
     last position, then every generated step) must be bit-exact with the
     decode-only run's rows for the same positions, and the tokens identical.

    python3 compiler/tests/test_c6p.py --out build/c6p/stories --prefill 8 [--check] [--generate 4] [--mb 64]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = int(os.environ.get("SA_D", "8"))   # the array size (the overlay's; compile_sa.sh passes --iree-sa-d)
COMPILER = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(COMPILER, ".."))
PY = sys.executable
RUN = os.path.join(REPO, "build", "iree", "build-sa-host", "runtime", "plugins", "hal", "drivers", "sa", "sa-llm-run")
# stories15M's prompt (llama2.c tokenizer): "Once upon a time, there was a little girl named Lily. She"
STORIES_PROMPT = [1, 9038, 2501, 263, 931, 29892, 727, 471, 263, 2217, 7826, 4257, 365, 2354, 29889, 2296]


def sh(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd[:3])} ...: exit {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    return r.stdout


def run_sim(out, tokens, generate, mb, prefill=0):
    """sa-llm-run on the functional simulator: (tokens, logits rows, stats lines)."""
    sock = f"/tmp/sa_c6p_{os.getpid()}_{prefill}.sock"
    srv = subprocess.Popen([PY, os.path.join(COMPILER, "sim", "sa_sim_server.py"), "--d", str(D), "--mb", str(mb),
                            "--socket", sock, "--once"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    lp = os.path.join(out, f"logits_sim_p{prefill}.f32")
    try:
        res = sh([RUN, "--device=sa", f"--module={os.path.join(out, 'sa.vmfb')}",
                  f"--parameters=model={os.path.join(out, 'sa_packed.irpa')}", "--tokens=" + ",".join(map(str, tokens)),
                  f"--generate={generate}", f"--logits_out={lp}", f"--pad={D}"] +
                 ([f"--prefill={prefill}"] if prefill else []), env=dict(os.environ, SA_SIM_SOCKET=sock))
    finally:
        srv.wait(timeout=60)
    lines = res.strip().splitlines()
    toks = [int(t) for t in lines[0].split(":")[1].split()]
    rows = len(tokens) + generate if not prefill else 1 + generate
    lg = np.fromfile(lp, np.float32)
    return toks, lg.reshape(rows, -1), lines[1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="an export with --prefill M (qllama.mlir, qllama.irpa)")
    ap.add_argument("--prefill", type=int, required=True, help="the exported chunk size M")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--check", action="store_true", help="dispatch_check at T = 16, 80, 256")
    ap.add_argument("--generate", type=int, default=4)
    ap.add_argument("--mb", type=int, default=64)
    ap.add_argument("--flags", default="", help="extra iree-compile flags")
    ap.add_argument("--model", help="the HuggingFace directory (tokenizer.json, config.json) for the board bundle; "
                    "default stories15M")
    ap.add_argument("--board-bundle", help="stage the board test here")
    ap.add_argument("--board-generate", type=int, default=24)
    args = ap.parse_args()
    out = args.out
    ok = True
    if not args.skip_compile:
        t0 = time.time()
        for sub in ("sa_bin", "sa_sources"):
            shutil.rmtree(os.path.join(out, sub), ignore_errors=True)
        sh([os.path.join(COMPILER, "scripts", "compile_sa.sh"), out,
            f"--iree-hal-dump-executable-binaries-to={os.path.join(out, 'sa_bin')}"],
           env=dict(os.environ, SA_COMPILE_FLAGS=args.flags))
        n = len(os.listdir(os.path.join(out, "sa_sources")))
        print(f"compiled for sa in {time.time() - t0:.1f} s: {n} executables (decode and prefill)")
    if args.check:
        for t in (16, 80, 256):
            r = subprocess.run([PY, os.path.join(HERE, "dispatch_check.py"), os.path.join(out, "sa_sources"),
                                os.path.join(out, "sa_bin"), "--t", str(t)], capture_output=True, text=True)
            print(f"per-dispatch differential check, T = {t}: {r.stdout.strip().splitlines()[-1]}")
            if r.returncode:
                print("\n".join(l for l in r.stdout.splitlines() if not l.endswith(" OK")))
                ok = False
    pn = os.path.join(out, "prompt.npy")
    prompt = [int(t) for t in np.load(pn)] if os.path.exists(pn) else STORIES_PROMPT
    if len(prompt) < 2 * args.prefill:            # at least two chunks and an overlapping last one
        prompt = [int(t) for t in np.resize(np.array(prompt), 2 * args.prefill + args.prefill // 2)]
    t0 = time.time()
    dt, dl, dstat = run_sim(out, prompt, args.generate, args.mb)
    t1 = time.time()
    pt, pl, pstat = run_sim(out, prompt, args.generate, args.mb, prefill=args.prefill)
    t2 = time.time()
    P = len(prompt)
    ref = dl[P - 1:]                      # decode only: the prompt's last position, then the generated steps
    exact = [pl[i].tobytes() == ref[i].tobytes() for i in range(len(pl))]
    same = pt == dt
    good = all(exact) and same
    print(f"sim, a {P}-token prompt + {args.generate} generated: decode only {t1 - t0:.0f} s, prefill + decode "
          f"{t2 - t1:.0f} s; logits bit-exact in {sum(exact)}/{len(exact)} rows (the prompt's last position, then "
          f"the generated steps); tokens {'identical' if same else 'DIFFERENT'} ({'OK' if good else 'FAILED'})")
    for l in pstat:
        if l.startswith("prefill"):
            print(f"  {l}")
    ok &= good
    if args.board_bundle:
        stage(args, out, prompt)
    print("C6.P PASS" if ok else "C6.P FAILED")
    return 0 if ok else 1


def stage(args, out, prompt):
    """Board bundle: module, packed parameters, the sim's decode-only run over the
    prompt and --board-generate tokens (board_llm.py compares the prefill run
    with its rows from the prompt's last position on), the tokenizer, model.json,
    sa_args.txt (--pad=D --prefill=M), board.txt (reference, window MB)."""
    import json
    dst = args.board_bundle
    os.makedirs(dst, exist_ok=True)
    for f in ("sa.vmfb", "sa_packed.irpa"):
        shutil.copy(os.path.join(out, f), os.path.join(dst, f))
    t0 = time.time()
    tokens, logits, _ = run_sim(out, prompt, args.board_generate, args.mb)
    np.save(os.path.join(dst, "prompt.npy"), np.array(prompt, np.int64))
    np.save(os.path.join(dst, "expected_tokens.npy"), np.array(tokens[:len(logits)], np.int64))
    np.save(os.path.join(dst, "expected_logits.npy"), logits.astype(np.float32))
    if args.model:                                        # a HuggingFace decoder
        shutil.copy(os.path.join(args.model, "tokenizer.json"), os.path.join(dst, "tokenizer.json"))
        shutil.copy(os.path.join(COMPILER, "frontend", "hf_tokenizer.py"), os.path.join(dst, "hf_tokenizer.py"))
        eos = json.load(open(os.path.join(args.model, "config.json"))).get("eos_token_id")
        mj = {"tokenizer": "hf", "bos": None, "stop": eos[0] if isinstance(eos, list) else eos, "context": 256}
    else:                                                 # stories15M (llama2.c)
        shutil.copy(os.path.join(REPO, "build", "llm_cache", "tokenizer.bin"), os.path.join(dst, "tokenizer.bin"))
        shutil.copy(os.path.join(REPO, "llm", "tokenizer.py"), os.path.join(dst, "tokenizer.py"))
        mj = {"tokenizer": "llama2", "vocab": int(logits.shape[1]), "bos": 1, "stop": 1, "context": 256}
    mj["prefill"] = args.prefill
    json.dump(mj, open(os.path.join(dst, "model.json"), "w"))
    with open(os.path.join(dst, "sa_args.txt"), "w") as f:
        f.write(f"--pad={D} --prefill={args.prefill}\n")
    with open(os.path.join(dst, "board.txt"), "w") as f:
        f.write(f"the functional simulator {args.mb}\n")
    print(f"board bundle: {dst} ({len(tokens)} expected tokens from the sim in {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())

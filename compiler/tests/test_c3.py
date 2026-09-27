#!/usr/bin/env python3
"""C3 (docs/iree_compiler_plan.md §6.8): the whole stories15M compiled by IREE
for the sa device, bit-exact with DeviceModel(SfuExact) (= the hand-written L5
path).

  1. export: compiler/frontend/export.py --static-len (attention over seq_len
     positions masked by pos; checks the llvm-cpu build against eager / DeviceModel);
  2. compile: compiler/scripts/compile_sa.sh (weights packed at compile time,
     every dispatch through the sa code generator; executables dumped);
  3. per dispatch: compiler/tests/dispatch_check.py (the dispatch's IR run by the
     oracle vs its template in the functional simulator, every call site);
  4. end to end: sa-llm-run on the sim device (sa_sim_server.py), the logits of
     every step bit-exact with DeviceModel(SfuExact); --board-bundle also stages
     the board test (board_c3.py) with the expected tokens and logits.

    python3 compiler/tests/test_c3.py [--steps 12] [--generate 20] [--board-bundle DIR]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMPILER = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(COMPILER, ".."))
sys.path.insert(0, os.path.join(REPO, "llm"))
PY = sys.executable
RUN = os.path.join(REPO, "build", "iree", "build-sa-host", "runtime", "plugins", "hal", "drivers", "sa", "sa-llm-run")
PROMPT = "Once upon a time, there was a little girl named Lily. She"


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd[:3])} ...: exit {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(REPO, "build", "c3", "stories15M"))
    ap.add_argument("--steps", type=int, default=12, help="prompt tokens fed on the sim")
    ap.add_argument("--generate", type=int, default=0, help="then greedy tokens on the sim")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--static-len", action="store_true",
                    help="attention over seq_len positions (the C3 form) instead of a dynamic length")
    ap.add_argument("--board-bundle", help="stage the board test here")
    ap.add_argument("--board-generate", type=int, default=60, help="greedy tokens in the board test")
    args = ap.parse_args()
    out = args.out
    ok = True
    if not args.skip_export:
        print(sh([PY, os.path.join(COMPILER, "frontend", "export.py"), "--out", out]
                 + (["--static-len"] if args.static_len else [])).strip().splitlines()[-2])
    run_args = [] if args.static_len else ["--pad=8"]
    t0 = time.time()
    shutil.rmtree(os.path.join(out, "sa_bin"), ignore_errors=True)
    shutil.rmtree(os.path.join(out, "sa_sources"), ignore_errors=True)
    sh([os.path.join(COMPILER, "scripts", "compile_sa.sh"), out,
        f"--iree-hal-dump-executable-binaries-to={os.path.join(out, 'sa_bin')}"])
    print(f"compiled for sa in {time.time() - t0:.1f} s: {len(os.listdir(os.path.join(out, 'sa_sources')))} "
          f"dispatches, sa.vmfb {os.path.getsize(os.path.join(out, 'sa.vmfb')) / 1e6:.1f} MB, "
          f"sa_packed.irpa {os.path.getsize(os.path.join(out, 'sa_packed.irpa')) / 1e6:.1f} MB")
    # per dispatch
    r = subprocess.run([PY, os.path.join(HERE, "dispatch_check.py"), os.path.join(out, "sa_sources"),
                        os.path.join(out, "sa_bin")], capture_output=True, text=True)
    last = r.stdout.strip().splitlines()[-1]
    print(f"per-dispatch differential check (oracle vs simulator, every call site): {last}")
    if r.returncode:
        print("\n".join(line for line in r.stdout.splitlines() if not line.endswith(" OK")))
        ok = False
    # end to end on the sim device
    import checkpoint
    from ref_model import DeviceModel, SfuExact, generate
    from tokenizer import Tokenizer
    cfg, w = checkpoint.load(os.path.join(REPO, "build", "llm_cache", "stories15M.bin"))
    kv = np.load(os.path.join(REPO, "build", "llm_cache", "kv", "stories15M_kv_p99.99.npy"))
    tok = Tokenizer(os.path.join(REPO, "build", "llm_cache", "tokenizer.bin"), cfg.vocab)
    prompt = tok.encode(PROMPT)
    toks = prompt[:args.steps]
    sock = f"/tmp/sa_c3_{os.getpid()}.sock"
    srv = subprocess.Popen([PY, os.path.join(COMPILER, "sim", "sa_sim_server.py"), "--d", "8", "--socket", sock,
                            "--once"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    logits_path = os.path.join(out, "logits_sim.f32")
    res = sh([RUN, "--device=sa", f"--module={os.path.join(out, 'sa.vmfb')}",
              f"--parameters=model={os.path.join(out, 'sa_packed.irpa')}", "--tokens=" + ",".join(map(str, toks)),
              f"--generate={args.generate}", f"--logits_out={logits_path}"] + run_args,
             env=dict(os.environ, SA_SIM_SOCKET=sock))
    srv.wait()
    sim_tokens = [int(t) for t in res.splitlines()[0].split(":")[1].split()]
    got = np.fromfile(logits_path, np.float32).reshape(-1, cfg.vocab)
    dm = DeviceModel(cfg, w, kv, d=8, sfu=SfuExact)
    exact = 0
    for pos in range(len(got)):
        ref = dm.forward(sim_tokens[pos], pos).astype(np.float32)
        exact += got[pos].tobytes() == ref.tobytes()
    print(f"sim device, {len(got)} steps ({res.splitlines()[1] if len(res.splitlines()) > 1 else ''}): "
          f"logits bit-exact with DeviceModel(SfuExact) in {exact}/{len(got)} steps")
    if args.generate:
        print(f"  text: {tok.decode(sim_tokens)!r}")
    ok &= exact == len(got)
    if args.board_bundle:
        stage(args.board_bundle, out, prompt, dm, cfg, w, kv, tok, args.board_generate)
        with open(os.path.join(args.board_bundle, "sa_args.txt"), "w") as f:
            f.write(" ".join(run_args))
    print("C3 PASS" if ok else "C3 FAILED")
    return 0 if ok else 1


def stage(dst, out, prompt, dm, cfg, w, kv, tok, n_gen):
    """Board bundle: the module, the packed parameters, the expected greedy run
    (tokens and logits from DeviceModel(SfuExact)) and the tokenizer."""
    from ref_model import DeviceModel, SfuExact, generate
    os.makedirs(dst, exist_ok=True)
    for f in ("sa.vmfb", "sa_packed.irpa"):
        shutil.copy(os.path.join(out, f), os.path.join(dst, f))
    ref = DeviceModel(cfg, w, kv, d=8, sfu=SfuExact)
    tokens, logits = generate(ref, prompt, len(prompt) + n_gen, keep_logits=True)
    np.save(os.path.join(dst, "expected_tokens.npy"), np.array(tokens, np.int64))
    np.save(os.path.join(dst, "expected_logits.npy"), np.stack(logits).astype(np.float32))
    np.save(os.path.join(dst, "prompt.npy"), np.array(prompt, np.int64))
    shutil.copy(os.path.join(REPO, "build", "llm_cache", "tokenizer.bin"), os.path.join(dst, "tokenizer.bin"))
    shutil.copy(os.path.join(REPO, "llm", "tokenizer.py"), os.path.join(dst, "tokenizer.py"))
    print(f"board bundle: {dst} ({len(tokens)} expected tokens)")


if __name__ == "__main__":
    sys.exit(main())

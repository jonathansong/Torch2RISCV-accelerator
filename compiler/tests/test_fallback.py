#!/usr/bin/env python3
"""Host fallback (docs/iree_compiler_plan.md §8.15): dispatches the sa backend
cannot compile run on the ARM host (VMVX), the rest on the accelerator.

  1. a small model: tanh (no sa lowering: the host) then exp (the accelerator)
     on the sim; the result against numpy (the device's exp is the SFU's), and
     the split (SA_STATS): 1 descriptor list, 1 host dispatch;
  2. stories15M with every decode linear layer forced to the host
     (--iree-sa-host-dispatches=matvec): a 16-token prompt + 4 generated
     tokens, logits bit-exact with the accelerator-only build (the linear
     layers are integer products and fp32 multiplies: the same bits on both).

    python3 compiler/tests/test_fallback.py [--skip-model]
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
sys.path.insert(0, HERE)
import test_c6p as T  # noqa: E402

IREE_BUILD = os.environ.get("IREE_BUILD", os.path.join(REPO, "build", "iree", "build-compiler"))
COMPILE = os.path.join(IREE_BUILD, "tools", "iree-compile")
RUN_MODULE = os.path.join(REPO, "build", "iree", "build-sa-host", "tools", "iree-run-module")
OUT = os.path.join(REPO, "build", "fallback")
FALLBACK = ["--iree-sa-host-fallback", "--iree-hal-link-executables=false"]


def sim(cmd):
    """cmd on the sa device (sim): (stdout, the SA_STATS line)"""
    sock = f"/tmp/sa_fb_{os.getpid()}.sock"
    srv = subprocess.Popen([T.PY, os.path.join(COMPILER, "sim", "sa_sim_server.py"), "--d", "8", "--mb", "64",
                            "--socket", sock, "--once"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, SA_SIM_SOCKET=sock, SA_STATS="1"))
    finally:
        srv.wait(timeout=60)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    stats = [l for l in r.stderr.splitlines() if l.startswith("sa:")]
    return r.stdout, stats[-1] if stats else ""


def toy():
    os.makedirs(OUT, exist_ok=True)
    vmfb = os.path.join(OUT, "toy.vmfb")
    T.sh([COMPILE, os.path.join(HERE, "fallback_toy.mlir"), "--iree-hal-target-device=sa", *FALLBACK, "-o", vmfb])
    x = np.linspace(-3, 3, 64).astype(np.float32)
    out, stats = sim([RUN_MODULE, "--device=sa", f"--module={vmfb}", "--function=main",
                      "--input=64xf32=" + " ".join(f"{v:.9g}" for v in x)])
    y = np.array([float(v) for v in out.split("64xf32=")[1].split()], np.float32)
    ref = np.exp(np.tanh(x.astype(np.float64)))
    err = float(np.max(np.abs(y - ref) / ref))
    ok = err < 1e-4 and "1 descriptor lists" in stats and "1 dispatches on the host" in stats
    print(f"tanh (host) -> exp (accelerator): max relative error {err:.2e} vs numpy; {stats} "
          f"({'OK' if ok else 'FAILED'})")
    return ok


def model():
    src = os.path.join(REPO, "build", "c6p", "stories_m8")
    out = os.path.join(OUT, "stories_hostlin")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for f in ("qllama.mlir", "qllama.irpa"):
        shutil.copy(os.path.join(src, f), os.path.join(out, f))
    for d, flags in ((out, "--iree-sa-host-dispatches=matvec"),):
        T.sh([os.path.join(COMPILER, "scripts", "compile_sa.sh"), d],
             env=dict(os.environ, SA_COMPILE_FLAGS=flags, SA_HOST_FALLBACK="1"))
    ref = os.path.join(OUT, "stories_sa")
    shutil.rmtree(ref, ignore_errors=True)
    os.makedirs(ref)
    for f in ("qllama.mlir", "qllama.irpa"):
        shutil.copy(os.path.join(src, f), os.path.join(ref, f))
    T.sh([os.path.join(COMPILER, "scripts", "compile_sa.sh"), ref], env=dict(os.environ, SA_HOST_FALLBACK="0"))
    a = T.run_sim(ref, T.STORIES_PROMPT, 4, 64)
    b = T.run_sim(out, T.STORIES_PROMPT, 4, 64)
    same = [x.tobytes() == y.tobytes() for x, y in zip(a[1], b[1])]
    ok = all(same) and a[0] == b[0]
    print(f"stories15M, decode linear layers on the host: logits bit-exact in {sum(same)}/{len(same)} rows, "
          f"tokens {'identical' if a[0] == b[0] else 'DIFFERENT'} ({'OK' if ok else 'FAILED'})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()
    ok = toy()
    if not args.skip_model:
        ok &= model()
    print("fallback PASS" if ok else "fallback FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

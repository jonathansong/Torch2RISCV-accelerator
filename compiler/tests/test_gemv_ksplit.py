#!/usr/bin/env python3
"""WP6 boundary tests (docs/generality_implementation_plan.md §10.3): decode's
linear layer (y[t, j] = sum_k x[k] W[t, k, j], int8 x and packed weights, the
int32 accumulator stored) at K = 4080 (one GEMV block), 4096, 4112 (a 16 / 32
tail block), 8192 (4080 + 4080 + 32), random values and the worst case
(|sum| = K * 2^14 = 2^27 at K = 8192), on the sim of the 4-port overlay,
against numpy's exact sums; each compiled with the K split on and off
(--iree-sa-gemv-k-split: off, a K beyond 4080 runs on EX). From the sim's
command trace: the GEMV commands (LD mode GEMV) per call, and whether the
layer ran on the GEMV unit.

x is int8 in [-127, 127] (the frontends' to_i8 saturates symmetrically):
the EX path's x goes through the VE's int8 conversion, which takes -128 to
-127; the GEMV unit reads the packed bytes as they are (exact for -128 too).
So the worst case is -128 x -128 on GEMV, -127 x -128 on EX.

    python3 compiler/tests/test_gemv_ksplit.py
"""
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMPILER = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(COMPILER, ".."))
sys.path.insert(0, HERE)
import perf_model as PM  # noqa: E402
import test_c6p as T  # noqa: E402

IREE_BUILD = os.environ.get("IREE_BUILD", os.path.join(REPO, "build", "iree", "build-compiler"))
COMPILE = os.path.join(IREE_BUILD, "tools", "iree-compile")
RUN_MODULE = os.path.join(REPO, "build", "iree", "build-sa-host", "tools", "iree-run-module")
OUT = os.path.join(REPO, "build", "gemv_ksplit")
D, NT = 16, 8                                     # the array size; output tiles (N = 128)
BLOCK = 4080

MLIR = """func.func @main(%x: tensor<{K}xi8>, %w: tensor<{NT}x{K}x16xi8>) -> tensor<{NT}x16xi32> {{
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<{NT}x16xi32>
  %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<{NT}x16xi32>) -> tensor<{NT}x16xi32>
  %y = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d2, d1)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel", "reduction"]}}
      ins(%x, %w : tensor<{K}xi8>, tensor<{NT}x{K}x16xi8>) outs(%z : tensor<{NT}x16xi32>) {{
  ^bb0(%a: i8, %b: i8, %o: i32):
    %a32 = arith.extsi %a : i8 to i32
    %b32 = arith.extsi %b : i8 to i32
    %p = arith.muli %a32, %b32 : i32
    %s = arith.addi %o, %p : i32
    linalg.yield %s : i32
  }} -> tensor<{NT}x16xi32>
  return %y : tensor<{NT}x16xi32>
}}
"""


def compile_k(K, split):
    d = os.path.join(OUT, f"k{K}_{'on' if split else 'off'}")
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, "mv.mlir")
    with open(src, "w") as f:
        f.write(MLIR.format(K=K, NT=NT))
    vmfb = os.path.join(d, "mv.vmfb")
    r = subprocess.run([COMPILE, src, "--iree-hal-target-device=sa", f"--iree-sa-d={D}", "--iree-sa-gemv-ports=4",
                        f"--iree-sa-gemv-k-split={'true' if split else 'false'}", "--iree-sa-codegen-report",
                        "-o", vmfb], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    return d, vmfb, [l for l in r.stderr.splitlines() if l.startswith("sa codegen")]


def run(d, vmfb, x, w):
    np.save(os.path.join(d, "x.npy"), x)
    np.save(os.path.join(d, "w.npy"), w)
    out, trace = os.path.join(d, "y.npy"), os.path.join(d, "mv.trace")
    sock = f"/tmp/sa_ks_{os.getpid()}.sock"
    srv = subprocess.Popen([T.PY, os.path.join(COMPILER, "sim", "sa_sim_server.py"), "--d", str(D), "--mb", "16",
                            "--socket", sock, "--once", "--trace", trace], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    time.sleep(3)
    try:
        r = subprocess.run([RUN_MODULE, "--device=sa", f"--module={vmfb}", "--function=main",
                            f"--input=@{d}/x.npy", f"--input=@{d}/w.npy", f"--output=@{out}"],
                           capture_output=True, text=True, env=dict(os.environ, SA_SIM_SOCKET=sock, SA_STATS="1"))
    finally:
        srv.wait(timeout=120)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    gemv = sum(c.gemv for _, cmds in PM.decode_trace(trace, PM.Arch(gemv_ports=4)) for c in cmds)
    return np.load(out), gemv


def main():
    rng = np.random.default_rng(0)
    cases = [(K, "random") for K in (4080, 4096, 4112, 8192)] + [(8192, "worst")]
    ok = True
    for K, kind in cases:
        if kind == "random":
            x = rng.integers(-127, 128, K, dtype=np.int8)
            w = rng.integers(-128, 128, (NT, K, D), dtype=np.int8)
        else:
            w = np.full((NT, K, D), -128, np.int8)
        for split in (True, False):
            if kind != "random":                  # (-128 x -128 on GEMV; x -127 on EX, see above)
                x = np.full(K, -128 if K <= BLOCK or split else -127, np.int8)
            ref = np.einsum("k,tkj->tj", x.astype(np.int64), w.astype(np.int64))
            d, vmfb, rep = compile_k(K, split)
            y, gemv = run(d, vmfb, x, w)
            exact = y.dtype == np.int32 and np.array_equal(y.astype(np.int64), ref)
            blocks = -(-K // BLOCK)
            # the GEMV unit: K <= 4080 always; beyond only with the split, in ceil(K / 4080) blocks per strip group
            want_gemv = K <= BLOCK or split
            good = exact and (gemv > 0) == want_gemv and (not want_gemv or gemv % blocks == 0)
            ok &= good
            print(f"K = {K:5d} {kind:>6}{f' x {int(x[0])}' if kind != 'random' else ''}, split {'on ' if split else 'off'}: {'bit-exact' if exact else 'MISMATCH'} "
                  f"(max |y| {int(np.abs(ref).max())}), {gemv} GEMV commands"
                  f"{f' ({gemv // blocks} x {blocks} K blocks)' if gemv and blocks > 1 else ''}"
                  f"{'' if gemv else ' (EX)'}; {'; '.join(r.split(': ', 2)[-1] for r in rep)}"
                  f"{'' if good else '  <-- FAIL'}", flush=True)
    print("gemv_ksplit PASS" if ok else "gemv_ksplit FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

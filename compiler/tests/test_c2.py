#!/usr/bin/env python3
"""C2 (docs/iree_compiler_plan.md §10): the first vertical slice, a torch int8
linear layer through iree-compile with the sa plugin to the accelerator.

For each case (x i8 or i32; stories15M-like shapes, and a layer long enough
for the LOOP_END form):
  1. export: a torch module y = dequant(x_q W_q^T) (qllama.qlinear, weights
     from export_w8a8.quantize_rows) with iree-turbine, parameters externalized;
  2. compile: iree-compile --iree-hal-target-device=sa, parameters imported and
     the packed weights exported to a new archive (sa-pack-linear-weights +
     const-eval), executables dumped;
  3. the executable holds one export with the schedule of
     plugins/sa/templates/reference.qlinear: the same kinds and numbers of
     commands (LD, ST, EX, VE, SETREG, LOOP_END; since C5 the code generator is
     the C5 pipeline, not a byte-for-byte port of the reference);
  4. the exported archive holds pack_b(W_q) byte for byte;
  5. run: iree-run-module --device=sa (sim transport: a simulator service is
     started) -> y bit-exact with DeviceModel.linear (SfuExact);
     --board-bundle DIR also stages everything for the board (run_c2.sh).

    python3 compiler/tests/test_c2.py [--d 8] [--keep DIR] [--board-bundle DIR]
"""
import argparse
import collections
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMPILER = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(COMPILER, ".."))
for p in ("llm", "driver"):
    sys.path.insert(0, os.path.join(REPO, p))
sys.path.insert(0, os.path.join(COMPILER, "plugins", "sa", "templates"))
sys.path.insert(0, os.path.join(COMPILER, "runtime", "tools"))
import reference  # noqa: E402
import sadesc  # noqa: E402
from export_w8a8 import pack_b  # noqa: E402
from ref_model import DeviceModel, SfuExact, quantize_rows  # noqa: E402

BUILD = os.path.join(REPO, "build", "iree")
IREE_COMPILE = os.path.join(BUILD, "build-compiler", "tools", "iree-compile")
IREE_RUN = os.path.join(BUILD, "build-sa-host", "tools", "iree-run-module")
CASES = [  # name, k, n, x i32
    ("qkv_i8", 288, 864, False),
    ("qkv_i32", 288, 864, True),
    ("w2_i32", 768, 288, True),
    ("long_i8", 288, 2048, False),
]


def export_case(out, k, n, x_i32, seed):
    import torch
    import iree.turbine.aot as aot
    sys.path.insert(0, os.path.join(COMPILER, "frontend"))
    from qllama import qlinear

    rng = np.random.default_rng(seed)
    wq, sw = quantize_rows((rng.standard_normal((n, k)) * 0.05).astype(np.float32))

    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(wq, dtype=torch.int8), requires_grad=False)
            self.s_w = torch.nn.Parameter(torch.tensor(sw, dtype=torch.float32), requires_grad=False)

        def forward(self, xq, s_x):
            return qlinear(xq, s_x, self.w, self.s_w)

    m = Linear()
    aot.externalize_module_parameters(m, external_scope="model")
    xdt = torch.int32 if x_i32 else torch.int8
    e = aot.export(m, args=(torch.zeros(k, dtype=xdt), torch.ones((), dtype=torch.float32)))
    e.save_mlir(os.path.join(out, "model.mlir"))
    aot.save_module_parameters(os.path.join(out, "model.irpa"), m)
    xq = rng.integers(-127, 128, k).astype(np.int32 if x_i32 else np.int8)
    s_x = np.array(0.0173, np.float32)
    np.save(os.path.join(out, "x.npy"), xq)
    np.save(os.path.join(out, "s_x.npy"), s_x)
    return wq, sw, xq, s_x


def compile_case(out, d):
    dump = os.path.join(out, "binaries")
    os.makedirs(dump, exist_ok=True)
    cmd = [IREE_COMPILE, os.path.join(out, "model.mlir"), "--iree-hal-target-device=sa", f"--iree-sa-d={d}",
           f"--iree-parameter-import=model={os.path.join(out, 'model.irpa')}",
           "--iree-parameter-import-maximum-size=4294967295",
           f"--iree-parameter-export=model={os.path.join(out, 'packed.irpa')}",
           "--iree-parameter-export-minimum-size=256",
           f"--iree-hal-dump-executable-binaries-to={dump}",
           "-o", os.path.join(out, "model.vmfb")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"iree-compile failed:\n{r.stderr[-4000:]}")
    return [os.path.join(dump, f) for f in sorted(os.listdir(dump)) if f.endswith(".sadesc")]


def read_params(path):
    import iree.runtime as rt
    idx = rt.ParameterIndex()
    idx.load(path)
    return {k: bytes(v.file_view) for k, v in idx.items()}


class SimServer:
    def __init__(self, d):
        self.sock = f"/tmp/sa_c2_{os.getpid()}.sock"
        self.p = subprocess.Popen([sys.executable, os.path.join(COMPILER, "sim", "sa_sim_server.py"), "--d", str(d),
                                   "--socket", self.sock], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            if os.path.exists(self.sock):
                break
            time.sleep(0.1)

    def close(self):
        self.p.terminate()
        self.p.wait()


def run_case(out, sim):
    y = os.path.join(out, "y.npy")
    env = dict(os.environ, SA_TRANSPORT="sim", SA_SIM_SOCKET=sim.sock)
    cmd = [IREE_RUN, "--device=sa", f"--module={os.path.join(out, 'model.vmfb')}",
           f"--parameters=model={os.path.join(out, 'packed.irpa')}", "--function=main",
           f"--input=@{os.path.join(out, 'x.npy')}", f"--input=@{os.path.join(out, 's_x.npy')}", f"--output=@{y}"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(f"iree-run-module failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return np.load(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--keep", help="work directory to keep (default: a temporary one)")
    ap.add_argument("--board-bundle", help="also stage the compiled cases for the board here")
    args = ap.parse_args()
    d = args.d
    work = args.keep or tempfile.mkdtemp(prefix="sa_c2_")
    sim = SimServer(d)
    fails = 0
    try:
        for i, (name, k, n, x_i32) in enumerate(CASES):
            out = os.path.join(work, name)
            os.makedirs(out, exist_ok=True)
            wq, sw, xq, s_x = export_case(out, k, n, x_i32, seed=100 + i)
            sadescs = compile_case(out, d)
            ok = True
            # 3. the template
            if len(sadescs) != 1:
                print(f"{name}: {len(sadescs)} executables (expected 1)")
                ok = False
            else:
                blob = open(sadescs[0], "rb").read()
                dd, _, exps = sadesc.read(blob)
                got = exps[0][1] if len(exps) == 1 else np.zeros((0, 8), np.uint64)
                prefix = sadesc.read_ext(blob)[0]["prefix"] if len(exps) == 1 else 0
                kinds = lambda rows: collections.Counter(int(r[0]) & 0xFF for r in rows if int(r[0]) & 0xFF != sadesc.OP_RET)
                ref = reference.qlinear(d, k, n, x_i32, 0, 1, 2, 3, 4).array()
                same = kinds(got) == kinds(ref)
                print(f"{name}: k={k} n={n} x={'i32' if x_i32 else 'i8'}: executable D={dd}, "
                      f"{len(exps)} export(s) '{exps[0][0] if exps else ''}', {len(got)} descriptors ({prefix} in the prefix); "
                      + ("the reference schedule (same commands)" if same else
                         f"commands DIFFERENT from the reference: {dict(kinds(got))} vs {dict(kinds(ref))}"))
                ok &= same and dd == d
            # 4. the packed weights
            packed = read_params(os.path.join(out, "packed.irpa"))
            want = pack_b(wq, d).tobytes()
            hits = [key for key, v in packed.items() if v == want]
            print(f"   packed.irpa: {sorted(packed)} -> pack_b(W) {'found: ' + hits[0] if hits else 'NOT FOUND'}")
            ok &= bool(hits)
            # 5. run on the sim device
            y = run_case(out, sim)
            dm = DeviceModel.__new__(DeviceModel)
            dm.d, dm.sfu = d, SfuExact
            ref = dm.linear(xq.astype(np.int8), s_x, (wq.astype(np.float64), sw)).astype(np.float32)
            same = y.shape == ref.shape and y.tobytes() == ref.tobytes()
            print(f"   iree-run-module --device=sa (sim): y {y.shape} "
                  f"{'bit-exact with DeviceModel.linear' if same else 'DIFFERENT, max diff %g' % np.abs(y - ref).max()}")
            ok &= same
            np.save(os.path.join(out, "y_ref.npy"), ref)
            fails += not ok
    finally:
        sim.close()
    if args.board_bundle:
        stage(work, args.board_bundle)
    print("C2 PASS" if fails == 0 else f"C2 FAILED ({fails} case(s))")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if fails else 0


def stage(work, dst):
    """Board bundle: per case model.vmfb, packed.irpa, x / s_x / y_ref .npy, plus
    run_c2.sh (armv7 iree-run-module through the C1 launcher)."""
    os.makedirs(dst, exist_ok=True)
    for name, *_ in CASES:
        os.makedirs(os.path.join(dst, name), exist_ok=True)
        for f in ("model.vmfb", "packed.irpa", "x.npy", "s_x.npy", "y_ref.npy"):
            shutil.copy(os.path.join(work, name, f), os.path.join(dst, name, f))
    print(f"board bundle: {dst}")


if __name__ == "__main__":
    sys.exit(main())

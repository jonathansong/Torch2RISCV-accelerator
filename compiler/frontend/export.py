#!/usr/bin/env python3
"""Stage C0: export the quantized llama with iree-turbine, compile it, run it, list its dispatches.

Steps (docs/iree_compiler_plan.md §3):
1. QLlama (compiler/frontend/qllama.py) of stories15M (or the random tiny model
   of llm/test_l4.py) -> iree-turbine aot.export: decode(token, pos, valid[T])
   with T dynamic, the KV cache as mutable globals, the weights externalized
   to a parameter archive (.irpa, scope "model");
2. iree-compile (the pip compiler of L0, llvm-cpu for the host), dumping every
   dispatch's source. Constant-expression hoisting is off: with it, IREE
   converts the int8 weights to transposed int32 copies at load time (4x the
   memory, and the matmul would see i32 x i32); without it every linear
   dispatch loads int8 weights and sign-extends them inside (the form the sa
   templates match, docs/iree_compiler_plan.md §6.2);
3. run --tokens decode steps with the IREE runtime (the KV state lives in the
   module's globals) and compare with eager QLlama and with
   DeviceModel(sfu=Sfu64) (the same arithmetic in float64-rounded functions);
4. the dispatch inventory: every dispatch's name, linalg operations, element
   types, shapes and the hand-written piece it corresponds to
   (compile_layer / compile_model), written to <out>/dispatches.md.

    python3 compiler/frontend/export.py --out build/c0/stories15M            # stories15M
    python3 compiler/frontend/export.py --tiny --out build/c0/tiny           # random tiny model
    python3 compiler/frontend/export.py --out build/c0/stories15M --armv7    # + the board bundle

--armv7 also compiles for the board's Cortex-A9 (the llvm-cpu flags, libm
shim and import check of iree-sa/l0) and stages build/deploy_c0: the
module, the parameter archive, the L0 iree-run-module and two single-step
cases (pos 0, and pos 5 with T = 8, each starting from an empty KV cache
since iree-run-module makes one call per process) with the host's outputs
as the expected values.
"""
import argparse
import collections
import os
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "llm"))
import checkpoint  # noqa: E402
import qllama as Q  # noqa: E402
from ref_model import DeviceModel, Sfu64  # noqa: E402

CACHE = os.path.join(REPO, "build", "llm_cache")


def load_model(args):
    if args.tiny:
        from test_l4 import tiny_model
        cfg, w, kv = tiny_model()
        name = "tiny"
    else:
        cfg, w = checkpoint.load(args.checkpoint)
        kv = np.load(args.kv)
        name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    return name, cfg, w, kv


def prefill_inputs(tokens, start, d):
    """(tokens, positions, valid) of a prefill chunk at start (plan §8.13):
    positions = start .. start + M - 1, valid's length the chunk's end rounded up to D."""
    n = start + len(tokens)
    return (torch.tensor(tokens), torch.arange(start, n, dtype=torch.int64),
            torch.zeros((n + d - 1) // d * d, dtype=torch.float32))


def prefill_starts(P, M):
    """Chunk starts of a P-token prompt (P >= M): 0, M, 2M, ... and P - M last
    (it recomputes rows of the one before, so its last row is the prompt's last)."""
    return list(range(0, P - M, M)) + [P - M]


def export(m, cfg, d, out, static_len=None, prefill=0):
    """static_len: attention over a fixed number of positions (valid[static_len],
    masked by pos) instead of a dynamic T; the sa backend's form. prefill = M:
    also prefill(tokens[M], positions[M], valid[T]) (m.prefill), sharing the parameters
    and the KV cache with main (decode)."""
    import iree.turbine.aot as aot
    aot.externalize_module_parameters(m, external_scope="model")
    t0 = time.time()
    if prefill:
        fxb = aot.FxProgramsBuilder(m)
        T = torch.export.Dim("T", min=1, max=cfg.seq_len)
        T2 = torch.export.Dim("T2", min=1, max=cfg.seq_len)

        @fxb.export_program(name="main", args=Q.step_inputs(1, 0, d),
                            dynamic_shapes={"token": None, "pos": None, "valid": {0: T}}, strict=False)
        def _decode(module, token, pos, valid):
            return module(token, pos, valid)

        @fxb.export_program(name="prefill", args=prefill_inputs([1] * prefill, 0, d),
                            dynamic_shapes={"tokens": None, "positions": None, "valid": {0: T2}}, strict=False)
        def _prefill(module, tokens, positions, valid):
            return module.prefill(tokens, positions, valid)

        exp = aot.export(fxb)
    elif static_len:
        exp = aot.export(m, args=Q.step_inputs(1, 0, d, static_len))
    else:
        T = torch.export.Dim("T", min=1, max=cfg.seq_len)
        exp = aot.export(m, args=Q.step_inputs(1, 0, d), dynamic_shapes={"token": None, "pos": None, "valid": {0: T}})
    mlir_path, irpa_path = os.path.join(out, "qllama.mlir"), os.path.join(out, "qllama.irpa")
    with open(mlir_path, "w") as f:
        f.write(str(exp.mlir_module))
    aot.save_module_parameters(irpa_path, m)
    print(f"exported in {time.time() - t0:.1f} s: {mlir_path} ({os.path.getsize(mlir_path) / 1e6:.1f} MB), "
          f"{irpa_path} ({os.path.getsize(irpa_path) / 1e6:.1f} MB)")
    return mlir_path, irpa_path


COMPILE_FLAGS = ["--iree-opt-const-expr-hoisting=false"]


def compile_host(mlir_path, out):
    import iree.compiler as ic
    disp = os.path.join(out, "dispatches")
    os.makedirs(disp, exist_ok=True)
    for f in os.listdir(disp):
        os.remove(os.path.join(disp, f))
    t0 = time.time()
    vmfb = ic.compile_file(mlir_path, target_backends=["llvm-cpu"], input_type="torch",
                           extra_args=["--iree-llvmcpu-target-cpu=host",
                                       f"--iree-hal-dump-executable-sources-to={disp}"] + COMPILE_FLAGS)
    path = os.path.join(out, "qllama_host.vmfb")
    with open(path, "wb") as f:
        f.write(vmfb)
    print(f"compiled (llvm-cpu, host) in {time.time() - t0:.1f} s: {path} ({len(vmfb) / 1e6:.1f} MB)")
    return path, disp


class IreeModel:
    def __init__(self, vmfb_path, irpa_path):
        import iree.runtime as rt
        self.config = rt.Config("local-sync")
        idx = rt.ParameterIndex()
        idx.load(irpa_path)
        params = rt.create_io_parameters_module(self.config.vm_instance, idx.create_provider(scope="model"))
        hal = rt.create_hal_module(self.config.vm_instance, self.config.device)
        main = rt.VmModule.mmap(self.config.vm_instance, vmfb_path)
        self.modules = rt.load_vm_modules(params, hal, main, config=self.config)
        self.main = self.modules[-1]

    def __call__(self, token, pos, valid):
        return self.main.main(token.numpy(), pos.numpy(), valid.numpy()).to_host()

    def prefill(self, tokens, positions, valid):
        return self.main.prefill(tokens.numpy(), positions.numpy(), valid.numpy()).to_host()


def check_prefill(vmfb, irpa, eager, prompt, M, d, step, gen=4):
    """The prompt through prefill chunks and through decode steps, each on a fresh
    module: the same last-position argmax, correlation >= 0.9999, the same greedy
    tokens after it; eager prefill (eager: a fresh model) for reference
    (correlation > 0.98, as decode). step(token, pos, d): decode inputs."""
    P = len(prompt)
    if P < M:
        print(f"prefill: a prompt of {P} tokens is shorter than a chunk ({M})")
        return False
    t0 = time.time()
    dec, pre, e = IreeModel(vmfb, irpa), IreeModel(vmfb, irpa), eager
    with torch.no_grad():
        for pos, t in enumerate(prompt):
            last = dec(*step(t, pos, d))
        for s in prefill_starts(P, M):
            a = prefill_inputs(prompt[s:s + M], s, d)
            got, ref = pre.prefill(*a), e.prefill(*a).numpy()
        seqs = []
        for model, first in ((pre, got), (dec, last)):
            seq = [int(first.argmax())]
            for i in range(gen):
                seq.append(int(model(*step(seq[-1], P + i, d)).argmax()))
            seqs.append(seq)
    corr_d, corr_e = float(np.corrcoef(got, last)[0, 1]), float(np.corrcoef(got, ref)[0, 1])
    exact = got.tobytes() == last.tobytes()
    ok = got.argmax() == last.argmax() and corr_d >= 0.9999 and corr_e > 0.98 and seqs[0] == seqs[1]
    print(f"prefill (M = {M}, chunks at {prefill_starts(P, M)}) in {time.time() - t0:.1f} s: last-position logits vs "
          f"decode only: argmax {got.argmax()} / {last.argmax()}, correlation {corr_d:.6f}"
          f"{' (bit-exact)' if exact else ''}; vs eager prefill {corr_e:.5f}; then {seqs[0]} "
          f"({'=' if seqs[0] == seqs[1] else '!='} decode only)")
    return bool(ok)


def compare(name, cfg, w, kv, d, iree_model, tokens, static_len=None):
    m = Q.QLlama(cfg, w, kv, d).eval()
    dm = DeviceModel(cfg, w, kv, d=d, sfu=Sfu64)
    worst_e = worst_d = 0.0
    agree = 0
    t0 = time.time()
    with torch.no_grad():
        for pos, t in enumerate(tokens):
            args = Q.step_inputs(t, pos, d, static_len)
            got = iree_model(*args)
            e = m(*args).numpy()
            ref = dm.forward(t, pos)
            worst_e = max(worst_e, float(np.abs(got - e).max() / np.abs(e).max()))
            worst_d = max(worst_d, float(np.abs(got - ref).max() / np.abs(ref).max()))
            agree += int(got.argmax() == ref.argmax())
    n = len(tokens)
    print(f"{name}: {n} decode steps (pos 0..{n - 1}) in {time.time() - t0:.1f} s; max |diff| / max|logit|: "
          f"vs eager QLlama {worst_e:.2e}, vs DeviceModel(Sfu64) {worst_d:.2e}; argmax agrees {agree}/{n}")
    return agree == n


# ------------------------------------------------------------ dispatch inventory
PIECES = [  # (predicate on the dispatch summary, hand-written piece)
    (lambda s: "_initializer_" in s["name"], "initializer (runs once at load)"),
    (lambda s: "vecmat" in s["ops"] and s["n_out"] >= 4096, "classifier: int8 GEMV + dequant (compile_layer.linear, looped)"),
    (lambda s: "vecmat" in s["ops"] or "matvec" in s["ops"], "int8 linear: GEMV + dequant (compile_layer.linear)"),
    (lambda s: "batch_matmul" in s["ops"] and re.search(r"x1x\d+xD_", s["name"]), "attention P·V (compile_model.attention)"),
    (lambda s: "batch_matmul" in s["ops"], "attention scores Q·K^T (compile_model.attention)"),
    (lambda s: "scatter" in s["ops"], "KV cache row write (compile_model.kv_append)"),
    (lambda s: "gather" in s["ops"] or s["index"], "gather: embedding / RoPE row (compile_model.embed, rope)"),
    (lambda s: s["reduction"], "reduction: RMSNorm sum, amax, softmax max / sum"),
    (lambda s: True, "element-wise chain (VE expression compiler)"),
]


def summarize(path):
    txt = open(path).read()
    name = re.search(r"func\.func @(\S+)\(", txt)
    name = name.group(1) if name else os.path.basename(path)
    body = txt[txt.find("func.func"):]
    ops = collections.Counter(o.split(".")[-1] for o in
                              re.findall(r"(linalg\.[a-z_]+|iree_linalg_ext\.[a-z_]+|tensor\.(?:extract|insert)_slice)", body)
                              if not o.endswith("yield") and not o.endswith(".index"))
    types = sorted(set(re.findall(r"tensor<[^>]*x(i8|i32|i64|f32)>", txt)))
    shapes = re.findall(r"tensor<(?:readonly:|writeonly:|readwrite:)?tensor<([^>]*)>", txt)
    iters = re.findall(r"iterator_types = \[([^\]]*)\]", txt)
    reduction = any("reduction" in it for it in iters)
    m = re.search(r"(?:vecmat|matvec)_(\d+)x(\d+)", name)
    n_out = int(m.group(1)) if m else 0
    s = {"name": name, "ops": dict(ops), "types": types, "shapes": sorted(set(shapes)), "reduction": reduction,
         "index": "linalg.index" in body, "n_out": n_out}
    s["piece"] = next(p for f, p in PIECES if f(s))
    return s


def inventory(disp, out):
    files = sorted((f for f in os.listdir(disp) if f.endswith(".mlir")),
                   key=lambda f: int(re.search(r"(\d+)\.mlir$", f).group(1)) if re.search(r"(\d+)\.mlir$", f) else 0)
    rows = [summarize(os.path.join(disp, f)) for f in files]
    by_piece = collections.Counter(r["piece"] for r in rows)
    lines = ["# Dispatch inventory (stage C0)", "",
             f"{len(rows)} dispatches (llvm-cpu, host), grouped by the hand-written piece they correspond to:", "",
             "| piece | dispatches |", "|---|---|"]
    lines += [f"| {p} | {n} |" for p, n in by_piece.most_common()]
    lines += ["", "| # | dispatch | linalg ops | types | tensor shapes | piece |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows):
        ops = ", ".join(f"{k}×{v}" if v > 1 else k for k, v in r["ops"].items())
        shapes = "; ".join(r["shapes"][:4]) + (" …" if len(r["shapes"]) > 4 else "")
        lines.append(f"| {i} | `{r['name']}` | {ops} | {' '.join(r['types'])} | {shapes} | {r['piece']} |")
    path = os.path.join(out, "dispatches.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(rows)} dispatches -> {path}")
    for p, n in by_piece.most_common():
        print(f"  {n:4d}  {p}")
    return rows


def board_bundle(mlir_path, irpa_path, host_vmfb, cfg, d, out, deploy):
    import shutil
    import subprocess
    sys.path.insert(0, os.path.join(REPO, "iree-sa", "l0"))
    import export_and_compile as L0
    import iree.compiler as ic
    L0.build_shim()
    # embedded_ld.sh finds IREE's lld next to iree-compile on PATH unless IREE_LLD says where it is
    os.environ.setdefault("IREE_LLD", os.path.join(os.path.dirname(ic.__file__), "_mlir_libs", "iree-lld"))
    dump = os.path.join(out, "armv7_binaries")
    os.makedirs(dump, exist_ok=True)
    t0 = time.time()
    vmfb = ic.compile_file(mlir_path, input_type="torch",
                           extra_args=L0.ARMV7_FLAGS + COMPILE_FLAGS + [f"--iree-hal-dump-executable-binaries-to={dump}"])
    L0.check_no_imports(dump)
    os.makedirs(deploy, exist_ok=True)
    with open(os.path.join(deploy, "qllama_armv7.vmfb"), "wb") as f:
        f.write(vmfb)
    print(f"compiled (llvm-cpu, armv7 Cortex-A9) in {time.time() - t0:.1f} s ({len(vmfb) / 1e6:.1f} MB)")
    shutil.copy(irpa_path, os.path.join(deploy, "qllama.irpa"))
    shutil.copy(os.path.join(REPO, "build", "deploy_iree_l0", "iree-run-module"), deploy)
    lines = ["#!/usr/bin/env bash", "# On the board (no sudo): the C0 module on the ARM (llvm-cpu), two single-step cases.",
             'cd "$(dirname "$0")"', "ok=1"]
    for pos, token in ((0, 1), (5, 400)):
        args = Q.step_inputs(token, pos, d)
        want = IreeModel(host_vmfb, irpa_path)(*args)          # fresh module: empty KV cache
        tag = f"pos{pos}"
        np.save(os.path.join(deploy, f"{tag}_token.npy"), args[0].numpy())
        np.save(os.path.join(deploy, f"{tag}_pos.npy"), args[1].numpy())
        np.save(os.path.join(deploy, f"{tag}_valid.npy"), args[2].numpy())
        np.save(os.path.join(deploy, f"{tag}_expected.npy"), want)
        lines += [f'echo "== pos {pos}, token {token}, T = {args[2].shape[0]}"',
                  "./iree-run-module --device=local-task --module=qllama_armv7.vmfb --parameters=model=qllama.irpa "
                  f"--function=main --input=@{tag}_token.npy --input=@{tag}_pos.npy --input=@{tag}_valid.npy "
                  f"--expected_output=@{tag}_expected.npy --expected_f32_threshold=1e-3 || ok=0"]
    lines += ['[ $ok = 1 ] && echo "C0 board PASS" || echo "C0 board FAIL"']
    with open(os.path.join(deploy, "run_c0.sh"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"board bundle: {deploy} ({subprocess.run(['du', '-sh', deploy], capture_output=True, text=True).stdout.split()[0]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=os.path.join(CACHE, "stories15M.bin"))
    ap.add_argument("--kv", default=os.path.join(CACHE, "kv", "stories15M_kv_p99.99.npy"))
    ap.add_argument("--tiny", action="store_true", help="the random tiny model instead of a checkpoint")
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--static-len", action="store_true",
                    help="attention over seq_len positions masked by pos (static shapes; the sa backend's form)")
    ap.add_argument("--armv7", action="store_true", help="also compile for the board and stage build/deploy_c0")
    ap.add_argument("--prefill", type=int, default=0, help="also export prefill with chunks of this many tokens")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    name, cfg, w, kv = load_model(args)
    m = Q.QLlama(cfg, w, kv, args.d).eval()
    static_len = cfg.seq_len if args.static_len else None
    mlir_path, irpa_path = export(m, cfg, args.d, args.out, static_len, prefill=args.prefill)
    vmfb_path, disp = compile_host(mlir_path, args.out)
    rng = np.random.default_rng(5)
    if args.tiny:
        tokens = [int(t) for t in rng.integers(0, cfg.vocab, min(args.tokens, cfg.seq_len))]
    else:
        from tokenizer import Tokenizer
        tok = Tokenizer(os.path.join(os.path.dirname(args.checkpoint), "tokenizer.bin"), cfg.vocab)
        tokens = tok.encode("Once upon a time, there was a little girl named Lily. She")[:args.tokens]
    ok = compare(name, cfg, w, kv, args.d, IreeModel(vmfb_path, irpa_path), tokens, static_len)
    if args.prefill:
        ok &= check_prefill(vmfb_path, irpa_path, Q.QLlama(cfg, w, kv, args.d).eval(), tokens, args.prefill,
                            args.d, Q.step_inputs)
    inventory(disp, args.out)
    if args.armv7:
        board_bundle(mlir_path, irpa_path, vmfb_path, cfg, args.d, args.out, os.path.join(REPO, "build", "deploy_c0"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""C6.5 (docs/iree_compiler_plan.md §8.22): a KV cache longer than one SPAD_B
bank holds (T x head_dim): decode attention's q . K^T goes over T in blocks in
a hardware loop whose count comes from T.

A truncated model (2 layers) exported at --max-len (the export writes
--iree-sa-max-dynamic into compile_flags.txt), compiled for the four-port
overlay with the executables dumped; every executable must be on the
accelerator, and every dispatch must match the oracle (dispatch_check.py) at
lengths T around the block boundaries (multiples of D: the runtime pads T).

    python3 compiler/tests/test_long_context.py [--model build/llm_cache/gemma-3-270m] [--max-len 512]
        [--out build/hfgen/gemma_l2_t512] [--t 16,240,256,272,304,496,512]
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMPILER = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(COMPILER, ".."))
PY = sys.executable


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="build/llm_cache/gemma-3-270m")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out", default="build/hfgen/gemma_l2_t512")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--t", default="16,240,256,272,304,496,512")
    args = ap.parse_args()
    env = dict(os.environ, SA_KV260_CONFIG="d16_100mhz_w128_p6_gemv_np4", SA_D="16", SA_GEMV_PORTS="4")
    out = os.path.join(REPO, args.out)
    if not os.path.exists(os.path.join(out, "qllama.mlir")):
        sys.path.insert(0, os.path.join(COMPILER, "frontend"))
        import hf_generic as H
        w = H.load(os.path.join(REPO, args.model), args.max_len, quant=True, rope=True, attn=True, layers=args.layers,
                   kv_channel="k", kv_calib=H.kv_calib_arg("build/llm_cache/wikitext-2-raw-v1-train.txt:1024"))
        H.export(w, out, 16)
        del w
    flags = open(os.path.join(out, "compile_flags.txt")).read().strip() if os.path.exists(
        os.path.join(out, "compile_flags.txt")) else ""
    if f"--iree-sa-max-dynamic={args.max_len}" not in flags:
        print(f"FAIL: {out} was not exported at --max-len {args.max_len} (compile_flags.txt: {flags!r})")
        return 1
    bins = os.path.join(out, "sa_bins")
    subprocess.run(["rm", "-rf", bins], check=True)
    r = subprocess.run([os.path.join(COMPILER, "scripts", "compile_sa.sh"), out], capture_output=True, text=True,
                       env=dict(env, SA_COMPILE_FLAGS=f"--iree-sa-codegen-report --iree-hal-dump-executable-binaries-to={bins}"))
    if r.returncode:
        print(r.stderr[-3000:])
        return 1
    ok_n, host = r.stderr.count(": ok\n"), [l for l in r.stderr.splitlines() if ": on the host" in l]
    print(f"compiled at --iree-sa-max-dynamic={args.max_len}: {ok_n} executables for the accelerator, {len(host)} for the host")
    for l in host[:5]:
        print("  " + l)
    fails = len(host) > 0
    for t in args.t.split(","):
        c = subprocess.run([PY, os.path.join(HERE, "dispatch_check.py"), os.path.join(out, "sa_sources"), bins, "--t", t],
                           capture_output=True, text=True)
        last = c.stdout.strip().splitlines()[-1] if c.stdout.strip() else c.stderr[-500:]
        bad = [l for l in c.stdout.splitlines() if l.startswith("#") and not l.rstrip().endswith(" OK")]
        print(f"  per-dispatch differential check, T = {t}: {last}")
        for l in bad[:5]:
            print("    " + l)
        fails |= c.returncode != 0
    print("long context FAIL" if fails else "long context PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

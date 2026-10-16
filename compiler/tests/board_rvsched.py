#!/usr/bin/env python3
"""Plan B of the RISC-V backend on the board (docs/riscv_backend_design.md
§7.1): the decode loop run by the PicoRV32 (rt_fw v3, RT_RUN_SCHED) instead of
the ARM. sa-llm-run --rvsched records the decode step's descriptor lists once
per attention-length bucket and hands the generation to the core, which writes
the inputs, patches the lists at bucket changes, submits them, takes the argmax
(the block maxima of argmax.vmfb, then a scan on the core) and feeds the token
back; the ARM waits for the run to end.

1. The bundle's prompt (board_llm.py's run) with the generation on the core:
   the tokens identical to the sim's (expected_tokens.npy), the last step's
   logits bit-exact with expected_logits.npy.
2. A longer generation (--gen, default 256 tokens after the prompt) on the ARM
   and on the core: the tokens identical, the time per step of both.

Files: the hfgen bundle (board_llm.py's) plus argmax.vmfb, rt_fw.bin version 3
and an sa-llm-run with --rvsched (compiler/scripts/deploy_rvsched.sh).

    python3 board_rvsched.py [--gen 256]     (or python3 board_run.py rvsched)
--rv (plan A inside plan B, docs/riscv_backend_design.md §0, S3): the core's loop
calls each dispatch's generated RISC-V code (SA_RV=rv/, compiler/scripts/
deploy_rvcode.sh) instead of submitting its descriptor list; the ARM's loop of
run 2 stays on descriptor lists (the reference). Writes rvsched_rv_board.json.

Writes rvsched_board.json.
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board_launcher as BL  # noqa: E402


def run(env, extra, tokens, gen, rvsched, logits_path, rv=False):
    """rv: False, or the SA_RV value (an image directory, or "embedded")"""
    cmd = [os.path.join(HERE, "sa-llm-run"), "--device=sa", f"--module={HERE}/argmax.vmfb",
           f"--module={HERE}/sa.vmfb", f"--parameters=model={HERE}/sa_packed.irpa",
           "--tokens=" + ",".join(str(int(t)) for t in tokens), f"--generate={gen}",
           f"--logits_out={logits_path}"] + extra + (["--rvsched"] if rvsched else [])
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env=dict(env, SA_STATS="1", **({"SA_RV": rv} if rv else {})))
    if r.returncode:
        raise SystemExit(f"sa-llm-run{' --rvsched' if rvsched else ''} failed ({r.returncode})\n"
                         f"{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    out = r.stdout
    toks = [int(t) for t in out.strip().splitlines()[0].split(":")[1].split()]
    res = {"tokens": toks, "stdout": out, "stderr": [l for l in r.stderr.splitlines() if l.startswith(("sa:", "rvsched:"))]}
    m = re.search(r"rvsched: (\d+) decode steps on the RISC-V core, ([\d.]+) ms per step .*core ([\d.]+) ms per step, "
                  r"its lists ([\d.]+) ms", out)
    if m:
        res.update(steps=int(m.group(1)), ms_per_step=float(m.group(2)), core_ms=float(m.group(3)),
                   lists_ms=float(m.group(4)))
    m = re.search(r"(\d+) decode steps, ([\d.]+) ms per step", out) or re.search(r"(\d+) steps, ([\d.]+) ms per step", out)
    if m and not rvsched:
        res.update(steps=int(m.group(1)), ms_per_step=float(m.group(2)))
    m = re.search(r"rvsched: .*", r.stderr)
    if m:
        res["table"] = m.group(0)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gen", type=int, default=256, help="tokens generated in the long run (0: skip it)")
    ap.add_argument("--rv", action="store_true", help="the core calls the generated RISC-V code (S3)")
    ap.add_argument("--embedded", action="store_true",
                    help="with --rv: the code iree-compile embedded in sa.vmfb (S4, SA_RV=embedded)")
    args = ap.parse_args()
    prompt = np.load(os.path.join(HERE, "prompt.npy"))
    want_tokens = [int(t) for t in np.load(os.path.join(HERE, "expected_tokens.npy"))]
    want_logits = np.load(os.path.join(HERE, "expected_logits.npy"))
    n_gen = len(want_tokens) - len(prompt)
    mb = int(open(os.path.join(HERE, "board.txt")).read().split()[-1])
    extra = open(os.path.join(HERE, "sa_args.txt")).read().split()
    vlen = max([int(a.split("=")[1]) for a in extra if a.startswith("--valid_len=")] or [256])
    lp = os.path.join(HERE, "logits_rvsched.f32")
    mm, buf, env = BL.start(os.path.join(HERE, "picorv32.bit"), os.path.join(HERE, "rt_fw.bin"), mb=mb, ring=16)
    out = {}
    try:
        # 1. the bundle's prompt, the generation on the core, against the sim's reference
        rv = ("embedded" if args.embedded else os.path.join(HERE, "rv")) if args.rv else False
        a = run(env, extra, prompt, n_gen, True, lp, rv)
        got = np.fromfile(lp, np.float32).reshape(-1, want_logits.shape[1])
        same = a["tokens"][:len(want_tokens)] == want_tokens
        last = got[-1].tobytes() == want_logits[-1].tobytes()
        print(f"1. prompt of {len(prompt)}, {n_gen} tokens generated on the RISC-V core: tokens "
              f"{'identical to' if same else 'DIFFERENT from'} the sim's, the last step's logits "
              f"{'bit-exact' if last else 'DIFFERENT'}")
        print("   " + a.get("table", ""))
        print(f"   {a.get('steps')} steps, {a.get('ms_per_step')} ms per step (core {a.get('core_ms')} ms, "
              f"its lists {a.get('lists_ms')} ms)")
        out["prompt"] = {k: v for k, v in a.items() if k != "stdout"}
        out["prompt"].update(tokens_identical=same, last_logits_exact=last)
        ok = same and last
        # 2. a longer generation, the ARM's loop against the core's
        g = min(args.gen, vlen - len(prompt))
        if g > 0:
            h = run(env, extra, prompt, g, False, os.path.join(HERE, "logits_arm.f32"))
            c = run(env, extra, prompt, g, True, lp, rv)
            same2 = h["tokens"] == c["tokens"]
            arm = np.fromfile(os.path.join(HERE, "logits_arm.f32"), np.float32).reshape(-1, want_logits.shape[1])
            last2 = np.fromfile(lp, np.float32).reshape(-1, want_logits.shape[1])[-1].tobytes() == arm[-1].tobytes()
            print(f"2. {g} tokens generated: ARM {h.get('ms_per_step')} ms per step, RISC-V core "
                  f"{c.get('ms_per_step')} ms per step (core {c.get('core_ms')} ms, its lists {c.get('lists_ms')} ms); "
                  f"tokens {'identical' if same2 else 'DIFFERENT'}, the last step's logits "
                  f"{'bit-exact' if last2 else 'DIFFERENT'}")
            print("   " + c.get("table", ""))
            out["long"] = {"gen": g, "arm": {k: v for k, v in h.items() if k not in ("stdout", "tokens")},
                           "core": {k: v for k, v in c.items() if k not in ("stdout", "tokens")},
                           "tokens_identical": same2, "last_logits_exact": last2}
            ok = ok and same2 and last2
    finally:
        BL.stop(mm, buf)
    out_name = ("rvsched_rv_embedded_board.json" if args.embedded else "rvsched_rv_board.json") if args.rv \
        else "rvsched_board.json"
    with open(os.path.join(HERE, out_name), "w") as f:
        json.dump(out, f, indent=1)
    print("rvsched PASS" if ok else "rvsched FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

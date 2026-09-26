#!/usr/bin/env python3
"""L5b on the board: 8 sequences decoded at once, one weight pass per step (docs/llm_inference_plan.md §11.1).

Files in one directory on the board: picorv32.bit / .hwh (L2 build),
rt_fw.bin, pynq_matmul.py, the llm/ modules (runtime, compile_batch,
compile_model, compile_layer, export_w8a8, tokenizer), stories15M_d8.w8a8,
tokenizer.bin.

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \\
                  && cd /home/xilinx/l5b && python3 l5b_batch_demo.py'

1. reference: each of the 8 prompts generated alone (LlamaDevice, L5),
   greedy, --steps positions; the MD5 of every position's logits;
2. batched, staggered starts (sequence b starts at step b, so the
   positions differ inside a step): every sequence's logits bit-exact
   with its single run at every position, identical text;
3. throughput: all 8 sequences from step 0, total tok/s vs one sequence,
   cycles per step and the counter breakdown of one step.
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import Device, perf_breakdown  # noqa: E402
from runtime import BatchLlamaDevice, LlamaDevice  # noqa: E402

PROMPTS = ["Once upon a time", "One day, a little girl named Lily", "The cat sat on the mat and",
           "Tom and his dog went to the park", "There was a big red ball", "Mom said to Timmy",
           "In a small town, there lived", "The sun was shining and the birds"]


def md5(x):
    return hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("--model", default=os.path.join(HERE, "stories15M_d8.w8a8"))
    ap.add_argument("--steps", type=int, default=64, help="positions per sequence")
    args = ap.parse_args()
    dev = Device(args.bit, os.path.join(HERE, "rt_fw.bin"), ring_size=16, use_irq=True)
    tokp = os.path.join(HERE, "tokenizer.bin")
    ok = True

    # ---- 1. one sequence at a time
    single = LlamaDevice(args.model, tokp, device=dev)
    ref_tok, ref_md5 = [], []
    t0 = time.time()
    for p in PROMPTS:
        single.reset()
        pt = single.tok.encode(p)
        toks, hs = [pt[0]], []
        for pos in range(args.steps):
            lg = single.forward(toks[-1])
            hs.append(md5(lg))
            if pos + 1 < args.steps:
                toks.append(pt[pos + 1] if pos + 1 < len(pt) else int(np.argmax(lg)))
        ref_tok.append(toks)
        ref_md5.append(hs)
    t_single = time.time() - t0
    n_tok = len(PROMPTS) * args.steps
    cyc1 = np.mean([s[1] for s in single.stats])
    print(f"1. single: {len(PROMPTS)} x {args.steps} positions in {t_single:.1f} s = {n_tok / t_single:.1f} tok/s wall "
          f"(device {50e6 / cyc1:.1f} tok/s, {cyc1:.0f} cycles per token)")
    single.buf.freebuffer()

    # ---- 2. batched, staggered
    bd = BatchLlamaDevice(args.model, tokp, device=dev)
    got_md5 = {}
    t0 = time.time()
    out = bd.generate(PROMPTS, args.steps, stagger=1, on_logits=lambda b, p, lg: got_md5.__setitem__((b, p), md5(lg)))
    t_stag = time.time() - t0
    bad = [(b, p) for b in range(len(PROMPTS)) for p in range(args.steps) if got_md5.get((b, p)) != ref_md5[b][p]]
    same = [out[b] == ref_tok[b] for b in range(len(PROMPTS))]
    good = not bad and all(same)
    ok &= good
    print(f"2. batched, staggered starts ({len(bd.stats)} steps): "
          f"{'all' if not bad else f'{len(bad)} NOT'} {len(PROMPTS)} x {args.steps} positions bit-exact with the single "
          f"runs, texts {'identical' if all(same) else 'DIFFERENT'}: {'PASS' if good else 'FAIL'} ({t_stag:.1f} s)")
    for b in range(len(PROMPTS)):
        print(f"   [{b}] {bd.tok.decode(out[b])[:110]!r}")

    # ---- 3. throughput with all 8 sequences live from step 0
    bd.stats.clear()
    t0 = time.time()
    bd.generate(PROMPTS, args.steps, stagger=0)
    t_b = time.time() - t0
    cyc = np.mean([s[0] for s in bd.stats])
    print(f"3. batched, 8 sequences x {args.steps} positions: {len(bd.stats)} steps in {t_b:.1f} s = "
          f"{n_tok / t_b:.1f} tok/s wall ({t_single / t_b:.1f}x the "
          f"single-sequence rate); device {cyc:.0f} cycles per step = {8 * 50e6 / cyc:.1f} tok/s "
          f"({8 * cyc1 / cyc:.1f}x)")
    bd.step([10] * 8, [1] * 8, perf=True)
    b = perf_breakdown(dev.mm._perf(), dev.d)
    print(f"   one step: EX useful {100 * b['ex_useful']:.1f}%, LD busy {100 * b['ld_busy']:.1f}%, "
          f"VE active {100 * b['ve_active']:.1f}%, head blocked "
          + ", ".join(f"{e} {100 * v:.1f}%" for e, v in b["head_blocked"].items()) + f"; commands {b['commands']}")
    bd.close()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""L5 on the board: end-to-end text generation with LlamaDevice (docs/llm_inference_plan.md §9).

Files in one directory on the board: picorv32.bit / .hwh (L2 build),
rt_fw.bin, pynq_matmul.py, the llm/ modules (runtime, compile_model,
compile_layer, export_w8a8, fp32, sfu, tokenizer), stories15M_d8.w8a8,
tokenizer.bin, l5_ref.npz (llm/prepare_l5.py).

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \\
                  && cd /home/xilinx/l5 && python3 l5_generate.py'

Acceptance (plan §9.2), every token waited for by the notify interrupt:
1. correctness: the prompt + 128 greedy tokens, each position's logits
   bit-exact with DeviceModel (MD5s in l5_ref.npz), identical text;
2. accuracy: teacher-forced on the fp32 model's greedy sequences of the
   L0 evaluation prompts: top-1 vs fp32 >= 95%, and the device's argmax
   equal to DeviceModel's at every position;
3. demo: a sampled story (temperature 0.8, top-p 0.9), tok/s and the
   counter breakdown of one token (recorded only);
4. stability: 10 sequences of 256 tokens (sampled, not stopping at BOS),
   no errors or timeouts, interrupts received == tokens.
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import perf_breakdown  # noqa: E402
from runtime import LlamaDevice  # noqa: E402

F = np.float32


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("--model", default=os.path.join(HERE, "stories15M_d8.w8a8"))
    ap.add_argument("--ref", default=os.path.join(HERE, "l5_ref.npz"))
    ap.add_argument("--tests", default="1234", help="which acceptance parts to run")
    ap.add_argument("--prompt", default="Once upon a time, in a big forest,")
    args = ap.parse_args()
    ld = LlamaDevice(args.model, os.path.join(HERE, "tokenizer.bin"), bitfile=args.bit,
                     firmware=os.path.join(HERE, "rt_fw.bin"), use_irq=True)
    dev = ld.dev
    ref = np.load(args.ref)
    ok = True
    print(f"LlamaDevice: D = {dev.d}, {ld.cfg}, list {len(ld.dl)} descriptors, notify interrupt on")

    if "1" in args.tests:
        fed, want = [int(t) for t in ref["greedy_tokens"]], [str(h) for h in ref["greedy_md5"]]
        ld.reset()
        bad = 0
        for pos, t in enumerate(fed):
            lg = ld.forward(t)
            bad += hashlib.md5(lg.tobytes()).hexdigest() != want[pos]
            last = int(np.argmax(lg))
        text = ld.tok.decode(fed + [last])
        same = text == str(ref["greedy_text"])
        good = bad == 0 and same
        ok &= good
        print(f"1. correctness: {len(fed)} positions, {'all' if not bad else f'{bad} NOT'} bit-exact with "
              f"DeviceModel, text {'identical' if same else 'DIFFERENT'}: {'PASS' if good else 'FAIL'}\n   {text!r}")

    if "2" in args.tests:
        t0 = time.time()
        top1 = n = mism = 0
        for i, p in enumerate(ref["eval_prompts"]):
            toks = [int(t) for t in ref[f"eval{i}_tokens"]]
            top5, dmx = ref[f"eval{i}_fp32_top5"], ref[f"eval{i}_dev_argmax"]
            got = np.array([a for a, _ in ld.generate(None, forced=toks)])
            k = min(len(got), len(dmx))
            top1 += int((got[:k] == top5[:k, 0]).sum())
            mism += int((got[:k] != dmx[:k]).sum()) + abs(len(got) - len(dmx))
            n += k
        acc = top1 / n
        good = acc >= 0.95 and mism == 0
        ok &= good
        print(f"2. accuracy: teacher-forced top-1 vs fp32 {100 * acc:.1f}% over {n} positions "
              f"(gate 95%), argmax {'identical to' if not mism else f'{mism} positions differ from'} "
              f"DeviceModel: {'PASS' if good else 'FAIL'} ({time.time() - t0:.0f} s)")

    if "3" in args.tests:
        print("3. demo (temperature 0.8, top-p 0.9):\n   ", end="", flush=True)     # generate() yields the prompt too
        ld.stats.clear()
        t0 = time.time()
        n = 0
        for _, piece in ld.generate(args.prompt, steps=256, temperature=0.8, topp=0.9, seed=42):
            print(piece, end="", flush=True)
            n += 1
        dt = time.time() - t0
        cyc = np.array([s[1] for s in ld.stats])
        print(f"\n   {len(ld.stats)} positions in {dt:.1f} s = {len(ld.stats) / dt:.1f} tok/s wall "
              f"(device {50e6 / cyc.mean():.1f} tok/s at 50 MHz; {cyc.min()} .. {cyc.max()} cycles per token)")
        ld.reset()
        for t in ld.tok.encode(args.prompt)[:-1]:
            ld.forward(t)
        ld.forward(ld.tok.encode(args.prompt)[-1], perf=True)
        b = perf_breakdown(dev.mm._perf(), dev.d)
        print(f"   one token at pos {ld.pos - 1}: EX useful {100 * b['ex_useful']:.1f}%, EX fill "
              f"{100 * b['ex_fill']:.1f}%, LD busy {100 * b['ld_busy']:.1f}% "
              f"({b['ld_bytes_per_busy_cycle']:.2f} B/busy cycle), ST busy {100 * b['st_busy']:.1f}%, "
              f"VE active {100 * b['ve_active']:.1f}%, all idle {100 * b['all_idle']:.1f}%; head blocked "
              + ", ".join(f"{e} {100 * v:.1f}%" for e, v in b["head_blocked"].items())
              + f"; commands {b['commands']}")

    if "4" in args.tests:
        t0 = time.time()
        irq0, tokens, errors = dev.irq_events, 0, 0
        for s in range(10):
            try:
                for _ in ld.generate(str(ref["eval_prompts"][s % len(ref["eval_prompts"])]), steps=256,
                                     temperature=1.0, topp=0.9, seed=s + 1, stop_at_bos=False):
                    pass
            except Exception as e:                                       # noqa: BLE001
                errors += 1
                print(f"   sequence {s}: {type(e).__name__}: {e}")
            tokens += ld.pos
        irqs = dev.irq_events - irq0
        good = errors == 0 and tokens == 2560 and irqs == tokens
        ok &= good
        print(f"4. stability: 10 sequences, {tokens} tokens, {errors} errors, {irqs} interrupts received: "
              f"{'PASS' if good else 'FAIL'} ({time.time() - t0:.0f} s, {tokens / (time.time() - t0):.1f} tok/s)")

    ld.close()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

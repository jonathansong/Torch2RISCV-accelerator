#!/usr/bin/env python3
"""A/B on the board: the stories15M token list with and without weight prefetching.

Prefetching (compile_model.ModelCompiler(prefetch=True)) loads the first
weight chunk of every linear layer while the vector steps before it run
(norms, quantization, RoPE, attention, SiLU, residuals), instead of after
them. Same bitstream (L2), same tokens: both lists must give logits
bit-exact with DeviceModel (l4_golden.npz); the script compares the device
cycles per token and the counter breakdown of the last token.

Files: as l4_decoder_demo.py (picorv32.bit / .hwh, rt_fw.bin, pynq_matmul.py,
the llm/ modules, stories15M_d8.w8a8, l4_golden.npz).

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \\
                  && cd /home/xilinx/l4ab && python3 l4_prefetch_ab.py'
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compile_model as CM  # noqa: E402
from export_w8a8 import W8A8  # noqa: E402
from pynq_matmul import Device, allocate, perf_breakdown  # noqa: E402

F = np.float32
IO_BYTES = 0x40000


def run(dev, m, dl, toks, want):
    """All golden tokens through one list; returns (all exact, cycles per token, last-token counters)."""
    c, d = m.cfg, dev.d
    rows = dl.array()
    io_off = -(-m.raw.size // 4096) * 4096
    kv_off = io_off + IO_BYTES
    kvb = CM.kv_bytes(c)
    lst_off = kv_off + -(-kvb // 4096) * 4096
    buf = allocate(shape=(lst_off + rows.nbytes,), dtype=np.uint8)
    buf[:m.raw.size] = m.raw
    buf[io_off:lst_off] = 0
    buf[lst_off:] = np.frombuffer(rows.tobytes(), np.uint8)
    buf.flush()
    base = buf.physical_address
    bases = [base, base + io_off, base + kv_off]
    ok, cycles = True, []
    for pos, t in enumerate(toks):
        buf[io_off:io_off + 8] = np.frombuffer(CM.arg_block(pos, t), np.uint8)
        buf.flush()
        rec = dev.wait(dev.submit(base + lst_off, bases=bases, params=CM.token_params(pos, d, c.head_size),
                                  perf=pos == len(toks) - 1), timeout=10.0)
        buf.invalidate()
        lo = io_off + CM.IO_LOGITS
        ok &= rec["status"] == 0 and np.array(buf[lo:lo + 4 * c.vocab]).tobytes() == want[pos].tobytes()
        cycles.append(rec["cycles"])
    perf = dev.mm._perf()
    buf.freebuffer()
    return ok, np.array(cycles), perf


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("--model", default=os.path.join(HERE, "stories15M_d8.w8a8"))
    ap.add_argument("--golden", default=os.path.join(HERE, "l4_golden.npz"))
    args = ap.parse_args()
    dev = Device(args.bit, os.path.join(HERE, "rt_fw.bin"), ring_size=16)
    m = W8A8(args.model)
    gold = np.load(args.golden)
    toks, want = [int(t) for t in gold["tokens"]], gold["logits"]
    res = {}
    for name, pf in (("without prefetch", False), ("with prefetch", True)):
        dl = CM.ModelCompiler(m, prefetch=pf).build()
        ok, cyc, perf = run(dev, m, dl, toks, want)
        res[name] = cyc
        b = perf_breakdown(perf, dev.d) if perf else None
        print(f"{name:>17}: {len(toks)} tokens {'all bit-exact' if ok else 'NOT bit-exact'}, "
              f"{cyc.mean():.0f} cycles per token = {50e6 / cyc.mean():.2f} tok/s (device)")
        if b:
            print(f"{'':>19}last token: EX useful {100 * b['ex_useful']:.1f}%, LD busy {100 * b['ld_busy']:.1f}%, "
                  f"VE active {100 * b['ve_active']:.1f}%, head blocked "
                  + ", ".join(f"{e} {100 * v:.1f}%" for e, v in b["head_blocked"].items()))
        if not ok:
            print("FAIL")
            return 1
    a, b = res["without prefetch"], res["with prefetch"]
    print(f"prefetch saves {a.mean() - b.mean():.0f} cycles per token ({100 * (1 - b.mean() / a.mean()):.1f}%), "
          f"{a.mean() / b.mean():.3f}x; per position: min {(a - b).min()}, max {(a - b).max()}")
    dev.close()
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

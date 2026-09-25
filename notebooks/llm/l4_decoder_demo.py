#!/usr/bin/env python3
"""L4 on the board: the whole stories15M decoder as one static descriptor list (docs/llm_inference_plan.md §8).

Files in one directory on the board: picorv32.bit / .hwh (L2 build),
rt_fw.bin, pynq_matmul.py, the llm/ modules (compile_model,
compile_layer, export_w8a8, sa_funcsim, fp32, sfu, ref_model, checkpoint,
tokenizer), stories15M_d8.w8a8, tokenizer.bin and l4_golden.npz
(llm/test_l4.py --save: the tokens fed at each position and the logits
of DeviceModel with the bit-exact SFU).

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \\
                  && cd /home/xilinx/l4 && python3 l4_decoder_demo.py'

One CMA buffer holds the model (BASE0), the io area (BASE1: argument
block pos / token, logits) and the int8 KV cache (BASE2, cleared at the
start). The list (embedding, 6 layers with a head loop each, final norm,
looped classifier) is built once; per token the ARM writes the argument
block and submits the list with that token's PARAM block (rt_fw ring).

Per token: logits bit-exact with DeviceModel (golden file), argmax, device
cycles. With --funcsim N the first N tokens also run in the functional
simulator here (bit-exact logits and KV cache). The greedy continuation
is printed as text.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compile_model as CM  # noqa: E402
from export_w8a8 import W8A8  # noqa: E402
from pynq_matmul import Device, allocate, perf_breakdown  # noqa: E402
from sa_funcsim import SaFuncSim  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

F = np.float32
IO_BYTES = 0x40000


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("--model", default=os.path.join(HERE, "stories15M_d8.w8a8"))
    ap.add_argument("--golden", default=os.path.join(HERE, "l4_golden.npz"))
    ap.add_argument("--funcsim", type=int, default=0, help="also run the first N tokens in the functional simulator")
    args = ap.parse_args()
    dev = Device(args.bit, os.path.join(HERE, "rt_fw.bin"), ring_size=16)
    d = dev.d
    m = W8A8(args.model)
    c = m.cfg
    if m.d != d:
        raise SystemExit(f"{args.model} is packed for D = {m.d}, the overlay has D = {d}")
    gold = np.load(args.golden)
    toks, want, n_prompt = [int(t) for t in gold["tokens"]], gold["logits"], int(gold["n_prompt"])
    tok = Tokenizer(os.path.join(HERE, "tokenizer.bin"), c.vocab)

    mc = CM.ModelCompiler(m)
    dl = mc.build()
    rows = dl.array()
    io_off = -(-m.raw.size // 4096) * 4096
    kv_off = io_off + IO_BYTES
    kvb = CM.kv_bytes(c)
    lst_off = kv_off + -(-kvb // 4096) * 4096
    size = lst_off + rows.nbytes
    buf = allocate(shape=(size,), dtype=np.uint8)
    buf[:m.raw.size] = m.raw
    buf[io_off:lst_off] = 0                                           # io area, empty KV cache
    buf[lst_off:] = np.frombuffer(rows.tobytes(), np.uint8)
    buf.flush()
    base = buf.physical_address
    bases = [base, base + io_off, base + kv_off]
    print(f"rt_fw ready: D = {d}; model {m.raw.size / 2**20:.1f} MB, KV cache {kvb / 1024:.0f} KB; "
          f"list {len(dl)} descriptors; {len(toks)} tokens ({n_prompt} prompt + {len(toks) - n_prompt} greedy)")
    sim = None
    if args.funcsim:
        sim = SaFuncSim(d, base, size)
        sim.ddr_write(base, np.array(buf).tobytes())

    ok = True
    cycles, wall, out_toks = [], [], []
    for pos, t in enumerate(toks):
        buf[io_off:io_off + 8] = np.frombuffer(CM.arg_block(pos, t), np.uint8)
        buf.flush()
        prm = CM.token_params(pos, d, c.head_size)
        t0 = time.perf_counter()
        rec = dev.wait(dev.submit(base + lst_off, bases=bases, params=prm, perf=pos == len(toks) - 1), timeout=10.0)
        wall.append(time.perf_counter() - t0)
        buf.invalidate()
        lo = io_off + CM.IO_LOGITS
        got = np.array(buf[lo:lo + 4 * c.vocab]).view(F)
        exact = rec["status"] == 0 and got.tobytes() == want[pos].tobytes()
        fs = ""
        if sim is not None and pos < args.funcsim:
            sim.ddr_write(base + io_off, CM.arg_block(pos, t))
            sim.run_list(base + lst_off, bases=bases, params=prm)
            fs_ok = (sim.ddr_read(base + lo, 4 * c.vocab).tobytes() == got.tobytes() and
                     sim.ddr_read(base + kv_off, kvb).tobytes() == np.array(buf[kv_off:kv_off + kvb]).tobytes())
            ok &= fs_ok
            fs = f", funcsim {'exact (logits + KV)' if fs_ok else 'DIFFERENT'}"
        ok &= exact
        cycles.append(rec["cycles"])
        nxt = int(np.argmax(got))
        out_toks.append(nxt)
        print(f"  pos {pos:3d} token {t:5d} -> argmax {nxt:5d} {tok.decode([t, nxt])[len(tok.decode([t])):]!r:14s} "
              f"{'bit-exact' if exact else 'DIFFERENT'}{fs}, {rec['descriptors']} descriptors, "
              f"{rec['cycles']} cycles ({1e3 * wall[-1]:.1f} ms)"
              + (f", status {rec['status']:#x}" if rec["status"] else ""))
    perf = dev.mm._perf()
    text = tok.decode(toks[:1] + out_toks)
    print(f"device text (argmax after each fed token):\n  {text!r}")
    cyc = np.mean(cycles[1:]) if len(cycles) > 1 else cycles[0]
    print(f"per token: {cyc:.0f} cycles = {1e3 * cyc / 50e6:.1f} ms at 50 MHz ({50e6 / cyc:.1f} tok/s device), "
          f"wall {1e3 * np.mean(wall[1:]):.1f} ms incl. submit / wait")
    if perf:
        b = perf_breakdown(perf, d)
        print(f"last token counters: EX useful {100 * b['ex_useful']:.1f}%, LD busy {100 * b['ld_busy']:.1f}%, "
              f"VE active {100 * b['ve_active']:.1f}%, head blocked "
              + ", ".join(f"{e} {100 * v:.1f}%" for e, v in b["head_blocked"].items())
              + f"; commands {b['commands']}")
    buf.freebuffer()
    dev.close()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""L3 on the board: W8A8 linear layers of stories15M as descriptor lists (docs/llm_inference_plan.md §7).

Files in one directory on the board: picorv32.bit / .hwh (L2 build, no
hardware change in L3), rt_fw.bin, pynq_matmul.py, the llm/ modules
(compile_layer, export_w8a8, test_l3, sa_funcsim, fp32, sfu, ref_model,
checkpoint, tokenizer), stories15M_d8.w8a8 (llm/export_w8a8.py) and
l3_cases.npz (llm/test_l3.py --save: real layer inputs, the DeviceModel
outputs and the fp32 reference outputs). Staged by the L3 deploy step.

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \\
                  && cd /home/xilinx/l3 && python3 l3_linear_demo.py'

The .w8a8 file and an io area share one CMA buffer (BASE0 = model,
BASE1 = io). For Wqkv / Wo / W13 / W2 of layers 0 and 5 and the
classifier (32000 outputs, a LOOP_END list), the list runs on the device
(rt_fw ring, performance counters on) and its output must be

- bit-exact with the functional simulator running the same list here,
- bit-exact with DeviceModel (computed on the host, in l3_cases.npz),
- within 5% (relative L2) of the fp32 reference.

Reported per layer: device cycles, weight bytes per cycle, EX useful /
LD busy. Acceptance (plan §7.2): the classifier's LD is busy >= 90%.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compile_layer as CL  # noqa: E402
from export_w8a8 import W8A8  # noqa: E402
from pynq_matmul import Device, allocate, perf_breakdown  # noqa: E402
from sa_funcsim import SaFuncSim  # noqa: E402
from test_l3 import linear_list  # noqa: E402

F = np.float32
IO_BYTES = 1 << 20


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("--model", default=os.path.join(HERE, "stories15M_d8.w8a8"))
    ap.add_argument("--cases", default=os.path.join(HERE, "l3_cases.npz"))
    ap.add_argument("--no-funcsim", action="store_true", help="skip the functional simulator (faster)")
    args = ap.parse_args()
    dev = Device(args.bit, os.path.join(HERE, "rt_fw.bin"), ring_size=16)
    d = dev.d
    m = W8A8(args.model)
    if m.d != d:
        raise SystemExit(f"{args.model} is packed for D = {m.d}, the overlay has D = {d}")
    cases = np.load(args.cases)
    lay = CL.Layout(d)

    io_off = -(-m.raw.size // 4096) * 4096
    buf = allocate(shape=(io_off + IO_BYTES,), dtype=np.uint8)
    buf[:m.raw.size] = m.raw
    buf.flush()
    base = buf.physical_address
    bases = [base, base + io_off]
    lbuf = allocate(shape=(4096, 8), dtype=np.uint64)                 # list area (up to 4096 descriptors)
    print(f"rt_fw ready: D = {d}; {args.model}: {m.raw.size / 2**20:.1f} MB at {base:#x}; "
          f"prompt {str(cases['prompt'])!r}")
    sim_img = None
    if not args.no_funcsim:
        sim_img = SaFuncSim(d, base, io_off + IO_BYTES)
        sim_img.ddr_write(base, m.raw.tobytes())

    ok = True
    keys = [k[:-2] for k in cases.files if k.endswith("_x")]
    print(f"{'layer':>9} {'in->out':>12} {'result':>6} {'funcsim':>8} {'golden':>7} {'vs fp32':>8} "
          f"{'desc':>5} {'cycles':>8} {'W B/cyc':>8} {'EX useful':>9} {'LD busy':>8}")
    for key in keys:
        layer_s, name = key.split("_", 1)
        layer = None if layer_s == "cls" else int(layer_s)
        x, want, ref = cases[key + "_x"], cases[key + "_want"], cases[key + "_ref"]
        dl, n_out, nc = linear_list(lay, m, layer, name)
        rows = dl.array()
        lbuf[:len(rows)] = rows
        lbuf.flush()
        io = buf[io_off:io_off + IO_BYTES]
        io[:] = 0
        io[:x.nbytes] = np.frombuffer(x.tobytes(), np.uint8)
        buf.flush()
        rec = dev.wait(dev.submit(lbuf.physical_address, bases=bases, perf=True))
        perf = dev.mm._perf()
        buf.invalidate()
        got = np.array(buf[io_off + 0x10000:io_off + 0x10000 + 4 * n_out]).tobytes()
        # functional simulator on the same list and inputs
        fs = "-"
        if sim_img is not None:
            sim_img.ddr_write(base + io_off, np.array(io).tobytes()[:x.nbytes])
            la = base + io_off + IO_BYTES - 64 * (len(dl) + 1)
            sim_img.ddr_write(la, rows.tobytes())
            sim_img.run_list(la, bases=bases)
            fs_ok = sim_img.ddr_read(base + io_off + 0x10000, 4 * n_out).tobytes() == got
            fs = "exact" if fs_ok else "DIFF"
        gold_ok = got == want.tobytes()
        y = np.frombuffer(got, F).astype(np.float64)
        rel = np.linalg.norm(y - ref) / np.linalg.norm(ref.astype(np.float64))
        good = rec["status"] == 0 and gold_ok and fs in ("exact", "-") and rel < 0.05
        k = len(x)
        cyc = rec["cycles"]
        b = perf_breakdown(perf, d) if perf else None
        pct = lambda v: f"{100 * v:8.1f}%" if b else "      n/a"
        ok &= good
        print(f"{key:>9} {f'{k}->{n_out}':>12} {'PASS' if good else 'FAIL':>6} {fs:>8} "
              f"{'exact' if gold_ok else 'DIFF':>7} {100 * rel:7.2f}% {rec['descriptors']:5d} {cyc:8d} "
              f"{k * n_out / max(cyc, 1):8.2f} {pct(b and b['ex_useful'])} {pct(b and b['ld_busy'])}")
        if rec["status"]:
            print(f"          status {rec['status']:#x}")
    # acceptance: the classifier GEMV keeps the LD DMA busy (rerun on the inputs left in the io area)
    dl, n_out, nc = linear_list(lay, m, None, "wcls")
    rows = dl.array()
    lbuf[:len(rows)] = rows
    lbuf.flush()
    t0 = time.perf_counter()
    rec = dev.wait(dev.submit(lbuf.physical_address, bases=bases, perf=True))
    t = time.perf_counter() - t0
    perf = dev.mm._perf()
    if perf:
        b = perf_breakdown(perf, d)
        ld_ok = b["ld_busy"] >= 0.90
        ok &= ld_ok
        print(f"classifier: {rec['cycles']} cycles ({1e3 * t:.1f} ms wall), LD busy {100 * b['ld_busy']:.1f}% "
              f"(acceptance >= 90%: {'PASS' if ld_ok else 'FAIL'}), LD {b['ld_bytes_per_busy_cycle']:.2f} B per "
              f"busy cycle, EX useful {100 * b['ex_useful']:.1f}%, VE active {100 * b['ve_active']:.1f}%, "
              f"head blocked {', '.join(f'{e} {100 * v:.1f}%' for e, v in b['head_blocked'].items())}")
    else:
        print("classifier: no performance counters in this overlay")
    lbuf.freebuffer()
    buf.freebuffer()
    dev.close()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

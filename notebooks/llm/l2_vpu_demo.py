#!/usr/bin/env python3
"""L2 on the board: fp32 vector engine, SFU, TRANSPOSE (docs/llm_inference_plan.md §6),
every result bit-exact against the functional simulator.

Files in one directory on the board: picorv32.bit / .hwh (L2 build),
rt_fw.bin (firmware/rt), pynq_matmul.py (driver/), gen_desc_cases.py
(firmware/desc_run/), sa_funcsim.py, fp32.py, sfu.py (llm/), this script.

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \\
                  && cd /home/xilinx/l2 && python3 l2_vpu_demo.py'

1. The descriptor-list regression of firmware/desc_run (L0 GEMM / vector,
   L1 loops / CALL / LDPARAM, L2 softmax, GEMM + dequant, RMSNorm +
   quantization, attention scores with TRANSPOSE), built for a DDR
   buffer here: output bytes and descriptor counts vs the functional
   simulator / NumPy, device cycles recorded.
2. Random fp32 vector commands (ops, SFU functions, index modes, row
   reductions, immediates, conversions; inputs with NaN, inf, zeros,
   subnormals and huge values) vs the functional simulator, bit-exact.
3. Error isolation: fp commands the simulator rejects fail on the device
   too, and the next entry runs normally.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_desc_cases as gdc  # noqa: E402
from pynq_matmul import MEM_ACC, Device, DescList, allocate, laddr  # noqa: E402
from sa_funcsim import SaError, SaFuncSim  # noqa: E402

F32 = 3


def run_both(dev, d, mem, base, dl, chunks, out_off, nbytes):
    """Run a list on the device and in the simulator over the same DDR image.
    Returns (device record, device bytes, simulator bytes or SaError, simulator count)."""
    mem[:] = 0
    lst = dl.array().tobytes()
    for off, data in chunks + [(gdc.LIST_OFF, lst)]:
        mem[off:off + len(data)] = np.frombuffer(data, np.uint8)
    mem.flush()
    sim = SaFuncSim(d, base, gdc.DDR_BYTES)
    sim.ddr_write(base, np.array(mem).tobytes())
    try:
        n = sim.run_list(base + gdc.LIST_OFF)
        want = sim.ddr_read(base + out_off, nbytes).tobytes()
    except SaError as e:
        n, want = None, e
    rec = dev.run(base + gdc.LIST_OFF)
    mem.invalidate()
    return rec, np.array(mem[out_off:out_off + nbytes]).tobytes(), want, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("-n", type=int, default=300, help="random fp32 lists in test 2")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    dev = Device(args.bit, os.path.join(HERE, "rt_fw.bin"), ring_size=16)
    d = dev.d
    mem = allocate(shape=(gdc.DDR_BYTES,), dtype=np.uint8)
    base = mem.physical_address
    gdc.DDR_BASE = base                                  # the case builder's DDR image lives here
    ok = True
    print(f"rt_fw ready: D = {d}, DDR image at {base:#x}")

    # ---- 1. descriptor-list regression
    t0 = time.perf_counter()
    cases = gdc.build_cases(d)
    print(f"1. {len(cases)} lists built ({time.perf_counter() - t0:.1f} s, simulator expectations)")
    bad = 0
    for name, dl, chunks, out_off, exp, n_exec in cases:
        rec, got, want, n = run_both(dev, d, mem, base, dl, chunks, out_off, len(exp))
        good = rec["status"] == 0 and rec["descriptors"] == n_exec and got == exp and want == exp
        bad += not good
        tag = "PASS" if good else f"FAIL (status {rec['status']:#x}, {rec['descriptors']}/{n_exec} descriptors)"
        if not good and rec["status"] == 0 and got != exp:
            nd = sum(a != b for a, b in zip(got, exp))
            tag += f", {nd} bytes differ"
        print(f"   {tag:8s} {rec['cycles']:8d} cycles  {name}")
    ok &= bad == 0
    print(f"1. descriptor lists: {'PASS' if not bad else f'FAIL ({bad})'}")

    # ---- 2. random fp32 commands
    rng = np.random.default_rng(args.seed)
    bad = done = rejected = 0
    first = None
    t0 = time.perf_counter()
    while done < args.n:
        dl, chunks, nbytes, desc = gdc.random_fp_list(rng, d)
        rec, got, want, n = run_both(dev, d, mem, base, dl, chunks, gdc.O_OFF, nbytes)
        if isinstance(want, SaError):
            rejected += 1
            if rec["status"] == 0:                       # the simulator rejects it: so must the device
                bad += 1
                first = first or f"device accepted a command the simulator rejects ({want}): {desc}"
            continue
        done += 1
        if rec["status"] != 0 or got != want:
            bad += 1
            if first is None:
                g, w = np.frombuffer(got, np.uint32), np.frombuffer(want, np.uint32)
                i = int(np.argmax(g != w)) if rec["status"] == 0 else -1
                first = (f"status {rec['status']:#x}" if rec["status"] else
                         f"{int((g != w).sum())} words differ, first [{i}] {g[i]:#010x} vs {w[i]:#010x}") + f": {desc}"
    ok &= bad == 0
    print(f"2. {done} random fp32 lists (+{rejected} rejected by both): {'PASS' if not bad else f'FAIL ({bad})'}"
          f" in {time.perf_counter() - t0:.1f} s")
    if first:
        print(f"   first failure: {first}")

    # ---- 3. error isolation
    n = 8 * d
    dl = DescList().ld(base + gdc.X_OFF, laddr(MEM_ACC, 0), 1, 4 * n, 4 * n)
    dl.ve(laddr(MEM_ACC, 0), 0, laddr(MEM_ACC, 64), n, 5, F32 | 1 << 2, fp=True)   # I16 output: illegal
    dl.end(0x61)
    x = gdc.rand_f32(rng, n)
    rec_bad, _, want_bad, _ = run_both(dev, d, mem, base, dl, [(gdc.X_OFF, x.tobytes())], 0, 4)
    name, dl, chunks, out_off, exp, n_exec = cases[-4]                      # L2 softmax
    rec, got, want, _ = run_both(dev, d, mem, base, dl, chunks, out_off, len(exp))
    good = isinstance(want_bad, SaError) and rec_bad["status"] != 0 and rec["status"] == 0 and got == exp
    ok &= good
    print(f"3. error isolation: {'PASS' if good else 'FAIL'} (bad entry status {rec_bad['status']:#x}, "
          f"then the softmax list)")

    mem.freebuffer()
    dev.close()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

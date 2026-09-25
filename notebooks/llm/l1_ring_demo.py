#!/usr/bin/env python3
"""L1 on the board: resident runtime firmware, command ring, notify interrupt,
compiler-facing command extensions (docs/llm_inference_plan.md §5).

Files in one directory on the board: picorv32.bit / .hwh (L1 build),
rt_fw.bin (firmware/rt), pynq_matmul.py (driver/), this script.

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \\
                  && cd /home/xilinx/l1 && python3 l1_ring_demo.py'

1. 200 asynchronous submissions (GEMM and vector lists mixed, ring of 64:
   the ARM runs ahead until the ring is full), every result vs NumPy.
2. One static list, only the parameter block and the bases change per run:
   a looped int8 add whose chunk count comes from PARAM2 (LOOP_END count
   field) and whose addresses advance by PARAM strides.
3. A gather by an index table in DDR: LDPARAM + CALL / RET + SETREG add.
4. Error isolation: a list with an invalid descriptor, then a good one.
5. The notify interrupt (matmul_0/notify_irq via the AXI interrupt
   controller): 20 entries each waited for by interrupt.
6. Latency: submit -> completion of a NOP and of a small list (recorded only).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import (MEM_SPAD_A, MEM_SPAD_B, VOPS, Device, DescList, allocate,  # noqa: E402
                         build_gemm_list, build_vector_list, golden, laddr, vector_golden)

P = DescList.REG_PARAM


def buf(data):
    b = allocate(shape=data.shape, dtype=data.dtype)
    b[:] = data
    b.flush()
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("-n", type=int, default=200, help="asynchronous submissions in test 1")
    args = ap.parse_args()
    dev = Device(args.bit, os.path.join(HERE, "rt_fw.bin"), ring_size=64)
    d = dev.d
    rng = np.random.default_rng(1)
    ok = True
    print(f"rt_fw ready: D = {d}, ring of {dev.size}")

    # ---- 1. asynchronous batch
    jobs = []
    t0 = time.perf_counter()
    for i in range(args.n):
        if i % 3 == 0:
            m, n, k = (int(rng.integers(1, 5)) * d for _ in range(3))
            a = buf(rng.integers(-128, 128, (m, k), dtype=np.int8))
            b = buf(rng.integers(-128, 128, (k, n), dtype=np.int8))
            c = buf(np.zeros((m, n), np.int32))
            dl = build_gemm_list(d, m, n, k, a.physical_address, b.physical_address, c.physical_address)
            jobs.append((dev.submit(dl), "gemm", (a, b, c)))
        else:
            n = int(rng.integers(1, 200)) * d
            x = buf(rng.integers(-128, 128, n, dtype=np.int8))
            y = buf(rng.integers(-128, 128, n, dtype=np.int8))
            o = buf(np.zeros(n, np.int8))
            dl = build_vector_list(d, "add", 0, 0, n, n, x.physical_address, y.physical_address, o.physical_address)
            jobs.append((dev.submit(dl), "vec", (x, y, o)))
    submitted = time.perf_counter() - t0
    bad = 0
    for seq, kind, bufs in jobs:
        rec = dev.wait(seq)
        bufs[-1].invalidate()
        if kind == "gemm":
            good = rec["status"] == 0 and np.array_equal(np.array(bufs[2]), golden(bufs[0], bufs[1]))
        else:
            good = rec["status"] == 0 and np.array_equal(np.array(bufs[2]), vector_golden("add", bufs[0], bufs[1]))
        bad += not good
        for b in bufs:
            b.freebuffer()
    total = time.perf_counter() - t0
    ok &= bad == 0
    print(f"1. {args.n} async submissions: {'PASS' if not bad else f'FAIL ({bad})'}; "
          f"submitted in {1e3 * submitted:.1f} ms, all done in {1e3 * total:.1f} ms")

    # ---- 2. one static list, parameters only
    sa, sb, so = laddr(MEM_SPAD_A, 0), laddr(MEM_SPAD_B, 0), laddr(MEM_SPAD_B, 1024)
    dl = DescList()
    dl.ld(0, sa, 1, 512, 512, base=0, dyn=[("ddr", 0, True)])
    dl.ld(0, sb, 1, 512, 512, base=1, dyn=[("ddr", 0, True)])
    dl.ve(sa, sb, so, 512, VOPS["add"], 0)
    dl.st(0, so, 1, 512, 512, base=2, dyn=[("ddr", 1, True)])
    dl.loop_end(-4, 1, k1=0, s1=512, k2=1, s2=512, dyn=[("count", 2)]).end(0x2)
    lst = buf(dl.array())
    res = []
    for chunks in (1, 5, 16):
        n = 512 * chunks
        x = buf(rng.integers(-128, 128, n, dtype=np.int8))
        y = buf(rng.integers(-128, 128, n, dtype=np.int8))
        o = buf(np.zeros(n + 64, np.int8))
        rec = dev.run(lst.physical_address, bases=[x.physical_address, y.physical_address, o.physical_address],
                      params=[0, 0, chunks])
        o.invalidate()
        good = (rec["status"] == 0 and rec["descriptors"] == 5 * chunks + 1 and
                np.array_equal(np.array(o[:n]), vector_golden("add", x, y)) and not np.array(o[n:]).any())
        res.append(good)
        for b in (x, y, o):
            b.freebuffer()
    lst.freebuffer()
    ok &= all(res)
    print(f"2. static list, PARAM-driven length and strides (1, 5, 16 chunks): {'PASS' if all(res) else 'FAIL'}")

    # ---- 3. gather by an index table: LDPARAM + CALL / RET + SETREG add
    rows = buf(rng.integers(0, 256, (64, 64), dtype=np.uint8))
    idx = np.array([7, 0, 63, 12, 12, 40], "<u4")
    tab = buf(idx)
    out = buf(np.zeros((len(idx), 64), np.uint8))
    dl = DescList().setreg((P + 3, tab.physical_address), (P + 4, out.physical_address))
    dl.ldparam(0, 2, mul=64, add=rows.physical_address, dyn=[("addr", 3)])
    dl.call(3, rel=True)
    dl.loop_end(-2, len(idx), k1=3, s1=4, k2=3, s2=0)
    dl.end(0x3)
    dl.ld(0, sa, 1, 64, 64, dyn=[("ddr", 2)])
    dl.st(0, sa, 1, 64, 64, dyn=[("ddr", 4)])
    dl.setreg((P + 4, 64, True))
    dl.ret()
    rec = dev.run(dl)
    out.invalidate()
    good = rec["status"] == 0 and np.array_equal(np.array(out), np.array(rows)[idx])
    ok &= good
    print(f"3. LDPARAM gather + CALL / RET: {'PASS' if good else 'FAIL'} ({rec['descriptors']} descriptors)")
    for b in (rows, tab, out):
        b.freebuffer()

    # ---- 4. error isolation
    badl = DescList().end()
    badl.rows[0][0] = 0                                  # opcode 0: invalid
    s_bad = dev.submit(badl)
    x = buf(rng.integers(-128, 128, 64 * d, dtype=np.int8))
    o = buf(np.zeros(64 * d, np.int8))
    s_good = dev.submit(build_vector_list(d, "copy", 0, 0, 64 * d, 64 * d, x.physical_address, x.physical_address,
                                          o.physical_address))
    r_bad, r_good = dev.wait(s_bad), dev.wait(s_good)
    o.invalidate()
    good = r_bad["status"] != 0 and r_good["status"] == 0 and np.array_equal(np.array(o), np.array(x))
    ok &= good
    print(f"4. error isolation: {'PASS' if good else 'FAIL'} (bad entry status {r_bad['status']:#x})")
    for b in (x, o):
        b.freebuffer()

    # ---- 5. notify interrupt
    try:
        from pynq import Interrupt
        dev.irq = Interrupt("matmul_0/notify_irq")
        lat = []
        for _ in range(20):
            t = time.perf_counter()
            rec = dev.wait(dev.submit(None, kind=2, irq=True))   # NOP
            lat.append(time.perf_counter() - t)
        dev.irq = None
        print(f"5. notify interrupt: PASS, 20 NOPs waited by interrupt, median {1e6 * np.median(lat):.0f} us")
    except Exception as e:                                      # no interrupt in the hwh, UIO missing, ...
        ok = False
        dev.irq = None
        print(f"5. notify interrupt: FAIL ({type(e).__name__}: {e})")

    # ---- 6. latency (polling)
    lat_nop, lat_list = [], []
    for _ in range(100):
        t = time.perf_counter()
        dev.wait(dev.submit(None, kind=2, irq=False))
        lat_nop.append(time.perf_counter() - t)
    x = buf(np.zeros(d, np.int8))
    small = build_vector_list(d, "copy", 0, 0, d, d, x.physical_address, x.physical_address, x.physical_address)
    lbuf = buf(small.array())
    for _ in range(100):
        t = time.perf_counter()
        dev.wait(dev.submit(lbuf.physical_address, irq=False))
        lat_list.append(time.perf_counter() - t)
    x.freebuffer()
    lbuf.freebuffer()
    print(f"6. latency submit -> completion (polling): NOP median {1e6 * np.median(lat_nop):.0f} us, "
          f"small list median {1e6 * np.median(lat_list):.0f} us")

    dev.close()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

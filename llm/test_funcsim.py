#!/usr/bin/env python3
"""Checks of the functional simulator (llm/sa_funcsim.py), D = 8 and 16.

1. The RTL co-simulation cases (firmware/desc_run/gen_desc_cases.py: the
   lists and expected bytes that tb_system.v checks the RTL against) give
   the same output bytes on the functional simulator.
2. Random GEMM and vector-operation lists built by the driver
   (build_gemm_list / build_vector_list) match the driver's NumPy goldens.
3. Control flow: JUMP chains, count limits, BASE relocation, END status.
4. Illegal descriptors / commands raise the RTL engine and error code.

    python3 llm/test_funcsim.py        (exit status 0 = all passed)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "driver"))
sys.path.insert(0, os.path.join(ROOT, "firmware", "desc_run"))
import gen_desc_cases as gdc  # noqa: E402
import pynq_matmul as pm  # noqa: E402
from sa_funcsim import (ENG_EX, ENG_FETCH, ENG_LD, ENG_ST, ENG_VE, XERR_RANGE, XERR_SHAPE,  # noqa: E402
                        SaError, SaFuncSim)

BASE = 0x10000000
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        fails.append(name)


def put_list(sim, dl, off):
    sim.ddr_write(BASE + off, dl.array().tobytes())
    return BASE + off


def cosim_cases(d):
    for name, dl, chunks, out_off, exp in gdc.build_cases(d):
        sim = SaFuncSim(d, gdc.DDR_BASE, gdc.DDR_BYTES)
        for off, data in chunks:
            sim.ddr_write(gdc.DDR_BASE + off, data)
        sim.ddr_write(gdc.DDR_BASE + gdc.LIST_OFF, dl.array().tobytes())
        n = sim.run_list(gdc.DDR_BASE + gdc.LIST_OFF)
        got = sim.ddr_read(gdc.DDR_BASE + out_off, len(exp)).tobytes()
        check(f"D={d} cosim case: {name}", got == exp and n == len(dl),
              f"{sum(a != b for a, b in zip(got, exp))} bytes differ")


def random_gemms(d, rng, count):
    for t in range(count):
        m, n, k = (int(rng.integers(1, 5)) * d, int(rng.integers(1, 9)) * d, int(rng.integers(1, 17)) * d)
        quant = pm.Requant(int(rng.integers(-300, 300)), int(rng.integers(0, 20)), int(rng.integers(-5, 5)),
                           -128, 127) if rng.random() < 0.4 else None
        bias = rng.random() < 0.5
        relu = bool(quant is not None and rng.random() < 0.5)
        bsplit = ["auto", True, False][int(rng.integers(0, 3))]
        prefetch = bool(rng.random() < 0.7)
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b = rng.integers(-128, 128, (k, n), dtype=np.int8)
        bv = rng.integers(-2**16, 2**16, n if quant else (m, n), dtype=np.int32) if bias else None
        sim = SaFuncSim(d, BASE, 1 << 22)
        A, B, C, BI, L = BASE, BASE + 0x40000, BASE + 0x80000, BASE + 0x100000, 0x200000
        sim.ddr_write(A, a.tobytes())
        sim.ddr_write(B, b.tobytes())
        if bias:
            sim.ddr_write(BI, bv.tobytes())
        try:
            dl = pm.build_gemm_list(d, m, n, k, A, B, C, BI if bias else None, quant, relu,
                                    prefetch=prefetch, bsplit=bsplit)
        except Exception as e:                   # shapes the schedule cannot take
            print(f"  skip  {m}x{n}x{k}: {e}")
            continue
        sim.run_list(put_list(sim, dl, L))
        if quant is not None:
            exp = pm.qgemm_golden(a, b, bv, quant, relu)
            got = sim.ddr_read(C, m * n).view(np.int8).reshape(m, n)
        else:
            exp = pm.golden(a, b) + (bv if bias else 0)
            got = sim.ddr_read(C, 4 * m * n).view("<i4").reshape(m, n)
        check(f"D={d} random gemm {m}x{n}x{k} quant={quant is not None} bias={bias} relu={relu} "
              f"bsplit={bsplit} prefetch={prefetch}", np.array_equal(got, exp))


def random_vectors(d, rng, count):
    dts = {0: np.int8, 1: np.int16, 2: np.int32}
    for t in range(count):
        it = int(rng.integers(0, 3))
        ops = ["add", "sub", "max", "min", "copy"] + (["mul"] if it != 2 else [])
        op = ops[int(rng.integers(0, len(ops)))]
        ot = 2 if op == "mul" and rng.random() < 0.7 else int(rng.integers(0, 3))
        n = int(rng.integers(1, 700)) * d
        period = int(rng.choice([0, 0, 1, int(rng.integers(2, 40))]))
        ny = n if period == 0 else d * period
        if ny > n:
            ny = n
        requant = pm.Requant(int(rng.integers(-2000, 2000)), int(rng.integers(0, 24)), int(rng.integers(-50, 50)),
                             int(rng.integers(-2**20, 0)), int(rng.integers(0, 2**20))) if rng.random() < 0.4 else None
        relu = bool(rng.random() < 0.3)
        info = np.iinfo(dts[it])
        lo, hi = (-2**24, 2**24) if it == 2 else (info.min, info.max + 1)
        x = rng.integers(lo, hi, n, dtype=np.int64).astype(dts[it])
        y = rng.integers(lo, hi, ny, dtype=np.int64).astype(dts[it])
        sim = SaFuncSim(d, BASE, 1 << 22)
        X, Y, O, L = BASE, BASE + 0x100000, BASE + 0x200000, 0x300000
        sim.ddr_write(X, x.tobytes())
        sim.ddr_write(Y, y.tobytes())
        dl = pm.build_vector_list(d, op, it, ot, n, ny, X, Y, O, relu, requant)
        sim.run_list(put_list(sim, dl, L))
        exp = pm.vector_golden(op, x, y, dts[ot], relu, requant)
        got = sim.ddr_read(O, exp.nbytes).view(exp.dtype)
        check(f"D={d} random vector {op} {np.dtype(dts[it]).name}->{np.dtype(dts[ot]).name} n={n} "
              f"period={period} relu={relu} requant={requant is not None}", np.array_equal(got, exp))


def control_flow(d):
    sim = SaFuncSim(d, BASE, 1 << 20)
    x = np.arange(256, dtype=np.uint8)
    # list 1 at 0x1000: LD x (BASE0-relative) -> SPAD_A, JUMP 0x2000;  list 2: ST (BASE1-relative), END
    l1 = pm.DescList().ld(0, pm.laddr(pm.MEM_SPAD_A, 0), 1, 256, 256, base=0).jump(BASE + 0x2000)
    l2 = pm.DescList().st(0x40, pm.laddr(pm.MEM_SPAD_A, 0), 1, 256, 256, base=1).end(0xABC)
    sim.ddr_write(BASE + 0x8000, x.tobytes())
    put_list(sim, l1, 0x1000)
    put_list(sim, l2, 0x2000)
    n = sim.run_list(BASE + 0x1000, bases=[BASE + 0x8000, BASE + 0x9000, 0, 0])
    check(f"D={d} JUMP + BASE relocation + END status",
          n == 4 and sim.dl_status == 0xABC and np.array_equal(sim.ddr_read(BASE + 0x9040, 256), x))
    sim2 = SaFuncSim(d, BASE, 1 << 20)
    sim2.ddr_write(BASE + 0x8000, x.tobytes())
    put_list(sim2, l1, 0x1000)
    put_list(sim2, l2, 0x2000)
    n = sim2.run_list(BASE + 0x1000, count=1, bases=[BASE + 0x8000, BASE + 0x9000, 0, 0])
    check(f"D={d} count limit stops after 1 descriptor", n == 1 and sim2.cmds["LD"] == 1 and sim2.cmds["ST"] == 0)


def errors(d):
    def expect(name, fn, eng, code):
        try:
            fn()
        except SaError as e:
            check(f"D={d} error: {name}", (e.engine, e.code) == (eng, code), f"got engine {e.engine} code {e.code}")
            return
        check(f"D={d} error: {name}", False, "no error")

    sim = SaFuncSim(d, BASE, 1 << 20)
    expect("Kt = 0", lambda: sim.ex(0, 0, 0, 0), ENG_EX, XERR_SHAPE)
    expect("EX C out of range", lambda: sim.ex(0, 0, sim.acc_words - 2, 1), ENG_EX, XERR_RANGE)
    expect("LD INTERLEAVE into ACC", lambda: sim.ld(BASE, pm.laddr(pm.MEM_ACC, 0), 1, 64, 64, 1), ENG_LD, XERR_SHAPE)
    expect("LD row_bytes not multiple of 8", lambda: sim.ld(BASE, pm.laddr(pm.MEM_SPAD_A, 0), 1, 12, 16), ENG_LD,
           XERR_SHAPE)
    expect("ST local range", lambda: sim.st(BASE, pm.laddr(pm.MEM_SPAD_B, sim.spad_words - 1), 2, d, d), ENG_ST,
           XERR_RANGE)
    expect("VE MUL on int32", lambda: sim.ve(pm.laddr(pm.MEM_ACC, 0), pm.laddr(pm.MEM_ACC, 8),
                                             pm.laddr(pm.MEM_ACC, 16), 1, 2, 2 | 2 << 2), ENG_VE, XERR_SHAPE)
    expect("VE int8 in ACC", lambda: sim.ve(pm.laddr(pm.MEM_ACC, 0), 0, pm.laddr(pm.MEM_SPAD_A, 0), 1, 5, 0),
           ENG_VE, XERR_RANGE)
    bad = pm.DescList().end()
    bad.rows[0][0] |= 1 << 20                     # reserved header bit
    sim.ddr_write(BASE + 0x1000, bad.array().tobytes())
    expect("reserved header bit", lambda: sim.run_list(BASE + 0x1000), ENG_FETCH, XERR_SHAPE)
    sim.ddr_write(BASE + 0x2000, bytes(64))
    expect("opcode 0 (zeroed memory)", lambda: sim.run_list(BASE + 0x2000), ENG_FETCH, XERR_SHAPE)
    sim.ddr_write(BASE + 0x3000, pm.DescList().jump(BASE + 0x3010).array().tobytes())
    expect("misaligned JUMP", lambda: sim.run_list(BASE + 0x3000), ENG_FETCH, XERR_SHAPE)


def main():
    rng = np.random.default_rng(2026)
    for d in (8, 16):
        print(f"== D = {d}")
        cosim_cases(d)
        random_gemms(d, rng, 40)
        random_vectors(d, rng, 60)
        control_flow(d)
        errors(d)
    print("PASS" if not fails else f"FAIL: {len(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())

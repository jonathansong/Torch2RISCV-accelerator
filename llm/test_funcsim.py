#!/usr/bin/env python3
"""Checks of the functional simulator (llm/sa_funcsim.py), D = 8 and 16.

1. The RTL co-simulation cases (firmware/desc_run/gen_desc_cases.py: the
   lists and expected bytes that tb_system.v checks the RTL against) give
   the same output bytes on the functional simulator.
2. Random GEMM and vector-operation lists built by the driver
   (build_gemm_list / build_vector_list) match the driver's NumPy goldens.
3. Control flow: JUMP chains, count limits, BASE relocation, END status.
4. Illegal descriptors / commands raise the RTL engine and error code.
5. L1 command extensions (plan §5.3): BASE4-15, dynamic fields, SETREG,
   LOOP_END (nested), relative JUMP, CALL / RET, LDPARAM, and their errors;
   expected values are computed independently of the simulator.
6. L2 fp32 vector engine (plan §6): every op / type / conversion, affine,
   RELU, FUNC, SWAPNEG, index modes (LIN / MOD / DIV / IMM), VALID, row
   reductions, TRANSPOSE, softmax and RMSNorm as command sequences, errors;
   references in NumPy float32 (normal-range data: IEEE = the FTZ model).

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
    for name, dl, chunks, out_off, exp, n_exec in gdc.build_cases(d):
        sim = SaFuncSim(d, gdc.DDR_BASE, gdc.DDR_BYTES)
        for off, data in chunks:
            sim.ddr_write(gdc.DDR_BASE + off, data)
        sim.ddr_write(gdc.DDR_BASE + gdc.LIST_OFF, dl.array().tobytes())
        n = sim.run_list(gdc.DDR_BASE + gdc.LIST_OFF)
        got = sim.ddr_read(gdc.DDR_BASE + out_off, len(exp)).tobytes()
        check(f"D={d} cosim case: {name}", got == exp and n == n_exec,
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
    bad.rows[0][0] |= 1 << 15                     # reserved header bit
    sim.ddr_write(BASE + 0x1000, bad.array().tobytes())
    expect("reserved header bit", lambda: sim.run_list(BASE + 0x1000), ENG_FETCH, XERR_SHAPE)
    sim.ddr_write(BASE + 0x2000, bytes(64))
    expect("opcode 0 (zeroed memory)", lambda: sim.run_list(BASE + 0x2000), ENG_FETCH, XERR_SHAPE)
    sim.ddr_write(BASE + 0x3000, pm.DescList().jump(BASE + 0x3010).array().tobytes())
    expect("misaligned JUMP", lambda: sim.run_list(BASE + 0x3000), ENG_FETCH, XERR_SHAPE)


def l1_extensions(d):
    DL, P = pm.DescList, pm.DescList.REG_PARAM
    SA = pm.laddr(pm.MEM_SPAD_A, 0)
    rng = np.random.default_rng(d)

    def fresh():
        sim = SaFuncSim(d, BASE, 1 << 20)
        src = rng.integers(0, 256, 1 << 14, dtype=np.uint8)
        sim.ddr_write(BASE + 0x10000, src.tobytes())
        return sim, src

    def run(sim, dl, **kw):
        return sim.run_list(put_list(sim, dl, 0x1000), **kw)

    # BASE4..15 relocation
    sim, src = fresh()
    dl = DL()
    for b in (4, 9, 15):
        dl.ld(0, SA, 1, 64, 64, base=b).st(0, SA, 1, 64, 64, base=15 - b + 1 if b != 15 else 1)
    dl.end()
    bases = [0] * 16
    for b in (4, 9, 15):
        bases[b] = BASE + 0x10000 + 64 * b
        bases[15 - b + 1 if b != 15 else 1] = BASE + 0x20000 + 64 * b
    run(sim, dl, bases=bases)
    check(f"D={d} L1 BASE4..15 relocation",
          all(np.array_equal(sim.ddr_read(BASE + 0x20000 + 64 * b, 64), src[64 * b:64 * b + 64]) for b in (4, 9, 15)))

    # dynamic fields: rows (replace) and DDR address (add)
    sim, src = fresh()
    dl = DL().ld(BASE + 0x10000, SA, 1, 64, 64, dyn=[("rows", 3), ("ddr", 1, True)]) \
             .st(BASE + 0x20000, SA, 1, 64, 64, dyn=[("rows", 3)]).end()
    run(sim, dl, params=[0, 128, 0, 5])
    got = sim.ddr_read(BASE + 0x20000, 6 * 64)
    check(f"D={d} L1 dynamic rows (replace) + DDR address (add)",
          np.array_equal(got[:320], src[128:448]) and not got[320:].any())

    # dynamic VE LEN and EX Kt against the static commands
    sim, src = fresh()
    x = rng.integers(-128, 128, 64 * d, dtype=np.int8)
    sim.ddr_write(BASE + 0x30000, x.tobytes())
    n = 5 * d
    dl = DL().ld(BASE + 0x30000, SA, 1, 64 * d, 64 * d) \
             .ve(SA, SA, pm.laddr(pm.MEM_SPAD_B, 0), 8 * d, pm.VOPS["add"], 0, dyn=[("len", 2)]) \
             .st(BASE + 0x40000, pm.laddr(pm.MEM_SPAD_B, 0), 1, 8 * d, 8 * d).end()
    run(sim, dl, params=[0, 0, n])
    got = sim.ddr_read(BASE + 0x40000, 8 * d).view(np.int8)
    exp = np.concatenate([np.clip(2 * x[:n].astype(int), -128, 127), np.zeros(3 * d, int)])
    check(f"D={d} L1 dynamic VE LEN", np.array_equal(got, exp.astype(np.int8)))
    a = rng.integers(-128, 128, (d, 4 * d), dtype=np.int8)
    b = rng.integers(-128, 128, (4 * d, d), dtype=np.int8)
    outs = []
    for dyn in (False, True):
        sim, _ = fresh()
        sim.ddr_write(BASE + 0x30000, a.tobytes())
        sim.ddr_write(BASE + 0x38000, b.tobytes())
        dl = DL().ld(BASE + 0x30000, SA, d, 4 * d, 4 * d, pm.LD_INTERLEAVE) \
                 .ld(BASE + 0x38000, pm.laddr(pm.MEM_SPAD_B, 0), 4 * d, d, d)
        dl.ex(0, 0, 0, 1 if dyn else 3, dyn=[("kt", 0)] if dyn else ())
        dl.st(BASE + 0x40000, pm.laddr(pm.MEM_ACC, 0), d, 4 * d, 4 * d).end()
        run(sim, dl, params=[3])
        outs.append(sim.ddr_read(BASE + 0x40000, 4 * d * d).view("<i4").reshape(d, d))
    exp = a[:, :3 * d].astype(np.int32) @ b[:3 * d].astype(np.int32)
    check(f"D={d} L1 dynamic EX Kt", np.array_equal(outs[0], exp) and np.array_equal(outs[1], exp))

    # SETREG (replace, add, BASE and PARAM)
    sim, src = fresh()
    dl = DL().setreg((P + 0, 5), (P + 1, 7), (6, BASE + 0x10000 + 256)).setreg((P + 0, 3, True)) \
             .ld(0, SA, 1, 64, 64, base=6).st(BASE + 0x20000, SA, 1, 64, 64).end()
    run(sim, dl)
    check(f"D={d} L1 SETREG replace / add, BASE via SETREG",
          sim.params[:2] == [8, 7] and np.array_equal(sim.ddr_read(BASE + 0x20000, 64), src[256:320]))

    # LOOP_END: 8 blocks copied by a 2-descriptor body with PARAM address strides
    sim, src = fresh()
    dl = DL().ld(BASE + 0x10000, SA, 1, 256, 256, dyn=[("ddr", 0, True)]) \
             .st(BASE + 0x20000, SA, 1, 256, 256, dyn=[("ddr", 1, True)]) \
             .loop_end(-2, 8, k1=0, s1=256, k2=1, s2=256).end(0x77)
    n = run(sim, dl, params=[0, 0])
    check(f"D={d} L1 LOOP_END x8 with PARAM strides",
          np.array_equal(sim.ddr_read(BASE + 0x20000, 2048), src[:2048]) and sim.params[:2] == [2048, 2048]
          and n == 8 * 3 + 1 and sim.dl_status == 0x77)

    # nested loops (3 x 4), count 1, and relative JUMP skipping a descriptor
    sim, _ = fresh()
    dl = DL().setreg((P + 4, 1, True)).loop_end(-1, 4, k1=2, s1=1).loop_end(-2, 3, k1=3, s1=1) \
             .loop_end(-1, 1, k1=5, s1=10).jump(2, rel=True).setreg((P + 6, 99)).end()
    run(sim, dl, params=[0] * 8)
    check(f"D={d} L1 nested loops 3x4, count 1, relative JUMP",
          sim.params[4] == 12 and sim.params[2] == 12 and sim.params[3] == 3 and sim.params[5] == 10
          and sim.params[6] == 0)

    # CALL / RET: a subroutine called twice, and nesting to depth 4
    sim, _ = fresh()
    main_l = DL().call(BASE + 0x3000).call(BASE + 0x3000).call(BASE + 0x3100).end()
    sub = DL().setreg((P + 0, 1, True)).ret()
    chain = DL().call(2, rel=True).ret().call(2, rel=True).ret().call(2, rel=True).ret() \
                .setreg((P + 1, 1, True)).ret()                    # 4 frames deep at the SETREG
    sim.ddr_write(BASE + 0x3000, sub.array().tobytes())
    sim.ddr_write(BASE + 0x3100, chain.array().tobytes())
    run(sim, main_l, params=[0, 0])
    check(f"D={d} L1 CALL / RET (twice, and depth 4)", sim.params[:2] == [2, 1])

    # LDPARAM: gather a row whose index is in DDR
    sim, src = fresh()
    sim.ddr_write(BASE + 0x5004, np.array([37], "<u4").tobytes())
    dl = DL().ldparam(4, 2, mul=64, add=0x10000, base=0) \
             .ld(BASE, SA, 1, 64, 64, dyn=[("ddr", 2, True)]).st(BASE + 0x20000, SA, 1, 64, 64).end()
    run(sim, dl, bases=[BASE + 0x5000])
    check(f"D={d} L1 LDPARAM (x mul + add, relocated) as a gather index",
          sim.params[2] == 37 * 64 + 0x10000 and np.array_equal(sim.ddr_read(BASE + 0x20000, 64), src[37 * 64:38 * 64]))

    # count limit inside a loop
    sim, _ = fresh()
    dl = DL().setreg((P + 0, 1, True)).loop_end(-1, 10).end()
    n = run(sim, dl, count=7, params=[0])
    check(f"D={d} L1 count limit inside a loop", n == 7 and sim.params[0] == 4)

    def expect(name, dl, **kw):
        sim, _ = fresh()
        try:
            run(sim, dl, **kw)
        except SaError as e:
            check(f"D={d} L1 error: {name}", (e.engine, e.code) == (ENG_FETCH, XERR_SHAPE),
                  f"got engine {e.engine} code {e.code}")
            return
        check(f"D={d} L1 error: {name}", False, "no error")

    bad = DL().fence().end()
    bad.rows[0][0] |= 1 << 16                                   # a dynamic field on FENCE
    expect("dynamic field undefined for the opcode", bad)
    expect("LOOP_END count 0", DL().loop_end(-1, 0).end())
    expect("loop stack overflow (3 levels)",
           DL().setreg((P, 1)).loop_end(-1, 2).loop_end(-2, 2).loop_end(-3, 2).end())
    expect("call stack overflow (depth 5)", DL().call(0, rel=True).end())
    expect("RET without CALL", DL().ret().end())
    expect("SETREG register 24", DL().setreg((24, 1)).end())
    expect("LDPARAM not 4-byte aligned", DL().ldparam(BASE + 0x5002, 0).end())


def l2_fp(d, rng):
    import sfu
    DL = pm.DescList
    F32, I32, I8 = 3, 2, 0
    ACC, SA, SB = pm.MEM_ACC, pm.MEM_SPAD_A, pm.MEM_SPAD_B
    X, Y, O, L = BASE, BASE + 0x40000, BASE + 0x80000, 0xC0000
    f32 = np.float32

    def place(sim, t, data, addr, mem, word):
        """write data (flat, element type t) to DDR and a LD into mem/word"""
        arr = {F32: data.astype(f32), I32: data.astype(np.int32), I8: data.astype(np.int8)}[t]
        sim.ddr_write(addr, arr.tobytes())
        return lambda dl: dl.ld(addr, pm.laddr(mem, word), 1, arr.nbytes, arr.nbytes)

    def run(t_in, t_out, x, y, n_out, **ve):
        sim = SaFuncSim(d, BASE, 1 << 21)
        m_in = ACC if t_in in (F32, I32) else SA
        m_in2 = ACC if t_in in (F32, I32) else SB
        m_out = ACC if t_out in (F32, I32) else SB
        w2 = 1024 if m_in2 == ACC else 0
        w_out = 2048
        dl = DL()
        place(sim, t_in, x, X, m_in, 0)(dl)
        if y is not None:
            place(sim, t_in, y, Y, m_in2, w2)(dl)
        dl.ve(pm.laddr(m_in, 0), pm.laddr(m_in2, w2), pm.laddr(m_out, w_out), ve.pop("length", x.size),
              ve.pop("op", 5), t_in | t_out << 2, fp=True, **ve)
        esz = 1 if t_out == I8 else 4
        dl.st(O, pm.laddr(m_out, w_out), 1, n_out * esz, n_out * esz).end()
        sim.run_list(put_list(sim, dl, L))
        raw = sim.ddr_read(O, n_out * esz)
        return raw.view({F32: f32, I32: np.int32, I8: np.int8}[t_out])

    def same(got, want):
        return np.array_equal(got.view(np.uint32) if got.dtype == f32 else got,
                              np.asarray(want).astype(got.dtype).view(np.uint32) if got.dtype == f32 else want)

    n = 32 * d
    x = rng.standard_normal(n).astype(f32) * 3
    y = rng.standard_normal(n).astype(f32) * 3
    ops = {"add": (0, x + y), "sub": (1, x - y), "mul": (2, x * y), "max": (3, np.maximum(x, y)),
           "min": (4, np.minimum(x, y)), "copy": (5, x)}
    for name, (op, want) in ops.items():
        got = run(F32, F32, x, None if op == 5 else y, n, op=op)
        check(f"D={d} L2 fp {name} F32->F32", same(got, want))
    A, Bc = f32(0.37), f32(-1.25)
    got = run(F32, F32, x, y, n, op=2, A=A, B=Bc)
    check(f"D={d} L2 fp affine (x*y)*A+B", same(got, (x * y) * A + Bc))
    got = run(F32, F32, x, y, n, op=0 | 0x10)
    check(f"D={d} L2 fp ADD + RELU", same(got, np.where(x + y < 0, f32(0), x + y)))
    for fname, fn in (("exp", sfu.exp), ("recip", sfu.recip), ("rsqrt", sfu.rsqrt), ("abs", sfu.fabs)):
        xin = np.abs(x) + f32(0.1) if fname == "rsqrt" else x
        got = run(F32, F32, xin, None, n, op=5, func=fname)
        check(f"D={d} L2 fp FUNC {fname}", np.array_equal(got.view(np.uint32), fn(xin.view(np.uint32))))
    xi = rng.integers(-2**30, 2**30, n)
    got = run(I32, F32, xi, None, n, op=5)
    check(f"D={d} L2 fp I32 -> F32 (RNE)", same(got, xi.astype(f32)))
    xb = rng.integers(-128, 128, n)
    yb = rng.integers(-128, 128, n)
    got = run(I8, F32, xb, yb, n, op=2)
    check(f"D={d} L2 fp I8 x I8 -> F32 (SPAD inputs)", same(got, (xb * yb).astype(f32)))
    got = run(F32, I8, x * 40, None, n, op=5)
    check(f"D={d} L2 fp F32 -> I8 (RNE, +-127)", np.array_equal(got, np.clip(np.rint(x * 40), -127, 127)))
    got = run(F32, I32, x * 1e5, None, n, op=5)
    check(f"D={d} L2 fp F32 -> I32 (RNE)", np.array_equal(got, np.rint(x * 1e5).astype(np.int64)))
    # SWAPNEG (RoPE): (x0, x1) -> (-x1, x0), times src2
    sw = np.empty_like(x)
    sw[0::2], sw[1::2] = -x[1::2], x[0::2]
    got = run(F32, F32, x, y, n, op=2, swapneg=True)
    check(f"D={d} L2 fp SWAPNEG", same(got, sw * y))
    # index modes: src2 MOD 4 groups, DIV 8 groups, IMM; src1 DIV D (replicate)
    g = np.arange(n // d)
    ymod = y[: 4 * d].reshape(4, d)[g % 4].ravel()
    got = run(F32, F32, x, y[: 4 * d], n, op=2, m2="mod", period=4)
    check(f"D={d} L2 fp src2 MOD 4", same(got, x * ymod))
    ydiv = y[: (n // d // 8) * d].reshape(-1, d)[g // 8].ravel()
    got = run(F32, F32, x, y[: (n // d // 8) * d], n, op=2, m2="div", period=8)
    check(f"D={d} L2 fp src2 DIV 8", same(got, x * ydiv))
    got = run(F32, F32, x, None, n, op=1, m2="imm", imm=2.5)
    check(f"D={d} L2 fp src2 IMM", same(got, x - f32(2.5)))
    got = run(F32, F32, x[: 4 * d], None, 4 * d * d, op=5, m1="div", p1=d, length=4 * d * d)
    check(f"D={d} L2 fp src1 DIV D (replicated rows)",
          same(got, x[: 4 * d].reshape(4, d)[np.arange(4 * d) // d].ravel()))
    # VALID: rows of 4 groups, first 3*d - 1 elements valid, others written as 0
    got = run(F32, F32, x, None, n, op=5, rowlen=4, valid=3 * d - 1)
    e = np.arange(n) % (4 * d)
    check(f"D={d} L2 fp VALID masking", same(got, np.where(e < 3 * d - 1, x, f32(0))))

    # reductions: order = per lane sequentially over the groups of a row, then a pairwise lane tree
    def ref_reduce(v, rowlen, valid, kind):
        ident = f32(0) if kind == "sum" else f32(-np.inf)
        fn = (lambda a, b: (a + b).astype(f32)) if kind == "sum" else np.maximum
        v = v.reshape(-1, rowlen, d).copy()
        if valid:
            ee = np.arange(rowlen)[:, None] * d + np.arange(d)[None, :]
            v = np.where(ee[None] < valid, v, ident)
        acc = v[:, 0, :]
        for gi in range(1, rowlen):
            acc = fn(acc, v[:, gi, :])
        while acc.shape[1] > 1:
            acc = fn(acc[:, 0::2], acc[:, 1::2])
        return np.repeat(acc, d, axis=1).ravel()

    for kind in ("sum", "max"):
        for rowlen, valid in ((32, 0), (8, 0), (8, 5 * d + 3)):
            got = run(F32, F32, x, None, (n // d // rowlen) * d, op=5, reduce=kind, rowlen=rowlen, valid=valid)
            check(f"D={d} L2 fp REDUCE {kind} rowlen {rowlen} valid {valid}",
                  same(got, ref_reduce(x, rowlen, valid, kind)))
    got = run(F32, F32, x, x, (n // d // 8) * d, op=2, reduce="sum", rowlen=8)
    check(f"D={d} L2 fp REDUCE sum of x*x", same(got, ref_reduce((x * x).astype(f32), 8, 0, "sum")))

    # TRANSPOSE: rows of S words (K = pos x hs, S = hs / d) -> B strips
    for t, mem in ((I8, SA), (F32, ACC)):
        R, S = 3 * d, 3
        sim = SaFuncSim(d, BASE, 1 << 21)
        data = (rng.integers(-128, 128, (R, S * d)) if t == I8 else rng.standard_normal((R, S * d))).astype(
            np.int8 if t == I8 else f32)
        sim.ddr_write(X, data.tobytes())
        dst = SB if t == I8 else ACC
        wd = 0 if t == I8 else 2048
        dl = DL().ld(X, pm.laddr(mem, 0), 1, data.nbytes, data.nbytes)
        dl.transpose(pm.laddr(mem, 0), pm.laddr(dst, wd), R * S * d, t | t << 2, S)
        dl.st(O, pm.laddr(dst, wd), 1, data.nbytes, data.nbytes).end()
        sim.run_list(put_list(sim, dl, L))
        got = sim.ddr_read(O, data.nbytes).view(data.dtype).reshape(-1, d)
        # block (kb, ks): positions kb*d.., dims ks*d..; output word kb*S*d + ks*d + i = data[kb*d:(kb+1)*d, ks*d + i]
        want = np.concatenate([data[kb * d:(kb + 1) * d, ks * d + i][None]
                               for kb in range(R // d) for ks in range(S) for i in range(d)])
        check(f"D={d} L2 TRANSPOSE {'I8 SPAD' if t == I8 else 'F32 ACC'} ({R} x {S} words)",
              np.array_equal(got, want))

    # softmax of rows of 4*d with VALID = 3*d + 1: max, exp(x - max), sum, recip, multiply
    rows, rl = 4, 4
    xs = (rng.standard_normal(rows * rl * d) * 4).astype(f32)
    valid = 3 * d + 1
    sim = SaFuncSim(d, BASE, 1 << 21)
    sim.ddr_write(X, xs.tobytes())
    a0, m, e, s_, r = 0, 1024, 2048, 3072, 3200
    la = lambda w: pm.laddr(ACC, w)
    dl = DL().ld(X, la(a0), 1, xs.nbytes, xs.nbytes)
    dl.ve(la(a0), 0, la(m), xs.size, 5, F32 | F32 << 2, fp=True, reduce="max", rowlen=rl, valid=valid)
    dl.ve(la(a0), la(m), la(e), xs.size, 1, F32 | F32 << 2, fp=True, m2="div", period=rl, func="exp",
          rowlen=rl, valid=valid)
    dl.ve(la(e), 0, la(s_), xs.size, 5, F32 | F32 << 2, fp=True, reduce="sum", rowlen=rl)
    dl.ve(la(s_), 0, la(r), rows * d, 5, F32 | F32 << 2, fp=True, func="recip")
    dl.ve(la(e), la(r), la(e), xs.size, 2, F32 | F32 << 2, fp=True, m2="div", period=rl)
    dl.st(O, la(e), 1, xs.nbytes, xs.nbytes).end()
    sim.run_list(put_list(sim, dl, L))
    got = sim.ddr_read(O, xs.nbytes).view(f32).reshape(rows, -1).astype(np.float64)
    xv = xs.reshape(rows, -1).astype(np.float64)[:, :valid]
    want = np.exp(xv - xv.max(1, keepdims=True))
    want /= want.sum(1, keepdims=True)
    check(f"D={d} L2 softmax as 5 VE commands (VALID {valid} of {rl * d}): max error "
          f"{np.abs(got[:, :valid] - want).max():.1e}",
          np.abs(got[:, :valid] - want).max() < 1e-5 and not got[:, valid:].any())
    # RMSNorm: sum(x*x), rsqrt(ss / n + eps), x * r, * g
    nn = 8 * d
    xr = rng.standard_normal(nn).astype(f32)
    gw = rng.uniform(0.5, 1.5, nn).astype(f32)
    sim = SaFuncSim(d, BASE, 1 << 21)
    sim.ddr_write(X, xr.tobytes())
    sim.ddr_write(Y, gw.tobytes())
    dl = DL().ld(X, la(0), 1, xr.nbytes, xr.nbytes).ld(Y, la(512), 1, gw.nbytes, gw.nbytes)
    dl.ve(la(0), la(0), la(1024), nn, 2, F32 | F32 << 2, fp=True, reduce="sum", rowlen=nn // d)
    dl.ve(la(1024), 0, la(1025), d, 5, F32 | F32 << 2, fp=True, func="rsqrt", A=1.0 / nn, B=1e-5)
    dl.ve(la(0), la(1025), la(1100), nn, 2, F32 | F32 << 2, fp=True, m2="div", period=nn // d)
    dl.ve(la(1100), la(512), la(1100), nn, 2, F32 | F32 << 2, fp=True)
    dl.st(O, la(1100), 1, xr.nbytes, xr.nbytes).end()
    sim.run_list(put_list(sim, dl, L))
    got = sim.ddr_read(O, xr.nbytes).view(f32).astype(np.float64)
    want = xr.astype(np.float64) / np.sqrt((xr.astype(np.float64) ** 2).mean() + 1e-5) * gw
    check(f"D={d} L2 RMSNorm as 4 VE commands: max rel error {np.abs(got / want - 1).max():.1e}",
          np.abs(got / want - 1).max() < 2e-6)

    def expect(name, fn):
        try:
            fn()
        except SaError as e_:
            check(f"D={d} L2 error: {name}", e_.engine == ENG_VE, f"engine {e_.engine}")
            return
        check(f"D={d} L2 error: {name}", False, "no error")

    sim = SaFuncSim(d, BASE, 1 << 20)
    expect("F32 in SPAD", lambda: sim.ve(pm.laddr(SA, 0), 0, pm.laddr(ACC, 0), 1, 5, F32 | F32 << 2, flags=1))
    expect("REQUANT with FP", lambda: sim.ve(pm.laddr(ACC, 0), 0, pm.laddr(ACC, 8), 1, 0x25, F32 | F32 << 2, flags=1))
    expect("REDUCE to I8", lambda: sim.ve(pm.laddr(ACC, 0), 0, pm.laddr(SB, 0), 1, 5, F32, flags=1 | 1 << 8))
    expect("DIV with period 0", lambda: sim.ve(pm.laddr(ACC, 0), pm.laddr(ACC, 8), pm.laddr(ACC, 16), 1, 0,
                                               F32 | F32 << 2, flags=1 | 2 << 6))
    expect("L2 field without FP", lambda: sim.ve(pm.laddr(ACC, 0), 0, pm.laddr(ACC, 8), 1, 5, 2 | 2 << 2, a=1))
    expect("TRANSPOSE with stride 0", lambda: sim.ve(pm.laddr(ACC, 0), 0, pm.laddr(ACC, 64), d, 6, F32 | F32 << 2))


def main():
    rng = np.random.default_rng(2026)
    for d in (8, 16):
        print(f"== D = {d}")
        cosim_cases(d)
        random_gemms(d, rng, 40)
        random_vectors(d, rng, 60)
        control_flow(d)
        errors(d)
        l1_extensions(d)
        l2_fp(d, rng)
    print("PASS" if not fails else f"FAIL: {len(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())

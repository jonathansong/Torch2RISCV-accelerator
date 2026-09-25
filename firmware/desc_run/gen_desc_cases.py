#!/usr/bin/env python3
"""Co-simulation cases for the descriptor-list path (firmware/sim/tb_system.v, DESC_TEST).

The lists are built by the real driver code (driver/pynq_matmul.py:
build_gemm_list / build_vector_list, DescList) with the simulated DDR
addresses, and the expected outputs come from the driver's NumPy goldens. The
testbench loads each case into its DDR model, runs desc_run_fw on PicoRV32 +
the accelerator RTL and compares the output bytes, so the Python descriptor
encoding and schedules are checked against the RTL before they reach the board.

Writes into --out: case<i>_ddr.hex (sparse, @byte-offset lines), case<i>_exp.hex
(expected output bytes) and desc_cases.vh (case count, parameters, a load task).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "driver"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "llm"))
import pynq_matmul as pm  # noqa: E402  (pynq itself is optional off the board)
from sa_funcsim import SaError, SaFuncSim  # noqa: E402

DDR_BASE = 0x18000000                  # tb_system.v
A_OFF, B_OFF, C_OFF, BIAS_OFF = 0x0000, 0x4000, 0x8000, 0x10000
X_OFF, Y_OFF, O_OFF, LIST_OFF = 0x18000, 0x20000, 0x28000, 0x40000
IDX_OFF = 0x30000
DDR_BYTES = 0x50000


def write_sparse(path, chunks):
    """chunks: [(byte offset, bytes)] -> $readmemh file with @offset lines."""
    with open(path, "w") as f:
        for off, data in chunks:
            f.write(f"@{off:x}\n")
            f.write("\n".join(f"{b:02x}" for b in data) + "\n")


def build_cases(d, seed=11):
    """[(name, DescList, [(DDR offset, bytes)], output offset, expected bytes,
    descriptors decoded)] for array size d; DDR offsets are relative to
    DDR_BASE. The last two cases use the L1 command extensions (loops,
    dynamic fields, SETREG, CALL / RET, LDPARAM); expected bytes always come
    from NumPy, independently of the lists."""
    rng = np.random.default_rng(seed)
    cases = []   # (name, dl, chunks, out_off, expected bytes)

    def gemm_case(name, m, n, k, bias=False, quant=None, relu=False, bsplit="auto"):
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b = rng.integers(-128, 128, (k, n), dtype=np.int8)
        chunks = [(A_OFF, a.tobytes()), (B_OFF, b.tobytes())]
        bias_v = None
        if bias:
            bias_v = rng.integers(-2**16, 2**16, n if quant else (m, n), dtype=np.int32)
            chunks.append((BIAS_OFF, bias_v.tobytes()))
        dl = pm.build_gemm_list(d, m, n, k, DDR_BASE + A_OFF, DDR_BASE + B_OFF, DDR_BASE + C_OFF,
                                DDR_BASE + BIAS_OFF if bias else None, quant, relu, bsplit=bsplit)
        if quant is not None:
            exp = pm.qgemm_golden(a, b, bias_v, quant, relu)
        else:
            exp = (pm.golden(a, b) + (bias_v if bias else 0)).astype(np.int32)
        cases.append((name, dl, chunks, C_OFF, exp.tobytes(), len(dl)))

    def vector_case(name, op, it, ot, n, ny, relu=False, requant=None):
        dt = {0: np.int8, 1: np.int16, 2: np.int32}
        info = np.iinfo(dt[it])
        x = rng.integers(info.min, info.max + 1, n, dtype=np.int64).astype(dt[it])
        y = rng.integers(info.min, info.max + 1, ny, dtype=np.int64).astype(dt[it])
        dl = pm.build_vector_list(d, op, it, ot, n, ny, DDR_BASE + X_OFF, DDR_BASE + Y_OFF,
                                  DDR_BASE + O_OFF, relu, requant)
        exp = pm.vector_golden(op, x, y, dt[ot], relu, requant)
        cases.append((name, dl, [(X_OFF, x.tobytes()), (Y_OFF, y.tobytes())], O_OFF, exp.tobytes(), len(dl)))

    gemm_case(f"gemm {4*d}x{4*d}x{8*d}", 4 * d, 4 * d, 8 * d)
    gemm_case(f"gemm {3*d}x{2*d}x{5*d} + bias rows", 3 * d, 2 * d, 5 * d, bias=True)
    gemm_case(f"gemm {2*d}x{8*d}x128 (B split)", 2 * d, 8 * d, 128, bsplit=True)
    gemm_case(f"int8 gemm {4*d}x{4*d}x{8*d} + bias, relu", 4 * d, 4 * d, 8 * d, bias=True,
              quant=pm.Requant(181, 15, -3), relu=True)
    vector_case("vector add i8 n=20000", "add", 0, 0, 20000, 20000)
    vector_case("vector add i32->i8 relu requant, bias period 8", "add", 2, 0, 4096, 8 * d, True,
                pm.Requant(300, 16, 5))
    vector_case("vector mul i16->i32 n=4112", "mul", 1, 2, 4112, 4112)

    # ---- L1: int8 add of 8 chunks of 512 by one loop body (PARAM address strides)
    P = pm.DescList.REG_PARAM
    x = rng.integers(-128, 128, 4096, dtype=np.int8)
    y = rng.integers(-128, 128, 4096, dtype=np.int8)
    sa, sb, so = pm.laddr(pm.MEM_SPAD_A, 0), pm.laddr(pm.MEM_SPAD_B, 0), pm.laddr(pm.MEM_SPAD_B, 1024)
    dl = pm.DescList().setreg((P + 0, 0), (P + 1, 0))
    dl.ld(DDR_BASE + X_OFF, sa, 1, 512, 512, dyn=[("ddr", 0, True)])
    dl.ld(DDR_BASE + Y_OFF, sb, 1, 512, 512, dyn=[("ddr", 0, True)])
    dl.ve(sa, sb, so, 512, pm.VOPS["add"], 0)
    dl.st(DDR_BASE + O_OFF, so, 1, 512, 512, dyn=[("ddr", 1, True)])
    dl.loop_end(-4, 8, k1=0, s1=512, k2=1, s2=512).end(0x11)
    exp = np.clip(x.astype(int) + y, -128, 127).astype(np.int8)
    cases.append(("L1 loop: int8 add, 8 x 512 through PARAM strides", dl,
                  [(X_OFF, x.tobytes()), (Y_OFF, y.tobytes())], O_OFF, exp.tobytes(), 1 + 8 * 5 + 1))

    # ---- L1: gather 4 rows by indices in DDR (LDPARAM), a CALLed body, SETREG add
    rows = rng.integers(0, 256, (32, 64), dtype=np.uint8)
    idx = np.array([5, 0, 17, 3], "<u4")
    dl = pm.DescList().setreg((P + 3, DDR_BASE + IDX_OFF), (P + 4, DDR_BASE + O_OFF))
    dl.ldparam(0, 2, mul=64, add=DDR_BASE + X_OFF, dyn=[("addr", 3)])      # 1: PARAM2 = row address
    dl.call(3, rel=True)                                                  # 2: -> 5
    dl.loop_end(-2, 4, k1=3, s1=4, k2=3, s2=0)                            # 3: next index
    dl.end(0x22)                                                          # 4
    dl.ld(0, sa, 1, 64, 64, dyn=[("ddr", 2)])                             # 5
    dl.st(0, sa, 1, 64, 64, dyn=[("ddr", 4)])                             # 6
    dl.setreg((P + 4, 64, True))                                          # 7
    dl.ret()                                                              # 8
    cases.append(("L1 LDPARAM gather + CALL / RET + SETREG add", dl,
                  [(X_OFF, rows.tobytes()), (IDX_OFF, idx.tobytes())], O_OFF, rows[idx].tobytes(), 1 + 4 * 7 + 1))

    # ---- L2 (fp32 VE, TRANSPOSE): expected = functional simulator, sanity-checked against float64
    F32, I32, I8 = 3, 2, 0
    ACC = pm.MEM_ACC
    la = lambda m, w: pm.laddr(m, w)

    def l2_case(name, dl, chunks, out_off, nbytes, check):
        sim = SaFuncSim(d, DDR_BASE, DDR_BYTES)
        for off, data in chunks:
            sim.ddr_write(DDR_BASE + off, data)
        sim.ddr_write(DDR_BASE + LIST_OFF, dl.array().tobytes())
        n = sim.run_list(DDR_BASE + LIST_OFF)
        out = sim.ddr_read(DDR_BASE + out_off, nbytes).tobytes()
        assert check(out), f"{name}: functional simulator far from the float64 reference"
        cases.append((name, dl, chunks, out_off, out, n))

    f32 = np.float32
    # softmax of 4 rows of 4d elements, the first 3d + 1 valid
    rows, rl, valid = 4, 4, 3 * d + 1
    xs = (rng.standard_normal(rows * rl * d) * 3).astype(f32)
    dl = pm.DescList().ld(DDR_BASE + X_OFF, la(ACC, 0), 1, xs.nbytes, xs.nbytes)
    T = F32 | F32 << 2
    dl.ve(la(ACC, 0), 0, la(ACC, 256), xs.size, 5, T, fp=True, reduce="max", rowlen=rl, valid=valid)
    dl.ve(la(ACC, 0), la(ACC, 256), la(ACC, 512), xs.size, 1, T, fp=True, m2="div", period=rl, func="exp",
          rowlen=rl, valid=valid)
    dl.ve(la(ACC, 512), 0, la(ACC, 768), xs.size, 5, T, fp=True, reduce="sum", rowlen=rl)
    dl.ve(la(ACC, 768), 0, la(ACC, 800), rows * d, 5, T, fp=True, func="recip")
    dl.ve(la(ACC, 512), la(ACC, 800), la(ACC, 512), xs.size, 2, T, fp=True, m2="div", period=rl)
    dl.st(DDR_BASE + O_OFF, la(ACC, 512), 1, xs.nbytes, xs.nbytes).end(0x51)

    def sm_ok(out):
        got = np.frombuffer(out, f32).reshape(rows, -1).astype(np.float64)
        xv = xs.reshape(rows, -1).astype(np.float64)[:, :valid]
        want = np.exp(xv - xv.max(1, keepdims=True))
        want /= want.sum(1, keepdims=True)
        return np.abs(got[:, :valid] - want).max() < 1e-5 and not got[:, valid:].any()
    l2_case("L2 softmax (4 rows, VALID): REDUCE MAX, EXP, REDUCE SUM, RECIP, MUL", dl,
            [(X_OFF, xs.tobytes())], O_OFF, xs.nbytes, sm_ok)

    # int8 GEMM (d x 4d x 2d) + fp32 dequantization: acc * s_w (per column, MOD) * s_x (IMM)
    m, k, n = d, 4 * d, 2 * d
    a = rng.integers(-127, 128, (m, k), dtype=np.int8)
    b = rng.integers(-127, 128, (k, n), dtype=np.int8)
    sw = rng.uniform(0.001, 0.01, n).astype(f32)
    sx = f32(0.05)
    dl = pm.build_gemm_list(d, m, n, k, DDR_BASE + A_OFF, DDR_BASE + B_OFF, DDR_BASE + C_OFF)
    dl.rows.pop()                                                   # drop its END, append the epilogue
    dl.ld(DDR_BASE + BIAS_OFF, la(ACC, 1000), 1, sw.nbytes, sw.nbytes, fence_before=True)
    dl.ld(DDR_BASE + C_OFF, la(ACC, 1100), 1, 4 * m * n, 4 * m * n)
    dl.ve(la(ACC, 1100), la(ACC, 1000), la(ACC, 1100), m * n, 2, I32 | F32 << 2, fp=True, m2="mod", t2=F32,
          period=n // d, A=float(sx))
    dl.st(DDR_BASE + O_OFF, la(ACC, 1100), 1, 4 * m * n, 4 * m * n).end(0x52)

    def dq_ok(out):
        got = np.frombuffer(out, f32).reshape(m, n).astype(np.float64)
        want = (a.astype(np.int64) @ b.astype(np.int64)) * sw.astype(np.float64) * float(sx)
        return np.allclose(got, want, rtol=1e-6, atol=0)
    l2_case("L2 int8 GEMM + fp32 dequant (I32 -> F32, x s_w MOD, x s_x)", dl,
            [(A_OFF, a.tobytes()), (B_OFF, b.tobytes()), (BIAS_OFF, sw.tobytes())], O_OFF, 4 * m * n, dq_ok)

    # RMSNorm + dynamic int8 quantization, written as D replicated rows (A strip)
    nn = 4 * d
    xr = rng.standard_normal(nn).astype(f32)
    gw = rng.uniform(0.5, 1.5, nn).astype(f32)
    SPA = pm.MEM_SPAD_A
    dl = pm.DescList().ld(DDR_BASE + X_OFF, la(ACC, 0), 1, xr.nbytes, xr.nbytes)
    dl.ld(DDR_BASE + Y_OFF, la(ACC, 64), 1, gw.nbytes, gw.nbytes)
    dl.ve(la(ACC, 0), la(ACC, 0), la(ACC, 128), nn, 2, T, fp=True, reduce="sum")            # sum x^2
    dl.ve(la(ACC, 128), 0, la(ACC, 129), d, 5, T, fp=True, func="rsqrt", A=1.0 / nn, B=1e-5)
    dl.ve(la(ACC, 0), la(ACC, 129), la(ACC, 136), nn, 2, T, fp=True, m2="div", period=nn // d)
    dl.ve(la(ACC, 136), la(ACC, 64), la(ACC, 136), nn, 2, T, fp=True)                        # * g
    dl.ve(la(ACC, 136), 0, la(ACC, 200), nn, 5, T, fp=True, func="abs", reduce="max")         # amax
    dl.ve(la(ACC, 200), 0, la(ACC, 201), d, 5, T, fp=True, func="recip", A=1.0 / 127)         # 127/amax
    dl.ve(la(ACC, 136), la(ACC, 201), la(ACC, 210), nn, 2, T, fp=True, m2="div", period=nn // d)
    dl.ve(la(ACC, 210), 0, la(SPA, 0), nn * d, 5, F32 | I8 << 2, fp=True, m1="div", p1=d)    # replicate rows
    dl.st(DDR_BASE + O_OFF, la(SPA, 0), 1, nn * d, nn * d).end(0x53)

    def rq_ok(out):
        got = np.frombuffer(out, np.int8).reshape(nn // d, d, d)                 # word w*D + g = x chunk w
        rows_ok = all(np.array_equal(got[:, g, :], got[:, 0, :]) for g in range(d))
        xn = xr.astype(np.float64) / np.sqrt((xr.astype(np.float64) ** 2).mean() + 1e-5) * gw
        want = np.clip(np.rint(xn * 127 / np.abs(xn).max()), -127, 127)
        return rows_ok and np.abs(got[:, 0, :].ravel() - want).max() <= 1
    l2_case("L2 RMSNorm + dynamic int8 quantization into replicated A-strip rows", dl,
            [(X_OFF, xr.tobytes()), (Y_OFF, gw.tobytes())], O_OFF, nn * d, rq_ok)

    # attention scores: K (2d positions x hs = 2d, int8) -> TRANSPOSE -> B strips; q replicated; EX; x scale
    pos, hs = 2 * d, 2 * d
    K = rng.integers(-127, 128, (pos, hs), dtype=np.int8)
    q = rng.integers(-127, 128, hs, dtype=np.int8)
    S_ = hs // d
    SPB = pm.MEM_SPAD_B
    dl = pm.DescList().ld(DDR_BASE + A_OFF, la(SPA, 0), 1, K.nbytes, K.nbytes)             # K rows, LINEAR
    dl.ld(DDR_BASE + B_OFF, la(SPA, 512), 1, hs, hs)                                        # q
    dl.transpose(la(SPA, 0), la(SPB, 0), pos * hs, I8 | I8 << 2, S_)                         # Kt strips
    dl.ve(la(SPA, 512), 0, la(SPA, 600), hs * d, 5, I8 | I8 << 2, fp=True, m1="div", p1=d)   # q rows
    dl.ex(600, 0, 0, hs // d, repeat=pos // d, bstep=hs, cstep=1, crow=pos // d)
    dl.ve(la(ACC, 0), 0, la(ACC, 0), pos * d, 5, I32 | F32 << 2, fp=True, A=0.125)           # scores / sqrt(64)
    dl.st(DDR_BASE + O_OFF, la(ACC, 0), 1, 4 * pos, 4 * pos).end(0x54)                        # row 0 = the scores

    def at_ok(out):
        got = np.frombuffer(out, f32).astype(np.float64)
        want = (K.astype(np.int64) @ q.astype(np.int64)) * 0.125
        return np.array_equal(got, want)
    l2_case("L2 attention scores: TRANSPOSE K, replicate q, EX, I32 -> F32 x 1/8", dl,
            [(A_OFF, K.tobytes()), (B_OFF, q.tobytes())], O_OFF, 4 * pos, at_ok)

    # ---- L3: a W8A8 linear layer as compile_layer builds it (quantize x, chunked GEMM over
    # double-buffered weight chunks in a LOOP_END with PARAM-stepped DDR addresses, fp32 dequant)
    import compile_layer as CL
    from export_w8a8 import pack_b
    from ref_model import quantize_rows
    k, n_out = 64, 2880
    wf = rng.standard_normal((n_out, k)).astype(f32) * f32(0.05)
    wq, sw = quantize_rows(wf)
    xl = (rng.standard_normal(k) * 2).astype(f32)
    W_OFF, SW_OFF, XL_OFF, YL_OFF = 0x0, 0x2D000, 0x30000, 0x31000
    lay = CL.Layout(d)
    dl = pm.DescList().ld(DDR_BASE + XL_OFF, la(ACC, lay.acc0), 1, 4 * k, 4 * k)
    s_x = CL.quant_act(dl, lay, lay.acc0, k, lay.acc0 + 128)
    CL.linear(dl, lay, k, n_out, DDR_BASE + W_OFF, DDR_BASE + SW_OFF, s_x, out_ddr=DDR_BASE + YL_OFF, loop=True)
    dl.end(0x55)

    def lin_ok(out):
        got = np.frombuffer(out, f32).astype(np.float64)
        s = np.abs(xl).max() / 127
        want = (wq.astype(np.float64) @ np.clip(np.rint(xl / s), -127, 127)) * sw * s
        return np.allclose(got, want, rtol=1e-5, atol=1e-6)
    l2_case(f"L3 linear {k} -> {n_out}: quantize, looped double-buffered GEMM, dequant", dl,
            [(W_OFF, pack_b(wq, d).tobytes()), (SW_OFF, sw.tobytes()), (XL_OFF, xl.tobytes())],
            YL_OFF, 4 * n_out, lin_ok)

    return cases


FP_OPS = ["add", "sub", "mul", "max", "min", "copy"]
FP_FUNCS = ["none", "exp", "recip", "rsqrt", "abs"]


def rand_f32(rng, n):
    """Mostly N(0, s) with random scales, plus special values."""
    x = (rng.standard_normal(n) * 10.0 ** rng.integers(-3, 4, n)).astype(np.float32).view(np.uint32)
    special = np.array([0x7FC00000, 0x7F800001, 0x7F800000, 0xFF800000, 0x00000000, 0x80000000,
                        0x00000001, 0x807FFFFF, 0x7F7FFFFF, 0xFF7FFFFF, 0x00800000, 0x3F800000], np.uint32)
    pick = rng.random(n) < 0.05
    x[pick] = rng.choice(special, pick.sum())
    return x


def random_fp_list(rng, d):
    """One random fp32 VE command between an LD and an ST; returns (dl, DDR chunks,
    output bytes, description). The simulator may reject it (random_fp_case redraws)."""
    G = int(rng.integers(1, 9)) * 8
    n = G * d
    x1, x2 = rand_f32(rng, n), rand_f32(rng, n)
    it = int(rng.choice([3, 3, 2]))
    if it == 2:
        x1 = rng.integers(-2**31, 2**31, n, dtype=np.int64).astype(np.int32).view(np.uint32)
    op = int(rng.integers(0, 6))
    func = FP_FUNCS[int(rng.integers(0, 5))]
    m2 = ["lin", "mod", "div", "imm"][int(rng.integers(0, 4))]
    red = ["none", "none", "sum", "max"][int(rng.integers(0, 4))]
    rowlen = int(rng.choice([0, 1, 2, 4, 8])) if red != "none" or rng.random() < 0.3 else 0
    if rowlen and G % rowlen:
        rowlen = 0
    ot = 3 if red != "none" else int(rng.choice([3, 3, 2]))
    period = int(rng.choice([1, 2, 4])) if m2 in ("mod", "div") else 0
    kw = dict(fp=True, func=func, m2=m2, reduce=red, rowlen=rowlen, period=period,
              swapneg=bool(rng.random() < 0.2),
              imm=float(np.float32(rng.standard_normal())), A=float(np.float32(rng.choice([1.0, 0.5, -3.0]))),
              B=float(np.float32(rng.choice([0.0, 1e-5, 2.0]))))
    if rowlen and red == "none" and rng.random() < 0.5:
        kw["valid"] = int(rng.integers(1, rowlen * d + 1))
    relu = 0x10 if ot == 3 and rng.random() < 0.2 else 0
    acc = lambda w: pm.laddr(pm.MEM_ACC, w)
    dl = pm.DescList().ld(DDR_BASE + X_OFF, acc(0), 1, 4 * n, 4 * n)
    dl.ld(DDR_BASE + Y_OFF, acc(1024), 1, 4 * n, 4 * n)
    dl.ve(acc(0), acc(1024), acc(2048), n, op | relu, it | ot << 2, **kw)
    # a reduction writes one word per row: store only those (ACC is not cleared between lists)
    nout = 4 * d * (G // (rowlen or G)) if red != "none" else 4 * n
    dl.st(DDR_BASE + O_OFF, acc(2048), 1, nout, nout).end(0x60)
    desc = f"G={G} op={FP_OPS[op]} it={it} ot={ot} {kw}"
    return dl, [(X_OFF, x1.tobytes()), (Y_OFF, x2.tobytes())], nout, desc


def random_fp_case(rng, d):
    """A random fp32 command list as a case tuple (expected = functional simulator);
    lists the simulator rejects are redrawn."""
    while True:
        dl, chunks, nbytes, desc = random_fp_list(rng, d)
        sim = SaFuncSim(d, DDR_BASE, DDR_BYTES)
        for off, data in chunks:
            sim.ddr_write(DDR_BASE + off, data)
        sim.ddr_write(DDR_BASE + LIST_OFF, dl.array().tobytes())
        try:
            n = sim.run_list(DDR_BASE + LIST_OFF)
        except SaError:
            continue
        return (f"L2 random fp32: {desc}"[:150], dl, chunks, O_OFF, sim.ddr_read(DDR_BASE + O_OFF, nbytes).tobytes(), n)


def check_funcsim(d, case):
    """The functional simulator gives the expected bytes and decode count."""
    name, dl, chunks, out_off, exp, n_exec = case
    sim = SaFuncSim(d, DDR_BASE, DDR_BYTES)
    for off, data in chunks:
        sim.ddr_write(DDR_BASE + off, data)
    sim.ddr_write(DDR_BASE + LIST_OFF, dl.array().tobytes())
    n = sim.run_list(DDR_BASE + LIST_OFF)
    assert n == n_exec and sim.ddr_read(DDR_BASE + out_off, len(exp)).tobytes() == exp, f"funcsim: {name}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--out", default=".")
    ap.add_argument("--random", type=int, default=12, help="random fp32 lists appended")
    args = ap.parse_args()
    d = args.d
    cases = build_cases(d)
    rng = np.random.default_rng(23)
    cases += [random_fp_case(rng, d) for _ in range(args.random)]

    os.makedirs(args.out, exist_ok=True)
    vh = [f"// generated by {os.path.basename(__file__)} --d {d}",
          f"localparam integer NDCASE = {len(cases)};"]
    load = ["task load_dcase(input integer c, output [31:0] n_desc, output [31:0] out_off, output [31:0] out_bytes);",
            "    case (c)"]
    for i, case in enumerate(cases):
        check_funcsim(d, case)
        name, dl, chunks, out_off, exp, n_exec = case
        rows = dl.array()
        chunks = chunks + [(LIST_OFF, rows.tobytes())]
        assert all(off + len(data) <= DDR_BYTES for off, data in chunks), name
        assert O_OFF + len(exp) <= LIST_OFF or out_off == C_OFF, name
        write_sparse(os.path.join(args.out, f"case{i}_ddr.hex"), chunks)
        write_sparse(os.path.join(args.out, f"case{i}_exp.hex"), [(0, exp)])
        load.append(f'        {i}: begin $readmemh("case{i}_ddr.hex", ddr); $readmemh("case{i}_exp.hex", dexp); '
                    f'n_desc = {n_exec}; out_off = 32\'h{out_off:x}; out_bytes = {len(exp)}; '
                    f'$display("TB   case {i}: {name}"); end')
    load += ["    endcase", "endtask"]
    with open(os.path.join(args.out, "desc_cases.vh"), "w") as f:
        f.write("\n".join(vh + load) + "\n")
    print(f"{len(cases)} cases for D = {d} in {args.out}")


if __name__ == "__main__":
    main()

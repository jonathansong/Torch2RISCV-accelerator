#!/usr/bin/env python3
"""Cases for tb_sa_vefp: random memory images (words 0 .. NW-1 of SPAD_A,
SPAD_B, ACC; ACC mixes normal fp32 values, integers and specials: NaN,
+-inf, +-0, subnormals), one fp32-VE / TRANSPOSE command each, and the
memories after it from the functional simulator (llm/sa_funcsim.py).
Writes vc<i>_{a,b,c}.hex (initial), vc<i>_{A,B,C}.hex (expected) and
vefp_cases.vh (count, NW, a task with the command fields)."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "llm"))
from sa_funcsim import MEM_ACC, MEM_SPAD_A, MEM_SPAD_B, SaFuncSim  # noqa: E402

NW = 1024
F32, I32, I8 = 3, 2, 0
FUNC = {"none": 0, "exp": 1, "recip": 2, "rsqrt": 3, "abs": 4}
IDX = {"lin": 0, "mod": 1, "div": 2, "imm": 3}
RED = {"none": 0, "sum": 1, "max": 2}


def f32b(x):
    return int(np.float32(x).view(np.uint32))


def acc_image(rng, d, kind):
    n = NW * d
    if kind == "int":
        return rng.integers(-2**31, 2**31, n, dtype=np.int64).astype(np.uint32)
    v = (rng.standard_normal(n) * np.exp(rng.uniform(-6, 6, n))).astype(np.float32).view(np.uint32)
    if kind == "exp":                                     # inputs for EXP: -110 .. 95
        v = rng.uniform(-110, 95, n).astype(np.float32).view(np.uint32)
    sp = np.array([0x7FC00000, 0x7F800001, 0x7F800000, 0xFF800000, 0, 0x80000000, 0x00000005, 0x807FFFFF,
                   0x00800000, 0x42B20000, 0xC2D00000, 0x7F7FFFFF], np.uint32)
    pick = rng.random(n) < 0.04
    v[pick] = rng.choice(sp, pick.sum())
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    d = args.d
    rng = np.random.default_rng(20 + d)
    cases = []   # (name, fields dict, acc kind)

    def c(name, src1, src2, dst, groups, op, it, ot, acc="float", **kw):
        cases.append((name, dict(src1=src1, src2=src2, dst=dst, groups=groups, op=op, types=it | ot << 2, **kw), acc))

    A = lambda w: (MEM_ACC << 28) | w
    SA = lambda w: (MEM_SPAD_A << 28) | w
    SB = lambda w: (MEM_SPAD_B << 28) | w

    def fl(func="none", m1="lin", m2="lin", red="none", swapneg=0):
        return 1 | FUNC[func] << 1 | IDX[m1] << 4 | IDX[m2] << 6 | RED[red] << 8 | swapneg << 10

    one = f32b(1.0)
    for op in range(6):
        c(f"op {op} F32", A(0), A(256), A(512), 64, op, F32, F32, flags=fl(), a=one)
    c("add I32 -> F32", A(0), A(256), A(512), 48, 0, I32, F32, acc="int", flags=fl(), a=one)
    c("mul I8 x I8 -> F32", SA(0), SB(0), A(512), 64, 2, I8, F32, flags=fl(), a=one)
    c("add F32 -> I8 (SPAD_B)", A(0), A(256), SB(512), 64, 0, F32, I8, flags=fl(), a=f32b(20.0))
    c("sub F32 -> I32", A(0), A(256), A(512), 64, 1, F32, I32, flags=fl(), a=f32b(1000.0), b=f32b(0.5))
    c("mul + affine + RELU", A(0), A(256), A(512), 64, 2 | 0x10, F32, F32, flags=fl(), a=f32b(-0.75), b=f32b(3.0))
    for fn in ("exp", "recip", "rsqrt", "abs"):
        c(f"FUNC {fn}", A(0), A(256), A(512), 32, 5, F32, F32, acc="exp" if fn == "exp" else "float",
          flags=fl(fn), a=one)
    c("FUNC exp after affine (x*-1)", A(0), A(0), A(512), 32, 5, F32, F32, acc="exp", flags=fl("exp"), a=f32b(-1.0))
    c("FUNC recip + RELU -> I8", A(0), A(0), SB(600), 16, 5 | 0x10, F32, I8, flags=fl("recip"), a=f32b(0.01))
    c("src1 MOD 3, src2 DIV 5", A(0), A(256), A(512), 60, 0, F32, F32, flags=fl(m1="mod", m2="div"), a=one,
      p1=3, period=5)
    c("src1 DIV D (replicate), src2 MOD 4", A(0), A(256), A(512), 8 * d, 2, F32, F32,
      flags=fl(m1="div", m2="mod"), a=one, p1=d, period=4)
    c("src2 IMM", A(0), A(256), A(512), 64, 1, F32, F32, flags=fl(m2="imm"), imm=f32b(2.5), a=one)
    c("SWAPNEG x src2", A(0), A(256), A(512), 64, 2, F32, F32, flags=fl(swapneg=1), a=one)
    c("VALID rows of 4 groups", A(0), A(256), A(512), 64, 0, F32, F32, flags=fl(), a=one, rowlen=4, vld=3 * d - 1)
    for kind in ("sum", "max"):
        c(f"REDUCE {kind} rows of 16", A(0), A(0), A(512), 64, 2 if kind == "sum" else 5, F32, F32,
          flags=fl(red=kind), a=one, rowlen=16)
        c(f"REDUCE {kind} rows of 8, VALID", A(0), A(0), A(512), 64, 5, F32, F32, flags=fl(red=kind), a=one,
          rowlen=8, vld=5 * d + 3)
        c(f"REDUCE {kind} whole, exp", A(0), A(0), A(512), 16, 5, F32, F32, acc="exp",
          flags=fl(func="exp", red=kind), a=one)
    # groups = words of the R x S matrix (R = 2d / 3d rows); source and destination must not overlap
    c("TRANSPOSE F32 (2d rows, S = 3)", A(0), 0, A(512), 2 * d * 3, 6, F32, F32, s=3)
    c("TRANSPOSE I8 (3d rows, S = 2)", SA(0), 0, SB(512), 3 * d * 2, 6, I8, I8, s=2)
    c("TRANSPOSE F32 (6d rows, S = 1)", A(0), 0, A(600), 6 * d, 6, F32, F32, s=1)

    os.makedirs(args.out, exist_ok=True)
    vh = [f"localparam integer NVC = {len(cases)}, NW = {NW};",
          "task vc_cmd(input integer i, output [31:0] s1, output [31:0] s2, output [31:0] dst, output [15:0] g,",
          "            output [7:0] op, output [5:0] ty, output [15:0] p2, output [10:0] fl, output [31:0] imm,",
          "            output [31:0] a, output [31:0] b, output [15:0] rl, output [15:0] vl, output [15:0] p1,",
          "            output [15:0] s);",
          "    case (i)"]
    for i, (name, f, acc) in enumerate(cases):
        sim = SaFuncSim(d)
        sim.mem[MEM_SPAD_A][:NW * d] = rng.integers(0, 256, NW * d, dtype=np.uint8)
        sim.mem[MEM_SPAD_B][:NW * d] = rng.integers(0, 256, NW * d, dtype=np.uint8)
        sim.mem[MEM_ACC][:NW * 4 * d] = acc_image(rng, d, acc).view(np.uint8)
        init = [sim.mem[m][:NW * (4 * d if m == MEM_ACC else d)].copy() for m in (MEM_SPAD_A, MEM_SPAD_B, MEM_ACC)]
        sim.ve(f["src1"], f["src2"], f["dst"], f["groups"], f["op"], f["types"], f.get("period", 0), 1, 0, 0,
               -2**31, 2**31 - 1, flags=f.get("flags", 0), imm=f.get("imm", 0), a=f.get("a", 0), b=f.get("b", 0),
               rowlen=f.get("rowlen", 0), valid=f.get("vld", 0), p1=f.get("p1", 0), s=f.get("s", 0))
        exp = [sim.mem[m][:NW * (4 * d if m == MEM_ACC else d)] for m in (MEM_SPAD_A, MEM_SPAD_B, MEM_ACC)]
        for tag, imgs in (("abc", init), ("ABC", exp)):
            for t, img, wb in zip(tag, imgs, (d, d, 4 * d)):
                words = img.reshape(NW, wb)[:, ::-1]           # hex: most significant byte first
                with open(os.path.join(args.out, f"vc{i}_{t}.hex"), "w") as fo:
                    fo.write("\n".join(w.tobytes().hex() for w in words) + "\n")
        vh.append(f'        {i}: begin s1 = 32\'h{f["src1"]:x}; s2 = 32\'h{f["src2"]:x}; dst = 32\'h{f["dst"]:x}; '
                  f'g = {f["groups"]}; op = 8\'h{f["op"]:x}; ty = 6\'h{f["types"]:x}; p2 = {f.get("period", 0)}; '
                  f'fl = 11\'h{f.get("flags", 0):x}; imm = 32\'h{f.get("imm", 0):x}; a = 32\'h{f.get("a", 0):x}; '
                  f'b = 32\'h{f.get("b", 0):x}; rl = {f.get("rowlen", 0)}; vl = {f.get("vld", 0)}; '
                  f'p1 = {f.get("p1", 0)}; s = {f.get("s", 0)}; $display("TB   case %0d: {name}", i); end')
    vh += ["    endcase", "endtask"]
    open(os.path.join(args.out, "vefp_cases.vh"), "w").write("\n".join(vh) + "\n")
    print(f"{len(cases)} fp VE cases for D = {d}")


if __name__ == "__main__":
    main()

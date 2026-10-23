"""Descriptor template for two dynamic expert GEMVs and fp32 weighted combine."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "driver"))

import sadesc  # noqa: E402
from pynq_matmul import DescList, MEM_ACC, MEM_SPAD_A, VOPS, laddr  # noqa: E402


VT_I32, VT_F32 = 2, 3
T_IF = VT_I32 | VT_F32 << 2
T_FF = VT_F32 | VT_F32 << 2


def top2_projection_template(dim, hidden, d=16):
    if dim <= 0 or hidden <= 0 or dim % d or hidden % d or hidden // d > 8:
        raise ValueError("M3 requires aligned dimensions and at most 8 GEMV strips")
    strips, row_bytes = hidden // d, dim * d
    y0, y1 = 0, strips + 1
    return (DescList()
            .ld(0, laddr(MEM_SPAD_A, 0), 1, dim, dim, base=0)
            .ldparam(12, 6, base=1)
            .ldparam(20, 7, base=1)
            .gemv(0, laddr(MEM_ACC, y0), strips, row_bytes, row_bytes, 0,
                  base=2, dyn=[("ddr", 6, True)])
            .ve(laddr(MEM_ACC, y0), 0, laddr(MEM_ACC, y0), hidden, VOPS["copy"], T_IF,
                fp=True, dyn=[("A", 7)])
            .ldparam(16, 6, base=1)
            .ldparam(24, 7, base=1)
            .gemv(0, laddr(MEM_ACC, y1), strips, row_bytes, row_bytes, 0,
                  base=2, dyn=[("ddr", 6, True)])
            .ve(laddr(MEM_ACC, y1), 0, laddr(MEM_ACC, y1), hidden, VOPS["copy"], T_IF,
                fp=True, dyn=[("A", 7)])
            .ve(laddr(MEM_ACC, y0), laddr(MEM_ACC, y1), laddr(MEM_ACC, y0), hidden,
                VOPS["add"], T_FF, fp=True)
            .st(0, laddr(MEM_ACC, y0), 1, hidden * 4, hidden * 4, base=3)
            .ret())


def make_executable(dim, hidden, d=16):
    executable = sadesc.Executable(
        d, caps=sadesc.CAPS_DESC | sadesc.CAPS_CMDX | sadesc.CAPS_GEMV | sadesc.CAPS_FPVE)
    executable.add("moe_top2_gate", top2_projection_template(dim, hidden, d), bindings=4)
    return executable.to_bytes()

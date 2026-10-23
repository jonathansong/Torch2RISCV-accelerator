"""Descriptor template for a complete device-side top-2 W8A8 MoE FFN."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "driver"))
sys.path.insert(0, os.path.join(REPO, "llm"))

import compile_layer as CL  # noqa: E402
import sadesc  # noqa: E402
from pynq_matmul import DescList, MEM_ACC, VOPS, laddr  # noqa: E402


I32, F32 = 2, 3
T_IF = I32 | F32 << 2
T_FF = F32 | F32 << 2


def _align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def expert_layout(dim, hidden, alignment=64):
    """Offsets within one record produced by frontend/moe.py:pack_parameters."""
    gate_w = 0
    gate_s = gate_w + hidden * dim
    up_w = _align(gate_s + hidden * 4, alignment)
    up_s = up_w + hidden * dim
    down_w = _align(up_s + hidden * 4, alignment)
    down_s = down_w + dim * hidden
    stride = _align(down_s + dim * 4, alignment)
    return {
        "gate_w": gate_w, "gate_s": gate_s,
        "up_w": up_w, "up_s": up_s,
        "down_w": down_w, "down_s": down_s,
        "stride": stride,
    }


def top2_ffn_template(dim, hidden, d=16, alignment=64, ret=True):
    if dim <= 0 or hidden <= 0 or dim % d or hidden % d:
        raise ValueError("M3b requires positive dimensions aligned to D")
    if max(dim, hidden) // d > 8:
        raise ValueError("M3b currently supports at most 8 GEMV strips")

    offsets = expert_layout(dim, hidden, alignment)
    lay = CL.Layout(d)
    dw, hw = dim // d, hidden // d

    # Persistent and reusable ACC allocations. Each word contains D lanes.
    x = lay.acc0
    tmp_x = x + dw
    gate = tmp_x + 3
    gate_s = gate + hw
    up = gate_s + hw
    up_s = up + hw
    hidden_v = up_s + hw
    tmp_h = hidden_v + hw
    down = tmp_h + 3
    down_s = down + dw
    combined = down_s + dw
    if combined + dw > lay.cbank:
        raise ValueError("M3b activation layout does not fit ACC bank 0")

    dl = DescList().ld(0, laddr(MEM_ACC, x), 1, dim * 4, dim * 4, base=0)

    def quant_vector(src, elements, tmp, dst):
        """Dynamic fp32 -> compact int8 quantization for the direct GEMV unit."""
        words = elements // d
        dl.ve(laddr(MEM_ACC, src), 0, laddr(MEM_ACC, tmp), elements,
              VOPS["copy"], T_FF, fp=True, func="abs", reduce="max")
        dl.ve(laddr(MEM_ACC, tmp), 0, laddr(MEM_ACC, tmp + 1), d,
              VOPS["copy"], T_FF, fp=True, A=CL.INV127)
        dl.ve(laddr(MEM_ACC, tmp), 0, laddr(MEM_ACC, tmp + 2), d,
              VOPS["copy"], T_FF, fp=True, A=CL.INV127, func="recip")
        dl.ve(laddr(MEM_ACC, src), laddr(MEM_ACC, tmp + 2),
              laddr(1, dst), elements, VOPS["mul"], F32, fp=True,
              m2="div", period=words)
        return tmp + 1

    sx = quant_vector(x, dim, tmp_x, 0)

    def projection(w_off, s_off, k, n, out, scale, a_strip):
        strips = n // d
        row_bytes = k * d
        dl.gemv(w_off, laddr(MEM_ACC, out), strips, row_bytes, row_bytes, a_strip,
                base=2, dyn=[("ddr", 6, True)])
        dl.ld(s_off, laddr(MEM_ACC, scale), 1, n * 4, n * 4,
              base=2, dyn=[("ddr", 6, True)])
        dl.ve(laddr(MEM_ACC, out), laddr(MEM_ACC, scale), laddr(MEM_ACC, out), n,
              VOPS["mul"], T_IF, fp=True, t2=F32)

    def expert(route_offset, route_weight, first):
        dl.ldparam(route_offset, 6, base=1)
        dl.ldparam(route_weight, 7, base=1)
        projection(offsets["gate_w"], offsets["gate_s"], dim, hidden, gate, gate_s, 0)
        dl.ve(laddr(MEM_ACC, gate), laddr(MEM_ACC, sx), laddr(MEM_ACC, gate), hidden,
              VOPS["mul"], T_FF, fp=True, m2="div", period=hw)
        projection(offsets["up_w"], offsets["up_s"], dim, hidden, up, up_s, 0)
        dl.ve(laddr(MEM_ACC, up), laddr(MEM_ACC, sx), laddr(MEM_ACC, up), hidden,
              VOPS["mul"], T_FF, fp=True, m2="div", period=hw)

        # hidden = SiLU(gate) * up = gate / (1 + exp(-gate)) * up.
        dl.ve(laddr(MEM_ACC, gate), 0, laddr(MEM_ACC, hidden_v), hidden,
              VOPS["copy"], T_FF, fp=True, A=-1.0, func="exp", B=-0.0)
        dl.ve(laddr(MEM_ACC, hidden_v), 0, laddr(MEM_ACC, hidden_v), hidden,
              VOPS["copy"], T_FF, fp=True, B=1.0, func="recip")
        dl.ve(laddr(MEM_ACC, gate), laddr(MEM_ACC, hidden_v), laddr(MEM_ACC, hidden_v), hidden,
              VOPS["mul"], T_FF, fp=True, B=-0.0)
        dl.ve(laddr(MEM_ACC, hidden_v), laddr(MEM_ACC, up), laddr(MEM_ACC, hidden_v), hidden,
              VOPS["mul"], T_FF, fp=True, B=-0.0)

        sh = quant_vector(hidden_v, hidden, tmp_h, dw)
        projection(offsets["down_w"], offsets["down_s"], hidden, dim, down, down_s, dw)
        dl.ve(laddr(MEM_ACC, down), laddr(MEM_ACC, sh), laddr(MEM_ACC, down), dim,
              VOPS["mul"], T_FF, fp=True, m2="div", period=dw)
        target = combined if first else down
        dl.ve(laddr(MEM_ACC, down), 0, laddr(MEM_ACC, target), dim,
              VOPS["copy"], T_FF, fp=True, dyn=[("A", 7)])
        if not first:
            dl.ve(laddr(MEM_ACC, combined), laddr(MEM_ACC, down), laddr(MEM_ACC, combined), dim,
                  VOPS["add"], T_FF, fp=True)

    expert(12, 20, True)
    expert(16, 24, False)
    dl.st(0, laddr(MEM_ACC, combined), 1, dim * 4, dim * 4, base=3)
    return dl.ret() if ret else dl.end(0x5A)


def make_executable(dim, hidden, d=16, alignment=64):
    executable = sadesc.Executable(
        d, caps=sadesc.CAPS_DESC | sadesc.CAPS_CMDX | sadesc.CAPS_GEMV | sadesc.CAPS_FPVE)
    executable.add("moe_top2_ffn", top2_ffn_template(dim, hidden, d, alignment), bindings=4)
    return executable.to_bytes()

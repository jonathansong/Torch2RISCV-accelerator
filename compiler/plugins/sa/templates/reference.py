"""Python reference of the sa backend's C++ templates (docs/iree_compiler_plan.md §6.4).

Each function builds, with the hand-written generator (llm/compile_layer.py),
the descriptor template the C++ backend must emit for one dispatch form,
under the sa-desc-v1 calling convention (§4.3): binding i -> BASE i, push
constants in PARAM0..5, PARAM6 / 7 private, relative jumps only, RET at the
end. The C++ output is compared with these byte for byte
(compiler/tests/test_c2.py), and these are run in the functional simulator
against DeviceModel.

qlinear: the int8 linear dispatch after the sa weight packing
(sa-pack-linear-weights), y[t, j] = float(sum_k x[k] * Wp[t, k, j]) * s_w[t, j] * s_x:
    x   i8[k] or i32[k] (int8 values)        binding b_x
    Wp  i8[n / D, k, D]  (= export_w8a8.pack_b(W))   binding b_w
    s_w f32[n]                               binding b_sw
    s_x f32 (scalar)                         binding b_sx
    y   f32[n]                               binding b_y
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "llm"))
sys.path.insert(0, os.path.join(REPO, "driver"))
import compile_layer as CL  # noqa: E402
from compile_layer import COPY, F32, I8, I32, T_FF, acc  # noqa: E402
from pynq_matmul import MEM_SPAD_A, DescList, laddr  # noqa: E402

PRIVATE_PARAMS = (6, 7)                     # PARAM6 / 7: the template's own (LOOP_END offsets)


def qlinear(d, k, n, x_i32, b_x, b_w, b_sw, b_sx, b_y):
    """DescList of the qlinear template (ends with RET)."""
    lay = CL.Layout(d)
    dl = DescList()
    # the replicated int8 A strip in SPAD_A words [0, k)
    if x_i32:
        xa = lay.acc0                                          # k / D ACC words
        dl.ld(0, acc(xa), 1, 4 * k, 4 * k, base=b_x)
        dl.ve(acc(xa), 0, laddr(MEM_SPAD_A, 0), k * d, COPY, I32 | I8 << 2, fp=True, m1="div", p1=d)
        sx = xa + k // d
    else:
        dl.ld(0, laddr(MEM_SPAD_A, k), 1, k, k, base=b_x)      # raw x after the strip
        dl.ve(laddr(MEM_SPAD_A, k), 0, laddr(MEM_SPAD_A, 0), k * d, COPY, I8 | I8 << 2, fp=True,
              m1="div", p1=d)
        sx = lay.acc0
    # s_x: one fp32 (the DMA moves 8 bytes), broadcast over the D lanes of the next word:
    # a max reduction over the first element (index modes address whole words, a
    # reduction writes its result to every lane)
    dl.ld(0, acc(sx), 1, 8, 8, base=b_sx)
    dl.ve(acc(sx), 0, acc(sx + 1), d, COPY, T_FF, fp=True, reduce="max", valid=1)
    CL.linear(dl, lay, k, n, 0, 0, sx + 1, out_ddr=0, wbase=b_w, sbase=b_sw, iobase=b_y,
              params=PRIVATE_PARAMS)
    return dl.ret()

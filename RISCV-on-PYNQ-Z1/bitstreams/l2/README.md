# L2: fp32 vector engine, SFU, TRANSPOSE (prebuilt)

This build is the L1 D = 8 accelerator plus the L2 additions of
`docs/llm_inference_plan.md` §6:

- **fp32 vector engine** `sa_vefp.v` next to the integer `sa_ve.v`: OP,
  y * A + B, EXP / RECIP / RSQRT / ABS, RELU, VALID mask, row reductions
  (SUM / MAX), index modes LIN / MOD / DIV / IMM, SWAPNEG, I8 / I32 / F32
  inputs and outputs; bit-exact with `llm/sa_funcsim.py`.
- **TRANSPOSE** (VE op 6) of D x D blocks.
- Lane folding: FL = 4 physical fp lanes for D = 8 (§6.9).
- **CAPS**: bits 22, 23 and 24 are set.

It is used with `firmware/rt` (`rt_fw.bin`) and `driver/pynq_matmul.py`
`Device`; board test `notebooks/llm/l2_vpu_demo.py`.

Built with Vivado 2024.1 (`scripts/build_bitstream.sh -jobs 4 -sa_d 8`):
44,005 LUT (82.7 %), 33,622 FF, 116 DSP, 130 BRAM36, setup WNS +1.836 ns
at 50 MHz. The L1 build was 27,136 LUT. `sa_vefp` alone is 17.2k LUT (OOC).

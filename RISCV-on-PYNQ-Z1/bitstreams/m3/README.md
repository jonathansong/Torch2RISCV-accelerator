# M3 overlay (prebuilt)

Double-buffered accelerator (`rtl/sysarray`, D = 8, one DMA port) with the
M3 vector engine (`sa_ve.v`, VL = 8, funct7 = 2 `vec_cfg` / `vec_run`),
fused after matmul or standalone. Built with Vivado 2024.1
(`scripts/build_bitstream.sh -jobs 4`). Setup WNS +0.435 ns at 50 MHz (the
worst path is in the clock wizard's AXI-Lite reset; the accelerator alone
has +2.5 ns OOC); 21720 LUT, 17000 FF, 96 DSP, 130 BRAM36.

Verified on the board (`notebooks/m3_vector_demo.py`), all bit-exact vs NumPy.

Fused int8 GEMM, `C = requant(relu(A @ B + bias))` with the epilogue on the
VE, vs the same GEMM with int32 output:

| GEMM (M×N×K) | bias | relu | result | cycles | MAC/cycle | int32 cycles | int32 MAC/cycle |
|---|---|---|---|---|---|---|---|
| 8×8×8 | yes | yes | PASS | 1762 | 0.3 | 1059 | 0.5 |
| 24×48×40 | – | – | PASS | 2750 | 16.8 | 2283 | 20.2 |
| 64×64×64 | yes | yes | PASS | 8539 | 30.7 | 6994 | 37.5 |
| 128×128×128 | yes | – | PASS | 46200 | 45.4 | 42166 | 49.7 |
| 256×256×256 | yes | yes | PASS | 313398 | 53.5 | 297155 | 56.5 |

Standalone vector ops (DDR → SPAD/ACC → VE → DDR, `vector_fw.bin`):

| op | in→out | n | src2 | post | cycles | elem/cycle | B/cycle |
|---|---|---|---|---|---|---|---|
| add | i8→i8 | 4096 | 4096 | – | 3925 | 1.04 | 3.13 |
| sub | i8→i16 | 4096 | 4096 | – | 5365 | 0.76 | 3.05 |
| mul | i8→i16 | 4096 | 4096 | – | 5358 | 0.76 | 3.06 |
| mul | i16→i32 | 4096 | 4096 | – | 7551 | 0.54 | 4.34 |
| max | i16→i16 | 2048 | 8 (broadcast) | – | 3578 | 0.57 | 2.29 |
| min | i8→i8 | 4000 | 4000 | clamp | 3953 | 1.01 | 3.04 |
| add | i32→i8 | 8192 | 64 (period 8) | relu+requant | 8668 | 0.95 | 4.75 |
| copy | i32→i32 | 8200 | – | relu | 9615 | 0.85 | 6.82 |
| add | i32→i32 | 16384 | 16384 | – | 26103 | 0.63 | 7.53 |
| sub | i8→i32 | 12000 | 24 (period 3) | – | 9549 | 1.26 | 6.29 |

- The int8 epilogue costs 3–8 % at 128³–256³ (not yet profiled; likely the
  longer per-strip chain EX → VE → ST, which delays reusing the strip's bank
  two strips later). In return C traffic drops 4× and the ARM does no
  post-processing.
- Standalone ops are DMA-bound at small element sizes (many 256-byte rows
  plus firmware issue overhead, not profiled); int32 streams reach 7.5 B/cycle, near the
  7.9 B/cycle HP port limit.
- Legacy firmware unchanged: `matmul_insn_fw` 1024/1024 (339.6 cycles/job).

Needs `firmware/{gemm,vector,matmul_insn}/*.bin` and `driver/pynq_matmul.py`.

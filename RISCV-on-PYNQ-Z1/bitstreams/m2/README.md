# M2 overlay (prebuilt)

Double-buffered accelerator (`rtl/sysarray`, D = 8, one DMA port) with the
M2 changes: repeat `mat_exec` (a row of C tiles per command, EX_* cfg keys)
and 8 outstanding bursts per DMA port. Built with Vivado 2024.1
(`scripts/build_bitstream.sh -jobs 4`). Setup WNS +1.498 ns at 50 MHz;
11965 LUT, 13102 FF, 72 DSP, 130 BRAM36.

Verified on the board (`notebooks/m1_gemm_demo.py` with the M2 `gemm_fw.bin`,
`notebooks/m2_bw_test.py`):

| GEMM (M×N×K) | bias | result | cycles | MAC/cycle | M1 MAC/cycle |
|---|---|---|---|---|---|
| 8×8×8 | – | PASS | 875 | 0.6 | 0.5 |
| 24×16×40 | yes | PASS | 1985 | 7.7 | 4.3 |
| 64×64×64 | – | PASS | 6845 | 38.3 | 13.0 |
| 128×128×128 | – | PASS | 42146 | 49.8 | 28.4 |
| 32×256×64 | yes | PASS | 18088 | 29.0 | 13.1 |
| 256×256×256 | – | PASS | 297134 | 56.5 | 54.5 |

- 64×64×64 vs 512 independent 8×8×8 jobs on the Phase 4 path: 25.4×.
- Legacy firmware unchanged: `matmul_insn_fw` 1024/1024 (339.3 cycles/job),
  `matmul_fw` 1024/1024 (522.6 cycles/job).
- DMA (one HP port): contiguous LD/ST 7.9 B/cycle (99 %), LD+ST together
  15.76 B/cycle, single-beat bursts 3.81 B/cycle (M1: 1.93).

Needs `firmware/{gemm,matmul,matmul_insn,bwtest}/*.bin` and `driver/pynq_matmul.py`.

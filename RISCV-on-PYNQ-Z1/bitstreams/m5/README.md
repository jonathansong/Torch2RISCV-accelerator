# M5: descriptor DMA (prebuilt)

The M4 accelerator (D = 16, one DMA port) with the performance counters and
the descriptor fetch unit `rtl/sysarray/sa_cmdfetch.v` (`mat_submit` =
funct7 1 / funct3 6, CAPS bits 20 and 21; docs/double_buffer_design.md §8.6).
Built with Vivado 2024.1 (`scripts/build_bitstream.sh -jobs 4 -sa_d 16`).
Setup WNS +2.460 ns at 50 MHz; 48928 LUT (92.0 %), 36350 FF, 184 DSP,
130 BRAM36 (m4p: 48267 LUT).

Verified on the board:

- `notebooks/m5_desc_demo.py` — the same work issued by the PCPI firmware and
  as one descriptor list built by the ARM (`desc_run_fw.bin`: one
  `mat_submit`), all results bit-exact vs NumPy:

  | Work | PCPI cycles | list cycles | speedup | front end idle PCPI → list |
  |---|---|---|---|---|
  | GEMM 16³ | 1801 | 430 | 4.19× | 99.4 % → 62.9 % |
  | GEMM 64³ | 4154 | 3383 | 1.23× | 86.8 % → 24.3 % |
  | GEMM 32×256×64 | 7409 | 7312 | 1.01× | 61.9 % → 37.4 % |
  | GEMM 128³ | 13384 | 13366 | 1.00× | 23.2 % → 9.2 % |
  | GEMM 256³ | 82798 | 82784 | 1.00× | 4.7 % → 2.7 % |
  | int8 GEMM 64³ | 4984 | 3288 | 1.52× | 68.3 % → 10.6 % |
  | int8 GEMM 256³ | 92215 | 92424 | 1.00× | 3.3 % → 0.9 % |
  | vector int8 add, 4096 | 3376 | 2224 | 1.52× | |
  | vector int16 mul → int32, 4096 | 6448 | 5296 | 1.22× | |
  | vector int32 add, 16384 | 24952 | 23806 | 1.05× | |
  | vector int32 → int8 + bias, relu, requant, 8192 | 8163 | 6449 | 1.27× | |
  | vector int16 max (broadcast), 2048 | 3801 | 1460 | 2.60× | |

  List cycles run from `mat_submit` to the end of `mat_fence`; the list is
  built by the ARM beforehand (and can be reused: one list run three times
  on different buffers through the relocation bases, all PASS).
- `notebooks/m4_perf.py`: 69/69 counter checks; `notebooks/m4_demo.py`:
  unchanged (256³ 202.6 MAC/cycle, legacy 1024/1024).

Needs `firmware/{gemm,vector,bwtest,matmul,matmul_insn,desc_run}/*.bin` and
`driver/pynq_matmul.py` from the same commit.

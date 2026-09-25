# M4 + performance counters (prebuilt)

The M4 accelerator (D = 16, 8 DSP + 8 LUT columns, VL = 16, one DMA port)
with the performance counters of `rtl/sysarray/sa_perf.v` (`PERF = 1`, CAPS
bit 20; `mat_perf` = funct7 1 / funct3 5, CSR mirror 0x40–0xBC). Built with
Vivado 2024.1 (`scripts/build_bitstream.sh -jobs 4 -sa_d 16`). Setup WNS
+1.904 ns at 50 MHz; 48267 LUT (90.7 %), 35110 FF, 184 DSP, 130 BRAM36
(M4: 47618 LUT, WNS +1.319 ns).

Verified on the board:

- `notebooks/m4_perf.py`: 7 GEMMs and 4 vector operations bit-exact vs NumPy,
  69/69 counter checks (tiles, useful steps × D² = M·N·K, bytes stored,
  commands, counting window) pass. Cycle breakdown in
  `docs/double_buffer_design.md` §10.4, e.g. 256³: EX useful 79.0 %,
  fill 9.3 %, CPU stalled on a full queue 61.3 %; 64³: head empty or all
  idle 86.8 % (front-end bound).
- `notebooks/m4_demo.py`: unchanged against M4 with the tuned firmware
  schedule — 256³ 202.6, 512×256×256 213.9, 256×128×1024 222.2 MAC/cycle,
  int8 256³ 181.9, all vector cases, DMA 7.94 / 15.76 B/cycle, legacy
  1024/1024 (386.9 and 533.7 cycles/job).

Firmware images must stay below 0x1E00 (the counters are copied to the
program BRAM 0x1E00–0x1EFF). Needs `firmware/{gemm,vector,bwtest,matmul,matmul_insn}/*.bin`
and `driver/pynq_matmul.py` from the same commit.

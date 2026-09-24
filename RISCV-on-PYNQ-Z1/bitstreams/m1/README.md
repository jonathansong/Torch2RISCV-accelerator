# M1 overlay (prebuilt)

Double-buffered accelerator (`rtl/sysarray`, D = 8, one DMA port on
S_AXI_HP2) with the funct7 = 1 ISA and the Phase 2–4 legacy interface.
Built with Vivado 2024.1 by `RISCV-on-PYNQ-Z1/scripts/build_bitstream.sh`.
Setup WNS +2.519 ns at 50 MHz; 10601 LUT, 12558 FF, 70 DSP, 130 BRAM36
(128 accelerator + 2 program BRAM).

Verified on the board with `notebooks/m1_gemm_demo.py`:

| GEMM (M×N×K) | bias | result | RISC-V cycles | MAC/cycle |
|---|---|---|---|---|
| 8×8×8 | – | PASS | 1111 | 0.5 |
| 24×16×40 | yes | PASS | 3589 | 4.3 |
| 64×64×64 | – | PASS | 20200 | 13.0 |
| 128×128×128 | – | PASS | 73969 | 28.4 |
| 32×256×64 | yes | PASS | 40169 | 13.1 |
| 256×256×256 | – | PASS | 308066 | 54.5 (85 % of peak) |

- 64×64×64 as 512 independent 8×8×8 jobs on the Phase 4 path: 173568
  cycles + partial sums on the ARM → 8.6× slower than the new ISA.
- Legacy firmware unchanged on this hardware: `matmul_insn_fw` 1024/1024,
  339.4 cycles/job (Phase 4: 311.9); `matmul_fw` 1024/1024, 522.6 cycles/job
  (Phase 3: 522.6).

Needs `firmware/{gemm,matmul,matmul_insn}/*.bin` and `driver/pynq_matmul.py`.

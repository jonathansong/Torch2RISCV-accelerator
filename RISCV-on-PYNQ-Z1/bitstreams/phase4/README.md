# Phase 4 overlay (prebuilt)

PicoRV32 + 8×8×8 int8 matmul unit with both control paths: AXI4-Lite CSRs
and the custom-0 matrix instructions on the PicoRV32 PCPI port
(`docs/custom_isa_encoding.md`). Built with Vivado 2024.1 by
`RISCV-on-PYNQ-Z1/scripts/build_bitstream.sh` from commit `0a299be`.
Setup WNS +3.591 ns at 50 MHz; 6640 LUT, 7421 FF, 64 DSP, 16 BRAM.

| File | |
|---|---|
| `picorv32.bit`, `picorv32.hwh` | overlay; keep the same basename for `pynq.Overlay` |
| `utilization.rpt`, `timing_summary.rpt` | implementation reports |

Verified on the board with `notebooks/phase4_insn_demo.py`:

- custom-instruction firmware (`firmware/matmul_insn`, no CSR access):
  64-tile demo, 2560/2560 random matmuls and int8 extremes match NumPy
- same 256 tiles through both paths:

| Path | cycles/job | µs/job @ 50 MHz | accelerator | CPU side |
|---|---|---|---|---|
| CSR (`firmware/matmul`) | 522.6 | 10.45 | 131.9 | 390.7 |
| custom instructions (`firmware/matmul_insn`) | 311.9 | 6.24 | 149.4 | 162.4 |

Also runs the Phase 3 CSR firmware unchanged. Needs the firmware `.bin`
files and `driver/pynq_matmul.py` next to it.

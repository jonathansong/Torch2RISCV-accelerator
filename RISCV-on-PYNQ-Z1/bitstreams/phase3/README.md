# Phase 3 overlay (prebuilt)

PicoRV32 + 8×8×8 int8 matmul unit for PYNQ-Z1, built with Vivado 2024.1 by
`RISCV-on-PYNQ-Z1/scripts/build_bitstream.sh` from the sources at commit
`194e756` (block design + `rtl/matmul`). Setup WNS +5.785 ns at 50 MHz.

| File | |
|---|---|
| `picorv32.bit`, `picorv32.hwh` | overlay; keep the same basename for `pynq.Overlay` |
| `utilization.rpt`, `timing_summary.rpt` | implementation reports |

Verified on the board with `notebooks/phase3_matmul_demo.py` (2560/2560
random matmuls match NumPy). Needs `firmware/matmul/matmul_fw.bin` and
`driver/pynq_matmul.py` next to it; see `docs/memory_model.md`.

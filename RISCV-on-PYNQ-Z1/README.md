# RISCV-on-PYNQ-Z1 — the PYNQ-Z1 overlay

The FPGA overlay of [Torch2RISCV-accelerator](../README.md): the Zynq PS7, a
PicoRV32 control core with an 8 KB program BRAM and its own DDR access (HP0),
and the double-buffered accelerator `rtl/sysarray` (`sa_unit`, DMA on HP2).

## Build

```sh
./scripts/build_bitstream.sh -jobs 4 -sa_d 16     # Vivado 2024.1, 15-30 min
```

Output: `build/output/picorv32.bit` and `.hwh` (+ timing and utilization
reports); the Vivado project is `build/picorv32_z1/picorv32_z1.xpr`. `-sa_d 8`
builds the D = 8 configuration of M1–M3. `build/` is not tracked.

## Contents

| Path | What |
|---|---|
| `scripts/build_bitstream.sh`, `build_bitstream.tcl` | one-shot build (project, block design, synthesis, implementation, bitstream, `.hwh`) |
| `scripts/pico_bit.tcl`, `pico_processor.tcl` | the block design: PS7, the PicoRV32 hierarchy, `matmul_0` (module reference to `sa_unit`), address map |
| `ip/` | IP repository used by the build: `picorv32_axi` (PicoRV32 with AXI and PCPI) and the `pcpi` interface definition |
| `picorv32/` | `picorv32.v` from [YosysHQ/picorv32](https://github.com/YosysHQ/picorv32) (license in `COPYING`), used by `ip/picorv32_axi` and the system simulations |
| `constrs/PYNQ-Z1.xdc` | board constraints |
| `bitstreams/<milestone>/` | board-verified `.bit` / `.hwh` with reports and a README of the results: phase3, phase4, m1–m4, m4p (performance counters), m5 (descriptor DMA) |
| `tests/ddr_access/` | the Phase 1 RISC-V → DDR access test (firmware, simulation, board script) |
| `pynq-z1-riscv-accelerator-plan-v2.md` | the original project plan (Chinese) |

The accelerator RTL, firmware, driver and board scripts live at the top of the
repository (`rtl/`, `firmware/`, `driver/`, `notebooks/`); design documents in
`docs/`.

## Acknowledgements

The PicoRV32-on-Zynq bring-up started from
[drichmond/RISC-V-On-PYNQ](https://github.com/drichmond/RISC-V-On-PYNQ) and
[JacoboJin/RISCV-on-PYNQ-Z2](https://github.com/JacoboJin/RISCV-on-PYNQ-Z2)
(MIT, `LICENSE`).

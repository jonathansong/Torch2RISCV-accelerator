# Kria KV260 overlay (docs/kv260_upgrade_plan.md)

The PicoRV32 + systolic-array overlay on the Kria KV260 (K26 SOM,
`xck26-sfvc784-2LV-c`, Vivado 2024.1). K1a: the PYNQ-Z1 L2 accelerator
unchanged (D = 8, 128 KB per SPAD, 256 KB ACC, one 64-bit DMA port, 50 MHz);
the plan's later stages change one thing at a time (K1b 100 MHz, K1c D = 16).

```sh
KV260/scripts/build_bitstream.sh -jobs 4                 # K1a: D = 8, 50 MHz
KV260/scripts/build_bitstream.sh -jobs 4 -sa_mhz 100     # K1b
KV260/scripts/build_bitstream.sh -jobs 4 -sa_d 16 -sa_mhz 100   # K1c
KV260/scripts/build_bitstream.sh -bd_only                # block design + address map only (minutes)
```

Outputs in `KV260/build/output/`: `picorv32.bit` / `picorv32.hwh` (same
basename for PYNQ), `timing_summary.rpt`, `utilization.rpt`, `clocks.rpt`,
`check_timing.rpt`, `address_map.txt`.

## Block design (`scripts/kv260_bd.tcl`)

The PL of the Z1 design (`RISCV-on-PYNQ-Z1/scripts/pico_bit.tcl`): the same
PicoRV32 hierarchy (`pico_processor.tcl`, IP `RISCV-on-PYNQ-Z1/ip`) and
`sa_unit` (`rtl/sysarray`) on the Zynq UltraScale+ PS (board preset of
`kv260_som` for DDR4 and MIO).

| | KV260 | PYNQ-Z1 |
|---|---|---|
| ARM -> PL | `M_AXI_HPM0_FPD` (32-bit) | `M_AXI_GP0` |
| PicoRV32 -> DDR (rings) | `S_AXI_HP0_FPD` (64-bit, AXI4) | `S_AXI_HP0` |
| accelerator DMA | `S_AXI_HP1_FPD` (64-bit, AXI4, direct) | `S_AXI_HP2` via an AXI4 -> AXI3 converter |
| PL clock | `pl_clk0` = `-sa_mhz`, the only clock | FCLK0 50 MHz + clk_wiz 50 MHz |
| RISC-V reset | EMIO GPIO[0] (1 = hold) | EMIO GPIO[0] |
| interrupts | axi_intc -> `pl_ps_irq0` (In0 PicoRV32 trap, In1 `matmul_0/notify_irq`) | axi_intc -> IRQ_F2P |
| PL I/O | none | unused LED / button / Arduino / PMOD ports |

## Address map (`build/output/address_map.txt`)

| Master | Slave | Address | Range |
|---|---|---|---|
| ARM (`zynq_ultra_ps_e_0/Data`) | program BRAM (`psBramController`), mailbox at +0x1F00, perf area at +0x1E00 | `0xA001_0000` | 8 KB |
| ARM | interrupt controller | `0xA002_0000` | 64 KB |
| PicoRV32 | DDR (HP0, low 2 GB, identity-mapped) | `0x0000_0000` | 2 GB |
| PicoRV32 | accelerator CSRs | `0x8000_0000` | 4 KB |
| PicoRV32 | program BRAM (reset vector) | `0xC000_0000` | 8 KB |
| accelerator DMA | DDR (HP1, low 2 GB, identity-mapped) | `0x0000_0000` | 2 GB |

The RISC-V / accelerator side is the Z1's (firmware unchanged); on the ARM
side only the window moves (Z1: BRAM at `0x4001_0000`, so the driver's
`BRAM_ARM_BASE` becomes `0xA001_0000`, `SA_BOARD_MBOX` `0xA001_1F00`).
DDR buffers for the accelerator must be in the low 2 GB
(docs/kv260_upgrade_plan.md §2.4; the runtime checks it).

## To confirm on the board

- The carrier's fan is controlled from the PL in AMD's reference designs;
  this design drives no PL pin. Check the fan's behaviour (speed, noise,
  temperature) with this overlay loaded before long runs.
- The EMIO GPIO numbering seen by PYNQ (`GPIO.get_gpio_pin(0)`) on the ZynqMP.
- Overlay loading on the chosen image (Kria-PYNQ vs Ubuntu for Kria) with this `.bit` / `.hwh`.

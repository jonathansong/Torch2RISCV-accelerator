# Kria KV260 overlay (docs/kv260_upgrade_plan.md)

The PicoRV32 + systolic-array overlay on the Kria KV260 (K26 SOM,
`xck26-sfvc784-2LV-c`, Vivado 2024.1). K1a: the PYNQ-Z1 L2 accelerator
unchanged (D = 8, 128 KB per SPAD, 256 KB ACC, one 64-bit DMA port, 50 MHz);
the plan's later stages change one thing at a time (K1b 100 MHz, K1c D = 16).

```sh
KV260/scripts/build_bitstream.sh -jobs 2                 # K1a: D = 8, 50 MHz (~25 min)
KV260/scripts/build_bitstream.sh -jobs 2 -sa_mhz 100     # K1b
KV260/scripts/build_bitstream.sh -jobs 2 -sa_d 16 -sa_mhz 100   # K1c
KV260/scripts/build_bitstream.sh -bd_only                # block design + address map only (minutes)
KV260/scripts/build_bitstream.sh -jobs 2 -synth_only     # stop after synthesis (~9 min)
```

Outputs per configuration in `KV260/build/output/d<D>_<MHz>mhz/` (K1a
`d8_50mhz`, K1b `d8_100mhz`, K1c `d16_100mhz`; the Vivado project in
`build/picorv32_kv260_<config>`, the log `build/vivado_<config>.log`):
`picorv32.bit` / `picorv32.hwh` (same basename for PYNQ), `build_info.txt`
(configuration, commit, date, WNS / WHS), `timing_summary.rpt`,
`utilization.rpt`, `clocks.rpt`, `check_timing.rpt`, `address_map.txt`,
`post_synth.dcp`, `post_route.dcp`.

## Builds

| Build | Date | LUT | FF | DSP | BRAM36 | URAM | Setup WNS / hold WHS | Notes |
|---|---|---|---|---|---|---|---|---|
| K1a: D = 8, 50 MHz (`-jobs 2`) | 2026-10-04 | 40,351 (34.5%) | 31,576 (13.5%) | 115 (9.2%) | 130 / 144 (90.3%) | 0 / 64 | +9.050 / +0.010 ns | no unclocked or unconstrained endpoints (`check_timing`); one clock `clk_pl_0` (20 ns) |
| K1b: D = 8, 100 MHz (`-sa_mhz 100`, LDPARAM fix `b4bd8a6`) | 2026-10-04 | 40,275 (34.4%) | 31,550 (13.5%) | 115 | 130 / 144 | 0 / 64 | +1.171 / +0.010 ns | `clk_pl_0` 10 ns; worst path LD `q_lane` -> DSP -> ACC BRAM write enable (7.8 ns, 13 levels): about 113 MHz |

K1a: the worst setup path (10.5 ns data path, 14 levels including a DSP
multiplier) runs from the DMA port's read data (`PS8_i/SAXIGP3RCLK`, HP1)
to the descriptor fetch unit's `fetch/param_reg`, so about 90-95 MHz is the
limit as built: K1b (100 MHz) needs a register stage on that path - done
(`sa_cmdfetch.v`: the word is registered on the last beat, multiplied the
cycle after; `tb_sa_unit` passes at D = 8 / 16). The next paths of the K1a
routing (20 ns target, so not pushed): VE TRANSPOSE address (`tr_word0`, a
DSP) -> ACC BRAM address / write enable (9.5-10.1 ns, 11 levels), LD
`q_rp` -> ACC BRAM address (~10.1 ns, 13 levels), the RISC-V reset -> EX
`bsr_reg` synchronous resets (~10.2 ns, high fanout); K1b's 100 MHz build
will show whether the tool closes them, K2a's 200-250 MHz needs stages there. BRAM is
at 90%: SPAD / ACC growth (K3, D = 32's wider words) goes to URAM.

Memory: the build runs in one Vivado process (global IP synthesis, in-process
synth / place / route) inside a systemd scope with MemoryMax = 12G; the
project-mode out-of-context runs reloaded the UltraScale+ device data per IP
run and systemd-oomd killed the terminal / the build on this 15 GB machine.
`-synth_only -jobs 2`: peak RSS 11.9 GB (with the mapped device data), at
least 3.1 GB left.

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

## Address map (`build/output/<config>/address_map.txt`)

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

## Board tests

The driver (`driver/pynq_matmul.py` `overlay_info`) reads the board, D, the
ARM-side BRAM address and the accelerator clock from the `.hwh`, so the
launcher, `bwtest` and the LLM tests need no board switch. The deploy
scripts take `SA_BOARD=kv260` (`compiler/scripts/board_env.sh`: the aarch64
runtime and the overlay of `build/output/$SA_KV260_CONFIG`, default
`d8_50mhz`, or `SA_BIT_DIR`); the whole regression of the Z1 freeze:

```sh
compiler/scripts/deploy_z1_freeze.sh --board kv260        # K1a (d8_50mhz) -> build/deploy_kv260
SA_KV260_CONFIG=d8_100mhz compiler/scripts/deploy_z1_freeze.sh --board kv260   # K1b
scp -r build/deploy_kv260 <user>@<kv260>:~/kv260
# on the board (PYNQ venv, root): cd ~/kv260 && python3 board_regress.py
```

and compare `results/` with `tests/baselines/pynq-z1/` (K1a: the token files
must match). Checked without the board: the aarch64 `sa-llm-run` of a KV260
bundle under qemu with the ring emulator at the KV260's addresses
(`sa_board_emu.py --mbox 0xA0011F00 --base 0x7C000000`): stories15M 22/22
logits rows bit-exact.

## To confirm on the board

- The carrier's fan is controlled from the PL in AMD's reference designs;
  this design drives no PL pin. Check the fan's behaviour (speed, noise,
  temperature) with this overlay loaded before long runs.
- The EMIO GPIO numbering seen by PYNQ (`GPIO.get_gpio_pin(0)`) on the ZynqMP.
- Overlay loading on the chosen image (Kria-PYNQ vs Ubuntu for Kria) with this `.bit` / `.hwh`.

# Kria KV260 overlay (docs/kv260_upgrade_plan.md)

The PicoRV32 + systolic-array overlay on the Kria KV260 (K26 SOM,
`xck26-sfvc784-2LV-c`, Vivado 2024.1). K1a: the PYNQ-Z1 L2 accelerator
unchanged (D = 8, 128 KB per SPAD, 256 KB ACC, one 64-bit DMA port, 50 MHz);
the plan's later stages change one thing at a time (K1b 100 MHz, K1c D = 16).
K1a and K1b are accepted on the board (2026-10-08, tokens identical to the
PYNQ-Z1 baselines; results below).

```sh
boards/kv260/scripts/build_bitstream.sh -jobs 2                 # K1a: D = 8, 50 MHz (~25 min)
boards/kv260/scripts/build_bitstream.sh -jobs 2 -sa_mhz 100     # K1b
boards/kv260/scripts/build_bitstream.sh -jobs 2 -sa_d 16 -sa_mhz 100   # K1c
boards/kv260/scripts/build_bitstream.sh -bd_only                # block design + address map only (minutes)
boards/kv260/scripts/build_bitstream.sh -jobs 2 -synth_only     # stop after synthesis (~9 min)
```

Outputs per configuration in `boards/kv260/build/output/d<D>_<MHz>mhz/` (K1a
`d8_50mhz`, K1b `d8_100mhz`, K1c `d16_100mhz`; the Vivado project in
`build/picorv32_kv260_<config>`, the log `build/vivado_<config>.log`):
`picorv32.bit` / `picorv32.hwh` (same basename for PYNQ), `build_info.txt`
(configuration, commit, date, WNS / WHS), `timing_summary.rpt`,
`utilization.rpt`, `clocks.rpt`, `check_timing.rpt`, `address_map.txt`,
`post_synth.dcp`, `post_route.dcp`.

## Board (K0, 2026-10-07)

| | |
|---|---|
| Image | Ubuntu 22.04.4 LTS (Ubuntu for Kria), kernel 5.15.0-1027-xilinx-zynqmp |
| Boot firmware | K26-BootFW-01.02 (XilinxSom_QspiImage-k26-v2.1), U-Boot 2023.01; image A / B identical |
| DDR | low `0x0000_0000`-`0x7FEF_FFFF`, high `0x8_0000_0000`-`0x8_7FFF_FFFF` |
| CMA | 1000 MiB at `0x3780_0000` (`cma=1000M`), entirely in the low 2 GB: the accelerator's windows are allocated there (the runtime checks the window ends at or below 2 GB) |
| PL at boot | the `k26-starter-kits` app is loaded; it must be unloaded (`xmutil unloadapp`) before an overlay: the driver (`kria_unload_app`) and `ddr_test.py` do it |
| Network | MAC 00:0a:35:2a:df:b8, 192.168.0.121 (DHCP) |

PYNQ (Kria-PYNQ, which installs the PYNQ venv and XRT setup under
`/etc/profile.d/`) is needed by the driver; Vivado 2024.1 builds the overlay.

## Builds

| Build | Date | LUT | FF | DSP | BRAM36 | URAM | Setup WNS / hold WHS | Notes |
|---|---|---|---|---|---|---|---|---|
| K1a, first build: D = 8, 50 MHz (`-jobs 2`) | 2026-10-04 | 40,351 (34.5%) | 31,576 (13.5%) | 115 (9.2%) | 130 / 144 (90.3%) | 0 / 64 | +9.050 / +0.010 ns | before the LDPARAM register stage; no unclocked or unconstrained endpoints (`check_timing`); one clock `clk_pl_0` (20 ns) |
| **K1a: D = 8, 50 MHz** (`output/d8_50mhz`, commit `0b9c414`) | 2026-10-04 | 40,257 (34.4%) | | 115 | 130 / 144 | 0 / 64 | +7.800 / +0.010 ns | with the LDPARAM stage; the overlay for K1a's board tests |
| **K1b: D = 8, 100 MHz** (`output/d8_100mhz`, LDPARAM fix `b4bd8a6`) | 2026-10-04 | 40,275 (34.4%) | 31,550 (13.5%) | 115 | 130 / 144 | 0 / 64 | +1.171 / +0.010 ns | `clk_pl_0` 10 ns; worst path LD `q_lane` -> DSP -> ACC BRAM write enable (7.8 ns, 13 levels): about 113 MHz |
| **K1c: D = 16, 100 MHz** (`output/d16_100mhz`, commit `4bd8389`) | 2026-10-08 | 74,951 (64.0%) | 59,944 (25.6%) | 347 (27.8%) | 130 / 144 | 0 / 64 | +0.630 / +0.011 ns | every PE column on DSPs (`DSP_COLS = D`: 256 for the array); the fp VE at FL = 8; worst path LD `q_rp` -> DSP -> ACC BRAM write enable (8.4 ns, 12 levels) |

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

The PL of the Z1 design (`RISCV-on-PYNQ-Z1/scripts/pico_bit.tcl` on the
`pynq-z1` branch): the same PicoRV32 hierarchy (`scripts/pico_processor.tcl`,
IP `rtl/ip`) and
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
scripts default to `SA_BOARD=kv260` (`compiler/scripts/board_env.sh`: the aarch64
runtime and the overlay of `build/output/$SA_KV260_CONFIG`, default
`d8_100mhz`, or `SA_BIT_DIR`; the frozen Z1's bundles: `pynq-z1` branch); the whole
regression of the Z1 freeze:

```sh
compiler/scripts/deploy_z1_freeze.sh                     # K1b (d8_100mhz) -> build/deploy_kv260_d8_100mhz
SA_KV260_CONFIG=d8_50mhz compiler/scripts/deploy_z1_freeze.sh    # K1a -> build/deploy_kv260_d8_50mhz
SA_KV260_CONFIG=d16_100mhz compiler/scripts/deploy_z1_freeze.sh  # K1c (D = 16: compiled with --iree-sa-d=16)
scp -r build/deploy_kv260_d8_100mhz <user>@<kv260>:~/kv260_d8_100mhz
# on the board: cd ~/kv260_d8_100mhz && sudo ./run_board.sh
# back: results/ -> build/deploy_kv260_d8_100mhz/results
$SA_PY compiler/tests/compare_z1_baselines.py build/deploy_kv260_d8_100mhz/results
```

and compare `results/` with `tests/baselines/pynq-z1/` (K1a: the token files
must match). Checked without the board: the aarch64 `sa-llm-run` of a KV260
bundle under qemu with the ring emulator at the KV260's addresses
(`sa_board_emu.py --mbox 0xA0011F00 --base 0x7C000000`): stories15M 22/22
logits rows bit-exact.

## K1a on the board (2026-10-08): passed

All K1a steps pass on the board with the `d8_50mhz` overlay (`REGRESSION
PASS`), and `compiler/tests/compare_z1_baselines.py` finds every run's tokens
identical to the PYNQ-Z1 baselines. Device cycles within 0.03% of the Z1
(same accelerator, clock and DMA width); wall time 1-4% faster (the A53
host): stories15M 18.27 tok/s (Z1 17.62), SmolLM2 qhf 2.27 tok/s (Z1 2.22),
generic 1.88 tok/s (Z1 1.87); DMA 397 MB/s.

Board setup the tests need (each boot):

```sh
# u-dma-buf (https://github.com/ikwzm/udmabuf, built on the board with the kernel headers):
# the runtime's window, since /dev/mem refuses RAM (CONFIG_STRICT_DEVMEM=y)
sudo insmod ~/udmabuf/u-dma-buf.ko udmabuf0=536870912     # 512 MB in CMA (0x37f00000 here)
cd ~/kv260 && sudo ./run_board.sh                          # the PYNQ venv; xmutil unloadapp by the driver
```

## K1b on the board (2026-10-08): passed

The same regression with the `d8_100mhz` overlay: `REGRESSION PASS`, every
run's tokens identical to the PYNQ-Z1 baselines. Wall time per decode step
halves (the device time dominates; the A53 host adds little):

| Run | Z1 (50 MHz) | K1a (50 MHz) | **K1b (100 MHz)** | K1b device cycles / Z1 |
|---|---|---|---|---|
| stories15M decode (c3) | 17.62 tok/s | 18.27 | **35.44** | |
| stories15M prefill + decode | 17.41 | 18.06 | **34.84** | 2,719,808 / 2,709,281 (+0.39%) |
| SmolLM2 qhf decode (c55) | 2.22 | 2.27 | **4.43** | |
| SmolLM2 qhf prefill + decode | 2.17 | 2.22 | **4.30** | 22,272,216 / 22,220,320 (+0.23%) |
| SmolLM2 generic (hfgen) | 1.87 | 1.88 | **3.73** | 26,729,622 / 26,671,269 (+0.22%) |

DMA (`bwtest`): contiguous LD / ST 794 MB/s (99.2% of 8 B/cycle), LD + ST
concurrently 1574 MB/s. The DDR latency is fixed in ns, so it costs twice
the cycles at 100 MHz: 1-beat bursts drop from 47.7% to 38.0% of 8 B/cycle,
and the LLM runs take 0.2-0.4% more cycles than at 50 MHz.

## K1c on the board (2026-10-08): passed

The regression with the `d16_100mhz` overlay (D = 16): every LLM test
(`c1`, `c3`, `c55`, `c6p_stories`, `c6p_smollm2`, `hfgen`) is bit-exact with
the D = 16 sim; `ddr`, `bwtest` and `desc` pass. Tokens vs the Z1 baselines:
`c3` and `hfgen` identical, the others differ from token 16-22 on (the fp32
reductions group by D lanes; expected). The `gemm` and `vector` demos failed
on D = 8-only shapes (fixed in `6658deb`, which skips them); rerun on the
board, `fwdemo` passes. GEMM 256x256x256: 202.6 MAC/cycle (79% of the 256
peak; K1b 58.9).

| Run | K1b (D = 8) | **K1c (D = 16)** |
|---|---|---|
| stories15M decode (c3) | 35.44 tok/s | **39.36** |
| SmolLM2 qhf decode (c55) | 4.43 | **4.77** |
| SmolLM2 qhf prefill (M = D) | 14.04 | **22.22** |
| SmolLM2 generic decode (hfgen) | 3.73 | **4.39** |
| SmolLM2 generic prefill | 6.84 (M = 8) | **10.04** (M = 16; 4.17 with M = 8) |

Decode gains are small: the array consumes D bytes of weights per cycle but
the read path is still 8 B/cycle (K2a). The generic path's prefill first ran
slower (4.17): its M = 8 export missed the prefill micro-kernel (rows must be
a multiple of D) and fell back to one GEMV per row (8x the EX steps);
`564a0de` exports M = D (10.04 on the board, bit-exact). The rest is its
gate / up matmul's fused SwiGLU epilogue on the fp VE / SFU (30% of the
prefill), which the qhf path does not have (22.2).

## To confirm on the board

- The carrier's fan is controlled from the PL in AMD's reference designs;
  this design drives no PL pin. Check the fan's behaviour (speed, noise,
  temperature) with this overlay loaded before long runs.

Confirmed by K1a / K1b: the overlay loads on Ubuntu 22.04 for Kria with
Kria-PYNQ (after `xmutil unloadapp`, done by the driver), and the RISC-V
reset through EMIO GPIO[0] (`GPIO.get_gpio_pin(0)`) works.

# PYNQ-Z1 → Kria KV260 upgrade plan

Status: draft (2026-10-02, revised after three rounds of pre-implementation review: K1 split into two steps at 50 / 100 MHz, K2 changed to a restructuring of the DMA and on-chip memory interfaces, K4 split into correctness and performance acceptance). **2026-10-04: Z1 frozen** (§7.2 done: tag `v1.0-pynq-z1`, branch `pynq-z1`, baseline files in `tests/baselines/pynq-z1/`, including the measured time breakdown `z1_profile.md`); all later optimization happens on the KV260, including the compiler's lifetime-based memory planning (a compiler work item of K2b). **2026-10-08: K1 accepted** (K1a 50 MHz, K1b 100 MHz, board token sequences identical to the Z1 baseline; at 100 MHz stories15M 35.44 token/s, SmolLM2 4.43 token/s); the `kv260` branch merged into main, which is the KV260 version from then on (§7.3). **2026-10-10: K2b accepted as a 100 MHz streaming GEMV (H4) + 4 read ports** (D = 16 unchanged, no longer premised on D = 32; design in [`k2b_gemv_design.md`](k2b_gemv_design.md)): decode Qwen3-0.6B 2.22 → **6.28 token/s**, SmolLM2 7.71 → 17.04, prefill unchanged; K5-M step 1 (a 1280 MiB accelerator memory pool reserved at boot) done. **2026-10-13**: through the generic frontend ([`generality_implementation_plan.md`](generality_implementation_plan.md) §13.2) SmolLM2-135M, Gemma-3-270M, Qwen3-0.6B and Llama-3.2-1B run whole on this overlay, every dispatch on the accelerator: decode 22.1 / 15.0 / 7.0 / 4.0 token/s, prefill 60.5 / 62.6 / 20.8 / 15.1 token/s. This document gives the complete plan for moving from the PYNQ-Z1 (Zynq-7020) to the Kria KV260 (K26 SOM, Zynq UltraScale+): staged goals, the changes to each module, acceptance criteria and risks. Performance numbers are estimates except where marked "measured", and need verification on the board.

Related documents: [`iree_compiler_plan.md`](iree_compiler_plan.md) §8.18 (FPGA size estimates), [`double_buffer_design.md`](double_buffer_design.md) (accelerator architecture), [`memory_model.md`](memory_model.md) (addresses and coherence).

---

## 0. Goals and principles

**Goals**

1. All existing functionality runs bit-exactly on the KV260 (stories15M, SmolLM2-135M, prefill + decode).
2. Raise decode's bottleneck, weight read bandwidth, from about 0.39 GB/s to a **measured effective bandwidth into the SPAD of 6–10 GB/s**. This needs a restructuring of the DMA receive side and the on-chip memory write interface (K2), not just a wider data path.
3. Get **Qwen3-0.6B (int8, all 28 layers)** running correctly on the board first, then tune from the measured time breakdown (10–15 token/s is the stretch goal; at 6 GB/s the upper bound from weight reads alone is about 10 token/s).
4. Leave room for later int4, 64-bit addresses, larger models and an RVV core.

**Principles** (following project conventions)

- **Change one kind of thing at a time**: first the platform (clock and parameters unchanged), then the clock, then the bandwidth architecture, then memory, and finally the large model. A problem at any step can be traced to that step's changes.
- **Correctness and performance accepted separately**: the platform port and the large model on the board are accepted first on bit-exactness; performance targets are set from measurements once the hardware is stable.
- **Each step: sim first, then the board, commit once bit-exact**. Bit-exact reference: the functional simulator (`llm/sa_funcsim.py`) and the results verified on the PYNQ-Z1.
- **K1 does not touch the compiler**: hardware parameters are passed in through the target configuration (`--iree-sa-d`, `--iree-sa-spad-kb`, `--iree-sa-acc-kb`), and the golden corpus (4002 dispatches) guards the code generation against changes. **K2b and K3 have explicit compiler work** (alignment, sub-bank conflicts, read-port assignment, chunk choice), listed under those stages.
- **The PYNQ-Z1 is frozen** (2026-10-04): tag `v1.0-pynq-z1`, branch `pynq-z1`, baseline in `tests/baselines/pynq-z1/`. Until K1 is done main is still the Z1 version; the new platform lives on the `kv260` branch with new directories and parameters, without breaking existing bitstreams and tests; after K1's acceptance main moves completely to the KV260 (see §7; merged 2026-10-08).

---

## 1. Platform comparison

| Item | PYNQ-Z1 (current) | KV260 (K26 SOM) |
|---|---|---|
| Device | Zynq-7020 (xc7z020clg400-1) | Zynq UltraScale+ (XCK26, comparable to ZU5EV) |
| Processor | 2 × Cortex-A9 (armv7, 32-bit), 650 MHz | 4 × Cortex-A53 (aarch64), about 1.3 GHz; also 2 × R5F |
| Logic resources | About 53K LUT, 220 DSP48E1, 140 BRAM36; LUT about 83% used | About 117K LUT, 1248 DSP48E2, 144 BRAM36, **64 URAM (about 2.25 MB)**. The BRAM count is close to Z1's; the new memory capacity comes mainly from URAM |
| DDR | 512 MB DDR3 | **4 GB DDR4 (64-bit)**, about 19 GB/s theoretical |
| PS-PL data ports | HP0–HP3, each **64-bit**, AXI3 | HP0–HP3 (and HPC), each **128-bit**, AXI4 |
| Accelerator clock | 50 MHz (measured timing constraint) | K1a 50 MHz → K1b 100 MHz → K2 target 200–250 MHz |
| Weight read bandwidth | **Measured about 0.39 GB/s** (1 HP port, about 7.8 B/cycle, 99% utilization; the 64-bit × 50 MHz LD entry caps it at 0.4 GB/s) | Target: effective bandwidth into the SPAD of 6–10 GB/s (multi-port receive + banked memory, see K2) |
| Software | PYNQ (armv7 Linux) | Kria-PYNQ or Ubuntu for Kria (aarch64) |

Resource numbers come from public material, checked against AMD's data sheets.

---

## 2. Platform-dependent parts of the existing design

An item-by-item inventory, classified as "unchanged / small change / rewrite".

### 2.1 Hardware (`rtl/`, `RISCV-on-PYNQ-Z1/`)

| Module | Class | Notes |
|---|---|---|
| `rtl/sysarray/*.v` (array, VE, LD/ST, scheduler, scoreboard) | Small change | Generic Verilog. `sa_unit` already has the parameters `D`, `DSP_COLS`, `NPORTS`, `SPAD_WORDS`, `ACC_WORDS`, `PERF`. DSP inference changes from DSP48E1 to DSP48E2 (usually automatic), and the DSP/LUT column split of `DSP_COLS` needs re-evaluation; the BRAM inference templates (`sa_tdpram.v`, `sa_bankmem.v`) need their inference results confirmed on UltraScale+. Larger SPAD/ACC use URAM, but `sa_tdpram.v` is a read-first BRAM template with 1-cycle read latency and power-up clear, while URAM's port behavior, read latency and initial values differ, so the memory interface has to be abstracted first (K3) |
| `rtl/sysarray/sa_ld.v`, `sa_st.v` (DMA) and the SPAD/ACC write interface | Unchanged in K1 → **restructured in K2** | Today each port carries 64 data bits (`m_rdata[NPORTS*64-1:0]`), the return side selects only one port per cycle (`r_sel`), and local writes take one 64-bit lane per cycle; more ports only hide latency, receive bandwidth does not add up. K1 configures the HP ports directly as 64-bit, so no width converter is needed; K2a widens the entry to 128 bits, K2b adds multi-port receive + banked memory |
| `picorv32.v`, PCPI (`sa_pcpi.v`) | Unchanged | Pure RTL |
| `scripts/pico_bit.tcl` (block design) | **Rewrite** | The device is hard-coded as `xc7z020clg400-1`; the PS is `processing_system7` and becomes `zynq_ultra_ps_e`; GP0 → HPM0_FPD/LPD; HP0/HP2 (64-bit, with the AXI4→AXI3 converter `matmulHpConverter`) → HP0–HP3 (128-bit AXI4, no AXI3 conversion needed) |
| `scripts/build_bitstream.tcl/.sh` | Small change | Add `-board kv260`; keep `-sa_d`, `-jobs` |
| `constrs/PYNQ-Z1.xdc` | Rewrite | The K26 needs only a few constraints such as clocks (the SOM's PL pins are basically unused) |
| RISC-V reset (EMIO GPIO[0]) | Small change | ZynqMP has EMIO GPIO too, with different numbering and Linux driver; or switch to AXI GPIO |
| Completion interrupt (`mat_notify`) | Small change | Connected to `pl_ps_irq0`; device tree / UIO configuration changes accordingly |

### 2.2 Firmware (`firmware/`)

| Module | Class | Notes |
|---|---|---|
| `rt/rt_fw.c` (resident command processor), `include/sysarray_intrinsics.h` | Unchanged | RISC-V-side addresses (BRAM `0xC0000000`, CSR `0x80000000`) do not depend on the ARM |
| `common/link.ld`, `include/mailbox.h` | Unchanged (to be confirmed) | The mailbox is at BRAM `0x1F00`, unchanged on the RISC-V side; the ARM-side base address is in the driver |
| Other milestone firmware (`gemm/`, `bwtest/` …) | Unchanged | K1 uses them for regression and bandwidth tests |

### 2.3 Host software

| Module | Class | Notes |
|---|---|---|
| `driver/pynq_matmul.py`, `notebooks/` | Small change | Base addresses such as `BRAM_ARM_BASE = 0x40010000` change to ZynqMP's HPM address windows (e.g. from `0xA000_0000`, as per the block design); overlay name, CMA allocation |
| IREE runtime cross-compilation (`iree-sa/l0/toolchain-armv7hf.cmake`, `build_runtime_armv7.sh`, `compiler/scripts/build_sa_runtime.sh armv7`) | Small change | New aarch64 toolchain file and `build_sa_runtime.sh aarch64`; `armv7_libm_shim.c` may no longer be needed on aarch64 (to be confirmed) |
| `compiler/runtime/sa/sa_transport_board.c` | Small change | Maps the device window through `/dev/mem` (`SA_BOARD_MEM=<phys>:<bytes>`, `SA_BOARD_MBOX=<phys>`); physical addresses are currently `uint32_t` (`sa_devmem_map(uint32_t phys, ...)`). Change to 64-bit parsing and checks; descriptors still use 32 bits, and the window must be in the low 2 GB (see 2.4) |
| `compiler/tests/board_*.py`, `scripts/deploy_*.sh` | Small change | Paths, architecture, window size, mailbox address |

### 2.4 Key constraint: 32-bit DDR addresses

- The DDR address field of descriptors is 32 bits (`DescList`: `ddr & M32`), PicoRV32 is a 32-bit core, and the runtime passes physical addresses as `uint32_t`.
- The KV260's 4 GB of DDR sits in two segments of the ZynqMP address space: **the low 2 GB from `0x0000_0000`, the high 2 GB from `0x8_0000_0000`**.
- **The approach for K1–K4: every buffer for the accelerator (weights, KV cache, activations, descriptor lists, ring queue) lives in the low 2 GB.** That way the descriptor address format need not change right away.
- **Order of address checks**: physical addresses (from environment variables, the driver or the allocator) are always parsed as a 64-bit type (`uint64_t`) first; only after checking that **the whole range** `[phys, phys + size)` lies within the reserved region and ends ≤ 2 GB are they converted to 32 bits for the descriptors. Truncating before checking would silently fold high-2 GB addresses back to low addresses.
- **Allocation and mapping**: reserved-memory only sets memory aside; it does not mean existing drivers can allocate and map it directly. To be made explicit:
  - the allocation method (device-tree reserved-memory + `/dev/mem`, CMA, or a driver like udmabuf);
  - mapping attributes (non-cached / write-combining / cacheable plus explicit flushes on the ARM side);
  - the CPU ↔ device synchronization points (what to do before submitting descriptors and before reading results).
  These go into the KV260 section of `docs/memory_model.md`.
- **ABI audit**: on aarch64 `long` and pointers are 64-bit. Structures shared by host and firmware (mailbox, ring entries, completion records, descriptors) **all use fixed-width fields** (`uint32_t` etc.), with no pointers, `long` or `size_t`; `static_assert` checks sizes and field offsets, confirming alignment and padding match the firmware. Merely recompiling the host program does not prove the ABI is unchanged.
- Capacity is sufficient: Qwen3-0.6B int8 weights are about 0.6 GB; plus INT8 KV (about 112 MiB for 2048 tokens, excluding scales and padding) and activations, far below the available space.
- 64-bit addresses are deferred to K5 (optional), needed only for 1.7B int8 or 3B+ models.

### 2.5 Compiler (`compiler/plugins/sa/`)

| Module | Class | Notes |
|---|---|---|
| sa backend plugin, `sahl`/`sahw` pipeline | Unchanged in K1; reviewed from K2 | Hardware parameters come from the target configuration. If K2 / K3 change the bank organization, read latency, DMA alignment or parallelism, the target configuration, memory planning, prefetch and scheduling need matching updates to use the new hardware |
| `TargetConfig::invalid()` | Depends | Currently requires D ∈ {8, 16} and on-chip memories of at most 2^16 words (16-bit local word addresses). SPAD/ACC beyond this or D = 32 need simultaneous changes to the descriptor encoding, the RTL and this check (see K3) |
| Host fallback (VMVX) | Unchanged | Independent of the CPU architecture |
| Functional simulator, `dispatch_check.py` | Unchanged | Already supports any D and memory size |

---

## 3. Staged plan

| Stage | Content | Main risk | Effort |
|---|---|---|---|
| **K0** | Preparation: pin the system image, boot firmware, Vivado / Kria-PYNQ versions; keep a reproducible Z1 baseline | Version combinations | Small |
| **K1a** | Platform port with the accelerator **completely unchanged**: D = 8 (Z1's L2 bitstream), 1 64-bit HP port, **50 MHz**, original memory capacity | Block design, address map, aarch64 runtime | Medium |
| **K1b** | Raise only the clock to 100 MHz; re-record timing and performance | Timing | Small |
| **K1c** | Change only D from 8 to 16 (on Z1 the LUTs only allowed a D = 8 LLM accelerator) | Resources, timing; the VE's fp lane folding (FL) changes with D | Small |
| **KC** | Compute side (H1, H2 of §3.1): multi-group interleaving in the VE sequencer mode (P6), out-of-order issue / fine-grained conflict checks in the scheduler; after K1c, can run in parallel with K2 | Bit-exactness with the functional simulator; scoreboard correctness | Medium |
| **K2a** | Single 128-bit port: widen the LD entry, DMA and SPAD writes; measure the end-to-end effective DMA → SPAD bandwidth | Changes to the DMA and memory write interfaces | Medium |
| **K2b** | Multi-port receive + banked memory, working toward 6–10 GB/s; **with D = 16 → 32 at the same time** (in decode the array consumes only D bytes of weights per cycle, see §3.1). **What was actually done**: D = 16 unchanged, decode goes through a streaming GEMV unit (H4) + 2 / 4 read ports, 4 ports at 64 B / cycle = 6.4 GB/s, accepted on the board 2026-10-10 (`k2b_gemv_design.md`) | **Restructuring of DMA + on-chip memory interfaces**, D = 32 resources and timing, memory planning | Large |
| **K3** | URAM to extend SPAD/ACC capacity (after K2b with D = 32: SPAD words of 256 bits, ACC words of 1024 bits) | URAM port semantics, initial values, read latency | Medium |
| **K4a** | Qwen3-0.6B whole model running correctly on the board | Addresses and capacity, long contexts | Medium |
| **K4b** | Long-context tiling (C6.5) and performance tuning, targets set from the measured time breakdown | Attention T tiling, scheduling | Medium |
| **K5** | Optional extensions: int4, 64-bit addresses, RVV core, MoE, each evaluated separately | Independent | Large |
| **Q (parallel)** | Qwen3 quantization quality (C6.2), on the CPU / sim, not waiting for K4 | Mathematical equivalence of quantization methods and scale semantics | Medium |

### 3.1 Hardware improvement items (Z1 measurements + RTL confirmation, 2026-10-04)

Basis: the event counters in `tests/baselines/pynq-z1/z1_profile.md` (SmolLM2, per decode step / per prefill chunk, hand-written path / generic path), each confirmed in the RTL.

| No. | Improvement | Evidence (counters + RTL) | Cost | Stage |
|---|---|---|---|---|
| **H0** | **D grows with bandwidth** (D = 16 → 32) | In decode `EX_STEP` almost equals bytes read ÷ 8 (SmolLM2 hand-written 17.36M vs 17.30M, stories 2.00M vs 1.98M). `sa_ex.v`: one SPAD_B word read per step (`sb_addr = b_base + c`, D bytes = D output columns of one k row); decode's A strip is x replicated to D rows (VE's DIV replication), the D rows compute the same result, array utilization 1/D. So weights consumed per cycle = D bytes: D = 8 / 16 / 32 cap at 2 / 4 / 8 GB/s at 250 MHz. Each tile also has 2(D−1) cycles of fill / drain (at K = 576 about 2.4% for D = 8, about 11% for D = 32) | See K3 item 6 | D = 16: K1c; D = 32: K2b |
| **H1** | **Multi-group interleaving in the VE sequencer mode** (P6) | `sa_vefp.v`: EXP / RECIP / RSQRT and REDUCE use the sequencer mode with **only one group in flight at a time** (`cap = seqmode ? 1 : OQ - 4`), the microprogram waiting out the M2 / A2 pipeline latency step by step; the streaming mode already allows 28 groups in flight and is not the bottleneck. Bit 26 of counters 24..27 on the fp VE is `sq_act` (sequencer working, ORed with the integer VE's "output FIFO full", named `VE_CREDIT`): decode 1.56M / 2.65M, prefill 12.2M / 19.4M, **51–58%** of VE busy time. Port conflicts `VE_RDBLOCK` ≈ 0 | Small to medium: the sequencer advances several groups at once (microprogram state per group), reusing idle pipeline stages of the existing fp units. **Passed on the board (2026-10-09, `d16_100mhz_w128_p6`)**: batched mode (EXP / RECIP / RSQRT without REDUCE) about 4.5× per group; Qwen3 prefill 89.9M → 70.0M cycles per chunk, 11.4 → 14.0 token/s, bit-exact (`iree_compiler_plan.md` P6); with REDUCE batched (`d16_100mhz_w128_p6_red`) further to 62.2M cycles, 15.45 token/s | KC |
| **H2** | **Scheduler: out-of-order issue + fine-grained conflict checks** | `sa_sched.v`: dispatches strictly in program order, and the scoreboard only has the granularity of 6 banks (two halves each of SPAD_A, SPAD_B, ACC); when the head is blocked everything behind it is blocked. Head-of-queue blocking per prefill chunk (hand-written): LD 13.2M, ST 11.3M, engine queue full 9.3M — the next block's weights cannot be read ahead during the VE tail; on the generic path the serialization of a decode linear layer's tail with EX (about 4.0M) has the same origin. Frontend starvation `STARVE` and fully idle are low; issue itself is not the problem | Medium: conflict checks by address range (instead of 6 banks), letting non-conflicting later commands pass a blocked head (independent issue windows per engine); correctness compared command by command with the functional simulator | KC |
| **H3** | **VE width** (physical fp lanes FL) | `sa_vefp.v`: `FL = D / 2` (only D / 2 or D supported), **grows with D**; the `sa_vefp` of L2 (D = 8, FL = 4) is about 17.2k LUT, roughly proportional to FL, so D = 32 with FL = 16 is about 69k LUT (estimate). Streaming-mode throughput grows with FL; the sequencer mode is limited by H1 | If LUTs are short at D = 32, support FL = D / 4 (4 half-beats per group, an RTL change) | Evaluated with H0 |
| **H4** | **A matrix-vector path dedicated to decode** (an alternative or complement to H0's D = 32) | H0: in decode only 1/D of the array's multiply-accumulates are useful, so D = 32 trades 1024 multiply-accumulates for 32 bytes per cycle; an int8 dot-product unit doing 32 bytes per cycle needs only a few dozen. Or fill the other rows with batched decode / speculative decoding (software) | Medium: one more unit and compiler path; D = 32 is more general if prefill also needs speeding up | Compared before K2b |
| **H5** | **int4 weights, unpacked on the LD path** | decode is limited by reads: half the bytes, double the ceiling | Medium to large, with quantization-quality issues (GPTQ / AWQ family) | K5 |
| **H6** | **Keeping intermediate results on chip** (URAM) | Activations between dispatches do not go through DDR; the fixed overhead is only 1–2%, a moderate gain | Medium | K3 |

Not needing changes: the issue frontend (`STARVE` 0.4M, fully idle 0.3M), the VE memory ports (`VE_RDBLOCK` ≈ 0), AXI read back-pressure (`LD_ARSTALL` ≈ 0 on Z1).

**Order**: H1 and H2 are cheap and directly target the part not hidden by reads; do them right after K1c (the KC stage; the gains are measurable at 100 MHz); H0's D = 32 goes together with K2b's bandwidth, with H4 compared as an alternative before that; H3 is evaluated with H0; H5 in K5 (after Qwen3's quantization quality is solved). Redo the time breakdown at every step (`board_profile.py` + `SA_PROFILE_PERF`, `compiler/tests/z1_profile.py`).

**A note on counter naming**: `PC_VE_CREDIT` (bit 26) on the fp VE counts the sequencer working (`sq_act`), not the output FIFO being full; `PC_VE_GROUPS` (bit 27) on the fp VE counts words written (including TRANSPOSE). Interpret them this way when analyzing LLM workloads.

### K0 Preparation

1. **Tools**: install a Vivado / Vitis version that supports the K26, and download the KV260 board files. Confirm that the free license of the Vivado version in use covers the XCK26.
2. **System image and boot firmware**: Kria-PYNQ (recommended; the Python driver and notebooks can mostly be reused) or Ubuntu for Kria, written to microSD; **record and pin** the image version, the boot firmware version and the matching Vivado version in `boards/kv260/README.md`. At power-up confirm Linux, the network and `/dev/mem` access rights.
3. **ARM baseline**: run the ARM CPU baseline on the board (corresponding to `notebooks/llm/arm_baseline.sh`), recording llama inference token/s on the A53 as the accelerator's comparison point.
4. **Repository structure**: create `KV260/` on the `kv260` branch (alongside `RISCV-on-PYNQ-Z1/`) for the block design scripts, constraints, build scripts and bitstream result directories. Reorganize directories after K1's acceptance (§7.4).
5. **Z1 baseline**: before starting K1, complete §7.2's Z1 freeze and baseline files.

**Acceptance**: the board boots and can load an empty overlay with PYNQ (or Python + `/dev/mem`); the version combination is recorded.

**Progress** (2026-10-07): the board has arrived. Image Ubuntu 22.04.4 (Ubuntu for Kria, kernel 5.15.0-1027-xilinx-zynqmp), boot firmware K26-BootFW-01.02 / U-Boot 2023.01 (A / B identical); DDR low segment `0x0000_0000`–`0x7FEF_FFFF`, CMA 1000 MiB at `0x3780_0000` (entirely in the low 2 GB); `k26-starter-kits` is loaded at boot, and `xmutil unloadapp` is needed before loading an overlay (the driver and `ddr_test.py` handle it automatically). The version combination is recorded in `KV260/README.md`. Vivado 2024.1 supports the XCK26 (confirmed). To do: install Kria-PYNQ, the A53 CPU baseline, loading an overlay. The regression test bundle has gained K1a verification steps 1, 3, 4 (`ddr`, `fwdemo`, `c1`).

### K1a Platform port (accelerator completely unchanged)

The goal is to close the loop on the KV260 **with the same accelerator as Z1**, separating platform problems from timing problems: if this step fails, it can only be a platform problem (block design, addresses, runtime).

**Hardware**

1. New block design (`KV260/scripts/kv260_bd.tcl`):
   - `zynq_ultra_ps_e` with HPM0_FPD (ARM → BRAM, CSR), S_AXI_HP0_FPD (accelerator DMA), `pl_clk0` (**50 MHz**, as on Z1), `pl_ps_irq0`, EMIO GPIO (RISC-V reset) enabled;
   - PicoRV32 (`picorv32_axi`), program BRAM (8 KB, dual port for the ARM side and the RISC-V side), `sa_unit` (**D = 8**, `NPORTS = 1`, SPAD/ACC as in Z1's L2 bitstream; the L2 that ran LLMs on Z1 is D = 8, and the launcher of `board_llm.py` prints `overlay D = 8`);
   - **HP0 configured directly as 64-bit** (ZynqMP's HP ports support 32 / 64 / 128 bits); the accelerator's 64-bit AXI master connects directly, with no width converter and no AXI4→AXI3 conversion as on Z1;
   - PicoRV32's DDR port (formerly HP0, used to read the ring queue) on HP1 or shared through a SmartConnect.
   - **Clock domains**: as on Z1, PicoRV32 and the accelerator share **one clock** (in Z1's `pico_bit.tcl` both `riscv_clk` and `matmul_0/aclk` connect to `subprocessorClk`), and the PS-side AXI-Lite crosses clock domains through the interconnect. Later clock increases raise both together: PCPI is a tightly coupled interface, and splitting it into two clock domains needs clock-domain crossing on both PCPI and the command path. Splitting is evaluated only if a path in PicoRV32 or PCPI (`sa_pcpi.v`) becomes critical in the full implementation.
   **Written** (2026-10-04, `kv260` branch): `KV260/scripts/kv260_bd.tcl` + `build_bitstream.{tcl,sh}` (`-sa_d`, `-sa_mhz`, `-bd_only`); the block design validates in Vivado 2024.1 (`-bd_only`); differences from Z1: only one PL clock `pl_clk0` (no clk_wiz), the accelerator DMA directly on `S_AXI_HP1_FPD` (AXI4, no protocol conversion), no PL pins. **Full build passes** (2026-10-04): LUT 34.5%, DSP 115, BRAM36 130 / 144 (90%), URAM 0; at 50 MHz setup WNS +9.05 ns, hold WHS +0.010 ns, `check_timing` shows no unconstrained paths. The worst path goes from the DMA port's (HP1) read data through a DSP multiply to `fetch/param_reg` of the descriptor fetch unit (data path 10.5 ns), good for about 90–95 MHz as is; **K1b's 100 MHz needs one more register stage on this path**. The build runs in one Vivado process (on this 15 GB machine, child processes synthesizing IPs separately get killed by systemd-oomd). See `KV260/README.md`.
2. Address map: record the new ARM-side addresses of the BRAM, CSRs and mailbox in the KV260 section of `docs/memory_model.md` (**written**, including `STRICT_DEVMEM` and the u-dma-buf window). ARM side: BRAM `0xA001_0000` (mailbox `0xA001_1F00`, perf area `0xA001_1E00`), interrupt controller `0xA002_0000`; the RISC-V and accelerator sides are the same as Z1, DDR in the low 2 GB.
3. Add a K26 device option to `synth_ooc.tcl` and do out-of-context synthesis first to confirm resources.

**Software**

4. aarch64 cross-compilation: `toolchain-aarch64.cmake`, `build_sa_runtime.sh aarch64`, building `sa-llm-run`, `sa_hal_test` and L0's `iree-run-module`. **Done** (2026-10-04, `kv260` branch): `compiler/runtime/toolchains/aarch64-linux-gnu.cmake`, all three programs statically linked; a board-free check `compiler/scripts/test_aarch64_runtime.sh` (qemu-aarch64 user mode + the ring-queue emulator `sa_board_emu.py`): the C1 HAL test PASSes, and stories15M decode and prefill + decode logits are bit-exact with the reference.
5. **Addresses and ABI** (details in §2.4): `sa_transport_board.c` parses physical addresses with a 64-bit type and converts to 32 bits only after checking the whole window lies in the reserved region; make the allocation of the reserved region, mapping attributes and synchronization points explicit; audit the structures shared by host and firmware. The runtime part is **done**: `SA_BOARD_*` are parsed as 64-bit, and the 32-bit device address is formed only after checking the whole window (non-empty, end ≤ 2 GB); the mailbox can be at any physical address; the only things shared with the firmware are ring entries (64-bit words) and completion records (32-bit words), written by offset, with no structures, pointers or `long`. The allocation method and mapping attributes of the reserved region are decided once the board arrives.
6. Driver and deployment: `pynq_matmul.py` and `deploy_*.sh` gain `--board kv260` and read the new base addresses. **Done** (2026-10-04, `kv260` branch): the driver reads the board, D, the ARM-side BRAM address and the accelerator clock from the `.hwh` (`overlay_info`), and the launcher, bwtest and `ddr_test.py` (addresses from the overlay's `ip_dict`) no longer hard-code Z1's addresses; the deployment scripts use `SA_BOARD=kv260` (`compiler/scripts/board_env.sh`: the aarch64 runtime, the overlay in `KV260/build/output/<config>`, `SA_KV260_CONFIG` selects the configuration, default `d8_50mhz`, changed to `d8_100mhz` after K1's acceptance, and `SA_BOARD` defaults to `kv260` too; build outputs go into per-configuration directories such as K1b's `d8_100mhz`); the whole regression `deploy_z1_freeze.sh --board kv260` → `build/deploy_kv260`; board-free check: the aarch64 `sa-llm-run` from the KV260 test bundle runs under qemu with the KV260 addresses (mailbox `0xA001_1F00`, window below 2 GB), stories15M 22/22 rows bit-exact.
7. Compiler: unchanged; it produces the same modules as the Z1 baseline (D = 8, default SPAD/ACC), and the descriptors should be byte-identical to `tests/baselines/pynq-z1/golden_manifest.txt`.

**Verification order** (each bit-exact with the sim and compared with §7.2's Z1 baseline files; the test bundle and board scripts reuse `compiler/scripts/deploy_z1_freeze.sh` and `compiler/tests/board_regress.py` with `--board kv260`)

| Step | Test | Expected |
|---|---|---|
| 1 | `tests/ddr_access`: RISC-V reads and writes DDR, including the first and last addresses of the reserved region | Passes |
| 2 | `bwtest`: DMA bandwidth | Recorded; should be close to Z1 (both 64 bits × 50 MHz, entry limit 0.4 GB/s) |
| 3 | `gemm`, `vector`, `desc_run` firmware | Bit-exact with NumPy |
| 4 | C1: `sa_hal_test` (hand-written template) | Bit-exact |
| 5 | C3: stories15M (`board_llm.py`) | 76/76 steps bit-exact with DeviceModel, token sequence identical to the Z1 baseline |
| 6 | SmolLM2-135M prefill + decode | Bit-exact with the sim, token sequence identical to the Z1 baseline |

**Acceptance**: steps 5 and 6 pass. **Accepted on correctness only**; token/s is recorded as an observation (the ARM host, DDR latency and driver overhead have all changed, so no ratio to Z1 can be assumed).

**K1a accepted** (2026-10-08, on the board, `d8_50mhz` overlay, `compiler/scripts/deploy_z1_freeze.sh --board kv260` + `run_board.sh`): steps 1–6 all PASS (`ddr`, `bwtest`, gemm / vector / desc_run, C1, C3, SmolLM2 and two prefill + decode paths); `compiler/tests/compare_z1_baselines.py`: the token sequences of 5 tests identical to the Z1 baseline. Device cycles within 0.03% of Z1 (per decode step: stories15M 2,708,522 / Z1 2,709,281, SmolLM2 qhf 22,215,268 / 22,220,320, generic path 26,663,452 / 26,671,269), DMA 397 MB/s (99.3%); wall clock 1–4% faster (A53 host: stories15M 54.7 ms/step = 18.27 token/s, Z1 56.8; SmolLM2 qhf 440.9 ms/step = 2.27 token/s, Z1 449.6).

Platform problems solved on the board: Kria-PYNQ has no `/etc/profile.d/xrt_setup.sh` (the test bundle carries `run_board.sh`, which finds the environment script automatically); the `k26-starter-kits` loaded at boot occupies the PL (the driver does `xmutil unloadapp` first); PYNQ lists the BRAM controller in `mem_dict` (`ddr_test.py` once fell back to Z1's address and wrote into DDR inside the CMA); **the kernel has `CONFIG_STRICT_DEVMEM=y`, so `/dev/mem` refuses to map memory even for root**: the runtime's window is now provided by the u-dma-buf module (`/dev/udmabuf0`, `sync_mode` 2 = write-combining; `SA_BOARD_MEM_DEV`), while PL addresses still go through `/dev/mem`.

### K1b Raising the clock to 100 MHz

1. Change only `pl_clk0` to 100 MHz, nothing else; record WNS / TNS and the critical paths.
2. Rerun steps 2–6 of K1a's verification order.

K1a's build showed the worst path (DMA read data → DSP multiply → `fetch/param_reg` of the descriptor fetch unit, data path 10.5 ns) only reaches about 90–95 MHz: before 100 MHz, add a register stage on this path (`sa_cmdfetch.v`), accepted on bit-exactness in functional simulation and on the board. **Changed** (2026-10-04, `kv260` branch): LDPARAM only latches the fetched word on the last beat and does the multiply-add in the next cycle (LDPARAM takes one more cycle, same semantics); `tb_sa_unit` passes entirely at D = 8 / 16 (including the LDPARAM checks).

Among the top 40 paths after K1a's routing (`post_route.dcp`; under a 20 ns constraint the tool did not try hard on them), once this one is fixed the next are: the VE TRANSPOSE address (`tr_word0`, DSP) → ACC BRAM address / write enable (data path 9.5–10.1 ns, 11 levels); LD's `q_rp` → ACC BRAM address (about 10.1 ns, 13 levels); RISC-V reset → synchronous reset of EX's `bsr_reg` (about 10.2 ns, 4 levels, high fan-out, mostly routing). Under a 10 ns constraint the tool optimizes (replicating the reset network, placing close by), and most should close; the actual 100 MHz build decides. **K2a's 200–250 MHz needs pipeline stages at all of these places**.

**100 MHz build passes** (2026-10-04, `-sa_mhz 100`): setup WNS +1.171 ns, hold WHS +0.010 ns, `check_timing` shows no unconstrained paths, resources the same as K1a (LUT 34.4%, BRAM36 130); the worst path is now LD's `q_lane` → DSP → ACC BRAM write enable (7.8 ns, 13 levels), good for about 113 MHz as is. What remains of K1b is board verification (steps 2–6).

**K1b accepted** (2026-10-08, on the board, `d8_100mhz` overlay, `SA_KV260_CONFIG=d8_100mhz compiler/scripts/deploy_z1_freeze.sh --board kv260` + `run_board.sh`): steps 2–6 all PASS, the token sequences of 5 tests identical to the Z1 baseline. Wall clock almost halved: stories15M decode 28.2 ms/step = 35.44 token/s (K1a 18.27, Z1 17.62), SmolLM2 qhf decode 225.5 ms/step = 4.43 token/s (K1a 2.27), generic path 268.3 ms/step = 3.73 token/s (K1a 1.88). DMA sequential read/write 794 MB/s (99.2% of 8 B/cycle), concurrent read and write 1574 MB/s. DDR latency is fixed in nanoseconds, so at 100 MHz the equivalent cycle count doubles: single-beat bursts drop from 47.7% to 38.0%, and LLM device cycles are 0.2–0.4% more than at 50 MHz (stories15M 2,719,808 / 2,708,522, SmolLM2 qhf 22,272,216 / 22,215,268, generic path 26,729,622 / 26,663,452). Host overhead is small, so almost all of the clock gain lands in wall clock. **K1 (K1a + K1b) accepted**, ready to merge into main per §7.3.

**Acceptance**: timing closes; steps 5 and 6 bit-exact. Performance is an observation: `bwtest`'s entry limit becomes 0.8 GB/s (64 bits × 100 MHz); decode token/s is expected to improve, but host overhead and DDR latency mean it is not necessarily twice Z1's.

### K1c D = 8 → 16

1. Change only `sa_unit`'s `D` (`build_bitstream.sh -sa_d 16`), nothing else; record resources and timing (Z1's D = 8 L2 already used 83% of the LUTs, `sa_vefp` alone 17.2k LUT).
2. Compiler: `--iree-sa-d=16`; the golden corpus already has D = 16 cases (`cfg_d16`), and the functional simulator and `dispatch_check.py` support any D. D = 16 modules have descriptors different from the Z1 baseline; accepted on bit-exactness with the sim, establishing the D = 16 configuration's own baseline (§5).
3. Rerun steps 2–6 of K1a's verification order with the D = 16 sim as reference (fp32 reductions are grouped by D lanes and may differ from D = 8 in the last bit, so token sequences identical to the Z1 baseline are not required; record whether they are).

**Acceptance**: steps 5 and 6 bit-exact; prefill token/s recorded as an observation (the array has 4× the multiply-accumulates; decode is still bandwidth bound). Later stages use D = 16 (one SPAD word is 128 bits, exactly one 128-bit beat, see K2a).

**Build passes** (2026-10-08, `-sa_d 16 -sa_mhz 100`, `4bd8389`): setup WNS +0.630 ns, hold WHS +0.011 ns; LUT 74,951 (64.0%), FF 59,944, DSP 347, BRAM36 130 / 144. All PE columns are on DSPs (`kv260_bd.tcl` sets `DSP_COLS = D`; Z1's D = 16 build had only 220 DSPs, so only 8 columns used DSPs). The worst path is of the same kind as K1b's: LD's `q_rp` → DSP → ACC BRAM write enable (8.4 ns, 12 levels).

**Software**: D follows the overlay. `board_env.sh` takes `SA_D` from the configuration name (`d16_100mhz` → 16); `compile_sa.sh` passes `--iree-sa-d`; in the tests the sim's D, the reference models (`DeviceModel(d = D)`, qhf's `valid` length) and `--pad` all use `SA_D`; C1's test executables are generated per D (`test_d16`). The prefill block size is M = D (`c6p_*`; the generic path `hfgen` is still M = 8, which is fixed in its export). Host builds with D ≠ 8 go into `build/<test>/<model>_d16`, not overwriting the D = 8 directories the golden corpus reads; the export is independent of D (the attention length is dynamic), and `seed_export` copies the D = 8 export. On the host: the three C1 items and C3 (stories15M 12/12 steps bit-exact with `DeviceModel(d = 16)`, 58 dispatches each checked OK) pass.

**A compiler bug found by D = 16 end to end** (`a1fefb6`): C5.5 (SmolLM2) on the D = 16 sim was correct at step 0 and diverged completely from step 1; the per-dispatch checks (`dispatch_check`) were all OK, but with `--dirty` (on-chip memories starting from garbage) 30 attention softmaxes (`reduction_3x3xD`, GQA's 3 × 3 rows) failed. Cause: in row mode (dynamic length, one row at a time), a VE whose length equals a whole row (`maxDynamic` = 256) takes the dynamic length LEN; the per-row broadcast (`perElementBcast`) first replicates a block of w × D × D and then transposes it, and at D = 16 with at most 16 rows that block is exactly 256 elements, so it also got LEN = T, wrote only the block's first word, and the broadcast rows read channels left over from the previous step (at step 0 the on-chip memory is still 0 and only channel 0 is used, so it happened to be correct). At D = 8 the block is 128 and does not collide. Fix: fixed-size auxiliary VEs (per-row broadcast blocks, scalar broadcast words) keep a static length in row mode. The golden corpus is unchanged; a new `smollm2_d16` group was added (4232 dispatches in total). Lesson: a new target configuration needs an end-to-end sim, and `dispatch_check --dirty`.

**Board results** (2026-10-08, `d16_100mhz` overlay, `SA_KV260_CONFIG=d16_100mhz compiler/scripts/deploy_z1_freeze.sh` + `run_board.sh`, bundle from `fcc75ee`): steps 5 and 6 all PASS (`c1`, `c3`, `c55`, `c6p_stories`, `c6p_smollm2`, `hfgen` bit-exact with the D = 16 sim), `ddr`, `bwtest` (792–794 MB/s, the same as K1b) and `desc` PASS. Compared with the Z1 baseline, `c3` and `hfgen` have identical tokens, while `c55`, `c6p_stories`, `c6p_smollm2` differ from tokens 18, 16, 22 on (different fp32 reduction grouping, as expected). The two demos `gemm` and `vector` failed in their scripts: shapes hard-coded as multiples of D = 8 (8×8×8, 24×16×40), and the driver raises ValueError at D = 16; the firmware itself passes in the D = 16 system simulation (19 GEMMs, 7 vector operations). `6658deb` makes the demos skip shapes that are not multiples of D and the 8×8×8 Phase 4 legacy path, and adds non-square cases also legal at D = 16; rerunning `fwdemo` on the board all PASS (`gemm`, `vector`, `desc`). GEMM 256×256×256 at 202.6 MAC/cycle (79% of the peak 256; K1b 58.9, 3.4×), 128³ at 156.3 (K1b 53.2). **K1c accepted**.

| Run | K1b (D = 8) | **K1c (D = 16)** | |
|---|---|---|---|
| stories15M decode (c3) | 35.44 token/s | **39.36** | +11% |
| decode of stories15M prefill + decode | 34.84 | **37.84** | +9% |
| SmolLM2 qhf decode (c55) | 4.43 | **4.77** | +8% |
| decode of SmolLM2 qhf prefill + decode | 4.30 | **4.50** | +5% |
| SmolLM2 qhf prefill (M = D) | 14.04 | **22.22** | +58% (16 tokens per chunk) |
| SmolLM2 generic path decode (hfgen) | 3.73 | **4.39** | +18% (device cycles 26.73M → 22.68M) |
| SmolLM2 generic path prefill | 6.84 (M = 8) | **10.04** (M = 16; 4.17 with M = 8) | +47% |

decode gains less than 2×: D bytes of weights are consumed per cycle, but the read bandwidth is still 8 B/cycle (§3.1 H0); D = 16 only pays off together with K2a's 128-bit read channel. **The generic path's prefill got slower**: its export fixes M = 8, which uses only half the rows of the D = 16 array per chunk, and the device cycles of prefill's matmuls (`8x96x16x576` etc.) were about twice those at D = 8 (30.3M vs 16.6M). Cause (board counters): the down projection's `LD_BEATS` is close to D = 8's (weights read once), but `EX_USEFUL` is 26.5M (3.3M expected at D = 16, 8×), `VE_ACTIVE` 27.5M. prefill's linear micro-kernel requires the row count to be a multiple of D (`p.rows % d` in `matchLinear` of `SahlKernels.cpp`); with M = 8 < 16 it falls back to the generic contraction, which does a decode-style GEMV for each row of x separately (x replicated to D rows, array utilization 1/D), 8 passes for 8 rows. Fix (`564a0de`): `deploy_hfgen.sh` exports M = D per D (`build/hfgen/smollm2_p<D>`; the export fixes M, so the D = 8 one is no longer copied); the D = 16 sim PASSes (prefill + decode vs decode only 5/5 rows bit-exact), `dispatch_check --dirty` 1306 OK (the other 2 are on the host). Rerunning `hfgen` on the board PASSes (bit-exact with the sim): prefill of 40 tokens in 3 chunks at 10.04 token/s (4.17 with M = 8, 6.84 at D = 8), decode 4.38 token/s; the down projection's `EX_USEFUL` is back to 3.3M. The big remaining item is the gate / up projection (`matmul_like_16x96x16x576`, 30 calls, 35.0M cycles, 30% of prefill): `VE_ACTIVE` 26.4M, `VE_CREDIT` 20.6M, limited by the fused SwiGLU tail (fp VE / SFU), independent of D (33.1M at D = 8). The qhf path's matmul at the same position has no such tail (`VE_ACTIVE` 1.2M), so its prefill is more than twice as fast (22.2 token/s). This belongs to the scope of P6 (SFU multi-slot interleaving) / KC. Letting the micro-kernel support M < D directly (loading / storing only M rows) is left as later compiler work: only exports with a fixed M hit it.

### K2 Bandwidth: DMA and on-chip memory interfaces

decode is limited by weight read bandwidth, and this stage is the main source of performance gains. **It is not just a wider DMA but a joint restructuring of the DMA and the on-chip memory write interface.**

**Current state (`rtl/sysarray/sa_ld.v`, `sa_unit.v`)**

- `sa_ld` can issue reads to `NPORTS` ports in turn (`ar_rr`), but **the return side selects only one port per cycle** (`r_sel`): `lw_data = m_rdata[64*r_sel +: 64]` is only 64 bits, one `lw_en` per cycle, and `m_rready` is raised only for the selected port. More ports only hide latency; receive bandwidth does not add up.
- Write side: each `lw_en` writes only one 64-bit lane of a SPAD word (`spad_we = 8'hFF << (8*lw_lane)`). At D = 16 a SPAD word is 128 bits and takes two cycles to fill.
- The two banks of `sa_bankmem` are split by the top bit of the word address for double buffering, not for parallel multi-port writes; one of side A's two requesters is already taken by ST's local reads (`lr_word`).

**Entry bandwidth limits** (zero stalls, entry width × clock; measurements can only be lower)

| Entry | 50 MHz | 100 MHz | 200 MHz | 250 MHz |
|---|---|---|---|---|
| 64-bit (current) | 0.4 GB/s | 0.8 GB/s | 1.6 GB/s | 2.0 GB/s |
| 128-bit (K2a) | — | 1.6 GB/s | 3.2 GB/s | 4.0 GB/s |
| 256-bit (K2b, 2 words/cycle) | — | — | 6.4 GB/s | 8.0 GB/s |
| 384-bit (K2b, 3 words/cycle) | — | — | 9.6 GB/s | 12 GB/s |

Z1 measured 0.39 GB/s, already right at the 64-bit × 50 MHz entry limit. A measured 6–10 GB/s means **writing 24–40 bytes, i.e. 2–3 SPAD words, into on-chip memory per cycle** at 250 MHz (D = 16).

**Target bandwidth versus configuration** (measured target ÷ entry limit = required utilization)

| Measured target | Configuration | Entry limit | Required utilization | Verdict |
|---|---|---|---|---|
| 6 GB/s | 2 words/cycle × 200 MHz | 6.4 GB/s | About 94% | Too little margin; not a planned configuration |
| 6 GB/s | 2 words/cycle × 250 MHz | 8.0 GB/s | 75% | Feasible |
| 6 GB/s | 3 words/cycle × 200 MHz | 9.6 GB/s | About 63% | Feasible |
| 10 GB/s | 3 words/cycle × 200 MHz | 9.6 GB/s | > 100% | **Not reachable** |
| 10 GB/s | 3 words/cycle × 250 MHz | 12 GB/s | About 83% | Needs high utilization |
| 10 GB/s | 4 words/cycle × 200 MHz | 12.8 GB/s | About 78% | Feasible, more resources |

K2b's target must be written together with its configuration: e.g. "2 words/cycle × 250 MHz, measured ≥ 6 GB/s", not just "6–10 GB/s". The usable bandwidth of the DDR controller itself may also be below the entry limit; `bwtest` measurements decide.

**After bandwidth: the parts other than reading weights**

Once bandwidth goes up, the cycles other than reading weights (VE tails, SFU, EX, the fixed overhead of each DMA and dispatch) do not shrink with it; DDR latency is constant in nanoseconds, so after a clock increase it is even more cycles. At the Z1 freeze, a time breakdown was made on the board with the accelerator's event counters (`tests/baselines/pynq-z1/z1_profile.md`, `compiler/tests/z1_profile.py`; per decode step, one list per dispatch):

| Path | Z1 cycles per step | Bytes read | Read floor (7.94 B/cycle) | Part not hidden by reads | All engines idle (fixed overhead) | K2b (24 B/cycle) read floor | Share of the rest after K2b |
|---|---|---|---|---|---|---|---|
| stories15M hand-written path | 2.71M | 15.8 MB | 1.99M | 0.72M (26%) | 0.06M (2%) | 0.66M | About 52% |
| SmolLM2 hand-written qhf path (2.22 token/s) | 22.22M | 138.4 MB | 17.44M | 4.78M (22%) | 0.28M (1%) | 5.77M | About 45% |
| SmolLM2 compiler generic path (1.87 token/s) | 26.67M | 140.9 MB | 17.75M | 8.92M (33%) | 0.30M (1%) | 5.87M | About 60% |

(The rest is estimated as constant in cycles; it may actually be more.) What the measurements show:
- **Fixed overhead is small** (all engines idle 1–2%): the descriptor lists and rt_fw's design already keep issue overhead down; it will not be a main problem after K2 either.
- **Linear layers sit at the read floor**: SmolLM2's `matvec`-class dispatches read 135.8 MB with LD busy 17.2M cycles, and EX's 17.2M overlaps LD almost completely (16.4M cycles of overlap).
- **The part not hidden**: of the hand-written path's 4.78M, linear-layer tails are about 1.6M (total `matvec` cycles minus the read floor), RMSNorm / softmax reductions about 1.7M, elementwise 0.9M, attention 0.46M; of the generic path's 8.92M, linear-layer tails are about 4.0M (the generic contraction's tail is serialized with EX), attention 2.4M, reductions 2.6M. All of this is VE / SFU work that K2 will not speed up.
- **prefill** per chunk (8 tokens): VE / SFU take 52–54%, the read floor only 28–42%: prefill is limited by VE / SFU, not bandwidth (K1c's D = 16 only speeds up EX).

So after K2b, **performance depends mainly on the efficiency of the VE / SFU side and scheduling**: overlap of tails with EX, the throughput of reductions and the SFU (P6 SFU multi-slot interleaving), attention, not just bandwidth. Also, **in decode the array consumes only D bytes of weights per cycle** (§3.1 H0): at D = 8, however wide the DMA, it is stuck at 8 bytes per cycle; the bandwidth upgrade must go together with D (K1c's D = 16 pairs with K2a, D = 32 with K2b). The performance expectations of each K stage are based on `z1_profile.md`. A rough extrapolation to Qwen3-0.6B is in K4b.

#### K2a Single 128-bit port

1. **Clock**: raise `pl_clk0` to 200 MHz (then try 250 MHz). The critical paths will most likely be in the VE's fp32 units (`sa_fp32_*.v`, `sa_vefp.v`) and the SFU lookups; add pipeline stages as needed; locate critical paths with `synth_ooc.tcl`; **acceptance is by the timing of the full design after place and route** (see §5).
2. **Widen the LD entry to 128 bits**: HP0 configured as 128-bit; `sa_ld`'s `lw_data` becomes 128 bits (parameterized `DMA_W`), filling a SPAD word (D = 16) at once; ACC writes (64-byte words) by 128-bit lanes. `sa_st` widened accordingly.
   **Keep the old 8-byte-granularity semantics**: today LD / ST have a minimum granularity of 8 bytes (`row_bytes` a multiple of 8, DDR addresses 8-byte aligned), and `sa_st` has no WSTRB logic (every 64-bit beat is full). With 128 bits, beats with **only half valid** will occur and must be handled by the hardware, rather than tightening alignment constraints and invalidating existing dispatches:
   - LD: generate a local write mask from the valid bytes within a beat (half beats at the start and end of rows);
   - ST: generate the correct `WSTRB`, a half beat writing only its valid 8 bytes;
   - Compatibility list (each item needs a sim test and a bit-exactness check):

     | Item | Cases to cover |
     |---|---|
     | DDR start address | 16-byte aligned; only 8-byte aligned (first beat half valid) |
     | Row length `row_bytes` | A multiple of 16; an odd multiple of 8 (last beat half valid) |
     | `pitch` | Equal to the row length (merged into one row, `flat`); larger than the row length; not a multiple of 16 |
     | 4 KB boundary | Bursts split at 4 KB, with the split point on a half beat |
     | ACC lanes | 64-byte words, numbering and masks of 128-bit lanes |
     | LINEAR / INTERLEAVE | Both modes covered; INTERLEAVE chunks crossing beats |
     | D = 8 / 16 | **At D = 8 a SPAD word is only 64 bits and a 128-bit beat spans two words**; under INTERLEAVE the two halves go to two locations `rows` words apart, not adjacent words |
3. **Add performance counters**: besides the AXI counts on the DDR side, a new counter of "**effective bytes written into SPAD/ACC**" (`sa_perf.v`); `bwtest` reports both.
   **Bursts in flight**: `sa_ld`'s burst queue per port has 8 entries (`QD = 8`, index `q_rp[2:0]`; the file-header comment "up to 4 outstanding per port" disagrees with the code; fix it along the way). At 250 MHz with a DDR latency of about 40 cycles, 16 B/cycle per port needs about 640 B in flight: 3 long bursts (16 beats × 16 B = 256 B) suffice, but with **short bursts** (narrow rows, 4 KB splits, INTERLEAVE's short rows, 1–2 beats) 8 entries hold only 128–256 B in flight and bandwidth is latency bound. `bwtest` measures by burst length (1, 2, 4, 8, 16 beats) and counts the cycles with "burst queue full" (a new event; the existing `perf_ev[2]` only counts cycles with AR blocked by the port); if QD has to deepen, widen the index too.
4. **Compiler**: the goal is to **keep** the existing alignment and minimum granularity (8 bytes), with the hardware handling half beats; only if an item truly cannot be made compatible in hardware are constraints added to the target configuration and memory planning, reflected in the new configuration's golden corpus baseline (see the acceptance constraints in §5).

**Acceptance**: `bwtest`'s effective DMA → SPAD bandwidth close to the single 128-bit port's entry limit (a measured value within 3.2 GB/s at 200 MHz, with the actual ratio recorded); stories15M and SmolLM2 bit-exact.

**In two steps** (as with K1a / K1b, separating width from timing): **K2a-1** widens the DMA to 128 bits at 100 MHz (`d16_100mhz_w128`, entry limit 1.6 GB/s), and **K2a-2** then raises the clock to 200 MHz (item 1).

**K2a-1 RTL done** (2026-10-08, `119a524`): `sa_ld` / `sa_st` / `sa_cmdfetch` / `sa_unit` get a `DMA_W` parameter (64 | 128, default 64, old configurations unchanged); the KV260 build gets `-dma_w 128` (HP1 128-bit, configuration name suffix `_w128`).
- The 8-byte granularity is unchanged: a burst starts at the address rounded down to 16 bytes, and the burst queue records "the low lane of the first beat is not this command's" and "whether the high lane of the last beat is this command's"; LD drops lanes accordingly and ST generates `WSTRB` accordingly.
- When the two lanes of a beat fall into the same local word and the starting lane is even, they are written in one cycle (`lw_two`: D = 16's SPAD and ACC with 16-byte-aligned DDR addresses); otherwise one lane per cycle (R held one beat), so D = 8's SPAD, addresses aligned only to 8 bytes and INTERLEAVE blocks crossing words are all correct, just not accelerated. ST's local reads likewise.
- Descriptor fetch gets two 64-bit words per beat (4 beats per descriptor); LDPARAM picks its word by the offset within the beat.
- `LD_BEATS` / `ST_BEATS` still count in 8 bytes (`sa_perf` gains `ev2`: a two-lane write / beat counts twice), so software such as `z1_profile.py` needs no change; counter 19 is now exactly "bytes written into SPAD/ACC ÷ 8" (item 3's effective byte count). The bursts-in-flight figure in `sa_ld`'s file header corrected to 8.
- Verification: `tb_sa_dma`'s AXI model gains 128 bits and `WSTRB`, plus 40 random shapes (8-byte-aligned addresses, row lengths that are odd multiples of 8, crossing 4 KB, INTERLEAVE); deliberately wrong merge conditions or `WSTRB` are caught; `make test` / `test16` / `test128`, `tb_sa_unit` (365 counter checks, descriptor lists) and the firmware system simulations (gemm, vector, desc_run, rt, bwtest; D = 16 at 64 / 128 bits) all PASS. In simulation bwtest sequential read/write goes 3.62 → 7.26 B/cycle (the DDR model has random stalls; not a board number), and `tb_sa_unit`'s 64×64×128 GEMM 10107 → 6413 cycles. `-bd_only` validates. Compiler and runtime unchanged; the driver reads `DMA_W` from the `.hwh`, and `m2_bw_test.py` reports the share by width.
- Build (2026-10-08, `094d535`): setup WNS +0.311 ns (K1c +0.630), WHS +0.010; LUT 75503 (64.5%, 552 more than K1c), FF 60270 (+326), BRAM 130 (90.3%), DSP 347, the last two unchanged. The worst path is still K1c's EX `drain_active` → PE `sh_out` (1 level, 96% routing), not on the DMA.
- **Passed on the board** (2026-10-08, `build/deploy_kv260_d16_100mhz_w128`): REGRESSION PASS, all tests bit-exact with the sim, generated tokens identical to K1c (the differences from the Z1 baseline are the same as K1c's, coming from D = 16). `bwtest` sequential LD / ST 15.6–15.8 B/cycle (1.57 GB/s, 98% of the 128-bit limit; K1c 7.9), LD + ST concurrent 31.0 B/cycle; 8-byte small rows (1-beat bursts) unchanged at 3.03 B/cycle (limited by burst overhead, as expected).

| Test (token/s) | K1c | K2a-1 | Ratio |
|---|---|---|---|
| stories15M decode (c3) | 39.36 | **61.72** | 1.57 |
| stories15M decode (c6p) | 37.84 | **58.40** | 1.54 |
| stories15M prefill (c6p) | 191.4 | **227.6** | 1.19 |
| SmolLM2 qhf decode (c55) | 4.77 | **7.98** | 1.67 |
| SmolLM2 qhf decode (c6p) | 4.50 | **7.26** | 1.61 |
| SmolLM2 qhf prefill (M = D) | 22.22 | **27.17** | 1.22 |
| SmolLM2 generic decode (hfgen) | 4.38 | **6.89** | 1.57 |
| SmolLM2 generic prefill | 10.04 | **11.21** | 1.12 |
| GEMM 256³ (MAC/cycle) | 202.6 | **216.0** | 1.07 |

decode is bandwidth bound and improves 1.5–1.7× (less than 2×: VE tails, attention and scheduling overhead are independent of bandwidth); prefill and GEMM are compute / VE bound and improve little. Next: K2a-2 (200 MHz).

#### K2b Multi-port receive + banked memory

1. **Decide the architecture first**. The functional simulator (`sa_funcsim`, `sa_sim_server.py`) **has no timing model** (it returns 0 cycles) and cannot compare options; two other tools are used instead:
   - **An analytical performance model** (first, for screening): inputs are words written per cycle, number of sub-banks, port assignment, DDR latency, burst length distribution and the bandwidth of each HP port; outputs are the effective bandwidth and sub-bank conflict rate of the various LINEAR / INTERLEAVE transfer shapes; calibrated with the `bwtest` measurements of Z1 and K2a.
   - **An RTL testbench** (then to confirm): extend `rtl/sysarray/sim/tb_sa_dma.v` (which already has an NP-port AXI memory model, several bursts in flight per port, random stalls, SLVERR injection and protocol checks; `make sim TB=tb_sa_dma GEN="NP=3 D=16"`): the memory model gets configurable latency and per-port bandwidth, the effective bandwidth into SPAD is measured, run in xsim.

   Two candidates:

   | Option | Approach | Compiler impact | Risk |
   |---|---|---|---|
   | **A (preferred) merging + banking by the low bits of the word address** | One receive FIFO per HP port (in order within a port, unsynchronized between ports, see below); merged into a 256 / 384 / 512-bit internal bus; SPAD split into 2 / 4 sub-banks by the low bits of the word address (even / odd words), LINEAR loads writing 2–4 adjacent words per cycle | The address space stays contiguous; DMA blocks must be aligned to 2 / 4 words, which enters the alignment constraints of the target configuration and memory planning | Merging logic and FIFO resources; the EX / VE read side has to adapt to sub-banks too |
   | B independent banks per port | Each HP port fixed to its own group of memory banks | Memory planning has to place data by bank; the address-to-bank mapping is maintained jointly by compiler and hardware | Large compiler changes; bandwidth discounted when placement is uneven |

   Also to be made explicit: each sub-bank's read/write port assignment (LD write, EX / VE read, ST read), read latency, read-write collision semantics, byte write enables, and per-sub-bank conflict checks in the scoreboard.

   **Performance model (2026-10-09, `compiler/tests/perf_model.py`)**: not analytical formulas but a command-level event model whose input is the command trace of a real model.
   - **Trace**: `compiler/scripts/perf_trace.sh` runs a packed LLM test on the functional simulator (32 tokens in 2 prefill chunks + 2 decode steps, `SA_PROFILE=1` one list per dispatch), and `sa_sim_server.py --trace` records every executed LD / ST / EX / VE / FENCE (with dynamic fields and BASEs substituted, with descriptor index and owning export name; the sim transport's `RUN` lines carry the export name). Qwen3 is about 680k commands, about 5 minutes.
   - **Timing rules follow the RTL**: the scheduler dispatches in program order, a 6-bank scoreboard (EX and VE share ports, so any shared bank conflicts), engine queues of 4, in-flight masks of 8; LD / ST one command at a time, `latency + max(beats, bursts × 2.63)` (bwtest calibration: 64 KB in 4161 cycles, 2.63 cycles per burst for 8-byte rows), bursts split at 16 beats / 4 KB, half-beat writes taking two cycles; EX `K + 2(D−1)` steps per tile, the drain overlapping the next command; VE `a × groups + b` per kind of command (integer, TRANSPOSE, fp, fp function, fp reduction, reduction after a function), with a and b fit by non-negative least squares to the board's `VE_ACTIVE`; the frontend f cycles per descriptor.
   - **Calibration** (board `board_profile.py 40 --prompt 32 SA_PROFILE_PERF=…`, `d16_100mhz_w128_p6_red`): Qwen3 fits fp 2.33 cycles / group, fp function 30.0, fp reduction 7.26 + 89 / command, frontend 24 cycles / descriptor (correlation 0.996). Qwen3 prefill 61.37M per chunk vs 61.22M on the board (+0.2%), decode 43.35M per step vs 43.33M (+0.1%); the large matmuls / matvecs each within ±3%, LD busy and EX steps each matching. **The same parameters, not refit, applied to SmolLM2**: prefill +2.8%, decode +1.1%.
   - The board profile table only stores 512 exports (`sa_context.c`), so part of Qwen3 prefill's attention exports have no board data; the totals below use the full trace (excluding logits), so a prefill chunk is 76.3M rather than the profile's 61–62M.
   - **Current state of decode**: Qwen3 reads 607.5 MB per step, 14.0 B / cycle on average; LD busy 39.5M, EX 38.5M, VE 2.7M (43.4M per step). Reading and the array are **equally saturated** (both 16 bytes per cycle), so adding only DMA ports or only changing the array gains almost nothing.
   - **Options** (100 MHz, device cycles; token/s converted from device time, excluding host and submission overhead):

     | Option | Qwen3 prefill (16 tokens per chunk) | Qwen3 decode | SmolLM2 prefill | SmolLM2 decode |
     |---|---|---|---|---|
     | Baseline `d16_100mhz_w128_p6_red` | 76.3M (21.0 tok/s) | 43.4M (2.31 tok/s) | 28.6M (56.0) | 11.4M (8.80) |
     | LD 32 B / cycle (2 ports + 2 sub-banks) | ×1.07 | ×1.02 | ×1.08 | ×1.03 |
     | H4 GEMV unit 32 B / cycle (LD 16) | ×1.00 | ×1.02 | ×1.00 | ×1.05 |
     | **LD 32 + H4 GEMV 32** | ×1.07 | **×1.85 (4.26 tok/s)** | ×1.08 | **×1.76 (15.5)** |
     | LD 48 + GEMV 48 | ×1.10 | ×2.55 (5.89) | ×1.10 | ×2.31 (20.3) |
     | LD 64 + GEMV 64 | ×1.12 | ×3.15 (7.27) | ×1.12 | ×2.74 (24.1) |
     | LD 32 + D = 32 array | ×1.41 | ×1.84 | ×1.29 | ×1.73 |
     | LD 32 + GEMV 32 + LD command pipelining | ×1.09 | ×1.90 | ×1.09 | ×1.81 |
     | LD 32 + GEMV 32 + frontend 4 cycles / descriptor | ×1.08 | ×1.85 | ×1.09 | ×1.78 |
     | H2 out-of-order + conflict checks by address range (on the baseline) | ×1.08 | ×1.01 | ×1.08 | ×1.02 |
     | LD 32 + GEMV 32 + H2 | ×1.13 | ×1.88 | ×1.14 | ×1.80 |
     | LD 64 + GEMV 64 + LD pipelining + frontend 4 + H2 | ×1.18 | ×3.46 (7.99) | ×1.21 | ×3.06 (27.0) |

   - **Conclusions**:
     - decode must **widen the read side and the array side together**. At 100 MHz, LD at 32 B / cycle with GEMV at 32 B / cycle gives decode about 1.8–1.85×; 48 / 64 B / cycle about 2.3–3.2×, at which point LD itself becomes the bottleneck again (Qwen3 LD 11.1M, GEMV 9.5M), with VE (2.7M) and the frontend as the remainder.
     - H4 (a GEMV unit dedicated to decode) has the same effect on decode as D = 32 (×1.84 vs ×1.85). D = 32 also speeds up prefill 1.3–1.4×, but with LUT already at 75% and BRAM at 90%, a D = 32 array plus an FL = 16 VE is not expected to fit; H4 is 32–64 int8 multiply-accumulates per cycle plus output accumulators, far fewer resources.
     - H2 only helps prefill (×1.08; from G2 on LD and ST are also treated as sharing a port, so H2's three rows are slightly below the initial ×1.10–1.12), and hardly decode (decode's matvec is already a pipeline overlapping LD and EX). LD command pipelining and a faster frontend give 1–3% each.
     - The model does not include the efficiency of the DDR itself (the KV260's DDR4 peaks at about 19 GB/s; 64 B / cycle × 100 MHz = 6.4 GB/s needs 4 HP ports) or the GEMV unit's actual microarchitecture; these are confirmed in the RTL testbench and on the board.
   - **Microarchitecture** (`docs/k2b_gemv_design.md`): decode's weights are used only once, so H4 is built as a **streaming GEMV**: a new LD mode (`mode = 2`) streams weights from the DMA directly into the GEMV unit, a group of 16 int8 multiply-accumulates per read port, bypassing the SPAD. Two-port reading therefore needs no SPAD sub-banks, and option A's sub-banks, alignment and per-sub-bank conflict checks in this section are deferred until prefill needs the bandwidth; stages G1 (funcsim + compiler) → G2 (RTL, single port) → G3 (dual port + BD) → G4 (board).
   - **Progress (2026-10-10)**: G1 (functional simulator + compiler `--iree-sa-gemv-ports`) stories / SmolLM2 byte-identical to the EX path; G2 (`sa_gemv.v`, `sa_ld` GEMV mode, bitstream `-gemv 1`) passes simulation, and the single-port bitstream `d16_100mhz_w128_p6_gemv` has REGRESSION PASS on the board (existing paths unchanged), bwtest GEMV 15.64 B / cycle; G3 (`-nports 2`, m1 on HP3) dual-port bitstream `d16_100mhz_w128_p6_gemv_np2` (WNS +0.138 ns after the R-side skid buffer, LUT 82.2%) bwtest GEMV **30.55 B / cycle**; **G4 board acceptance (dual port)**: REGRESSION PASS, tokens identical to the EX path, decode **Qwen3 2.22 → 3.98 tok/s (×1.80, predicted ×1.85)**, SmolLM2 7.71 → 12.58 (×1.63, predicted ×1.76), prefill unchanged. G5 (4 read ports, model Qwen3 decode ×3.15, about 6.5–7 tok/s, `k2b_gemv_design.md` §8.1): bitstream `d16_100mhz_w128_p6_gemv_np4` WNS +0.168 ns, LUT 87.0%, board regression passes, bwtest GEMV **57.95 B / cycle** (steady state close to 64); **G5 board acceptance (4 ports)**: REGRESSION PASS, tokens identical, decode **Qwen3 3.98 → 6.28 tok/s (×2.83 over EX, predicted ×3.15)**, SmolLM2 12.58 → 17.04 (×2.21, predicted ×2.74), prefill unchanged; the fixed part of each step (Qwen3 about 67 ms, SmolLM2 about 38 ms) becomes the main bottleneck. Details in `k2b_gemv_design.md` §8.
2. **The first version only guarantees multi-word/cycle throughput for LINEAR**:
   - LINEAR (weight loads, decode's main traffic): consecutive words fall into different sub-banks in turn, so several words can be written per cycle.
   - INTERLEAVE (the array's A strips) has local addresses `word = base + chunk × rows + row`, adjacent chunks of the same row `rows` words apart. When `rows` is a multiple of the number of sub-banks (e.g. 4 sub-banks, `rows = 16`), all chunks of a row fall into the same sub-bank and cannot be written in parallel. Adjacent rows fall into different sub-banks, and if the bursts of two rows happen to return on different ports at the same time they can go in parallel, but that depends on return timing and is not a throughput guarantee.
   - So the first version **only guarantees correctness** for INTERLEAVE, with throughput measured separately (`bwtest` gains an INTERLEAVE mode); if it becomes a bottleneck, evaluate an address swizzle (e.g. sub-bank number = low bits of the word address XOR chunk number) or a staging transpose buffer.
3. **AXI IDs and return order**:
   - Each HP port uses one fixed AXI ID (consistent with the existing design: today `sa_ld` outputs no ARID, one burst queue per port consumed in `q_rp` order). Read transactions with the same ID return in request order, so **in order within a port**; returns from different ports are not synchronized.
   - Out-of-order returns within a port need handling only if a port ever uses several IDs.
4. **Receive FIFO contents and write-back rules**:
   - Each entry carries not just data but also the matching **local word address, byte mask, target memory and transaction status** (`rresp` error, whether it is the last beat of a burst).
   - The local addresses of the bursts of one command never overlap, so **write back by sub-bank availability**, without restoring a global order across ports; otherwise one slow port would block the others.
   - **The `done` condition**: all returned data has been written to local memory and all receive FIFOs are drained; seeing the last AXI beat arrive is not enough. (In the existing design data is written in the cycle it is received, so the two are equivalent; with FIFOs they no longer are.) `err` is reported after all beats are written back.
5. **RTL**: the return side of `sa_ld` receives on several ports at once (independent `m_rready` per port); merging and write-back; `sa_bankmem` gains a sub-bank dimension. Build a standalone memory and DMA testbench first, then connect EX / VE.
6. **Multiple ports**: `NPORTS` = 2, then try 3–4, on HP0–HP3; measure port by port, and compare HP with HPC ports.
7. **Compiler work items** (option A):
   - **Lifetime-based memory planning** (`iree_compiler_plan.md` §8.17 item 3, decided at the Z1 freeze to go here), in two steps. **3a framework**: lifetime analysis on the IR after `sahl-expand-kernels` expansion, with the planner producing `sa.word` / `sa.bank` / `sahl.scope`; the first version reproduces the existing allocation exactly (golden corpus unchanged), reads the number of banks, alignment, capacity and read ports from `TargetConfig`, with constraints as a replaceable layer; removes the coupling of `expandContraction` replaying the allocator. It does not change outputs and can be done on the `kv260` branch during K1. **3b optimization**: reuse by lifetime, avoid sub-bank conflicts, tile when things do not fit; the three constraints below are its rules; it changes outputs, done once the bank organization of K2b / K3 is settled, accepted by per-dispatch checks and board cycles;
   - **Alignment**: the DMA block alignment (2 / 4 words) required for multi-word writes goes into `TargetConfig` and memory planning; blocks that do not meet it fall back to single-word writes (correct but slow) instead of failing to compile;
   - **Sub-bank conflicts**: the scoreboard checks conflicts per sub-bank; memory planning tries to put concurrently accessed LD writes and EX / VE reads in different sub-banks, and scheduling serializes on conflicts;
   - **Read-port assignment**: the mapping of EX / VE / ST read ports to sub-banks is part of the target configuration, and the scheduler uses it to avoid same-cycle contention;
   - **Verification**: `dispatch_check` and the functional simulator support the new target parameters, and the new configuration gets its own golden corpus baseline (§5).
8. **Runtime**: with more bandwidth, the fixed overhead per dispatch becomes a larger share; use the performance counters (`board_profile.py`) to judge whether larger chunks or more cross-dispatch prefetch are needed.

**Acceptance**:

- `bwtest` reports both the DDR-side bandwidth and the **effective bandwidth written into SPAD**, separately for LINEAR and INTERLEAVE and by burst length, and reports the queue-full cycles; LINEAR's effective bandwidth reaches the target of the configuration chosen from the table above (e.g. 2 words/cycle × 250 MHz, measured ≥ 6 GB/s), recorded separately as "what the DDR can give" and "what on-chip memory can take" to find the actual bottleneck. INTERLEAVE's measured value is recorded.
- stories15M and SmolLM2 bit-exact; SmolLM2 decode token/s improves with effective bandwidth, with the measured ratio recorded (linearity not assumed).

### K3 URAM capacity extension

1. **Abstract the memory interface first**: `sa_tdpram.v` is UG901's BRAM template (read-first, 1-cycle read latency, byte-write); just changing `ram_style` does not keep the behavior. The memory interface becomes:
   - a configurable read latency (URAM usually needs extra pipeline registers to reach high clocks);
   - explicit read-write collision semantics (whether a same-address same-cycle read and write returns the old value, the new value, or is forbidden);
   - byte write enables;
   - support for K2's write widths and bank layout.
2. **Initial values**: URAM does not support arbitrary INIT contents and **is zero after configuration**. Three states are distinguished:
   - **power-up / after configuration**: both BRAM (all-zero INIT) and URAM are zero;
   - **reloading the overlay**: equivalent to reconfiguration, also zero;
   - **a logic reset while running**: neither BRAM nor URAM is cleared.
   The comment in `sa_tdpram.v` states that the legacy sequencer depends on "zero contents at power-up" (the reserved tile slots for D > 8), i.e. the first state, and the existing BRAM design is not cleared on a logic reset either. So with URAM the dependency is unchanged. If software later requires the reserved slots to be zero **after every logic reset**, initialize those slots explicitly; there is no need to clear the whole SPAD/ACC each time.
3. **Standalone memory tests**: the URAM version of the memory is first simulated and tested on the board alone (read latency, collisions, byte writes, sub-banks), and only then connected to EX / VE. Resources and timing re-verified.
4. **Capacity**: e.g. SPAD 256 KB × 2, ACC 512 KB. Larger chunks reduce the fixed overhead of each layer and are also a prerequisite for long-context attention.
   - Local addresses today are **16-bit word addresses** (`TargetConfig::invalid()` requires each memory to be at most 2^16 words). At D = 16 a SPAD word is 16 bytes, so 2^16 words = 1 MB; an ACC word is 64 bytes, with a larger limit, so the configuration above needs no change to the address width. Beyond that the descriptor encoding, RTL and compiler checks change together.
   - Compiler: `--iree-sa-spad-kb`, `--iree-sa-acc-kb` pass in the new sizes; C6 has already verified SPAD 256 KB / ACC 512 KB and larger configurations on the sim (D = 16).
5. **Compiler work items**: once SPAD/ACC grow, redo the choice of chunks (block sizes of each linear layer and attention), the space available for cross-dispatch prefetch, and the double-buffer split; the new configuration gets its own golden corpus baseline.
6. **D = 32 together with K2b** (originally planned after K4b; changed after the Z1 measurements and RTL confirmation, see §3.1 H0): in decode the array consumes only D bytes of weights per cycle, so K2b's 24–40 bytes per cycle are only usable with D = 32. Needs RTL (an array of 1024 int8 multiply-accumulates: about 512 DSPs with two per DSP48E2, or partly LUTs; 256-bit SPAD words, 1024-bit ACC words; the VE's FL), the descriptor encoding, `TargetConfig` (today only 8, 16 allowed) and the functional simulator to support it together, with a golden corpus baseline of the D = 32 configuration's own.

**Acceptance**: standalone memory tests pass; the new configuration first passes semantic verification (per-dispatch checks consistent with the functional simulator), then establishes **its own** golden corpus descriptor baseline without overwriting the baselines of Z1 and other configurations (see §5); bit-exact on the board.

### K4a Qwen3-0.6B whole model running correctly

1. **Whole model on the board**: 28 layers, int8 weights about 0.6 GB. C6.1 is already bit-exact per dispatch on the sim.
2. **Capacity and addresses**: weights, KV cache, activations and descriptor lists all inside the reserved region in the low 2 GB (§2.4). INT8 KV: 28 layers × 8 KV heads × head_dim 128 × 2 (K, V), about **112 MiB** for 2048 tokens, excluding scales and layout padding.
3. At short contexts (no longer than what the current untiled attention can handle), prefill + decode bit-exact with the sim.

**Acceptance**: Qwen3-0.6B prefill + decode on the board bit-exact with the sim, with normal generated text (text quality depends on the parallel Q work). **No token/s threshold.**

**Result (2026-10-09, done early on K2a-1's `d16_100mhz_w128`, PASS)**:
- **Compiler** (f2479cc): at D = 16, M = 16, Qwen3 had 29 prefill dispatches falling back to the host because they did not fit in ACC (host fp is not bit-exact with the device, so prefill and decode disagreed). Now sahl-tile's row tiling supports rank ≥ 2 (the 3-D RMSNorm application); when even 2 rows do not fit (`reduction_16x3072`, the quantization after SwiGLU), it tiles by 2 rows and marks `sahl.row_by_row`, and the lowering executes row by row, releasing temporary buffers after each row. Golden corpus unchanged.
- **Host**: per-dispatch comparison at T = 16 / 80 / 256, 1262 OK each; in the sim prefill + decode vs decode-only logits 5/5 rows bit-exact.
- **On the board** (`c6p_qwen3`, opt-in; `insmod u-dma-buf udmabuf0=671088640` right after boot, before the CMA leaks; after K5-M M1 replaced by the 1280 MiB device-tree pool loaded at boot): all 2049 descriptor lists on the accelerator, 0 host dispatches; logits bit-exact with the sim for 25/25 steps, generated tokens identical to the host reference; device memory peak 595.4 MB / 640 MB.
- **Speed** (100 MHz, 1.576 GB/s): prefill of 40 tokens 3500.9 ms (11.43 token/s, including loading in the first chunk); **decode 468.5 ms/step (2.13 token/s)**. The floor from weight reads alone is about 0.6 GB ÷ 1.576 GB/s ≈ 380 ms, so decode already reaches about 81% of the weight-read bound.
- **prefill time breakdown** (`c6p_qwen3_profile`, about 89.9M cycles per 16-token chunk): fp operations of the SwiGLU segment take the largest share: `reduction_16x3072` (row by row, 16.4%), `elementwise_16x3072` (14.8%), `matmul_like 16x384x16x1024` (15.2%). Row-by-row execution is prefill's first optimization point (K4b / KC).

### K4b Long contexts and performance

1. **Attention tiled by T (C6.5)**: one head's K rows (T × head_dim) must fit one SPAD_B bank, so T ≤ 256 for Gemma-3 (head_dim 256), 512 for Qwen3 (128), 1024 for Llama-3.2 (64); checked by compiling at `--max-len 512` (2026-10-13). Revised design without online softmax and without hardware changes: [`iree_compiler_plan.md`](iree_compiler_plan.md) §8.22.
2. **Review the compiler along with the hardware**: if K2 / K3 change the bank organization, read latency or DMA parallelism, the target configuration, memory planning, prefetch and scheduling all need review to make sure the new hardware's bandwidth is used; this is more than just T tiling.
3. **Time breakdown**: with `SA_PROFILE` and the performance counters, split each token's time into **weight reads, KV reads, compute (EX / VE / SFU), scheduling (descriptor processing, dispatch fixed overhead, host)**, measured at context lengths **128, 512, 2048**. The busy cycles of the engines (LD, EX, VE, ST) overlap and **cannot simply be added**: mark the overlap intervals on a timeline, and report for each token's wall clock the exclusive time of each part, the overlapped time, and the engines on the critical path.
4. **Performance targets set from the breakdown**: the ceiling from weight reads alone is effective bandwidth ÷ weight bytes read per token. Qwen3-0.6B's embedding and classifier share one weight (151936 × 1024, about 155 MB), the classifier reads it once per token, and the weights read per token total about 0.6 GB: a ceiling of about 10 token/s at 6 GB/s, about 16 token/s at 10 GB/s. KV reads and the non-overlapping parts of compute and scheduling come on top, so **10–15 token/s is a stretch goal, not a derived result**.
   **Rough extrapolation** (only to show the order of magnitude): at 24 B/cycle, reading 0.6 GB of weights is about 25M cycles; if the part other than weight reads scales from SmolLM2's about 9.5M cycles (generic path) by about 2× with layer width and hidden dimension, about 19M cycles, the total is about 44M cycles, about 5.7 token/s at 250 MHz. Reaching 10 token/s (within 25M cycles) requires the part other than weight reads to be essentially hidden behind the weight reads. So K4b's main work is **compressing and overlapping the part other than weight reads** (merging linear layers, SFU multi-slot interleaving, fixed overhead, cross-dispatch prefetch), prioritized from `z1_profile.md` (Z1 measurement: of the generic path decode's 8.92M cycles not hidden by reads, linear-layer tails about 4.0M, attention 2.4M, reductions 2.6M) and the breakdown after K2b.
5. **Tuning**: move the generic contraction path's linear layers onto micro-kernels (the approach of §8.19); decide which part to optimize first from the breakdown.

**Acceptance**: at T = 2048 consistent per dispatch and bit-exact on the board; time breakdowns given at the three context lengths; performance targets set from the breakdown and met.

### Q (parallel) Qwen3 quantization quality (C6.2)

No need to wait for K4; it can be done now on the CPU / sim.

1. **Locate the error sources**: torch W8A8 Qwen3 agrees with fp32 on top-1 for only 25/40. Following the method of `iree_compiler_plan.md`, switch items to fp32 one at a time (activations, KV, probabilities) to find the main error source.
2. **SmoothQuant-style methods**: divide activations by s per channel and multiply weights by s, a **paired transformation of activations and weights**. With the parts that can be folded into adjacent parameters folded (RMSNorm gain → q/k/v, gate/up; output rows of `up_proj` → `down_proj`; output rows of `Wv` → `wo`), hardware and compiler are unchanged, but the following must hold:
   - **Mathematical equivalence**: outputs before and after the transformation agree in fp32 (within tolerance);
   - **Scale semantics**: the transformation changes the numeric ranges of activations and KV, so the int8 per-token activation scales and KV scales must be recalibrated on the transformed values;
   - **GQA constraint**: `Wv` has only 8 × 128 output channels while `wo`'s input has 16 × 128 channels; Q heads sharing a KV head must use the same s, which limits the smoothing at `wo`;
   - **SwiGLU**: `down_proj`'s input is `silu(gate_proj·x) ⊙ (up_proj·x)`. Dividing the output rows of `up_proj` by s divides the matching channels of the product by s, compensated by multiplying the matching input columns of `down_proj` by s; the scaling **cannot pass through SiLU's gate branch** (nonlinear). (llama2.c's naming: `w1` = gate_proj, `w2` = down_proj, `w3` = up_proj; documents and code uniformly use HF's names.)
   - The classifier shares weights with the embedding and is not transformed.
3. Other candidates: finer KV scales, higher precision for individual layers (evaluated separately if they need hardware or compiler changes).

**Acceptance**: top-1 agreement with fp32 and perplexity within the acceptable range defined by C6.2; the chosen method bit-exact on the sim.

### K5 Optional extensions (evaluated separately)

| Item | Content | Gain | Dependencies |
|---|---|---|---|
| int4 weights | Unpack int4 on the LD path; per-group scales (one per 64/128 of K); the micro-kernel runs EX in K-group segments, VE scales then accumulates | Half the reads, about a 2× ceiling for decode | Quality needs GPTQ / AWQ-style methods; LD unpacking sits on K2's new entry |
| Using all 4 GB of DDR | See **K5-M** below: M1 has the device tree reserve most of the low 2 GB for the accelerator; M2 maps device address bit 31 to the high 2 GB (`0x8_0000_0000`) at `sa_unit`'s AXI outlet, descriptors staying 32-bit | M1: window about 1 GB → about 1.5 GB, and no more need to insmod right after boot; M2: about 3 GB+ (Qwen3-1.7B int8, 3B–4B int4) | M1 none; M2 changes the BD, `sa_unit`'s outlet, runtime with two windows |
| RVV RISC-V core | A standalone experiment or a replacement for PicoRV32; evaluate SiFive SKL and IREE's RISC-V backend | Irregular operations (sampling, top-k, MoE routing), an all-RISC-V stack | Resources, toolchain |
| MoE | Packing of expert weights, a linear micro-kernel with dynamic weight base addresses (`LDPARAM` + dynamic DMA addresses) | Small MoEs such as Granite-1B-A400M | K4, possibly 64-bit addresses |


#### K5-M: using all 4 GB of DDR

**Current state**: the KV260's 4 GB is in two segments, the low 2 GB at `0x0` (actually `0x0000_0000`–`0x7FEF_FFFF`) and the high 2 GB at `0x8_0000_0000`. The accelerator can only access the low 2 GB: the DDR addresses in descriptors are 32-bit, and the BD maps only each port's `HPx_DDR_LOW`. What is actually usable is the window u-dma-buf carves out of the global CMA (1000 MiB at `0x3780_0000`): Qwen3's 640 MB has to be allocated right after boot, otherwise once the CMA is taken by XRT / PYNQ allocations and movable pages no large contiguous block can be found; afterwards only about 350 MB is left.

**M1: the device tree gives the accelerator a dedicated pool (low 2 GB, no hardware change)**
- Add a dedicated `shared-dma-pool`, `reusable` pool under `reserved-memory` (e.g. 1536 MiB), used only by a u-dma-buf device node (`compatible = "ikwzm,u-dma-buf"`, `memory-region`, `sync-mode = <2>`); the global CMA shrinks to about 256 MiB (PYNQ / XRT's `pynq.allocate`). Linux itself can use all of the high 2 GB.
- `reusable` (a per-device CMA region) rather than `no-map`: the kernel's `dma_alloc_coherent` on a `no-map` pool allocates in powers of two, so a 1536 MiB request rounds up to 2 GiB and fails; `reusable` goes through `cma_alloc` with any size (1 MiB aligned).
- u-dma-buf loads automatically at boot (`/etc/modules-load.d`) and allocates the whole pool as soon as the device appears, so it is never taken afterwards; `run_board.sh` and the launcher need no change (the window is still `/dev/udmabuf0`).
- The pool must avoid the kernel image, the addresses where U-Boot places the DTB / initrd, and the PMU firmware region (above `0x7FF0_0000`): first run `boards/kv260/scripts/board_mem_inspect.sh` on the board to collect `/proc/iomem`, the boot arguments, the existing `reserved-memory` and how the DTB is loaded, then generate a new DTB from the running device tree (`/sys/firmware/fdt`) with `accel_mem_dtb.sh`; the installation method follows the inspection results, keeping the original DTB for rollback (the SD card can also be restored on a PC).
- **Board inspection results (2026-10-10, kernel 5.15.0-1080)**: the kernel at `0x0021_0000`–`0x0319_FFFF`; a small reserved piece at `0x0FFF_2000`; at boot the initrd at `0x75DA_E000`–`0x78FF_FFFF` (U-Boot places it below its own region `0x7BF0_0000`), the PMU firmware at `0x7FF0_0000`; the device tree has no `linux,cma` node, and the CMA size comes from `cma=1000M` hard-coded in the boot script; the boot script (flash-kernel's `bootscr.zynqmp.kria`) uses `/boot/firmware/user-override.dtb` instead of the DTB in the FIT when it exists. Therefore: the pool is set at **`0x1000_0000`–`0x5FFF_FFFF` (1280 MiB)**, and the global CMA via `LINUX_KERNEL_CMDLINE="cma=256M"` in `/etc/default/flash-kernel` (after `cma=1000M`, so it wins). Installation: `accel_mem_install.sh install` (DTB → `user-override.dtb`, change the CMA and regenerate the boot script, load u-dma-buf at boot, original files kept as `*.pre-accel`), `check` to verify, `uninstall` to roll back. **Installed (2026-10-10)**: after reboot the command line is `cma=1000M cma=256M`, the kernel sets up the 1280 MiB pool (`0x1000_0000`), the global CMA of 256 MiB is at `0x65C0_0000`, u-dma-buf loads at boot, and `udmabuf0` is at `0x1000_0000`, 1280 MiB, `sync_mode` 2; no more insmod right after boot. The dual-port GEMV regression rerun on it (`results_m1`): all pass, the windows (16–640 MB) all at `0x1000_0000`, tokens and speeds exactly as before.

**M2: the accelerator accesses the high 2 GB (device address remapping)**
- At `sa_unit`'s AXI outlet: device address bit 31 = 0 → physical `addr` (low 2 GB); bit 31 = 1 → physical `0x8_0000_0000 + addr[30:0]` (high 2 GB). The accelerator sees a contiguous 4 GB device address space, and **the descriptor, compiler and firmware formats are unchanged**; the AXI address widens to 40 bits, and the BD additionally maps `HPx_DDR_HIGH` / `HPC0_DDR_HIGH` for m0–m3.
- PicoRV32 unchanged: the ring queue and descriptor lists stay in the low 2 GB (above its `0x8000_0000` are the CSRs and BRAM).
- Software: the device tree reserves a piece of the high 2 GB as `reserved-memory` (the default CMA only uses physical addresses below 4 GB, and the high 2 GB is physically at 32 GB), mapped as a second window through u-dma-buf's `memory-region`; `sa_transport_board.c` supports two windows and the physical → device address conversion (high window `0x8_xxxx_xxxx` → `0x8000_0000 + offset`); the runtime puts weights and the KV cache into the high window. Address checks still follow §2.4: 64-bit parsing and whole-range checks before converting.
- Needed only for 1.7B+ models; independent of K2b / G5's 4 ports.

---

## 4. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Insufficient multi-port receive and memory write throughput** (K2b) | More HP ports add no bandwidth | Settle the merge + banking architecture in sim first; `bwtest` measures the DDR side and the SPAD write side separately; sub-bank conflicts included in the scoreboard checks |
| Buffers allocated in the high 2 GB (≥ `0x8_0000_0000`), or addresses truncated before checking | The accelerator reads and writes wrong addresses; wrong results that are hard to trace | Reserved region fixed in the low 2 GB; 64-bit parsing and whole-range checks before converting to 32 bits; the runtime errors out instead of truncating silently |
| Layout of host/firmware shared structures changing on AArch64 | Misaligned fields, intermittent errors | Shared structures use fixed-width fields only; `static_assert` on sizes and offsets; the firmware ABI does not change when the host is recompiled |
| Timing not closing above 200 MHz | Bandwidth targets missed | 200 MHz first; pipeline stages in the VE fp32 and SFU; separate clock domains for DMA and compute if necessary |
| URAM semantics differing from BRAM (read latency, collisions, no arbitrary INIT) | Wrong results or timing failures after K3 | Abstract the memory interface; make zero initial values depend only on the post-configuration state; standalone memory tests first |
| After the bandwidth increase the part other than weight reads dominates | token/s far below the bandwidth-derived figure | A time-breakdown baseline before the Z1 freeze; K4b prioritizes compressing and overlapping the non-weight-read part from the breakdown |
| Not enough bursts in flight with short bursts | Bandwidth limited by DDR latency | `bwtest` measures by burst length and counts queue-full cycles; deepen QD if necessary |
| INTERLEAVE conflicting on sub-banks | A-strip loads miss multi-word/cycle throughput | First version only guarantees correctness and measures separately; address swizzle or a transpose buffer if it becomes a bottleneck |
| Half-beat handling wrong after 128 bits | Misaligned data or out-of-bounds writes at row starts/ends, 4 KB boundaries, D = 8 | LD write masks and ST `WSTRB`; sim tests item by item from the compatibility list |
| Actual concurrent bandwidth of several HP ports below expectations | K2b's gain discounted | `bwtest` per port; compare HP with HPC ports; tune burst lengths and outstanding counts |
| PS-side cache coherence | The accelerator does not see the latest data written by the ARM | Keep the existing conventions (non-cached mappings / explicit flushes), add ZynqMP notes to `memory_model.md`; make each buffer's mapping attributes and synchronization points explicit; do not use HPC's coherent ports unless measurements show the need |
| Kria-PYNQ and Vivado versions not matching | Overlay fails to load | K0 pins the version combination; if necessary load with Ubuntu + a self-written FPGA manager script |
| New compiler corner cases with new parameters or bank organization | Some dispatches wrong | Per-dispatch checks (`dispatch_check.py --dirty --skew`), golden corpus, functional simulator first |
| Insufficient Qwen3 quantization quality | Poor output text quality | Q work in parallel: locate the error sources first, then choose the fix |

---

## 5. Summary of metrics per stage

| Stage | Accelerator clock | LD entry limit | Read bandwidth (measured / target) | Performance |
|---|---|---|---|---|
| PYNQ-Z1 (measured) | 50 MHz | 0.4 GB/s | Measured about 0.39 GB/s | SmolLM2-135M about 2.2 token/s (hand-written path; generic path 1.87); Qwen3-0.6B does not fit |
| K1a | 50 MHz | 0.4 GB/s | Measured 0.397 GB/s | Passed; stories15M 18.27 token/s, SmolLM2 2.27 (generic 1.88) |
| K1b | 100 MHz | 0.8 GB/s | Measured 0.794 GB/s | Passed; stories15M 35.44 token/s, SmolLM2 4.43 (generic 3.73) |
| K1c | 100 MHz (D = 16) | 0.8 GB/s | Measured 0.794 GB/s | Passed; stories15M 39.36 token/s, SmolLM2 4.77 (generic 4.39), SmolLM2 prefill 22.2 token/s; GEMM 256³ 202.6 MAC/cycle |
| K2a-1 | 100 MHz (D = 16, 128-bit) | 1.6 GB/s | Measured 1.576 GB/s | Passed; stories15M 61.72 token/s, SmolLM2 7.98 (generic 6.89), SmolLM2 prefill 27.2 token/s; GEMM 256³ 216.0 MAC/cycle |
| K2a-2 | 200–250 MHz | 3.2–4.0 GB/s | DMA → SPAD effective bandwidth close to the limit | Observed |
| K2b | 200–250 MHz | 6.4–12.8 GB/s (by the chosen configuration) | **LINEAR effective bandwidth into SPAD** reaches the chosen configuration's target (e.g. 2 words/cycle × 250 MHz ≥ 6 GB/s); INTERLEAVE's measured value recorded | SmolLM2 improves with effective bandwidth (measured ratio) |
| K4a | Same as K2b (actually done on K2a-1) | Same as K2b | — | **Passed** (2026-10-09): bit-exact on the board; decode 2.13 token/s (100 MHz, 1.576 GB/s), prefill 11.4 token/s |
| KC-H1 (P6) | 100 MHz (D = 16, 128-bit, `d16_100mhz_w128_p6`) | Same as K2a-1 | Bit-exact, SFU segments accelerated | **Passed** (2026-10-09): REGRESSION PASS; Qwen3 prefill 13.96 token/s (w128 11.43), decode 2.20; SmolLM2 prefill 34.9 (27.2); LUT 72.1%, WNS +0.113 ns |
| KC-H1b (REDUCE batched) | 100 MHz (`d16_100mhz_w128_p6_red`) | Same as K2a-1 | Bit-exact, reduction segments accelerated | **Passed** (2026-10-09): REGRESSION PASS, tokens identical to p6; Qwen3 70.0M → 62.2M cycles per chunk, prefill 15.45 token/s, decode 2.22; SmolLM2 prefill 38.4; LUT 75.4%, WNS +0.123 ns |
| K4b | Same as K2b | Same as K2b | — | Targets set from the time breakdown; ceiling from weight reads alone: about 10 token/s at 6 GB/s, about 16 token/s at 10 GB/s; 10–15 token/s is the stretch goal |

**Acceptance constraints common to all stages**

1. **Bandwidth targets paired with configurations**: every bandwidth target states its configuration (words/cycle × clock) and the required utilization (the table under §3 K2); combinations requiring more than about 85% utilization are not planned targets.
2. **Timing judged on the full implementation**: OOC synthesis with `synth_ooc.tcl` is only for locating critical paths. Acceptance looks at the reports of the full design after place and route: setup and hold both closed (WNS, WHS ≥ 0), all clocks constrained (`report_clocks`, `check_timing` without unconstrained paths), and clock-domain-crossing checks (`report_cdc`) without unhandled paths.
3. **Golden corpora kept separately per target configuration**: each target configuration (Z1's D = 8, the KV260's D = 16 and the K2 / K3 configurations …) has its own descriptor baseline. A new configuration first passes semantic verification (per dispatch consistent with the functional simulator, end to end bit-exact with the sim), then establishes its own baseline; it **does not overwrite** the Z1 baseline (§7.2) or those of other configurations.
4. **Time breakdowns mark overlaps**: the busy cycles of the engines overlap and cannot simply be added; report the exclusive and overlapped parts of the wall clock (K4b step 3).

Estimation method: decode reads all weights once per token, so **the ceiling from weight reads alone** ≈ effective bandwidth ÷ weight bytes read per token. In reality KV reads, quantization, SFU, descriptor processing and computation that cannot fully overlap come on top, to be measured with K4b's time breakdown.

---

## 6. Open questions

- Whether the Vivado version and free license cover the XCK26; the KV260 board file version.
- The current Kria-PYNQ version, the boot firmware version and the matching Vivado version.
- The HPM address windows through which the ARM accesses the PL on ZynqMP (as generated by the block design).
- How the low-2 GB reserved region is allocated (device-tree reserved-memory, CMA or udmabuf), its mapping attributes, and whether existing drivers can use it directly.
- Whether `armv7_libm_shim.c` is still needed in the aarch64 build.
- Whether the firmware and driver have other hard-coded ARM-side addresses (grep `0x4001`, `0x4000` etc. one by one).
- BRAM inference results of `sa_tdpram.v`, `sa_bankmem.v` on UltraScale+; URAM read-latency configuration and collision behavior.
- K2b's number of sub-banks and internal bus width (2 / 3 / 4 words per cycle), chosen from the target-bandwidth-versus-configuration table and the analytical performance model.
- D = 32 resources and timing: feasibility of two int8 multiplies per DSP48E2, the LUTs of FL = 16 (or FL = D / 4 instead), the timing of the D-lane reduction tree at 250 MHz; compared with H4 (a dedicated matrix-vector path).
- Measured DDR read latency on the KV260 (varying with load), for the analytical performance model and the choice of QD.
- Timing margins of PicoRV32 and PCPI at 200–250 MHz.
- In K2b option A, whether the assignment of EX / VE read ports to sub-banks lowers the existing compute throughput.
- The actual usable bandwidth of the DDR controller with several HP ports concurrent.

---

## 7. Repository strategy: freeze Z1, main moves to the KV260

**Decision**: no new repository. The PYNQ-Z1 version is frozen as a tag and a branch, kept as the legacy version; after K1's acceptance main moves completely to the KV260.

### 7.1 Why not a new repository

- The repository name `Torch2RISCV-accelerator` is board independent; the KV260 version keeps it.
- Stars, followers and the links in the LinkedIn post all stay with the original repository; readers need not be redirected.
- The full commit history and `git blame` stay in one repository, and the third-party sources listed in NOTICE remain traceable.
- The compiler, firmware, functional simulator and golden corpus are the same for both boards; in one repository, K1's bit-exact acceptance can compare directly.

### 7.2 Z1 freeze (before K1 starts)

1. **Finish Z1**: complete the remaining planned Z1 work; all tests pass on main (golden corpus, `dispatch_check.py`, C3's stories15M, SmolLM2).
2. **Save baseline files** in `tests/baselines/pynq-z1/` as the reference for the KV260's bit-exact acceptance (so the Z1 board need not be present):

   | File | Content |
   |---|---|
   | `golden_manifest.txt` | List and hashes of the golden corpus's 4002 dispatches |
   | `dispatch_check.json` | Per-dispatch results of `dispatch_check.py` |
   | `stories15m_tokens.txt` | stories15M's 76-step token sequence (fixed prompt and seed) |
   | `smollm2_tokens.txt` | SmolLM2-135M's prefill + decode token sequence |
   | `perf.md` | Measured token/s, `bwtest` bandwidth, resource usage and timing (one row per bitstream) |
   | `z1_profile.md` | Time breakdown: the hand-written and generic paths of stories15M and SmolLM2, with reads (and their floor), EX, VE / SFU, ST, fixed overhead and overlapped cycles per dispatch type (`board_profile.py` + `SA_PROFILE_PERF`: the runtime reads the accelerator's event counters); the basis for the performance expectations of each K stage |

   **Done** (2026-10-04): `compiler/scripts/deploy_z1_freeze.sh` produces all test bundles (bwtest, C3, C5.5, two C6.P, generic path), `compiler/tests/board_regress.py` is bit-exact for all of them on the board (`REGRESSION PASS`), and `compiler/tests/make_z1_baselines.py` writes the files in the table above (plus a `README.md`).

   Large binary outputs (logits, intermediate tensors) do not go into git; they go into Release attachments.
3. **Tag and branch**:

   ```sh
   git checkout main && git pull
   git tag -a v1.0-pynq-z1 -m "PYNQ-Z1: final release (stories15M 17.7 tok/s, SmolLM2-135M, bit-exact)"
   git branch pynq-z1 v1.0-pynq-z1
   git push origin v1.0-pynq-z1 pynq-z1
   ```

4. **GitHub Release**: create a Release on `v1.0-pynq-z1` with the final bitstream / `.hwh`, firmware binaries and the large files of 7.2 step 2 attached. Readers can reproduce on the Z1 without installing Vivado.

### 7.3 KV260 development (K0–K1)

- Develop on the `kv260` branch; main remains the usable Z1 version meanwhile.
- After K1 (K1a and K1b) is accepted, merge `kv260` into main. From then on main is the KV260 version.
- Update the README in the same merge PR (7.5).
- **Done** (2026-10-08): after the board acceptance of K1a and K1b, fast-forward merged into main, the top of the README replaced with the note of 7.5, and the deployment scripts' default board changed to `kv260` (`SA_BOARD=pynq-z1` still works). The directory reorganization of 7.4 is a separate commit.

### 7.4 Cleaning up main after the merge

1. **Directories**:

   ```
   rtl/              # shared RTL, parameterized (D, NPORTS, SPAD/ACC in BRAM or URAM)
   rtl/picorv32/     # moved from RISCV-on-PYNQ-Z1/picorv32/, keeping COPYING (ISC)
   firmware/         # unchanged
   compiler/         # TargetConfig defaults changed to the KV260
   boards/kv260/     # block design tcl, constraints, build scripts, system image notes
   driver/           # board addresses read from the board configuration, not hard-coded
   ```

2. **Remove Z1-specific parts**: the Z1 block design, constraints, bitstream result directories and armv7-related build options under `RISCV-on-PYNQ-Z1/`. They all remain on the `pynq-z1` branch and tag.
3. **Licenses**: if main still keeps scripts adapted from `RISCV-on-PYNQ-Z1` (Jinzzj, MIT), that NOTICE entry stays with updated paths; remove it from NOTICE once they are entirely gone.
4. **Bitstreams no longer in git**: the KV260's bitstreams and `.xsa` are much larger than Z1's; they all go into GitHub Releases (or Git LFS).
5. **Documents**: Z1-specific content in documents such as `memory_model.md` and `hardware_learning_path.md` becomes a "History: PYNQ-Z1" subsection, or notes "see the `pynq-z1` branch".

**Done** (2026-10-08):

- `KV260/` → `boards/kv260/` (block design, build scripts, README; build outputs remain in `boards/kv260/build/`, not in git); `RISCV-on-PYNQ-Z1/picorv32/` → `rtl/picorv32/`, `RISCV-on-PYNQ-Z1/ip/` → `rtl/ip/` (the relative path `../../picorv32/picorv32.v` in the IP packaging is unchanged), `pico_processor.tcl` → `boards/kv260/scripts/`, `RISCV-on-PYNQ-Z1/tests/ddr_access/` → `tests/ddr_access/`.
- Removed: the rest of `RISCV-on-PYNQ-Z1/` (the Z1 block design `pico_bit.tcl`, Z1 build scripts, constraints, all results of phase3–l2 under `bitstreams/`, the original plan), `iree-sa/l0` (armv7's L0), `build_sa_runtime.sh armv7`, `SA_BOARD=pynq-z1` in `board_env.sh`, `export.py --armv7`, `make_z1_baselines.py`. All on the `pynq-z1` branch.
- Licenses: `rtl/ip/` and `pico_processor.tcl` still derive from RISCV-on-PYNQ-Z2 (Jinzzj, MIT); the license moved to `LICENSES/RISCV-on-PYNQ-Z1-MIT.txt`, with NOTICE paths updated.
- The compiler's target configuration defaults need no change: K1's accelerator is the same as Z1's (D = 8, 128 KB SPAD, 256 KB ACC); from K1c on they follow the new configuration.
- Bitstreams: main no longer contains bitstreams; the KV260 overlays go into GitHub Releases.
- Documents: documents recording Z1 work note at the top that `RISCV-on-PYNQ-Z1/...` and `bitstreams/...` refer to the `pynq-z1` branch; the README's directory description and the build steps in `hardware_learning_path.md` changed to the KV260.
- Verification: `-bd_only` passes with the new paths (address map the same as K1), the system simulations of `tests/ddr_access` and `firmware/gemm` PASS, the aarch64 runtime builds.

### 7.5 README note

Placed at the very top of the README:

```markdown
> **Platform update:** Development has moved to the AMD Kria KV260.
> The PYNQ-Z1 version is frozen at tag
> [`v1.0-pynq-z1`](../../tree/v1.0-pynq-z1) (branch `pynq-z1`):
> stories15M at 17.7 tok/s and SmolLM2-135M, bit-exact on the board.
> It still works and is the cheapest way to try the full stack, but it
> only receives critical fixes.
```

Before the K1 merge, a teaser version can go up first: "KV260 port in progress on branch `kv260`".

### 7.6 Maintenance of the `pynq-z1` branch

- Only serious problems are fixed: wrong results, build failures, wrong steps in documents. No new features, no performance optimization.
- Fixes are committed on `pynq-z1` first with a minor version tag (`v1.0.1-pynq-z1` …); if main has the same problem, it is ported separately.
- Add a "Board: PYNQ-Z1 / KV260" option to the issue template to tell them apart.

### 7.7 Timeline

| When | main | Other |
|---|---|---|
| Now → Z1 finished | Z1 version | — |
| Z1 freeze (**done 2026-10-04**) | Z1 version | tag `v1.0-pynq-z1`, branch `pynq-z1`, Release, baseline files |
| K0–K1 | Z1 version (README announces the KV260) | Development on the `kv260` branch |
| K1 acceptance (**done 2026-10-08**) | **KV260 merged**, README updated, directories reorganized (as a separate commit afterwards) | `pynq-z1` only gets serious fixes |
| K2 and later | KV260 version | — |

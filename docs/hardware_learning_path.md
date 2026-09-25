# Hardware learning path

A guided route through the hardware side of this repository: from digital
design fundamentals to the double-buffered systolic-array accelerator
(`rtl/sysarray`) running next to a PicoRV32 on the PYNQ-Z1. Every stage
pairs outside reading with the files in this repo that put the idea into
practice, a hands-on lab that uses the real testbenches, build scripts or
board scripts, and questions to check yourself.

The project history is itself a curriculum: Phase 2 → Phase 4 built a
simple 8×8 unit, M1 → M4 rebuilt it as a double-buffered 16×16 accelerator
with a vector engine, and M5 made it observable (performance counters) and
added a descriptor DMA once the counters showed where time was lost.
Reading the commits in order (`git log --reverse`) shows each idea arriving
on its own. To follow one operation through every layer first, read
[`execution_walkthrough.md`](execution_walkthrough.md) (Chinese).

---

## How to use this path

- **Order.** Stages 0–2 are prerequisites; skip what you already know.
  Stages 3–8 follow the data from the ARM to the array and back, and are
  best done in order. Stages 9–10 are for going further.
- **Each stage has:** *Goal* · *Read* (outside material) · *In this repo*
  (files, in reading order) · *Lab* (something to run or change) ·
  *Check yourself* (answer without looking; the answers are in the files).
- **Tools you need:** Vivado 2024.1 (xsim for simulation, synthesis,
  implementation; the scripts use `/home/jon/Projects/Vivado/...`, adjust
  `VIVADO_SETTINGS`), clang/lld with the RISC-V target (firmware), Python 3
  with NumPy (golden vectors), and for the board stages a PYNQ-Z1 with the
  PYNQ image.
- **Limit Vivado to 4 threads/jobs** as the scripts do
  (`set_param general.maxThreads 4`, `-jobs 4`) - full builds take
  15-30 minutes; run them in your own terminal so they survive a closed
  session.

## The hardware at a glance

```
 ARM Cortex-A9 (Linux, PYNQ)                        PL (FPGA fabric, 50 MHz)
 ─────────────────────────                          ─────────────────────────────────────────────
 driver/pynq_matmul.py                              PicoRV32 (rv32imc, 8 KB program BRAM)
   │ allocate() DDR buffers                            │ firmware: firmware/{gemm,vector,...}
   │ writes firmware + mailbox ── AXI GP ──► BRAM ────►│ custom-0 instructions (.insn)
   │ GPIO EMIO[0] = RISC-V reset                        ▼ PCPI
   │ builds descriptor lists in DDR              sa_pcpi ─────┐
   │  (fetched after mat_submit)                 sa_cmdfetch ─┴► sa_sched (queue, decode, scoreboard)
   │                                                               │ in-order dispatch to 4 engines
   ▼                                                               │   (sa_perf counts events everywhere)
 DDR3 (512 MB) ◄── HP0 ── PicoRV32 data (DDR access)               ├─► LD  (DDR → SPAD/ACC)  ┐
      ▲                                                            ├─► ST  (SPAD/ACC → DDR)  ├─ AXI4 ─► HP2
      └────────── HP2 ◄── axi_protocol_converter ◄─────────────────┤                         ┘
                                                                   ├─► EX  (D×D systolic array, K-streaming)
                                                                   └─► VE  (vector engine, VL = D)
                                                     SPAD_A, SPAD_B (128 KB each), ACC (256 KB): 2 banks each
```

Numbers to keep in mind (M5 build, board-measured): D = 16, 256 MAC/cycle
peak at 50 MHz, 202.6 MAC/cycle on a 256³ GEMM (the array does useful work
79 % of the time), one HP port at 7.9 B/cycle (99 % of 8 B/cycle), 128 of
140 BRAM36 in the accelerator, 92 % of LUTs.

---

## Stage 0 - Digital design and Verilog

**Goal.** Read and write synthesizable Verilog fluently: combinational vs
sequential logic, blocking vs non-blocking assignment, FSMs, FIFOs,
parameters and `generate`, signed arithmetic.

**Read.**
- Harris & Harris, *Digital Design and Computer Architecture, RISC-V
  Edition* (2021), chapters 1-5 (logic, sequential design, HDL, building
  blocks).
- HDLBits (hdlbits.01xz.net) - do the Verilog language and circuits
  problem sets; they are short and graded automatically.

**In this repo.**
1. `rtl/sysarray/sa_tdpram.v` - a 50-line true-dual-port RAM: two
   `always @(posedge clk)` blocks, byte-write enables, read-first behavior.
2. `rtl/matmul/systolic_array.v` - the Phase 2 array: `generate` loops,
   a PE grid wired by index arithmetic.
3. `rtl/sysarray/sa_ld.v`, the `qst` generate block - a circular queue with
   read/write pointers.

**Lab.** Write a small testbench for `sa_tdpram` (write on port A, read on
port B the next cycle, write and read the same address on both ports) and
run it with `xvlog`/`xelab`/`xsim` (see the `sim` rule in
`rtl/sysarray/Makefile` for the exact commands).

**Check yourself.**
- Why does `doutA <= ram_block[addrA]` return the *old* data when the same
  cycle writes `addrA`? What is this mode called?
- In `sa_ld.v`, why is `used` a 4-bit wire (`q_wp - q_rp`) instead of a
  32-bit expression? (Hint: pointers wrap. This was a real bug in M1.)
- What does `$signed(x[15:0]) * $signed(y[15:0])` produce when one operand
  of an expression is an unsigned concatenation? (See the RELU/REQUANT
  comment in `sa_ve.v`.)

---

## Stage 1 - FPGA fabric, Zynq and AXI

**Goal.** Know what the tools turn your RTL into, how the Zynq's ARM
(PS) talks to the fabric (PL), and how AXI transfers work.

**Read.**
- AMD UG474 *7 Series CLB* (LUTs, carry chains, distributed RAM),
  UG473 *7 Series Memory Resources* (RAMB36/RAMB18, aspect ratios, byte
  writes), UG479 *7 Series DSP48E1* (the 25×18 multiplier + 48-bit
  accumulator). Skim, then come back when a report surprises you.
- AMD UG585 *Zynq-7000 Technical Reference Manual*: the PS-PL interfaces
  (GP ports, HP ports, EMIO GPIO) and the DDR controller.
- ARM IHI 0022 *AMBA AXI and ACE Protocol Specification*: channels,
  VALID/READY handshake, bursts (INCR, AxLEN, AxSIZE), the 4 KB rule,
  responses (OKAY/SLVERR/DECERR); AMD UG1037 *AXI Reference Guide* for the
  Xilinx view (AXI3 on the HP ports, protocol converters).
- PYNQ documentation (pynq.readthedocs.io): `Overlay`, `allocate`,
  `MMIO`, `GPIO`, and how the `.hwh` file describes the design.

**In this repo.**
1. `docs/memory_model.md` - the three address maps (ARM, RISC-V, DMA) and
   the coherency rules (`flush()` / `invalidate()`).
2. `RISCV-on-PYNQ-Z1/scripts/pico_bit.tcl` - the block design: PS7, the
   PicoRV32 hierarchy, `matmul_0` (a module reference to `sa_unit`), the
   HP0/HP2 connections, `assign_bd_address`.
3. `RISCV-on-PYNQ-Z1/scripts/build_bitstream.tcl` - project creation,
   synthesis/implementation runs, bitstream + `.hwh` export.
4. `RISCV-on-PYNQ-Z1/constrs/PYNQ-Z1.xdc` - pin constraints.
5. `RISCV-on-PYNQ-Z1/bitstreams/m4/utilization.rpt` and
   `timing_summary.rpt` - what a finished design costs and how fast it is.

**Lab.**
1. Build the overlay: `cd RISCV-on-PYNQ-Z1 && ./scripts/build_bitstream.sh -jobs 4 -sa_d 16`.
2. Open `build/picorv32_z1/picorv32_z1.xpr` in the Vivado GUI; open the
   block design and the implemented design; find the accelerator's BRAMs
   and DSPs on the device view.
3. In `build/output/picorv32.hwh`, find the `sa_unit` module and its
   `PARAMETER` entries (`D`, `SPAD_WORDS`, ...) - the driver reads D from
   here (`overlay_params()` in `driver/pynq_matmul.py`).

**Check yourself.**
- Why does the accelerator go through an `axi_protocol_converter` before
  HP2? What would break without it?
- A burst starts at DDR address `0x1000_0FF0` with 16 beats of 8 bytes.
  Is that legal AXI? What does `sa_ld.v` do to avoid it?
- Why must the ARM call `flush()` after writing an input buffer and
  `invalidate()` before reading an output buffer?

---

## Stage 2 - The RISC-V control core and custom instructions

**Goal.** Understand how a tiny in-order core drives an accelerator through
custom instructions, and how bare-metal firmware is built and loaded.

**Read.**
- *The RISC-V Instruction Set Manual, Volume I: Unprivileged ISA* - base
  encoding formats (R-type), the reserved *custom-0* opcode (0x0B), and the
  GNU assembler `.insn` directive.
- PicoRV32 README (github.com/YosysHQ/picorv32): configuration parameters,
  the AXI adapter, and the **PCPI** (Pico Co-Processor Interface) signals
  `pcpi_valid/insn/rs1/rs2` → `pcpi_wr/rd/wait/ready`.

**In this repo.**
1. `docs/custom_isa_encoding.md` - the Phase 4 instructions (funct7 = 0).
2. `firmware/include/sysarray_intrinsics.h` - every instruction as a C
   inline function (`.insn r 0x0B, funct3, funct7, rd, rs1, rs2`): funct7 = 0
   legacy, 1 matrix/DMA (incl. `mat_perf`, `mat_submit`), 2 vector;
   `sa_init()` reads D and the capability bits from CAPS (`sa_has_perf()`,
   `sa_has_desc()`: firmware must not issue an instruction the loaded
   overlay does not answer, or the core traps).
3. `firmware/common/start.S`, `firmware/common/link.ld`, `firmware/common.mk`
   - reset vector, stack, BRAM layout (code + data + stack below 0x1E00),
   clang flags (`rv32imc`).
4. `firmware/include/mailbox.h` - the ARM ↔ RISC-V contract: the mailbox in
   the last 256 bytes of the program BRAM and the performance counter area
   below it (0x1E00).
5. `rtl/matmul/matmul_pcpi.v`, then `rtl/sysarray/sa_pcpi.v` - the PCPI
   decoders (Phase 4, then M1-M5).
6. `docs/execution_walkthrough.md` - one GEMM traced from Python through the
   firmware, PCPI and scheduler to the engines (Chinese).

**Lab.**
1. `cd firmware/gemm && make` then
   `llvm-objdump -d gemm_fw.elf | grep unknown` - the disassembler does not
   know custom-0, so those lines are the accelerator instructions (bytes
   little-endian, e.g. `0b 00 65 03` = `0x0365000b`). Decode one by hand:
   opcode [6:0], rd [11:7], funct3 [14:12], rs1 [19:15], rs2 [24:20],
   funct7 [31:25].
2. `make sim` in `firmware/matmul_insn` - PicoRV32 runs the firmware
   against the real accelerator RTL (`firmware/sim/tb_system.v`).

**Check yourself.**
- PicoRV32 traps an unanswered custom instruction after 16 cycles unless
  `pcpi_wait` is held. How does `sa_pcpi.v` keep a long `mat_fence` from
  trapping?
- Why do queued commands (`mat_load`, `mat_exec`) return immediately while
  `mat_fence` stalls the core?

---

## Stage 3 - The first accelerator (Phase 2-4)

**Goal.** Understand a complete, simple accelerator end to end before the
complex one: a systolic array, CSRs, an AXI master, and a job sequencer.

**Read.**
- H. T. Kung, "Why Systolic Architectures?", *IEEE Computer*, 1982.
- Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing
  Unit", ISCA 2017 - a production systolic array (weight-stationary) and
  why memory bandwidth dominates.

**In this repo** (`rtl/matmul`, the Phase 2-4 unit, kept as reference).
1. `rtl/matmul/README.md` - CSR map, memory layout, timing of one run.
2. `systolic_array.v` - output-stationary 8×8 array: A flows right, B flows
   down, each PE keeps its C[i][j].
3. `matmul_unit.v` - AXI4-Lite CSR slave, AXI4 master (load A/B, store C),
   the state machine.
4. `matmul_pcpi.v` - the same unit driven by custom instructions and a
   descriptor in DDR.
5. `sim/tb_matmul_unit.v` + `sim/gen_vectors.py` - NumPy golden vectors.
6. `notebooks/phase3_matmul_demo.py`, `phase4_insn_demo.py`;
   `RISCV-on-PYNQ-Z1/bitstreams/phase3|phase4/README.md` for the board numbers.

**Lab.** `cd rtl/matmul && make sim`. Then draw, on paper, the operand
skew for a 3×3 output-stationary array: which A and B element enters which
PE at cycle t, and when the last product reaches PE(2,2).

**Check yourself.**
- The Phase 4 unit needed ≈ 312 cycles for 512 MACs. Where do the cycles
  go (compute vs DDR transfers vs control)? What does that suggest?
- Why is "one independent 8×8×8 job at a time, straight from DDR" a
  bandwidth problem, and what does on-chip buffering buy?

---

## Stage 4 - Accelerator architecture: memory hierarchy and dataflow

**Goal.** Reason about an accelerator the way its designer did: budget the
on-chip memory, choose a dataflow, decide what overlaps with what.

**Read.**
- Sze, Chen, Yang, Emer, *Efficient Processing of Deep Neural Networks*
  (Morgan & Claypool, 2020) - dataflows (weight/output/row stationary),
  memory hierarchy, the cost of data movement. Chen, Emer, Sze, "Eyeriss",
  ISCA 2016 for the dataflow taxonomy.
- Genc et al., "Gemmini: Enabling Systematic Deep-Learning Architecture
  Evaluation via Full-Stack Integration", DAC 2021 - the closest relative
  of this design (scratchpad + accumulator, decoupled load/store/execute
  queues, RISC-V custom instructions).
- Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance
  Model", *CACM* 2009.

**In this repo.** `docs/double_buffer_design.md`, in this order:
§1 goals and decisions → §2 architecture → §3 memories (BRAM budget, banks,
local addresses, data layouts) → §4.1-4.2 the EX command and K-streaming →
§7 the bank scoreboard → §10 expected performance → §12 milestones. Then
read the *(M1)*…*(M4)* notes to see which decisions changed after
measurements, and why (§10.1: one HP port already at 99 %, so M2 cut
command overhead instead of adding ports; §10.3: three ports not needed).

**Lab.** For a GEMM of M×N×K on D = 16 with one port at 7.9 B/cycle:
compute (a) the array time with 16 strips × N/D tiles × (K + 2(D−1) + 1)
cycles, (b) the DMA time for A + B + C, (c) the roofline bound. Compare
with the table in §10.3 for 256³ (93k cycles measured before the schedule
tuning, 82.8k after). Where did the remaining cycles go?

**Check yourself.**
- Why two banks per memory? What exactly overlaps with what in the GEMM
  schedule (`firmware/gemm/gemm_fw.c`)?
- Why is the SPAD : ACC split 1 : 1 : 2 (128 KB / 128 KB / 256 KB)?
- What does output-stationary + K-streaming save compared with loading a
  new B tile for every 8×8×8 job?

---

## Stage 5 - The accelerator RTL, module by module

**Goal.** Be able to modify any engine and predict the effect. Read in this
order; each module is small enough for one sitting. Keep
`rtl/sysarray/README.md` (file table) and `sa_defs.vh` (packet layout,
memory ids, keys, error codes) open next to it.

### 5.1 Memories - `sa_tdpram.v`, `sa_bankmem.v`
- UG901 byte-write TDP template kept literal so Vivado infers RAMB36 with
  byte enables; the zero `initial` block (BRAM INIT values).
- Banking: word address → bank = upper half; several requesters per side
  (`NA`/`NB`) multiplexed onto one port.
- *Check:* why did a "cleaner" RAM description fall back to LUTs or refuse
  to map in M1? (UG901 "RAM inference" section.)

### 5.2 Array and EX engine - `sa_array.v`, `sa_ex.v`
- PE variants: `sa_pe_dsp` (`use_dsp = "yes"`) and `sa_pe_lut`,
  `DSP_COLS` chooses per column (M4: 8 DSP + 8 LUT columns).
- K-streaming feed: A strip in k-tile-major layout, one SPAD_A and one
  SPAD_B word per cycle; `rowreg`/`bsr` build the skew.
- `swap` → shadow accumulators → D-cycle drain overlapping the next tile;
  accumulate mode reads ACC first (2D cycles).
- M2 repeat: one command = several C tiles (`bstep`, `cstep`, `crow`).
- *Check:* why does a command take K + 2(D−1) + 1 cycles? What limits
  back-to-back tiles?

### 5.3 DMA - `sa_ld.v`, `sa_st.v`
- Command shape: rows × row_bytes, pitch; LINEAR vs INTERLEAVE (A strips);
  flattening contiguous rows into one long row.
- Bursts of ≤ 16 beats, never across 4 KB; up to 8 outstanding per port;
  each burst remembers its local word/lane so beats land by address.
- Lanes: a 64-bit AXI beat is 1/(D/8) of a SPAD word and 1/(D/2) of an ACC
  word.
- *Check:* why are 8-byte rows at D = 16 single-beat bursts, and what does
  that do to bandwidth (bwtest test 4: 3.81 B/cycle)?

### 5.4 Scheduler - `sa_sched.v`
- Input queue → decode stage 1 (fields, span) → stage 2 (validation, bank
  masks) → dispatch. Two stages because one did not meet timing in M1.
- Scoreboard: per engine, a queue of read/write bank masks; RAW, WAR, WAW
  across engines; EX and VE additionally never share a bank (port sharing).
- Sticky error: a bad command or an AXI error stops dispatch until
  `clear_error`; queued commands stay put until then.
- *Check:* why is DDR **not** tracked, and what must software do about it
  (`mat_fence` between a store and a later load of the same bytes)?

### 5.5 PCPI front end - `sa_pcpi.v`
- funct7 = 0/1/2 decode; `mat_cfg` / `vec_cfg` registers captured into the
  command at queue time (the `pkt_*` functions of `sa_defs.vh`); `wait` held
  while the queue is full.
- `mat_perf` (funct3 = 5) reads / controls the counters; `mat_submit`
  (funct3 = 6) starts a descriptor list, after which queued PCPI commands
  wait until the list is done (program order).

### 5.6 Vector engine - `sa_ve.v`
- Read unit (src1 words, then src2 words; broadcast read once), four
  compute stages (op → RELU + requant multiply → rounding shift → zero
  point + clamp + saturate + pack), 8-entry output FIFO, credit counter.
- Types tied to memories: int8/int16 in SPAD, int32 in ACC.
- *Read:* Jacob et al., "Quantization and Training of Neural Networks for
  Efficient Integer-Arithmetic-Only Inference", CVPR 2018 - where
  `((x·scale + 2^(shift−1)) >> shift) + zp` comes from.
- *Check:* why can the FIFO never overflow? Why did splitting stage 3 need
  a deeper FIFO to keep the throughput (M3)?

### 5.7 Legacy sequencer - `sa_legacy.v`
- The Phase 2 CSR map and funct7 = 0 on the new datapath: one job = LD A,
  LD B, EX (Kt = 1), ST C through the normal scheduler, in a reserved slot.
- At D = 16 the 8×8 job is zero-padded by the slot's power-on zeros.
- `L_ERRW`: wait for idle engines before clearing a job's error (M4 fix).

### 5.8 Top level - `sa_unit.v`
- Parameters (`D`, `DSP_COLS`, `NPORTS`, `SPAD_WORDS`, `ACC_WORDS`, `PERF`),
  the three AXI master ports (NPORTS used), `X_INTERFACE_INFO` attributes
  that make Vivado see proper AXI interfaces, memory port wiring per engine.
- Command-source priority into the scheduler: legacy > descriptor fetch >
  PCPI. Port 0's read channel is shared by LD and the fetch unit: the AR
  grant is locked until its handshake, and an owner FIFO in AR order routes
  the returning beats (no AXI IDs, so bursts return in order).

### 5.9 Performance counters - `sa_perf.v`
- 32 counters fed by `perf_ev` ports of the scheduler, PCPI and engines -
  existing signals only, registered once so no engine path gets longer.
  Two combinational read ports (PCPI, CSR mirror 0x40–0xBC), control from
  either; `PERF = 0` removes the block.
- Read the definitions in
  [`perf_counters_and_desc_dma_plan.md`](perf_counters_and_desc_dma_plan.md)
  §1.2: which signal means "the head is blocked by a hazard", "the front end
  starves", "the array does useful work".
- *Check:* why does `EX_USEFUL × D²` have to equal M·N·K exactly, and why is
  that a better test than "the numbers look plausible"?

### 5.10 Descriptor fetch unit - `sa_cmdfetch.v`
- `mat_submit(list, count)`: 64-byte descriptors fetched with 8-beat bursts
  (4 in flight, LUTRAM FIFO), assembled, decoded with the same `pkt_*`
  functions as PCPI; FENCE / JUMP / END handled inside; DRAIN drops bursts
  still in flight after JUMP, END or an error.
- Format and semantics: `docs/double_buffer_design.md` §8.6.
- *Check:* why must JUMP drain the prefetched bursts instead of just
  changing the fetch address? Why is opcode 0x00 invalid on purpose?

**Lab (Stage 5).** Pick one change and carry it through simulation:
- make `sa_ld` issue at most 4 outstanding bursts (`QD`) and measure
  `tb_sa_dma` burst counts / cycles;
- add a new VE op (e.g. absolute value) end to end: `sa_defs.vh` op code,
  `sa_ve.v` stage 1, `sa_sched.v` validation, `tb_sa_ve.v` reference model;
- add a performance counter for "EX idle while its next command waits in the
  engine queue" and an invariant that checks it.

---

## Stage 6 - Verification

**Goal.** Build the confidence to change RTL: golden models, reference
models, randomized streams, and tests that are proven to catch bugs.

**Read.**
- Spear & Tumbush, *SystemVerilog for Verification* (3rd ed.) - chapters
  on testbench architecture, randomization and functional coverage (the
  ideas carry over to the plain-Verilog testbenches here).

**In this repo** (`rtl/sysarray/sim`, `firmware/sim`).
| Testbench | Technique to study |
|---|---|
| `tb_sa_ex.v` | behavioral golden model per command; commands issued back to back |
| `tb_sa_dma.v` | AXI memory model with random stalls, SLVERR injection and protocol checks (length, 4 KB, WLAST); byte-level reference incl. neighbours |
| `tb_sa_ve.v` | reference model of every op/type; per-command check, then 60 back-to-back commands and a full compare |
| `tb_sa_unit.v` | the whole unit: NumPy golden vectors (legacy), GEMM golden model, **random command stream vs a sequential reference** (checks the scoreboard), error paths; performance counters against **exact invariants**; the same GEMM and random stream **as descriptor lists** (same reference model) plus directed ordering / FENCE / JUMP / error tests |
| `firmware/sim/tb_system.v` | real firmware on PicoRV32 + the unit + DDR model (`GEMM_TEST`, `VEC_TEST`, `BW_TEST`, `DESC_TEST`); counters read back and checked |
| `firmware/desc_run/gen_desc_cases.py` | **co-simulation**: lists built by the Python driver (`DescList`, `build_gemm_list`) are executed by the RTL and compared byte by byte with NumPy - the driver's encoding is verified before it reaches the board |

**Lab.**
1. `cd rtl/sysarray && make test PYTHON=<python with numpy>` and
   `make test16`.
2. **Mutation check:** break something on purpose and confirm a test fails
   - e.g. drop the WAR term from the scoreboard hazard in `sa_sched.v`, or
   the rounding constant in `sa_ve.v`. A test that passes a mutated design
   is not testing that behavior. Restore the file afterwards.
3. `cd firmware/gemm && make sim SIM_D=16` - read how `tb_system.v` plays
   the ARM side.

**Pitfalls this project hit in xsim 2024.1** (see `rtl/sysarray/README.md`):
- Two calls of the same static (non-`automatic`) function in one
  expression returned the same value → reference models silently wrong.
  Use `function automatic` and one call per statement.
- `f(..) !== g(..)` on wide function results misreported mismatches.
  Compare through registers.
- The `Makefile` filters simulator output to lines starting with `TB` - a
  `$display("DBG ...")` is invisible there; run `xsim tb -R` in the build
  directory to see everything.
- Simulation timing is not board timing: the testbench memories answer
  faster (instruction fetch) or slower (the DDR model) than the board. A
  check derived from simulated cycle counts (the counter window tolerance)
  failed only on the board - check invariants, not simulated timings.

**Check yourself.**
- Why does the random stream compare *final* memory contents against a
  *sequential* model, and what class of bug does that catch that a
  per-command check cannot?
- Why do the testbenches check a margin around each destination region?

---

## Stage 7 - Synthesis, timing closure and resources

**Goal.** Read utilization and timing reports, find the critical path, and
fix it by pipelining or restructuring; trade DSPs, LUTs and BRAMs.

**Read.**
- AMD UG901 *Vivado Synthesis* (RAM/DSP inference, attributes), UG903
  *Using Constraints*, UG906 *Design Analysis and Closure Techniques*
  (reading timing paths, logic levels, WNS/TNS).

**In this repo.**
1. `rtl/sysarray/synth_ooc.tcl`, `make synth SYNTH_GEN="D=16 NPORTS=1"` -
   out-of-context place and route of the accelerator alone, 50 MHz.
2. Case studies from the history:
   - M1: scheduler at −0.72 ns → split decode into two stages.
   - M3: VE final stage 31 logic levels (WNS +0.78 ns) → split into two
     stages, WNS +2.49 ns, FIFO 4 → 8 to keep throughput.
   - M4: D = 16 costs 21.3k LUT in the array (128 LUT PEs) and 14.5k in the
     VE; the board build sits at 89.5 % LUT, 84 % DSP, 93 % BRAM.
   - M4: the block-design module reference froze `SPAD_WORDS`/`ACC_WORDS`
     at their D = 8 defaults → 258 RAMB36 → the build sets them with D.
   - M5: the counters cost +0.65k LUT, the descriptor fetch unit +0.66k
     (its FIFO is LUTRAM, 44 LUTs, so the last free BRAMs stay free); the
     build is at 92 % LUT - the next feature needs `DSP_COLS = 10`.
3. `docs/double_buffer_design.md` §11 - estimates vs measured resources.

**Lab.**
1. Run `make synth` at D = 8 and D = 16; compare
   `build/synth/utilization_hier.rpt` per module and the worst path in
   `timing_summary.rpt` (source, destination, logic levels).
2. Change `DSP_COLS` to 10 at D = 16 (`SYNTH_GEN="D=16 NPORTS=1 DSP_COLS=10"`)
   and predict, then measure, the LUT and DSP change.

**Check yourself.**
- A path has 7 logic levels and +1.3 ns slack at 20 ns; another has 31
  levels and +0.8 ns. Which one do you fix first before adding logic?
- Why is the DSP48E1 a natural fit for an int8 MAC with a 32-bit
  accumulator, and what does a LUT PE cost instead?

---

## Stage 8 - System integration, firmware scheduling and the board

**Goal.** Run the hardware from Python, measure it, and learn that the
schedule in firmware matters as much as the RTL.

**In this repo.**
1. `driver/pynq_matmul.py` - `MatmulOverlay`: loading firmware into the
   BRAM, the mailbox protocol (`_run`), `gemm()`, `vector()`, `bandwidth()`,
   NumPy golden models (`golden`, `vector_golden`, `qgemm_golden`); counters
   in `stats["perf"]` with `perf_breakdown()`; descriptor lists: `DescList`,
   `build_gemm_list` / `build_vector_list` (the firmware schedules as data),
   `run_list`, `gemm_list`, `vector_list`.
2. `firmware/gemm/gemm_fw.c` - the GEMM schedule: resident B, A strips
   alternating banks, one repeat `mat_exec` per strip, the int8 epilogue on
   the VE, **A prefetch** and **B split** (M4 tuning; `MBOX_GEMM_FLAGS`).
3. `firmware/vector/vector_fw.c` - streaming a long vector through the local
   memories in chunks that alternate banks.
4. `firmware/bwtest/bwtest_fw.c` + `notebooks/m2_bw_test.py` - measuring the
   HP port.
5. `firmware/desc_run/desc_run_fw.c` - the whole firmware side of a
   descriptor list: load the bases, one `mat_submit`, one `mat_fence`.
6. `notebooks/m4_demo.py`, `m4_sched_tune.py`, `m4_perf.py` (cycle
   breakdown), `m5_desc_demo.py` (PCPI vs list), and each
   `RISCV-on-PYNQ-Z1/bitstreams/<milestone>/README.md` for board results.

**Lab.**
1. Copy `build/deploy_m5/*` (or `bitstreams/m5/` + the firmware and driver)
   to the board and run `m4_demo.py` and `m4_perf.py` (command lines in
   their docstrings). Compare with `bitstreams/m5/README.md`.
2. Run `m4_sched_tune.py` and explain each column with the two schedule
   changes described in `gemm_fw.c`.
3. Change one schedule decision in `gemm_fw.c` (e.g. the B-split threshold
   or the order of the epilogue store), rebuild with `make`, verify with
   `make sim SIM_D=16`, then measure on the board. No new bitstream needed.
4. Build a `DescList` by hand for an operation the firmware does not have
   (e.g. two chained vector ops with a FENCE between a store and a reload),
   run it with `run_list`, and check it against NumPy. Then reuse it on new
   buffers through the relocation bases.

**Check yourself.**
- Why does in-order dispatch make the *order in which firmware queues
  commands* a performance decision?
- Why did splitting B help 256³ but hurt 64³?
- `m4_demo.py` shows 32×256×64 at 17 % of peak. Which resource bounds it,
  and why does the int8 epilogue help there more than extra HP ports would?
- Why are descriptor lists built by the ARM and not by PicoRV32?

---

## Stage 9 - Performance engineering

**Goal.** Predict performance before building, then explain every gap
between prediction and measurement.

**Material.** `docs/double_buffer_design.md` §10 (model), §10.1 (measured
bandwidth), §10.2 (M2 results), §10.3 (M4 results and the three-port
decision, schedule tuning table), §10.4 (the **measured** cycle breakdown
from the counters), §10.5 (descriptor lists vs PCPI).

**Exercises.**
1. Build a spreadsheet roofline for this board: compute ceiling D² MAC/cycle
   (D = 8, 16), bandwidth ceilings 7.9 B/cycle (one port, one direction) and
   15.8 B/cycle (LD + ST together). Place the measured GEMMs on it.
2. For 256³ at D = 16, estimate how the 82.8k cycles split into array
   time, skew fill, waiting for banks and front end - then compare with the
   counters in §10.4 (useful 79 %, fill 9.3 %, EX blocked 8.8 %). Which
   remaining item would a finer-grained scoreboard remove?
3. Estimate what three HP ports would give for 256³ and for 32×256×64 and
   compare with the decision in §10.3.
4. The counters predicted ≈ 1.6× for 64³ with descriptor lists; the board
   gave 1.23× (§10.5). With a list the head is still empty 24 % of the
   time: work out the fetch unit's minimum cycles per descriptor and what a
   deeper prefetch or a faster decoder would change.

---

## Stage 10 - Where to go next (capstone ideas)

Each is a self-contained project on top of M5; they are listed roughly by
difficulty. (Already done in this repository, as worked examples: the
performance counters and the descriptor DMA - see
[`perf_counters_and_desc_dma_plan.md`](perf_counters_and_desc_dma_plan.md)
for how such a project is planned, verified and measured.)

1. **Board stress test**: run the random command stream as descriptor lists
   (`DescList`) with a Python reference model - millions of commands on the
   board.
2. **Fix the int8 epilogue** (≈ 10 % at 256³; the counters show the VE
   command waiting at the head 80 % of the time): overlap strip i's VE with
   strip i+1's EX.
3. **Faster descriptor fetch**: decode while the next descriptor arrives,
   deeper prefetch - close the gap between 1.23× and the ≈ 1.6× the counters
   predicted for 64³.
4. **Event trace buffer**: log dispatch / done events with timestamps into
   a BRAM FIFO and render an engine timeline (Perfetto).
5. **ARM direct submission**: a doorbell register on the AXI GP port so the
   ARM submits lists without PicoRV32 (block design change).
6. **Finer-grained scoreboard**: track address ranges (or sub-banks)
   instead of whole banks; removes the need for the B split.
7. **DDR hazard tracking** in hardware so software needs fewer fences.
8. **10 DSP columns** to free LUTs, then spend them (e.g. a wider VE).
9. **A weight-stationary variant** of the array and a comparison on
   convolution-shaped GEMMs.
10. **Phase 5 hand-off**: the MLIR lowering can emit descriptor lists (data)
    instead of `.insn` code - which constraints (alignment, D multiples,
    fences, reserved slot, list format) must the compiler guarantee?

---

## Appendix A - Lessons from this project's bugs

| Symptom | Cause | Fix / where |
|---|---|---|
| PYNQ `KeyError 'S_AXI'` loading the overlay | block design container (BDC) not understood by the `.hwh` parser | hierarchical cell instead of a BDC (`pico_bit.tcl`) |
| Accelerator interface inferred as a BRAM port | Vivado guessed the interface from signal names | `X_INTERFACE_INFO` attributes (`sa_unit.v`) |
| RAM not mapped to RAMB36 | description differed from the inference template | UG901 byte-write TDP template, literal (`sa_tdpram.v`) |
| DMA hangs after many bursts | 32-bit compare of wrapping queue pointers | N-bit `used` wires (`sa_ld.v`, `sa_st.v`) |
| Legacy tile split into single-beat bursts | one burst per 8-byte row | flatten contiguous rows (`sa_ld.v`) |
| Scheduler fails timing (−0.72 ns) | decode + checks + hazard in one cycle | two-stage decode (`sa_sched.v`) |
| Negative REQUANT scale treated as unsigned | `?:` mixing a signed product with an unsigned concatenation | separate `if/else` branches (`sa_ve.v`) |
| VE results wrong only in the full-unit random test | xsim static-function aliasing in the *reference model* | `automatic` functions, one call per statement (testbenches) |
| Next legacy job fails after an AXI error (D = 16) | error cleared while another engine of the job was still running | `L_ERRW` waits for idle engines (`sa_legacy.v`, M4) |
| D = 16 build needs 258 RAMB36 | module reference froze derived parameters at D = 8 | set `SPAD_WORDS`/`ACC_WORDS` with `D` (`pico_bit.tcl`, `-sa_d`) |
| `make` warns "overriding recipe" | inline comment left trailing spaces in a variable | comments on their own line (`firmware/common.mk`) |
| `xelab` runs for minutes and takes 13 GB | two generics passed as one `-generic_top "NP=1 D=16"` | one `-generic_top` per generic (`rtl/sysarray/Makefile`) |
| Firmware slower after reading D at run time | divisions by the run-time D inside the strip loop (PicoRV32 `div`) | hoist them before the timed region (`gemm_fw.c`) |
| Counter check passed in simulation, failed on the board | the counting window includes a few instructions, fetched much slower on the board | check a constant offset range, not a simulated tolerance (`m4_perf.py`) |
| Read error of a descriptor fetch lost (found in review) | a non-blocking assignment before a `case` was overwritten by one inside it | apply the error after the `case` (`sa_cmdfetch.v`) |
| `xvlog`: redeclaration of ANSI port | a new port had the name of an existing internal register | rename the port (`sa_legacy.v`) |

## Appendix B - Glossary

- **D** - array size (8 or 16); also the vector length VL and the SPAD word
  size in bytes. ACC words are D × int32.
- **SPAD_A / SPAD_B / ACC** - the local memories (A strips, B strips,
  accumulators / int32 data); each has two banks.
- **LADDR** - local address: `mem << 28 | word` (`SA_LADDR`).
- **Strip / tile** - D rows of A (a strip) × one D-column block of B gives a
  D×D C tile; `mat_exec` with repeat computes a row of C tiles.
- **K-streaming** - the array consumes K one step per cycle from SPAD
  instead of loading D×D tiles; C stays in the PEs (output-stationary).
- **Scoreboard** - per-engine lists of banks being read/written; a command
  dispatches only when it has no RAW/WAR/WAW conflict.
- **Sticky error** - the first failing command stops dispatch until
  `mat_reset` (`clear_error`).
- **PCPI** - PicoRV32's co-processor port for custom instructions.
- **Mailbox** - the last 256 bytes of the program BRAM, shared by ARM and
  RISC-V (`firmware/include/mailbox.h`).
- **HP port** - Zynq high-performance AXI slave port into the DDR
  controller (64-bit, AXI3).
- **Performance counter area** - the 256 bytes below the mailbox
  (0x1E00) where firmware copies the counters after a measurement window.
- **Front end starve** - the scheduler has no command at its head while an
  engine is busy: the hardware waits for the next command.
- **Descriptor list** - 64-byte commands in DDR built by the ARM and run by
  the fetch unit after one `mat_submit`; FENCE / JUMP / END control it.

## Appendix C - References

Books and papers
- D. Harris, S. Harris, *Digital Design and Computer Architecture, RISC-V Edition*, 2021.
- C. Spear, G. Tumbush, *SystemVerilog for Verification*, 3rd ed., 2012.
- V. Sze, Y.-H. Chen, T.-J. Yang, J. Emer, *Efficient Processing of Deep Neural Networks*, Morgan & Claypool, 2020.
- H. T. Kung, "Why Systolic Architectures?", *IEEE Computer* 15(1), 1982.
- N. P. Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit", ISCA 2017.
- Y.-H. Chen, J. Emer, V. Sze, "Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks", ISCA 2016.
- H. Genc et al., "Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation via Full-Stack Integration", DAC 2021.
- B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018.
- S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", *CACM* 52(4), 2009.

Specifications and vendor guides
- ARM IHI 0022, *AMBA AXI and ACE Protocol Specification*.
- *The RISC-V Instruction Set Manual, Volume I: Unprivileged ISA*.
- AMD/Xilinx UG585 (Zynq-7000 TRM), UG474 (7 Series CLB), UG473 (7 Series Memory Resources), UG479 (7 Series DSP48E1), UG901 (Vivado Synthesis), UG903 (Using Constraints), UG906 (Design Analysis and Closure Techniques), UG1037 (AXI Reference Guide).
- PicoRV32: github.com/YosysHQ/picorv32 (README: PCPI, AXI adapter).
- PYNQ documentation: pynq.readthedocs.io.

In this repository
- `docs/double_buffer_design.md` - the accelerator design (M1-M5), incl. §8.6 descriptor lists and §10.4 / §10.5 measurements.
- `docs/perf_counters_and_desc_dma_plan.md` - plan, verification and results of the counters and the descriptor DMA (Chinese).
- `docs/execution_walkthrough.md` - one GEMM through every layer (Chinese).
- `docs/custom_isa_encoding.md`, `docs/memory_model.md` - Phase 4 ISA and address maps.
- `rtl/matmul/README.md`, `rtl/sysarray/README.md` - the two RTL units.
- `RISCV-on-PYNQ-Z1/bitstreams/*/README.md` - board results per milestone.
- `RISCV-on-PYNQ-Z1/pynq-z1-riscv-accelerator-plan-v2.md` - the original project plan (Chinese).

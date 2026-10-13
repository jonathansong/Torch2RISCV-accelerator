# Performance counters and descriptor DMA: plan, implementation and results

> English translation of [`perf_counters_and_desc_dma_plan.md`](perf_counters_and_desc_dma_plan.md) (Chinese original, translated 2026-10-13 at `db07c59`). The Chinese document is authoritative; if the two differ, follow the original.

> **PYNQ-Z1 paths**: this document records work done on the PYNQ-Z1. The `RISCV-on-PYNQ-Z1/...` (block design, constraints), `bitstreams/...` (builds verified on the board) and `iree-sa/l0` it mentions are on the `pynq-z1` branch (tag `v1.0-pynq-z1`); on main, PicoRV32 and its IP are in `rtl/picorv32/` and `rtl/ip/`, and the current overlay is in `boards/kv260/`.

Status: **both parts are done and verified on the board**. Below are first the completion records of each step (differences from the original plan are written into them),
then the original plan, kept as the design rationale.

- **Part 1 (performance counters, P1–P4) done and verified on the board** (`RISCV-on-PYNQ-Z1/bitstreams/m4p/`)
- **D0 decision done** (below)
- **Part 2 (descriptor DMA) D1–D5 done and verified on the board** (`RISCV-on-PYNQ-Z1/bitstreams/m5/`); D6 (direct submission from the ARM) not done

Design details follow the design document: §8.6 (descriptor format and semantics), §10.4 (measured counters), §10.5 (lists versus PCPI).

> **P1 completion** (simulation stage; board in P3): `rtl/sysarray/sa_perf.v` + event ports in each module + `mat_perf`
> (funct7 = 1, funct3 = 5) + CSR mirror 0x40–0xBC / `PERF_CTRL` 0x3C + CAPS bit 20.
> `tb_sa_unit` adds 365 counter checks (exact invariants, directed RAW conflicts, freeze / clear / CSR mirror);
> D = 8 / 16 and `PERF = 0` all pass; all 5 mutations are caught; `make test`, `make test16` and
> the 4 firmware system simulations do not regress. OOC (D = 16, 50 MHz): 42,819 LUT (+988, `sa_perf` 558),
> 29,729 FF (+953), BRAM / DSP unchanged, WNS +1.631 ns (M4: +1.864 ns).
>
> **P2 completion** (simulation): intrinsics (`mat_perf_ctl` / `mat_perf_read`, `sa_perf_begin` /
> `sa_perf_end`, which check CAPS bit 20 first and do not execute `mat_perf` on old hardware); program BRAM 0x1E00–0x1EFF is
> the counter area, `link.ld` shrunk to 0x1E00, all firmware relinked; `gemm_fw` / `vector_fw` switch the counters
> on and off around the timing window and dump them, with the count written to `MBOX_PERF_COUNT` (0x8C); `tb_system.v` reads them back and checks the invariants
> (gemm 178 items, vector 49, all passing at D = 8 / 16) and prints the cycle breakdown; driver `stats["perf"]`,
> `PERF_NAMES`, `perf_breakdown()` / `format_breakdown()`; board script `notebooks/m4_perf.py`
> (dry run with a fake overlay passes). Differences from 1.5: the driver always reads back the counters (`stats["perf"]`,
> None on old hardware), with no separate `perf=` parameter; the tolerance between `CYCLES` and the firmware's `rdcycle` window is +128
> cycles (a few instructions at each end fall inside the counting window).
>
> **P3 completion** (board): bitstream (D = 16, PERF = 1) WNS +1.904 ns, 48,267 LUT (90.7%);
> `m4_perf.py`: 7 GEMMs + 4 vector operations all correct, **69/69 counter checks pass**; the `m4_demo.py`
> regression all PASS, performance identical to the tuned M4. On the board `CYCLES` exceeds the firmware's `rdcycle` window by a **fixed**
> offset (GEMM 131, vector 175 cycles: a few instructions at each end are fetched over AXI, about 10 cycles each), so `m4_perf.py`
> checks 0 ≤ offset ≤ 512 instead of the simulation's +128.
>
> **P4 completion**: the measured cycle breakdown is in design document §10.4 (replacing the estimate of §10.3), and the bitstream archived in
> `bitstreams/m4p/`.
>
> **D0 conclusion: descriptor DMA is only worth it for small operations.**
> - Large GEMMs (256³ and above): the CPU spends 61–72% of its time stuck on a full queue and the head of the queue is empty only 2–4%; the frontend is not the bottleneck,
>   and descriptor DMA brings no gain. Array efficiency is 79–87%; the rest is mainly structural fill (2(D−1) steps per tile) and waiting for banks (5–11%).
> - Small GEMMs and short vector operations: an empty head of queue or fully idle accounts for 60–94% (64³ 86.8%, int8 add 4096 93.9%);
>   the bottleneck is PicoRV32 issuing commands. This is where descriptor DMA pays off; 64³ estimated to drop from about 4.2k to about 2.5k cycles
>   (then limited by the 16 KB C write-back and compute), subject to measurement.
> - DMA at 7.4–7.97 B/cycle while busy, close to the port limit, confirming again that a third port is not needed.
>
> **D1–D3 completion** (simulation): format and semantics written into design document §8.6; shared `pkt_*` functions (`sa_defs.vh`);
> `sa_cmdfetch.v` + port 0 read arbitration (ownership FIFO) + `mat_submit` (funct3 = 6) + BASE0–3 + status CSRs 0xC0–0xD0 +
> `ext_status` bit 24 + CAPS bit 21 + counters 28–31. `tb_sa_unit`: list GEMM, list random command streams (against an in-order reference model),
> 26 directed checks (mixed order, FENCE / no FENCE / FENCE_BEFORE, JUMP, END, count, RELOC, 5 kinds of errors and recovery, FETCH_DESC);
> all 4 mutations caught; `make test` / `test16` / `PERF = 0` and all firmware system simulations do not regress. Software: `firmware/desc_run`,
> driver `DescList` / `build_gemm_list` / `build_vector_list` / `run_list` / `gemm_list` / `vector_list`;
> **co-simulation**: 7 lists built by the driver (GEMM, B split, int8, vector) executed by the RTL on `tb_system`, byte-exact at D = 8 / 16;
> board script `notebooks/m5_desc_demo.py`. OOC (D = 16): 43,540 LUT (+721, `sa_cmdfetch` 816), 30,970 FF,
> BRAM / DSP unchanged, WNS +2.108 ns.
>
> Differences from the plan in 2.x: the descriptor FIFO uses LUTRAM (32 × 64 bits, 4 descriptors prefetched) instead of BRAM; only the header's reserved bits and the opcode are checked;
> END's IRQ flag is reserved and not implemented; FENCE is judged from the scheduler's queue occupancy and the engines' busy bits; **lists are built by the ARM** (PicoRV32 needs about 160 cycles
> to write one descriptor, slower than issuing over PCPI directly), so the list modes of `gemm_fw` / `vector_fw` were not done; instead the `desc_run`
> firmware + list building in the driver.
>
> **D4 / D5 completion** (board): bitstream WNS +2.460 ns, 48,928 LUT (92.0%). `m5_desc_demo.py` all correct:
> GEMM 16³ **4.19×**, 64³ 1.23×, int8 64³ 1.52×, 128³ and above 1.00×; vector int8 add 1.52×, broadcast max **2.60×**,
> int32→int8 1.27×; the same list reused three times through BASE relocation, all correct. `m4_perf.py` 69/69, `m4_demo.py` unchanged.
> Results in design document §10.5. The gain at 64³ (1.23×) is below D0's estimate of about 1.6×: with lists the head of the queue is still empty 24% of the time;
> the fetch unit needs at least 9 cycles per descriptor, and each waits for its own 8-beat read.
>
> **Later evolution (LLM plan L1–L5b, see [`llm_inference_plan.md`](llm_inference_plan.md))**: the descriptor format and fetch unit of this document are
> the foundation of LLM inference and were extended later:
> - **L1 command-processing extensions** (§5.3), CAPS bit 24:
>   - BASE0–15;
>   - PARAM0–7;
>   - w0[31:16] holds two dynamic slots (formerly reserved bits);
>   - new opcodes: 0x13 LOOP_END, 0x14 SETREG, 0x15 CALL, 0x16 RET, 0x17 LDPARAM;
>   - JUMP can use relative addresses.
> - **Host notification** is implemented with `mat_notify` and `notify_irq` (CAPS bit 22). END's IRQ bit is **still not implemented**:
>   the ARM cannot access the accelerator's CSRs, so notification is an edge interrupt raised by the resident firmware `firmware/rt` when a command-ring entry completes.
> - **L2's VE descriptors**: use w3[63:53], w5[63:32], w6, w7; types[5:4] is src2's type; op 6 is TRANSPOSE (CAPS bit 23).
> - **No new counter events**: L2's fp32 VE shares the existing VE events with the integer VE, ORed.
> - The current bitstream is `bitstreams/l2` (D=8, 44,005 LUT).
>   stories15M's whole list executes 2,477 descriptors per token (L4); batching 8 sequences executes 7,231 per step (L5b).

1. **Performance counters** (part 1): let the hardware report where its cycles go;
2. **Descriptor DMA** (part 2): software writes the commands as a table in DDR, and the hardware fetches and executes it itself.

There is an ordering dependency: performance counters first, to quantify the command-issue overhead with board data, then decide the investment and optimization focus of descriptor
DMA. This continues the project's "measure before building" approach; see the three-port DMA trade-off in
design document §10.1.

Related documents: [`double_buffer_design.md`](double_buffer_design.md) (architecture; §7 scoreboard, §8 instructions,
§10.3 M4 measurements), [`execution_walkthrough.md`](execution_walkthrough.md) (layer-by-layer execution of one GEMM).

---

## Contents

- [0. Background](#0-background)
- [1. Performance counters](#1-performance-counters)
  - [1.1 Questions to answer](#11-questions-to-answer)
  - [1.2 Counter definitions](#12-counter-definitions)
  - [1.3 Derived metrics](#13-derived-metrics)
  - [1.4 Hardware design](#14-hardware-design)
  - [1.5 Software interface](#15-software-interface)
  - [1.6 Verification](#16-verification)
  - [1.7 Resources and timing](#17-resources-and-timing)
  - [1.8 Implementation steps](#18-implementation-steps)
- [2. Descriptor DMA](#2-descriptor-dma)
  - [2.1 Motivation](#21-motivation)
  - [2.2 Descriptor format](#22-descriptor-format)
  - [2.3 Execution semantics](#23-execution-semantics)
  - [2.4 Hardware design](#24-hardware-design)
  - [2.5 Instructions, CSRs and status](#25-instructions-csrs-and-status)
  - [2.6 Software](#26-software)
  - [2.7 Verification](#27-verification)
  - [2.8 Resources](#28-resources)
  - [2.9 Implementation steps](#29-implementation-steps)
- [3. Overall plan](#3-overall-plan)
- [4. Risks and mitigations](#4-risks-and-mitigations)
- [Appendix A: the extended funct7 = 1 encoding](#appendix-a-the-extended-funct7--1-encoding)
- [Appendix B: list of modified files](#appendix-b-list-of-modified-files)

---

## 0. Background

On the board M4 (D = 16) measured a 256³ GEMM at 82,813 cycles, 202.6 MAC/cycle (79% of peak),
but the cycle breakdown (design document §10.3) was **estimated**: array compute about 73.5k, B load about 4.1k,
and about 5k unexplained. Two other obvious inefficiencies:

- 64³ reaches only 25% of peak;
- int8 vector addition only 1.19 elements/cycle.

Are they limited by command issue speed, or by DMA or scheduling? There is currently no data to answer this.

The only observability today is `rdcycle` in the firmware (total cycles) and `ext_status` (idle, errors).

---

## 1. Performance counters

### 1.1 Questions to answer

| # | Question | Answered by which counters |
|---|---|---|
| Q1 | How many cycles is the array doing useful multiply-accumulate? How much is pipeline fill, write-back waits, idle? | `EX_USEFUL`, `EX_STEP`, `EX_SWAPWAIT`, `CYCLES` |
| Q2 | Which kinds of commands does the scoreboard stall, and for how long? | `HAZ_LD/ST/EX/VE` |
| Q3 | Can the frontend (firmware issue) keep up? | `STARVE`, `PCPI_QFULL`, `ALL_IDLE` |
| Q4 | What is the actual DMA bandwidth and the share of AXI back-pressure? | `LD/ST_BUSY`, `LD/ST_BEATS`, `LD_ARSTALL`, `ST_WSTALL` |
| Q5 | What share of VE time goes to port conflicts and a full output FIFO? | `VE_ACTIVE`, `VE_RDBLOCK`, `VE_CREDIT` |

### 1.2 Counter definitions

All counters are 32 bits (overflow after about 86 seconds at 50 MHz) and count only inside the enable window. Event conditions
are all expressed with **existing RTL signals** and change no engine's behavior.

| No. | Name | +1 when | Signal source |
|---|---|---|---|
| 0 | `CYCLES` | Every cycle inside the enable window | — |
| 1 | `CMD_LD` | `dispatch && h_eng == ENG_LD` | `sa_sched.v` |
| 2 | `CMD_ST` | `dispatch && h_eng == ENG_ST` | `sa_sched.v` |
| 3 | `CMD_EX` | `dispatch && h_eng == ENG_EX` | `sa_sched.v` |
| 4 | `CMD_VE` | `dispatch && h_eng == ENG_VE` | `sa_sched.v` |
| 5 | `PCPI_QFULL` | `state == S_EXEC && q_valid && !q_ready` (CPU stuck on a full queue) | `sa_pcpi.v` |
| 6 | `PCPI_FENCE` | `state == S_EXEC && grp == 1 && op == 4 && !(fence_ok \|\| sched_err)` | `sa_pcpi.v` |
| 7 | `HAZ_LD` | `d2_v && !err && h_valid_cmd && hazard && h_eng == ENG_LD` | `sa_sched.v` |
| 8 | `HAZ_ST` | Same, `h_eng == ENG_ST` | `sa_sched.v` |
| 9 | `HAZ_EX` | Same, `h_eng == ENG_EX` | `sa_sched.v` |
| 10 | `HAZ_VE` | Same, `h_eng == ENG_VE` | `sa_sched.v` |
| 11 | `DISP_FULL` | `d2_v && !err && h_valid_cmd && !hazard && (e_full[h_eng] \|\| m_full[h_eng])` | `sa_sched.v` |
| 12 | `STARVE` | `!d2_v && eng_busy != 0` (no command at the head, but some engine busy) | `sa_sched.v` |
| 13 | `ALL_IDLE` | `pipe_empty && eng_busy == 0` (fully idle, pure software overhead) | `sa_sched.v` |
| 14 | `EX_STEP` | `step` (array enabled) | `sa_ex.v` |
| 15 | `EX_USEFUL` | `step && live_new` (new K data read in) | `sa_ex.v` |
| 16 | `EX_SWAPWAIT` | `streaming && c == t_end && drain_active` (waiting for the previous tile's write-back) | `sa_ex.v` |
| 17 | `EX_TILES` | `swap_now` | `sa_ex.v` |
| 18 | `LD_BUSY` | `active` | `sa_ld.v` |
| 19 | `LD_BEATS` | `lw_en` (each 8-byte data beat) | `sa_ld.v` |
| 20 | `LD_ARSTALL` | `\|(m_arvalid & ~m_arready)` | `sa_ld.v` |
| 21 | `ST_BUSY` | `active` | `sa_st.v` |
| 22 | `ST_BEATS` | `\|(m_wvalid & m_wready)` | `sa_st.v` |
| 23 | `ST_WSTALL` | `\|(m_wvalid & ~m_wready)` | `sa_st.v` |
| 24 | `VE_ACTIVE` | `active` | `sa_ve.v` |
| 25 | `VE_RDBLOCK` | `rd_can && rd_block` (a read lost the port to a write) | `sa_ve.v` |
| 26 | `VE_CREDIT` | `active && !reading && rg < ngroups && inflight == OQ` | `sa_ve.v` |
| 27 | `VE_GROUPS` | `wr_now && wr_last` (a group fully written) | `sa_ve.v` |
| 28–31 | Reserved | For descriptor DMA: `FETCH_DESC`, `FETCH_WAIT`, `FETCH_ARSTALL`, etc. (see 2.4) | `sa_cmdfetch.v` |

Notes:

- `HAZ_*` is classified by the type of the command at the **head of the queue**. Under in-order dispatch, when the head is stuck everything behind it is stuck.
- A high `STARVE` means the firmware issues too slowly, exactly the problem descriptor DMA solves; a high `PCPI_QFULL` means
  the firmware is already ahead of the hardware and issue is not the bottleneck.

### 1.3 Derived metrics

| Metric | Formula | Notes |
|---|---|---|
| Array useful utilization | `EX_USEFUL / CYCLES` | About 70% expected for 256³ |
| Pipeline fill overhead | `(EX_STEP − EX_USEFUL) / CYCLES` | The 2(D−1)-cycle skew per tile |
| Useful MACs | `EX_USEFUL × D²` | Should equal M·N·K (usable as a check) |
| LD bandwidth (while busy) | `LD_BEATS × 8 / LD_BUSY` B/cycle | Compare with bwtest's 7.9 |
| ST bandwidth (while busy) | `ST_BEATS × 8 / ST_BUSY` | |
| Frontend starvation share | `STARVE / CYCLES` | Large means descriptor DMA is needed |
| Conflict-wait share | `Σ HAZ_* / CYCLES` | Look at it split by type |
| VE throughput | `VE_GROUPS / VE_ACTIVE` groups/cycle | |

### 1.4 Hardware design

#### 1.4.1 New module `rtl/sysarray/sa_perf.v`

```verilog
module sa_perf #(
    parameter integer NCNT = 32,        // number of counters (28 defined + 4 reserved)
    parameter integer PERF = 1          // 0: the whole block is optimized away, reads are always 0
) (
    input  wire              clk, resetn,
    input  wire [NCNT-1:0]   ev,        // events, one bit per counter (bit 0 = CYCLES, always 1)
    input  wire              en,        // enable: 1 = count
    input  wire              clear,     // one-cycle pulse: clear all
    input  wire [4:0]        rsel,      // read select
    output reg  [31:0]       rdata      // registered output, 1 cycle of latency
);
```

- **Events are registered first** (`ev_q <= ev`) before entering the counters, so the engines' combinational logic is not stretched into
  new critical paths. Counting one cycle late does not matter for statistics.
- **Each counter**: `if (clear) cnt <= 0; else if (en && ev_q[i]) cnt <= cnt + 1;`
- **Read-out**: a 32-to-1 multiplexer + an output register.
- **With `PERF = 0`**: `generate` removes the counters and the multiplexer entirely, and reads return 0: a switch for when
  resources are tight.

#### 1.4.2 Bringing out the events (modules only gain output ports; logic unchanged)

| Module | New output port | Content |
|---|---|---|
| `sa_sched.v` | `perf_ev[12:0]` | `CMD_*` (4), `HAZ_*` (4), `DISP_FULL`, `STARVE`, `ALL_IDLE`, plus `dispatch` etc. |
| `sa_pcpi.v` | `perf_ev[1:0]` | `PCPI_QFULL`, `PCPI_FENCE` |
| `sa_ex.v` | `perf_ev[3:0]` | `EX_STEP`, `EX_USEFUL`, `EX_SWAPWAIT`, `EX_TILES` |
| `sa_ld.v` | `perf_ev[2:0]` | `LD_BUSY`, `LD_BEATS`, `LD_ARSTALL` |
| `sa_st.v` | `perf_ev[2:0]` | `ST_BUSY`, `ST_BEATS`, `ST_WSTALL` |
| `sa_ve.v` | `perf_ev[3:0]` | `VE_ACTIVE`, `VE_RDBLOCK`, `VE_CREDIT`, `VE_GROUPS` |
| `sa_unit.v` | — | Instantiates `sa_perf` and concatenates the ports above into `ev` by the numbering of 1.2 |

#### 1.4.3 Control and read-out paths

Two paths, reading the same set of counters:

1. **PCPI instruction `mat_perf`** (funct7 = 1, funct3 = 5): for firmware, a few cycles each; see 1.5.1.
2. **CSR mirror** (AXI-Lite, the read-out `case` of `sa_legacy.v`): for debugging and legacy firmware.
   - Address `0x40 + 4·i` (i = 0…31, i.e. 0x40–0xBC);
   - the current CSRs only use up to 0x28 (`R_EXT`); `addr[7:2]` gives 64 slots, enough room;
   - `0x3C` is the control register `PERF_CTRL`: write bit0 = clear, bit1 = enable; reads back the enable state.

**CAPS register**: new bit 20 = `PERF` (has counters), bit 21 = `DESC` (has descriptor DMA, part 2).
Software uses them to detect hardware capabilities; on old bitstreams both bits are 0.

### 1.5 Software interface

#### 1.5.1 Instruction `mat_perf`

| Form | rs1 | rs2 | rd |
|---|---|---|---|
| Read | bit31 = 0, [4:0] = counter number | — | Count value |
| Control | bit31 = 1 | bit0 = CLEAR, bit1 = ENABLE | Number of counters (NCNT; 0 with `PERF = 0`) |

```c
/* firmware/include/sysarray_intrinsics.h (new) */
#define SA_PERF_CLEAR   (1u << 0)
#define SA_PERF_ENABLE  (1u << 1)
enum { SA_PC_CYCLES = 0, SA_PC_CMD_LD, SA_PC_CMD_ST, SA_PC_CMD_EX, SA_PC_CMD_VE,
       SA_PC_PCPI_QFULL, SA_PC_PCPI_FENCE, SA_PC_HAZ_LD, SA_PC_HAZ_ST, SA_PC_HAZ_EX,
       SA_PC_HAZ_VE, SA_PC_DISP_FULL, SA_PC_STARVE, SA_PC_ALL_IDLE, SA_PC_EX_STEP,
       SA_PC_EX_USEFUL, SA_PC_EX_SWAPWAIT, SA_PC_EX_TILES, SA_PC_LD_BUSY, SA_PC_LD_BEATS,
       SA_PC_LD_ARSTALL, SA_PC_ST_BUSY, SA_PC_ST_BEATS, SA_PC_ST_WSTALL, SA_PC_VE_ACTIVE,
       SA_PC_VE_RDBLOCK, SA_PC_VE_CREDIT, SA_PC_VE_GROUPS, SA_PC_NUM = 32 };

static inline uint32_t mat_perf_ctl(uint32_t flags)     /* returns the number of counters */
{
    uint32_t rd;
    __asm__ volatile (".insn r 0x0B, 5, 1, %0, %1, %2" : "=r"(rd) : "r"(1u << 31), "r"(flags));
    return rd;
}
static inline uint32_t mat_perf_read(uint32_t idx)
{
    uint32_t rd;
    __asm__ volatile (".insn r 0x0B, 5, 1, %0, %1, x0" : "=r"(rd) : "r"(idx));
    return rd;
}
```

#### 1.5.2 Firmware

The counting window is aligned with the existing `rdcycle` timing window:

```c
uint32_t n = mat_perf_ctl(SA_PERF_CLEAR | SA_PERF_ENABLE);   /* before t0 */
uint32_t t0 = rdcycle();
... issue commands ...
uint32_t st = mat_fence(SA_ENG_ALL);
uint32_t t1 = rdcycle();
mat_perf_ctl(0);                                             /* freeze */
for (uint32_t i = 0; i < n && i < PERF_AREA_WORDS; i++)      /* write into the counter area (on-chip BRAM) */
    PERF_AREA[i] = mat_perf_read(i);
```

**Counter area: in the program BRAM, below the mailbox**

```
Program BRAM (8 KB, RISC-V 0xC0000000, ARM 0x40010000)
0x0000 ┌──────────────────────────┐
       │ firmware code + data +   │  at most 0x1E00 (7.5 KB), stack top 0x1E00
       │ stack                    │
0x1E00 ├──────────────────────────┤
       │ counter area PERF_AREA   │  256 B = 64 words (new)
0x1F00 ├──────────────────────────┤
       │ mailbox                  │  256 B, layout unchanged
0x2000 └──────────────────────────┘
```

- **Why not in DDR, and not squeezed into the mailbox**: the mailbox has only 29 free words left (0x8C–0xFC), not enough for
  32 counters, and filling it would leave no room for parameters. A separate 256 B counter area below the mailbox:
  - needs no DDR buffer, and the RISC-V does not write DDR through HP0;
  - the existing mailbox field offsets stay **exactly the same**;
  - holds up to 64 counters, including descriptor DMA's 4;
  - when the firmware hangs, the ARM can still read the counter area directly to see where it stopped.
- **Cost**: the space available to firmware shrinks from 0x1F00 to 0x1E00 (7.5 KB). The largest firmware today,
  `vector_fw.bin`, is about 1.4 KB; plenty of margin.
- **All firmware must be relinked** (`LENGTH` in `link.ld` becomes 0x1E00 and the stack top moves down accordingly); no code changes.
  **No new bitstream is needed**: the BRAM size and the mailbox offset are unchanged, so the archived m1–m4 bitstreams work as before with the new firmware.
- **New in `mailbox.h`**: `PERF_AREA_BASE 0xC0001E00`, `PERF_AREA_OFFSET 0x1E00`, `PERF_AREA_WORDS 64`,
  and `#define PERF_AREA ((volatile uint32_t *)PERF_AREA_BASE)`.
- **Overhead**: 32 `mat_perf` reads + 32 on-chip BRAM writes, about 100–200 cycles, all outside the timing window.
- **Scope**: added to both `gemm_fw.c` and `vector_fw.c`; optional for `bwtest_fw.c`.

#### 1.5.3 Driver and scripts

- `pynq_matmul.py`:
  - the size limit of `load_firmware()` changes from 0x1F00 to 0x1E00, and the counter area is cleared as well;
  - `gemm()` / `vector()` get a parameter `perf=False`. When True, after the run the 32 words are read back over MMIO from
    `PERF_AREA_OFFSET` (ARM address 0x40011E00), with no DDR buffer and no `invalidate()` needed;
  - `stats["perf"]` is a `{name: value}` dictionary, plus the derived metrics of 1.3;
  - a new `PERF_NAMES` list (in the same order as the C enum) and a formatting function `perf_breakdown(stats)`.
- New board script `notebooks/m4_perf.py`: for 64³, 128³, 256³, 512×256×256,
  256×128×1024, int8 256³ and several vector operations, prints the cycle breakdown and checks the invariants of 1.6.
  Example output (numbers are only illustrative):
  ```
  256x256x256   82813 cycles
    EX  useful 69.6%  fill 9.8%  swap-wait 1.2%
    head blocked: EX 7.1%  ST 3.0%  LD 0.4%   queue full 0.0%
    frontend starve 2.3%   all idle 0.5%   CPU stalled on full queue 61%
    LD 7.8 B/cycle busy 38%   ST 7.9 B/cycle busy 49%
  ```

### 1.6 Verification

**Unit test** (`rtl/sysarray/sim/tb_sa_unit.v`): after each GEMM the counters are checked against exact **invariants**.

| Invariant | Expected value |
|---|---|
| `EX_TILES` | Total tiles issued: Σ repeat of every EX |
| `EX_USEFUL` | Σ tiles × K |
| `EX_STEP` | Σ tiles × (K + 2(D−1)) |
| `EX_USEFUL × D²` | M·N·K |
| `LD_BEATS × 8` | Sum of the bytes of all LDs |
| `ST_BEATS × 8` | Sum of the bytes of all STs |
| `CMD_LD + CMD_ST + CMD_EX + CMD_VE` | Number of commands issued (rejected ones not counted) |
| `VE_GROUPS` | Σ groups of every VE |
| `CYCLES` | Cycles in the enable window (timed separately in the testbench for comparison) |
| After freezing | Two read-backs give the same values |
| After clearing | All 0 |

**Other checks**:

- **Read-out paths**: after a random command stream, the PCPI read-out and the CSR-mirror read-out agree.
- **`PERF = 0` build**: all reads 0; other tests pass as usual.
- **Negative test**: the existing check "funct7 = 1, funct3 = 5 must not be answered" changes to funct3 = 7.
- **System simulation** (`make sim` of `firmware/gemm`, `firmware/vector`): `tb_system.v` reads back the counters in DDR,
  prints the breakdown, and checks `EX_USEFUL × D² == M·N·K`.
- **Mutation test**: deliberately wire one event wrongly (e.g. `EX_USEFUL` to `EX_STEP`) and confirm the invariant check fails.
- **`make test` / `make test16`**: all pass.

### 1.7 Resources and timing

| Item | Estimate |
|---|---|
| Counters 32 × 32 bits | ≈ 1,024 FF |
| Incrementers | ≈ 32 × 32 LUT (carry chains) ≈ 1,000 LUT |
| Event registers | ≈ 32 FF |
| Read-out 32-to-1 × 32 bits | ≈ 250 LUT |
| **Total** | **≈ 1.2–1.3k LUT, ≈ 1.1k FF, 0 BRAM, 0 DSP** |

- M4 today: 47.6k / 53.2k LUT (89.5%), about 5.6k left; it fits, ending at about 92%.
- Timing: each counter is one level of 32-bit adder, the events are registered; plenty of slack at 50 MHz.
- If routing gets hard, `DSP_COLS` can go from 8 to 10: frees about 5k LUT, DSP from 184 to 216 / 220.

### 1.8 Implementation steps

| Step | Task | Acceptance criterion | Board needed |
|---|---|---|---|
| **P1** RTL | ① `sa_perf.v`; ② event ports from each module; ③ instantiation and concatenation in `sa_unit.v`; ④ `sa_pcpi.v` claims funct3 = 5 and implements `mat_perf`; ⑤ CSR mirror, `PERF_CTRL`, CAPS bit 20 in `sa_legacy.v`; ⑥ counter numbers defined in `sa_defs.vh` | `make test`, `make test16` all pass; the invariants of 1.6 hold at D = 8 and D = 16; mutation tests are caught | No |
| **P2** software | ① intrinsics; ② `PERF_AREA_*` in `mailbox.h`, `LENGTH` 0x1E00 in `link.ld`, all firmware relinked; ③ `gemm_fw.c`, `vector_fw.c` write the counters into the counter area; ④ `tb_system.v` reads back from `bram[0x1E00/4 + i]` and checks; ⑤ driver `perf=` parameter and formatting; ⑥ `notebooks/m4_perf.py` (dry run with a fake overlay first) | System simulation passes at `SIM_D = 8`, `16` and prints the breakdown; dry run passes | No |
| **P3** board | ① OOC synthesis for resources and timing; ② `build_bitstream.sh -jobs 4 -sa_d 16` (run by you in the terminal); ③ on the board run `m4_perf.py`, `m4_demo.py` (regression) | Timing met; regression all PASS; invariants hold on the board; measured cycle breakdowns for each shape | Yes |
| **P4** documentation | Replace the estimate in design document §10.3 with the measured breakdown; new §10.4 "Performance counters"; archive the bitstream in `bitstreams/m5/` (or `m4p/`) with a README; decide the scope of descriptor DMA (part 2) from the `STARVE` / `PCPI_QFULL` data | Documents updated; **whether descriptor DMA is implemented and which cases to optimize first is backed by data** | — |

Optional follow-up (P5): an **event trace buffer**. On every dispatch and completion, write (timestamp, engine, command number,
user tag) into a BRAM FIFO (64 bits × 512 deep, one BRAM36), exported by the driver as Perfetto /
Chrome trace JSON, giving a timeline (Gantt chart) of the engines in which "EX waiting for LD of the right half of B" is directly visible.
Works best together with the descriptors' user tag field (2.2).

---

## 2. Descriptor DMA

### 2.1 Motivation

Today every hardware command is one PCPI instruction executed by PicoRV32, plus a few `mat_cfg`s. For example
each GEMM strip: `mat_cfg ×4 → mat_exec → mat_cfg ×4 → mat_exec → mat_cfg ×4 →
mat_load → mat_store …`, a dozen or so PCPI instructions, plus ordinary instructions to compute the parameters.

| Problem | How descriptor DMA solves it |
|---|---|
| Issue overhead: small matrices and vector operations are limited by issue speed (to be confirmed by P3's `STARVE` data) | The hardware fetches by itself; a command costs one 64-byte burst read |
| The CPU is occupied for the whole job | After one `mat_submit` the CPU is free |
| The same network re-executes the issuing code on every inference | A list is built once and reused; switching buffers only changes base addresses (relocation) |
| The Phase 5 compiler has to generate code | It generates **data** (a descriptor table) instead: simpler and easier to check |

**Expected gains (subject to measurement)**:
- Large GEMMs basically unchanged: the firmware is ahead of the hardware today, and `PCPI_QFULL` should be high;
- small matrices such as 64³ and int8 vector operations should improve clearly;
- the biggest change is that the CPU is free during jobs, and the compiler gets a simpler target.

**Look at the P3 data first**: if `STARVE` is close to 0 on every shape of interest, part 2 should be deprioritized.

### 2.2 Descriptor format

Each descriptor is **64 bytes**, eight little-endian 64-bit words `w0…w7`, and must be 64-byte aligned. Why 64 bytes:

- exactly one 8-beat × 8-byte AXI burst;
- 64-byte aligned, it never crosses a 4 KB boundary;
- the widest VE command (the fields of the 261-bit command packet) fits;
- each carries all its configuration, independent of `mat_cfg` state, so descriptors can be reordered, reused and debugged individually.

**Common header `w0`** (this section is the M5 format; L1 and L2 extensions are in "Later evolution" at the top and in `llm_inference_plan.md` §5.3, §6.6, Appendix A)

| Bits | Field | Notes |
|---|---|---|
| [7:0] | `opcode` | 0x01 LD, 0x02 ST, 0x03 EX, 0x04 VE, 0x10 FENCE, 0x11 JUMP, 0x12 END; **0x00 illegal** (all-zero memory is treated as an error, preventing runaway when END is forgotten) |
| [8] | `RELOC` | DDR address field = offset + `BASE[BASESEL]` |
| [10:9] | `BASESEL` | Selects BASE0–BASE3 |
| [11] | `FENCE_BEFORE` | Wait for all engines to be idle before executing this one (as if a FENCE were inserted before it) |
| [12] | `IRQ` | END only: raise an interrupt on completion |
| [31:13] | Reserved | Must be 0, otherwise an error |
| [63:32] | `tag` | User tag: debugging, error location, trace buffer (P5) |

**Fields per opcode** (unlisted bits must be 0)

| opcode | `w1` | `w2` | `w3` | `w4` | `w5` |
|---|---|---|---|---|---|
| LD / ST | [31:0] DDR address (or RELOC offset) | [31:0] local address LADDR, [47:32] rows, [63:48] row_bytes | [31:0] pitch, [33:32] mode (LD only; must be 0 for ST) | — | — |
| EX | [15:0] A, [31:16] B, [47:32] C, [59:48] Kt, [60] accumulate | [11:0] repeat (0 means 1), [31:16] bstep, [47:32] cstep, [63:48] crow (0 means 1) | — | — | — |
| VE | [31:0] src1 LADDR, [63:32] src2 LADDR | [31:0] dst LADDR, [63:32] LEN (elements, a multiple of D) | [7:0] op byte (op / RELU / REQUANT, as in `vec_cfg`), [13:8] types, [31:16] period, [47:32] scale, [52:48] shift | [31:0] zp, [63:32] clamp lo | [31:0] clamp hi |
| FENCE | [3:0] engine mask (0 = all) | — | — | — | — |
| JUMP | [31:0] address of the next list segment (64-byte aligned) | — | — | — | — |
| END | [31:0] completion value (written into `DESC_STATUS`) | — | — | — | — |

VE's LEN counts elements rather than groups, consistent with the LEN semantics of `vec_cfg`; the decoder shifts right by log2(D) to get the group count.

### 2.3 Execution semantics

- **Start**: `mat_submit(list, count)` fetches from `list` and executes until END, or until `count` descriptors have executed
  (`count = 0` means no limit). Only one list runs at a time; a new `mat_submit` waits while the previous list has not finished.
- **Order**: within a list commands enter the scheduler in order, exactly equivalent to issuing them one by one over PCPI. On-chip dependencies between commands
  are still handled by the scoreboard.
- **Mixing with PCPI**: `mat_submit` itself counts as one command. While a list runs, newly arriving queued PCPI commands
  (`mat_load/store/exec`, `vec_run`) are held off by `pcpi_wait` until the list finishes. `mat_cfg`,
  `mat_perf` and `mat_fence` are unaffected. So mixed programs execute in the order they are written.
- **`mat_fence`**: its semantics extend to also waiting for the fetch unit to finish (the list to complete).
- **DDR dependencies**: still not tracked by the hardware. When the same DDR region is ST'd then LD'd, the list must contain a FENCE descriptor,
  or the LD must set `FENCE_BEFORE`: the same rule as `mat_fence` on the PCPI path.
- **FENCE**: the fetch unit stops sending commands until all engines in the mask are idle (reusing the scheduler's `fence_ok`).
- **JUMP**: rewrites the fetch address; can chain several list segments, e.g. a fixed prologue + per-layer lists.
- **END**: writes `DESC_STATUS`, increments the `DESC_DONE` count, raises the interrupt if IRQ is set, and the fetch unit returns to idle.
- **Errors**: the following set the sticky error with **engine number = 4 (FETCH)**, record the index of the failing descriptor in `DESC_ERR_IDX`,
  and stop the fetch unit; recovery is the same as today, with `mat_reset`:
  - illegal opcode, non-zero reserved bits, misaligned address;
  - an AXI read error during fetch (error code 3);
  - a decoded command judged illegal by the scheduler: the existing `reject` path, error code 1 or 2, with the command's own engine number.

### 2.4 Hardware design

```
                PCPI: mat_submit(list, count)
                       │
┌──────────────────────▼──────────────────────────────┐
│ sa_cmdfetch.v (new module)                           │
│  ① fetch FSM: IDLE → FETCH → RUN → (FENCE_WAIT) → IDLE│
│  ② AXI read: 8 beats each, at most 2 bursts in flight│──┐
│  ③ descriptor FIFO: 1 × BRAM36 (64 bits × 512 = 64)  │  │  AR/R
│  ④ decode: descriptor → 261-bit packet (shared fns)  │  │
└──────────────────────┬──────────────────────────────┘  │
                       │ priority: legacy > cmdfetch > PCPI  │
                       ▼                                    ▼
                 sa_sched (unchanged)              m0 read-channel arbiter (new)
                                                   ├── sa_ld (existing)
                                                   └── sa_cmdfetch
```

#### 2.4.1 Fetch unit `sa_cmdfetch.v`

- **States**:
  - `IDLE`: waiting for `mat_submit`;
  - `FETCH/RUN`: fetching and sending commands in parallel;
  - `FENCE_WAIT`: waiting for `fence_ok`;
  - `ERR`: stopped, waiting for `clear_error`.
- **Fetch**: address increments by 64, at most 2 bursts in flight. Fetching pauses when the FIFO has room for fewer than 16 64-bit words;
  on a JUMP the already prefetched following words are dropped and fetching restarts at the new address.
- **Sending commands**: decoding starts once a complete descriptor (8 words) is in the FIFO. Control descriptors (FENCE / JUMP /
  END) are handled inside the fetch unit; command descriptors go to the scheduler's input port.
- **Relocation**: four 32-bit registers `BASE0–3`, set by new `mat_cfg` keys (see 2.5).
  With `RELOC = 1`, DDR address = `BASE[BASESEL] + w1[31:0]`.
- **Performance counter events** (on the numbers 28–31 reserved in 1.2):
  - `FETCH_DESC`: +1 per decoded descriptor;
  - `FETCH_WAIT`: a complete descriptor is ready but the scheduler input does not accept it;
  - `FETCH_EMPTY`: in the RUN state but the FIFO is empty (fetch cannot keep up);
  - `FETCH_ARSTALL`: the fetch's AR request is back-pressured.

#### 2.4.2 Decoding shared with PCPI

The "fields → 261-bit command packet" assembly in `sa_pcpi.v` is extracted into shared functions in `sa_defs.vh`:

```verilog
function [`SA_PKT_W-1:0] pkt_ld (input [31:0] ddr, laddr, input [15:0] rows, rb,
                                 input [31:0] pitch, input [1:0] mode);
function [`SA_PKT_W-1:0] pkt_st (...);
function [`SA_PKT_W-1:0] pkt_ex (input [15:0] a, b, c, input [11:0] kt, input acc,
                                 input [11:0] rep_m1, input [15:0] bstep, cstep, crow);
function [`SA_PKT_W-1:0] pkt_ve (...);
```

Both the PCPI path and the descriptor path call these functions, so the command format is maintained in one place. This step is a separate refactoring, D1,
first making sure all existing tests pass.

#### 2.4.3 Read-port arbitration (in `sa_unit.v`)

- **AR channel**: LD or the fetch unit, round robin; fetch requests are rare and hardly affect LD.
- **R channel**: AXI IDs are not used, so read data on one port returns in request order. An **ownership FIFO**
  (16 deep, covering up to 8 LD plus up to 2 fetch bursts in flight) records who owns each issued burst;
  R data beats are routed by the FIFO head, popped on `RLAST`.
- **No extra HP port**: descriptor traffic is small, 64 bytes per command, while an LD usually moves several KB.
  This also avoids changing the block design.
- **Alternative**: connect the idle `m1` port to HP1/HP3. No arbitration needed, but it changes `pico_bit.tcl` and takes
  another HP port; not recommended.

#### 2.4.4 Interaction with PCPI (`sa_pcpi.v`)

- funct3 = 6 `mat_submit`: latches (list, count) and answers immediately when the fetch unit is idle; `pcpi_wait` when busy.
- With `fetch_busy = 1`, queued instructions (funct3 = 1/2/3 and `vec_run`) do not raise `q_valid` and
  hold `pcpi_wait` until the list finishes.
- `mat_fence`'s completion condition becomes `(fence_ok && !fetch_busy) || sched_err`.

### 2.5 Instructions, CSRs and status

**New instruction**

| Instruction | Encoding | Semantics |
|---|---|---|
| `mat_submit` | funct7 = 1, funct3 = 6; rs1 = list address (64-byte aligned), rs2 = maximum count (0 = until END) | Starts the list and returns immediately (waits if the previous list has not finished) |

**New `mat_cfg` keys** (after the existing keys 0–10)

| key | Name | Value |
|---|---|---|
| 11–14 | `BASE0`–`BASE3` | Relocation bases (32-bit physical addresses) |

**CSRs (AXI-Lite mirror, read-only, for debugging)**

| Address | Name | Content |
|---|---|---|
| 0xC0 | `DESC_ADDR` | Current fetch address |
| 0xC4 | `DESC_DONE` | Number of lists completed (END count) |
| 0xC8 | `DESC_STATUS` | Completion value of the latest END |
| 0xCC | `DESC_ERR_IDX` | Index of the failing descriptor in the current list |
| 0xD0 | `DESC_EXEC` | Descriptors decoded in the current list |

**`ext_status`**:
- bit 24 = `FETCH_BUSY` (a list is executing); [31:21] are free today;
- error engine number 4 = FETCH (the field [15:12] is 4 bits; it fits).

**CAPS**: bit 21 = `DESC`.

**Interrupt**: the existing `irq` output of `sa_unit` (used by legacy) is ORed with END's IRQ. If it is already connected to the PS interrupt controller
in the block design, the ARM can wait for an interrupt instead of polling; whether it is connected is confirmed in D4.

### 2.6 Software

#### 2.6.1 Firmware: `firmware/include/sa_desc.h` (new)

```c
typedef struct { uint64_t w[8]; } __attribute__((aligned(64))) sa_desc_t;

/* List building: each function writes one descriptor and returns a pointer to the next */
sa_desc_t *desc_load (sa_desc_t *d, uint32_t ddr, uint32_t laddr, uint32_t rows,
                      uint32_t row_bytes, uint32_t pitch, uint32_t mode, uint32_t flags);
sa_desc_t *desc_store(sa_desc_t *d, uint32_t ddr, uint32_t laddr, uint32_t rows,
                      uint32_t row_bytes, uint32_t pitch, uint32_t flags);
sa_desc_t *desc_exec (sa_desc_t *d, uint32_t a, uint32_t b, uint32_t c, uint32_t kt, uint32_t acc,
                      uint32_t repeat, uint32_t bstep, uint32_t cstep, uint32_t crow);
sa_desc_t *desc_vec  (sa_desc_t *d, uint32_t src1, uint32_t src2, uint32_t dst, uint32_t len,
                      uint32_t op, uint32_t types, uint32_t period, int32_t scale, uint32_t shift,
                      int32_t zp, int32_t lo, int32_t hi);
sa_desc_t *desc_fence(sa_desc_t *d, uint32_t mask);
sa_desc_t *desc_jump (sa_desc_t *d, const sa_desc_t *next);
sa_desc_t *desc_end  (sa_desc_t *d, uint32_t status, uint32_t irq);

static inline void mat_submit(const sa_desc_t *list, uint32_t count)
{
    __asm__ volatile (".insn r 0x0B, 6, 1, x0, %0, %1" :: "r"(list), "r"(count) : "memory");
}
```

- **Lists live in DDR**. PicoRV32 writes DDR through HP0 and the accelerator reads DDR through HP2; neither path goes through the ARM caches,
  so no cache coherence needs maintaining; but the writes must be complete after writing the list and before `mat_submit` (PicoRV32's
  AXI writes are blocking, which satisfies this).
- **List mode of `gemm_fw.c`**: `GEMM_FLAGS` bit 3 = `USE_DESC`. The list-building logic corresponds one to one with today's PCPI issue
  order (A prefetch and B split kept), for easy comparison. List building is timed separately (`MBOX_BUILD_CYCLES`),
  apart from execution time.
- **List mode of `vector_fw.c`**: the same.

#### 2.6.2 Driver: a Python list builder

- `pynq_matmul.py` gains `DescList`: descriptors appended one by one into a NumPy structured array (`dtype = [("w", "<u8", 8)]`),
  `allocate()`d into DDR, then `flush()`ed.
- The ARM builds the whole list directly, and the firmware only executes `mat_submit` (new firmware `firmware/desc_run/`: reads the
  list address and BASE0–3 from the mailbox, submits, fences, writes back the status). This completely separates "building" from "executing",
  matching the industry's command-buffer model.
- Relocation example: once a GEMM's list is built, switching input A only means writing the new address into BASE0 and resubmitting.

#### 2.6.3 Phase 5 compiler

The last step of the MLIR lowering can produce a descriptor array (data) instead of a `.insn` sequence (code):

- easier to check: can be printed and compared directly;
- can be built offline, with only the BASEs changed at run time;
- consistent with the "command buffer + runtime submission" model of frameworks such as IREE.

### 2.7 Verification

| Category | Test | Expected |
|---|---|---|
| Equivalence | `tb_sa_unit`'s GEMM, int8 GEMM and random command streams rewritten as descriptor lists | Final memories match **the same in-order reference model**, word for word identical to the PCPI path |
| Mixed order | PCPI commands alternating with lists | Execution order equals the written order |
| DDR dependencies | The same DDR region ST'd then LD'd: with FENCE / `FENCE_BEFORE` | Correct |
| Negative test | The same **without** FENCE | Stale reads observable in the testbench's DDR model (proves the test is effective) |
| Control descriptors | JUMP chaining three list segments; END status and count; IRQ; truncation by `count` | As in 2.3 |
| Relocation | Same list, different BASEs | Each result correct |
| Errors | Illegal opcode (including all zeros), non-zero reserved bits, misaligned list address, SLVERR on fetch, descriptor address out of range | Error code, engine number 4 or the command's engine, `DESC_ERR_IDX` correct; recovers after `mat_reset` |
| Port arbitration | Heavy LD reads concurrent with fetch under an AXI model with random latencies | R beats routed correctly; `tb_sa_dma`-style protocol checks (4 KB, length, RLAST) all pass |
| D = 8 / 16 | All of the above | Both sizes pass |
| System simulation | `gemm_fw` list mode, `desc_run` firmware | Correct results; build and execution cycles printed |

### 2.8 Resources

| Item | Estimate |
|---|---|
| Fetch FSM + address / count registers | ≈ 300 LUT, ≈ 200 FF |
| Decode (shared functions, mostly multiplexers) | ≈ 400–600 LUT |
| Read-port arbitration + ownership FIFO | ≈ 150 LUT, ≈ 50 FF |
| BASE0–3, status CSRs | ≈ 100 LUT, ≈ 250 FF |
| Descriptor FIFO | **1 × BRAM36** (M4 uses 130 / 140) |
| **Total** | **≈ 1.0–1.5k LUT, ≈ 0.5k FF, 1 BRAM36** |

Together with the performance counters about 2.5k LUT, bringing M4's LUT usage to about 94%. If routing gets hard, set `DSP_COLS`
to 10 (frees about 5k LUT; DSP 216 / 220).

### 2.9 Implementation steps

| Step | Task | Acceptance criterion | Board needed |
|---|---|---|---|
| **D0** decision | Look at the `STARVE`, `PCPI_QFULL`, `ALL_IDLE` data from P3 / P4; decide which cases to optimize and the expected gain | Conclusion written into the design document | — |
| **D1** specification + refactoring | ① 2.2 / 2.3 written into design document §8.6; ② shared `pkt_*` functions extracted into `sa_defs.vh`, `sa_pcpi.v` switched to them | Document review; `make test`, `make test16` all pass (behavior unchanged) | No |
| **D2** RTL | ① `sa_cmdfetch.v`; ② read-port arbitration + ownership FIFO; ③ `mat_submit` in `sa_pcpi.v`, holding off queued PCPI commands, extended fence; ④ BASE keys, CSRs, `ext_status`, CAPS bit 21; ⑤ performance counter events 28–31; ⑥ the unit tests of 2.7 | The unit-test part of 2.7 passes at D = 8 / 16; existing tests do not regress | No |
| **D3** software | ① `sa_desc.h`; ② list modes of `gemm_fw` / `vector_fw`; ③ `desc_run` firmware; ④ Python `DescList`; ⑤ new mode in `tb_system.v`; ⑥ board script `notebooks/m5_desc_demo.py` (dry run) | System simulation passes; dry run passes | No |
| **D4** board | ① OOC resources and timing; ② bitstream build (run by you in the terminal); ③ on the board: PCPI mode vs list mode (64³, 128³, 256³, vector operations); CPU idle share; all regressions | Timing met; regression all PASS; comparison data obtained; performance counters confirm `STARVE` drops | Yes |
| **D5** documentation | Results in design document §10.5; bitstream archived; README / learning path updated | — | — |
| **D6** (optional) | **Direct submission from the ARM**: map a set of "doorbell" registers (list address, count, start) to the ARM over AXI GP, bypassing PicoRV32 entirely; needs block design changes and arbitration between ARM and RISC-V submitting at the same time | An architectural decision, evaluated separately | Yes |

---

## 3. Overall plan

```
P1 ─► P2 ─► P3 (board) ─► P4 (docs + decision) ─► D0 ─► D1 ─► D2 ─► D3 ─► D4 (board) ─► D5
                                                  └────────── optional: P5 trace buffer, D6 direct ARM submission
```

| Stage | Main output | New resources (estimate) | New bitstream needed |
|---|---|---|---|
| P1–P4 | Performance counters, measured cycle breakdown | ≈ 1.3k LUT, 1.1k FF | Yes (once) |
| D1 | Packing-function refactoring | 0 | No |
| D2–D5 | Descriptor DMA | ≈ 1.0–1.5k LUT, 0.5k FF, 1 BRAM36 | Yes (once) |
| P5 (optional) | Event tracing | ≈ 0.5k LUT, 1 BRAM36 | Yes |

Both board builds are run by you in your own terminal with `./scripts/build_bitstream.sh -jobs 4 -sa_d 16`;
Claude does the OOC checks before the build and the timing and resource review after it.

---

## 4. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LUT at 92–94%, hard routing or worse timing | Build fails / negative WNS | OOC first; registered events; `PERF = 0` switch; `DSP_COLS = 10` if needed |
| Counter events wired wrongly, data misleading conclusions | Wrong optimization decisions | The exact invariants of 1.6 + mutation tests |
| Wrong order when mixing PCPI and lists | Wrong results that are hard to reproduce | The rule of 2.3 "hold off queued PCPI commands during a list"; mixed-order tests |
| Read-port arbitration breaks AXI ordering or the 4 KB rule | Corrupted data | Ownership FIFO routing in issue order; reuse `tb_sa_dma`'s protocol checks |
| A forgotten FENCE reads stale DDR data | Intermittent errors | Software rule written in the comments of `sa_desc.h` and in the design document; negative tests; hardware DDR dependency tracking could be considered later |
| An all-zero or corrupted list runs away | Garbage commands executed | opcode 0x00 illegal; reserved bits must be 0; `count` limit |
| Descriptor DMA gains less than expected | Investment not worth it | D0 decides from P3 data first; data decides whether to do it |

---

## Appendix A: the extended funct7 = 1 encoding

| funct3 | Instruction | Status |
|---|---|---|
| 0 | `mat_cfg` (keys 0–10 existing; 11–14 = BASE0–3, new) | Existing + extended |
| 1 | `mat_load` | Existing |
| 2 | `mat_store` | Existing |
| 3 | `mat_exec` | Existing |
| 4 | `mat_fence` (completion condition adds "fetch unit idle") | Existing + extended |
| 5 | `mat_perf` | New (part 1) |
| 6 | `mat_submit` | New (part 2) |
| 7 | Reserved (the negative test moves to this encoding) | — |

## Appendix B: list of modified files

| File | Performance counters | Descriptor DMA |
|---|---|---|
| `rtl/sysarray/sa_perf.v` | New | Events 28–31 connected |
| `rtl/sysarray/sa_cmdfetch.v` | — | New |
| `rtl/sysarray/sa_unit.v` | Instantiation, event concatenation | Instantiation, read-port arbitration, input-port priority |
| `rtl/sysarray/sa_pcpi.v` | funct3 = 5 | funct3 = 6, holding off queued commands, fence extension, switch to `pkt_*` |
| `rtl/sysarray/sa_sched.v` / `sa_ex.v` / `sa_ld.v` / `sa_st.v` / `sa_ve.v` | Event ports | `sa_sched`: `ext_status` bit 24 |
| `rtl/sysarray/sa_legacy.v` | CSR mirror, `PERF_CTRL`, CAPS bit 20 | Descriptor CSRs, CAPS bit 21, IRQ OR |
| `rtl/sysarray/sa_defs.vh` | Counter numbers | `pkt_*` functions, descriptor opcodes / bit definitions |
| `rtl/sysarray/sim/tb_sa_unit.v` | Invariant checks, negative test moved to funct3 = 7 | All tests of 2.7 |
| `firmware/include/sysarray_intrinsics.h` | `mat_perf_*` | `mat_submit`, BASE keys |
| `firmware/include/sa_desc.h` | — | New |
| `firmware/include/mailbox.h` | `PERF_AREA_BASE` / `OFFSET` / `WORDS` | List address, `MBOX_BUILD_CYCLES`, etc. |
| `firmware/common/link.ld` | `LENGTH` 0x1F00 → 0x1E00 (making room for the counter area; stack top moves down) | — |
| `firmware/gemm`, `firmware/vector` | Write the counters into the counter area | List mode |
| Other firmware (`matmul`, `matmul_insn`, `bwtest`) | Relink only | — |
| `firmware/desc_run/` | — | New |
| `firmware/sim/tb_system.v` | Read back the counters | List-mode tests |
| `driver/pynq_matmul.py` | `perf=`, `PERF_NAMES`, formatting | `DescList`, submission interface |
| `notebooks/m4_perf.py` | New | — |
| `notebooks/m5_desc_demo.py` | — | New |
| `docs/double_buffer_design.md` | §10.3 measurements, §10.4 | §8.6 format, §10.5 results |

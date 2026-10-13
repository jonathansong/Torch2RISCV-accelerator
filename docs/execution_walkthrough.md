# Execution walkthrough: from `m4_demo.py` to the RTL

> **PYNQ-Z1 paths**: this document records work done on the PYNQ-Z1. The `RISCV-on-PYNQ-Z1/...` (block design, constraints), `bitstreams/...` (builds verified on the board) and `iree-sa/l0` it mentions are on the `pynq-z1` branch (tag `v1.0-pynq-z1`); on main, PicoRV32 and its IP are in `rtl/picorv32/` and `rtl/ip/`, and the current overlay is in `boards/kv260/`.

This document takes one **int32 GEMM `mm.gemm(a, b)`** from `notebooks/m4_demo.py`
(256×256×256, the M4 board build, D = 16) as an example and follows one run layer by layer: the Python
driver → PicoRV32 firmware → PCPI custom instructions → scheduler → engines → BRAM / DDR,
and back to Python. int8 GEMM and vector operations take the same path, only with extra VE commands.
Section 10 is M5's descriptor-list path. Section 11 is one token of LLM inference (L4–L5, the D = 8 `l2` build):
the command ring, resident firmware, static lists with PARAM / LOOP, and interrupts.

Companion reading: [`double_buffer_design.md`](double_buffer_design.md) (the design;
§4 EX, §5 DMA, §7 scoreboard, §8 instructions), [`memory_model.md`](memory_model.md)
(address map), [`hardware_learning_path.md`](hardware_learning_path.md)
(learning path; Stage 5 reads module by module).

---

## Contents

0. [Overview](#0-overview)
1. [Python layer](#1-python-layer-m4_demopy--driverpynq_matmulpy)
2. [Firmware layer](#2-firmware-layer-firmwaregemmgemm_fwc-on-picorv32)
3. [PicoRV32 → PCPI → `sa_pcpi`](#3-picorv32--pcpi--rtlsysarraysa_pcpiv)
4. [Scheduler `sa_sched`](#4-scheduler-rtlsysarraysa_schedv)
5. [Engines and memories](#5-engines-and-memories)
6. [Completion and return](#6-completion-and-return)
7. [Data contracts between layers](#7-data-contracts-between-layers)
8. [Debug checklist by layer](#8-debug-checklist-by-layer)
9. [Watching the process yourself](#9-watching-the-process-yourself)
10. [Another path: descriptor lists](#10-another-path-descriptor-lists)
11. [A third path: one LLM token](#11-a-third-path-one-llm-token-l4l5)

---

## 0. Overview

```
m4_demo.py ──► driver/pynq_matmul.py ──► AXI GP0 ──► program BRAM (firmware + mailbox)
                                     └─► EMIO GPIO[0] releases the RISC-V reset
PicoRV32 fetches from 0xC0000000 ──► gemm_fw.c ──► .insn custom-0 instructions
   ──► PCPI ──► sa_pcpi (decode, pack commands) ──► sa_sched (FIFO → decode → scoreboard → dispatch)
   ──► sa_ld / sa_ex / sa_st / sa_ve ──► SPAD/ACC BRAM; DMA reads and writes DDR through HP2
   ──► engine done → scoreboard release → mat_fence returns ──► firmware writes mailbox DONE
   ──► Python polls DONE → invalidate → compare with NumPy
```

Clock: on the PL side PicoRV32 and the accelerator share 50 MHz (`subprocessorClk`).
There are only two control channels between the ARM and the PL: AXI GP0 (writes the program BRAM / mailbox) and
EMIO GPIO[0] (RISC-V reset). All data is exchanged through DDR.

---

## 1. Python layer: `m4_demo.py` → `driver/pynq_matmul.py`

### 1.1 Loading the overlay: `MatmulOverlay(bit, gemm_fw.bin)`

| Step | What it does | Hardware path |
|---|---|---|
| `Overlay(bitfile)` | Programs the `.bit` into the PL and parses the `.hwh` for the address map | PYNQ → Linux FPGA manager |
| `overlay_params()` | Reads D = 16, NPORTS = 1 from the `PARAMETER`s of `MODTYPE="sa_unit"` in the `.hwh` → `mm.d` | Pure file parsing |
| `GPIO(EMIO[0]).write(1)` | **Holds the RISC-V in reset** | PS GPIO → EMIO → `riscvReset` |
| `MMIO(0x40010000, 0x2000)` | Maps the ARM window of the 8 KB program BRAM | `/dev/mem` |
| `load_firmware()` | Clears the BRAM, then writes `gemm_fw.bin` word by word (32 bits) | **AXI GP0 → `psBramController` → `riscvBram` port A** |

`riscvBram` is a dual-port BRAM: port A is for the ARM, port B is the program memory PicoRV32 sees at
0xC0000000. The last 512 bytes are reserved: 0x1E00–0x1EFF is the performance
counter area, 0x1F00–0x1FFF the mailbox. The firmware must not exceed 0x1E00 bytes (`load_firmware` checks).

### 1.2 `mm.gemm(a, b)`

1. **Checks**: M, N, K are all multiples of D = 16; K×N ≤ 128 KB (B resident in SPAD_B);
   K ≤ 65536 / D (an A strip fits in one SPAD bank).
2. **Allocating DDR buffers**: `allocate()` allocates **physically contiguous** A, B, C from CMA
   and copies the data in.
3. **`flush()`**: writes dirty lines in the ARM caches back to DDR. The accelerator accesses
   DDR directly through the HP ports, bypassing the ARM caches; without a flush it reads stale data.
4. **Writing parameters**: `A_BASE`, `B_BASE`, `C_BASE` (physical addresses), `GEMM_M/N/K`,
   `BIAS_BASE`, `GEMM_FLAGS = 0` into the mailbox.
5. **`_run()`**:
   1. reset set to 1;
   2. clear all 64 words of the mailbox (from offset 0x1F00);
   3. write the parameters;
   4. **reset set to 0**; the RISC-V starts fetching from 0xC0000000;
   5. poll `MBOX_STATUS` until `0x600D600D` (with a timeout);
   6. reset set to 1 (stops the RISC-V), read `EXT_STATUS`; bit 1 = 1 means the accelerator reported an error
      (error code in [11:8], the failing engine in [15:12]).
6. **Fetching the result**: `cbuf.invalidate()` discards stale lines in the ARM caches, C is copied out, the buffers are freed.
7. **`m4_demo.py`**: `np.array_equal(c, golden(a, b))`, prints PASS, the cycle count and
   MAC/cycle.

---

## 2. Firmware layer: `firmware/gemm/gemm_fw.c` on PicoRV32

### 2.1 Startup: `firmware/common/start.S` + `link.ld`

- Reset vector `PROGADDR_RESET = 0xC0000000` (`pico_processor.tcl`).
- Sets the stack top `__stack_top = 0xC0001E00`, right below the performance counter area.
- Clears `.bss`, then `call main`.
- After `main` returns it executes `ebreak`: PicoRV32 traps and halts until the ARM resets it again.

### 2.2 Preparation in `main()`

1. Writes `STATUS_RUNNING` to the mailbox.
2. `sa_init()`: reads the **CAPS register (0x80000024)**. This is a PicoRV32
   AXI-Lite read, through the RISC-V's peripheral interconnect to the CSRs of `sa_legacy`, reading [7:0] = D = 16.
   So the same firmware runs on both the D = 8 and the D = 16 overlays.
3. Reads the mailbox parameters and computes all D-dependent divisions **before timing starts**:
   `nt = N/16 = 16` (C tiles per strip), `kt = K/16 = 16`, the bank sizes.
   PicoRV32's divide instruction takes dozens of cycles; inside the loop it would slow down command issue.
4. `mat_reset()` clears any leftover sticky error.
5. `sa_perf_begin()`: when CAPS bit 20 is 1 (hardware has performance counters), `mat_perf` clears and starts the counters;
   on older hardware it does nothing (otherwise `mat_perf` would raise an illegal-instruction exception). Then `t0 = rdcycle`.

### 2.3 Issuing commands

The firmware **only queues the commands**; it does not wait for them to execute:

```
B = 64 KB ≥ 16 KB, so B is split (B split):
  mat_cfg_load(K, 8*16, N, INTERLEAVE); mat_load(left half of B → SPAD_B bank0)
  load_strip(A0 → SPAD_A bank0)
  mat_cfg_load(K, 8*16, N, INTERLEAVE); mat_load(right half of B → SPAD_B bank1)
  mat_cfg_store(16, 4N, 4N)
for each strip i (16 rows of A; the SPAD_A / ACC banks alternate):
  mat_cfg_exec(8, K, 1, N/16); mat_exec(A_i, left B, C_i,     kt)    ← first 8 C tiles
  mat_cfg_exec(8, K, 1, N/16); mat_exec(A_i, right B, C_i + 8, kt)   ← last 8 C tiles
  load_strip(A_{i+1} → the other bank)   ← A prefetch: queued before ST(i)
  mat_store(the 16×N strip of C_i → DDR)
mat_fence(all engines)                   ← the only real wait
```

Why this order (the M4 scheduling optimization, design document §10.3):

- **A prefetch**: the scheduler dispatches in program order. If `LD A(i+1)` is queued after `ST(i)`,
  it has to wait for `ST(i)`, which waits for `EX(i)` to finish, so the next strip's load cannot
  overlap with the current computation.
- **B split**: the scoreboard tracks whole banks. With all of B in one bank, the first
  EX waits for the whole of B to load; split into two halves in two banks, the first EX only waits for the left half.
  Split only when B ≥ 16 KB (on small matrices the overhead of two short EX commands is larger).

### 2.4 Wrap-up

`st = mat_fence(0)` (wait for all commands) → `t1 = rdcycle` →
`sa_perf_end()`: freezes the counters, copies the 32 counter values into the counter area of the program BRAM (0x1E00), and writes their count into
`MBOX_PERF_COUNT` → writes `TOTAL_CYCLES`, `JOBS_DONE`, `EXT_STATUS`, `ERRORS`, and finally
`STATUS_DONE`. The driver reads back the counters into `stats["perf"]`.

### 2.5 What the instructions look like

Each command is an inline function in `firmware/include/sysarray_intrinsics.h`:

```c
mat_load(ddr, laddr)   →  .insn r 0x0B, 1, 1, x0, a0, a1
//                           opcode = custom-0, funct3 = 1 (load), funct7 = 1 (matrix/DMA group)
//                           rs1 = DDR physical address, rs2 = local address (mem << 28 | word)
mat_exec(a, b, c, kt, acc)
//  rs1 = B << 16 | A,  rs2 = acc << 28 | Kt << 16 | C
```

In `llvm-objdump -d gemm_fw.elf` these instructions show as `<unknown>`
(the disassembler does not know custom-0), e.g. `0b 00 65 03` = `0x0365000b`.

---

## 3. PicoRV32 → PCPI → `rtl/sysarray/sa_pcpi.v`

### 3.1 Handshake

When PicoRV32 meets an instruction it does not know (custom-0), it hands it to the coprocessor interface PCPI:

```
cycle    PicoRV32                      sa_pcpi
t0       pcpi_valid=1, insn, rs1, rs2  ours=1 → pcpi_wait=1 (combinational, same cycle)
                                       S_IDLE: latch funct7/funct3/rs1/rs2, pack the command → S_EXEC
t1..tn   keeps valid, waits            S_EXEC: handled by instruction group (table below)
tn       ←                             pcpi_ready=1 (for mat_fence also pcpi_wr=1, pcpi_rd=ext_status)
tn+1     valid=0, next instruction     S_DONE → S_IDLE
```

- **No exception**: if there is neither `pcpi_ready` nor `pcpi_wait` within 16 cycles, PicoRV32
  raises an illegal-instruction exception. `sa_pcpi` raises `pcpi_wait` in the same cycle it claims the instruction, so
  long stalls like `mat_fence` do not trigger it.
- **Encodings not claimed** (e.g. funct7 = 3) get no response, and the CPU reports an illegal instruction as usual.

### 3.2 What each instruction does in `S_EXEC`

| Instruction | Behavior | How long the CPU stalls |
|---|---|---|
| `mat_cfg` / `vec_cfg` | Write internal configuration registers, `respond` immediately | About 2 cycles |
| `mat_load` / `mat_store` / `mat_exec` / `vec_run` | `q_valid = 1`, hand the command packet to the top-level arbiter (the legacy sequencer has priority); accepted and `respond`ed when the scheduler's `in_ready` (the 8-deep FIFO not full) | A few cycles when the FIFO has room; **stalls as long as it is full**: this is the firmware's back-pressure |
| `mat_fence` | Waits for `fence_ok` (the input pipeline empty and the engines in the mask idle) or a sticky error → `respond(rd = ext_status)` | Until all queued commands have executed |
| funct7 = 0 (legacy) | Handed to the `sa_legacy` sequencer (one 8×8×8 job) | Depends on the instruction |

### 3.3 Command packets: configuration frozen at issue

`S_IDLE` builds the command packet the moment the instruction arrives, from the **current** configuration registers
(`SA_PKT_W` = 261 bits; layout in `sa_defs.vh`):

- `mat_load`: `{ld_mode, ld_pitch, ld_rb, ld_rows, rs2 (local address), rs1 (DDR), CMD_LD}`
- `mat_exec`: `{crow, cstep, bstep, repeat-1, acc, Kt, C, B, A, CMD_EX}`
- `vec_run`: `{hi, lo, zp, shift, scale, period, types, op, groups, dst, src2, src1, CMD_VE}`

So once the firmware has issued a command it can **immediately** change the configuration for the next one; queued commands are unaffected.
When a sticky error is set, new commands are dropped and answered directly, so the CPU does not hang on a stalled queue.

---

## 4. Scheduler: `rtl/sysarray/sa_sched.v`

Commands pass four stages in program order, at most one command per stage:

```
input FIFO (8 deep) → d1: split fields, compute the span (words for LD/ST, Kt×D for EX)
                    → d2: legality checks + bank masks
                    → dispatch: conflict check → engine queue (4 deep) + in-flight mask FIFO (8 deep)
```

### 4.1 d2: legality checks and bank masks

- **Checks**: DDR address, row bytes and pitch 8-byte aligned; INTERLEAVE only allowed for LD,
  not writing ACC, row bytes a multiple of D; address ranges in bounds; EX's Kt ≠ 0;
  VE types must live in the matching memory (int32 in ACC, int8/int16 in SPAD).
  Illegal → `reject` → sets the **sticky error** (error code and engine number go into `ext_status`).
- **Bank masks**: 6 bits = SPAD_A b0/b1, SPAD_B b0/b1, ACC b0/b1. The bits of the banks the word range of a command
  covers are set, separately as a read mask `c_r` and a write mask `c_w`.
  Example: `EX(strip 0, left B)` reads SPAD_A b0 + SPAD_B b0 and writes ACC b0.

### 4.2 Dispatch and conflicts

- Compared with the in-flight masks of **other engines**: RAW (I read, it writes), WAR (I write, it reads),
  WAW (both write).
- EX and VE share SPAD's port B and ACC's port A, so the two conflict whenever they touch the same bank
  (even if both only read).
- Commands of the **same engine** execute in order by construction and are not checked against each other.
- No conflict, engine queue and mask FIFO not full → dispatch.
- A mask is popped only when the engine signals **`done`**, not when the engine accepts the command: "in flight"
  lasts until the command has really finished.
- **DDR is not tracked**: when the same DDR region is first ST'd and then LD'd, software must put a
  `mat_fence` in between.

**In-order dispatch is performance critical**: when one command is stuck on a conflict, every command behind it waits.

### 4.3 Conflicts in this example

| Command | What it waits for |
|---|---|
| EX(0, left B) | LD left B (SPAD_B b0), LD A0 (SPAD_A b0), RAW. **Does not wait for the right half of B**: this is the gain of the B split |
| EX(0, right B) | LD right B (SPAD_B b1) |
| LD A1 → SPAD_A b1 | No bank conflict with EX(0); loads in parallel while EX(0) computes |
| ST(0), reads ACC b0 | Both EX(0) finished writing ACC b0 (RAW) |
| EX(1), writes ACC b1 | LD A1 done; can run in parallel with ST(0) |
| LD A2 → SPAD_A b0 | EX(0) finished reading SPAD_A b0 (WAR) |
| EX(2), writes ACC b0 | ST(0) finished reading ACC b0 (WAR) |

### 4.4 Errors

- An illegal command, or LD/ST receiving an AXI SLVERR/DECERR → sticky error, **dispatch stops**.
- Dispatched commands run to completion; queued commands stay in place until `clear_error`
  (`mat_reset`) clears them.
- `mat_fence` also returns when there is a sticky error; the firmware and the driver report it from there.

---

## 5. Engines and memories

### 5.1 Memories and port assignment (`sa_unit.v`)

| Memory | Size / word width (D = 16) | Port A | Port B |
|---|---|---|---|
| SPAD_A | 128 KB, 8192 words × 16 B, two banks | LD write / ST read | EX read / VE read-write |
| SPAD_B | 128 KB, same | LD write / ST read | EX read / VE read-write |
| ACC | 256 KB, 4096 words × 64 B (16 int32), two banks | EX / VE read-write | LD write / ST read |

Each memory is split into two banks by `sa_bankmem`, and each bank is a `sa_tdpram` (the UG901
byte-write true dual-port RAM template, inferred as RAMB36), 128 BRAM36 in total. Several
users on the same port (e.g. LD and ST) are multiplexed by their enables; the scoreboard guarantees they never access the same bank at the same time.

Local address (LADDR) = `mem << 28 | word`; mem: 1 = SPAD_A, 2 = SPAD_B, 3 = ACC.

### 5.2 LD engine: `sa_ld.v`

1. Command shape: rows rows of row_bytes bytes each, row distance pitch; LINEAR or INTERLEAVE.
   Contiguous rows (pitch = row_bytes and whole-word aligned) are merged into one row for longer bursts.
2. Split into AXI INCR bursts: 8 bytes per beat, at most 16 beats, **not crossing a 4 KB boundary**;
   at most 8 bursts in flight per port.
3. AR requests: `m0_axi` → `matmulHpConverter` (AXI4 → AXI3) → **HP2** → DDR controller.
4. Each burst remembers its starting local word and lane. Returning 64-bit data beats are written by address into
   the matching 8 bytes of the matching SPAD / ACC word (byte write enables): a SPAD word = D/8 lanes,
   an ACC word = D/2 lanes.
5. INTERLEAVE: local word = block number × rows + row number. This lays out the A EX needs
   (k-tile first: word w·D + g = A[g][wD .. wD+D−1]) and B in column strips
   (strip j starts at word j·K, word k = B[k][Dj .. Dj+D−1]).
6. `done` after the last beat is written; on an SLVERR/DECERR along the way `err` is reported with it.

### 5.3 EX engine: `sa_ex.v` + `sa_array.v`

1. **Streaming K**: each cycle one word is read from SPAD_A and one from SPAD_B (BRAM
   read latency is 1 cycle, so reads are issued one beat ahead). `rowreg` holds each row's current A word and `bsr`
   is a shift register of B words; together they produce the diagonal skew: PE(i,j) gets
   A[i][k] and B[k][j] in the same step.
2. **Multiply-accumulate**: A flows right, B flows down, PE(i,j) accumulates C[i][j] (output stationary).
   The first 8 PE columns use DSP48E1 (`sa_pe_dsp`), the last 8 LUT multipliers (`sa_pe_lut`).
3. **Tile switch**: after streaming K + 2(D−1) + 1 = 287 cycles per tile, `swap`: the accumulated
   results move into shadow registers, the array is cleared, and the next tile starts immediately.
4. **Write-back to ACC**: the shadow registers shift up row by row, writing the 16 rows into ACC at row address =
   c + r·cstep + i·crow (here crow = N/16, giving a row-major C strip). Write-back overlaps with
   the next tile's computation; accumulate mode has to read the old values first (two cycles per row).
5. **repeat**: one command computes 8 tiles; tile r's B base = b + r·bstep
   (bstep = K). `done` only after the last tile is written.

### 5.4 ST engine: `sa_st.v`

Reads SPAD / ACC by lanes and writes back to DDR in bursts over the AW/W channels (also through HP2,
also not crossing 4 KB). `done` after all B responses are received; `err` on an error response.
In this example one ST writes 16 × 256 × 4 = 16 KB, about 2k cycles, overlapping with the next strip's computation.

### 5.5 VE engine: `sa_ve.v` (appears for int8 GEMM / vector operations)

The read unit issues one read per cycle (src1's word first, then src2's; a broadcast src2 is read only once) →
a 4-stage compute pipeline (op → ReLU + requant multiply → rounding shift → zp + clamp + saturate + pack) →
an 8-deep output FIFO → write-back (writes have priority when sharing a port with reads). In int8 GEMM, VE reads
int32 from ACC, adds the bias vector (src2 period = N/D), applies ReLU and requant, writes into the upper bank of SPAD_A,
and ST then writes the int8 back to DDR.

---

## 6. Completion and return

1. Each engine's `done` makes the scheduler pop the matching in-flight mask, and commands stuck on conflicts behind it dispatch.
2. All commands done → `fence_ok` → `mat_fence` returns `ext_status` (bit 1 = 0 without errors).
3. The firmware records the cycle count, writes `STATUS_DONE`, then halts with `ebreak`.
4. Python polls DONE → `invalidate()` → reads C → compares with NumPy:
   **PASS, 82,813 cycles, 202.6 MAC/cycle** (measured on the board by `m4_sched_tune.py`).

### Where the cycles go (measured with the performance counters, `notebooks/m4_perf.py`)

| Part | Share of the counting window | Meaning |
|---|---|---|
| EX useful compute (`EX_USEFUL`) | **79.0%** | The array reading new K data: 16 × 16 tiles × 256 steps = 65,536 cycles |
| EX fill (`EX_STEP − EX_USEFUL`) | 9.3% | The 2(D−1) = 30 skew steps per tile, structural overhead |
| Head-of-queue EX stuck on a conflict (`HAZ_EX`) | 8.8% | EX waiting for a bank (the right half of B loading, the previous ST finishing its ACC read, etc.) |
| Frontend starvation + fully idle | 4.3% + 0.4% | No command at the head of the queue: time when firmware issue falls behind; rare |
| CPU stuck on a full queue (`PCPI_QFULL`) | 61.3% | The firmware is far ahead of the hardware; the frontend is not the bottleneck |
| LD / ST busy | 20.1% / 39.8% | 7.85 / 7.93 B/cycle while busy, close to the port limit |

"Head of queue stuck on ST" at 86% is the normal state with prefetch ordering: ST(i) waits at the head for EX(i) to finish while the commands behind it already execute in the engines.
Detailed analysis in design document §10.4.

---

## 7. Data contracts between layers

| Boundary | Contract | Defined in |
|---|---|---|
| ARM ↔ RISC-V | Mailbox: 256 bytes from offset 0x1F00 of the program BRAM; input parameters, output status and cycle counts | `firmware/include/mailbox.h`, `MBOX_*` in `driver/pynq_matmul.py` |
| ARM ↔ accelerator | DDR physical addresses, 8-byte aligned; flush / invalidate on the ARM side | `docs/memory_model.md` |
| Firmware ↔ hardware | custom-0 instruction encodings, the keys of `mat_cfg` / `vec_cfg` | `sysarray_intrinsics.h`, design document §8 |
| PCPI ↔ scheduler | 261-bit command packet | `rtl/sysarray/sa_defs.vh` |
| Scheduler ↔ engines | `cmd_valid/ready` + command fields; `done` (+ `err`) | Instantiation wiring in `sa_unit.v` |
| Engines ↔ memories | Local word addresses, byte write enables; bank = upper half of the address | `sa_bankmem.v` |
| Engines ↔ DDR | AXI4 INCR bursts, ≤ 16 beats, not crossing 4 KB, converted to AXI3 to HP2 | `sa_ld.v` / `sa_st.v`, `pico_bit.tcl` |
| Firmware ↔ ARM (counters) | Counter area at 0x1E00–0x1EFF of the program BRAM, count in `MBOX_PERF_COUNT` (0x8C) | `mailbox.h`, `PC_*` in `sa_defs.vh` |
| ARM ↔ fetch unit (lists) | 64-byte descriptors, 64-byte aligned, in DDR; address, count and BASE0–3 passed to the `desc_run` firmware through the mailbox (0x90–0xA4) | Design document §8.6, the driver's `DescList` |

---

## 8. Debug checklist by layer

| Symptom | Which layer first | How to check |
|---|---|---|
| `TimeoutError: ... not started` | The firmware did not start | Is the firmware loaded (`load_firmware`); the reset GPIO; does the firmware exceed 0x1E00 |
| `TimeoutError: ... running` | The firmware is stuck on an instruction | Most likely stuck in `mat_fence`: is there a command that can never dispatch (a conflict that never clears) or an engine that never signals `done`; reproduce with `make sim` |
| `RuntimeError: accelerator error` | Scheduler / DMA | Error code [11:8]: 1 = illegal shape / alignment, 2 = local address out of range or wrong memory number, 3 = AXI read response error, 4 = AXI write response error; engine number [15:12]: 0 = LD, 1 = ST, 2 = EX, 3 = VE |
| Partly wrong results | Data layout / caches / DDR ordering | flush / invalidate done; a `mat_fence` between ST and LD of the same DDR region; the INTERLEAVE layouts of A / B |
| All results wrong but simulation correct | Build configuration | Do D, SPAD_WORDS, ACC_WORDS of `sa_unit` in the `.hwh` match; the CAPS the firmware read |
| Lower performance than expected | Firmware command order | Under in-order dispatch, what is a command waiting for (see §4.3); compare scheduling variants with `GEMM_FLAGS` |
| CPU reports an illegal instruction | Instruction encoding | Are funct7 / funct3 within the `ours` range of `sa_pcpi`; `mat_perf` / `mat_submit` only work on overlays with CAPS bit 20 / 21 set |
| No idea where the time goes | Performance counters | `stats["perf"]` + `perf_breakdown()`, or `m4_perf.py`: array efficiency, which engine waits on conflicts, whether the frontend starves, DMA bandwidth while busy |
| List execution error (engine number 4) | Descriptors | Error code 1 = illegal opcode / non-zero reserved header bits / misaligned address, 3 = read error on fetch; CSR 0xCC is the index of the failing descriptor, 0xC0 / 0xD0 the address and count last decoded |
| An LD in a list reads stale data | DDR ordering | When the same DDR region is ST'd then LD'd, there must be a FENCE descriptor in between or FENCE_BEFORE set on the LD (the hardware does not track DDR) |

---

## 9. Watching the process yourself

**Waveforms**: the system simulation in `firmware/gemm` covers the whole path from firmware instructions to the RTL (`tb_system.v` plays the ARM side).

```sh
cd firmware/gemm && make sim SIM_D=16          # run once first to produce the build products
cd build/sim16
source /home/jon/Projects/Vivado/Vivado/2024.1/settings64.sh
xelab -debug typical tb_system -s tb_dbg
xsim tb_dbg -gui
```

Add to the waveform in order:

1. `pcpi_valid / pcpi_insn / pcpi_wait / pcpi_ready` (CPU ↔ `sa_pcpi`)
2. `mm/sched/in_valid`, `in_ready`, and `dispatch`, `hazard` (scheduler)
3. `mm/ld/cmd_valid`, `done`, `mm/ex/...`, `mm/st/...` (the engines)
4. `mm/m0_axi_ar*`, `r*`, `aw*`, `w*`, `b*` (the DMA bus; at the `tb_system` top level they are called `m_ar*`, `m_r*`, etc.)

Intervals with `dispatch` at 0 and `hazard` at 1 are commands waiting on the scoreboard.

**Looking at one engine alone**: `make sim TB=tb_sa_ex GEN=D=16` in `rtl/sysarray`
(or `tb_sa_dma`, `tb_sa_ve`); same method, just elaborate in the matching build directory.

---

## 10. Another path: descriptor lists

The same GEMM can also be issued not command by command by PicoRV32 but as a descriptor list built by the ARM,
with PicoRV32 executing a single `mat_submit` (M5, design document §8.6). From the scheduler on the path is identical to sections 4–6;
what differs is where the commands come from:

1. **Python**: `mm.gemm_list(a, b)` calls `build_gemm_list()`, which produces a `DescList` with exactly the same schedule as `gemm_fw.c`
   (resident B, B split, A prefetch): one 64-byte descriptor per command carrying all its configuration,
   ending with END. The list is written into a DDR buffer from `allocate()` and `flush()`ed.
2. **Firmware**: `firmware/desc_run/desc_run_fw.c` reads the list address, count and BASE0–3 from the mailbox,
   sets the BASEs with `mat_cfg`, then `mat_submit(list, count)` (returns immediately) and `mat_fence`.
3. **PCPI**: funct3 = 6 sends a start pulse to the fetch unit; while the list runs, new queued commands on PCPI
   are held off by `pcpi_wait`, and `mat_fence` also waits for the list to finish, so program order is preserved.
4. **Fetch unit** (`rtl/sysarray/sa_cmdfetch.v`): starting from the list address, one 8-beat burst per descriptor,
   at most 4 in flight, data into a 32 × 64-bit LUTRAM FIFO; decoded once 8 words have arrived:
   - LD / ST / EX / VE: packed into 261-bit command packets with the same `pkt_*` functions as PCPI and sent to the scheduler input
     (priority: legacy > fetch unit > PCPI);
   - FENCE: waits until the scheduler queue is empty and the engines in the mask are idle;
   - JUMP: continues at a new address, first dropping prefetched data still on the way (DRAIN);
   - END: records a status value; the list ends.
5. **Read port**: the fetch unit and the LD engine share the read channel of port 0. A granted AR request is locked until its handshake completes;
   an ownership FIFO recording "who this burst belongs to" in AR order routes returning R beats to LD or the fetch unit.

Measured (`notebooks/m5_desc_demo.py`, design document §10.5): small operations that were frontend bound get clearly faster
(16³ 4.19×, 64³ 1.23×, int8 64³ 1.52×, broadcast vector operations 2.60×); large GEMMs such as 256³ are unchanged,
because their frontend was already running ahead of the hardware.

---

## 11. A third path: one LLM token (L4–L5)

Take `LlamaDevice.forward(token)` in `llm/runtime.py`: stories15M, `bitstreams/l2` (D = 8, with the fp32 VE),
about 2.48M cycles (49.6 ms) per token. The full design is in [`llm_inference_plan.md`](llm_inference_plan.md) §5, §8, §9.
Compared with section 10 there are three more things:
- the command ring: resident firmware, the ARM can submit at any time;
- a static list: one for the whole model, varied only by parameters;
- interrupts: the ARM is notified on completion.

```
LlamaDevice.forward ──► write the parameter block (pos, token) + ring entry (list address, BASE0–2, PARAM block) ──► flush
   ──► write mailbox RING_TAIL (doorbell)
rt_fw (PicoRV32 resident loop) sees the new entry ──► mat_cfg sets BASE0–3, PARAM0–7 ──► mat_submit ──► mat_fence
   ──► sa_cmdfetch fetches 953 descriptors (executes 2,477): LDPARAM computes addresses, dynamic slots replace fields, LOOP_END loops
   ──► scheduler ──► LD / EX / ST / VE (integer) / VE (fp32, sa_vefp) ──► SPAD / ACC, DDR through HP2
   ──► END ──► rt_fw writes the completion record + RING_HEAD ──► mat_notify ──► notify_irq ──► AXI intc ──► ARM
LlamaDevice.wait woken by the interrupt ──► invalidate ──► read logits ──► NumPy samples the next token
```

**1. Python (`llm/runtime.py`)**
- Once at startup: the model file `.w8a8`, the io area, the KV cache and the static list go into one CMA buffer;
  the list is produced by `ModelCompiler.build()` in `llm/compile_model.py`.
- Per token:
  - write two words (pos, token) in the io area;
  - write a 64-byte ring entry with `Device.submit()`: type RUN_LIST, IRQ flag, seq, list address,
    BASE0 = model, BASE1 = io, BASE2 = KV, and the parameter block of PARAM0–5 (pos+1, pos_pad, etc., §8.5);
  - after a flush write the mailbox's `RING_TAIL`: this is the doorbell.

**2. Firmware (`firmware/rt/rt_fw.c`)**
- A resident loop polls `RING_TAIL`. On a new entry it:
  - writes BASE and PARAM with `mat_cfg` (keys 11–34);
  - starts the list with `mat_submit` and waits for it with `mat_fence`;
  - writes a 32-byte completion record in DDR (seq, status, cycles, executed count, END value) and updates `RING_HEAD`;
  - executes `mat_notify` if the entry carries the IRQ flag.
- On an error it does `mat_reset` first and continues; later entries are unaffected.

**3. Fetch unit (`sa_cmdfetch.v`)**: as in section 10, plus L1's extensions:
- **Dynamic slots**: two slots in descriptor w0[31:16], replaced by or added from PARAMs at decode time.
  E.g. attention's LD K reads with `rows = P1` (pos_pad) and `ddr += P6` (head offset),
  EX's `repeat = P2`, softmax's `VALID = P0`;
- **LDPARAM**: at fetch time reads one word from DDR, multiplies by mul, adds add, and writes it into a PARAM.
  The device uses it to compute the KV row offset (pos × dim), the RoPE row offset and the embedding row offset from the parameter block, and to read the embedding's scale;
- **LOOP_END**: attention loops over heads (P6, P7 advance by a stride each round), the classifier loops over blocks;
- **FENCE**: only after this token's K and V rows are ST'd into DDR can attention LD them back. The hardware does not track DDR dependencies,
  so this FENCE is required (`memory_model.md`).

**4. Engines**: LD, EX, ST and the integer VE are as in section 5, plus:
- **fp32 VE** (`sa_vefp.v`, executes commands with the FP bit or op = 6):
  - RMSNorm: REDUCE SUM, then RSQRT;
  - activation quantization: REDUCE MAX(ABS), then RECIP, finally int8 output replicated by DIV into an A strip;
  - dequantization: I32 × s_w (MOD) × s_x (DIV);
  - RoPE (SWAPNEG), softmax (VALID, EXP), SiLU;
  - TRANSPOSE turns K into a B strip.
- **One linear layer**: weights are read block by block into SPAD_B with LD, alternating the two banks; EX multiplies the replicated-row A strip by the weights;
  the next block's LD overlaps this block's EX (the classifier's LD is busy 93.9%).

**5. Return**: `Device.wait()` waits on `notify_irq` (PYNQ `Interrupt`), reads the completion record when woken,
and reads the logits after invalidating. A completion record must be read by the ARM before its ring slot can be reused (the command ring section of `memory_model.md`).

**Where the cycles go** (counters of the last token, `l5_generate.py`):
- EX useful about 77%, LD busy about 80%, VE active about 14%;
- about 80% of head-of-queue blocking is VE: post-processing and attention VE steps waiting for the preceding EX to finish;
- the ARM side adds about 13 ms per token: sampling, waiting for the interrupt, copying the logits.

**Batching** (L5b, `llm/compile_batch.py`): the same path, only 8 sequences share one weight stream per step.
Each sequence's PARAMs are read from a DDR parameter table by LDPARAM, and A strips make a round trip through DDR (ST → FENCE → LD INTERLEAVE).
8 sequences together 63.3 tok/s; the bottleneck moves to VE.

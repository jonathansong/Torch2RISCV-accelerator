# Double-buffered accelerator design (v2 of the matmul unit)

Status: **M1 done and verified on the board** (`rtl/sysarray`, D = 8, one
DMA port; bitstream and results in `RISCV-on-PYNQ-Z1/bitstreams/m1/`:
256×256×256 at 54.5 MAC/cycle, 85 % of the array peak, 35× Phase 4).
Notes from the implementation are marked *(M1)*. **M2 was redefined from
measurements** (§10.1): one HP port already runs at 99 % of its 8 B/cycle,
so M2 cuts command-issue overhead instead (repeat `mat_exec`, §4.1) and the
three DMA ports move to M4. Changes are marked *(M2)*. **M2 done and verified
on the board** (`RISCV-on-PYNQ-Z1/bitstreams/m2/`): 64³ 13.0 → 38.3 MAC/cycle,
128³ 28.4 → 49.8, 256³ 56.5 (88 % of peak). **M3 (vector engine)
done and verified on the board** (`RISCV-on-PYNQ-Z1/bitstreams/m3/`): fused
int8 GEMM (bias + RELU + requant on the chip) bit-exact at 53.5 MAC/cycle
for 256³ (int32 output: 56.5), standalone vector ops bit-exact for every
op/type combination; decisions taken while building it are marked *(M3)*
(§6.4, §8.4). **M4 (D = 16) done and verified on the board**
(`RISCV-on-PYNQ-Z1/bitstreams/m4/`): 256³ at 180.4 MAC/cycle (3.2× M3),
512×256×256 at 191.1, all M1–M3 tests bit-exact; the measurements show one
HP port is not the limit at D = 16, so the three ports are not built (§10.3).
Changes are marked *(M4)*. **Performance counters** (`sa_perf.v`, `mat_perf`;
[plan](perf_counters_and_desc_dma_plan.md) P1–P4) are verified on the board
(`RISCV-on-PYNQ-Z1/bitstreams/m4p/`) and give the measured cycle breakdown in
§10.4. **Descriptor lists** (`sa_cmdfetch.v`, `mat_submit`, §8.6) are verified on
the board (`RISCV-on-PYNQ-Z1/bitstreams/m5/`): front-end-bound work runs 1.2–4.2×
faster, large GEMMs unchanged (§10.5).

**LLM inference levels L0–L5b** ([`llm_inference_plan.md`](llm_inference_plan.md))
build on this unit. The LLM work uses a **D = 8** build (L0) to leave room for
new logic. The unit was extended twice:
- **L1**: compiler-facing command extensions in `sa_cmdfetch.v` (BASE0–15,
  PARAM0–7, dynamic fields, SETREG, LOOP_END, CALL / RET, LDPARAM); a host
  interrupt (`mat_notify`, `notify_irq`); a resident runtime firmware that
  serves a command ring. Bitstream: `bitstreams/l1`.
- **L2**: an fp32 vector engine (`sa_vefp.v`: fp32 ops, EXP / RECIP / RSQRT,
  row reductions, index modes, TRANSPOSE), next to the integer VE of §6.
  Bitstream: `bitstreams/l2`, 44.0k LUT, WNS +1.84 ns at 50 MHz.

stories15M runs end to end on the `l2` build, bit-exact with the reference
model. Changes are marked *(L1)* / *(L2)*; the full specifications are in the
LLM plan §5 and §6.

## 1. Goals and decisions

The Phase 3/4 unit processes one independent 8×8×8 job at a time, straight
from DDR: per job it reads a descriptor, A and B, computes 22 cycles and
writes C, all serialized. On the board a job costs 311.9 cycles (custom
instruction path), of which 149.4 are the unit and only 22 are compute —
**1.6 MAC/cycle out of a 64 MAC/cycle array**. This design makes the array,
not DDR latency or the CPU, the limit.

| # | Decision | Section |
|---|---|---|
| 1 | Clock: everything on `riscv_clk` = 50 MHz; no 100 MHz pipelining | — |
| 2 | Array size is a parameter **D**: build D = 8 first, then D = 16 | §4 |
| 3 | D = 16 array: **8 DSP columns + 8 LUT columns** (DSP margin for the vector engine) | §4.4 |
| 4 | **K-streaming** output-stationary array with shadow accumulators | §4 |
| 5 | On-chip memories: **128 BRAM36** = SPAD_A 32 + SPAD_B 32 + ACC 64, each **double-banked**; depths are parameters; program BRAM shrinks to 8 KB | §3 |
| 6 | Memory-centric: DMA, array and vector engine are **peer engines** on the same memories | §2 |
| 7 | Ordering: **hardware bank scoreboard** — software issues commands in program order | §7 |
| 8 | DMA: **3 symmetric HP ports** (HP1–HP3), bursts striped across ports, DDR ↔ any memory | §5 |
| 9 | Vector engine with **VL = D lanes**, used both fused after matmul (bias/ReLU/requant) and standalone | §6 |
| 10 | ISA: new groups funct7 = 1 (matrix/DMA) and funct7 = 2 (vector); funct7 = 0 and the CSRs stay compatible | §8 |

## 2. Architecture

```
                    PicoRV32 ──PCPI──► decoder ──► dispatch queue ──► scoreboard (§7)
                                                        │
                        ┌───────────────┬───────────────┼───────────────┬───────────────┐
                        ▼               ▼               ▼               ▼               │
                   LD engine       ST engine       EX engine       VE engine            │
                   (DMA read)      (DMA write)     (array)         (vector)             │
                        │               ▲               │  ▲            │  ▲            │
   ┌────────────────────┼───────────────┼───────────────┼──┼────────────┼──┼──────────┐ │
   │  SPAD_A [bank0|bank1]   SPAD_B [bank0|bank1]   ACC [bank0|bank1]   (§3)        │ │
   └────────────────────┼───────────────┼──────────────────────────────────────────┘ │
                        │               │                                              │
                   3 × AXI4 master (64-bit) ─► protocol converter ─► S_AXI_HP1/2/3 ─► DDR
```

- **LD** moves DDR → any memory, **ST** moves any memory → DDR. They share
  the three AXI ports (reads and writes are independent AXI channels).
- **EX** reads SPAD_A and SPAD_B, writes ACC.
- **VE** reads and writes any memory.
- All four engines run concurrently on different banks; the scoreboard only
  serializes commands that touch the same bank from different engines.
- Legacy paths (CSRs, funct7 = 0 instructions) become a small sequencer that
  issues LD/EX/ST commands (§8.5).

*(As built, M1–M5.)* One AXI port (`NPORTS = 1`, HP2) — §10.1/§10.3 showed
more ports are not needed — and two additions around the scheduler:

```
  PicoRV32 ──PCPI──► sa_pcpi ─────────────┐
  DDR list ──(port 0 reads)──► sa_cmdfetch ┼─► dispatch queue ─► scoreboard ─► LD / ST / EX / VE
  legacy sequencer (CSRs, funct7 = 0) ─────┘      (priority: legacy > list > PCPI)
  sa_perf: 32 event counters fed by the scheduler, PCPI and the engines (§10.4)
```

- **Descriptor fetch unit** (`sa_cmdfetch.v`, §8.6): runs lists of 64-byte
  commands the ARM built in DDR after one `mat_submit`; it shares port 0's
  read channel with LD.
- **Performance counters** (`sa_perf.v`, `mat_perf`, CSR mirror).

## 3. On-chip memories

### 3.1 Budget (xc7z020: 140 BRAM36)

| Use | BRAM36 |
|---|---|
| SPAD_A (int8 A operands, also vector data) | 32 |
| SPAD_B (int8 B operands, also vector data) | 32 |
| ACC (int32 results, bias, vector data) | 64 |
| PicoRV32 program BRAM (8 KB; ARM window shrinks from 64 KB to 8 KB) | 2 |
| **Total** | **130 (93 %)** |
| Reserve (debug ILA, FIFOs) | 10 |

Depths are Verilog parameters (`SPAD_BRAM`, `ACC_BRAM`), so a debug build
can shrink them to make room for an ILA. With the D = 8 array a 32-BRAM
configuration performs the same (§10); the full budget matters for D = 16,
for whole-layer residency and for fewer, larger commands.

### 3.2 Organization

| Memory | Word | Words (D = 8) | Words (D = 16) | Banks |
|---|---|---|---|---|
| SPAD_A | D bytes (one A tile row) | 16384 × 64 bit | 8192 × 128 bit | 2 (upper half of the index = bank 1) |
| SPAD_B | D bytes (one B tile row) | 16384 × 64 bit | 8192 × 128 bit | 2 |
| ACC | D × int32 (one C row) | 8192 × 256 bit | 4096 × 512 bit | 2 |

Each bank is its own set of BRAMs with two ports, so different banks never
contend. Port use per bank:

| Memory | Port A | Port B |
|---|---|---|
| SPAD_A / SPAD_B | LD / ST (DMA) | EX read, VE read/write |
| ACC | EX drain write, VE read/write | LD / ST (DMA) |

Within a port, EX and VE never touch the same bank at the same time (the
scoreboard orders them), so the port mux is a static select, not an arbiter.

### 3.3 Local addresses

Engines address memories with a 32-bit **local address (LADDR)**:

| Bits | Field |
|---|---|
| 31:28 | memory: 1 = SPAD_A, 2 = SPAD_B, 3 = ACC (0 and others: error) |
| 27:16 | reserved, 0 |
| 15:0 | word index in that memory (bank = MSB of the valid index range) |

### 3.4 Data layouts

- **A strip** (D rows × K columns of A, int8) in SPAD_A, *k-tile major*:
  word `base + w·D + g` = `A[g][w·D .. w·D+D-1]`, w = 0 .. K/D-1. This is
  what the array consumes one word per cycle (§4.2). Produced by an
  INTERLEAVE load from row-major DDR (§5.3).
- **B strip** (K rows × D columns of B) in SPAD_B, row per word:
  word `base + k` = `B[k][n0 .. n0+D-1]`. Produced by a LINEAR 2D load.
- **C tile** (D × D int32) in ACC: word `base + i` = `C[i][0 .. D-1]`.
- **Vector data**: elements packed little-endian in words; per word D int8,
  D/2 int16, or (ACC) D int32.

## 4. Array engine (EX)

### 4.1 Command

`mat_exec(A, B, C, Kt, flags)`: for one D×D output tile

```
C[i][j] (+)= Σ_{k < Kt·D} A[i][k] · B[k][j]
```

reading the A strip at SPAD_A word A, the B strip at SPAD_B word B, writing
D ACC words at C. `flags.accumulate` adds to the current ACC contents (bias
preload, or K split across several commands); otherwise ACC is overwritten.

*(M2)* One command can compute `repeat` output tiles with the same A strip
(configured with `mat_cfg` EX keys, §8.2): tile r reads the B strip at
B + r·Bstep and writes its row i to ACC word C + r·Cstep + i·Crow. With
Bstep = K, Cstep = 1, Crow = N/D a whole row of C tiles is computed by one
command and laid out row-major in ACC, so one `mat_store` writes the D × N
strip back. Tiles stream back to back like separate commands; the command
completes after its last tile.

### 4.2 K-streaming dataflow

Output-stationary, as in Phase 2, but the operands are streamed for all
Kt·D steps instead of 8, so one command computes the full dot products.

- Row g of the array needs `A[g][t-g]` at step t. With the k-tile-major
  layout, each A word serves one row for D steps and exactly one row needs a
  new word per cycle (row `t mod D`): **one SPAD_A read per cycle**, loaded
  into a rotating set of D row registers.
- Column j needs `B[t-j][j]`: B word k is needed from step k (column 0) to
  k+D-1 (column D-1), so a D-deep shift register of B words with **one SPAD_B
  read per cycle** feeds all columns.
- Compute time per command: `Kt·D + 2(D-1)` cycles (fill 14 for D = 8,
  30 for D = 16); peak D² MAC/cycle. The next command starts streaming
  after the previous one's fill completes (fill is not overlapped in M1).

### 4.3 Shadow accumulators and drain

Each PE has an active and a shadow accumulator. When a command's last
products are in, the active values move to the shadow (1 cycle) and the
next command starts streaming while the shadows drain: they shift down
column-wise and the bottom row writes one ACC word per cycle, **D cycles
per tile** (2D with `accumulate`, read-add-write on ACC port A). Drain is
therefore hidden; fill is not.

Later optimization (not in M1–M4): overlap the fill of the next command
with the tail of the current one by switching accumulators per PE along the
wavefront, saving 2(D-1) cycles per command (matters for small Kt).

### 4.4 PE implementation

Two PE modules with identical behavior, selected per column by a generate
parameter `DSP_COLS`:

| Variant | Resources per PE | Where |
|---|---|---|
| `systolic_pe_dsp` (`use_dsp = "yes"`) | 1 DSP48E1 (MAC + acc in P register) + shadow FFs | columns 0 .. DSP_COLS-1 |
| `systolic_pe_lut` (`use_dsp = "no"`) | ≈ 92 LUT + 36 FF (Phase 2 measurement) + shadow FFs | remaining columns |

D = 8: `DSP_COLS = 8` (64 DSP). D = 16: `DSP_COLS = 8` → 128 DSP PEs + 128
LUT PEs.

## 5. DMA (LD / ST engines)

### 5.1 Ports

Three identical AXI4 masters (64-bit data, 32-bit address, no IDs, INCR,
≤ 16 beats), each through an AXI4→AXI3 protocol converter to S_AXI_HP1,
HP2 and HP3. HP0 stays with the PicoRV32. All ports carry reads and writes.

### 5.2 Burst striping

A command is split into bursts of ≤ 16 beats that never cross a 4 KB page;
bursts are dispatched round-robin to the ports, up to 4 outstanding reads
and 4 outstanding writes per port. Each burst carries its own local address,
so read data is written by address as it returns from any port — no
reordering buffer. A command completes when all of its bursts have
completed (per-command burst counter).

### 5.3 Transfers

Shape comes from configuration (§8.2), captured when the command is queued:

| Field | Meaning |
|---|---|
| rows | number of DDR rows (1 .. 65535) |
| row_bytes | bytes per row, multiple of 8 |
| pitch | DDR bytes between row starts, multiple of 8 |
| mode | LINEAR: local words fill in row-major order. INTERLEAVE: local word = `chunk·rows + row` (chunk = D-byte piece of a row) — produces the A layout of §3.4 |
| zero_pad | a row shorter than a local word is zero-filled (8×8 legacy jobs on the D = 16 array) |

DDR addresses must be 8-byte aligned. A store reads local words and writes
`rows × row_bytes` with `pitch` (C tiles: row_bytes = 4D).

*(M1)* Contiguous LINEAR rows (`pitch == row_bytes`, whole local words) are
handled as one long row, so e.g. an 8×8 tile is one 8-beat burst. An
INTERLEAVE load of a whole K×N matrix places column strip j at word j·K —
the resident-B layout used by `firmware/gemm`. In M1 each engine runs one
command at a time (no overlap of consecutive LD commands), and LD writes
one 64-bit beat per cycle into local memory (8 B/cycle at D = 8). For M2
the three ports only pay off if LD commands to different memories run
concurrently or beats are combined into wider writes.

### 5.4 Errors

SLVERR/DECERR, a local address out of range or an illegal shape set a
sticky error (engine id + code, §8.3); the unit stops dispatching new
commands until `mat_reset`.

## 6. Vector engine (VE)

### 6.1 Lanes and throughput

VL = D lanes (8 or 16). Each lane: one DSP for int16 × int16, one DSP for
the 32 × 16 requantization multiply, plus an ALU (add/sub/min/max/shift/
saturate). Elements per cycle by source:

| Source | D = 8 | D = 16 |
|---|---|---|
| ACC (int32) | 8 (one word) | 16 (one word) |
| SPAD int8 | 8 | 16 |
| SPAD int16 | 4 per operand | 8 per operand |

### 6.2 Operations

`dst[e] = op(src1[e], src2[e])` for e < V_LEN, with src2 optionally
broadcast (stride 0 → one row repeated, e.g. a bias vector):

| Op | Semantics |
|---|---|
| ADD, SUB, MUL | elementwise, saturating to the output type |
| MAC | `dst += src1 · src2` |
| MAX, MIN | elementwise |
| RELU | `max(src1, 0)` |
| REQUANT | `clamp(((src1 · scale + 2^(shift-1)) >> shift) + zp, lo, hi)` — int32 → int8/int16 |
| COPY | type conversion / move between memories |

Types: in int8/int16/int32, out int8/int16/int32 (per-op legality in the RTL spec).

*(M3)* As built (`rtl/sysarray/sa_ve.v`):

| Op | Semantics |
|---|---|
| ADD, SUB | int32 saturating |
| MUL | int8/int16 inputs only (16 × 16 → 32, exact) |
| MAX, MIN | elementwise |
| COPY | unary (src2 not read): type conversion, move, or a pure post-op |

followed by optional flags applied in this order: **RELU**, **REQUANT**
(`((x · scale + 2^(shift-1)) >>> shift) + zp`, scale int16, shift 0–31),
then the clamp window `[lo, hi]` (always applied; default full int32) and
saturation to the output type. RELU and REQUANT are flags on any op, not
ops of their own, so "bias ADD + RELU + REQUANT" is one command. MAC was
dropped: `mat_exec` with `accumulate` covers the GEMM case, and a
read-modify-write of dst would need a third read port slot per group.

src2 is addressed by a **period** instead of a stride: group g reads src2
group g (period 0), 0 (period 1, broadcast) or g mod period. A bias vector
of N int32 is N/8 groups, so the fused epilogue over a row-major 8 × N strip
uses period N/8.
### 6.3 Uses

- **Fused after matmul**: `mat_exec → ACC bank b`, then `vec_run(REQUANT or
  bias ADD + RELU, ACC bank b → SPAD)` while EX computes into the other ACC
  bank; storing int8 cuts C write traffic to ¼ (the bottleneck for small K).
- **Standalone**: `mat_load x, y → SPAD`, `vec_run`, `mat_store` — no array
  involved.
- **Bias preload**: `mat_load bias → ACC`, then `mat_exec` with `accumulate`.

For data from DDR the three HP ports limit elementwise ops to ≈ 3 int16
elements/cycle, so VL matters most for fused ops and multi-pass work on data
that is already on chip.

### 6.5 fp32 vector engine *(L2)*

`rtl/sysarray/sa_vefp.v` sits next to the integer engine. `sa_unit` routes a
VE command to it when FLAGS bit 0 (FP) is set or when op = 6 (TRANSPOSE). The
two engines share the memory ports (ORed; only one is busy at a time) and the
VE performance events.
- **Numerics**: IEEE fp32 with round to nearest even, flush to zero, and a
  canonical NaN. Bit-exact with `llm/fp32.py` and `llm/sfu.py` through the
  functional simulator `llm/sa_funcsim.py`.
- **Per group**: `y = FUNC(OP(src1, src2) × A + B)`, then RELU and the VALID
  mask, then an output conversion (F32 / I32 to ACC, I8 to SPAD) or a row
  reduction (SUM / MAX, one broadcast word per row).
- **Index modes**: LIN, MOD P, DIV P, and IMM for src2; SWAPNEG for RoPE.
- **Lane folding**: FL = D / 2 physical fp lanes. A group passes through as two
  halves. The full-width engine did not fit next to the rest of the unit.
- **Two modes**: a stream mode for element-wise work, and a microsequencer
  for the special functions and reductions.

Encoding: descriptor w3[63:53], w5[63:32], w6, w7 and `vec_cfg` keys 10–15
(LLM plan §6.6). Implementation record and measurements: LLM plan §6.9.

### 6.4 Implementation *(M3)*

- **Types are tied to memories**: int32 lives only in ACC (one word per
  group of D), int8 only in SPAD (one word), int16 only in SPAD (two words:
  elements 0..D/2-1, then D/2..D-1). Any SPAD may hold either operand or the
  result; src1 and src2 have the same input type.
- **Pipeline**: a read unit issues one local-memory read per cycle (the
  src1 word(s), then the src2 word(s); a broadcast src2 is read once per
  command), four compute stages (op; RELU + REQUANT multiply; rounding
  shift; zero point + clamp + saturate + pack), an 8-entry output FIFO and a
  write stage (one word per cycle, priority over reads on the same memory).
  A credit counter keeps at most 8 groups between read and write, so the
  FIFO cannot overflow. Throughput: 1 group per max(reads + writes on the
  busiest port) cycles — the fused int32 → int8 epilogue with a periodic
  bias runs at ≈ 2 cycles/group (64 groups in 136 cycles).
- **DSPs**: 24 at D = 8 (16 × 16 MUL and 32 × 16 requant multiply per lane,
  the latter split by synthesis).
- **Scoreboard** (§7): VE reads banks(src1) ∪ banks(src2) and writes
  banks(dst). Besides RAW/WAR/WAW, EX and VE share the ACC and SPAD ports,
  so an EX and a VE command that touch any common bank never run together.
  Partial overlap of dst with a source inside one command is undefined
  (exact in-place, dst = src1, is allowed).
## 7. Ordering: bank scoreboard

Commands are dispatched **in program order** from the dispatch queue to the
engine queues. Each command declares the banks it reads and writes
(memory × bank, 6 entries). At dispatch, a command waits while any **other**
engine has an in-flight command with a conflicting access to one of its
banks:

| Hazard | Rule |
|---|---|
| RAW | reading bank X waits for in-flight writers of X on other engines |
| WAR | writing X waits for in-flight readers of X on other engines |
| WAW | writing X waits for in-flight writers of X on other engines |

**DDR is not tracked** *(M1)*: the scoreboard orders accesses to local
banks only. Software must `mat_fence` between a store and a later load of
the same DDR bytes (and before the ARM reads results).

Commands on the same engine execute in order, so they are not checked
against each other (back-to-back `mat_exec` overlap the drain of one tile
with the compute of the next).
Implementation (`sa_sched.v`, as built):
- Stage d2 computes each command's read and write bank masks (6 bits each)
  from its address ranges.
- Each engine keeps a mask FIFO of its in-flight commands: dispatched and
  not yet done, up to 8 entries (the 4-entry engine queue plus the running
  command). The entries are pushed at dispatch and popped when the engine
  reports done. Engines complete in order, so FIFO order is completion order.
- The OR of an engine's live entries is its current read set and write set.
- The head command dispatches only if its masks do not conflict with the
  sets of the *other* engines (RAW, WAR, WAW), and its engine queue has room.
- *(M3)* EX and VE share the memory ports of SPAD side B and ACC side A, so
  between these two any common bank conflicts, even two readers.

In-order dispatch makes it deadlock-free. Its cost is head-of-line blocking:
a blocked head also holds back later commands that could run. The
`HAZ_*` counters measure this (§10.4). In the LLM lists (L4 / L5), a VE
command waiting for its EX is the main cause.

Tracking is per bank, so two commands on disjoint addresses of the same
bank still wait for each other (a false dependency; see LLM plan §11.3 for
a finer-grained option). Software double-buffers by alternating banks:

```c
// C = A · B, M×K times K×N, D×D output tiles, one A strip resident per bank
cfg_load_interleave(D, K);             // A strips: D rows × K bytes
load(A strip 0 -> SPAD_A bank0);  load(B strip 0 -> SPAD_B bank0);
for (tile t = 0; t < tiles; t++) {
    int b = t & 1;
    if (t + 1 < tiles) { load(A strip t+1 -> bank !b);  load(B strip t+1 -> bank !b); }
    exec(SPAD_A bank b, SPAD_B bank b, ACC bank b, K/D);  // waits for its loads
    store(ACC bank b -> C[t]);                            // waits for its exec
}
fence(ALL);
```

## 8. Instruction set additions

All R-type in custom-0 (opcode 0x0B). funct7 = 0 is Phase 4, unchanged.
Commands are queued and the instruction returns; it stalls only when the
dispatch queue (8 entries) is full.

### 8.1 funct7 = 1: matrix / DMA

| funct3 | Mnemonic | rs1 | rs2 | rd |
|---|---|---|---|---|
| 0 | `mat_cfg` | key | value | — |
| 1 | `mat_load` | DDR address | destination LADDR | — |
| 2 | `mat_store` | DDR address | source LADDR | — |
| 3 | `mat_exec` | `{B word[31:16], A word[15:0]}` | `{flags[31:28], Kt[27:16], C word[15:0]}` | — |
| 4 | `mat_fence` | engine mask (bit0 LD, 1 ST, 2 EX, 3 VE; 0 = all) | — | extended status |
| 5 | `mat_perf` | read: counter index [4:0]; control: bit 31 = 1 | control: [0] clear, [1] enable | read: counter; control: number of counters |
| 6 | `mat_submit` | descriptor list address (64-byte aligned) | count (0 = until END) | — (§8.6) |
| 7 | `mat_notify` *(L1)* | — | — | — : one pulse on `notify_irq` (the host interrupt), `NOTIFY_COUNT` (CSR 0xD4) + 1; CAPS bit 22 |

`mat_exec` flags: bit28 accumulate. Kt: 1 .. 4095 k-tiles.

`mat_perf` exists only when CAPS bit 20 is set (`PERF = 1`); on older builds
the encoding is not answered and the core traps, so firmware checks CAPS
first (`sa_has_perf()`). The counters are also readable through the CSR
mirror (0x40 + 4·i, `PERF_CTRL` at 0x3C). Counter list: §10.4.

### 8.2 `mat_cfg` keys

| Key | Value |
|---|---|
| 0 LD_ROWS | rows |
| 1 LD_ROW_BYTES | bytes per row |
| 2 LD_PITCH | DDR pitch |
| 3 LD_MODE | bit0 INTERLEAVE, bit1 zero_pad |
| 4 ST_ROWS | rows |
| 5 ST_ROW_BYTES | bytes per row |
| 6 ST_PITCH | DDR pitch |
| 7 EX_REPEAT *(M2)* | output tiles per `mat_exec` (1 .. 4095, default 1) |
| 8 EX_B_STEP *(M2)* | SPAD_B words between the tiles' B strips |
| 9 EX_C_STEP *(M2)* | ACC words between tiles |
| 10 EX_C_ROW *(M2)* | ACC words between rows of a tile (default 1) |
| 11–26 BASE0–15 *(descriptor DMA, L1)* | descriptor relocation bases (the fetch unit's registers; keys 11–14 = BASE0–3 since M5) |
| 27–34 PARAM0–7 *(L1)* | descriptor parameters (dynamic fields, LOOP_END, LDPARAM) |

Configuration is copied into each command when it is queued, so software
may change it immediately after issuing.

### 8.3 Extended status (`mat_fence` rd)

| Bits | Field |
|---|---|
| 0 | all engines idle and queues empty |
| 1 | sticky error |
| 7:4 | engine busy (LD, ST, EX, VE) |
| 11:8 | error code (1 address/shape, 2 local range, 3 read response, 4 write response) |
| 15:12 | engine that raised the error |
| 20:16 | dispatch queue occupancy |
| 24 | a descriptor list is running (§8.6) |

### 8.4 funct7 = 2: vector

| funct3 | Mnemonic | rs1 | rs2 |
|---|---|---|---|
| 0 | `vec_cfg` | key | value |
| 1 | `vec_run` | src1 LADDR | src2 LADDR |

`vec_cfg` keys: V_OP, V_LEN (elements), V_DST (LADDR), V_TYPES (in/out),
V_SRC2_STRIDE (0 = broadcast one word), V_SCALE, V_SHIFT, V_ZP, V_CLAMP.
Captured at queue time like `mat_cfg`.

*(M3)* As built (`firmware/include/sysarray_intrinsics.h`):

| key | name | value |
|---|---|---|
| 0 | OP | `op[2:0]` (ADD 0, SUB 1, MUL 2, MAX 3, MIN 4, COPY 5), bit 4 RELU, bit 5 REQUANT |
| 1 | LEN | elements, a multiple of D (stored as groups) |
| 2 | DST | LADDR |
| 3 | TYPES | `in[1:0] | out[3:2]` (int8 0, int16 1, int32 2) |
| 4 | SRC2_MOD | src2 period in groups (0 elementwise, 1 broadcast) |
| 5–7 | SCALE, SHIFT, ZP | REQUANT (int16, 0–31, int32) |
| 8, 9 | CLAMP_LO, CLAMP_HI | int32 clamp window (reset: full range) |

Validation at dispatch (sticky error, like §5.4): zero groups, a type > 2,
op > COPY, MUL on int32, a type in the wrong memory, or a range past the end
of its memory. The queue packet is 261 bits wide (`SA_PKT_W`).

*(L2)* Keys 10–15 hold the fp32 engine's fields: FLAGS (FP, FUNC, M1, M2,
REDUCE, SWAPNEG), IMM, A, B, ROWLEN / VALID, P1 / S. TYPES gains VT_F32 = 3
(ACC only) and a src2 type in bits [5:4]. Op 6 is TRANSPOSE. The packet
grows to 432 bits.

### 8.5 Compatibility

- **CSRs and funct7 = 0** (`mat_trigger`, `mat_status`, `mat_reset`,
  `mat_wait`, `mat_cycles`) keep their encodings and behavior. A sequencer
  turns an 8×8×8 job into LD A, LD B, EX (Kt = 1), ST C using a reserved
  last tile slot in each memory (zero-padded on the D = 16 array).
  `firmware/matmul`, `firmware/matmul_insn` and `driver/pynq_matmul.py` run
  unchanged; their Phase 3/4 board numbers are the regression baseline.
- `mat_reset` also clears the sticky error of §5.4.
- *(M4)* At D = 16 an 8×8×8 job loads its 8-byte rows into the low half of
  the reserved slot; the rest of the slot must be zero. The RAMs power up
  zero (`sa_tdpram` initial values) and only the sequencer writes the slot,
  so the padding holds as long as new-ISA commands keep off the last D words.
- *(M4)* After a failed job command the sequencer waits for all engines to go
  idle before clearing the sticky error (another command of the same job
  could still be running and set it again; seen at D = 16).

### 8.6 Descriptor lists *(descriptor DMA)*

`rtl/sysarray/sa_cmdfetch.v`; plan and rationale in
[`perf_counters_and_desc_dma_plan.md`](perf_counters_and_desc_dma_plan.md) part 2,
decided by the counters of §10.4 (small operations are front-end bound).
The ARM builds a list in DDR (`driver/pynq_matmul.py` `DescList`,
`build_gemm_list`, `build_vector_list`); `firmware/desc_run` submits it with one
`mat_submit`; the fetch unit reads, decodes and feeds the commands to the
scheduler in list order, exactly like PCPI-issued commands (both use the
`pkt_*` functions of `sa_defs.vh`).

**Format**: 64 bytes = 8 little-endian 64-bit words `w0..w7`, 64-byte aligned.

| Word | LD / ST | EX | VE | FENCE / JUMP / END |
|---|---|---|---|---|
| w0 | header | header | header | header |
| w1 | [31:0] DDR address (or offset, RELOC) | [15:0] A, [31:16] B, [47:32] C, [59:48] Kt, [60] accumulate | [31:0] src1, [63:32] src2 (LADDR) | [3:0] engine mask / [31:0] target / [31:0] status |
| w2 | [31:0] LADDR, [47:32] rows, [63:48] row bytes | [11:0] repeat (0 = 1), [31:16] B step, [47:32] C step, [63:48] C row (0 = 1) | [31:0] dst LADDR, [63:32] length (elements) | — |
| w3 | [31:0] pitch, [33:32] mode (LD) | — | [7:0] op byte, [13:8] types, [31:16] period, [47:32] scale, [52:48] shift | — |
| w4 | — | — | [31:0] zero point, [63:32] clamp lo | — |
| w5 | — | — | [31:0] clamp hi | — |

Header `w0`: [7:0] opcode (0x01 LD, 0x02 ST, 0x03 EX, 0x04 VE, 0x10 FENCE,
0x11 JUMP, 0x12 END; **0x00 is invalid**, so zeroed memory stops the list),
[8] RELOC (DDR address += BASE[[10:9]], `mat_cfg` keys 11..14), [11]
FENCE_BEFORE, [12] IRQ (reserved, ignored), [31:13] must be 0, [63:32] tag
(free; the builders store the index). Every command carries its whole
configuration, no `mat_cfg` state is used.

**Semantics**

- One list at a time; `mat_submit` returns at once (it waits only while a
  previous list runs). Queued PCPI commands issued after it wait until the
  list is done, so mixed programs keep their program order; `mat_fence`
  also waits for the list.
- FENCE: no further command is fed until the scheduler queue is empty and
  the engines in the mask (0 = all) are idle. FENCE_BEFORE does the same
  for all engines before its command. DDR is still not tracked: a store and
  a later load of the same bytes need one of the two.
- JUMP continues at another list address; END records its status value and
  counts a finished list; a list also ends after `count` descriptors.
- Errors (sticky, engine 4): misaligned list or jump address, invalid
  opcode or reserved header bits → code 1; read response error on a fetch
  → code 3. A decoded command that the scheduler rejects reports as usual
  (its own engine). `mat_reset` recovers; the unit drops the bursts still
  in flight before it goes idle.
- Status CSRs (read-only, RISC-V 0x80000000 +): 0xC0 last decoded
  descriptor address, 0xC4 lists finished (END), 0xC8 last END value, 0xCC
  index of a failing descriptor, 0xD0 descriptors decoded in the current
  list. CAPS bit 21 = present.

*(L1)* Compiler-facing extensions (CAPS bit 24; LLM plan §5.3, appendix A):
- BASESEL widens to `{w0[14:13], w0[10:9]}` (BASE0–15);
- w0[31:16] holds two dynamic slots, each replacing or adding a PARAM into a
  numbered field;
- new opcodes: 0x13 LOOP_END (two nested levels, two PARAMs stepped per
  iteration), 0x14 SETREG, 0x15 CALL / 0x16 RET (depth 4), 0x17 LDPARAM
  (`PARAM = mem32 × mul + add`, read at fetch time);
- JUMP w2[0] = relative.

The END IRQ bit is still unused: the host is notified through `mat_notify`
from the runtime firmware. *(L2)* VE descriptors use w3[63:53] and w5–w7.

**Implementation**: descriptors are prefetched with one 8-beat burst each,
up to four in flight, into a 32 × 64-bit LUTRAM FIFO (no BRAM), assembled
and decoded one at a time. The fetch unit shares DMA port 0 with the LD
engine: an AR grant is held until its handshake (the fetch unit wins a new
grant), and an owner FIFO in AR order routes the returning beats (no AXI
IDs, bursts return in order). Performance counters 28–31: descriptors
decoded, command waiting for the scheduler, waiting for descriptor words,
fetch AR stalled.

## 9. Block design changes

- `pico_processor.tcl`: program BRAM window 64 KB → 8 KB (BRAM 16 → 2).
- `pico_bit.tcl`: enable S_AXI_HP1 and S_AXI_HP3 (HP2 already on), all on
  `riscv_clk`; three `axi_protocol_converter`s; `matmul_0/m_axi0..2` →
  HP1/HP2/HP3, each mapped 0x0000_0000–0x1FFF_FFFF.
- PCPI and the CSR window at 0x8000_0000 unchanged; ACC/SPAD are not
  memory-mapped to the CPU (only reachable through the engines).
- *(As built, M4)* One DMA port (HP2) only; the three-port option was
  dropped (§10.3), so HP1 / HP3 stay off.
- *(L1)* `irqConcat` gets a second input: `In1` = `matmul_0/notify_irq`
  (rising edge) into the AXI interrupt controller, next to the PicoRV32
  trap on `In0`.

## 10. Expected performance

Model: output-stationary tiling with double buffering, time =
max(compute, per-port traffic, DDR total); 50 MHz; HP port ≈ 6 B/cycle
(75 % of 64-bit), DDR ≈ 29 B/cycle total. MAC/cycle, peak D²:

| Workload | Today (measured) | D = 8, 1 port | D = 8, 3 ports | D = 16, 3 ports |
|---|---|---|---|---|
| independent 8×8×8 jobs | 1.6 | ≈ 8 | ≈ 23 | ≈ 11 (each job zero-padded to 16×16: 46 cycles) |
| GEMM 256³ | ≈ 1.6 (+ ARM sums partials) | 60.7 | 60.7 | 229 |
| GEMM 512³ | — | 62.3 | 62.3 | 242 |
| 1024×1024×64 (small K, C-write bound) | — | 52.5 | 52.5 | 174 |
| 16×4096×4096 (GEMV-like, B-read bound) | — | — | — | 254 |

- D = 8 is compute-bound for GEMM with any split of ≥ 32 BRAM; D = 16 needs
  the full 128 BRAM (C tiles up to 128×256) and the three ports.
- int8 output through the vector engine removes most of the small-K C
  traffic (not included above).
- The HP efficiency above is the design-time assumption (75 %); §10.1 has
  the measurement.

### 10.1 Measured DMA bandwidth *(M2)*

`firmware/bwtest` on the M1 bitstream (one port, HP2, 50 MHz; 64 KB per test):

| Transfer | B/cycle | MB/s | of 8 B/cycle |
|---|---|---|---|
| LD DDR → SPAD / ACC, contiguous | 7.94 | 397 | 99 % |
| ST SPAD / ACC → DDR | 7.91–7.93 | 396 | 99 % |
| LD, 8-byte rows (single-beat bursts) | 1.93 | 97 | 24 % |
| LD and ST at the same time | 15.76 | 788 | read + write both at 99 % |

One port is saturated by long bursts, and its read and write channels run
concurrently. The engines take one 64-bit beat per cycle, which equals one
port. Re-running the model with 8 B/cycle per direction: D = 8 is
compute-bound on every workload with one port; D = 16 is memory-bound only
for small K (1024×1024×64: 128 MAC/cycle), where the int32 C writes
dominate — fixed by int8 output (M3) or more write ports (M4 decision).
Single-beat bursts are latency-bound (≈ 16 cycles round trip, 4 in
flight); M2 raises the outstanding bursts per port from 4 to 8.

On the board, M1 GEMMs of size 64³ ran at 13 MAC/cycle against a model of
≈ 60: about 300 cycles per C tile went to PicoRV32 issuing commands (each
instruction is fetched over AXI) against 78 cycles of compute. M2's repeat
`mat_exec` issues 3 commands per A strip instead of 2+ per tile.

### 10.2 M2 on the board *(M2)*

| GEMM | M1 MAC/cycle | M2 MAC/cycle |
|---|---|---|
| 24×16×40 + bias | 4.3 | 7.7 |
| 64×64×64 | 13.0 | 38.3 |
| 128×128×128 | 28.4 | 49.8 |
| 32×256×64 + bias | 13.1 | 29.0 |
| 256×256×256 | 54.5 | 56.5 |

Single-beat LD bursts: 1.93 → 3.81 B/cycle with 8 outstanding bursts per
port (contiguous transfers unchanged at 99 %). What remains for small
problems is fixed cost (loading B, the first A strip, the final store and
fence) that later strips cannot hide.

### 10.3 M4 on the board, and the three-port decision *(M4)*

| GEMM | D = 8 (M3) | D = 16 (M4) | % of 256 |
|---|---|---|---|
| 64³ | 37.5 | 70.1 | 27 % |
| 128³ | 49.7 | 132.1 | 52 % |
| 256³ | 56.5 | 180.4 | 70 % |
| 512×256×256 | – | 191.1 | 75 % |
| 256×128×1024 | – | 180.2 | 70 % |

256³ in 93.0k cycles, against lower bounds of ≈ 73.5k for the array
(16 strips × 16 tiles × (K + 2(D-1) + 1)) and ≈ 48.6k for the DMA (A 64 KB +
B 64 KB + C 256 KB at 7.9 B/cycle; reads and writes overlap, LD + ST reach
15.76 B/cycle). One port is therefore **not** the bottleneck for GEMMs of this
size, and three ports would not remove the remaining ≈ 20k cycles, which
come from the schedule (estimated, not profiled):

- the resident-B load (64 KB ≈ 8.3k cycles) runs before the first exec;
- in-order dispatch: the firmware queues LD A(i), EX(i), ST(i) per strip, so
  LD A(i+1) waits behind ST(i), which waits for EX(i) - the next strip's A
  load (4 KB ≈ 520 cycles) does not overlap the current exec (16 × ≈ 600).

Decision: no three-port DMA. Small-K shapes (32×256×64: 17 %) are C-write
bound, which the int8 epilogue addresses (4× less C traffic) rather than
more ports.

The bullets above were estimates; §10.4 has the measured breakdown from the
performance counters.

**Firmware schedule tuning** (`firmware/gemm`, `notebooks/m4_sched_tune.py`,
same bitstream; MAC/cycle on the board, D = 16):

| GEMM | M4 | + A prefetch | + B split (forced) | both | default |
|---|---|---|---|---|---|
| 64³ | 64.1 | 63.0 | 48.4 | 47.3 | 63.0 |
| 128³ | 131.7 | 150.0 | 136.9 | **156.7** | 156.7 |
| 48×32×80 + bias | 32.9 | **34.0** | 27.2 | 26.0 | 34.0 |
| 32×256×64 + bias | 44.0 | **50.7** | 46.9 | 50.6 | 50.6 |
| 256³ | 180.3 | 197.2 | 184.8 | **202.6** | 202.6 (79 %) |
| 512×256×256 | 191.1 | 210.9 | 193.6 | **213.9** | 213.9 (84 %) |
| 256×128×1024 | 180.2 | 216.2 | 184.3 | **222.2** | 222.2 (87 %) |
| 256³ int8 + bias | 167.8 | **182.0** | 167.7 | 181.9 | 181.9 |

- *A prefetch*: queue EX(i), LD A(i+1), ST(i) - the next A strip loads during
  EX(i) instead of queueing behind ST(i). +9-20 % on 128³ and up.
- *B split*: the scoreboard tracks banks, so B's column halves go to the two
  SPAD_B banks (loaded half 0, A(0), half 1) and each strip runs two execs;
  the first exec starts after half of B. +3 % on top of the prefetch for
  large B, but -25 % at 64³ (two short execs per strip), so the firmware
  splits only when B >= 16 KB ("default" column; the table's default values
  are the measured cells that rule selects).

### 10.4 Measured cycle breakdown (performance counters)

`rtl/sysarray/sa_perf.v` counts 28 events taken from existing signals of the
scheduler, the PCPI front end and the four engines (definitions: plan §1.2,
`PC_*` in `sa_defs.vh`). The firmware opens the counting window around its
`rdcycle` window and copies the counters to the program BRAM below the
mailbox; `notebooks/m4_perf.py` checks them against exact invariants (tiles,
useful steps × D² = M·N·K, bytes stored, commands) and prints the breakdown.
Board, D = 16, tuned firmware schedule, 69/69 checks passing
(`RISCV-on-PYNQ-Z1/bitstreams/m4p/`):

| Workload | cycles | EX useful | EX fill | head: EX blocked | head: VE blocked | starve + all idle | CPU on full queue |
|---|---|---|---|---|---|---|---|
| 64³ | 4,154 | 23.9 % | 11.2 % | 0.0 % | – | **86.8 %** | 0.0 % |
| 128³ | 13,383 | 60.6 % | 14.2 % | 4.5 % | – | 23.2 % | 0.0 % |
| 32×256×64 | 7,409 | 27.2 % | 12.7 % | 6.4 % | – | **61.8 %** | 0.0 % |
| 256³ | 82,803 | **79.0 %** | 9.3 % | 8.8 % | – | 4.7 % | 61.3 % |
| 512×256×256 | 156,863 | **83.5 %** | 9.8 % | 4.6 % | – | 2.5 % | 70.0 % |
| 256×128×1024 | 151,022 | **86.7 %** | 2.5 % | 11.2 % | – | 1.9 % | 71.8 % |
| 256³ int8 | 92,216 | 71.0 % | 8.3 % | 7.4 % | 80.1 % | 3.4 % | 67.7 % |
| vector int8 add, 4096 | 3,388 | – | – | – | 0.4 % | **93.9 %** | 0.0 % |
| vector int16 mul, 4096 | 6,448 | – | – | – | 15.4 % | 68.8 % | 0.0 % |
| vector int32 add, 16384 | 24,948 | – | – | – | 61.2 % | 26.5 % | 0.0 % |

Percentages are of the counter window (`CYCLES`), which is the firmware's
`rdcycle` window plus a constant 131 (GEMM) / 175 (vector) cycles: the few
instructions between `mat_perf` and `rdcycle` at both ends, fetched over the
AXI interconnect at ~10 cycles each.

What the numbers say:

- **Large GEMMs are array-bound.** The array steps 88–89 % of the time
  (useful + fill). The fill is structural: every tile spends 2(D−1) = 30 of
  its K + 30 steps in the operand skew (≈ 10 % at K = 256, 2.5 % at
  K = 1024). The rest is the array waiting at the head for a bank
  (`HAZ_EX` 5–11 %). "Head blocked by ST" (86–93 %) is the normal state of the
  prefetch schedule: ST(i) sits at the head waiting for EX(i) while the
  commands behind it already run.
- **The front end is not the limit for large GEMMs.** The CPU spends
  61–72 % of the window stalled on a full queue; the head is empty only
  2–4 % of the time.
- **Small GEMMs and vector operations are front-end bound.** The head is
  empty (starve) or everything is idle 60–94 % of the time: the hardware
  waits for PicoRV32 to issue the next command.
- **The DMA is efficient whenever it runs**: 7.4–7.97 B/cycle while busy,
  close to the 8 B/cycle port limit (confirms §10.3: no three ports).
- **int8 GEMM**: the VE command waits at the head 80 % of the time (RAW on the
  strip's EX) while the VE itself is active only 9 %; the per-strip chain
  EX → VE → ST costs the ≈ 10 % against int32 output.

**Descriptor DMA decision (plan D0):** worth building for small operations
only. It would remove most of the 60–94 % front-end time of 64³-class GEMMs
and short vector operations (estimate for 64³: ≈ 4.2k → ≈ 2.5k cycles, then
bound by the 16 KB C store and compute), and does nothing for large GEMMs,
whose front end already runs ahead.

### 10.5 Descriptor lists on the board *(descriptor DMA)*

The same commands issued by the PCPI firmware and as one ARM-built list
(`notebooks/m5_desc_demo.py`, `RISCV-on-PYNQ-Z1/bitstreams/m5/`; D = 16, cycles
from the first command / `mat_submit` to the end of `mat_fence`, all
bit-exact):

| Work | PCPI | list | speedup | front end idle (PCPI → list) |
|---|---|---|---|---|
| GEMM 16³ | 1,801 | 430 | **4.19×** | 99.4 % → 62.9 % |
| GEMM 64³ | 4,154 | 3,383 | 1.23× | 86.8 % → 24.3 % |
| int8 GEMM 64³ | 4,984 | 3,288 | 1.52× | 68.3 % → 10.6 % |
| GEMM 128³ / 256³ | 13,384 / 82,798 | 13,366 / 82,784 | 1.00× | 23 % / 5 % → 9 % / 3 % |
| vector int8 add, 4096 | 3,376 | 2,224 | 1.52× | |
| vector int16 max (broadcast), 2048 | 3,801 | 1,460 | **2.60×** | |
| vector int32 → int8 + bias, 8192 | 8,163 | 6,449 | 1.27× | |
| vector int32 add, 16384 | 24,952 | 23,806 | 1.05× | |

This matches the decision of §10.4: work whose head was empty most of the
time gains, large GEMMs (front end already ahead) do not. The gain for 64³
(1.23×) is smaller than the ≈ 1.6× estimated from the counters: with a list
the head is still empty 24 % of the time — the fetch unit decodes one
descriptor per ≥ 9 cycles and each descriptor waits for its 8-beat fetch.
The list cycles exclude building the list on the ARM, which is done once
and can be reused with the relocation bases.

## 11. Resource estimate

Baseline: Phase 4 board build (6640 LUT, 7421 FF, 64 DSP, 16 BRAM), of which
the Phase 4 unit is 1545 LUT / 2116 FF / 64 DSP.

| | LUT (53.2k) | FF (106.4k) | DSP (220) | BRAM36 (140) |
|---|---|---|---|---|
| D = 8, VL = 8 | ≈ 14k (26 %) | ≈ 17k (16 %) | 80 (36 %) | 130 (93 %) |
| D = 16, VL = 16, 8 DSP columns | ≈ 28k (53 %) | ≈ 30k (28 %) | 160 (73 %) | 130 (93 %) |

*(M3)* Measured OOC (D = 8, NPORTS = 1, with the VE): 16.4k LUT (VE 8.1k:
eight lanes of saturating add, 48-bit rounding shift and two clamps),
11.6k FF, 96 DSP (array 64, VE 24), 128 BRAM36, WNS +2.5 ns at 50 MHz. The
VE is larger than estimated; LUTs stay at 31 %.

*(M4)* Board build at D = 16 (8 DSP columns + 8 LUT columns, VL = 16, one
port): 47.6k LUT (89.5 %; array 21.3k, VE 14.5k in OOC), 34.2k FF, 184 DSP
(84 %; array 128, VE 48), 130 BRAM36, WNS +1.3 ns. LUTs are the tight
resource: more logic would need 10 DSP columns (≈ 5k LUT freed, 216 DSP).
The module reference freezes derived parameter defaults, so the block design
sets `SPAD_WORDS` / `ACC_WORDS` together with `D` (`-sa_d`).
With the performance counters (`bitstreams/m4p`): 48.3k LUT (90.7 %; +0.65k),
35.1k FF, same DSP / BRAM, WNS +1.9 ns. With the descriptor fetch unit as
well (`bitstreams/m5`): 48.9k LUT (92.0 %; +0.66k), 36.4k FF, same DSP / BRAM,
WNS +2.5 ns.

*(L0–L2, D = 8)* The LLM builds go back to D = 8 to make room:
- `bitstreams/l0`: 23.5k LUT, WNS +1.56 ns.
- `bitstreams/l1`: command extensions and notify, 27.1k LUT (51 %), WNS +2.35 ns.
- `bitstreams/l2`: fp32 VE, 44.0k LUT (82.7 %), 33.6k FF, 116 DSP, 130 BRAM36,
  WNS +1.84 ns. In OOC, `sa_vefp` is 17.2k LUT with FL = 4 lanes; with 8 lanes
  it was 28.9k and the unit did not place.

Everything runs at 50 MHz; the Phase 2 LUT-PE array alone reached ≈ 99 MHz,
so timing margin is large. BRAM has no room for an ILA without shrinking
the memory parameters.

## 12. Milestones

Each milestone: unit + system simulation, board test script, Phase 3/4
regression, then the tested bitstream is committed under
`RISCV-on-PYNQ-Z1/bitstreams/`.

| | Scope | Acceptance |
|---|---|---|
| **M1** | D = 8: SPAD/ACC, LD/ST with one port (HP2), K-streaming EX, scoreboard, funct7 = 1 ISA, legacy sequencer, 8 KB program BRAM | NumPy-exact GEMMs (several shapes incl. non-square, K ≫ D) with double-buffered firmware; legacy tests pass; cycles measured *(done, board-verified)* |
| **M2** *(redefined)* | DMA bandwidth self-test (§10.1); repeat `mat_exec` + strip-wide `mat_store`; 8 outstanding bursts per port | Measured HP efficiency; same tests incl. repeat commands; GEMM command overhead reduced on the board *(done, board-verified)* |
| **M3** | Vector engine VL = D, funct7 = 2 | Fused GEMM + bias + ReLU + requant and standalone elementwise ops exact vs NumPy *(done, board-verified)* |
| **M4** | D = 16 build (8 DSP columns, VL = 16); three HP ports with striping if the D = 16 measurements need them (§10.1) | All of the above at D = 16; resource/timing report *(done, board-verified; three ports not needed, §10.3)* |
| **M4 + perf** | Performance counters (`sa_perf.v`, `mat_perf`); firmware schedule tuning | Counters checked against exact invariants on the board; measured cycle breakdown (§10.4) *(done, `bitstreams/m4p`)* |
| **M5** | Descriptor DMA (`sa_cmdfetch.v`, `mat_submit`, ARM-built lists) | Same results as the PCPI path; front-end-bound work faster, nothing slower (§10.5) *(done, `bitstreams/m5`)* |
| **L0–L5b** | LLM inference on this unit (D = 8): command ring and extensions (L1), fp32 VE (L2), W8A8 linears (L3), the whole decoder as one list (L4), the ARM runtime (L5), batched decode (L5b) | [`llm_inference_plan.md`](llm_inference_plan.md) §13 *(done, `bitstreams/l1`, `l2`)* |

## 13. Verification

- Unit testbenches per engine: DMA against an AXI memory model with random
  stalls, SLVERR injection and per-port protocol checks (length, 4 KB,
  WLAST, WSTRB); EX against NumPy for random (Kt, layout) combinations;
  VE per operation incl. saturation and rounding edge cases.
- Scoreboard: randomized command streams checked against a sequential
  reference model (same final memory contents as in-order execution).
- System: firmware on `picorv32_axi` against the full unit (as today).
- Board: GEMM, fused and vector demos vs NumPy; DMA bandwidth self-test;
  Phase 3/4 demo scripts unchanged.
- *(As built.)* Every testbench runs at D = 8 and 16 (`make test`,
  `make test16`). Performance counters: exact invariants (tiles, useful
  steps × D² = M·N·K, bytes stored, commands) in `tb_sa_unit`, `tb_system`
  and on the board (`m4_perf.py`). Descriptor lists: a GEMM and the random
  stream as lists against the same reference model, directed tests
  (ordering, FENCE, JUMP, END, count, relocation, errors), and a
  co-simulation in which lists built by the Python driver run on the RTL
  (`firmware/desc_run/gen_desc_cases.py`). Mutation checks confirm the
  tests catch deliberately broken scoreboard, VE, counter and fetch logic
  (`rtl/sysarray/README.md`).

## 14. Risks and open points

- **HP efficiency and DDR sharing**: the model assumed 75 % per port; M2
  measured 99 % (7.94 B/cycle) on an idle board. Heavy Linux traffic on
  the shared 16-bit DDR3 would lower it.
- **BRAM at 93 %**: placement is fine at 50 MHz, but debugging with an ILA
  needs a reduced-memory build.
- **LUT at 92 %** (M5 build, D = 16): further logic needs `DSP_COLS = 10`
  (≈ 5k LUT freed, 216 of 220 DSP) or a slimmer VE. *(L2)* The LLM build is
  D = 8 at 82.7 % with the fp32 VE. D = 16 with the fp32 VE would not fit.
- **Complexity**: the scoreboard and multi-port DMA were the risky parts;
  both got randomized tests against reference models before integration
  (the multi-port DMA is tested at `NPORTS = 3` but not built).
- **DDR is not tracked**: software must fence between a store and a later
  load of the same bytes (`mat_fence`, FENCE descriptor); a future compiler
  has to emit these.
- **Legacy sequencer** reserves one tile slot per memory; mixing legacy and
  new commands is allowed but documented as sequential (it fences first).

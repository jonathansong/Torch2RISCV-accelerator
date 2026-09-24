# Double-buffered accelerator design (v2 of the matmul unit)

Status: **M1 done and verified on the board** (`rtl/sysarray`, D = 8, one
DMA port; bitstream and results in `RISCV-on-PYNQ-Z1/bitstreams/m1/`:
256×256×256 at 54.5 MAC/cycle, 85 % of the array peak, 35× Phase 4).
Notes from the implementation are marked *(M1)*. **M2 was redefined from
measurements** (§10.1): one HP port already runs at 99 % of its 8 B/cycle,
so M2 cuts command-issue overhead instead (repeat `mat_exec`, §4.1) and the
three DMA ports move to M4. Changes are marked *(M2)*. **M2 done and verified
on the board** (`RISCV-on-PYNQ-Z1/bitstreams/m2/`): 64³ 13.0 → 38.3 MAC/cycle,
128³ 28.4 → 49.8, 256³ 56.5 (88 % of peak). Next: M3 (vector engine).

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
  issues LD/EX/ST commands (§8.4).

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
Implementation: per bank one reader count and one writer count, each tagged
with the engine; updated at dispatch and completion. In-order dispatch makes
it deadlock-free. Software double-buffers by alternating banks:

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

`mat_exec` flags: bit28 accumulate. Kt: 1 .. 4095 k-tiles.

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
| 23:16 | dispatch queue occupancy |

### 8.4 funct7 = 2: vector

| funct3 | Mnemonic | rs1 | rs2 |
|---|---|---|---|
| 0 | `vec_cfg` | key | value |
| 1 | `vec_run` | src1 LADDR | src2 LADDR |

`vec_cfg` keys: V_OP, V_LEN (elements), V_DST (LADDR), V_TYPES (in/out),
V_SRC2_STRIDE (0 = broadcast one word), V_SCALE, V_SHIFT, V_ZP, V_CLAMP.
Captured at queue time like `mat_cfg`.

### 8.5 Compatibility

- **CSRs and funct7 = 0** (`mat_trigger`, `mat_status`, `mat_reset`,
  `mat_wait`, `mat_cycles`) keep their encodings and behavior. A sequencer
  turns an 8×8×8 job into LD A, LD B, EX (Kt = 1), ST C using a reserved
  last tile slot in each memory (zero-padded on the D = 16 array).
  `firmware/matmul`, `firmware/matmul_insn` and `driver/pynq_matmul.py` run
  unchanged; their Phase 3/4 board numbers are the regression baseline.
- `mat_reset` also clears the sticky error of §5.4.

## 9. Block design changes

- `pico_processor.tcl`: program BRAM window 64 KB → 8 KB (BRAM 16 → 2).
- `pico_bit.tcl`: enable S_AXI_HP1 and S_AXI_HP3 (HP2 already on), all on
  `riscv_clk`; three `axi_protocol_converter`s; `matmul_0/m_axi0..2` →
  HP1/HP2/HP3, each mapped 0x0000_0000–0x1FFF_FFFF.
- PCPI and the CSR window at 0x8000_0000 unchanged; ACC/SPAD are not
  memory-mapped to the CPU (only reachable through the engines).

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

## 11. Resource estimate

Baseline: Phase 4 board build (6640 LUT, 7421 FF, 64 DSP, 16 BRAM), of which
the Phase 4 unit is 1545 LUT / 2116 FF / 64 DSP.

| | LUT (53.2k) | FF (106.4k) | DSP (220) | BRAM36 (140) |
|---|---|---|---|---|
| D = 8, VL = 8 | ≈ 14k (26 %) | ≈ 17k (16 %) | 80 (36 %) | 130 (93 %) |
| D = 16, VL = 16, 8 DSP columns | ≈ 28k (53 %) | ≈ 30k (28 %) | 160 (73 %) | 130 (93 %) |

Everything runs at 50 MHz; the Phase 2 LUT-PE array alone reached ≈ 99 MHz,
so timing margin is large. BRAM has no room for an ILA without shrinking
the memory parameters.

## 12. Milestones

Each milestone: unit + system simulation, board test script, Phase 3/4
regression, then the tested bitstream is committed under
`RISCV-on-PYNQ-Z1/bitstreams/`.

| | Scope | Acceptance |
|---|---|---|
| **M1** | D = 8: SPAD/ACC, LD/ST with one port (HP2), K-streaming EX, scoreboard, funct7 = 1 ISA, legacy sequencer, 8 KB program BRAM | NumPy-exact GEMMs (several shapes incl. non-square, K ≫ D) with double-buffered firmware; legacy tests pass; cycles measured |
| **M2** *(redefined)* | DMA bandwidth self-test (§10.1); repeat `mat_exec` + strip-wide `mat_store`; 8 outstanding bursts per port | Measured HP efficiency; same tests incl. repeat commands; GEMM command overhead reduced on the board |
| **M3** | Vector engine VL = D, funct7 = 2 | Fused GEMM + bias + ReLU + requant and standalone elementwise ops exact vs NumPy |
| **M4** | D = 16 build (8 DSP columns, VL = 16); three HP ports with striping if the D = 16 measurements need them (§10.1) | All of the above at D = 16; resource/timing report |

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

## 14. Risks and open points

- **HP efficiency and DDR sharing**: the model assumes 75 % per port;
  Linux traffic on the shared 16-bit DDR3 lowers it. M2 measures it.
- **BRAM at 93 %**: placement is fine at 50 MHz, but debugging with an ILA
  needs a reduced-memory build.
- **Complexity**: the scoreboard and multi-port DMA are the risky parts;
  both get randomized tests against reference models before integration.
- **Legacy sequencer** reserves one tile slot per memory; mixing legacy and
  new commands is allowed but documented as sequential (it fences first).

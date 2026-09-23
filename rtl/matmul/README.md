# matmul_unit — 8×8×8 int8 matrix multiply (Phase 2)

Loosely-coupled accelerator: software programs a few AXI4-Lite CSRs, the unit
fetches A and B from DDR through its own AXI4 master, computes
`C = A · B` on an 8×8 output-stationary systolic array (int8 × int8 → int32),
and writes C back to DDR.

```
            s_axi (AXI4-Lite CSR)           m_axi (AXI4, 64-bit) ──► DDR (PS7 S_AXI_HP*)
                    │                               ▲
              ┌─────▼─────┐   A/B rows   ┌──────────┴─────────┐
              │ CSR + FSM ├─────────────►│ abuf / bbuf (64 B) │
              └─────┬─────┘              └──────────┬─────────┘
                    │ start/en                      │ skewed feed
                    │                  ┌────────────▼────────────┐
                    └─────────────────►│ 8×8 PE array (64 DSP48) │── C (int32) ──► m_axi W
                                       └─────────────────────────┘
```

| File | Contents |
|---|---|
| `systolic_array.v` | `systolic_pe` (MAC, forwards a → right, b → down) and the N×N array |
| `matmul_unit.v` | CSRs, control FSM, AXI4 master, input skew |
| `sim/gen_vectors.py` | random + corner-case vectors, NumPy golden model |
| `sim/tb_matmul_unit.v` | xsim testbench (AXI memory model, CSR master) |
| `synth_ooc.tcl` | out-of-context synth/place/route for resource/timing numbers |

## CSR map

| Offset | Name | Access | Description |
|---|---|---|---|
| 0x00 | CTRL | RW | bit0 start (write 1, reads 0) · bit1 soft reset (clears STATUS / IRQ_STATUS, reads 0) · bit2 irq_enable |
| 0x04 | STATUS | RO | bit0 done · bit1 busy · bit2 error · bits[11:8] error code |
| 0x08 | SRC_A_ADDR | RW | DDR address of A |
| 0x0C | SRC_B_ADDR | RW | DDR address of B |
| 0x10 | DST_ADDR | RW | DDR address of C |
| 0x14 | DIM_M_N_K | RW | M = [9:0], N = [19:10], K = [29:20]; resets to 8×8×8, only 8×8×8 is accepted |
| 0x18 | IRQ_STATUS | RW1C | bit0 set on every completion (also on error), write 1 to clear |
| 0x1C | CYCLES | RO | cycles from start to done of the last run |
| 0x20 | ID | RO | `0x4D4D3038` ("MM08") |

- `irq = IRQ_STATUS[0] & irq_enable`. IRQ_STATUS latches even when irq_enable is 0, so clear it before enabling the interrupt.
- Writes to SRC/DST/DIM and start/soft reset are ignored while busy.
- Error codes: 1 DIM is not 8×8×8 · 2 address misaligned or crossing 4 KB · 3 read response SLVERR/DECERR (C is not written) · 4 write response SLVERR/DECERR. An error still sets done and IRQ_STATUS, so polling loops terminate.
- CSR writes are full 32-bit (WSTRB is ignored).

## Memory layout

All row-major, little-endian. Addresses must be 8-byte aligned and the buffer
must not cross a 4 KB page (true for `pynq.allocate` buffers of this size).

| Buffer | Size | Bus traffic |
|---|---|---|
| A (8×8 int8) | 64 B | 1 burst × 8 beats, one row per 64-bit beat |
| B (8×8 int8) | 64 B | 1 burst × 8 beats |
| C (8×8 int32) | 256 B | 2 bursts × 16 beats, beat g = C elements 2g, 2g+1 |

Bursts are ≤ 16 beats and INCR, so the master can connect to the AXI3 HP
ports through a protocol converter/interconnect.

## Timing of one run

read A, read B → 22 compute cycles (`t = i + j + k`, last at 2·7 + 7) → write
C. In simulation with 0–3 cycle random AXI stalls: 143–169 cycles per
8×8×8 matmul (512 MACs).

## Verify / build

```sh
make sim                 # PYTHON=... if python3 lacks numpy
make synth               # CLOCK_NS=10.0 by default
```

`make sim` runs 32 golden cases (zeros, identity, ±128/127 extremes, patterns,
24 random) plus CSR, interrupt, busy-write, soft-reset and all four error
paths, with protocol checks on every burst (length, INCR, 8-byte size,
4 KB, WSTRB, WLAST). Mutation checks (unsigned B, one compute step short,
early WLAST) all make it fail.

Out-of-context results on xc7z020clg400-1 (Vivado 2024.1):

| LUT | FF | DSP48E1 | BRAM | 100 MHz WNS |
|---|---|---|---|---|
| 1279 (2.4 %) | 2010 (1.9 %) | 64 (29 %) | 0 | +0.450 ns (≈105 MHz) |

Without the `use_dsp` attribute on `systolic_pe`, Vivado builds the 8×8
multipliers from LUTs: 7178 LUTs, 0 DSPs, and 100 MHz fails by 0.15 ns.

## Not done yet (Phase 3)

Integration into the overlay block design (CSR slave on the PicoRV32 and ARM
buses, master on S_AXI_HP2), PicoRV32 firmware driver, board test.

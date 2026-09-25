# sysarray — double-buffered matrix accelerator

Implementation of [`docs/double_buffer_design.md`](../../docs/double_buffer_design.md).
Replaces `rtl/matmul` (Phase 2–4) in the overlay from M1 on; the old unit's
behavior (CSRs, funct7 = 0 instructions) is kept by `sa_legacy`.

| File | Contents |
|---|---|
| `sa_unit.v` | top level: CSR slave, 3 AXI4 masters (NPORTS used), PCPI, memories, engines |
| `sa_sched.v` | input queue → 2-stage decode (validation, bank masks) → bank scoreboard → engine queues |
| `sa_pcpi.v` | custom-0 decoder: funct7 = 0 (legacy), funct7 = 1 (`mat_cfg/load/store/exec/fence`), funct7 = 2 (`vec_cfg/run`) |
| `sa_legacy.v` | Phase 2–4 CSR map + CAPS/EXT_STATUS; sequencer turning an 8×8×8 job into LD/LD/EX/ST |
| `sa_ld.v`, `sa_st.v` | DMA engines: 2D shapes (LINEAR / INTERLEAVE), ≤ 16-beat 4 KB-safe bursts striped over NPORTS |
| `sa_ex.v` | EX engine: K-streaming feed (one SPAD_A + one SPAD_B word per cycle), shadow drain, accumulate, repeat (M2: several C tiles per command with B / C strides) |
| `sa_perf.v` | performance counters: 32 × 32-bit event counters (28 defined, `PC_*` in `sa_defs.vh`), read by `mat_perf` (funct7 = 1, funct3 = 5) or the CSR mirror 0x40 + 4·i, control via `mat_perf` or `PERF_CTRL` (0x3C); `PERF = 0` removes them ([plan](../../docs/perf_counters_and_desc_dma_plan.md), P1) |
| `sa_cmdfetch.v` | descriptor fetch unit: `mat_submit(list, count)` reads 64-byte descriptors from DDR (shares DMA port 0 with LD), decodes LD/ST/EX/VE into the same packets as PCPI (`pkt_*` in `sa_defs.vh`), handles FENCE / JUMP / END, relocation bases, errors (engine 4), status CSRs 0xC0–0xD0 ([design §8.6](../../docs/double_buffer_design.md)) |
| `sa_ve.v` | M3 vector engine: VL = D lanes, ADD/SUB/MUL/MAX/MIN/COPY + RELU/REQUANT/clamp, int8/int16 (SPAD) and int32 (ACC), src2 period (broadcast / bias vector) |
| `sa_array.v` | D×D PEs (`sa_pe_dsp` / `sa_pe_lut` per column), shadow accumulators |
| `sa_bankmem.v`, `sa_tdpram.v` | double-banked byte-write TDP memories (UG901 template → RAMB36) |
| `sa_defs.vh`, `sa_macros.vh` | memory ids, command packet layout, cfg keys, error codes |

## Verify / build

```sh
make test          # all unit testbenches at D = 8 (PYTHON=... with numpy for tb_sa_unit)
make test16        # the same at D = 16 (M4)
make synth         # OOC synth/place/route, 50 MHz (SYNTH_GEN="D=16 NPORTS=1")
```

Every testbench takes D as a top-level generic (`make sim TB=tb_sa_ve GEN=D=16`,
`GEN="NP=3 D=16"`); fixed word addresses are written relative to the bank
boundaries or scaled with the memory depth.

| Testbench | Covers |
|---|---|
| `tb_sa_ex` | K-streaming vs golden model: Kt 1–64, strips across banks, back-to-back commands, accumulate, extremes, repeat commands (row-major C strip, gapped B strips, bank crossing) |
| `tb_sa_dma` (NP = 1, 3) | LD/ST shapes, INTERLEAVE, 4 KB splitting, bank crossing, partial ACC words, SLVERR; byte-exact vs reference incl. neighbours |
| `tb_sa_unit` | Phase 2–4 tests (CSR + mat_trigger, 32 NumPy vectors, errors, irq); tiled GEMMs on the new ISA (repeat exec, strip-wide store); quantized GEMMs with the VE epilogue (bias vector or preloaded bias rows, RELU, REQUANT → int8); 120-command random stream incl. repeat EX and VE commands vs a sequential reference (scoreboard); new-ISA error paths; performance counters: exact invariants after GEMMs / an int8 GEMM (tiles, useful steps, beats, commands), a directed RAW hazard, frozen / cleared / CSR-mirror reads, `GEN=PERF=0`; descriptor lists: a GEMM and the whole random stream as lists (same reference model), PCPI/list ordering, FENCE / no FENCE / FENCE_BEFORE on a DDR dependency, JUMP, END, count, RELOC, error paths and recovery, FETCH_DESC |
| `tb_sa_ve` | VE alone: every op × type, RELU/REQUANT/clamp/saturation edges, broadcast and periodic src2, in-place, bank crossing, all operands in one memory; 200 random commands checked one by one, then 60 issued back to back |

Mutation checks (B byte off by one, accumulate dropped, extra k step, no
4 KB split, lane mapping, dropped SLVERR, scoreboard without hazard check or
without WAR, VE without RAW/WAR/WAW hazards, REQUANT without rounding; counters: EX_USEFUL wired to the array step, hazards
on the wrong engine, LD beats wired to busy, PCPI read index off by one, CSR
clear ignored; descriptors: PCPI not held back during a list, FENCE not
waiting, count off by one, read beats not routed by owner) all make the
testbenches fail.

xsim note: compare function results through a `reg` (`x = f(..); if (x !== y)`).
Two calls of the same static function in one expression, or `f(..) !== g(..)`
on wide results, gave wrong values in xsim 2024.1; the testbenches avoid both.

System level: `firmware/{matmul,matmul_insn,gemm,vector,bwtest}` `make sim`
run the firmware on PicoRV32 against this unit.

## Board results

M1–M3 bitstreams with their board numbers: `RISCV-on-PYNQ-Z1/bitstreams/m1`,
`.../m2` (M2: 64³ at 38.3 MAC/cycle, 256³ at 56.5 — 88 % of the D = 8 peak),
`.../m3` (fused int8 GEMM 256³ at 53.5 MAC/cycle; standalone vector ops).

## M1 numbers (D = 8, NPORTS = 1)

OOC at 50 MHz: 5375 LUT, 7181 FF, 70 DSP, 128 BRAM36, WNS +2.9 ns.

Simulation (random-stall DDR model): single 8×8×8 exec 34 cycles; GEMM
32×32×64 in 3515 cycles from the testbench (18.7 MAC/cycle) and 4791
cycles with PicoRV32 issuing the commands (13.7 MAC/cycle).

## M3 numbers (D = 8, NPORTS = 1, with the VE)

OOC at 50 MHz: 16445 LUT (VE 8142), 11604 FF, 96 DSP (VE 24), 128 BRAM36,
WNS +2.5 ns (57 MHz).

Simulation: fused int32 → int8 epilogue (bias period 8, RELU, REQUANT)
64 groups in 136 cycles; quantized GEMM 32×32×64 in 3375 cycles from the
testbench vs 3211 for int32 output.

# sysarray — double-buffered matrix accelerator

Implementation of [`docs/double_buffer_design.md`](../../docs/double_buffer_design.md).
Replaces `rtl/matmul` (Phase 2–4) in the overlay from M1 on; the old unit's
behavior (CSRs, funct7 = 0 instructions) is kept by `sa_legacy`.

| File | Contents |
|---|---|
| `sa_unit.v` | top level: CSR slave, 3 AXI4 masters (NPORTS used), PCPI, memories, engines |
| `sa_sched.v` | input queue → 2-stage decode (validation, bank masks) → bank scoreboard → engine queues |
| `sa_pcpi.v` | custom-0 decoder: funct7 = 0 (legacy) and funct7 = 1 (`mat_cfg/load/store/exec/fence`) |
| `sa_legacy.v` | Phase 2–4 CSR map + CAPS/EXT_STATUS; sequencer turning an 8×8×8 job into LD/LD/EX/ST |
| `sa_ld.v`, `sa_st.v` | DMA engines: 2D shapes (LINEAR / INTERLEAVE), ≤ 16-beat 4 KB-safe bursts striped over NPORTS |
| `sa_ex.v` | EX engine: K-streaming feed (one SPAD_A + one SPAD_B word per cycle), shadow drain, accumulate, repeat (M2: several C tiles per command with B / C strides) |
| `sa_array.v` | D×D PEs (`sa_pe_dsp` / `sa_pe_lut` per column), shadow accumulators |
| `sa_bankmem.v`, `sa_tdpram.v` | double-banked byte-write TDP memories (UG901 template → RAMB36) |
| `sa_defs.vh`, `sa_macros.vh` | memory ids, command packet layout, cfg keys, error codes |

## Verify / build

```sh
make test          # all unit testbenches (PYTHON=... with numpy for tb_sa_unit)
make synth         # OOC synth/place/route, 50 MHz (SYNTH_GEN="D=8 NPORTS=1")
```

| Testbench | Covers |
|---|---|
| `tb_sa_ex` | K-streaming vs golden model: Kt 1–64, strips across banks, back-to-back commands, accumulate, extremes, repeat commands (row-major C strip, gapped B strips, bank crossing) |
| `tb_sa_dma` (NP = 1, 3) | LD/ST shapes, INTERLEAVE, 4 KB splitting, bank crossing, partial ACC words, SLVERR; byte-exact vs reference incl. neighbours |
| `tb_sa_unit` | Phase 2–4 tests (CSR + mat_trigger, 32 NumPy vectors, errors, irq); tiled GEMMs on the new ISA (repeat exec, strip-wide store); 120-command random stream incl. repeat EX with random strides vs a sequential reference (scoreboard); new-ISA error paths |

Mutation checks (B byte off by one, accumulate dropped, extra k step, no
4 KB split, lane mapping, dropped SLVERR, scoreboard without hazard check or
without WAR) all make the testbenches fail.

System level: `firmware/{matmul,matmul_insn,gemm}` `make sim` run the
firmware on PicoRV32 against this unit.

## Board results

M1 and M2 bitstreams with their board numbers: `RISCV-on-PYNQ-Z1/bitstreams/m1`,
`.../m2` (M2: 64³ at 38.3 MAC/cycle, 256³ at 56.5 — 88 % of the D = 8 peak).

## M1 numbers (D = 8, NPORTS = 1)

OOC at 50 MHz: 5375 LUT, 7181 FF, 70 DSP, 128 BRAM36, WNS +2.9 ns.

Simulation (random-stall DDR model): single 8×8×8 exec 34 cycles; GEMM
32×32×64 in 3515 cycles from the testbench (18.7 MAC/cycle) and 4791
cycles with PicoRV32 issuing the commands (13.7 MAC/cycle).

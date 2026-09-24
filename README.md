# Torch2RISCV-accelerator

An end-to-end compiler stack that lowers PyTorch models to a custom RISC-V
instruction set, running on a heterogeneous **ARM (host) + PicoRV32 (control
core) + systolic-array accelerator** built on a PYNQ-Z1 board.

```
PyTorch (ARM / Linux, host)
  → torch-mlir → linalg dialect
  → custom pattern-matching pass → sysarray dialect
  → lowering pass → RISC-V custom instruction (inline asm, .insn encoding)
  → PicoRV32 (RISC-V control core, bare-metal firmware)
  → PCPI (Pico Co-Processor Interface) decode → accelerator trigger
  → Matrix / Vector accelerator (PL, systolic array)
  → result written back to DDR → verified on the ARM host
```

This project covers the full hardware/software boundary of a small,
from-scratch AI accelerator: ISA extension design, MLIR dialect + lowering
passes, RTL accelerator implementation, and the runtime driver connecting the
ARM host and the RISC-V control core.

---

## Why this project

Most "RISC-V on FPGA" repos stop at booting a core. This one goes further:
it builds a full compiler path from a PyTorch model down to a **custom RISC-V
instruction** that a hand-written coprocessor decodes and executes, with a
working MLIR lowering pipeline generating that instruction automatically.

It intentionally mirrors the responsibilities of a compiler/accelerator
architect role:

| Responsibility | Where it shows up in this project |
|---|---|
| PyTorch → custom hardware compilation path | torch-mlir → custom dialect → RISC-V custom-instruction firmware |
| Op dispatch, kernel launch, tensor memory, runtime ABI | PyTorch backend dispatch → PYNQ driver → PCPI instruction ABI / CSR ABI |
| ISA extension / custom instruction design | Custom opcode encoding + PCPI coprocessor RTL + instruction-level MLIR lowering |
| MLIR / compiler infrastructure | Custom `sysarray` dialect, pattern-matching pass, inline-asm lowering |
| Defining DMA, memory model, CSR, and kernel ABI across the HW/SW boundary | Designed and implemented end to end, single-owner |
| Software/hardware co-verification | Golden-model (NumPy) comparison, latency tracing, loose- vs tight-coupling benchmark |

---

## Architecture

```
┌─────────────────────────────┐
│  ARM Cortex-A9 (PS, Linux)  │  Host: PyTorch, torch-mlir, driver
└──────────────┬──────────────┘
               │ AXI-Lite (GP) — control/CSR
               │ AXI (HP)      — bulk DDR data path
┌──────────────▼──────────────┐
│   PicoRV32 (PL, bare metal) │  Control core: decodes custom instruction
│      + PCPI coprocessor     │  and dispatches to the accelerator
└──────────────┬──────────────┘
               │ PCPI handshake (pcpi_insn / rs1 / rs2 / wr / wait / ready)
┌──────────────▼──────────────┐
│  8x8 Systolic Array (PL)    │  int8 in, int32 accumulate, AXI staging buffer
└──────────────────────────────┘
```

### Custom instruction encoding

Uses the RISC-V reserved `custom-0` opcode space (`0001011`), so it never
collides with the standard ISA. `funct3` selects the operation:

| funct3 | Mnemonic | Semantics |
|---|---|---|
| 0 | `mat_trigger` | rs1 = parameter block address, rs2 = destination address |
| 1 | `mat_status` | rd = accelerator status register |
| 2 | `mat_reset` | reset the accelerator |

No changes to the GCC/LLVM front end are required — instructions are emitted
via the standard `.insn` pseudo-op, both by hand-written firmware and by the
MLIR lowering pass (as an LLVM `InlineAsm` node).

```asm
# .insn r opcode, funct3, funct7, rd, rs1, rs2
.insn r 0x0B, 0, 0, x0, a0, a1   # mat_trigger(param_addr=a0, dst_addr=a1)
```

---

## Repository layout

```
├── rtl/
│   ├── sysarray/            # current accelerator (M1-M4): double-buffered D x D systolic
│   │   │                    #   array, LD/ST DMA, vector engine, bank scoreboard, PCPI + CSRs
│   │   ├── sa_*.v, *.vh     #   RTL (file table in rtl/sysarray/README.md)
│   │   ├── sim/             #   unit testbenches (make test / make test16)
│   │   └── synth_ooc.tcl    #   out-of-context synth + timing (make synth)
│   └── matmul/              # Phase 2-4 8x8x8 unit (reference; CSR + PCPI, NumPy vectors)
├── firmware/                # bare-metal PicoRV32 firmware (clang, rv32imc)
│   ├── include/             #   sysarray_intrinsics.h (.insn wrappers), mailbox.h, matmul_csr.h
│   ├── common/, common.mk   #   start.S, link.ld, build + system-simulation rules
│   ├── gemm/                #   tiled GEMM, int32 or int8 output (VE epilogue)
│   ├── vector/              #   standalone vector-engine operations
│   ├── bwtest/              #   DMA bandwidth self-test
│   ├── matmul/, matmul_insn/#   Phase 3 (CSR) / Phase 4 (custom-instruction) job firmware
│   └── sim/tb_system.v      #   firmware on PicoRV32 + the accelerator RTL (make sim)
├── driver/pynq_matmul.py    # PYNQ driver: MatmulOverlay (gemm, vector, bandwidth, matmul)
├── notebooks/               # board scripts per milestone (phase3/4, m1-m4, m4_sched_tune)
├── docs/                    # design docs and the hardware learning path (see below)
└── RISCV-on-PYNQ-Z1/        # the PYNQ-Z1 overlay (PicoRV32 + PS7 + accelerator)
    ├── scripts/             #   build_bitstream.sh/.tcl (one-shot build, -jobs, -sa_d),
    │                        #   pico_bit.tcl / pico_processor.tcl (block design)
    ├── bitstreams/          #   board-verified bit/hwh + results: phase3, phase4, m1-m4
    ├── ip/, gold_ip/        #   PicoRV32 IP repository used by the build (+ reference copy)
    ├── picorv32/            #   upstream PicoRV32 sources
    ├── constrs/             #   PYNQ-Z1 / PYNQ-Z2 XDC constraints
    ├── tests/ddr_access/    #   RISC-V -> DDR access test (firmware, sim, board script)
    └── notebooks/           #   upstream tutorial / example notebooks
```

The MLIR lowering (Phase 5) is not in the repository yet. Build output
(`build/`) is not tracked; tested bitstreams are copied to
`RISCV-on-PYNQ-Z1/bitstreams/`.

---

## Documentation

- [Hardware learning path](docs/hardware_learning_path.md) - a staged guide
  from digital design fundamentals to the accelerator RTL, verification,
  timing closure and board measurements, with labs on this repository
- [Double-buffered accelerator design](docs/double_buffer_design.md) - the
  `rtl/sysarray` architecture, ISA and board results (M1-M4)
- [Custom instruction encoding](docs/custom_isa_encoding.md) and
  [memory model](docs/memory_model.md) - Phase 4 ISA and address maps

---

## Project phases

| Phase | Scope | Status | Result / where |
|---|---|---|---|
| 0 | Environment, one-shot Vivado build | ✅ done | `RISCV-on-PYNQ-Z1/scripts/build_bitstream.sh` |
| 1 | PicoRV32 bring-up on PYNQ-Z1, RISC-V → DDR access | ✅ board-verified | `RISCV-on-PYNQ-Z1/tests/ddr_access` |
| 2 | 8×8×8 matrix unit (CSR + AXI master, loose coupling) | ✅ simulation-verified vs NumPy | `rtl/matmul` |
| 3 | End-to-end closed loop, CSR path, hand-written firmware | ✅ board-verified | 522.6 cycles/job; `bitstreams/phase3` |
| 4 | Custom instructions (PCPI, `.insn`) | ✅ board-verified | 311.9 cycles/job; `bitstreams/phase4` |
| M1–M4 | Accelerator rebuild: double-buffered systolic array (table below) | ✅ board-verified | `rtl/sysarray`, `bitstreams/m1`–`m4` |
| 5 | MLIR dialect + instruction-level lowering | ⏳ next | targets `firmware/include/sysarray_intrinsics.h` |
| 6 | Loose- vs tight-coupling benchmark | 🟡 partial | Phase 3 vs 4 board numbers exist; write-up pending |
| 7 | Vector unit | ✅ done in M3 | fused int8 epilogue + standalone vector ops |
| 8 | Auto-tiling + double buffering | 🟡 double buffering done (M1); compiler auto-tiling with Phase 5 | tiling is hand-written in `firmware/gemm` today |
| 9 | Documentation and write-up | 🟡 in progress | `docs/` (design doc, learning path) |

The accelerator rebuild replaced the Phase 2–4 unit with a double-buffered
design (`docs/double_buffer_design.md`); each milestone was measured on the
board before choosing the next one:

| Milestone | Scope | Board result (MAC/cycle at 50 MHz) |
|---|---|---|
| M1 | D = 8 array, SPAD/ACC banks, LD/ST DMA, bank scoreboard, funct7 = 1 ISA, legacy compatibility | 256³ GEMM at 54.5 (35× Phase 4) |
| M2 | Repeat `mat_exec`, 8 outstanding bursts, DMA bandwidth test (one HP port at 99 %) | 64³ 13.0 → 38.3, 256³ 56.5 |
| M3 | Vector engine (funct7 = 2): bias/RELU/requant → int8, standalone vector ops | int8 GEMM 256³ at 53.5 |
| M4 | D = 16 (8 DSP + 8 LUT columns, VL = 16); three HP ports measured as unnecessary; firmware schedule tuning | 256³ at 202.6 (79 % of peak), 256×128×1024 at 222.2 |

**Minimum viable target: Phase 5** — a PyTorch model compiles end to end
into firmware that issues a custom RISC-V instruction, executed by a
hand-built accelerator on real FPGA hardware. The hardware side of that
target is complete.

---

## Acknowledgements

- [PicoRV32](https://github.com/YosysHQ/picorv32) — the RISC-V core used as
  the control core.
- [drichmond/RISC-V-On-PYNQ](https://github.com/drichmond/RISC-V-On-PYNQ) and
  [JacoboJin/RISCV-on-PYNQ-Z2](https://github.com/JacoboJin/RISCV-on-PYNQ-Z2) —
  reference PS/PL bridging IP and PCPI integration used as a starting point
  for the PicoRV32-on-Zynq bring-up.
- [torch-mlir](https://github.com/llvm/torch-mlir) / [MLIR](https://mlir.llvm.org/) —
  the compiler infrastructure this project's lowering passes build on.

## License

MIT (or your choice — update before publishing).

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
┌──────────────────────────────────┐
│   ARM Cortex-A9 (PS, Linux)      │  Host: PyTorch, driver/pynq_matmul.py
└──┬──────────────────┬────────────┘
   │ AXI GP           │ EMIO GPIO[0]
   │ firmware +       │ RISC-V reset
   │ mailbox (BRAM)   │
┌──▼──────────────────▼────────────┐         ┌──────────────────────────┐
│  PicoRV32 (PL, rv32imc, 50 MHz)  │── HP0 ─►│                          │
│  8 KB program BRAM, bare metal   │         │   DDR3 (512 MB, shared)  │
└──┬────────────────────┬──────────┘         │                          │
   │ PCPI: custom-0     │ AXI-Lite: legacy   └──────────────▲───────────┘
   │ instructions       │ CSRs (0x80000000)                 │
┌──▼────────────────────▼─────────────────────────────┐     │ HP2
│  sa_unit — double-buffered accelerator (PL)         │     │ (AXI4 → AXI3)
│                                                     │     │
│  command queue → decode → bank scoreboard           │     │
│      │ in-order dispatch to four engines            │     │
│      ├─► LD  DDR → local memory   ┐                 │     │
│      ├─► ST  local memory → DDR   ┴─ DMA, 64-bit ───┼─────┘
│      ├─► EX  D×D systolic array, int8 × int8 → int32 (K-streaming)
│      └─► VE  vector engine, VL = D: add/sub/mul/max/min/copy,
│              RELU, requant → int8/int16/int32
│                                                     │
│  SPAD_A 128 KB · SPAD_B 128 KB · ACC 256 KB         │
│  (each two banks: DMA fills one while EX/VE use the other)
└─────────────────────────────────────────────────────┘
```

The board build (M4) uses D = 16: a 16×16 array (8 DSP + 8 LUT columns),
256 MAC/cycle peak at 50 MHz, one HP port for the accelerator. Details:
[`docs/double_buffer_design.md`](docs/double_buffer_design.md).

### Custom instruction encoding

All instructions are R-type in the RISC-V reserved `custom-0` opcode space
(`0x0B`), so they never collide with the standard ISA. `funct7` selects the
instruction group and `funct3` the operation:

| funct7 | funct3 | Mnemonic | Operands | Semantics |
|---|---|---|---|---|
| 0 | 0 | `mat_trigger` | rs1 = descriptor, rs2 = C | one 8×8×8 job (Phase 4, kept for compatibility) |
| 0 | 1 | `mat_status` | rd | STATUS, never stalls |
| 0 | 2 | `mat_reset` | — | clear done/error (also the sticky error of the new ISA) |
| 0 | 3 | `mat_wait` | rd | stall until idle, return STATUS |
| 0 | 4 | `mat_cycles` | rd | cycles of the last job |
| 1 | 0 | `mat_cfg` | rs1 = key, rs2 = value | DMA shape, exec repeat/strides |
| 1 | 1 | `mat_load` | rs1 = DDR, rs2 = local addr | queue DDR → SPAD/ACC |
| 1 | 2 | `mat_store` | rs1 = DDR, rs2 = local addr | queue SPAD/ACC → DDR |
| 1 | 3 | `mat_exec` | rs1 = B<<16 \| A, rs2 = acc<<28 \| Kt<<16 \| C | queue C (+)= A strip × B strips |
| 1 | 4 | `mat_fence` | rs1 = engine mask, rd | stall until the engines are idle, return extended status |
| 1 | 5 | `mat_perf` | rs1 = counter index (read) or bit 31 + rs2 = clear/enable (control), rd | performance counters (CAPS bit 20; board build `bitstreams/m4p`) |
| 1 | 6 | `mat_submit` | rs1 = descriptor list (DDR), rs2 = count | run a list of 64-byte descriptors built by the ARM (CAPS bit 21; board build `bitstreams/m5`) |
| 2 | 0 | `vec_cfg` | rs1 = key, rs2 = value | op, length, types, dst, src2 period, requant, clamp |
| 2 | 1 | `vec_run` | rs1 = src1, rs2 = src2 | queue one vector command |

Queued commands return immediately; ordering between the engines is kept by
the hardware scoreboard (DDR is not tracked, so software fences between a
store and a later load of the same bytes). Full semantics:
[`docs/custom_isa_encoding.md`](docs/custom_isa_encoding.md) (funct7 = 0)
and [`docs/double_buffer_design.md`](docs/double_buffer_design.md) §8
(funct7 = 1, 2); C wrappers in
[`firmware/include/sysarray_intrinsics.h`](firmware/include/sysarray_intrinsics.h).

No changes to the GCC/LLVM front end are required — instructions are emitted
via the standard `.insn` pseudo-op, both by hand-written firmware and by the
MLIR lowering pass (as an LLVM `InlineAsm` node).

```asm
# .insn r opcode, funct3, funct7, rd, rs1, rs2
.insn r 0x0B, 1, 1, x0, a0, a1   # mat_load(ddr=a0, laddr=a1)
.insn r 0x0B, 3, 1, x0, a2, a3   # mat_exec(a2 = B<<16 | A, a3 = acc<<28 | Kt<<16 | C)
.insn r 0x0B, 4, 1, a0, x0, x0   # a0 = mat_fence(all engines)
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
    ├── ip/                  #   IP repository used by the build (picorv32_axi, pcpi interface)
    ├── picorv32/            #   picorv32.v from upstream PicoRV32 (+ license)
    ├── constrs/             #   PYNQ-Z1 XDC constraints
    └── tests/ddr_access/    #   RISC-V -> DDR access test (firmware, sim, board script)
```

The MLIR lowering (Phase 5) is not in the repository yet. Build output
(`build/`) is not tracked; tested bitstreams are copied to
`RISCV-on-PYNQ-Z1/bitstreams/`.

---

## Documentation

- [Hardware learning path](docs/hardware_learning_path.md) - a staged guide
  from digital design fundamentals to the accelerator RTL, verification,
  timing closure and board measurements, with labs on this repository
- [Execution walkthrough (中文)](docs/execution_walkthrough.md) - one GEMM
  traced from `m4_demo.py` through the driver, firmware, PCPI, scheduler and
  engines down to BRAM/DDR, with a per-layer debugging checklist
- [Performance counters and descriptor DMA (中文)](docs/perf_counters_and_desc_dma_plan.md) -
  plan, implementation record and board results of the two M5 additions:
  counter definitions, descriptor format, verification, milestones
- [Double-buffered accelerator design](docs/double_buffer_design.md) - the
  `rtl/sysarray` architecture, ISA and board results (M1-M5)
- [Custom instruction encoding](docs/custom_isa_encoding.md) and
  [memory model](docs/memory_model.md) - the Phase 4 instructions with an
  overview of all three instruction groups; address maps, coherency, DDR
  ordering and alignment rules

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
| M4 + perf | Performance counters (28 events, `mat_perf`): measured cycle breakdown | 256³: array useful 79 %, front end idle 5 %; 64³: front end 87 % → descriptor DMA planned for small ops |
| M5 | Descriptor DMA: ARM-built lists of 64-byte descriptors, one `mat_submit` (`sa_cmdfetch.v`) | GEMM 16³ 4.2×, 64³ 1.2×, int8 64³ 1.5×, vector ops up to 2.6× faster; large GEMMs unchanged |

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

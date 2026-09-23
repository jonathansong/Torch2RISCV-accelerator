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

## Repository layout (planned)

```
├── rtl/                  # PicoRV32 integration, PCPI coprocessor, systolic array
├── firmware/             # Bare-metal C firmware, sysarray_intrinsics.h
├── mlir/                 # sysarray dialect, pattern-matching + lowering passes
├── driver/               # PYNQ Python driver (MMIO / allocate based)
├── constrs/              # XDC constraints for PYNQ-Z1
├── notebooks/            # End-to-end demo notebooks
├── docs/
│   ├── baseline_utilization.md
│   ├── custom_isa_encoding.md
│   ├── ps_pl_datapath.md
│   └── tight_vs_loose_coupling.md
└── scripts/               # Vivado .tcl build scripts
```

---

## Project phases

| Phase | Scope | Milestone |
|---|---|---|
| 0 | Environment setup, base overlay resource baseline | — |
| 1 | PicoRV32 bring-up on PYNQ-Z1 | **M1** — bare-metal hello world |
| 2 | Matrix accelerator (CSR/AXI, loose coupling) | **M2** — simulation-verified against NumPy |
| 3 | End-to-end closed loop, CSR path, hand-written firmware | **M3** — first demo-able ARM→PicoRV32→accelerator run |
| 4 | Custom instruction (PCPI) design + implementation | **M4** — `.insn`-triggered accelerator run, verified |
| 5 | MLIR dialect + instruction-level lowering | **M5** — PyTorch model compiles straight to custom-instruction firmware |
| 6 | Loose- vs tight-coupling benchmark | **M6** — quantified CSR vs PCPI comparison |
| 7 | Vector unit (optional, resource-permitting) | M7 |
| 8 | Auto-tiling + double buffering (optional) | M8 |
| 9 | Documentation and write-up | M9 |

**Minimum viable target: M5** — a PyTorch model compiles end to end into
firmware that issues a custom RISC-V instruction, executed by a hand-built
accelerator on real FPGA hardware.

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

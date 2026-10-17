# Torch2RISCV-accelerator

> **Platform update:** Development has moved to the AMD Kria KV260
> ([`boards/kv260/`](boards/kv260/README.md), [plan](docs/kv260_upgrade_plan.md)). Its first
> stage (K1: the same accelerator, then 100 MHz) is accepted on the board:
> every test's tokens are identical to the PYNQ-Z1 baselines, stories15M at
> 35.4 tok/s and SmolLM2-135M at 4.4 tok/s (generic HuggingFace path 3.7).
> Since then: 128-bit DMA, a batched SFU and REDUCE, Qwen3-0.6B (int8, all 28
> layers) on the board, and a streaming GEMV unit for decode fed by four DMA
> ports (K2b, 64 B/cycle at 100 MHz). Decode on the board, bit-exact with the
> sim: Qwen3-0.6B 6.28 tok/s (prefill 15.9), SmolLM2-135M 17.0, stories15M
> 131.7. The accelerator's buffers come from a 1280 MiB pool reserved at boot.
> Unmodified HuggingFace models now compile whole through the generic path
> ([generality plan](docs/generality_implementation_plan.md) §13.2), every
> dispatch on the accelerator, bit-exact with the sim, decode / prefill tok/s:
> SmolLM2-135M 22.1 / 60.5, Gemma-3-270M 15.0 / 62.6, Qwen3-0.6B 7.0 / 20.8,
> Llama-3.2-1B 4.0 / 15.1 (public W8A8 checkpoints, int8 KV cache; board
> perplexity within 1-16% of fp32).
> **Consolidated results (2026-10-17, tag `v2.2-kv260`,
> [results_v2.2](docs/results_v2.2.md))**: all four models compiled for 2048
> positions (Gemma 512), decode / prefill tok/s SmolLM2-135M 23.6 / 57.8,
> Gemma-3-270M 16.5 / 68.5, Qwen3-0.6B 7.21 / 19.0, Llama-3.2-1B 4.42 / 15.4;
> decode reads weights at 66–96% of the four-port peak; board teacher forcing
> over two context-length windows within 3% of fp32 PPL for all four; the
> RISC-V paths below on all four models (SmolLM2 added 2026-10-17).
> Long context (2026-10-16): Qwen3-0.6B and Llama-3.2-1B compiled for 2048
> positions, Gemma-3-270M for 512, prefill vs decode bit-exact on the board over
> a 2032-token prompt; Qwen3 decode 7.2 tok/s at T = 64 and 3.2 at T = 2048,
> Qwen3 teacher-forcing PPL over two 2048-token windows 18.58 (fp32 18.77).
> RISC-V backend pre-research (C7, [design](docs/riscv_backend_design.md)), all
> on the board for Gemma-3, Qwen3 and Llama-3.2, bit-exact:
> - **Plan B**: the decode loop runs on the PicoRV32 control core instead of the
>   ARM. The ARM records the step's lists once per attention-length bucket; the
>   core patches them at bucket changes, submits them and takes the argmax.
> - **Plan A**: every dispatch as RISC-V code that iree-compile generates
>   (`sahw` → LLVM IR → rv32imc) and embeds in `sa.vmfb`, which issues the
>   commands over PCPI with no descriptors, also inside plan B's loop.
> - **R5**: a scalar kernel compiled by IREE's llvm-cpu for the core.
> - **One executable per kernel, not per layer** (on by default): a decode
>   step's rv32 code is 20–41 KB instead of 65–288 KB, small enough for a
>   32–64 KB instruction memory.
>
> The counters put plan A's remaining gap to the lists (Qwen3 163 against 136
> ms per step) entirely on instruction fetch from DDR: about 27 cycles per
> instruction, 9–11 instructions per command.
> The PYNQ-Z1 version is frozen at tag
> [`v1.0-pynq-z1`](../../tree/v1.0-pynq-z1) (branch `pynq-z1`): stories15M at
> 17.6 tok/s and SmolLM2-135M (prefill + decode, the hand-written and the
> generic HuggingFace path), all bit-exact on the board; its baselines are in
> [`tests/baselines/pynq-z1/`](tests/baselines/pynq-z1/). It still works and
> is the cheapest way to try the full stack, but it only receives critical
> fixes. The project phases below were done on the PYNQ-Z1; their bitstreams are on
> that branch.

An end-to-end compiler stack that lowers PyTorch models to a custom RISC-V
instruction set, running on a heterogeneous **ARM (host) + PicoRV32 (control
core) + systolic-array accelerator** on an FPGA SoC: the AMD Kria KV260
(Zynq UltraScale+) now, a PYNQ-Z1 (Zynq-7020) until the freeze.

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
│   ├── sysarray/            # the accelerator: double-buffered D x D systolic array, LD/ST DMA,
│   │   │                    #   descriptor fetch, int8 + fp32 vector engines, scheduler, PCPI + CSRs
│   │   ├── sa_*.v, *.vh     #   RTL (file table in rtl/sysarray/README.md)
│   │   ├── sim/             #   unit testbenches (make test / make test16)
│   │   └── synth_ooc.tcl    #   out-of-context synth + timing (make synth)
│   ├── picorv32/            # picorv32.v from upstream PicoRV32 (+ license)
│   ├── ip/                  # Vivado IP repository: picorv32_axi, the PCPI interface
│   └── matmul/              # Phase 2-4 8x8x8 unit (reference; CSR + PCPI, NumPy vectors)
├── boards/kv260/            # the Kria KV260 overlay (README: builds, address map, board setup)
│   └── scripts/             #   build_bitstream.sh/.tcl (-sa_d, -sa_mhz, -bd_only), kv260_bd.tcl,
│                            #   pico_processor.tcl (block design); outputs in build/output/<config>
├── firmware/                # bare-metal PicoRV32 firmware (clang, rv32imc)
│   ├── include/             #   sysarray_intrinsics.h (.insn wrappers), mailbox.h, matmul_csr.h
│   ├── common/, common.mk   #   start.S, link.ld, build + system-simulation rules
│   ├── rt/                  #   the resident runtime firmware (command ring)
│   ├── gemm/, vector/       #   tiled GEMM (VE epilogue), standalone vector-engine operations
│   ├── bwtest/, desc_run/   #   DMA bandwidth self-test, descriptor lists
│   ├── matmul/, matmul_insn/#   Phase 3 (CSR) / Phase 4 (custom-instruction) job firmware
│   └── sim/tb_system.v      #   firmware on PicoRV32 + the accelerator RTL (make sim)
├── compiler/                # the IREE compiler (sa backend), the sa HAL runtime, tests, deploy scripts
├── llm/                     # the hand-written LLM path (export, compile_model, runtime, funcsim)
├── driver/pynq_matmul.py    # PYNQ driver: MatmulOverlay (reads board, D, addresses, clock from the .hwh)
├── notebooks/               # board scripts per milestone (phase3/4, m1-m5, llm/)
├── tests/                   # ddr_access (RISC-V -> DDR test), baselines/pynq-z1 (freeze baselines)
├── docs/                    # design docs and the hardware learning path (see below)
└── LICENSES/                # licenses of third-party parts (see NOTICE)
```

Build output (`build/`, `boards/kv260/build/`) is not tracked; KV260
bitstreams are published with the GitHub releases. The PYNQ-Z1 overlay
(block design, constraints, the board-verified bitstreams of every
milestone: phase3, phase4, m1-m5, l0-l2) is on the `pynq-z1` branch (tag
`v1.0-pynq-z1`), as `RISCV-on-PYNQ-Z1/`; paths `RISCV-on-PYNQ-Z1/...` and
`bitstreams/...` in the documents refer to it.

---

## Documentation

- [Results v2.2](docs/results_v2.2.md) - the consolidated board numbers
  (tag `v2.2-kv260`): speed, quality, long context, decode bandwidth and the
  RISC-V backend paths for SmolLM2, Gemma-3, Qwen3 and Llama-3.2, with the
  bundle, commit and log behind each
- [LLM learning path](docs/llm_learning_path.md) - the models, their
  numerics, W8A8 and KV-cache quantization, the inference system, prefill /
  decode, long context, and measuring speed and quality on the board
- [Compiler learning path](docs/compiler_learning_path.md) - MLIR and IREE,
  the export, the `sa` backend pass by pass, the HAL driver, verification,
  the RISC-V backend and upstreaming to IREE
- [Hardware learning path](docs/hardware_learning_path.md) - a staged guide
  from digital design fundamentals to the accelerator RTL, verification,
  timing closure and board measurements, with labs on this repository
- [Execution walkthrough](docs/execution_walkthrough.md) - one GEMM
  traced from `m4_demo.py` through the driver, firmware, PCPI, scheduler and
  engines down to BRAM/DDR, with a per-layer debugging checklist
- [Dispatch walkthrough](docs/dispatch_walkthrough.md) - one IREE dispatch (decode
  attention scores) through every pass of the sa compiler to its sa-desc
  descriptor template, bit by bit, and what the driver and accelerator do with it
- [Performance counters and descriptor DMA](docs/perf_counters_and_desc_dma_plan.md) -
  plan, implementation record and board results of the two M5 additions:
  counter definitions, descriptor format, verification, milestones
- [LLM inference plan](docs/llm_inference_plan.md) - the next stage:
  an industry-style LLM inference system (ARM host runtime, command ring and
  completion IRQ, resident RISC-V firmware, compiler-facing command
  extensions for MLIR/IREE, fp32 vector/special-function unit, W8A8 + int8
  KV cache) built in levels L0-L6, 50 MHz with 75 MHz as a stretch goal
- [LLM compiler plan](docs/iree_compiler_plan.md) - the design of the
  end-to-end compiler (level L6-IREE): a quantized PyTorch llama exported with
  iree-turbine, an IREE target-backend plugin that emits descriptor-list
  templates, and a C HAL driver on the ARM that submits them through the
  command ring, in stages C0-C7 (C4 closed 2026-10-15, its fusions continued on
  the KV260; C7, a RISC-V backend, is pre-research for an ASIC NPU:
  [design](docs/riscv_backend_design.md); since 2026-10-16 the PicoRV32 can run
  the whole decode loop on the board, bit-exact, the ARM out of the loop, every
  dispatch also as RISC-V code generated and embedded by iree-compile). Done: a full MLIR code generator (the `sahl`
  / `sahw` dialects, templates as micro-kernels); stories15M (17.7 tok/s) and
  HuggingFace SmolLM2-135M (2.22 tok/s) run on the board bit-exact with the
  references, and generate text interactively there (on the KV260 also the
  generic bundles, with multi-turn chat for Llama-3.2-1B-Instruct and Qwen3);
  larger models are the
  current stage (C6, on sim: Qwen3-0.6B with all 28 layers compiles bit-exact
  per dispatch and generates correct text end to end). A model now compiles
  with both prefill and decode: on the board the prompt is processed 8 tokens
  per call on the full array, 4.3-4.7x faster per token than decode, bit-exact.
  The code generator's decisions (kernels, gathers, pieces, memory places,
  schedules) are separate passes visible in the IR, guarded by a
  byte-for-byte golden corpus of 4002 dispatches (C8); the micro-kernels'
  schedules (strips, EX, SPAD_B banks, prefetch, loops) are written out as
  `sahl` operations, so the last stage only translates one to one. Dispatches the
  accelerator backend cannot compile run on the ARM host (VMVX host fallback).
  An unmodified HuggingFace decoder (transformers' own modeling code) now
  compiles too: generic rewrites (W8A8, RoPE tables, an int8 KV cache and the
  accelerator's attention) put it on the accelerator, prefill and decode, with
  the quantized model's quality; its decode runs entirely on the accelerator,
  and Qwen3 compiles through the same path without code changes (sim).
  Since then Qwen3-0.6B, SmolLM2-135M, Llama-3.2-1B and Gemma-3-270M run whole
  on the KV260 through this path, every dispatch on the accelerator
  ([generality plan](docs/generality_implementation_plan.md) §13.2).
  See [compiler/README.md](compiler/README.md)
- [PYNQ-Z1 → Kria KV260 upgrade plan](docs/kv260_upgrade_plan.md) - the
  port in stages (platform at the same accelerator, clock, DMA and on-chip
  memory bandwidth, URAM capacity, Qwen3-0.6B), the measured Z1 time breakdown
  it starts from, and the repository strategy (Z1 frozen at `v1.0-pynq-z1`).
  K1 (K1a 50 MHz, K1b 100 MHz), K2a (128-bit DMA), Qwen3-0.6B (K4a) and the
  four-port streaming GEMV (K2b, [design](docs/k2b_gemv_design.md)) are
  accepted on the board; the overlay, its
  builds, address map and board setup are in [boards/kv260/README.md](boards/kv260/README.md)
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
| 0 | Environment, one-shot Vivado build | ✅ done | `RISCV-on-PYNQ-Z1/scripts/build_bitstream.sh` (pynq-z1 branch; KV260: `boards/kv260/scripts/build_bitstream.sh`) |
| 1 | PicoRV32 bring-up on PYNQ-Z1, RISC-V → DDR access | ✅ board-verified | `tests/ddr_access` |
| 2 | 8×8×8 matrix unit (CSR + AXI master, loose coupling) | ✅ simulation-verified vs NumPy | `rtl/matmul` (tag `v1.0-pynq-z1`; its interface lives on in `rtl/sysarray/sa_legacy.v`) |
| 3 | End-to-end closed loop, CSR path, hand-written firmware | ✅ board-verified | 522.6 cycles/job; `bitstreams/phase3` |
| 4 | Custom instructions (PCPI, `.insn`) | ✅ board-verified | 311.9 cycles/job; `bitstreams/phase4` |
| M1–M4 | Accelerator rebuild: double-buffered systolic array (table below) | ✅ board-verified | `rtl/sysarray`, `bitstreams/m1`–`m4` |
| 5 | MLIR dialect + instruction-level lowering | ✅ superseded | done as the IREE compiler (`sahl` / `sahw` dialects → descriptor lists, [`compiler/`](compiler/README.md)) instead of `.insn` firmware |
| 6 | Loose- vs tight-coupling benchmark | 🟡 partial | Phase 3 vs 4 board numbers exist; write-up pending |
| 7 | Vector unit | ✅ done in M3 | fused int8 epilogue + standalone vector ops |
| 8 | Auto-tiling + double buffering | ✅ done | double buffering (M1); compiler tiling in the IREE backend (K blocks, `sahl-tile` pieces, GEMV K split); attention over long T (C6.5 / C6.5b, sim: Gemma 512, Qwen3 2048, a 1104-token prompt end to end; on the board 2026-10-14: whole Qwen3 at 2048 / Gemma at 512, prefill vs decode only bit-exact over a 2032-token prompt, Qwen3 decode 1.60 tok/s at T = 2048; since 2026-10-15 on the sim: p · V on the GEMV unit with a packed V cache, the tiled q · Kᵀ's blocks of 128, query rows in one EX and a software-pipelined T loop: attention at T = 2048 431.6 → 132.7 ms per token by the cycle model; on the board (2026-10-15) all bit-exact and the device time as projected (Qwen3 decode at 2048 618.6 → 310.4 ms device), a host copy of the packed V cache per step, found on the board, fixed with per-layer V buffers written as a scatter: on the board 2026-10-16 wall = device, Qwen3 decode at 2048 316.8 ms per step (was 626.1), at 64 138.1, prefill 2.1× faster; KV scales calibrated on the context length: Qwen3 board KL at 2048 0.073 → 0.055; [plan §8.22](docs/iree_compiler_plan.md)) |
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

Apache-2.0 WITH LLVM-exception. See LICENSE and NOTICE; third-party parts keep their own licenses.

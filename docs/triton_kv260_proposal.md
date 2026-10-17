# Project proposal: Triton kernels for LLM inference on the KV260 accelerator

Status: proposal (2026-10-17), not started. Planned start after the IREE
community meeting (2026-11-25). Working name of the new repository:
`triton-kv260`.

## 1. Summary

This repository compiles unmodified HuggingFace models to the KV260
accelerator automatically, through IREE and the `sa` backend (tag
`v2.2-kv260`, [`results_v2.2.md`](results_v2.2.md)). The new project keeps
the hardware, the numerics and the verification method, and changes the
programming model: **LLM inference written as a small library of hand-fused
Triton kernels**, compiled by a new Triton backend that lowers Triton's block
operations to the existing `sahl` tile dialect, and run by a lightweight
runtime that records one decode step as a graph and replays it.

The two projects answer different questions and are meant to be compared:
automatic whole-model compilation on one side, a kernel programming model on
the other, on the same board, the same bitstream and the same bit-exact
contract.

## 2. Questions the project answers

1. **Programmability.** Can block-level Triton kernels be written for a
   command-driven accelerator with explicit local memories (SPAD / ACC, banks,
   a scoreboard, DMA), and at what cost in code and effort compared with the
   IREE path?
2. **Performance.** Do hand-fused kernels reach the cross-operator fusion
   bound the performance model gives
   ([`fusion_design.md`](fusion_design.md) §2: decode −4.5% to −14%, prefill
   −10% to −20% against the unfused IREE build), or beat the IREE path with its
   own fusions?
3. **Mapping limits.** Which Triton concepts map to this kind of hardware and
   which do not: the launch grid, masks, dynamic shapes, atomics, `tl.dot`
   shapes, reductions with a defined order.

## 3. Success criteria

| Criterion | Target |
|---|---|
| Correctness | SmolLM2-135M decode on the board: logits bit-exact with the functional simulator (17/17 steps), every kernel bit-exact with its oracle |
| Cross-check | The same model, prompt and quantization as the IREE bundle: logits compared (bit-exact where the formulas are the same) |
| Speed | SmolLM2 decode no slower than `v2.2-kv260` (42.4 ms per step, 23.6 tok/s); goal under 37 ms |
| Scale | Qwen3-0.6B decode and prefill on the board |
| Report | Kernel code size and development time against the IREE path; a list of Triton features that map and that do not |

## 4. Architecture

```
Triton kernels (Python: tl.load / tl.dot / element-wise / reductions on blocks)
  │ Triton frontend → Triton IR (tt dialect)
  │ the new backend (Triton's out-of-tree backend: compiler stages + driver)
  │   ├ pointer analysis: pointer arithmetic → strided memref accesses, masks → bounds
  │   │   (from triton-shared's PtrAnalysis / MaskAnalysis)
  │   └ tt → sahl: block load → sahl.load, tl.dot → sahl.mma / sahl.gemv,
  │       element-wise → VE ops, the K loop → sahl.loop, reductions → the device's order
  │ the existing sahl → sahw pipeline: memory planning, scheduling, registers, serialization
  ▼
sa-desc templates (optionally rv32 code: the C7 backend)
  │ runtime (C): load templates, allocate in the u-dma-buf window, record a step graph,
  │ replay it through rt_fw's ring (one list per step), read back logits
  ▼
KV260 overlay d16_100mhz_w128_p6_gemv_np4_pcpi (Release v2.2.1: bitstream, rt_fw v7)
```

**Orchestration.** A Python model description on the host builds one decode
step (and one prefill chunk) as a sequence of kernel launches with their
buffers, once; the board-side C runner replays it per token. Launching kernels
one by one from Python on the A53 would cost tens of microseconds each, for
hundreds per step (the IREE runtime's fixed part is already 3.5–6.5 ms per
step, `results_v2.2.md` §2).

**Weights.** Packed offline with the existing rules (`ct_import.py`, the
`[N/D, K, D]` B-tile layout), loaded once.

**Why lower to `sahl`, not to linalg.** Triton's block operations correspond
almost one to one with `sahl`'s tile operations, which already carry local
memory placement, strips, loops and the GEMV unit. Going through linalg (as
triton-shared does for CPUs) would make the backend re-match patterns the
kernel author already wrote explicitly.

## 5. What carries over from this repository

| Asset | Use |
|---|---|
| Overlay, rt_fw, u-dma-buf setup | Unchanged (Release v2.2.1 assets) |
| `llm/sa_funcsim.py`, `llm/fp32.py`, `llm/sfu.py` | The simulator and the numerics, unchanged |
| `sahl` / `sahw` dialects and passes (`compiler/plugins/sa/`) | Reused, ported to the LLVM revision Triton pins (IREE pins another) |
| Runtime transports (`sa_transport_board.c`, `sa_transport_sim.c`), `board_launcher.py`, `compiler/sim/` | The new runtime's lower layer |
| The oracle approach | Triton's interpreter mode (`TRITON_INTERPRET=1`, numpy) with the device's numerics substituted: a per-kernel oracle |
| Quality and board tooling (`board_quality.py`, WikiText-2 windows, `results/` layout) | Reused for the reports |
| The IREE bundles | The baseline: same model, prompt, quantization |

## 6. Kernel library (decode of the Llama / Qwen3 family)

About eight kernels, four to five launches per layer (the IREE path runs about
30 dispatches per layer):

| Kernel | Content |
|---|---|
| `rmsnorm_quant` | residual add, RMSNorm, per-token int8 quantization |
| `gemv_w8a8` | int8 GEMV on the four-port unit; epilogue variants: scales, residual add, int8 output |
| `swiglu_fused` | gate and up GEMVs, SiLU · mul, amax, int8 output (the hand-written form of the SwiGLU fusion) |
| `rope_kv_write` | q / k norm, RoPE, the KV cache write |
| `attention_decode` | q · Kᵀ in T blocks, qsum softmax, p · V on the GEMV unit |
| `gemm_w8a8_rows`, `attention_prefill_rows` | prefill's 16-row chunk |
| `argmax` | the token |

Kernels see the hardware's constraints through small helpers: block sizes in
multiples of D = 16, packed weight views, the local memory budget.

## 7. Milestones

| Milestone | Content | Estimate |
|---|---|---|
| **M0 feasibility** | Triton built from source at a pinned release; backend skeleton; `sahl` / `sahw` built against Triton's LLVM; a vector-add kernel through to an sa-desc template, run on the simulator | 2 weeks |
| M1 first real kernel | `gemv_w8a8`: pointer analysis, tt → sahl, memory planning; bit-exact on the simulator, then on the board | 2–3 weeks |
| M2 SmolLM2 decode | the remaining kernels; end to end on the simulator against the IREE bundle's logits | 3–4 weeks |
| M3 runtime | step graph record / replay on the board; speed against `v2.2-kv260` | 2 weeks |
| M4 scale | prefill, Qwen3 / Llama, long context (T blocks) | 3–4 weeks |
| M5 optional | torch.compile / Inductor generating Triton for this device | open |

About three to four months to the first comparison on the board (SmolLM2),
one person working with an AI assistant at this project's pace.

### M0 in detail

| # | Task | Done when |
|---|---|---|
| 0.1 | Pin a Triton release; build it from source in a venv within the 15 GB / 10 GB-cap limits; record its LLVM revision | `python -c "import triton"` works; build time and disk recorded |
| 0.2 | Out-of-tree backend skeleton: a target name, compiler stages (ttir → our stages → binary), a driver stub; confirm how the Triton release discovers external backends | a `@triton.jit` kernel reaches our first stage and dumps its tt IR |
| 0.3 | Port `sahl` / `sahw` (dialects, `SahlToSahw`, `SahwSerialize`, `DescList`) to Triton's LLVM; list the API changes | the dialects' lit tests pass against the new LLVM |
| 0.4 | tt → sahl for one element-wise kernel (vector add / scale: `tl.load`, `+`, `tl.store` over one block) | a `sahl` function equivalent to what `sa-to-sahl` gives for the same operation |
| 0.5 | The binary: sa-desc via the existing serializer; the driver loads it into the simulator transport | the kernel runs on `sa_funcsim` and matches numpy |
| 0.6 | Interpreter oracle: Triton's interpreter with `llm/fp32.py` arithmetic | the vector-add kernel bit-exact against it |
| 0.7 | Go / no-go note: the porting cost of 0.3, the shape of the pointer analysis needed for M1, revised estimates | written up in the new repository |

**Stop condition.** If 0.3 shows the dialects cannot reasonably move to
Triton's LLVM, the fallback is a sahl-equivalent tile dialect written fresh
in the new repository (its passes ported by hand), and the estimates are
revised before M1.

## 8. Risks

| Risk | Mitigation |
|---|---|
| LLVM revision mismatch (IREE and Triton pin different ones) | M0 task 0.3 measures it first; fallback above |
| Triton API changes between releases (backend interface, IR) | Pin one release for the whole project |
| Hardware constraints leaking into kernels (D = 16 blocks, packed weights, two dynamic fields per command, local memory size) | Kernel-side helpers; the backend rejects what does not fit with a clear message |
| Host overhead on the A53 | Record / replay of step graphs from the start (M3 design in M0's note) |
| Build resources (two large builds on a 15 GB machine, disk) | One build at a time under the memory cap; the new repository on its own build tree |
| Scope growth (generality across many models) | The kernel library targets the Llama / Qwen3 family; Gemma's variants only after M4 |
| Schedule | Starts after 2026-11-25; the talk and the SwiGLU fusion's board results first |

## 9. Repository layout (planned)

```
triton-kv260/
  backend/        Triton backend: compiler stages, driver, pointer analysis, tt → sahl
  dialects/       sahl / sahw ported (or the fresh tile dialect), their passes and lit tests
  runtime/        C runtime: template loader, buffer allocation, step-graph record / replay, transports
  kernels/        the Triton kernel library (Section 6)
  models/         model descriptions building the step graphs; weight packing
  sim/            the functional simulator and numerics (from this repository)
  tests/          per-kernel oracle tests, end-to-end checks against the IREE bundles, board scripts
  docs/
```

## 10. Relationship to this repository

This repository stays the baseline and is not changed for the new project.
Shared code is copied at the start (simulator, numerics, transports) and the
dialects ported; whether they become a shared library is decided after M0.
Results are reported side by side: the same model on the same bitstream, the
IREE path against the Triton path, for correctness, speed, code size and
development time.

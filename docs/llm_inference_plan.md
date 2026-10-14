# LLM inference: architecture plan and implementation steps

> **PYNQ-Z1 paths**: this document records work done on the PYNQ-Z1. The `RISCV-on-PYNQ-Z1/...` (block design, constraints), `bitstreams/...` (builds verified on the board) and `iree-sa/l0` it mentions are on the `pynq-z1` branch (tag `v1.0-pynq-z1`); on main, PicoRV32 and its IP are in `rtl/picorv32/` and `rtl/ip/`, and the current overlay is in `boards/kv260/`.

Status: **L0–L5 and L5b done and verified on the board**. stories15M runs entirely on the device, bit-identical to the reference model:
- single sequence 15.1 tok/s (wall clock; device 18.9 tok/s);
- 8 sequences batched 63.3 tok/s (device 77 tok/s);
- teacher-forced top-1 agreement with fp32 96.2%.

L6-IREE (the compiler) is mostly done: PyTorch / HuggingFace models compile through IREE with our own `sa` backend (two dialect levels, sahl / sahw), stories15M and SmolLM2-135M run on the board with logits bit-identical to the reference (stories15M 17.7 tok/s, SmolLM2 2.22 tok/s); both can generate text interactively on the board; larger models (C6) are verified on the sim: Qwen3-0.6B, all 28 layers, bit-exact per dispatch, with correct end-to-end generation. On the KV260 (2026-10-14): four unmodified HuggingFace models (SmolLM2-135M, Gemma-3-270M, Qwen3-0.6B, Llama-3.2-1B) run whole on the board with every dispatch on the accelerator ([`generality_implementation_plan.md`](generality_implementation_plan.md)), with interactive multi-turn chat for Llama-3.2-1B-Instruct and Qwen3; decode attention over long contexts (Qwen3 2048, Gemma 512) is done in the compiler and checked on the board (2026-10-14: prefill vs decode only bit-exact over a 2032-token prompt, board teacher forcing PPL 18.99 against fp32 18.77). Progress and the remaining work (C4 performance, C6 larger models) are in [`iree_compiler_plan.md`](iree_compiler_plan.md).

Not done yet: L5.5 (75 MHz). The implementation records and board results of each level are in
§4 (L0), §5.5 (L1), §6.9 (L2), §7.3 (L3), §8.7 (L4), §9.3 (L5), §11.1 (L5b), summarized in §13.

**Goal**: build on the PYNQ-Z1 a complete system **whose architecture matches mainstream industrial LLM inference accelerators**,
running a llama2-structured model (TinyStories) end to end on the board to generate text. The system has four parts:

- **host**: the ARM, the runtime;
- **command processor**: PicoRV32 running resident firmware;
- **accelerator**: DMA, an int8 matrix engine, an fp32 vector engine and a special function unit;
- **compiler**: targeting **MLIR/IREE**.

**Position on speed**: architectural alignment is the first goal. Speedups that resources allow and that are cheap are done anyway (§0).

**Clock**: **50 MHz is a target that must be met**; 75 MHz is an extra target after end-to-end works, a level of its own (L5.5).

Related documents:

- [`double_buffer_design.md`](double_buffer_design.md): the current accelerator; §3 data layout, §6 VE, §7 scoreboard, §8 instructions and descriptors;
- [`perf_counters_and_desc_dma_plan.md`](perf_counters_and_desc_dma_plan.md): performance counters and descriptor DMA (M5);
- [`iree_compiler_plan.md`](iree_compiler_plan.md): the detailed design of L6-IREE (PyTorch → MLIR/IREE → descriptor lists);
- [`../notebooks/llm/README.md`](../notebooks/llm/README.md): the first measurements. Conclusion: single-sequence decode is bandwidth bound; at D=16, 50 MHz, counting only GEMV gives 24.5 tok/s (stories15M), about the same as the ARM's int8 version at 22.9 tok/s.

---

## Contents

- [0. Principles](#0-principles)
- [1. Comparison with industrial architectures](#1-comparison-with-industrial-architectures)
- [2. Overall architecture and the data flow of one token](#2-overall-architecture-and-the-data-flow-of-one-token)
- [3. Numeric scheme](#3-numeric-scheme)
- [4. L0: baselines, reference model, functional simulator and toolchain verification](#4-l0-baselines-reference-model-functional-simulator-and-toolchain-verification)
- [5. L1: host interface and command-processing extensions](#5-l1-host-interface-and-command-processing-extensions)
- [6. L2: fp32 vector engine and special function unit](#6-l2-fp32-vector-engine-and-special-function-unit)
- [7. L3: quantization format and GEMM post-processing](#7-l3-quantization-format-and-gemm-post-processing)
- [8. L4: one decoder layer executing on the device](#8-l4-one-decoder-layer-executing-on-the-device)
- [9. L5: ARM runtime and end-to-end generation](#9-l5-arm-runtime-and-end-to-end-generation)
- [10. Compiler interface: MLIR / IREE](#10-compiler-interface-mlir--iree)
- [11. L5b, L5.5, L6: concurrent sequences, 75 MHz and later extensions](#11-l5b-l55-l6-concurrent-sequences-75-mhz-and-later-extensions)
- [12. Resource and timing budget](#12-resource-and-timing-budget)
- [13. Overall plan and acceptance](#13-overall-plan-and-acceptance)
- [14. Risks and mitigations](#14-risks-and-mitigations)
- [Appendix A: summary of encoding changes](#appendix-a-summary-of-encoding-changes)
- [Appendix B: new and modified files](#appendix-b-new-and-modified-files)

---

## 0. Principles

1. **Architectural alignment first; cheap speedups are done anyway**. The array goes back to D=8, giving the LUTs to the fp32 vector unit. This costs single-sequence decode nothing:
   decode is bandwidth bound, and EX at D=8 consumes exactly 8 weights per cycle, matching LD's about 7.9 B/cycle.
   Where resources allow:
   - the VE's elementwise operations are pipelined (§6.7);
   - scheduling overlaps LD, EX and VE;
   - the paths for concurrent sequences and prefill are kept (§11).
2. **Everything can be verified bit for bit**. A Python **functional simulator** executes descriptor lists directly.
   - Board results, RTL co-simulation and the functional simulator are compared bit for bit;
   - accuracy is evaluated in only one place, "functional simulator vs the fp32 reference model".
3. **Measure before deciding**. The quantization scheme's accuracy is first measured on the host with the reference model (L0), and only then is hardware built.
   Whether the IREE runtime runs on 32-bit ARM is also verified in L0.
4. **Design the interfaces for the compiler**. The hardware provides the abstractions IREE needs: bindings, push constants, dynamic shapes, command buffers,
   barriers, semaphores (§10). Hand-written list generators come first (L3–L5) and are later replaced by an IREE target backend; both output the same descriptors.
5. **Clock**:
   - existing modules get no timing changes; they already have margin at 50 MHz;
   - **newly written modules** (fp32 units, SFU, the fetch unit's new decoding) **are all pipelined for 13.3 ns (75 MHz)**,
     using only some more FFs (34% of FFs are used today), leaving room for L5.5.
6. **Each level can go on the board alone**, with its own board script and acceptance criteria, as in M1–M5.
7. **Backward compatibility**:
   - the semantics of existing instructions, descriptors and the integer VE are unchanged;
   - new functions come in through new opcodes or keys, formerly reserved bits and new CAPS bits;
   - all existing tests are kept.

---

## 1. Comparison with industrial architectures

The common structure of mainstream inference accelerators (data-center NPUs, phone NPUs, AI chips with RISC-V control cores), and this project's counterparts:

| Layer | Common industrial practice | This project today | This plan |
|---|---|---|---|
| Compiler | Framework → an IR such as MLIR → operator fusion, tiling, layout → device code (e.g. IREE, XLA, TVM) | Hand-written firmware and drivers | IREE: a template library first, then full code generation (§10) |
| Host runtime | Load the model, allocate the KV cache, schedule prefill/decode, tokenization, sampling | Python driver, one operator per call | L5 brings it up in Python; L6-IREE replaces it with an IREE HAL driver in C |
| Host ↔ device interface | Command queues in memory, doorbells, completion interrupts or semaphores | mailbox parameters, one firmware per operator, ARM polling | L1: DDR command ring, doorbell, completion records, PL→PS interrupt |
| Command processor | An on-device microcontroller (many chips use RISC-V) fetches commands, dispatches, synchronizes | PicoRV32, firmware swapped per operator | L1: resident firmware `rt_fw` |
| Command buffers and executables | Record once, replay with changed parameters (CUDA Graph, Vulkan command buffers); bindings and push constants; position-independent kernels | Descriptor lists + 4 BASEs (M5) | L1: 16 BASEs, 8 PARAMs, dynamic fields, SETREG, LOOP, CALL/RET, LDPARAM |
| Matrix engine | int8/fp8/bf16 systolic arrays or tensor cores | int8 16×16, int32 accumulation | Keep int8, change to D=8 |
| Vector unit and special function unit | fp32/bf16 vector ALUs, reductions, table-based exp/rsqrt etc. | Only integer elementwise operations and requant | L2: fp32 VPU and SFU, plus reductions, transpose and masks |
| Quantization | W8A8: weights per channel, activations dynamic per token; KV cache int8/fp8 with static scales | A single integer scale only | L3: the same scheme |
| Attention | KV cache in device memory; in decode the whole attention runs on the device | None | L4: q·Kᵀ, softmax, p·V all on the device |
| Observability | Hardware counters, tracing | 32 counters (M4p) | Kept, with new VPU and fetch events |

**Not done**:
- multiple cores and a network on chip;
- HBM;
- paged KV cache (optional in L6);
- continuous batching;
- fp8/bf16 matrix multiplication.

These are differences of scale, not of structure.

**EX stays int8**:
- W8A8's int32 accumulation is exact, and the scale is multiplied once in the VE post-processing, mathematically equivalent to dequantizing first and multiplying in floating point;
- 64 fp32 multiply-add units would need about 32k LUT and 128 DSPs (estimate), which do not fit.

---

## 2. Overall architecture and the data flow of one token

```
┌──────────────── ARM Cortex-A9 (host) ────────────────┐
│ Runtime: Python LlamaDevice in L5 / IREE + HAL driver in L6 │
│  model loading → DDR │ KV cache │ static descriptor lists (one per model) │
│  per token: write the parameter block (PARAM) + a ring entry │ tokenization, sampling │
└──────┬───────────────────────────────────▲───────────┘
       │ ① write the entry + flush          │ ⑤ IRQ (PL→PS), read the completion record
       │ ② doorbell: mailbox RING_TAIL      │
┌──────▼───────────────────────────────────┴───────────┐
│ PicoRV32 resident firmware rt_fw (command processor)    │
│  ③ fetch the entry → set BASE0-3, PARAM0-7 → mat_submit → mat_fence │
│  ④ write the completion record → mat_notify            │
└──────┬───────────────────────────────────────────────┘
       │ PCPI
┌──────▼─────────────── accelerator sa_unit (D = 8) ──────┐
│ Descriptor fetch: BASE0-15 │ PARAM0-7 │ dynamic fields │ SETREG │ │
│             LOOP │ CALL/RET │ LDPARAM                  │
│ → scheduler (scoreboard)                                │
│   LD / ST (DMA, HP2) │ EX (int8 array) │ VE (int + fp32 VPU/SFU) │
│   SPAD_A 128 KB │ SPAD_B 128 KB │ ACC 256 KB (int32 / fp32) │
└──────────────────────────────────────────────────────┘
DDR: weights (int8 + scales), KV cache (int8), RoPE tables, embedding table, logits, lists, command ring, parameter blocks
```

**The execution of one token**:

1. After the ARM samples the previous token, it writes a 32-byte parameter block: pos, pos+1, pos_pad, token×dim, etc. (§8.5).
   **The list itself is not changed**;
2. submit a command-ring entry, then ring the doorbell;
3. `rt_fw` loads the parameter block into the PARAMs and executes the model's static list. Each layer in the list is in turn:
   RMSNorm → QKV → RoPE → append KV → attention → Wo → residual → RMSNorm → SwiGLU FFN → residual;
   layers are chained with LOOP; finally the classifier;
4. the logits (fp32) are written back to DDR;
5. `rt_fw` writes the completion record, then raises the interrupt;
6. the ARM samples.

---

## 3. Numeric scheme

### 3.1 Formats

| Data | Format | Location |
|---|---|---|
| Linear-layer weights W (out × in) | int8 symmetric, **one fp32 scale per output channel** `s_w[o] = max|W[o,:]| / 127`; stored transposed in DDR as Wᵀ (in × out), used as the B operand | DDR |
| Linear-layer input activations | int8 symmetric, **computed dynamically per token** `s_x = max|x| / 127`, on the device | SPAD |
| Matrix multiply-accumulate | int32 | ACC |
| Residual stream, intermediates of norm, RoPE, softmax, SiLU | **fp32** | ACC |
| KV cache | int8, **one static scale each for K and V per layer** (calibrated offline in L0) | DDR |
| q (attention) | int8, dynamic scale per head | SPAD |
| softmax probabilities p | int8, static scale 1/127 | SPAD |
| Embedding table | int8, one fp32 scale per row | DDR |
| RMSNorm weights, RoPE cos/sin tables | fp32 | DDR |
| logits | fp32 | DDR |

This is the SmoothQuant-style W8A8 approach plus an int8 KV cache, the common combination of industrial int8 inference.
llama2.c's Q8_0 (one scale per group of 32 elements along K) would need K split and accumulated separately on a systolic array,
so this plan does not use it; if L0 finds per-channel quantization insufficiently accurate, it is a fallback (§14).

### 3.2 Conventions for fp32 arithmetic

Hardware, functional simulator and compiler all implement this convention. It is also the numeric convention given to IREE: declared as fast-math (§10.4).

- Addition and multiplication follow IEEE-754 binary32 with **round to nearest even (RNE)**.
- **Flush to zero for denormals (FTZ)**: denormal inputs and outputs are treated as a 0 of the same sign.
- Overflow gives ±inf. NaN propagates by the IEEE rules, but the RTL tests do not cover NaN.
- No fused multiply-add: `a·x + b` is one multiply and one add, rounded twice.
- Type conversions:
  - int32 → fp32 with RNE;
  - fp32 → int8/int32 also with RNE, then saturation; the int8 range is [−127, 127];
  - (optional) bf16 ↔ fp32: truncation, plus round to nearest even.
- The **order of additions in reductions is fixed** (§6.4).
- The special functions EXP, RECIP, RSQRT are implemented with fixed algorithms (§6.5), and the models implement **the same algorithms**, not libm.
  So division = RECIP plus a multiply, **not IEEE exact division**.

Under these conventions every fp32 step can be reproduced exactly with NumPy float32 plus FTZ handling.

---

## 4. L0: baselines, reference model, functional simulator and toolchain verification

L0 does not change the RTL.

### 4.1 D=8 baseline bitstream and timing survey

- Build `scripts/build_bitstream.sh -jobs 4 -sa_d 8`, keeping both PERF and DESC, at 50 MHz.
- Record resources as the budget baseline of L1/L2. M3 at D=8 was 21,720 LUT, 96 DSP, 130 BRAM36;
  with the counters and the fetch unit about 24k LUT estimated.
- **Timing survey**: one OOC synthesis of the same RTL under a 75 MHz constraint, recording the 10 worst paths as input to L5.5.
  m5's (D=16) known worst path is `ex/drain_active` driving the enables (CE) of all PE registers:
  16.8 ns, 15.8 ns of it routing, a fan-out problem.
- Board regression: `m4_perf.py`, `m5_desc_demo.py`, `m4_demo.py` all run at D=8.
- Archived as `bitstreams/l0/`.

> **L0 measured** (`bitstreams/l0/`):
> - **Resources**: 23,452 LUT (44.1%), 19,193 FF, 96 DSP, 130 BRAM36;
> - **Timing**: WNS +1.555 ns at 50 MHz;
> - **Board regression all PASS**: `m4_demo` (256³ 59.0 MAC/cycle, 92% of peak), `m4_perf` 69/69, `m5_desc_demo`.
>
> **75 MHz OOC timing survey** (`rtl/sysarray`, `make synth CLOCK_NS=13.333`, D=8): WNS −0.717 ns, estimated maximum about 71.2 MHz.
> The 20 worst paths are **all in `sa_ld.v`**, in two kinds:
> 1. **Burst generation**: starting from `off`, the address is computed, then the minimum of three quantities decides the burst length (what remains of the row, the distance to the 4 KB boundary, 16 beats),
>    then multiplied by `step` (through a DSP) to give `cur_word`. 17 logic levels, 13.9 ns.
> 2. **The write-back address of read data**: starting from `q_rp`, computing `lw_word = h_word + (h_sum >> lpwl) * step` (through a DSP),
>    driving the SPAD/ACC address, data and write enables. 6 logic levels, 13.5 ns, about 60% routing.
>
> Both have the `* step` multiply in a combinational path. Fix: replace the multiply with an accumulator incremented per beat; register the write-back address, data and enables one more cycle.
> These two are L5.5's fix list. The full build also adds PicoRV32 and the AXI interconnect; L5.5's actual build decides.

### 4.2 Reference model `llm/ref_model.py`

Reads llama2.c's fp32 checkpoints and implements two forward computations in NumPy:

- **fp32 mode**: the same order of computation as `run.c`.
  - Acceptance: stories15M greedy decoding of 256 tokens gives text identical to `run.c`.
- **Device numeric mode**: simulates the device's computation with the quantization of §3.1. Special functions first use float64 approximations, replaced by the functional simulator after L2.

Accuracy evaluation (`llm/eval_quant.py`):
- **teacher forcing**: both modes are fed the token sequence generated by the fp32 mode, comparing at each position whether their top-1 agree,
  and computing the KL divergence of the logits.
- Models: stories15M, 42M, 110M;
- Data: about 8 prompts, 256 tokens each;
- Also calibrates each layer's static KV scales: absolute maximum or the 99.99th percentile, both tried.
- **Acceptance threshold**: top-1 agreement ≥ 95% for 15M, and fluent greedy-generated text.
  - If not met, change the scheme in the order of §14;
  - this 95% is a provisional value, written into this document after L0's measurement as the accuracy baseline of later levels.

> **L0 measured** (host, `llm/eval_quant.py`, details in [`llm/README.md`](../llm/README.md)):
> - All three models and both KV calibrations meet the 95% threshold; the device model's argmax is within fp32's top 5 100% of the time;
> - top-1 agreement: 15M 95.6% (absmax) / **96.3%** (99.99th percentile), 42M 96.0% / 95.9%, 110M 97.9% / 98.1%;
> - mean KL divergence: 15M about 0.011, 42M 0.007–0.010, 110M about 0.004.
>
> **Accuracy baseline set to**: stories15M, KV scales at the 99.99th percentile, top-1 96.3%.
> After L2 replaces the special functions with the hardware algorithms it is re-evaluated, and must still be ≥ 95%.

### 4.3 Functional simulator `llm/sa_funcsim.py`

Simulates the execution of descriptor lists in Python:

- **State**: DDR (a byte array), SPAD_A, SPAD_B, ACC, stored by words with the layout of design document §3.4;
  plus the BASE and PARAM registers.
- **LD / ST**: LINEAR, INTERLEAVE, ZERO_PAD.
- **EX**: including repeat, bstep, cstep, crow, accumulate.
- **VE**: the existing integer operations; L2 adds all fp32 operations.
- **Control descriptors**: FENCE, JUMP, END, BASE relocation; L1's extensions (§5.3) too.
- **Cycle estimation model**: used by the compiler as a cost model. First estimated from each command's theoretical cycles, calibrated with the counters after L1.

**Acceptance**:
- with the 7 cases of the existing `gen_desc_cases.py` as input, the functional simulator's output is bit-identical to the bytes expected from RTL co-simulation;
- a few lists extracted from `m5_desc_demo.py` compared with board results.

From then on, every new hardware function is **first implemented in the functional simulator with test vectors**, before writing RTL.

### 4.4 IREE toolchain verification (the largest software risk, placed first)

- **Host**: install or build the IREE compiler. Compile a small torch model (a 2-layer MLP plus one softmax) with iree-turbine to the `llvm-cpu` target,
  with target triple `armv7-linux-gnueabihf`.
- **Board**: cross-compile the IREE runtime (armhf, hard-float) and run the model above with `iree-run-module`, matching PyTorch.
  IREE is mostly tested on aarch64 and 32-bit ARM is not guaranteed; this step confirms it works.
- **Also confirm**:
  - the interfaces of the plugin mechanisms for custom HAL drivers and target backends in the chosen IREE version;
  - how the C runtime accesses CMA buffers (u-dma-buf or PYNQ's CMA interface) and interrupts (UIO).
- **If it fails**: use only the IREE compiler, and write a small executor in C as the runtime, reading the executables and scheduling information IREE outputs.
  The hardware interface of §10 is unaffected.

> **L0 measured: passes** (IREE 3.11.0, details in [`iree-sa/l0/README.md`](../iree-sa/l0/README.md)).
> - **Flow**: torch → iree-turbine → iree-compile (llvm-cpu, armv7) → the IREE runtime on the board
>   (cross-compiled with clang-18, statically linked).
> - **Results**: the test model matches torch with both the local-sync and local-task drivers, and all 15 single-operator tests PASS.
>
> Two armv7-specific problems were solved along the way, both with a linker wrapper script plus a tiny libm shim:
> 1. **Missing `fmaxf`/`fminf`**: the Cortex-A9 has no IEEE-semantics max/min instructions, so LLVM calls these two functions,
>    and the embedded ELF has no libm.
> 2. **`fmaf` in IREE's bundled musl is an empty implementation (`unreachable`)**: the Cortex-A9 has no FMA, `exp`'s polynomial approximation calls it,
>    and `exp` always returned the constant 4. The shim provides a correctly rounded software `fmaf`, bit-identical to glibc on 40 million inputs.
>    This may be an IREE bug worth reporting upstream.
>
> The build script checks every armv7 executable on the host, allowing neither external imports nor empty libm implementations.

---

## 5. L1: host interface and command-processing extensions

L1's hardware changes have two parts, built together:
- **command ring and completion interrupt**: for the host runtime;
- **command-processing extensions**: for the compiler, covering IREE's bindings, push constants, command buffers and dynamic shapes.

> **L1 done and verified on the board** (`bitstreams/l1/`).
> - **Simulation**:
>   - 272 functional-simulator checks;
>   - 15 new L1 checks in `tb_sa_unit`; all 8 deliberately introduced bugs caught;
>   - two Python-generated L1 lists pass RTL co-simulation, at both D=8 and 16;
>   - `RING_TEST`: 9 entries submitted to a command ring of size 4; all 3 deliberately introduced firmware bugs caught;
>   - no old test regresses.
> - **Board**: all 6 tests of `l1_ring_demo.py` pass; `m4_demo`, `m4_perf`, `m5_desc_demo` do not regress.
> - **Differences from the plan**:
>   1. the completion interrupt became an edge interrupt `notify_irq`, because the ARM cannot access the accelerator's CSRs;
>   2. the driver must reuse a command-ring slot only after reading out its completion record. On the board it turned out that judging ring-full by `RING_HEAD` alone
>      let the firmware overwrite completion records not yet read; fixed.
>   3. the fetch unit measured 4.4k LUT, above the estimate.

### 5.1 Command ring

**Submission ring**: allocated by the ARM in DDR, uncached, or flushed after every write. The number of entries is a power of 2, 64 by default;
64 bytes per entry.

| Word | Content |
|---|---|
| w0 | [7:0] type: 0x01 RUN_LIST, 0x02 NOP, 0x03 RESET (continue after `mat_reset`); [8] interrupt on completion; [9] dump counters; [63:32] `seq` |
| w1 | List address (64-byte aligned) |
| w2 | [31:0] count (0 means run until END) |
| w3–w6 | BASE0–BASE3 |
| w7 | [31:0] **parameter block address**: 8 32-bit PARAMs, 32-byte aligned; 0 means do not load |

BASE4–15 are set by SETREG inside the list (§5.3).

**Completion record ring**: the same indices as the submission ring, 32 bytes per record.

| Word | Content |
|---|---|
| c0 | [31:0] `seq`; [63:32] status: 0 for success, otherwise `ext_status`, containing the error engine number and error code |
| c1 | [31:0] cycles; [63:32] number of descriptors executed (`DL_EXEC`) |
| c2 | [31:0] END's completion value (`DL_STATUS`); [63:32] reserved |
| c3 | Reserved |

**New mailbox fields** (from BRAM 0x1F00; 0xB0–0xD0 are free today):

| Offset | Name | Writer | Notes |
|---|---|---|---|
| 0xB0 | `RING_BASE` | ARM | Physical address of the submission ring |
| 0xB4 | `RING_SIZE` | ARM | Number of entries (a power of 2) |
| 0xB8 | `RING_TAIL` | ARM | **Doorbell**: the new tail pointer |
| 0xBC | `RING_HEAD` | Firmware | Number of completed entries (monotonically increasing) |
| 0xC0 | `CPL_BASE` | ARM | Physical address of the completion record ring |
| 0xC4 | `FW_STATE` | Firmware | 0x52554E00 `RUN\0` means ready; 0xDEADxxxx on error |
| 0xC8 | `FW_VERSION` | Firmware | Interface version |
| 0xCC | `HEARTBEAT` | Firmware | Idle loop count, for debugging |

**Doorbell**: the ARM writes `RING_TAIL`, and `rt_fw` polls it in its idle loop. This is a simplification:
real chips usually write an MMIO doorbell register that raises a device-side interrupt. Switching to interrupts is deferred to L6.

**Completion interrupt** (changed to an edge interrupt in the implementation; reasons below):
- A new instruction `mat_notify` (funct7=1, funct3=7): a new output of `sa_unit`, **`notify_irq`**, produces a pulse
  (high for 4 cycles, then low for at least 1; consecutive notifies queue so each has its own rising edge), and `NOTIFY_COUNT` increments.
- BD: `irqConcat` changes to `NUM_PORTS 2`, with `In1` connected to `matmul_0/notify_irq` (interface declared `EDGE_RISING`),
  through the existing `psInterruptController` (axi_intc) to `IRQ_F2P`. The axi_intc latches the edge, and the ARM acknowledges at the intc.
- **Why not the originally planned level interrupt with W1C**: in the current BD, the CSRs of `matmul_0` are only mapped in PicoRV32's address space (0x80000000),
  so the ARM cannot access them, nor write 1 to clear a status. An edge interrupt needs no state cleared in the accelerator, similar in nature to PCIe's MSI message interrupts.
- ARM side: Python uses `pynq.Interrupt("matmul_0/notify_irq")`, the C runtime UIO; both keep a fallback path polling `RING_HEAD`.
  Several edges may merge into one interrupt before the ARM acknowledges, so after waking the ARM processes all completed entries by `RING_HEAD`.
- Why the firmware raises the interrupt: the firmware must finish writing the completion record first, so the ARM reads a consistent state when woken.
  The IRQ bit of the END descriptor is left for direct submission from the ARM (L6).

**New CSR** (AXI-Lite, accessible only to PicoRV32): 0xD4 `NOTIFY_COUNT` (read-only).
**CAPS bit 22** means `mat_notify` and `notify_irq` are present.

### 5.2 Firmware `firmware/rt/rt_fw.c`

```c
init: check CAPS (needs bit 21 DESC, bit 22 NOTIFY, bit 24 CMDX); FW_STATE = RUN;
loop:
  while (RING_HEAD == RING_TAIL) HEARTBEAT++;
  e = &ring[RING_HEAD % RING_SIZE];               // PicoRV32 reads DDR through HP0
  switch (e->type):
    RUN_LIST: mat_cfg(BASE0..3); if (e->params) mat_cfg(PARAM0..7 ← parameter block);
              t0 = rdcycle; mat_submit(e->list, e->count); mat_fence();
              st = ext_status;  if (error) mat_reset();
    NOP:      st = 0;
    RESET:    mat_reset(); st = 0;
  write the completion record;  [optional] dump counters;  RING_HEAD++;  if (e->flags & IRQ) mat_notify();
```

- The firmware is **resident**: the ARM loads it once and never swaps firmware afterwards.
- The program must stay below 0x1E00; this loop is small, estimated within 1 KB.
- The existing `gemm_fw`, `vector_fw`, `desc_run_fw` are kept for the old scripts.

### 5.3 Command-processing extensions (for the compiler)

All extensions are in `sa_cmdfetch.v` and take effect at decode. Fetch and decode are already in order, so later descriptors naturally see new register values;
addresses and lengths in command packets already sent are already computed and unaffected by later changes.

| # | Extension | Encoding | Purpose |
|---|---|---|---|
| 1 | **BASE0–15** | BASESEL widened to 4 bits: `{w0[14:13], w0[10:9]}`. Old lists have w0[14:13] = 0, same meaning | An IREE dispatch often has 5–8 bindings |
| 2 | **PARAM0–7** (32-bit) | `mat_cfg` keys 27–34; the command ring's parameter block; SETREG; LDPARAM | Dynamic shapes and positions, corresponding to IREE's push constants |
| 3 | **Dynamic fields** | w0[31:16] holds two 8-bit dynamic slots (formerly reserved): `[3:0]` field number (0 = unused), `[6:4]` which PARAM, `[7]` 0 = replace, 1 = add | Lengths, counts and address offsets can come from PARAMs, so one model needs only one static list |
| 4 | **SETREG** (opcode 0x14) | w1: 3 7-bit items at [6:0], [14:8], [22:16], each {[6] valid, [5:0] register number: 0–15 = BASE, 16–23 = PARAM}; w2–w4: the matching values; w5[2:0]: per item replace (0) or add (1); written in item order | Binds parameters inside a list, like Vulkan's bind descriptor set and push constants |
| 5 | **LOOP_END** (opcode 0x13) | w1: the jump-back target (relative offset in descriptors, signed); w2[15:0]: count (≥ 1, can also come from a PARAM through a dynamic slot); w2[18:16] / w2[21:19]: PARAMs k1 / k2 incremented each round; w3 / w4: their strides (signed 32-bit) | Tile loops, per-head loops, per-layer loops; corresponds to IREE's workgroup count |
| 6 | **Relative JUMP, CALL, RET** | JUMP / CALL (0x15): with w2[0] = REL, w1 is a signed offset (in descriptors), otherwise a 64-byte-aligned absolute address; CALL pushes the address of the next descriptor onto a return stack of **depth 4**; RET (0x16): pops and jumps | Position-independent executables: a command buffer = SETREG … CALL some kernel |
| 7 | **LDPARAM** (opcode 0x17) | w1: DDR address (relocatable, 4-byte aligned); w2[2:0]: which PARAM to write; w3[15:0]: multiplier; w4: addend. Result `PARAM = mem32 × mul + add` | Data-dependent addresses: embedding lookup, KV update by the position tensor, gather; if the data was just written by ST, a FENCE must precede it |

**LOOP semantics**:
- The loop body precedes LOOP_END and executes `count` times; count must be ≥ 1;
- the counter stack has depth 2, and a LOOP_END identifies its level by its own address: if the top of the stack has the same address as the current LOOP_END, the top is used;
  otherwise a new counter is pushed; when it counts down to 0 it pops and execution continues;
- at the end of each round `P[k1] += s1`, `P[k2] += s2`; after the loop the PARAMs keep their final values, and the compiler resets them with SETREG when needed;
- jumping back discards prefetched descriptors and refetches, reusing JUMP's implementation; each round rereads the loop body from DDR, with no loop buffer.

**Dynamic field numbers** (final after L1's implementation, consistent with `sa_cmdfetch.v` and `DYN_FIELDS` in `llm/sa_funcsim.py`):

| opcode | Field numbers |
|---|---|
| LD / ST | 1 DDR address, 2 LADDR (word number), 3 rows, 4 row_bytes, 5 pitch |
| EX | 1 A, 2 B, 3 C, 4 Kt, 5 repeat, 6 bstep, 7 cstep |
| VE | 1 src1, 2 src2, 3 dst, 4 LEN, 5 VALID, 6 ROWLEN, 7 P1, 8 P2, 9 A (fp32 bit pattern), 10 B, 11 IMM |
| LOOP_END | 1 count |
| LDPARAM | 1 address |
| JUMP / CALL | 1 target |
| SETREG | 1–3 the values of items 1–3 |

- Address-type fields are computed in the order `BASE[sel] + field + PARAM`;
- values after replacement or addition still pass the scheduler's existing range checks; out of range raises an error.

**Errors**: the following raise an error with error engine number FETCH (4); recovery is still `mat_reset`:
- return-stack or counter-stack overflow or underflow;
- a count of 0;
- an undefined field number;
- an AXI error on LDPARAM's read.

**CAPS bit 24 (CMDX)** means all these extensions are present.

**New `mat_cfg` keys**: 15–26 set BASE4–15, 27–34 set PARAM0–7.

### 5.4 Driver

- `driver/pynq_matmul.py` gains `class Device`:
  - `Device(bitfile)`: loads `rt_fw`, allocates the two rings, waits for `FW_STATE == RUN`;
  - `submit(dl, bases, params=None, irq=True) -> seq`;
  - `wait(seq)` and `async wait_async(seq)`;
  - `run(dl, bases, params)`: a convenience for synchronous calls.
- `DescList` gains `setreg`, `loop_end`, `call`, `ret`, `ldparam`, and the dynamic-slot argument `dyn=[(field, param, add)]`.

### 5.5 Verification

- **`tb_sa_unit`**:
  - `mat_notify`: one 4-cycle pulse each time, two in a row give two separate rising edges, `NOTIFY_COUNT` counts, the legacy `irq` is unaffected;
  - the negative test used funct3=7 as an illegal encoding; it is now taken, so funct7=3 is used instead;
  - **directed tests of the command-processing extensions**:
    - replace and add for each dynamic field;
    - SETREG takes effect on later descriptors without affecting commands already sent;
    - LOOP with 1 and N iterations, two nested levels, PARAMs incremented each round;
    - CALL/RET nested to depth 4;
    - LDPARAM's multiply-add, and the order "ST → FENCE → LDPARAM";
    - each kind of error and recovery;
    - mixed with FENCE, END, BASE0–15;
  - **random lists**: with dynamic slots, loops and calls, compared bit for bit with the functional simulator.
- **New `RING_TEST` in `tb_system.v`**: N entries in the ring, covering wrap-around, NOP, a deliberately failing list, RESET, and entries with parameter blocks;
  checking every completion record and the number of interrupts.
- **On the board**, `notebooks/llm/l1_ring_demo.py`:
  - submit 200 lists asynchronously, all results correct, interrupt count equal to completion count;
  - with one static list containing LOOP and dynamic fields, give correct results for inputs of different lengths by changing only the parameter block;
  - measure the submission-to-completion latency, recorded only.

---

## 6. L2: fp32 vector engine and special function unit

### 6.1 The operation pipeline (fp32 mode)

VE commands gain an **FP bit**. With FP=0 the behavior is exactly as today; with FP=1 each group (VL = D elements) passes in turn through:

```
src1 ─ fetch (index mode M1) ─ to fp32 ─ [SWAPNEG] ─┐
                                                  ├─ OP ─ ×A + B ─ FUNC ─ [RELU] ─┬─ output conversion → dst
src2 ─ fetch (M2, or immediate IMM) ─ to fp32 ─────┘                              └─ reduction → dst
```

| Stage | Content |
|---|---|
| Input | src1 and src2 can be I8 (SPAD), I32 or F32 (ACC), converted to fp32 after reading; src2 can also be an fp32 immediate `IMM` in the command |
| SWAPNEG (src1, optional) | Each adjacent pair (2i, 2i+1) becomes (−x[2i+1], x[2i]), for RoPE |
| OP | ADD, SUB, MUL, MAX, MIN, COPY (same encoding as the integer mode) |
| Affine | y = y × A + B, with A, B fp32 immediates, defaulting to 1.0 and 0.0 |
| FUNC | NONE, EXP, RECIP, RSQRT, ABS |
| RELU | Same as the existing bit |
| Output | F32 written to ACC; I32 converted with RNE and saturated, written to ACC; I8 converted with RNE, saturated to ±127, written to SPAD; or into the reduction |

New type encoding **VT_F32 = 3**, only in ACC. (Optional) VT_BF16 = 4 in SPAD, laid out like int16, only a storage format and conversions.

### 6.2 Index modes

Unified for src1 and src2. Let the output group number be g:

| Mode | src word | Use |
|---|---|---|
| LINEAR | g | Ordinary elementwise operations |
| MOD P | g mod P | Column broadcast: per-channel scales, RMSNorm weights, RoPE tables (the existing period mode) |
| DIV P | ⌊g / P⌋ | Row broadcast: one scalar per row (max, sum, s_x); **row replication**: with P = D a vector expands into the D rows of an A strip (§8.2) |
| IMM (src2 only) | — | An fp32 constant in the command |

Each source has its own P: src1 uses the new field P1, src2 the existing period field P2.

### 6.3 TRANSPOSE (op=6)

Transposes a matrix of "R rows × S words", S·D elements per row, in D×D blocks:
- word i of output block b = element i of each word of input block b;
- word j of input block b is at `src + (⌊b/S⌋·D + j)·S + (b mod S)`;
- word i of output block b is at `dst + b·D + i`.

Effect: a row-major K (pos × hs) turns directly into the matrix engine's B-strip layout (Kᵀ, one segment per D positions), see §8.3.
- Supports I8 (SPAD → SPAD) and F32/I32 (ACC → ACC);
- in hardware a D×D register buffer, with a throughput of one word per cycle.

### 6.4 Reductions and tail masks

REDUCE ∈ {NONE, SUM, MAX}, per row: a row = `ROWLEN` groups. Within a row, accumulate per lane first, then merge the D lanes.
**Each row outputs a single fp32 word, broadcast to all D lanes**, so later DIV-mode accesses can take it per row directly.

- **VALID**: only the first VALID elements of each row are valid; 0 means all are valid.
  - invalid elements take the identity in reductions: 0 for SUM, −inf for MAX;
  - in elementwise output, invalid elements are written as 0.
- **Order of additions** (normative; the models must follow it):
  1. each lane accumulates **sequentially** in group order: `acc = ((v0 + v1) + v2) + …`;
  2. then the D lanes are merged by a binary tree, pairing by lane number: `(l0+l1)+(l2+l3)…`.
- Reductions take the multi-cycle path (§6.7), each group waiting out the adder latency, about one group every 3–5 cycles, in exchange for simple hardware and a deterministic order.

### 6.5 Special functions

| Function | Algorithm | Target accuracy (vs float64) |
|---|---|---|
| EXP | t = x·log2(e), split into n = ⌊t⌋ and f = t − n; 2^f by linear interpolation in a 64-segment table T[i] with slopes S[i]; the result multiplied by 2^n (added directly to the exponent); x < −87 gives 0, x > 88 gives +inf | Relative error ≤ 3e-5 |
| RECIP | x = m·2^e; initial 1/m from linear interpolation in a 64-segment table; one Newton step r = r·(2 − m·r); finally the exponent negated | Relative error ≤ 2e-7 |
| RSQRT | m reduced to [1, 4) by the parity of the exponent; initial value from a table; one Newton step r = r·(1.5 − 0.5·m·r²) | Relative error ≤ 5e-7 |
| ABS | Clear the sign bit | Exact |

The tables are generated by `llm/sfu_tables.py`, shared by the RTL and the functional simulator.
Other activation functions are compositions of these basic functions, e.g. sigmoid(x) = RECIP(1 + EXP(−x)), SiLU(x) = x·sigmoid(x).

### 6.6 Encoding

**VE descriptors**: an extension of the M5 format. With FP=0 the new fields must be 0, and old lists keep their meaning.

| Word | Existing content | New |
|---|---|---|
| w1 | src1, src2 LADDR | — |
| w2 | dst LADDR, LEN | — |
| w3 | [7:0] op (op 6 = TRANSPOSE), [13:8] types ([1:0] input, [3:2] output, [5:4] src2 type, see §6.9), [31:16] P2 (period), [47:32] scale, [52:48] shift | [53] FP, [56:54] FUNC, [58:57] M1, [60:59] M2 (3 = IMM), [62:61] REDUCE, [63] SWAPNEG |
| w4 | [31:0] zp, [63:32] lo | — |
| w5 | [31:0] hi | [63:32] IMM (fp32) |
| w6 | Reserved | [31:0] A (fp32), [63:32] B (fp32) |
| w7 | Reserved | [15:0] ROWLEN, [31:16] VALID, [47:32] P1, [63:48] S (transpose row width, in words) |

- **New PCPI `vec_cfg` keys**: 10 FLAGS (corresponding to w3[63:53]), 11 IMM, 12 A, 13 B, 14 ROWLEN/VALID, 15 P1/S.
- The scheduler command packet `SA_PKT_W` widens beyond 261 bits; the scheduler FIFO uses LUTRAM, so widening does not affect BRAM.
- **CAPS bit 23** means an fp32 VE is present.

### 6.7 Implementation: a hybrid structure

Each lane's data path:

```
cvt(src1) ─ SWAPNEG ─┐
                      ├─ [M1 mul | A1 add/sub | compare] ─ [M2 mul A] ─ [A2 add B] ─┬─ cvt/saturate ─► output FIFO
cvt(src2)/IMM ────────┘                                                            │
                                        FUNC ≠ NONE or REDUCE ≠ NONE: back to M2/A2 for multi-cycle micro-operations
                                        (lookup, interpolation, Newton steps, accumulation)
```

- **Pipelined part**: input conversion → OP → affine → output conversion. Each lane has 2 multipliers (M1, M2) and 2 adders (A1, A2).
  **Elementwise operations and the affine step process one group per cycle**: this is the path of the frequent operations post-processing, RoPE, residuals and quantization.
- **Multi-cycle part**: special functions and reductions **reuse M2 and A2**, stepped by one microprogram controller shared by all lanes.
  - EXP about 8 steps, RECIP/RSQRT about 10, reductions 1 step per group, plus adder latency;
  - meanwhile the read unit is paused by `fp_ready`;
  - special functions only occur in softmax, norms and SiLU, on few elements.
- **Why not make everything multi-cycle**:
  - for single-sequence decode a multi-cycle VE would take only about 6% more time;
  - but with concurrent sequences and prefill the VE's work multiplies with the number of rows while the weights read stay the same.
    E.g. decoding 8 sequences at once, a multi-cycle VE would take about 40% of the time, a pipelined one about 5% (estimate).
  - the resources fit (§12), so pipelined it is.
- **New files**:
  - `sa_fp32_add.v`, `sa_fp32_mul.v`, `sa_fp32_cvt.v`: written ourselves, not Xilinx IP, so xsim needs no IP simulation libraries;
    all **pipelined for 13.3 ns**: adder 5–6 stages, multiplier 3–4 stages, 2 DSPs per multiplier;
  - `sa_sfu_seq.v`: the microprogram controller and LUTRAM tables;
  - `sa_ve_fp.v`: D lanes, with index generation, reductions and the transpose buffer.
- **Relationship with `sa_ve.v`**: the read/write ports, command interface, FIFO credit counting, write-back stage and scoreboard interface are shared, the FP bit selecting the path.
  - the integer path is untouched, so M3's tests are unaffected;
  - FIFO credits increase from 8 to 16 to cover the longer pipeline.
- **Scheduler**: the **exact access ranges** in the new modes (DIV/MOD, IMM, TRANSPOSE, reductions):
  - the last word in DIV P mode = start + ⌊(groups − 1)/P⌋;
  - IMM mode has no src2;
  - TRANSPOSE computed by the block formula.
  These serve both the range checks and the scoreboard's bank masks. The scoreboard still tracks by bank; finer per-region tracking is optional, decided by data (§11).
- **New counter events**: originally planned `VE_FP_ACTIVE`, `VE_SFU_BUSY`, `VE_REDUCE`. None were actually added: the fp32 VE shares the existing VE events with the integer VE (§6.9).

### 6.8 Verification

- **Testbenches of the basic units**: `fp32_add`, `fp32_mul`, `fp32_cvt` with 10⁶ random inputs each, plus boundary values:
  ±0, denormals, ±inf, rounding exactly at .5, overflow. Compared bit for bit with NumPy float32 plus FTZ handling.
- **`tb_sa_ve_fp`**: test vectors generated by the functional simulator, compared bit for bit. Coverage:
  - every combination of OP, FUNC, index mode and type;
  - reductions (with VALID), transpose, SWAPNEG;
  - saturation and rounding of fp32 → int8.
- **SFU accuracy test**: a sweep of 10⁶ inputs; the maximum relative error must meet the targets of §6.5.
- **`tb_sa_unit`**: scoreboard conflict tests with the fp32 VE concurrent with EX and LD, and tests of the exact access ranges.
- **System level**: `gen_desc_cases.py` adds lists for softmax, RMSNorm, RoPE, transpose etc., co-simulated on `tb_system`.
- **On the board**, `notebooks/llm/l2_vpu_demo.py`: the same operators, board results bit-identical to the functional simulator.
- No existing test regresses: D=8 and 16, `PERF = 0`.

### 6.9 Implementation record (differences from the design above)

**Encoding additions**
- **src2 type**: types[5:4] gives src2's type separately: 0 same as src1, 1 I8, 2 I32, 3 F32.
  E.g. in dequantization src1 is the I32 accumulator result and src2 the F32 per-channel scales. Only valid with FP=1.
  The driver writes it as `DescList.ve(..., t2=...)`, the firmware as `SA_VT2(t)`.
- **PCPI path**: `sa_pcpi` keeps the values of keys 10–15, with A reset to 1.0.
  - only FP commands put these fields into the command packet; S is sent only with TRANSPOSE;
  - so integer VE commands are unaffected by earlier `vec_cfg` settings.

**Range rules** (identical in the scheduler and the functional simulator; both conservative)
- DIV-mode sources and REDUCE outputs are range-checked as G words, not by the exact formula of §6.7.
- The scoreboard still tracks by bank; this does not affect correctness, it only rejects a few legal commands, which the compiler must respect.
- TRANSPOSE requires (G/D) mod S = 0. The functional simulator checks this, the RTL does not (the result is undefined).
- LEN must be a multiple of ROWLEN. The functional simulator reports a SHAPE error, the RTL does not check.

**Structure** (`rtl/sysarray/sa_vefp.v`)
- The integer `sa_ve.v` stays as is; the fp32 part is a new parallel module `sa_vefp`:
  - `sa_unit` routes commands to one of the two by the FP bit and op=6;
  - the two are never busy at the same time, and their memory ports and events are ORed;
  - counter events reuse the existing VE events, with no new numbers.
- **Lane folding**: `sa_vefp` has only FL = D/2 physical fp lanes (a parameter; FL = D also possible).
  - a group of D elements is split in two halves entering the pipeline in two consecutive cycles, and the output is reassembled into one word;
  - reason: the D-lane version made sa_unit about 51k LUT after synthesis at D=8, which with PicoRV32 did not fit in the xc7z020 (placement failed);
    folded, sa_vefp drops from 28.9k to 17.6k LUT, sa_unit about 40k;
  - cost: binary operations already read twice per group, so their speed is unchanged; unary operations and the multi-cycle mode are half as fast;
  - the instruction set, numerics and functional simulator are unchanged; results remain bit-exact.
- **Pipelined mode**: FUNC is NONE or ABS, with no REDUCE.
  - one fixed-latency pipeline per lane: i2f ×2 → SWAPNEG → A1/M1/compare → M2 (×A) → A2 (+B) → RELU/VALID → f2i;
  - half a group issued per cycle, an output FIFO of 32 entries.
- **Multi-cycle mode**: EXP, RECIP, RSQRT or REDUCE.
  - one group at a time; after A2 the microprogram controller reuses each lane's M2, A2, i2f, f2i, executing the steps of `llm/sfu.py` half group by half group;
  - there is only one lookup ROM, read lane by lane;
  - then RELU/VALID, then the reduction (A2 or the comparator, finally the lane binary tree) or the output conversion.
- **TRANSPOSE**: a state machine reads D words with stride S into a D×D register buffer, then writes out D transposed words.
  Supports I8 (SPAD) and I32/F32 (ACC).
- **Basic units**: `sa_fp32_add.v` 4 stages, `sa_fp32_mul.v` 4 stages (DSP), `sa_fp32_cvt.v` i2f and f2i 2 stages each.
- **SFU tables**: `rtl/sysarray/gen_sfu_tables.py` generates `sa_sfu_tables.vh` from `llm/sfu.py`, one 256-entry ROM.
- **Measured SFU accuracy** (vs float64): EXP 1.84e-5, RECIP 1.18e-7, RSQRT 1.17e-7.

**Verification**
- `tb_fp32`: over 560k vectors, including RNE cases exactly at the midpoint; mutation tests catch errors.
- `tb_sa_vefp`: 31 cases at D=8 and 16, tested with both FL = D/2 and FL = D; mutation tests of the folding logic catch errors.
- `tb_sa_unit`: CAPS checks, and directed tests of fp32 and TRANSPOSE through PCPI `vec_cfg`.
- System-level `desc_run` co-simulation:
  - 4 L2 lists: softmax, GEMM + dequantization, RMSNorm + quantization, attention scores with TRANSPOSE;
  - 12 random fp32 lists with inputs containing NaN, inf, denormals;
  - expected values from the functional simulator, bit-identical with the RTL.
- On the board, `notebooks/llm/l2_vpu_demo.py`:
  - the same lists, plus a few hundred random fp32 lists per run;
  - commands rejected by the functional simulator must also raise errors on the device.
- **Board results** (`bitstreams/l2`: 44,005 LUT, 82.7%; WNS +1.836 ns at 50 MHz):
  - all 13 descriptor lists pass: softmax 4,235 cycles, GEMM + dequantization 783, RMSNorm + quantization 997, attention scores 524;
  - 300 random fp32 lists bit-identical to the functional simulator; error isolation passes;
  - the L1, M5 and M4 regressions all pass.

---

## 7. L3: quantization format and GEMM post-processing

### 7.1 Model export `llm/export_w8a8.py`

Exports llama2.c's fp32 checkpoint to the device's `*.w8a8` file: a header plus several sections, each 64-byte aligned.

| Section | Content |
|---|---|
| header | Magic, version, dim, hidden, layers, heads, vocab, seq_len, D, section offsets |
| embed | Embedding table int8 (vocab × dim), plus s_emb fp32 (vocab) |
| Per layer (all layers have the same structure, stored at a fixed stride so LOOP can advance addresses per layer) | rms_att, rms_ffn (fp32, dim); **Wqkvᵀ** (dim × 3·dim, int8) plus s_w; Woᵀ plus s_w; **W13ᵀ** (dim × 2·hidden, W1 and W3 concatenated by output channel) plus s_w; W2ᵀ plus s_w; static scales of K and V (written as immediates into that layer's descriptors) |
| final | rms_final (fp32); Wclsᵀ (dim × vocab) plus s_w |
| rope | cos and sin tables, fp32, seq_len × hs/2, each element stored once at each position of a pair (matching SWAPNEG's pairing), D aligned |

KV scales differ per layer, while LOOP repeats the same body for every layer. So each layer stores a small parameter section (fp32 bit patterns),
read into a PARAM with LDPARAM and substituted into VE's A field through a dynamic slot.

### 7.2 Command sequence of GEMM and post-processing

y = x·Wᵀ:

1. **Activation quantization** (x is an fp32 row vector):
   - `amax = REDUCE MAX(ABS(x))`;
   - `inv = RECIP(amax × (1/127))`, giving 127/amax;
   - `s_x = amax × (1/127)`;
   - `xq = x × inv` (src2 in DIV mode), output I8, with src1 in **DIV D** mode replicating it into the D rows of an A strip (§8.2).
2. **GEMM**: Wᵀ goes into SPAD_B in column blocks, the two banks **alternating**: while block i computes, block i+1 loads.
   A is the row-replicated xq; EX outputs to ACC (int32). The block loop is expressed with LOOP, and the block address advances per round through a PARAM.
3. **Post-processing** (overlapped with the next block's load):
   - `y = acc × s_w`: src2 in MOD mode, input I32, output F32;
   - `y = y × s_x`: src2 in DIV mode.

**L3 acceptance**:
- on the board, `l3_linear_demo.py` uses stories15M's real weights to compare the outputs of QKV, Wo, W13, W2 and the classifier layer by layer,
  bit-identical to the functional simulator, with errors versus the fp32 reference within the range calibrated in L0;
- **counter metric**: LD busy ≥ 90% for the classifier GEMV, showing that LD and EX really overlap.

### 7.3 Implementation record and results

**Files**
- `llm/export_w8a8.py`: exports and reads `.w8a8`. After the header comes a tensor table (name, offset, bytes, whether per layer).
  - Linear-layer weights are **pre-packed** as B strips: block t is the (in × D) row-major int8 of output channels [tD, (t+1)D);
  - so any run of consecutive blocks is contiguous in DDR and loads into SPAD_B with one LINEAR LD, with no strided DMA;
  - the packing depends on D, which the header records. stories15M at D=8 is 24.2 MB.
- `llm/compile_layer.py`: the hand-written list generator.
  - `quant_act`: 4 VEs computing amax, s_x, 1/s_x in turn, and finally outputting the row-replicated int8 A strip with src1 DIV D;
  - `linear`: NC output blocks per chunk. NC is the largest value that divides the number of output blocks while fitting one SPAD_B bank and the ACC staging area;
  - DDR addresses are all relocated through BASEs (BASE0 = model, BASE1 = input/output);
  - when the output goes to DDR and there are many blocks (the classifier), LOOP_END loops over a pair of blocks, with PARAM0/1 advancing the DDR addresses by a stride:
    the list has only 32 descriptors but executes 1046.
- `ref_model.SfuExact`: DeviceModel uses an SFU bit-identical to the hardware.
- `llm/test_l3.py`: host test. Runs layers 0, 5 and the classifier with real inputs, compared bit for bit with DeviceModel;
  `--save` outputs case files for the board test.
- `notebooks/llm/l3_linear_demo.py`: board test. L3 changes no hardware and uses `bitstreams/l2`.

**Verification**
- On the host, every layer is bit-identical to DeviceModel at D=8 and 16, with relative errors versus the fp32 reference of 0.37–3.27% (W2 the largest).
  The loop form and the unrolled form give exactly the same results.
- Mutation test: changing the dequantization order to multiply by s_x before s_w is detected as a mismatch in every layer.
- RTL co-simulation: `desc_run` gains a synthetic L3 linear layer (64 → 2880, forced into the loop form), passing at D=8 and 16.

**Board results** (stories15M, D=8, 50 MHz)

| Layer | in → out | Cycles | Weight B/cycle | EX useful | LD busy |
|---|---|---|---|---|---|
| Wqkv | 288 → 864 | 42,121 | 5.91 | 73.7% | 75.9% |
| Wo | 288 → 288 | 16,997 | 4.88 | 60.7% | 63.5% |
| W13 | 288 → 1536 | 67,517 | 6.55 | 81.8% | 84.0% |
| W2 | 768 → 288 | 36,319 | 6.09 | 75.9% | 78.4% |
| Classifier | 288 → 32000 | 1,258,768 | 7.32 | 91.5% | **93.9%** |

- The board results of 9 linear layers are bit-identical to both the functional simulator run on the board and the host's DeviceModel.
- **The classifier's LD busy is 93.9%; acceptance (≥ 90%) passes**. LD moves 7.91 B per busy cycle, close to the limit of one HP port.
- The small layers have LD busy of only 64–84%: the first block's load and the last block's post-processing cannot overlap, a large share when there are few blocks.
  L4 chains a layer's linear layers into one list, so the next linear layer's first block can be loaded ahead.
- A layer's 4 linear layers total about 163k cycles (3.3 ms), 6 layers about 19.7 ms; plus the classifier's 25 ms,
  the linear-layer part is about 45 ms/token (a ceiling of about 22 tok/s, excluding attention and other VE operations).

---

## 8. L4: one decoder layer executing on the device

### 8.1 On-chip memory allocation (D=8, illustrative; refined during L4's design)

| Region | Use |
|---|---|
| ACC bank 0 | Residual stream x (fp32, dim/D words); xn; q, k, v; attention output; FFN intermediate h (fp32); scalars (amax, s_x, sum etc., one per word) |
| ACC bank 1 | GEMM outputs (int32, replicated rows D × N); scores |
| SPAD_A | Row-replicated xq, q, p (int8); the current row of the RoPE table; various scale vectors |
| SPAD_B | Column blocks of Wᵀ (the two banks alternating); Kᵀ strips and V during attention |

The hand-written path (L4/L5) keeps the residual stream in ACC throughout. On the IREE path data between dispatches goes through DDR (§10.3).

### 8.2 The replicated-row layout

An A strip's layout is: word `w·D + g` = `A[g][w·D .. w·D+D−1]`.
If **all D rows hold the same vector v**, word `w·D + g` is the w-th segment of v, which is exactly what VE's src1 **DIV D** mode
produces when expanding the compactly stored v.

Then the D rows of C are identical, and row 0 stored word-contiguously (Cstep=1) is the result vector.
- Single-sequence decode is bandwidth bound, so the D-fold redundant computation does not affect speed;
- EX needs no change;
- for concurrent sequences and prefill, these D rows are simply replaced by D different sequences or D tokens (§11).

### 8.3 Attention

The KV cache has two blocks per layer, both row major: position t is row t of `Smax × dim` (int8, all heads together).
- **Appending a token**: one row each for k and v, dim bytes each, a multiple of 8, one ST each;
- **Reading per head**: rows = pos_pad, row_bytes = hs, pitch = dim.

For each head h (pos_pad = ⌈(pos+1)/D⌉·D, supplied by a PARAM), this loop body is repeated heads times with LOOP, the head offset advancing per round through a PARAM:

| Step | Engine | Operation |
|---|---|---|
| 1 | LD | K_h (pos_pad × hs) read LINEAR into SPAD, hs/D words per row. **FENCE_BEFORE**: this token's K was just written to DDR by ST |
| 2 | VE | TRANSPOSE (S = hs/D) → SPAD_B, giving the Kᵀ strip |
| 3 | VE | q_h: amax → s_q; quantized to I8, replicated by DIV D into an A strip (K dimension = hs) |
| 4 | EX | scores = Q_rep · Kᵀ: M = D, K = hs, N = pos_pad (repeat = pos_pad/D, bstep = hs) |
| 5 | VE | Row 0 of the scores × s_q (DIV mode); affine A = s_k/√hs; output F32 |
| 6 | VE | softmax with VALID = pos + 1: m = REDUCE MAX; e = EXP(s − m); sum = REDUCE SUM(e); r = RECIP(sum); p = e × r |
| 7 | VE | pq = p × 127 output as I8, replicated by DIV D into an A strip (K dimension = pos_pad) |
| 8 | LD | V_h (pos_pad × hs) read INTERLEAVE into SPAD_B, the layout being the B strip |
| 9 | EX | out_h = P_rep · V_h: M = D, K = pos_pad, N = hs |
| 10 | VE | × s_v/127 (affine), written into segment h of the attention output (fp32) |

The K, V rows padded after position pos are cleared by the ARM at allocation, and p is 0 at those positions, so the result is unaffected.

### 8.4 The complete sequence of one layer

| # | Content | Main commands |
|---|---|---|
| 0 | (before the first layer) x = embedding | LD the embedding row (address = BASE + PARAM[token×dim]); LDPARAM reads s_emb[token]; VE CVT × A (dynamic slot from a PARAM) |
| 1 | xn = RMSNorm(x) | VE: REDUCE SUM(x·x); RSQRT(ss·(1/dim) + eps); x × r (DIV); × g (MOD) |
| 2 | q, k, v = GEMM(xn, Wqkv) | §7.2 |
| 3 | RoPE(q), RoPE(k) | LD row pos of the cos/sin tables (address + PARAM[pos×row bytes]); VE: t1 = x·cos, t2 = SWAPNEG(x)·sin, x = t1 + t2 |
| 4 | Append KV | VE: k × (1/s_k) output as I8; ST to row pos of the K cache (address = BASE + layer offset + PARAM[pos×dim]); likewise v |
| 5 | Attention | §8.3 |
| 6 | x += GEMM(att, Wo) | §7.2, plus a VE ADD |
| 7 | xn = RMSNorm(x) | As step 1 |
| 8 | h13 = GEMM(xn, W13) | §7.2 |
| 9 | h = SiLU(h1) · h3 | VE: EXP(−h1); RECIP(1 + ·); × h1; × h3 |
| 10 | x += GEMM(h, W2) | §7.2, plus a VE ADD |

Steps 1–10 are one loop body repeated layers times with LOOP, each layer's weight and KV addresses advancing by the layer stride through PARAMs.
Finally: RMSNorm → classifier GEMM (column blocks, LOOP) → post-processing → ST writes the logits back to DDR → END.

**Size estimate**: with LOOP, a model's static list is about 150–300 descriptors (about 1.5–2k when unrolled).

### 8.5 PARAM assignment (hand-written path)

| PARAM | Content | Computed by |
|---|---|---|
| P0 | pos + 1 (VALID) | ARM, per token |
| P1 | pos_pad (rows, LEN) | ARM |
| P2 | pos_pad / D (repeat, Kt) | ARM |
| P3 | pos × dim (offset for appending KV) | ARM |
| P4 | pos × RoPE row bytes | ARM |
| P5 | token × dim (offset of the embedding row) | ARM; on the IREE path computed on the device with LDPARAM |
| P6, P7 | Loop variables (layer offset, head offset, block offset) | Advanced per round by LOOP; reset with SETREG before entering a loop |

### 8.6 Verification

- **List generator `llm/compile_layer.py`**: takes the model configuration and memory allocation, outputs a static list with LOOP and PARAMs. L6-IREE replaces it with the compiler.
- **Co-simulation of a small model**: random weights, dim=32, hidden=64, 2 heads, 2 layers (to test LOOP), vocab=64, seq=16.
  - the same list executed on `tb_system` (RTL) and the functional simulator gives bit-identical logits;
  - the functional simulator compared with the device numeric mode of `ref_model.py`, errors within tolerance;
  - covering pos = 0, pos = D−1, pos = D, **changing only the parameter block, the list unchanged**.
- **On the board**, `l4_layer_demo.py`: layer 0 of stories15M at several pos, bit-identical to the functional simulator.

### 8.7 Implementation record and results

**Implementation** (`llm/compile_model.py`, `ModelCompiler.build()`)
- One static list: embedding → layers → final RMSNorm → classifier (loop form) → END.
- Only two things change per token:
  - the ARM's parameter block P0–P5: pos+1, pos_pad, pos_pad/D, scratch, pos_pad·hs, pos_pad·D;
  - the two-word parameter block in DDR (pos, token).
- The other addresses are computed by the device itself with LDPARAM (value read × mul): the KV row offset, the RoPE row offset, the embedding row offset;
  the embedding's scale is read into a PARAM and substituted into VE's A field through a dynamic slot.
- Attention loops over heads with LOOP_END: P6 is the byte offset into K/V, P7 the word offset into q/att.
  - per head: K_h is read and turned into a Kᵀ strip with TRANSPOSE; q_h is quantized; EX computes the scores; the scores are multiplied by s_q·a_k;
  - softmax with the VALID mask; (e·r)·127 quantized into an int8 P strip; V_h read with INTERLEAVE; EX computes P·V; multiplied by a_v.
- Differences from §8.4:
  - layers are **unrolled one after another**, each layer's constants (1/s_k, a_k etc.) and weight offsets written as immediates rather than a per-layer LOOP.
    A per-layer LOOP needs these constants read into dynamic slots with LDPARAM; left for L5 to decide;
  - stories15M: the list has 953 descriptors and executes 2,477 per token.
- `B = −0.0`: y + (−0) = y holds for every y and keeps the sign of zero, making the result bit-identical to DeviceModel.
- `DeviceModel.forward(n_layers=…)` can run only the first N layers.

**Verification**
- Host (`llm/test_l4.py`), functional simulator vs `DeviceModel(sfu=SfuExact)`:
  - tiny model (dim 32, 2 heads, 2 layers, vocab 64, seq 16): at 16 positions with D=8 and 16, logits and KV cache all bit-identical;
  - stories15M: 12 tokens bit-identical.
- RTL co-simulation (`desc_run`): the tiny decoder with tokens at pos 0, D−1, D, logits and KV cache all bit-identical (D=8 and 16).
  About 221 descriptors and about 30k cycles per token.
- Mutation tests: removing SWAPNEG, or removing V's dynamic bstep parameter, are both detected.
  Removing VALID from softmax's max is not detected: the scores at padded positions are exactly 0, and the int8 quantization of p absorbs the rounding differences.
  The mask only matters when all valid scores are very negative (exp underflows, giving NaN), which this test does not cover.

**Board results** (`notebooks/llm/l4_decoder_demo.py`, `bitstreams/l2`, stories15M, D=8, 50 MHz)
- A 16-token prompt plus 20 greedy tokens: **the logits of all 36 tokens are bit-identical to DeviceModel**;
  the logits and KV cache of the first 2 tokens are also bit-identical to the functional simulator run on the board.
- Text generated by the device: "Once upon a time, there was a little girl named Lily. She loved to play outside in the
  sunshine. One day, she saw a big, red ball in".
- **About 2.48M cycles per token = 49.6 ms, i.e. 20.2 tok/s**; 51.1 ms including submission and waiting.
  - every D more positions add about 18.5k cycles (attention);
  - for comparison: llama2.c's int8 version on the ARM, 2 threads, 22.9 tok/s.
- Counters of the last token:
  - EX useful 76.1%, LD busy 78.2%, VE active 15.4%;
  - VE accounts for 80.0% of head-of-queue blocking: post-processing and attention VE steps waiting for the preceding EX to finish;
  - commands executed per token: 634 LD, 172 ST, 340 EX, 1,187 VE.

### 8.8 Per-token cycle breakdown and optimizations (after L4)

**Optimization 1: prefetch block 0 of the next linear layer** (`ModelCompiler(prefetch=True)`, on by default)
- As soon as the EX of a linear layer's last block is issued, the weights of the next layer's block 0 are LD'd into the freed SPAD_B bank; attention uses the other bank;
- board A/B (`notebooks/llm/l4_prefetch_ab.py`): 2,477,683 → 2,399,968 cycles per token (**−3.1%**), 20.18 → 20.83 tok/s;
  LD busy 78.2% → 80.7%; both lists bit-exact;
- the gain is below the initial estimate: before the change, block 0's LD already ran in parallel with the layer's activation quantization, so the new overlap is only the VE steps before the quantization.

**Optimization 2: SiLU fused into W13's block loop** (`ModelCompiler(fuse_silu=True)`, on by default)
- W13 computes h1 blocks and h3 blocks interleaved, and after each pair computes that segment's SiLU (elementwise, results unchanged);
- the FFN activations (H13, U) live in the other ACC bank, avoiding the staging area of the even-numbered blocks that follow. EX and VE share the ACC port
  and serialize on the same bank, so this lets the first 3 SiLU segments run in parallel with the next h1 block's EX;
- the list grows from 953 to 1,025 descriptors (3 more segments per layer, 4 VEs each).

**Three-way board comparison** (stories15M, `bitstreams/l2`, 36 tokens, all bit-identical to DeviceModel)

| List | Cycles per token | tok/s (device) | vs baseline | LD busy |
|---|---|---|---|---|
| Baseline | 2,477,684 | 20.18 | — | 78.2% |
| Prefetch | 2,399,974 | 20.83 | −3.1% | 80.7% |
| Prefetch + SiLU fusion | **2,293,970** | **21.80** | **−7.4%** | 84.4% |

- SiLU fusion saves another ~106k cycles (4.4%) on top of prefetch, matching the estimate (about 4%);
- the tiny model (only 1 pair of blocks) is about 5% slower with fusion: splitting blocks adds overhead, and its single SiLU segment has no EX after it to overlap with.
  Fusion only pays off on layers with many blocks.

**Each segment timed separately** (`notebooks/llm/l4_phase_cost.py`, pos 35, no prefetch, one small list per segment)

| Segment (per layer) | Cycles | Of which the floor for moving weights (7.9 B/cycle) | Other |
|---|---|---|---|
| RMSNorm ×2 | 4,367 | — | 4.4k |
| Quantization + Wqkv 288→864 | 41,513 | 31,498 | 10.0k |
| RoPE, KV append, residual ×2 | 2,367 | — | 2.4k |
| Attention (6 heads, grows with pos) | 20,513 | — | 20.5k |
| Quantization + Wo 288→288 | 16,676 | 10,499 | 6.2k |
| Quantization + W13 288→1536 | 66,578 | 55,996 | 10.6k |
| SiLU × h3 | 24,780 | — | 24.8k |
| Quantization + W2 768→288 | 35,746 | 27,998 | 7.7k |
| **Total per layer** | **212,540** | **125,991 (59%)** | **86.5k** |

- The classifier (including the final norm): 1,260,410 cycles, floor 1,166,582, 94k extra (gaps between blocks); embedding 476 cycles.
- The segments add up to 2.536M; the whole list measures 2.40M, showing about 136k cycles already overlap between segments.
- **Ceiling**: with all time outside the weight stream hidden, 1.92M cycles per token, about 26 tok/s (single port, 50 MHz). Today 20.8 tok/s.

**Where the remaining time is and what to do about it**

| Source | Per token | Approach | Kind |
|---|---|---|---|
| SiLU (EXP, RECIP of 768 elements in multi-cycle mode) | About 150k | Fused into W13's block loop: h1, h3 blocks interleaved, each pair's SiLU computed as soon as it is available, in parallel with the next block's EX | Software, **done**, saves about 106k (optimization 2) |
| Attention (a dozen or so serial small VE commands per head) | About 120k, grows with pos | Online softmax to merge passes; interleave two adjacent heads | Software |
| Gaps between blocks (two SPAD_B banks allow only one block ahead) | About 130k | A fine-grained scoreboard (§11.3) allows two or more blocks ahead | Hardware, about 400–600 LUT |
| The multi-cycle mode itself is slow (folding + waiting at each step) | Included in the first two rows | Multi-group interleaved execution in the microprogram controller | Hardware, LUTs tight |

---

## 9. L5: ARM runtime and end-to-end generation

### 9.1 `llm/runtime.py`: `LlamaDevice` (Python, for bring-up and as a reference)

- **Loading**: read `*.w8a8` into one CMA buffer. stories15M is about 16 MB, 110M about 115 MB; the CMA size needs checking.
  Also allocate the KV cache (cleared), RoPE tables, logits, scratch, the two command rings and the parameter block.
- **List**: generate one static list with `compile_layer.py`, loaded once.
- **Per token**: write the parameter block (§8.5) → submit an entry → wait for the interrupt → read back the logits → sample (argmax or temperature/top-p, in NumPy).
- **Prompt**: L5 sends the prompt token by token through the decode path, just without sampling; real prefill is left to L6.
- **tokenizer**: `llm/tokenizer.py`, porting the encoding and decoding of llama2.c's `tokenizer.bin` (BPE).

### 9.2 Acceptance

On the board, `notebooks/llm/l5_generate.py`:

1. **Correctness**: stories15M with a given prompt, 128 greedy tokens. Every token's logits bit-identical to the functional simulator, identical text.
2. **Accuracy**: under teacher forcing, top-1 agreement with the fp32 reference reaches the threshold set in L0.
3. **Demo**: generate fluent TinyStories text, printing tok/s and the per-token cycle breakdown, recorded only.
   Estimated about 25 tok/s at 50 MHz (stories15M).
4. **Stability**: 10 consecutive generations of 256 tokens each, no errors or timeouts, interrupt count equal to token count.

### 9.3 Implementation record and results

**Files**
- `llm/runtime.py`: `LlamaDevice` and `Sampler`.
  - the model, io area, KV cache and static list share one CMA buffer, and the list is generated and written once;
  - per token: write the parameter block (pos, token) → submit through rt_fw's command ring with a PARAM block → wait for the notify interrupt → read the logits → sample on the ARM;
  - `Sampler` is ported from run.c: temperature, top-p and the xorshift random numbers;
  - the prompt goes through the same decode path, and `generate()` also supports teacher forcing.
- `driver/pynq_matmul.py`: `Device.irq_events` counts the interrupts received.
- `llm/prepare_l5.py`: host-side reference data.
  - greedy: MD5 of the DeviceModel gold-standard logits at every position;
  - accuracy: L0's 8 evaluation prompts, fp32's greedy sequences with the top-5 at every position, and DeviceModel's argmax.
- `notebooks/llm/l5_generate.py`: board acceptance, covering the 4 items of §9.2.

**Board results** (stories15M, D=8, 50 MHz, `bitstreams/l2`, waiting for the interrupt on every token)

| Item | Result |
|---|---|
| 1 Correctness | Prompt plus 128 greedy tokens, 132 positions in all, logits all bit-identical to DeviceModel, identical text: **pass** |
| 2 Accuracy | Teacher-forced top-1 agreement with fp32 over 1,554 positions **96.2%** (threshold 95%); every position's argmax identical to DeviceModel: **pass** |
| 3 Demo | Stories generated at temperature 0.8, top-p 0.9 are fluent. Wall clock **15.1 tok/s**, device 18.9 tok/s (2.44–2.85M cycles per token, growing with pos). Counters of one token: EX useful 77%, LD busy 80% (7.90 B per busy cycle), VE accounts for 81% of head-of-queue blocking |
| 4 Stability | 10 runs × 256 tokens, 2,560 tokens in all, 0 errors, 2,560 interrupts received: **pass** (14.7 tok/s) |

- The gap between wall-clock and device speed is about 13 ms/token, spent on the ARM: NumPy softmax and sampling over 32000-dimensional logits, waiting for the interrupt, copying the logits.
- For comparison, llama2.c's int8 version on the ARM (2 threads) does 22.9 tok/s. This plan aims at an industrial-style architecture, not speed.
  The main speedups are left to L6: overlapping VE with EX, L5.5's 75 MHz, prefill.

---

## 10. Compiler interface: MLIR / IREE

### 10.1 IREE's structure and integration points

```
PyTorch ──(iree-turbine / torch-mlir)──► MLIR (linalg / tensor)
  IREE compiler: Flow (operator fusion, dispatch formation) → Stream (async scheduling, buffer allocation)
             → HAL (each dispatch compiled into a target executable; VM bytecode generated for the host)
IREE runtime (C on the ARM): VM → HAL driver → command buffers → queue submission → semaphores
```

Integrating a new accelerator takes two parts:
- **Compiler side**: a **target backend** compiling dispatches into our executables, i.e. position-independent descriptor-list templates with metadata:
  number of bindings, number of push constants, entry points.
- **Runtime side**: a **HAL driver** (C code) implementing the device, memory allocation, executable cache, command buffers, queues and semaphores.

### 10.2 Concept mapping and required hardware

| IREE concept | Hardware and firmware counterpart | Where |
|---|---|---|
| Dispatch bindings | BASE0–15 + relocation | §5.3 #1 |
| Push constants, dynamic dimensions | PARAM0–7 + dynamic fields | §5.3 #2, #3 |
| Workgroup count (can be dynamic) | LOOP_END, with the count from a PARAM | §5.3 #5 |
| Command buffer: recording several "bind + dispatch" | SETREG + CALL an executable, forming one list | §5.3 #4, #6 |
| Executables (loaded once) | Position-independent descriptor templates (relative JUMP/LOOP, CALL/RET) | §5.3 #6 |
| Execution barrier | FENCE descriptor. IREE requires explicit synchronization at barriers, consistent with the rule "DDR dependencies are not tracked by hardware" | Existing |
| Queue submission and timeline semaphores | Command-ring entries + the seq of completion records + interrupts | §5.1 |
| Data-dependent addresses (gather, embedding, writes by position) | LDPARAM | §5.3 #7 |
| Buffer flush / invalidate | CMA buffers + cache maintenance | Software |
| fill, copy, update buffer commands | Executable templates: DDR→SPAD→DDR copy; VE writes an immediate, then ST | Software |
| Data tiling (`tensor.pack`, mmt4d) | A and B strips are the pack layout of [D, D] inner tiles; LD INTERLEAVE packs while moving; weights packed ahead at compile time | Existing |
| Multiple devices (some dispatches on the CPU) | The `llvm-cpu` target on the ARM; CMA buffers accessible from both sides | Software |

### 10.3 Conventions that come with IREE (no hardware change)

1. **Data between dispatches goes through DDR**. The on-chip SPAD/ACC are only like a GPU's shared memory, not kept across dispatches.
   In decode activations are only a few KB, so the impact is small; how large dispatches are is decided by IREE's fusion.
2. **Numeric convention**: §3.2 declared as fast-math, as GPUs do:
   - FTZ;
   - EXP/RECIP/RSQRT are approximate;
   - `arith.divf` lowers to RECIP plus a multiply;
   - `math.exp`, `math.rsqrt` map directly to FUNC.
3. **Elementwise bodies**: a `linalg.generic` that exceeds the VE's fixed operation chain is split into several VE commands, with intermediates kept on chip.
4. **Layout types**: the compiler internally defines the following layouts and propagates them:
   - compact vectors;
   - A strips, B strips;
   - broadcast words (one scalar per row);
   - Kᵀ strips.
   There are three conversion operators: TRANSPOSE, DIV-D row replication, LD INTERLEAVE through DDR.
5. **Cost model**: the functional simulator's cycle estimates (§4.3), calibrated with the counters.

> **Detailed design**: this section is an outline. The full design of the compiler and runtime, the staged plan (C0–C7), implementation records and acceptance are in
> [`iree_compiler_plan.md`](iree_compiler_plan.md).

### 10.4 Two ways of integrating

- **A. Template library** (first, L6-IREE): match linalg named operators to pre-written executable templates,
  covering matmul and its quantized forms, softmax, RMSNorm-style reductions, elementwise operations, transpose, gather.
  These templates generalize the segments of `compile_layer.py`; unsupported dispatches run on `llvm-cpu`.
- **B. Full code generation** (later): define an `sa` dialect. After tiling, packing per SPAD and memory planning, linalg
  lowers to operations such as `sa.ld`, `sa.ex`, `sa.ve`, `sa.loop`, then serializes to descriptors.

Both approaches have **the same** hardware requirements, met by §5.3 and §6.

**Implementation status**: both were done. A is C3 (the template library); B is C5, actually built as two dialect levels: `sahl` (on-chip buffers and DDR reads/writes, where micro-kernels and memory planning live) and `sahw` (one operation per accelerator command); A's templates were rewritten as `sahl`-level micro-kernels, and without micro-kernels the generic lowering still compiles every dispatch (`iree_compiler_plan.md` §8).

### 10.5 L6-IREE deliverables and acceptance

(Actual locations: the compiler plugin is in `compiler/plugins/sa`, the HAL driver in `compiler/runtime/sa`, see `compiler/README.md`. The current state and later plans are in [`iree_compiler_plan.md`](iree_compiler_plan.md) §8.12–§8.19: prefill + decode, the layered restructuring of code generation, host fallback, a generic frontend for original HuggingFace models (prefill + decode bit-exact on the board), the FPGA size larger models need.)

- **`iree-sa/compiler`**: the HAL target backend plugin, using the template library, with executable format `sa-desc-v1`.
- **`iree-sa/runtime`**: a HAL driver in C:
  - accesses the mailbox, command ring and parameter blocks (through mmap);
  - UIO interrupts;
  - allocates buffers from CMA;
  - semaphores map to the seq of completion records.
- **Acceptance**:
  1. an MLP, a softmax and a single decoder layer compiled by IREE, in mixed "CPU + accelerator" execution, bit-identical to the functional simulator;
  2. stories15M compiled with IREE generates text on the board identical to L5's hand-written path (bit-exact when the numeric conventions are the same).
- **Result**: item 2 was reached in C3 (board logits bit-identical to the hand-written path), item 1 is covered by per-dispatch differential tests (every dispatch bit-identical to the oracle); afterwards C5 switched to full code generation, and C5.5 compiled SmolLM2-135M (on the board) and the Qwen3 structure (sim) with no dedicated code written for them. C6: Qwen3-0.6B, all 28 layers (sim).

---

## 11. L5b, L5.5, L6: concurrent sequences, 75 MHz and later extensions

### 11.1 L5b: concurrent multi-sequence decode (no hardware change)

The D rows of the replicated-row layout become D sequences:
- each sequence has its own KV cache and pos;
- the A strip is made of the activations of D sequences: converted into a strip through DDR with LD INTERLEAVE, or written row by row by VE;
- attention is done separately per sequence, since their pos differ;
- the same weights serve D rows, so throughput approaches D times the single sequence.

Acceptance: 8 sequences generated at once, each one's text identical to generating it alone; total tok/s recorded.

**Implementation record** (`BatchCompiler` in `llm/compile_batch.py`, `BatchLlamaDevice` in `llm/runtime.py`; D=8 only, D=16's ACC is too small)
- **Activations**: a row-major D × n fp32 matrix, row b being sequence b.
  - elementwise VEs simply run D times as long;
  - per-sequence reductions use REDUCE with ROWLEN = n/D, in the same order as a single sequence, so the results are bit-exact;
  - per-sequence scalars are broadcast with DIV, per-channel vectors with MOD.
- **A strip** (word kb·D + b = segment kb of sequence b): the VE's index modes cannot express this mapping, so it detours through DDR: ST the quantized matrix → FENCE → LD INTERLEAVE.
- **GEMM**: EX writes C directly into the output matrix (C's row stride = one matrix row), with block boundaries aligned to the segments of q/k/v and h1/h3,
  each of which becomes its own matrix; dequantization is done once per linear layer.
  The classifier is processed in blocks, writing the D × vocab logits matrix with strided STs.
- **Attention**: unrolled per sequence, with a head loop inside each sequence; that sequence's PARAMs are read from its entry in a DDR parameter table with LDPARAM.
- **Each sequence is independent**: each has its own KV cache and pos and can start at a different step (an early form of continuous batching);
  slots without a sequence run a dummy token at pos 0.
- The buffers of the attention phase and the FFN phase reuse the same space in ACC: stories15M uses 3,820 words.
- The list has 2,137 descriptors and executes 7,231 per step.

**Verification**
- Host: the tiny model for 16 steps and stories15M for 12 steps, 8 sequences starting at staggered times, each sequence's logits and KV cache bit-identical to the single-sequence DeviceModel.
- RTL co-simulation: the batched tiny decoder, bit-exact at steps 0 and 8 (positions 8..1).
- Mutation test: changing the RMSNorm weights' MOD to DIV is detected.

**Board results** (`notebooks/llm/l5b_batch_demo.py`, stories15M, `bitstreams/l2`)

| Mode | Wall-clock tok/s | Device |
|---|---|---|
| Single sequence (L5's `LlamaDevice`) | 17.5 | 2.51M cycles/token, 19.9 tok/s |
| 8 sequences batched | **63.3 (3.6×)** | 5.19M cycles/step, **77.0 tok/s (3.9×)** |

- With the 8 sequences starting at staggered times, all 64 positions of each are **bit-identical** to generating it alone, with identical text.
- Not 8×, because the work proportional to the number of sequences grows 8-fold per step: VE elementwise operations and reductions, per-sequence attention, the A strip's round trip through DDR (a FENCE each time).
  Counters of one step: EX useful 40.5%, LD busy 42.6%, VE active 51.3%, head-of-queue blocking VE 28.8%, EX 24.1%, ST 12.3%; 4,104 VE commands.
  So weight bandwidth is no longer the bottleneck; VE is. Going faster needs VE overlapped with EX, a wider VE (resources are tight), or fewer FENCEs.

### 11.2 L5.5: 75 MHz (extra target)

- Change `subprocessorClk` in `pico_bit.tcl` to 75 MHz. PicoRV32, the accelerator and the HP ports all use this clock; the HP ports themselves support up to 150 MHz.
  The frequency constants in the driver and runtime change accordingly.
- Following L0's timing survey, fix only the necessary paths. One known path is the fan-out of `ex/drain_active`:
  replicate the register, or pipeline the enable one stage driving it in row groups, delaying the data by one cycle to match.
- The new modules are already designed for 13.3 ns and in principle need no change.
- **Acceptance**:
  - WNS ≥ 0;
  - all L1–L5 board scripts pass at 75 MHz;
  - performance unchanged in cycles and 1.5× faster in time (estimated about 37 tok/s for stories15M).
- **Failure does not affect earlier results**: keep using 50 MHz.

### 11.3 L6: later extensions (optional, as needed)

| Item | Content |
|---|---|
| Real prefill | GEMM on batches of D tokens: ST to DDR, then read back with LD INTERLEAVE to get A strips; the causal mask within a block uses a VALID that varies by row |
| Fine-grained scoreboard | Track each memory in 8 regions instead of 2 banks (masks widened from 6 to 24 bits), reducing false dependencies. Conditioned on L4/L5 counters showing a significant `HAZ_*` share; about 400–600 LUT |
| bf16 | The VE's bf16 storage format and conversions (if not done in §6), for torch models exported in bf16 |
| int4 weights | Unpack int4 to int8 on the LD path. **Only pays off if EX can consume more weights per cycle**; no gain at D=8 |
| Direct submission from the ARM | Implement END's IRQ bit; the ARM hands lists directly to the fetch unit |
| Paged KV cache | K and V stored in blocks, the ARM maintains a block table, block addresses read from it with LDPARAM |
| Device-side interrupt doorbell | An MMIO doorbell register triggers a PicoRV32 interrupt, replacing polling |
| Full code generation | Approach B of §10.4 |

---

## 12. Resource and timing budget

xc7z020: 53,200 LUT, 106,400 FF, 220 DSP, 140 BRAM36. All figures below are **estimates**; synthesis results decide:

| Part | LUT | DSP | BRAM36 | Basis |
|---|---|---|---|---|
| D=8 baseline (M3 measured 21.7k, plus PERF and DESC) | ~24k | 96 | 130 | M3 report; at D=16 counters and fetch together add about 1.8k |
| L1 command ring, interrupt, `mat_notify` | < 0.3k | 0 | 0 | A few registers |
| L1 command-processing extensions (16 BASEs, PARAMs, dynamic fields, SETREG, LOOP, CALL/RET, LDPARAM) | 1.5–2.5k (**measured about 3.6k**) | 1 (measured 2) | 0 | **OOC measurement** (D=8, 50 MHz): fetch unit 4,408 LUT, 2,106 FF, 2 DSP, the whole `sa_unit` 21,787 LUT, WNS +2.37 ns. The excess is mainly the register file's multiple read/write ports and the field-rewrite logic |
| L2 fp32 lanes × 8 (2 multipliers, 2 adders, conversions, comparisons, tables) | 11–14k (**measured: 8 lanes about 18.6k, does not fit; folded to 4 lanes about 9.7k**) | 32 (measured 16) | 0 | Measured about 2.3k LUT and 2 DSP per lane (i2f ×2 and f2i take half); the tables are one 256-entry ROM. Folding: see §6.9 |
| L2 microprogram control, transpose buffer, reductions, index generation, wider command packet, exact access ranges | 2–3k (**measured about 7.5k + scheduler / fetch +1.2k**) | 0 (measured 2) | 0 | **OOC measurement** (D=8, FL=4, 50 MHz): `sa_vefp` 17,197 LUT, 11,696 FF, 18 DSP; the whole `sa_unit` 39,062 LUT (73.4%), 28,232 FF, 116 DSP, 128 BRAM36, WNS +1.62 ns. Overall estimate about 44.4k (83%) |
| **Total** | **~39–44k (73–83%)** | **~129** | **130** | D=16 still met 50 MHz at 92% |

- **BRAM** is the tightest resource, with only 10 left. All new storage uses LUTRAM or FFs.
- **FF** expected at about 50–55k (about 50%), plenty of margin for pipeline registers.
- **Timing**: at 50 MHz the existing modules have margin and the new modules are designed for 13.3 ns. Above 80% LUT routing congestion worsens, but 50 MHz still has margin.
- **If LUTs exceed the budget, cut in this order**:
  1. remove the integer VE's int16 path;
  2. remove CALL/RET, with executables' absolute addresses patched by the ARM at load time;
  3. merge M1 and M2 of the fp32 lanes (MUL and the affine multiply cannot happen at once; two cycles instead);
  4. reduce the SFU tables to 32 segments.

---

## 13. Overall plan and acceptance

| Level | Content | What changes | Bitstream build needed | Acceptance | Status (commit) |
|---|---|---|---|---|---|
| **L0** | D=8 baseline and 75 MHz timing survey; reference model and quantization accuracy; functional simulator; IREE on armv7 | Python; one D=8 build | Yes (no RTL change) | D=8 regression all passes; the fp32 reference's text identical to `run.c`; top-1 agreement ≥ threshold; functional simulator bit-identical to the co-simulation cases; IREE's `llvm-cpu` small model runs correctly on the board, or a fallback decided | **Done** (2915e89, 602f79c) |
| **L1** | Command ring, `rt_fw`, interrupt; command-processing extensions | sa_cmdfetch, sa_pcpi, sa_legacy, sa_unit, sa_defs, BD, firmware, driver, functional simulator | Yes | §5.5: directed and random tests bit-exact; `RING_TEST`; 200 asynchronous submissions all correct on the board; a static list runs correctly with only the parameter block changed | **Done** (0d626b8, `bitstreams/l1`) |
| **L2** | fp32 VPU (hybrid structure), SFU, reductions, transpose, masks | sa_ve and the new fp32 modules, sa_sched, sa_defs, sa_pcpi, sa_cmdfetch, driver, functional simulator | Yes | §6.8: bit-exact; SFU accuracy met; board operators bit-identical to the functional simulator; no regression | **Done** (98a4a66, `bitstreams/l2`; fp lanes folded to D/2, 44.0k LUT) |
| **L3** | W8A8 export; GEMM and post-processing (B double buffering) | Python | No | Every linear layer on the board bit-identical to the functional simulator; classifier GEMV LD busy ≥ 90% | **Done** (318df51; LD busy 93.9%) |
| **L4** | One decoder layer (with attention) executing on the device, static list plus PARAMs | Python | No | Small model compared three ways between RTL, functional simulator and reference model; layer 0 bit-exact on the board | **Done** (a452ca4; the whole model on the device, 20.2 tok/s) |
| **L5** | Python runtime; end-to-end text generation | Python | No | The 4 items of §9.2 | **Done** (89741a3; 15.1 tok/s wall clock, top-1 96.2%) |
| L5b | Concurrent multi-sequence decode | Python | No | §11.1 | **Done** (faf7bc7; 8 sequences 63.3 tok/s) |
| L5.5 | 75 MHz | Clock and a few timing fixes | Yes | §11.2 | Not done (LUT already at 83%, timing harder to close) |
| **L6-IREE** | Target backend (template library) + HAL driver in C | IREE plugin, C runtime | No | §10.5; detailed design in [`iree_compiler_plan.md`](iree_compiler_plan.md) | **Mostly done** (C0–C3 and C5 done: stories15M 17.7 tok/s on the board, SmolLM2-135M 2.22 tok/s on the board, both bit-exact; C4 partly done; C6 larger models in progress: Qwen3-0.6B all 28 layers bit-exact on the sim; generic frontend: the original HF SmolLM2 prefill + decode bit-exact between board and sim, Qwen3 passes without code changes) |
| L6 | Other extensions | — | Depends | Defined per item | Not done |

- **Suggested order**: L0 → L1 → L2 → L3 → L4 → L5 → L6-IREE, with L5b and L5.5 interleaved.
  The RTL of L1 and L2 can be merged into one bitstream build but verified separately.
- **Build convention**: Vivado limited to 4 jobs; long builds are run by the user in their own terminal.
- **Commit convention**: commit after each level passes on the board, archiving the bitstream in `bitstreams/l<n>/`.

---

## 14. Risks and mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| Per-channel W8A8 is not accurate enough on a model as small as 15M | L0's top-1 agreement misses the threshold | Try in order: ① switch to 42M or 110M; ② dynamic KV scales (each token's scale stored as a broadcast word, rearranged with TRANSPOSE when read back); ③ an fp32 bypass for layers with many outliers (GEMV on the VE as elementwise multiply-add plus per-row reduction); ④ per-group quantization: K split into several EXs by group, results in different ACC regions, each group's scale multiplied in the VE before accumulating |
| The IREE runtime does not run on armv7 | L0 §4.4 fails | Use only the IREE compiler, with a self-written executor in C as the runtime; the hardware interface is unaffected |
| IREE's plugin interfaces change between versions | Compilation breaks after an upgrade | Pin one IREE version; keep plugin code isolated from IREE itself |
| LUTs over budget | Over 83%, or routing fails | Cut in the order of §12 |
| Bugs in the dynamic-field and control-flow descriptor logic | Wrong list execution results | Directed tests for every kind, plus random list tests against the functional simulator; all errors are sticky and record the index of the failing descriptor |
| PicoRV32 and the ARM seeing different DDR contents | The firmware reads stale entries or parameter blocks, or the ARM reads stale completion records | Uncached buffers for the command ring, parameter blocks and lists, or strict flush / invalidate; completion records write data first and update `RING_HEAD` last |
| PYNQ or UIO cannot find the interrupt | `matmul_0/irq` not found | Check the hwh offline with pynqmetadata first; polling `RING_HEAD` as a fallback path |
| 75 MHz cannot meet timing | L5.5's WNS < 0 | Keep using 50 MHz; earlier results are unaffected |
| Functional simulator and RTL disagree | Bit-exact comparison fails | This is exactly what it is meant to find. This document's specification decides; fix both sides to it and add test vectors |
| Not enough program BRAM | `rt_fw` exceeds 0x1E00 | The loop is small, estimated within 1 KB |

---

## Appendix A: summary of encoding changes

| Item | Encoding | Level |
|---|---|---|
| `mat_notify` | funct7=1, funct3=7: one rising-edge pulse on `notify_irq`, `NOTIFY_COUNT` incremented | L1 |
| CSR | 0xD4 `NOTIFY_COUNT` (read-only) | L1 |
| `sa_unit` port | `notify_irq` (EDGE_RISING), connected to `irqConcat/In1` in the BD | L1 |
| CAPS | bit 22 NOTIFY/IRQ, bit 23 FP32 VE, bit 24 CMDX (command-processing extensions) | L1 / L2 |
| mailbox | 0xB0–0xCC: command ring and firmware status (§5.1) | L1 |
| Command-ring entry | w7 = parameter block address | L1 |
| Descriptor w0 | BASESEL = `{w0[14:13], w0[10:9]}`; w0[31:16] holds two dynamic slots | L1 |
| New opcodes | 0x13 LOOP_END, 0x14 SETREG, 0x15 CALL, 0x16 RET, 0x17 LDPARAM; JUMP's w2[0] = REL | L1 |
| `mat_cfg` keys | 15–26 BASE4–15, 27–34 PARAM0–7 | L1 |
| VE types | VT_F32 = 3 (only in ACC); types[5:4] is src2's type (0 same as src1, 1 I8, 2 I32, 3 F32). The optional VT_BF16 is not implemented | L2 |
| VE op | 6 = TRANSPOSE (7 reserved) | L2 |
| VE descriptor | w3[63:53], w5[63:32], w6, w7 (§6.6) | L2 |
| `vec_cfg` keys | 10 FLAGS, 11 IMM, 12 A, 13 B, 14 ROWLEN/VALID, 15 P1/S | L2 |

## Appendix B: new and modified files

| Path | Level | Content |
|---|---|---|
| `llm/ref_model.py`, `llm/eval_quant.py` | L0 | Reference model, quantization accuracy evaluation and KV scale calibration |
| `llm/sa_funcsim.py` | L0–L2 | Functional simulator (the bit-exact golden reference) and cycle estimates |
| `llm/sfu_tables.py`, `llm/gen_ve_fp_vectors.py` | L2 | SFU tables, fp32 VE test vectors |
| `llm/export_w8a8.py` | L3 | Model export |
| `llm/compile_layer.py`, `llm/test_l3.py` | L3, L4 | Hand-written list generator (LOOP, PARAM) and its host test |
| `llm/compile_model.py`, `llm/test_l4.py` | L4 | Static list of the whole decoder (only parameters change per token) and its host test |
| `llm/runtime.py`, `llm/prepare_l5.py` | L5 | ARM runtime `LlamaDevice`, sampler; reference data for acceptance |
| `llm/compile_batch.py`, `llm/test_l5b.py` | L5b | Static list for concurrent multi-sequence decode (`BatchLlamaDevice`) and its host test |
| `llm/runtime.py`, `llm/tokenizer.py` | L5 | Python runtime |
| `firmware/rt/` | L1 | Resident firmware `rt_fw` |
| `firmware/include/mailbox.h`, `sysarray_intrinsics.h` | L1, L2 | Command-ring fields, `mat_notify`, new `mat_cfg` / `vec_cfg` keys |
| `rtl/sysarray/sa_cmdfetch.v` | L1 | Command-processing extensions |
| `rtl/sysarray/sa_pcpi.v`, `sa_legacy.v`, `sa_unit.v`, `sa_defs.vh` | L1, L2 | `mat_notify`, interrupt, new CSRs and CAPS, new keys, wider command packet |
| `rtl/sysarray/sa_ve.v`, new `sa_ve_fp.v`, `sa_fp32_add.v`, `sa_fp32_mul.v`, `sa_fp32_cvt.v`, `sa_sfu_seq.v` | L2 | fp32 VPU and SFU |
| `rtl/sysarray/sa_sched.v` | L2 | Exact access ranges in the new modes |
| `rtl/sysarray/sim/tb_sa_ve_fp.v`, `tb_sa_unit.v`, `firmware/sim/tb_system.v` | L1, L2, L4 | New tests |
| `RISCV-on-PYNQ-Z1/scripts/pico_bit.tcl` | L1, L5.5 | `irqConcat` changed to 2 ports; clock changed to 75 MHz |
| `driver/pynq_matmul.py` | L1, L2 | `Device` (command ring, parameter blocks), new descriptors and fields in `DescList` |
| `compiler/plugins/sa`, `compiler/runtime/sa` (the originally planned `iree-sa/…`) | L6-IREE | IREE target backend plugin and HAL driver |
| `notebooks/llm/l1_ring_demo.py` … `l5_generate.py` | L1–L5 | Board scripts |

# LLM compiler: an end-to-end plan based on MLIR / IREE (PyTorch → descriptor lists)

> **PYNQ-Z1 paths**: this document records work done on the PYNQ-Z1. The `RISCV-on-PYNQ-Z1/...` (block design, constraints), `bitstreams/...` (builds verified on the board) and `iree-sa/l0` it mentions are on the `pynq-z1` branch (tag `v1.0-pynq-z1`); on main, PicoRV32 and its IP are in `rtl/picorv32/` and `rtl/ip/`, and the current overlay is in `boards/kv260/`.

Status: **C0–C3 done** (§3.5, §5.7, §6.7, §6.9); **C4 partly done, on hold** (§7.1: 2.591M cycles / token on the board, 17.7 tok/s, 13% behind the hand-written path against a target of ≤ 10%; the remaining fusions are left until after C5); **C5 done** (§8.7: full code generation with the two dialect levels sahl / sahw, templates turned into micro-kernels; stories15M 2.589M cycles / token on the board; SmolLM2-135M 2.23 tok/s on the board, the Qwen3 structure on the sim; both can generate interactively on the board); **C6 in progress** (§8.12: target-configuration cross-validation, K tiling, C6.0 single copy of the embedding, C6.1 Qwen3-0.6B all 28 layers bit-exact on the sim, C6.P prefill + decode bit-exact on the board done; next C6.2 quantization quality or prefill optimization); on the KV260 Qwen3-0.6B already runs on the board, with decode on the streaming GEMV (`--iree-sa-gemv-ports`, 4 ports 6.28 tok/s, [`k2b_gemv_design.md`](k2b_gemv_design.md)); **generality plan** ([`generality_implementation_plan.md`](generality_implementation_plan.md), 2026-10-13): unmodified HuggingFace SmolLM2-135M, Gemma-3-270M, Qwen3-0.6B and Llama-3.2-1B whole on the KV260, every dispatch on the accelerator (decode 22.1 / 15.0 / 7.0 / 4.0 tok/s), which completes C6.2 and C6.4; C6.5 / C6.5b done in the compiler (2026-10-14, §8.22; p · V on the GEMV unit with a packed V cache the default since 2026-10-15): decode attention over T in blocks without online softmax, everything on the accelerator at T = 512 for Gemma and 2048 for Qwen3, a 1104-token Qwen3 prompt end to end on the sim; on the board (2026-10-14) the whole Qwen3 at 2048 and Gemma at 512: prefill vs decode only bit-exact over a 2032 / 496-token prompt, 0 dispatches on the host, board teacher forcing PPL within 1.2% of fp32 (Qwen3) over two 2048-token windows (§8.22); interactive generation with multi-turn chat on the board for the generic bundles (`board_generate.py --chat`, generality plan §4). This is the detailed design of the L6-IREE level of [`llm_inference_plan.md`](llm_inference_plan.md),
replacing the outline of §10.4, §10.5 there.

**Goal**: starting from a llama-like model written in PyTorch, compile automatically with MLIR / IREE into a complete program running on the PYNQ-Z1:
- the IREE runtime runs on the ARM;
- computation is executed by the accelerator, in the form of descriptor lists entering the engines through the command ring, `rt_fw` and the fetch unit;
- finally stories15M compiled by IREE generates text on the board identical to L5's hand-written path.

**No hardware change**: the hardware interfaces were already designed around IREE's concepts in L1 and L2 (§1.3); the compiler and runtime are all software.

Related documents:
- [`llm_inference_plan.md`](llm_inference_plan.md): §3 numeric conventions, §5 command ring and command-processing extensions, §6 fp32 VE, §7–§9 the hand-written path, §8.8 scheduling optimizations, §10 the IREE outline;
- [`double_buffer_design.md`](double_buffer_design.md): the accelerator, scoreboard, instruction and descriptor formats;
- [`memory_model.md`](memory_model.md): address map, cache coherence, command-ring buffers;
- [`../iree-sa/l0/README.md`](../iree-sa/l0/README.md): L0 verified that IREE works on the board's ARM (armv7).

---

## Contents

0. [Principles](#0-principles)
1. [Overall structure](#1-overall-structure)
2. [Inheriting existing results](#2-inheriting-existing-results)
3. [Frontend: quantized model and export (C0)](#3-frontend-quantized-model-and-export-c0)
4. [Executable format sa-desc-v1](#4-executable-format-sa-desc-v1)
5. [Runtime: the sa HAL driver (C1)](#5-runtime-the-sa-hal-driver-c1)
6. [Compiler: the sa target backend plugin (C2, C3)](#6-compiler-the-sa-target-backend-plugin-c2-c3)
7. [Performance: scheduling across dispatches (C4)](#7-performance-scheduling-across-dispatches-c4)
8. [Full code generation: the sa dialect (C5)](#8-full-code-generation-the-sa-dialect-c5)
9. [Verification system](#9-verification-system)
10. [Staged plan and acceptance](#10-staged-plan-and-acceptance)
11. [Directory structure](#11-directory-structure)
12. [Risks and mitigations](#12-risks-and-mitigations)
13. [Open decisions](#13-open-decisions)

---

## 0. Principles

- **First open the narrowest vertical slice, then widen it**: the first milestone supports only one quantized matmul, but end to end from torch to board results.
  Then add one kind of dispatch at a time, each step verifiable.
- **Numerics are set by the specification, not the implementation**: `DeviceModel(sfu=SfuExact)` is the numeric specification (§3.2).
  No compiler transformation may change the order of operations or the number of roundings; otherwise the strongest acceptance criterion, "bit-identical to the hand-written path", is lost.
- **Keep this project's verification ladder**: functional simulator → RTL co-simulation → board. Most compiler development happens on the host (the simulator transport of §5.5).
- **Measure before optimizing**: first make it correct (C2, C3), then find bottlenecks with the counters and per-segment timing like §8.8 (C4).
- **The hand-written path is the reference**: L3–L5's `compile_layer.py` and `compile_model.py` already compile the whole decoder into descriptor lists;
  they are both the reference implementation of the compiler's templates and the performance comparison.

---

## 1. Overall structure

### 1.1 Data flow

```
PyTorch model (an explicitly quantized llama, §3.1)
   │  iree-turbine: aot.export → torch dialect (weights externalized as a parameter file .irpa)
   ▼
torch-mlir → linalg / tensor / arith / math
   │  preprocessing (§6.2): keep high-level operators such as softmax, forbid splitting reductions, update the KV cache in place
   │  IREE DispatchCreation / Flow: fusion, dispatch formation
   │  data tiling (§6.3): matmul operands carry encodings, resolved into packs of [D, D] inner tiles
   │  IREE Stream: async scheduling, buffer allocation (transient buffers, constants)
   │  IREE HAL: each dispatch handed to the target backend; host-side scheduling becomes VM bytecode
   ├──► sa target backend plugin (C++, §6): dispatch → sa-desc-v1 executable (position-independent descriptor template + metadata)
   └──► VM bytecode .vmfb (call order, bindings, push constants, barriers, semaphores)
ARM: IREE runtime (C)
   └──► sa HAL driver (C, §5)
          command buffer → one descriptor list (SETREG bindings + CALL templates + FENCE)
          → command-ring entry → doorbell → wait for the notify interrupt → completion record → semaphore
PicoRV32 rt_fw, sa_cmdfetch, sa_sched, the engines: unchanged
```

### 1.2 The two parts to write

| Part | Language | Runs where | Purpose |
|---|---|---|---|
| **sa target backend plugin** | C++, MLIR | iree-compile on the host | Compiles each dispatch into an sa-desc-v1 executable |
| **sa HAL driver** | C | The IREE runtime on the board's ARM | Implements IREE's device abstraction: memory, executables, command buffers, queues, semaphores |

Plus two auxiliary parts:
- the frontend's quantized model (Python, §3);
- a simulator service (Python, reusing `llm/sa_funcsim.py`, §5.5), letting the driver run against the functional simulator on the host.

### 1.3 Mapping of IREE concepts to hardware (following `llm_inference_plan.md` §10.2)

| IREE concept | Hardware and firmware | Use in this plan |
|---|---|---|
| Dispatch bindings | BASE0–15 + relocation | binding i → BASE i (§4.3) |
| Push constants, dynamic dimensions | PARAM0–7 + dynamic fields | PARAM0–5 passed directly, extra ones through LDPARAM (§4.3) |
| Workgroup count | LOOP_END (count can come from a PARAM) | Loops inside templates |
| Executables | Position-independent descriptor templates | Only relative JUMP / CALL / LOOP, copied whole into CMA at load |
| Command buffers | SETREG + CALL, forming one list | §5.2 |
| Execution barrier | FENCE descriptor | Barrier → FENCE (DDR dependencies are not tracked by hardware) |
| Queue submission, timeline semaphores | Command-ring entries, the seq of completion records, interrupts | §5.4 |
| Data-dependent addresses | LDPARAM | gather (embedding table), writing KV by position |
| Data tiling (pack, mmt4d) | A and B strips are [D, D] inner tiles | Weights packed at compile time (§6.3) |

---

## 2. Inheriting existing results

| Existing result | Role in the compiler |
|---|---|
| `DeviceModel(sfu=SfuExact)` in `llm/ref_model.py` | **The numeric specification**. The frontend model, compiler transformations and device results all follow it (§3.2) |
| `llm/compile_layer.py`, `compile_model.py`, `compile_batch.py` | **Reference implementations of the templates**: quantization, block-double-buffered linear layers, RMSNorm, RoPE, attention, SiLU, residuals, embedding gather; §8.8's prefetch and SiLU fusion are exactly the scheduling C4 is to automate |
| `llm/export_w8a8.py` | **Weight packing rules**: B-strip layout, per-channel scales. IREE's data tiling must produce the same layout |
| `llm/sa_funcsim.py` | **The gold standard and cost model**: executes descriptor lists bit for bit; the core of the simulator service (§5.5) |
| `llm/runtime.py` (`LlamaDevice`) | **Behavioral model of the HAL driver**: CMA layout, parameter blocks, command ring, interrupts, cache maintenance |
| `DescList` in `driver/pynq_matmul.py` | Reference for the descriptor encoding; the plugin's serializer is byte-identical to it |
| `firmware/rt`, command ring, fetch unit, scheduler | Unchanged |
| `iree-sa/l0` | The armv7 cross-compilation environment, libm shim, embedded ELF checks; IREE 3.11.0 (e4a3b0405d) |
| Board scripts, counters, `l4_phase_cost.py` | C4's measurement tools |

---

## 3. Frontend: quantized model and export (C0)

### 3.1 Model definition (`iree-sa/frontend/qllama.py`)

A llama with **explicit quantization**, written in PyTorch, corresponding step by step to `DeviceModel.forward`.
It does not rely on the compiler's "automatic quantization", because then the frontend's and the device's numerics would not match.

- **Weights**: int8 tensors plus per-channel fp32 scales (the quantization results of `export_w8a8`), as module parameters.
- **Activation quantization**: written out explicitly as `amax = max(|x|)`, `s = amax * (1/127)`, `inv = 1 / s`,
  `q = clamp(round_half_even(x * inv), -127, 127).to(int8)`.
- **int8 matmul**: `int32 = x_q @ W_qᵀ` (e.g. `torch._int_mm`, or an int32 matmul), then `y = float(acc) * s_w * s_x`, two roundings, order unchanged.
- **RMSNorm, RoPE, softmax, SiLU, residuals**: the same formulas and order of operations as `DeviceModel`:
  - RMSNorm uses `ss * (1/dim) + 1e-5`, then rsqrt;
  - RoPE uses the SWAPNEG form: `x*cos + swapneg(x)*sin`;
  - softmax does max, exp, sum, recip and the multiply as one step each;
  - SiLU uses `h1 * recip(1 + exp(-h1))`.
- **KV cache**: int8, one static scale each for K and V per layer; in the exported module a **mutable global** (`util.global`), one row written per position.
- **Signature of the decode function**: `logits = decode(token: i32, pos: i32)`. token and pos are scalar inputs; quantities depending on the sequence length are computed from pos.
  - attention only looks at the first pos+1 positions: a dynamic length of `pos_pad = ceil((pos+1)/D)·D` plus a mask;
  - pos_pad is a dynamic dimension that becomes a push constant after compilation.
- **Embedding**: `emb_q[token] * emb_s[token]`, a gather.

### 3.2 Numeric contract

Compiler and device must both keep the following conventions, so that IREE's compiled results are bit-identical to `DeviceModel(sfu=SfuExact)`:

| Item | Convention |
|---|---|
| fp32 arithmetic | RNE, FTZ, canonical NaN (§3.2 of the LLM plan); one rounding per operator |
| Forbidden transformations | Reassociation, contracting multiply-add into FMA, splitting reductions (split-k / split reduction), changing the semantics of max/min for NaN and ±0 |
| Reduction order | First each lane accumulates sequentially in group order, then the D lanes are merged by a binary tree pairing them (LLM plan §6.4). The compiler can only generate this order, and the frontend's reference (`DeviceModel`) computes in this order too |
| Special functions | `math.exp` → SFU EXP, `1/x` and `arith.divf` → RECIP (then a multiply), `math.rsqrt` → RSQRT; all specified approximation algorithms (`llm/sfu.py`) |
| int8 quantization | RNE, saturation to ±127, NaN → 0 |

- torch itself is not a bit-exact reference (its exp and reduction order differ); torch results are only used to check accuracy (error ranges and L0's top-1 threshold).
- These conventions are declared in the compiler as fast-math attributes and checked by the plugin when matching. Dispatches that do not satisfy them raise an error; there is no silent approximation.
- **Tiled reductions** (needed for large models, §8.9): integer accumulation (EX accumulating int32 across K blocks) is order independent and can be tiled freely. An fp32 reduction (softmax's max / sum, sums over long vectors) that needs tiling must be rewritten **explicitly** into tiled form in the IR by preprocessing (e.g. online softmax: the hardware order above within a block, blocks merged in block order, with a fixed scaling scheme), and the oracle executes the IR, so it stays bit-exact. Code generation must not change the order of an fp32 reduction on its own; max / min are the exception (order independent, e.g. an abs-max halving tree).

### 3.3 Export

- `iree.turbine.aot.export` exports `decode`; the weights are externalized into a parameter file (.irpa) with `externalize_module_parameters`,
  loaded into CMA by the runtime rather than embedded in the .vmfb.
- The KV cache is exported as a mutable global; confirm that the generated program **updates it in place** rather than copying the whole cache every token (§12).
- First compile with the existing `llvm-cpu` (armv7) backend and run on the host and the board.

### 3.4 C0 outputs and acceptance

- `qllama.py` compared with `DeviceModel` on the host: logit errors within fp32 rounding, argmax agreement ≥ 99%.
- The `llvm-cpu` version runs on the board, matching the host results (within the accuracy of armv7's math library).
- **Dispatch inventory**: export each dispatch's linalg IR with `--iree-hal-dump-executable-sources-to`, listing:
  - the shape, fusion and count of each kind of dispatch;
  - which hand-written segment each corresponds to (which part of `compile_layer` / `compile_model`).

  This inventory decides which templates C3 writes; it is the plan's **first decision point** (§13).

### 3.5 C0 results (2026-09-27)

**Implementation**: `compiler/frontend/qllama.py` (the quantized model), `compiler/frontend/export.py` (export, compile, compare, inventory, board bundle).

**Numerics**
- **eager QLlama**:
  - versus `DeviceModel(sfu=Sfu64)`, each token's logits differ relatively by about 1e-7 and the KV cache is identical, showing the operation steps correspond one to one;
  - versus the `SfuExact` version, EXP's approximation (error about 1e-5) occasionally flips int8 quantization by ±1, about 2% difference, with argmax still identical.
    What the device must finally match bit for bit is the `SfuExact` version.
- **IREE compiled (llvm-cpu, host)**: stories15M for 12 consecutive steps within 4e-7 of the eager version, argmax 12/12 identical to `DeviceModel`.
  The KV cache, a mutable global, persists between calls.
- **On the board (armv7 Cortex-A9, L0's `iree-run-module`)**: two single-step cases, pos 0 and pos 5 (T = 8), match the host output (threshold 1e-3): **C0 board PASS**.
  Multi-step decode needs consecutive calls within one process and is tested with C1's runtime.

**Practices settled during export and compilation** (kept by C2, C3)

| Problem | Practice |
|---|---|
| Indexing with a 0-D tensor like `emb_q[tok]` is treated as a data-dependent scalar and export fails | gathers are always written as `index_select` |
| Writing the KV cache per layer (`kc[l].index_copy_`) exports as index_put plus a sliced write-back, which IREE's CPU backend rejects ("write affecting operations on global resources are restricted to workgroup distributed contexts") | The KV cache is flattened to (layers·seq_len, kv_dim), with one `index_copy_` per layer writing row l·S + pos |
| The dynamic length of attention | T comes from the dimension of an input `valid[T]`, a dynamic shape backed by an input, so export is stable; it becomes a push constant after compilation |
| Constant-expression hoisting transposes int8 weights ahead of time and extends them to int32 (4× the memory, and the matmul sees i32×i32) | Compile with `--iree-opt-const-expr-hoisting=false`, so linear-layer dispatches read int8 weights directly |
| armv7 link failure: `__aeabi_idiv` undefined (the Cortex-A9 has no hardware integer division, needed by dynamic-shape index computations) | Add `__aeabi_{u,}idiv{,mod}` to `iree-sa/l0/armv7_libm_shim.c`, matching C's `/` and `%` on 4 million inputs |
| Weights | Externalized into a parameter file (.irpa, 16.9 MB, the embedding table and the classifier sharing one copy) |

**Dispatch inventory** (`compiler/frontend/inventory/stories15M_dispatches.md`): **241** dispatch calls per token, 65 executables after deduplication.

| Category | Calls per token | Corresponding hand-written segment | Template |
|---|---|---|---|
| Elementwise chains | 122 | RoPE, SiLU, residuals, quantization scaling and rounding, softmax's exp, etc. | VE expression compiler |
| Reductions | 56 | RMSNorm's sum of squares, quantization's amax, softmax's max and sum | `reduce` |
| int8 linear layers (GEMV + dequantization) | 24 + classifier 1 | `compile_layer.linear` | `qlinear` |
| gather | 14 | Embedding rows, RoPE rows | `gather` |
| KV cache row writes (scatter) | 12 | `compile_model.kv_append` | `kv_write` |
| Attention Q·Kᵀ, P·V (per-head batch_matmul, dynamic length T) | 6 + 6 | `compile_model.attention` | `attn_scores`, `attn_pv` |

**Implications for C2, C3**
1. **The linear layers' form is exactly what the template wants**: the dispatch reads a row-major int8 W (out × in), sign-extends and transposes it internally,
   the vecmat accumulates in i64 and truncates to i32 (int8 × int8 cannot overflow for k ≤ 768, so it can be treated as int32 accumulation),
   and the dequantization tail (`sitofp`, × s_w, × s_x) is already fused into the same dispatch.
2. **Weight layout**: the parameter file has row-major W, while the device wants the packed B-strip layout. Two ways:
   - compiler side: data tiling, packing at compile time;
   - driver side: rearrange once when loading the parameter file.

   C2 uses the driver-side way first (simple, and identical to `export_w8a8.pack_b`), leaving data tiling to C4.
3. **Elementwise operations and reductions make up 178/241**, and are cut very finely (one quantization = a reduction dispatch plus an elementwise dispatch).
   The VE expression compiler is C3's focus; C4's fusion and cross-dispatch scheduling matter most for speed.
4. **Per-layer constants** (a_k, a_v, 1/s_k …) are written into the dispatch bodies, so layers with the same shape cannot share executables (one attention per layer).
   Improvement: the frontend passes them in as parameter tensors so the layers' dispatches deduplicate.
5. **Data between dispatches goes through DDR**: 241 dispatches, possibly with a FENCE between each.
   After C3, first quantify this overhead with per-segment timing, then set C4's priorities.

---

## 4. Executable format sa-desc-v1

### 4.1 Design points

- **An executable = several entry points**, each corresponding to a dispatch function, whose content is a **position-independent** descriptor template:
  - only relative JUMP / CALL / LOOP;
  - ending with RET, CALLed from the command buffer.
- **Loaded once**: the driver copies the whole executable into CMA and records each entry point's physical address. Each later dispatch just CALLs that address.
- **All DDR addresses are relocated through BASEs**; no absolute addresses appear in templates, so one template can serve different buffers.

### 4.2 File layout (little endian)

| Part | Content |
|---|---|
| Header | Magic `SADESC1\0`, version, D, number of entry points, bytes of the template area, hardware requirements (CAPS bits: CMDX, FPVE) |
| Entry point table | Per entry: name, offset and count of the template in the template area, number of bindings, number of push constants, PARAMs used and the size of the LDPARAM constant block, SPAD / ACC usage, estimated cycles (from the cost model) |
| Template area | 64-byte-aligned descriptors, entry points back to back |
| Debug section (optional) | The source dispatch and linalg location of each descriptor, for locating errors |

Generating it with IREE's existing flatbuffer tools is fine too; what matters is that the content matches the table. The plugin's serializer must be byte-identical to `DescList` (unit tested).

### 4.3 Calling convention

| Resource | Convention |
|---|---|
| BASE0–15 | binding i → BASE i. A dispatch has at most 16 bindings (constant weights, scales, inputs, outputs, KV cache …). A BASE holds the binding's physical address plus offset |
| PARAM0–5 | The first 6 push constants (pos+1, pos_pad, pos_pad/D, offsets …), written by the command buffer with SETREG |
| PARAM6, PARAM7 | **Used inside templates**: loop variables, intermediate results of LDPARAM. Callers cannot assume they survive a CALL |
| More constants | Beyond 6, the command buffer writes the constants into a block of DDR (a constant block, the last binding), and the template reads them into PARAM6/7 with LDPARAM as needed |
| On-chip memories | A template owns SPAD and ACC exclusively; no on-chip data is kept between dispatches (data goes through DDR, LLM plan §10.3) |
| Barriers | A template does not require the engines to be idle at its end; dependencies between dispatches are handled by the command buffer's FENCE (§5.2) |

### 4.4 The actual format (C1–C4)

The implementation does not use flatbuffers but a custom little-endian binary; its definition and read/write tools are in `compiler/runtime/tools/sadesc.py` (the C loader `runtime/sa/sa_loader.c` reads the same layout):
- **v1** (C1): header, entry point table (name, template offset and count, number of bindings / push constants, estimated cycles), template area, name area.
- **v2** (C3): each entry point can carry a **load table** (`BASE r = binding b + f(constant)`, `PARAM r = f(constant)`, `f(c) = ((c·mul) >> shift) + add`), evaluated by the driver. IREE passes binding offsets as push constants, so templates contain only static offsets, and one executable can serve several layers.
- **v3** (C4): a 32-byte extension per entry point: a **prefix** (weight LDs that can execute early, through BASE15, ending with RET), a **head** (the LDs at the start of the body, ending with RET, after which the driver can insert another dispatch's prefix), read / write binding masks, whether SPAD_B is used, and the BASE register the prefix originally used. Templates cannot use BASE15.

---

## 5. Runtime: the sa HAL driver (C1)

### 5.1 Object mapping

Starting point: the skeleton of a new driver in the IREE runtime (per the chosen version, e.g. the null driver), following the structure of the local-sync driver.

| IREE HAL object | Implementation in the sa driver |
|---|---|
| driver / device | Opens the overlay (the program BRAM holding the mailbox and command ring, through mmap), loads `rt_fw.bin`, waits for `FW_STATE = READY` |
| allocator / buffer | Allocates physically contiguous buffers from CMA (u-dma-buf or the CMA interface PYNQ uses); each buffer records its physical address |
| Memory types | Device local, host visible but **not coherent**: invalidate on `map`, write back caches on `flush` (the rules of memory_model.md) |
| executable / executable cache | Parses sa-desc-v1, copies the template area into CMA, entry point → physical address |
| command buffer | Generates a descriptor list directly while recording (§5.2) |
| queue_execute | Submits the list as one command-ring entry (doorbell = `RING_TAIL`) |
| semaphore | Timeline value ↔ command-ring seq; waiting uses the UIO interrupt (`notify_irq`) |
| Parameter file (.irpa) | Weights loaded into a CMA buffer as a constant binding |

### 5.2 Command-buffer encoding

Each recorded operation appends descriptors to the list:

| HAL operation | Descriptors generated |
|---|---|
| `dispatch(executable, entry, bindings, constants, workgroups)` | Several SETREGs (3 registers each) writing BASE0..n and PARAM0..5; with more than 6 constants, the constant block is written first (by the host, as a binding); then `CALL` the entry point's absolute address |
| `execution_barrier` | `FENCE` (mask 0 = all engines). IREE inserts barriers only where needed, consistent with "DDR dependencies are not tracked by hardware" |
| `fill_buffer` / `copy_buffer` / `update_buffer` | Built-in templates of the driver: LD / ST copies; VE writes an immediate, then ST; small updates are written into CMA directly by the host and flushed |
| End | `END` |

- One `queue_execute` corresponds to **one** command-ring entry, so all dispatches of a token pass through PicoRV32 only once (as with L4's static list).
- Command buffers can be reused (IREE's reusable command buffers): dynamic values come in through push constants and the list itself is unchanged, just like the hand-written path's "only the parameter block changes per token".
  If IREE re-records every time, that is only ARM-side overhead, functionally the same.
- **Implementation (C4, §7.1)**: the command buffer records dispatches first, and the driver lays out the list at flush; FENCEs at barriers choose their masks by conflicts in the DDR read/write ranges, and prefixes are placed into earlier dispatches. C1–C3 used one list per dispatch.

### 5.3 Memory and addresses

- **Physical addresses**: the device only sees physical addresses. A buffer's physical address is fixed at allocation; binding offsets are added at SETREG.
- **Alignment**: DMA needs 8-byte alignment, descriptors and templates 64-byte alignment; the allocator aligns everything to 64 bytes.
- **Transient buffers**: IREE Stream allocates transient buffers for intermediate results between dispatches; the driver carves them from a CMA pool, avoiding frequent system calls.
- **Sizes**: stories15M's weights are about 24 MB, the KV cache 0.9 MB, transient buffers small; 110M about 115 MB, needing the CMA size checked (LLM plan §9.1).

### 5.4 Synchronization

- The command ring completes in order, so wait semaphores on the same queue are satisfied naturally; the driver only needs to handle host waits.
- Each entry carries the IRQ flag: on completion `rt_fw` issues `mat_notify`, the driver waits for the interrupt on UIO, then reads the completion record and advances the semaphore's value.
- Errors: when a completion record's status is non-zero, the semaphore is put into a failed state, reporting the index of the failing descriptor (in the completion record), mapped back to the dispatch name (debug section).

### 5.5 Transport layer: board or simulator

The driver's bottom layer is a replaceable "transport":

| Transport | Implementation | Use |
|---|---|---|
| `board` | mmap of the program BRAM (mailbox, doorbell), CMA, UIO | Running on the board |
| `sim` | Connects over a local socket to a Python simulator service: the service holds a simulated DDR (`SaFuncSim`), executes command-ring entries and returns completion records | **Developing the compiler on the host**: the whole IREE program runs on the host, with the device part executed bit for bit by the functional simulator |

- Under `sim`, physical addresses are addresses within the simulated DDR window, and the allocator allocates within the window.
- The same .vmfb must give bit-identical results on both transports. This is also a test of the driver itself.

### 5.6 C1 acceptance

- Without the compiler: wrap templates generated by `compile_layer.py` into sa-desc-v1 by hand and execute them with a small IREE program
  (or a C test program calling the HAL API directly):
  - a quantized linear layer bit-identical to `DeviceModel.linear` on both `sim` and the board;
  - 1000 consecutive submissions without errors or timeouts, with semaphores and interrupt counts consistent.

### 5.7 C1 implementation and results (2026-09-27)

The code is in `compiler/runtime/` (driver, tests), `compiler/sim/` (simulator service, command-ring emulation), `compiler/runtime/tools/` (sa-desc-v1 read/write, test executables).

**Structure**: the sa device = IREE's local-sync device + three components of our own, without rewriting HAL objects:

| Component | File | Purpose |
|---|---|---|
| context and arena | `sa/sa_context.c` | A first-fit allocator over the device-visible window (CMA on the board / the shared-memory DDR of sim), used as the heap allocator's `data_allocator` so every buffer has a physical address; assembles and submits dispatch lists |
| loader | `sa/sa_loader.c` | Parses sa-desc-v1 and copies templates into the arena; `issue_call` submits only at workgroup (0,0,0): `SETREG` (BASE i = physical address of binding i, PARAM j = push constant j) + `CALL` the template + `END 0x5A` |
| transport | `sa/sa_transport_sim.c`, `sa/sa_transport_board.c` | `sim`: Unix socket + a shared-memory file, served by `compiler/sim/sa_sim_server.py` (`SaFuncSim`). `board`: rt_fw's command ring (the ring at the start of the window, completion records at +0x2000), polling `RING_HEAD` |
| registration | `sa/sa_driver_module.c`, `iree_runtime_plugin.cmake` | An IREE external HAL driver `sa` (`-DIREE_EXTERNAL_HAL_DRIVERS=sa`), visible in `iree-run-module --list_drivers` |

**Differences from §5.1–5.4** (simplifications in C1, same functionality):
- **One list and one command-ring entry per dispatch**, not one list per command buffer (§5.2). Barriers are satisfied naturally (each executed synchronously). Merging into one list is left to C4.
- **Polling `RING_HEAD`**, not using `notify_irq` (§5.4).
- **Board memory**: a static armv7 program cannot use PYNQ's CMA interface, so the PYNQ launcher (`runtime/test/board_launcher.py`) allocates the window, sets up the command ring and starts rt_fw; the program maps the window and mailbox through `/dev/mem` (`O_SYNC`, which on ARM is an uncached write-combine mapping for RAM), so **no cache maintenance is needed**. It can later switch to `sa_board_attach()` (the caller passes in already-mapped memory).
- **Errors**: the completion record's extended status is decoded into "engine + error type" (`sa_sched.v`: [11:8] code, [15:12] engine, [24] fetch busy), along with the number of descriptors decoded; sim and emu report in the same format.
- An IREE issue hit along the way: when the data and host allocators differ (split mode), heap buffers are allocated with `malloc_aligned` but freed with `free` on the aligned pointer (`hal/buffer_heap.c`). The arena looks up the containing block by address, so any pointer inside a block can be freed.

**Tests** (`runtime/test/sa_hal_test.c`, using only the IREE HAL API: registry → device → executable cache → allocator → command buffer → queue_execute → semaphore):

| Item | sim (host) | Board emulation (host, `sim/sa_board_emu.py`) | On the board (L2 bitstream, rt_fw) |
|---|---|---|---|
| qlinear (`compile_layer`'s quantized linear layer, K=64, N=160) vs `DeviceModel.linear` | Bit-exact | Bit-exact | Bit-exact |
| axpb (push constant → PARAM → dynamic field) | Bit-exact | Bit-exact | Bit-exact |
| Binding with an offset (a subrange of the same buffer) | Bit-exact | Bit-exact | Bit-exact |
| fault (LD out of range): submission fails and reports the device status | LD range error, 0x1000203, 3 descriptors | Same | Same |
| qlinear run again after the error | Bit-exact | Bit-exact | Bit-exact |
| Stress: qlinear + barrier + axpb submitted repeatedly, checking results and semaphores each time | — | 200 times, 0 errors (ring wrapped 25 times) | **1000 times, 0 errors**, semaphore 1003 = expected; 152 µs per submission (2 dispatches) |
| An executable with a mismatching D | INCOMPATIBLE reported at load | — | — |

Build: `compiler/scripts/build_sa_runtime.sh host|armv7` (runtime only, about 1 minute); board deployment `compiler/scripts/deploy_c1.sh` → `build/deploy_c1`.

---

## 6. Compiler: the sa target backend plugin (C2, C3)

### 6.1 Plugin structure

- iree-compile's target backends are compile-time plugins, so **iree-compile must be built from source**: the pip package cannot load custom backends.
  - the same version as L0's runtime (IREE 3.11.0, e4a3b0405d), with the plugin compiled in through `-DIREE_CMAKE_PLUGIN_PATHS=<iree-sa/compiler>`;
  - the build takes a lot of time and memory; limited to 4 jobs (the same convention as Vivado), run by the user in their own terminal.
- The plugin implements IREE HAL's `TargetDevice` and `TargetBackend` (per the chosen version's interfaces):

| Interface | Purpose |
|---|---|
| Default executable target | Declares the `sa` device and the `sa-desc-v1` format, with D, CAPS, SPAD / ACC sizes in the configuration |
| `buildConfigurationPassPipeline` | Chooses a template for each dispatch and checks the numeric conventions (§3.2); if none fits, errors out and prints the dispatch's IR |
| `buildTranslationPassPipeline` | Turns a dispatch's linalg IR into a descriptor template (C3: template matching plus parameterization; C5: full code generation) |
| `serializeExecutable` | Outputs sa-desc-v1 (§4) |

### 6.2 Dispatch formation and preprocessing

How IREE cuts dispatches decides what shapes the templates must handle. The plugin adds preprocessing before and after IREE's pipeline:
- **Keep high-level operators**: softmax and the like are matched before decomposition and kept as one dispatch (or with a custom dispatch region).
- **Forbid** transformations that change numerics: split reductions (split-k), reassociation; turned off in the plugin's device configuration.
- **Control fusion**: a quantized linear layer has the shape "quantize → int8 matmul → dequantize", ideally fused into one or two dispatches;
  elementwise chains fused into neighboring dispatches are acceptable, handled by the VE expression compiler (§6.4).
- **In-place KV cache updates**: confirm that a `tensor.insert_slice` into the global is implemented as writing one row in the original buffer.

### 6.3 Data tiling and weight packing

- ~~matmul operands carry encodings, resolved into packs for the sa device by the plugin~~. C2's experiments showed this path does not work today: IREE's data tiling needs the i8 × i8 → i32 matmul form (the frontend writes a matmul after extending to i32) and a plugin-defined encoding resolver; packing by hand in the torch graph gets undone by IREE's layout canonicalization (back to a runtime transpose).
  **What was done (C2)**: the plugin's preprocessing pass `iree-sa-pack-linear-weights` (`plugins/sa/target/PackLinearWeights.cpp`), run only when the target is the sa device:
  `vecmat(x, transpose(extsi(W)))` (W an i8 weight that is a constant or an immutable global) → `Wp = linalg.pack W inner_tiles=[D]` (i8 [N/D, K, D], i.e. `pack_b`) + a contraction reading Wp directly + `collapse_shape`, with the same arithmetic as the original vecmat.
- Constant weights are evaluated at compile time (const-eval), so the pack happens at compile time with no rearrangement at run time: parameters are imported at compile time (`--iree-parameter-import=model=<irpa> --iree-parameter-import-maximum-size=...`), and the packed weights are exported into a new archive (`--iree-parameter-export=model=<packed.irpa>`). The results are byte-identical to `export_w8a8.pack_b` (checked by `compiler/tests/test_c2.py`).
  Since the weights are packed as i8 first, constant hoisting can be turned back on (threshold `--iree-opt-const-expr-max-size-increase-threshold=0`, so weights are no longer hoisted into i32 as in C0): the whole stories15M model compiles in 4.5 seconds, all 5 kinds of linear layers in the pack layout, 16.1 MB of packed parameters.
- Activation side: the row-replicated A strips are generated inside templates (VE's DIV-D replication, or L5b's detour through DDR + LD INTERLEAVE), not through IREE's pack.

### 6.4 Template library and matching (C3, approach A)

Matching goes by the **structure** of the dispatch function body: which linalg operators, iterator types, indexing maps, element types, and arith / math operations in the body.
Each template is a parameterized descriptor generator (implemented in C++, ported from `compile_layer.py` / `compile_model.py`):

| Dispatch form | Template | Reference implementation |
|---|---|---|
| Activation quantization (max(abs) reduction + multiply by the reciprocal + convert to int8) | `quant_act` | `compile_layer.quant_act` |
| int8 × int8 → int32 matmul, with or without a dequantization tail (multiply by the scale vector, by s_x) | `qlinear`: tiling, double buffering, dequantization | `compile_layer.linear` |
| Elementwise chains (`linalg.generic` with all parallel iterators) | **VE expression compiler**: each arith / math operation in the body → one VE (OP, ×A+B, FUNC), indexing maps → index modes (LIN, MOD, DIV, IMM); intermediates kept on chip | RoPE, SiLU, residuals |
| Row reductions (sum / max, reduction iterators) | `reduce`: REDUCE + ROWLEN | RMSNorm's sum of squares, amax |
| softmax (masked) | `softmax`: max, exp, sum, recip, multiply (VALID = dynamic length) | The softmax part of `compile_model.attention` |
| Attention scores / P·V (small int8 matmuls, dynamic length) | `attn_scores`, `attn_pv` (TRANSPOSE, INTERLEAVE, dynamic repeat / Kt) | `compile_model.attention` |
| transpose | `transpose` | TRANSPOSE |
| gather (embedding rows) | `gather`: LDPARAM computes the address + LD + read the scale | `compile_model.embed` |

- The VE expression compiler is the most general part of C3: it is already a small code generator, just limited to elementwise operations.
- The templates' on-chip memory planning and block sizes follow the rules of `compile_layer.Layout` (§6.5).

### 6.5 Memory planning within a dispatch

- SPAD_A: A strips; SPAD_B: weight blocks (the two banks alternating); ACC: staging and fp32 intermediates.
- Block size: the largest that divides the number of output blocks and fits one SPAD_B bank and the ACC staging area (`Layout.chunk_tiles`).
- When on-chip space is insufficient (long vectors of large models), the template errors out and preprocessing cuts the dispatch smaller.

### 6.6 Unsupported dispatches

- **Default**: error out, printing the dispatch's IR and the closest template. That way the coverage is always explicit.
- **Optional**: mixed CPU and accelerator execution, with unsupported dispatches on `llvm-cpu`, using IREE's multi-device and affinity support.
  This part of IREE is still evolving; a fallback, not the main path (§12).
- The kinds of dispatches llama decode uses are limited (the hand-written path covers them all), so the goal is **everything on the accelerator**.

### 6.7 C2 results (2026-09-27)

The first vertical slice: a torch int8 linear layer → iree-turbine → iree-compile with the sa plugin → .vmfb + packed .irpa → `iree-run-module --device=sa` (the C1 driver).

| Component | File (`compiler/plugins/sa/`) | Purpose |
|---|---|---|
| Preprocessing | `target/PackLinearWeights.cpp` | Linear-layer weights → B-strip layout (§6.3) |
| Translation | `target/LowerWorkgroupCount.cpp` | Each export's workgroup count is (1, 1, 1): one dispatch = one template, executed once |
| Matching | `target/Match.cpp` | Strictly matches the dispatch's structure: the load of x (i8 or i32), `acc[t, j] += ext(x[k]) * ext(Wp[t, k, j])` (fill 0), the tail `(f32(i32(acc)) * s_w) * s_x` (order of operations, no fast-math), whole-buffer bindings; yields k, n and each operand's binding |
| Templates | `target/Templates.cpp`, `target/DescList.cpp` | C++ ports of `compile_layer.linear` and `DescList`; Python reference `templates/reference.py` |
| Serialization | `target/SATarget.cpp` | sa-desc-v1 (the same layout as `runtime/tools/sadesc.py`); dispatches without a template error out and print the whole dispatch |

The qlinear template (reference `templates/reference.py`):
- A strip: with x i8, LD into SPAD_A, then replicated into D rows with a DIV-D VE COPY; with x i32, LD into ACC, then an I32 → I8 COPY.
- s_x: LD one fp32 (DMA works in 8-byte units), broadcast to the whole word with a max reduction over only the first element (`valid = 1`). VE's index modes work in words and cannot broadcast within a word.
- Then the same as `compile_layer.linear` (tiling, double buffering, dequantization; LOOP_END beyond 4 blocks, with offsets in the template-private PARAM6/7).

Tests (`compiler/tests/test_c2.py`, on the board `compiler/tests/board_c2.py`):

| Case | Shape | Template vs Python reference | Packed weights vs `pack_b` | sim (D=8, 16) | Board (D=8) |
|---|---|---|---|---|---|
| qkv_i8 | k=288, n=864, x i8 | Byte-identical | Identical | Bit-exact | Bit-exact |
| qkv_i32 | k=288, n=864, x i32 | Byte-identical | Identical | Bit-exact | Bit-exact |
| w2_i32 | k=768, n=288 | Byte-identical | Identical | Bit-exact | Bit-exact |
| long_i8 | k=288, n=2048 (LOOP_END form) | Byte-identical | Identical | Bit-exact | Bit-exact |

Also: the C++ template generator, compiled alone, was compared byte for byte with the Python reference on 56 configurations (D=8/16, including the classifier shape with n=32000, and shuffled binding orders); non-sa targets (vmvx) are unaffected by the preprocessing pass.

### 6.8 C3 implementation plan (2026-09-27, based on stories15M's actual dispatches under the C2 flow)

Compiling stories15M with C2's flow (weight packing + constant hoisting + imported parameters) gives 64 dispatches (`build/c3/sources`). Compared with C0's inventory, a few points decide C3's approach:

| Observation | Example | Response |
|---|---|---|
| One dispatch is reused by several layers, binding offsets come from push constants (several constants packed in one resource), and several tensors at different offsets share a binding | Linear layers (shared by 6 layers), quantization | **sa-desc's register load table** (A below) |
| More than 6 push constants; 64-bit values assembled from two constants (`extui`/`shli`/`ori`); the dynamic length T appearing as a workload ordinal | Attention's 7 constants | The load table puts only the constants the device needs into PARAMs |
| Data-dependent masks `select(valid[t] > 0.5, x, -inf)` | softmax's max / exp | The frontend switches to a **length-based mask** `arange(T) <= pos` (the mainstream approach: attention kernels take the sequence length), and the compiler recognizes "index ≤ scalar" prefix masks → VE's VALID field (LDPARAM reads pos) |
| torch's exp, 1/x, rsqrt | SiLU, softmax, RMSNorm, quantization | Mapped to the hardware SFU (exp, recip, rsqrt): results as defined by DeviceModel(SfuExact), bit-identical to L5 |
| `to_i8` expanded into NaN→0, ±inf clamping, roundeven, ±127 clamping, fptosi | Quantization, KV writes | Recognize the whole pattern → VE's I8 output conversion (which has exactly these semantics) |
| gather: the row number of `tensor.extract` comes from an i64 scalar on the device | Embedding, RoPE tables | LDPARAM reads the row number to compute the address → an LD with a dynamic address |
| scatter: KV rows written into the cache | K, V of each layer | LDPARAM → an ST with a dynamic address |
| i64 scalar arithmetic on the device | `pos + l·S` | I32 arithmetic on the VE (values < 2^24, exact in fp32); the high word cleared |
| Attention's two small int8 matmuls (dynamic T) | Q·Kᵀ, P·V | Port the templates of `compile_model.attention` |

The implementation has several parts (C++, `plugins/sa/target/`; each with a Python reference or funcsim tests):

- **A. sa-desc register load table**: each export carries a table, `BASE r = physical address of binding b + push constant j` (or + 0), `PARAM i = push constant j`, computed by the driver at SETREG. Templates contain only static offsets. This is like Vulkan's dynamic offsets, with no extra cost on the device. The format version goes to 2 (`sadesc.py` and the driver's loader updated together).
- **B. Dispatch analysis**: parse the dispatch function into a program of "tensors + operations": each load / store's (binding, offset expression, static slice, shape, type), and the use of each push constant (offset, length, other).
- **C. On-chip memory model**: tensors flattened row major, element e in lane e%D of word e/D; fp32 / i32 in ACC, i8 in SPAD; scalars and "one per row" values stored as broadcast words (all lanes identical). Dynamic dimensions allocated by their upper bound (a plugin option, default seq_len).
- **D. VE expression compiler** (C3's core): the body of a `linalg.generic` (elementwise or row reduction) → a sequence of VE instructions.
  - operand access patterns: as is, scalar broadcast, row broadcast, pair swap with negation (SWAPNEG), gathered rows;
  - constants folded into A / B / imm; exp / recip / rsqrt / abs on the SFU;
  - reductions with REDUCE (sum / max, ROWLEN, VALID); to_i8 with the I8 output conversion; prefix masks with VALID.
- **E. Contraction templates**: qlinear generalized to bindings with offsets and fused tails (e.g. a residual add, with the rest of the tail handed to D); attention's Q·Kᵀ, P·V.
- **F. gather / scatter / integer scalars**: see the table above.
- **G. Runner**: a C program (IREE runtime API) calling the model token by token and writing out the logits, the counterpart of L5's `LlamaDevice` (the KV cache is a global of the module and must persist across tokens within one process).
- **H. Verification**:
  - each template has a Python reference or funcsim tests;
  - end to end: IREE-compiled stories15M on the sim, per-token logits bit-identical to DeviceModel(SfuExact);
  - the text generated on the board identical to L5.

Order: H's framework + frontend changes → A → B/C/D (elementwise, reductions) → E (generalized qlinear) → F → E (attention) → G → end to end → board.

### 6.9 C3 results (2026-09-27)

**The text that IREE-compiled stories15M generates on the board is identical to L5's hand-written path, and every step's logits are bit-identical to DeviceModel(SfuExact).**

| Item | Result |
|---|---|
| Compile | `compile_sa.sh`: 4.4 s; all 64 dispatches get templates from the sa backend (no CPU fallback); sa.vmfb 10.3 MB + sa_packed.irpa 16.1 MB |
| Per-dispatch differential test (`tests/dispatch_check.py`: the oracle executes the IR semantics vs the template run in funcsim, at every call site) | 64/64 bit-exact |
| End to end, sim (`tests/test_c3.py`, `sa-llm-run` + `sa_sim_server.py`) | 20 steps (12 prompt + 8 greedy) bit-exact; about 1.4 s / token (Python funcsim) |
| End to end, board (`tests/board_llm.py`, L2 bitstream, rt_fw) | 76/76 steps bit-exact, generated tokens identical to the host reference; **12.6 tok/s** (L5 hand-written: 15.1 tok/s); 242 dispatches per token |

Implementation (`plugins/sa/target/`):

| File | Content |
|---|---|
| `Codegen.cpp` | Dispatch program analysis: push constants → `Lin` (a constant or f(one push constant), including 64-bit assembly), binding regions and BASE / PARAM allocation (the load table), on-chip tensors (two layouts, packed and broadcast, converted with replication + TRANSPOSE), the VE expression compiler, qlinear (with fused tails), attention, scatter, i64 scalars |
| `CloneCheapProducers.cpp` | Preprocessing: elementwise values depending only on indices and scalars (masks) are cloned into each consumer, keeping the "index ≤ pos" structure |
| `Templates.cpp` | `emitLinear`: `compile_layer.linear` with offsets and a per-block tail callback (C2's byte comparison unchanged) |
| `DescList.cpp` | Adds dynamic fields, TRANSPOSE, LDPARAM, FENCE |

Key points of the expression compiler: each VE is `FUNC(OP(s1', s2)·A + B)`, and the IR's operations are folded in order into these stages (constants → A / B / IMM, `1/x` / exp / rsqrt / abs → SFU, negf → A = −1); an identity B uses −0 (`y + (−0) = y`, keeping the sign of zero); scalar broadcast uses "a max reduction over only the first element"; prefix masks → VALID (LDPARAM reads pos); the whole `to_i8` chain → the I8 output conversion; gather indices are decided by evaluating them point by point over the iteration space (row gather, pair swap).

Decisions compared with the plan:
- **Static attention length** (`export.py --static-len`: valid fixed at seq_len, masked by pos). A masked softmax over a dynamic T needs 4 dynamic fields, LEN, ROWLEN, VALID and the per-row period, while a descriptor has at most 2. With a static length the results are still bit-exact (masked positions have exp = 0, the extra +0 in the sum does not change partial sums, and rows with P = 0 contribute 0 to P·V), at the cost of attention always computing over 256 positions. Compiling several versions bucketed by length is an option for C4.
- **Frontend constants written as exact fp32 values** (`qllama.f32c`): `1.0 / 288` became, after export, a constant one ulp smaller than `F(1/288)`, the only source of end-to-end differences (all per-dispatch tests passed; located by bisecting layer by layer and step by step with `QLlama(n_layers, tap)`).
- One executable reused by several layers: offsets are computed in the driver through the load table, so 64 executables serve 242 dispatches.

---

## 7. Performance: scheduling across dispatches (C4)

After C3, the IREE version will be slower than the hand-written path (21.8 tok/s), because:
- data between dispatches goes through DDR, with a FENCE between every pair of dispatches;
- the hand-written path's cross-layer optimizations of §8.8 (prefetching the next layer's block 0, SiLU fused into W13's block loop) are outside the scope of a single dispatch.

C4's approach, by gain:
1. **FENCE only where needed**: add a FENCE only when DDR reads and writes really depend on each other; on-chip dependencies are left to the scoreboard.
2. **Cross-dispatch prefetch**: the command buffer knows what the next dispatch is, so the driver (or the plugin at link time) can move the LD of the next dispatch's first weight block earlier.
   This needs templates to declare an interface like "block 0 is prefetched by the caller" (the same as `compile_layer.linear(preloaded=…)`).
3. **Larger dispatches**: fuse everything fusable within a layer in preprocessing (e.g. W13 + SiLU), then schedule inside the template as in §8.8.
4. **Cost model**: extend the functional simulator into cycle estimates calibrated with board counters; the plugin uses it to choose block sizes and fusion.

**Acceptance**: the IREE version's cycles per token within ≤ 10% of the hand-written path, with per-segment timing in the style of `l4_phase_cost.py` explaining where the remaining gap is.

### 7.1 C4 results (2026-09-27, on hold)

**On the board: 2,591,234 cycles per token (C3 started at 3.63M), wall clock 17.7 tok/s (C3: 12.6; L5 hand-written path: 15.1), 76/76 steps bit-exact, text identical to L5.** The hand-written path (prefetch + SiLU fusion) is 2,293,970 cycles, a 13% gap, short of ≤ 10%. By the user's decision the remaining optimizations are on hold and C5 comes first: large fusions are more natural in generic code generation, and C4's runtime mechanisms carry over to C5 as they are.

| Step (commit) | Content | Cycles / token | tok/s (wall clock) |
|---|---|---|---|
| C3 start (05d3c66) | Static attention length, 242 dispatches, one list per dispatch | 3.63M | 12.4 |
| 1 Dynamic attention (a377e75) | Attention over the actual length (`--pad=8`); [H, T] values processed row by row so that each descriptor has at most 2 dynamic fields | 2.93M | 14.4 |
| 2 Aggressive fusion (173457e) | `--iree-dispatch-creation-enable-aggressive-fusion`: 242 → 200 dispatches | 2.91M | 14.9 |
| 3 Compute once (322c271) | Scalar / per-row values computed only once; identical LDs within a region shared | 2.69M | 15.9 |
| 4 Batched submission (2a24101) | One list per command buffer (the driver's own sa device / command buffer, modeled on IREE's local-sync); 7 lists per token; host overhead about 9 ms → 3.4 ms | 2.66M | 17.4 |
| 5 List scheduling (7b47707) | sa-desc v3; the driver lays out the list at flush: the LD of a linear layer's block-0 weights is placed after the head of an earlier dispatch that does not use SPAD_B, overlapping its VE computation; FENCE masks chosen by DDR conflicts (read-after-write waits only for ST, write-after-read only for LD) | 2.610M | 17.9 |
| 6 abs-max halving tree (e0f094e) | \|x\| in pipelined mode, elementwise MAX halving, and finally a REDUCE over a few groups (REDUCE processes one group at a time); values ≥ +0, so order does not affect the result | 2.591M | 17.7 |

Measurement tools: `SA_PROFILE=1` (one list per dispatch, device cycles and host time per entry point), `SA_PROFILE=batch` (per list, plus counts of barriers / FENCEs / prefixes), `board_profile.py [--batch] [NAME=VALUE]`.

Tried and dropped:
- **Placing the prefix before this dispatch's FENCE**: saves only 6k cycles. The scheduler dispatches in order, and the prefix sits after the previous dispatch's ST, which waits for VE to finish before dispatching, so nothing actually moves earlier. Conclusion: prefetch must come in the list before the VE work it is to overlap (hence step 5).
- **Making block 1's weights a prefix too**: about 80k cycles slower. The template's double buffering already runs block 1's LD in parallel with block 0's EX; moving it earlier only occupies the single LD engine and delays the host dispatch's loads.
- IREE's `fuse-multi-use` and `element-wise-fuse-multi-reduction` do not change the dispatch count; `enable-aggressive-reshape-movement` brings 200 down to 173 (61 executables), not yet verified.

Decision: **cross-dispatch scheduling lives in the driver** (§13 item 5 originally leaned toward the compiler). At flush the driver sees the real binding addresses and can judge read/write conflicts exactly; the compiler provides the information needed through sa-desc v3 (prefix, head, read/write masks, SPAD_B).

The remaining gap (per-dispatch timing, per token) and the items on hold:
- **SiLU**: 3 dispatches (exp; 1 + reciprocal, two multiplies, abs-max; quantization), about 30k cycles per layer, not overlapped. The hand-written path fuses SiLU into W13's block loop, in parallel with EX, saving about 106k cycles. This needs preprocessing to form a dispatch region itself (W13 and SiLU together), interleaved inside the template.
- **Quantization chain**: before each linear layer, 3 dispatches abs-max → scalar → quantization, each boundary a DDR round trip + FENCE; all 168 barriers are real dependencies.
- **Cost model** (§7 item 4): not done. The lesson from prefetching block 1 shows that deciding which loads go into which host needs cycle estimates.
- The classifier (1.26M, the same as the hand-written path) and each layer's weight stream are the floor: moving the weights takes about 1.92M cycles.

---

## 8. Full code generation: the sa dialect (C5)

Status: **plan** (2026-09-27). C4 is on hold (§7.1) and C5 comes first; C4's remaining fusions are done on C5's scheduler.

### 8.1 Goals, scope and acceptance

The code generator of C3/C4 is "a template library + a VE expression compiler": linear layers, attention and scatter each have a dedicated piece of C++ that only recognizes the few dispatch forms appearing in stories15M. C5 replaces it with a generic MLIR pipeline: the dispatch's linalg IR → tiling → layout → bufferization → `sahl` (tile level) → `sahw` (command level) → memory allocation, pipelining, scheduling → descriptors (§8.3). New model structures no longer need new templates.

The templates are **not deleted** but rewritten as **micro-kernels** at the `sahl` level (§8.8): generic code generation guarantees everything compiles, and hand-written micro-kernels handle the most critical operators worth tuning by hand (the same coexistence as IREE's ukernels, or cuBLAS alongside Triton).

**Unchanged**: frontend and export (C0), preprocessing (weight packing, `CloneCheapProducers`), the sa-desc v3 format, the driver and list scheduling (C1, C4), the numeric conventions (§3.2), all test tools. Hardware unchanged.

**Acceptance**:
1. **Correct**: the old code generator that emits descriptors directly (the template path of `Codegen.cpp` and the DescList version of `Templates.cpp`) is retired, and stories15M is compiled only with the new pipeline: all 58 dispatches pass `dispatch_check.py`; sim and board 76/76 steps bit-exact, text identical to L5. **Both** the default configuration (micro-kernels enabled) and `--iree-sa-ukernels=none` (generic code generation only) must satisfy this.
2. **Not slower**: in the default configuration, ≤ 2,591,234 cycles per token on the board (C4's result, `board_profile.py --batch`); per-dispatch cycles (`SA_PROFILE=1`) at most +2% over C4. The cycles of `--iree-sa-ukernels=none` are also recorded as a quality metric of the generic path.
3. **Generic**: models other than stories15M with no dedicated code written for them (§8.7 C5.5: SmolLM2-135M and small configurations of the Qwen3 structure) compile with `--iree-sa-ukernels=none`; per dispatch bit-exact with the oracle, end-to-end error versus torch fp32 within range (these models have no hand-written DeviceModel). The compiler hard-codes no numbers of any board or model (§8.9).

### 8.2 Where each part of the existing code generator goes in C5

| Part of C3/C4 (`plugins/sa/target/`) | C5 |
|---|---|
| Push constant analysis `Lin` (including 64-bit assembly, `util.assume.int`), load tables | Kept as an analysis, used by `sahw-legalize-dynamic` and `sahw-assign-registers` |
| Binding regions, BASE / PARAM allocation (PARAMs from 0 upward, private ones from 7 downward, BASE15 reserved for prefixes) | `sahw-assign-registers` |
| The two layouts of on-chip tensors (packed, broadcast word) and row strides (dynamic T) | Layout attribute `#sa.layout` + `sa-layout` propagation + `sahl-plan-memory` / `sahw-allocate` |
| The VE expression compiler (each VE = `FUNC(OP(s1', s2)·A + B)`, to_i8 chains, prefix-mask VALID, gather index detection, SWAPNEG) | Split in two steps: the body of `sahl.elementwise` lowered operation by operation into single-stage `sahw.ve` (`sahl-to-sahw`), then merged into multi-stage ones by `sahw-fuse-ve`; same rules |
| `emitLinear` (tiling, double buffering, per-block tails) | Tiling (`sa-tile`) + A / B strip layouts + `sahl.matmul` + software pipelining (`sahl-pipeline`); also rewritten as the `linear` micro-kernel (§8.8) |
| The attention template (Kᵀ strips, per-row dynamic length) | `batch_matmul` takes the same contraction path, with Kᵀ's TRANSPOSE inserted by layout propagation; the original template rewritten as a micro-kernel (§8.8) |
| scatter, gather, i64 scalars | `sahl.gather / scatter / scalar`, lowered to `sahw` (LDPARAM, dynamic DDR addresses, the SWAPNEG trick) |
| Row-by-row processing (at most 2 dynamic fields per descriptor) | `sahw-legalize-dynamic`: cut into a row loop when exceeded |
| Scalars / per-row values computed once, identical LDs shared (C4 step 3) | `sahl-uniform` (CSE with memory effects) |
| Prefix and head (C4 step 5) | Chosen by `sahl-schedule`, represented by `sahw.template` |
| abs-max halving tree (C4 step 6) | One lowering of max reductions in `sahl-to-sahw`, chosen by the cost model |
| `DescList`, sa-desc v3 serialization | Kept, now serializing from `sahw` |

### 8.3 Two dialect levels: `sahl` and `sahw`

```
linalg / scf (IREE's dispatch)
  → sahl: tile level, hardware semantics, independent of command encoding
  → sahw: command level, one operation = one accelerator command
      ├→ descriptor serialization (sa-desc v3; C5)
      └→ LLVM dialect → LLVM IR → riscv32 object file (PicoRV32 issuing commands with PCPI instructions; C7, §8.10)
```

Why two levels: high-level decisions (layouts, tile operations after tiling, memory planning, pipelining, micro-kernels) are separated from the details of command encoding (VE stages, at most 2 dynamic fields per command, registers, LOOP_END). A hardware generation change (D, number of VE stages, number of dynamic fields) mainly touches `sahw` and the target configuration; micro-kernels are written at `sahl` and not tied to the encoding; the same set of commands can be compiled into descriptors or issued one by one by RISC-V code. The descriptor path does not go through LLVM IR; LLVM is only used for code running on the processor.

**Shared attributes**:
- `#sa.mem<spad_a | spad_b | acc>`: on-chip memory spaces (with bank planning at `sahl`, word offsets at `sahw`).
- `#sa.layout<packed | bcast | a_strip | b_strip | kt_strip>`: the layout types of LLM plan §10.3 (compact vector, broadcast word, A / B strips, Kᵀ strip), part of on-chip memref types.
- Target configuration (§8.9 item 1): the verifiers and passes of both levels take capacities and capabilities from it.

**`sahl` (tile level)**: operands are memrefs (on the DDR side subspans of bindings, on the on-chip side carrying `#sa.mem` and `#sa.layout`); loops remain scf.
- `sahl.load`, `sahl.store`: moves between DDR and on-chip tiles (any shape and stride; the layout is decided by the target memref);
- `sahl.matmul`: an int8 tile matrix multiply with int32 accumulation (optionally accumulating into an existing result);
- `sahl.elementwise`: an elementwise operation with a region containing the original arith / math, with indexing maps kept;
- `sahl.reduce`: sum / max reductions (in the order of §3.2);
- `sahl.relayout`: layout conversions (transpose, DIV-D replication, packing into broadcast words);
- `sahl.gather`, `sahl.scatter`: reads and writes with data-dependent addresses (embedding rows, KV writes);
- `sahl.scalar`: i64 scalar operations (positions, row numbers).

**`sahw` (command level)**: one operation per command, plus structural operations.
- `sahw.ld`, `sahw.st`: rows, row bytes, pitch, mode (LINEAR / INTERLEAVE);
- `sahw.ex`: kt, accumulate, repeat, B / C strides, C row count;
- `sahw.ve` (stages: `op`, `func`, `m1 / m2 / p1 / period`, `A`, `B`, `imm`, `swapneg`, `reduce`, `rowlen`, `valid`, input / output types), `sahw.transpose`;
- `sahw.ldparam`, `sahw.fence`, `sahw.setreg`;
- `sahw.loop` (with a region, lowered to LOOP_END and PARAM strides; a real loop for the RISC-V backend);
- `sahw.template { sahw.prefix {…} sahw.head {…} sahw.body {…} }`: the structure of an export, matching sa-desc v3's prefix, head and body.
- **Dynamic values**: sizes and offsets are SSA values of type `index`, coming from push constants. The descriptor backend requires at most 2 dynamic fields per operation (checked by the verifier after `sahw-legalize-dynamic`); the RISC-V backend has no such limit.

**Interfaces** (implemented at both levels): memory effects (which segment of which on-chip bank, which binding is read or written), for CSE and scheduling; cost (estimated cycles, §8.5).

### 8.4 The pipeline (`buildTranslationPassPipeline`)

The serializer only translates `sahw.template` into bytes; all decisions are made in passes, each of which can be run and tested alone with `iree-opt`.

| # | Level | Pass | Key points |
|---|---|---|---|
| 1 | linalg | `sa-normalize` | Generalize named operators (batch_matmul, fill); fold extsi / trunci into contractions; recognize i8 × i8 → i32 contractions |
| 2 | linalg | `sa-tile` | TilingInterface (`scf::tileConsumerAndFuseProducersUsingSCF`): contractions tiled along N with block sizes decided by the SPAD_B bank and the ACC staging area (as `Layout.chunk_tiles`), tails fused into each block; when K does not fit one B strip, tile along K with EX accumulating across blocks via the accumulate flag (int32, order independent); overly long vectors cut into segments; all capacities from the target configuration (§8.9), block sizes optionally chosen by the cost model |
| 3 | linalg | `sa-layout` | A of a contraction → A strip (DIV-D replication of packed x), B → B strip (packed weights LD'd directly), Kᵀ → transpose, scalars → broadcast words, everything else packed; conversions inserted where layouts disagree |
| 4 | linalg | Bufferization | tensor → memref. Try IREE's one-shot bufferize first (on-chip buffers = allocs with `#sa.mem`); if it does not work out, our own bufferization (C3 already has the same analysis), see §8.11 |
| 5 | linalg → sahl | `sa-to-sahl` | copy → `sahl.load / store`; contraction → `sahl.matmul`; elementwise / reduction → `sahl.elementwise / reduce` (bodies kept as they are); scatter / gather / i64 scalars; **micro-kernels are chosen here** (§8.8) |
| 6 | sahl | `sahl-uniform` | Scalar / per-row values computed once, identical loads shared (CSE with memory effects) |
| 7 | sahl | `sahl-plan-memory` | Plan on-chip buffers by bank (the scoreboard judges dependencies by bank), double-buffered blocks in different banks, capacity checks; if things do not fit, error out with a suggested block size |
| 8 | sahl | `sahl-pipeline` | Two-stage software pipelining: after the matmul of block i, load block i+1 into the other bank (the templates' double buffering) |
| 9 | sahl | `sahl-schedule` | Coarse scheduling: hoist independent loads; choose the prefix (weight loads that can go early) and head |
| 10 | sahl → sahw | `sahl-to-sahw` | Each arith / math of a body → a single-stage `sahw.ve` (indexing map → LIN / MOD / DIV / IMM); matmul → `sahw.ex`; relayout → TRANSPOSE / DIV-D replication; gather / scatter → LDPARAM + dynamic addresses; max reductions optionally as a halving tree (chosen by the cost model) |
| 11 | sahw | `sahw-fuse-ve` | C3's folding rules: constant multiply → A, constant add → B, SFU function → FUNC, negation → A = −1, identity B as −0; to_i8 chains → I8 output; prefix masks → VALID. Only merges that do not change rounding |
| 12 | sahw | `sahw-legalize-dynamic` | Dynamic values → PARAMs (load table or `SETREG +=` on private PARAMs); more than 2 dynamic fields → cut into a row loop (descriptor backend) |
| 13 | sahw | `sahw-allocate` | Final on-chip word offsets (including VE intermediates), following the bank plan of `sahl-plan-memory` |
| 14 | sahw | `sahw-schedule` | Fine-grained list scheduling: in-order dispatch, engine queues, bank conflicts; interleave VEs between EXs; loops with many iterations → `sahw.loop`, few unrolled |
| 15 | sahw | `sahw-assign-registers` | BASE / PARAM allocation and load tables (§4.4), BASE15 reserved for prefixes |
| 16 | sahw → bytes | Serialization | `sahw.template` → sa-desc v3 through `DescList` (for the RISC-V backend `sahw-to-llvm` instead, §8.10) |

**Implementation status (2026-09-29, after C8)**: tiling (elementwise tiling: `sahl-tile`), memory placement and layout (`sahl-plan-memory`), linear-layer scheduling (`sahl-schedule`) and kernel and gather recognition (`sa-to-sahl`) are already separate passes; item 1's normalization, item 12, the fine-grained scheduling of item 14, and contraction tiling and double buffering are still inside translation. See §8.14.

### 8.5 Cost model

Needed for tiling, scheduling and the fusions C4 put on hold. The lesson of prefetching block 1 (§7.1) shows loads cannot be placed by intuition.

- **Per-operation estimates**:
  - LD / ST: bytes / 7.9 plus latency;
  - EX: kt × repeat;
  - VE: about 2 cycles per group in pipelined mode (lane folding), a fixed cost per group in multi-cycle mode (different for EXP, RECIP, RSQRT, REDUCE);
  - the start-up latency of each command.
- **Per-dispatch estimates**: a small simulator with in-order dispatch, engine queues and a bank scoreboard, implemented in C++, running at `sahl` and `sahw` (`sahl` for choosing tiling, pipelining and micro-kernels, `sahw` for fine-grained scheduling).
- **Calibration**: C4 already has per-dispatch board cycles for 58 exports (`SA_PROFILE=1`). Add a set of micro-benchmark lists measured with the same `board_profile` flow. Target ±10% error per dispatch.
- Estimates go into the "estimated cycles" field of sa-desc's entry point table, which the driver can later use to choose host dispatches.

### 8.6 Verification

| Level | Approach |
|---|---|
| Pass unit tests | `compiler/plugins/sa/test/`: `iree-opt` running a single pass + lit / FileCheck (needs the FileCheck target built) |
| Per dispatch | `dispatch_check.py` unchanged (the oracle executes the IR semantics vs funcsim running the generated list, at every call site): the main correctness guarantee |
| Old vs new | A switch `--iree-sa-codegen=templates|dialect`, both paths coexisting until C5.4: the same dispatch must give bit-identical results on both, and estimated cycles can be compared |
| Micro-kernel comparison | The same dispatch compiled once with the micro-kernel and once with `--iree-sa-ukernels=none`: results bit-identical, cycles compared (the value of micro-kernels must be backed by data) |
| End to end | `test_c3.py` (sim), `board_llm.py`, `board_profile.py`; new `compare_profiles.py`: two per-dispatch cycle tables compared item by item |
| Mutation tests | Deliberately break VE folding rules, index modes, loop counts, confirming the tests catch them |

### 8.7 Step-by-step plan (vertical slices, each step sim → board → commit)

| Step | Content | Acceptance | Effort |
|---|---|---|---|
| **C5.0 infrastructure** | The `sahw` dialect (ODS / TableGen), verifiers, serializer, the `--iree-sa-codegen` switch, pipeline skeleton, FileCheck; the existing code generator changed to output `sahw` operations, finished by the new `sahw-assign-registers` and serializer (prefix / head still given by the existing code); **target configuration** (§8.9 item 1): the config of `#hal.executable.target` carries all hardware parameters, and new code reads only from it | The sa-desc of the 58 dispatches **byte-identical** to C4 (verifying the back half of the pipeline first) | Medium |
| **C5.1 elementwise and reductions** | The `sahl` dialect; pipeline: normalize, layout (packed / broadcast word), bufferization, `sa-to-sahl`, `sahl-uniform`, `sahl-plan-memory`, `sahl-to-sahw`, `sahw-fuse-ve`, `sahw-legalize-dynamic`, `sahw-allocate`, `sahw-schedule`; funcsim and `dispatch_check` support the target configuration's parameters (D, SPAD / ACC sizes) | The roughly 40 non-contraction dispatches all pass `dispatch_check` on the new path (D = 8, cross-validated with configurations such as D = 16); mixed-path end-to-end sim bit-exact; per-dispatch cycles no higher than C4 | Large |
| **C5.2 linear layers** | Contraction tiling (N, and K tiling with cross-block accumulation when K does not fit), A / B strips, per-block tail fusion, software pipelining (`sahl-pipeline`), `sahw.loop`, prefix / head (`sahl-schedule`) | The 5 kinds of linear layers pass; plus standalone linear-layer tests such as K = 8192 (covering K tiling); board per-dispatch cycles within ≤ 1% of C4 (the template's schedule is the reference) | Large |
| **C5.3 attention, scatter, gather, i64 scalars** | batch_matmul (Kᵀ strips, TRANSPOSE), row loops for dynamic T, masked softmax, KV writes (LDPARAM), embedding gather; GQA's index mapping (KV head = Q head / group size) implemented by design | All 58 dispatches on the new path; GQA covered by a standalone attention test | Large |
| **C5.4 templates become micro-kernels** | The linear-layer and attention templates rewritten as micro-kernels that output `sahl` operations (§8.8), with `--iree-sa-ukernels`; the old direct-descriptor path and the `--iree-sa-codegen` switch retired; board acceptance | Items 1 and 2 of §8.1 (both configurations) | Medium |
| **C5.5 generality** | The two models below; frontend (HuggingFace import, generic quantization, structures such as QK-norm), runtime (configurable device window, allocator up to GBs), end-to-end tests accepted on torch error | Item 3 of §8.1; SmolLM2-135M on the board | Medium |

The frontend (§8.9 item 2) is independent Python work unrelated to code generation and can be done first, interleaved with C5.1–C5.3: try compiling SmolLM2 / Qwen3 with the existing template path to see which unsupported dispatches come up, checking C5's coverage in return.

C5.5's models (the first step of §8.9):
- **SmolLM2-135M**: Llama structure, GQA (9 Q heads, 3 KV heads), dim 576, hidden 1536, 30 layers, vocabulary 49152, embedding shared with the output layer. About 135 MB in int8, fits on the PYNQ-Z1. Tests HuggingFace import, generic quantization, GQA and different shapes; the goal is running on the board.
- **Small configurations of the Qwen3 structure**: QK-norm, GQA, head_dim 128, small dimensions (random or truncated weights). Accepted on the sim only, testing Qwen3-specific operators.
- Acceptance (§8.1 item 3): compiles with `--iree-sa-ukernels=none`; bit-exact with the oracle per dispatch; end to end within error range of torch fp32, with normal generation quality (these models have no hand-written DeviceModel).

**C5.0 results (2026-09-27)**: done.
- The `sahw` dialect (`plugins/sa/dialect/SahwOps.td`): `sahw.template` with its three sections prefix / head / body, registers (`sahw.base`, `sahw.param`, `sahw.private`, types `!sahw.base` / `!sahw.param`), all commands (ld, st, ex, ve, transpose, ldparam, fence, setreg, loop).
- Difference from the plan: instead of rewriting the existing code generator, its output (descriptor lines + load table) is **raised** field by field into `sahw` (`transforms/SahwRaise.cpp`, pass `iree-sahw-legacy-codegen`; LOOP_END → `sahw.loop` regions, register numbers → values). The old generator is replaced step by step in C5.1–C5.3; the raising is only transitional.
- New passes: `iree-sahw-split-head`, `iree-sahw-assign-registers`; the serializer `transforms/SahwSerialize.cpp` (sahw → sa-desc v3, including read/write masks, SPAD_B, prefix registers). Code generation moved into the translation pipeline, and `serializeExecutable` only serializes.
- Target configuration: the config of `#hal.executable.target` carries `d`, `spad_bytes`, `acc_bytes`, `max_dynamic`, `dyn_fields`, `bases`, `params` (`TargetConfig` in `transforms/SahwPasses.h`); `Layout`'s capacities now come from the configuration.
- Switch `--iree-sa-codegen=dialect|templates` (default dialect).
- Acceptance (`tests/test_c50.py`): the lit tests (`plugins/sa/test/`: dialect round trip, the two passes) pass; the 58 executables of stories15M are **byte-identical** on both paths; C2 (D = 8, 16) and C3 (58 dispatches, 12/12 steps bit-exact) pass on the new path. The executables are identical to the board-verified C4, so no new board run is needed.

**C5.1 results (2026-09-28)**: all elementwise and reduction dispatches go through the new pipeline.
- Pipeline: IREE's comprehensive bufferize (§13 item 6: usable directly) → `iree-sa-to-sahl` (DDR reads and writes made explicit: computation only reads and writes on-chip buffers, with `sahl.load` / `sahl.store`; i64 scalars stay in DDR) → `iree-sahl-to-sahw` → `iree-sahw-fuse-ve`.
- `sahl-to-sahw`: on-chip buffer allocation; DDR views → BASE + offset; dynamic lengths → PARAMs (from push constants), [H, T] buffers arranged by rows with DMA row by row (a PARAM accumulating the row offset); one single-stage VE per arith / math operation; values depending only on scalars / rows computed once; reductions (including the abs-max halving tree); int8 / int32 outputs; reads (a single element, a whole row by position, pair swap with negation → SWAPNEG), with rows read usable directly as results; i64 scalar arithmetic; prefix masks → VALID (LDPARAM reads the position); generics over dynamic T lowered row by row (LEN as a dynamic field).
- `sahw-fuse-ve`: merges single-stage VEs by C3's rules (A, B, FUNC, VALID, REDUCE, output conversion in turn), merging only when the dynamic LEN is the same.
- Computation is still represented as `linalg.generic` over on-chip memrefs; `sahl` has only the two operations load / store so far.
- Driver: each dispatch first tries the new pipeline and falls back to the C3/C4 generator on failure (`--iree-sa-new-codegen=on|only|off`, `--iree-sa-codegen-report`).
- Results (`tests/test_c51.py`): 34 of stories15M's 58 executables go through the new pipeline (all elementwise and reduction kinds), all 58 pass `dispatch_check`, end-to-end sim bit-exact; descriptor / VE counts the same as C3/C4, with dispatches 5 and 30 each one VE shorter (REDUCE merged into the last step of the halving). The 23 of C5.1a have per-dispatch cycles on the board no higher than C4, overall 2,590,630 cycles / token.
- The remaining 24: 5 linear layers (C5.2), 12 attention batch_matmuls and 7 KV-writing scatters (C5.3).

**C5.2 (2026-09-28)**: linear layers go through the new pipeline.
- "int8 contraction + elementwise tail" is recognized at the `sahl` level: x (i8, or int8 values in i32) × packed W (i8 [N/D, K, D]) → int32 accumulator → an elementwise tail over [N/D, D] → write-back. All buffers of the contraction and tail may only be used by this structure.
- Lowering: tiling (block size by the SPAD_B bank and the ACC staging area, capacities from the target configuration); the A strip by DIV-D replication of x; after block i's EX, prefetch block i+1's weights and the tail's per-block inputs (the other bank); the tail lowered per block with C5.1's elementwise lowering (any elementwise operations, no longer limited to the template's few forms), temporary buffers in that block's bank; LOOP_END over a pair of blocks when there are many blocks; block 0's weights in the prefix.
- Results: all 5 kinds of linear layers pass `dispatch_check`; the kinds, order and number of commands are the same as the template (the classifier's loop structure identical), differing only in where tail intermediates live, residuals prefetched with the next block, and register numbers.
- Debugging tool: `compiler/runtime/tools/sadis.py` (sa-desc disassembler).
- On the board: per-dispatch cycles the same as the template or slightly fewer (33: −1,071; 22: −582, residual prefetch), overall 2,588,992 cycles / token. **C5.2 accepted.**

**C5.3 (2026-09-28)**: attention and KV writes go through the new pipeline, so all 58 executables of stories15M are generated by the C5 pipeline (`--iree-sa-new-codegen=only` compiles).
- scatter (KV writes): LDPARAM (position × row bytes) + an ST with a dynamic DDR address; the "copy the cache to itself" write-back is deleted in `sa-to-sahl`; only multi-level elementwise loops with identity maps are flattened to one dimension.
- Attention: `sa-to-sahl` keeps the batch_matmul and the expanding generic over the cache slice; `sahl-to-sahw` treats "expand + batch_matmul + tail + write-back" as one structure and lowers it per head the way C3 did (scores: K rows → TRANSPOSE → Kᵀ blocks, q replicated into an A strip, EX, `×s_q[h]×a_k`; P·V: p replicated into an A strip, V loaded with INTERLEAVE, EX, `×a_v`; T dynamic). This effectively ports the attention template to the new pipeline, becoming the `attention` micro-kernel in C5.4; the generic lowering of batch_matmul (with the Kᵀ layout inserted by layout propagation) is postponed.
- On the board: all 58 executables from the C5 pipeline, 2,588,991 cycles / token, bit-exact. **C5.3 accepted.**

**C5.4 results (2026-09-28)**: the C3/C4 generator retired, templates turned into micro-kernels, and stories15M compiles without micro-kernels too.
- Retired: `Codegen.cpp` (C3/C4's direct-descriptor generator), `Templates` (the DescList port of the reference implementation), C5.0's raising layer, `--iree-sa-codegen` / `--iree-sa-new-codegen`. Code generation has a single path `iree-sa-codegen`; dispatches that cannot be compiled raise errors (`--iree-sa-allow-unsupported`: produce an export that errors when run).
- Micro-kernels (`--iree-sa-ukernels=all|none|<list>`): `linear` (C5.2's tuned linear-layer lowering), `attention` (C5.3's attention lowering).
- Generic contraction lowering (without micro-kernels): each operand is traced back to DDR along `sahl.load` and expanding generics (including permutations), giving "element offset = Σ loop dimension × coefficient", with loops classified as batch / N / K; matrices become B blocks by layout (already packed: direct LD; K contiguous: row-by-row LD + TRANSPOSE; N contiguous: INTERLEAVE LD), x replicated into an A strip; per batch, per output block: B block, EX, tail (elementwise lowering, with identity-mapped inputs loaded per block), write-back; dynamic N / K through PARAMs; serial schedule (no prefetch, no LOOP_END). On-chip buffers used by a block are reclaimed at the block's end, so the tail's broadcast words must be produced before the block loop (a bug found in the first board run at T ≥ 80).
- Acceptance: both `--iree-sa-ukernels=all` and `none` compile 58 executables, `tests/test_c5.py` passes `dispatch_check` throughout at T = 16, 80, 256, sim bit-exact for 76 steps; on the board both are bit-exact with text identical to L5.
- The value of micro-kernels (board, cycles / token): with micro-kernels 2,588,975; generic lowering only 4,395,723 (10.3 tok/s). The difference is almost all in the linear layers (no prefetch, no loops): classifier 1.26M → 2.43M, W13 0.41M → 0.70M, Wqkv 0.26M → 0.40M, W2 0.23M → 0.35M; attention differs very little.
- Tests: `tests/test_c5.py` (lit, both configurations, T sweep, which dispatches the micro-kernels change); `test_c2.py` now checks that the kinds and numbers of commands match the reference schedule (no longer byte by byte); `test_c50.py`, `test_c51.py` deleted.

**C5.5 results (2026-09-28)**: two models with no dedicated code written for them pass; SmolLM2-135M passes on the board.
- Frontend: `frontend/qhf.py` (HuggingFace configuration, safetensors reading without third-party packages (BF16 → f32), byte-level BPE tokenization, an fp32 reference model, KV calibration, the quantized model `QModel`; rotate-half RoPE turned into a row permutation to interleaved pairs; GQA written as batch = KV heads, rows = the Q heads of a group; Qwen3's QK-norm), `frontend/export_hf.py` (export, `--layers` to truncate the layer count).
- The acceptance reference: the device's EXP / RECIP / RSQRT are hardware approximations, and with torch's exact functions as the reference the first step's correlation is only 0.9875. `qhf.device_sfu()` makes the eager `QModel` use the device's approximations (`SfuExact` in `llm/ref_model.py`): the first step then differs from the sim only in summation order (relative error 2e-5); afterwards the summation order occasionally flips an int8 rounding, amplified layer by layer, so acceptance is argmax identical and correlation > 0.99 (`tests/test_c55.py`). The board is compared with the sim bit for bit.
- Runtime: the device window is configurable (sim `--mb`, board `board.txt`); `sa_device_queue_read` reads files straight into device buffers (the old streaming path first allocated a staging buffer as large as the transfer, so the largest parameter needed two copies: SmolLM2's peak 184.5 → 160.5 MB); `sa-llm-run` prints the device memory peak, and arena out-of-memory errors include the current usage.
- Compiler: the row-by-row lowering for dynamic T reclaims each row's temporary buffers after the row (Qwen3's softmax has 16 rows and used to fill ACC).
- SmolLM2-135M (30 layers, GQA 9/3, dim 576, vocabulary 49152): 230 dispatches, all bit-exact with the oracle at T = 16 / 80 / 256; sim end to end 16 steps with argmax 16/16, minimum correlation 0.9922, generating "Once upon a time, there was a little girl named Lily. She lived in a big house with her family, and she loved to play with her toys. …"; window 168 MB (peak 160.6 MB); sim 9 s / token. Board bundle: `scripts/deploy_c55.sh` (`tests/board_llm.py`: shared by C3 and C5.5, comparing bit for bit against the sim's results).
- **On the board**: 38 steps of logits bit-exact with the sim, identical generated tokens, 1.90 tok/s (526 ms / token). The PYNQ-Z1's CMA defaults to 128 MB, too small for the window: add `uEnv.txt` to the boot partition (`bootargs=<original command line> cma=320M`, imported by `boot.scr`); with 256M the CMA fragments after loading the overlay and no contiguous 168 MB block can be found (the kernel has no compaction). When window allocation fails the launcher drops the page cache and retries; stop Jupyter before running.
- **Repackaging and interactive generation (2026-09-28)**: repackaged with the latest compiler (after K tiling and piece mode) and re-verified on the board: stories15M 76/76 steps bit-exact with DeviceModel, 17.75 tok/s; SmolLM2 38/38 steps bit-exact with the sim, 2.23 tok/s (the bundle now has `--pad=8`, so the attention length grows with the position; computing over 256 gave 1.90). `tests/board_generate.py`: the overlay starts once, then prompts are read line by line, tokenized on the ARM (`frontend/hf_tokenizer.py`, standard library only), and generated with `sa-llm-run --stop_token`, printing tokens as they come; both bundles carry `model.json` (tokenizer, BOS, stop tokens, context length). The module and parameters are reloaded for every prompt: about 11 s before the first token for SmolLM2, 0.4–1.7 s for stories15M. The window must be allocated before loading the tokenizer: SmolLM2's tokenizer takes tens of MB in Python, and pages the CMA lent to the kernel cannot be migrated out, so no contiguous 168 MB block can be found.
- Qwen3-0.6B truncated to 2 layers (QK-norm, head_dim 128, q_dim 2048 ≠ dim 1024, GQA 16/8, θ = 1e6, vocabulary 151936): all 42 dispatches bit-exact; sim 8 steps argmax 8/8, correlation 1.00000; peak 330.5 MB.
- Problem found (left for C6): the shared embedding is stored twice (the classifier's packed weights in the parameter file, and the original table used by gather as a constant in the vmfb): 28 MB extra for SmolLM2, 155 MB for Qwen3; gather could read the packed layout directly, or the original table could be exported as a parameter too.

### 8.8 Extension points and micro-kernels

**Micro-kernels** (ukernels):
- A micro-kernel = a match condition + a C++ function that rewrites the matched linalg operator (with its tail) into a piece of `sahl` operations (or, when needed, directly a `sahw` fragment). It does **not** output descriptors directly; it then goes through memory planning (or declares the on-chip regions it occupies), scheduling, `sahl-to-sahw`, register allocation and serialization as usual, so it is equally bound by the verifiers, `dispatch_check`, and the driver's prefix prefetch and FENCE rules.
- Selection happens in `sa-to-sahl`: for each operator the micro-kernel table is checked first, used if it matches, otherwise the generic lowering is taken; when both are possible the cost model or a switch decides. Switch `--iree-sa-ukernels=all|none|<name,…>`.
- Initial micro-kernels: `linear` (`emitLinear`: tiling, double buffering, per-block tails; the byte-comparison tests from C2 on are kept), `attention` (scores / P·V). Operators later worth hand-tuning (e.g. C4's continuation, the interleaved W13 + SiLU loop) only need one more micro-kernel.
- Each micro-kernel needs comparison data: bit-identical to the generic path, with the cycle difference recorded.

**Extension points**: code generation is an MLIR pass pipeline, every pass registered in the plugin and runnable and testable alone with `iree-opt`. New optimization passes go into one of the following places, switched by compile options for easy A/B measurement:

| Extension point | IR it acts on | Suitable optimizations | Existing examples |
|---|---|---|---|
| Preprocessing (global, before dispatch formation) | The whole model's linalg | Controlling fusion, forming dispatch regions ourselves, constant packing | `iree-sa-pack-linear-weights`, `iree-sa-clone-cheap-producers` |
| linalg level (before and after `sa-tile`) | A single dispatch's linalg / scf | Tiling strategy, operator rewriting, micro-kernel selection | — |
| `sahl` level | Tile-level operations | Memory planning, pipelining, coarse scheduling, prefetch, tile-level rewriting | — |
| `sahw` level | Commands | Fine-grained scheduling, peephole optimization, VE stage merging | abs-max halving tree (C4 step 6, moved to `sahl-to-sahw` in C5) |
| Driver (runtime) | A command buffer's whole list | Cross-dispatch scheduling | C4 step 5's list scheduling |

- Experiments need no pipeline changes: `iree-opt --pass-pipeline=...` can be tried directly on IR exported at any stage.
- Optional: hook up IREE transform dialect scripts to specify tiling and fusion strategies per operator without rebuilding the compiler.

### 8.9 Extensibility: larger models and boards

**Goal**: Qwen3 and Llama3 are not required to run on the PYNQ-Z1, but when moving to a board with more resources the compiler should compile these models without structural changes. So the compiler must not hard-code any number of the PYNQ-Z1 or stories15M.

**Limits of the PYNQ-Z1** (hardware, not the compiler's problem): 512 MB of DDR shared with Linux; weight bandwidth about 390 MB/s.

| Model | Parameters | int8 weights | PYNQ-Z1 |
|---|---|---|---|
| stories15M / 110M | 15M / 110M | 16 / 115 MB | Yes (110M needs a larger CMA window, about 3 tok/s) |
| SmolLM2-135M (Llama structure) | 135M | About 135 MB | Yes (about 2.5 tok/s) |
| Qwen3-0.6B | About 0.6B | About 600 MB | No: more than all the DDR |
| Llama-3.2-1B | 1.24B | About 1.2 GB | No |

Descriptor DDR addresses and BASE registers are 32-bit, so even on another board all of a model's data must be within 4 GB of physical addresses; beyond that needs hardware changes.

**1. Parameterized target configuration**: all hardware parameters come from `#hal.executable.target<"sa", …, {config}>` (which already has `d`), with no constants in the compiler:
- D, SPAD / ACC sizes and bank counts, the number of dynamic fields per descriptor, the numbers of BASEs / PARAMs, CAPS (FPVE, later int4 unpacking, etc.);
- the cost model's parameters (bandwidth, engine latencies, multi-cycle costs);
- the driver reads D and CAPS from the device and checks them against the executable (the D check exists).
Verification: compile the same model with different configurations (D = 8 / 16, larger SPAD) and verify per dispatch on a funcsim configured with the same parameters (funcsim needs to support them).

**2. Frontend**:
- Import from HuggingFace models (transformers → torch → iree-turbine), no longer relying on the hand-written `qllama.py`; `qllama.py` stays as the reference for the stories series.
- A generic quantization rewrite: replace `nn.Linear` with quantized linear layers (W8A8, per-output-channel weight scales); optionally outlier handling such as SmoothQuant (only folding scales into neighboring operators, no hardware change). The hardware's EX is int8 × int8 only; lower-bit weights (int4) need hardware support.
- Structural features: GQA, QK-norm (Qwen3), RoPE variants (theta, Llama3's frequency scaling: precomputed into tables), shared embedding and output layer, SwiGLU / GELU, LayerNorm.
- Long contexts: `max-dynamic` and the KV cache size come from the model configuration, not a fixed 256 (done 2026-10-13: `hf_generic` / `test_hf_generic.py --max-len`, the export's `compile_flags.txt`; generality plan §9.3).

**3. Code generation** (covered by C5's generic parts):
- K-dimension tiling with cross-block accumulation (Llama3-1B's W2: K = 8192 is exactly one SPAD_B bank);
- attention tiled by T (long sequences' K / V do not fit on chip): ~~online softmax written explicitly into the IR by preprocessing (§3.2)~~ revised in §8.22: with int8 probabilities (qsum) only integer accumulations and the order-independent max cross T, so the blocks are a dynamic-count `LOOP_END` over integer-accumulating contractions, bit-identical to the untiled form;
- GQA: KV head index = Q head / group size, expressed with the DIV index mode or a loop;
- classifiers with large vocabularies (128K–152K rows): just more blocks;
- when a command buffer has many dispatches (about 900 for 28 layers), the driver already splits it into several lists.

**4. Runtime**: the device window size comes from the startup configuration (fixed at 64 MB today); the arena allocator supports hundreds of MB to several GB; parameter files are loaded into the window on demand.

**5. Verification methods adapt to scale**:
- Large models have no hand-written DeviceModel: per dispatch still bit-exact with the oracle (the oracle executes the IR, independent of the model); end to end, errors are compared with torch fp32, then generation quality (top-1 agreement, perplexity).
- The Python funcsim takes about 1.4 s / token on stories15M, so a whole 0.6B model on the sim is impractical: end-to-end checks with truncated layer counts (like `QLlama(n_layers)`), while per-dispatch checks still cover all dispatches; port funcsim to C if needed (§13 item 2).

**6. Prefill and batching** (optional): batched prefill is a matrix multiply with M > 1 and can use the whole array; the weight-packing pass only recognizes vecmat today and must be generalized to matmul.

**Schedule of the items** (principle: whatever is costlier to change later is built into C5's passes from the start):

| # | Item | When | Reason |
|---|---|---|---|
| 1 | Parameterized target configuration | Throughout C5 from C5.0; funcsim parameterization and cross-validation **made up after C5.5** (see below) | New passes contain no constants from their first line, at almost no extra cost; changing it afterwards means going through every pass |
| 3a | K-dimension tiling, cross-block accumulation | Originally in C5.2, not actually done; **made up after C5.5** (see below) | Part of generic contraction tiling; stories15M does not need it, covered by synthetic-model tests |
| 3b | GQA index mapping | Implemented in C5.3, first used in C5.5 | Designed into the attention lowering per group; SmolLM2 needs it |
| 2 | Frontend: HuggingFace import, generic quantization, structures such as QK-norm | C5.5 (can be done first, interleaved with C5.1–C5.3) | Independent of code generation; needed by SmolLM2 and the small Qwen3 configurations |
| 4 | Runtime: configurable device window, allocator up to GBs | C5.5 | SmolLM2-135M's weights are about 135 MB and do not fit today's 64 MB window; small change |
| 5 | Verification: errors versus torch, truncated layer counts | From C5.5; C funcsim in C6 if needed | New models have no hand-written DeviceModel; the Python sim is only too slow on large models |
| 3c | Attention tiled by T (revised: integer tiling, no online softmax, §8.22) | C6 | Needed: the decode attention holds T × head_dim in one SPAD_B bank (T ≤ 256 for Gemma-3, 512 for Qwen3); no change to the numeric conventions |
| 3d | Large vocabularies, multiple lists | Already supported, verified in C6 | No new code needed |
| 6 | Batched prefill | Moved up to after C6.1: **done** (C6.P, §8.13) | Only affects speed, not whether things compile |

**Made up (2026-09-28)**: checking after C5.5 found that item 1's cross-validation and item 3a had not actually been done; they were added:
- **Target-configuration cross-validation**: funcsim's SPAD / ACC sizes are configurable (`SaFuncSim(d, …, spad_bytes, acc_bytes)`; on-chip addresses are 16-bit word addresses, at most 2^16 words per memory); the compiler gets `--iree-sa-spad-kb` / `--iree-sa-acc-kb` and errors on an invalid target configuration (`TargetConfig::invalid`); `dispatch_check.py` reads D and memory sizes from the dispatch's target, and the sim service gets `--spad-kb` / `--acc-kb`. `tests/test_c5.py --configs`: stories15M under three configurations, D = 16, SPAD 256 KB / ACC 512 KB, and D = 16 + SPAD 512 KB / ACC 1 MB, compiles 58 executables with and without micro-kernels, all bit-exact with the oracle at T = 16 / 256. (On the first run dispatch_check's regex did not match the target, so everything was checked with the default configuration and reported a wave of false errors; all pass after the fix.)
- **K tiling**: the generic contraction lowering executes by K blocks when K exceeds capacity (within a block: that block's B block, the matching segment of x replicated into an A strip, EX; the first block writes the accumulator, later ones accumulate, the tail after all blocks). The block length has three limits: one SPAD bank; the row length of one DMA (16 bits: block length × D ≤ 65535, 8184 at D = 8); the range check of the DIV index mode used for A-strip replication (the hardware has no divider and checks G words, so x's position + K/D + block length must not exceed the memory x lives in). The linear micro-kernel does not match when K exceeds its capacity, leaving it to the generic lowering. Llama-3.2-1B's W2 (K = 8192) would have exceeded the DMA row length, and now runs in 2 blocks.
- Two capacity problems found and fixed at the same time: contiguous DMAs larger than 64 KB are split into "rows × row length" plus a tail (`contiguousDma`); elementwise dispatches with more elements than ACC execute in pieces (piece mode: each piece reruns the whole dispatch body with DDR offsets and local buffers per piece, released after the piece; max reductions over the whole vector (including abs-max) are merged piece by piece, still bit-exact because max is order independent; merging sums piece by piece would change the summation order and is not supported for now).
- Verification: `frontend/synth_hf.py` generates a HuggingFace directory with random weights (dim 256, hidden 16384, 1 layer, SmolLM2's tokenizer): all 26 dispatches bit-exact at T = 16 / 80 / 256, sim end to end with 0 error at the first step, argmax 8/8, correlation 1.0. Regression: C3 (stories15M bit-exact with DeviceModel), C5 (two configurations + three hardware configurations), SmolLM2 (230 dispatches) all pass.

**Stages**:
- **C5**: items 1, 3a, 3b (C5.0–C5.3 of §8.7).
- **C5.5** (§8.7): items 2, 4, 5; SmolLM2-135M (board) and small configurations of the Qwen3 structure (sim).
- **C6 larger models**: item 3c, item 5's C funcsim (if needed); Qwen3-0.6B and Llama-3.2-1B compiled on the sim and verified per dispatch (end to end with truncated layer counts); run on the board after moving to a new board. Item 6 optionally after that. Step-by-step plan in §8.12.

### 8.10 After C5

- **C4 continued**: the fusions on hold, done on C5's scheduler and cost model. W13 + SiLU formed into one dispatch region by preprocessing and interleaved inside the template; the quantization chain merged into one dispatch; `enable-aggressive-reshape-movement`. The goal remains within ≤ 10% of the hand-written path.
- Optional extensions (§10): batched decode, prefill, larger models.
- **C7 RISC-V backend** (optional, any time after C5, not affecting the C5 / C6 main line):
  - the same accelerator commands can be issued in two ways: the ARM builds a descriptor list executed by the fetch unit, or PicoRV32 issues them one by one with PCPI custom instructions (custom-0, funct7 = 1 matrix / DMA, funct7 = 2 vector). The command fields are basically the same, so `sahw` can feed a second backend: `sahw` + scf / arith → LLVM dialect (commands as `.insn` inline assembly) → LLVM IR → riscv32 object files.
  - The value is flexibility: full control flow and scalar computation (data-dependent loops and branches, argmax / top-k sampling on the device), without the "at most 2 dynamic fields per command" limit; it also prepares for hardware with a stronger RISC-V core.
  - General computation the accelerator cannot do, which has to run on the RISC-V, does not take this path but reuses IREE's llvm-cpu backend (linalg → vector → LLVM, supporting riscv32 / riscv64).
  - Limits on the PYNQ-Z1: PicoRV32 is rv32imc with only 8 KB of BRAM program memory, so only small kernels fit (whether it can fetch from DDR needs checking); issuing PCPI commands one by one is slower than descriptor DMA (M5: 1.2–4.2× for small operators), so on this board it is not a speedup.
  - Prerequisite: today's iree-compile does not include llvm-cpu, and LLVM's RISC-V target may not be compiled in; needs checking, and if necessary rebuilding the compiler (about 2 hours).
  - Acceptance: one dispatch executed by PicoRV32 code issuing PCPI instructions, bit-identical to the descriptor path; plus one riscv32 dispatch compiled with IREE llvm-cpu as a demonstration.

### 8.11 Risks and mitigations

| Risk | Mitigation |
|---|---|
| IREE's bufferization does not handle custom memory spaces and layouts smoothly | Fall back to our own bufferization: the dispatches have a simple structure (load → on chip → store), and C3 already has the same analysis |
| The generic lowering does not reach the templates' speed, especially linear-layer block sizes and pipelining | C5.2 is accepted by per-dispatch cycles against the templates; the cost model chooses block sizes; the templates' schedules written into FileCheck tests as references |
| Descriptor constraints (at most 2 dynamic fields per command, index modes by whole words, partial overlap of VE source and destination undefined) | Verifier checks plus dedicated legalization passes; violations raise errors instead of producing wrong lists |
| Bit-exactness broken (folding changing rounding, reduction order changed) | Folding rules carried over from C3, merging only operations that do not change rounding; the order of sum reductions fixed to the hardware order (only max can be reordered, as in the halving tree); `dispatch_check` run at every step |
| Large effort (MLIR C++) | Vertical slices, both paths coexisting until C5.4 so the model always runs end to end; the plugin needs only incremental builds (a few minutes) |
| Micro-kernels hiding defects of the generic path | Acceptance also requires `--iree-sa-ukernels=none`; each micro-kernel needs cycle comparison data |


---

### 8.12 C6 detailed plan: larger models

**Goal**: after moving to a board with more resources, the compiler compiles models of the Qwen3-0.6B and Llama-3.2-1B class without structural changes. The PYNQ-Z1's 512 MB of DDR cannot hold them, so on this board acceptance is on the sim only: all dispatches bit-exact with the oracle, end to end (with truncated layer counts if necessary) compared with the eager model (device SFU) by argmax and correlation, and generation quality measured separately against an fp32 reference.

**State as checked after C5.5**:
- Available: HuggingFace import (Llama, Qwen3 structures), GQA, QK-norm, shared embedding, configurable device window, end-to-end tests against torch error, truncated layer counts, synthetic models (`frontend/synth_hf.py`).
- The two items the plan listed as done but that were not have been made up (§8.9 "Made up"): target-configuration cross-validation, K tiling; DMAs above 64 KB and elementwise dispatches exceeding ACC (piece mode) were fixed along the way.
- Known capacity limits: attention's K rows must fit in one SPAD bank (D = 8, 128 KB SPAD: T ≤ 512 for head_dim 128, T ≤ 1024 for 64); elementwise dispatches in piece mode do not support sums over the whole vector (they would change the summation order).

| Step | Content | Acceptance | Status |
|---|---|---|---|
| **C6.0 cleanup** | The shared embedding is stored twice (the classifier's packed weights in the parameter file, and the original table used by gather as a constant in the vmfb): gather reads the packed layout directly, or the original table is exported as a parameter too | SmolLM2 still bit-exact on the board, with a smaller window (Qwen3 saves 155 MB, SmolLM2 28 MB) | **Done** (below) |
| **C6.1 Qwen3-0.6B all 28 layers** | Export, compile, per-dispatch checks, a few end-to-end sim steps (estimated about 40 s per token) | All dispatches bit-exact; argmax identical to the eager model (device SFU), correlation > 0.98; fluent generated text | **Done** (below) |
| **C6.2 quantization quality** | Measure top-1 agreement and perplexity against an fp32 reference; if insufficient, add outlier handling such as SmoothQuant (scales folded into neighboring operators, hardware unchanged) | top-1 and perplexity versus fp32 within an acceptable range | **Done** in the generality plan (WP3/WP4: external W8A8 checkpoints, K per-channel KV, qsum; board PPL within 1–16% of fp32, §13.2) |
| **C6.3 K tiling** | The generic contraction accumulates over K blocks; the linear micro-kernel hands off to the generic lowering when it does not fit | Synthetic model with K = 16384 bit-exact per dispatch and end to end | **Done** (§8.9 made up) |
| **C6.4 Llama-3.2-1B** | Frontend: Llama3's RoPE frequency scaling (precomputed into tables; raises an error today), 128K vocabulary, GQA 32/8; weights need Meta's authorization (or a public mirror) | As C6.1 | **Done** in the generality plan (WP5–WP7: RoPE scaling, 128K vocabulary, GEMV K split; on the board 4.04 tok/s, PPL 23.20 vs fp32 22.77) |
| **C6.5 attention tiled by T** | ~~Online softmax written explicitly into the IR by preprocessing (§3.2)~~ **Revised 2026-10-13 (§8.22)**: integer tiling of the decode attention contractions over T with a dynamic-count `LOOP_END`, bit-identical to the untiled form, no online softmax; KV cache length and `max-dynamic` set by the model configuration (`--max-len`, `compile_flags.txt`, done) | Consistent per dispatch at T = 2048 | **Done on the sim** (§8.22, 2026-10-14: C6.5 + C6.5b, Qwen3 2048 / Gemma 512 all on the accelerator, per-dispatch checks and a 1104-token end to end pass; **on the board** 2026-10-14: whole Qwen3 at 2048, Gemma at 512, prefill vs decode bit-exact) |
| **C6.6 target-configuration cross-validation** | funcsim parameterized by the target configuration; the same model compiled and verified with different D, SPAD / ACC | `test_c5.py --configs` all bit-exact | **Done** (§8.9 made up) |
| C6.7 (optional) | Batched prefill: matrix multiplies with M > 1; weight packing generalized from vector × matrix to matrix multiply | Consistent with token-by-token decode | **Done**, as C6.P (§8.13) |

**C6.0 results (2026-09-28)**: gather reads the packed layout (the reverse, making the classifier read the original layout, would need a whole-table TRANSPOSE every token, about +40% cycles for stories15M).
- Preprocessing (`target/PackLinearWeights.cpp`): each constant weight is packed only once (placed after its definition); its gathers (`tensor.extract W[r, c]`) are changed to read the packed copy `Wp[r / D, c, r % D]`, so the original table has no readers and no longer goes into the vmfb. Condition: K × D ≤ 65535 (LDPARAM's 16-bit multiplier, one DMA row).
- Lowering (`packedRowGather` in `sahl-to-sahw`): the command processor cannot divide, so r / D and r % D are computed by VE (exact in fp32: `round(r / D − (D − 1) / 2D)`), written out through this dispatch's result buffer in DDR (later overwritten by the result), FENCE, read into PARAMs with LDPARAM; the tile (K words) is loaded by PARAM; the one-hot word is made with VALID (the difference of two word prefixes `prefix(D + l + 1) − prefix(D + l)`, because VALID = 0 means all valid); the tile is multiplied by the one-hot and each word REDUCEd (only one non-zero lane, so the sum is exact), and `packBcast` packs the K scalars back into K / D words. stories15M's embedding dispatch is 56 descriptors.
- Results: vmfb stories15M 10.3 → 1.1 MB, SmolLM2 30.6 → 2.2 MB, Qwen3 157.3 → 1.7 MB; device memory peak SmolLM2 160.6 → 133.6 MB (window 168 → 144 MB), Qwen3 (2 layers) 330.5 → 182.1 MB. stories15M's 58 dispatches and 12/12 sim steps bit-exact with DeviceModel; SmolLM2's 230 and Qwen3's 42 dispatches bit-exact at T = 16 / 80 / 256, end to end passing; the configuration cross-check (two micro-kernel settings × 4 hardware configurations) all passes.
- On the board: SmolLM2 38/38 steps bit-exact with the sim, 2.22 tok/s, window 144 MB (still needing a retry or two after dropping the page cache: the CMA fragments after loading the overlay); stories15M 76/76 steps bit-exact with DeviceModel, 17.65 tok/s (previously 17.75: the new embedding gather adds about 56 descriptors per token).
- Long-running tests: `compiler/scripts/run_tests.sh [-j jobs] [-t threads per test] test…` (one log per test; processes started from a Claude Code session end with the session, so run it in your own terminal).

**C6.1 results (2026-09-28)**: Qwen3-0.6B all 28 layers (dim 1024, hidden 3072, 16 / 8 heads, head_dim 128, QK-norm, vocabulary 151936), with no compiler changes:
- Export 45.6 s (`qllama.irpa` 613 MB); on the host IREE llvm-cpu vs the eager model argmax 8/8, minimum correlation 0.989 (28 layers flip more int8 roundings, so the end-to-end correlation threshold goes from 0.99 to 0.98, while the first step must still be within 1e-4); QModel's top-1 versus fp32 is only 5/8 (quantization quality, C6.2).
- sa compile 70.6 s: 276 dispatches, 610.7 MB of parameters; all 276 bit-exact with the oracle at T = 16 / 80 / 256.
- sim end to end: relative error 2.1e-6 at the first step, argmax 8/8 identical to the eager model (device SFU), minimum correlation 0.9937, generating "Once upon a time, there was a man"; device memory peak 587.4 MB (window 768 MB), sim 24 s / token.
- Test memory: the oracle used to evaluate over the whole iteration space at once, so checking the classifier (155M points) took over 5 GB, and the desktop's systemd-oomd killed it together with the terminal when the session's memory pressure exceeded 50% (several earlier "terminal exits" were this); the oracle now evaluates in pieces along the outermost parallel loop (reduction order unchanged), down to 2 GB; `run_tests.sh -m` gives each test a systemd scope with a memory limit.

**Order**: C6.0 → C6.1 → **C6.P (prefill, §8.13, brought forward: first compile a complete LLM)** → C6.2 → C6.4 → C6.5. C6.0–C6.2 only need Qwen3 and no new code generation, giving results for a whole large model fastest; C6.5 is the only new compiler feature left. The Python funcsim's time per token scales with model size (estimated about 40 s for Qwen3-0.6B, about 80 s for Llama-1B); running only a few end-to-end steps is acceptable, and a C funcsim is built only when it is really too slow.

### 8.13 C6.P prefill: compiling a complete LLM (prefill + decode)

**Goal**: one module with two functions, prefill and decode, sharing parameters and the KV cache; the prompt is prefilled in chunks (weights read once serve M tokens, using the whole D×D array), followed by token-by-token decode. Originally an optional item (§8.9 item 6, C6.7), brought forward before C6.2.

**Design**:
- **Fixed chunk length M** (an export parameter; originally defaulting to 16, defaulting to 8 after the board comparison, see P5): a prompt of length P is split into ⌈P / M⌉ chunks, the last padded with filler tokens; the KV written by filler rows is at positions ≥ P, which decode overwrites before reading and attention masks by position, so it is harmless. That way T remains the only dynamic dimension (not touching "at most 2 dynamic fields per descriptor").
- `prefill(tokens[M], start, valid[T])`: row i's causal mask is position ≤ start + i; M rows of KV written at once; only the last row's logits computed (the classifier is the largest layer). Both functions are exported with iree-turbine's `CompiledModule`, the KV cache a shared mutable global.
- **The main acceptance basis**: on the device, the logits of prefill's last position are bit-identical to the decode-only path (the linear layers compute each row the same way: exact int32 accumulation, elementwise dequantization; attention sums in the same order when T's padding is the same).

| Step | Content | Acceptance |
|---|---|---|
| **P0 frontend** | QModel gains `prefill` (per-row causal mask, M rows of KV, logits of the last row only); `prefill` and `decode` exported as one module | eager prefill + decode generates the same tokens as decode only; IREE llvm-cpu consistent with eager |
| **P1 linear layers** | The packing pass also recognizes matmul (packed weights shared with decode, C6.0's "pack only once"); the A strip from M rows of X through TRANSPOSE; micro-kernel and generic lowering | Bit-exact per dispatch; EX efficiency per chunk |
| **P2 attention and the rest** | Scores / P·V of M rows; softmax masks per row (row i's valid length start + i + 1, LDPARAM's addend); RoPE takes M rows; the KV cache written M rows at once | Bit-exact per dispatch, T sweep |
| **P3 runtime** | `sa-llm-run` prefills in chunks, then decodes; `board_generate.py` uses it | sim: tokens identical to decode only, last-position logits bit-exact |
| **P4 board** | stories15M, SmolLM2; M = 8 / 16 / 32 compared | Bit-exact; prompt tok/s (expected 5–10× faster than token by token). **Done** (below): M = 8 is the default; M = 16 is only 1.7% faster per token; M = 32 not done |

**P0 results (2026-09-28)**: `prefill(tokens[M], positions[M], valid[T])` (positions = start .. start + M − 1 given by the caller: the command processor cannot write registers back to DDR and VE has no iota, so computing i64 vectors on the device is awkward); the last chunk overlaps the previous one instead of being padded (starting from P − M; recomputed rows write back the same KV), so the last row is always the prompt's last token. eager prefill bit-exact with decode (logits, KV cache); on IREE llvm-cpu stories15M's prefill is bit-exact with decode only, SmolLM2's correlation 1.000000, with identical tokens generated afterwards. Export: `FxProgramsBuilder`, `main` (decode) and `prefill` sharing parameters and the KV globals.

**Tasks of P1 / P2** (stories15M's prefill on the existing compiler: all 58 decode executables compile, 47 prefill ones fail, in 19 kinds):
1. The packing pass recognizes `matmul(X, Wᵀ)`, pulling X's extsi into the matmul (X stays int8 in DDR); an i64 accumulator + trunci accumulate as i32 (identical modulo 2³²).
2. A multi-row linear micro-kernel: EX's A strip is "one block per D words of K, word i being the D elements of row i", exactly the layout DMA INTERLEAVE produces when loading D rows of X (int8), so one LD gives the strip of D rows; output written row by row with `crow`; the tail's s_w per column, s_x per row. At M = 16 two blocks of rows share one weight load.
3. i64 index vectors: gather / scatter / masks reading `positions[m]`, `tokens[m]` per row (one LDPARAM per row); `vector + constant` folded into LDPARAM's addend.
4. Generics over three loop levels (head × row × T), with VALID per row.
5. Multi-row attention contractions (batch_matmuls of M rows per head).
6. DDR views strided by rows (`h13[:, :hidden]` etc.: the DMA's row pitch).
7. Odds and ends: i32 fills, alignment.
Also: the dispatch source files of the two functions are numbered separately (`module_main$…_N`, `module_prefill$…_N`), so `dispatch_check` must match binaries by function name.

**P1–P3 results (2026-09-28)**:
- **Change of approach**: attention, gathers (embedding, scales, RoPE tables), scatters (KV rows) and the QK-norm / RoPE of HF models are done row by row in the frontend, each row exactly in the form of one decode step (scalar indices `positions[m:m+1]`, `tokens[m:m+1]`), reusing the verified decode lowerings; what is batched is the linear layers (where the gain is) and the row-wise elementwise operations and normalization. Items 3–5 above therefore need not be done; the number of dispatches row-by-row processing adds can be merged later.
- **Compiler**:
  - the packing pass recognizes `matmul(X, Wᵀ)` (item 1); a weight global is packed only once: a `W$packed` global set by an initializer, evaluated by IREE at compile time, used by prefill and decode and by every load of the same weight (previously the two functions each packed a copy, and SmolLM2's parameters became 272 MB); gathers of a packed global are redirected by global name to the packed copy (C6.0 holds in two-function modules too).
  - The multi-row linear micro-kernel `linearRows` (item 2): one INTERLEAVE LD loads the A strip of D rows of X, and the D×D array computes D rows at once; tail inputs per column (MOD nc) and per row (DIV nc); without a tail the accumulator is stored directly. Serial schedule, no prefetch yet.
  - Two-level loop merging can split at any point (rows [0, s), inner [s, n)); DDR views strided by rows (item 6, only for load / store); 4-byte-aligned fp32 scalars through LDPARAM; scalar gathers inside loops; the result of a packed gather is an exact integer value (sitofp passes straight through); a buffer in broadcast layout reused by an elementwise result is replaced with a fresh compact buffer; **row tiling**: when a row-wise dispatch does not fit in ACC, it is grouped by rows (an even number of rows per group).
  - `sa-llm-run --prefill=M`; `dispatch_check` matches binaries by function name; `test_c6p.py` (bit-for-bit comparison of the logits of both runs); `deploy_c6p.sh stories|smollm2`, with `board_llm.py [--decode-only]` and `board_generate.py` on the board using prefill.
- **Results** (sim, M = 8):
  - stories15M: all 181 executables (decode + prefill) bit-exact with the oracle; a 16-token prompt prefilled in 2 chunks, with the logits of the prompt's last position and the next 4 steps bit-identical to decode only, identical tokens; parameters 16.1 MB (one copy).
  - SmolLM2-135M: all 777 executables bit-exact at T = 16 / 80 / 256; a 14-token prompt, prefill + decode bit-exact with decode only (5/5 rows), identical tokens; parameters 137.4 MB (one copy).
  - On host llvm-cpu, prefill versus decode only is compared by correlation only: llvm-cpu generates different code for prefill's shapes (vectorized sums), and int8 rounding flips can change close argmaxes; on the sa device bit-exactness is required.
- **P4 on the board (2026-09-28, M = 8)**: for both models prefill and decode only are bit-exact with the sim, with identical generated text.
  - stories15M: a 16-token prompt in 2 chunks, **258.6 ms (including loading)**; with decode only the prompt takes about 16 × 57 ms ≈ 915 ms (plus loading): **at least 3.5×**. Generation 17.4 tok/s (unchanged).
  - SmolLM2-135M: 14 tokens in 2 chunks, **2.43 s (including loading)**; decode only about 14 × 456 ms ≈ 6.4 s (plus loading): **at least 2.6×**. Generation 2.18 tok/s (unchanged).
  - The prompts are still short (2 chunks, the second overlapping) and loading is counted in prefill, so these are lower bounds. Still to do: prefetch and loops in `linearRows` (serial scheduling today), M = 16 / 32, merging the row-by-row attention and gather / scatter into batches.
- **P5 optimizations (2026-09-28)**:
  - `linearRows` prefetch: after the first EX, the next block's weights are loaded into the other SPAD_B bank; loops that only load weights prefetch per column group.
  - Skipping the classifier: a third exported function `prefill_kv` (writes KV only, no classifier), used for every chunk but the last; saves about 1.1M cycles per chunk for stories.
  - Fix for M = 16: flat piece mode only for contiguous views (column slices tiled by flat offset read the wrong data: SmolLM2's SiLU input at M = 16); gather's domain upper bound 65536; test prompts at least 2M long; `deploy_c6p.sh <model> [M]` with `--check`.
  - **On the board, M = 8** (bit-exact with the sim; cycles per chunk excluding loading):
    - stories15M: per chunk 6.43M → **4.82M (−25%)**, the 16-token prompt 258.6 → **211.6 ms**;
    - SmolLM2: per chunk 53.9M → **42.1M (−22%)**, 14 tokens 2426 → **1871.5 ms**;
    - linear layers 30–43% faster (SmolLM2 W13 14.2M → 8.14M, W2 6.83M → 3.85M).
    - About 12 ms per token (stories; decode 57 ms: 4.7×), about 105 ms (SmolLM2; decode 456 ms: 4.3×).
  - **M = 8 versus M = 16 (stories15M, a 64-token prompt)**:
    - both bit-exact with the sim;
    - 981.7 ms (8 chunks) versus 964.6 ms (4 chunks), only 1.7% apart per token (4.82M and 9.80M cycles per chunk);
    - the linear layers are already close to the compute bound at M = 8: W13 takes 83k cycles per layer per 8 rows (79k at M = 16), against an ideal of 55k for the D×D array (about 66% utilization); weight loading is already hidden by prefetch, and more rows only amortize the small remaining overhead.
    - **Conclusion**: default M = 8; the ACC problem of M = 32 is not pursued.
  - **The remaining bottleneck** is VE: the SiLU chain (`reduction_Mx768` + `elementwise_Mx768`) takes 30% for stories and 34% for SmolLM2, about 30k cycles per layer per token, the same as decode (elementwise work, which batching does not save).
    - Analysis: the generated VE instructions are already minimal (one exp, one recip, each covering all elements); the time goes to the hardware's multi-cycle SFU: about 19 cycles per element (exp 115k cycles / 6144 elements).
    - Fusing SiLU into W13's block loop (the approach of §8.8, purely in the compiler) can at most hide W13's EX time (83k per layer against SiLU's 238k), about **−10%**.

**P6: SFU multi-slot interleaving** (the microprogram controller in `rtl/sysarray/sa_vefp.v`; accepted on the board 2026-10-09, results at the end of this section)
- **Current state**:
  - the multi-cycle mode processes one group at a time; each micro-step issues one operation for one half group (FL = 4 lanes), then waits out the unit latency (M2 / A2 4 cycles, i2f / f2i 2 cycles, plus about 6 cycles for issue and write-back) before the next step.
  - EXP 11 steps, RECIP 8, RSQRT 10, with lookups reading one ROM lane by lane.
  - A group of 8 elements: EXP about 150 cycles, RECIP about 110; the units are idle about 80% of the time.
- **Approach**:
  - **Collect**: the pipeline feeds up to NB = 8 half groups (4 groups) into slots, instead of stopping at one group.
  - **Interleaved issue**: each micro-step issues over slots 0 … NB−1 back to back, one per cycle; by the time the last slot issues, slot 0's result is back, so the next step has no bubble. About 1 cycle per half group per step (about 6 today).
  - **Per-slot registers**: RF[8], TT / SS, inputs and results. Flip-flops would be 8× today's plus a 64:1 operand multiplexer, about 5k LUT, which does not fit; LUTRAM instead (64 × 32 bits per lane, two copies for two read ports), about 500–800 LUT.
  - **Everything else unchanged**:
    - lookups stay serial by lane at first (4 cycles per half group), changed to a dual-port BRAM reading 2 lanes per cycle if they become the bottleneck;
    - the output order and destination addresses of groups are unchanged;
    - SUM reductions are not interleaved: the accumulation order must be kept, which bit-exactness depends on.
  - **Parameter NB**: 1 = today's behavior; 2 / 4 / 8 trade LUTs against speed by synthesis results.
- **Numerics**: each element's operations and their order are completely unchanged, the functional simulator is unchanged, results remain bit-exact; only timing changes.
- **Expected (NB = 8, serial lookups)**:
  - EXP about 150 → 33 cycles/group (4.5×), RECIP about 110 → 25;
  - prefill: stories per chunk 4.82M → about 3.7M (**−23%**); SmolLM2 about **−26%**;
  - decode only about **−5%**: decode's time is mainly weight loading and linear layers, with SiLU only 6–8% of a step.
  - So it mainly shortens time to first token (more gain for longer prompts, and each turn of a multi-turn conversation benefits too), not generation speed.
- **Verification**:
  - `tb_fp32`, `tb_sa_vefp` (31 cases), `tb_sa_unit`, system-level `desc_run` co-simulation; NB = 1 / 2 / 8, FL = D/2 and D, D = 8 / 16, bit-exact;
  - new cycle-count tests for EXP / RECIP / RSQRT, and mutation tests of the slot index;
  - out-of-context synthesis of `sa_unit`: LUTs (83% today) and 13.3 ns timing;
  - the user runs the bitstream build; on the board `board_llm.py` bit-exact for both models, `board_profile.py` before/after comparison.
- **Risks**:
  - LUT margin: OOC synthesis first; if NB = 8 does not fit, NB = 4 still gives about 3×.
  - Timing: LUTRAM read plus the operand multiplexer must fit in 13.3 ns; otherwise register the operands one more stage, 1 more cycle of latency, which 8 slots still hide.
- **Status**: on hold (2026-09-28). The bottleneck of generation speed is decode; first profile SmolLM2's decode on the board, then decide the next step.
- **Implementation (2026-10-09)**:
  - Batched mode only for commands with FUNC = EXP / RECIP / RSQRT and no REDUCE; reductions still go through the original sequencer. sa_unit parameter `VEFP_NB` (default 8, 0 = the original behavior, built only for NB ≥ 6).
  - Each micro-step issues cycle by cycle over NB slots; lookups use one ROM per lane (a slot read in one cycle), so they did not stay serial.
  - Per-slot registers: one LUTRAM per lane (NB × 8 × 32 bits; read ports: register 0 and operands a, b; one write port: collected inputs, otherwise the stage-5 write-back — results of 2-latency units are delayed two more cycles, and x·2^-17 and END are computed at issue and go down the same pipeline). The a / b fields of the END microcode name the register fres reads; the result is written to register 1, and integer outputs take one more f2i step; the output FIFO takes one half group per cycle.
  - The first version with flip-flops + per-slot multiplexers: sa_unit +33k LUT (96%), unusable; the LUTRAM version **+2.65k LUT** (KV260 OOC, D = 16, 128-bit DMA: 82.2k vs 79.6k, 70%), flip-flops −0.8k; the worst path is still `ld` → BRAM enable (the same as NB = 0), fmax estimated 123 MHz (134 for NB = 0).
  - Simulation: `tb_sa_vefp` gains 7 batched cases (partial batch, VALID, RECIP after OP + IMM, I32 / I8 outputs, I8 input, MOD), and 38 cases are bit-exact at D = 8 / 16, NB = 0 / 8 / 16, FL = D; `make test / test16 / test128` all PASS.
  - Cycles (D = 16, 32 groups of EXP): 4502 → 1042 (4.3×); per group EXP about 150 → 32, RECIP about 116 → 26, RSQRT about 140 → 30. Qwen3 prefill estimated to save about 18M of 89.9M cycles per chunk (about −20%).
- **Board acceptance (2026-10-09, overlay `d16_100mhz_w128_p6`, 100 MHz, WNS +0.113 ns, LUT 84.5k / 72.1%, 9.0k more than w128)**: REGRESSION PASS, all tests bit-exact with the sim, all models' tokens identical to w128.
  - Qwen3-0.6B prefill per chunk (profile) 89.9M → **70.0M cycles (−22%)**: `elementwise_16x3072` (exp) 13.34M → 3.24M, `reduction_16x3072` (with RECIP) 14.72M → 7.02M; prefill 11.43 → **13.96 token/s**, decode 2.13 → 2.20.
  - Other prefills (token/s, w128 → p6): stories15M 227.6 → 284.3, SmolLM2 (qhf) 27.2 → 34.9, hfgen 11.2 → 13.3; decode improves slightly (SmolLM2 7.26 → 7.67, hfgen 6.89 → 7.45).
  - Qwen3 prefill's largest items now are the W13 matmul (19.5%), the q/k/v matmul (13.1%) and `reduction_16x3072` (10.0%, whose SUM / MAX reductions still go through the single-group sequencer).
- **Batched REDUCE (2026-10-09, `e6f0575`, `b10da84`)**:
  - Batched mode extended to commands with REDUCE (SUM / MAX) and any FUNC: collection and function micro-steps are batched as usual; then the state `Q_BRED` reduces in slot order (the first slot stores; MAX takes 1 cycle per slot with fmax; SUM goes through A2, the accumulation chains of the two half groups interleaved, stalling issue while a result of the same half group is in flight). Each lane's accumulation order is unchanged, so results are bit-exact. At the end of a row, after A2 drains, the original tree / write-back runs, then it returns to `Q_BRED` for the next slot; non-final groups are retired by `b_ret`.
  - FUNC = NONE / ABS (absmax, sum of squares) do no function micro-steps; the reduction takes the value directly from register 0 for the VALID mask and RELU (ABS is applied during collection).
  - The first build failed timing (−0.730 ns: register file → fres → relu → fmax → `ACCR`; FUNC is a run-time register, so synthesis does not know NONE / ABS never use fres); with the reduction value of NONE / ABS bypassing fres, WNS +0.123 ns.
  - A bug found in simulation: with 1 group per row, `retire_seq` was computed in the cycle after FIN (when `sq_last` was already cleared), so the in-flight count was decremented twice and underflowed (hang); in batched mode that term no longer counts.
  - Simulation: `tb_sa_vefp` gains 7 cases (1 / 3 / 5 / 6 / 10 / 24 groups per row and whole segments, ABS, VALID + RELU, reduction after a function, I8 input, MOD), and 45 cases are bit-exact at D = 8 / 16, NB = 0 / 8 / 16, FL = D; `make test / test16 / test128` PASS. Cycles (D = 16): absmax with 24 groups per row 2108 → 746, MAX after exp 6026 → 1414, SUM after rsqrt 4607 → 1284.
  - **Board acceptance (overlay `d16_100mhz_w128_p6_red`, 100 MHz, WNS +0.123 ns, LUT 88.25k / 75.4%, 3.7k more than p6)**: REGRESSION PASS, all tokens identical to p6.
    - Qwen3 per chunk (profile) 70.0M → **62.2M cycles (−11%)**: `reduction_16x3072` (absmax) 7.02M → 5.39M, the RMSNorm sums of squares 244 / 242 / 246 together 6.41M → 2.82M, decode's `main` 18 (softmax) 2.05M → 0.97M. prefill 13.96 → **15.45 token/s**, decode 2.20 → 2.22.
    - Others (token/s, p6 → red): prefill stories15M 284.3 → 311.8, SmolLM2 (qhf) 34.9 → 38.4, hfgen 13.3 → 14.2; decode SmolLM2 7.67 → 7.71, hfgen 7.45 → 7.62, stories 61.3 → 61.9.
    - Qwen3 prefill's largest items now are the W13 matmul (21.9%), the q/k/v matmul (14.7%), down (10.1%), `reduction_16x3072` (8.7%).

### 8.14 C8 layered restructuring of code generation: splitting `sahl-to-sahw` into passes

**Motivation (assessment of 2026-09-29)**: §8.3 / §8.4 designed 16 passes, each decision made in its own pass, runnable and testable alone with `iree-opt`. What was actually implemented is only:

| Actual pass | What it does |
|---|---|
| IREE's comprehensive bufferize | tensor → memref |
| `iree-sa-to-sahl` (140 lines) | Only replaces copies with `sahl.load / store`; the `sahl` dialect has just these two operations |
| **`iree-sahl-to-sahw` (3052 lines, one `Lowerer` class)** | Almost everything else: micro-kernel matching and selection, recognition of contraction / attention / linear layers, tiling and piecing (flat pieces, row pieces, K blocks, N blocks), on-chip memory allocation (sequential), prefetch and double buffering, DDR views and dynamic DMA, gather / scatter, translation of elementwise bodies |
| `iree-sahw-fuse-ve`, `iree-sahw-split-head`, `iree-sahw-assign-registers` | As designed |

Sizes of the parts of `SahlToSahw.cpp`: elementwise operations (`generic`) about 730 lines, generic contraction about 520, linear layers (including `linearRows`) about 430, attention about 250, DDR and DMA about 300, piecing about 200.

**Problems**:
- **Special cases interfere with each other**: the member state of the one `Lowerer` (`piece`, `rowPiece`, `locals`, `bcastOf`, `accTop`) is shared by all paths. For example C6.P's M = 16 bug: the offsets of flat pieces were applied to the view of a column slice.
- **Decisions are invisible**: tiling, piecing, memory placement and prefetch exist only in C++ control flow, invisible in the IR and not testable alone (`test/` only has lit tests for `sahw`).
- **Hard to extend**: T-tiled attention (C6.5), the cost model (§8.5), compile-time scheduling (the fusions C4 put on hold), host fallback and C7 all need tiling and memory planning visible in the IR.

**Goal**: move decisions into separate passes following the layering of §8.3 / §8.4; `sahl-to-sahw` finally does only one-to-one translation (target < 1000 lines). **The generated descriptors stay byte-identical at every step**: this is a restructuring, not an optimization, so no new board run is needed.

**The acceptance guard rail: a golden corpus** (done first)
- Save the `.sadesc` of every executable used by today's tests:
  - stories15M decode + prefill (M = 8, 181);
  - SmolLM2 prefill M = 8 (777);
  - the Qwen3 two-layer configuration;
  - the synthetic K = 16384 model;
  - the `--iree-sa-ukernels=none` configuration;
  - configurations with D = 16 and larger SPAD / ACC.
- `compiler/tests/golden.py`: `--record` saves; the default mode recompiles and compares byte for byte, listing the executables that differ; added to `run_tests.sh`.
- When a step has to change the output (e.g. the order of memory placement changes), the reason is stated separately; those executables are accepted instead by `dispatch_check` (T = 16 / 80 / 256) and end-to-end bit-exactness on the sim, comparing `board_profile` cycles.

**Steps**

| Step | Content | Acceptance |
|---|---|---|
| **R0 guard rail** | The golden corpus and `golden.py`; a lit-test framework for the existing passes (each pass runnable alone with `iree-opt`) | The corpus is reproducible (two recordings with the same compiler identical) |
| **R1 `sahl` operations** | Complete `sahl.matmul` (int8 tile multiply, int32 accumulation, accumulable), `sahl.elementwise` (with a region, keeping indexing maps), `sahl.reduce`, `sahl.relayout` (transpose, DIV-D replication, packing into broadcast words), `sahl.gather`, `sahl.scatter`, `sahl.scalar` as in §8.3; on-chip memrefs carry `#sa.mem` and `#sa.layout`. `sa-to-sahl` does the recognition: the matching of to_i8 chains, contractions, linear layers and attention moves out of `Lowerer`; `sahl-to-sahw` reads `sahl` operations instead of re-matching linalg | Golden corpus byte-identical; roundtrip and verifier lit tests for each kind of `sahl` operation |
| **R2 micro-kernels** | The linear-layer (including `linearRows`) and attention micro-kernels become expansions at the `sahl` level (§8.8): a pass expands matched operations into sequences of `sahl.load / matmul / elementwise`, selected by `--iree-sa-ukernels` | Byte-identical; both micro-kernel and `ukernels=none` configurations in the corpus |
| **R3 `sahl-tile`** | Tiling and piecing become IR: contractions' N blocks and K blocks, and elementwise operations' flat and row pieces, are all rewritten as scf loops (or unrolled `sahl` operations) acting on subviews; replacing `Lowerer`'s practice of rerunning `lowerBody` with member state | Byte-identical; lit: given the target configuration's capacities, check the tiling results |
| **R4 `sahl-plan-memory`** | The bank and offset of on-chip buffers are written as attributes by one pass (double-buffered blocks in different banks; when capacity is insufficient, error out with a suggested block size); replacing the sequential allocation of `allocAcc`, `newLocal` | Byte-identical (copying today's allocation order first); lit: capacity checks, bank conflicts |
| **R5 `sahl-pipeline`** | Double buffering and prefetch become IR: after block i's EX, load block i + 1 into the other bank (`linearRows`' prefetch, the linear layers' double buffering) | Byte-identical |
| **R6 translation and dynamic values** | `sahl-to-sahw` only translates one to one; dynamic values (row loops, dynamic DMA, PARAMs) move to `sahw-legalize-dynamic` | Byte-identical; `SahlToSahw.cpp` < 1000 lines; every pass has lit tests |
| R7 (later) | The cost model (§8.5) hooked into the choices of `sahl-tile` and `sahl-pipeline`; only now may the output change (accepted by cycles) | Board per-dispatch cycles not worse |

**R0 results (2026-09-29)**:
- **Golden corpus**: `compiler/tests/golden.py`, stored in `build/golden` (deduplicated copies of the sources, unaffected by model re-exports).
  - 9 groups: stories15M M = 8 (with / without micro-kernels), stories15M M = 16, SmolLM2 M = 8 / 16, Qwen3 all 28 layers, the K = 16384 synthetic model, D = 16, larger SPAD / ACC;
  - 3112 dispatches in total, 2959 after deduplication;
  - each source file compiled alone with `--compile-mode=hal-executable` gives descriptors byte-identical to the whole-model compile's exports (checked for all 7 groups that have exports);
  - recompiling and comparing takes only about 20 seconds (6 in parallel); two recordings are identical; a deliberately corrupted expected file is caught.
  - `run_tests.sh golden`: runs lit first, then compares the golden corpus.
- **Running passes alone**:
  - the `sahl` dialect was not registered with `iree-opt`, so `sahl`-level IR could not be read: now registered;
  - `iree-sahl-to-sahw` gains pass options for the target configuration (`d`, `spad-kb`, `acc-kb`, `ukernels`) and can run alone in `iree-opt`;
  - new lit tests: `sa_to_sahl.mlir`, and `sahl_to_sahw.mlir` (three configurations: single-stage VE, after fusion, D = 16; two functions, exp and per-row quantization).
- **An old gap found**: with `--iree-sa-ukernels=none`, prefill's 5 multi-row linear layers did not compile (`contraction epilogue input layout`: the generic contraction's tail does not support per-row inputs). §8.7's "everything compiles without micro-kernels too" only held for decode. The corpus records them as "expected failures"; R2 fixes this while moving micro-kernels to `sahl`-level expansions.

**R1 split and progress**:
- **R1a kernel grouping (done, 2026-09-29)**:
  - the three matchers (linear layer, attention, generic contraction) are extracted from `Lowerer` into `KernelMatcher` in `SahlKernels.{h,cpp}`, shared by both passes;
  - a new operation `sahl.kernel "linear" | "attention" | "contraction"`: `sa-to-sahl` runs the matchers and moves each matched set of operations, in their original order, into a `sahl.kernel` placed where its contraction (marked `sahl.anchor`) was; the allocations, dims, constants and views used by later operations are hoisted before the kernel (they generate no commands);
  - `sa-to-sahl` also gets the target-configuration pass options (the same as `sahl-to-sahw`);
  - `sahl-to-sahw` no longer searches the whole function for kernels itself, getting the plan only from each `sahl.kernel`'s anchor;
  - result: which micro-kernel is used and which operations are lowered together is visible in the IR (e.g. the same W13 is `"linear"` by default and `"contraction"` with `ukernels=none`); golden corpus byte-identical; new lit test `sahl_kernels.mlir`; `SahlToSahw.cpp` 3052 → 2579 lines.
- **R1b patterns within bodies (done, 2026-09-29)**:
  - elementwise operations and reductions stay represented as `linalg.generic`: it already is the "elementwise operation with a region and indexing maps" envisioned in §8.3, and inventing `sahl.elementwise` would just rename it. What really needed making explicit are two kinds of pattern decisions the lowering made inside bodies:
  - **`sahl.to_i8`**: the whole `qllama.to_i8` chain (NaN, ±inf, rounding, clamping, fptosi, 12 operations) is replaced in `sa-to-sahl` by one operation, which the lowering treats directly as VE's int8 output conversion;
  - **`sahl.gather "row" | "scalar" | "packed" | "swap"`**: memref.loads in bodies are replaced by gathers with a kind. The kind is decided in `sa-to-sahl` by the shared `gatherForm` (`SahlKernels.cpp`) over the iteration space the lowering will use; the lowering generates commands by kind and checks its own iteration space gives the same kind, erroring out otherwise;
  - in the corpus: gathers of whole rows 70, single scalars 63, packed rows 60, pair swaps 118; to_i8 1148; kernels linear 59, attention 60, contraction 179 (counted per dispatch function);
  - golden corpus byte-identical; new lit test `sahl_gather.mlir`, with to_i8 added to `sa_to_sahl.mlir`.
  - **One place left to keep the bytes unchanged**: the capacity estimate of flat pieces (`pieceWords`) used to count the operands of the body, and `sahl.to_i8` still counts as the 12 operations it replaces. Counting it as 1 would make 2 dispatches no longer pieced (smaller descriptors, possibly an improvement); left to be handled with R4's memory planning.
  - **Not made into operations**: shape normalization (flattening multi-level loops, merging by rows, dynamic lengths row by row) belongs to R3's tiling; the i64 scalar `x + c` is still a generic (used only for position computation; little value).
- **R1c scatter (done)**: a one-row `iree_linalg_ext.scatter` (KV write) becomes `sahl.scatter` in `sa-to-sahl`; lit test `sahl_scatter.mlir`. Layout conversions (transpose, DIV-D replication, packing into broadcast words) were not made into operations: they are inserted by the lowering where used, belonging to memory layout, see R4.

**R2–R6 results (2026-09-29, golden corpus byte-identical at every step, except R2's new functionality)**:

| Step | What was actually done | Differences from the plan and why |
|---|---|---|
| **R2 micro-kernels** | The generic contraction gains the three forms of prefill's linear layers (per-column tail inputs, int8 x in SPAD_A bank 1, storing the int32 accumulator directly without a tail): `--iree-sa-ukernels=none` now compiles every dispatch of every model (previously 5 prefill dispatches per model did not compile). The newly compiled dispatches are bit-exact with the oracle at T = 16 / 80 / 256, and stories15M's prefill + decode without micro-kernels is bit-exact with decode only on the sim; the corpus adds `stories_m16_none`, `smollm2_m8_none` (3962 dispatches in total) | The micro-kernels were not rewritten as `sahl`-level expansions: the scheduling of linear layers and attention is essentially command level (EX strips, alternating SPAD_B banks, LOOP_END pair loops, prefix prefetch), and expressing it at `sahl` would mean reinventing `sahw`'s concepts. The micro-kernels stay in the lowering, split into files by kind (R6) |
| **R3 `sahl-tile`** | The flat and row pieces used when an elementwise dispatch does not fit ACC become IR: one `sahl.scope` per piece (allocations within a piece released at its end), DDR views replaced by the piece's subviews (flat pieces through `collapse_shape`), local buffers replaced by piece-sized ones; reduction results shared across pieces are preallocated with `sahl.reserve` and filled only once, with later pieces' reductions marked `sahl.accumulate`. The capacity estimate moves from the lowering to this pass. In the corpus 17 dispatches are pieced (55 pieces); the lowering no longer has piece state | Contractions' N / K blocks stay in the kernel lowering (K blocks depend on the address where x lands during lowering); the row-by-row lowering of dynamic lengths also stays in the lowering (it is driven by "at most 2 dynamic fields per descriptor", part of command encoding) |
| **R4 memory** | `sahl-plan-memory`: each local buffer's placement (`sa.mem`: `spad_a` / `acc`) and layout (`sa.layout`: `packed` / `bcast` / `rows`) written as attributes, which the lowering follows and checks; the allocator extracted as `LocalMemory` (`SahlLocalMemory.h`: sequential allocation, released by markers at the end of pieces, blocks and rows) | Addresses are not decided in the pass: the lowering's temporaries and buffers are allocated interleaved at first use, so fixing addresses in advance would necessarily change the output; lifetime-based allocation would change addresses and the bank conflicts the scoreboard sees, and needs board cycle measurements first (moved into R7) |
| **R5 `sahl-schedule`** | The linear micro-kernel's schedule (`chunk_tiles`: output tiles per chunk; `loop`: LOOP_END pair loops when the decode form exceeds 4 chunks) is written onto the `sahl.kernel` by this pass, with the formula shared (`linearSchedule`), and the lowering generates accordingly | Where double buffering and prefetch go is still decided by the micro-kernel's lowering (command level, see R2); the generic contraction's blocks are decided by the lowering (see R3) |
| **R6 translation only** | The lowering's declarations move to `SahlLower.h`, with definitions split into files by responsibility: `SahlToSahw.cpp` 541 lines (the pass, block traversal, registers, local memory, DDR / DMA, VE, scatter), `SahlLowerGeneric.cpp` (elementwise operations, reductions, gathers), `SahlLowerKernels.cpp` (linear layers, attention, generic contraction); lit tests added for `sa-pack-linear-weights` and `sa-clone-cheap-producers` (10 lit tests in total) | No `sahw-legalize-dynamic`: dynamic values (PARAMs, row loops, dynamic DMA) are handled while generating commands according to the descriptor encoding's limits, and `sahw` currently has no intermediate form of "symbolic dynamic fields"; to be split out when C7 (the RISC-V backend, without the 2-dynamic-field limit) needs it |

**C8 results**:
- **The pass pipeline** (each runnable alone with `iree-opt`, each with lit tests): IREE bufferize → `iree-sa-to-sahl` (DDR traffic explicit; kernel grouping, gather kinds, to_i8, scatter) → `iree-sahl-tile` (pieces) → `iree-sahl-plan-memory` (placement and layout) → `iree-sahl-schedule` (linear-layer scheduling) → `iree-sahl-to-sahw` (translation) → `iree-sahw-fuse-ve` → `iree-sahw-split-head` → `iree-sahw-assign-registers` → serialization.
- **Decisions visible in the IR**: which kernel is used, gather kinds, pieces, buffer placement and layout, linear-layer chunking.
- **Code**: the original 3052-line `SahlToSahw.cpp` split into:
  - shared matching and classification `SahlKernels.cpp`, about 720 lines;
  - the passes: `SaToSahl.cpp` about 290 lines, `SahlTile.cpp` about 350 lines, `SahlPlanMemory.cpp`, `SahlSchedule.cpp`;
  - translation: `SahlToSahw.cpp` 541 lines, `SahlLowerGeneric.cpp` 754 lines, `SahlLowerKernels.cpp` 710 lines, plus `SahlLower.h`, `SahlLocalMemory.h`.
- **Guard rail**: `run_tests.sh golden` (lit + byte comparison of 3962 dispatches, about 30 seconds). Throughout the restructuring no existing dispatch's output changed, so no new board run is needed.
- **Not achieved, left for later**:
  - `sahl`-level expansion of micro-kernels and an IR representation of their double buffering / prefetch;
  - lifetime-based memory planning (changes the output, needs board cycle acceptance);
  - `sahw-legalize-dynamic`;
  - R7 cost model.
  - The one estimate kept to preserve the bytes (`sahl.to_i8` counted as 12 operations in the piece estimate), handled together with lifetime-based memory planning.

**Order and dependencies**: R0 → R1 → R2 → R3 → R4 → R5 → R6, each committed separately. R1 is the largest step (all recognition logic moves); R3 and R4 are prerequisites of C6.5 (T-tiled attention) and the cost model.

**Risks**:
- **Byte-identical is too strict**: in some steps the order of operations or allocations is hard to keep as is. Mitigation: copy today's order in the new pass first; what truly cannot be kept is accepted by the exception rule above.
- **bufferize and memory spaces**: once on-chip memrefs carry `#sa.mem`, IREE's bufferize and memref canonicalization may not recognize them; if necessary `#sa.mem` appears only after `sa-to-sahl`.
- **Effort**: large (about the same as C5.1–C5.3). This stage has no visible features; its value shows in later extensions (T tiling, cost model, host fallback, C7, a generic frontend).

**Suggested coverage work afterwards** (outside this section, planned separately):
1. **Host fallback**: dispatches sa cannot compile go to llvm-cpu on the ARM (IREE's heterogeneous devices); today they can only error out.
2. **A generic frontend**: quantization and formulation rewrites (gather, how KV is written, prefill's row-by-row unrolling) move from the hand-written models into the compiler's preprocessing passes, so HuggingFace's original modeling code compiles directly.
3. **Non-Llama structures**: e.g. GPT-2 (LayerNorm, GELU), to validate generality.

### 8.15 Host fallback: dispatches sa cannot compile run on the ARM

**Goal**: any model runs: dispatches the sa backend cannot compile go to the ARM host, the rest stay on the accelerator as before, both executing in order on the same device within the same command buffer. The standard practice of industrial compilers (operators the accelerator does not support fall back to the CPU).

**Approach** (done 2026-09-29):
- **VMVX for host code**: IREE's reference CPU backend, compiling dispatches into VM bytecode interpreted at run time (with micro-kernels). It is already in the compiler build and the ARM runtime carries the VMVX loader, so no LLVM toolchain and no compiler rebuild are needed. Slow, but only for the fallback; if speed is needed later it can be replaced by llvm-cpu (the compiler would need the LLVM_CPU backend enabled and armv7 cross-compilation).
- **Compiler** (`--iree-sa-host-fallback`, on by default in `compile_sa.sh`, off with `SA_HOST_FALLBACK=0`):
  - the sa device's target lists both executable formats `sa-desc-v1` and `vmvx-bytecode-fb`, so every dispatch is compiled into two variants (sa first);
  - when the sa backend cannot compile a dispatch, instead of erroring out it adds an always-false `hal.executable.condition` to the sa variant:
    - after translation IREE generates the variant selection at the dispatch's call site (the first variant whose format is available and whose condition holds), so this dispatch selects the VMVX variant;
    - the same holds for the selection at executable creation (`MaterializeResourceCaches`);
  - IREE's executable linking is turned off (`--iree-hal-link-executables=false`): VMVX linking would merge all VMVX variants into one executable and redirect references to the original executables wholesale (marked as risky in IREE's source), conflicting with executables that still keep their sa variant (the compiler crashed). sa never links anyway; without VMVX linking each dispatch just has a small module;
  - for testing: `--iree-sa-host-dispatches=<name fragment,...>` forces the given dispatches onto the host.
- **Runtime**:
  - the sa driver additionally registers an IREE VMVX loader, and the executable cache picks the loader by format;
  - when the command buffer meets a non-sa executable, it first submits the queued descriptor list (its inputs may come from the accelerator), then executes on the ARM workgroup by workgroup the way IREE's inline command buffers do;
  - buffers are the device window (sim: a shared file; board: the CMA window), so both sides see the same memory;
  - `SA_STATS=1`: prints the number of descriptor lists on the accelerator and dispatches on the host at the end.
- **Approaches considered but not used**:
  - deleting the sa variants that cannot compile at link time: the variant selection at dispatch call sites is generated before linking, so deleting would leave dangling references;
  - two devices (sa + local) with per-dispatch affinity: whether a dispatch is supported is only known during sa code generation, while affinity must be decided before dispatch formation.

**Verification** (`run_tests.sh fallback`, `test_fallback.py`; lit `sa_host_fallback.mlir`):
- A small model: `tanh` (sa has none, host) → `exp` (the accelerator's SFU): maximum relative error versus numpy 1.6e-5 (from the SFU's exp approximation); the split is 1 list on the accelerator, 1 dispatch on the host;
- stories15M with all decode linear layers (25 dispatches / token) forced onto the host: a 16-token prompt plus 4 generated steps, logits bit-identical to the accelerator-only version on 20/20 rows (the linear layers are integer multiply-adds plus fp32 multiplies, the same bits on both sides);
- with the fallback on by default, stories15M's prefill + decode remains bit-exact, and the 182 sa executables are byte-identical to before; the golden corpus is unaffected.
- **On the board (2026-09-29)**:
  - stories15M with the linear layers on the ARM (`deploy_fallback.sh`): 9/9 steps bit-exact with the sim, identical text; the split is 354 lists on the accelerator, 201 dispatches on the host (8 steps × 25, plus the classifier of prefill's last chunk). Very slow: 16.8 s per decode step, prefill 10.5 s (with the 288 × 32000 classifier also on the host). About 0.7 µs per weight byte: VMVX interprets on the ARM, reading operands directly from the uncached CMA window;
  - ordinary stories15M with the fallback on by default and nothing forced: 61/61 steps bit-exact, 0 host dispatches, prefill 211.3 ms (before 211.6), 64 tokens 981.0 ms (before 981.7), decode 57.5 ms / step (before 57): **no cost when unused**.
- **Cost**: SmolLM2 M = 8 compiles in 261 instead of 140 seconds, vmfb 6.6 → 10.1 MB (every dispatch gets an extra VMVX variant); no run-time cost (dispatches sa can compile still select the sa variant).
- **Possible later**:
  - compile VMVX variants only for dispatches that cannot be compiled (needs deciding compilability before translation);
  - run host dispatches in parallel with the accelerator (today they first wait for the accelerator to finish);
  - llvm-cpu instead of VMVX;
  - copy host dispatches' operands into cached memory first (or give the CPU a cached mapping and flush caches around accelerator lists).

### 8.16 A generic frontend: compiling HuggingFace's original modeling code directly

**Goal**: no more hand-written models (qhf.py rewrote Llama / Qwen3 in the form of QLlama). transformers' modeling code is exported as is, and quantization plus the formulations prepared for the accelerator (gather, KV layout and int8, RoPE tables, pair swap, the form of attention) are done by generic graph rewrites in the compiler frontend, not specific to any model.

**F0 / F1: exported as is, correct end to end (done, 2026-09-29)** (`compiler/frontend/hf_generic.py`, `run_tests.sh hfgen`):
- **Environment**: the frontend venv has transformers 5.17 installed (torch 2.14, iree-turbine 3.9).
- **Export**:
  - `HFDecoder`: `decode(input_ids[1,1], position_ids[1,1]) -> logits`, wrapping `AutoModelForCausalLM`;
  - transformers 5.x's static KV cache tracks the position itself (a scalar buffer incremented every step), which errors on import into turbine; `PositionedCache` instead writes at the position given by the caller, into a buffer of the wrapper module (accessed through module attributes, so torch.export recognizes it as a buffer mutation and the KV cache becomes a mutable global);
  - transformers' own export wrapper (`TorchExportableModuleForDecoderOnlyLM`) in 5.17 drops `cache_position` in its outer forward, so it is not used;
  - flow: torch.export (non-strict) → `run_decompositions()` (functional ATen) → graph rewrites → externalized parameters (must come after decomposition: marking them first makes aot_autograd fail) → iree-turbine.
- **Graph rewrites** (`GRAPH_PASSES`): currently only `pow(x, 2) → x * x` (VMVX has no `fpowi`; 61 places).
- **Running**: `sa-llm-run --abi=hf` (`main(input_ids, position_ids)` every step, no valid).
- **Results** (SmolLM2-135M, sim):
  - 32 executables, **3 on the accelerator, 30 on the host**;
  - 9 steps versus transformers' fp32: minimum correlation 1.000000, maximum error 2.6e-3, argmax 9/9 identical;
  - very slow (about 3.8 seconds per step); the transposed fp32 weights are evaluated at compile time and embedded in the vmfb (538 MB), and the sim window must be opened to 1.1 GB.

**F2: W8A8 for linear layers and the embedding (done, 2026-09-29)**:
- Replacement by module type: before export every `nn.Linear` (211 in SmolLM2) becomes a `QLinear` and `nn.Embedding` a `QEmbedding`, with the same formulas as qhf (weights int8 per row, activations int8 dynamic per token, int32 multiply-add, then the two scales in turn); an embedding shared with the classifier is quantized once with a shared int8 table. Independent of the specific model (the same as torch.ao's per-module quantization).
- The decomposition table switches to iree-turbine's (the one `aot.export(module)` uses): torch's default table decomposes the quantization chain (`nan_to_num`, `round`, `clamp`) into forms the compiler does not recognize.
- **A bug of the sa compiler found and fixed as a result**:
  - HF's linear layers are `batch_matmul(x, extsi(Wᵀ))`, with unpacked weights, lowered by the generic contraction;
  - when the tail only truncates and outputs int32, the per-block tail still wrote results as fp32, converting the int32 into floats, so all results were wrong;
  - now `Chunk` carries the output type, and the contraction allocates and writes back by the tail's actual type;
  - found by the per-dispatch checks; after the fix the existing corpus is byte-identical. This model's dispatches were added to the golden corpus (4002 in total).
- **Results** (SmolLM2, sim):
  - 13 executables on the accelerator, 27 on the host (F0 had 3 and 30);
  - versus the quantized model in torch: 9 steps with minimum correlation 1.000000, maximum error 1.9e-4, argmax 9/9 identical;
  - parameters 176 MB (int8).

**F3: RoPE (done, 2026-09-29)** (`rope_rewrite`):
- Rotary position embedding modules (class names ending in `RotaryEmbedding`) are replaced by `RopeTable`: the original module computes positions 0 … max_len−1 once, giving `cos` / `sin` tables, which are then looked up by position (works for any RoPE variant; frequency scaling is in the tables too);
- `rotate_half` becomes a pair swap: each head's dimensions in the q / k projections have their two halves interleaved (q·k unchanged: both sides permuted the same way), and the tables' columns and the q / k norm weights are permuted the same way; in every transformers model module defining `rotate_half`, it is replaced by `swapneg` (the two are identical under the interleaved permutation);
- **The fp32 model's output is exactly the same as the original model's** (maximum error 0);
- SmolLM2: 17 executables on the accelerator, 20 on the host; 9 steps argmax 9/9 identical; all 17 compiled by sa consistent with the oracle.

**RMSNorm and lookups** (done): modules whose class names end in `RMSNorm` are replaced by `SaRMSNorm` (`mean` becomes a multiply by the constant 1/n, as in qhf); embedding and RoPE table lookups are written as `index_select` (the compiler's row gather). SmolLM2: 22 on the accelerator, 15 on the host.

**F4: int8 KV cache and attention (in progress, 2026-09-29)**:
- Approach:
  - register an `"sa"` attention through transformers' `AttentionInterface` (`sa_attention`, qhf's formulas: q quantized per head, int32 scores, softmax masked by position, int8 probabilities, int32 P·V; since 2026-10-13 the probabilities quantize the unnormalized `e = exp(s − max)` and are normalized by Σ i8(e·127) (qsum, `generality_implementation_plan.md` §12.4));
  - `SaCache`: one int8 buffer each for K and V, `[layers × T, kv_dim]`, written by rows, with static scales per layer;
  - `calibrate_kv`: the fp32 model collects max |K|, |V| on qhf's calibration text (the same as qhf).
- **A second bug of the sa compiler fixed as a result**: a multi-row static generic with a prefix mask (static-T softmax, 9 rows × 256) was merged into one VE whose VALID only truncated the first pos + 1 elements of the whole vector, so everything from row 1 on was wrong; now lowered row by row (`RowSel::fixed`). The existing corpus is byte-identical.
- **An IREE 3.11 issue found** (`compiler/tests/iree_global_merge_repro.py`, reproducible in 20 lines, unrelated to sa, also occurs on VMVX): two mutable int8 globals with identical initial contents (compiled with imported parameters), each written one row in the same call, get the second one's data written into both. Workaround: give each buffer different initial contents; and, as qhf does, have all layers share one K and one V buffer.
- **The wrapper's causal mask bug (fixed)**: transformers 5.x builds the causal mask from the length the cache records itself, which `PositionedCache` did not update, so the mask always assumed position 0 and attention only looked at position 0. The tests of F1–F3 compared the compiled results with the same wrapper, wrong in the same way, so it was not noticed then; F4's calibration was biased by it too. Now each step sets this length to the input position (an attribute assignment, not mutable state); the tests first check the wrapper agrees with the original HF (computing the whole sequence at once; maximum error 1e-4). F1–F3's conclusions were re-verified accordingly: the W8A8 + RoPE configuration versus the torch quantized model has correlation 0.984 and argmax 8/9 (the earlier "completely identical" was because attention effectively looked at only one position).
- **Results** (SmolLM2, sim):
  - 171 executables on the accelerator (KV writes and attention all moved to the accelerator), 33 on the host (mainly linear layers whose tails have int8 outputs: the K / V projections fused with the quantization for the cache write);
  - every dispatch sa compiles is bit-exact with the oracle (T = 16 / 256);
  - step-by-step comparison with the torch quantized model is not a suitable criterion: int8 rounding is sensitive to the order of operations (the device and torch sum in different orders), and flips amplify with the number of steps; two correct programs (hand-written qhf and the generic path, with the same KV scales) also drift apart gradually on the device.
  - **Quality** (a fixed text fed position by position for 40 tokens, top-1 agreement with the original fp32 HF): **generic path 35/40 (mean correlation 0.9567), hand-written qhf 36/40 (0.9569)**, 39/40 between the two (this time the generic path used qhf's KV scales); with its own calibrated KV scales (differing from qhf's only in the last bit) **34/40 (0.9566)**, and `run_tests.sh hfgen` passes.
  - The test criterion becomes this quality metric (≥ 85%, mean correlation > 0.95). The reference model must be computed before any rewrite: `rope_rewrite`'s replacement of `rotate_half` applies to the whole process.
- **What remains of F4**: linear layers whose tails have int8 outputs onto the accelerator (the generic contraction's tail supporting int8 output). **Done** (§8.19: int8 tails, int32 elementwise inputs for SwiGLU; decode entirely on the accelerator).

**Reasons for landing on the host (F0's measurement) and the matching generic rewrites**:

| Count | Reason | Rewrite | Step |
|---|---|---|---|
| 6 | Linear layers are fp32 (EX only does int8) | W8A8: weights int8 per channel, activations int8 dynamic per token (the form of qhf's qlinear), applied to the `aten.linear` of every constant weight | F2 |
| 6 | ACC too small (large fused fp32 dispatches) | Sizes shrink after F2; what remains depends on tiling | F2 |
| 2 | `sin` / `cos`: HF computes RoPE every step | The position-independent part constant-folded into a table `[max_len, dim/2]`, gathered by position | F3 |
| 1 | `rotate_half` (concatenating the two halves) | The q / k projection weights rearranged per head into interleaved pairs, so RoPE becomes a pair swap (the approach of llama2.c's export and qhf) | F3 |
| 3 + 2 + 3 | Attention is an fp32 `batch_matmul`; KV written as `[B, H, T, D]`; positions are vectors | KV becomes int8 `[T, H, D]` (static per-layer scales calibrated with the fp32 model), SDPA rewritten as scores, a masked softmax and P·V, in the same form as qhf | F4 |
| The rest | Division etc. | As needed | F5 |

**The main acceptance basis**: originally bit-exactness with the hand-written qhf path; in practice the two frontends differ in formula details (HF computes the RoPE tables in fp32, qhf in float64; attention's summation length), and int8 rounding is sensitive to the order of operations, so bit-exactness is not the right target. Instead: every sa dispatch bit-exact with the oracle (correctness of the compiler), plus quality versus the original fp32 HF no worse than the hand-written path (correctness of the frontend), and it runs on the board; then Qwen3 verifies the same rewrites need no code changes. After that F6: non-Llama structures (GPT-2: LayerNorm, GELU, learned position embeddings).

### 8.18 How large an FPGA a usable Qwen3 needs (estimate, 2026-09-29)

Single-sequence generation is bound by memory bandwidth: each generated token reads all weights (plus the classifier) from DDR once, so generation speed ≈ usable bandwidth ÷ bytes read per token. The following is extrapolated from the public Qwen3 configurations, not measured; "usable" means interactive generation ≥ 10 token/s with a 2K context.

| Model | Weights read per token (int8 / int4) | KV cache (int8, 2K) | Bandwidth needed for 10 token/s (int8 / int4) |
|---|---|---|---|
| Qwen3-0.6B (28 layers, hidden 1024) | About 0.6 GB / 0.3 GB | About 117 MB | About 6 / 3 GB/s |
| Qwen3-1.7B (28 layers, 2048) | About 1.7 GB / 0.9 GB | About 117 MB | About 17 / 9 GB/s |
| Qwen3-4B (36 layers, 2560) | About 4.0 GB / 2.0 GB | About 150 MB | About 40 / 20 GB/s |
| Qwen3-8B (36 layers, 4096) | About 7.6 GB / 3.8 GB | About 150 MB | About 76 / 38 GB/s |

- In small models the classifier (150K vocabulary × hidden dimension) is a large part of the bytes read per token (about a quarter for 0.6B); generation also reads the KV cache (about 60–75 MB per token at an average context of 1K).
- Compute requirements are modest: decode is about "parameter count" multiply-adds per token (about 6 GMAC/s for 0.6B at 10 token/s); only prefill is compute hungry (512 tokens is about 300 GMAC; finishing in 1–2 seconds needs 150–300 GMAC/s).
- **The PYNQ-Z1 cannot**: 512 MB of DDR cannot hold 0.6B's int8 weights (int4 would fit); the measured weight read rate is about 0.39 GB/s (one 64-bit HP port, 50 MHz), so even int4 gives only about 1 token/s; an 8×8 array at 50 MHz peaks at 3.2 GMAC/s; LUTs are already 83% used.

| Class | Typical device / board | Memory and bandwidth | Qwen3 that runs |
|---|---|---|---|
| Entry Zynq UltraScale+ | Kria KV260 (K26 SOM, about 250k logic cells, 1248 DSPs, 4 GB 64-bit DDR4, about 19 GB/s theoretical, about 10 GB/s usable from the PL) | 4 GB | 0.6B int8 about 10–15 token/s; 1.7B int4 about 8–10 token/s |
| Mid-range Zynq UltraScale+ | ZCU104 / ZCU102 (DDR4 on both PS and PL) | 4–6 GB, about 20–30 GB/s combined | 1.7B int8 about 10 token/s; 4B int4 about 5 token/s |
| With HBM | Alveo U55C / U280, Versal HBM | 8–16 GB, about 400–460 GB/s | 4B / 8B int8 tens of token/s |

**Conclusion**: a "usable" Qwen3-0.6B / 1.7B needs a KV260 (K26) class board; 4B / 8B need HBM. The accelerator has to change with it: DMA widened to 2–4 128-bit HP ports at 200–300 MHz (the main source of speedup); int4 unpacking on the LD path (half the bandwidth); an array of D = 16 or 32 (compute for prefill; M4 already verified D = 16); larger SPAD / ACC in URAM; the compiler's T-tiled attention (C6.5). The compiler side is ready: hardware parameters come from the target configuration, Qwen3-0.6B's 28 layers are verified per dispatch on the sim, and the generic frontend can use HF's Qwen3 code directly.

### 8.17 Next steps (decided 2026-09-29)

In order:

| Order | Work | Content | Acceptance | Risks and dependencies |
|---|---|---|---|---|
| 1 | **Prefill on the generic path** (end-to-end prefill + decode) | Export `prefill(tokens[M], positions[M])` in the same module as decode, sharing parameters and the KV cache; `SaCache` writes M rows at once; the `"sa"` attention adds causal masks per row (qhf's row-by-row form); `sa-llm-run --abi=hf` supports chunked prefill; F4's remaining int8 tails along the way | Bit-exact on the device with the same model using decode only (C6.P's criterion); no drop in the quality metric; Qwen3 without code changes; on the board | Mostly reuses the existing compiler (multi-row linear layers and row-by-row attention already exist); may expose more compiler corner cases. **Done** (2026-09-29, sim and board, §8.19) |
| 2 | **Micro-kernels expanded at the `sahl` level** | The scheduling of linear layers, attention and the generic contraction (tiling, alternating SPAD_B banks, prefetch, LOOP_END loops) expressed explicitly with `sahl` operations and expanded by a pass; `sahl-to-sahw` only translates one to one; `sahl` needs operations that can express banks, strips and loops | Golden corpus byte-identical, no board run needed | Large effort; not done in C8 (§8.14 R2: it amounts to reinventing `sahw`'s concepts). Its value now is making scheduling visible to and changeable by the cost model and memory planning; also a prerequisite of item 3. **Done** (2026-10-04, golden corpus all identical, the old kernel lowerings deleted, §8.20). After item 1 there is also a direct benefit: the linear-layer schedule can wrap any tail, while about 66% of the generic path's decode weights go through the generic contraction today, about 50% slower than the hand-written path (§8.19); that part changes the output and is accepted the way item 3 is (consistent per dispatch, board cycles) |
| 3 | **Lifetime-based memory planning** | Once all local buffers and temporaries are in the IR (item 2), allocate SPAD / ACC by lifetime, avoid bank conflicts, and let the planning decide pieces when things do not fit (replacing today's estimates, including the place where `sahl.to_i8` counts as 12 operations) | Changes the output: per-dispatch checks and sim bit-exact; board per-dispatch cycles not worse | Changes in addresses and banks affect the scoreboard's concurrency, possibly faster or slower; must be measured on the board. **Moved to the KV260** (2026-10-04, §8.21): on Z1 decode is bandwidth bound, so the planning's gain cannot be measured, while the constraints it must meet (K2b's sub-banks, K3's capacity) are only settled on the KV260; split into 3a framework (reproducing the existing allocation, output unchanged) and 3b optimization, written into the compiler work items of K2b in `kv260_upgrade_plan.md` |

Reasons: item 1 is functional completeness (the generic frontend catching up with the hand-written path); item 2 is a pure restructuring guarded by the golden corpus; item 3 depends on item 2 and is the only step needing performance measurement on the board. After items 1 and 2 were done on Z1, Z1 was frozen (§8.21), and later work follows `kv260_upgrade_plan.md`.

### 8.19 Prefill on the generic path (result of §8.17 item 1, 2026-09-29)

**Results** (sim; SmolLM2-135M and Qwen3-0.6B truncated to 2 layers, both with the original HF modeling code, W8A8, M = 8):

| | SmolLM2-135M | Qwen3-0.6B (2 layers, no code changes) |
|---|---|---|
| prefill + decode vs decode only | Bit-exact (5/5 rows, identical tokens) | Bit-exact (5/5 rows, identical tokens) |
| Per decode step | One descriptor list, **0 host dispatches** | One descriptor list, 0 host dispatches |
| Executables (decode + prefill) | 794 on the accelerator, 2 on the host (Q's RoPE in prefill) | — |
| Quality (teacher-forced top-1 vs fp32 HF) | Device 34/40 (0.9573), torch W8A8 34/40 (0.9558) | Device 22/40, torch W8A8 21/40 |
| Per-dispatch checks (`--dirty`) | All 794 consistent | — |

**What was done**:
- Frontend: `prefill(tokens[1, M], positions[1, M])` / `prefill_kv` exported together with decode (`aot.FxPrograms`, sharing parameters and the KV cache); `SaCache` writes KV row by row; the `"sa"` attention adds prefix masks row by row; embedding and RoPE tables gathered row by row; `QLinear` multiplies over 2-D rows (a 3-D int32 matmul would export as a `batch_matmul` with i64 accumulation, which can only run on the host); `sa_attention` turns the query back into `[M, H, hd]` (each row contiguous). `sa-llm-run --abi=hf` supports chunked prefill.
- Compiler: the generic contraction's tail supports int8 outputs (the K / V projections fused with KV quantization) and int32 elementwise inputs (the MLP's `silu(gate) · up` fused into gate's tail); `sa-to-sahl` unrolls three-level loops the lowering does not accept (an outer small static row loop: RoPE and KV quantization over M rows) by rows into decode's two-level form, slicing the gather sources by row.
- Tests: `test_hf_generic.py --prefill M` (prefill + decode bit-exact with decode only), `--layers N` (truncated models), `--board-bundle`; `dispatch_check.py` gains i64 vectors with distinct values, no writes outside the storage region (except within 8-byte-aligned ranges), `--skew`, `--dirty`; `sa_sim_server.py --wlog`.

**An IREE defect found**: prefill at first did not match decode at all (correlation ≈ 0), while every dispatch checked alone was consistent. Bisecting by moving batches of prefill dispatches to the host with `--iree-sa-host-dispatches` located one sa dispatch (27); its output was bit-correct, but the host dispatch right after it (28) had identical inputs and different outputs in the two runs, with different workgroup counts (1 and 3). The cause is IREE's `MergeIndexSwitchPattern`, which merges adjacent `scf.index_switch` ops comparing only case values, not the selector; with host fallback two adjacent dispatches have different variant selectors (one selecting sa, one VMVX), so after merging the host dispatch used the sa variant's workgroup count and computed only a third. The fix is one line (require identical selectors), kept in `compiler/patches/iree/`, applied automatically by `fetch_iree_sources.sh`, guarded by `merge_index_switch.mlir`. A second defect in the same pattern (2026-10-13, `0002-util-merge-index-switch-yield-outer-values.patch`): the second switch's yields were looked up with `lookupOrNull`, so a value defined above both switches (e.g. a constant hoisted out of the cases) became a null operand and crashed the compiler; fixed with `lookupOrDefault`, guarded by the same test (golden corpus unchanged). Both fixes are submitted upstream (2026-10-13): iree-org/iree [#25037](https://github.com/iree-org/iree/pull/25037) (same selector) and [#25038](https://github.com/iree-org/iree/pull/25038) (outer yields); upstream main 17fb04d still had both defects. Lesson: per-dispatch checks cannot see problems between dispatches or in the runtime; bisection plus comparing hashes of the bound contents per dispatch across two runs is an effective method.

**The quality criterion changed**: it used to be "top-1 agreement with fp32 HF ≥ 85%", which actually judges the quantization scheme, not the compiler: on the full Qwen3-0.6B torch's W8A8 model itself only gets 25/40 (21/40 with 2 layers). The criterion is now that the device keeps the quantized model's quality: agreement no lower than the torch W8A8 model's minus 2, mean correlation no lower than its minus 0.01; both are printed. **Qwen3's W8A8 quality is a separate problem** (C6.2 quantization quality: per-token activations + per-row weights + per-layer static int8 KV + int8 probabilities lose noticeably on Qwen3), not part of item 1.

**On the board** (`compiler/scripts/deploy_hfgen.sh` → `build/deploy_hfgen`, `board_llm.py`, window 168 MB): SmolLM2 prefill (M = 8) + 16 decode steps, logits bit-exact with the sim for 17/17 steps, identical generated tokens, **board PASS**. 211 descriptor lists on the accelerator, 180 host dispatches (2 × 30 layers × 3 chunks, all in prefill; 0 in decode). Prefill of 20 tokens 8.5 s (2.35 token/s, the first chunk including loading), decode 681 ms/step (1.47 token/s) — about 50% slower than the hand-written qhf path (about 450 ms/step, 2.22 token/s); the dispatch split differs, not yet profiled.

**Why the generic path's decode is about 50% slower than the hand-written path** (681 vs about 450 ms/step; structural comparison on the sim, plus per-export board cycles from `board_profile.py`):
- Dispatches per step: 910 hand-written, 1000 generic, only 10% more; the difference is in the linear layers. The hand-written path has 4 linear layers per layer (q/k/v merged into one, gate/up merged into one), the generic path HF's 7. The weight bytes are the same and decode is bound by weight read bandwidth, so the fixed overhead of the extra dispatches cannot explain 50%.
- The main cause: **about 66% of the weights go through the generic contraction instead of the linear micro-kernel**. The linear micro-kernel only matches tails of the hand-written path's kind (`(acc · s_w) · s_x` with fp32 output); on the generic path IREE fused more into the tails, or moved them away:

| Linear layer (per layer) | Weights | Tail | Lowering |
|---|---|---|---|
| gate | 1536×576 | `silu(gate) · up` (up's int32 accumulation as an elementwise input) | Generic contraction |
| up | 1536×576 | None (int32 accumulation output directly) | Generic contraction |
| k, v | 2 × 192×576 | Dequantize, multiply by the KV scale, then int8 (cache write) | Generic contraction (one executable per layer: the KV scale is a constant) |
| One of q, o | 576×576 | | Generic contraction |
| The other of q, o, down, classifier | | `(acc · s_w) · s_x` | Linear micro-kernel |

  The generic contraction lacks the micro-kernel's scheduling (SPAD_B double buffering, chunk 0 prefetch, LOOP_END loops). **Measured on the board** (about 34.8M device cycles per decode step ≈ 696 ms, matching the wall clock of 681–713 ms: decode's time is essentially all on the device; `board_profile.py`'s "per step" divided by half again as many calls, so cycles must be multiplied by 2):

| Linear layer (per layer) | Weights | Lowering | Cycles/call | Weight-read floor (7.8 B/cycle) | Ratio |
|---|---|---|---|---|---|
| gate (fused SwiGLU) | 885 KB | Generic contraction | 282k | 113k | 2.5× |
| up (int32 output directly) | 885 KB | Generic contraction | 225k | 113k | 2.0× |
| down | 885 KB | Linear micro-kernel | 124k | 113k | 1.1× |
| One of q / o | 331 KB | Generic contraction | 85k | 42k | 2.0× |
| The other of q / o | 331 KB | Linear micro-kernel | 53k | 42k | 1.25× |
| v (fused KV quantization) | 110 KB | Generic contraction | 15k | 14k | 1.05× |

  What is slow is the large generic contractions (the small k / v are fine); gate + up take 45% of decode device time. At the micro-kernel's efficiency these three would save about 290k cycles per layer, about 8.7M per step (34.8M → about 26M, about 520 ms); the remaining gap to the hand-written path comes from the hand-written path merging q / k / v and gate / up into one linear layer each (avoiding repeated activation quantization and each linear layer's fixed overhead).
- **Prefill has a bigger problem of its own**: in prefill, the gate fused with SwiGLU (`matmul_like_8x192x8x576`) alone takes **48.9%** of prefill device cycles, 1.64M cycles per call, 12.8× the same-sized up without a tail (128k): at M = 8 its tail is done per chunk and per row (small DMAs of elementwise inputs serialized with the SFU).
- **Prefill's gate fixed** (2026-09-29): the linear micro-kernel only accepted fp32 elementwise tail inputs, while SwiGLU's up accumulation is int32, so gate fell to the generic contraction (at M = 8 one EX per row, a replicated strip, the tail per row: 1940 descriptors); it now accepts int32 and takes the micro-kernel (one EX for 8 rows: 229 descriptors). On the board (bit-exact, board PASS):

| | Before | After |
|---|---|---|
| prefill's gate per call | 1.64M cycles | 552k |
| Device cycles per prefill chunk (8 tokens) | 100.9M | 68.1M (−32%) |
| prefill wall clock (20 tokens, including loading) | 8.5 s | 6.5 s (3.06 token/s) |
| decode's gate per call | 282k | 177k |
| Device cycles / wall clock per decode step | 34.8M / 681 ms | 30.8M / 618 ms (1.62 token/s) |

  gate is still 4.3× up: SwiGLU's exp and reciprocal (8 × 1536 elements, 2 SFU operations each, about 19 cycles) take about 470k cycles, the SFU's latency (the same as the hand-written path's SiLU; P6 SFU multi-slot interleaving, §8.13, on hold). decode still has up (225k/call, 22%, no tail: the decode form of the micro-kernel only took tailed layers) and one of q / o (85k/call) on the generic contraction, estimated to save about 4.5M cycles/step more (about 530 ms). With prefill, `sa-llm-run`'s per-step `SA_PROFILE` numbers used to be divided by all steps (leaving only half); now divided by the counted decode steps.
- **decode's up, one of q / o and prefill's k / v fixed too** (2026-09-29): the decode form of the linear micro-kernel also accepts "no tail, store the int32 accumulation directly" (the multi-row form already did); the multi-row form also accepts int8 outputs (k / v fused with KV quantization; decode's v is already near its floor and unchanged). On the board (bit-exact, board PASS):

| | Before | After |
|---|---|---|
| decode's up per call | 225k cycles | 123k (the same as down) |
| decode's one of q / o per call | 85k | 51k (the same as the other) |
| Device cycles / wall clock per decode step | 30.8M / 618 ms | 26.7M / 536 ms (1.87 token/s) |
| prefill's k / v per call | 152k | 23.7k |
| Device cycles / wall clock per prefill chunk (20 tokens) | 68.1M / 6.5 s | 65.0M / 6.3 s (3.18 token/s) |

  decode's large linear layers are now all near their weight-read floors. About 19% remains to the hand-written path (about 450 ms/step): the hand-written path merges q / k / v and gate / up into one linear layer each (avoiding repeated activation quantization and each linear layer's fixed overhead); the v projection takes 29k per call (twice its floor, about 0.9M per step); of gate's 177k, about 58k is the SFU's SiLU.
- **Direction of the fix** (later, item 2): let the linear micro-kernel accept any elementwise tail (like the generic contraction, handing the accumulated results per chunk to `generic`'s tail lowering; outputs fp32 / int32 / int8, elementwise inputs fp32 / int32), storing the int32 accumulation directly when there is no tail. This is exactly what item 2 (micro-kernels expanded at the `sahl` level) does: once scheduling and tails are expressed separately, the linear-layer schedule can wrap any tail.
  **The last piece, int8 results (2026-10-12, `14f5612`)**: decode's linear layers only accepted fp32 results (int8 only for prefill's row form), so the v projection quantized straight into the KV cache fell to the generic contraction (weights through SPAD_B and EX, no GEMV): 136k cycles per layer on Qwen3, versus 17.6k for the k projection of the same shape. Now the decode form also accepts int8 results (placed in SPAD_A; LOOP_END has only two strides and the int8 DDR stride is a third, so the chunks of such linear layers are written out unrolled). Qwen3 decode on the board 211.1 → 178.4 ms/step (5.60 tok/s), bit-exact.
  **Attention over the actual length (2026-10-12, `1860823`)**: the generic frontend used to compute attention over the whole cache (T = 256), masked by position; now, like the hand-written path, it carries `valid[T]` (dynamic T) and uses only the first T rows of the cache (`sa-llm-run --abi=hfv --pad=D`). torch-mlir takes a min for the upper bound of the cache slice, which IREE cannot prove equals T, so each layer keeps a runtime assertion that cuts the command buffer; `compile_sa.sh` adds `--iree-opt-strip-assertions`. Qwen3 on the board decodes at 145.4 ms/step (6.88 tok/s), prefill 773 ms per chunk, logits unchanged bit for bit.

**Remaining**: the generic path's decode linear layers on the micro-kernel (above); the two host dispatches of Q's RoPE in prefill (IREE materializes the swapped q in a head-outer layout, so the gather carries a transpose; changing how the frontend writes the swap instead put 31 on the host, so it was not adopted); host fallback with native ARM code (`llvm-cpu` instead of the VMVX interpreter, needing a compiler rebuild).

### 8.20 Micro-kernels expanded at the `sahl` level (result of §8.17 item 2, 2026-10-04)

**Results**: a new pass `sahl-expand-kernels` (after `sahl-schedule`, before `sahl-to-sahw`) writes the scheduling of the three kinds of micro-kernels — linear layers (the decode form and the multi-row prefill form), attention (decode's scores / PV, dynamic T), and the generic contraction (static, dynamic K or N) — as explicit `sahl` operations; the kernel lowering in `sahl-to-sahw` (`SahlLowerKernels.cpp`, 747 lines) is **deleted**, and any remaining `sahl.kernel` raises an error (that dispatch goes to the host). The descriptors of the golden corpus's 4002 dispatches are **all identical** (384 differ only in the numbering of BASE / PARAM registers, see below); after deleting the old lowering the corpus is still all identical, i.e. no kernel is left for it (also countable with `SA_EXPAND_STATS=1`). A pure restructuring, no board run needed.

**New `sahl` operations and attributes** (enough to express all of the old lowering's scheduling):

| | Meaning |
|---|---|
| `sahl.strip` | DDR rows → an interleaved LD into an A strip; or a local vector → a VE DIV replication into a strip |
| `sahl.mma {accumulate}` | One EX (strip × a B block in SPAD_B → ACC); KT / BSTEP / REPEAT come from the operand shapes, dynamic dimensions become dynamic fields |
| `sahl.bcast` | Per-row / scalar values → broadcast layout (one word per value) |
| `sahl.transpose` | Rows in SPAD_A → a B block in SPAD_B (TRANSPOSE) |
| `sahl.loop {count, strides}` | A LOOP_END loop; loads / stores inside the loop name the stride they advance per round with `sa.advance` |
| `sahl.claim_spad {words}` | `[0, words)` of SPAD_A belongs to the fixed-position buffers that follow (the strip at 0, x or the rows to transpose in bank 1) |
| `sahl.cursor_reset` | Resets the row cursor (a DDR offset accumulated on the device, SETREG) to zero; loads / stores with `sa.cursor` add it to the address and advance by one row's bytes after `sa.cursor_step` — for rows of dynamic length and dynamic pitch (prefill attention's p rows and scores rows) |
| `sahl.scope {keep, spad_from}` | The lifetime of local buffers (chunk / row); `keep` keeps the outer caches, `spad_from` sets the start of the SPAD temporary area |
| `sa.word` / `sa.bank` | Fixed buffer positions: word addresses in SPAD_A / ACC, the bank in SPAD_B |
| `sa.accumulator` | IREE's i64 accumulator (actually int32 words, truncated by the tail) |
| `sa.prefix`, `sa.rows`, `sa.interleave`, `sa.prefer_bank` | Weights placed into the template prefix (which can run during the previous dispatch), one DMA row per row, K-first interleave, the preferred bank of tail inputs |

Dynamic sizes (T) are expressed with `?` dimensions of `memref`s, with sizes as SSA values (the sizes of the cache subview; the block count `T >> log2 D` with `arith.shrui`); the lowering puts the upper bound (`max_dynamic`) into static fields and the `Lin` (a linear function of push constants) into dynamic fields.

**How it is guaranteed to be a pure restructuring**: after each expansion step the golden corpus runs, compared byte for byte with the old lowering. The expanded IR creates registers in a different order than the old lowering (BASEs and setup PARAMs numbered by first use), so `golden.py` compares a normalized disassembly: BASEs and setup PARAMs are named by their setup expressions (binding + offset, linear function of push constants), not by number; prefix register fields are ignored when there is no prefix. Private PARAMs (from 7 downward) follow creation order without normalization — which is why the row cursor's position also matches the old lowering exactly.

**Steps** (each committed once the golden corpus is identical):
- E1 prefill's multi-row linear layers: one interleaved-LD strip per row block, B blocks per chunk, an EX per row block, the tail, storing D rows.
- E2 decode's linear layers: the chunk loop (`sahl.loop` + `sa.advance`), alternating SPAD_B banks, chunk 0's weights in the prefix (`sa.prefix`), fixed ACC positions (`sa.word`).
- E3a the generic contraction (static): three matrix layouts (packed as is, K rows → transpose, N rows → interleave), K-block accumulation, the tail's four kinds of inputs (elementwise, per column, per row, scalar). The "dimension → loop" mapping of DDR operands is derived from their strides and each loop's element offset (views can be transposed relative to the contraction's operands). When x is in ACC, the K blocks depend on its address: obtained by replaying the local memory allocator (only when the kernel is the first command-issuing operation of the dispatch; this coupling is removed by item 3's memory planning).
- E4 attention (decode, dynamic T): K rows loaded with `sa.rows` and transposed into `T / D` blocks, V rows loaded interleaved, q / p strips, EX, per-head tails; the H rows of scores stored contiguously.
- E4a the generic contraction with dynamic K / N (prefill attention's `batch_matmul`: N = T for `q · K^T`, K = T for `p · V`; also decode's attention with `ukernels=none`): one K block, one chunk; the rows of p and of the scores have dynamic pitches (dimensions of size 1 may follow the dynamic one), walked row by row with the row cursor.
- E5 delete `SahlLowerKernels.cpp` and the chunk path in `generic` used only by it; the LOW checks of `sahl_kernels.mlir` now go through expansion.

**Verification**: the golden corpus (4002 dispatches); lit: `sahl_expand_kernels.mlir` (E1), the EXP / EXPGEN checks of `sahl_kernels.mlir` (E2 / E3a), `sahl_expand_dynamic.mlir` (E4a: cursors and dynamic fields of scores / PV), 14 lit tests in total; SmolLM2 generic path (prefill M = 8 + decode): 794 executables on the accelerator, no kernel left after expansion, `dispatch_check --dirty` all consistent, prefill + decode bit-exact with decode only (5/5 rows), teacher-forced 34/40.

**Code size**: `SahlExpandKernels.cpp` 989 lines (expansion), `SahlToSahw.cpp` 892 lines (one-to-one translation), `SahlLowerGeneric.cpp` 771 lines.

**Next**: scheduling is now in the IR, visible and changeable: (1) the direction of §8.19 — wrapping the linear-layer schedule around any tail (changes the output, accepted the way item 3 is); (2) item 3's lifetime-based memory planning: after expansion all local buffers are `memref.alloc`s, and `sa.word` / `sa.bank` / `sahl.scope` are decided by the planning rather than copying the old lowering's addresses.

### 8.21 Z1 freeze (2026-10-04)

After items 1 and 2 of §8.17 were done, the PYNQ-Z1 version was frozen at tag `v1.0-pynq-z1` (branch `pynq-z1`, a GitHub Release with the bitstream, firmware, runtime and the full board results); all later optimization happens on the KV260 (`kv260_upgrade_plan.md`, with item 3 moved into K2b there).

**KV260 K1** (2026-10-08): compiler unchanged; the same regression with the aarch64 runtime and the KV260 overlay (`deploy_z1_freeze.sh --board kv260`), and on the board the token sequences of 5 tests in both K1a (50 MHz) and K1b (100 MHz) are identical to the Z1 baseline (`compiler/tests/compare_z1_baselines.py`); at 100 MHz stories15M 35.44 token/s, SmolLM2 qhf 4.43, generic path 3.73, about twice Z1. From then on main is the KV260 version.

**Board regression before the freeze** (test bundles produced by `compiler/scripts/deploy_z1_freeze.sh`, run on the board with `compiler/tests/board_regress.py`, L2 bitstream, D = 8, 50 MHz): all bit-exact.

| Test | On the board |
|---|---|
| bwtest | LD / ST 397 MB/s (99% of the 64-bit × 50 MHz limit), LD and ST concurrently 788 MB/s |
| C3 stories15M decode | 76/76 steps bit-exact with DeviceModel (the hand-written L5 path), 17.60 token/s |
| C5.5 SmolLM2 decode (qhf) | 38/38 steps, 450 ms/step (2.22 token/s) |
| C6.P stories15M prefill + decode | 61/61 steps; prefill 75.6 token/s, decode 17.39 token/s |
| C6.P SmolLM2 prefill + decode (qhf) | 25/25 steps; prefill 7.13 token/s, decode 460 ms/step |
| Generic path SmolLM2 prefill + decode | 17/17 steps; prefill 3.18 token/s, decode 535.5 ms/step (1.87 token/s) |

**Baselines** (`tests/baselines/pynq-z1/`, `compiler/tests/make_z1_baselines.py`): the descriptor hashes of the golden corpus re-recorded with the compiler at the freeze (4002 dispatches), per-dispatch checks of five models (`--dirty`, all consistent; the two host dispatches in the generic path's prefill are UNSUPPORTED), board token sequences, speeds and bandwidth (`perf.md`), time breakdowns (`z1_profile.md`). The KV260's K1a (the same accelerator) uses them as its reference.

**Time breakdown**: the runtime gains `SA_PROFILE_PERF=<file>`: in profile mode each list carries `RT_F_PERF`, rt_fw writes the accelerator's 32 event counters (`PC_*` in `rtl/sysarray/sa_defs.vh`) into the perf area of the BRAM, and the runtime reads them out, accumulates them per export and writes a CSV; `compiler/tests/z1_profile.py` splits them by dispatch type into reads (and their floor at bwtest's 7.94 B/cycle), EX, VE / SFU, ST, all engines idle (fixed overhead) and overlap between engines. Per decode step:

| Path | Device cycles | Read floor | Not hidden by reads | Fixed overhead |
|---|---|---|---|---|
| stories15M hand-written | 2.71M | 1.99M | 0.72M (26%) | 2% |
| SmolLM2 hand-written qhf | 22.22M | 17.44M | 4.78M (22%) | 1% |
| SmolLM2 generic | 26.67M | 17.75M | 8.92M (33%) | 1% |

Linear layers sit at the read floor (EX and LD overlap almost completely); fixed overhead is small; what is not hidden is VE / SFU work (linear-layer tails, RMSNorm / softmax reductions, attention) — of the generic path's 8.92M, linear-layer tails about 4.0M, attention 2.4M, reductions 2.6M. In prefill VE / SFU take 52–54% of each chunk. Once bandwidth goes up (the KV260's K2) this part becomes the main cost, the basis of the performance work there.

### 8.22 C6.5 revised: attention tiled by T without online softmax (design 2026-10-13, implemented 2026-10-14)

**Why it is needed.** The compiler plans the SPAD for dynamic lengths up to `--iree-sa-max-dynamic` (default 256; the export now sets it from `--max-len`, generality plan §9.3). The decode attention contractions are the limit: `SahlExpandKernels.cpp:665` requires the whole K block of QKᵀ, T × head_dim bytes, in one SPAD_B bank (4096 words of 16 bytes at D = 16), and line 666 / `nk > 1 && dyn` forbid splitting a dynamic K (PV, whose K is T). Hence T ≤ 65536 / head_dim: Gemma-3 (256) 256, Qwen3 (128) 512, Llama-3.2 / SmolLM2 (64) 1024. Checked with 2-layer models at `--max-len 512` (generality plan §12.4): Qwen3 all on the accelerator, Gemma's decode QKᵀ and PV on the host. Prefill attention (row by row) is limited separately by `spadBytes / d ≥ 2 · maxDynamic` (T ≤ 2048 at D = 16) and is not part of this step.

**Why no online softmax.** Decode attention after qsum (generality plan §12.4) is: QKᵀ int8 × int8 → int32 scores → fp32 by a scale; max over T; p = i8(exp(s − max) · 127); Σp an integer sum; PV int8 × int8 accumulated in int32 over T; division by Σp. The only fp32 reduction over T is the max, which is order independent; everything else accumulates integers, which §3.2 allows to be tiled freely. Tiling T therefore keeps the results bit-identical to the untiled form, with no change to the numeric contract or to the oracle. Online softmax (block-local max, rescaling) would instead change p's int8 rounding and break bit-exactness with the existing semantics.

**What the hardware allows (checked in the RTL and funcsim).**
- A command with a zero length is rejected with the sticky `XERR_SHAPE` (`sa_sched.v`: LD / ST rows or row bytes 0, GEMV rows 0, EX Kt 0, VE groups 0; `sa_cmdfetch.v`: `LOOP_END` count 0); funcsim does the same (`sa_funcsim.py`, `test_funcsim.py` "LOOP_END count 0"). Unrolling to the maximum number of blocks and giving the unused ones LEN = 0 would need an RTL change and a new bitstream: rejected.
- `LOOP_END`'s count is a patchable dynamic field (slot 1, 16 bits, from a PARAM; `pynq_matmul.py` `LOOP_END: {"count": 1}`), and each iteration adds strides to two PARAMs. A loop whose trip count is set at dispatch time therefore needs no hardware change. With nb = ⌈T / Tb⌉ and T ≥ 1, the count is never 0.

**Design.**
1. **Block count on the host.** The loader computes PARAMs as `((push constant × mul) >> shift) + add` (`compiler/runtime/sa/sa_loader.c`), which can divide by a power of two but cannot round up (the add comes after the shift). Add a pre-shift addend to the setup entry so that nb = (T + Tb − 1) >> log2(Tb); `Lin` in the compiler gains the same term. Tb is a power of two chosen so that Tb × head_dim fits one SPAD_B bank (Gemma Tb = 256).
2. **`sahl.loop` with a dynamic count.** Today `sahl.loop` / `sahw.loop` take a static `I64Attr` count (decode's linear-layer chunk-pair loop). Allow the count to come from a `Lin`, lowered to the `LOOP_END` count's dynamic slot; the DDR views advance by PARAM strides as they already do.
3. **Contraction expansion over dynamic T** (`SahlExpandKernels`): QKᵀ as a loop over T blocks, each loading its K strip (Tb × head_dim) and writing its scores to the block's slice of the score buffer (heads × nb·Tb × 4 bytes; Gemma 8 KB at T = 512); PV as a loop over the same blocks, accumulating int32 in ACC (first block writes, later blocks accumulate, as the GEMV K split of WP6 does with static K).
4. **The last, partial block.** Every block is computed at full size Tb; positions beyond T read KV cache rows that exist but are unused (the cache is allocated at max_len, so the reads stay in bounds). The softmax VEs keep their dynamic LEN = T, and p beyond T must be zero before PV (a fill of the p buffer's tail, or VE VALID masking), so those rows contribute nothing. At most Tb − 1 positions of waste, in the last block only.
5. **Tests.** Synthetic attention at block boundaries (T = Tb − 1, Tb, Tb + 1, 2·Tb) per dispatch against the oracle; the golden corpus (T ≤ 256) byte-identical; a 2-layer Gemma at `--max-len 512` with more than 256 positions on the sim (teacher forcing over a 512-token window); then the original acceptance, per dispatch at T = 2048 and bit-exact on the board.

**Out of scope.** Gemma-3's sliding window beyond 512 tokens (a window mask on the local layers, or a ring-buffered KV cache) is a separate frontend item (`hf_generic.py` raises for `max_len` above the window). Prefill rows longer than 2048 would need the same tiling of the row-mode VEs.

**Effect.** Gemma's limit goes from 256 to 512 (to the sliding window), and Qwen3 / Llama / SmolLM2 are then bound by prefill's 2048 and by device memory (Llama about 2200 positions, generality plan §9.3). Compiler and runtime only, no new bitstream; the K3 URAM evaluation of generality plan §16 waits until this is done.

**Implementation (2026-10-14).** Done in the compiler only (no runtime, format or bitstream change); it differs from the design above in four places:
- **Block count without a ceiling.** Instead of a pre-shift addend in the loader, the loop count is `(T >> log2 Tb) + 1`, which the existing `Lin` (mul, shift, add) expresses. When T is a multiple of Tb this computes one block past T; its scores are never stored and its K reads stay inside the cache allocation (max_len rows), so it costs one block of work and nothing else. Tb = nb · D rows with nb = bit_floor(SPAD_B bank words / head_dim words): Gemma Tb = 256, Qwen3 Tb = 512.
- **`sahl.loop` / `sahw.loop` with a dynamic count** (`Optional<Index>:$dyn_count`, `DYN_LOOP_COUNT` = 1 in `DescList.h`, serialized into `LOOP_END`'s count slot from a PARAM). Sequential loops share their PARAM registers (`loopPool` in `SahlToSahw.cpp`): one fresh pair per loop overflowed the 8 PARAMs on Qwen3.
- **QKᵀ (dynamic N = T over the K rows)** in `SahlExpandKernels`: per batch b (in a scope that releases ACC afterwards) the strips of all G query rows go to SPAD_A bank 0; each iteration loads Tb cache rows (`sa.rows`, DDR stride Tb × row pitch, `sa.advance`), transposes them into SPAD_B bank 0, runs G EXs into a scratch ACC area of (G − 1 + D) · Tb elements (an EX writes D output rows, so a block result cannot go straight into a per-row buffer without clobbering its neighbours), then G VE copies (`memref.copy` local → local, integer) move each row's block into its row buffer at a destination that advances by nb words per iteration (`DYN_VE_DST`). After the loop each row runs the existing epilogue and cursor store (`finish`).
- **PV (dynamic K = T)** keeps the untiled form, now allowing several N chunks of the output; with G > 1 a `sahl.cursor_reset` with `sa.skip` = b · G repositions the cursor per chunk. The int32 accumulation over T was already within one bank at these lengths.

Everything is integer except the fp32 max over T, so tiled results are bit-identical to the untiled oracle, as argued above.

**Tests.** `compiler/tests/test_long_context.py` (run_tests.sh `longctx` / `longctxq`): a 2-layer model exported at `--max-len`, compiled with the bound from `compile_flags.txt`, 0 executables on the host required, then `dispatch_check.py` at T around the block boundaries (T is a multiple of D, the runtime pads it). Golden corpus cases `gemma_t512` and `qwen3_t1024` added; the 7046 earlier dispatches byte-identical. Fixed on the way: `dispatch_check.py` filled int8 inputs with zeros unless `--full-i8` (a regression since 6511ab0, 2026-10-10, which had hidden int8 path errors from the per-dispatch checks); inputs are now in [−127, 127] by default, [−128, 127] with `--full-i8` (where PV shows known differences from saturating −128 · −128, not a compiler bug). `test_c5.py --configs` passes again on all configurations with real data.

**Results (sim).**

| Model (2 layers) | `--max-dynamic` | executables (accelerator / host) | per-dispatch check |
|---|---|---|---|
| Gemma-3-270M | 512 | 214 / 0 (before: 4 decode attention dispatches on the host) | OK at T = 16, 240, 256, 272, 304, 496, 512 |
| Qwen3-0.6B | 1024 | 202 / 0 | OK at T = 16, 496, 512, 528, 1024 |

Gemma end to end with a 320-token prompt (past the old 256 limit, `--max-len 512`, prefill 16): 324 steps against the torch W8A8 model, min corr 0.9979, 0 dispatches on the host; teacher forcing over 40 positions device top-1 37/40 = torch W8A8 37/40, device vs torch argmax differs at 0; prefill vs decode only 5/5 rows bit-exact.

**Limits after C6.5.** Gemma 512 (its sliding window, out of scope above); Qwen3 1024. At 2048 Qwen3's decode QKᵀ fits, but 17 dispatches went to the host: the attention probabilities over T (`reduction_16xD`: the masked max over T, then p = i8(exp(s − max) · 127), one decode dispatch and one per prefill row) hold [16, max T] fp32 buffers and their temps in ACC, and `sahl-tile` skipped dynamic shapes ("ACC full").

**C6.5b (2026-10-14): row pieces over dynamic-length rows.**
- `sahl-tile` accepts [R, ?] (a dynamic innermost length as the whole row; T estimated at max_dynamic): the pieces are scopes over rows [r0, r0 + n), with local [n, ?] buffers and DDR views `subview %b[r0, 0] [n, dim(%b, 1)]` of the [R, ?] bindings.
- The lowering (`ddrOf`) reads such a view as the binding's dynamic rows with a static row skip. A [H, ?] binding was already moved row by row (the row offset PARAM accumulating T · element bytes with `SETREG +=`, each DMA with two dynamic fields, address and row bytes); a piece starts by adding the bytes per row r0 times. No new PARAMs, no shared cursor between the input (fp32, 4T bytes per row) and the output (int8, T bytes per row): each DMA sequence resets the offset first.
- The estimate at max_dynamic is far above what the lowering takes (the untiled [16, 1024] form compiles), so dynamic rows are split only on `SaCodegen`'s retry after ACC full, then for ACC as estimated (no shrink on that first retry). Every dispatch that compiled before is byte-identical (golden corpus).
- Tests: lit `sahl_tile_rows_dyn.mlir` (the pieces, untouched without the retry, the lowered row skips); `run_tests.sh longctxq2k` (Qwen3 2 layers at `--max-len 2048`: 202 executables on the accelerator, 0 on the host; per-dispatch checks OK at T = 16, 1024, 1040, 2032, 2048); golden case `qwen3_t2048`.
- End to end on the sim (2026-10-14): Qwen3 2 layers at `--max-len 2048`, a 1104-token WikiText prompt (past the C6.5 limit of 1024), prefill 16: 1106 decode steps against the torch W8A8 model, min corr 0.9981, argmax equal in 1044/1106, 0 dispatches on the host; teacher forcing over 40 positions device top-1 39/40 (torch W8A8 37/40), device vs torch argmax differs at 2; prefill (69 chunks) vs decode only 3/3 rows bit-exact.

**Board check (2026-10-14; the 2-layer dry run, then the whole Qwen3 at 2048 and Gemma at 512 on the board).** The whole model has no sim reference at these lengths (the 2-layer, 1104-token run above takes hours), so the device is checked against itself and against fp32:
- `compiler/scripts/deploy_longctx.sh qwen3|gemma` (hours; `systemd-run -p MemoryMax=10G`; `LC_LAYERS=2`: a dry run of the flow with a truncated model, ~40 min): the whole model at `--max-len` 2048 (Qwen3) / 512 (Gemma), the usual 40-token `board_llm.py` bundle bit-exact with the sim, plus two teacher-forcing windows of the context length with the fp32 reference (`board_quality.py prep`) → `build/deploy_hfgen_<model>_lc<len>` (its own directory: such a bundle is slower at short contexts, below).
- `board_longctx.py` (`board_run.py longctx`): a prompt of the window's length minus G (default 16) run as prefill + G decode steps and as decode only; the rows from the prompt's last position on must be bit-exact between the two (prefill's [M, T] attention against decode's T loop; the sim checks the same equality), the generated tokens identical; the decode-only rows are scored as teacher forcing by position range ([0, 256), [256, 512), [512, 1024), [1024, 2048)) against fp32. It also gives the speeds at long T (prefill tokens/s, decode ms per step at T ≈ 2k). `--sweep 64,256,512,1024,2048` measures instead, for each T, S decode steps (`--steps`, default 8) after a prefill to T − S, with `SA_PROFILE=batch` (wall and device ms per decode step, device ms per prefill chunk) → `longctx_sweep_board.json`: the board baseline of steps a–e below, and, run at the same short T on a bundle compiled for 256 and one compiled for 2048, the static block size's cost at short contexts.
- `board_teacher.py` streams the logits through a FIFO (`run_streamed`) and scores them row block by row block: a 2048-token window's logits (Qwen3: 1.2 GB) are never held whole; `board_quality.py sim` uses the same path.
- Dry run of the whole flow (2026-10-14, `LC_LAYERS=2`): Qwen3 at 2048 (202 executables on the accelerator, `--valid_len=2048` in `sa_args.txt`, `model.json` context 2048, qemu 3/3, two 2048-token windows) and Gemma at 512 (215 executables, prefill_all 40/40, window 640) staged without errors; `--sweep` checked on the sim. Device memory of the whole models (`mem_preflight.py`): Qwen3 at 2048 824 MiB, Gemma at 512 297 MiB of the 1280 MiB pool.
- **On the board, 2-layer Qwen3 at 2048 (2026-10-14, `build/deploy_hfgen_qwen3_lc2048_l2`, four-port overlay; the first hardware run past 256 positions)**: `board_llm.py` 17/17 bit-exact with the sim (40-token prompt); `board_longctx.py` with a 2032-token prompt: prefill (127 chunks) vs decode only (2048 steps) 17/17 rows bit-exact from position 2031, generated tokens identical, 0 dispatches on the host in both runs; teacher forcing over the prompt top-1 vs fp32 87.2% (86.2–88.3% in every position range up to 2031, the 2-layer model's PPL on a par with its fp32: 858k against 868k). `--sweep` (device ms, 2 layers):

  | T | decode per step | prefill per 16-token chunk |
  |---|---|---|
  | 64 | 39.6 | 131.1 |
  | 256 | 41.1 | 135.8 |
  | 512 | 46.9 | 153.0 |
  | 1024 | 54.7 | 215.3 |
  | 2048 | 70.2 | 339.8 |

  Decode grows by 15.3 ms per layer from T = 64 to 2048; the cycle model above gives 13.7 ms (`attn_cycles.py`, the table interpolated at 64), so it reads about 11% low at long T (T = 512: 3.65 against 3.27 ms per layer). Extrapolated to the whole Qwen3 (28 layers): decode about 0.6 s per token at T = 2048 (1.7 tok/s), prefill about 3.7 s per chunk near 2048 (104 ms per layer per chunk above T = 64).
- **On the board, the whole models (2026-10-14, four-port overlay, `build/deploy_hfgen_qwen3_lc2048` window 809 MB, `build/deploy_hfgen_gemma_lc512` window 640 MB)**. The builds: Qwen3 1138 executables, Gemma 791, all on the accelerator; sim 40-token checks PASS (Qwen3 decode 9/9 argmax vs torch W8A8; Gemma 7/9, within the quantized-model teacher-forcing gate), prefill vs decode only 5/5, Gemma prefill_all 40/40, qemu 3/3. The compile of Qwen3 at 2048 peaks at 7.5 GB and was OOM-killed next to the exporting Python process under the 10 GB cap: `deploy_longctx.sh` now runs the export (`test_hf_generic.py --export-only`), the compile and the checks as separate processes.
  - `board_llm.py`: 17/17 bit-exact with the sim, both models.
  - `board_longctx.py`: Qwen3, a 2032-token prompt: prefill (127 chunks, 590.6 s, 3.44 tok/s) vs decode only (2048 steps, 384.8 ms per step on average) 17/17 rows bit-exact from position 2031, generated tokens identical, 0 dispatches on the host, device memory peak 694 of 809 MB. Gemma, a 496-token prompt: 17/17 bit-exact, prefill 28.98 tok/s, decode only 85.3 ms per step on average, peak 272 of 640 MB.
  - Teacher forcing over the prompt (decode-only rows against fp32):

    | Positions | Qwen3 PPL (fp32) | top-1 vs fp32 | Gemma PPL (fp32) | top-1 vs fp32 |
    |---|---|---|---|---|
    | [0, 256) | 29.84 (29.63) | 86.7% | 62.93 (53.03) | 80.1% |
    | [256, 512) | 12.96 (12.93) | 90.6% | 18.89 (17.64) | 87.0% |
    | [512, 1024) | 8.37 (8.43) | 93.4% | | |
    | [1024, 2031) | 16.76 (15.66) | 86.8% | | |
    | all | 14.65 (14.17) | 88.9% | 35.20 (31.17) | 83.5% |

    Qwen3 holds fp32's PPL up to 1024 positions; past 1024 the gap grows to +7% (top-1 86.8%). Not yet attributed; a candidate is the int8 attention probabilities (qsum, i8(e · 127)): with the mass spread over more positions, more of them round to 0.
  - `board_teacher.py`, two windows of the context length (WikiText-2 test from token 0): Qwen3 through decode (no prefill_all in this module; 384.8 ms per step) PPL **18.990 against fp32 18.767** (+1.2%); Gemma through prefill_all (20 tok/s) PPL **43.59 against fp32 43.98**. Scored (`board_quality.py score`, `build/quality/{qwen3_lc2048,gemma_lc512}_board.json`): Qwen3 top-1 vs fp32 86.8%, KL(fp32 || device) over the top 64 0.073; Gemma 76.9%, 0.213. Against the same models' 256-token windows (generality plan §12.4, qsum: Qwen3 88.1% / 0.038, Gemma 80.3% / 0.157) KL is 1.9× / 1.35× higher: the loss grows with the context, consistent with the per-range table above (Qwen3 past 1024). Gemma's PPL below fp32 is noise over 1024 positions with a 262k vocabulary; its KL is the measure.
  - `--sweep` (wall / device ms per decode step, device ms per 16-token prefill chunk):

    | T | Qwen3 decode | Qwen3 prefill chunk | Gemma decode | Gemma prefill chunk |
    |---|---|---|---|---|
    | 64 | 197.0 / 189.9 | 1623 | 72.4 / 66.0 | 317 |
    | 128 | | | 75.4 / 67.3 | 322 |
    | 256 | 218.2 / 211.1 | 1787 | 86.8 / 75.6 | 347 |
    | 512 | 298.9 / 291.8 | 2039 | 104.3 / 86.6 | 435 |
    | 1024 | 408.0 / 400.7 | 2917 | | |
    | 2048 | 626.1 / 618.6 | 4663 | | |

    Qwen3's decode grows by 15.3 ms per layer from T = 64 to 2048, exactly the 2-layer dry run's figure, so the 2-layer extrapolation holds (decode 1.60 tok/s at T = 2048); prefill grows by 109 ms per layer per chunk (2-layer: 104). At short T this bundle pays for its static block size: Qwen3 decode at T = 64 197 ms against 145 ms for the bundle compiled for 256 (`deploy_hfgen_qwen3_qs`, 6.88 tok/s), +36% (the cycle model above: +35 ms per token at T = 16); Gemma 72.4 against 66.7 ms (+9%). This is the baseline for steps a–e below.
- Checked on the sim with the 2-layer Qwen3 at 2048 (48-token prompt, 4 generated): prefill vs decode only 5/5 rows bit-exact, 0 dispatches on the host; `board_longctx.py`'s scores equal `board_quality.py sim`'s for the same positions.

**Limits now.** Qwen3 (and by the same rule SmolLM2 / Llama, head_dim 64) compile for 2048 with everything on the accelerator; Gemma stays at 512 (sliding window). The target check `spadBytes / d ≥ 2 · max_dynamic` allows max_dynamic up to 4096 at D = 16. **Checked at 4096 (2026-10-14, 2-layer Qwen3, `test_long_context.py --max-len 4096`, report in `build/hfgen/qwen3_l2_t4096/compile_report.txt`)**: 200 executables on the accelerator, 2 on the host, the decode p · V of both layers (`batch_matmul_8x2x128xD`: `SahlExpandKernels.cpp:673`, the dynamic contraction length above kmax = 4080 needs a K split); prefill (row attention) and everything else pass, the other 200 OK in the per-dispatch checks at T = 16, 2064, 4096. So 4096 needs only the p · V dynamic K split. Device memory bounds Llama at about 2200 positions (generality plan §9.3); Qwen3's whole model at 4096 fits (`mem_preflight.py`: T ≤ 6213). Not yet run on the board.

**Performance at long T and double buffering of the T loop (analysis 2026-10-14; not implemented).** The method (`compiler/tests/attn_cycles.py`, 2026-10-14; first done with scratch scripts): the decode dispatches over T of the 2-layer Qwen3 (`build/hfgen/qwen3_l2_t2048lc`) are run per dispatch on the functional simulator (`dispatch_check.py`, so each run is also checked against the oracle, `sim.trace` capturing the commands as executed), and timed with `perf_model.py`'s `run_list` (four-port overlay, the `qwen3_dyn_cal.json` calibration); `--max-dynamic` recompiles the same dispatches for other bounds. The figures are model cycles per layer (8 KV heads, 100 MHz), not board measurements.

| Dispatch (per layer) | T = 2048 total | LD busy | EX busy | VE busy |
|---|---|---|---|---|
| q · Kᵀ (the C6.5 T loop) | 629k | 168k | 404k | 48k |
| p · V | 788k | 187k | 266k | 333k |
| attention probabilities (qsum) | 100k | 9k | 0 | 90k |

At T = 2048 attention alone is about 15 ms per layer, about 430 ms per token for Qwen3's 28 layers (decode about 2 tok/s, against 6.88 at short T on the board).

By T, the four dispatches together (q · Kᵀ, the masked max / exp, the sum, p · V; per layer and × 28 per token), compiled for 2048 against compiled for 256 / 512 (`attn_cycles.py build/hfgen/qwen3_l2_t2048lc --max-dynamic 256,512`; all 52 runs OK against the oracle):

| T | compiled for 2048: per layer | ms / token | compiled for 512 (256 the same up to 256): per layer | ms / token |
|---|---|---|---|---|
| 16 | 163k | 45.5 | 37k | 10.4 |
| 128 | 211k | 59.1 | 95k | 26.7 |
| 256 | 267k | 74.9 | 162k | 45.5 |
| 512 | 502k | 140.5 | 297k | 83.1 |
| 1024 | 848k | 237.5 | — | — |
| 2048 | 1541k | 431.6 | — | — |

The blocked form costs more at every T it shares with the untiled one, not only at short T: at T = 512 1.69× (q · Kᵀ 254k against 132k, p · V 212k against 128k).

*Findings.*
1. **EX, not LD, bounds the T loop.** Decode has G query rows per KV head (Qwen3 2, Gemma / Llama 4, SmolLM2 3), and each row is its own EX over a strip of the row replicated over the D rows: 1 of D output rows is used. Even perfect overlap of LD, transpose and EX only brings q · Kᵀ from 629k to its EX busy time, about 404k (−36%).
2. **p · V costs more than q · Kᵀ at long T**, mostly VE (p copied over the D rows, per chunk of output tiles), which the T loop does not touch.
3. **Regression at short T: the block size is static.** Every block is computed at full Tb (Qwen3 512 rows: EX repeat nb = 32 tiles), however short T is; the untiled form's EX repeat is dynamic. Per layer, the same q · Kᵀ dispatch compiled for max_dynamic 256 / 2048: T = 16 16k / 128k cycles, T = 128 42k / 129k; p · V at T = 128 38k / 68k (several N chunks). A Qwen3 compiled with `--max-len` 2048 would decode at short contexts about 33–35 ms (+23%) slower per token (table above: 45.5 against 10.4 ms of attention at T = 16). The board bundles (compiled for 256) are not affected: the tiled form applies only when T × head_dim exceeds one SPAD_B bank (Qwen3 above 512, Gemma above 256).

*On-chip memory for double buffering.* The scoreboard tracks hazards per bank (`double_buffer_design.md` §7: two buffers on disjoint addresses of one bank still wait for each other; EX and VE also conflict on any common bank of SPAD_B / ACC), so double-buffered blocks need different banks, not just room. In the T loop:

| Memory | Use per block now | Double buffering |
|---|---|---|
| SPAD_A bank 0 (4096 words) | the strips of x, G · K words (Qwen3 256, Gemma 1024) | — |
| SPAD_A bank 1 | the raw block of cache rows (Tb × K bytes: the whole bank) | a second raw block has no free bank: in bank 0 the LD waits for the EXs reading the strips |
| SPAD_B bank 0 | the transposed tiles (the whole bank) | — |
| SPAD_B bank 1 | free | the tiles' second buffer fits |
| ACC (4096 words) | scratch (G − 1 + D) · Tb and the rows' score buffers: a few hundred words | ample |

So the tiles can alternate between SPAD_B banks 0 and 1 at no capacity cost; the raw block cannot be double-buffered within the four SPAD banks (5 roles: strips, raw × 2, tiles × 2). A smaller Tb does not change that (it is the bank count, not the size). The LD cannot transpose (INTERLEAVE places D-byte chunks of a row, `double_buffer_design.md` §5.3), so the VE transpose and its raw block stay.

*Plan (in this order; step a done 2026-10-14, step d replaced by p · V on the GEMV unit 2026-10-15, both below the table):*

| Step | Content | Expected (model) | Estimate |
|---|---|---|---|
| a (**done**) | **Smaller Tb** (e.g. 128 rows): the waste is at most one block, the block past T of a multiple of Tb shrinks too; the extra commands per block (one LD, one transpose, the EXs and copies, `LOOP_END`) cost little | removes the short-T regression | ½ day |
| b | **The G query rows in one strip** (rows g in lanes 0..G − 1 of the A strip), one EX per block for all rows (its output rows 0..G − 1), instead of one EX per row; the scratch slots go away | q · Kᵀ EX time ÷ G | 1–1.5 days |
| c | **Double buffering of the T loop**: tiles alternating SPAD_B banks 0 / 1, the loop body reordered (software pipelining, in-order dispatch: block i + 1's LD and transpose issued before block i's EXs / copies, the copies of block i not ahead of the next transpose on the VE queue) | after b, EX and LD are comparable: q · Kᵀ about 629k → 210k at T = 2048 | 1 day |
| d (**replaced**) | ~~**p · V**: the G rows of p in one strip as in b, no VE copies of p over the D rows~~ p · V on the GEMV unit with a packed V cache (below, the default since 2026-10-15) | p · V 788k → 74k cycles per layer at T = 2048 | done |
| e (optional, larger) | ~~**q · Kᵀ on the GEMV unit**~~ **Corrected 2026-10-15**: the GEMV unit reads each strip as [K, 16] contiguous (16 output columns per k), so q · Kᵀ would need the K cache transposed within blocks of 16 positions, and a decode step writes one byte per channel into it (the DMA needs 8-byte alignment): not practical. **p · V** is the GEMV's shape instead (below: the packed V cache) | — | — |

Each step's acceptance: the golden corpus (byte-identical outside the changed kernels, the changed ones re-recorded after review), `run_tests.sh longctx / longctxq / longctxq2k` per-dispatch checks, the cycle model as above, then the board.

**p · V on the GEMV unit with a packed V cache (feasibility 2026-10-15, the default since the same day; replaces step d).** The GEMV unit (`sa_funcsim.gemv`) computes per strip s (at most 8 a command) ACC[s] (16 int32) = x[K] · W_s[K, 16], W_s the K × 16 bytes at ddr + s · pitch, rb = K · 16 a multiple of 256. With the V cache per head as [hd / 16, T, 16] instead of [T, hd], p · V is exactly that: the 8 groups of 16 channels are the 8 strips, rb = T · 16 (the one dynamic field), the pitch static, x one row of p (T int8). No transpose, no copies of p over the D rows, no EX, the cache streamed through the four ports.
- **Frontend** (`hf_generic.py`; the default, `SA_V_PACKED=0` gives the earlier row-major cache): the V cache buffer [L, kv_dim / 16, max_len + 16, 16] (16 positions more than attended: a slice [:, :T] is then never the whole buffer, otherwise torch.export guards T ≠ max_len on its layout); written per row as a slice at the position taken as an integer (`.item()`, checked in range; a tensor index became an element-wise scatter with 3 i64 indices per byte); p · V as Σₜ (Hk, G, 1, T, 1) · (Hk, 1, hd / 16, T, 16), one contraction x (h, g, t), V (h, c, t, j), output (h, g, c, j) (a matmul broadcast p over the groups on the host and transposed after). The same int8 values: eager logits identical to the row-major cache (30 steps, max |diff| 0).
- **Compiler**: the matcher accepts a packed matrix with a dynamic K whose blocks are at least max_dynamic rows apart; `expandContraction` puts such a contraction (`gemvDyn`: a GEMV target, int8 x; more than 8 groups of 16 channels in commands of 8, Gemma's 16 in two) on the GEMV unit, per row of x one `sahl.gemv` after the row's load at the cursor; `ddrOf` takes an [R, ?, C] view (`dynMid`), `gemv` lowers a dynamic K as the LD's row-bytes field (T · 16); `replaceDynStores` (sa-to-sahl) turns a store at one dynamic offset computed from an i64 in DDR (torch's slice normalization: selects, compares, adds; the position taken as is) into `sahl.scatter` along dimension 1, lowered to LDPARAM + one ST of R rows of 16 bytes a pitch apart. `dispatch_check.py`'s binding probe now skips operations on loaded data (the store offset). `SA_EXPAND_STATS` prints why a kernel was not expanded.
- **Results** (Qwen3 2 layers at `--max-len 2048`, `build/hfgen/qwen3_l2_t2048vp`): 216 executables, all on the accelerator; per-dispatch checks OK at T = 16, 128, 144, 1040, 2032, 2048 (216 each); golden corpus identical (7664, the option off). End to end on the sim (`build/hfgen/qwen3_l2_vp_e2e`): 9 decode steps against torch W8A8 min corr 1.000000, 9/9; teacher forcing 39/40 (torch 37/40); prefill vs decode only 5/5 rows bit-exact (prefill's row attention runs the same p · V executables as decode, IREE sharing them across the module: on the GEMV unit too). Cycle model per layer (`attn_cycles.py`, `build/perfmodel/attn_vpacked.json`), p · V alone and the four attention dispatches per token (28 layers, with step a):

| T | p · V before (EX) | p · V GEMV | attention ms / token before | GEMV |
|---|---|---|---|---|
| 16 | 26.5k | 5.9k | 20.9 | 15.1 |
| 128 | 68.1k | 9.6k | 34.5 | 18.1 |
| 512 | 212.2k | 22.2k | 110.1 | 57.0 |
| 2048 | 788.3k | 74.3k | 412.7 | 212.8 |

  p · V at T = 2048 is 10.6× faster (72k cycles of LD for 4 MB: 58 B/cycle as measured on the board), at T = 16 4.5× (below the untiled form's 12.4k). Attention per token at T = 2048 halves (q · Kᵀ is now 74% of it: steps b, c); at short T it is within 5 ms of a model compiled for 512.
- **Adopted as the default (2026-10-15)**: `hf_generic.py` exports the packed V cache unless `SA_V_PACKED=0` (the `SA_FP32` attribution modes read it in either layout: kv / probs / q identical to the row-major cache); `expandContraction` splits more than 8 groups into GEMV commands of 8 (Gemma head_dim 256: 16 groups, two commands per row of p, each into its 128 columns of the output); `mem_preflight.py` counts the cache's 16 extra positions (KV bytes and the packed length from the K cache alone). Per-dispatch checks (`test_long_context.py`), all on the accelerator, 0 on the host, OK at every T: Gemma at 512 (230 executables, T = 16, 112, 128, 144, 240, 256, 272, 496, 512), Qwen3 at 1024 (216, T = 16 … 1024) and at 2048 (216, T = 16 … 2048), SmolLM2-135M at 2048 (226, Hk 3, G 3, 4 groups; T = 16, 144, 1040, 2048), Llama-3.2-1B at 2048 (226, Hk 8, G 4, 4 groups; same T). Gemma 2 layers end to end on the sim (`--prefill 16 --prefill-all --max-len 512`, `build/hfgen/gemma_l2_e2e_vp`): 9/9 decode steps against torch W8A8 (min corr 0.9991), teacher forcing 37/40 (torch 37/40), prefill vs decode only 5/5 rows bit-exact, prefill_all 40/40. The golden corpus: the earlier sources unchanged by the compiler change (7664 identical), then `gemma_t512` / `qwen3_t1024` / `qwen3_t2048` re-recorded from the packed exports (7708 unique dispatches; the row-major exports kept as `build/hfgen/*_rowmajor` for the cycle data above). Left: the four bundles rebuilt and checked on the board.

**Step a done (2026-10-14).** Two changes to the tiled q · Kᵀ:
- **Block size** `--iree-sa-attn-block` (target config `attn_block`, `TargetConfig::attnBlock`): rows of T per block, default 0 = **128**, at most what one SPAD_B bank holds (Qwen3 was 512, Gemma 256).
- **Exact block count.** The count was `(T >> log2 Tb) + 1`, one block too many whenever T is a multiple of Tb (T = 2048 ran 5 blocks of 512). T is a multiple of D (the runtime pads it, the untiled form already relies on it: its tile count is T >> log2 D), so ceil(T / Tb) = `((T · (M − 1)) >> log2(M · Tb)) + 1` with M = bit_ceil(max_dynamic / D): for T = k · Tb the product is k · Tb − k · Tb / M, just under k · Tb (k ≤ M); for any other multiple of D it stays at or above the next lower multiple of Tb (M · D ≥ T). The existing `Lin` (mul, shift, add) expresses it with no loader change, within the setup entry's 4-bit shift (log2(M · Tb) ≤ 15; else the old form). `arith.muli` marked `sa.index` is now accepted by `sahl-to-sahw` like the count's `addi`.

The choice of Tb (`attn_cycles.py --attn-block 32,64,128,256`, all 200 + 128 runs OK against the oracle; build/perfmodel/attn_block_sweep2.json, attn_block_gemma.json). Qwen3 (2 layers compiled for 2048), the four attention dispatches per token (28 layers), ms:

| T | Tb 512 (before) | Tb 32 | Tb 64 | **Tb 128** | Tb 256 | compiled for 512 (untiled) |
|---|---|---|---|---|---|---|
| 16 | 45.5 | 14.8 | 16.8 | **20.9** | 29.1 | 10.4 |
| 128 | 59.1 | 38.4 | 35.8 | **34.5** | 42.7 | 26.7 |
| 256 | 74.9 | 67.4 | 62.3 | **59.7** | 58.5 | 45.5 |
| 512 | 140.5 | 125.4 | 115.2 | **110.1** | 107.6 | 83.1 |
| 1024 | 237.5 | 241.5 | 221.1 | **211.0** | 205.9 | — |
| 2048 | 431.6 | 473.6 | 433.0 | **412.7** | 402.6 | — |

The q · Kᵀ dispatch alone at T = 16: 128.5k → 40.6k cycles per layer; at T = 2048 629k → 562k (the extra block gone). Smaller blocks win at short T, larger ones at long T (the per-block LD, transpose, `LOOP_END` and VE copies); 128 is the best at T = 128 and within 3% of the best from 256 up and saves 25 ms per token at T = 16 against the 512 blocks. Gemma (18 layers, Tb 256 before): T = 16 6.4 → 4.4 ms, T = 512 25.2 → 21.2 ms. What remains of the short-T regression against the untiled form (20.9 against 10.4 ms at T = 16) is about 60% q · Kᵀ (one block of 128 against 16 rows) and 40% p · V (26.5k against 12.4k cycles per layer: its output tiles in several N chunks at max_dynamic 2048, step d).

Checks: golden corpus 7658 of 7664 dispatches byte-identical, the 6 changed are exactly the tiled q · Kᵀ of `gemma_t512`, `qwen3_t1024`, `qwen3_t2048` (2 layers each; re-recorded); per-dispatch checks around the new block boundaries all OK (Gemma 512 at T = 16, 112, 128, 144, 240, 256, 272, 496, 512; Qwen3 1024 at 16, 128, 144, 496, 512, 528, 1024; Qwen3 2048 at 16, 128, 144, 1024, 1040, 2032, 2048). Bundles compiled for 256 are unchanged (the tiled form only applies past one bank). Not yet rebuilt for the board.

## 9. Verification system

| Level | Content | Tools |
|---|---|---|
| Plugin unit tests | Matching and lowering: MLIR fragments → expected template parameters or descriptors (FileCheck) | lit |
| Template tests | Each template executed on the functional simulator for random shapes, compared bit for bit with the corresponding NumPy / `DeviceModel` functions; serialization compared byte for byte with `DescList` | pytest + `sa_funcsim` |
| End to end (host) | The whole .vmfb runs on the `sim` transport, logits bit-identical to `DeviceModel(sfu=SfuExact)` | IREE runtime + simulator service |
| RTL co-simulation | A few command buffers generated by IREE (list + DDR image) extracted into `desc_run` and executed on `tb_system` | The existing `gen_desc_cases.py` flow |
| Board | Generated text identical to L5's hand-written path; stability (10 × 256 tokens); speed and counters | Board scripts |

Mutation tests as before: deliberately break templates (index modes, number of roundings, loop counts) and confirm each level of tests catches it.

---

## 10. Staged plan and acceptance

| Stage | Content | Acceptance | Effort (relative to L1–L5) |
|---|---|---|---|
| **C0 frontend** | Quantized torch model; iree-turbine export; `llvm-cpu` running on host and board; dispatch inventory | §3.4; a table mapping the inventory to the hand-written segments. **Done** (§3.5) | Small |
| **C1 driver** | sa HAL driver (board and sim transports); simulator service; sa-desc-v1 read/write | §5.6: hand-wrapped templates bit-identical to `DeviceModel` on sim and board. **Done** (§5.7) | Medium |
| **C2 first vertical slice** | Build iree-compile with the plugin from source; the plugin supports only the quantized linear layer kind of dispatch, erroring on the rest; tested with a "single linear layer" torch model | torch → iree-compile → .vmfb → sim and board, bit-identical to `DeviceModel.linear`. **Done** (§6.7) | Medium (the build environment is the main difficulty) |
| **C3 template library** | All templates of §6.4; the VE expression compiler; data tiling and weight packing; in-place KV cache updates | Text generated on the board by IREE-compiled stories15M identical to L5's hand-written path, logits bit-exact. **Done** (§6.9, 12.6 tok/s on the board) | Large |
| **C4 performance** | Fewer FENCEs, cross-dispatch prefetch, larger dispatches, cost model | Cycles per token within ≤ 10% of the hand-written path. **Partly done, on hold** (§7.1: 13% behind, 17.7 tok/s on the board) | Medium |
| **C5 code generation** | The two dialect levels `sahl` / `sahw` and the full pipeline (§8), templates turned into `sahl`-level micro-kernels, in steps C5.0–C5.5 (§8.7) | Generic code generation alone (`--iree-sa-ukernels=none`) compiles stories15M with results identical to C3; default-configuration cycles per token no higher than C4; one model variant compiles (§8.1). **Done** (§8.7: stories15M with / without micro-kernels bit-exact on the board; SmolLM2-135M 2.23 tok/s on the board, interactive generation) | Large |

| **C6 larger models** | HuggingFace import and generic quantization, parameterized target configuration, K / T tiling, GQA / QK-norm (§8.9) | Qwen3-0.6B, Llama-3.2-1B compiled on the sim, bit-exact per dispatch, end-to-end errors with truncated layer counts within range; on the board once a board with more resources is available. **In progress** (§8.12: target-configuration cross-validation, K tiling, C6.0 single embedding copy, C6.1 Qwen3-0.6B all 28 layers bit-exact on the sim done; C6.P prefill + decode bit-exact on the board for two models, prompts 4.3–4.7× faster per token than decode, §8.13) | Large |

| **Generic frontend** | HuggingFace's original modeling code exported directly, quantization and formulation done by generic graph rewrites (§8.16) | sa dispatches bit-exact with the oracle, quality (top-1 versus the original fp32 HF) no lower than the hand-written qhf path; Qwen3 without code changes; on the board; non-Llama structures. **In progress** (F0–F4 done; the generic path's prefill + decode bit-exact with decode only (sim and board), decode entirely on the accelerator, Qwen3 passes without code changes, §8.19) | Large |
| **Host fallback** | Dispatches sa cannot compile run on the ARM with VMVX (§8.15) | Any dispatch runs; mixed execution bit-identical to accelerator only (stories15M with the linear layers on the host). **Done** (sim and board) | Medium |
| **C8 layered restructuring of code generation** | Split `sahl-to-sahw` into the passes designed in §8.4: `sahl` operations, micro-kernel expansion, tiling, memory planning, pipelining, dynamic values (§8.14) | Golden corpus descriptors byte-identical at every step; `SahlToSahw.cpp` < 1000 lines; lit tests for each pass. **Done** (2026-09-29, §8.14: R0–R6; the `sahl` expansion of micro-kernels done 2026-10-04 (§8.20, `sahl-expand-kernels`, the old kernel lowering deleted); lifetime-based memory planning is next (§8.17 item 3); `sahw-legalize-dynamic` and R7's cost model left for later) | Large |
| **Z1 freeze** | Freeze the PYNQ-Z1 version after items 1 and 2, moving later optimization to the KV260 (§8.21) | Board regression all bit-exact; baseline files (including time breakdowns); tag `v1.0-pynq-z1`, branch `pynq-z1`, Release. **Done** (2026-10-04) | Small |
| **C7 RISC-V backend** (optional) | `sahw` → LLVM → riscv32, PicoRV32 issuing commands with PCPI instructions (§8.10) | One dispatch executed by PicoRV32 code, bit-identical to the descriptor path | Medium |

- **Order**: C0 → C1 → C2 → C3 → C4 → C5. C1 and C0 can run in parallel; C2 depends on C1's driver and sim.
- **Every step passes on the host first** (the sim transport), then on the board; commit after passing on the board, as by the existing convention.
- **Optional extensions**: batched decode (L5b's layout), prefill (batches of D tokens), larger models (stories42M / 110M), GQA.

---

## 11. Directory structure

All compiler-related code lives in the top-level `compiler/` of the repository (environment and build instructions in [`../compiler/README.md`](../compiler/README.md)):

```
compiler/
  env.sh                  paths and settings (IREE version, sources, build directory, Python environment)
  scripts/                fetch the IREE compiler's submodules; build iree-compile with the plugin from source
  plugins/sa/             the sa target backend plugin (C++, compiled into iree-compile through IREE_CMAKE_PLUGIN_PATHS)
    target/               TargetDevice / TargetBackend, preprocessing (weight packing), matching, C++ templates and DescList, sa-desc-v1 serialization (C2)
    templates/            Python references of the templates (reference.py; the C++ output must be byte-identical to it)
    dialect/              the two dialect levels sahl and sahw: ODS, attributes, operations, interfaces (C5, §8.3)
    transforms/           C5's pipeline (§8.4) and cost model (§8.5)
    ukernels/             sahl-level micro-kernels (linear, attention, §8.8)
    riscv/                the sahw → LLVM RISC-V backend (C7, §8.10)
    test/                 lit tests (iree-opt + FileCheck)
  frontend/               C0: qllama.py (the quantized model), export.py (iree-turbine export), comparison scripts
  runtime/                C1: the sa HAL driver (C, sa/), board / sim transports; tools/ (sa-desc-v1); test/
  sim/                    simulator service sa_sim_server.py, command-ring emulation sa_board_emu.py (based on llm/sa_funcsim.py)
  tests/                  test_c2 / test_c3 (host), board_c2 / board_llm (board); oracle.py (dispatch reference interpreter),
                          dispatch_check.py (per-dispatch differential tests), summarize.py
iree-sa/l0/               existing: armv7 toolchain verification (L0)
build/iree/               not in git: IREE sources, Python environment, compiler build directory
```

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Building iree-compile from source**: long, memory hungry, interfaces change between versions | Pin L0's version (3.11.0, e4a3b0405d); script the build; follow that version's built-in backends as examples for the plugin interfaces; migrate only on major upgrades |
| **Dispatches cut differently than expected** (too much or too little fusion) | C0 obtains the dispatch inventory before designing templates; preprocessing controls the cuts; the VE expression compiler catches arbitrary elementwise chains |
| **Numeric mismatches** (reassociation, FMA, split reductions, library functions) | The conventions of §3.2 are checked in the plugin, with violations raising errors; end-to-end tests compare bit for bit with `DeviceModel`, so any deviation shows immediately |
| **The KV cache copied whole every token** | C0 checks the generated IR and the runtime's copy operations; if necessary the KV update becomes a custom operator, or the driver manages the KV buffers directly |
| **Data through DDR between dispatches and too many FENCEs**, making C3 clearly slower than the hand-written path | Expected; C4 handles it specifically; with the hand-written path as the reference the gap can be quantified |
| **CMA and physical addresses**: IREE's default allocation does not fit | The driver brings its own CMA pool; parameter files are loaded directly into CMA |
| **Immature multi-device (CPU + accelerator) support** | The main path has all dispatches on the accelerator; mixed execution is only a fallback |
| **Problems of the armv7 runtime** (L0 ran into libm) | Keep L0's cross-compilation configuration and shim; the driver is pure C without floating-point library dependencies |
| **Effort**: C3 and C5 need a lot of C++ / MLIR | Vertical slices first; templates ported from verified Python implementations and compared one by one on the sim |

---

## 13. Open decisions

1. **After C0**: decide the list and granularity of templates from the dispatch inventory (whether preprocessing is needed to control the cuts).
2. **Implementation of the simulator transport**: a Python service (fast to build, reusing existing code) or porting `sa_funcsim` to C (no Python dependency, faster). Python first.
3. **Executable encoding**: a custom binary or flatbuffer. Leaning toward flatbuffer (IREE's convention), with content per §4.2. What was actually used is a custom binary (§4.4).
4. **Where constants live**: parameter files (.irpa, recommended) or embedded in the .vmfb.
5. **Where C4's cross-dispatch scheduling lives**: the driver (seeing adjacent dispatches while recording the command buffer) or the compiler (at link time). Originally leaning toward the compiler; **decided: the driver** (§7.1), with the compiler providing information through sa-desc v3.
6. **C5's bufferization**: IREE's one-shot bufferize (with the `#sa.mem` memory space) or our own. Try the former first (§8.4 step 4, §8.11).
7. **C5.5's model variants**: **decided**: SmolLM2-135M (board) and small configurations of the Qwen3 structure (sim) (§8.7, §8.9).
8. **C5's dialect layering**: **decided**: two levels, `sahl` (tile level) and `sahw` (command level); `sahw` feeds two backends, descriptors and RISC-V (C7) (§8.3, §8.10).

# Compiler learning path

A guided route through the compiler side of this repository: from MLIR and
IREE's structure, through the `sa` HAL target plugin that turns each IREE
dispatch into descriptor lists (or RISC-V code) for the KV260 accelerator, to
the runtime driver, the verification system and the patches sent upstream to
IREE. It is written against tag `v2.2-kv260` (2026-10-17): IREE `e4a3b0405d`
(3.11.0) plus `compiler/patches/iree/0001`–`0003`, four unmodified
HuggingFace models compiled whole (665–1127 executables each, all on the
accelerator), bit-exact with the functional simulator on the board.

Like the [hardware learning path](hardware_learning_path.md), each stage
pairs outside reading with the files that put the idea into practice, a lab
and questions to check yourself. The [LLM learning path](llm_learning_path.md)
covers the models, their quantization and the inference system; this one
covers how they are compiled. Two walkthroughs follow single dispatches
through every pass and are the best companions to stages 5–7:
[`dispatch_walkthrough.md`](dispatch_walkthrough.md) (decode attention
scores) and [`gemv_walkthrough.md`](gemv_walkthrough.md) (a decode linear
layer on the streaming GEMV unit).

The compiler's history, in [`iree_compiler_plan.md`](iree_compiler_plan.md):
- **C0** export of a quantized llama with iree-turbine; **C1** the `sa` HAL
  driver running hand-built executables; **C2** linear layers compiled;
  **C3** a whole model (stories15M) compiled by matching templates;
- **C4** scheduling across dispatches (one list per command buffer);
- **C5** a real code generator: the `sahl` and `sahw` dialects; **C8** its
  restructuring into separate passes, guarded by a golden corpus;
- **C6** larger models, prefill, host fallback, the generic HuggingFace
  frontend; **C6.5** attention tiled over a dynamic T (long context);
- **C7** ([`riscv_backend_design.md`](riscv_backend_design.md)): `sahw` →
  LLVM IR → rv32imc, code embedded in `sa.vmfb`; one executable per kernel
  instead of one per layer (IREE patch 0003, upstream as iree-org/iree#25059).

---

## How to use this path

- **Order.** Stages 0–1 are MLIR and IREE; stages 2–4 are the inputs and
  outputs of the backend (the exported program, the hardware's contract,
  what IREE hands over); stages 5–7 are the code generator; stages 8–9 the
  runtime and verification; stages 10–12 performance, the RISC-V backend and
  upstreaming. Do 0–6 in order.
- **Each stage has:** *Goal* · *Read* · *In this repo* · *Lab* · *Check
  yourself*.
- **Tools.** cmake ≥ 3.28, ninja, clang / lld 18, ccache. The compiler build:
  `compiler/scripts/fetch_iree_sources.sh`, then
  `compiler/scripts/build_iree_compiler.sh` (1–2 h at 4 jobs; run it in your
  own terminal). After plugin changes:
  `cmake --build $IREE_BUILD -j 4 --target iree-compile iree-opt` (minutes,
  ccache). `source compiler/env.sh` sets `$IREE_BUILD`, `$SA_PY` and the
  rest. With 15 GB of RAM, run one compile of a full model at a time, under a
  memory cap (`systemd-run --user -p MemoryMax=10G …`).
- **Environment for the KV260 overlay:** `SA_D=16 SA_GEMV_PORTS=4`
  (`--iree-sa-d=16 --iree-sa-gemv-ports=4`); D = 8 is the default and the
  PYNQ-Z1's.
- **The guard rail:** after any change to the plugin, run
  `compiler/scripts/run_tests.sh golden` (lit tests, then every executable of
  the corpus byte for byte; about 30 s).

## The compiler at a glance

```
 HuggingFace model (transformers' own code)          compiler/frontend/hf_generic.py
   │ module rewrites (QLinear, RopeTable, SaRMSNorm, SaCache + sa_attention)
   │ torch.export → ATen decompositions → graph passes → iree-turbine
   ▼
 qllama.mlir (torch dialect) + qllama.irpa (externalized weights)
   │ compile_sa.sh → iree-compile --iree-hal-target-device=sa
   ▼
 IREE input conversion → linalg / tensor / arith
   │ sa preprocessing:  iree-sa-pack-linear-weights, iree-sa-clone-cheap-producers
   │ DispatchCreation / Flow: fusion, dispatch regions, executables
   │   + iree-flow-generalize-executables (patch 0003), DeduplicateExecutables
   │ Stream: async scheduling, resources, transient buffers
   │ HAL: one executable variant per target; per export, the sa backend:
   │     iree-sa-lower-workgroup-count, iree-sa-codegen:
   │       bufferize → iree-sa-to-sahl → iree-sahl-tile → iree-sahl-plan-memory
   │       → iree-sahl-schedule → iree-sahl-expand-kernels → iree-sahl-to-sahw
   │       → iree-sahw-fuse-ve
   │     iree-sahw-split-head, iree-sahw-assign-registers
   │     serialize: sahw.template → sa-desc (DescList)  [+ rv32 code, --iree-sa-riscv]
   ▼
 sa.vmfb (VM bytecode: calls, bindings, push constants, barriers) + sa_packed.irpa
   │ ARM: IREE runtime + the sa HAL driver (compiler/runtime/sa)
   │   command buffer → one descriptor list (SETREG + CALL templates + FENCE)
   ▼
 PicoRV32 rt_fw → sa_cmdfetch → sa_sched → LD / ST / EX / VE / GEMV engines
```

---

## Stage 0 - MLIR fundamentals

**Goal.** Read and write MLIR: operations, regions, blocks, attributes,
types, dialects; run a pass with `iree-opt`; write a rewrite pattern.

**Read.**
- The MLIR Toy tutorial (chapters 1–6) and *MLIR: Scaling Compiler
  Infrastructure for Domain Specific Computation* (Lattner et al., 2021).
- MLIR docs: *Language Reference*, *Defining Dialects* (ODS), *Pattern
  Rewriting*, *Pass Infrastructure*, *Bufferization*.
- The `linalg` rationale (*Linalg Dialect Rationale*) and the `linalg.generic`
  indexing-map model; the TilingInterface (`scf::tileConsumerAndFuseProducersUsingSCF`).
- Optional background: Cooper & Torczon, *Engineering a Compiler* (SSA,
  scheduling, register allocation).

**In this repo.**
1. `compiler/plugins/sa/dialect/SahlOps.td`, `SahwOps.td`: two small ODS
   dialects (about 200 lines each). Read them as an example of ops with
   regions (`sahl.kernel`, `sahl.loop`, `sahw.template`), custom types
   (`!sahw.base`, `!sahw.param`) and verifiers.
2. `compiler/plugins/sa/test/*.mlir`: lit tests; each is a `RUN:` line with
   `iree-opt --pass` and `CHECK:` lines. `sahw_roundtrip.mlir` is the
   simplest.

**Lab.**
```sh
source compiler/env.sh
$IREE_BUILD/llvm-project/bin/llvm-lit -v compiler/plugins/sa/test/sahw_roundtrip.mlir
$IREE_BUILD/tools/iree-opt --iree-sahw-split-head --iree-sahw-assign-registers \
    compiler/plugins/sa/test/sahw_passes.mlir | head -60
```
Then change one `CHECK` line and watch the test fail.

**Check yourself.**
- What makes `sahw.template` `IsolatedFromAbove`, and why does the backend
  want that?
- What is the difference between a pass on `func.func` and one on
  `builtin.module`, and which does `buildSahlPipeline` use for each pass?
- Why are `linalg.generic`'s indexing maps enough to decide whether an
  operation is an elementwise op, a reduction or a contraction?

## Stage 1 - IREE's architecture and the plugin API

**Goal.** Know what each IREE phase does, what an *executable*, *export*,
*binding*, *push constant* and *workgroup* are, and where a target plugin
plugs in.

**Read.**
- IREE docs: *Developer overview*, *Glossary*, the *HAL* and *Stream* design
  docs, *Extending IREE with a custom target* (the plugin API), and the
  `iree-compile --help` sections on dumping (`--mlir-print-ir-after-all`,
  `--iree-hal-dump-executable-sources-to`, `…-binaries-to`).
- The IREE source: `compiler/src/iree/compiler/Dialect/HAL/Target/TargetBackend.h`
  (the interface the plugin implements) and `Dialect/Flow/Transforms/`
  (dispatch formation, `DeduplicateExecutables`).

**In this repo.**
1. [`iree_compiler_plan.md`](iree_compiler_plan.md) §1: the data flow, the
   two parts to write (backend plugin, HAL driver), and §1.3's mapping of
   IREE concepts to the hardware (bindings → BASE registers, push constants →
   PARAM, workgroup count → LOOP_END, barriers → FENCE).
2. [`compiler/README.md`](../compiler/README.md): layout, environment, the
   cut-down build (only `sa` and `vmvx` backends, Release, no llvm-cpu by
   default) and why.
3. `compiler/plugins/sa/iree_compiler_plugin.cmake`,
   `compiler/plugins/sa/target/SATarget.cpp`: registration of
   `#hal.device.target<"sa">` and `#hal.executable.target<"sa", "sa-desc-v1">`,
   the options (`SAOptions::bindOptions`: read every `--iree-sa-*` flag
   there), `extendPreprocessingPassPipeline`,
   `buildTranslationPassPipeline`, `serializeExecutable`.

**Lab.**
```sh
compiler/scripts/build_iree_compiler.sh --configure      # checks CMake and the plugin
$IREE_BUILD/tools/iree-compile --iree-hal-list-target-backends   # must list sa
$IREE_BUILD/tools/iree-compile --help | grep -A1 iree-sa-
```

**Check yourself.**
- Which IREE phase decides how many executables a model has, and which one
  decides the order of dispatches in a step?
- What does the sa backend receive per export: a `func.func` on tensors, or
  on memrefs? (Stage 5's first pass answers it.)
- Why does the build exclude llvm-cpu, and what turns it back on?
  (`SA_WITH_LLVM_CPU=1`, needed for stage 11's R5.)

## Stage 2 - The frontend: from PyTorch to an IREE input program

**Goal.** Export a model so that the KV cache is mutable state, the weights
are parameters, the attention length is dynamic, and every quantization step
is explicit.

**Read.**
- PyTorch docs: `torch.export`, *ExportedProgram*, decompositions
  (`run_decompositions`), buffer mutation.
- iree-turbine docs: `aot.export`, externalized parameters, `.irpa`.
- [`frontend_analysis.md`](frontend_analysis.md) §2: how other LLM compiler
  frontends (TVM / MLC, ExecuTorch, TensorRT-LLM, torch-mlir) capture and
  lower graphs.

**In this repo.**
1. `iree_compiler_plan.md` §3 (C0): the quantized model, the numeric
   contract ("the compiler never quantizes by itself"), the export.
2. `compiler/frontend/qllama.py`, `compiler/frontend/export.py`: the
   hand-written path (stories15M); why the KV cache is one buffer written with
   `index_copy_` per layer (the module docstring explains the IREE CPU
   backend's limit).
3. `compiler/frontend/hf_generic.py`: the generic path. `HFDecoder` +
   `PositionedCache` (buffer mutations become mutable globals); the module
   rewrites; `GRAPH_PASSES` (FX graph rewrites, today `square_pow`);
   `export()` (`torch.export` non-strict → decompositions → graph passes →
   iree-turbine, weights externalized, `prefill` / `prefill_all` exports).
4. `iree_compiler_plan.md` §8.16 (the generic frontend) and §8.19 (prefill on
   it).
5. `compiler/frontend/ct_import.py`: a compressed-tensors checkpoint's int8
   weights imported as they are.

**Lab.**
```sh
$SA_PY compiler/frontend/hf_generic.py --model build/llm_cache/SmolLM2-135M \
    --out build/learn/smol_l2 --layers 2 --quant --rope --attn --prefill 16
grep -c util.global build/learn/smol_l2/qllama.mlir
grep -n "func.func" build/learn/smol_l2/qllama.mlir | head
```
Find the KV cache globals, the `main` / `prefill` functions and their
arguments (`input_ids`, `position_ids`, `valid`).

**Check yourself.**
- Why does the attention length come in as the length of a `valid` tensor
  rather than as an integer argument?
- Why must the weights be externalized for a 1B model on a 15 GB host?
- What would happen to the KV cache if the decoder returned it instead of
  mutating a buffer?

## Stage 3 - The target's contract: descriptors, registers and sa-desc

**Goal.** Know exactly what the backend must emit: the commands, the
registers, the control flow the fetch unit understands, and the file format
the runtime loads.

**Read.** The hardware side is in the
[hardware learning path](hardware_learning_path.md) stages 4–5 and 8 (the
engines, the scheduler, the descriptor fetch unit); read it at least to the
level of "what one LD, EX, VE command does".

**In this repo.**
1. [`custom_isa_encoding.md`](custom_isa_encoding.md): the PCPI instruction
   encoding and the 64-byte descriptor.
2. `llm_inference_plan.md` §5.3: BASE / PARAM registers, dynamic fields
   (at most two per command), SETREG, LOOP_END, CALL / RET, LDPARAM: the
   extensions made *for* a compiler.
3. `iree_compiler_plan.md` §4: sa-desc-v1 (position-independent templates,
   relative jumps only, binding i → BASE i, push constants → PARAM0–5,
   PARAM6/7 private), and §4.4 the actual format up to v3 (prefix, head,
   body, register setup entries).
4. `compiler/runtime/tools/sadesc.py` (writer, reader, the header layout
   including the optional RISC-V code table at word 48),
   `compiler/runtime/tools/sadis.py` (disassembler),
   `compiler/plugins/sa/target/DescList.cpp` (the C++ encoder, byte-identical
   to `driver/pynq_matmul.py`'s `DescList`).
5. [`memory_model.md`](memory_model.md): address maps, the uncached device
   window, ordering and alignment rules the code generator must respect.

**Lab.**
```sh
$SA_PY compiler/runtime/tools/make_test_exec.py --d 16 --out build/learn/c1
$SA_PY compiler/runtime/tools/sadis.py build/learn/c1/*.sadesc | head -40
```

**Check yourself.**
- Why must a template be position independent, and what in the format
  enforces it?
- A command needs three dynamic values. What does the backend do? (Stage 7.)
- How does the driver compute a PARAM from a push constant? (`sa_loader.c`:
  `((c × mul) >> shift) + add`.) What can and cannot be expressed that way?

## Stage 4 - What IREE hands over: preprocessing and dispatch formation

**Goal.** See a whole model's dispatches, know why there are as many as there
are, and what the backend does before IREE forms them.

**Read.**
- IREE's `DispatchCreation` and `Flow` passes (fusion heuristics, dispatch
  regions, executable outlining) and `DeduplicateExecutables`.

**In this repo.**
1. `compiler/plugins/sa/target/PackLinearWeights.cpp`
   (`iree-sa-pack-linear-weights`): weights rewritten into the B-tile layout
   at compile time, folded by const-eval, exported to `sa_packed.irpa`
   (`iree_compiler_plan.md` §6.3).
2. `compiler/plugins/sa/target/CloneCheapProducers.cpp`: index-only producers
   cloned into their consumers so dispatches stay self-contained.
3. `compiler/plugins/sa/target/LowerWorkgroupCount.cpp`: a dispatch runs as
   one workgroup; loops live inside the template.
4. `compiler/scripts/compile_sa.sh`: the full command line (host fallback,
   executable generalization on by default, linking off, the dispatch
   sources dumped into `sa_sources/`; extra flags through
   `SA_COMPILE_FLAGS`).
5. `compiler/patches/iree/0003-flow-generalize-executables.patch` and
   [`riscv_backend_design.md`](riscv_backend_design.md) §9.11: executables
   differing only in index / f32 constants or static offsets into a global
   get those as dispatch operands, and `DeduplicateExecutables` merges them
   (Qwen3 1132 → 1051 executables; a decode step calls 59 distinct exports
   instead of 140).
6. `compiler/patches/iree/0001`, `0002`: two fixes to IREE's
   `MergeIndexSwitchPattern` (iree-org/iree#25037, #25038): adjacent
   `scf.index_switch` ops merged although they switch on different values,
   and a merged region's yield of a value defined outside it lost.
   `compiler/plugins/sa/test/merge_index_switch.mlir` is the reduced test.

**Lab.** Compile the 2-layer export of stage 2 twice and compare:
```sh
SA_D=16 SA_GEMV_PORTS=4 compiler/scripts/compile_sa.sh build/learn/smol_l2
ls build/learn/smol_l2/sa_sources | wc -l
SA_GENERALIZE=0 SA_D=16 SA_GEMV_PORTS=4 compiler/scripts/compile_sa.sh build/learn/smol_l2
ls build/learn/smol_l2/sa_sources | wc -l
```
Open two attention sources that differ only in their KV offset (with
`SA_GENERALIZE=0`) and find the operand that replaces it.

**Check yourself.**
- Why are per-layer attention dispatches *not* identical before
  generalization, when their code is the same?
- Why does patch 0003 generalize an offset only into a global's contents?
  (The pass's header comment in the patch, and §9.11.)
- Why is `--iree-hal-link-executables=false` needed with the host fallback?
  (`iree_compiler_plan.md` §8.15.)

## Stage 5 - The sa code generator, pass by pass

**Goal.** Follow one dispatch from linalg on tensors to a `sahw.template`,
and be able to say which pass made each decision.

**Read.** [`dispatch_walkthrough.md`](dispatch_walkthrough.md) end to end,
then [`gemv_walkthrough.md`](gemv_walkthrough.md) §3–5. They are this stage's
textbook.

**In this repo** (in pipeline order; `transforms/SahlPipeline.cpp`
`buildSahlPipeline`, driven per export by `SaCodegen.cpp`):

| Pass | File | Decides |
|---|---|---|
| bufferize | IREE's `ComprehensiveBufferize` | tensors → memrefs on the bindings |
| `iree-sa-to-sahl` | `SaToSahl.cpp`, `SahlKernels.cpp` | explicit DDR ↔ local traffic (`sahl.load` / `store`, `gather`, `scatter`); which ops form a micro-kernel (linear, attention, GEMV) or a generic contraction, grouped into `sahl.kernel` |
| `iree-sahl-tile` | `SahlTile.cpp` | pieces: flat and row pieces of elementwise work, N and K blocks |
| `iree-sahl-plan-memory` | `SahlPlanMemory.cpp`, `SahlLocalMemory.h`, `target/Layout.cpp` | SPAD / ACC bank placement, double buffering in different banks, capacity |
| `iree-sahl-schedule` | `SahlSchedule.cpp` | the linear layer's chunks, prefix (weights loadable early) and head |
| `iree-sahl-expand-kernels` | `SahlExpandKernels.cpp` | each micro-kernel's schedule written out in `sahl`: strips, `sahl.mma` / `sahl.gemv`, ping-pong, `sahl.loop` |
| `iree-sahl-to-sahw` | `SahlToSahw.cpp`, `SahlLowerGeneric.cpp`, `SahlLower.h` | one `sahw` op per command; elementwise bodies → VE stages; dynamic values → PARAMs |
| `iree-sahw-fuse-ve` | `SahlPipeline.cpp` | VE stages merged where rounding is unchanged (constant multiply → A, add → B, SFU → FUNC, to_i8 → I8 output) |
| `iree-sahw-split-head` | `SahwPasses.cpp` | the template's prefix / head / body |
| `iree-sahw-assign-registers` | `SahwPasses.cpp` | BASE / PARAM allocation and the setup table |
| serialize | `SahwSerialize.cpp`, `target/DescList.cpp` | bytes of sa-desc v3 |

Background for the table: `iree_compiler_plan.md` §8.3 (why two dialect
levels), §8.4 (the designed 16-pass pipeline and what was implemented),
§8.14 (C8: how a 3052-line `Lowerer` was split into these passes, the
descriptors byte-identical at every step), §8.20 (micro-kernels expanded at
the `sahl` level), `SaLin.h` (values derived from push constants).

**Lab.** Walk a real dispatch:
```sh
compiler/scripts/walk_dispatch.sh build/learn/smol_l2 7 build/learn/walk
ls build/learn/walk          # after_<pass>.mlir for every sa pass, d7.sadesc, d7.dis.txt
```
Pick a dispatch whose name contains `matvec_like` and one with
`batch_matmul`; diff consecutive `after_*.mlir` files and name each pass's
change.

**Check yourself.**
- Where is the choice "this contraction is a linear layer micro-kernel"
  made, and where is it lowered?
- Why does `sahl-plan-memory` put double-buffered blocks in *different*
  banks? (The scoreboard tracks dependencies per bank.)
- Which rules may `sahw-fuse-ve` *not* apply, and why does that matter for
  bit-exactness?

## Stage 6 - Writing and testing a pass

**Goal.** Change the code generator safely: a lit test for the pass, the
golden corpus for the whole, and per-dispatch checks when the output must
change.

**In this repo.**
1. `compiler/plugins/sa/test/`: one lit file per pass
   (`sa_to_sahl.mlir`, `sahl_tile*.mlir`, `sahl_expand_*.mlir`,
   `sahl_to_sahw.mlir`, `sahw_passes.mlir`, …).
2. `compiler/tests/golden.py`: the corpus (dispatch sources of stories15M,
   SmolLM2, a 2-layer Qwen3, a synthetic K = 16384 model, `ukernels=none`,
   D = 16 configurations…) recompiled with `--compile-mode=hal-executable`
   and compared byte for byte (`iree_compiler_plan.md` §8.14).
3. `iree_compiler_plan.md` §8.14's rule: a change that must alter the output
   states why, and the changed executables are accepted by `dispatch_check`
   and end-to-end bit-exactness instead.

**Lab.** Make a deliberate change and catch it:
1. In `SahlPipeline.cpp` (`SahwFuseVePass`), disable one fusion rule.
2. Rebuild (`cmake --build $IREE_BUILD -j 4 --target iree-compile iree-opt`).
3. `compiler/scripts/run_tests.sh golden`, and read
   `build/c6/logs/golden.log`: which executables changed?
4. Revert, rebuild, rerun.

**Check yourself.**
- Why is the corpus made of dispatch *sources* rather than whole models?
- When is a byte-identical corpus the wrong criterion?

## Stage 7 - Hard cases: dynamic shapes, tiling and capacity

**Goal.** Understand the problems that made the backend more than a template
matcher: dynamic lengths, contractions that do not fit local memory, and
data-dependent addresses.

**In this repo.**
1. **Dynamic values.** `SaLin.h` (`Lin`: mul, shift, add of a push
   constant), `sa_loader.c`'s setup table, the two-dynamic-field limit and
   loops over rows (`iree_compiler_plan.md` §8.3, `sahw-legalize-dynamic` in
   §8.4).
2. **Attention over a dynamic T** (`iree_compiler_plan.md` §8.22): T tiled in
   blocks in a hardware loop whose count is `(T >> log2 Tb) + 1`; why no
   online softmax (the result stays bit-identical); `sahl.loop` with a
   dynamic count (`DYN_LOOP_COUNT`); the scratch ACC area and VE copies;
   `loopPool` sharing PARAMs between sequential loops. Tests:
   `sahl_expand_tiled_t.mlir`, `sahl_tile_rows_dyn.mlir`,
   `compiler/tests/test_long_context.py`.
3. **GEMV over a long K** (`generality_implementation_plan.md` §10, WP6):
   decode's linear layers with K > 4080 in K blocks accumulated in ACC
   (Llama's `down_proj`, K = 8192; ×2.89 decode). Test:
   `compiler/tests/test_gemv_ksplit.py`.
4. **Gather and scatter.** `sahl.gather` (embedding rows, RoPE rows by
   position: LDPARAM), `sahl.scatter` (the KV write at a dynamic row);
   `sahl_gather.mlir`, `sahl_scatter*.mlir`.
5. **When it does not fit.** `SaCodegen.cpp` retries a dispatch with smaller
   pieces on "ACC full", and the diagnostics handler that must take only its
   own thread's errors (read the comment in `compile`).
6. **The host fallback** (`iree_compiler_plan.md` §8.15): a VMVX variant for
   every dispatch, an always-false condition on the sa variant of one the
   backend cannot compile; `--iree-sa-host-dispatches` to force it;
   `test_fallback.py`.

**Lab.**
```sh
$SA_PY compiler/tests/test_long_context.py --model build/llm_cache/gemma-3-270m --max-len 512
```
It exports a 2-layer Gemma at 512 positions, compiles it and checks every
dispatch against the oracle at T around the block boundaries. Then read one
tiled q · Kᵀ dispatch with `walk_dispatch.sh`.

**Check yourself.**
- Why can the loop count be one block too many when T is a multiple of Tb,
  and why is that harmless?
- A dispatch with `exp` falls back to the host. Why does that break
  bit-exactness with the simulator? (The host uses libm; the device its SFU.)
- Why does a per-row buffer need a scratch ACC area instead of letting each
  EX write into it directly?

## Stage 8 - The runtime: the sa HAL driver

**Goal.** Know what happens between `iree_runtime_call` and the doorbell: how
executables load, how a command buffer becomes one descriptor list, how
transports and the host fallback work.

**Read.** IREE's HAL API (`runtime/src/iree/hal/`), in particular
`command_buffer.h`, `executable_cache.h`, `semaphore.h`, and the inline
command buffer in `hal/local/inline_command_buffer.c` the sa one is modeled
on.

**In this repo.**
1. `iree_compiler_plan.md` §5: object mapping, command-buffer encoding,
   memory, synchronization, transports.
2. `compiler/runtime/sa/`: `sa_driver.c`, `sa_driver_module.c` (the
   driver's registration, with a second loader, VMVX, for the fallback),
   `sa_device.c`, `sa_loader.c` (templates copied into device memory,
   register setup), `sa_command_buffer.c` (dispatches appended to one batch
   list, barriers → FENCE), `sa_context.c`, `sa_transport_board.c` /
   `sa_transport_sim.c`.
3. `compiler/sim/sa_sim_server.py` (the sim transport: `llm/sa_funcsim.py` on
   shared-memory DDR) and `compiler/sim/sa_board_emu.py` (rt_fw's ring on the
   host).
4. `compiler/runtime/tools/sa_llm_run.c`: the decode loop in one process
   (`--abi`, `--pad`, `--prefill`, `--prefill_all_out`, `--rvsched`), the
   tool every test runs.
5. `compiler/scripts/build_sa_runtime.sh host|aarch64`,
   `compiler/scripts/check_bundle_aarch64.sh` (the aarch64 build under qemu,
   checked against the sim before a bundle goes to the board).

**Lab.**
```sh
compiler/scripts/build_sa_runtime.sh host
$SA_PY compiler/runtime/tools/make_test_exec.py --d 8 --out build/c1/d8
$SA_PY compiler/sim/sa_sim_server.py --d 8 --once &
build/iree/build-sa-host/runtime/plugins/hal/drivers/sa/sa_hal_test build/c1/d8
```
Then run a whole model on the sim with `SA_STATS=1` (lists on the
accelerator, dispatches on the host) through `test_hf_generic.py`.

**Check yourself.**
- Why does one command buffer become *one* list, and what does the driver do
  when it meets a VMVX executable in the middle?
- Where would a per-step host copy show up in the runtime, and why can
  neither the sim nor the cycle model see its cost? (Stage 9.)

## Stage 9 - Verification: from per-dispatch oracle to the board

**Goal.** Know the layers of checking and which bug each one catches.

**In this repo.**
1. `iree_compiler_plan.md` §9 and `generality_implementation_plan.md` §3.1:
   the layers of correctness.
2. `compiler/tests/oracle.py`: a reference interpreter of dispatch IR with
   the device's numerics (fp32 of `llm/fp32.py`, SFU functions, the device's
   reduction order), independent of the code generator.
3. `compiler/tests/dispatch_check.py`: every dispatch's source on the oracle
   against its binary on the functional simulator, all written bytes equal.
4. `compiler/tests/golden.py` (stage 6), `compiler/tests/rv_equiv.py`
   (stage 11).
5. `compiler/tests/host_copies.py`: the module compiled to the stream level,
   `stream.cmd.copy` ops counted per function; a copy of a global fails the
   build (a strided KV write once cost 640 ms per Qwen3 step on the board).
6. `compiler/tests/test_hf_generic.py`: a whole model on the sim against
   transformers; `compiler/scripts/run_tests.sh` lists the host suites
   (`golden`, `fallback`, `hfgen`, `ctimport`, …).
7. Board: `board_run.py llm` (logits bit-exact with the sim in 17/17 steps),
   `longctx` (prefill against decode only), [`results_v2.2.md`](results_v2.2.md).

**Lab.**
```sh
SA_D=16 SA_GEMV_PORTS=4 \
  SA_COMPILE_FLAGS=--iree-hal-dump-executable-binaries-to=build/learn/smol_l2/sa_binaries \
  compiler/scripts/compile_sa.sh build/learn/smol_l2
$SA_PY compiler/tests/dispatch_check.py build/learn/smol_l2/sa_sources \
    build/learn/smol_l2/sa_binaries --d 16 --t 16
$SA_PY compiler/tests/host_copies.py build/learn/smol_l2
```

**Check yourself.**
- What does the oracle know that the code generator does not, and why must
  it not share code with it?
- Which layer would have caught the `firstToEmit` bug (an `arith.bitcast`
  missing from a list of ops that emit no command), and which would not?
- Why is bit-exactness, not a tolerance, the right criterion between the
  board and the sim?

## Stage 10 - Performance engineering in the compiler

**Goal.** Measure before changing; know which tool answers which question.

**In this repo.**
1. Profiling: `SA_PROFILE=1 SA_PROFILE_PERF=<csv>` (one list per dispatch,
   the unit's 32 event counters per export); `compiler/tests/board_profile.py`,
   `dispatch_stats.py` (time and MACs per category, coverage),
   `kv260_fixed.py` (the per-step fixed part), `z1_profile.py`.
2. `compiler/tests/perf_model.py` + `compiler/scripts/perf_trace.sh`: a
   command-level model following the RTL's timing rules, calibrated on board
   counters; used to choose hardware before writing RTL
   ([`kv260_upgrade_plan.md`](kv260_upgrade_plan.md) K2b,
   [`k2b_gemv_design.md`](k2b_gemv_design.md)).
3. `compiler/tests/attn_cycles.py`, `issue_rate.py`: focused models of
   attention and of command issue.
4. Case studies, each a decision made from a measurement:
   - C4 (`iree_compiler_plan.md` §7.1): cross-dispatch scheduling, and why
     its fusions were put on hold;
   - the V projection on GEMV and attention over the actual length
     (`generality_implementation_plan.md` §13.1);
   - p · V on GEMV and the pipelined T loop (`iree_compiler_plan.md` §8.22);
   - one executable per kernel (`riscv_backend_design.md` §9.11).

**Lab.** From a board run's `profile_batch/perf.csv`, compute the read rate
and the share of all-idle cycles per step as [`results_v2.2.md`](results_v2.2.md)
§5 does; then run `dispatch_stats.py` on the per-dispatch `profile/perf.csv`
of the same bundle and find the top three exports.

**Check yourself.**
- Why does the per-dispatch profile inflate submission time, and which
  numbers can still be trusted from it?
- What does a high "all engines idle" share point to, and which compiler
  change addresses it?

## Stage 11 - A second backend: sahw → LLVM IR → RISC-V

**Goal.** Lower the same `sahw` templates to code for the PicoRV32 control
core, embed it in the executable, and prove it issues the same commands.

**Read.**
- The LLVM Language Reference (IR basics), `llvm::IRBuilder`, and the RISC-V
  assembler's `.insn` directive.
- [`riscv_backend_design.md`](riscv_backend_design.md) §0–4, §7, §9 (this
  stage's design document).

**In this repo.**
1. `compiler/plugins/sa/transforms/SahwToRiscv.cpp`: one function per
   export, `void f(const uint32_t *bindings, const uint32_t *push_constants)`;
   registers become variables, commands inline `.insn` (no custom LLVM:
   §9.1), `sahw.loop` a real loop, `ldparam` a fence and a volatile load;
   built with `IRBuilder` in a private `LLVMContext` so it is thread-safe
   inside the parallel serialization.
2. `--iree-sa-emit-riscv=<dir>` (portable `.ll`, host checks) and
   `--iree-sa-riscv` (in-process rv32imc, the code table in sa-desc: S4).
3. `firmware/include/sysarray_intrinsics.h`, `sa_rv_rt.h`,
   `firmware/rv/sa_rv_rt.c`: the PCPI primitives, shared with the firmware.
4. `compiler/tests/rv_equiv.py` + `rv_host_main.c`: per dispatch, the
   generated function on the host (PCPI recorded) against the descriptor
   template on the simulator, the same command stream.
5. `compiler/tests/rv_codesize.py`: a step's rv32 working set against an
   instruction memory (decode 20–41 KB at v2.2).
6. Runtime: `compiler/runtime/sa/sa_rv.c` (`SA_RV=embedded`),
   `sa_rvsched.c` (plan B), `compiler/tests/r5_kernel.py` (R5: a scalar
   kernel compiled by IREE's own llvm-cpu for riscv32).
7. Results: `riscv_backend_design.md` §9.3, §9.10 (plan A's gap is
   instruction fetch from DDR: about 27 cycles per instruction, 9–11
   instructions per command), `results_v2.2.md` §6.

**Lab.**
```sh
$SA_PY compiler/tests/rv_equiv.py --golden --t 16,304
$SA_PY compiler/tests/rv_codesize.py build/deploy_hfgen_qwen3g_pcpi/sa.vmfb \
    build/deploy_hfgen_qwen3g_pcpi/results/*/*/*/profile/perf.csv
```
(`rv_equiv.py` needs the system clang-18 with the RISC-V target; the second
command needs a bundle compiled with `--iree-sa-riscv` and its profile.)

**Check yourself.**
- Why is inline assembly acceptable here, and what would a custom LLVM
  target buy for an ASIC? (§9.1.)
- Why must the embedded functions contain no relocation or data section?
- Why did one executable per kernel shrink the rv32 code far more than the
  descriptor templates?

## Stage 12 - Working with upstream IREE

**Goal.** Carry patches against a pinned IREE, reduce a bug to a test, and
send it upstream properly.

**In this repo.**
1. `compiler/patches/iree/`: the three patches applied to `build/iree/src`
   (pinned at `e4a3b0405d`); `compiler/scripts/fetch_iree_sources.sh`.
2. The path of patch 0003 from a local fix to iree-org/iree#25059: a pass in
   Flow, a lit test with two `RUN` lines (`generalize_executables.mlir`: four
   cases, with and without deduplication), off by default behind
   `--iree-flow-enable-executable-generalization`, rebased onto IREE main,
   `git clang-format`, and a PR description with the motivation, numbers and
   open questions.
3. The rules a contributor follows: the DCO `Signed-off-by` is the author's
   own (never added for them), and IREE's AI tool policy (the author reviews
   and understands all generated code, marks AI assistance, ideally writes
   the description).

**Lab.** Read `compiler/patches/iree/0001-util-merge-index-switch-same-selector.patch`
and `compiler/plugins/sa/test/merge_index_switch.mlir`: state the bug in one
sentence, then find the smallest input that shows it.

**Check yourself.**
- Why was the generalization pass put in Flow, before deduplication, rather
  than in the sa backend?
- What makes a backend-motivated change acceptable upstream when no upstream
  backend needs it yet?

## Stage 13 - Where to go next (capstone ideas)

- **The designed passes not yet split out** (`iree_compiler_plan.md` §8.4
  status): `sa-normalize`, `sahw-legalize-dynamic`, a fine-grained
  `sahw-schedule`; each with its lit test and the corpus unchanged.
- **A cost model that picks block sizes** (§8.5) instead of capacity rules.
- **Compile-time cross-dispatch fusion**: C4's on-hold fusions (§7.1) as
  Flow-level dispatch formation for the sa target.
- **Generalization on Stream executables**, or with two dynamic address
  terms: the PR's open questions, implemented and measured.
- **Plan A with a TCM**: once an instruction memory exists, specialize the
  configuration writes at compile time and measure the remaining issue cost
  (`riscv_backend_design.md` §8).
- **W4A8 in the backend**: an unpack stage for GEMV, the packing pass and the
  oracle extended (`generality_implementation_plan.md` §17.6).

---

## Appendix A - Lessons from this project's compiler bugs

- **IREE's `MergeIndexSwitchPattern` merged `scf.index_switch` ops with
  different selectors**: with the host fallback, a dispatch run on the host
  took the sa variant's workgroup count (1 instead of 3: a third of its
  output). Patch 0001, upstream #25037; 0002 (#25038) fixes a lost yield in
  the same pattern. Lesson: a bug in the host program shows up as wrong
  device output.
- **A strided write into a global became a per-step `stream.cmd.copy`** run
  on the CPU over the uncached window: 640 ms per Qwen3 step, invisible to
  the sim and the cycle model. Now `host_copies.py` runs in every build.
- **A list of "quiet" ops (no commands) was incomplete**: once the f32 scale
  arrived as a push constant through `arith.bitcast`, contractions stopped
  expanding (`SahlExpandKernels.cpp` `firstToEmit`). Lesson: a pass that
  enables another can expose an allow-list.
- **Stale outputs pass for results**: `sa_sources/` left from an earlier
  compile had no binary; `compile_sa.sh` now clears it first.
- **Parallel compilation and diagnostics**: executables compile on several
  threads with one context; a handler that took other threads' errors made
  the "ACC full" retry random (`SaCodegen.cpp`).
- **Hardware limits are part of the contract**: a zero-length command is a
  sticky `XERR_SHAPE`, so a dynamic loop count must never be 0; sequential
  loops each taking two PARAMs overflowed the eight registers (`loopPool`).
- **A refactor is only safe with a byte-level guard**: C8 split a 3052-line
  lowering into passes with the golden corpus unchanged at every step.

## Appendix B - Glossary

| Term | Meaning here |
|---|---|
| dispatch / executable / export | one unit of device work in IREE / its compiled container / one entry point in it |
| binding, push constant | a buffer argument of a dispatch (→ BASE register) / a 32-bit scalar argument (→ PARAM) |
| `sahl` | the tile-level dialect: DDR ↔ local traffic, micro-kernels, loops |
| `sahw` | the command-level dialect: one op per accelerator command, inside a `sahw.template` |
| template | an export's position-independent descriptor list: prefix, head, body |
| micro-kernel | a matched pattern (linear, attention, GEMV) lowered by a hand-written schedule in `sahl` |
| `Lin` | a value `((c × mul) >> shift) + add` of a push constant, computed by the driver |
| dynamic field | a descriptor field replaced or offset by a PARAM at fetch time (at most two per command) |
| LDPARAM | a PARAM loaded from DDR at run time (data-dependent addresses) |
| host fallback | the VMVX variant of a dispatch the sa backend cannot compile, run on the ARM |
| generalization | promoting constants that differ between otherwise-equal executables to dispatch operands (patch 0003) |
| golden corpus | dispatch sources whose sa-desc bytes must not change (`golden.py`) |
| oracle | the reference interpreter of dispatch IR with the device's numerics (`oracle.py`) |
| plan A / plan B / R5 | dispatches as rv32 code / the decode loop on the PicoRV32 / an llvm-cpu kernel on the core |

## Appendix C - References

- Lattner et al., *MLIR: Scaling Compiler Infrastructure for Domain Specific
  Computation*, CGO 2021; the MLIR Toy tutorial and dialect docs
  (mlir.llvm.org).
- IREE documentation (iree.dev): developer overview, HAL, Stream, plugins;
  the IREE source at the pinned revision.
- Vasilache et al., *Composable and Modular Code Generation in MLIR* (the
  linalg / structured codegen paper), 2022.
- Chen et al., *TVM: An Automated End-to-End Optimizing Compiler for Deep
  Learning*, OSDI 2018; Lattner & Adve, *LLVM*, CGO 2004.
- Ragan-Kelley et al., *Halide* (algorithm / schedule separation), PLDI
  2013.
- Jouppi et al., *In-Datacenter Performance Analysis of a Tensor Processing
  Unit*, ISCA 2017 (a command-driven accelerator and its compiler contract).
- PyTorch `torch.export` and iree-turbine documentation; torch-mlir.
- The LLVM Language Reference and the RISC-V assembler manual (`.insn`).
- Cooper & Torczon, *Engineering a Compiler*, 3rd ed.

# C7: a RISC-V backend for the sa accelerator (design, 2026-10-15)

Status: **design and pre-research**; plan B (§7.1, the decode loop on the PicoRV32) implemented and **accepted on the board 2026-10-16** (Gemma-3-270M, Qwen3-0.6B, Llama-3.2-1B: tokens identical, logits bit-exact). C7 is the optional stage of
[`iree_compiler_plan.md`](iree_compiler_plan.md) §8.10. Its purpose here is to
prepare a future ASIC NPU: a small programmable core next to the
accelerator that issues its commands and runs the scalar code the
accelerator cannot (control flow over data, sampling). The KV260 overlay
already has such a core (PicoRV32), so the questions an ASIC design would
ask can be measured on it before any silicon decision: how fast must the
core issue commands, where must its code live, and which parts of a model
should it run.

## 0. The paths at a glance (2026-10-16)

Two independent questions decide how a decode step reaches the accelerator: **who sequences the step** (the ARM or the PicoRV32) and **how each command is encoded** (a descriptor list the fetch unit executes, or PCPI instructions the core issues). The paths below are combinations of the two, plus the scalar code of model C.

| Path | Who sequences a step | How commands reach the unit | Switch | Status |
|---|---|---|---|---|
| **Default** (L6-IREE) | the ARM: IREE VM + sa HAL driver build one or two lists per step; `rt_fw` only `mat_submit`s ring entries | descriptor lists (templates from the descriptor backend, `SETREG` + `CALL` per dispatch) | none | the main line; every board result of the compiler plan |
| **Plan B** (§7.1, model B) | **the PicoRV32** runs the decode loop (`rt_fw` `RT_RUN_SCHED`); the ARM records each attention-length bucket's lists once, then waits | the same descriptor lists, recorded, patched per bucket | `sa-llm-run --rvsched` | **accepted on the board** 2026-10-16 (Gemma-3, Qwen3, Llama-3.2: tokens identical, logits bit-exact) |
| **Plan A** (§9, model A) | per dispatch the ARM (`RT_CALL`, S2); from S3 plan B's loop on the core | **generated RISC-V code** (sahw → LLVM IR → rv32imc) issues each command over PCPI; no descriptors | `--iree-sa-emit-riscv` (compiler), `SA_RV` (runtime) | **S1 done**, **S2 accepted on the board** 2026-10-16: the whole Qwen3-0.6B with every dispatch as generated RISC-V code (34039 `RT_CALL`s, no descriptor list), logits bit-exact in 17/17 steps (§9.6, §9.8) |
| R2 / R3 first form (§6) | a test firmware (`firmware/rvb`) | straight-line C generated from a trace, over PCPI | `board_run.py rvb` | done (12 runs bit-identical on the board); a single-dispatch experiment, superseded by plan A's code generator |
| **Model C / R5** (§3, §5) | — (scalar kernels, not a scheduling path) | the core computes itself: kernels (argmax, top-k sampling) compiled by IREE's llvm-cpu for riscv32 | — | not started; needs iree-compile rebuilt with the RISC-V target and llvm-cpu (the same rebuild as plan A's S4) |

How they relate:

- **Plan B changes the sequencer, plan A the command encoding.** Plan B is the ASIC shape "small control core + command DMA": the core decides what runs next, the fetch unit streams the commands. Plan A removes the descriptors: the core's code is the command stream, which needs a real compiler backend (§9) and measures what an ASIC core without a command DMA would have to sustain.
- **They compose.** Plan A's S3 runs inside plan B's loop: the step table marks each dispatch "call" (plan A's function) or "list" (its recorded descriptors), so a dispatch not yet covered by plan A falls back to the list in the same step. The end state is a decode step driven entirely by the core and generated code, the ARM idle.
- **Model C is orthogonal**: it adds scalar computation on the core (today plan B's argmax is a block-max dispatch plus a scan in rt_fw, §7.1), usable from either sequencer.
- **All coexist.** Each is opt-in (the switches above); the default path's binaries, the golden corpus and every existing bundle are unchanged. Plan B and plan A use the existing `_pcpi` overlay (GEMV over PCPI, CAPS bit 26); no path needs a new bitstream. rt_fw v3 serves the list entries, `RT_RUN_SCHED` and (S2) `RT_CALL` side by side.

## 1. What the KV260 overlay provides (checked 2026-10-15)

| Item | Fact | Source |
|---|---|---|
| Core | PicoRV32 rv32imc (MUL, DIV, compressed), `ENABLE_PCPI`, reset at 0xC0000000, stack at 0xC0002000; clocked by `riscv_clk` | `boards/kv260/scripts/pico_processor.tcl` |
| Program memory | 8 KB BRAM at 0xC0000000 (`link.ld`: 0x1E00 for code + data, the stack above), dual-ported to the ARM (`S_AXI_MEM`) | `firmware/common/link.ld` |
| DDR | 0x00000000–0x7FFFFFFF through `M_AXI_DDR` (HP0) on the same AXI interconnect as the BRAM: **the core can execute from DDR**, uncached (every instruction fetch is an AXI read) | `kv260_bd.tcl` address map |
| Accelerator CSRs | 0x80000000 (4 KB), `M_AXI_PERIPH` | `kv260_bd.tcl` |
| Resident firmware | `firmware/rt/rt_fw.c`, 688 bytes: serves the ARM's submission ring, one `mat_submit` per entry, completion records, `mat_notify` | `firmware/rt` |
| Command issue by the core | PCPI custom-0: funct7 = 1 matrix / DMA (`mat_cfg` keys, `mat_load`, `mat_store`, `mat_exec`, `mat_fence`, `mat_perf`, `mat_submit`, `mat_notify`), funct7 = 2 vector (`vec_cfg` keys 0–15 including the fp32 L2 fields, `vec_run`, TRANSPOSE through op 6 + S) | `docs/custom_isa_encoding.md`, `firmware/include/sysarray_intrinsics.h`, `rtl/sysarray/sa_pcpi.v` |
| Command arbitration | legacy sequencer > descriptor fetch unit > PCPI into the same in-order scheduler queue; PCPI commands are held while a list runs | `rtl/sysarray/sa_unit.v` |
| Not reachable by PCPI | **LD mode GEMV** (CAPS bit 25: "descriptors only": strips, C step and x word have no `mat_cfg` keys), and the fetch unit's own constructs (`LOOP_END`, `SETREG`, `LDPARAM`, `CALL` / `RET`, dynamic fields), which a program replaces by its own loops and registers | `sysarray_intrinsics.h` |
| Earlier measurement | M5 on the Z1: descriptor lists run front-end-bound work 1.2–4.2× faster than the same commands issued over PCPI, large GEMMs unchanged | `double_buffer_design.md` §10.5 |
| Toolchain | the system clang builds rv32imc firmware (`firmware/common.mk`); **iree-compile has neither the llvm-cpu backend nor LLVM's RISC-V target** (`IREE_TARGET_BACKEND_LLVM_CPU=OFF`, `LLVM_TARGETS_TO_BUILD=X86`) | `build/iree/build-compiler/CMakeCache.txt` |

## 2. How fast commands must be issued

The cycle model over the decode executables of a 2-layer Qwen3 (`compiler/tests/issue_rate.py build/hfgen/qwen3_l2_e2e_qf`, T = 64, unique executables, four-port overlay):

| Dispatch kind | Commands | Cycles | Cycles per command |
|---|---:|---:|---:|
| GEMV (weight streams) | 7467 | 3.00M | 402 |
| attention (q · Kᵀ, p · V) | 396 | 69.5k | 176 |
| reductions (norms, softmax) | 166 | 26.2k | 158 |
| other elementwise | 158 | 62.8k | 398 |
| all | 8187 | 3.16M | 386 |

The tightest executables need a command every **75 cycles** (p · V) to **90–140 cycles** (the softmax and norm reductions); SmolLM2 (head_dim 64) down to **64 cycles** (p · V), 400 on average; the descriptor fetch unit spends about 24 cycles per descriptor (`perf_model` `f_desc`). A command over PCPI is a few `mat_cfg` / `vec_cfg` writes plus the command itself (a load: 4 + 1 custom instructions; an fp VE: up to 8 + 1) and the address arithmetic: on PicoRV32 (no pipelining, a few cycles per instruction from BRAM, more for a PCPI handshake) that is several tens of cycles from BRAM and several hundred from DDR (each instruction an uncached AXI read). So:
- **code in BRAM** keeps up with GEMV-bound work and is close to the limit for attention and reductions;
- **code in DDR** is too slow for anything but the weight streams;
- an ASIC core needs either tightly coupled instruction memory (or an I-cache) of the size of the hot loops, or a command format the core can issue in one instruction (a descriptor pushed from registers instead of a sequence of `mat_cfg`). These estimates are to be replaced by measurements (step R1 below).

## 3. Execution models

**A. The core runs a dispatch's commands.** The compiler emits one RISC-V function per export instead of (or beside) its descriptor list: BASE / PARAM registers become function arguments and local variables, `LOOP_END` a loop, dynamic fields plain register values, `LDPARAM` a load. The ARM launches it through the ring (a new entry type "call": code address, argument block with the binding addresses and push constants); `rt_fw` calls it and writes the completion record as for a list. Strength: full control flow and no limit of two dynamic fields per command. Weakness on this overlay: issue speed (§2) and GEMV, which only descriptors reach.

**B. The core sequences descriptor lists.** The compiler still emits descriptor lists; the core runs the small control program around them: it computes dynamic values (lengths, offsets, loop counts) with full scalar code, patches or selects lists, and `mat_submit`s them. The heavy command streams keep the fetch unit's rate and GEMV; the core only adds what descriptors cannot express. This is the likely shape of an ASIC design (a scalar core plus a command DMA), and it is what `rt_fw` already does in its simplest form.

**C. The core runs general scalar code** compiled by IREE's llvm-cpu backend for riscv32 (argmax / top-k sampling, small reductions the accelerator does badly), as a host-fallback variant whose host is the PicoRV32 instead of the ARM.

The pre-research should measure A against B on the same dispatches (§5) and demonstrate C on one sampling kernel.

## 4. Mapping `sahw` to the core

| `sahw` | Descriptor path | Model A (RISC-V code) |
|---|---|---|
| `sahw.template` / `body` / `head` / `prefix` | one export's list, the driver's list scheduling | one function `void f(const uint32_t *base, const uint32_t *pc)`; prefix / head as separate functions or dropped (cross-dispatch scheduling becomes a call order) |
| `sahw.base`, `sahw.param` (setup entries: binding + f(push constant)) | patched by the runtime into BASE / PARAM | computed at function entry from the arguments (the `Lin` form: `((pc * mul) >> shift) + add`) |
| `sahw.private` | a PARAM owned by the template | a local variable |
| `sahw.ld` / `st` (LINEAR, INTERLEAVE) | LD / ST descriptor | `mat_cfg_load` / `mat_cfg_store` + `mat_load` / `mat_store`; dynamic fields as register operands |
| `sahw.ld` mode GEMV | LD descriptor (CAPS bit 25) | **not available today**: model B (a one-command list), or the proposed `gemv_run` (§4.1, hardware change) |
| `sahw.ex` | EX | `mat_cfg_exec` (repeat, B / C steps, C row) + `mat_exec` |
| `sahw.ve`, `sahw.transpose` | VE | `vec_cfg` keys 0–15 (only the changed ones: the emitter tracks the configuration state) + `vec_run` |
| `sahw.loop` (`LOOP_END`, dynamic count) | fetch-unit loop, strides added to two PARAMs | a `for` loop; strides added to the variables |
| `sahw.setreg` | SETREG | assignments |
| `sahw.ldparam` | LDPARAM (mem32 × mul + add) | a 32-bit load from DDR (uncached, coherent with the ARM's writes after its flush) |
| `sahw.fence` (mask) | FENCE | `mat_fence(mask)` |
| fence-before flags, list END | | `mat_fence(0)` before return |

### 4.1 GEMV over PCPI (`gemv_cfg` / `gemv_run`, funct7 = 3; RTL and simulation 2026-10-15, not on the board)

GEMV is LD mode 2 with three fields only the descriptor path fills (`xword`, `cstep`, `acc`: descriptor w3[58:34], `pkt_ldg` in `sa_defs.vh`); the PCPI decoder builds LDs with `pkt_ld`, so a GEMV over PCPI would go out with cstep = 0 (a shape error). The proposal gives it its own group, in the style of `mat_exec`: what changes per chunk in the two source registers, what stays per layer in sticky configuration.

| funct7 / funct3 | Mnemonic | Operands | Semantics |
|---|---|---|---|
| 3 / 0 | `gemv_cfg` | rs1 = key, rs2 = value | key 0 `KBYTES`: bytes per strip = K × 16 (a multiple of 256, ≤ 65535; reset 256); key 1 `PITCH`: DDR bytes between strips (reset 256); key 2 `XWORD`: SPAD_A word of x (reset 0); key 3 `CSTEP`: ACC words between strips' results (1 … 255, reset 1). Registers of the PCPI front end (as `mat_cfg`'s LD keys): no wait |
| 3 / 1 | `gemv_run` | rs1 = DDR address of strip 0 (16-byte aligned); rs2 = `acc << 28 \| (strips − 1) << 16 \| C` | One LD in mode GEMV: for s < strips (1 … 8), ACC word C + s · CSTEP (+)= x · W_s, x = KBYTES / 16 int8 from SPAD_A word XWORD, W_s at rs1 + s · PITCH. Same command as the descriptor's (`pkt_ldg`), so the same engine (LD), scoreboard, ports and errors; returns once queued |

- A decode GEMV chunk is then one `gemv_run` (K split: the second block's `acc` = 1 in the same register), against a `mat_cfg` per changed field plus `mat_load` with LD mode keys extended.
- **Implemented** (RTL, simulation): `sa_pcpi.v` decodes funct7 = 3 funct3 0–1 when built with `GEMV` (four configuration registers, the descriptor's `pkt_ldg`; the funct7 = 1 checks now compare the whole funct7, which bit 0 alone no longer identifies); CAPS bit 26 (`sa_legacy.v`, set with bit 25); `sysarray_intrinsics.h`: `SA_CAPS_GEMV_PCPI`, `gemv_cfg`, `gemv_cfg_all`, `gemv_run` (rv32imc output checked: funct7 = 3, funct3 0 / 1). `tb_sa_unit` (`GEN="D=16 DMA_W=128 GEMV=1"`): the descriptor cases repeated over PCPI against the same reference (3 strips C step 2, accumulate with x resident and the x words read 4 then 0, 8 strips K = 256 across the SPAD_A banks, the configuration kept between runs), errors (DDR not 16-byte aligned, K bytes not a multiple of 256: the descriptor's error codes), funct7 = 3 funct3 = 2 unclaimed; without `GEMV` funct7 = 3 is unclaimed (traps). Two GEMV chunks issue as `gemv_cfg_all` once and one `gemv_run` each (about 20 rv32 instructions for the pair, the configuration included). Logic: four registers and the decode; a new bitstream for the board (not built).
- The encodings funct7 = 3 funct3 ≥ 2 and funct7 ≥ 4 stay unclaimed (trap).

The emitter needs the same register setup the serializer has (`SahwSerialize.cpp`) and the VE / EX field packing (`DescList.h`); writing it beside the serializer keeps the two encodings consistent.

## 5. Plan (R1–R5; R2 and R3 first forms done, §6)

| Step | Content | Needs | Estimate |
|---|---|---|---|
| R1 | **Measure the issue cost**: a firmware loop issuing the commands of one recorded dispatch (from a funcsim trace) over PCPI from BRAM and from DDR, against the same list through `mat_submit`; on the board with the performance counters (scheduler starvation, all-idle) | the trace, firmware, the board | 1 day |
| R2 | (first form done 2026-10-15, §6: straight-line from the trace) **`sahw` → C emitter** (model A) with `sysarray_intrinsics.h`, compiled by the system clang to rv32imc: no LLVM rebuild. Functional check on the host: the same C compiled for x86 with the intrinsics redirected to the functional simulator (a C shim over `sa_funcsim`), compared with the descriptor path's results | compiler plugin, a host shim | 2–3 days |
| R3 | (RTL part done 2026-10-15, §6: six dispatches bit-identical) **RTL co-simulation** of one small dispatch (a norm or the scalar index store) on `tb_system` (PicoRV32 + sa_unit), bit-identical to the descriptor path; then the board through a ring "call" entry in `rt_fw` | RTL sim, firmware | 2 days |
| R4 | **Model B**: a control program that computes the T loop's counts and offsets and submits the list (the dynamic fields no longer needed), as a prototype of a scalar core + command DMA | firmware, runtime | 2 days |
| R5 | **Model C**: iree-compile rebuilt with llvm-cpu and the RISC-V target (about 2 hours; the user runs it), one sampling kernel (argmax over the logits) compiled for riscv32 and run on the core | compiler rebuild | 1–2 days |

Acceptance of the C7 stage (§10 of the compiler plan): one dispatch executed by PicoRV32 code issuing PCPI instructions, bit-identical to the descriptor path (R3), and one riscv32 dispatch compiled with IREE llvm-cpu (R5). For the ASIC pre-research the output is the measured issue cost per command kind (R1), the A / B comparison on the decode step's tightest dispatches, and the code size of the hot paths (whether a tightly coupled memory of a few KB is enough).

## 6. First results: real dispatches over PCPI in RTL simulation (2026-10-15)

**Setup.** `compiler/tests/rv_dispatch.py` takes one compiled dispatch, lays it out in the system testbench's DDR window as `dispatch_check.py` does for the functional simulator (template, bindings with generated inputs, the driver's list), runs it on the functional simulator (checked against the oracle, commands traced) and writes the DDR image, the expected image and `rvb_gen.c`: the traced commands as `sysarray_intrinsics.h` calls, configuration keys written only when their value changes (a straight-line program for the given T: the first form of R2; the structural emitter keeping loops and registers is still to do). `firmware/rvb` runs the dispatch twice on the RTL (`firmware/sim/tb_system.v`, mode `RVB_TEST`: PicoRV32 + sa_unit at D = 16 with the GEMV unit, one clock for both as on the KV260's `pl_clk0`): its descriptor list (`mat_submit`), then `rvb_gen()` over PCPI, each timed with `rdcycle`; the testbench compares the bindings with the functional simulator's after each run and restores them in between. The generated code sits in the program BRAM or, `RVB_CODE=ddr`, in DDR, fetched uncached with an extra read latency (`SIM_CPU_DDR_LAT`, cycles; the board's HP0 latency is not measured yet).

Six decode dispatches of SmolLM2-135M (2 layers, `build/hfgen/smollm2_l2_e2e_qf`, T = 64; `build/rvb/results_*.txt`), PCPI time as a multiple of the descriptor list's:

| Dispatch | Commands | Code | Descriptor list | PCPI, code in BRAM | DDR, +20 cycles | DDR, +60 cycles |
|---|---:|---:|---:|---:|---:|---:|
| RMSNorm reduction | 3 | 196 B | 1465 | 0.87 | 1.14 | 2.26 |
| quantization | 6 | 308 B | 1405 | 0.87 | 1.55 | 3.70 |
| q projection (GEMV) | 9 | 212 B | 87162 | 0.99 | 0.99 | 1.00 |
| q · Kᵀ | 54 | 1724 B | 12233 | 0.98 | 1.11 | 2.31 |
| p · V (GEMV + VE) | 57 | 1360 B | 12723 | 0.98 | 1.12 | 1.80 |
| up projection (GEMV + epilogue) | 67 | 1664 B | 241980 | 0.99 | 1.00 | 1.01 |

- **Bit-identical** in all 18 runs (descriptor list, PCPI from BRAM, PCPI from DDR at both latencies) to the functional simulator, including the GEMV chunks issued with `gemv_run` (§4.1).
- **Code in BRAM keeps up**: 0.87–0.99 of the descriptor list's time; on the smallest dispatches PCPI is even faster, the list paying its SETREG / CALL / descriptor fetches.
- **Code in DDR does not**: the issue-bound dispatches (attention, norms, quantization) lose 11–55% at 20 cycles per fetch and 1.8–3.7× at 60; the GEMV-bound ones are unaffected (the weight stream hides the issue).
- **Issue cost**: the generated code needs about 5 PCPI instructions per command for attention (q · Kᵀ: 289 for 54 commands), most of them configuration writes (a VE has up to 16 `vec_cfg` keys); a GEMV chunk is one `gemv_run`.
- **Code size**: 0.2–1.7 KB per dispatch as straight-line code at T = 64; the decode step's 30 executables of a 2-layer SmolLM2 would not fit the 7.5 KB BRAM next to the firmware, and straight-line attention grows with T: the structural emitter (loops) and a TCM / I-cache (§1) are needed beyond single dispatches.
- **Not modeled**: the board's DDR and HP0 latencies (the testbench's DMA model is idealized; the GEMV times are not the board's) and contention between the core's fetches and the unit's DMA. These are R1 on the board.
- Testbench changes kept the other system tests passing (matmul, matmul_insn, gemm, vector, desc_run, rt; desc_run at D = 16 with GEMV).

**On the board (2026-10-15, `_pcpi` overlay, `board_run.py rvb`): all 12 runs bit-identical**, the GEMV cases through `gemv_run` included. PCPI time as a multiple of the list's (rdcycle at 100 MHz):

| Dispatch | List | BRAM | DDR |
|---|---:|---:|---:|
| RMSNorm reduction | 963 | 1.41 | 2.05 |
| quantization | 763 | 2.02 | 3.56 |
| q projection (GEMV) | 5851 | 1.03 | 1.08 |
| q · Kᵀ | 8662 | 1.18 | 1.67 |
| p · V (GEMV + VE) | 3305 | 2.02 | 3.82 |
| up projection (GEMV + epilogue) | 23807 | 1.05 | 1.11 |

The board is less favorable than the testbench (0.87–0.99 from BRAM there): its DMA is faster than the testbench's model, so the lists' times shrink while the core's issue time does not. From BRAM the GEMV-bound dispatches stay within 5%, the issue-bound ones take up to twice as long; from DDR up to 3.8×. The configuration writes per command are the cost an ASIC core must avoid (a one-instruction command, §8).

**The board flow (staged as below).** The same cases relocated: `rv_dispatch.py` writes the generated code relative to the window's base (`rvb_gen(W)`, W from the mailbox) and the list's contents (`rvb_list.json`: SETREG values, BASE ones relative to the window), `board_rvb.py` allocates the window wherever pynq places it, rebuilds the list there with the driver's `DescList`, loads `fw_bram.bin` or `fw_ddr.bin` + the position-independent `code.bin` (no PC-relative references), and compares as the testbench does (`compiler/scripts/deploy_rvb.sh` → `build/deploy_rvb_<config>`, `python3 board_run.py rvb`). The relocated form passes the testbench in both code placements; the rebuilt lists are byte-identical to the simulated ones at the simulated base. `compiler/scripts/build_board_bundles.sh` stages it with the long-context bundles of the `_pcpi` overlay.

So for an ASIC: a small in-order core issuing commands from a tightly coupled memory matches a descriptor DMA's rate on these kernels; from uncached DRAM it does not. The remaining lever is the configuration traffic per command (a one-instruction command format or a few command templates in registers).

## 7. Future plans: a whole Qwen3 driven by the RISC-V core

Today a Qwen3-0.6B decode step is about 1074 dispatches, all on the accelerator, but the ARM schedules them: the IREE VM records a command buffer per step (the push constants and buffer sizes depend on T, so it is re-recorded: 6.5 µs per dispatch on the host, §13.1 of the generality plan), the sa HAL driver (`compiler/runtime/sa/sa_context.c`, `sa_command_buffer.c`) turns each dispatch into a list (SETREG of the BASE / PARAM values from the binding addresses and push constants, CALL of the export's template, END, `sadesc.dispatch_list` builds the same), `rt_fw` runs the lists, and the ARM samples the next token (`sa_llm_run.c`) and detokenizes. Two plans move that scheduling onto the PicoRV32, so that the ARM only loads the model, tokenizes and reads tokens back: plan B keeps the descriptor lists, plan A issues every command from RISC-V code. Plan B is implemented (below), plan A not started.

### 7.1 Plan B: the core schedules the descriptor lists (existing bitstream)

The core becomes the per-step scheduler; the commands stay in the compiled lists (the fetch unit's rate, GEMV through descriptors). The shape of an ASIC with a small control core and a command DMA.

- **Step schedule (compile and initialization time).** The decode step's dispatch sequence is fixed: only T changes, and every buffer is allocated once (the KV cache, the activations, the parameters at fixed addresses in the pool). A tool records one step through the sa driver (the dispatches in order, each with its export's template address and its setup entries: BASE = binding address + f(push constant), PARAM = f(push constant), `Lin` = `((pc · mul) >> shift) + add`, and how each push constant depends on the step: position, T = position + 1 rounded up to 16, constants) and writes a table into DDR: per dispatch the template address, the setup entries with their resolved binding addresses, and the push constant indices; plus the step's inputs (token, position) and the logits buffer.
- **Firmware (`firmware/rvsched`, replacing `rt_fw` for the run).** Per step: write the token and position into the input buffers; for each table entry compute the register values (a multiply, a shift and an add per entry), write them with `mat_cfg` (BASE0–15, PARAM0–7) and `mat_submit` the template (or build a tiny list SETREG + CALL + END in a ring, as the driver does), fencing only where the driver's command buffer has a barrier; then sample. A few hundred bytes of code; the table is data read from DDR.
- **Sampling on the device.** Greedy argmax over the vocabulary (151936 for Qwen3): a scalar scan on the PicoRV32 is about 1.5M cycles (15 ms, 10% of a step); instead a VE max reduction over the logits (one dispatch compiled for it, or appended by the tool) and a scalar search for the index of the maximum in one block. Top-k / sampling with a temperature later (scalar code, model C's llvm-cpu, §3).
- **Prefill**: the same with the prefill function's table (chunks of 16 positions), then decode.
- **Acceptance**: on the board, the token sequence of a prompt identical to the ARM-driven run (`board_llm.py`'s 17-token check), and the logits of the first steps bit-identical (dumped on request); the per-step time against the ARM path (expected: the host part, about 7 ms per Qwen3 step, replaced by about 1–2 µs of core time per dispatch, 1–2 ms per step; the device part unchanged).
- **Effort**: about 3–5 days (the schedule tool from the driver, the firmware, sampling, the board script). **Risks**: the IREE VM's work that is not a dispatch (buffer copies, fills, transfers between allocations) must appear in the table as commands or be removed; a T-dependent buffer size must not change an address (allocations at the maximum, as today).

**Implemented (2026-10-16).** The shape above with three simplifications the code showed possible:
- **The step is recorded, not described.** With the T-loop changes a decode step is pure dispatches (no host copies or fills, `host_copies.py`; `main` has one `stream.cmd.execute`) batched into one or two lists, and the lists depend only on the attention-length bucket (T rounded up to 16, `--pad`): the token and position are data in DDR (LDPARAM). So the driver records each bucket's lists in a **dry mode** (`sa_context_t.record`: a flush copies the list it built and does not run it) while `sa-llm-run --rvsched` calls `main` (and the block maxima, below) once per bucket with throw-away inputs; the inputs' and outputs' physical addresses come from the buffer views. Two passes (`compiler/runtime/sa/sa_rvsched.c`): the first sizes the table, the table is allocated in the device window, the second records the addresses the core will use with the table in place; one more step of the first and of the last bucket must give the same lists (checked). Any host operation or host dispatch in the recorded step is an error.
- **The table** (`firmware/include/rvsched.h`): bucket 0's lists copied once; the 32-bit list words that differ between buckets (SETREG values and CALL targets that depend on T or on a T-sized buffer's address) with their value per bucket; per bucket the addresses of the token, position, valid mask, logits and block maxima. Whole Qwen3-0.6B at 2048 positions: 128 buckets, 2 lists per step (3926 descriptors), **2678 words patched at a bucket change**, a 2.0 MB table; SmolLM2 / Gemma 2-layer: 190 / 210 words.
- **The core's loop** (`rvs_run`, shared by rt_fw and the sim transport's emulation): per step write the token and position (i64) and the valid mask entry (the whole mask at a bucket change, after patching the words), submit the lists one after another (`mat_submit` + `mat_fence`), then the argmax: a second module (`compiler/frontend/argmax_module.py`, `sa_argmax.blockmax`: one f32 max reduction per block of 128–256 logits, on the accelerator; Qwen3 1187 blocks of 128) whose list is part of the step, the core scanning the block maxima and the winning block (the first position with the largest value, as the host's argmax; vocab / block + block uncached reads instead of vocab). rt_fw v3 runs it as ring entry `RT_RUN_SCHED` (1364 bytes, was 688); the ARM writes one entry and waits for the run to end.
- **Coexistence**: everything is opt-in. Without `--rvsched` the driver and rt_fw behave as before (`RT_RUN_LIST` unchanged; the board transport offers schedules only with rt_fw ≥ 3); the model's bundle is not recompiled, `compiler/scripts/deploy_rvsched.sh` adds `argmax.vmfb`, the new rt_fw and sa-llm-run, `board_rvsched.py`.

Checks: on the sim (the transport runs the same `rvs_run` C over the shared window, its lists through the functional simulator) against the host's loop: 2-layer SmolLM2 24 tokens over 2 buckets, 20 after a prefill, 200 over 13 buckets; 2-layer Gemma 40 after a prefill over 3 buckets: tokens identical, the last step's logits bit-exact. The aarch64 build under qemu against the sim: identical tokens. RTL (`firmware/rt`, `make sim`, RING_TEST): two schedules on picorv32 + sa_unit (two buckets, patched words, block maxima with negative values, a stop token) among the ring's entries, 11/11 completions. **On the board (2026-10-16, `_pcpi` overlay, the long-context bundles with `deploy_rvsched.sh`; `run_board_tests.sh` rvsched, `board_run.py rvsched`): all three PASS.** (1) The bundle's 40-token prompt, the 16 generated tokens on the core: tokens identical to the sim's, the last step's logits bit-exact. (2) 256 tokens after the prompt (17 buckets) on the ARM's loop and on the core's: tokens identical, the last logits bit-exact. Time per step of run (2):

| Model | ARM loop | RISC-V core loop | of which the lists (device) | core's own work | lists per step (descriptors) | words patched per bucket | table |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma-3-270M | 62.3 ms | 62.5 ms | 60.2 ms | 2.3 ms | 2 (2660) | 1780 | 332 KB |
| Qwen3-0.6B | 148.3 ms | **145.7 ms** | 143.2 ms | 2.5 ms | 2 (3926) | 2678 | 490 KB |
| Llama-3.2-1B | 231.7 ms | 230.6 ms | 229.3 ms | 1.3 ms | 2 (2002) | 1378 | 256 KB |

The ARM is out of the loop: it records the buckets (0.2–0.4 s for 17 buckets) and waits. The core loop is 2.6 ms per step faster than the ARM's for Qwen3, 1.1 ms for Llama, the same for Gemma: the ARM's per-step cost (the IREE VM's command buffer, the list build, the ring) is gone, and the core spends 1.3–2.5 ms of its own per step. That is almost all the argmax scan: Gemma 1024 + 256, Qwen3 1187 + 128, Llama 501 + 256 uncached 32-bit reads over HP0 at about **170–190 core cycles each** (the inputs, the valid entry and the amortized bucket patches are a few hundred accesses). The lists include the block-max dispatch (about 1 ms). For an ASIC: the control core's load latency to DRAM, not its instruction rate, bounds its scalar work; the argmax's index should come from the accelerator (an index-carrying max reduction, so the core reads one word) or the reduced values should land in core-local memory.

### 7.2 Plan A: the core issues every command (new bitstream, research)

The core runs the whole step as RISC-V code issuing PCPI commands (§3 model A), on top of plan B's scheduler.

- **Prerequisites**: the structural `sahw` → C emitter (R2 in full: loops, registers and dynamic fields kept, so the code does not grow with T); GEMV over PCPI in a bitstream (§4.1, built by the user); code storage: the step's executables are an estimated 100 KB or more of code, beyond the 7.5 KB BRAM, so either fetching from DDR (§6: 1.8–3.7× slower on the issue-bound dispatches at 60 cycles a fetch) or a tightly coupled memory in URAM / an I-cache (a block design change).
- **Expected**: on this board slower than plan B (the issue-bound part of a decode step, about 20% of it, slowed by the fetch latency and the configuration traffic: an estimated 30–60% per step from DDR, close to plan B with a TCM). Its value is the measurements for an ASIC: code size per model, issue rate per command kind over a whole model, the share of configuration writes, and the effect of a one-instruction command format.
- **Effort**: weeks (the emitter, the bitstream, the code placement), after plan B.

## 8. Open questions for an ASIC NPU

- **Command format**: one-instruction issue (a descriptor written from registers, or a command FIFO the core stores to) instead of `mat_cfg` sequences; which fields must be dynamic.
- **Instruction memory**: the hot loops' size (R2 gives it per dispatch) against a TCM or I-cache; PicoRV32's BRAM is 8 KB with `rt_fw` resident.
- **Core performance**: whether a simple in-order core issuing at the §2 rates is enough, or the command DMA must stay (model B).
- **Scalar work on the core**: sampling, small reductions, KV cache bookkeeping, the per-token scales (today the VE's REDUCE and SFU).

## 9. The sahw → LLVM → riscv32 backend (plan A's code generator; plan 2026-10-16)

Purpose: **feasibility**, not speed. Every dispatch the descriptor backend compiles also gets a RISC-V function that issues the same commands over PCPI; the whole model's decode must then run with every dispatch through that code, bit-exact with the descriptor path. Speed, code size and issue cost are recorded for the ASIC questions (§8), not optimized.

### 9.1 Custom instructions: no custom LLVM

The PCPI instructions are R-type instructions in the custom-0 opcode space (`docs/custom_isa_encoding.md`). LLVM's RISC-V assembler accepts any such encoding through the **`.insn` directive** (`.insn r CUSTOM_0, funct3, funct7, rd, rs1, rs2`), so they are emitted as inline assembly: `firmware/include/sysarray_intrinsics.h` already does exactly this for the firmware, and the same header serves the generated code. An unmodified LLVM (the system clang, or IREE's own once rebuilt with the RISC-V target) compiles it. What inline assembly costs: the compiler treats each instruction as opaque (`volatile`, a memory clobber), so it neither reorders nor schedules them; for issuing commands in program order that is the required semantics anyway. A custom LLVM (the instructions in TableGen, as intrinsics) would only matter for an ASIC: scheduling the configuration writes, a register-file model of the command fields, or proposing an ISA extension. Not needed here.

### 9.2 Lowering

A pass on each compiled `sahw.template` (after the descriptor serialization, which it reuses) emits one function per export:

    void <export>(const uint32_t *bindings, const uint32_t *push_constants)

- **Registers become variables.** `sahw.base` = binding address + f(push constant) and `sahw.param` = f(push constant), f(c) = add + ((c · mul) >> shift) in 64-bit arithmetic as the driver's setup table (`sa_loader.c`); `sahw.private` starts at 0. No limit of 16 BASE / 8 PARAM registers.
- **Commands become one call each**: `sa_rv_cmd(desc, base, p0, p1)` with the command's 64-byte descriptor as the serializer encodes it (a constant), the value of its BASE register and of its one or two dynamic-field PARAMs. The runtime function (`firmware/include/sa_rv_rt.h`) does what the fetch unit does with a descriptor (dynamic fields replaced or added, the DDR address relocated, FENCE_BEFORE) and issues it over PCPI (`mat_cfg` / `vec_cfg` / `gemv_cfg` keys, then `mat_load` / `mat_store` / `mat_exec` / `vec_run` / `gemv_run` / `mat_fence`: the mapping proven on the board by the R2 / R3 programs, `rv_dispatch.py`). S1 keeps this split, the encoding in one place; the configuration writes can later be specialized at compile time (static fields folded) without changing the interface.
- **Control flow is compiled**: `sahw.loop` (`LOOP_END`: the count, possibly from a PARAM, read when it is first reached; k1 += s1, k2 += s2 after every pass) becomes a loop; `sahw.setreg` assignments (with their own dynamic fields and add mask); `sahw.ldparam` a `mat_fence` (the value may come from a store just issued) and a 32-bit load from DDR, times mul plus add. The template's prefix, head and body run in order (the driver's overlap of prefixes is an optimization of the list path).
- **Output**: LLVM IR built directly with `llvm::IRBuilder` in a private `LLVMContext` (`compiler/plugins/sa/transforms/SahwToRiscv.cpp`; thread-safe inside the parallel serialization, no MLIR dialect registration), without a target triple or data layout: the IR only uses i32 / i64 and the two pointer arguments, so the same file compiles for riscv32 (the generated code) and for the host (the checks).

### 9.3 Build stages

| Stage | LLVM IR → object | Rebuild |
|---|---|---|
| S1–S3 | `iree-compile ... --iree-sa-emit-riscv=<dir>` writes one `.ll` per export; the system `clang-18 --target=riscv32 -march=rv32imc -O2` compiles it with `sa_rv_rt.h`, `ld.lld` links | none |
| S4 | in-process: iree-compile rebuilt with `LLVM_TARGETS_TO_BUILD="X86;RISCV"` (and lld) emits the riscv32 object itself into the sa-desc code section; the same rebuild with `IREE_TARGET_BACKEND_LLVM_CPU=ON` also enables R5 (scalar kernels for the core through llvm-cpu) | the user runs it (about 2 h) |

### 9.4 Runtime (S2, S3)

- **sa-desc**: an optional code section (one entry offset per export); a loader without it ignores it, the descriptor templates stay as they are (the golden corpus compares them; the option is off by default).
- **rt_fw**: ring entry `RT_CALL` (function address, argument block: binding addresses and push constants); the code lives in DDR (speed does not matter; the 8 KB BRAM holds rt_fw).
- **Driver**: `SA_RV=1` (or a list of exports) sends a dispatch as `RT_CALL` instead of SETREG + CALL in a list; dispatches without code fall back to the list. Plan B's step table can mark each dispatch as "call" or "list", so the whole decode loop runs on the core.

### 9.5 Verification

1. **Command-stream equivalence (S1, host)**: per dispatch, the `.ll` compiled for the host with the PCPI instructions replaced by a recorder (`SA_RV_HOST`) issues a sequence of PCPI operations; the descriptor template run on the functional simulator with the same bindings and push constants (`dispatch_check`'s layout, `--trace`) gives the executed commands, mapped to PCPI by the same rules. The two sequences must be identical, for every dispatch of the reference exports.
2. **riscv32 build**: every `.ll` compiles for rv32imc; code size per dispatch and per model recorded.
3. **RTL** (S2): a few dispatches on `tb_system` (PicoRV32 + sa_unit), as R3.
4. **Board** (S2 one dispatch, S3 the whole decode): tokens identical and logits bit-exact against the descriptor path.

### 9.6 S1 results (2026-10-16)

Implemented: `compiler/plugins/sa/transforms/SahwToRiscv.cpp` (the lowering, behind `--iree-sa-emit-riscv=<dir>` in `SATarget::serializeExecutable`), `firmware/include/sa_rv_rt.h` + `firmware/rv/sa_rv_rt.c` (the runtime: 1.3 KB of rv32imc), `compiler/tests/rv_equiv.py` + `rv_host_main.c` (the equivalence harness: each dispatch's descriptor path on the functional simulator against its generated function on the host, at T = 16 and 304; the LDPARAM reads replayed in order and their addresses compared, since a dispatch may load values it stored before).

| Corpus | Dispatches | PCPI streams identical | rv32imc code (generated functions) |
|---|---:|---:|---|
| Qwen3-0.6B long-context bundle (`qwen3_lc2048_pcpi`, D = 16, 4 GEMV ports, max 2048) | 1132 | **1132** | 1.45 MB in all, median 344 B, max 57 KB |
| golden corpus (all cases: D = 8 / 16, EX and GEMV paths, micro-kernels on / off, max 256–2048) | 7708 | **7682** (+ 26 the corpus records as having no template: `hf_smollm2_q`'s host-fallback dispatches, failing identically without the option) | 31.0 MB in all, median 780 B, max 301 KB |

The default build is unchanged (option off: the golden corpus byte-identical). Observations for S2 / the ASIC questions: the generated code is the descriptor stream's control flow (LLVM fully unrolls short `LOOP_END`s, e.g. the q_proj GEMV's 3 passes) plus one 64-byte constant per command, so its size follows the templates': most dispatches are under 1 KB, but straight-line prefill templates reach 300 KB (a template's descriptors are 64 bytes each, the code adds a call per command); a whole Qwen3 decode needs about 1.5 MB, so the code lives in DDR (§9.4), and specializing the configuration writes at compile time (§9.2) would replace the 64-byte constants by the few fields each command really sets.

### 9.8 S2 results (2026-10-16)

Implemented: `rv_dispatch.py --code compiled` (firmware/rvb calls an export's generated function: BRAM, or DDR position independent: `-mcmodel=medany -fno-jump-tables -mno-relax`, no unwind tables), rt_fw v4's `RT_CALL` entry, `compiler/tests/rv_image.py` (a model's functions + runtime as one position-independent `rvcode.bin` + map, and `rvcode_host.so` for the sim), the driver's `SA_RV=<dir>` (`sa_rv.c`: an export with a function is flushed out of the batch and sent as `RT_CALL` with its raw binding addresses and push constants; the sim calls the host build, each command run on the functional simulator as a one-command list), `deploy_rvcode.sh` / `board_rvcode.py`, `deploy_rvb.sh RVB_GEN=compiled`.

| Check | Result |
|---|---|
| RTL (`RVB_TEST`): the six SmolLM2 dispatches (norm reduction, quantization, q / up projections on GEMV, q · Kᵀ, p · V) as their generated functions | **bit-identical in BRAM and in DDR mode**; time against the list ×1.01–4.75 (BRAM), ×1.10–31 (DDR, 40 cycles per instruction fetch) |
| RTL (`RING_TEST`): `RT_CALL` of a function in DDR | 12/12 completions |
| sim: SmolLM2 2-layer, prefill + 12 decode steps, `SA_RV` | all 1536 dispatches as RISC-V code, logits identical to the descriptor path |
| **board** (`run_board_tests.sh rvbc`): the six dispatches, BRAM and DDR | **12/12 runs bit-identical** |
| **board** (`board_rvcode.py`, Qwen3-0.6B long-context bundle): the 40-token prompt (3 prefill chunks) + 16 decode steps, `SA_RV` | **34039 dispatches as RISC-V code, 0 descriptor lists; logits bit-exact with the sim in 17/17 steps, tokens identical** |

Speed on the board (not the aim of plan A): decode 1434 ms per step against 138 ms with descriptor lists (×10.4), prefill 36.7 s for the 3 chunks against 2.6 s. Every command costs the core its 64-byte descriptor read from DDR, the dynamic fields applied in 64-bit software arithmetic, all of its configuration keys written over PCPI and the instruction fetches from uncached DDR (the 1.4 MB image cannot sit in the 8 KB BRAM); the accelerator mostly waits. These are the numbers §8 asks for: without a command DMA a control core needs local instruction memory and a command format it can issue in a few instructions; the next step on the compiler side is specializing the configuration writes at compile time (§9.2: only the fields a command really sets, static ones as immediates), which removes most of the per-command work.

### 9.7 Coexistence

Every part is opt-in: the compiler option (off: identical output), the sa-desc section (absent: identical files), `SA_RV` (unset: the list path), rt_fw's new entry (as `RT_RUN_SCHED`), the `_pcpi` overlay (no new bitstream). Existing bundles and board tests are unchanged.

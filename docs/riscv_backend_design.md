# C7: a RISC-V backend for the sa accelerator (design, 2026-10-15)

Status: **design and pre-research only**. C7 is the optional stage of
[`iree_compiler_plan.md`](iree_compiler_plan.md) §8.10. Its purpose here is to
prepare a future ASIC NPU: a small programmable core next to the
accelerator that issues its commands and runs the scalar code the
accelerator cannot (control flow over data, sampling). The KV260 overlay
already has such a core (PicoRV32), so the questions an ASIC design would
ask can be measured on it before any silicon decision: how fast must the
core issue commands, where must its code live, and which parts of a model
should it run.

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

### 4.1 Proposed: GEMV over PCPI (`gemv_cfg` / `gemv_run`, funct7 = 3)

GEMV is LD mode 2 with three fields only the descriptor path fills (`xword`, `cstep`, `acc`: descriptor w3[58:34], `pkt_ldg` in `sa_defs.vh`); the PCPI decoder builds LDs with `pkt_ld`, so a GEMV over PCPI would go out with cstep = 0 (a shape error). The proposal gives it its own group, in the style of `mat_exec`: what changes per chunk in the two source registers, what stays per layer in sticky configuration.

| funct7 / funct3 | Mnemonic | Operands | Semantics |
|---|---|---|---|
| 3 / 0 | `gemv_cfg` | rs1 = key, rs2 = value | key 0 `KBYTES`: bytes per strip = K × 16 (a multiple of 256, ≤ 65535; reset 256); key 1 `PITCH`: DDR bytes between strips (reset 256); key 2 `XWORD`: SPAD_A word of x (reset 0); key 3 `CSTEP`: ACC words between strips' results (1 … 255, reset 1). Waits while a list runs (as `mat_cfg`) |
| 3 / 1 | `gemv_run` | rs1 = DDR address of strip 0 (16-byte aligned); rs2 = `acc << 28 \| (strips − 1) << 16 \| C` | One LD in mode GEMV: for s < strips (1 … 8), ACC word C + s · CSTEP (+)= x · W_s, x = KBYTES / 16 int8 from SPAD_A word XWORD, W_s at rs1 + s · PITCH. Same command as the descriptor's (`pkt_ldg`), so the same engine (LD), scoreboard, ports and errors; returns once queued |

- A decode GEMV chunk is then one `gemv_run` (K split: the second block's `acc` = 1 in the same register), against a `mat_cfg` per changed field plus `mat_load` with LD mode keys extended.
- Hardware: `sa_pcpi.v` decodes funct7 = 3 funct3 0–1 (four configuration registers, `pkt_ldg`); a CAPS bit (26, "GEMV over PCPI"); `sysarray_intrinsics.h` wrappers `gemv_cfg` / `gemv_run`; `tb_sa_unit` cases issuing the same GEMVs over PCPI and as descriptors (results, scoreboard order against a following VE, errors: cstep 0 is no longer reachable, bad alignment, strips 0 impossible by encoding). Logic: four registers and the decode, tens of LUTs; a new bitstream for the board.
- The encodings funct7 = 3 funct3 ≥ 2 and funct7 ≥ 4 stay unclaimed (trap).

The emitter needs the same register setup the serializer has (`SahwSerialize.cpp`) and the VE / EX field packing (`DescList.h`); writing it beside the serializer keeps the two encodings consistent.

## 5. Plan (not started)

| Step | Content | Needs | Estimate |
|---|---|---|---|
| R1 | **Measure the issue cost**: a firmware loop issuing the commands of one recorded dispatch (from a funcsim trace) over PCPI from BRAM and from DDR, against the same list through `mat_submit`; on the board with the performance counters (scheduler starvation, all-idle) | the trace, firmware, the board | 1 day |
| R2 | **`sahw` → C emitter** (model A) with `sysarray_intrinsics.h`, compiled by the system clang to rv32imc: no LLVM rebuild. Functional check on the host: the same C compiled for x86 with the intrinsics redirected to the functional simulator (a C shim over `sa_funcsim`), compared with the descriptor path's results | compiler plugin, a host shim | 2–3 days |
| R3 | **RTL co-simulation** of one small dispatch (a norm or the scalar index store) on `tb_system` (PicoRV32 + sa_unit), bit-identical to the descriptor path; then the board through a ring "call" entry in `rt_fw` | RTL sim, firmware | 2 days |
| R4 | **Model B**: a control program that computes the T loop's counts and offsets and submits the list (the dynamic fields no longer needed), as a prototype of a scalar core + command DMA | firmware, runtime | 2 days |
| R5 | **Model C**: iree-compile rebuilt with llvm-cpu and the RISC-V target (about 2 hours; the user runs it), one sampling kernel (argmax over the logits) compiled for riscv32 and run on the core | compiler rebuild | 1–2 days |

Acceptance of the C7 stage (§10 of the compiler plan): one dispatch executed by PicoRV32 code issuing PCPI instructions, bit-identical to the descriptor path (R3), and one riscv32 dispatch compiled with IREE llvm-cpu (R5). For the ASIC pre-research the output is the measured issue cost per command kind (R1), the A / B comparison on the decode step's tightest dispatches, and the code size of the hot paths (whether a tightly coupled memory of a few KB is enough).

## 6. Open questions for an ASIC NPU

- **Command format**: one-instruction issue (a descriptor written from registers, or a command FIFO the core stores to) instead of `mat_cfg` sequences; which fields must be dynamic.
- **Instruction memory**: the hot loops' size (R2 gives it per dispatch) against a TCM or I-cache; PicoRV32's BRAM is 8 KB with `rt_fw` resident.
- **Core performance**: whether a simple in-order core issuing at the §2 rates is enough, or the command DMA must stay (model B).
- **Scalar work on the core**: sampling, small reductions, KV cache bookkeeping, the per-token scales (today the VE's REDUCE and SFU).

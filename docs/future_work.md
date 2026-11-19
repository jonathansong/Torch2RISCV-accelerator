# Future work

Status (2026-11-19): plan, nothing here is started unless noted.

Where the project stands:
- Release `v2.4-kv260` ([`results_v2.4.md`](results_v2.4.md)): four dense models and one
  MoE model on the KV260, all executables on the accelerator, bit-exact with the
  functional simulator.
- Decode runs at about 91% of the 6.4 GB/s weight-streaming peak (Llama-3.2-1B,
  [`results_v2.3.md`](results_v2.3.md)).
- Cross-operator fusion is closed ([`fusion_design.md`](fusion_design.md) §7): what
  is left is worth a few percent at most.

The remaining levers are therefore fewer weight bytes, the hardware, the missing
functions (MoE prefill, new model families), and the engineering around the
compiler. This note lists them with their expected gain, cost and first step, and
proposes an order.

## 1. Performance

| Item | Expected gain | Effort | Basis |
|---|---|---|---|
| **W4A8 weights** | decode ×1.48 (Qwen3, performance model); upper bound by GEMV share ×1.55–1.87 | 3–5 weeks | [`generality_implementation_plan.md`](generality_implementation_plan.md) §17.6 |
| **MoE: one kernel over a layer's k experts** | up to about 18% of a Granite decode step | 1–2 weeks | [`moe_native_lowering_design.md`](moe_native_lowering_design.md) §22, §24 |
| **MoE: router + Top-k in one RISC-V program** | up to 8.8% of a Granite step (24 Top-8 calls, 0.38 ms each) | days | same, §25 |
| **PicoRV32 instruction memory (32–64 KB)** | plan A (rv32 code) on par with descriptor lists | 1–2 weeks (RTL) | [`riscv_backend_design.md`](riscv_backend_design.md) §9.10–§9.11 |
| H2 / LD command pipelining | ×1.04–1.05 alone, ×1.89 stacked with W4A8 | medium | `generality_implementation_plan.md` §17.6 |
| Wider LD / GEMV | medium | large | LUT 87%, BRAM 138 / 144 on the current overlay |

### 1.1 W4A8

Decode is weight-streaming bound, so halving the bytes per step is the only lever
close to 2×. Llama-3.2-1B's weights go from 1.24 GB to about 0.65 GB.

Order of work, each step gated on the previous one:
1. **Quality on the host.** A W4A8 checkpoint (GPTQ / AWQ-style, per-group
   scales) through `quality_eval.py` against fp32, on the same protocol as the W8A8
   tables (2 × 2048 WikiText-2 tokens, board-style KV cache). Go on only if top-1
   and KL stay close to W8A8.
2. **Numeric contract.** Int4 weights unpacked to int8 in the GEMV path. Per-group
   scales need a K split with a rescale per group, or per-channel int4 if the
   quality allows it, which is cheaper in hardware.
3. **Functional simulator, then RTL**: unpacking in `sa_ld` / `sa_gemv`.
4. **Compiler**: packing of int4 tables, the GEMV kernel's group rescale.
5. **Board**: bit-exact with the simulator; the speed against §17.6's prediction.

### 1.2 MoE kernel over k experts

At I = 512 (Granite) an expert's gate + up is only 1 MB, so each FFN's
unoverlapped tail (last-chunk gating, amax, int8 output: about 4.8K cycles) is 23%
of its stream. One kernel that streams expert i + 1's weights during expert i's tail
recovers it. It needs:
- a region shape in `FormMoeRegions` behind a flag;
- k dynamic bases in one dispatch (2 PARAMs per expert).

Acceptance: logits byte-equal with the split form.

### 1.3 Instruction memory for the control core

Plan A's gap to descriptor lists is instruction fetch from DDR, about 27 cycles per
instruction. Since executable generalization, the decode working set is 20–41 KB
per model. A 32–64 KB instruction memory, BRAM or URAM (URAM is unused today),
would let generated RISC-V code issue commands at list speed. That turns the
RISC-V path into the default and opens device-side control flow: the decode loop,
sampling, MoE routing.

## 2. Functions

| Item | What it adds | Effort |
|---|---|---|
| **MoE prefill (M6: rows grouped by expert)** | Granite prefill; also fixes the export's memory (a 24-layer export with prefill rows traces past 10 GB) | 2–3 weeks |
| **FP16 GEMV** ([`fp16_gemv_design.md`](fp16_gemv_design.md)) | unquantized FP16 checkpoints in decode (Qwen2.5-0.5B prototype, 50 MHz, two ports) | 3–4 weeks (stages F1–F6) |
| Shared experts and sigmoid routers (Qwen2-MoE, DeepSeek-style) | today `moe_rewrite` raises `NotImplementedError` | 1–2 weeks |
| New model families (Phi, Mistral, …) | the most direct measure of generality: count the frontend and compiler changes each needs | days each |
| Contexts beyond 2048, batch > 1 | long-document use; serving several requests | medium to large |

**MoE prefill.** Today a prefill chunk's rows each run their own k experts, one
row at a time. Grouping a chunk's rows by expert means:
- a device-side histogram and gather of the rows per expert;
- one GEMM per expert, on the EX path;
- a scatter-add of the weighted results.

Each expert's weights then stream once per chunk instead of once per row. Plan in
[`moe_implementation_plan.md`](moe_implementation_plan.md) §M6.

## 3. Compiler engineering

| Item | Why | Effort |
|---|---|---|
| **CI** | every compiler change is checked by hand today | 2–3 days |
| **Executable generalization upstream** (iree-org/iree #25059, design issue #25124) | removes patch 0003 from the local stack | depends on review |
| Patch 0005 upstream (`insert_slice` → store fold) | a miscompile in IREE, fixed only locally | days |
| Rebase onto a current IREE | pinned at `e4a3b0405d` (3.11.0) plus five patches; the cost grows with time | 1 week or more |
| Lifetime-based local memory planning (3a / 3b) | prefill and long context; designed in [`local_memory_planning.md`](local_memory_planning.md) | 2–3 weeks |

**CI.**
- **Tier 1, every push, hosted runner:** Python unit tests and synthetic frontend
  tests.
- **Tier 2, every push:**
  - fetch the pinned IREE and apply the patches;
  - build `iree-compile` with ccache, cached per IREE revision and patch set;
  - run lit and the golden corpus.
  
  This needs the corpus (`build/golden`, about 15 MB) moved into the repository,
  e.g. `compiler/tests/golden/`. A deliberate codegen change then shows in the
  diff.
- **Tier 3, nightly or on demand, self-hosted on the development machine:**
  end-to-end runs on the simulator, `dispatch_check` and the RTL testbenches.
  - Only on `schedule` / `workflow_dispatch` events on `main`, never on pull
    requests: the repository is public.
  - Each job under the memory cap.
- **Board runs stay manual.**

**Upstream.** The generalization pass is to be presented at the IREE community
meeting (2026-11-25); the level it belongs at (Flow before dedup, Stream, or region
formation) is the open question in #25124. Patch 0004, the plugin hook after
dispatch region formation, is the other candidate once there is interest.

## 4. Measurements and releases

| Item | Why |
|---|---|
| ARM A53 baseline on the KV260 | a like-for-like comparison point (the same models with an int8 CPU runtime); the only one so far is from the PYNQ-Z1 |
| Power and energy per token | the board has power monitors; energy per token is the natural figure for an edge accelerator |
| Release `v2.5-kv260` | after the next result lands (W4A8, MoE prefill or FP16): a frozen `results_v2.5.md` and a tag, as for v2.2–v2.4 |

## 5. Proposed order

1. **CI** (days). It protects everything after it.
2. **The MoE small items**: router + Top-k in one program, then the k-expert
   kernel. About a quarter faster for Granite, with no hardware change.
3. **One large item**, chosen by goal:
   - **performance**: W4A8 (§1.1), starting with the host quality check;
   - **FP16 workloads**: the FP16 GEMV stages F1–F3, then F4–F5.
4. **MoE prefill (M6).**
5. **Instruction memory** for the control core, then device-side control flow.
6. **Release v2.5** once 3 and 4 are on the board.

Running alongside, as the opportunities come:
- the IREE rebase and the upstreaming of 0003 / 0005;
- new model families as generality tests;
- the ARM and power measurements.

## 6. Risks

- **FPGA resources.** The current overlay uses 87% of LUTs and 96% of BRAM. W4A8
  unpacking is small, but wider ports, the instruction memory and FP16 lanes
  compete for the same budget; variant overlays (as for FP16) keep the main one
  intact.
- **W4A8 quality.** Per-group scales may be needed for acceptable quality, and they
  cost hardware (a rescale per group). The host check comes first for that reason.
- **IREE drift.** Each month on the pinned revision makes the rebase larger; the
  patches that are not upstream (0003–0005) are the risk.
- **Bit-exactness under new numerics.** FP32 accumulation (FP16) and group rescales
  (W4A8) make the order of operations part of the semantics; each design fixes it
  explicitly so the simulator stays bit-exact (see `fp16_gemv_design.md` §4).

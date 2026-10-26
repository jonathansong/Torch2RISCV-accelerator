# MoE learning path

A guided route through the mixture-of-experts (MoE) implementation in this
repository: from what a sparse MoE layer computes, through the W8A8 frontend
block, the preprocessing and region passes, the backend's device-computed
expert base and the Top-k as embedded RISC-V code, to verification, the board
and the open problems. It is written against tag `v2.4-kv260` (2026-10-25):
Granite-3.1-1B-A400M (24 layers, Top-8 of 32 experts) decoding on the KV260,
bit-exact with the functional simulator, 104.4 ms per step (9.58 tok/s; 8.98
tok/s with the runtime's own overhead), PPL 6.528 against fp32's 6.386.

Like the [hardware](hardware_learning_path.md), [LLM](llm_learning_path.md)
and [compiler](compiler_learning_path.md) learning paths, each stage pairs
outside reading with the files that put the idea into practice, a lab and
questions to check yourself. This path assumes the compiler path's stages
0–7 (MLIR, IREE, the `sa` plugin, `sahl` / `sahw`) or equivalent; stage 0
below lists exactly which parts are used.

The canonical design record is
[`moe_native_lowering_design.md`](moe_native_lowering_design.md) (§1–§25,
referred to as "native §N" below). The earlier, abandoned path is
[`moe_implementation_plan.md`](moe_implementation_plan.md) (M0–M5, in
Chinese with English milestone logs; "plan §N"). The release record is
[`results_v2.4.md`](results_v2.4.md).

The MoE history in one paragraph: **M0–M4** (2026-10-21..24) built MoE by
hand: a numpy / PyTorch reference, a packed parameter file, hand-written
descriptor lists for the expert FFN, firmware Top-1 / Top-2 kernels, one
Mixtral layer at a time on the board. **M5** (2026-10-24) moved it into the
compiler, but as a template that replaced the whole module: one MoE layer
per VMFB, side-car weights, a Q15 route on RISC-V the sim could not run.
**The native lowering** (S0–S8, 2026-10-25) threw that away: experts are
ordinary SA dispatches in the same module as everything else, selected by a
dynamic DDR base the device computes. Then a trained checkpoint (Granite,
native §21–§25), Top-k up to 64 experts, the board and the quality run, all
on 2026-10-25. The M0–M4 code is preserved at tag `moe-m0-m4-final`.

---

## How to use this path

- **Order.** Stage 0 is prerequisites; stages 1–2 are the model side (what
  MoE computes, how HF writes it, how this repo rewrites it); stage 3 is the
  design decisions and why the first attempt failed; stages 4–7 are the
  compiler (preprocessing, regions, backend, Top-k); stage 8 verification;
  stages 9–10 the board, performance and quality; stage 11 open problems as
  projects. Do 1–7 in order; 3 can be read at any point after 2.
- **Each stage has:** *Goal* · *Read* · *In this repo* · *Lab* · *Check
  yourself*.
- **Environment.** `source compiler/env.sh`; Python is `$SA_PY`
  (`build/iree/venv/bin/python`). The board's configuration is
  `SA_D=16 SA_GEMV_PORTS=4`; MoE builds also pass `--iree-sa-form-moe`
  (not on by default in `compile_sa.sh`: give it through `SA_COMPILE_FLAGS`).
- **Memory.** 15 GB of RAM. Exports and compiles of a full model run under
  `systemd-run --user --unit=<name> -p MemoryMax=10G -p MemorySwapMax=0
  --working-directory=$PWD …`, one at a time. A Granite export *with*
  prefill rows does not fit at all (native §24): use `--prefill 0`.
- **Bundles already built** (use them instead of rebuilding when a lab only
  reads): `build/moe/native/tm` (tiny-mixtral, the S6/S7 bundle),
  `build/moe/native/tm_topk` (the same after the Top-k rewrite),
  `build/moe/granite/l2` (Granite, 2 layers), `build/moe/granite/granite24`
  (24 layers, decode only, bit-exactness / speed), `build/moe/granite/granite24q`
  (24 layers, quality bundle), `build/moe/granite_proxy/gp` (Granite-shaped
  Top-2 proxy). Board copies: `build/deploy_moe_{tm,granite24,granite24q}`.
- **Checkpoints:** `build/llm_cache/TitanML-tiny-mixtral` (random weights:
  functional and speed checks only), `build/llm_cache/granite-3.1-1b-a400m-base`.

## MoE at a glance

```
 HF model: MixtralSparseMoeBlock / GraniteMoeMoE  (data-dependent loop over experts: not exportable)
   │ hf_generic.moe_rewrite  ->  QMixtralMoe (W8A8)
   │   router QLinear padded to 16k rows (-inf on padding) -> torch.topk(k)
   │   -> exp / sum / reciprocal (no linalg.softmax)
   │   -> for k: index_select(table, id_k) -> SwiGLU expert (_qlinear x3) -> w_k * y_k, summed
   ▼
 qllama.mlir + qllama.irpa (int8 tables [E, N, K], scales [E, N]) + compile_flags.txt
   │ iree-compile --iree-sa-form-moe  (preprocessing, SATarget.cpp)
   │   PackLinearWeights: rewriteExpertGather   gather at a Top-k id -> extract_slice T[id]
   │                      packedWeights          T -> linalg.pack [E, N/D, K, D], evaluated at compile time
   │   FormMoeRegions:    Top-k region (sa.moe.kind = "topk", sa.moe.top_k), readIdsAsI32
   │   FormFfnRegions:    one SwiGLU region per selected expert (the dense pass, unchanged)
   │   isolate-moe-experts (k > 2): optimization_barrier per expert term ... dropped after
   │                      region formation by drop-moe-isolation (IREE patch 0004 hook)
   ▼
 per decode MoE layer: router GEMV | Top-k (rv_only) | softmax (2 VE dispatches)
                       | k x expert FFN (expandFfn) | k x down (+ combine)
   │ sa backend: memref.load ids[c] + subview offset  ->  Lowerer::expertId -> tagged BASE
   │   applyExpertBases: FENCE; P = id * stride (LDPARAM, or LD -> VE mul -> ST -> FENCE -> LDPARAM);
   │                     every DMA on the tagged BASE gets a dynamic slot (DDR field += P)
   │ Top-k: sahl.topk -> sahw template with sa.riscv_only -> SahwToRiscv buildRawTopk (rv32 code)
   ▼
 sa.vmfb: descriptor lists for everything, embedded rv32 code for the Top-k exports only
   │ runtime: rv_only exports always run their RISC-V code (sa_rv.c required=true);
   │          the sim runs the same code compiled for the host (rvcode_host.so)
   ▼
 KV260: 25 descriptor lists + 24 Top-8 calls per Granite decode step
```

---

## Stage 0 - Prerequisites from the other paths

**Goal.** Have the dense pieces MoE reuses in your head before you start;
the MoE implementation adds surprisingly little on top of them.

**Read** (in this order, skipping what you know):
- LLM path: the decoder layer and SwiGLU FFN; W8A8 (`QLinear`: int8 rows,
  per-row f32 scales, per-token activation quantization `_quant_act`).
- Hardware path: the GEMV unit (streaming weights through LD mode 2, 4 ports,
  57.95 B/cycle measured = 5.8 GB/s at 100 MHz); descriptors, BASE / PARAM
  registers, **LDPARAM** (`mem32[addr] * mul + add`, 16-bit `mul`) and the
  two **dynamic slots** per descriptor (a field replaced or added to from a
  PARAM, applied by `sa_cmdfetch.v`).
- Compiler path: stage 5–7 (bufferize → `sahl` kernels → `sahw`), the
  `ddrOf` helper in `SahlLower.h`; [`fusion_design.md`](fusion_design.md) §4
  (SwiGLU fusion: `FormFfnRegions`, `expandFfn`, overlapped chunks);
  [`riscv_backend_design.md`](riscv_backend_design.md) §0, §9 (plan A,
  embedded rv32 code, `rv_image.py`, RT_CALL); the packing of dense weights
  (`PackLinearWeights`, `linalg.pack` with inner tile D, evaluated at compile
  time into `sa_packed.irpa`).

**Check yourself.**
- A decode linear layer of `N × K` int8 weights at D = 16: what is the
  packed layout, and how many bytes does the GEMV stream per token?
- What can a dynamic slot do that a PARAM alone cannot? How many slots does
  one LD have?
- What does `rv_image.py` produce, and which of its outputs does the sim use?

## Stage 1 - What a sparse MoE layer computes

**Goal.** Know the arithmetic and the routing semantics exactly, and why
MoE decode on this machine is a pure weight-streaming problem.

**Read.**
- Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer* (2017): gating, Top-k, load balancing.
- Lepikhin et al., *GShard* (2020) and Fedus et al., *Switch Transformers*
  (2021): Top-2 / Top-1 routing, capacity factor, expert parallelism.
- Jiang et al., *Mixtral of Experts* (2024): the architecture implemented
  here (8 experts, Top-2, SwiGLU experts, softmax over the selected logits).
- IBM's Granite 3.0 / 3.1 model card and technical report for the
  `granite-3.1-1b-a400m` MoE (32 experts, Top-8, I = 512).
- Optional: Dai et al., *DeepSeekMoE* (2024) (shared + fine-grained
  experts; not supported here, stage 11); Gale et al., *MegaBlocks* (2022)
  (grouped / block-sparse GEMM for training and prefill; the idea behind M6).

**In this repo.**
1. [`moe_implementation_plan.md`](moe_implementation_plan.md) §4: the
   layer as pseudo-code and the three compilation problems: a
   data-dependent weight address, k expert passes and their sum, and in
   prefill an uneven number of tokens per expert.
2. Native §4 "Semantics": Mixtral softmaxes all E logits, takes the Top-k
   and renormalizes. That equals the softmax over the k selected logits.
3. Native §21 (Granite's shape) and §18 / §22 (bytes per step).

**Lab.**
1. Prove the routing identity numerically:
   ```python
   import torch
   s = torch.randn(32)
   p = torch.softmax(s, -1); v, i = p.topk(8); a = v / v.sum()      # HF: softmax, Top-k, renormalize
   v2, i2 = s.topk(8); b = torch.softmax(v2, -1)                     # this repo: Top-k, softmax
   assert torch.equal(i, i2) and torch.allclose(a, b)
   ```
   Then write the softmax as `QMixtralMoe.route` does (`exp(v - v[0])`,
   sum, reciprocal) and explain why no max is needed.
2. Compute, for Granite (H = 1024, I = 512, E = 32, k = 8, 24 layers,
   vocabulary 49152, tied embedding), the int8 bytes of: one expert (gate +
   up + down), all experts of the model, the experts read per decode step.
   Add the attention projections and the classifier and compare with the
   429 MB per step and 67 ms stream bound of native §22. Repeat for
   tiny-mixtral (H = 1024, I = 3584, E = 8, k = 2, 2 layers): native §18
   says about 82 MB and 14.1 ms.

**Check yourself.**
- Why is decode MoE bound by the GEMV's weight stream, and what fraction of
  Granite's weights does one token read?
- Why does prefill change the picture (several tokens per expert, so
  matrix–matrix work again)? Why is "every token through its own k experts"
  correct but wasteful?
- Is the renormalized Top-k equal to "softmax then Top-k" without
  renormalization? Which models use which (look at the HF router classes)?

## Stage 2 - From HF's MoE block to an exportable W8A8 block

**Goal.** Understand why HF's MoE code cannot be exported, and every
choice in `QMixtralMoe` that makes the exported graph compile for the SA.

**Read.**
- transformers 5.x: `MixtralSparseMoeBlock`, `MixtralTopKRouter`,
  `MixtralExperts` (3-D `gate_up_proj [E, 2I, H]`, `down_proj [E, H, I]`),
  `GraniteMoeMoE`. Note the loop over experts with data-dependent masks.
- PyTorch: `torch.export` and data-dependent control flow (why
  `nonzero` / boolean masks break tracing).
- Native §4 (the whole section) and §11.1 (what the naive form compiled to:
  almost everything on the host).

**In this repo.**
1. `compiler/frontend/hf_generic.py`: `_qlinear` (the shared W8A8 product),
   `ROUTER_PAD`, `MOE_ROUTERS`, `QMixtralMoe` (`__init__`, `route`, `_row`,
   `forward`), `moe_rewrite`. About 120 lines; read every one.
2. The same file, `export()`: `compile_flags.txt` gets
   `--iree-flow-inline-constants-max-byte-length=1` for MoE models only
   (native §13 "Constant inlining").
3. `quantize()`: Granite's `logits_scaling` folded into the classifier's
   row scales (native §23 "Frontend").
4. `compiler/tests/test_hf_moe.py`: the synthetic check (E = 4, H = 64,
   I = 128) and the checkpoint check (`--model`, `--layers`, `--export`).
   Read the pitfall in native §12.2: the HF reference must run *before*
   hf_generic patches transformers.

**Lab.**
```sh
source compiler/env.sh
$SA_PY compiler/tests/test_hf_moe.py --export build/moe/synth   # synthetic, keeps the export
grep -c 'torch.aten.topk' build/moe/synth/qllama.mlir           # 2 layers x (1 + 16 + 16) rows = 66
grep -c 'softmax' build/moe/synth/qllama.mlir                   # 0
cat build/moe/synth/compile_flags.txt
```
Then, in a Python shell, build `QMixtralMoe` from one layer of
tiny-mixtral and check: the router has 16 rows, rows 8..15 are zero and
`pad[8:]` is `-inf`; `gq.shape == (8, 3584, 1024)`, `gq.dtype == int8`.
Break one choice at a time and look at the exported MLIR: slice the scores
to E before the Top-k; use `torch.softmax`; compute the router in fp32.

**Check yourself.**
- Why pad the router to 16 rows with `-inf` instead of slicing the scores
  to E? (Two reasons: the dequantization's innermost size, and the Top-k
  recognition, native §4, §11.1.)
- Why can't a padded lane ever be selected? What happens if a real score is
  NaN? (Commit `b181104`: Top-k never selects NaN.)
- Why `index_select(0, i)[0]` with a one-element `i`, and what does IREE do
  with it if the compiler does nothing (native §11.1: an i32 copy of the
  whole expert, then an i64 matmul on the host)?
- Why are prefill rows processed one at a time, and why does that make a
  24-layer Granite export with prefill blow past 10 GB?
- Why does `_quant_act(x)` run only once per row although router and every
  expert call it?

## Stage 3 - The design: why native lowering, and what M0–M5 taught

**Goal.** Be able to argue each of the five key decisions of the native
lowering, and recognize the failure modes of the earlier path.

**Read.**
- Plan §1.1 (M0–M4 results) and §M5s/§M5t (the compiler-owned HAL chain).
- Native §1 (the seven problems of M5, as a table), §2 (scope), §3 (the
  key decisions), §8 (stages and gates), §10 (risks).

**In this repo.**
1. `git ls-tree -r --name-only moe-m0-m4-final | grep -i moe`: the M0–M4
   reference path (`frontend/moe.py`, `moe_hf.py`,
   `runtime/tools/moe_m3b.py` with its hand-written expert descriptor list,
   `firmware/moe_top2`, the board scripts). Skim `moe_m3b.py`: it is what the
   compiler now generates.
2. `git log --oneline -- docs/moe_native_lowering_design.md`: one commit per
   stage S0–S8, each with its gate.

**Lab.** For each of native §1's seven M5 problems, write one sentence
naming the native mechanism that removes it (for instance problem 4,
side-car weights → expert tables are module parameters packed at compile
time; problem 6, Q15 route not runnable on the sim → softmax on the VE,
Top-k as compiler-generated code the sim executes).

**Check yourself.**
- Why is "one executable per expert" the wrong design (code size, IREE
  dispatch count, generalization), and why is "run all experts and mask" the
  wrong one?
- Decision 2 says expert selection is "a dynamic view, not a new kernel".
  Which existing kernels then compile the experts unchanged?
- Why must the ARM never read the routing result? (One host round trip per
  layer per token, and the descriptor list split it causes.)
- Why does only the Top-k run as RISC-V code, and why was keeping
  `SA_RV=embedded` for the whole program unacceptable (native §1 problem 7)?

## Stage 4 - Preprocessing: expert gathers to packed slices

**Goal.** Follow an expert weight from the exported gather to a slice of a
packed table, read on the device at a dynamic offset.

**Read.**
- MLIR: `tensor.extract_slice` (rank-reducing), `linalg.pack`
  (`inner_dims_pos`, `inner_tiles`), constant folding of initializers in IREE
  (`--iree-opt-const-eval`, parameter export).
- Native §5.1, §11.2 (the id stays in the dispatch in all three placements),
  §11.3 (packing at compile time), §13 (S2 results).

**In this repo.**
1. `compiler/plugins/sa/target/PackLinearWeights.cpp`:
   - `rewriteExpertGather` (around line 150): the match conditions (constant
     table, one-element index, body a single `tensor.extract`, the index
     coming from a sort's / topk's indices). Find the check that keeps the
     decode embedding gather out.
   - `expertSlice`, `packedWeights` (around line 209–230): a weight that is a
     slice of a constant `[E, N, K]` int8 table packs the whole table
     (`T$packed : [E, N/D, K, D]`) and reads `extract_slice Tp[id, 0, 0, 0]`.
2. Lit: `compiler/plugins/sa/test/sa_pack_expert_weights.mlir` (a gate weight
   and its scales at a sort's index become packed slices; a gather at a
   non-Top-k index is unchanged).
3. `compiler/tests/moe_s0_spike.py`: the S0 spike's four hand-written
   variants (`static`, `dyn`, `sliceout`, `noregion`).

**Lab.**
```sh
source compiler/env.sh
$IREE_BUILD/llvm-project/bin/llvm-lit -v compiler/plugins/sa/test/sa_pack_expert_weights.mlir
# the RUN line shows the pass pipeline; run it by hand and read the output
grep RUN compiler/plugins/sa/test/sa_pack_expert_weights.mlir
```
Then check in the tiny-mixtral bundle that the packing really happened at
compile time: list the parameter names of `build/moe/native/tm/sa_packed.irpa`
(`layers.{0,1}.mlp.{gq,uq,dq}$packed`, 29,360,128 bytes each; `rq$packed`
16 KiB) with IREE's `iree-dump-parameters` or the `iree.runtime` Python
API, and grep the dispatch sources (`sa_sources/`) for
`dispatch.tensor.load … offsets = [%…, 0, 0, 0]`.

**Check yourself.**
- After packing, what is the byte stride between two experts of a gate
  table? Of its scale table? (`N * K` and `N * 4`.)
- Why must the gather rewrite require the index to come from a Top-k, given
  that the embedding lookup has exactly the same shape?
- Why did the router fall back to the host until the inline-constant
  threshold was set to 1 (native §13), and why is that flag written only for
  MoE exports?

## Stage 5 - Regions: Top-k, experts, and keeping epilogues apart

**Goal.** Know which dispatches one MoE layer becomes, which pass forms
each, and the two IREE behaviours the passes had to work around.

**Read.**
- IREE: dispatch region formation, elementwise fusion of producers into
  consumers, `util.optimization_barrier`, and how Stream splits command
  buffers.
- Native §5.2, §5.3 (the dispatch table), §14 (S3: ids read as i32), §23
  ("isolate-moe-experts"), §24 ("Command buffer splits").

**In this repo.**
1. `compiler/plugins/sa/target/FormMoeRegions.cpp` (read the header comment
   first):
   - the Top-k anchor: an `iree_linalg_ext.topk` or the recognized Torch
     sort prefix, wrapped in its own region with `sa.moe.kind = "topk"` and
     `sa.moe.top_k = k`; the limits (1 ≤ k < E, k ≤ 16, 4 ≤ E ≤ 64);
   - `rewriteTorchTop2Sort` / `readIdsAsI32`: the ids read as i32 because
     `index_cast(extsi(x)) == index_cast(x)`; without it IREE made one
     host dispatch per id extension ("i64 scalar computation");
   - `iree-sa-isolate-moe-experts` (k > 2): a barrier on each expert's term,
     so IREE does not fuse 8 down-projection epilogues and their sum into one
     dispatch of 30 bindings (the SA has 15 BASE registers);
   - `iree-sa-drop-moe-isolation`: removes those barriers after region
     formation (the IREE patch 0004 hook), because in Stream each barrier
     split the command buffer: 217 lists per step → 25, 141.4 → 104.4 ms.
2. `FormFfnRegions` (fusion_design.md §4): unchanged, forms one SwiGLU
   region per selected expert.
3. `compiler/plugins/sa/target/SATarget.cpp`,
   `extendPreprocessingPassPipeline`: the order of the passes.
4. Lit: `sa_form_moe_regions.mlir` (incl. `@expert_id_as_index` and
   Granite's Torch spelling), `sa_isolate_moe_experts.mlir`.

**Lab.**
1. Run both lit tests with `-v`, then run their RUN lines by hand.
2. Count the decode dispatches of one MoE layer in a real build. In
   `build/moe/granite/l2/sa_sources`, identify: the router GEMV, the Top-8,
   the softmax's reduction and elementwise dispatches, the expert FFN
   executable (called 8 times per layer), the down executables. Compare with
   native §5.3's table (written for k = 2).
3. On the flow IR of a compile with `--mlir-print-ir-after=iree-flow-…` (or
   the codegen report), confirm there is no `flow.tensor.load`: the ids never
   leave the device (native §10 R1).

**Check yourself.**
- Why does tiny-mixtral need 3 down executables but Granite (k = 8) a
  different structure? Where does IREE fuse the combine?
- Why did the S3 change make the two expert FFNs separate executables
  (different static offsets into the ids buffer), and what does that cost?
- Why did wrapping each expert's chain in an explicit dispatch region fail,
  where barriers worked (native §24: the `collapse_shape` before the
  epilogue)?

## Stage 6 - The backend: a dynamic expert base computed on the device

**Goal.** Understand how `table[id]` becomes descriptors that add
`id × stride` to the DDR address of every weight and scale DMA, without the
host, and the exactness and register constraints that come with it.

**Read.**
- Native §6 (representation, the id → byte offset, PARAM budget, slots),
  §11.4 (where codegen stopped before S4), §11.5 (real-size templates), §15
  (S4 results, incl. a pre-existing reduction bug it uncovered).
- Hardware path: LDPARAM, dynamic slots and their composition in
  `sa_cmdfetch.v` (two add slots on one field apply one per cycle, each
  writing the descriptor word back).

**In this repo.**
1. `compiler/plugins/sa/transforms/SahlLower.h` (around line 185–200):
   `expertId`, `applyExpertBases`, the tagged BASE in `ddrOf`.
2. `compiler/plugins/sa/transforms/SahlToSahw.cpp`: `expertIdLoad`,
   `Lowerer::expertId` (a subview offset that is `index_cast(memref.load
   ids[c])` of an i32 / i64 DDR buffer), and `applyExpertBases`:
   1. a FENCE at the start of the body (the ids are the previous dispatch's
      output; such dispatches give up the head prefetch);
   2. one P = id × stride per distinct (ids address, stride): one LDPARAM
      with `mul = stride` if stride ≤ 65535; otherwise LD the id into ACC,
      a VE multiply by `float(stride)`, ST into the first word of an
      output-only binding, FENCE, LDPARAM;
   3. every DMA on the tagged BASE re-pointed at the real BASE, plus a
      dynamic slot (add mode, P) on its DDR field; refused (host fallback)
      when both slots are taken or a slot is in replace mode.
   Note the `MapVector`: with a `DenseMap` the prologue's order followed
   pointer hashes and golden caught it (native §17).
3. `compiler/plugins/sa/transforms/SahlLowerGeneric.cpp`: `packedRowGather`
   (the same id → offset trick, older, for the embedding), and the 1-D
   reduction fix (VALID = n, ROWLEN = ⌈n / D⌉) that the softmax's sum over
   2 exposed.
4. `compiler/sim/sa_funcsim.py`: LDPARAM and the slot application, to see
   the semantics the RTL implements.
5. Lit: `sahl_expert_slice.mlir` (stride 256 → one LDPARAM; 131072 → VE
   path; 2^24 + 1 → refused as inexact).

**Lab.**
1. Run `sahl_expert_slice.mlir`, read the `sahw` output of the 131072 case
   and point at: the FENCE, the LD of the id, the VE op, the ST, the
   LDPARAM, and the slot on a looped GEMV LD.
2. For Granite's gate (N = 512, K = 1024), up (same), down (N = 1024,
   K = 512) and their scales, decide by hand which path each takes and
   whether the exactness check passes (every `e × stride` for e < E exact in
   f32, i.e. odd part < 2^24, and < 2^31). Do the same for tiny-mixtral
   (3,670,016 = 7 × 2^19). Then construct an (E, N, K) at D = 16 that fails.
3. Read the compiled descriptors of one tiny-mixtral expert FFN
   (`build/moe/native/tm/sa_binaries`, parsed with the library
   `compiler/runtime/tools/sadesc.py`) and count PARAMs in use; compare with
   native §15's table (router 0, FFN 2, down 1, down + combine 4, of 8).

**Check yourself.**
- Why does the VE path go through an f32 multiply at all? (No integer
  multiplier on the VE; LDPARAM's `mul` is 16 bits.)
- Why must P be stored into an *output* binding's first word, and why is
  that safe (the dispatch writes that word later)?
- A looped GEMV LD already uses one slot for the loop's PARAM. What happens
  to a DMA that already uses both? Where is that refusal?
- What does the leading FENCE cost, and why is it needed?
- In plan A (all exports as RISC-V code), what does the same `sahw` program
  become? (A load and a 32-bit multiply: no special case.)

## Stage 7 - Top-k as embedded RISC-V code

**Goal.** Know how the only non-descriptor part of an MoE layer is
generated, executed on the board, and executed bit-exactly on the sim.

**Read.**
- Native §3 decisions 3–5, §7 (the plan), §16 (S5: what was actually
  done; the sim needed no semantic record), §23 (Top-k of up to 64,
  `buildRawTopk`).
- [`riscv_backend_design.md`](riscv_backend_design.md): RT_CALL, the
  argument block, `sa_rv_ldparam`, the host build for the sim
  (`SA_RV_SIM`), CPI ~27 when fetching from DDR (§9.10: why instruction
  count matters).
- The IEEE-754 trick for ordering floats as unsigned integers.

**In this repo.**
1. `compiler/plugins/sa/dialect/SahlOps.td`: `sahl.topk` (attribute `k`).
   `SaToSahl.cpp` forms it; `SahlToSahw.cpp` records `sa.moe.raw_topk` (E),
   `sa.moe.top_k` and `sa.riscv_only` on the template.
2. `compiler/plugins/sa/transforms/SahwToRiscv.cpp`, `buildRawTopk` (around
   line 419): each score → an unsigned key in f32 order (±0 one key); a
   descending list of k (key, id) entries on the stack; a score enters only
   when it beats the k-th key; ids scanned upwards with a strict compare, so
   ties keep the lower id. Reads through `sa_rv_ldparam`, writes through
   `sa_rv_store32` (plain volatile accesses with `--iree-sa-riscv`, calls in
   the portable IR).
3. `SATarget.cpp` (around line 269–282): without `--iree-sa-riscv`, only
   `sa.riscv_only` exports get embedded rv32 code.
4. Runtime: `compiler/runtime/sa/sa_rv.c` (`sa_rv_lookup(…, required, …)`),
   `sa_loader.c` (`rv_only`): an `rv_only` export runs its code whatever
   `SA_RV` says; `SA_RV_ONLY=rv_only` keeps every other export on
   descriptor lists. `firmware/rv/sa_rv_rt.c`, `firmware/include/sa_rv_rt.h`:
   `sa_rv_store32` for the PicoRV32, the recorder and the sim.
5. `compiler/tests/oracle.py`, `op_iree_linalg_ext_topk`: the reference
   (ordered `>`, equal values by id).
6. Lit: `sahl_moe_topk.mlir` (Top-8 of 32).

**Lab.**
1. Write the key transform and the insertion in Python, and check it
   against `torch.topk` on random scores, on scores with exact ties, with
   `-0.0` and `+0.0`, with `-inf` padding, and with a NaN. Decide what your
   version does with NaN, then find what `buildRawTopk` does.
2. Count instructions: the old fully unrolled Top-2 network at Top-8 of 32
   was about 5,000 instructions (native §23). Estimate the new kernel's
   dynamic instruction count for E = 32, k = 8 and, at CPI ≈ 27 and
   100 MHz, its time. Compare with the measured 0.38 ms per call (native
   §25), which also includes the RT_CALL and the list split around it.
3. Run the Top-k dispatch of `build/moe/granite/l2` through
   `dispatch_check.py` with and without `--rv build/moe/granite/l2/rv` and
   see the "RISC-V only" report without it.

**Check yourself.**
- Why is an exact comparison on bit patterns bit-exact between the
  PicoRV32 and the host build, while M5's Q15 softmax was not?
- Why did M5's Top-2 only ever run on the board (raw pointers to physical
  addresses), and what replaced the raw accesses?
- Why does the descriptor body of an `rv_only` export contain only a RET?
- Each Top-k call ends a descriptor list. How many lists per Granite decode
  step does that give, and what is the 25th?

## Stage 8 - Verification: bit-exact where it is defined

**Goal.** Run the repository's verification ladder on an MoE build and
understand why "device ≡ host model" is *not* one of the gates for a model
with near-tie routing.

**Read.**
- Native §7 (gates), §15–§17 (per-dispatch, sim e2e, golden), §23
  ("Granite, 2 layers"), §24 ("The checks against HF and the host model").
- CLAUDE.md "Verification".

**In this repo.**
1. `compiler/tests/dispatch_check.py`: `--ids LIST` (fills the bindings read
   as indices: force every expert, the last one = the highest DDR offset),
   `--rv DIR` (runs `rv_only` exports' host build), `--dirty`, `--pos 0`,
   `--t T`.
2. `compiler/tests/test_moe_sim.py`: runs A (decode, Top-k as RISC-V
   code), B (prefill: byte-equal to A), C (plan A: byte-equal to A), and
   the host model reference; writes `<bundle>/s6/*.npy`, the board's
   reference.
3. `compiler/tests/golden.py`: cases `moe_mixtral` and `moe_granite`.
4. `compiler/tests/test_moe_compiler_riscv.py`: router → Top-k.

**Lab.**
```sh
source compiler/env.sh
B=build/moe/native/tm
$SA_PY compiler/tests/dispatch_check.py $B/sa_sources $B/sa_binaries --t 16 --dirty --ids 7,0 --rv $B/rv
$SA_PY compiler/tests/dispatch_check.py $B/sa_sources $B/sa_binaries --t 16 --pos 0 --ids 2,5 --rv $B/rv
compiler/scripts/run_tests.sh golden
```
(`sa_binaries/` holds the bundle's compiled executables, from
`--iree-hal-dump-executable-binaries-to`.) Then, under `systemd-run` with
the memory cap, run `test_moe_sim.py --bundle $B --model
build/llm_cache/TitanML-tiny-mixtral` and read its three byte-equality
results. Optional: repeat with `--ids` covering 31 on `build/moe/granite/l2`
(`--ids 7,0,31,12,5,19,26,3`) at T = 16, 32, 48, 64.

**Check yourself.**
- The device matches the host W8A8 Granite in argmax only 20 / 28. Why is
  that not a bug? (The host against *itself* with 1e-6 noise on the router
  inputs gives 22 / 28: 68 of 672 layer-steps have the 8th and 9th score
  within 0.01.) Which checks then carry bit-exactness?
- Why are the expert ids not read back from a device run, and what covers
  them instead (native §17)?
- What did golden catch that no other test would have (native §17)?
- Why did dense-model golden cases stay byte-identical through every MoE
  change, and what guarantees MoE recognition never fires on a dense model?

## Stage 9 - The board and where the time goes

**Goal.** Deploy and run an MoE bundle on the KV260, and account for every
millisecond of a Granite decode step against the weight-stream bound.

**Read.**
- Native §18 (tiny-mixtral on the board), §20 (v2 measured: why merging
  dispatches saves 0.08%), §22 (the Granite proxy estimate), §24 (Granite on
  the board, the pool, the command buffer splits), §25 ("Speed in context").
- [`results_v2.4.md`](results_v2.4.md).
- [`kv260_upgrade_plan.md`](kv260_upgrade_plan.md) K5-M (the u-dma-buf pool,
  now 1536 MiB; CMA 128 MiB).

**In this repo.**
1. `compiler/scripts/deploy_moe.sh` (the window = sim peak in MiB + 32; the
   MB-vs-MiB bug fixed in `9797dbf`), `compiler/scripts/run_board_tests.sh`
   (bundle `moe`: `llm`, `llm --decode-only`).
2. `compiler/scripts/moe_granite_proxy.sh`, `build/moe/granite_proxy/run.sh`:
   how a model the backend could not yet compile was priced from a proxy with
   its shapes.
3. `compiler/tests/perf_model.py`, `compiler/scripts/perf_trace.sh`,
   `compiler/tests/moe_v2_bound.py`.
4. Board results: `build/deploy_moe_granite24/results/35db39f/…`,
   `build/deploy_moe_granite24q/results/15dcf0e/…` (the `profile` run).

**Lab.**
1. Model a tiny-mixtral decode step from a sim trace (native §20):
   ```sh
   SA_RV=build/moe/native/tm/rv SA_RV_ONLY=rv_only DEPLOY=build MODEL=deploy_moe_tm MB=320 OUT=moe_tm \
       compiler/scripts/perf_trace.sh
   $SA_PY compiler/tests/moe_v2_bound.py --trace build/perfmodel/moe_tm.trace \
       --cal build/perfmodel/qwen3g_cal.json
   ```
   Read the per-dispatch efficiency against the 64 B/cycle stream bound
   (FFN 90%, down 92–97%).
2. Build Granite's step budget from the `profile` result (111.3 ms = device
   103.2 ms (25 lists 94.1 + 24 Top-8 calls 9.1) + runtime 5.8 ms) and from
   native §22's table (experts' FFN 41%, downs + combine 25%, attention /
   norms / router 25%, classifier 9%). Where are the 94.1 − 67 = 27 ms over
   the stream bound?
3. If you have the board: `deploy_moe.sh build/moe/granite/granite24`, copy,
   and run `run_board_tests.sh` as CLAUDE.md describes; compare your step
   time with 104.4 ms.

**Check yourself.**
- Why is Granite's device at 68–72% of the stream bound while dense
  Llama-3.2-1B reaches about 91%? (I = 512: an expert's gate + up is only
  1 MB, so each FFN's unoverlapped tail, 4.8K cycles, is 23% of its stream.)
- Why did merging tiny-mixtral's expert dispatches save nothing (the
  batched runtime already runs them in one list; the tails are dependencies,
  not dispatch overhead)? Why could the same idea be worth ~18% on Granite?
- Why does decode slow from ~104 ms to 157.7 ms on average over a
  2048-token window?

## Stage 10 - Quality: routing under quantization

**Goal.** Measure an MoE model's quantization loss, and separate the
router's own quantization error from the error of the hidden states it sees.

**Read.**
- Native §10 R5 (router precision), §21 (Granite on the host), §25 (on the
  board).
- LLM path: perplexity, top-1 agreement, KL against fp32; teacher forcing
  on the board (`board_teacher.py`, `board_quality.py`).

**In this repo.**
1. `compiler/tests/quality_eval.py --moe-routing`: per MoE layer and
   position, the experts each variant selects against fp32's, and the Top-k
   of an fp32 router on the variant's *own* hidden state.
2. `build/quality/granite_moe_2048.json` (host), `granite_board_2048.json`
   (board).
3. The quality bundle's export flags (native §25): `--max-len 2048
   --kv-channel k --kv-calib build/llm_cache/wikitext-2-raw-v1-train.txt:2048
   --prefill 0`.

**Lab.** Read the two JSON files and reproduce native §21's table: PPL,
top-1, KL, experts in common with fp32 (97.7%), same set of 8 (81.7%),
W8A8 vs fp32 router on the same input (99.3%). Then explain in two
sentences why a changed expert set moves the output little at Top-8 (it
usually swaps one low-weight expert).

**Check yourself.**
- Routing differs from fp32 at 2.3% of the experts. Is the router's
  quantization to blame? What experiment answers it?
- Why are tiny-mixtral's numbers functional only (random weights), and
  what made Granite the right trained checkpoint (Apache-2.0, native in
  transformers, Mixtral's routing semantics, no shared expert, fits the
  pool at int8)?
- Granite's first position differs from fp32 by 137% (rel L2), the others by
  about 20%. Why, and why does it not show in the window's PPL?

## Stage 11 - Open problems as projects

Each is a real next step, with its measured upper bound. Pick one, write the
design note first (the repository's habit: estimate, then build, then
measure against the estimate).

1. **A kernel over a layer's k experts** (native §9, §20, §22; up to ~18% of
   a Granite step). Stream expert i + 1's gate/up weights during expert i's
   tail (last chunk's gating, amax, int8). Starts from `expandFfn`'s
   overlapped schedule (fusion_design.md §4.9); needs a region shape in
   `FormMoeRegions` behind a flag and k dynamic bases in one dispatch (PARAM
   budget: 2 per expert; slot budget per DMA unchanged). Acceptance:
   byte-equal logits with the split form.
2. **The Top-k calls** (8.8% of the step, 0.38 ms each). Options: run the
   router GEMV and the Top-k from one RISC-V program (no list split), a
   cheaper kernel (fewer DDR instruction fetches), or a VE-side Top-k.
   Measure first with `board_rvperf.py` / `rv_perf.py`.
3. **M6: grouped prefill** (plan §M6). Group a chunk's rows by expert so
   each expert's weights stream once per chunk; a grouped-GEMM kernel on the
   EX path; also fixes the export's memory (rows traced one at a time).
   Read MegaBlocks first.
4. **Shared experts and sigmoid routers** (Qwen2-MoE, DeepSeek-style):
   today `moe_rewrite` raises `NotImplementedError`. A shared expert is a
   dense SwiGLU next to the routed ones; a sigmoid router changes `route`
   and the semantics check.
5. **W4A8 experts** (the dense path's next lever, CLAUDE.md): MoE decode is
   the most weight-bound workload here, so halving the expert bytes is the
   largest single gain available; the dynamic base's stride halves too.
6. **Expert offloading beyond the pool**: a model whose experts exceed the
   1536 MiB pool (expert caching in DDR, prefetch by predicted routing).
   Research-level on this board.

## Reading map (files by stage)

| Stage | Docs | Code | Tests |
|---|---|---|---|
| 2 | native §4, §12 | `compiler/frontend/hf_generic.py` (`QMixtralMoe`, `moe_rewrite`, `export`) | `test_hf_moe.py` |
| 3 | native §1–§3, plan §1.1, §M5 | tag `moe-m0-m4-final` | |
| 4 | native §5.1, §11, §13 | `target/PackLinearWeights.cpp` | `sa_pack_expert_weights.mlir`, `moe_s0_spike.py` |
| 5 | native §5.2–§5.3, §14, §23, §24 | `target/FormMoeRegions.cpp`, `SATarget.cpp` | `sa_form_moe_regions.mlir`, `sa_isolate_moe_experts.mlir` |
| 6 | native §6, §15 | `transforms/SahlLower.h`, `SahlToSahw.cpp`, `SahlLowerGeneric.cpp`, `sim/sa_funcsim.py` | `sahl_expert_slice.mlir`, `dispatch_check.py --ids` |
| 7 | native §16, §23 | `dialect/SahlOps.td`, `SaToSahl.cpp`, `SahwToRiscv.cpp` (`buildRawTopk`), `runtime/sa/sa_rv.c`, `sa_loader.c`, `firmware/rv/sa_rv_rt.c` | `sahl_moe_topk.mlir`, `oracle.py`, `test_moe_compiler_riscv.py` |
| 8 | native §17, §24 | | `dispatch_check.py --rv`, `test_moe_sim.py`, `golden.py` |
| 9 | native §18, §20, §22, §24, results_v2.4 | `deploy_moe.sh`, `run_board_tests.sh`, `moe_granite_proxy.sh` | `perf_model.py`, `moe_v2_bound.py` |
| 10 | native §21, §25 | | `quality_eval.py --moe-routing`, `board_quality.py` |

(`target/`, `transforms/`, `dialect/`, `test/` are under
`compiler/plugins/sa/`; tests without a path are under `compiler/tests/`.)

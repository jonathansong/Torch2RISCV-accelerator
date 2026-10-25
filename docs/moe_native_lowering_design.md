# Native MoE lowering: Mixtral experts as ordinary SA dispatches

> Status (2026-10-25): S0–S8 done; **accepted on the KV260** (§18): tiny-mixtral, prefill + decode, bit-exact with the sim, decode 16.0 ms per step. M5's chain removed (§19). v2 measured and not pursued (§20: at most 0.1% of a step). A trained checkpoint, Granite-3.1-1B-A400M (32 experts, Top-8), evaluated on the host (§21): W8A8 quality in line with the dense models; its compilation needs Top-k > 2 and E > 16 in the backend. Decode estimated at about 100–110 ms per step on the board (§22). The backend takes Top-k of up to 64 experts since §23: the 2-layer Granite runs end to end on the sim, bit-exact across its forms. **Granite (24 layers, Top-8 of 32) accepted on the KV260 (§24)**: decode bit-exact with the sim, 104.4 ms per step (9.58 tok/s). **Quality on the board (§25)**: PPL 6.528 against fp32's 6.386, top-1 92.1%, KL 0.030 (2 × 2048 tokens). Released as `v2.4-kv260` ([`results_v2.4.md`](results_v2.4.md)).
> It replaces M5's compiler-owned
> HAL chain (`LowerMoeChain`, [moe_implementation_plan.md](moe_implementation_plan.md)
> §M5s/M5t) with a lowering that keeps the rest of the model in the same module.
> It must land before M6 (grouped prefill) and full-model MoE generation.

## 1. Why M5's path cannot be extended

M5 was accepted on the KV260 for one Mixtral layer (`TitanML/tiny-mixtral`
layer 0, `dim=1024`, `hidden=3584`, 8 experts, Top-2), with a one-input VMFB.
The path works, but most of the work happens outside the compiler, and it
cannot share a module with anything else:

| # | M5 mechanism | Consequence |
|---|---|---|
| 1 | `LowerMoeChain.cpp:236` clears the whole module and parses a text template (`MoeChainTemplate.inc`, `MoeModelTemplate.inc`); `:96` requires exactly one recognized MoE | Attention, other layers, embedding and classifier cannot be in the same VMFB. A full model, or M6's model integration (P4b), cannot use it. |
| 2 | `FormMoeRegions.cpp:234` requires exactly three expert gathers in the whole function and `:355` sets the attributes on the function | A function with two or more MoE layers is not recognized. |
| 3 | The FFN executable is `moe_m3b.make_executable` (a Python descriptor list, `moe_device_bundle.py:176`); router and Top-2 binaries are picked out of `--iree-hal-dump-executable-binaries-to` by file-name regexes; the route is a text-substituted fixture; six `iree-compile` runs | The compiler packages hand-written code. None of the SwiGLU fusion (`expandFfn`: overlapped chunks, per-bank placement) is reused, and no codegen guard (golden, `dispatch_check`) covers it. |
| 4 | The module's fp32 expert weights are discarded; the device reads `moe_params.bin`, packed separately by `moe_hf.pack_layer`; only shapes and the stride are checked | A stale or mismatched parameter file compiles and runs and gives wrong results. |
| 5 | The router matmul is the VMVX host fallback (fp32, on the ARM) | The ARM is back in the middle of every MoE layer (M4f had the router on the SA). |
| 6 | The route weights are an integer Q15 softmax in RISC-V code (`SahwToRiscv.cpp` `buildRouteTop2`); the functional simulator cannot run RISC-V and builds the route record on the host in fp32 | Board and sim are not bit-exact (board vs W8A8 reference within 6.3e-4, tolerance 3e-3); every dense model in this repository is bit-exact. |
| 7 | An export marked `rv_only` needs `SA_RV=embedded` (`sa_loader.c:253`), which switches the whole program to plan A | A MoE model runs every dense dispatch through plan A, which is slower (Qwen3 decode 163 vs 136 ms per step, [riscv_backend_design.md](riscv_backend_design.md) §9.10). |

## 2. Goal and scope

**Goal**: an exported W8A8 Mixtral decoder compiles with
`compiler/scripts/compile_sa.sh` like the four dense models. Each MoE block
lowers to ordinary SA dispatches inside the same module as attention and the
other layers. Expert selection is a dynamic DDR base that the device resolves,
so the ARM never reads routing results. Board and sim logits are byte-equal.

**In scope (v1)**

- Mixtral-style block: router linear, Top-2, softmax over the two selected
  logits, gated SwiGLU experts, weighted sum. 4 to 16 experts, D = 16, batch 1.
- Decode (one row). Prefill rows go through the same dispatches one row at a
  time. This is correct but costs one decode-sized expert pass per row; M6
  replaces it.
- Any number of MoE layers per function. Executable generalization (IREE patch
  0003) shares the dispatches across layers.

**Not in scope (v1)**: shared experts and sigmoid routers (Qwen2-MoE),
Top-k > 2, grouped prefill (M6), fusing both experts into one dispatch (§9, v2).

## 3. Key decisions

1. **The weights live in the module.** The frontend exports int8 expert tables
   and per-row scales as parameters; the compiler packs them at compile time
   the same way it packs dense weights. No side-car parameter file, so
   problem 4 goes away.
2. **Expert selection is a dynamic view, not a new kernel.** Before packing,
   `table[id_k]` becomes a `tensor.extract_slice` of the table. IREE keeps it,
   and the id it reads, in the consuming dispatch, so the offset is loaded
   from DDR on the device (§11.2). The existing linear
   and FFN kernels then compile each selected expert unchanged, except that the
   weight and scale LDs get a dynamic DDR field.
3. **Only Top-2 runs as RISC-V code.** The router is the existing W8A8 linear
   kernel. The softmax over two values is an ordinary generic on the VE, the
   same SFU `exp` / `recip` as attention, modelled bit-exactly by funcsim. The
   Q15 route record and `sahl.route_top2` are removed.
4. **RISC-V only where needed.** `rv_only` exports always run their embedded
   code. Every other export follows `SA_RV` (descriptor lists by default).
5. **The simulator executes Top-2.** The sa-desc file carries a small semantic
   record for each `rv_only` export, and funcsim / `sa_sim_server` /
   `dispatch_check` execute it in Python. Selection is an exact comparison, so
   this is bit-exact with the RISC-V code.

## 4. Frontend: the W8A8 Mixtral block (`hf_generic`)

**Implemented (S1, 2026-10-25)**: `QMixtralMoe` and `moe_rewrite` in
`compiler/frontend/hf_generic.py`, called by `quantize()`; test
`compiler/tests/test_hf_moe.py` (§12).

transformers' `MixtralSparseMoeBlock` (5.x: a `MixtralTopKRouter` `gate` with
a weight `[E, H]`, and `MixtralExperts` with `gate_up_proj [E, 2I, H]` and
`down_proj [E, H, I]`) loops over the experts with data-dependent masks and
cannot be exported. `moe_rewrite` replaces it with `QMixtralMoe`, which reuses
`QLinear`'s conventions (`_quant_act`, `_quantize_rows`; its product is now the
shared `_qlinear(x, wq, s_w)`):

```python
class QMixtralMoe(torch.nn.Module):
    # rq, rs: the router, E rows padded to ROUTER_PAD = 16 (zero rows)
    # pad: f32 [16], 0 for the E experts, -inf for the padding
    # gq, uq: int8 [E, I, H]   gs, us: f32 [E, I]   (gate_up_proj split)
    # dq:     int8 [E, H, I]   ds:     f32 [E, H]
    def route(self, x):                                    # x: [1, H] f32
        scores = _qlinear(x, self.rq, self.rs) + self.pad  # [1, 16], no slice
        vals, ids = torch.topk(scores, 2, dim=-1)          # descending
        e = torch.exp(vals - vals[:, :1])                  # no linalg.softmax
        return vals, ids, e * torch.reciprocal(e.sum(dim=-1, keepdim=True))

    def _row(self, x):
        _, ids, w = self.route(x)
        y = 0
        for k in range(2):                                 # unrolled
            i = ids[0, k:k + 1]                            # [1] i64
            g = _qlinear(x, self.gq.index_select(0, i)[0], self.gs.index_select(0, i)[0])
            u = _qlinear(x, self.uq.index_select(0, i)[0], self.us.index_select(0, i)[0])
            yk = _qlinear(self.act_fn(g) * u, self.dq.index_select(0, i)[0],
                          self.ds.index_select(0, i)[0])
            y = y + w[:, k:k + 1] * yk
        return y

    def forward(self, hidden_states):                      # [.., H]: rows one at a time
        ...
```

- The graph per expert is the same as a dense SwiGLU FFN except for the weight
  operand, so `FormFfnRegions` and the GEMV matchers see the forms they
  already know. `index_select` exports as a gather generic, which the compiler
  rewrites into a dynamic slice (§5.1, S2). Left as it is, IREE fuses the
  gather with the `.to(int32)` into a dispatch that writes an i32 copy of the
  expert, followed by an i32 × i32 → i64 matmul, and both run on the host
  (§11.1, §12.3).
- The router's expert axis stays padded to 16, and `-inf` scores mark the
  padding. Slicing the scores to E < D puts the dequantization on the host
  ("innermost size not a multiple of D"), and the slice between the projection
  and the sort also hides the router from Top-2 recognition (§11.1). With
  finite real scores, Top-2 never picks a padded lane. With the padding, the
  existing recognition finds the Top-2 and it compiles for the SA (§12.3).
- `torch.softmax` exports as `linalg.softmax`, which the SA backend does not
  lower (§11.1). `route` writes it out as exp, sum and recip. The first value
  is the larger one, so no max is needed. Its `[2]`-sized reduction and
  elementwise op compile for the SA (§12.3).
- The activation is quantized once. Router and both experts compute
  `_quant_act(x)` on the same `x`, and IREE CSE merges them.
- **Semantics.** Mixtral's router softmaxes all E logits, takes the Top-2 and
  renormalizes. That equals a softmax over the two selected logits, which is
  what `route` computes. Ties follow `torch.topk`'s order on the host; the
  device takes the lower expert ID. Exact ties are not expected on real
  scores.
- Prefill: for `T > 1` rows, `forward` runs row by row, each row with its own
  Top-2. The rows-by-row result equals one row at a time exactly (§12.1).
- Only Top-2 of at most 16 experts, without shared experts. Anything else
  raises `NotImplementedError` rather than being approximated.
- The router is W8A8. On tiny-mixtral, an fp32 router on the same hidden
  states picks the same ids as the W8A8 router at every step (§12.2), so
  router quantization is not where routing differs from HF.

Calibration and quality evaluation reuse the existing `hf_generic` and
`quality_eval.py` flows. The packed `sa-moe-parameters-v1` format remains only
for the M0–M5 tests. The dense path's export is unchanged: a random 2-layer
Llama exports to byte-identical MLIR and parameters before and after the
`_qlinear` refactor.

## 5. Preprocessing passes

Inside `extendPreprocessingPassPipeline`: the new expert-gather rewrite (§5.1)
runs first, then `PackLinearWeights` → `CloneCheapProducers` →
`FormMoeRegions` (rewritten) → `FormFfnRegions` → `FormAttnRegions` (after
IREE region formation). `LowerMoeChain` is deleted.

### 5.1 Expert gathers to slices, then `PackLinearWeights`

**Gather to slice** (implemented in `PackLinearWeights.cpp`,
`rewriteExpertGather`, run before its packing patterns). The pattern is a gather generic `G[n, k] = T[e[0], n, k]` (or
`[n]` for a scale table) where:

- `T` is a constant table;
- `e` is a one-element index tensor;
- the body is a single `tensor.extract` whose first index is
  `index_cast(e[0])` and whose other indices are the loop indices in order;
- `e` comes, through slices and reshapes, from a sort's or topk's indices: the
  decode embedding is the same one-row gather of a constant table at a token
  id and must keep its form (the packed-gather path, plan §8.12 C6.0).

It becomes `tensor.extract e[0]` → `index_cast` →
`tensor.extract_slice T[i, 0, 0] [1, N, K]` (rank-reduced). The rewrite may run
at flow level, outside any region: IREE clones the scalar extract and the
slice into the consuming dispatch (§11.2), so no host readback appears.

**`PackLinearWeights`, new pattern**: `vecmat(x, transpose(extsi(W)))` where
`W = extract_slice(T, [i, 0, 0])` of a constant `[E, N, K]` int8 table. The
rewrite:

```
Tp = linalg.pack T inner_dims_pos = [1] inner_tiles = [D]   -> [E, N/D, K, D] i8
Wp = extract_slice Tp[i, 0, 0, 0] [1, N/D, K, D]            (rank-reduced)
acc[t, j] += ext(x[k]) * ext(Wp[t, k, j])                   (the packed form, as dense)
```

`Tp` depends only on constants. With `compile_sa.sh`'s flags, IREE evaluates
it at compile time and exports it into the packed parameter archive (§11.3),
as it does for dense weights. Each expert is then a contiguous `N * K`-byte
slab: the expert stride is `N * K` bytes, and a scale table's stride is
`N * 4`.

### 5.2 `FormMoeRegions`, rewritten

The anchor is each Top-2 (an `iree_linalg_ext.topk` with K = 2, or the
recognized Torch sort prefix, as today). Matching is local to that anchor:

1. **Top-2 region.** Wrap the topk in its own region with
   `sa.moe.kind = "top2"`. This is today's behaviour; the attributes go on the
   region, never on the function.
   The padded scores reach the sort directly, so the existing recognition
   (projection → sort prefix) applies. Its expert count is the padded `Ep`.
2. **Expert dispatches.** No region work is needed to keep the selection on
   the device. In all three placements the spike tried, IREE gave one
   dispatch with ids, table and scales as bindings, and no
   `flow.tensor.load` (§11.2):
   - the id extracted inside a region;
   - the id and the slices outside it;
   - no explicit region at all.

   The gate/up chain is still wrapped by `FormFfnRegions` as for a dense
   SwiGLU, and the down projection forms on its own.
3. **Validation.** Reject with a diagnostic, falling back to the ordinary
   path, when:
   - the expert count is outside 4..16;
   - a gather's index does not come from this Top-2;
   - a table is not packed;
   - `N * K` fails the exactness check of §6.2.

There is no exactly-one condition and no function attribute. Each Top-2
anchor is matched independently, so the number of MoE layers in a function
does not matter.

The softmax over `[1, 2]` and the weighted sum are left to IREE's own region
formation. They are small elementwise and reduction dispatches that the
generic lowering already supports (v2 folds them into the expert kernel).

### 5.3 The dispatches of one decode MoE layer (v1)

| Dispatch | Content | Kernel | New |
|---|---|---|---|
| router | `x_q · W_r` → scores `[1, E]` (padded rows sliced off) | linear (GEMV) | none |
| top2 | scores → values `[1, 2]` f32, ids `[1, 2]` | `sahl.top2`, `rv_only` | runtime / sim (§7) |
| softmax | `w = softmax(values)` | generic (VE exp / recip) | none |
| ffn_k (k = 0, 1) | gate/up GEMV of expert `ids[k]`, SiLU · up, amax, int8 `h_q` | `expandFfn` | dynamic weight / scale base |
| down_k | `h_q · W_down[ids[k]]` → `y_k` f32 | linear (GEMV) | dynamic weight / scale base |
| combine | `w0 · y0 + w1 · y1` (+ residual) | generic | none |

As compiled for tiny-mixtral after S3 (§14), one decode MoE layer is 8
dispatches:

1. router;
2. Top-2;
3. the softmax's reduction and elementwise op (2 dispatches);
4. the FFN of expert 0, then of expert 1 (2 dispatches);
5. down of expert 0, which stores its raw i32 accumulator;
6. down of expert 1 with both dequantizations and the combine (IREE fuses
   the combine into it).

## 6. Backend: dynamic expert base

### 6.1 Representation

After bufferization the slice is a `memref.subview` of the table's binding,
with a dynamic offset in dimension 0 that comes from `memref.load` of the ids
buffer (DDR). `ddrOf` (`SahlLower.h`) gains a dynamic-index term:

```
Ddr { base, off, ..., dynIndex: { Ddr scalar (the i64 id), int64 strideBytes } }
```

`matchLinear` (`SahlKernels.cpp`) accepts the dynamic view unchanged: in the
spike, `sahl.kernel "linear"` forms over a weight `memref.subview` with
`offset: ?`. Compilation stops only in `sahl-to-sahw`, at the id's
`memref.load` (§11.4). `matchFfn` takes its weights the same way (`FfnPlan::w`
is a DDR value), but the spike did not exercise it; S4 checks it.

The lowering's work: `memref.load` of an i64 from a binding becomes a DDR
scalar for LDPARAM, and `ddrOf` folds a subview with that offset into the
dynamic-index term. Every LD the kernels emit for a weight tile or scale slice
then adds one PARAM to its DMA DDR field (`dyn` slot, add mode). M3b's
descriptor list did the same on the board.

### 6.2 Turning the id into a byte offset

LDPARAM computes `mem32[addr] * mul + add` with a 16-bit `mul`
(`sa_funcsim.py:522`). A weight slab is `N * K` bytes (3,670,016 for
tiny-mixtral), more than 16 bits.

- **Scales** (`N * 4` bytes ≤ 65535 for `N` ≤ 16383): one LDPARAM with
  `mul = N * 4`.
- **Weights**: the technique of `packedRowGather` (`SahlLowerGeneric.cpp:7`).
  1. LD the id word into a zeroed lane.
  2. VE copy I32 → F32, then a VE multiply by the immediate `float(N * K)`,
     then a VE copy F32 → I32.
  3. ST the result into a scratch word of the dispatch's output buffer, then
     a FENCE.
  4. LDPARAM with `mul = 1`.

  The product is exact when `N * K` and `(E - 1) * N * K` are integers that
  f32 represents exactly (odd part below 2^24) and below 2^31. Dimensions that
  are multiples of D always have many factors of two. `FormMoeRegions` checks
  this at compile time (§5.2 step 3).
- In plan A (RISC-V code) the same `sahw` program becomes a load and a 32-bit
  multiply. No special case is needed.

This costs one short LD → VE → ST → FENCE → LDPARAM sequence per expert and
dispatch, about 1–2 µs, against about 2 ms of weight streaming per expert.

**PARAM budget and dynamic slots (measured on the linear kernel, §11.5).**

- At tiny-mixtral sizes the decode linear kernel is one `sahw.loop` with two
  PARAMs: one steps the weight address, one steps the scale and output
  address.
- Each weight and scale LD inside the loop already uses one of its two
  dynamic slots (DDR field, add mode) for the loop PARAM. The expert term
  takes the second slot.
- `sa_cmdfetch.v` applies the slots one after the other, one cycle each,
  updating the descriptor word, so two adds on the same field compose. This
  is to confirm in an RTL test at S4.
- PARAMs: 2 (loop) + 2 (expert weights, expert scales) + 1 temporary = 5 of 8.

`expandFfn` is still to count at S4. If it leaves fewer than 2 free, the
scale offset is derived from the weight id with a second LDPARAM of the
same scratch word.

### 6.3 What goes away

- `sahl.route_top2`, `buildRouteTop2`, `routeScoreQ16`, `routeQ15ToF32`.
- `LowerMoeChain.cpp`, `MoeChainTemplate.inc`, `MoeModelTemplate.inc`, and the
  options `--iree-sa-moe-bundle` and `--iree-sa-moe-model-abi`.
- `moe_device_bundle.py`'s binary extraction, and `moe_iree_chain.py`.
- The `sa.moe.*` function attributes.

`sahl.top2` and `buildRawTop2` stay. `--iree-sa-form-moe` stays as the switch;
it turns on by default once v1 is accepted, like the fusions.

## 7. Runtime, sim and verification

**Runtime (`sa_loader.c`, `sa_rv.c`)**

- An `rv_only` export with embedded code runs it whatever `SA_RV` and
  `SA_RV_ONLY` say. This replaces the error at `sa_loader.c:253`.
- Every other export keeps the descriptor path by default.
- Without `--iree-sa-riscv`, the compiler emits embedded code only for
  `rv_only` exports. This needs a new state in `SahwToRiscv`: today it emits
  all or nothing.

**Sim**

- The sa-desc extension gets a per-export semantic record for `rv_only`
  exports: kind `top2`, `E`, and for scores / values / ids the
  `(binding, base Lin, offset)` triples already in `sa.moe.addresses`.
- `sa_funcsim.py` implements Top-2 with the comparison of `rawTop2Greater`:
  - compare on f32 bit patterns;
  - NaN and ±0 pairs are never "greater";
  - ties go to the lower ID.

  `sa_sim_server.py` and `dispatch_check.py` call it for such exports.
- Alternative, not chosen for v1: a small rv32im interpreter in the sim. It is
  more general (R5 kernels, future `rv_only` code) but slower and more code.

**Verification gates**, the repository's usual set:

1. Lit: new `sa_form_moe_regions.mlir` cases. Two layers in one function,
   in-region slices, rejections: a foreign index, an unpacked table, a stride
   that fails the exactness check.
2. Golden guard: the dense corpus compiles to the same bytes (MoE recognition
   must not fire on dense models). Then add the MoE dispatches (synthetic
   `E=4, H=32, I=64` and tiny-mixtral) with `golden.py --record`.
3. `dispatch_check.py` per dispatch:
   - every expert ID (forced IDs, as M1 did);
   - ties, `-inf`, a NaN score;
   - the last expert, the highest DDR offset.
4. End to end on the sim: tiny-mixtral (2 layers), decode and per-row prefill,
   through `compile_sa.sh` and `sa-llm-run`. Logits byte-equal between the
   descriptor and plan A builds. Top tokens and expert IDs match the
   `hf_generic` host reference.
5. Board: the same bundle, board vs sim logits byte-equal, and expert IDs
   reported per layer and step.
6. Speed, against M5's numbers: two-expert FFN 408.6K cycles per layer,
   Top-2 + route 8.4K, M4f dense part 228K. Report device and wall time per
   token. The weights are resident in the u-dma-buf pool, so there is no
   per-layer reload (`board_moe_full_device.py:103` today).

**Numerics.** v1's routing weights come from the VE's fp32 `exp`, not M5's
Q15 integer softmax, so M5's saved references do not apply. The host reference
is `hf_generic`'s fp32-emulated W8A8 graph. Board ≡ sim is exact; against HF
the usual quantization gap applies (tiny-mixtral is random-initialized, so
only token and ID agreement are reported, not quality).

## 8. Stages

| Stage | Work | Gate |
|---|---|---|
| S0 spike | **Done 2026-10-25 (§11).** Hand-written MLIR of one expert linear, four placements, two sizes | IR results recorded; R1 resolved |
| S1 frontend | **Done 2026-10-25 (§12).** `QMixtralMoe` in `hf_generic` (padded router, explicit softmax), `test_hf_moe.py` | Host model: top tokens equal to HF; ids equal where HF's routing is clear; no softmax op, Top-2 over `[1, 16]` |
| S2 packing | **Done 2026-10-25 (§13).** Gather-to-slice rewrite, `PackLinearWeights` expert tables | Lit; tables packed at compile time into the parameter archive |
| S3 regions | **Done 2026-10-25 (§14).** One FFN region per expert (already `FormFfnRegions`'), ids read as i32 | Lit; no id-extension dispatches; golden identical |
| S4 backend | **Done 2026-10-25 (§15).** Tagged expert BASEs, the P prologue, the dynamic slots | Every tiny-mixtral dispatch on the SA; `dispatch_check` bit-exact except the Top-2 (S5) |
| S5 Top-2 | **Done 2026-10-25 (§16).** Top-2 through the RISC-V runtime's load / store, host build on the sim, runtime `rv_only` policy, compiler embeds only `rv_only` code | `dispatch_check` 198/198 incl. the Top-2; sim decode runs |
| S6 e2e sim | **Done 2026-10-25 (§17).** tiny-mixtral through `compile_sa.sh`, `test_moe_sim.py` | Prefill ≡ decode and plan A ≡ descriptors byte-equal; host model argmax 28/28; golden + 198 MoE dispatches |
| S7 board | **Accepted 2026-10-25 (§18).** `deploy_moe.sh` → `build/deploy_moe_tm`, `run_board_tests.sh moe` | Board ≡ sim 5/5 (prefill) and 28/28 (decode) steps; decode 16.0 ms per step |
| S8 cleanup | **Done 2026-10-25 (§19).** §6.3 and M5's milestone scripts deleted | Lit 28/28, golden identical, the accepted bundle recompiles identically |

Estimate: S0 1 day, S1–S3 3–4 days, S4 3–4 days, S5 2 days, S6–S8 2–3 days.

## 9. v2: one dispatch for both experts (after v1 is accepted)

*Measured on 2026-10-25 and not pursued: §20. The bound below assumed
dispatch boundaries that the batched runtime does not have.*

v1 has about 8 dispatches per MoE layer against 2–3 for a dense FFN. Each
boundary costs the small fixed LD / ST latency that the fusions removed
elsewhere (fusion_design.md §7). v2 merges ffn_0, down_0, ffn_1, down_1 and
combine into one region (kernel `moe_ffn`):

- Both experts' `expandFfn` chunks back to back.
- Expert 1's first GEMV issued before expert 0's down-projection epilogue, so
  the GEMV stream does not drain.
- `y_0` stays in ACC and `w_k` comes in as VE scalars.

The bound: the FFN stream is 22.1 MB per layer per token. At the measured
57.95 B/cycle that is about 381K cycles, against 408.6K today, so about 7% at
most on the FFN plus the removed dispatch overhead. Router + Top-2 in one
`rv_only` export (the GEMV issued from the same RISC-V code) is a separate,
smaller step.

## 10. Risks and open questions

- **R1: host readback of the ids. Resolved by S0** (§11.2). IREE keeps the
  scalar extract and the dynamic slice in the consuming dispatch in every
  placement tried. Watch for it again when the full model compiles (S6):
  `grep flow.tensor.load` on the flow IR.
- **R2: PARAM pressure.** 5 of 8 for the linear kernel; `expandFfn` still to
  count (§6.2).
- **R3: dynamic DMA fields.** The looped GEMV LD uses one slot, and the
  expert term takes the other (§6.2). An LD that already uses both cannot
  take an expert base; S4 rejects such a dispatch with a diagnostic
  (host fallback) rather than miscompiling it.
- **R4: plan A call latency.** Each `rv_only` dispatch is an ARM → PicoRV32
  call per layer. M5 measured Top-2 + route at 8.4K cycles. The ARM-side call
  overhead is measured at S7.
- **R5: router precision.** On tiny-mixtral the W8A8 router picks the same
  ids as an fp32 router on the same hidden states (§12.2). The ids that differ
  from HF are near ties (HF's 2nd vs 3rd logit 0.002–0.008 apart) moved by the
  hidden states' W8A8 error upstream, which an fp32 router would not change.
  This is to recheck on a trained checkpoint, whose routing margins are
  larger. Rechecked on Granite (§21): the W8A8 router agrees with an fp32
  router on 99.3% of the selected experts on the same hidden states. The fp32 router stays an option: E rows times H as VE multiply +
  reduce-sum over fp32 weights.
- **R6: tie semantics against HF.** HF's `torch.topk` does not specify tie
  order. Exact ties are rare in real scores; the reference defines
  lower-ID-first.

## 11. S0 spike results (2026-10-25)

`compiler/tests/moe_s0_spike.py --out <dir> [--experts E --n N --k K]` writes
the four variants below and their parameter archive. Each was compiled with
`compile_sa.sh`'s flags (D = 16, 4 GEMV ports, generalization, FFN fusion,
host fallback) plus `--iree-sa-codegen-report`, and the dispatch sources were
run through the SA pipeline with `iree-opt` up to `iree-sahl-to-sahw`. The
sizes were the synthetic `E=4, N=64, K=32`, and tiny-mixtral's gate
(`E=8, N=3584, K=1024`) and down (`E=8, N=1024, K=3584`).

### 11.1 The frontend form as exported today

A Top-2 block written with `index_select`, `torch.softmax` and a router
sliced to E = 4 (synthetic, through `hf_generic`'s `_quant_act`) compiles as
follows (codegen report, with and without `--iree-sa-form-moe`):

| Dispatch | Decision | Cause |
|---|---|---|
| activation amax / int8, router GEMV (padded to 16) | SA | — |
| router dequantization `[1, 4]` | host | innermost size not a multiple of D (the slice to E) |
| Top-2 as `iree_linalg_ext.sort` + iota | host | not recognized: the slice sits between projection and sort |
| `linalg.softmax` over 2 | host | not lowered by the SA backend |
| gather + `.to(int32)`: `elementwise_broadcast_2048_i64xi32` per weight | host | a materialized i32 copy of the selected expert |
| `matmul_1x64x32_i32xi32xi64` per projection | host | not the packed linear form |
| amax / int8 of h | SA | — |

There is no `flow.tensor.load`: even this form keeps the ids on the device
(they are dispatch operands), but almost everything around them runs on the
host. §4 and §5.1 are written against these findings.

### 11.2 The id stays in the dispatch

The hand-written form (packed table `[E, N/D, K, D]`, slice at the id, decode
linear with a dequantization epilogue on `[N/D, D]`):

| Variant | flow | Dispatch operands |
|---|---|---|
| `dyn`: extract + slices inside an explicit region | no `flow.tensor.load` / `flow.tensor.slice` | ids `tensor<2xi64>`, packed table, scales, x, s_x |
| `sliceout`: extract + slices before the region | same: IREE clones them into the dispatch | same |
| `noregion`: no explicit region | same: IREE forms one dispatch | same |

In the dispatch source, the slice is folded into the binding load at the
id's offset:

```mlir
%extracted = tensor.extract %6[%c0] : tensor<2xi64>
%9 = arith.index_cast %extracted : i64 to index
%10 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [%9, 0, 0, 0], sizes = [1, 4, 32, 16], ...
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4x32x16xi8>> -> tensor<4x32x16xi8>
```

### 11.3 Packing at compile time

`linalg.pack T inner_dims_pos = [1] inner_tiles = [16]` of the `[E, N, K]`
table in an initializer is evaluated at compile time. The flow IR of the
synthetic case holds `@__parameter_model_gq_packed_tensor_4x4x32x16xi8` (a
`#flow.parameter.named`), with no pack dispatch left. The exported archive
for tiny-mixtral's gate is 29,364,224 bytes: the 8 × 3584 × 1024 =
29,360,128 bytes of packed weights plus a 4 KiB header. The synthetic scale
table became an inline constant, reshaped by IREE to `[E, N/D, D]`.

### 11.4 Bufferized form and where the backend stops

```mlir
%6 = memref.load %0[%c0] : memref<2xi64, #hal.descriptor_type<storage_buffer>>
%7 = arith.index_cast %6 : i64 to index
%subview = memref.subview %1[%7, 0, 0, 0] [1, 4, 32, 16] [1, 1, 1, 1]
    : memref<4x4x32x16xi8, ...> to memref<4x32x16xi8, strided<[512, 16, 1], offset: ?>, ...>
%subview_2 = memref.subview %3[%7, 0, 0] [1, 4, 16] [1, 1, 1]
    : memref<4x4x16xf32, ...> to memref<4x16xf32, strided<[16, 1], offset: ?>, ...>
sahl.kernel "linear" attributes {chunk_tiles = 4 : i64, loop = false} {
  sahl.load %subview, %alloc_1 : memref<4x32x16xi8, strided<[512, 16, 1], offset: ?>, ...>, memref<4x32x16xi8>
  ...
```

| Variant | Codegen |
|---|---|
| `static` (expert 1 at a constant offset) | **ok**, on the SA: GEMV LD at `ddr = 1 · N · K` (2048; 3,670,016 at tiny-mixtral sizes), scale LD at `1 · N · 4` |
| `dyn`, `sliceout`, `noregion` | kernel `linear` formed; host fallback with `sahl-to-sahw: unsupported operation memref.load` |

The kernel matcher already accepts the dynamic view, so S4's work is confined
to the lowering of the id load and the dynamic subview offset.

The spike also found a constraint on the form: the epilogue must read the
accumulator directly on `[N/D, D]`. A unit row dimension (`(m, t, j, k)`, the
prefill form) or a `collapse_shape` between the accumulator and the epilogue
makes `matchLinear` fail, ending in "fill with a non-constant or non-fp32
value". Dense decode gets this form from IREE's unit-dimension folding, which
skips ops already in a region; `FormFfnRegions` folds them before wrapping
for the same reason. The expert dispatches inherit this as long as no new
region is formed before the folding.

### 11.5 Real-size templates (static offset)

| Shape | Template | GEMV LDs | Dynamic slots used | PARAMs |
|---|---|---|---|---|
| gate `N=3584, K=1024` | `sahw.loop 27` with the weight PARAM stepping 131072 B and the scale/output PARAM stepping 512 B, 2 LDs per iteration, then a 2-LD tail | 4 (2 in the loop, 2 in the tail) | 1 per looped weight / scale LD / ST (DDR, add); none in the tail | 2 |
| down `N=1024, K=3584` | same structure, pitch 57344 | 4 | same | 2 |

The expert base therefore takes the second dynamic slot of the looped LDs and
the first of the tail's (§6.2).

## 12. S1 results (2026-10-25)

`compiler/tests/test_hf_moe.py` (synthetic by default; `--model DIR` for a
checkpoint; `--export DIR` keeps the synthetic export).

### 12.1 Synthetic block and export

The synthetic model is a random 2-layer Mixtral: E = 4, H = 64, I = 128, its
router weights drawn with σ = 0.5 so the routing is spread.

- **Block**, 64 random rows: Top-2 ids equal to HF's on all 62 clearly routed
  rows (2nd vs 3rd logit more than 5% apart). The output's relative L2
  against HF fp32 is 0.020 on average and 0.059 at most (W8A8). Four rows
  computed together are exactly equal to the rows computed one at a time.
- **Export** (`hf_generic.export`, decode + 16-token prefill + prefill_kv):
  no softmax op, and 66 `torch.aten.topk` (2 layers × (1 + 16 + 16) rows),
  each over `!torch.vtensor<[1,16],f32>`.

### 12.2 tiny-mixtral (host, 3 decode steps of tokens 1, 42, 43)

The W8A8 model is `SaDecoder`: int8 KV cache calibrated on the three tokens,
sa attention, `QLinear`, `QMixtralMoe`.

| Pos | Top token (HF) | Logit rel L2 vs HF | Layer 0 experts | Layer 1 experts (HF) | HF layer-1 gap, 2nd vs 3rd |
|---:|---:|---:|---|---|---:|
| 0 | 25591 (25591) | 0.298 | [7, 1] = HF | [6, 4] ([6, 2]) | 0.0023 |
| 1 | 13899 (13899) | 0.035 | [7, 1] = HF | [2, 5] = HF | 0.0173 |
| 2 | 26962 (26962) | 0.323 | [7, 0] = HF | [2, 6] ([2, 0]) | 0.0075 |

- Top tokens all equal to HF's, and all layer-0 ids equal (HF gaps
  0.06–0.18).
- The layer-1 ids that differ are near ties. The W8A8 router agrees with an
  fp32 router on the same hidden states at all six steps (max logit
  difference 0.0085–0.020), so the flips come from the W8A8 hidden states
  upstream.
- The two steps with a flipped expert have the large logit errors (0.30,
  0.32). The step without one has 0.035, the size M4d measured for its
  hybrid. The model is random-initialized: this is a functional check, not a
  quality one.
- M4's board run took the same layer-1 ids [2, 5] and [2, 6] at positions 1
  and 2. Its HF reference was the W8A8-expert hybrid trace (M4c/M4d), not
  pure fp32 HF.

**Test pitfall.** hf_generic's rewrites (RoPE, RMSNorm, attention
registration) patch transformers functions shared by every model in the
process. A reference HF model must therefore run before them; the test's
first version ran it after and reported wrong HF ids and tokens.

### 12.3 What the synthetic export compiles to today

`compile_sa.sh` with D = 16, 4 GEMV ports, `--iree-sa-form-moe` and
`--iree-sa-riscv`:

- **On the SA**:
  - the Top-2 (`topk_1x16xf32`, the `rv_only` export);
  - the softmax's `reduction_2_f32` and `elementwise_2_f32`;
  - the activation and hidden-state quantization.
- **On the host, MoE-specific**:
  - the expert gathers (`elementwise_broadcast_8192_i64xi32`: gather + i32
    cast);
  - the expert projections (`matmul_1x64x128_i32xi32xi64`,
    `matmul_1x128x64_i32xi32xi64`).

  Both are S2's work.
- **On the host, not MoE-related**: at this tiny size (H = 64) the attention
  projections (`matvec_like_4x16x64_i8xi8xi32` and its prefill forms, error
  `bufferization.to_buffer`) also fall back. A random dense Llama of the same
  size shows the same fallbacks with the same flags, so S2's compiled fixture
  needs larger dimensions (or this is fixed separately).

## 13. S2 results (2026-10-25)

**Implementation** (`compiler/plugins/sa/target/PackLinearWeights.cpp`):

- `rewriteExpertGather`: the one-row gather at a Top-2 id becomes
  `extract_slice T[id, 0, ..]`, and a collapse dropping its unit dimension
  becomes the rank-reduced slice.
- `expertSlice` / `packedWeights`: `rewriteMatmul` / `rewriteVecmat` accept a
  weight that is such a slice of a constant int8 `[E, N, K]` table. They pack
  the table along N (`packOf` with a packed dimension, `T$packed : [E, N/D,
  K, D]`) and read `extract_slice Tp[id, 0, 0, 0] [1, N/D, K, D]`.
- The existing gather redirection (embedding rows of a packed classifier)
  only considers 2-D tables.

Lit `sa_pack_expert_weights.mlir`: an expert gate weight and its scales
gathered at a sort's index become a packed slice and a scale slice, and a
gather at a non-Top-2 index is unchanged. The suite passes (28 tests). The
golden corpus is identical (636 dispatches). That corpus is codegen only; for
preprocessing, the new rewrites fire only on a sort's or topk's indices and on
3-D expert tables, and the recent dense exports (SmolLM2, Gemma, Qwen3, Llama)
contain no sort or topk.

**tiny-mixtral, whole model** (`test_hf_moe.py --model ... --export`, decode
+ 16-token prefill, `compile_sa.sh` with D = 16, 4 ports, `--iree-sa-form-moe
--iree-sa-riscv`):

- `sa_packed.irpa` (214 MB) holds `layers.{0,1}.mlp.{gq,uq,dq}$packed`,
  29,360,128 bytes each, plus the padded router `rq$packed` (16 KiB).
- In the flow IR there is no `flow.tensor.load`. The 64 `flow.tensor.slice`
  are prefill RoPE rows, present in a dense model of the same form too.
- 192 dispatches on the SA, 7 on the host:

| Host dispatch | Cause | Stage |
|---|---|---|
| gate/up of both experts (`matvec_like_224x16x1024`), down of each expert (`matvec_like_64x16x3584`, 3 executables) | `memref.load` of the id | S4 |
| a prefill scale slice materialized as its own copy (`slow_memcpy`) | `memref.load` of the id | S4 (better: keep the slice in its consumer) |
| `id_k = extsi(ids[k])` to i64, one dispatch per k | i64 scalar computation / 8-byte alignment | S3: `index_cast(extsi(x)) = index_cast(x)`, consumers take the i32 ids |

**Constant inlining.** At first the decode router (`matvec_like_16x1024`)
also fell back, with `bufferization.to_buffer`. Its scales and padding (16
floats each, 64 bytes) are below IREE's
`--iree-flow-inline-constants-max-byte-length` (256) and were inlined into
the dispatch as constant tensors, which the SA backend cannot read. This is
also why the H = 64 synthetic models' projections fall back (§12.3): their
scales are under 256 bytes too. A threshold of 1 keeps splats inlined (they
are always inlined) and keeps every other tensor constant a binding.
`hf_generic.export` writes it into `compile_flags.txt` for models with
`QMixtralMoe` only, so dense models compile as before. With it, the router
runs on the SA.

**Dispatch structure.** *Corrected in S3 (§14): an earlier version of
this paragraph said that the gate and up projections of both experts share
one dispatch. That was a misreading. The dumped source holds two variants
(the SA one and the VMVX host fallback), each with one expert's two table
loads.* `FormFfnRegions` already forms one FFN region per expert. The
dispatch is one executable called once per expert, at ids `%46` and `%48`.

## 14. S3 results (2026-10-25)

**Regions.** Split mode (v1) needs no new region pass:

- `FormFfnRegions` already forms one FFN region per expert. Each expert's
  gate and up contractions share x, and the other expert's slice is separate.
- IREE forms the down projections, and fuses the combine into the second
  one.
- The Top-2 region and its attributes stay on the region, as before.
- Merged mode (v2, §9) will be a second region shape in `FormMoeRegions`
  behind a flag. Its acceptance is byte-equal logits against split mode
  (same per-expert arithmetic, another schedule).

**Ids read as i32** (`FormMoeRegions.cpp`, `readIdsAsI32`):

- `rewriteTorchTop2Sort` replaces the sort with a Top-2 that returns i32 ids
  and extends them to i64 for the rest of the graph.
- IREE turned each expert's id into its own one-element extension dispatch,
  on the host ("i64 scalar computation", "load not 8-byte aligned").
- Now every `tensor.extract` of the ids that feeds an `index_cast` reads the
  i32 ids through the same view chain, rebuilt on i32. This holds because
  `index_cast(extsi(x)) = index_cast(x)`.
- Lit: `sa_form_moe_regions.mlir` `@expert_id_as_index`.

**tiny-mixtral** (same flags as §13): 192 dispatches on the SA, 6 on the host,
all with `memref.load` of the id (S4):

- the FFN of each expert;
- the down projections (3 executables);
- the prefill scale copy.

The two id extensions are gone. The two FFN dispatches are now separate
executables, because they read the id at different static offsets of the
ids buffer (`ids[0]`, `ids[1]`). That costs one more descriptor template per
FFN shape.

**Prefill scale copy.** In the row-by-row prefill, IREE materializes one
expert's down-projection scales as a 4 KiB copy (`slow_memcpy`, dynamic
offset) instead of cloning the slice into its consumer, as it does in
decode. S4's lowering takes it to the SA as a dynamic LD + ST. M6's grouped
prefill replaces this path, so it is not worth fighting IREE's fusion here.

Lit 28/28; golden identical (636 dispatches).

**Not done in S3:** removing the M5 full-graph match (`mixtralExpertShape`,
the function attributes) and `LowerMoeChain`. The new frontend never
matches it, and it is deleted with the rest of M5's chain in S8.

## 15. S4 results (2026-10-25)

**Implementation** (`SahlToSahw.cpp`, `SahlLower.h`):

- `ddrOf` recognizes a subview offset that is `index_cast(memref.load
  ids[c])` of an i32 / i64 DDR buffer (`expertId`). The view gets a **tagged
  BASE**: a `sahw.base` of its own, keyed by the binding, the id's address
  and the byte stride.
- `memref.load` ops whose only uses are such offsets are skipped by
  `lowerBlock`.
- After the lowering, `applyExpertBases`:
  1. puts a FENCE at the start of the body. The ids are the previous
     dispatch's. With the FENCE, no leading LD reaches `sahw.head` ahead of
     it; such dispatches give up the head prefetch.
  2. computes each distinct P = id × stride once:
     - a stride ≤ 65535: one LDPARAM (`mul = stride`);
     - a larger one: LD the ids' 8-byte word into ACC word 0, one VE (I32 →
       I32, A = stride) into word 1, ST into the first word of an
       output-only binding (written by the dispatch later), FENCE, LDPARAM.
       Refused unless every `e × stride` (e < rows) is exact in fp32 and
       below 2^31.
  3. points every DMA on a tagged BASE (LD, ST, LDPARAM) at the real BASE and
     appends a dynamic slot (its DDR / address field, add mode, P). The base
     is replaced first: appending the slot's operand reallocates the op's
     operands. Refused, with a host fallback, when a DMA has both slots taken
     or its address field is a replace-mode slot.

**Hardware check.** `sa_cmdfetch.v` applies the slots one per cycle and
writes each result back into the descriptor word (lines 360–362). Two add
slots on one field therefore compose, as `sa_funcsim.py` does. An RTL run of
this case is left to the board stage.

**A pre-existing bug found by the per-dispatch check** (`SahlLowerGeneric.cpp`):

- A 1-D reduction whose length is not a multiple of D (the Top-2 softmax's
  sum over 2) reduced the whole word, garbage lanes included: NaN with
  `--dirty`, 11.26 instead of 1.195 on clean memory.
- It now passes VALID = n and ROWLEN = ⌈n / D⌉, so the masked lanes take the
  identity.
- No dense dispatch has such a reduction; the golden corpus is unchanged.

**Verification.**

- Lit `sahl_expert_slice.mlir`:
  - a 256-byte stride (one LDPARAM);
  - a 131072-byte stride (the VE path through the output's first word);
  - a stride of 2^24 + 1, refused as inexact.
- `dispatch_check.py --ids LIST` fills the bindings read as indices with
  those ids.
- S0 spike, synthetic size: ids 0..3, all equal to the oracle, which depends
  on the id (outputs for ids 0 and 3 differ).
- tiny-mixtral's gate size, a looped GEMV with two add slots on the DDR
  field: ids 0, 1, 3, 6, 7 equal.
- tiny-mixtral whole model: 198 dispatches on the SA, none on the host.
- `dispatch_check` with `--dirty --ids 7,0` and with `--pos 0 --ids 2,5`:
  197 of 198 equal to the oracle. The Top-2 (`topk_1x16xf32`) is not
  checkable: the oracle has no `iree_linalg_ext.topk`, and the functional
  simulator cannot run its RISC-V code (S5).
- Lit 29/29; golden identical (636).

**PARAMs** (from the compiled descriptors):

| Dispatch | PARAMs in use |
|---|---:|
| router | 0 |
| FFN of expert k (`expandFfn`, unrolled at this size) | 2 (P_w, P_s) |
| down of expert 0 | 1 |
| down of expert 1 + combine (both experts' scales) | 4 |

The register file has 8, so this fits with room to spare.

## 16. S5 results (2026-10-25)

**The sim needed no semantic record.** §7 planned a Python model of the Top-2
for the simulator. The repository already runs generated RISC-V code on the
sim: the portable LLVM IR (`--iree-sa-emit-riscv`) is compiled for the host
(`compiler/tests/rv_image.py` → `rvcode_host.so`). `sa_transport_sim.c` calls
it, its commands run as one-descriptor lists on the functional simulator, and
its loads read the shared window. That executes the same code as the board,
so it replaces the semantic record.

The obstacle was the Top-2 kernel itself. M5's `buildRawTop2` read and wrote
DDR through raw pointers to physical addresses. That works on the PicoRV32
but not in a host process, so M5's Top-2 was only ever tested on the board.

**Changes:**

- **`SahwToRiscv.cpp`**:
  - `buildRawTop2` reads the scores through `sa_rv_ldparam` and writes the
    values and ids through a new `sa_rv_store32`.
  - With `--iree-sa-riscv` (inline PCPI, in-process) both stay plain
    volatile accesses; the load follows a FENCE.
  - In the portable IR they are calls.
- **Runtime shim** (`firmware/rv/sa_rv_rt.c`, `sa_rv_rt.h`):
  `sa_rv_store32` for the PicoRV32 (a store), the host recorder (a note) and
  the sim (a callback registered by `sa_rv_sim_init_store`, found with
  `dlsym`, so an older `rvcode_host.so` still loads).
- **`SATarget.cpp`**: without `--iree-sa-riscv`, the exports marked
  `sa.riscv_only` (and only those) get embedded rv32 code. The old "requires
  `--iree-sa-riscv`" error is gone. In tiny-mixtral one executable of 198
  has a code table (4.5 KB).
- **Runtime policy** (`sa_rv.c`, `sa_loader.c`): `sa_rv_lookup(..., required)`.
  - An `rv_only` export runs RISC-V code whatever `SA_RV` and `SA_RV_ONLY`
    say: on the board its executable's embedded code (the argument block
    allocated on demand), on the sim the `SA_RV` directory's host build.
  - Without either it fails with a message saying which is needed.
  - `SA_RV_ONLY=rv_only` keeps every other export on descriptor lists.
- **`dispatch_check.py --rv DIR`**: an `rv_only` export runs its host build,
  with the functional simulator executing its commands, loads and stores.
  `oracle.py` executes `iree_linalg_ext.topk` (IREE's scalar insertion, the
  ordered `>` comparator, equal values ordered by id). `sadesc.read_ext`
  reports `rv_only`.
- **M5 tests**: `test_moe_compiler_riscv.py` and `test_moe_compiler_route.py`
  now expect compilation without `--iree-sa-riscv` to succeed. The former
  links its host run of the Top-2 IR with a two-function shim. Both pass.

**Verification:**

- `dispatch_check` on tiny-mixtral with `--rv`, `--dirty --ids 7,0` and
  `--pos 0 --ids 2,5`: 198 / 198 equal to the oracle, the Top-2 included.
  Without `--rv` the Top-2 reports "RISC-V only".
- Sim decode of tokens 1, 42, 43 (`sa-llm-run --abi=hfv --pad=16
  --valid_len=64`, `SA_RV=<rv dir> SA_RV_ONLY=rv_only`):
  - 0 dispatches on the host, 6 as RISC-V code (the Top-2 of 2 layers ×
    3 steps);
  - top tokens 25591, 13899, 26962, the host W8A8 model's and HF fp32's.
- Lit 29/29; golden identical (636); the host and aarch64 runtimes build.

S6 compares the sim's logits with the host model and checks prefill against
decode. S7 adds the board.

## 17. S6 results (2026-10-25)

**Bundle.** `build/moe/native/tm` holds the bundle, and S7 uses it:
`test_hf_moe.py --model build/llm_cache/TitanML-tiny-mixtral --export`, then
`compile_sa.sh` with D = 16, 4 ports and `--iree-sa-form-moe
--iree-sa-emit-riscv=<bundle>/rv_ll`, then `rv_image.py` into `<bundle>/rv`.

- 198 dispatches, all on the SA, none on the host.
- `sa_packed.irpa` 214 MB.
- The KV scales are calibrated on tokens 1, 42, 43. The checkpoint is
  random-initialized, so this is a functional check, not a quality one.

**`compiler/tests/test_moe_sim.py`** (on the sim, a 24-token prompt and 4
generated tokens):

| Run | Dispatches | Result |
|---|---|---|
| A: decode of the prompt, descriptor lists, the Top-2 as RISC-V code (`SA_RV_ONLY=rv_only`) | 84 lists, 56 RISC-V (Top-2), 0 host | the reference for B and C |
| B: the prompt through `--prefill 16` (two chunks, the rows' MoE one row at a time) | 82 lists, 72 RISC-V (64 prefill rows + 8 decode), 0 host | last prompt position and 4 steps **byte-equal** to A |
| C: plan A, every export as generated RISC-V code (host build) | 1568 RISC-V, 0 lists | 28 steps **byte-equal** to A |
| ref: the host W8A8 model (`SaDecoder`, `QMixtralMoe`), teacher-forced on A's tokens | — | argmax equal in 28 / 28, correlation min 0.99957, mean 0.99982 |

The expert ids cannot be read from the device run. They are covered
indirectly: the per-dispatch check verifies the Top-2 and the expert slices
for given ids, and a flipped expert changes a step's logits by far more than
this correlation (§12.2: rel L2 ≈ 0.3 against ≈ 0.03 at the same step).

**Golden.**

- New case `moe_mixtral` (`build/moe/native/tm/sa_sources`): 198 dispatches,
  all compiling.
- The corpus is now 834 unique dispatches. The 610 recorded before are
  byte-identical, and the 26 that already failed to compile (host fallbacks
  of the older cases) are unchanged.
- **Nondeterminism found and fixed.** The first comparison after recording
  differed in 2 MoE dispatches. `applyExpertBases` iterated a `DenseMap`
  keyed by `Value`, so the prologue's order (P computations, PARAMs)
  followed pointer hashes. It is now a `MapVector`, in creation order.
  Recording plus two comparisons pass.
- The bundle was recompiled with the fix (sources unchanged). The S6 test
  and `dispatch_check --dirty --ids 7,0 --rv` pass on it (198 / 198).

## 18. S7: the board (accepted 2026-10-25)

`compiler/scripts/deploy_moe.sh [build/moe/native/tm]` stages
`build/deploy_moe_tm`, containing:

- `sa.vmfb`, `sa_packed.irpa`;
- the sim's decode-only run as the reference (`test_moe_sim.py` now writes
  `prompt.npy`, `expected_tokens.npy`, `expected_logits.npy` into
  `<bundle>/s6`);
- the aarch64 `sa-llm-run`;
- `rt_fw.bin` (v7: RT_CALL for the Top-2's embedded code);
- the `d16_100mhz_w128_p6_gemv_np4_pcpi` overlay, `board_llm.py`,
  `board_run.py`.

The window is 260 MB, the sim's device peak (238 MB) plus 32. `sa_args.txt`
is `--abi=hfv --pad=16 --valid_len=64 --prefill=16`.

On the board, the Top-2 runs as the executable's embedded RISC-V code (the
`rv_only` policy, §16, without `SA_RV`), and everything else runs as
descriptor lists. `run_board_tests.sh moe` runs two tests:

- `llm`: prefill + decode, compared from the prompt's last position on;
- `llm --decode-only`: the whole prompt as decode steps.

Both compare the logits bit for bit with the sim. `board_run.py` gives
decode-only a results directory of its own (`llm_decode`) and keeps
`logits_board.f32`. `board_llm.py` prints the ids when the checkpoint has no
tokenizer.

**Board results** (commit `b065671`, `d16_100mhz_w128_p6_gemv_np4_pcpi`, 100 MHz,
rt_fw v7; `build/deploy_moe_tm/results/b065671/moe_tm/.../llm` and `.../llm_decode`):

| Test | Logits vs sim | Tokens | Speed | Dispatches |
|---|---|---|---|---|
| `llm`: prefill 16 (2 chunks) + 4 decode steps | bit-exact 5 / 5 steps | identical | prefill 24 tokens 310.3 ms (77.4 tok/s, the first chunk includes loading); decode 16.1 ms per step (62.2 tok/s) | 82 lists, 72 RISC-V (Top-2), 0 host |
| `llm --decode-only`: 28 steps | bit-exact 28 / 28 | identical | 16.0 ms per step (62.5 tok/s) | 84 lists, 56 RISC-V, 0 host |

What this confirms on the hardware, beyond the sim:

- the Top-2 running as the executable's embedded rv32 code through RT_CALL,
  without `SA_RV`, between descriptor lists;
- the expert offset computed on the device (LDPARAM, and the VE product
  through the output's first word);
- two add slots on one DDR field in `sa_cmdfetch.v` (looped GEMV LDs):
  §15 had only read the RTL for this.

**Speed in context.** Each decode step reads about 82 MB of weights:

- two layers × two experts × 11.0 MB of expert weights;
- the dense projections, about 2.6 MB per layer;
- the router;
- the 32.8 MB classifier.

At the measured GEMV stream of 57.95 B/cycle (5.8 GB/s) that is about
14.1 ms, so the 16.0 ms wall time per step is about 88% of the weight-stream
bound. M4f's full-device run had no meaningful wall time: it reloaded each
layer's 88 MB expert slot from a file at every step (§1 problem 5, the
`board_moe_full_device.py` CMA path). This bundle keeps every expert table
resident in the u-dma-buf window (260 MB).

The checkpoint is random-initialized: these are functional and speed
results, not quality ones. Quality needs a trained MoE checkpoint that fits
the board (§10 R5).

## 19. S8: M5's chain removed (2026-10-25)

**Deleted (46 files):**

- **Compiler**:
  - `LowerMoeChain.cpp`, `MoeChainTemplate.inc`, `MoeModelTemplate.inc`;
  - the options `--iree-sa-moe-bundle` and `--iree-sa-moe-model-abi`;
  - `sahl.route_top2` and its lowerings (`SahlToSahw`; `buildRouteTop2`,
    the Q15 softmax `routeScoreQ16` / `routeQ15ToF32` in `SahwToRiscv`);
  - `FormMoeRegions`' full-graph match (`mixtralExpertShape`,
    `packedExpertStride`, the `sa.moe.arch` / `dim` / `hidden` /
    `expert_stride` function attributes).

  The Top-2 region's `sa.moe.kind` is now `"top2"`.
- **Frontend**: `moe_device_bundle.py`, `moe_iree_chain.py`, `moe_native.py`
  (the fp32 M5h module; `hf_generic.QMixtralMoe` replaces it).
- **Tests and scripts**:
  - M5's board probes (`board_moe_{compiled_ffn, compiler_top2, iree_chain,
    iree_ffn, m5t_diagnose, model_route, native_layer, native_model,
    route_from_top2}.py`);
  - the bundle builders `moe_m5{d,e,f,g,h,k_route,m,n,q,t}_board_bundle.py`
    and `moe_m5t_diagnose.py`;
  - the route and raw Top-2 kernels and their tests;
  - `test_moe_{compiled_ffn_real, compiler_route, device_bundle,
    model_route_host, native, native_compile, native_real, native_sim,
    route_from_top2, route_from_top2_int, top2_raw}.py`;
  - the MLIR fixtures `moe_parameter_probe`, `moe_route_top2_export`;
  - the lit test `sahl_moe_route_top2.mlir`.
- **Firmware**: `moe_top2_raw`, `moe_route_from_top2`,
  `moe_route_from_top2_int` (M5's hand-written route kernels; the Top-2 is
  now the compiler's code).

**Kept:**

- `sahl.top2` and its lowering, `buildRawTop2` (the compiler's Top-2), and
  the `rv_only` flag and runtime policy;
- `test_moe_compiler_riscv.py` (router → Top-2, now with its own
  `topk_reference`);
- the M0–M4 hand-written path (`frontend/moe.py`, `moe_hf.py`,
  `runtime/tools/moe_m*.py`, `moe_model.py`, `firmware/moe_top1`,
  `moe_top2`, their tests and board scripts): the reference implementation
  M0–M4 were accepted with. (Deleted on 2026-10-25 after §24, at the tag `moe-m0-m4-final`.)

`test_moe_m3b.py` is back to its pre-M5 state, which drops M5's route-bridge
case.

**Verification:**

- Lit 28 / 28; golden 834 dispatches identical (with `moe_mixtral`);
  `test_moe_compiler_riscv`, `test_moe_m3b` pass.
- tiny-mixtral recompiled with the reduced compiler: the dispatch sources
  and `sa_packed.irpa` are identical. The vmfb, compiled at an equal-length
  path, differs from the board-accepted one only in the source paths it
  records and the zip CRCs over them.

## 20. v2 measured: not worth implementing (2026-10-25)

**Method.** `compiler/tests/moe_v2_bound.py` runs the command-level
performance model (`perf_model.py`, the 4-port overlay's calibration
`qwen3g_cal.json`) on a sim trace of the board bundle:

```
SA_RV=build/moe/native/tm/rv SA_RV_ONLY=rv_only DEPLOY=build MODEL=deploy_moe_tm MB=320 OUT=moe_tm \
    compiler/scripts/perf_trace.sh
```

**The model matches the board.** It predicts a decode step of 1.524M cycles
(15.24 ms). The board measures 16.0 ms of wall time, which also includes the
four Top-2 RISC-V calls and the host's submissions.

| Dispatch (one call) | Modeled cycles | Weights | GEMV stream bound (64 B / cycle) | Efficiency |
|---|---:|---:|---:|---:|
| FFN of an expert (gate + up, gating, amax, int8) | 127,265 | 7.34 MB | 114,688 | 90% |
| down of expert 0 | 58,858 | 3.67 MB | 57,344 | 97% |
| down of expert 1 + combine | 62,645 | 3.67 MB | 57,344 | 92% |
| classifier (for comparison) | 562,590 | 32.8 MB | 512,000 | 91% |

**What merging would save.** Per decode step (2 layers × 4 MoE dispatches):

| Schedule | Cycles | Saved |
|---|---:|---:|
| the dispatches separate | 752,068 | |
| one list per layer | 751,754 | 314 |
| one list per layer, without any FENCE (not legal: an upper bound) | 750,891 | 1,177 = 0.08% of the step |

Keeping the intermediates on chip (h: 3.5 KB, expert 0's accumulator: 4 KB,
one ST and one LD each) adds a few hundred cycles more.

**Why §9 overestimated it:**

- The batched runtime already runs a step's dispatches in a few descriptor
  lists: 3 per decode step on the board, split at the Top-2 RISC-V calls. A
  dispatch boundary is therefore not a list overhead; at most it is a
  FENCE, whose drain is short next to the weight streams.
- Each expert FFN's remaining 10% is inside its kernel: the per-chunk
  gating on the VE and the final amax / int8 pass. A merged kernel would
  keep that.

**Where the step's other 16% goes** (against the 1.28M-cycle stream bound
of the step's ~82 MB of weights):

- the decode attention, about 41K cycles per layer;
- the FFN tails, about 12.6K per expert;
- the small dispatches (norms, quantization, router);
- on the board, the Top-2 calls.

None of these is specific to MoE. Speeding up decode further is the dense
path's agenda (CLAUDE.md: W4A8 weights, LD / GEMV width, command issue).

## 21. A trained checkpoint: Granite-3.1-1B-A400M on the host (2026-10-25)

**Why Granite.** tiny-mixtral has random weights, so its numbers say nothing
about quality. `ibm-granite/granite-3.1-1b-a400m-base` (Apache-2.0, native in
transformers) is a trained MoE that is close to what v1 implements: 24
layers, H = 1024, 32 experts of I = 512, Top-8, the softmax over the Top-k
logits (Mixtral's renormalized Top-k probabilities), gate / up / down tables
laid out as Mixtral's, no shared expert. About 1.33 GB in int8, 400M
parameters active per token.

**Frontend.** `QMixtralMoe` now takes any Top-k router of that semantics
(`hf_generic.MOE_ROUTERS`: Mixtral's, Granite MoE's); the router's expert
axis is padded to a multiple of `ROUTER_PAD`, and `moe_rewrite` also matches
`GraniteMoeMoE` (its router is `router`, not `gate`). Granite's scalar
multipliers (embedding, residual, attention, logits) stay in the HF code and
are traced with it. The tiny-mixtral export is byte-identical to S6's
(`qllama.mlir`, `qllama.irpa`), and `test_hf_moe.py` passes. The backend is
unchanged: it still takes Top-2 of 4 to 16 experts (`FormMoeRegions`), so
Granite does not compile yet.

**`quality_eval.py --moe-routing`** records, per MoE layer and position, the
experts each variant selects, against the fp32 model's; and, for a quantized
variant, the Top-k of an fp32 router on the variant's own hidden state (the
router's quantization alone).

**Results** (WikiText-2 test, 2048 tokens, windows of 256, teacher-forced,
against the fp32 HF model; `build/quality/granite_moe_2048.json`):

| Variant | PPL | top-1 vs fp32 | KL | Experts in common with fp32 | Same set of 8 | W8A8 router vs fp32 router, same input |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 10.32 | | | | | |
| w8a8 | 10.99 | 91.8% | 0.062 | 97.7% | 81.7% | 99.3% (94.4% same set) |
| sa:kch (the board's form) | 10.90 | 90.3% | 0.069 | 97.4% | 79.5% | 99.3% (94.4% same set) |

`sa:kch` is W8A8 plus the int8 KV cache with K per channel (scales from 1024
WikiText-2 train tokens, as the dense bundles), RoPE tables and the device's
attention and SFU. The worst layer (5) still shares 96.2% of its experts.

**Reading.**

- The quantization loss is in the dense models' range under the same
  protocol (self-quantized W8A8, 2048 tokens): KL 0.062 against SmolLM2's
  0.102 and Qwen3's 0.039; top-1 91.8% against 87.6% and 89.1%. MoE adds no
  visible loss of its own.
- Routing differs from fp32 mostly through the hidden states, not the
  router: on the same input the W8A8 router keeps 99.3% of the experts. The
  differences are near ties, as on tiny-mixtral (R5). An fp32 router is not
  needed.
- With 8 of 32 experts, a changed set usually swaps one low-weight expert
  (97.7% of the experts in common while the whole set matches at 81.7% of
  the positions), so a routing difference moves the output little.

**What compiling it needs.** Top-k for k up to 8 and E up to 32 in
`FormMoeRegions` and the RISC-V Top-k; 24 layers × 8 expert calls per decode
step (each expert 1 MB gate + up and 0.5 MB down), so the per-dispatch fixed
cost weighs more than on tiny-mixtral (§20's v2 conclusion to recheck); and
on the board, a u-dma-buf pool larger than today's 1280 MiB.

## 22. Granite decode estimated from a proxy (2026-10-25)

The backend cannot compile Granite yet (§21), but the cost of each dispatch
depends on its shapes, not on k or E. `compiler/scripts/moe_granite_proxy.sh`
builds a proxy with Granite's shapes in a form v1 compiles: a random-weight
Mixtral, 2 layers, H = 1024, I = 512, 16 heads of 64 (8 KV heads), vocabulary
49152 (tied), 16 experts, Top-2. It exports, compiles (`--iree-sa-form-moe`,
D = 16, 4 ports), builds the RISC-V image and traces 2 prefill chunks and 2
decode steps on the sim (`build/perfmodel/moe_gp.trace`). The command-level
model (`perf_model.py`, the 4-port calibration `qwen3g_cal.json`) gives
each dispatch's cycles. The proxy compiles to 54 decode dispatches per step,
all on the SA (the Top-2 as `rv_only` code, not in the trace).

**One proxy layer, decode, modeled cycles** (T ≈ 34):

| Part | Cycles | GEMV bound (64 B / cycle) |
|---|---:|---:|
| norms (2) | 7.7K | |
| q, k, v, o projections | 55.6K | 49.2K |
| RoPE, cache writes | 1.9K | |
| attention core | 28.1K | |
| router (16 × 1024, EX path) | 4.3K | |
| softmax of the Top-2 | 0.8K | |
| expert FFN, one call (gate + up, gating, amax, int8) | 21.2K | 16.4K (77%) |
| expert down, first expert | 9.3K | 8.2K (88%) |
| expert down, second expert + combine | 13.1K | 8.2K (62%) |

**Extrapolated to Granite** (24 layers, Top-8 of 32; the router twice the
proxy's rows; the down of every expert after the first priced with the
combine, 13.1K):

| Part | Cycles per step | Share |
|---|---:|---:|
| experts' FFN (24 × 8 × 21.2K) | 4.07M | 41% |
| experts' down + combine (24 × (9.3K + 7 × 13.1K)) | 2.42M | 25% |
| attention, norms, router (24 × 102.6K) | 2.46M | 25% |
| classifier + embedding | 0.90M | 9% |
| **step** | **9.85M = 98.5 ms** | |

If the combine is paid once per layer rather than per expert, the downs take
1.88M and the step 93 ms. The weight-stream bound of the step's 429 MB is
67 ms, so the device sits at 68–72% of it, against about 91% for the dense
Llama-3.2-1B (`results_v2.3.md`).

**On the board**, add the Top-k calls: the board measured 0.76 ms more than
the model per tiny-mixtral step (§20), for 2 MoE layers, which covers the
Top-2 on the PicoRV32, the descriptor list split around it and the host's
submission. Scaled to 24 layers that is about 9 ms, more if the Top-8 of 32
takes noticeably longer than the Top-2 of 8. **Estimate: about 100–110 ms per
decode step, 9–10 tok/s**, at short context; the attention core grows with T.

**Where the gap to the bound is.** The experts' time over their stream bound
is 1.78M cycles (18 ms), from two places:

- each expert FFN's tail (the gating of the last chunk, amax, int8 output:
  4.8K per call) is not overlapped with anything. At I = 512 an expert is only
  1 MB of gate + up, so the tail is 23% of its stream, against 10% for
  tiny-mixtral's I = 3584 (§20);
- the down projections' fixed costs and the combine.

Joining a layer's expert dispatches into one list, or dropping the FENCEs
(§20's measure, on the proxy's Top-2), saves 1.3K cycles per 2 layers: the
tails are dependencies, not dispatch overhead. Recovering them takes a
schedule that streams expert i + 1's weights during expert i's tail, i.e. one
kernel over the k experts (§9's v2, now with a real target: up to about 18%
of the step for Granite, against 0.1% for tiny-mixtral). Measured on the
board, the Top-k calls are the other item to look at.

## 23. Top-k of up to 64 experts: Granite on the sim (2026-10-25)

**What changed.**

- `FormMoeRegions` takes a router's Top-k for 1 ≤ k < E, k ≤ 16, of 4 to
  64 experts (the Torch sort's [1, k] prefix, or a topk with the sentinel
  initialization); the region carries `sa.moe.top_k` = k and
  `sa.moe.kind = "topk"`.
- `sahl.top2` became `sahl.topk` with a `k` attribute; `sahl-to-sahw`
  records `sa.moe.raw_topk` (E) and `sa.moe.top_k` on the template.
- **The RISC-V Top-k** (`SahwToRiscv.cpp`, `buildRawTopk`) replaces the fully
  unrolled Top-2 (E × k compare-and-swap networks: 256 at Top-8 of 32, about
  5,000 instructions, each fetched from DDR at about 27 cycles). Each score
  becomes an unsigned key in the order of its f32 value (±0 one key), and
  enters a descending list of k entries in the stack only when it beats the
  k-th key. Most scores cost a load, the key and one compare; the ids are
  scanned in increasing order with a strict comparison, so equal scores keep
  the lower id first, the topk's order. The outputs are those of the old
  Top-2: tiny-mixtral's decode logits are byte-equal to the board-accepted
  reference (below).
- **`iree-sa-isolate-moe-experts`** (after `FormFfnRegions`, k > 2 only).
  With 8 experts, IREE's elementwise fusion merged the 8 down-projection
  epilogues and their sum into one dispatch of 30 bindings (each expert's
  int32 result, its row scales selected by id, its s_x), past the 15 BASE
  registers. A `util.optimization_barrier` on each expert's term (the end of
  its epilogue, before the sum) keeps the epilogue in the expert's
  down-projection dispatch, and the sum reads one f32 vector per expert.
  Top-2 builds are unchanged.
- **Frontend.** Granite divides the logits by `logits_scaling` (6); the VE
  divides only as 1 / x, so `hf_generic.quantize` folds the factor into the
  classifier's row scales (the reference changes by rounding only).
  `test_hf_moe.py` and `test_moe_sim.py` take `--layers N` and both router
  layouts (Mixtral's `mlp.gate`, Granite's `block_sparse_moe.router`); on a
  truncated model, whose logits are nearly flat, the top token is reported,
  not checked, and the router check skips the fp32 router's own near ties.

**Granite, 2 layers** (`test_hf_moe.py --model .../granite-3.1-1b-a400m-base
--layers 2 --export build/moe/granite/l2`, `compile_sa.sh` with
`--iree-sa-form-moe`, D = 16, 4 ports):

- 211 dispatches, all on the SA, none on the host; the Top-8 as RISC-V code.
- `dispatch_check --ids 7,0,31,12,5,19,26,3 --rv`: 211 / 211 at T = 16
  (`--pos 0`), 32 (`--dirty`), 48 and 64. At T = 17 and 33 the decode
  attention kernel reports a TRANSPOSE shape error; the export's `--pad=16`
  never runs those lengths (T is a multiple of 16), and the kernel passes at
  every multiple.
- `test_moe_sim.py --layers 2`: prefill ≡ decode (byte-equal) and plan A ≡
  descriptor lists (byte-equal, 28 steps); against the host model, argmax 27 /
  28 and correlation min 0.991, mean 0.997. The truncated model's logits are
  nearly flat and its routing has near ties: a different expert from a
  rounding difference changes a step more than on tiny-mixtral (0.9996).
- Decode on the sim: 2 Top-8 calls per step (one per layer), 19 descriptor
  lists per step.

**Regression.**

- tiny-mixtral recompiled (`build/moe/native/tm_topk`): 198 dispatches on the
  SA; `test_moe_sim.py` PASS; its decode logits byte-equal to
  `build/deploy_moe_tm/expected_logits.npy`, the reference the board matched
  in S7; `dispatch_check --ids 7,0 --rv` 198 / 198 at T = 16 and 48.
- Golden: the only change was tiny-mixtral's Top-2 dispatch, whose embedded
  RISC-V code is new (golden's disassembly does not see the code, so it
  reported the dispatch as equal up to register numbering). Re-recorded, and
  a case `moe_granite` (211 dispatches) added: 1045 unique dispatches, equal
  in two comparisons. Lit: `sa_isolate_moe_experts.mlir`,
  `sahl_moe_topk.mlir` (Top-8 of 32), `sa_form_moe_regions.mlir` (Granite's
  Torch spelling).

**Next.** The whole 24-layer model on the sim, then the board (§24).

## 24. Granite on the KV260 (accepted 2026-10-25)

**The pool.** About 1.33 GB of int8 weights: the board's u-dma-buf pool went
from 1280 to 1536 MiB, the global CMA from 256 to 128 MiB
(`docs/kv260_upgrade_plan.md` K5-M).

**Export.** `test_hf_moe.py --model .../granite-3.1-1b-a400m-base --export-only
--prefill 0`: decode only. With prefill rows (one at a time, each with its 8
experts), the traced graph of 24 layers grows past 10 GB (a 4-layer export:
0.27 GB of parameters, 1.9 GB more while exporting, linear in the layers);
grouping a chunk's rows by expert is M6. The KV scales are calibrated on 3
tokens and the cache is 64 long: a bit-exactness and speed bundle, not the
quality one (that is K per channel on 1024 WikiText-2 tokens, §21).

**The checks against HF and the host model.** Two of `test_hf_moe.py`'s checks
and one of `test_moe_sim.py`'s are reported, not asserted, for this model
(`--hf-reported`, `--host-reported`):

- W8A8 against HF fp32: at the first position the logits differ by 137%
  (rel L2), against about 20% at the next ones. The first token's hidden
  state has large outliers that per-token int8 quantization handles badly. The
  W8A8 top token differs from fp32's at about 10% of positions (§21), and
  through 24 layers the hidden states' error changes some experts where HF's
  routing is clear. The router's own check (W8A8 against an fp32 router on the
  same hidden state, where its gap is clear) stays asserted and passes.
- The device against the host W8A8 model: argmax 20 / 28, correlation min
  0.955, mean 0.985. **The host against itself with 1e-6 relative noise on the
  routers' inputs: 22 / 28, min 0.965, mean 0.989**, the same spread. The
  routing has near ties (8th vs 9th score within 0.01 in 68 of 672 layer-steps,
  within 0.001 in 5), so a rounding difference swaps an expert and the change
  grows through the layers. Bit-exactness is checked where it is defined:
  `dispatch_check` (60 / 60 at T = 16, 32, 48, 64), plan A against the
  descriptor lists (byte-equal), and the board against the sim.

**Command buffer splits.** The first board run took 141.4 ms per step with 217
descriptor lists per step. §23's `util.optimization_barrier`s reached Stream,
where each split the command buffer, one list per expert. They are now marked
`sa.moe.isolate` and `iree-sa-drop-moe-isolation` removes them in the dispatch
region formation hook (IREE patch 0004), once they have kept the experts'
epilogues apart: 25 lists per step (one per layer, split at the Top-8 calls,
and the classifier), the dispatches and logits unchanged. Wrapping each
expert's chain in a dispatch region instead failed: the contraction's
`collapse_shape` stayed inside before the epilogue (IREE's reshape propagation
moves it out of the way otherwise), and the backend fell back to the host.

**Board** (commit `35db39f`, `d16_100mhz_w128_p6_gemv_np4_pcpi`, 100 MHz,
rt_fw v7, window 1315 MB = the sim's peak 1282.4 MiB + 32;
`build/deploy_moe_granite24/results/35db39f/moe_granite24/.../llm` and
`.../llm_decode`):

| Test | Logits vs sim | Tokens | Speed | Dispatches per step |
|---|---|---|---|---|
| `llm --decode-only`: 28 steps | bit-exact 28 / 28 | identical | 104.4 ms per step (9.58 tok/s) | 25 lists, 24 RISC-V (Top-8) |
| `llm` (no prefill export: the same decode run) | bit-exact 28 / 28 | identical | 104.4 ms per step | 25 lists, 24 RISC-V |

Against §22's estimate (93–99 ms of device time, about 100–110 ms on the
board with the Top-k calls) the step lands at the low end of the board
range. The weight-stream bound of the step's 429 MB is 67 ms: the gap is the
experts' tails (§22) and the 24 Top-8 calls. `deploy_moe.sh` read the sim's
peak as 10^6 bytes (it is MiB): the first board run ran out of device memory
loading the parameters (fixed in `9797dbf`).

**Next.**

- A kernel over a layer's k experts (§9's v2): up to about 18% of the step
  (§22), the tails overlapped with the next expert's weight stream.
- The quality bundle: K per channel, the KV scales on 1024 WikiText-2 tokens, a
  longer cache; teacher-forced on the board.
- Prefill: M6, grouped by expert (also the export's memory).

## 25. Granite's quality on the KV260 (2026-10-25)

**The bundle** (`build/moe/granite/granite24q` → `build/deploy_moe_granite24q`,
commit `15dcf0e`): the dense models' long-context settings. `test_hf_moe.py
--export-only --prefill 0 --max-len 2048 --kv-channel k --kv-calib
build/llm_cache/wikitext-2-raw-v1-train.txt:2048`: K per channel, the KV scales
on 2048 WikiText-2 train tokens (`hf_generic.load`), a 2048-token cache,
decode only. 60 executables on the SA; `dispatch_check` 60 / 60 at T = 16,
1024 and 2048 (the attention's tiled form); sim device peak 1330 MiB, window
1362 MB. `board_quality.py prep --tokens 4096 --window 2048`: two windows of
the WikiText-2 test set from token 0, fp32 PPL 6.386.

**Board** (`d16_100mhz_w128_p6_gemv_np4_pcpi`, 100 MHz, rt_fw v7;
`build/deploy_moe_granite24q/results/15dcf0e/...`, scored into
`build/quality/granite_board_2048.json`):

| Test | Result |
|---|---|
| `llm` (decode only, 28 steps) | logits bit-exact with the sim 28 / 28, tokens identical |
| `teacher` (2 × 2048 tokens through decode) | **PPL 6.528** (fp32 6.386), **top-1 vs fp32 92.1%**, **KL(fp32 ‖ device) 0.030** over the top 64 (94.5% of the mass); 157.7 ms per step on average over positions 0–2047 |
| `profile 40 --prompt 32 --batch` | 111.3 ms per step (8.98 tok/s): device 103.2 ms = 25 descriptor lists 94.1 ms + 24 Top-8 calls 9.1 ms (8.8%); the rest of the runtime 5.8 ms |

**Against the dense models** (board teacher forcing, `results_v2.2.md` §3;
SmolLM2, Qwen3, Llama at 2048, Gemma at 512): top-1 90.7%, 88.4%, 91.9%,
76.9%; Llama's KL 0.031. Granite's W8A8 loss is among the smallest, and its
first position's outliers (§24) do not show in the windows' totals.

**Speed in context.** The descriptor lists' 94.1 ms match §22's 93–99 ms of
modeled device time. The Top-8 calls cost 0.38 ms each, as the tiny-mixtral
measurement predicted (§22). Decode slows with the context: 157.7 ms per step
on average over a 2048-token window.

**Next.** v2's kernel over a layer's experts (up to about 18% of the step,
§22); the Top-8 calls (8.8%); prefill (M6).

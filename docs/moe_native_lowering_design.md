# Native MoE lowering: Mixtral experts as ordinary SA dispatches

> Status (2026-10-25): design; S0 spike done (§11), the rest not implemented.
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

`hf_generic` replaces each `MixtralSparseMoeBlock` (its forward loops over the
experts with data-dependent masks and cannot be exported) with `QMixtralMoe`.
It reuses `QLinear`'s conventions (`_quant_act`, `_quantize_rows`):

```python
class QMixtralMoe(torch.nn.Module):
    # router: QLinear with E rows padded to Ep = a multiple of D (zero rows),
    #         pad: f32 [Ep], 0 for the E experts, -inf for the padding
    # gate_q, up_q: int8 [E, I, H]   gate_s, up_s: f32 [E, I]
    # down_q:       int8 [E, H, I]   down_s:       f32 [E, H]
    def forward(self, x):                                  # x: [1, H] f32
        scores = self.router(x) + self.pad                 # [1, Ep] f32, no slice
        vals, ids = torch.topk(scores, 2, dim=-1)          # descending
        w = _softmax2(vals)                                # exp / sum / recip, no linalg.softmax
        y = 0
        for k in range(2):                                 # unrolled
            e = ids[:, k]                                  # [1] i64
            g = _qlinear_rows(x, self.gate_q.index_select(0, e)[0],
                              self.gate_s.index_select(0, e)[0])
            u = _qlinear_rows(x, self.up_q.index_select(0, e)[0],
                              self.up_s.index_select(0, e)[0])
            h = torch.nn.functional.silu(g) * u
            yk = _qlinear_rows(h, self.down_q.index_select(0, e)[0],
                               self.down_s.index_select(0, e)[0])
            y = y + w[:, k:k + 1] * yk
        return y
```

- `_qlinear_rows` is `QLinear.forward` with the weight and scale passed in. The
  graph per expert is then the same as a dense SwiGLU FFN except for the
  weight operand, so `FormFfnRegions` and the GEMV matchers see the forms they
  already know. `index_select` exports as a gather generic, which the compiler
  rewrites into a dynamic slice (§5.1). Left as it is, IREE fuses the gather
  with the `.to(int32)` into a dispatch that writes an i32 copy of the expert,
  followed by an i32 × i32 → i64 matmul, and both run on the host (§11.1).
- The router's expert axis stays padded to a multiple of D, and `-inf` scores
  mark the padding. Slicing the scores to E < D puts the dequantization on the
  host ("innermost size not a multiple of D"), and the slice between the
  projection and the sort also hides the router from Top-2 recognition
  (§11.1). With finite real scores, Top-2 never picks a padded lane.
- `torch.softmax` exports as `linalg.softmax`, which the SA backend does not
  lower (§11.1). `_softmax2` writes it out as max, exp, sum and recip, as
  `hf_generic`'s attention already does.
- The activation is quantized once. Router and both experts compute
  `_quant_act(x)` on the same `x`, and IREE CSE merges them.
- **Semantics.** Mixtral renormalizes the Top-2 probabilities, which equals a
  softmax over the two selected logits. Ties go to the lower expert ID. The
  host reference uses a stable descending sort, because `torch.topk` does not
  specify the order of ties.
- Prefill: for `T > 1` rows, `forward` runs row by row (the `_attention_rows`
  idea), each row with its own Top-2.
- The router is W8A8, like M4f (8/8 routing agreement with fp32 in M4b).
  §10 R5 has the fp32 option.

Calibration and quality evaluation reuse the existing `hf_generic` and
`quality_eval.py` flows. The packed `sa-moe-parameters-v1` format remains only
for the M0–M5 tests.

## 5. Preprocessing passes

Inside `extendPreprocessingPassPipeline`: the new expert-gather rewrite (§5.1)
runs first, then `PackLinearWeights` → `CloneCheapProducers` →
`FormMoeRegions` (rewritten) → `FormFfnRegions` → `FormAttnRegions` (after
IREE region formation). `LowerMoeChain` is deleted.

### 5.1 Expert gathers to slices, then `PackLinearWeights`

**Gather to slice** (a rewrite in `FormMoeRegions`'s file, run before
packing). The pattern is a gather generic `G[n, k] = T[e[0], n, k]` (or
`[n]` for a scale table) where:

- `T` is a constant table;
- `e` is a one-element index tensor;
- the body is a single `tensor.extract` whose first index is
  `index_cast(e[0])` and whose other indices are the loop indices in order.

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
| S1 frontend | `QMixtralMoe` in `hf_generic` (padded router, `_softmax2`), synthetic fixture, tiny-mixtral export | Host reference agrees with HF top tokens and IDs; no `linalg.softmax`, no sliced scores |
| S2 packing | Gather-to-slice rewrite, `PackLinearWeights` expert tables | Lit; table packed at compile time (in the parameter archive) |
| S3 regions | `FormMoeRegions` rewrite | Lit, including 2 layers per function |
| S4 backend | `ddrOf` dynamic index, LDPARAM sequence, kernels accept dynamic weights | `dispatch_check` on the synthetic model, all IDs |
| S5 Top-2 | Semantic record, funcsim Top-2, runtime `rv_only` policy, compiler emits only `rv_only` code | `dispatch_check` of top2; sim e2e with `SA_RV` unset |
| S6 e2e sim | tiny-mixtral through `compile_sa.sh` | Gates 1–4 of §7; golden re-recorded |
| S7 board | Bundle, board run | Gates 5–6 |
| S8 cleanup | Delete §6.3 and the M5 milestone scripts; update `moe_implementation_plan.md` and CLAUDE.md | Lit + golden |

Estimate: S0 1 day, S1–S3 3–4 days, S4 3–4 days, S5 2 days, S6–S8 2–3 days.

## 9. v2: one dispatch for both experts (after v1 is accepted)

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
- **R5: router precision.** W8A8 routing flips on real checkpoints would show
  up as ID disagreement in S6 on a trained model. The option is an fp32
  router: E rows times H as VE multiply + reduce-sum over fp32 weights (E·H/D
  = 512 VE words for tiny-mixtral), a new generic form, still bit-exact with
  funcsim.
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

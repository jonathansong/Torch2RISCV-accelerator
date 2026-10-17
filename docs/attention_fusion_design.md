# Attention fusion: design

Status: design (2026-10-18), not implemented. The second fused chain after
SwiGLU ([`fusion_design.md`](fusion_design.md) §4, on by default since
2026-10-18). Bounds from `compiler/tests/fusion_whatif.py` on the `v2.2-kv260`
traces (§2); the attention kernels it builds on are those of C6.5
([`iree_compiler_plan.md`](iree_compiler_plan.md) §8.22).

**In one paragraph.** A decode step spends 13 to 18 dispatches per layer on
attention: the preparation of q and k (dequantization, QK-norm, RoPE, int8),
the two cache writes, the scores, the probabilities, their sum, p · V and the
quantization before `o_proj`. Each boundary costs a round trip through DDR
on the dependency path. Fusing the q path with the attention core and the
output quantization into one dispatch per layer is worth up to −3.3% of
decode and −5.6% of prefill on Qwen3. Up to −6.4% / −11% with every piece
on SmolLM2. Prefill gains more because its attention runs row by row, each
row as a decode step, so every boundary is paid 16 times per chunk. The
gain at long contexts is larger still and not in these bounds: inside one
dispatch the SFU-bound probabilities of one KV head can overlap the EX- and
LD-bound scores and the GEMV-bound p · V of the next. The numerics need no
change: the only fp32 reduction over T is a max, so the fused build is
bit-exact with the unfused one. The hard part is not capacity (everything
fits) but the per-bank scoreboard (§5): SPAD_A has two banks for three roles,
so the schedule is to be prototyped in the performance model before the
compiler work starts.

---

## 1. The chain today (Qwen3-0.6B decode, one layer)

Numbering of the 2-layer build (`build/fuse/qwen3_l2`); the other models
have the same chain without QK-norm (SmolLM2, Llama) or with Gemma's own
norms and scaling.

| # | Dispatch kind | Work | Stage (§3) |
|---|---|---|---|
| 7 | `matvec_like_128x16x1024` | q_proj → int32 [2048] | 3 |
| 8 | `elementwise_2048` | q dequantized (s_x · s_w) → fp32 | **1** |
| 9 | `reduction_16x128` | q QK-norm per head (sum of squares, rsqrt, weight) | **1** |
| 10 | `matvec_like_64x16x1024` | k_proj → int32 | 2 |
| 11 | `reduction_8x128` | k dequantized, QK-norm per head | 2 |
| 12 | `matvec_like_64x16x1024` | v_proj → int8 (WP9) | 2 |
| 13 | `elementwise_broadcast_2048` | q rearranged for RoPE (a copy) | **1** |
| 14 | `elementwise_8x128` | k RoPE, int8 per channel → the K row | 2 |
| 15 | `slow_memcpy` | the K row into the K cache | 2 |
| 16 | `scatter` | the V row into the packed V cache (one ST of the groups' rows) | 2 |
| 17 | `elementwise_16x128` | q RoPE | **1** |
| 18 | `reduction_16x128` | q amax per head → s_q | **1** |
| 19 | `elementwise_16x128` | q → int8 (held as i32) | **1** |
| 20 | `batch_matmul_8x2xDx128` | scores q · Kᵀ (EX; the T loop of §8.22 past one bank) × s_q · s_k → fp32 [16, T] | **1** |
| 21 | `reduction_16xD` | masked max over T; p = i8(exp(s − max) · 127) (SFU) | **1** |
| 22 | `reduction_16xD` | Σp (fp32 sum of integers, exact) | **1** |
| 23 | `matmul_like_8x2x8x16xD` | p · V on the GEMV unit (packed V), ÷ Σp, × s_v → fp32 [2048] | **1** |
| 24 | `reduction_2048` | amax of the attention output | **1** |
| 25 | `elementwise_2048` | → int8 | **1** |
| 26 | `matvec_like_64x16x2048` | o_proj + residual | 3 |

The dispatches between the projections run on local data of a few KB (the
q row is 2048 values). Their time is mostly the boundary itself: the
producer's last ST, which the next dispatch waits for, then the consumer's
first LDs, which its work waits for ([`fusion_design.md`](fusion_design.md)
§2.2).

**Prefill** (`hf_generic._attention_rows`) computes the q / k / v
projections for the 16 rows of a chunk together, then runs each row exactly
as a decode step (dispatches 8–25 at the row's position, the same executables:
IREE shares them across the module), then `o_proj` for the 16 rows. A
fused row kernel therefore serves decode and prefill alike, and prefill
pays every boundary it removes 16 times per chunk.

## 2. What it is worth

### 2.1 Upper bounds (performance model)

Method as in [`fusion_design.md`](fusion_design.md) §2.1: the intermediates'
STs and the LDs that read them back are removed from the joined step, with
no barriers in between. The KV cache is persistent, so the cache writes →
scores round trip is not counted, whatever the region. Chains selected by
role (`--chain`, §5 there), prefill chunk / decode step:

| Chain | SmolLM2 | Gemma | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| Attention + q / k preparation + quantization before `o_proj` | −9.0 / −5.6% | −7.2 / −2.0% | −7.6 / −3.5% | −3.5 / −1.2% |
| … with the q / k / v and `o_proj` projections | −11.0 / −6.4% | −8.7 / −2.4% | −9.3 / −3.9% | −4.2 / −1.4% |

Qwen3 in stages (`--only` by kind: Qwen3's attention kinds do not collide
with the others; `build/perfmodel/attn_stages.sh`):

| Fused | Prefill | Decode |
|---|---:|---:|
| Core: scores, probabilities, Σp, p · V (20–23) | −2.1% | −1.1% |
| + the output quantization (24, 25) | −2.1% | −1.7% |
| **+ the q path (8, 9, 13, 17–19): stage 1** | **−5.6%** | **−3.3%** |
| + q_proj | −6.5% | −3.4% |
| + the k / v path and projections (10–12, 14–16): stage 2 | −7.5% | −3.7% |
| + o_proj | −7.5% | −3.8% |
| The k / v path alone, with its projections | −1.0% | −0.5% |

The q path is where most of the gain is. It sits on the dependency path
from q_proj to the scores, and its five dispatches are small.

### 2.2 Long contexts (beyond the bounds)

The bounds keep each dispatch's commands and order, and the traces are at a
32-token prompt. At long T the attention dispatches dominate, and each is
bound by a different engine. Qwen3 per layer at T = 2048 (cycle model, §8.22
"Where decode attention's time goes"):

| Dispatch | Total | Bound by |
|---|---:|---|
| q · Kᵀ (T loop, steps a–c) | 276k | EX 172k, LD 140k, VE 88k |
| masked max + exp | 100k | VE / SFU (≈ 90% busy) |
| Σp | 24k | VE |
| p · V (GEMV) | 74k | LD (72k) |
| **sum, as today** | **474k** | |
| per engine, if overlapped | | LD ≈ 212k, VE ≈ 200k, EX ≈ 172k |

In one dispatch, KV head h's probabilities (VE) can run while head h + 1's
K blocks stream (LD) and its scores compute (EX). Head h's p · V (GEMV, on
the LD engine) then fills the LD engine's gaps. The floor is the busiest
engine, about 212k cycles against 474k. That would cut up to about
−73 ms of the 310 ms Qwen3 decode step at 2048 (board, `qwen3_lc2048_pcpi`),
about −24%, if the overlap were perfect. It will not be: LD and VE are both
near the floor, and §5's false dependencies cost some of it. The schedule
prototype (§6 step 1) gives the real figure. Gemma has a single KV head
(Hk = 1), so it gets no head-level overlap. Its attention at T ≤ 512 is small
anyway (38k cycles per layer).

## 3. Scope and stages

1. **Stage 1 (this design): q path + core + output quantization**,
   dispatches 8, 9, 13, 17–25 → one dispatch per layer (12 → 1 for Qwen3;
   SmolLM2 / Llama have no QK-norm, Gemma has its own norms, the same
   structure). Inputs: q_proj's int32 output and its scales, the QK-norm
   weights, the RoPE rows at the position, the K / V caches (after their
   writes), the position (the masked max reads it), the scales. Output:
   the int8 attention row and its scale, which `o_proj` reads. Decode and
   prefill rows.
2. **Stage 2: the k / v path** (11, 14–16 with the k / v projections'
   epilogues) as a second region ending in the cache writes. Small (−0.5 /
   −1.0% on Qwen3); it keeps the cache writes before the attention region's
   reads, so no DDR RAW inside a dispatch (DDR is not tracked by the
   scoreboard; a fused write-then-read would need a FENCE).
3. **Later**: q_proj (and o_proj) inside the region, as SwiGLU does with its
   projections (−0.1 / −0.9%); all 16 prefill rows in one dispatch (each with
   its own T; beyond the bounds above).

## 4. Design (stage 1)

### 4.1 Frontend / preprocessing: `iree-sa-form-attn-regions`

Next to `iree-sa-form-ffn-regions` (`compiler/plugins/sa/target/FormFfnRegions.cpp`),
same phase (before dispatch creation, after weight packing and
`clone-cheap-producers`):

- **Match** structurally, anchored on the scores contraction (a batched
  contraction i32 × i8 → … whose B operand is a slice `[:, :T]` of a KV cache
  global, T dynamic). Walk back along the q operand through elementwise and
  per-head reduction generics to the q_proj contraction (stopping there).
  Walk forward through the masked max / exp generic (the one reading the
  position), the Σp reduction, the p · V contraction over the V cache slice
  and its epilogue, the amax and the quantization.
- **Wrap** them in one `flow.dispatch.region` yielding the int8 row and the
  scale. The region captures T (the cache slice's size) and the position
  value; its result shapes are static.
- Flag `--iree-sa-fuse-attn` (default off; `compile_sa.sh` `SA_FUSE_ATTN`),
  golden corpus identical with it off. Lit: a decode attention in linalg (one
  with QK-norm, one without) → one region; a near miss (scores not on a
  cache slice) left alone.
- **Risk to check first** (§6 step 2): regions with dynamic dims this
  early. IREE's own dispatch formation produces today's dynamic attention
  dispatches, but a region formed by hand must carry the workload ordinals
  for T. The FFN regions were static.

### 4.2 Backend: one `sahl.kernel "attn_layer"`

`SahlKernels.cpp` matches the fused dispatch as a whole (an `AttnLayerPlan`:
the q-path generics, the scores contraction and its epilogue, the two
reductions over T, the p · V contraction and its epilogue, the amax, the
quantization). `SahlExpandKernels.cpp` gets `expandAttnLayer`, written as
`expandFfn` is: the commands in sahl, reusing the pieces that exist:

- the q path per head as generic pieces in scopes (the generic lowering, as
  the FFN's gating pieces);
- the A tile of a KV head's G query rows (step b of §8.22, built from int8
  q in SPAD_A instead of loaded from DDR);
- the T loop of q · Kᵀ (steps a–c: blocks of Tb = 128, software-pipelined,
  `sa.advance_final`, no read past T);
- the probabilities as row pieces over the dynamic T (C6.5b), per head;
- p · V on the GEMV unit (`gemvDyn`, x = the head's p rows from SPAD_A, more
  than 8 groups in commands of 8);
- the amax accumulated per head (`sahl.accumulate`, as the FFN's), the
  quantization after the last head, then the two STs.

The schedule, per KV head h of Hk (G query rows each):

```
q path for all heads: dequantize, QK-norm, RoPE, amax per head, int8      (VE; small)
A tiles of all KV heads                                                    (VE)
for h in 0 … Hk − 1:
    scores(h):  T loop over K blocks: LD, transpose, EX, copies            (LD, VE, EX)
    probs(h−1): masked max, exp → p, Σp                                    (VE / SFU)
    pv(h−1):    GEMV over the packed V cache, ÷ Σp, × s_v, amax piece      (LD mode 2, VE)
probs(Hk−1), pv(Hk−1)
output: int8 by the amax; ST row, ST scale
```

The skew by one head is the FFN's skew by one chunk: commands issue in order
into queues of four, so head h + 1's LDs and EXs must be issued before head
h's VE work, or the VE waits behind them. Whether probs(h − 1) sits before or
after the T loop of h, or is interleaved with its blocks, is what the
prototype (§6 step 1) decides.

### 4.3 Local memory (D = 16; words: SPAD 16 B, ACC 64 B = 16 fp32)

Per KV head at the compiled maximum T (SmolLM2 / Qwen3 / Llama 2048, Gemma
512), two heads in flight:

| | SmolLM2 | Gemma | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| Hq / Hk / head_dim (G) | 9 / 3 / 64 (3) | 4 / 1 / 256 (4) | 16 / 8 / 128 (2) | 32 / 8 / 64 (4) |
| ACC: scores G · T fp32, per head | 384 | 128 | 256 | 512 |
| ACC: block scratch (D slots × Tb / 16) | 128 | 128 | 128 | 128 |
| ACC: the attention output (Hq · hd fp32) | 36 | 64 | 128 | 128 |
| SPAD_A: p (G · T int8), per head | 384 | 128 | 256 | 512 |
| SPAD_A: raw K block (Tb · hd) | 512 | 2048 | 1024 | 512 |
| SPAD_A: q tiles (Hq · hd int8) | 36 | 64 | 128 | 128 |
| SPAD_B: transposed tiles (as the raw block) | 512 | 2048 | 1024 | 512 |
| Capacity: ACC bank / SPAD bank | 2048 / 4096 | 2048 / 4096 | 2048 / 4096 | 2048 / 4096 |

Everything fits within one bank per role, even with two heads in flight.
Capacity is not the constraint; banks are (§5).

### 4.4 Numerics

The same operations on the same values in the same order per element. The
only fp32 reduction over T is the max (order-free). Σp sums integers below
2²⁴ (exact in any order). Scores and p · V accumulate int32. The amax is
order-free and the quantization unchanged. So the fused build must be **bit-exact**
with the unfused one, and prefill rows bit-exact with decode (they run the
same fused executable). As in C6.5, no online softmax: block-local maxima
would change p's int8 rounding.

## 5. The hard part: banks

The scoreboard tracks hazards per bank (SPAD_A ×2, SPAD_B ×2, ACC ×2): RAW,
WAR and WAW against in-flight commands of *other* engines. EX and VE also
conflict on any common SPAD_B / ACC bank, even two readers
([`double_buffer_design.md`](double_buffer_design.md) §7). The GEMV unit is the LD
engine. It reads x from SPAD_A and writes ACC. The roles:

| Role | Written by | Read by | Bank |
|---|---|---|---|
| q tiles (A operand) | VE | EX | SPAD_A ? |
| raw K block | LD | VE (transpose) | SPAD_A ? |
| p (GEMV x) | VE | LD (GEMV) | SPAD_A ? |
| transposed tiles | VE | EX | SPAD_B 0 |
| block scratch, scores(h) | EX, VE | VE | ACC h mod 2 |
| p · V accumulators(h), output | LD (GEMV), VE | VE | ACC h mod 2 |

ACC and SPAD_B work with the heads alternating ACC banks, as the FFN's
chunks do. SPAD_A does not: three roles, two banks, and every pairing makes a
false dependency.

- p with the q tiles (bank 0): VE writing head h's p waits for every in-flight
  EX (head h + 1's scores read their tile there), which is the overlap we
  want gone.
- p with the raw K (bank 1): VE writing p waits for the in-flight K block LDs
  (WAW), and the next block's LD waits for p's writes (WAW). The two only
  serialize where they meet, and LD and GEMV are one engine, in order.

Options, to be measured in the prototype:

- (a) p in bank 1 with the raw K blocks, and the q tiles of all heads built
  once, before the head loop, in bank 0, which then has no writer inside the
  loop. Issue p's writes between two K block LDs, so they meet an LD queue
  that is drained there, not full.
- (b) A finer hazard granularity in the scoreboard (LLM plan §11.3: bank
  halves or address ranges). This is a bitstream change outside this design's
  scope, to be weighed against the loss that (a) measures.

EX's A operand and the GEMV's x must both be in SPAD_A, so moving a role to
SPAD_B or ACC is not an option.

The FFN's experience: the per-bank hazards decided the result, half the
gain lost until every chunk's VE work sat in its own bank (§4.8–4.9 there).
Here the cost is not known until (a) is modelled.

## 6. Plan

| Step | Content | Estimate |
|---|---|---|
| 1 | **Schedule prototype in the performance model**: one layer's fused command list built from the unfused dispatches' commands (intermediates dropped, heads skewed, banks assigned as §5 (a)), run with `perf_model.run_list` at T = 32, 512, 2048. Gives stage 1's real gain at short and long T, and whether (a)'s false dependencies matter. Go / no-go | 2 days |
| 2 | Region pass with dynamic T: the formed region through IREE's dispatch creation, `--compile-to=stream` and the backend unchanged (the region compiled as today's pieces would be, unfused inside) | 3–4 days |
| 3 | `AttnLayerPlan` match and `expandAttnLayer`, decode at T within one bank first, then the T loop | 8–12 days |
| 4 | Verification: lit; golden corpus with the flag off; `dispatch_check.py` per dispatch at T = 16, Tb ± D, 2·Tb, max T; end to end on the sim bit-exact with the unfused build (four models, 2 layers); prefill vs decode rows bit-exact; `test_long_context.py` at 2048 (512 Gemma); `host_copies.py` | 4–5 days |
| 5 | Board: the four bundles (`run_board_tests.sh`: `llm`, `profile --batch`, long context and its sweep, plan A / plan B) | 2 days |

About four weeks after a positive step 1. Expected: decode −1.4% to −6.4%,
prefill −4.2% to −11% at short T (§2.1, stage 1 slightly less), more at long
T (§2.2).

**Risks.** (1) The false dependencies of §5 eating the overlap, the reason
for step 1. (2) Dynamic-shape regions in IREE, the reason for step 2 before
the backend. (3) The per-layer generalization (`--iree-flow-enable-executable-generalization`)
must still merge the fused executables across layers: the caches' offsets and
scales as push constants, as today (checked by the decode working-set size,
`rv_codesize.py`). (4) The T loop's guarantees (no read past T, the
`sa.advance_final` epilogue) carried into a loop that now sits inside a
head loop: the per-dispatch checks at block boundaries cover it.

# Cross-operator fusion: evaluation and the SwiGLU design

Status: evaluation done (2026-10-17, `compiler/tests/fusion_whatif.py`,
[`iree_compiler_plan.md`](iree_compiler_plan.md) §8.23); the SwiGLU fusion is
implemented (`97a3323`, `--iree-sa-fuse-ffn` / `SA_FUSE_FFN=1`, default off)
and accepted on the board on all four models (2026-10-17, §4.8): bit-exact,
decode −1.1% to −2.7% of device time, about half of the model's −2.3% to
−4.4%, for a reason found in §4.8 (the fused gating is not chunked enough to
overlap the weight streaming). Measured against tag `v2.2-kv260`.

**In one paragraph.** Every IREE dispatch on the sa device stores its outputs
to DDR and the next dispatch loads them back. Keeping those intermediates in
the accelerator's local memory (SPAD / ACC) would save at most 10–20% of a
prefill chunk and 4.5–14% of a decode step (the model's upper bound, four
models). The gain is not the bytes (under 1.5% of a decode step's reads):
it is the round trip on the dependency path, a producer's last STs that the
next dispatch waits for and the consumer's first LDs that its work waits for.
Loads off that path cost nothing to remove (batching the constant scale loads
saves 0–0.7%). The gain is spread over the whole layer: no single operator chain carries more than about a third
of it. The SwiGLU chain alone (the two FFN projections, the gating, and the
quantization of its output) is worth 2.3–4.4% of decode and 2.4–4.2% of
prefill; it is the first step because its structure is the simplest and it
builds what every later chain needs (several contractions in one dispatch,
intermediates in local memory).

---

## 1. Where the time goes at dispatch boundaries

IREE forms one dispatch per contraction (with the elementwise work fused
around it), plus separate dispatches for reductions that cannot fuse with
their producer. The sa backend compiles each dispatch to one descriptor-list
template; the runtime joins a command buffer's templates into one list with a
FENCE where IREE placed a barrier. A Qwen3-0.6B decode step is 932
dispatches; one layer's FFN is five of them:

| Dispatch (Qwen3, decode) | Work | DDR traffic besides the weights |
|---|---|---|
| `31` matvec 192×16×1024 | x_q · W₂ → int32 [3072] | LD x (1 KB); ST the int32 result (8 × 1.5 KB) |
| `32` matvec 192×16×1024 | x_q · W₁ → int32; epilogue silu(a) · b with both scales | LD x again; per chunk LD `31`'s result and two scale slices (24 × 1.5 KB); ST h fp32 (8 × 1.5 KB) |
| `33` reduction 3072 | amax \|h\| | LD h (12 KB); ST the scalar |
| `34` elementwise 3072 | h → int8 by the amax | LD h, LD the scalar; ST h_q (3 KB) |
| `35` matvec 64×16×3072 | h_q · W_down (+ residual) | LD h_q, the scalar, … |

The GEMV unit streams weights through the LD engine, so the small LDs of a
dispatch (its epilogue inputs) queue behind and between the weight streams.

## 2. Evaluation

### 2.1 Method

`compiler/tests/fusion_whatif.py`, on the command-level model
`compiler/tests/perf_model.py`:

1. A trace of each bundle on the functional simulator
   (`compiler/scripts/perf_trace.sh`: two prefill chunks and two decode
   steps, one list per dispatch, every command as executed).
2. The model calibrated on the same bundle's per-dispatch board profile
   (`profile/perf.csv`, weighted error 2.3–6.2%).
3. A step's dispatches joined into one list as the batched runtime runs them,
   a FENCE before each dispatch that touches DDR an earlier dispatch wrote (or
   writes what it read) since the last FENCE. The joined list's device time is
   within 4% of the board's `profile --batch` for all four models.
4. *Intermediates*: STs whose bytes a later dispatch of the same step reads
   back and that no later step reads before rewriting them (IREE's transient
   buffers are rewritten within a step and tracked by their latest writer;
   the KV cache, logits and parameters are not intermediates).
5. Scenarios: the joined list; no barriers between dispatches; the
   intermediates' STs and the LDs reading only them removed (local memory
   unlimited); both. `--only REGEX` restricts the removal to one chain
   (producer and consumer kinds both matching).

The removal keeps every other command and the engines' timing, so each figure
is an upper bound: no capacity limit and no extra local copies are modelled.
The model is validated on the unfused lists only; a fused build must be
measured on the board.

### 2.2 Everything fused (upper bound)

Time per prefill chunk (16 tokens) / decode step, against the joined list:

| | SmolLM2-135M | Gemma-3-270M | Qwen3-0.6B | Llama-3.2-1B |
|---|---:|---:|---:|---:|
| Prefill: intermediates ST / LD read back (share of LD bytes) | 17.9 / 30.3 MB (19%) | 16.7 / 28.4 MB (19%) | 50.7 / 82.5 MB (13%) | 39.7 / 65.1 MB (6%) |
| **Prefill** | **−19.6%** | **−17.7%** | **−14.6%** | **−10.2%** |
| Decode: intermediates ST / LD read back (share of LD bytes) | 1.2 / 2.1 MB (1.5%) | 1.0 / 1.8 MB (0.7%) | 3.4 / 5.4 MB (0.9%) | 2.7 / 4.4 MB (0.4%) |
| **Decode** | **−14.1%** | **−6.8%** | **−7.6%** | **−4.5%** |
| Barriers removed, traffic kept | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |

- **Barriers cost nothing**: consecutive dispatches are serial through their
  data anyway.
- **The round trip, not the bytes**: on Qwen3's decode step removing only
  the intermediates' STs gains −4.0% (0.43M ST-engine cycles), only their LDs
  −4.1% (0.57M LD-engine cycles), both −7.6%: each is almost entirely on the
  critical path. A dispatch's last ST must finish before the dependent
  dispatch starts (and consecutive templates allocate the same local words
  from 0, so the bank hazards serialize them even without a FENCE), and the
  consumer's first LDs gate its work. Loads that are not on such a path are
  hidden in the LD engine's idle time (about 20% in decode): batching the
  constant scale loads (§5) removes 1.3–1.8 thousand LDs per decode step and
  saves nothing. Small models gain most because their steps have the same
  number of boundaries over less weight-streaming time.
- **Decode is close to reachable** (its intermediates are vectors of a few
  KB); **prefill less so**: a chunk's FFN activations (16 × 8192 fp32 for
  Llama, 512 KB) exceed ACC (256 KB).

### 2.3 One chain at a time (Qwen3)

| Chain fused alone | Prefill | Decode |
|---|---:|---:|
| SwiGLU: W₁, W₂, silu · mul, amax, quantization of h | −2.4% | −2.3% |
| Attention core: scores → max → exp / sum → p · V | −3.7% | −1.1% |
| Attention core + q / k preparation (norm, RoPE) + quantization before `o_proj` | −6.1% | −3.7% |
| Residual + RMSNorm + quantization of x | −0.7% | −0.7% |
| … with the projections before and after it | −3.7% | −1.6% |
| **Everything** | **−14.6%** | **−7.6%** |

The chains add up to roughly the whole: the gain needs the whole layer fused
eventually. (The standalone cycles of the removed commands rank the chains
differently, SwiGLU 15–31%; those overlap with other engines' work and are not
the marginal time. The per-chain runs above are.)

SwiGLU on all four models:

| SwiGLU fused | SmolLM2 | Gemma (GeGLU) | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| Prefill | −3.6% | −4.2% | −2.4% | −3.5% |
| Decode | −4.4% | −2.5% | −2.3% | −2.3% |

### 2.4 Against hardware changes (the same model, Qwen3, `perf_model.py --whatif next`)

| | Prefill | Decode |
|---|---:|---:|
| LD commands pipelined (the next LD's bursts under the previous latency) | ×1.02 | ×1.02 |
| H2: out-of-order issue with address-range checks | ×1.07 | ×1.04 |
| W4A8 decode weights | ×1.00 | ×1.53 |
| Fusion, everything (upper bound) | ×1.17 | ×1.08 |

Fusion is compiler work only (no bitstream), and its prefill bound is larger
than any of the scheduling changes; for decode, W4A8 remains the large step.

## 3. How to fuse: options

| Option | What | Verdict |
|---|---|---|
| A. IREE's `FuseHorizontalContractions` (`--iree-dispatch-creation-enable-fuse-horizontal-contractions`, off by default) | Contractions with a common LHS become one multi-result `linalg.generic` in one dispatch, the elementwise consumer fused into it | Covers W₁ / W₂ + gating only (about 45% of the SwiGLU chain's commands); the reduction and quantization stay separate; the sa backend would need a multi-result contraction anyway |
| **B. A sa preprocessing pass forming the region** | Before dispatch creation, wrap the chain's ops in a `flow.dispatch.region`; IREE's `FormDispatchRegions` leaves ops already in a region alone (`isNonNullAndOutsideDispatch`); IREE's own `MakeSingleDispatchForFunction` forms regions at the same phase | **Chosen**: the region is exactly the chain, any length, behind a flag; the backend sees ordinary linalg ops |
| C. Residency across dispatches | Dispatches stay separate; the runtime / compiler keep a buffer in local memory from one template to the next | Breaks IREE's model (dispatches communicate through bindings), needs local-memory allocation across templates; larger than B for the same gain |

## 4. The SwiGLU fusion (decode)

### 4.1 Target

One dispatch per layer replacing `31`–`34` (Qwen3 numbering):

```
LD x_q (once)                                     SPAD_A, K / D words
GEMV x_q · W₂  → b  int32 [F]                     ACC, kept (not stored)
GEMV x_q · W₁  → a  int32 [F], chunk by chunk:
   epilogue per chunk: s = a · s₁ · s_x;  h = s · σ(s) · (b · s₂ · s_x)   (VE, the same op sequence as today)
amax |h| over F                                    VE reduction into a broadcast word
h_q = to_i8(h · 127 / amax)                        VE, int8 into SPAD_A
ST h_q, ST amax                                    the only DDR writes (down_proj reads both)
```

Removed per layer: `31`'s ST of b and `32`'s LDs of it, `32`'s second LD of
x, `32`'s ST of h, `33`'s and `34`'s LDs of h and their own LD / ST of the
scalar, three list boundaries. The per-channel scales s₁, s₂ stay loads from
the parameters.

### 4.2 Frontend / preprocessing: `iree-sa-form-ffn-regions`

A pass in `extendPreprocessingPassPipeline` after
`iree-sa-pack-linear-weights` and `iree-sa-clone-cheap-producers` (the weights
are packed, the scales materialized):

- **Match** (structurally, not by name): two contractions (`linalg.generic`
  with one reduction dimension, i8 × i8 → i32) with the same LHS value and the
  same N; an elementwise `linalg.generic` consuming both results (the gating:
  SiLU for SmolLM2 / Qwen3 / Llama, GELU-tanh for Gemma, recognized only as
  "consumes both"); optionally the amax reduction over its result and the
  elementwise quantization consuming the result and the amax.
- **Wrap** them, with their fills and empties, in one `flow.dispatch.region`
  yielding what is used outside (h_q and the amax; h too if anything else
  reads it, which today nothing does).
- **Decode only** in the first version: M = 1 (vectors). Prefill's chain
  (M = 16) is left to IREE: its intermediates do not fit (§4.4).
- Flag `--iree-sa-fuse-ffn` (default off), `compile_sa.sh` `SA_FUSE_FFN=1`.
  With the flag off nothing changes (golden corpus identical).
- Lit test: a two-layer FFN in linalg, the region formed; a near miss (two
  contractions with different LHS) left alone.

### 4.3 Backend

What the sa pipeline already has: a dispatch with one GEMV "linear" kernel and
its elementwise tail (`32` today), reductions and broadcast words, int8
quantization chains (`sahl.to_i8`), several pieces in one template. What it
needs:

1. **Kernel matching** (`SahlKernels.cpp`): a dispatch may hold two linear
   kernels. The first (W₂) has no tail and its result is a local buffer, not a
   store; it becomes a `sahl.kernel "linear"` whose output is an ACC buffer
   kept for the rest of the dispatch (`sahl.reserve`). The second (W₁) has
   the gating as its tail, with one tail input that is that local buffer.
2. **Tail input from local memory** (`SahlExpandKernels.cpp`): today a tail
   input is loaded from DDR per chunk (kinds 2–4 of `epiIn`). A new kind reads
   a slice of a local buffer: the chunk's view of b (no LD; the VE reads ACC
   directly).
3. **x shared**: the second GEMV reads x from the same SPAD_A words (the GEMV
   unit keeps x resident when the word and K match); the second
   `sahl.load` of the same DDR view is dropped (a load of an already loaded,
   unmodified view: a small CSE in `sa-to-sahl`).
4. **After the kernels**: the amax reduction and the quantization run on local
   buffers through the generic lowering, as in other dispatches; h is a local
   buffer (ACC), the amax a broadcast word, h_q int8 in SPAD_A, then the two
   stores.
5. **Local memory** (3a's allocator as it is: in order at first use): b, h
   and the chunks' accumulators must fit ACC above the chunk scratch at once,
   and h_q in SPAD_A next to x. A dispatch that does not fit fails in the
   backend and is compiled unfused (the region is formed only where §4.4's
   check passes).

### 4.4 Local memory budget (decode, D = 16: an ACC word is 16 × 4 B, a SPAD_A word 16 B)

| | SmolLM2 | Gemma | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| K (hidden) / F (FFN) | 576 / 1536 | 640 / 2048 | 1024 / 3072 | 2048 / 8192 |
| ACC: b + h (int32, fp32: F / 16 words each) | 192 | 256 | 384 | 1024 |
| SPAD_A: x + h_q (K / 16 + F / 16 words) | 132 | 168 | 256 | 640 |
| Capacity (ACC 4096 words, SPAD_A 8192 words) | fits | fits | fits | fits |

The chunk scratch of the linear kernel comes on top (the C3 layout, at the
start of each ACC bank). Prefill (M = 16) multiplies the ACC need by 16:
Llama 16 384 words, beyond ACC; a prefill version needs h kept per N chunk
and the amax as a second pass, left for later.

### 4.5 Numerics

The fused dispatch computes the same operations in the same order on the same
values: the int32 accumulators are exact, b is read from ACC instead of from
its DDR copy (the same bits), the gating's VE sequence is unchanged, the amax
is order-free, the quantization identical. **The fused build must be
bit-exact with the unfused one**: logits identical, not within a tolerance.

### 4.6 Verification

1. Lit: the region pass; `sahl_expand_kernels`-style tests of the two-kernel
   dispatch (local tail input, shared x).
2. Golden corpus with the flag off: byte-identical.
3. `dispatch_check.py` on the fused dispatches against the oracle (it
   interprets any sequence of linalg ops), T irrelevant (no attention).
4. End to end on the sim, four models: decode logits bit-exact with the
   unfused build of the same model; `host_copies.py` 0.
5. Board: `run_board_tests.sh` with the fused bundles: `llm` (17/17
   bit-exact), `profile --batch` (device ms per step against v2.2), `rvsched`
   and `rvcode --embedded` (the fused templates through plan B and plan A).
6. The model's prediction checked: `fusion_whatif.py --only` gave −2.3% to
   −4.4%; the board's batched decode should land near it.

### 4.7 Expected gain, effort, risks

- **Gain**: decode −2.3% (Qwen3, Llama) to −4.4% (SmolLM2) per step; prefill
  unchanged in the first version.
- **Effort**: about two weeks: the region pass and its tests (2–3 days); the
  backend's two-kernel dispatch, local tail input and shared x (4–6 days);
  verification on the sim, four bundles, board runs (3–4 days).
- **Risks**: IREE passes between preprocessing and dispatch creation not
  expecting regions this early (mitigated by IREE's own precedent, and caught
  by the lit test on the whole pipeline); the backend's fixed ACC layout of
  the linear kernel (chunk scratch at fixed words) leaving too little room
  next to b and h for some shape (caught at compile time, falls back
  unfused); the generalization pass merging the fused executables per layer
  (it should: they differ only in offsets).

### 4.8 Board results (2026-10-17)

Bundles `build/deploy_hfgen_{smollm2f,gemmaf,qwen3f,llamaf}_pcpi`: the
`v2.2` generalized bundles (`smollm2_lc2048`, `gemmag`, `qwen3g`, `llamag`)
with `sa.vmfb` recompiled at `97a3323` with `SA_FUSE_FFN=1`, same parameters,
prompt and overlay. `run_board_tests.sh smollm2f gemmaf qwen3f llamaf`; logs
`build/bundles_pcpi/board_tests_s3/<b>f.*.log`, results
`build/deploy_hfgen_<b>f_pcpi/results/`.

**Correctness: all PASS.** `llm`: logits bit-exact with the sim in 17/17
steps; long context (SmolLM2 / Qwen3 / Llama 2032-token prompt, Gemma 496):
prefill + decode bit-exact with decode only; `rvsched` (plan B) and
`rvcode --embedded` / `rvsched --rv --embedded` (plan A): tokens identical,
logits bit-exact. Every dispatch on the accelerator.

**Speed** (decode after a 32-token prompt, `profile --batch`, one list per step):

| | SmolLM2 | Gemma | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| Dispatches per decode step | 878 → 788 | 618 → 564 | 932 → 848 | 472 → 424 |
| Device ms per step, unfused → fused | 36.25 → 35.28 | 56.25 → 55.45 | 132.11 → 130.72 | 222.95 → 220.19 |
| Device time | **−2.7%** | **−1.4%** | **−1.1%** | **−1.2%** |
| Decode ms per step (with the runtime) | 42.4 → 40.9 | 60.7 → 59.6 | 138.7 → 136.8 | 226.5 → 223.5 |
| Model's prediction (§2.3, `--only`) | −4.4% | −2.5% | −2.3% | −2.3% |
| Plan B (`rvsched`), ms per step | 37.8 → 36.8 | 61.2 → 60.4 | 136.7 → 135.2 | 225.7 → 222.9 |

The long-context sweep shows the same saving at every T (decode 1–3 ms per
step less from T = 64 to 2048). Prefill is unchanged (decode only, as
designed): 276.7 / 233.6 / 843.0 / 1041.7 ms per chunk.

**Why half of the prediction.** The board's per-dispatch profile (decode,
cycles per layer) shows where it goes:

| Cycles per layer | SmolLM2 | Gemma | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| gate + up GEMV dispatches, unfused | 38 676 | 56 646 | 119 501 | 581 433 |
| the fused dispatch | 37 518 | 55 167 | 118 425 | 573 729 |
| amax + quantization dispatches removed | 2 571 | 3 120 | 4 198 | 9 842 |
| Saved per layer | 3 729 | 4 599 | 5 274 | 17 546 |
| The model's saving per layer | ≈ 5 300 | ≈ 7 800 | ≈ 10 900 | ≈ 32 000 |

The fused dispatch is barely faster than the two GEMV dispatches it replaces,
although it no longer stores b, loads it back or loads x twice. The model
agrees with the board on the fused code (SmolLM2: a trace of the fused bundle
modelled with the unfused bundle's calibration gives −2.9% against −2.7%
measured), so the difference is the code, not the model:

- **The gating is not overlapped with the streaming.** The unfused gating
  dispatch (`expandLinear`) runs in chunks (SmolLM2: 4 × 24 tiles, ACC banks
  alternating): chunk i's SiLU · mul runs on the VE while chunk i + 1's
  weights stream into the GEMV unit. `expandFfn` sizes its chunks by ACC
  capacity only, so SmolLM2 and Gemma get one chunk (N = 1536 / 2048) and
  Qwen3 two: the whole gating, with its two SFU passes over N (about 3
  thousand cycles each on SmolLM2), runs after the last weight byte. That is
  what the removed round trip saved, spent again.
- **The amax and the quantization run after all chunks** (in pieces only
  for capacity), serially after the streaming as the separate dispatches
  did: only their dispatch boundaries are saved (Llama: the 9.8 thousand
  cycles of the two dispatches are mostly VE work over 8192 elements).

**Next (compiler only, `expandFfn`).** (1) At least four chunks whatever the
capacity allows (as `expandLinear` chooses), banks alternating; (2) chunk i's
amax piece right after its gating (`sahl.accumulate` into the reserved
scalar), leaving only the quantization (one VE pass over N) after the last
chunk; (3) chunk i + 1's GEMVs emitted before chunk i's gating, so that the VE
queue (4 deep, in-order dispatch) never holds back the next chunk's streaming.
Each step is checked with a sim trace and `perf_model.py` on the two-layer
builds before rebuilding the bundles; the numerics do not change (the same
operations on the same values; the amax is order-free).

## 5. After SwiGLU

1. **Batching the constant tail inputs: not worth it** (estimated
   2026-10-17, `fusion_whatif.py --merge-consts`). A GEMV dispatch loads its
   per-channel scale slices chunk by chunk (`32`: 16 LDs of 1.5 KB); merging
   each dispatch's chains of constant, ACC-bound loads into one removes
   367–1830 LDs per decode step and 1.8–11.7 thousand per prefill chunk, and
   changes the time by +0.1% to +0.7% (decode) and −0.4% to −0.7% (prefill),
   within the model's error: these loads are off the critical path.
2. **Attention with its preparation** (q / k norm, RoPE, the quantization
   before `o_proj`): −3.7% decode, −6.1% prefill on Qwen3; needs the region
   pass to take a dynamic-T chain and the T-block loop of §8.22 inside the
   fused template.
3. **Residual + RMSNorm into the projections**: −1.6% / −3.7%.
4. **Prefill**: the chains above with capacity-aware chunking (h per N chunk,
   the amax as a second pass over local memory), where the bound is largest
   (−10% to −20%).

## Files

- `compiler/tests/fusion_whatif.py`: the evaluation (`--only` for one chain).
- `compiler/tests/perf_model.py`: the command-level model (a FENCE now holds
  back later commands; it did not before 2026-10-17).
- `build/perfmodel/<bundle>.trace`, `<bundle>_cal.json`,
  `<bundle>_fusion.json`, `swiglu_whatif.txt`: the traces, calibrations and
  results of §2 (not in git).

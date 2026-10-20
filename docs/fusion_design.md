# Cross-operator fusion: evaluation, the SwiGLU and attention designs

Status: evaluation done (2026-10-17, `compiler/tests/fusion_whatif.py`,
[`iree_compiler_plan.md`](iree_compiler_plan.md) §8.23); the SwiGLU fusion is
implemented (`97a3323`, `--iree-sa-fuse-ffn`; on by default in
`compile_sa.sh` since 2026-10-18, `SA_FUSE_FFN=0` turns it off) and accepted on the board on all four models (2026-10-17, §4.8): bit-exact,
decode −1.1% to −2.7% of device time, about half of the model's −2.3% to
−4.4%, for a reason found in §4.8 (the fused gating is not chunked enough to
overlap the weight streaming). The schedule is fixed in `ea7ca80` (§4.9)
and accepted on the board (2026-10-18): bit-exact, decode −3.3% to −6.1% of
device time, within 0.4 points of the model's −3.4% to −6.3%. The attention
fusion (§6) is implemented for decode (2026-10-18/19: a region pass after IREE's
region formation, IREE patch 0004, and a kernel `attn_layer`). It was accepted
on the board on all four models (§6.4.8): bit-exact, decode device time
−1.1% to −4.8% against the SwiGLU-fused builds. It has been on by default in
`compile_sa.sh` since 2026-10-19 (`SA_FUSE_ATTN=0` turns it off). Together
against v2.2: decode device time −4.4% to −9.3%. Measured against tag
`v2.2-kv260`. Prefill: a chunk kernel (§6.7, 2026-10-19/20) runs a chunk's 16
attention rows in one dispatch per layer, bit-exact on the sim on Gemma, Qwen3
and Llama up to 2048 positions; in the model a layer's prefill attention costs
−51% to −64%. Accepted on the board on all four models (2026-10-20, §6.7.1):
bit-exact, prefill −9% to −25% at a 40-token prompt and −20% to −45% at a
2032-token one (Gemma 496), decode unchanged; on by default since 2026-10-20
(`SA_FUSE_ATTN_CHUNK=0` turns it off). The fusion work is closed there (§7):
the chains left are worth a few percent each and are deferred.

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
- Flag `--iree-sa-fuse-ffn` (off in `iree-compile`; `compile_sa.sh` passes it
  by default since 2026-10-18, `SA_FUSE_FFN=0` to leave it out). With the
  flag off nothing changes (golden corpus identical).
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

**Next (compiler only, `expandFfn`; done in §4.9).** (1) At least four chunks whatever the
capacity allows (as `expandLinear` chooses), banks alternating; (2) chunk i's
amax piece right after its gating (`sahl.accumulate` into the reserved
scalar), leaving only the quantization (one VE pass over N) after the last
chunk; (3) chunk i + 1's GEMVs emitted before chunk i's gating, so that the VE
queue (4 deep, in-order dispatch) never holds back the next chunk's streaming.
Each step is checked with a sim trace and `perf_model.py` on the two-layer
builds before rebuilding the bundles; the numerics do not change (the same
operations on the same values; the amax is order-free).

### 4.9 The overlapped schedule (2026-10-17, `ea7ca80`)

`expandFfn` now writes:

```
LD x; LD each scalar input into both banks
GEMV chunk 0 (W₂, then W₁); LD its scale slices
for chunk i (bank i mod 2):
    GEMV chunk i+1, first projection            (streams while the VE works on chunk i)
    scope: gating(i) → h slice i; amax piece(i)  (VE; everything in bank i mod 2)
    GEMV chunk i+1, second projection; LD its scale slices
int8 output, piece by piece (it needs the whole amax); ST h_q, ST amax
```

- **Chunks**: about six whatever the capacity allows (`kFfnChunks`; three to
  eight model within 2% of each other on SmolLM2, six is the best on all four
  models). SmolLM2 6 × 16 tiles, Gemma 8 × 16, Qwen3 6 × 32, Llama 16 × 32
  (capacity-bound).
- **Issue order**: commands issue in order with engine queues of four, so
  chunk i+1's second projection is issued after chunk i's gating. The
  gating's VE commands then get dispatched while the first projection
  streams, and the GEMV unit never waits on the dispatcher.
- **Banks**: the scoreboard tracks hazards per bank half, so a VE command
  reading or writing a bank the in-flight GEMV writes would wait for the whole
  GEMV. Everything one chunk's VE work touches is therefore in its own bank:
  its accumulators and scale slices (as before), its slice of h (even chunks'
  in bank 0, odd chunks' in bank 1, in the chunk scratch), its own copy of the
  scalar inputs (broadcast inside the chunk's scope), and its temporaries.
  h is no longer one buffer; the amax and int8-output pieces read the slices.
- **The amax** is reduced piece by piece right after each chunk's gating
  (`sahl.accumulate`), overlapped like the gating. Only the int8 output (one
  VE pass per piece) follows the last chunk.

Same operations on the same values: bit-exact with the unfused build on the
two-layer models of all four (sim logits, 40-token prompt + 8 decode steps),
every dispatch OK against the oracle, golden corpus otherwise unchanged.

**Decode cycles saved per layer** (performance model on sim traces of the
two-layer builds, calibrated on each full bundle's board profile; the old
schedule's model numbers match its board run within the model's error):

| | SmolLM2 | Gemma | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| Chunks × tiles | 6 × 16 | 8 × 16 | 6 × 32 | 16 × 32 |
| Old schedule: model / board | 3.6k / 3.7k | 4.7k / 4.6k | 5.6k / 5.3k | 20.3k / 17.5k |
| Overlapped schedule: model | **7.6k** | **11.8k** | **16.0k** | **49.5k** |
| Projected decode device time (× layers, against v2.2) | **−6.3%** | **−3.8%** | **−3.4%** | **−3.6%** |

**Board (2026-10-18, bundles `deploy_hfgen_*f2_pcpi`, logs
`build/bundles_pcpi/board_tests_s3/*f2.*`).** All tests PASS: `llm`
(logits bit-exact with the sim, 17/17 steps), long context and its sweep,
plan B and plan A (`rvsched`, `rvcode`, tokens identical, logits bit-exact).
Decode device time per step (`profile --batch`, 32-token prompt):

| | SmolLM2 | Gemma | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| v2.2, unfused (ms) | 36.25 | 56.25 | 132.11 | 222.95 |
| Old schedule, `97a3323` (ms) | 35.28 | 55.45 | 130.72 | 220.19 |
| Overlapped schedule, `ea7ca80` (ms) | **34.04** | **54.15** | **127.77** | **215.42** |
| Against v2.2 | **−6.1%** | **−3.7%** | **−3.3%** | **−3.4%** |
| Projected (model) | −6.3% | −3.8% | −3.4% | −3.6% |
| Cycles saved per layer, board / model | 7.4k / 7.6k | 11.7k / 11.8k | 15.5k / 16.0k | 47.1k / 49.5k |

Decode end to end (host loop included): 39.7 / 58.3 / 133.9 / 218.7 ms per
step (25.2 / 17.2 / 7.47 / 4.57 tokens/s). Prefill and device memory are
unchanged (the fusion is decode-only). Plan B (the decode loop on the RISC-V
core) gains the same: 35.5 / 59.1 / 132.2 / 218.1 ms per step against the old
schedule's 36.8 / 60.4 / 135.2 / 222.9.

This beats the SwiGLU bound of §2.3 (−2.3% to −4.4%). That bound kept the
unfused code's commands and order, including its own serialization: the
unfused gating dispatch also leaves part of its gating after its last weight
chunk. The overlapped schedule removes that too, so a whole-layer bound
computed the same way also understates what fused, re-scheduled code can reach.

## 5. After SwiGLU

1. **Batching the constant tail inputs: not worth it** (estimated
   2026-10-17, `fusion_whatif.py --merge-consts`). A GEMV dispatch loads its
   per-channel scale slices chunk by chunk (`32`: 16 LDs of 1.5 KB); merging
   each dispatch's chains of constant, ACC-bound loads into one removes
   367–1830 LDs per decode step and 1.8–11.7 thousand per prefill chunk, and
   changes the time by +0.1% to +0.7% (decode) and −0.4% to −0.7% (prefill),
   within the model's error: these loads are off the critical path.
2. **Attention with its preparation** (q / k norm, RoPE, the quantization
   before `o_proj`; table below): the largest chain left, −1.4% to −6.4%
   of decode and −4.2% to −11% of prefill; needs the region pass to take a
   dynamic-T chain and the T-block loop of §8.22 inside the fused template.
   Designed in §6 (2026-10-18): stage 1 (the q path + the core + the output quantization)
   is most of it, −3.3% / −5.6% on Qwen3; more at long T.
3. **Residual + RMSNorm into the projections** (table below): −0.9% to
   −4.0% of decode, −2.5% to −4.8% of prefill; structurally the closest to
   SwiGLU (a reduction and an elementwise pass between GEMVs).

**The remaining chains on all four models** (2026-10-18, `fusion_whatif.py
--chain`, on the same v2.2 traces and calibrations as §2; upper bounds, "on
chip, no barriers"). Kinds alone cannot separate these chains (SmolLM2's and
Llama's attention output has the hidden size, so its quantization has the
same kind as the RMSNorm's), so `--chain` selects by role: the dispatches
between two projections form a run, a run with a dynamic-T kind is the
attention (in decode also v's preparation and the quantization before
`o_proj`; in prefill the whole row attention), and the runs after it are,
in order, residual + RMSNorm, SwiGLU, residual + RMSNorm, then q / k
preparation up to the next attention. `--with-proj` adds the intermediates
between the chain and its projections, and between two projections when the
run after the consumer belongs to the chain (gate → up for SwiGLU, the
residual stream across a block for the norm).

| Chain fused alone (prefill / decode) | SmolLM2 | Gemma | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| Attention + q / k preparation + quantization before `o_proj` | −9.0 / −5.6% | −7.2 / −2.0% | −7.6 / −3.5% | −3.5 / −1.2% |
| … with the q / k / v and `o_proj` projections | **−11.0 / −6.4%** | **−8.7 / −2.4%** | **−9.3 / −3.9%** | **−4.2 / −1.4%** |
| Residual + RMSNorm + quantization of x | −1.2 / −2.2% | −2.7 / −1.5% | −0.7 / −0.7% | −0.9 / −0.4% |
| … with the projections before and after it | **−4.4 / −4.0%** | **−4.8 / −2.1%** | **−2.7 / −1.5%** | **−2.5 / −0.9%** |
| SwiGLU with its projections (check against §2.3) | −4.1 / −4.7% | −4.7 / −2.7% | −2.7 / −2.5% | −3.7 / −2.4% |

The SwiGLU row is within 0.5 points of §2.3's (selected there by kind);
the role selection also keeps h_q's load into the down projection. The Qwen3
rows differ from §2.3's (−6.1 / −3.7% and −3.7 / −1.6%) because the earlier
regexes, by kind, also caught pairs of other chains. As SwiGLU showed
(§4.9), re-scheduled fused code can beat these bounds, which keep the
unfused code's order. The decode figures are at a 32-token prompt: the
attention's share grows with the context.

4. **Prefill**: the chains above with capacity-aware chunking (h per N chunk,
   the amax as a second pass over local memory), where the bound is largest
   (−10% to −20%).

## 6. The attention fusion (design)

Status: design (2026-10-18), schedule prototyped in the performance model
(§6.2.2). The region pass (§6.4.1) and the backend's kernel in its in-order
form (§6.4.5) are implemented, steps 3 and 4a of §6.6: the fused dispatch runs
on the accelerator, bit-exact with the unfused build; decode only (§6.4.7), on
the board since 2026-10-19 (§6.4.8: decode −1.1% to −4.8% against f2), on by
default in `compile_sa.sh`.
The second fused chain after
SwiGLU (§4, on by default since 2026-10-18). Bounds from
`compiler/tests/fusion_whatif.py` on the `v2.2-kv260` traces (§6.2); the
attention kernels it builds on are those of C6.5
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
numerics need no change: the only fp32 reduction over T is a max, so the
fused build is bit-exact with the unfused one. A schedule prototype on the
real commands (§6.2.2) gives Qwen3 −4.0% to −4.3% of decode from T = 48 to
2048 with a schedule the compiler can emit, a little above the bound. It also
finds the larger lever at long contexts. The VE is the busiest engine, and
the q · Kᵀ loop's transposes wait for its EXs on a shared SPAD_B bank.
Alternating that bank (implemented, §6.2.2) is worth −1.8% of decode at
T = 2048 by default. With 256 rows per loop iteration it is worth −7.4%,
which costs short contexts +3%, and −11.6% together with the fusion (§6.5,
§6.6).

### 6.1 The chain today (Qwen3-0.6B decode, one layer)

Numbering of the 2-layer build (`build/fuse/qwen3_l2`); the other models
have the same chain without QK-norm (SmolLM2, Llama) or with Gemma's own
norms and scaling.

| # | Dispatch kind | Work | Stage (§6.3) |
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
first LDs, which its work waits for (§2.2).

**Prefill** (`hf_generic._attention_rows`) computes the q / k / v
projections for the 16 rows of a chunk together, then runs each row exactly
as a decode step (dispatches 8–25 at the row's position, the same executables:
IREE shares them across the module), then `o_proj` for the 16 rows. A
fused row kernel therefore serves decode and prefill alike, and prefill
pays every boundary it removes 16 times per chunk.

### 6.2 What it is worth

#### 6.2.1 Upper bounds (performance model)

Method as in §2.1: the intermediates'
STs and the LDs that read them back are removed from the joined step, with
no barriers in between. The KV cache is persistent, so the cache writes →
scores round trip is not counted, whatever the region. Chains selected by
role (`--chain`, §5), prefill chunk / decode step:

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

#### 6.2.2 The schedule prototype (performance model, 2026-10-18)

`compiler/tests/attn_schedule.py` builds stage 1's fused command list from
the real commands. Each unfused dispatch of the chain (Qwen3 2-layer build at
`--max-len 2048`, `build/hfgen/qwen3_l2_t2048lc`) is compiled by the current
compiler and run on the functional simulator at T through `dispatch_check.py`.
So every dispatch is also checked against the oracle, with the executed
commands captured. The list is then edited and timed by `perf_model.run_list`
(`qwen3g_cal.json`):

- the LDs and STs of intermediates are dropped: those on transient (`Indirect`)
  bindings, except the chain's input (q_proj's output) and its outputs;
- the GEMV reads p where the probabilities wrote it;
- the commands are segmented by unit: the probabilities' row pieces, which
  for Qwen3 are 4 query rows = 2 KV heads, so 4 units;
- unit u's ACC work goes in bank u mod 2.

Then several orders are timed. Per layer, cycles of the attention chain
(dispatches 8, 9, 13, 17–25):

| T | Unfused | In order (the bound's form) | **Fused, implementable** | + alternating tiles | Alternating tiles alone (unfused) |
|---|---:|---:|---:|---:|---:|
| 48 | 84.2k | 69.3k | **65.7k** | 61.8k | 82.5k |
| 512 | 191.7k | 172.1k | **165.5k** | 145.1k | 174.0k |
| 2048 | 592.4k | 555.8k | **548.2k** | 464.3k | 510.8k |
| Busiest engine (VE), unfused | 43.5k / 100.2k / 310.0k | | | | |

Per token (× 28 layers), and against the board's decode device time
(127.8 ms at T ≈ 40 with SwiGLU fused, `*f2`; 168.8 / 310.1 ms at
T = 512 / 2048 from the `qwen3_lc2048_pcpi` sweep):

| T | In order | Fused, implementable | + alternating tiles | Alternating tiles alone |
|---|---:|---:|---:|---:|
| 48 | −4.2 ms (−3.3%) | **−5.2 ms (−4.1%)** | −6.3 ms (−4.9%) | −0.5 ms (−0.4%) |
| 512 | −5.5 ms (−3.3%) | **−7.3 ms (−4.3%)** | −13.0 ms (−7.7%) | −5.0 ms (−3.0%) |
| 2048 | −10.2 ms (−3.3%) | **−12.4 ms (−4.0%)** | −35.9 ms (−11.6%) | −22.8 ms (−7.4%) |

- **The prototype reproduces the bound.** "In order" (the intermediates
  dropped, nothing moved) gives −3.3% of decode at T ≈ 40, exactly
  `fusion_whatif.py`'s stage 1 (§6.2.1).
- **The implementable schedule** puts unit u − 1's probabilities, Σp and
  p · V between the two KV heads' T loops of unit u. The commands stay
  whole, with no loop split by T, so the structure is valid for any T (a
  `LOOP_END` count must not be 0). This gains another 1–2 ms per token
  over "in order".
- **Forms that are not implementable or not better.**
  - Spreading the previous unit's commands one after each EX ("interleaved")
    reaches −6.3 / −8.6 / −19.0 ms. But that needs the T loop unrolled,
    which a dynamic T forbids.
  - Cutting them into one piece per K block, a fixed share in the loop body
    ("blocked"), pays each piece's fixed cost on the VE. That is worse than
    "in order" up to T = 512, and better only at 2048 (−23.5 ms).
- **p's bank does not matter.** Placing p in SPAD_A bank 0 (with the A tiles)
  or bank 1 (with the raw K blocks) changes the cycles by at most 1% in every
  form. The false dependencies feared in §6.5 are small.
- **The VE is the busiest engine at every T**: 52% of the unfused chain at
  T = 2048, against LD 39% and EX 29%.
  - What keeps the fused list far above the VE floor is the q · Kᵀ loop
    itself. A block's transpose into SPAD_B bank 0 waits for the previous
    block's EX reading that bank, because EX and VE conflict on any common
    bank. And the VE is held behind that block's copies, which wait for the
    same EX.
  - With the tiles alternating between SPAD_B banks 0 and 1, and the next
    block's transpose issued ahead of the current block's copies, the VE
    transposes while the EX runs. This is the double buffering that §8.22
    step c left out.
  - Alone it is worth −7.4% of decode at T = 2048. Together with the fusion
    the two compound: −11.6% at 2048, −7.7% at 512.

The other three models are not modelled yet (`--dispatches` takes their
numbering; Gemma's Hk = 1 has a single unit).

**The alternating tiles, implemented (2026-10-18, §6.6 step 2).** Correction
to the table above: its "alternating tiles" columns tile blocks of 128 rows
in alternating banks at even block counts. That is 256 rows per loop
iteration. At T ≤ 128 such a loop computes a second block, which the
prototype did not count.

The compiler (`SahlExpandKernels.cpp`, the q · Kᵀ T loop):
- Each iteration covers its block of Tb rows (`--iree-sa-attn-block`,
  default 128) as two halves whose tiles alternate SPAD_B banks.
- Per half: its LD, the previous half's EX, its transpose, then the previous
  half's copies.
- Only where a half's raw rows are at least 8 KB (Tb / 2 · head_dim). At the
  default that is Qwen3 (128) and Gemma (256). SmolLM2 and Llama (64) get
  halves only with `--iree-sa-attn-block 256`; otherwise their code is
  byte-identical.
- The block count, the rows read (never past ⌈T / Tb⌉ · Tb) and the guard
  block are unchanged.

The q · Kᵀ dispatch per layer, compiled, timed by the cycle model, and
checked against the oracle at every T from 16 to the compiled maximum around
the block boundaries:

| T | Qwen3 default | Qwen3, 256 | Gemma default | Gemma, 256 | Llama, 256 | SmolLM2, 256 |
|---|---:|---:|---:|---:|---:|---:|
| 48 | −9.0% | +29.5% | −12.1% | +30.2% | +25.7% | +23.7% |
| 256 | −7.6% | −10.0% | −12.1% | −8.7% | −10.3% | −11.3% |
| 512 | −6.5% | −15.6% | −12.3% | −14.4% | −16.2% | −17.9% |
| 2048 | −5.3% | −21.7% | | | −22.9% | −25.6% |

Per token on Qwen3:
- The default saves 1.2 ms at T = 48 (−0.9% of decode) and 5.6 ms at 2048
  (−1.8%).
- 256 rows per iteration costs 3.9 ms at T ≤ 128 (+3.1%) and saves 22.9 ms
  at 2048 (−7.4%), the figure above; 294.0k cycles, exactly the prototype's.

So the default helps every T a little, and 256 is the choice for a bundle
meant for long contexts.

Checks:
- lit: `sahl_expand_tiled_t.mlir` updated.
- Golden: exactly the three tiled q · Kᵀ dispatches changed, re-recorded.
- The 2-layer Qwen3 at 2048, end to end on the sim: logits bit-exact with the
  single-bank form (`--iree-sa-attn-block 16`).

### 6.3 Scope and stages

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

### 6.4 Design (stage 1)

#### 6.4.1 Dispatch creation: `iree-sa-form-attn-regions` (implemented 2026-10-18)

The pass (`compiler/plugins/sa/target/FormAttnRegions.cpp`, flag
`--iree-sa-fuse-attn`, default off) does not wrap linalg ops before dispatch
creation, as `iree-sa-form-ffn-regions` does. It **merges the
`flow.dispatch.region` ops that IREE has formed**, right after region
formation and before they become `flow.dispatch.workgroups`. The first plan
was the FFN's: wrap the ops in the preprocessing phase. The IR at that point
showed why not. The attention is still raw: an i32 × i32 → i64
`batch_matmul`, `linalg.transpose`, the RoPE halves as `tensor.concat`, some
40 unfused generics. GlobalOptimization and dispatch creation's fusion turn
these into the forms the backend knows today. A region formed before them
keeps them out of both, because GlobalOptimization skips ops already in a
region. Merged after formation, every piece keeps exactly the body of
today's dispatch. `expandAttnLayer` (step 4) can then reuse the existing
piece matchers, and T stays a dynamic dimension of each piece as IREE
formed it. `ConvertDispatchRegionsToWorkgroups` derives the merged
dispatch's workload (three ordinals for T: T, min(T, rows of the K cache),
min(T, rows of the V cache)).

IREE has no plugin hook in dispatch creation, so
`compiler/patches/iree/0004-pluginapi-dispatch-region-formation-hook.patch`
adds one: `PipelineExtensions::extendDispatchRegionFormationPassPipeline`,
run after `addDispatchRegionCreationPasses`. DispatchCreation takes it as an
optional callback, so it does not depend on the plugin API. The default is a
no-op.

- **Match**, structurally, over the regions of a block:
  - *Anchor*: a region with a contraction whose result has a dynamic dim
    (the scores). A contraction is a linalg op with a reduction loop and two
    operands of rank ≥ 1.
  - *Backward*: producer regions without a contraction, all of whose results
    are read only by the set (the q path). q_proj, the K / V paths and values
    shared with them, such as RoPE tables, stay outside.
  - *Forward*: consumer regions without a static contraction whose other
    operands do not depend on the set (probabilities, Σp, p · V, amax,
    int8). o_proj stops it.
  - *Accept* when a second contraction runs over a dynamic dim (p · V) and
    every value used outside is static.
- **Merge**:
  - The bodies are cloned in block order into one region at the set's last
    op, together with the reshapes between the regions.
  - The cache writes stay outside, before it: no DDR RAW inside a dispatch.
  - The dynamic sizes inside get `util.assume.int` with umax
    `--iree-sa-max-dynamic`. A piece's dynamic intermediate is now a tensor
    inside the dispatch, not a result stored to DDR, and IREE's range
    analysis leaves T unbounded. Without the bound the host fallback's
    (VMVX) allocas are 2⁵³ elements and the run fails with
    `RESOURCE_EXHAUSTED`.
  - A collapse that only drops unit dims of a dynamic tensor (Gemma's
    scores `1x4x?` → `4x?`) becomes a rank-reducing `extract_slice`.
    Inside the dispatch that tensor is a subview of a bounded allocation,
    and VMVX fails to legalize its `collapse_shape` ("unresolved
    materialization ... `memref<4x?xf32, strided<[256, 1]>>`").
- **Lit**: `sa_form_attn_regions.mlir`. A decode attention at region level
  becomes one region, with q_proj, o_proj and a value shared with the
  outside left out. A near miss without p · V is left alone.
- **Result** (2-layer builds, D = 16, 4 ports): Qwen3 merges 12 regions per
  decode layer (§6.1: dispatches 8, 9, 13, 17–25) and 9 per prefill row: 66
  chains, 2 in decode and 2 layers × 16 rows in each of the two prefill
  functions. A prefill row's fused dispatch reads q already dequantized and
  QK-normed for the whole chunk (`1x16x16x128xf32`). Those pieces are shared
  by the 16 rows, so they stay outside. The
  backend is unchanged, so each fused dispatch runs on the host (VMVX).
  Gemma merges 8 per decode layer and 6 per prefill row, Llama 9 and 6, 66
  chains each.

  End to end on the sim (40-token prompt in chunks of 16, 8 decode steps),
  every model's tokens equal the unfused build's:

  | Model | Logits, fused vs unfused |
  |---|---|
  | Gemma | bit-exact |
  | Qwen3 | within 1.9·10⁻⁶ (one row) |
  | Llama | within 0.62 (correlation 0.998) |

  The fused dispatch computes exp and 1 / x in IEEE fp32 on VMVX, not with
  the device's SFU (`qhf.device_sfu`). An int8 probability that flips
  grows through the layers. For Llama this accounts for the whole
  difference: the unfused build with only its probabilities dispatch on the
  host (`--iree-sa-host-dispatches=reduction_32xD_i64xf32xf32`) gives
  logits bit-identical to the fused build's. With the flag off the golden
  corpus is unchanged (patch 0004 adds nothing then).

#### 6.4.2 Backend: one `sahl.kernel "attn_layer"`

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

The schedule (§6.2.2). A unit is one row piece of the probabilities: Qwen3
has 4 query rows = 2 KV heads, and units alternate ACC banks.

```
q path for all heads: dequantize, QK-norm, RoPE, amax per head, int8      (VE; small)
A tiles of all KV heads, in SPAD_A                                         (VE)
for unit u:
    scores, first KV head of u:   T loop over K blocks                     (LD, VE, EX)
    post(u − 1): masked max, exp → p, Σp; p · V (GEMV), ÷ Σp, × s_v, amax piece
    scores, second KV head of u:  T loop over K blocks
post(last unit)
output: int8 by the amax; ST row, ST scale
```

post(u − 1)'s commands are whole and sit between two T loops. No loop is
split by T, so the list is valid for any T. The T loop itself, unfused or
fused, should alternate its tiles between SPAD_B banks 0 and 1. That means a
body unrolled by two, with the next block's transpose issued ahead of the
current block's copies. In the prototype this matters as much as the fusion
at long T (§6.2.2).

#### 6.4.3 Local memory (D = 16; words: SPAD 16 B, ACC 64 B = 16 fp32)

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
Capacity is not the constraint; banks are (§6.5).

#### 6.4.4 Numerics

The same operations on the same values in the same order per element. The
only fp32 reduction over T is the max (order-free). Σp sums integers below
2²⁴ (exact in any order). Scores and p · V accumulate int32. The amax is
order-free and the quantization unchanged. So the fused build must be **bit-exact**
with the unfused one, and prefill rows bit-exact with decode (they run the
same fused executable). As in C6.5, no online softmax: block-local maxima
would change p's int8 rounding.

#### 6.4.5 Implemented: the in-order kernel (step 4a, 2026-10-18)

What the backend does with the fused dispatch today. It differs from §6.4.2
in where the cut lies and in the schedule, which is in order. The interleaved
schedule is step 4b.

- **What the generic lowering keeps**: the q path before the core, and the
  amax and int8 output after it. They are generics on local buffers, so
  sa-to-sahl normalizes what bufferization leaves for them:
  - `expandLocalViews`: a static local buffer seen through an `expand_shape`
    is allocated in the expanded shape, its other uses through a collapse of
    it. sahl-to-sahw also takes a static `expand_shape` of a contiguous local
    buffer as the same words.
  - `freshReductionInits`: bufferization merges equal fills, so reductions
    share one filled buffer. An identity copy of a filled buffer becomes a
    fill, and a reduction whose filled init was used in between gets a fresh
    filled buffer.
- **The core, an `AttnLayerPlan`** (`SahlKernels.cpp`, kernel kind
  `attn_layer`), anchored at the scores' batch matmul with a local q. It runs
  from the scores through p · V's epilogue: the extension of the K cache
  slice, the scores' epilogue (s_q local), the masked max and p (the
  position), Σp, p's and the V cache's extensions (V's local copy: its DDR
  source), p · V and its epilogue. Each intermediate is the core's own. Its
  readers are checked over the value's lifetime (to the buffer's next write),
  because bufferization reuses buffers.
- **`expandAttnLayer`**, all in one keep-scope:
  - Per KV head, the scores of its G rows:
    - *untiled*, when max_dynamic × head_dim, plus q's int8, fits one SPAD_B
      bank (Qwen3, Llama at 256): one LD of the head's T rows, one transpose,
      then per row its strip and its EX straight into the row's accumulator;
    - *tiled* otherwise (Gemma's head_dim 256; long contexts): the T loop of
      §8.22, copied from `expandContraction`.
  - Then per query row, each a 1-D piece of the matched generic over the
    row's T for the generic lowering: the scores' epilogue, the masked max, p
    (int8, SPAD_A), Σp, p · V on the GEMV unit (p as its x), and the epilogue
    into the row's slice of the output.
  - T is the K slice's length for every piece. The IR's shapes require p's
    and V's lengths to equal it, and one length parameter lets
    `sahw-fuse-ve` merge across the pieces.
- **SPAD_A**: q's int8 may already be in SPAD_A when the kernel starts. The
  kernel's claim is therefore relative (`sahl.claim_spad` "sa.relative"): its
  placed words follow the buffers already there.
- **A defect found and fixed**: the lowering writes `select(mask, x, -inf)` as
  a copy carrying VALID, whose masked lanes are 0, and relies on
  `sahw-fuse-ve` merging that copy into the max REDUCE. The kernel's rows
  reuse buffers, so the merge was refused, and rows with one to three keys
  unmasked gave NaN.
  - Now, for such a mask only, the merge counts readers only of the words no
    later VE of the same length has overwritten. An EX reads ACC only when it
    accumulates.
  - A mask left unmerged fails the compile (`sa.neginf`) instead of computing
    the wrong value.
  - `dispatch_check.py --pos 0` tests rows with one key unmasked; the default
    position, T − 3, had missed it.
- **Verified** (2-layer Gemma / Qwen3 / Llama, D = 16, 4 ports, max_dynamic
  256): no dispatch on the host; logits end to end on the sim bit-identical to
  the unfused build's.
  - `dispatch_check` passes every fused dispatch (decode and the 64 prefill
    rows) at T = 16, 128, 144, 256, with `--dirty`, each at the default
    position and at `--pos 0`.
  - Golden: the corpus without the flag unchanged; three new cases,
    `attn_gemma`, `attn_qwen3`, `attn_llama` (8932 dispatches).
- **Model** (decode, last step, cycles saved against the default build):

  | Model | Saved per layer | Decode, whole model |
  |---|---:|---:|
  | Qwen3 | 15.9K | ≈ −3.5% |
  | Llama | 14.8K | ≈ −1.1% |
  | Gemma (tiled form) | 4.3K | ≈ −1.4% |

  Qwen3 lands at the in-order bound of §6.2.2 (−3.3%). At first the kernel
  always ran the T loop, with integer copies, and saved nothing on Qwen3. The
  untiled form removed that.
  - VE busy time is still above the unfused path's: per-row pieces pay the
    VE's fixed cost once per row.

#### 6.4.6 Step 4b: the interleaved schedule (2026-10-18)

The untiled form now follows §6.4.2's schedule and §6.5's banks:

- **A tiles before the loop.** Each tile holds D query rows of all heads in
  order: staging rows in SPAD_A bank 1, two transposes, the A form in bank
  0. Qwen3's 16 rows are one tile, Llama's 32 two. One EX a head reads its
  rows' tile. Its output rows from the head's first row on are the head's
  scores; the other rows are garbage nobody reads. Before, each row was a
  strip and an EX of its own; a tile per head cost two transposes each,
  almost 470 VE cycles of fixed cost apiece.
- **Order per head b**: the LD of its K rows (SPAD_A bank 1), the transpose
  into SPAD_B bank b mod 2, its EX, then head b − 1's per-row post (VE, GEMV).
  The VE works on head b − 1 while the EX runs head b and the LD fetches
  head b + 1.
- **ACC**: the EX output [D, max_dynamic] by the head's parity in ACC bank 0
  or 1 (`sahl.reserve` with `sa.prefer_bank`), the post's buffers and temps
  in the same bank as its rows. EX and VE conflict on any shared ACC bank.
- **SPAD_A**: p's row in bank 1 after the K rows, or in bank 0 after the A
  tiles when bank 1 is full. Gemma (head_dim 256) is now untiled too. The
  relative claim offsets only bank-0 placements: q's int8 sits at the start
  of bank 0.
- **Two defects the interleaving exposed, fixed:**
  - *The EX writes all D rows of its output*, row i at `c + i · crow`, here
    crow = max_dynamic / D words. The accumulator had room for one row, so
    the 15 garbage rows overwrote the next buffers. That was harmless while
    each row's post came right after its EX; with the interleaving they
    clobbered head b − 1's post buffers. The accumulator is now [D, T].
    `expandContraction`'s untiled form allocates the same way, harmlessly
    so far, since nothing live follows its accumulator.
  - *The q path's generics never released their temporaries.* The RoPE
    gather's per-element broadcasts alone take about 2048 ACC words, so the
    kernel's buffers no longer fit. The generics before the kernel are now
    marked `sa.release_temps`: results allocated first, temps released after
    each. Only fused dispatches are marked, so golden is unchanged without
    the flag.
- **Verified** as §6.4.5 (2-layer Gemma / Qwen3 / Llama): no host dispatch,
  logits end to end bit-identical to the unfused build's, `dispatch_check` OK
  at T = 16 / 128 / 144 / 256 with `--dirty`, at the default position and at
  `--pos 0`. Golden: the unfused cases identical, `attn_*` re-recorded.
- **Model** (decode, last step; Qwen3's fused dispatch per layer, unfused /
  4a / 4b: 62.8K / 46.4K / 39.2K cycles; VE busy 29.0K, the busiest engine):

  | Model | Saved per layer, 4a → 4b | Decode, whole model, 4a → 4b |
  |---|---:|---:|
  | Qwen3 | 15.9K → 23.0K | −3.5% → **−5.0%** |
  | Gemma | 4.3K → 7.8K | −1.4% → **−2.6%** |
  | Llama | 14.8K → 27.7K | −1.1% → **−2.1%** |

  Qwen3 passes the prototype's −4.0% to −4.3% for this schedule (§6.2.2).
  Left: per-row pieces still pay the VE's fixed cost per row, the tiled form
  (long contexts) is still in order, and none of it is on the board yet.

#### 6.4.7 Decode only, for now (2026-10-18)

- **Prefill rows are no longer fused** (superseded by the chunk kernel, §6.7). The model (2-layer builds, a chunk of
  16 rows) gave prefill slower with every row fused: Qwen3 4423K → 5158K
  cycles (+17%), Llama +4%. Each fused row is a decode step's kernel: it loads
  and transposes the cache of every KV head once per row, 16 times a chunk.
  `iree-sa-form-attn-regions` therefore leaves a prefill chunk's rows alone
  (option `rows=true` merges them as before). It recognizes them by their
  q: rows of one producer read by nothing but attention chains (the chunk's
  q, sliced per row), where a decode layer's chain has its own q. Values
  shared with the k path, such as the RoPE tables, are read outside the
  chains and do not count. The K cache cannot tell them apart: all layers
  share one global, and decode's two layers read it the same way.
- **A kernel for a whole chunk** is the real prefill fusion: the chunk's 16
  query rows of a head as one A tile, the head's cache loaded and transposed
  once, one EX for all 16 rows, each row with its own causal position.
  Implemented 2026-10-19/20 (§6.7).
- **Long context (max_dynamic 2048) checked**, the tiled form: on the 2-layer
  Qwen3 compiled for 2048 positions, logits bit-identical to the unfused
  build, `dispatch_check` OK at T = 16 to 2048 and at `--pos 0`. It also
  showed that `sa.release_temps` (§6.4.6) had never applied to generics
  whose result is a view of a local buffer: the check looked the result up
  only among cached buffers. With the tiled form's larger buffers, every
  fused dispatch ran out of ACC. The result is now recomputed through its
  buffer.

#### 6.4.8 Board results (2026-10-19, bundles `*a`, `673a82e`)

The four models compiled as for the board, with the defaults plus
`SA_FUSE_ATTN=1`: 2048 positions (Gemma 512), decode attention fused (30 / 18
/ 28 / 16 chains, one per layer), prefill rows unfused. Bundles
`deploy_hfgen_{smollm2,gemma,qwen3,llama}a_pcpi`, `run_board_tests.sh
smollm2a gemmaa qwen3a llamaa`. All tests pass on all four:
- logits bit-exact with the sim (17/17 steps);
- long context at 2048 (Gemma 512), prefill against decode-only rows
  bit-exact;
- the length sweep;
- plan B and plan A (`rvsched`, `rvcode --embedded`, `rvsched --rv
  --embedded`): tokens identical, last logits bit-exact.

Decode device time per step (`profile --batch`, 32-token prompt), against the
f2 bundles (§4.9, SwiGLU fused):

| | SmolLM2 | Gemma | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| f2 (ms) | 34.04 | 54.15 | 127.77 | 215.42 |
| f2 + attention (ms) | **33.09** | **53.49** | **122.33** | **213.41** |
| Against f2 | **−2.8%** | **−1.2%** | **−4.3%** | **−0.9%** |
| Against v2.2 | −8.7% | −4.9% | −7.4% | −4.3% |
| Decode end to end, ms per step (f2) | 37.5 (39.7) | 57.0 (58.3) | 126.8 (133.9) | 216.0 (218.7) |
| tokens / s | 26.7 | 17.5 | 7.89 | 4.63 |

Decode device time at long context (sweep), f2 → f2 + attention: SmolLM2
90.6 → 87.3 ms at T = 2048 (−3.6%), Qwen3 306.4 → 286.6 (−6.5%), Llama
310.0 → 301.4 (−2.8%), Gemma 60.7 → 59.7 at 512 (−1.6%).

Reading:
- **The f2 bundles predate the tile alternation (§6.2.2, `21888b0`).** It is on
  by default for head_dim ≥ 128, so Qwen3's and Gemma's differences include
  it. Their prefill is the alternation alone: Qwen3 843.3 → 811.0 ms a chunk
  at T = 64 (−3.8%), Gemma 233.7 → 229.5 (−1.8%). SmolLM2's and Llama's
  prefill are unchanged (hd 64: no alternation by default; prefill rows
  unfused).
- **For SmolLM2 and Llama the difference is the fusion alone**: −2.8% and
  −0.9% at short context, −3.6% and −2.8% at 2048.
- **Below §6.4.6's model** (Qwen3 −5.0%, Gemma −2.6%, Llama −2.1%). The model
  ran at max_dynamic 256, where the kernel takes the untiled form with the
  interleaved schedule. At 2048 (Gemma 512) every model takes the tiled form,
  which still runs in order (§6.4.5). The gain grows with T: the boundaries
  removed are per layer, the in-order T loop is per block.
- **Next**: the interleaved schedule for the tiled form (post(b − 1) between
  head b's T loop blocks, accumulators by parity) is what the board's
  configurations use; a chunk kernel for prefill (§6.4.7).

**Second run (2026-10-19, `d391e13`): the tiled form interleaved.** The
same bundles rebuilt, with the tiled form on §6.4.6's schedule (A tiles
shared, head b − 1's post after head b's T loop). Llama falls back to the
in-order form (SaCodegen's ACC-full retry: G = 4 rows' buffers twice do not
fit). All tests bit-exact again. Decode device time, run 1 → run 2:

| | SmolLM2 | Gemma | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| short context (`profile --batch`, ms) | 33.09 → **32.87** | 53.49 → 53.49 | 122.33 → **121.64** | 213.41 → 213.12 |
| against f2 | −3.4% | −1.2% | −4.8% | −1.1% |
| T = 2048 (Gemma 512, ms) | 87.3 → 87.0 | 59.7 → 59.7 | 286.6 → 288.8 | 301.4 → 301.1 |

The interleaving gains about 0.6% at short context on SmolLM2 and Qwen3,
nothing on Gemma. Qwen3 at T = 2048 is 0.8% slower. That is far below the
model (Qwen3 at 2048 positions −1.2% of a 2-layer step, §6.4.6 ×28 ≈ −4.6%
in all, about 1 point over run 1). Not yet explained: the tiled loop's own
VE work (transposes, integer copies) sits on the same VE queue as the
previous head's post, so the overlap the untiled form gets is mostly absent,
and the extra buffers move the post's temps.

**On by default since 2026-10-19**: `compile_sa.sh` passes
`--iree-sa-fuse-attn` unless `SA_FUSE_ATTN=0` (`host_copies.py` the same);
`iree-compile`'s flag itself stays off, like the SwiGLU fusion's.

### 6.5 Banks (measured in §6.2.2)

The scoreboard tracks hazards per bank (SPAD_A ×2, SPAD_B ×2, ACC ×2): RAW,
WAR and WAW against in-flight commands of *other* engines. EX and VE also
conflict on any common SPAD_B / ACC bank, even two readers
([`double_buffer_design.md`](double_buffer_design.md) §7). The GEMV unit is the LD
engine. It reads x from SPAD_A and writes ACC. The roles:

| Role | Written by | Read by | Bank |
|---|---|---|---|
| q tiles (A operand), all heads, before the loop | VE | EX | SPAD_A 0 |
| raw K block | LD | VE (transpose) | SPAD_A 1 |
| p (GEMV x) | VE | LD (GEMV) | SPAD_A 1 (0 measured the same) |
| transposed tiles | VE | EX | SPAD_B 0 / 1 alternating |
| block scratch, scores, p · V accumulators of unit u | EX, VE, GEMV | VE | ACC u mod 2 |

The design feared SPAD_A, which has three roles in two banks. The prototype
measured that cost as small: p's bank changes the cycles by at most 1%.
SPAD_B is the costly bank. With the tiles in one bank, each block's transpose
waits for the previous block's EX. Alternating the tiles is now done (§6.2.2):
−1.8% of decode at T = 2048 by default, −7.4% with 256 rows per iteration.
A finer scoreboard (LLM plan §11.3) is not needed for this design.

### 6.6 Plan

| Step | Content | Estimate |
|---|---|---|
| 1 | ~~Schedule prototype~~ **done 2026-10-18** (§6.2.2, `attn_schedule.py`): stage 1 −4.0% to −4.3% of decode (Qwen3, T = 48 to 2048) with an implementable schedule; with alternating tiles −4.9% to −11.6%. **Go.** | done |
| 2 | ~~Alternating tiles in the existing q · Kᵀ T loop~~ **done 2026-10-18** (§6.2.2): each block as two halves in alternating SPAD_B banks, where a half is at least 8 KB. Qwen3 −0.9% to −1.8% of decode by default, −7.4% at 2048 with `--iree-sa-attn-block 256`. Bit-exact; golden: the three tiled q · Kᵀ dispatches changed | done |
| 3 | ~~Region pass with dynamic T~~ **done 2026-10-18** (§6.4.1): the regions IREE formed merged after formation (plugin hook, IREE patch 0004), `--iree-sa-fuse-attn` / `SA_FUSE_ATTN=1`. 66 chains per 2-layer model (decode layers and prefill rows), workload over T from IREE. The backend is unchanged, so they run on the host. Gemma / Qwen3 / Llama end to end with the same tokens; the logit difference is the VMVX exp, shown on Llama | done |
| 4 | (decode only since §6.4.7: prefill rows unfused) ~~4a: `AttnLayerPlan` and `expandAttnLayer`, in order~~ **done 2026-10-18** (§6.4.5): the fused dispatch on the accelerator, decode and prefill rows, untiled and tiled forms; bit-exact end to end on Gemma / Qwen3 / Llama; model decode −3.5% / −1.1% / −1.4%. ~~4b: the interleaved schedule~~ **done 2026-10-18** (§6.4.6): A tiles shared across heads, one EX a head, post(b − 1) after head b's EX, ACC banks by parity, Gemma untiled; model decode Qwen3 −5.0%, Gemma −2.6%, Llama −2.1%. Left: row pieces per head, the tiled form's schedule | done (the rest optional) |
| 5 | Verification: lit; golden corpus with the flag off; `dispatch_check.py` per dispatch at T = 16, Tb ± D, 2·Tb, max T; end to end on the sim bit-exact with the unfused build (four models, 2 layers); prefill vs decode rows bit-exact; `test_long_context.py` at 2048 (512 Gemma); `host_copies.py` | 4–5 days |
| 6 | Board: the four bundles (`run_board_tests.sh`: `llm`, `profile --batch`, long context and its sweep, plan A / plan B) | 2 days |

About four weeks for steps 2–6. Expected on Qwen3 (model): decode −4%
(fusion) and up to −12% at T = 2048 with step 2. Prefill gains more than
decode, because its rows each pay the boundaries (§6.2.1). The other models
remain to be modelled with `attn_schedule.py --dispatches`.

**Risks.** (1) The model's accuracy for a fused list. It was calibrated per
dispatch, but SwiGLU's fused lists landed within 0.4 points of it on the board
(§4.9). (2) Dynamic-shape regions in IREE, the reason for step 3 before
the backend. (3) The per-layer generalization (`--iree-flow-enable-executable-generalization`)
must still merge the fused executables across layers: the caches' offsets and
scales as push constants, as today (checked by the decode working-set size,
`rv_codesize.py`). (4) The T loop's guarantees, no read past T and the
`sa.advance_final` epilogue, now hold inside a head loop and with a body
unrolled by two. An odd block count must not prefetch past the cache. The
per-dispatch checks at block boundaries cover it.

### 6.7 Prefill: the chunk kernel (2026-10-19/20)

A prefill chunk ran its attention row by row, each row a decode step's
dispatches (§6.1): about 12 dispatches a row, 16 rows a chunk, and every row
loading and transposing the K cache of every head again. Now the chunk's
rows are one dispatch per layer and one kernel, `attn_chunk`.

**Frontend** (`hf_generic._attention_rows`). The rows' K / V cache writes all
come first (each row's module forward, its query recorded, `sa_collect`),
then the attention of the M rows over the cache after the writes
(`sa_attention` with `query_rows`), every row's q quantized before the
first row's scores. Each row's mask is its own position, so the later rows'
keys never reach it: the max is over the same keys and p is 0 elsewhere.
Eager logits are bit-identical to the row-by-row export (prefill and
decode), and so is the unfused build on the sim.

**Region pass** (`--iree-sa-fuse-attn-chunk`, `FormAttnRegions.cpp`, option
`chunks`). The chains of a chunk's rows (found by their shared q producer,
§6.4.7) are merged into one region:
- each row's small inputs computed outside are copied in for it: its slice
  of the chunk's q, the RoPE rows at its position (small regions without a
  contraction, also read by the k path). The region then reads the chunk's
  q, the positions and the tables once: 16 rows' own inputs were 48 bindings
  and base registers, beyond the 16 / 15 there are;
- the rows' results, put together by an `insert_slice` chain into a
  `tensor.empty` (the `torch.cat`), are stacked as they are, `[rows, ...]`,
  and reshaped after the region: one result, and no collapse in front of each
  row's store (IREE's VMVX codegen mis-typed the binding of Gemma's rows,
  `[1, 4, 16, 16]` collapsed to `[4, 256]`).
- **An IREE miscompile found and fixed**: `FoldInsertSliceWithTensorStoreOp`
  rewrote `store(insert_slice(src into dest))` as a store of `src` alone,
  dropping `dest`. Correct only for an undefined `dest`; a chain kept only its
  last row, the other 15 rows' work then dead code. `dest` is now stored first
  unless it is a `tensor.empty`
  (`compiler/patches/iree/0005-tensorext-fold-insert-slice-store-keep-dest.patch`).
  The golden corpus is unchanged by it.

**Backend.** `KernelMatcher::matchAttnChunk` matches the dispatch as rows:
an `AttnLayerPlan` each (decode's matcher), all reading the same K / V cache
views, plus each row's q path (the last writers of its q and s_q, back to
DDR) and the store of its output. `expandAttnChunkBlocked`
(`SahlExpandKernels.cpp`), for M = D = 16 rows:
- *q paths*: each row's in a scope of its own (its buffers released after
  it: a dispatch's buffers are otherwise allocated for the whole dispatch,
  and 16 rows' RoPE and norm temps do not fit ACC); its int8 q into row m of
  a staging matrix `[16, H G hd]`, its s_q into row m of `[16, U]`
  (U = H G units, a unit = a KV head b and query head g).
- *A tiles*: the staging matrix transposed once into columns, then one
  transpose per unit: unit u's tile holds the 16 rows' q of that head.
  s_q per unit by an fp32 transpose of `[16, U]` (`sahl.transpose` now
  also ACC → ACC).
- *per KV head, two passes over the cache in blocks of Tb keys* (`sahl.loop`,
  the count ⌈T / Tb⌉ from T; the loads advance with the loop):
  - pass 1: the block's K rows loaded and transposed once for the head's G
    units; per unit one EX `[16, Tb]`, the scores' epilogue, the mask, the
    block's maximum per row folded into the running maximum (the reduction
    with `sahl.accumulate`, now over several rows);
  - pass 2: K again and V's block (its packed groups are the B tiles of
    p · V as they are), followed by a tile of ones; per unit the scores again,
    p = i8(exp(s − max) · 127) for the 16 rows, p into the A form (two
    transposes), then p · V and Σp as one EX each, accumulated over the
    blocks (the ones tile gives every lane the row's sum: no VE reduction);
  - after: p · V's epilogue by the sums, the 16 rows' outputs stored.
  Every VE operation covers the 16 rows `[16, Tb]`; the bodies are the
  matched generics' (scores' epilogue, max, p, p · V's epilogue), so the
  arithmetic is the IR's.
- *The mask is arithmetic*, each row at its own position:
  s' = min(s, (pos − key + ½) · 10³⁰). A key past the row's position gets
  about −10³⁰: the max is over the earlier keys as before, exp gives 0, p is
  0 as the IR's select gives. No VALID, so blocks partly or wholly past T and
  any positions (the last chunk of a prompt starts anywhere) need nothing
  special, and the loop body is the same for every block. The key indices
  are `sahl.iota` (lanes from VALID prefixes, then + D per word), the
  positions `sahl.scalar_f32` (an i64 scalar as an fp32 word), new sahl ops,
  as is `sahl.splat` (an accumulator zeroed, the ones tile).
- No online softmax: two passes keep the max global, so p's int8 rounding is
  the unfused build's.

Infrastructure on the way: the private PARAMs are numbered by liveness when a
template has more than 8 (16 rows' positions and RoPE rows; LDPARAM-written
ones reused, the others kept), and a `sa.fresh_params` scope forgets the VALID
counts it loaded; a DDR view through `expand_shape`; `sahl.loop` takes two
strides always. None changes a template that compiled before (golden
identical).

**Verification** (2-layer builds, D = 16, 4 ports): `dispatch_check` of the
chunk dispatch OK at T = 16, 64, 128, 144, 256 (`--dirty`, and `--pos 0`)
on Gemma, Qwen3 and Llama at 256 positions, and on Qwen3 at 2048 positions
at T = 16 to 2048; logits end to end on the sim bit-identical to the unfused
builds (all four builds). `dispatch_check` now gives fp32 push constants a
scale's value (it gave t's bits, a denormal: every output 0) and can save the
executed commands (`SA_DC_TRACE`) for the cycle model.

**Model** (Qwen3, one layer's prefill attention for a chunk of 16 rows, cycles;
rows unfused: each row's scores, probabilities, sum and p · V dispatches at
T, × 16, plus the q paths, without the dispatch boundaries):

| T | rows unfused | chunk, Tb = 32 | **Tb = 64 (default)** | Tb = 128 |
|---|---:|---:|---:|---:|
| 64 | ≈ 1.16M | 0.45M | **0.42M (−64%)** | 0.52M |
| 512 | ≈ 2.81M | 1.58M | **1.32M (−53%)** | 1.19M |
| 2048 | ≈ 9.0M | 5.45M | **4.38M (−51%)** | 3.85M |

`--iree-sa-attn-chunk-block` sets Tb. The VE is the busy engine (≈ 75%):
at T = 64 half of it is the 16 rows' q paths (RMSNorm, RoPE, quantization:
the unfused build's work, ≈ 11K cycles a row), the rest the units' blocks;
exp is the largest single operation. A 2-layer prefill chunk (the model,
T = 32): 503 → 217 dispatches.

#### 6.7.1 Board results (2026-10-20, bundles `*c`)

The four models exported again by this frontend (`qllama.irpa` byte-identical
to the earlier exports: the sim's references of the bundles stay valid),
compiled with the defaults and `SA_FUSE_ATTN_CHUNK=1` (+ `--iree-sa-riscv`) at
2048 positions (Gemma 512), bundles `deploy_hfgen_{smollm2,gemma,qwen3,llama}c_pcpi`
(`build/chunk/bundles_chunk.sh`), `run_board_tests.sh smollm2c gemmac qwen3c llamac`.
On the host first: no dispatch on the host, `host_copies.py` PASS, the chunk
dispatch's `dispatch_check` OK at T = 16, 144 and the maximum, each with `--pos 0`.
All eight tests pass on all four: logits bit-exact with the sim (17/17 steps);
long context, prefill against decode-only rows bit-exact (17/17), the generated
tokens identical; the length sweep; plan B and plan A (`rvsched`, `rvcode
--embedded`, `rvsched --rv --embedded`).

Prefill time against the `*a` bundles (§6.4.8: the same build without the
chunk kernel), board, ms:

| | SmolLM2 | Gemma | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| 40-token prompt, 3 chunks (`llm`) | 924.9 → **703.9 (−24%)** | 764.8 → **658.2 (−14%)** | 2547.5 → **1914.5 (−25%)** | 3181.5 → **2898.0 (−9%)** |
| 32-token prompt, 2 chunks (`profile --batch`) | 620.0 → **472.5 (−24%)** | 522.6 → **452.0 (−14%)** | 1705.8 → **1288.0 (−24%)** | 2132.4 → **1948.6 (−9%)** |
| 2032-token prompt, 127 chunks (Gemma 496, 31) | 95.8 s → **62.9 s (−34%)** | 9.06 s → **7.19 s (−21%)** | 277.3 s → **151.2 s (−45%)** | 227.7 s → **182.9 s (−20%)** |
| prefill tokens / s at 2032 (Gemma 496) | 21.2 → **32.3** | 54.8 → **69.0** | 7.33 → **13.44** | 8.92 → **11.11** |

Decode is unchanged (37.3 / 57.0 / 126.1 / 215.7 ms a step). The gain grows
with the context: a row by row chunk loaded and transposed every head's cache
16 times. Llama gains least: its prefill is mostly the linear layers (hidden
2048, 8192 in the MLP), the attention a smaller share.

**A fix the whole Gemma needed** (the 2-layer build passed): its RoPE rows are
computed in small regions together with each row's k norm, so the copies of
§6.7's region pass captured those norms, 32 inputs a chunk. In the 2-layer
build their offsets were constants (one base register for the binding); with
the per-layer generalization (§9.11 of the RISC-V design) each became a push
constant and a base register of its own, past 15. The copying now goes on in
rounds: a copied region's own small inputs (at most 1024 elements: a row's
vectors, not the chunk's tensors) are copied too, and the chunk reads the k
projection once. Gemma's chunk region: 448 regions with a 4096-element bound
(it copied each row's dequantized k chunk too: ACC full), 256 with 1024.

**On by default since 2026-10-20**: `compile_sa.sh` passes
`--iree-sa-fuse-attn-chunk` with `--iree-sa-fuse-attn` unless
`SA_FUSE_ATTN_CHUNK=0` (`host_copies.py` the same); `iree-compile`'s flag stays
off. An export from before the frontend change (each row reading the cache after
its own write) keeps its rows unfused: the region pass merges a chunk only when
its rows read the same int8 tensors from outside (the caches, through views);
checked on such an export (no chunk formed, logits bit-identical to before) and on
this frontend's 2-layer Qwen3 and Gemma with the defaults (chunks formed, logits
bit-identical to the unfused builds).

Not done (§7): pass 1 recomputes what pass 2 needs (K's loads and transposes, the
EX, the scores' epilogue and mask; a single pass with the scores kept in ACC for T
up to a few hundred would save an estimated 30–40% of the units' blocks there);
the q paths batched over the rows (they are bound by the VE's throughput, not its
fixed cost: about 5–10% of them).

## 7. Where it stands (2026-10-20): closed

| Fusion | Status | Board, against the build before it |
|---|---|---|
| SwiGLU, decode (§4) | on by default | decode −3.3% to −6.1% |
| Decode attention, stage 1 (§6.4) | on by default | decode −1.1% to −4.8% (−2.8% to −6.5% at T = 2048) |
| Prefill chunk kernel (§6.7) | on by default | prefill −9% to −25% (40 tokens), −20% to −45% (2032) |
| The tiled form's interleaving (§6.4.8) | done | about +0.6%, below the model; not explained |
| Decode attention stage 2, k / v path (§6.3) | deferred | bound −0.5% decode / −1.0% prefill (Qwen3) |
| q_proj / o_proj in the region (§6.3) | deferred | bound −0.1% to −0.9% |
| Residual + RMSNorm + projection (§5) | deferred | bound −0.9% to −4.0% decode, −2.5% to −4.8% prefill |
| SwiGLU for prefill | deferred | bound −2.4% to −4.2% prefill |

Against v2.2: decode device time −4.4% to −9.3% (the upper bound of §2.2 was
−4.5% to −14%); prefill well past §2.2's −10% to −20%, because the chunk kernel
also removed work (every row reading and transposing every head's cache), not
only the round trips.

**Why stop here.** Each chain left is a few percent at most in the model, and the
board has given about half of the model before (§4.8, §6.4.8); each costs a
region pattern, a kernel and a round of 2-layer, golden, bundle and board checks.
Decode is bound by the GEMV's weight streaming (about 80% of a step), prefill now
by the linear layers. The larger levers are elsewhere: fewer weight bytes (W4A8,
the performance model's answer to the next step, `perf_model.py --whatif`), the
hardware side (LD / GEMV width, command issue), and the results for the talk.

## Files

- `compiler/tests/fusion_whatif.py`: the evaluation (`--only` for one chain by kind, `--chain` by role).
- `compiler/tests/attn_schedule.py`: the attention schedule prototype (§6.2.2).
- `SahlKernels.cpp` (`matchAttnChunk`), `SahlExpandKernels.cpp` (`expandAttnChunkBlocked`):
  the prefill chunk kernel (§6.7); `compiler/patches/iree/0005-tensorext-fold-insert-slice-store-keep-dest.patch`.
- `compiler/plugins/sa/target/FormAttnRegions.cpp`, `test/sa_form_attn_regions.mlir`:
  the attention region pass (§6.4.1); `compiler/patches/iree/0004-pluginapi-dispatch-region-formation-hook.patch`:
  its hook.
- `compiler/tests/perf_model.py`: the command-level model (a FENCE now holds
  back later commands; it did not before 2026-10-17).
- `build/perfmodel/<bundle>.trace`, `<bundle>_cal.json`,
  `<bundle>_fusion.json`, `swiglu_whatif.txt`: the traces, calibrations and
  results of §2 (not in git).

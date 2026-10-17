# Cross-operator fusion: evaluation, the SwiGLU and attention designs

Status: evaluation done (2026-10-17, `compiler/tests/fusion_whatif.py`,
[`iree_compiler_plan.md`](iree_compiler_plan.md) §8.23); the SwiGLU fusion is
implemented (`97a3323`, `--iree-sa-fuse-ffn`; on by default in
`compile_sa.sh` since 2026-10-18, `SA_FUSE_FFN=0` turns it off) and accepted on the board on all four models (2026-10-17, §4.8): bit-exact,
decode −1.1% to −2.7% of device time, about half of the model's −2.3% to
−4.4%, for a reason found in §4.8 (the fused gating is not chunked enough to
overlap the weight streaming). The schedule is fixed in `ea7ca80` (§4.9)
and accepted on the board (2026-10-18): bit-exact, decode −3.3% to −6.1% of
device time, within 0.4 points of the model's −3.4% to −6.3%. The attention fusion is designed in §6 (2026-10-18, not implemented).
Measured against tag `v2.2-kv260`.

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
(§6.2.2), not implemented. The second fused chain after
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

#### 6.4.1 Frontend / preprocessing: `iree-sa-form-attn-regions`

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
- **Risk to check first** (§6.6 step 3): regions with dynamic dims this
  early. IREE's own dispatch formation produces today's dynamic attention
  dispatches, but a region formed by hand must carry the workload ordinals
  for T. The FFN regions were static.

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
| 3 | Region pass with dynamic T: the formed region through IREE's dispatch creation, `--compile-to=stream` and the backend unchanged (the region compiled as today's pieces would be, unfused inside) | 3–4 days |
| 4 | `AttnLayerPlan` match and `expandAttnLayer` with §6.4.2's schedule, decode at T within one block first, then the T loop | 8–12 days |
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

## Files

- `compiler/tests/fusion_whatif.py`: the evaluation (`--only` for one chain by kind, `--chain` by role).
- `compiler/tests/attn_schedule.py`: the attention schedule prototype (§6.2.2).
- `compiler/tests/perf_model.py`: the command-level model (a FENCE now holds
  back later commands; it did not before 2026-10-17).
- `build/perfmodel/<bundle>.trace`, `<bundle>_cal.json`,
  `<bundle>_fusion.json`, `swiglu_whatif.txt`: the traces, calibrations and
  results of §2 (not in git).

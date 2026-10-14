# Compiler frontend analysis: current state, industry comparison and gaps to generality

Status: 2026-10-03, based on reading `compiler/frontend/` and `docs/iree_compiler_plan.md`. Not verified by actually running other models; points marked "inferred" are conclusions from reading the code.

Related documents: [`iree_compiler_plan.md`](iree_compiler_plan.md) (§3 frontend, §8.9 HuggingFace import, §8.16 generic frontend), [`../compiler/README.md`](../compiler/README.md).

---

## 1. The two frontend paths

| Path | Files | Approach | Status |
|---|---|---|---|
| Hand-written quantized model | `qllama.py`, `qhf.py`, `export_hf.py` | The decoder rewritten in PyTorch by hand (`QLlama`, `QModel`), quantization and numeric steps written out explicitly | stories15M, SmolLM2-135M, Qwen3 |
| Generic path | `hf_generic.py` | transformers' original modeling code exported as is; quantization and formulation done by module rewrites and graph rewrites | F0–F4 done, F6 (non-Llama structures) not done |

Both paths share the export and compile flow of `export.py` and then enter the same `sa` backend.

### 1.1 Hand-written path (`qhf.py`)

```
config.json + *.safetensors
  → HFConfig (GQA, head_dim, QK-norm, theta, eps, tie; rope_scaling raises an error)
  → load_weights (wq / wk and q_norm / k_norm reordered per head from rotate-half to interleaved pairs)
  → Fp32Model calibrates the KV scales (amax / 127)
  → QModel (int8 weights + per-channel scales, int8 KV cache)
  → aot.export → qllama.mlir + qllama.irpa
```

- **GQA**: the kv heads are the batch dimension of the attention matmul; the G = heads / kv_heads Q heads of a group are the row dimension.
- **KV cache**: flattened into an int8 buffer of `(layers * seq_len, kv_dim)`, one `index_copy_` per layer.
- **prefill**: linear layers batched over M rows; gather / RoPE / KV write / attention done row by row in decode's scalar form.
- **Verification**: there is no hand-written device model to compare against. Each dispatch is checked against its own IR (`dispatch_check.py` + `oracle.py`); end to end it is compared with fp32 within tolerances (correlation > 0.98 etc.).

### 1.2 Generic path (`hf_generic.py`)

| Mechanism | Role |
|---|---|
| `HFDecoder` + `PositionedCache` | Wraps `AutoModelForCausalLM`; the static KV cache is written into module buffers, so `torch.export` sees the buffer mutation |
| `quantize()` | By module type, `nn.Linear` → `QLinear`, `nn.Embedding` → `QEmbedding` (W8A8); an embedding shared with the classifier is quantized once |
| `rope_rewrite()` | `*RotaryEmbedding` → `RopeTable` (cos / sin lookup); q / k projections reordered per head into interleaved pairs; `rotate_half` → `swapneg` |
| `norm_rewrite()` | `*RMSNorm` → `SaRMSNorm` (`mean` becomes a multiply by the constant 1/n) |
| `SaCache` + `sa_attention` | Registers `"sa"` through `AttentionInterface`; int8 KV, softmax masked by position, GQA form |
| `GRAPH_PASSES` | Currently only `pow(x, 2) → x * x` |
| `export()` | `torch.export` (non-strict) → `run_decompositions` (iree-turbine's decomposition table) → graph rewrites → externalized parameters → iree-turbine |

**Criterion**: bit-exactness with the hand-written path is not a goal (the two paths differ in the details of their formulas, and int8 rounding is sensitive to the order of operations). Instead: every sa dispatch is bit-exact with the oracle, plus quality versus the original fp32 HF model (top-1, correlation) no worse than the hand-written path.

---

## 2. Mainstream LLM compiler frontends in industry

### 2.1 Graph capture

| Approach | Representatives | Characteristics |
|---|---|---|
| Tracing | `torch.jit.trace`, `torch.fx` | Loses data-dependent control flow; largely superseded |
| Bytecode-level capture | TorchDynamo (`torch.compile`) | Takes original HF code; unsupported Python triggers a graph break |
| Symbolic export | `torch.export` | Whole graph, no graph breaks, dynamic shape constraints; requires a certain coding style |
| Rebuild the model | TensorRT-LLM, vLLM, MLC-LLM, llama.cpp | Does not capture arbitrary Python; rebuilds the structure from its own operator library; the most common in industry |
| JAX tracing | JAX `jit`, PyTorch/XLA | Produces StableHLO; shapes mostly static, variable length handled by bucketing / padding |

This repository: the hand-written path is "rebuild the model"; the generic path uses `torch.export` (through iree-turbine).

### 2.2 Lowering to a canonical operator set

- PyTorch: ATen → Core ATen → Prims; the MLIR family: torch-mlir → `linalg` / `tensor` / `arith` / `math`, or StableHLO / TOSA.
- LLM compilers usually **keep** high-level operators such as attention, RMSNorm and softmax, because after decomposition re-fusion rarely recovers FlashAttention-level kernels. This repository recovers them by pattern matching in `sa-to-sahl` (`sahl.kernel`).

### 2.3 Common approaches to LLM-specific problems

| Problem | Approach |
|---|---|
| Dynamic shapes | Symbolic shapes (`SymInt`, `torch.export.Dim`); bucketing + padding; inflight batching |
| KV cache | Explicit inputs and outputs; mutable global state (this repository); PagedAttention (vLLM) |
| prefill / decode | Compiled as two graphs, or specialized by token count |
| Generation loop, sampling | Outside the graph, driven by the runtime |
| Custom operators | `torch.library`; the compiler treats them as black boxes or maps them to kernels |
| Parallelism | DTensor / GSPMD / Shardy describe sharding; the compiler inserts communication |

### 2.4 Where quantization sits in the frontend

- PT2E / `torch.ao`: Q/DQ nodes inserted into the graph, fused by the backend.
- Weight quantization (GPTQ / AWQ / FP8 / INT4): converted offline into a packed format; the graph uses dedicated dequant-matmuls.
- This repository writes the whole quantization chain explicitly in PyTorch; **the numeric specification is fixed by the frontend**, guaranteeing bit-exactness with the hardware. Industry more often tolerates small numeric differences in exchange for flexibility.

### 2.5 System comparison

| System | Frontend approach | Backend |
|---|---|---|
| `torch.compile` | Dynamo + AOTAutograd + FX | Inductor: Triton / C++ |
| TensorRT-LLM | Own Python model definitions + plugins | TensorRT engine, CUTLASS |
| vLLM | Own model implementations + PagedAttention | Own CUDA kernels |
| XLA / JAX | JAX tracing → StableHLO | XLA fusion and code generation |
| IREE | `torch.export` / iree-turbine → linalg | HAL backends (including this repository's `sa`) |
| MLC-LLM / TVM | Relax + model builder | TIR scheduling |
| llama.cpp | Hand-written graph + GGUF quantization | Hand-written kernels |

---

## 3. GQA and the `attention` micro-kernel (backend interface)

Conclusions from reading `SahlKernels.cpp` (inferred):

- `matchAttention` only recognizes the MHA form: scores `bmm(ext(Kᵀ), q)` with q of `[H, hs, 1]`; in P·V, p is `[H, 1, T]`; the cache is read by an expanding generic `(t,h,j) -> (h,t,j)`.
- GQA (the form of `QModel` and `sa_attention`): `q[Hk, G, hs] @ K[Hk, hs, T]`, with the KV heads as batch and G as rows. This does not satisfy the conditions above.
- So GQA goes through `matchContraction` and becomes `sahl.kernel "contraction"`:
  - `bLoop` = KV heads (in both x and M), `gLoop` = Q heads within a group (only in x and the output), `nLoop` = T or hs, `kLoop` = the reduction dimension;
  - the lowering is done by `Lowerer::contraction()`: the B block is loaded once per KV head, and for each `(b, g)` one row of x is loaded and replicated by DIV-D into an A strip;
  - limits: no dynamic batch / rows; G ≠ 1 not allowed when K needs tiling; serial schedule, no prefetch.
- Consequence: `--iree-sa-ukernels=all` has no extra effect on GQA attention; the room for tuning is in `contraction()`.

---

## 4. Gaps to generality

### 4.1 Structural coverage

F6 in the plan (GPT-2: LayerNorm, GELU, learned positions) is not done.

| Feature | Current state |
|---|---|
| LayerNorm (mean, bias) | `norm_rewrite` only matches `RMSNorm` by class name |
| GELU and other activations | No rewrite; the SFU only has EXP, RECIP, RSQRT, nothing for `tanh` / `erf` |
| Learned / ALiBi position encodings | `rope_rewrite` only handles RoPE; `sa_attention` has no score bias |
| Sliding window | The mask is only `arange(T) <= pos` |
| MoE | No routing, top-k, or per-expert gather |
| MLA / low-rank KV | `SaCache` assumes `[T, kv_heads * head_dim]` |
| Encoder-decoder / multimodal | The `HFDecoder` signature has only `input_ids, position_ids` |

### 4.2 Hard-coded assumptions in the rewrites

- `rope_rewrite`: assumes an even `head_dim`, projections laid out per head, and module names `q_proj` / `k_proj` / `q_norm` / `k_norm`; partial rotary is not handled.
- **The `rotate_half` replacement applies to the whole process** (its docstring says so): a non-rewritten model of the same family loaded afterwards computes RoPE wrongly.
- Matching by class name / attribute name (`endswith("RMSNorm")`, `hasattr(mod, "q_proj")`); fused QKV (`c_attn`, `qkv_proj`) does not match.
- `quantize()` treats every `nn.Linear` the same, with no configuration to "skip sensitive layers" (router, LoRA, etc.).
- `from_pretrained(..., dtype=torch.float32)` first loads the whole model in fp32, which is not feasible in memory for large models.

### 4.3 Quantization quality and calibration

- Only naive per-channel W8A8 quantization; no GPTQ / AWQ / SmoothQuant, no outlier handling.
- Static KV scales calibrated on a short fixed text; a different domain may overflow or waste precision.
- Coarse quality criteria: top-1 agreement ≥ 85%, mean correlation > 0.95; no perplexity or downstream-task regression.
- The generic path has no bit-exact end-to-end reference.

### 4.4 Tight coupling to the transformers version

Depends on 5.x internal interfaces: `StaticCache.layers`, `layer.update`, `layer.cumulative_length`, `early_initialization`, `AttentionInterface`. The plan already records one bug caused by version behavior (the causal mask built from the cache's internal length).

### 4.5 Export and shapes

- batch fixed at 1; decode is `[1,1]`, prefill a block of fixed M.
- `max_len` is static per build (`--max-len`, default 256; the export also sets the compiler's `--iree-sa-max-dynamic`); long contexts are limited by on-chip capacity: on the KV260 (D = 16) the decode attention holds T × head_dim in one SPAD_B bank, so T ≤ 256 for Gemma-3 (head_dim 256), 512 for Qwen3 (128), 1024 for Llama-3.2 / SmolLM2 (64), before the T tiling of `iree_compiler_plan.md` §8.22 (C6.5 / C6.5b, done 2026-10-14): now Qwen3 compiles for 2048 (and 4096 but for one dispatch) and Gemma for 512 (its sliding window) with everything on the accelerator, checked on the board.
- All layers share one K and one V buffer to work around the mutable-global merge problem of IREE 3.11 (`compiler/tests/iree_global_merge_repro.py`); a workaround for an upstream defect.

### 4.6 Backend coverage

- Tiled elementwise dispatches do not support a sum over a whole vector (it would change the summation order).
- With `--iree-sa-ukernels=none`, multi-row prefill linear layers once failed to compile (recorded in §8.14).
- The host fallback interprets with VMVX: with SmolLM2's linear layers forced to the host, decode takes 16.8 s per step. "Compiles" does not mean "usable".

---

## 5. Suggested priorities

| Priority | Item | Reason |
|---|---|---|
| High | Finish F6: LayerNorm, GELU, learned positions, validated on GPT-2 | Already planned; tests whether the generality is real |
| High | Remove the process-global `rotate_half` replacement | Affects correctness; easy to trip over in batch tests |
| Medium | Fused QKV, partial rotary, sliding window | Covers families such as Phi, Mistral, GPT-NeoX |
| Medium | Configurable quantization: skipped layers, SmoothQuant, richer calibration sets | Key to the accuracy of large models |
| Medium | A transformers version-adaptation layer and a pinned version | Lowers the maintenance risk |
| Low | MoE, MLA, multimodal | Need new backend capabilities; large effort |

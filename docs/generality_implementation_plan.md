# After the 4-port GEMV: implementation plan for generality and quantization quality

Status: implementation draft (reviewed 2026-10-10 against the code and the four-port board results: WP0 and WP1 are basically done, WP5's EX baseline already exists, WP6 only lacks compiler integration). **Progress 2026-10-12**: WP2, WP3 (SmolLM2), WP7 (Llama-3.2-1B) accepted on the board; WP4 Qwen3 K per-channel accepted on the board; first version of WP9 done (§13.1: statistics, board teacher forcing, decode / prefill breakdown; with the V projection on GEMV and attention over the actual length, Qwen3 decode 6.88 tok/s, prefill 20.5 tok/s); WP6 switch and boundary tests done; WP6 decode K split integrated into the compiler; SmolLM2 per-channel KV evaluation done (kch 28.82 / 87.7%), SmolLM2 whole model with kch + actual length accepted on the board (decode 22.17, prefill 60.09 tok/s, board PPL 28.55, §7). **Progress 2026-10-13**: Llama-3.2-1B K per-channel rebuild accepted on the board (4.04 / 14.97 tok/s, §11); 4-port performance model recalibrated, §17.6 answers question 6 (next hardware investment W4A8); **WP8 Gemma-3-270M whole model accepted on the board** (790 executables all on the accelerator, 17/17 bit-exact, decode 14.96, prefill 62.5 tok/s, board PPL 158.5, §12.4), §13.2 main table has all four rows, §17.4 answers question 4; Gemma's quality loss traced to the int8 attention probabilities, changed to normalization by the quantized sum (qsum, 1a0ba3d): Gemma, SmolLM2, Qwen3 rebuilt and accepted on the board, board KL −25% / −31% / −29% (Gemma PPL 158.5 → 142.8), speed unchanged (§12.4); Llama being rebuilt; §17 now answers all six questions (17.1–17.6)  
Date: 2026-10-10 (updated 2026-10-12)  
Baseline: `main@b42fec6` (four-port acceptance `fb01909`)  
Milestones: talk content freeze 2026-11-15; IREE community meeting 2026-11-25

## 1. Goals and scope

After the G5 four-port GEMV, this stage addresses two problems first:

1. **Generality**: one PyTorch / Hugging Face → IREE → SA descriptor flow covering four model structures: Qwen3, SmolLM2, Llama 3.2 and Gemma 3.
2. **Quantization quality**: import ready-made compressed-tensors W8A8 weights, and measure "the checkpoint's own quantization loss" separately from "the extra numeric loss this project introduces".

High-risk hardware work such as 200 MHz, W4A8, SPAD sub-banks and K3 URAM is not started before the meeting. The only exceptions are the GEMV K split, whose hardware support is done and only needs compiler integration, and row-by-row prefill execution if time allows.

### 1.1 Success criteria before the meeting

| Level | Deliverables |
|---|---|
| Must | ~~Full four-port GEMV acceptance~~ (done 2026-10-10); compressed-tensors W8A8 importer; SmolLM2 validation; Qwen3 quality re-measurement; Llama 3.2 correctness path and memory conclusion |
| Should | GEMV K split for Llama's down_proj; Llama end to end on the board; first ablation tables |
| Show as in progress | Gemma 3 running on the sim, coverage matrix, main fallback reasons |
| Decide after the meeting | 200 MHz, W4A8, H2/LD pipelining, SPAD sub-banks, T tiling and URAM |

### 1.2 Non-goals

- The first importer does not support int4, FP8, asymmetric quantization, static activation quantization, or per-group/per-block weight quantization.
- Gemma's first version is not required to run long contexts fully on the KV260.
- top-1 over 40 positions is not taken as the final quality conclusion; it is only a fast regression metric.
- Host wall-clock time and device cycles are not simply added, unless the runtime confirms they do not overlap.

## 2. Overall order of execution

```text
G5-4 wrap-up (done; only perf_model calibration left)
  → compressed-tensors importer
  → synthetic checkpoints + SmolLM2 validation
  → Qwen3 quality attribution
  → Llama frontend, memory preflight and EX baseline
  → GEMV K split
  → full Llama sim / board acceptance
  → first ablations
  → Gemma short-context sim and coverage matrix
  → decide the next optimization from the time breakdown
```

Key principle: Llama first establishes a correctness baseline on the EX path, and only then enables the K split. This lowers the coupling in debugging and naturally yields a set of compiler ablation experiments.

## 3. Common acceptance criteria

### 3.1 Layers of correctness

Each model passes the following layers in order:

1. **Checkpoint reference**: the original checkpoint as recognized by compressed-tensors / Transformers or vLLM. The convention for dynamic activation quantization follows **vLLM's runtime semantics** (per token, `scale = amax / 127`, the same as the device and `qllama`); the fake-quant used by llm-compressor during calibration may use `amax / 127.5` and clamp to [-128, 127] (to be verified) and is not used as the reference, otherwise this layer would mix in convention differences unrelated to this project.
2. **Importer eager reference**: the PyTorch eager model built by the project's importer.
3. **SA oracle**: a host reference using the device's quantization semantics, approximate functions and rounding rules.
4. **funcsim / dispatch_check**: every dispatch's inputs and outputs bit-exact with the SA oracle.
5. **Board execution**: logits bit-exact with funcsim, generated tokens identical to the EX path.

"Bit-exact" only applies to layers 3–5, which use the same integer semantics; the quantized model is not required to be bit-exact with the bf16/fp32 model.

### 3.2 Quality metrics

| Metric | Use | Minimum data |
|---|---|---|
| Fixed 40-position top-1 | Fast regression, daily tests | 40 fixed positions |
| top-1 agreement | Formal reporting | At least several thousand token positions |
| Mean correlation / cosine | Overall logit drift | Same positions as top-1 |
| KL divergence | Shift of the probability distribution | Same positions as top-1 |
| WikiText-2 perplexity or an equivalent PPL | End-to-end language-modeling quality | A fixed, reproducible data split |

**Where to run them**: the functional simulator is too slow (4 decode steps of Qwen3 take about 819 s on the sim), so the formal metrics over thousands of positions do not run on funcsim. Two ways: ① the SA oracle on the host (eager, device quantization semantics), for layer 3; ② teacher forcing with prefill on the board (Qwen3 prefill about 16 tok/s, 4000 positions in about 4 minutes), for layer 5. ② needs the runtime to export the logits of every prefill position (today only the last position), listed under WP9.

Calibration sets and quality test sets must be separate; all implementations use exactly the same token IDs, context length, chat template and truncation rules.

### 3.3 Coverage metrics

Four kinds of coverage are reported together, to avoid the misleading "a high dispatch count but the hot spots still on the host":

- Static coverage: the share of dispatch types in the compiled graph that can go to the accelerator.
- Dynamic coverage: accelerator dispatches actually run / total dispatches.
- Time coverage: accelerator execution time / end-to-end execution time, and the share of host-fallback wall clock.
- Work coverage: the share of estimated MACs and weight/activation bytes carried by the accelerator.

The fallback list contains at least: dispatch name, operator category, shape, call count, cumulative time, share, fallback reason.

## 4. Work package WP0: baseline freeze and documentation sync

**Status (2026-10-10)**: documentation sync done and pushed (`b42fec6`): root README, `compiler/README.md`, `kv260_upgrade_plan.md` (K2b now states D = 16 + streaming GEMV with 2 / 4 ports, no longer premised on D = 32), `k2b_gemv_design.md`, `iree_compiler_plan.md`. The `summary.txt` of the board result directories records the commit, bitstream hash, overlay configuration and WNS. What remains is unifying the naming of test prompts / result directories.

### Implementation

- Record the `main` commit, the Vivado/Vitis version, the IREE version, the board firmware, the device tree and the u-dma-buf layout.
- Fix the test prompts, token IDs, context length and generation length.
- Define a uniform naming for result directories: `<commit>/<model>/<bitstream>/<mode>/`.
- Immediately fix the numbers in the root README that are still K1's.
- Update K2b in `kv260_upgrade_plan.md` to the 100 MHz streaming GEMV with 2/4 ports, removing the old description of D=32 as K2b's premise; fill in G4/G5 resources and status from the accepted results.

### Acceptance

- Every later table can be traced to a commit, a bitstream and run parameters.
- No document describes K2b as D=32 at 200–250 MHz and as a 100 MHz four-port GEMV at the same time.

## 5. Work package WP1: G5 four-port GEMV wrap-up

**Status (2026-10-10, `fb01909`)**: done except the perf_model calibration. Board REGRESSION PASS (including Qwen3), logits bit-exact with funcsim, tokens identical to EX / 2 ports; bwtest GEMV 57.44–57.95 B / cycle. decode (tok/s, EX → 2 ports → 4 ports): Qwen3 2.22 → 3.98 → **6.28** (predicted ×3.15, measured ×2.83), SmolLM2 7.71 → 12.58 → **17.04**, hfgen 7.62 → 12.73 → 17.19, stories 61.9 → 100.8 → 131.65; prefill unchanged. Solving from the 2-port / 4-port points, each step = a fixed part F + a GEMV part: **F about 67 ms (Qwen3, 42% of the 4-port step), about 38 ms (SmolLM2, 65%)**; the GEMV part already shrinks nearly linearly with the number of ports. Results in `k2b_gemv_design.md` §8.1 (G5-4) and `build/deploy_kv260_*_gemv{,_np2,_np4}/results`.

### Implementation

- Staging of Qwen3, SmolLM2, generic and stories with `--iree-sa-gemv-ports=4`.
- Keep the executables, bitstream identifiers and full run logs for EX and GEMV with 1/2/4 ports.
- Collect decode device cycles, DDR read bytes, command counts and per-stage times with `board_profile.py`.
- Feed the profile into `perf_model.py`, keeping predictions, measurements and errors.

### Acceptance

- The full regression passes, including Qwen3.
- Board logits bit-exact with funcsim; token sequences identical to the EX path.
- The four-port bandwidth test stays at about 57.95 B/cycle, with the actual range of variation recorded.
- Qwen3 decode measured and predicted values both in the table; the 6.5–7 token/s estimate is not written as a measurement.

### Output

- No separate closeout document: the results are already in `k2b_gemv_design.md` §8.1; the remaining perf_model calibration (prediction / measurement / error, and the categorized breakdown of the fixed part F) also goes there.
- The four rows EX, 1 port, 2 ports, 4 ports of the hardware ablation table (the board decode data for 1 port has not been retrieved; optional)

## 6. Work package WP2: compressed-tensors W8A8 importer

**Progress (2026-10-10)**: first version done, verified on the host.
- `compiler/frontend/ct_import.py`: checks `quantization_config` according to §6.1 (a rejection names the config group / layer, the field and the supported range); safetensors read per tensor by mmap (single file or shards); each quantized linear layer's `(q int8, s fp32)` is attached to the weight Parameter (`_sa_qs`), the weight itself being the dequantized `q * s` (for the fp32 reference and KV calibration); `ignore`d layers and non-Linear tensors are loaded as floats.
- `hf_generic.py`: `hf_model()` goes through the importer when it sees `quantization_config`; `rope_rewrite()` also permutes the rows and scales of `_sa_qs`; `quantize()` uses `_sa_qs` as is when present (not through `_quantize_rows`, -128 kept). The export of ordinary models is byte-identical to before the change (SmolLM2 truncated to 2 layers: same irpa / mlir).
- `synth_hf.py --ct VARIANT`: synthetic checkpoints (scale = amax / 127.5, [-128, 127], like llm-compressor's minmax); accepted ok / neg128 / bf16-scale, rejected group / static-act / asym / int4 / fp8 / missing-scale / bad-shape / kv-scheme.
- `compiler/tests/test_ct_import.py [--e2e]` (`run_tests.sh ctimport`): all 8 rejections report the right field; for the 3 accepted ones the int8 and scales of the 14 linear layers are byte-identical to the file (q / k permuted for RoPE), -128 kept, the embedding and classifier share one table, dequantized weights = q × s; mutations (`quantize()` not using `_sa_qs`, `rope_rewrite` not permuting `_sa_qs`) are all caught. `--e2e` (D = 16, 4 GEMV ports): the Llama-shaped neg128 / bf16-scale export, compile (everything on the accelerator), and sim match torch W8A8 step by step, teacher-forced 40/40 identical to torch. Host memory peak 436 MB (small models).
- `dispatch_check.py --full-i8`: int8 data in [-128, 127] (the default remains [-127, 127], the range of the frontend's own quantization).

**Problems found in this step** (none introduced by the importer):
1. **The embedding's row gather goes through VE**, whose int8 output saturates to ±127: a -128 in the table becomes -127 (`elementwise_broadcast_*_i64xi8` in `dispatch_check --full-i8`). Self-quantized tables have no -128; compressed-tensors only quantizes Linear layers and the embedding is quantized by this project. This only arises when "the embedding shares a table with a quantized (not `ignore`d) classifier", which `quantize()` now explicitly rejects. Root fix: do the gather with LD (dynamic address) instead of a VE copy; evaluated separately.
2. **The generic frontend's form of Qwen3 (solved, 2026-10-10)**: with real Qwen3-0.6B (truncated to 2 layers) through the generic frontend, decode was already entirely on the accelerator and the fallbacks were all in prefill: QK-norm (`reduction_16x16x128` / `16x8x128`), Q's RoPE and transpose, the RoPE table gather (`elementwise_broadcast_32768`), K's permutation plus quantization, dequantization of the attention output (`elementwise_16x2048`), all tensors of M × heads × head_dim (32768 fp32 = 2048 words, more than one ACC bank's 1536 words), and HF's `transpose(1, 2)` after the norm fused into the same dispatch, which the existing row tiling does not accept. Two fixes:
   - **Frontend** (`hf_generic._attention_rows`): for prefill (M > 1) the attention module's forward runs row by row, the same structure as qhf.prefill: the q / k / v projections are computed for the M rows together (integer, identical to row by row), then the module's **original** forward is called for each row (projections replaced by that row's results, o_proj temporarily removed; QK-norm, RoPE, KV write and sa attention exactly as in one decode step), and finally o_proj is computed for the M rows together. HF's attention code is reused as is, generic for Llama / Qwen3 / Gemma, and prefill and decode are bit-exact by construction. decode (M = 1) is unchanged.
   - **Compiler** (`SaCodegen`): when lowering reports `ACC full`, retry with `tileShrink` = 1, 2, 3 (sahl-tile plans tiles by ACC / 2^n); dispatches that compile the first time are unchanged (golden corpus byte-identical). `elementwise_16x2048` thereby moves to the accelerator. In addition, the report only takes the first non-empty error diagnostic (with multi-threaded compilation it used to print "failed ()").
   Result: Qwen3 (2 layers) prefill + decode with 0 host fallbacks, prefill bit-exact with decode-only 5/5, teacher-forced 22/40 (torch W8A8 21/40); SmolLM2 (30 layers) decode numbers identical to before the change, prefill 5/5. Cost: the exported graph grows (SmolLM2's mlir 18.7 → 35.4 MB, export 7.5 minutes / 3.0 GB, compile 21 minutes / 4.5 GB), and the board prefill speed is to be measured (each row adds the attention dispatches). Remaining: for models with head_dim = 64 and QK-norm (only synthetic ones), the 256-byte norm weights are inlined by IREE as constants inside the dispatch (`bufferization.to_buffer` does not support it); the target models (Qwen3 128, Gemma3 256, Llama without QK-norm) are unaffected.
3. **Bit-exactness between prefill + decode and decode-only depended on the data (located and fixed, 2026-10-10)**: Llama-shaped random-weight models (self-quantized, not through the importer) with seeds 1–3 gave 0/5, 4/5, 4/5. How it was located: comparing DDR at the end of the sim → only layer 1's K / V differ by ±1 at positions 18, 28, 36 (layer 0 identical); torch eager's prefill and decode KV are identical, and the sim's decode matches eager → the sim's prefill deviates; the per-dispatch checks pass for every T / dirty, leaving only 3 UNSUPPORTED, i.e. **dispatches that fall back to the host**, among them `prefill:161`, layer 0's gate projection fused with SwiGLU (`silu(g)·u`, with exp and division). Its code generation failed (`sahl-to-sahw: ACC full`) and it was computed on the host with libm's exp / division, while the same operation in decode runs on the device SFU (approximate exp / RECIP); they occasionally differ by one rounding unit, which layer 1's activation quantization turns into a KV ±1. Root cause: the block size of prefill's row-wise linear kernel (`linearSchedule`, `chunk_tiles`) was chosen only by SPAD_B, 16 tiles for K = 256, while each ACC bank has only 1536 words above the scratch, buffers do not cross banks, and the scope has to hold 15 buffers of D × nc words (accumulation, result, up's int32 input, one temporary per epilogue step). Real models have K ≥ 576 with small blocks and never hit it. Fix: the block size is also capped by the space available per bank and the number of buffers (here 16 → 12); the golden corpus of 4232 dispatches is byte-identical; seeds 1–3 and neg128 are all 5/5, and `test_ct_import --e2e` gets `--prefill` back. Remaining: in such synthetic shapes 2 prefill dispatches are still on the host (`elementwise_broadcast_*`: fixed-index gathers other than the pair swap; `elementwise_transpose_4x16x64`: more than two loop levels), all data movement and multiply-add, bit-exact for now, but if the host's multiply-add were contracted into an FMA the same kind of problem would reappear: listed as a follow-up; also, the whole-module `--iree-sa-codegen-report` only prints "failed ()" for such failures (the diagnostic is not passed back with multi-threaded compilation), and the reason only shows when compiling that dispatch alone: to be fixed.
   **Lesson**: a host-fallback (VMVX) dispatch containing exp / division or other operations approximated by the SFU is not bit-exact with the same operation on the device; if either prefill or decode falls back, their equivalence breaks. The fallback list (WP9) should flag such dispatches.

### 6.1 Accepted formats

The first version accepts only:

- `format: int-quantized`, state compressed.
- Weights: int8, static, per output channel, symmetric, no zero point, `group_size = null`.
- Activations: int8, dynamic, per token, symmetric.
- The `ignore` list may contain unquantized layers such as `lm_head`.
- scales may be bf16/fp16/fp32, converted to fp32 on load, without changing the int8 weights.

Any other format must produce an explicit error naming the layer, the actual field and the supported range; no silent fallback or requantization.

### 6.2 Memory-safe data structures

Keeping int8 weights, scales and a `q * s` fp32 copy of the whole model at the same time for long is forbidden. Implement an external quantized linear layer, e.g.:

```python
QLinearExternal(
    qweight: Tensor[int8],
    scale: Tensor[fp32],
    bias: Optional[Tensor],
    source_name: str,
)
```

Requirements:

- safetensors loaded with mmap / lazily; read and released layer by layer.
- The compile archive consumes `(qweight, scale)` directly, not through `_quantize_rows`.
- The eager reference dequantizes temporarily only when the current layer or operator needs it, then releases it.
- When `rope_rewrite` permutes the rows of the Q/K projections, it permutes the qweight rows and the matching scales together, without creating a full fp32 weight copy.
- Calibration logic that cannot handle quantized weights directly gets a per-layer temporary dequantization interface.

### 6.3 Numerics and packing

- Imported int8 constants keep the full `[-128, 127]` range.
- Audit the packing, funcsim, oracle and DMA paths and remove any implicit assumption that "the minimum negative value is -127". Known places: the frontend's `hf_generic._to_i8` / `_quantize_rows` and `qllama`'s conversion all clamp to ±127 (the self-quantization convention; the import path must not go through them); the hardware's int8 multiply (`sx8`) and funcsim itself have no problem with -128. Activation quantization is still done by the device and stays ±127.
- Keep the importer's semantics apart from this project's self-quantization semantics; the import path does not call `_to_i8` or `_quantize_rows` to modify existing integers.
- SmoothQuant needs nothing from the importer: llm-compressor has already folded the smoothing factors into the norm weights (norm→QKV, norm→gate/up) and up_proj (→down_proj); the weights in the checkpoint are final values and are read as is.

### 6.4 Model packing and ignore

- Layers in `ignore` go through the existing path, but the manifest must record whether they are "original bf16/fp16" or "requantized by this project".
- A tied embedding / lm_head keeps a single device-side int8 table, and it is verified that the importer did not untie the weight sharing.
- The manifest records the source model, revision, configuration hash, safetensors SHA256, license and conversion parameters.

### 6.5 Tests

Generate at least the following cases with `synth_hf.py`:

1. Normal channel-wise W8A8.
2. Weights containing -128.
3. bf16 scales.
4. `ignore: [lm_head]` with a tied embedding/lm_head.
5. Per-group weight scales: must be rejected.
6. Static activation scales: must be rejected.
7. Asymmetric / zero point: must be rejected.
8. int4 and FP8: must be rejected.
9. Missing weight tensor or scale, mismatched shapes: must be rejected.

### Acceptance

- After import, the integer weights of every supported linear layer are byte-identical to the safetensors.
- Scales converted to fp32 equal the values in the original file.
- Synthetic models are bit-exact through the archive, funcsim and dispatch_check.
- The importer's peak host memory is measured and does not grow by orders of magnitude from building a whole-model fp32 copy.

## 7. Work package WP3: SmolLM2-135M validation

**Progress (2026-10-10)**: host and sim done, board pending.
- **Generation**: `compiler/frontend/llmc_w8a8.py`, run in a separate environment `build/llmc_venv` (CPU torch 2.14.0, llm-compressor 0.14.0, compressed-tensors 0.19.0; the export environment is untouched): SmoothQuant 0.8 + GPTQ (W8A8, `ignore: [lm_head]`), calibrated on 256 passages × 512 tokens of 200+ characters from WikiText-2 train (sampled with `default_rng(0)`, line numbers recorded in `sa_manifest.json`), 7.6 minutes. GPTQ's `actorder: static` (with per-channel quantization only GPTQ's column order, already folded back into the weights, no g_idx stored) is now accepted.
- **Quality** (`compiler/tests/quality_eval.py`: WikiText-2 test, 2048 tokens, window 256, teacher-forced, reference the original fp32; results in `build/quality/smollm2_w8a8_2048.json`):

  | Variant | PPL | top-1 (vs fp32) | corr | KL |
  |---|---|---|---|---|
  | fp32 | 26.98 | — | — | — |
  | Self-quantized W8A8 (fp32 attention) | 29.92 | 87.6% | 0.9899 | 0.102 |
  | llm-compressor W8A8 (SmoothQuant + GPTQ) | 28.16 | 90.3% | 0.9932 | 0.050 |
  | Same, GPTQ only (no SmoothQuant) | 29.83 | 87.9% | 0.9903 | 0.106 |
  | Self-quantized, SA form (int8 KV, device attention, SFU) | 30.77 | 83.9% | 0.9853 | 0.138 |
  | **llm-compressor W8A8, SA form** | **28.77** | **87.1%** | 0.9887 | **0.074** |

  Under the device's way of computing, the external checkpoint has about half the KL of self-quantization and a PPL 2.0 lower; nearly all of the improvement comes from SmoothQuant (GPTQ alone is on par with self-quantization). The extra loss of the SA form (int8 KV, attention probabilities, SFU) is similar for both (KL +0.024 / +0.036). A single 256-token window once gave the opposite conclusion: evaluation needs at least thousands of positions (§3.2).
- **KV scales (2026-10-12, WP4's approach applied to SmolLM2)**: same checkpoint, KV calibrated on the first 1024 tokens of WikiText-2 train (`build/quality/smollm2_kvch_2048.json`): ct-sa (per layer) PPL 28.86, top-1 86.7%, KL 0.074; **ct-sa:kch 28.82, 87.7%, 0.072**; kch+vch 28.83, 86.6%, 0.076. SmolLM2's K has no channel outliers like Qwen3's; the three differ within noise (the original CALIB per layer was 28.77 / 87.1%). For consistency with Qwen3 / Llama, whole-model builds use `--kv-channel k`.
- **Whole model + K per channel + actual length (2026-10-12, sim PASS)**: `test_hf_generic.py --model build/llm_cache/SmolLM2-135M-W8A8 --out build/hfgen/smollm2_kch_dyn --quant --rope --attn --prefill 16 --prefill-all --kv-channel k --kv-calib build/llm_cache/wikitext-2-raw-v1-train.txt:1024` (SA_D=16, 4 GEMV ports): export 812 s, compile 700 s, 1220 executables all on the accelerator; prefill vs decode-only 5/5 and prefill_all 40/40 bit-identical; teacher forcing over 40 positions device vs fp32 34/40 (torch W8A8 35/40), device vs torch mean correlation 0.9957, argmax differs at 6, passing the criterion for quantized models. The free-generation comparison over 9 steps is only 2/9 (minimum correlation 0.47: once a generated token differs, all later inputs differ; the old build with static T and per-layer KV was 9/9, 0.9995). Localization: 2 layers + kch match torch to 1.5e-5 (9/9, TF 0 differences), 2 layers with per-layer KV match too; the whole model's per-position correlation is 0.9999 at position 0 and then smoothly 0.983–0.998, with no jump at the T boundaries 16 / 32: it is the accumulation of int8 rounding over 30 layers (device vs torch order of operations), not an error; K per-channel scales enlarge q, making it more sensitive to rounding. Real quality is judged by board teacher forcing over 2048 positions.
- **On the board (2026-10-12, PASS, `build/deploy_hfgen_smollm2_kch_dyn`)**: 17/17 steps of logits bit-exact with the sim, identical tokens, 0 host dispatches. board_llm: prefill 45.34 tok/s (40 tokens, including loading), decode 21.90 tok/s (45.7 ms/step). Batched (positions 32–40): decode 45.1 ms/step (**22.17 tok/s**), of which device 38.64 ms and rest of the runtime 6.45 ms; prefill 266.3 ms per 16-token chunk (**60.09 tok/s**, 26.6M cycles, 7 lists); device memory peak 134.7 / 149.9 MB. Per dispatch (prof3.csv → stats_decode.json): 70.8% of decode device time is GEMV (28.6 ms, LD 135.7 MB, 47.4 B/cycle, 82% of the four-port peak 57.95; lm_head alone 54.6 B/cycle, 94%; the 576-wide small matrices inside the layers are less efficient), reduction 10.0%, attention (batch_matmul) 9.8%; in a prefill chunk matmul 47.8%, reduction 20.7%, row attention 18.3%. Time / MAC coverage both 100%. Board teacher forcing (8 × 256, WikiText-2 test): **PPL 28.545 (fp32 26.978), top-1 86.5%, KL 0.055**; host ct-sa:kch 28.82 / 87.7% / 0.072 (the host uses torch's order of computation, the board the device's actual rounding; the two are within noise).
  Compared with the WP3 acceptance (static T, per-layer KV, `results_wp3b`): decode 17.12 → 22.17 tok/s (×1.29), prefill (both as measured by board_llm) 27.18 → 45.34 tok/s (×1.67); presumably mainly from attention over the actual length and the V projection on GEMV (not separated).
- **qsum rebuild (2026-10-13, `build/deploy_hfgen_smollm2_qs`, §12.4)**: 1221 executables all on the accelerator; prefill vs decode-only 5/5 and prefill_all 40/40 bit-identical, qemu 3/3. On the board 17/17 bit-exact with the sim, 0 host dispatches; batched decode 45.2 ms/step (22.12 tok/s, device 38.48, rest of the runtime 6.71 ms), prefill 264.5 ms per chunk (60.50 tok/s), peak 134.7 MB. Board teacher forcing **PPL 28.54 → 28.26, top-1 86.5% → 89.8%, KL 0.055 → 0.038**.
- **sim** (`test_hf_generic.py`, D = 16, 4 GEMV ports): export + compile of 1710 accelerator executables, decode 9/9 argmax identical to torch W8A8, prefill vs decode-only 5/5 bit-exact. The 40-position teacher-forced gate became statistical: with d positions where the device and the torch model's argmax differ, max(2, ⌈2√d⌉) is tolerated, and the mean correlation of their logits must also be ≥ 0.98. The original gate (−2) failed on this checkpoint: the d = 3 flips happened to all fall on the device's side (34 vs 37), while the device's mean correlation to torch, 0.9963, was higher than self-quantization's 0.9911 (d = 6, 3 on each side).
- **On the board**: `deploy_z1_freeze.sh hfgen_w8a8` (not in the default set; `HFGEN_MODEL` / `HFGEN_TAG` passed to `deploy_hfgen.sh`), with a matching entry in `board_regress.py`.
- **First time on the board (2026-10-11, 4 ports)**: hfgen_w8a8 PASS (17/17 bit-identical to the sim), **hfgen FAIL (0/17)**. Both had 1440 host dispatches: 16e5e5d's row-by-row prefill, on models without QK-norm (SmolLM2, Llama), fused each row's RoPE + int8 quantization into the scatter that writes KV (6-D, with a permuting gather, `more than two loops`), 480 per chunk computed on the host; ARM and x86 occasionally differ by one ulp on these operations, which int8 quantization turns into a KV ±1 (w8a8 happened not to flip). Staging's check only counted decode's fallbacks and missed them. Running the aarch64 `sa-llm-run` in qemu with the device still the x86 sim reproduced the board result (x86 3/3, aarch64 0/3).
  Fix: in row-by-row prefill each row's cos / sin are looked up in the RoPE table by that row's scalar position (as a decode step does) instead of slicing the whole block's table; each row's computation graph is then the same as decode's, RoPE + quantization is a separate 2-D dispatch (the compiler supports the pair-swap gather), and the scatter only moves data. SmolLM2 / Qwen3 (truncated to 2 layers) prefill with 0 host dispatches; the q / k / v projections are still computed for the M rows together. New `compiler/scripts/check_bundle_aarch64.sh`: compares the test bundle bit for bit with the aarch64 runtime in qemu (old bundles: hfgen 0/3, hfgen_w8a8 3/3, matching the board), run automatically at the end of `deploy_hfgen.sh`.
- **Board acceptance (2026-10-11, 4 ports, staging `3e6cc5d`, `results_wp3b`)**: REGRESSION PASS. hfgen (self-quantized) and hfgen_w8a8 (llm-compressor W8A8) are both 17/17 steps bit-identical to the sim, identical tokens, 0 host dispatches (1440 the previous time); 2206 executables compiled, all on the accelerator, staging's aarch64 check 3/3. Speed: **prefill 27.18 tok/s** (14.20 before row attention, ×1.91; descriptor lists per run 1465 → 37), decode 17.12 tok/s (unchanged; both weights have the same amount of computation).
  **Why prefill got faster**: not because the accelerator computes faster (board_profile's prefill device cycles 20.9M → 20.0M), but because the host fallbacks are gone. Before the change batched prefill had 60 host dispatches per chunk (2 per layer: tensors of M × heads × head_dim that do not fit in ACC or have unsupported shapes, like the RoPE gather and Q transpose seen on Qwen3), 180 in total; at each one the runtime has to submit the queued commands and wait for the accelerator to finish, interpret on the ARM with VMVX, and start a new list (211 lists). After row attention there are 0 host dispatches and a prefill chunk is one or two consecutive lists (37 in total); what is saved is the synchronization round trips and the interpretation on the ARM. (The first board run with 1440 host scatters reached 19.96 tok/s: they came in a row and cut fewer lists.)

### Implementation

- Generate the W8A8 checkpoint with a fixed llm-compressor recipe, freezing the recipe, the calibration sample IDs and the software versions.
- Complete checkpoint reference, importer eager, SA oracle, funcsim and board validation in order.
- A/B against the existing self-quantization path with the same prompts/token IDs.

### Acceptance

- Every dispatch bit-exact with the SA oracle, the board bit-exact with funcsim.
- 40-position top-1 no lower than the existing self-quantized result, as a fast gate.
- Also record top-1, correlation/KL and PPL over at least thousands of tokens; if not better than self-quantization, this does not block merging the importer, but where the loss occurs between the reference layers must be located.

## 8. Work package WP4: Qwen3-0.6B quality re-measurement and attribution

### Input

First evaluate a ready-made compressed-tensors W8A8 checkpoint (first check whether a public W8A8 version of Qwen3-0.6B exists; if not, generate one with llm-compressor using WP3's fixed recipe); keep the current self-quantized model as a control. Pin the revision and file hashes when downloading; weights are not checked into the repository.

### Four-layer quality comparison

| Layer | Purpose |
|---|---|
| Original bf16/fp32 HF | Final quality reference |
| Standard runtime of the external W8A8 checkpoint | Measures the checkpoint's own quantization loss |
| importer eager | Checks the importer, RoPE rewrite, weight sharing and tokenizer |
| SA end to end | Measures the extra loss of KV, attention probabilities, approximate functions and device rounding |

### Decision gates

- If the external W8A8 checkpoint is clearly better than today's 25/40 and the importer eager is close to it, make the external weights the default demo path.
- If the standard runtime's quality is good but the importer eager drops clearly, fix the importer / tokenizer / rewrites first.
- If the importer eager is good but SA drops clearly, switch items to high precision one at a time to locate the loss: KV cache, QK logits, softmax/exp/recip, activation quantization, final classifier.
- Only once KV is confirmed as the main cause, evaluate per-head scales; no hardware or format complexity is added before the loss is located.

### Acceptance

- Output the four-layer quality table and the error-attribution conclusion.
- The data and scripts for 40-position top-1, formal top-1, correlation/KL and PPL are reproducible.

### Results (2026-10-11)

Qwen3-0.6B, the first 2048 tokens of WikiText-2 test, window 256; the checkpoint is a W8A8 (ct) generated with llm-compressor. top-1/corr/KL are all relative to fp32's per-position logits. To reproduce:

    compiler/tests/quality_eval.py --model build/llm_cache/Qwen3-0.6B --ct build/llm_cache/Qwen3-0.6B-W8A8 --tokens 2048 --window 256 \
        --variants fp32,w8a8,ct-w8a8,sa:act,ct-sa:act --out build/quality/qwen3_w8a8_2048.json
    same with --variants fp32,ct-sa,ct-sa:act+kv,ct-sa:act+q,ct-sa:act+probs,ct-sa:act+sfu,ct-sa:act+kv+q+probs+sfu --out build/quality/qwen3_attr.json

Four-layer quality table:

| Layer | Variant | PPL | top-1 | corr | KL |
|---|---|---|---|---|---|
| Original | fp32 | 39.21 | — | — | — |
| Checkpoint standard semantics | ct-w8a8 | 40.05 | 92.3% | 0.9957 | 0.022 |
| Self-quantized (control) | w8a8 | 41.05 | 89.1% | 0.9928 | 0.039 |
| SA end to end (ct weights) | ct-sa | 53.50 | 71.5% | 0.9485 | 0.411 |

The importer is covered by test_ct_import; the ct weights are about 3 points better than self-quantization and become the default demo weights.

Attribution (`act+X`: linear-layer activations and X kept in fp32, the rest computed as SA; baseline ct-sa:act):

| Variant | PPL | top-1 | KL |
|---|---|---|---|
| ct-sa:act | 52.91 | 74.3% | 0.394 |
| + kv | **38.60** | **91.7%** | **0.025** |
| + q | 53.30 | 73.4% | 0.391 |
| + probs | 53.41 | 75.2% | 0.382 |
| + sfu | 53.00 | 73.6% | 0.396 |
| + kv+q+probs+sfu | 39.27 | 97.9% | 0.0016 |

Conclusions:

- Nearly all of SA's extra loss comes from the int8 KV cache: switching KV alone back to fp32 brings top-1 from 74.3% back to 91.7% and lowers KL about 15×. q, probs and the SFU approximation, switched alone, are each within ±1 point.
- Activation quantization of the linear layers (ct-sa → ct-sa:act) accounts for only about 3 points.
- With KV fixed, q+probs+sfu together are about 6 points (91.7% → 97.9%), a secondary item.
- Per the gates, next evaluate per-head (or K per-channel) KV scales and re-measure ct-sa; no hardware or data-format changes before that.

KV scale improvement (no hardware or data-format change): K per channel (`kch`, scales folded into q, the int8 dot product unchanged), V per channel (`vch`, the PV result scaled per output channel), KV calibrated on the first 1024 tokens of WikiText-2 **train** (in max_len windows, taking the maximum over windows; previously the ~45-token `export_hf.CALIB`). To reproduce:

    compiler/tests/quality_eval.py --model build/llm_cache/Qwen3-0.6B --ct build/llm_cache/Qwen3-0.6B-W8A8 --tokens 2048 --window 256 \
        --variants fp32,ct-sa:kch,ct-sa:kch+vch --kv-calib build/llm_cache/wikitext-2-raw-v1-train.txt:1024 --out build/quality/qwen3_kvch_2048.json

| Variant | Calibration | tokens | PPL | top-1 | corr | KL |
|---|---|---|---|---|---|---|
| ct-sa (per layer) | CALIB | 2048 | 53.50 | 71.5% | 0.9485 | 0.411 |
| ct-sa (per layer) | train 1024 | 512 | 44.15* | 68.8% | 0.9306 | 0.500 |
| ct-sa:kch | CALIB | 512 | 33.32* | 81.1% | 0.9797 | 0.129 |
| ct-sa:kch | train 1024 | 2048 | **39.14** | **87.4%** | 0.9907 | 0.054 |
| ct-sa:kch+vch | train 1024 | 2048 | 39.12 | 87.1% | 0.9919 | **0.046** |
| (reference) ct-sa:act+kv | — | 512 | 30.56* | 92.0% | 0.9959 | 0.021 |

\* The fp32 PPL over 512 tokens is 31.16 (39.21 over 2048); PPL can only be compared at the same token count. 512-token data in build/quality/qwen3_kvch_512.json, qwen3_kvcal_512.json.

Conclusions:

- The main cause is K's channel outliers (per-channel max/median 7–120×). More calibration data with per-layer scales does not help (68.8%); only K per channel together with sufficient calibration works: top-1 71.5% → 87.4%, KL 7.6× lower, PPL on par with fp32.
- V per channel over 2048 tokens is basically the same as K only (KL slightly better, top-1 slightly worse). **The default is `--kv-channel k`**: no extra per-channel multiply on the V side.
- About 5 points remain to the checkpoint itself (ct-w8a8 92.3%), of which about 4 are the remaining KV quantization error (compare act+kv at 92.0%). Alternative: percentile-clipping calibration.

**On the board (2026-10-12, PASS)**: Qwen3-0.6B-W8A8, all 28 layers, generic frontend, `--kv-channel k --kv-calib build/llm_cache/wikitext-2-raw-v1-train.txt:1024`, four-port overlay d16_100mhz_w128_p6_gemv_np4:

    SA_D=16 SA_GEMV_PORTS=4 compiler/tests/test_hf_generic.py --model build/llm_cache/Qwen3-0.6B-W8A8 --out build/hfgen/qwen3_kch \
        --quant --rope --attn --prefill 16 --kv-channel k --kv-calib build/llm_cache/wikitext-2-raw-v1-train.txt:1024 \
        --board-bundle build/hfgen/qwen3_kch_bundle                                         # (MemoryMax=10G)
    HFGEN_TAG=qwen3_kch HFGEN_BUNDLE=build/hfgen/qwen3_kch_bundle compiler/scripts/deploy_hfgen.sh   # -> build/deploy_hfgen_qwen3_kch

- 1138 executables compiled, all on the accelerator, host 0; sim decode 9/9 argmax identical to torch W8A8, prefill vs decode-only 5/5 bit-exact; qemu aarch64 3/3.
- On the board: 17/17 steps of logits bit-exact with the sim, 0 host dispatches; **prefill 9.33 tok/s, decode 4.73 tok/s (211 ms/step)**, window 613 MB.
- **Later (2026-10-12, §13.1)**: with decode's v projection moved to GEMV (`14f5612`), the same configuration decodes at **5.60 tok/s (178.4 ms/step)**; board teacher-forcing quality PPL 39.62, top-1 86.9% (host 39.14 / 87.4%).
- **qsum rebuild (2026-10-13, `build/deploy_hfgen_qwen3_qs`, §12.4)**: attention probabilities normalized by the quantized sum, 1139 executables all on the accelerator, 17/17 bit-exact on the board; teacher forcing PPL 39.62 → 39.68 (flat), top-1 86.9% → 88.1%, KL 0.053 → 0.038; decode 6.96 tok/s (143.6 ms/step), prefill 20.79 tok/s.
- Not directly comparable with WP1's 6.28 tok/s: that was the hand-written export path (export_hf), this is the generic frontend (row attention etc.) plus K per channel; how much each contributes has not been separated (that would need a whole-model Qwen3 board point with the generic frontend and per-layer KV).
- Note: `test_hf_generic.py` reads D only from `SA_D` (default 8), not derived from `SA_KV260_CONFIG`; the first attempt compiled for D = 8, the sim passed too, and the qemu check caught it.
- `kv_cache_scheme` in llm-compressor 0.14 / compressed-tensors 0.19 can produce static int8 KV scales (observed after RoPE), but only supports the `tensor` and `attn_head` granularities; `channel` explicitly errors for attention. So it can be one input source for the importer, but cannot replace K per-channel calibration.

## 9. Work package WP5: Llama-3.2-1B frontend and memory preflight

### 9.1 Model constraints

Following the actual `config.json`, the expected structure is 16 layers, hidden size 2048, FFN 8192, 32/8 GQA, head_dim 64, vocab 128256, tied embedding, and Llama 3 RoPE scaling. The public W8A8 version (RedHatAI) is the **Instruct** model: the quality reference is the Instruct bf16, with inputs built using the chat template; such checkpoints usually have `ignore: [lm_head]`, and the embedding / classifier table is quantized by this project (recorded in the manifest).

**Preparation (2026-10-11)**:
- **checkpoint**: `RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8` (revision `9c406c48`, no login needed) downloaded to `build/llm_cache/Llama-3.2-1B-Instruct-W8A8` (1.9 GB). The configuration is exactly within the importer's scope (int8 per-channel symmetric, activations dynamic per token, `ignore: [lm_head]`, no act-order); the recipe is SmoothQuant 0.7 + GPTQ (observer mse). The file stores the tied `lm_head.weight` **a second time** (bit-identical to the embedding): the importer now checks that the two are identical and rejects otherwise (it used to write whichever was read last into the shared tensor, silently replacing the other); covered by the synthetic variants `lm-head-dup` / `lm-head-diff`.
- **Original reference**: `meta-llama/Llama-3.2-1B-Instruct` needs a login and accepting a license; the unrestricted mirror `unsloth/Llama-3.2-1B-Instruct` (revision `5a8abab4`, `build/llm_cache/Llama-3.2-1B-Instruct`, 2.4 GB) is used. Check: its embedding table is bit-identical to RedHatAI's; for SmoothQuant-invariant quantities (weights × the preceding RMSNorm's weights), RedHatAI's dequantized values correlate 0.99987 with it, with a median error of 0.26 quantization steps: the same base model.
- **Tokenizer**: `hf_tokenizer.py` used to approximate pre-tokenization with one fixed regex; for Llama 3 only 114/305 segments matched transformers (Llama 3 groups digits by 3). It now follows tokenizer.json's normalizer (NFC), pre-tokenizer (Split regex / Digits / ByteLevel's GPT-2 regex) and `ignore_merges`, matching special tokens whole first: Llama 3, SmolLM2 and Qwen3 are **exactly identical** to transformers on 1009 segments each (including chat-template special tokens, Chinese, full-width characters, a 20,000-character text). Needs the `regex` module (Unicode categories); without it (the board's system Python: `sudo apt install python3-regex`) it falls back to the old approximation. The tokenization of SmolLM2 / Qwen3's calibration text and prompts is unchanged, so existing models' KV scales are unaffected; Llama's calibration text goes from 47 to 45 tokens.
- **RoPE**: `RopeTable` generates the whole cos / sin table from HF's rotary module, naturally including llama3 scaling; `test_hf_generic`'s before/after-rewrite comparison verifies it on the real model.
- **Memory preflight**: `compiler/tests/mem_preflight.py` (reads only the safetensors headers). Calibrated against board measurements: peak ≈ packed parameters + KV + a few MiB; the sum of the items is 2.0% (Qwen3 570.5 vs 582.4 MiB) / 1.5% (SmolLM2) below the packed file, so the estimate is multiplied by 1.025. Llama-3.2-1B: int8 linear layers 928.0 + int8 embedding 250.5 + scales etc. ≈ 1210 MiB packed, KV for T = 256 4 MiB, margin 32 MiB, total **1246 MiB / 1280 MiB: it fits (34 MiB to spare)**; the longest KV is T ≈ 2400. Note: staging currently sets the runtime heap to "irpa × 1.1 + 24" (1323 MiB for Llama, more than the pool); before going on the board change it to "packed + KV + margin".

### 9.2 Frontend implementation

- The importer first strictly validates that the compressed-tensors configuration is within WP2's supported scope.
- `rope_rewrite` generates the frequency table with the model's RotaryEmbedding, and compares eager results before and after the rewrite.
- Llama 3 tokenizer special tokens and the chat template are compared one by one with Transformers; if the local tokenizer is still incomplete before the talk, the host may generate token IDs with `tokenizers`.
- Embedding gather gets boundary and packing tests; a tied classifier/embedding keeps a single copy in the device archive.

### 9.3 Byte-level memory preflight

Before starting the full sim, produce a memory manifest counting item by item:

- Each layer's int8 weights and scales.
- Whether embedding/classifier are shared, and whether a bf16 copy remains.
- Constant area, descriptors, workspaces and alignment waste.
- Growth of the KV cache with context length.
- The peak lifetimes of host staging buffers and temporary device buffers.

Note: the original checkpoint file is about 2 GB, which is not the device archive size; the estimate of about 1.24 GB / 1178 MiB only holds once the tied embedding/classifier is requantized and deduplicated.

**Estimate** (orders of magnitude before the manifest): int8 weights = embedding 2048 × 128256 (262.7M) + 16 layers × 60.8M (Q / O 2 × 4.19M, K / V 2 × 1.05M, gate / up / down 3 × 16.8M) ≈ 1236 MB = 1178 MiB; KV cache per token 16 layers × 8 heads × 64 × 2 = 16 KB, 8 MiB at T = 512; pool 1280 MiB, margin about 90 MiB (Qwen3's peak 595 MB / heap 640 MB, workspaces and scales are small): **it very likely fits**. The host side needs a preflight too: outside the pool and the global CMA, Linux has about 2.4 GB left, and the host memory peak while loading the 1.2 GB irpa counts toward the gate. decode estimate: the GEMV part scaled by bytes from Qwen3 (about 92 ms / 607.5 MB at 4 ports) is about 190 ms; adding the fixed part, about **4 tok/s** (lower with down_proj on EX).

Memory gate:

- Going on the board is allowed only if `archive_bytes + peak_workspace + kv_bytes(T) + safety_margin ≤ pool_bytes`.
- A safety margin of at least 32–64 MiB is recommended, fixed after measurement.
- "Enlarging the 1280 MiB pool" is not the default solution; the existing CMA, initrd and low-2 GB address space must first be proven conflict free.
- If even a short context cannot meet the gate, the high 2 GB of DDR (K5-M M2) becomes a prerequisite of Llama's board acceptance.

### 9.4 EX correctness baseline

**Already available**: when K does not fit, splitting by K on EX is work completed in C6.3 (`iree_compiler_plan.md` §8.9; a synthetic model with K = 16384 bit-exact, Llama's W2 executed in 2 blocks); the compiler uses GEMV only when `K × 16 ≤ 65535` (`expandLinear` in `SahlExpandKernels.cpp`), so down_proj already goes to EX automatically. This step only needs to run through and record data.

- The K=8192 down_proj goes through the EX path (current state).
- Complete per-dispatch and end-to-end sim; record EX baseline cycles and token/s.
- If the memory gate passes, do board correctness acceptance.

**Result (2026-10-12, given by WP6's off / on ablation, §10)**: `--iree-sa-gemv-k-split=false` is the EX baseline (the K = 8192 down_proj split by K on EX, the other linear layers still on GEMV). Llama-3.2-1B whole model, four-port overlay: on the board 17/17 bit-exact with the sim, 0 host dispatches, logits identical to K split on; decode **716.3 ms / step (1.40 tok/s)**, down_proj 3.20M cycles each (5.3 B/cycle), the 16 layers together about 72% of decode device time. With the K split on the same layer takes 0.275M cycles (61 B/cycle), decode 4.03 tok/s (×2.89). The memory gate passed (§9.3, board peak 1190 / 1203 MB); the quality reference is the original Instruct model (§11).

### Acceptance

- tokenizer/token IDs identical to the reference implementation, or the host's standard tokenizer explicitly adopted.
- eager outputs before and after the RoPE rewrite agree within the agreed tolerance.
- The memory manifest gives a clear fits / does-not-fit conclusion.
- The EX path correct per dispatch, with the original bf16 model as the quality reference.

## 10. Work package WP6: GEMV K split

**Progress (2026-10-11)**: decode's K split done.
- Previously, when K exceeded one SPAD_B bank (4096 at D = 16), decode's linear micro-kernel was **not selected at all** (the schedule computed a block size of 0 from SPAD_B), the whole layer went to the generic contraction's K-split EX path, and GEMV was not used. Three changes, effective only when "GEMV is present and K exceeds the old limit": `KernelMatcher::gemvHolds` (K is limited only by where x is stored: packed x in SPAD_A, int32 x in a bank of the ACC scratch), `linearSchedule` (block size by the ACC scratch only), `expandLinear` (K split by 4080, x and weights taken as subviews, the first block writes the accumulator, the others set acc to accumulate; the epilogue reads after the last block).
- **Three 4-port GEMV cases added to the golden corpus** (`qwen3_gemv4`, `smollm2_gemv4`, `stories_gemv4`, the `_gemv_np4` board builds, 2814 dispatches), recorded with the compiler before the change; after the change the 16 cases' 7046 dispatches are byte-identical: deployed GEMV models are unchanged.
- **Synthetic Llama, K = 8192** (dim 256, FFN 8192, `run_tests.sh gemvk`): down_proj becomes 3 GEMVs (4080 + 4080 + 32), decode has 0 host dispatches, step-by-step argmax vs torch W8A8 9/9, teacher-forced 40/40 (correlation 1.0000), prefill vs decode-only 5/5 bit-exact.
- **Two kinds of host fallback remain in prefill** (correctness unaffected; board prefill speed affected; listed under WP6 / WP7): ① prefill's down_proj (`matmul_like_16x…x8192`; the row-wise EX path has no K split: "contraction not expanded"), integer arithmetic so still bit-exact, but computed on the ARM; ② prefill's KV-writing scatter (`scatter_…x32x2x1xi8`): in models without QK-norm (Llama) RoPE's swapneg gather is fused into the scatter that writes the cache ("more than two loops"); Qwen3 is separated by QK-norm and unaffected. The real Llama-3.2-1B meets both.

### 10.1 Splitting rule

The GEMV descriptor's `row_bytes` (w2[63:48]) is 16 bits and equals K × 16, so K ≤ 4095; K must also be a multiple of 16, and the x buffer is 4096 bytes per lane. So the safe limit is:

```text
K_BLOCK_MAX = floor(4095 / 16) * 16 = 4080
```

4096 must not be used. The K=8192 example splits as:

```text
4080 + 4080 + 32
```

The last block uses the existing GEMV tail semantics; if the hardware requires complete strips, the compiler pads explicitly, and padding must not cross the archive's boundaries.

### 10.2 Compiler implementation

The hardware and the lower layers are ready: `sahl.gemv` has an accumulate flag, the `SahlToSahw` lowering already sets the descriptor's `acc` bit (w3[58]), and funcsim implements the acc = 1 semantics; the work is only in `expandLinear` (today `useGemv = gemvPorts && K × d ≤ 65535`; otherwise the whole layer goes to EX).

- In `expandLinear`, check K, the x-buffer capacity, the descriptor fields and alignment requirements.
- For each K block compute the x/weight base offsets, the effective K and the number of strips.
- The first block `acc=0`, later blocks `acc=1`, accumulating in the same ACC rows.
- Only the last block does requantize, bias, activation and write-back, avoiding per-block rounding.
- Add a development switch `--iree-sa-gemv-k-split=0|1` for correctness comparison and ablation.
- The compile log prints the split, e.g. `K=8192 -> [4080, 4080, 32]`.
- **x residency**: the GEMV unit's x buffer judges residency by (xword, K). If the K blocks are the inner loop of the strip groups, each command switches x segment and re-reads about 255 words (down_proj has about 8160 cycles of data per command, about 3%). Optional: K blocks as the outer loop (the same K segment runs through all strip groups, partial sums stay in ACC, needing N × 4 bytes = 8 KB of ACC), or accept the 3%; implement the simple one first and let the profile decide.

### 10.3 Overflow and numeric semantics

Accumulation is int32 wrap-around with no saturation or rounding in between (the semantics of `k2b_gemv_design.md` §4), and |Σ| ≤ K × 128 × 128 = 8192 × 2¹⁴ = 2²⁷ ≪ 2³¹, so overflow is impossible for K ≤ 131071 and the order of addition does not matter. Requantize happens in the epilogue (after reading ACC); nothing is truncated at the end of each block. Tests only need:

- one worst case of all -128 × -128 (confirming sums of order 2²⁷ are correct).
- 4080, 4096, 4112, 8192 (multiples of 16) near the K-block boundaries.

**Results (2026-10-12, `compiler/tests/test_gemv_ksplit.py`, `run_tests.sh gemvks`)**: a single decode linear layer (N = 128) on the sim of the 4-port overlay, compared with numpy's exact sum; compiler switch `--iree-sa-gemv-k-split=true|false` (off: layers with K > 4080 go entirely to EX, i.e. the behavior before WP6, the ablation baseline).

| K | On (GEMV commands = groups × K blocks) | Off |
|---|---|---|
| 4080 | Bit-exact, 8 × 1 | Bit-exact, GEMV (no split needed) |
| 4096 | Bit-exact, 8 × 2 (4080 + 16) | Bit-exact, EX |
| 4112 | Bit-exact, 1 × 2 (4080 + 32) | Bit-exact, EX |
| 8192 | Bit-exact, 1 × 3 (4080 + 4080 + 32) | Bit-exact, EX |
| 8192 worst case | x = w = −128: 2²⁷, bit-exact | x = −127, w = −128, bit-exact |

- **Numeric convention**: activation int8 is [−127, 127] (the symmetric saturation of every frontend's `to_i8`). On the EX path x goes through VE's int8 conversion, which turns −128 into −127; the GEMV unit reads the packed bytes as they are, so −128 is exact too. So EX's worst case uses −127 × −128; inputs outside the convention (x = −128) are exact only on GEMV.
- **Board ablation (2026-10-12, Llama-3.2-1B, `deploy_hfgen_llama1b_dyn` / `_nosplit`, the same export and parameters, differing only in `--iree-sa-gemv-k-split`)**: both 17/17 bit-exact with the sim, 0 host dispatches, identical logits (the off build's qemu check passes directly against the on build's sim reference).

  | | K split off (down_proj on EX) | K split on (4080 + 4080 + 32) |
  |---|---:|---:|
  | down_proj per call (16 per step) | 3.20M cycles, 5.3 B/cycle | 0.275M cycles, 61 B/cycle (×11.6) |
  | decode device / per step | 711.8 ms / 716.3 ms (1.40 tok/s) | 243.6 ms / 248.1 ms (**4.03 tok/s, ×2.89**) |
  | prefill (batched, per 16-token chunk) | 1066 ms | 1066 ms (prefill's row form does not use GEMV; unaffected) |

- **Found along the way**: for K ≤ 4096, the chunk of a decode linear layer is still sized by the capacity of one SPAD_B bank (`sbank / K`: 1 tile per chunk at K = 4096, 4 per chunk for Qwen3's K = 1024), while GEMV streams the weights and does not use SPAD_B; sizing chunks by the ACC scratch could reduce the number of chunks per layer (to be evaluated).

### Acceptance

- Synthetic models with K>4080 bit-exact with the EX path.
- Llama's down_proj clearly shown as several GEMV descriptors, consistent per dispatch.
- Llama outputs identical with the K split on and off; the performance difference goes into the compiler ablation table.
- No behavior change on the existing K≤4080 golden corpus.

## 11. Work package WP7: full Llama acceptance

### Implementation

- Recompile Llama with the K split enabled.
- Complete the sim end to end; after the memory gate passes, run on the board with four ports.
- Collect cycles, DDR bytes, token/s and the dispatch breakdown before and after down_proj moves from EX to GEMV.

### Acceptance

- Per dispatch consistent with the SA oracle; the board bit-exact with the sim.
- Report decode, prefill, quality, peak memory and coverage.
- Performance numbers distinguish estimates, sim and board measurements.

### Results (2026-10-11, board PASS)

Llama-3.2-1B-Instruct-W8A8 (compressed-tensors, 16 layers, GQA 32/8, K=8192 down_proj), four-port overlay d16_100mhz_w128_p6_gemv_np4:

    compiler/tests/test_hf_generic.py --model build/llm_cache/Llama-3.2-1B-Instruct-W8A8 --out build/hfgen/llama1b \
        --quant --rope --attn --prefill 16 --mb 1400 --board-bundle build/hfgen/llama1b_bundle     # (MemoryMax=10G)
    HFGEN_TAG=llama1b HFGEN_BUNDLE=build/hfgen/llama1b_bundle compiler/scripts/deploy_hfgen.sh   # -> build/deploy_hfgen_llama1b

- Compile: 971 executables all on the accelerator, host 0; export 273 s, compile 253 s.
- Memory: mem_preflight estimates 1246.2 MiB / 1280 MiB pool; actual sa_packed.irpa 1182.5 MiB, window 1203 MB.
- sim: prefill (M=16, 40 tokens) vs decode-only logits bit-exact on 5/5 rows; teacher-forced 40 positions top-1 vs fp32 30/40 (torch W8A8 32/40, tolerance 5), device vs torch mean corr 0.9924.
- aarch64 (qemu) vs sim: 3/3 rows bit-exact.
- On the board: 17/17 steps of logits bit-exact with the sim, identical generated tokens, 0 host dispatches; **prefill 9.38 tok/s (40 tokens in 4264 ms, including loading in the first chunk), decode 3.50 tok/s (286 ms/step)**.
- decode agrees with the bandwidth model: about 1.18 GB of parameters / 57.95 B/cycle (100 MHz) ≈ 214 ms, plus about 70 ms of fixed overhead.
- **Later (2026-10-12, V projection on GEMV `14f5612` + attention over the actual length `1860823`)**: 17/17 bit-exact on the board, **decode 4.03 tok/s (248.1 ms/step, device 243.6 ms, GEMV 226 ms = 92.6%, 54.7 B/cycle)**, prefill 12.62 tok/s (`board_llm.py`, including loading in the first chunk; batched 1066 ms per chunk, 15.21 tok/s, matmul 73% of prefill). The K split on / off comparison is in §10.
- **K per-channel KV rebuild (2026-10-12/13, `build/hfgen/llama1b_kch` → `build/deploy_hfgen_llama1b_kch`)**: the original board build used per-layer KV calibrated on the ~45-token `CALIB`, inconsistent with WP4's approach for Qwen3 / SmolLM2; the rebuild uses `--kv-channel k --kv-calib build/llm_cache/wikitext-2-raw-v1-train.txt:1024` (export 330 s, 715 executables all on the accelerator, `sa_packed.irpa` 1240 MB). sim: teacher forcing over 40 positions top-1 agreement with fp32 35/40 (torch W8A8 also 35/40), prefill vs decode-only 5/5 rows bit-identical; qemu aarch64 3/3. Board PASS: 17/17 bit-exact, 0 host dispatches, board_llm decode 4.03 tok/s (248.4 ms/step), prefill 12.58 tok/s (including loading); batched decode 247.6 ms/step (**4.04 tok/s**, device 243.9 ms, rest of the runtime 3.6 ms), prefill 1068.9 ms per chunk (**14.97 tok/s**), device memory peak 1189.8 / 1202.9 MB; per dispatch: decode GEMV 226.5 ms (92.5%, 1238.8 MB, 54.7 B/cycle = 94%), prefill matmul 72.5%, array 200 MAC/cycle (78%). Speed identical to the per-layer build (only the KV scales changed). Quality (host, 2048 tokens, `build/quality/llama1b_2048_*.json`; the board's memory cannot fit prefill_all): fp32 PPL 22.77; **ct-sa:kch PPL 23.33, top-1 90.7%, correlation 0.991, KL 0.036**.
- The default prompt is SmolLM2 token ids, meaningless for Llama; the generated text is only used for bit-exact comparison and does not represent quality; quality is judged by the teacher-forced metrics.
- Test memory: only one fp32 model kept at a time (autograd off, gc.collect after del); peak 7.1 GB.

## 12. Work package WP8: Gemma-3-270M short-context bring-up

**Preparation (2026-10-11)**:
- **Model**: `google/gemma-3-270m` needs a login; the unrestricted mirror `unsloth/gemma-3-270m` is used (base, revision `edcf1fe3`, `build/llm_cache/gemma-3-270m`, 0.5 GB). Structure: 18 layers, hidden 640, 4 / 1 GQA, head_dim 256, vocabulary 262144, FFN 2048, GELU-tanh, sliding window 512 (5 of every 6 layers sliding), RoPE local base 1e4 and global 1e6, `query_pre_attn_scalar` 256, no logit softcap. Memory preflight: about 300 MiB at T = 512, plenty of room.
- **eager comparison item by item** (original HF fp32 as the reference, 24 tokens) found and fixed four issues, all effective only on Gemma-like structures: ① the wrapped decoder already differed from the original (correlation 0.73): HF infers positions from layer 0's cache length, Gemma's layer 0 is a sliding layer whose counter is `cumulative_length_int`, which was not set (now `_at` sets it too; at export the sa attention does not use HF's mask) → bit-exact level (7e-5); ② completely wrong after W8A8 (correlation 0.13): `Gemma3TextScaledWordEmbedding` multiplies by √640 and `QEmbedding` dropped it; `embed_scale` is now kept and applied after dequantization (it cannot be folded into the scale: the classifier shares the table without the multiply), and an embedding subclass with a custom forward but no `embed_scale` is explicitly rejected; ③ `RopeTable` errored: Gemma has a single rotary module called with `layer_type`; now one table per type; ④ the RMSNorm gain is `1 + w`: `SaRMSNorm` probes the module's actual gain with an all-ones vector, decides between `w` and `1 + w`, and rejects other forms. Also: the score scaling of `sa_attention` now uses the module's own `scaling` (when it equals 1/√head_dim the original algorithm is kept, so existing models are bit-identical); a cache longer than the sliding window is explicitly rejected (the device attention has no sliding window; the semantics are the same for T ≤ 512).
  Result: after wrapping and the RoPE / norm rewrites it matches the original (7e-5, argmax 24/24), W8A8 correlation 0.992, SA form (int8 KV, device attention, SFU approximation) correlation 0.9906, argmax 23/24. The exports of SmolLM2 and Qwen3 (truncated to 2 layers) are byte-identical to before the change.
- **Tokenizer**: Gemma uses SentencePiece-style BPE (the normalizer replaces spaces with `▁`, BPE runs directly on characters, byte fallback `<0xNN>`). `hf_tokenizer.py` gained this mode (normalizer Replace / Prepend, `MergedWithPrevious` Split, byte fallback and the matching decoding), and BPE moved to a heap + linked list (n log n; for Gemma a whole text is one "word"). Gemma, Llama 3, SmolLM2 and Qwen3 match transformers exactly on 1009 segments each, and decoding round-trips; Gemma does not need the `regex` module, so the board's system Python is exact too.
- **Not done yet** (WP8 formal stage): export and compile (the device form of GELU-tanh, the attention micro-kernel and SPAD capacity for head_dim 256), the Gemma3 structure in `synth_hf.py`.

### 12.1 Scope of the first stage

- Mainly on the sim; a few host fallbacks allowed.
- At first the tokenizer pre-generates token IDs with the standard `tokenizers` / Transformers, so SentencePiece/byte fallback does not block the compiler coverage study.
- Increase T step by step over 16/32/64/128, without assuming T=512 fits in the current SPAD.

### 12.2 Key adaptations

- Support the layer pattern of about 5 sliding-window layers + 1 full-attention layer; for T ≤ window the semantics are the same, but the local/global RoPE bases are still verified separately.
- Support 4 RMSNorms, the `1 + w` form, the embedding multiplied by `sqrt(d)`, and QK norm.
- GELU-tanh rewritten into a graph the device can execute, but the correctness reference must use the approximate semantics of the device's EXP/RECIP.
- Add tests for large positive/negative inputs, clamping, overflow and rounding of `x / (1 + exp(-2u))`, without claiming bit-exactness with PyTorch's GELU.
- Explicit SPAD capacity preflight in the planner for attention's K/V, temporary logits and workspaces.
- head_dim = 256 is twice Qwen3's (128): besides capacity, first check the head_dim assumptions and limits of the attention micro-kernel, QK-norm and the RoPE rewrite.

### 12.3 The T=512 risk

With head_dim=256 a single K tensor at T=512 is already about 131072 bytes, possibly more than the current capacity of one bank. Therefore:

- The first version only commits to the context lengths that actually pass the capacity check.
- If T=512 does not pass, the document explicitly marks C6.5 T tiling as a prerequisite for long contexts.
- "Local and global layers use the same algorithm for T≤512" is not taken to imply "the existing SPAD can certainly hold it".

### Acceptance

- At least one short context completes end to end on the sim.
- A complete fallback list and the four kinds of coverage are produced.
- Every operator that cannot be moved down has an actionable reason category: frontend, rewrite, lowering, hardware primitive, memory capacity, or tokenizer/runtime.

### 12.4 Results of the formal stage (2026-10-13)

**Compile coverage**: the whole model, 18 layers (`build/hfgen/gemma_kch`, `--quant --rope --attn --kv-channel k --kv-calib train:1024 --prefill 16 --prefill-all`), 790 executables all on the accelerator; the fallback list is empty. The changes made for this (all effective only on Gemma-like structures; the golden corpus's 7046 dispatches unchanged):
- **GELU-tanh**: HF's `GELUTanh` exports as `math.fpowi` and lands on host VMVX. `hf_generic.act_rewrite` replaces it with `SaGELUTanh`: `x * recip(exp(x·(A + B·x²)) + 1)` (A = −2√(2/π), B = 0.044715·A), using only the device's EXP / RECIP; the module is probed first, and non-tanh GELU forms are rejected. Maximum error versus PyTorch on [−20, 20] 4.8e−7, the two ends giving −0 / x.
- **Attention with a single KV head**: 4 / 1 GQA gives attention's batch_matmul a batch dimension of 1 (`1x4xDx256`); `SahlExpandKernels` judged the batch loop as −1 and left the whole thing to the host. Fix: viewMap recognizes size-1 dimensions before looking at dynamic strides; the epilogue input analysis skips size-1 dimensions. Also a new attribute `sa.unexpanded = "SahlExpandKernels.cpp:LINE"`, so the reason a kernel was not expanded shows directly in the codegen report (6772516).
- **head_dim 256**: the capacity check passes for contexts with T ≤ 256 without C6.5's T tiling; T = 512 not verified (the risk of §12.3 remains).

**sim acceptance**: prefill vs decode-only 5/5 rows bit-exact, prefill_all 40/40; teacher forcing over 40 positions device 33/40 (mean corr 0.979), torch W8A8 36/40 (0.978), device vs torch differ at 6, corr 0.981 (within tolerance). aarch64 (qemu) 3/3 bit-exact.

**On the board (2026-10-13, four-port overlay, `build/deploy_hfgen_gemma_kch`)**: board_llm 17/17 bit-exact with the sim, 0 host dispatches; decode 66.8 ms / step (14.96 tok/s), prefill 256 ms per 16-token chunk (62.5 tok/s); device memory peak 266 MB. Teacher forcing (8 × 256) PPL 158.54 (fp32 123.76), top-1 76.0%. Note: the window in `board.txt` is set from the peak of a 40-token run (277 MB); teacher forcing's 256-token windows (16 MB of logits per chunk) need the window raised to 640 MB.

**Quality attribution** (host, WikiText-2 test 2048 tokens, window 256, `build/quality/gemma_2048_*.json`; with a vocabulary of 262144 the PPL over 2048 tokens is dominated by a few tokens and its ranking is unstable, so KL and top-1 are what count):

| Variant | PPL | top-1 | corr | KL |
|---|---:|---:|---:|---:|
| torch W8A8 (fp32 attention) | 126.15 | 81.8% | 0.9870 | 0.129 |
| sa:kch (device semantics) | 166.76 | 77.8% | 0.9792 | 0.238 |
| +act (fp32 activations in linear layers) | 163.98 | 83.1% | 0.9895 | 0.133 |
| +kv (fp32 KV cache) | 175.47 | 79.5% | 0.9811 | 0.216 |
| +q (attention query not quantized) | 169.36 | 76.3% | 0.9769 | 0.265 |
| +sfu (exact exp / recip / rsqrt) | 158.35 | 75.4% | 0.9771 | 0.274 |
| **+probs (fp32 attention probabilities)** | 137.09 | 80.2% | 0.9830 | 0.181 |
| +q+probs+kv+sfu (attention fully fp32) | 136.07 | 82.3% | 0.9873 | 0.126 |

The extra loss of the device semantics over W8A8 comes mainly from the **int8 attention probabilities** `i8(p·127)`: the resolution is 1/127, Gemma's attention distributions are spread out, and many p < 0.004 are rounded to 0. KV and query quantization and the SFU approximation are within noise. Unlike Qwen3 (§8, where the main cause is the K-channel outliers of the int8 KV), this is another kind of error determined by the model's structure.

**Candidate fix: unnormalized probability quantization**. Quantize `e = exp(s − max)` (its maximum is exactly 1, and e ≥ p) instead of `p = e·r`: `pq = i8(e·127)`, and the int32 PV result is then multiplied by the row's r and a_v. The same 8 bits, a per-row resolution no worse than today, and r goes from multiplying T probabilities to multiplying hd outputs; no hardware change, only the graph exported by `hf_generic.sa_attention` changes. Host evaluation (same setup): Gemma sa:kch PPL 166.76 → 145.27, top-1 77.8% → 80.5%, KL 0.238 → 0.167. SmolLM2 (ct-sa:kch, `build/quality/smollm2_pe_*.json`) improves too: PPL 28.82 → 28.25, top-1 87.7% → 88.7%, KL 0.072 → 0.053. It changes the numeric semantics of every model (golden, the sim and board acceptance of all four models have to be redone).

**Adopted: normalization by the quantized sum (qsum, 2026-10-13)**. `eq = i8(e·127)`, `out = (eq·V) · 127/Σeq · a_v`: the weights actually taking part in PV sum to exactly 1, and the tail mass rounded to 0 is redistributed proportionally over the remaining keys; Σeq is an exact integer (≤ 127·T), independent of summation order. It differs from fsum (`· 1/Σe`) only by one scalar per row (same output direction).

| Host, 2048 tokens | Current | fsum | **qsum** |
|---|---|---|---|
| Gemma PPL / top-1 / KL | 166.76 / 77.8% / 0.238 | 145.27 / 80.5% / 0.167 | 148.52 / 80.3% / 0.176 |
| SmolLM2 PPL / top-1 / KL | 28.82 / 87.7% / 0.072 | 28.25 / 88.7% / 0.053 | 28.50 / 89.9% / 0.053 |

Looking only at the probability-quantization step (fp32 Gemma's q/k/v, one 256-token window, 16200 attention rows): relative error of the attention output is 0.121 today, 0.0185 for fsum, 0.0159 for qsum, with qsum more accurate on 81% of the rows; the tail mass averages 1.7% (p90 4.0%). The end-to-end difference between the two (Gemma KL 0.009) is smaller than this evaluation's noise (e.g. exact SFU end to end was even +0.036), so qsum was chosen for its per-step accuracy and the exactness of Σeq.

Implementation (`hf_generic.sa_attention` and the attribution branches): Σeq must be summed directly over the int8 values (`eq.to(float32)`, sitofp): extending to i32 first and summing (`extsi` is not in to_i8's form) or summing `round(e·127)` (`math.roundeven` in the reduction body) both land on the host. 2-layer Gemma (214) and SmolLM2 (212) executables all on the accelerator, no device vs torch differences in teacher forcing, prefill 5/5 bit-exact; golden passes.

**qsum board acceptance (2026-10-13, four-port overlay, `build/deploy_hfgen_{gemma,smollm2,qwen3}_qs`)**: whole-model rebuilds (Gemma 791 / SmolLM2 1221 / Qwen3 1139 executables, all on the accelerator), sim prefill 5/5 and prefill_all 40/40 bit-exact, qemu 3/3; on the board board_llm 17/17 bit-exact with the sim, 0 host dispatches.

| Board (8 × 256 teacher forcing) | Old semantics PPL / top-1 / KL | **qsum** | decode (batched) | prefill per chunk |
|---|---|---|---:|---:|
| Gemma-3-270M (fp32 PPL 123.76) | 158.54 / 76.0% / 0.209 | **142.83 / 80.3% / 0.157** | 66.8 → 66.9 ms | 256.0 → 255.5 ms |
| SmolLM2-135M (fp32 PPL 26.98) | 28.54 / 86.5% / 0.055 | **28.26 / 89.8% / 0.038** | 45.1 → 45.2 ms | 266.3 → 264.5 ms |
| Qwen3-0.6B (fp32 PPL 39.21) | 39.62 / 86.9% / 0.053 | **39.68 / 88.1% / 0.038** | 145.4 → 143.6 ms | 773 → 769.7 ms |

Gemma's quantization PPL loss 34.8 → 19.1, KL −25%; SmolLM2 KL −31%, top-1 +3.3 points; Qwen3 PPL flat (+0.06, within noise over 2048 positions), KL −29%, top-1 +1.2 points. Speed unchanged: one more Σeq reduction dispatch per layer and row (decode Gemma 690 → 708, SmolLM2 1000 → 1030, Qwen3 1046 → 1074; prefill +16 per layer per chunk), the original softmax normalization is no longer needed, and the times of reduction and batch_matmul move in opposite directions (SmolLM2 prefill: reduction −5.6 ms, batch_matmul +4.5 ms); the batched rest of the runtime +0.1–0.3 ms. Llama being rebuilt.

## 13. Work package WP9: measurement infrastructure and coverage matrix

### 13.1 Runtime statistics

Extend `SA_STATS=1` to record for every dynamic dispatch:

- Name, operator category and key shapes.
- Where it executes: host / accelerator.
- Call count.
- Wall-clock time of host dispatches.
- Accelerator device cycles, command count, estimated DDR bytes and MACs.
- Fallback reason.

In addition: prefill exports the logits of every position (needed by the board teacher forcing of §3.2), and the fixed part of each decode step (attention EX, VE / SFU, LD command latency, dispatch / host overhead) is timed by category, for the decisions of §16.

If host and accelerator may overlap asynchronously, report the two sides' times separately and use the end-to-end wall clock as the total; do not add them directly.

**Implementation (2026-10-12, d9fea48, f85482e, 14f5612)**:

- Runtime: host fallbacks are recorded as `host:<export>` rows (CPU ns); the `SA_PROFILE_PERF` csv is output per section (prefill / decode), also written on the sim; the profile table holds 4096 exports (Qwen3 prefill has about 2800; the old 512 truncated the row attention).
- `compiler/tests/dispatch_stats.py`: per category the calls, cycles, MACs (from the loop ranges in the dispatch name), LD/ST bytes, dynamic / time / MAC coverage; `--report` (`--iree-sa-codegen-report`) gives static coverage and a classification of fallback reasons (forced / memory capacity / hardware primitive / lowering).
- Board teacher forcing: export `prefill_all` (`--prefill-all`, the logits of every position, bit-exact with decode only), `sa-llm-run --prefill_all_out`, `board_teacher.py` (on the board) + `board_quality.py prep / sim / score` (host: fp32 reference, PPL, top-1, KL over the reference's top 64).

**Qwen3-0.6B board results** (generic frontend, W8A8 + K per-channel KV, 4-port overlay; 1140 executables all on the accelerator):

- Quality (WikiText-2 test 8 windows × 256, teacher forcing): board PPL 39.62 (fp32 39.21), top-1 86.9%, KL 0.053 (the top 64 cover 87.8% of the reference's mass); host ct-sa:kch 39.14 / 87.4% / 0.054; the board agrees with the host model.
- decode per step (batched): device 175.0 ms, rest of the runtime 3.4 ms — **the fixed overhead is on the device, not the host**. After the V-projection fix (14f5612: decode's V projection is quantized straight into the KV cache; its int8 result was not accepted by matchLinear and fell to the generic EX path, 136k cycles per layer vs 17.6k for the K projection of the same shape) 211.1 → 178.4 ms/step (4.74 → 5.60 tok/s), 17/17 bit-exact.

| decode category (after the fix) | Share of cycles | ms/step | Notes |
|---|---:|---:|---|
| matvec (weight GEMV, including lm_head 27.6 ms) | 63.0% | 111.4 | Reads 598 MB, 53.7 B/cycle = 93% of the GEMV peak 57.95; bandwidth floor about 103 ms |
| batch_matmul (attention) | 26.3% | 46.5 | Computed over the static T=256, independent of position |
| reduction / elementwise / other | 10.7% | 18.7 | RMSNorm, RoPE, softmax, quantization |

- prefill 1.44 s per 16-token chunk (11.15 tok/s, batched 142.8M cycles): row attention (batch_matmul, 56 exports × 16 rows) 51.7%, matmul 27.9%, reduction 12.5%. **The next step for prefill is attention, not GEMM.**
- Next candidates (for the decisions of §16): decode attention over the actual length instead of a static T (≤46 ms); prefill attention by blocks instead of by rows; GEMV is already close to the bandwidth limit.

**Attention over the actual length (2026-10-12, `1860823`, generic frontend default)**: at export each function gets an extra `valid[T]` (T dynamic, ≤ cache length) and only the first T rows of the cache are used; `sa-llm-run --abi=hfv --pad=16` picks T from the position (position + 1 rounded up to 16). The dynamic T makes torch-mlir leave a "mismatching contracting dimension" assertion before each layer's matmul, each cutting the command buffer (SmolLM2 with 2 layers: 3 lists per step); with `--iree-opt-strip-assertions` in `compile_sa.sh` it is back to 1 per step (Qwen3 2, the same as static). Qwen3 on the board (positions 32–48):

| | Static T = 256 | Actual length |
|---|---:|---:|
| decode ms/step (tok/s) | 178.4 (5.60) | **145.4 (6.88)** |
| 　device / rest of the runtime | 175.0 / 3.4 ms | 136.9 / 6.9 ms |
| 　attention (batch_matmul) | 46.5 ms | 11.8 ms |
| prefill per 16-token chunk (batched) | 1428 ms (11.15 tok/s) | **773 ms (20.53 tok/s)** |
| 　row attention | 744.7 ms | 145.4 ms |
| teacher forcing (2048 positions) | PPL 39.624 | 39.624, argmax / NLL / top-64 bit-identical |

GEMV is now 80% of decode device time (111.4 ms, 93% of peak); the generic path is faster than the hand-written path (WP1's 6.28 tok/s). The rest of the runtime grew from 3.4 to 6.9 ms (dynamic-shape computation on the host, to be investigated). `dispatch_stats.py` does not count MACs for dynamic dimensions (`D` in the name), so attention's MACs are not in the coverage.

### 13.2 Final main table

| Model | Weight source | Correctness | Quality | Dynamic/time/MAC coverage | decode | prefill | Bandwidth utilization | Memory peak |
|---|---|---|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | External W8A8 (ct) + K per-channel KV | Board 17/17 bit-exact | PPL 39.68 (fp32 39.21), top-1 88.1% (qsum) | 100% / 100% / 100% | 6.96 tok/s | 20.79 tok/s | GEMV 93% (decode 4.38 GB/s average) | 590 / 613 MB |
| SmolLM2-135M | llm-compressor W8A8 (ct) + K per-channel KV | Board 17/17 bit-exact | PPL 28.26 (fp32 26.98), top-1 89.8% (qsum) | 100% / 100% / 100% | 22.12 tok/s | 60.50 tok/s | GEMV 82% (decode 3.43 GB/s average) | 135 / 150 MB |
| Llama-3.2-1B | RedHatAI W8A8 (ct) + K per-channel KV | Board 17/17 bit-exact | PPL 23.33 (fp32 22.77), top-1 90.7% (host 2048-token evaluation; not enough memory for prefill_all on the board) | 100% / 100% / 100% | 4.04 tok/s | 14.97 tok/s | GEMV 94% (decode 5.08 GB/s average) | 1190 / 1203 MB |
| Gemma-3-270M | Self-quantized W8A8 (per-token dynamic activations) + K per-channel KV | Board 17/17 bit-exact | PPL 142.83 (fp32 123.76), top-1 80.3% (board 8×256 teacher forcing, qsum; old semantics 158.54 / 76.0%, §12.4) | 100% / 100% / 100% | 14.95 tok/s | 62.62 tok/s | GEMV 88% (decode 4.38 GB/s average) | 266 / 640 MB (window set for teacher forcing) |

**Main-table breakdown (2026-10-12, board, four-port overlay)**: category times come from per-dispatch timing (`board_profile.py 40 --prompt 32 SA_PROFILE_PERF=...csv` → `dispatch_stats.py`, one list per dispatch); total per-step time, device time and the rest of the runtime come from batched measurement (`--batch`); the two agree on device time within about 1%; submission time in per-dispatch mode is inflated (90–270 ms per step) and not listed. Data: `build/deploy_hfgen_llama1b_kch/prof_llama_*.json`, `build/deploy_hfgen_{qwen3,smollm2,gemma}_qs/prof_{decode,prefill}.json`.

Attention-probability quantization: the Qwen3, SmolLM2 and Gemma rows (and the three columns of the two breakdown tables below) are qsum (2026-10-13, §12.4); the Llama row is still the old semantics, updated after the rebuild.

decode per step:

| | Qwen3-0.6B | SmolLM2-135M | Llama-3.2-1B | Gemma-3-270M |
|---|---:|---:|---:|---:|
| Per step (batched) | 143.6 ms | 45.2 ms | 247.6 ms | 66.9 ms |
| 　device / rest of the runtime | 136.60 / 7.02 ms | 38.48 / 6.71 ms | 243.9 / 3.6 ms | 60.91 / 5.75 ms |
| dispatches (host) | 1074 (0) | 1030 (0) | 538 (0) | 708 (0) |
| **GEMV** (matvec_like) | 111.3 ms (80%) | 28.6 ms (71%) | 226.5 ms (93%) | 52.7 ms (85%; lm_head 31.7 ms) |
| 　GEMV bandwidth (peak 57.95 B/cycle) | 53.8 (93%) | 47.4 (82%) | 54.7 (94%) | 51.2 (88%) |
| attention (batch_matmul) | 12.2 ms (9%) | 4.2 ms (10%) | 6.9 ms (3%) | 3.0 ms (5%) |
| reduction (norm, softmax) | 7.3 ms (5%) | 3.7 ms (9%) | 6.7 ms (3%) | 3.3 ms (5%) |
| elementwise kinds | 7.4 ms (5%) | 3.5 ms (9%) | 4.6 ms (2%) | 2.8 ms (5%) |
| scatter (KV write) | 0.3 ms | 0.3 ms | 0.2 ms | 0.2 ms |
| DDR read | 607 MB, 4.38 GB/s | 138 MB, 3.43 GB/s | 1243 MB, 5.08 GB/s | 272 MB, 4.38 GB/s |

prefill per 16-token chunk:

| | Qwen3-0.6B | SmolLM2-135M | Llama-3.2-1B | Gemma-3-270M |
|---|---:|---:|---:|---:|
| Per chunk (batched) | 769.7 ms | 264.5 ms | 1068.9 ms | 255.5 ms |
| dispatches (host) | 6838 (0) | 5404 (0) | 2642 (0) | 3876 (0) |
| **GEMM** (matmul_like) | 402.1 ms (51%) | 132.1 ms (48%) | 779.0 ms (73%) | 113.5 ms (44%) |
| 　array utilization (peak 256 MAC/cycle) | 175 (68%) | 129 (50%) | 200 (78%) | 141 (55%) |
| row attention (batch_matmul) | 152.9 ms (20%) | 55.2 ms (20%) | 88.7 ms (8%) | 38.6 ms (15%) |
| reduction | 113.9 ms (15%) | 51.6 ms (19%) | 108.3 ms (10%) | 46.3 ms (18%) |
| elementwise kinds | 81.3 ms (10%) | 26.3 ms (10%) | 50.1 ms (5%) | 25.8 ms (10%) |
| GEMV (lm_head, last row) | 27.6 ms (4%) | 5.2 ms (2%) | 45.6 ms (4%) | 31.7 ms (12%) |
| scatter | 6.0 ms | 4.8 ms | 2.7 ms | 3.0 ms |

- decode is mostly bandwidth-bound GEMV (Llama 93%, Qwen3 80%); the remaining fixed part is mainly attention and norms; SmolLM2's matrices inside the layers are only 576 wide, so the fixed overhead of GEMV commands is a larger share (82%).
- prefill is not only GEMM: the GEMM array utilization of the small models is only 50–68%, and row attention plus reduction take 30–40%.
- Gemma's vocabulary of 262144 makes lm_head (168 MB int8) the largest single item of decode (60% of GEMV, 51% of device time); the layers are only 640 wide, prefill array utilization 55%.
- The 100% MAC coverage excludes attention: `dispatch_stats.py` does not count MACs for dynamic dimensions (to be fixed).

### 13.3 Two ablation tables

Do not mix hardware resources and pure compiler features in the same table.

**Hardware ablation**: EX → GEMV 1 port → 2 ports → 4 ports, with bitstream features such as P6/REDUCE listed separately; columns token/s, device cycles, DDR bandwidth, LUT, FF, BRAM, DSP, WNS.

**Compiler/model ablation**: self-quantization → external W8A8, K split off → on, row-by-row prefill off → on, host fallback → accelerator; columns quality, coverage, cycles, token/s and memory peak. The K split does not repeat the same hardware resources.

**Hardware ablation (first version, 2026-10-12)**: the same hand-written-path tests (`c6p_*`, hfgen, the compiler at staging time), each overlay with its own build, board logits all bit-exact with the sim; decode is `board_llm.py`'s per-step wall clock (tok/s). Resources and timing from the build table in `boards/kv260/README.md`, GEMV bandwidth from bwtest (64 KB, x resident).

| overlay | Qwen3-0.6B | SmolLM2-135M | SmolLM2 hfgen | stories15M | GEMV B/cycle | LUT | FF | BRAM36 | DSP | WNS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EX (`p6_red`) | 2.22 | 7.71 | 7.62 | 61.9 | — (LD 15.70) | 88,250 (75.4%) | 60,823 | 130 | 347 | +0.123 |
| GEMV 1 port (`_gemv`) | 2.28 | 8.21 | 8.34 | 68.07 | 15.60 | 92,329 (78.8%) | 61,566 | 132 | 349 | +0.274 |
| GEMV 2 ports (`_gemv_np2`) | 3.98 | 12.58 | 12.73 | 100.8 | 30.55 | 96,294 (82.2%) | 63,675 | 134 | 349 | +0.138 |
| GEMV 4 ports (`_gemv_np4`) | **6.28** | **17.04** | **17.19** | **131.65** | 57.95 | 101,878 (87.0%) | 67,688 | 138 | 349 | +0.168 |

- One-port GEMV has the same LD bandwidth as EX (about 15.6 B/cycle), and decode is only 3–10% faster (saving the SPAD_B round trip and the EX pipeline); the gain comes from the number of ports: 2 ports ×1.8, 4 ports ×2.8 (Qwen3). The cost of 4 ports: 13.6k LUT, 6.9k FF, 8 BRAM36 and 2 DSP more than EX.
- Other bitstream features (Qwen3, hand-written path): P6 batched SFU prefill 11.4 → 14.0 tok/s; batched REDUCE prefill → 15.45 tok/s (LUT 72.1% → 75.4%); decode unchanged (2.20–2.22).

**Compiler / model ablation (first version, 2026-10-12)**: same hardware (4-port overlay), same model, only one compiler or frontend item changed; everything except quality is measured on the board, and bit-exact items are noted.

| Item | Model | Off | On | Notes |
|---|---|---:|---:|---|
| Self-quantization → external W8A8 (llm-compressor) | SmolLM2 (host, SA form, 2048 tokens) | PPL 30.77, top-1 83.9% | PPL 28.77, top-1 87.1% | §7; KL 0.138 → 0.074 |
| Per-layer KV → K per channel (+1024-token calibration) | Qwen3 (host, 2048 tokens) | PPL 53.50, top-1 71.5% | PPL 39.14, top-1 87.4% | §8; board 39.62 / 86.9% |
| Host fallback → accelerator (row-by-row prefill attention) | SmolLM2 hfgen | prefill 14.20 tok/s, 1440 host dispatches | 27.18 tok/s, 0 | §7; lists 1465 → 37 |
| K split (`--iree-sa-gemv-k-split`) | Llama-3.2-1B | decode 1.40 tok/s | **4.03 tok/s (×2.89)** | §10; logits bit-identical |
| V projection's int8 result on GEMV (`14f5612`) | Qwen3 | decode 4.74 tok/s | 5.60 tok/s | §13.1; bit-identical |
| Attention over the actual length (`1860823`) | Qwen3 | decode 5.60 / prefill 11.15 tok/s | **6.88 / 20.53 tok/s** | §13.1; 2048-position teacher forcing bit-identical |
| Attention probabilities `i8(p·127)` → qsum (`1a0ba3d`) | Gemma-3-270M (board 8 × 256 teacher forcing) | PPL 158.54, top-1 76.0%, KL 0.209 | **PPL 142.83, top-1 80.3%, KL 0.157** | §12.4; SmolLM2 / Qwen3 KL −31% / −29%; speed unchanged |

The generic frontend's Qwen3 decode (6.88 tok/s, 6.96 after qsum) is now faster than the hand-written path on the same overlay (6.28).

## 14. Schedule and stop conditions

| Dates | Goal | Stop condition / fallback action |
|---|---|---|
| ~~10/11–10/14~~ | WP0–WP1: four-port wrap-up | **Done 10/10** (perf_model calibration left, can run in parallel with WP2); bitstream frozen as `d16_100mhz_w128_p6_gemv_np4` |
| 10/11–10/18 | WP2: importer schema and synthetic tests (**first version done 10/10**; the prefill equivalence problem and the generic-frontend Qwen3 fallbacks are solved) | If format parsing is unstable by 10/18, narrow to a single verified checkpoint schema |
| 10/19–10/24 | WP3–WP4: SmolLM2 (**accepted on the board 10/11**), Qwen3 | Even if Qwen's quality has not improved by 10/24, the four-layer attribution conclusion must be delivered; no further blind scale tuning |
| 10/25–10/29 | WP5: Llama frontend, EX baseline, memory preflight | If the memory gate fails, pause board staging, keep the sim results and start the M2 evaluation |
| 10/30–11/03 | WP6–WP7: K split and Llama | If the K split is unstable, the talk uses the EX correctness baseline and marks the performance optimization as in progress |
| 11/04–11/09 | First WP9, ablation tables | Finish reproducible data first; do not wait for Gemma |
| 11/04–11/12 | WP8: Gemma short-context sim | If not end to end by 11/10, show the import coverage, the first blocking point and the fallback list |
| 11/13–11/15 | Tables frozen, talk material | After 11/15 no new hardware or large compiler refactoring |
| 11/16–11/25 | Slides, reproduction experiments, rehearsal | Only blocking bugs fixed; every new result must be fully reproducible |

## 15. Suggested commit split

Each commit should be independently testable and revertible:

1. ~~`docs: sync KV260 G4/G5 status and metrics`~~ (`fb01909`, `b42fec6`)
2. ~~`frontend: parse compressed-tensors W8A8 config`~~
3. ~~`frontend: add external int8 weight/scale representation`~~
4. ~~`frontend: preserve imported int8 values including -128`~~
5. ~~`tests: add synthetic compressed-tensors checkpoints`~~ (2–5 merged into one commit)
6. `frontend: import SmolLM2 and Qwen3 W8A8 checkpoints` (SmolLM2 done)
7. ~~`tools: add quantization quality comparison pipeline`~~ (`quality_eval.py`)
8. `frontend: add Llama 3.2 tokenizer and RoPE coverage` (tokenizer done)
9. ~~`tools: add device memory manifest and preflight`~~ (`mem_preflight.py`)
10. ~~`compiler: split GEMV K at aligned 4080-element boundary`~~ (decode)
11. `tests: cover GEMV K split boundaries and overflow`
12. `runtime: report per-dispatch host/device timing and fallback reason`
13. `frontend: add Gemma 3 graph rewrites and short-context bring-up`

Avoid changing the importer, lowering, RTL and golden outputs in the same commit; otherwise, when bit-exactness breaks, it is hard to tell which layer is responsible.

## 16. Rules for choosing post-meeting performance items

Decide from the time breakdowns of the three models Qwen, Llama and Gemma:

| Observation | Priority item |
|---|---|
| A high share of fixed per-step cost in decode (measured with four ports: Qwen3 about 42%, SmolLM2 about 65%) | H2 out-of-order issue + LD command pipelining + fewer dispatches and less host overhead; find the largest item first with WP9's categorized timing. **WP9 measurement (2026-10-12, Qwen3 generic path)**: host / runtime only 3.4 ms, the fixed part is on the device; after the v-projection fix attention (static T = 256) 46.5 ms, other small operators 18.7 ms, GEMV 111 ms (93% of peak); next: attention over the actual length |
| Most decode time is weight traffic, four ports close to the port limit | W4A8 (only shrinks the GEMV part); quality feasibility experiment first, then hardware unpacking |
| Several models still limited by 100 MHz arithmetic/control cycles, with a clear timing-fix path | K2a-2 200 MHz |
| LD command bubbles consistently above about 8–10% across models | H2 + LD command pipelining |
| A high share of SwiGLU/elementwise segments in Qwen prefill | Row-by-row prefill execution |
| Gemma/long-context attention fails on capacity | C6.5 T tiling, then evaluate K3 URAM |
| prefill clearly limited by SPAD bank conflicts | SPAD sub-banks |

Each candidate must give at the same time: the models that benefit, an upper bound on its share of time, the expected gain, the resource budget, the timing risk, the quality risk and the verification cost.

## 17. Final definition of done

When this stage is complete, the same body of evidence should answer:

1. Which public W8A8 checkpoints can the compiler import, which formats does it reject, and why?
2. How much of Qwen3's quality loss comes from the checkpoint, and how much more does this project introduce?
3. How is Llama 3.2's K=8192 mapped onto GEMV without changing the hardware, while staying bit-exact?
4. Across three model families, how much of the dynamic work runs on the accelerator, and where are the remaining hot spots?
5. What are the measured gain, bandwidth utilization and resource cost of the four-port GEMV?
6. Why is the next hardware investment 200 MHz, W4A8, URAM or one of the other candidates?

As long as these questions have reproducible data behind their answers, even if Gemma still has some host fallback, this stage already forms a stronger compiler and systems contribution than simply chasing another 10% of performance.

### 17.1 Question 1: which W8A8 checkpoints import, and which formats are rejected (2026-10-13)

`compiler/frontend/ct_import.py` (§6) accepts exactly the compressed-tensors scheme the hardware executes as is, and rejects everything else with the layer, field and supported range named; there is no silent fallback or requantization.

| | Accepted | Rejected (and why) |
|---|---|---|
| Format | `quant_method: compressed-tensors`, `format: int-quantized`, `quantization_status: compressed` | Any other method / format / status |
| Weights | int8, static, per output channel, symmetric (zero point 0), `group_size` unset; scales fp32 / bf16 / fp16 (converted to fp32, int8 values untouched, -128 kept); GPTQ `actorder: static` (only a column order, already folded into the weights) | Per-group / per-block scales (EX accumulates a whole K in int32; per-group scales would need a split K with a rescale per group); asymmetric / zero points (no zero-point path in EX or the GEMV unit); int4 and FP8 (EX and the GEMV unit multiply int8 × int8 only; W4A8 is §17.6's candidate) |
| Activations | int8, dynamic, per token, symmetric: the device computes `amax / 127` per token itself | Static activation scales; output activations quantized |
| KV cache | — (the project calibrates K per channel, §8) | `kv_cache_scheme` set: llm-compressor 0.14 offers only `tensor` / `attn_head` granularity, which loses Qwen3's quality (§8) |
| Layers | `ignore` lists (e.g. `lm_head`, then quantized by the project and recorded in the manifest) | A tied `lm_head` stored separately and different from the embedding; an embedding shared with a quantized (not ignored) classifier (the embedding gather goes through VE, which saturates -128 to -127) |

Tested: `test_ct_import.py` (`run_tests.sh ctimport`) covers 8 rejection variants and 3 accepted ones on synthetic checkpoints, byte-identical int8 and scales, and an end-to-end synthetic model on the sim. Imported in practice and accepted on the board (§13.2): RedHatAI `Llama-3.2-1B-Instruct-quantized.w8a8` (SmoothQuant 0.7 + GPTQ, public), SmolLM2-135M and Qwen3-0.6B W8A8 generated with llm-compressor (SmoothQuant 0.8 + GPTQ, `compiler/frontend/llmc_w8a8.py`); Gemma-3-270M uses the project's own W8A8 quantization. The external checkpoints beat self-quantization under the device's arithmetic (SmolLM2 KL 0.138 → 0.074, Qwen3 top-1 +3 points), almost entirely thanks to SmoothQuant.

### 17.2 Question 2: Qwen3's quality loss, checkpoint versus this project (2026-10-13)

Qwen3-0.6B, WikiText-2 test, 2048 positions in windows of 256, reference the original fp32 model (PPL 39.21); details in §8.

| Layer | PPL | top-1 | KL | Loss attributed to |
|---|---:|---:|---:|---|
| Checkpoint, standard semantics (ct-w8a8, host) | 40.05 | 92.3% | 0.022 | The checkpoint itself |
| Device semantics, first version (per-layer static int8 KV, host) | 53.50 | 71.5% | 0.411 | This project |
| + K per channel, KV calibrated on 1024 train tokens (host) | 39.14 | 87.4% | 0.054 | This project |
| Same, on the board (teacher forcing 8 × 256) | 39.62 | 86.9% | 0.053 | This project |
| + attention probabilities normalized by their quantized sum (qsum, board) | **39.68** | **88.1%** | **0.038** | This project |

- **The checkpoint costs** PPL +0.84 (2%), 7.7 points of top-1 and KL 0.022.
- **This project's first version added** far more (top-1 −21 points, KL ×19). The attribution (switching one item at a time to fp32, §8) put almost all of it on the **int8 KV cache**: fp32 KV alone gives back 91.7%; q, probabilities and the SFU approximations were each within ±1 point. The cause was outlier channels of K (per-channel max/median 7–120×), so the fix is per-channel K scales folded into q (int8 dot product unchanged, no hardware change) plus enough calibration data; neither alone helps.
- **What remains today**: on the board, versus the checkpoint, top-1 4.2 points lower and KL 0.016 higher, at the same PPL; the board agrees with the host model of the device semantics (39.62 vs 39.14 PPL, within noise). Most of the gap is the remaining KV quantization error (fp32 KV on top of per-channel K reaches 92.0% on the host); percentile-clipping calibration is the next candidate.

### 17.3 Question 3: Llama 3.2's K = 8192 on GEMV without hardware changes (2026-10-13)

The GEMV descriptor's `row_bytes` is 16 bits and equals K × 16, and the unit's x buffer is 4096 bytes per lane, so one GEMV command takes K ≤ 4080 (the largest multiple of 16 below 4096). Llama-3.2-1B's `down_proj` has K = 8192. Before WP6 such layers did not match the decode linear micro-kernel at all and ran entirely on EX through SPAD_B.

- **Mapping** (compiler only, `expandLinear`, switch `--iree-sa-gemv-k-split`): K is split into 4080 + 4080 + 32; each block is one GEMV command over the matching slices of x and the packed weights; the first writes the ACC words (`acc = 0`), the others add into them through the descriptor's existing `acc` bit (w3[58]); the epilogue (dequantization, requantization) runs once after the last block.
- **Why it is bit-exact**: accumulation is int32 with no saturation or rounding between blocks, and |Σ| ≤ K × 128 × 128 = 2²⁷ ≪ 2³¹ (no overflow for any K ≤ 131071), so the sum is independent of the order and the split. `test_gemv_ksplit.py` checks K = 4080, 4096, 4112, 8192 and the −128 × −128 worst case against numpy's exact sum, with the split both on and off; the golden corpus (7046 dispatches of the existing GEMV models) is byte-identical.
- **Result on the board** (§10): logits identical with the split on and off; `down_proj` 3.20M → 0.275M cycles per call (5.3 → 61 B/cycle), Llama decode **1.40 → 4.03 tok/s (×2.89)**. Today GEMV runs at 94% of the four-port peak in Llama's decode (§13.2).

### 17.4 Question 4: coverage and remaining hot spots (2026-10-13)

Four models (Qwen3, SmolLM2 (a small model with the Llama structure), Llama 3.2, Gemma 3), whole models, the same compile flow, per-dispatch timing on the board (§13.2): **dynamic dispatch, device time and MAC coverage are all 100%**, with no host fallback (Gemma's GELU-tanh and single-KV-head attention were the last two, §12.4). The remaining hot spots move with the shape ratios:

- decode: dominated by weight traffic; GEMV is Qwen3 80%, SmolLM2 71%, Llama 93%, Gemma 85% of device time (Gemma's lm_head alone 51%, vocabulary 262144), at 82–94% of the 4-port peak bandwidth; the rest is attention, norms and the fixed overhead of each command, a larger share on narrow models (SmolLM2 576, Gemma 640).
- prefill: GEMM only 44–73%, array utilization 50–78% (lower for narrow models); row attention and reduction together 18–39%, the target of the next compiler-side steps (attention batching, norm fusion).
- Not yet counted: attention's MACs (`dispatch_stats.py` does not count dynamic dimensions).

### 17.5 Question 5: gain, bandwidth utilization and cost of the four-port GEMV (2026-10-13)

Data from §5, §13.3 and `k2b_gemv_design.md` §8.

| | EX path | GEMV 1 port | 2 ports | **4 ports** |
|---|---:|---:|---:|---:|
| bwtest GEMV, 64 KB (B/cycle; peak 16 × ports) | — (LD 15.70) | 15.64 | 30.55 | **57.95** (90.5%) |
| Qwen3 decode, hand-written path (tok/s) | 2.22 | 2.28 | 3.98 | **6.28** (×2.83) |
| SmolLM2 decode, hand-written path (tok/s) | 7.71 | 8.21 | 12.58 | **17.04** (×2.21) |
| LUT | 88,250 (75.4%) | 92,329 | 96,294 | **101,878 (87.0%)** |
| FF / BRAM36 / DSP | 60,823 / 130 / 347 | 61,566 / 132 / 349 | 63,675 / 134 / 349 | 67,688 / 138 / 349 |
| WNS (100 MHz) | +0.123 | +0.274 | +0.138 | +0.168 |

- **Gain**: one port alone gains only 3–10% (it saves the SPAD_B round trip but has the same 16 B/cycle as LD); the gain comes from the ports: decode ×1.8 at 2 ports and ×2.8 at 4 ports for Qwen3, against a model prediction of ×3.15. The gap is a fixed per-step part (about 67 ms for Qwen3 at the time) that does not scale with ports; compiler work then cut it (V projection on GEMV, attention over the actual length), so on the generic path Qwen3 decodes at 6.96 tok/s (§13.2).
- **Bandwidth utilization**: the 64 KB microbenchmark reaches 90.5% of 64 B/cycle (a fixed ~100 cycles per command on top of 1024 data cycles; steady state close to 64). Inside the models GEMV runs at 82–94% of the 57.95 B/cycle measured peak (Llama 94%, Qwen3 93%, Gemma 88%, SmolLM2 82%: narrow 576-wide layers pay the per-command overhead more), and GEMV is 71–93% of decode device time.
- **Cost**: versus the EX overlay, +13.6k LUT (75.4% → 87.0%), +6.9k FF, +8 BRAM36 (one x buffer per lane), +2 DSP; timing still closes at 100 MHz (the worst path is the PicoRV32 reset fan-out, not the GEMV data path). prefill is unchanged by construction (it uses EX).

### 17.6 Question 6: the next hardware investment (2026-10-12)

Basis: the categorized decode timing of three models in §13.2 (GEMV's share of decode time: Qwen3 80%, SmolLM2 71%, Llama 93%, at 82–94% of the 4-port peak bandwidth), and the 4-port performance model recalibrated on Qwen3's generic path (prefill +2.3%, decode +3.6%; what-if table in `k2b_gemv_design.md` §8.1).

| Candidate | decode | prefill | Cost / risk |
|---|---|---|---|
| **W4A8** | Qwen3 ×1.48 (model); upper bound by GEMV share Llama ×1.87, SmolLM2 ×1.55 | Unchanged | RTL limited to int4 unpacking in GEMV / LD; weight memory halved (Llama 1.24 GB → about 0.65 GB); the quality risk must first be evaluated on the host with a W4A8 checkpoint |
| 200 MHz | ×1.79 | ×1.88 | Whole-design timing closure (at 100 MHz LUT 87%, WNS +0.168 ns), less DDR bandwidth headroom; lower project priority than architecture |
| 6 GEMV ports | ×1.28 | Unchanged | The PS only has HPC1 left; PicoRV32's port has to move |
| H2 / LD command pipelining | ×1.04–1.05 | ×1.01–1.07 | Small gains alone; ×1.89 stacked with W4A8 |
| URAM / SPAD sub-banks | Cannot be evaluated by the model (changes the tiling) | prefill array utilization 50–78% | Only brought forward when Gemma / long contexts fail on capacity (§16) |

**Conclusion**: the next hardware investment is **W4A8**. The decode of all three models is dominated by weight traffic and already close to the 4-port bandwidth limit; W4A8 is the only item that shortens GEMV time and memory footprint at the same time, and its changes are confined to the streaming GEMV unit without touching global timing. 200 MHz has the largest gain and also helps prefill, but has the highest risk, so it comes after W4A8; stacked, the model predicts decode ×2.53.

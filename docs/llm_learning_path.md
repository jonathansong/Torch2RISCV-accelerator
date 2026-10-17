# LLM inference learning path

A guided route through the LLM side of this repository: from what a decoder
computes, through quantization, the inference system around the accelerator
and long context, to measuring speed and quality on the board. It is written
against tag `v2.2-kv260` (2026-10-17): four unmodified HuggingFace models
(SmolLM2-135M, Gemma-3-270M, Qwen3-0.6B, Llama-3.2-1B) compiled whole for the
KV260 accelerator, every dispatch on the accelerator, bit-exact with the
functional simulator. The numbers this path refers to are in
[`results_v2.2.md`](results_v2.2.md).

Like the [hardware learning path](hardware_learning_path.md), every stage
pairs outside reading with the files in this repo that put the idea into
practice, a lab that uses the real scripts, and questions to check yourself.
Its companion, the [compiler learning path](compiler_learning_path.md),
follows the same models through IREE and the `sa` backend; this one stays on
the model, its numerics and the system that runs it.

The project history is a curriculum here too:
- **L0–L5b** ([`llm_inference_plan.md`](llm_inference_plan.md)): llama2.c's
  stories15M, a NumPy reference model with the device's numerics, a
  bit-exact functional simulator, then hand-built descriptor lists for one
  layer (L3), the whole decoder (L4) and an ARM runtime that generates text
  (L5) and batches 8 sequences (L5b);
- **C0–C6** ([`iree_compiler_plan.md`](iree_compiler_plan.md)): the same
  model, then HuggingFace models, compiled by IREE instead of by hand;
- **WP0–WP9** ([`generality_implementation_plan.md`](generality_implementation_plan.md)):
  unmodified HuggingFace models, public W8A8 checkpoints, quality measured on
  the board;
- **C6.5 and long context** (`iree_compiler_plan.md` §8.22): 2048 positions;
- **C7** ([`riscv_backend_design.md`](riscv_backend_design.md)): the decode
  loop on the PicoRV32 control core instead of the ARM.

---

## How to use this path

- **Order.** Stages 0–2 are the model and its arithmetic; stages 3–5 are
  quantization, the part that decides quality; stages 6–8 are the system
  (runtime, prefill and decode, long context); stages 9–10 measure it; stage
  11 is for going further. Do 0–5 in order; after that follow your interest.
- **Each stage has:** *Goal* · *Read* (outside material) · *In this repo*
  (files, in reading order) · *Lab* · *Check yourself* (the answers are in the
  files or the docs named).
- **Tools.** Python 3 with NumPy for `llm/` (nothing else); the IREE venv
  `build/iree/venv` (torch, transformers, iree-turbine) for
  `compiler/frontend/` and `compiler/tests/`; `source compiler/env.sh` first.
  The llm-compressor environment `build/llmc_venv` only for stage 4's
  checkpoint generation. The board stages need a KV260 with a bundle from
  `compiler/scripts/build_board_bundles.sh` (stage 10).
- **Environment for the generic models:** `SA_D=16 SA_GEMV_PORTS=4` (the
  four-port overlay; the tests default to D = 8 otherwise), and
  `SA_KV260_CONFIG=d16_100mhz_w128_p6_gemv_np4_pcpi` for bundles.
- **Model files** live in `build/llm_cache/` (git-ignored): `stories15M.bin`
  and `tokenizer.bin` (from `notebooks/llm/prepare_llm.sh`), the HuggingFace
  directories, their `-W8A8` compressed-tensors checkpoints, and WikiText-2
  raw (`wikitext-2-raw-v1-{train,test}.txt`).
- **Memory.** The host has 15 GB: run one model-sized job at a time;
  `compiler/scripts/run_tests.sh` caps each test (`-m`, default 6 GB).

## One token at a glance

```
 tokenizer (hf_tokenizer.py, on the ARM)
   │ token id, position
   ▼
 embedding row (int8 + per-row scale)  ─┐
                                        │ × layers (16–30):
   ┌────────────────────────────────────┘
   │ RMSNorm → Q K V projections (W8A8 GEMV) → RoPE (cos/sin tables, pair swap)
   │   → K, V appended to the int8 KV cache at `pos`
   │   → scores q·Kᵀ over T positions (int8 × int8 → int32, GQA: kv head = batch)
   │   → softmax (max, exp on the SFU, p = i8(e·127), divide by Σp)  → p·V
   │   → O projection → residual
   │ RMSNorm → gate / up (W8A8) → SiLU or GELU-tanh → down → residual
   ▼
 final norm → lm_head (W8A8, often the tied embedding) → logits (fp32)
   ▼
 argmax / sampling (ARM, or the PicoRV32 in plan B) → next token
```

Decode does this for one token and reads every weight once: it is bound by
DDR bandwidth (stage 9). Prefill does it for 16 tokens at a time and is bound
by compute.

The four models of the main table (from their `config.json`):

| | SmolLM2-135M | Gemma-3-270M | Qwen3-0.6B | Llama-3.2-1B |
|---|---:|---:|---:|---:|
| Family | Llama | Gemma 3 | Qwen3 | Llama 3.2 |
| Layers | 30 | 18 | 28 | 16 |
| Hidden / FFN | 576 / 1536 | 640 / 2048 | 1024 / 3072 | 2048 / 8192 |
| Q heads / KV heads / head_dim | 9 / 3 / 64 | 4 / 1 / 256 | 16 / 8 / 128 | 32 / 8 / 64 |
| Vocabulary | 49 152 | 262 144 | 151 936 | 128 256 |
| Activation | SiLU | GELU-tanh | SiLU | SiLU |
| Particulars | tied embedding | sliding window 512, QK norm | QK norm, tied embedding | tied embedding |
| Board decode / prefill (tok/s) | 23.6 / 57.8 | 16.5 / 68.5 | 7.21 / 19.0 | 4.42 / 15.4 |

---

## Stage 0 - What a decoder computes

**Goal.** Be able to write one decoder step from memory: shapes of every
tensor, where the KV cache is read and written, and what changes between
prefill and decode.

**Read.**
- Vaswani et al., *Attention Is All You Need* (2017), §3.
- Andrej Karpathy, `llama2.c` (`run.c`): a whole Llama inference in one C
  file, and the origin of stories15M.
- Jay Alammar, *The Illustrated GPT-2*; Lilian Weng, *The Transformer
  Family* (v2) for the variants.
- Su et al., *RoFormer* (RoPE); Zhang & Sennrich, *RMSNorm*; Shazeer, *GLU
  Variants Improve Transformer* (SwiGLU); Ainslie et al., *GQA*.

**In this repo.**
1. `llm/checkpoint.py`, `llm/tokenizer.py`: llama2.c's checkpoint and BPE
   tokenizer in Python.
2. `llm/ref_model.py` `Fp32Model`: `run.c` in NumPy float32. Read `forward`
   top to bottom; it is the decoder step above, without quantization.
3. `compiler/frontend/qhf.py` `HFConfig`, `Fp32Model`: the same for a
   HuggingFace Llama or Qwen3 directory (`config.json` + safetensors), still
   without transformers. Note how `load_weights` permutes `wq` / `wk` per
   head from HuggingFace's rotate-half RoPE to llama2.c's interleaved pairs.
4. `compiler/frontend/hf_tokenizer.py`: a HuggingFace BPE tokenizer in plain
   Python (it also runs on the board).

**Lab.**
```sh
python3 llm/ref_model.py build/llm_cache/stories15M.bin build/llm_cache/tokenizer.bin -n 64
```
Compare the text with `run.c`'s (`build/llm_cache/runc_15M.txt`): it is
byte-identical. Then print the shapes of `q`, `k`, `v`, the scores and the
cache inside one step.

**Check yourself.**
- For Qwen3-0.6B at T = 2048, how many bytes does the int8 KV cache take?
  (layers × T × kv_heads × head_dim × 2.)
- Why can GQA's query heads of one group share a single pass over K and V?
  (`qhf.py`'s module docstring writes the kv head as the batch.)
- Why are rotate-half and interleaved-pair RoPE the same function up to a
  permutation of the head's channels?

## Stage 1 - Architectures: what differs between the four models

**Goal.** Know which parts of a "Llama-like" model actually vary, and which of
them a compiler or accelerator has to care about.

**Read.**
- The model cards and `config.json` of SmolLM2, Gemma 3 (the technical
  report: local and global layers, sliding window), Qwen3 (QK norm),
  Llama 3.2.
- HuggingFace transformers' `modeling_llama.py`, `modeling_qwen3.py`,
  `modeling_gemma3.py`: the code the generic path exports unmodified.

**In this repo.**
1. [`frontend_analysis.md`](frontend_analysis.md) §1, §4: the hand-written
   path (`qhf.py`) against the generic path (`hf_generic.py`), and the gaps
   to generality as seen on 2026-10-03.
2. `compiler/frontend/hf_generic.py`: `HFDecoder` wraps
   `AutoModelForCausalLM`; the rewrites are module swaps by type:
   `rope_rewrite` (`RopeTable`, `swapneg`), `norm_rewrite` (`SaRMSNorm`),
   `act_rewrite` (`SaGELUTanh`), `quantize` (`QLinear`, `QEmbedding`), and
   `SaCache` + `sa_attention` registered through transformers'
   `AttentionInterface`.
3. [`generality_implementation_plan.md`](generality_implementation_plan.md)
   §12 (Gemma: GELU-tanh, a single KV head, the 512 window), §9 (Llama-3.2:
   memory preflight, K = 8192 in `down_proj`).

**Lab.** Export a truncated model and look at what it became:
```sh
source compiler/env.sh
$SA_PY compiler/frontend/hf_generic.py --model build/llm_cache/gemma-3-270m \
    --out build/learn/gemma_l2 --layers 2 --quant --rope --attn
```
Open `build/learn/gemma_l2/qllama.mlir` and find the norm, the GELU, the
attention and the KV cache globals.

**Check yourself.**
- Gemma-3-270M has a 262 144-token vocabulary on a 640-wide model. What share
  of its weights is the embedding, and what does that mean for decode time?
- Which of the rewrites change the arithmetic, and which only its form?
- Why does a model with one KV head make the attention a matrix-vector
  product even in prefill?

## Stage 2 - Numerics: an arithmetic you can reproduce bit for bit

**Goal.** Understand why every fp32 step on the device is specified exactly,
and how that turns "close enough" into "bit-exact" testing.

**Read.**
- David Goldberg, *What Every Computer Scientist Should Know About
  Floating-Point Arithmetic*.
- Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 4 (summation
  order).

**In this repo.**
1. [`llm_inference_plan.md`](llm_inference_plan.md) §3: the formats table and
   the fp32 conventions: round to nearest even, denormals flushed, no FMA,
   fixed reduction order, EXP / RECIP / RSQRT by fixed algorithms (so a
   division is RECIP and a multiply).
2. `llm/fp32.py`, `llm/sfu.py`, `llm/test_fp32.py`: bit-exact models of the
   fp32 unit and the special functions.
3. `llm/ref_model.py` `DeviceModel`: the decoder in the device's numerics;
   `red_sum` is the reduction order.
4. `llm/sa_funcsim.py`: the functional simulator of the accelerator, the
   golden reference every board run is compared with.
5. [`generality_implementation_plan.md`](generality_implementation_plan.md)
   §3.1: the five layers of correctness (checkpoint, importer, SA oracle,
   funcsim, board) and which of them are bit-exact.

**Lab.**
```sh
python3 llm/test_fp32.py
python3 llm/test_funcsim.py
python3 llm/ref_model.py build/llm_cache/stories15M.bin build/llm_cache/tokenizer.bin -n 64 --device
```
Change one reduction in `DeviceModel` to `np.sum` and see how many steps stay
bit-exact with the device path.

**Check yourself.**
- Why is a max reduction order-free but a sum is not?
- Why does the plan forbid FMA even though the result would be more accurate?
- Which layers of §3.1 are allowed to differ, and by what measure are they
  judged instead?

## Stage 3 - W8A8 quantization

**Goal.** Quantize a model to int8 weights and int8 activations, and measure
what it costs.

**Read.**
- Jacob et al., *Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference* (2018): scales and int32 accumulation.
- Dettmers et al., *LLM.int8()* (outlier channels).
- Xiao et al., *SmoothQuant* (moving activation outliers into the weights).
- Frantar et al., *GPTQ* (second-order weight rounding).

**In this repo.**
1. `llm_inference_plan.md` §3.1 and §7 (L3): per-output-channel weight
   scales, per-token dynamic activation scales `s_x = amax / 127`, int32
   accumulation, two roundings `float(acc) · s_w · s_x`.
2. `llm/export_w8a8.py`, `llm/eval_quant.py`: the first quantizer and its
   evaluation (top-1, top-5, KL, where greedy text diverges); results in
   `llm/README.md`.
3. `compiler/frontend/qllama.py`: the same quantization as a PyTorch module
   with every step written out ("the compiler never quantizes by itself").
4. `compiler/frontend/hf_generic.py` `quantize`, `QLinear`, `QEmbedding`:
   the generic path's self-quantization.
5. `compiler/frontend/llmc_w8a8.py`: a compressed-tensors W8A8 checkpoint
   made with llm-compressor (SmoothQuant 0.8 + GPTQ, lm_head left in floating
   point).
6. `compiler/frontend/ct_import.py`: imports such checkpoints as they are;
   everything it does not accept is refused with the layer and field, never
   silently re-quantized. Read its docstring's list of accepted formats.
7. `generality_implementation_plan.md` §6 (WP2) and §17.1–17.2: which public
   checkpoints import, and why external checkpoints beat self-quantization
   (mostly SmoothQuant).

**Lab.**
```sh
python3 llm/eval_quant.py build/llm_cache/stories15M.bin build/llm_cache/tokenizer.bin
$SA_PY compiler/tests/quality_eval.py --model build/llm_cache/SmolLM2-135M \
    --ct build/llm_cache/SmolLM2-135M-W8A8 --tokens 1024 --window 256 \
    --variants fp32,w8a8,ct-w8a8
```
Compare self-quantized `w8a8` with the checkpoint's `ct-w8a8`.

**Check yourself.**
- Why does the device use `amax / 127` and the range [−127, 127], not −128?
  (`generality_implementation_plan.md` §3.1 on vLLM's runtime semantics.)
- Why is per-group quantization along K (Q8_0) awkward on a systolic array?
  (`llm_inference_plan.md` §3.1.)
- Which layers does llm-compressor leave unquantized, and what does the
  frontend do with them?

## Stage 4 - KV cache and attention quantization: where quality is lost

**Goal.** Find, by attribution, which part of a quantized model loses the
most, and fix it without breaking bit-exactness.

**Read.**
- Liu et al., *KIVI* (K per channel, V per token) and Hooper et al.,
  *KVQuant*: why the K cache has outlier channels.
- Xiao et al., *StreamingLLM* (attention sinks), for why the first positions
  matter.

**In this repo.**
1. `compiler/tests/quality_eval.py`: the variants `sa:X+Y` keep parts of the
   device form in fp32 (`act`, `kv`, `q`, `probs`, `sfu`) to attribute the
   loss; `kch` / `vch` are per-channel KV scales.
2. `generality_implementation_plan.md` §8 (WP4) results: the int8 KV cache is
   the main loss; K per channel with calibration on 1024 train tokens moved
   Qwen3's top-1 from 71.5% to 87.4%.
3. `hf_generic.py` `calibrate_kv`, `kv_calib_arg`, `--kv-channel`,
   `--kv-calib`: KV scales from WikiText-2 train text, never from the test
   split.
4. `generality_implementation_plan.md` §12.4: Gemma's loss traced to the
   int8 attention probabilities; *qsum*: p = i8(e · 127), normalized by the
   int8 sum instead of the fp32 one (KL −25% to −31% on three models, same
   speed).
5. `iree_compiler_plan.md` §8.22: KV scales calibrated on at least the
   context length (Qwen3 board KL at 2048: 0.073 → 0.055).

**Lab.**
```sh
$SA_PY compiler/tests/quality_eval.py --model build/llm_cache/Qwen3-0.6B \
    --ct build/llm_cache/Qwen3-0.6B-W8A8 --tokens 1024 --window 256 \
    --variants fp32,ct-sa,ct-sa:kv,ct-sa:kch
```
Which variant recovers most of the gap?

**Check yourself.**
- Why does qsum keep the device bit-exact with the oracle when online softmax
  would not? (`iree_compiler_plan.md` §8.22, "Why no online softmax".)
- Why must the calibration text be at least as long as the context the model
  is evaluated at?
- With the KV scales static per layer (per channel for K), what happens to a
  token whose values exceed the calibrated range?

## Stage 5 - The inference system: one token on the board

**Goal.** Follow one token from the ARM to the accelerator and back:
parameter block, command ring, firmware, descriptor list, completion,
interrupt.

**Read.**
- NVIDIA's FasterTransformer / TensorRT-LLM design notes, or the vLLM paper
  (Kwon et al., *PagedAttention*) for how a production server lays out a
  step; [`llm_inference_plan.md`](llm_inference_plan.md) §1 compares this
  project's structure with them.

**In this repo.**
1. `llm_inference_plan.md` §2 (the data flow of one token) and §5 (L1: the
   command ring, `firmware/rt/rt_fw.c`, BASE / PARAM / LOOP / CALL /
   LDPARAM).
2. `llm/compile_model.py` (L4): the whole decoder as one static list; per
   token only the PARAM block and a (pos, token) argument block change.
3. `llm/runtime.py` `LlamaDevice` (L5): load once into the device window,
   submit per token, wait for the notify interrupt, sample.
4. `llm/compile_batch.py`, `runtime.BatchLlamaDevice` (L5b): 8 sequences per
   step, each bit-exact with its own single-sequence run.
5. [`memory_model.md`](memory_model.md): the address maps, coherency of the
   uncached device window, the command ring's ordering.
6. The IREE version of the same loop: `compiler/runtime/tools/sa_llm_run.c`
   (`--abi=hfv`, `--pad`, `--prefill`, `--rvsched`).

**Lab.**
```sh
python3 llm/test_l4.py          # the static list token after token on the simulator
python3 llm/test_l5b.py         # 8 sequences per step
```
Then read `sa_llm_run.c`'s flags and match each to a field of the L1 design.

**Check yourself.**
- What does the ARM write per token in L4, and why is the list itself never
  rewritten?
- Where does the position-dependent address of the KV write come from?
  (LDPARAM, `llm_inference_plan.md` §5.3.)
- Why does L5b decode 8 sequences for nearly the price of one?

## Stage 6 - Prefill and decode

**Goal.** Know why prefill and decode are different programs, and how to make
them agree bit for bit.

**Read.**
- Pope et al., *Efficiently Scaling Transformer Inference* (2022): the
  arithmetic intensity of prefill against decode.
- Agrawal et al., *Sarathi* (chunked prefill).

**In this repo.**
1. `iree_compiler_plan.md` §8.13 (C6.P): prefill in chunks of M = 16 rows
   (one array height), a separate `prefill` export, the KV cache shared.
2. `iree_compiler_plan.md` §8.19 and `hf_generic.py` `_attention_rows`:
   prefill attention row by row, so each row computes exactly what decode
   computes at that position.
3. `sa_llm_run.c` `--prefill`, `--prefill_all_out`: chunks at 0, M, 2M, …;
   `prefill_all` returns every position's logits (teacher forcing on the
   device, stage 10).
4. [`gemv_walkthrough.md`](gemv_walkthrough.md): decode's linear layer on
   the streaming GEMV unit (four DMA ports), from PyTorch to the descriptor.
5. Commit 393ee81, the prefill-vs-decode mismatch: prefill's chunk tiles
   are now capped by the ACC banks, and dispatches that had fallen back to
   the host with `exp` / `div` (libm, not the SFU) broke bit-exactness.

**Lab.** On the simulator, check that prefill + decode gives the same logits
as decode only:
```sh
SA_D=16 SA_GEMV_PORTS=4 $SA_PY compiler/tests/test_hf_generic.py \
    --model build/llm_cache/SmolLM2-135M --out build/learn/smollm2_l2 \
    --layers 2 --quant --rope --attn --prefill 16
```

**Check yourself.**
- For a 2032-token prompt, how many prefill chunks run, and why is the last
  one at P − M rather than at a multiple of M?
- Why is decode's linear layer a GEMV and prefill's a GEMM, and which unit
  runs each?
- What would break if prefill used a different reduction order from decode?

## Stage 7 - Long context

**Goal.** Run 2048 positions: dynamic attention length, tiling over T, device
memory, and what the window of Gemma means.

**Read.**
- Dao et al., *FlashAttention* (tiling attention over T; why this project
  does not use online softmax is in stage 4).
- Beltagy et al., *Longformer*, and the Gemma 3 report for sliding windows.

**In this repo.**
1. `generality_implementation_plan.md` §13.1 (WP9): attention over the
   actual length (`--abi=hfv --pad=16`): decode reads T rounded up to 16, not
   the whole cache.
2. `iree_compiler_plan.md` §8.22: T tiled in blocks of 128 in a hardware
   loop whose count comes from T; p · V on the GEMV unit with a packed V
   cache; the software-pipelined T loop; on the board decode at T = 2048
   halves.
3. `compiler/tests/mem_preflight.py`: what the accelerator pool must hold,
   item by item, from the safetensors header only (calibrated on the board
   within 2.5%).
4. `compiler/scripts/deploy_longctx.sh` (`qwen3`, `llama`, `gemma`,
   `smollm2`): bundles compiled for 2048 (Gemma 512), KV calibrated on 2048
   train tokens.
5. `compiler/tests/board_longctx.py`: a prompt that fills the context;
   prefill against decode only, bit-exact; per-position loss.
6. [`results_v2.2.md`](results_v2.2.md) §4: decode ms per step from T = 64
   to 2048 for all four models.

**Lab.**
```sh
$SA_PY compiler/tests/mem_preflight.py build/llm_cache/Llama-3.2-1B-Instruct-W8A8 --t 2048
$SA_PY compiler/tests/test_long_context.py --model build/llm_cache/gemma-3-270m --max-len 512
```

**Check yourself.**
- Why does Gemma stop at 512 while the others go to 2048? (Two reasons: the
  window and the frontend.)
- Llama-3.2-1B at 2048 uses 1219 of 1259 MB. Which items dominate?
- Why does decode time grow linearly with T, and with what slope for each
  model? (`results_v2.2.md` §4.)

## Stage 8 - Sampling, chat and the control core

**Goal.** Close the loop: greedy and chat generation on the board, and the
loop moved off the ARM.

**In this repo.**
1. `compiler/tests/board_generate.py`: interactive generation, `--chat`
   (Llama-3.2-1B-Instruct, Qwen3), driven by `model.json`
   (`compiler/tests/stage_model_json.py`: stop tokens, chat template ids).
2. `compiler/frontend/argmax_module.py`: the argmax as a module of its own
   (block maxima), so the control core can finish it.
3. [`riscv_backend_design.md`](riscv_backend_design.md) §7.1 (plan B): the
   ARM records each step's descriptor lists once per length bucket; the
   PicoRV32 patches them, submits them and takes the argmax.
   `compiler/runtime/sa/sa_rvsched.c`, `firmware/include/rvsched.h`.
4. `compiler/tests/r5_kernel.py`: the block argmax compiled by IREE's
   llvm-cpu for rv32 and run inside plan B's loop (R5).
5. `results_v2.2.md` §6: plan B matches the ARM on three models and beats it
   on SmolLM2.

**Check yourself.**
- Why does recording the lists once per *bucket* of positions suffice?
- What does the ARM still do while the core runs the loop?
- Why is plan B faster than the ARM on the smallest model only?

## Stage 9 - Performance: decode is a bandwidth problem

**Goal.** Predict a model's decode time from its size and the read bandwidth,
then explain the rest with counters.

**Read.**
- Williams, Waterman, Patterson, *Roofline* (2009).
- Kim et al., *Full Stack Optimization of Transformer Inference: a Survey*
  (2023), on memory-bound decode.

**In this repo.**
1. `results_v2.2.md` §5: bytes per decode step and the read rate from the
   accelerator's event counters: 66–96% of the four-port peak (58 B/cycle).
2. `compiler/tests/kv260_fixed.py`, `generality_implementation_plan.md`
   §13.4: the per-step fixed part (descriptor fetch, fences, the ARM's
   runtime).
3. `compiler/tests/dispatch_stats.py` and `generality_implementation_plan.md`
   §13.2: decode time per dispatch category (GEMV 71–93%).
4. `compiler/tests/perf_model.py`, [`k2b_gemv_design.md`](k2b_gemv_design.md)
   §8.1: a command-level model calibrated within a few percent, and its
   what-if table; §17.6 picks W4A8 as the next hardware investment.
5. [`kv260_upgrade_plan.md`](kv260_upgrade_plan.md): the steps that got
   here (128-bit DMA, batched SFU, one to four GEMV ports).

**Lab.** From `results_v2.2.md`'s table, compute bytes per step / (58 B/cycle
× 100 MHz) for each model and compare with the measured device time. Then
read one `profile_batch/perf.csv` and recompute the read rate as
`results_v2.2.md` §5 does (bytes = 8 × `PC_LD_BEATS`).

**Check yourself.**
- At 5.8 GB/s, what is the decode floor of a 1.2 GB model?
- Why does SmolLM2 reach only 66% of the peak?
- What would W4A8 change in decode, and what in prefill?

## Stage 10 - Measuring quality and running on the board

**Goal.** Report quality and speed that someone else can reproduce: fixed
data, separate calibration and test splits, and a log for every number.

**Read.**
- Jurafsky & Martin, *Speech and Language Processing*, ch. 3 (perplexity).
- The WikiText-2 paper (Merity et al., *Pointer Sentinel Mixture Models*).

**In this repo.**
1. `generality_implementation_plan.md` §3.2–3.3: quality metrics (top-1,
   KL, PPL) and the four kinds of coverage.
2. `compiler/tests/board_quality.py`: `prep` (WikiText-2 windows and the fp32
   reference's argmax, loss and top-K), `sim`, `score`; and
   `compiler/tests/board_teacher.py`, its board side (through `prefill_all`,
   or `--decode` step by step when it does not fit).
3. `compiler/scripts/build_board_bundles.sh`: the release bundle set
   (`smollm2`, `gemma`, `qwen3`, `llama`, `<model>_rv`); every bundle carries
   a `VERSION` file.
4. `compiler/tests/board_run.py`: one entry point for the board tests
   (`llm`, `teacher`, `profile [--batch]`, `longctx [--sweep]`, `rvsched`,
   `rvcode`, `rvperf`, `generate`), each run in
   `results/<commit>/<test>/<overlay>/<mode>/`;
   `compiler/scripts/run_board_tests.sh` runs a bundle's list and skips what
   passed.
5. [`results_v2.2.md`](results_v2.2.md): the consolidated table, with its
   provenance section (§7) as the model for reporting.

**Lab.** Score a board run from its saved file:
```sh
$SA_PY compiler/tests/board_quality.py score --bundle build/deploy_hfgen_qwen3g_pcpi
```
(from a bundle whose `tf_board.npz` came back from the board). Then read the
same model's `longctx` log and explain why its PPL differs from the teacher
score.

**Check yourself.**
- Why is the top-1 agreement measured against fp32's argmax, not against the
  text?
- Why are the calibration windows drawn from the train split?
- What does a `VERSION` line marked `(dirty)` tell a reader, and how does
  `results_v2.2.md` §7 deal with it?

## Stage 11 - Where to go next (capstone ideas)

Each is sized as a project of its own; the documents named say what is known.

- **W4A8 decode** (`generality_implementation_plan.md` §17.6, `k2b_gemv_design.md`
  §8.1): halve the bytes per step; needs an unpack stage in front of GEMV.
- **Gemma beyond 512**: a window mask on the local layers or a ring-buffered
  KV cache (`iree_compiler_plan.md` §8.22, "Out of scope").
- **Batched decode on the generic path**: L5b did it by hand; what does IREE
  need to batch sequences with separate positions?
- **Sampling on the control core**: top-k / temperature in rt_fw or as an R5
  kernel, so the ARM only sees tokens.
- **Speculative decoding**: a draft model needs the target's vocabulary, so
  SmolLM2 (49 152 tokens) cannot draft for Llama-3.2-1B (128 256); a
  layer-truncated Llama-3.2 (`--layers`) can. A prefill chunk verifies up to
  16 drafted tokens in one call, but costs about 4.6 decode steps on Llama
  (1042 against 227 ms): work out the acceptance rate that pays first.
- **A fifth architecture** (a Phi or a Mistral-style model): run it through
  `hf_generic.py` and `mem_preflight.py`, list what fails, and fix the
  frontend, not the model.

---

## Appendix A - Lessons from this project

- **The int8 KV cache, not the weights, was the main quality loss** (WP4);
  attribution by keeping one part in fp32 at a time found it in an afternoon.
- **Calibrate on the context you evaluate at**: scales from 256-token windows
  clipped at 2048 (Qwen3 KL 0.073 → 0.055 once calibrated on 2048 tokens).
- **Normalize what you quantized**: dividing int8 probabilities by their fp32
  sum biased every row; dividing by the int8 sum (qsum) fixed Gemma.
- **Prefill and decode must share every rounding**: a prefill chunk that
  split ACC differently, or a dispatch that ran on the host with libm `exp`,
  broke bit-exactness (393ee81).
- **The board sees what the simulator cannot**: a strided write into the KV
  cache became a per-step host copy (640 ms per Qwen3 step); now every build
  runs `compiler/tests/host_copies.py`.
- **Per-head permutations are easy to get wrong**: RoPE's pair order must be
  applied to `q_norm` / `k_norm` weights too (`qhf.py` `load_weights`).
- **Numbers need provenance**: every row of `results_v2.2.md` names its
  bundle, commit and log, because bundles are assembled in steps.

## Appendix B - Glossary

| Term | Meaning here |
|---|---|
| W8A8 | int8 weights (per output channel) and int8 activations (per token, dynamic) |
| compressed-tensors | the checkpoint format of llm-compressor and vLLM; `ct_import.py` reads its int8 form |
| KV cache | the K and V of every past position, int8 with static scales (K per channel) |
| qsum | attention probabilities p = i8(e · 127) normalized by their int8 sum |
| GQA | grouped-query attention: several query heads share one KV head |
| prefill / decode | the prompt in chunks of 16 rows / one token per step |
| T, `valid_len`, `--pad` | the attention length; the compiled maximum; T rounded up at run time |
| teacher forcing | feeding the reference tokens and scoring each position's prediction |
| top-1 vs fp32 | share of positions where the device's argmax equals the fp32 model's |
| bucket | a range of positions with the same padded T: plan B records lists per bucket |
| bundle | a directory staged for the board: overlay, firmware, `sa.vmfb`, parameters, tests, `VERSION` |
| plan A / plan B / R5 | dispatches as rv32 code / the decode loop on the PicoRV32 / an llvm-cpu kernel on the core |

## Appendix C - References

- Vaswani et al., *Attention Is All You Need*, 2017.
- Touvron et al., *LLaMA* and *Llama 2*, 2023; Meta, *Llama 3* report, 2024.
- Gemma Team, *Gemma 3 Technical Report*, 2025; Qwen Team, *Qwen3 Technical
  Report*, 2025; Allal et al., *SmolLM2*, 2025.
- Su et al., *RoFormer*, 2021; Zhang & Sennrich, *Root Mean Square Layer
  Normalization*, 2019; Shazeer, *GLU Variants Improve Transformer*, 2020;
  Ainslie et al., *GQA*, 2023.
- Jacob et al., *Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference*, 2018; Dettmers et al., *LLM.int8()*,
  2022; Xiao et al., *SmoothQuant*, 2023; Frantar et al., *GPTQ*, 2023.
- Liu et al., *KIVI*, 2024; Hooper et al., *KVQuant*, 2024; Xiao et al.,
  *Efficient Streaming Language Models with Attention Sinks*, 2024.
- Dao et al., *FlashAttention*, 2022; Kwon et al., *Efficient Memory
  Management for Large Language Model Serving with PagedAttention*, 2023;
  Pope et al., *Efficiently Scaling Transformer Inference*, 2022.
- Williams et al., *Roofline*, 2009; Merity et al., *Pointer Sentinel Mixture
  Models* (WikiText), 2016.
- Andrej Karpathy, `llama2.c`; HuggingFace `transformers`; vLLM;
  llm-compressor.

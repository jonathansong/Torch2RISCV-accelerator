# llm/: LLM inference software (host side)

This directory implements level L0 of
[docs/llm_inference_plan.md](../docs/llm_inference_plan.md).

| File | What |
|---|---|
| `checkpoint.py` | Loads llama2.c fp32 checkpoints (the legacy `.bin` format). |
| `tokenizer.py` | The llama2.c BPE tokenizer (`tokenizer.bin`), ported from `run.c`. |
| `ref_model.py` | Reference models. `Fp32Model` is `run.c` in NumPy float32. `DeviceModel` uses the plan §3 numerics: per-channel W8A8, a static int8 KV cache, fp32 VE steps and the fixed reduction order. |
| `eval_quant.py` | Measures the accuracy of `DeviceModel` against fp32. It calibrates the KV scales, then reports teacher-forced top-1 and top-5 agreement, KL divergence, and where free-running greedy text first diverges. |
| `sa_funcsim.py` | Functional simulator of `rtl/sysarray`. It runs descriptor lists bit-exactly (M5 hardware) and is the golden reference for the later levels. |
| `fp32.py`, `sfu.py`, `test_fp32.py` | L2: bit-exact models of the fp32 arithmetic and the special functions (EXP, RECIP, RSQRT). |
| `export_w8a8.py` | L3: exports a checkpoint as a `.w8a8` device file. Weights are int8 per channel, pre-packed as B tiles for D. The file also holds the KV parameter blocks and the RoPE tables. |
| `compile_layer.py` | L3: builds descriptor lists for activation quantization and W8A8 linear layers. It uses chunked, double-buffered weights, BASE relocation, and LOOP_END + PARAM for long layers. |
| `test_l3.py` | L3: runs real stories15M layer inputs through the lists in the functional simulator. The results must be bit-exact with `DeviceModel(sfu=SfuExact)` and close to fp32. `--save` writes the cases for the board test. |
| `compile_model.py` | L4: emits the whole decoder as one static list. Per token only the PARAM block and a (pos, token) argument block change; the device derives its own addresses with LDPARAM. Attention runs as a head loop. |
| `test_l4.py` | L4: runs the list token after token in the functional simulator. Logits and KV cache must be bit-exact with `DeviceModel(sfu=SfuExact)`. Models: a random tiny model and stories15M. `--save` writes the golden run for the board. |
| `runtime.py` | L5: `LlamaDevice` is the ARM runtime. It loads the model once into CMA, then per token submits the static list through the rt_fw ring, waits for the notify interrupt and samples on the ARM (run.c's `Sampler`). |
| `prepare_l5.py` | L5: builds the reference data for the board acceptance: golden logits MD5s, and the fp32 sequences with their top-5 for the accuracy test. |
| `test_funcsim.py` | Checks the functional simulator against the RTL co-simulation cases, the driver's golden models, control flow and error codes. |

All scripts run on the host and need only NumPy. The checkpoints and
`tokenizer.bin` are staged in `build/llm_cache/` by
`notebooks/llm/prepare_llm.sh`.

```sh
python3 llm/test_funcsim.py
python3 llm/ref_model.py build/llm_cache/stories15M.bin build/llm_cache/tokenizer.bin -n 256 [--device]
python3 llm/eval_quant.py build/llm_cache/stories{15M,42M,110M}.bin build/llm_cache/tokenizer.bin --out eval.json
```

## L0 results (host)

- **fp32 reference**: stories15M with greedy decoding for 256 steps, prompt
  "Once upon a time". The text is byte-identical to `run.c`.
- **Functional simulator**: 238 checks at D = 8 and 16 all pass. They
  cover:
  - the 7 RTL co-simulation cases, byte-identical;
  - 80 random GEMM lists and 120 random vector lists, identical to the
    driver goldens;
  - control flow (JUMP, relocation, END, count limit);
  - 10 error cases.

  Four injected bugs were all caught: requant rounding, INTERLEAVE
  addressing, src2 period, and EX accumulate.
- **Quantization accuracy** (`eval_quant.py`, `DeviceModel` against fp32):
  - KV scales are calibrated on 4 prompts and the models are evaluated on 8
    different prompts, 256 steps each;
  - "top-1" is teacher-forced agreement: both models are fed the fp32 token
    sequence and their argmax predictions are compared at every position;
  - "diverges at" is the first differing token of free-running greedy text,
    one value per evaluation prompt.

| Model | KV scale | Top-1 | Device argmax in fp32 top-5 | KL mean / p99 | Greedy text diverges at |
|---|---|---|---|---|---|
| stories15M | absmax | 95.6% | 100% | 0.0113 / 0.094 | 35 34 23 9 40 27 100 16 |
| stories15M | p99.99 | **96.3%** | 100% | 0.0113 / 0.075 | 55 34 85 9 80 106 59 57 |
| stories42M | absmax | 96.0% | 100% | 0.0073 / 0.063 | 42 25 11 19 8 83 17 48 |
| stories42M | p99.99 | 95.9% | 100% | 0.0096 / 0.093 | 42 25 12 52 8 26 16 19 |
| stories110M | absmax | 97.9% | 100% | 0.0036 / 0.038 | 53 117 115 62 21 176 33 63 |
| stories110M | p99.99 | 98.1% | 100% | 0.0039 / 0.038 | 34 117 62 212 76 176 108 19 |

All models pass the 95% gate. Every device argmax is in the fp32 top 5.
Accuracy improves with model size. The two KV calibrations differ by less
than 1 point. stories15M with p99.99 KV scales (96.3%) is the baseline for
the later levels.

## L3 results

```sh
python3 llm/export_w8a8.py build/llm_cache/stories15M.bin --kv build/llm_cache/kv/stories15M_kv_p99.99.npy --d 8 -o build/llm_cache/stories15M_d8.w8a8
python3 llm/test_l3.py build/llm_cache/stories15M.bin build/llm_cache/stories15M_d8.w8a8 --kv build/llm_cache/kv/stories15M_kv_p99.99.npy --save l3_cases.npz
```

On the board (`notebooks/llm/l3_linear_demo.py`, L2 bitstream):

- The 9 linear layers tested are Wqkv, Wo, W13 and W2 of layers 0 and 5, plus the classifier. All are bit-exact with the functional simulator and with `DeviceModel`, and 0.37–3.27% from fp32.
- The classifier (288 → 32000) takes 1.26 M cycles and moves 7.32 weight bytes per cycle, with LD busy 93.9% (the plan requires ≥ 90%).
- See plan §7.3 for the per-layer table.

## L4 results

```sh
python3 llm/test_l4.py --d 8                    # tiny random model, 16 positions
python3 llm/test_l4.py --checkpoint build/llm_cache/stories15M.bin --kv build/llm_cache/kv/stories15M_kv_p99.99.npy --tokens 12 --save l4_golden.npz
```

On the board (`notebooks/llm/l4_decoder_demo.py`, L2 bitstream), stories15M runs entirely on the device:

- All 36 tokens (16 prompt + 20 greedy) are bit-exact with `DeviceModel`, and the text matches.
- Each token takes 2.48 M cycles: 49.6 ms, or 20.2 tok/s. For comparison, the ARM llama2.c int8 build runs at 22.9 tok/s with 2 threads.
- See plan §8.7 for the details.

## L5 results

On the board, `notebooks/llm/l5_generate.py` runs stories15M end to end on the device, with every token waited for by the notify interrupt:

- **Correctness**: 128 greedy tokens after the prompt are bit-exact with `DeviceModel`, and the text is identical.
- **Accuracy**: teacher-forced top-1 against fp32 is 96.2% over 1,554 positions, above the 95% gate.
- **Speed**: sampled generation runs at 15.1 tok/s wall clock (18.9 tok/s on the device).
- **Stability**: 10 × 256 tokens with 0 errors and 2,560 interrupts received.
- See plan §9.3 for the details.

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

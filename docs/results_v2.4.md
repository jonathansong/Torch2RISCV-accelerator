# KV260 results v2.4 (tag `v2.4-kv260`)

A trained mixture-of-experts model on the board. IBM's
Granite-3.1-1B-A400M (24 layers, 32 experts per layer, 8 per token, 1.3B
parameters with about 400M read per token) compiles from the unmodified
HuggingFace checkpoint through the same flow as the four dense models of
[`v2.3-kv260`](results_v2.3.md). Every executable runs on the accelerator, the
board's logits are bit-exact with the simulator, and its quality under
teacher forcing on the board is within 2.2% of fp32 PPL. Every number was
measured on the board on 2026-10-25 and comes from a log named in §6.

Since `v2.3-kv260`:

- **MoE lowered natively** ([`moe_native_lowering_design.md`](moe_native_lowering_design.md)
  S0–S8, §21–§25):
  - each MoE layer becomes ordinary SA dispatches in the same module as the
    rest of the model;
  - the expert tables are packed at compile time, and each expert's weights
    are a dynamic slice whose DDR offset the device computes from the id;
  - the router's Top-k (k ≤ 16 of 4 to 64 experts) is compiled to RISC-V
    code that the PicoRV32 runs between descriptor lists;
  - nothing is read back to the ARM.
- **The accelerator's pool** grew from 1280 to 1536 MiB, and the global CMA
  shrank from 256 to 128 MiB ([`kv260_upgrade_plan.md`](kv260_upgrade_plan.md) K5-M).
- **The dense models are unchanged.** The compiler changes since v2.3 touch
  only MoE patterns. The golden corpus keeps every dense dispatch
  byte-identical, and the v2.4 runtime gives Llama-3.2-1B's v2.3 bundle the
  same logits and speed (§4). So the dense numbers of [`results_v2.3.md`](results_v2.3.md) hold.
- **Retired**: the hand-written MoE path M0–M4 and M5's compiler-owned
  chain (tag `moe-m0-m4-final`).

## 1. Setup

| | |
|---|---|
| Board | AMD Kria KV260, Ubuntu 22.04 (kernel 5.15.0-1080), u-dma-buf pool 1536 MiB at `0x1000_0000`, global CMA 128 MiB |
| Overlay | `d16_100mhz_w128_p6_gemv_np4_pcpi` (D = 16, 4 GEMV ports, 100 MHz), as v2.2.1 / v2.3 |
| Firmware | `rt_fw.bin` v7, byte-identical to v2.3's |
| Runtime | `sa-llm-run` (aarch64) of this release: the expert bases and the `rv_only` policy (the Top-k as the executable's embedded RISC-V code) |
| Compiler | IREE `e4a3b0405d` + `compiler/patches/iree` 0001–0005; `compile_sa.sh` defaults plus `--iree-sa-form-moe` |
| Model | `ibm-granite/granite-3.1-1b-a400m-base`, W8A8 by the frontend (`hf_generic.QMixtralMoe`), int8 KV cache; decode only (§5) |

## 2. Granite-3.1-1B-A400M on the board

Two bundles of the same model:

- **Speed** (`build/deploy_moe_granite24`): a 64-position cache, the KV scales
  on 3 token ids. For bit-exactness and speed.
- **Quality** (`build/deploy_moe_granite24q`): the dense models'
  long-context settings. K per channel, the KV scales on 2048 WikiText-2
  train tokens, a 2048-position cache.

| | Speed bundle | Quality bundle |
|---|---:|---:|
| Executables (all on the accelerator) | 60 | 60 |
| Logits vs the simulator (decode, 28 steps) | bit-exact 28 / 28 | bit-exact 28 / 28 |
| Decode, ms per step | 104.4 (prompt of 24, cache 64) | 111.3 (after a 32-token prompt) |
| Decode, tok/s | 9.58 | **8.98** |
| of which device / rest of the runtime, ms | | 103.2 / 5.8 |
| Device: descriptor lists / Top-8 calls, ms | | 94.1 / 9.1 (24 calls, 8.8%) |
| Descriptor lists per step | 25 | 25 |
| Device memory peak / window, MB | 1282 / 1315 | 1330 / 1362 |

The decode tok/s is measured as v2.3's main table (`profile 40 --prompt 32
--batch`, wall time). In context:

- Granite reads about 429 MB of weights per step, against Llama-3.2-1B's
  1.24 GB, and decodes 1.9× faster (8.98 tok/s against 4.64).
- The descriptor lists take 94.1 ms against a weight-stream bound of 67 ms.
  The rest is each expert FFN's tail (§22 of the design) and the attention.
- The Top-8 runs on the PicoRV32 at 0.38 ms a call.

## 3. Quality

**Board teacher forcing** (quality bundle; `board_run.py teacher`, two windows
of 2048 tokens of the WikiText-2 test set, scored by `board_quality.py score`
against the fp32 HuggingFace model, as `results_v2.2.md` §3):

| | Granite-3.1-1B-A400M | (v2.2 / v2.3: SmolLM2, Gemma, Qwen3, Llama) |
|---|---:|---|
| Windows | 2 × 2048 | 2 × 2048 (Gemma 2 × 512) |
| PPL (fp32) | **6.528** (6.386) | 13.79 (13.48), 43.59 (43.98), 18.58 (18.77), 13.56 (13.21) |
| top-1 vs fp32 | **92.1%** | 90.7%, 76.9%, 88.4%, 91.9% |
| KL(fp32 ‖ device), top 64 | **0.030** (94.5% of the mass) | Llama 0.031 |

Decode over a 2048-token window takes 157.7 ms per step on average (the
attention grows with the context).

**On the host** (`quality_eval.py --moe-routing`, 2048 tokens in windows of 256,
§21 of the design):
- W8A8 alone: PPL 10.99, against fp32's 10.32.
- The board's numerics (`sa:kch`): PPL 10.90, top-1 90.3%, KL 0.069.
- The W8A8 router picks the same experts as an fp32 router 99.3% of the time
  on the same hidden states.

## 4. The dense models with this release's runtime

Llama-3.2-1B's v2.3 bundle (`deploy_hfgen_llamac_pcpi`, the largest window,
1259 MB), with two changes, one at a time:

| Run | Logits vs the simulator (v2.3's reference) | Decode device time per step |
|---|---|---:|
| v2.3 (`results_v2.3.md` §2) | bit-exact 17 / 17 | 213.1 ms (21,312,418 cycles) |
| the 1536 MiB pool, v2.3 runtime | bit-exact 17 / 17 | 213.14 ms (21,313,966 cycles) |
| the 1536 MiB pool, v2.4 runtime | bit-exact 17 / 17 | 213.12 ms (21,311,518 cycles); 4.64 tok/s wall, as v2.3 |

## 5. Limits of this release

- **Decode only for Granite.** Its prefill rows go through the experts one row
  at a time. At 24 layers, the traced graph of the prefill functions exceeds
  the 10 GB build limit. The board tests run the prompt as decode steps.
  Grouping a prefill chunk's rows by expert (M6) is the next step there.
- **Routing near ties.** In 68 of 672 layer-steps, the 8th and 9th expert
  scores are within 0.01. The host model against itself, with 1e-6 noise on
  the routers' inputs, differs as much as the device does. So bit-exactness
  is checked where it is defined:
  - per dispatch (`dispatch_check`, 60 / 60 at T = 16, 32, 48, 64, 1024 and 2048);
  - plan A against the descriptor lists;
  - the board against the simulator.

  The host model is not a bit-exact reference for this model.
- **Supported MoE form**: softmax over the Top-k logits (Mixtral, Granite), no
  shared expert, k ≤ 16, at most 64 experts.

## 6. Provenance

| Item | Built at | Logs (copied back) | Results |
|---|---|---|---|
| `build/deploy_moe_granite24` (speed) | `35db39f` | `build/board_results/granite_board_35db39f/` | `results/35db39f/` |
| `build/deploy_moe_granite24q` (quality) | `15dcf0e` | `build/board_results/granite_q_15dcf0e/` | `results/15dcf0e/`; scores `build/quality/granite_board_2048.json` |
| Llama, 1536 MiB pool, v2.3 runtime | `6a0640a` bundle | `build/board_results/pool1536_check/` | `results/6a0640a/.../llm.3`, `profile_batch.3` |
| Llama, v2.4 runtime | `6a0640a` bundle, `sa-llm-run` of `15dcf0e` | `build/board_results/v24_runtime_check/` | `results/6a0640a/.../llm.4`, `profile_batch.4` |

Board commands: `python3 board_run.py llm` / `llm --decode-only` / `teacher` /
`profile 40 --prompt 32 --batch` in each bundle (as root, after
`/etc/profile.d/pynq_venv.sh`). The bundles are built by `test_hf_moe.py
--export-only --prefill 0 [--max-len 2048 --kv-channel k --kv-calib
...train.txt:2048]`, `compile_sa.sh` with `--iree-sa-form-moe
--iree-sa-emit-riscv`, `rv_image.py`, `test_moe_sim.py --prefill 0
--host-reported`, `deploy_moe.sh`, and for the quality bundle
`board_quality.py prep --tokens 4096 --window 2048`.

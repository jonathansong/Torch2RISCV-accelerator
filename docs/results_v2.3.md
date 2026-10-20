# KV260 results v2.3 (tag `v2.3-kv260`)

The board numbers after cross-operator fusion: the same four unmodified
HuggingFace models, the same overlay, runtime and firmware as
[`v2.2.1-kv260`](results_v2.2.md), compiled with three fusions on by default
([`fusion_design.md`](fusion_design.md)). Every number was measured on the board
on 2026-10-17..20 and comes from a log named in §7.

Since `v2.2.1-kv260`:

- **SwiGLU in decode** (§4): the two FFN projections, the gating, the amax and
  the int8 output in one dispatch, each chunk's gating overlapping the next
  chunk's weight streaming;
- **attention in decode** (§6.4): the q path, the scores, the probabilities,
  p · V and the int8 output in one dispatch per layer (a region pass after
  IREE's region formation, IREE patch 0004);
- **attention in prefill** (§6.7): a chunk's 16 rows in one kernel per layer,
  each head's cache read and transposed once a chunk instead of once a row
  (the frontend writes the rows' K / V first; IREE patch 0005);
- the arithmetic is unchanged: every fusion is bit-exact with the unfused
  build, so the quality figures of v2.2 (§3) still hold.

## 1. Setup

| | |
|---|---|
| Board | AMD Kria KV260, Ubuntu, runtime aarch64 (`sa-llm-run`, `rt_fw.bin` v7: byte-identical to v2.2.1's) |
| Overlay | `d16_100mhz_w128_p6_gemv_np4_pcpi`, as v2.2 |
| Compiler | IREE `e4a3b0405d` + `compiler/patches/iree` 0001–0005, `compile_sa.sh` defaults (generalization, `SA_FUSE_FFN`, `SA_FUSE_ATTN`, `SA_FUSE_ATTN_CHUNK`) |
| Models | as v2.2: public W8A8 checkpoints, int8 KV cache with K per channel, KV scales calibrated on 2048 WikiText-2 train tokens; exported again by the current `hf_generic` (the parameters byte-identical to v2.2's) |
| Context | compiled for 2048 positions (Gemma 512) |

All four models: every dispatch on the accelerator (0 on the host), board
logits bit-exact with the functional simulator in 17/17 steps, prefill vs
decode-only bit-exact over a prompt that fills the context (17/17 rows, tokens
identical), no per-step host copies of the KV cache, and plan A / plan B on the
RISC-V core bit-exact with the ARM's loop.

## 2. Main table

As v2.2: decode is one token per step after a 32-token prompt (`profile
--batch`, wall time); prefill is one 16-token chunk at T = 64 (device time,
`longctx --sweep`). In parentheses: v2.2.

| | SmolLM2-135M | Gemma-3-270M | Qwen3-0.6B | Llama-3.2-1B |
|---|---:|---:|---:|---:|
| Executables (all on the accelerator) | 1070 (1127) | 699 (739) | 1007 (1051) | 608 (665) |
| Decode, ms per step | 37.1 (42.4) | 56.9 (60.7) | 126.0 (138.7) | 215.6 (226.5) |
| Decode, tok/s | **27.0** (23.6) | **17.6** (16.5) | **7.94** (7.21) | **4.64** (4.42) |
| 　of which device / rest of the runtime, ms | 32.9 / 4.2 | 53.5 / 3.4 | 121.6 / 4.3 | 213.1 / 2.5 |
| Decode against v2.2 | 1.14× | 1.07× | 1.10× | 1.05× |
| Prefill, ms per chunk of 16 | 217.5 (276.7) | 202.9 (233.7) | 617.7 (843.5) | 953.4 (1041.7) |
| Prefill, tok/s | **73.6** (57.8) | **78.9** (68.5) | **25.9** (19.0) | **16.8** (15.4) |
| Prefill against v2.2 | 1.27× | 1.15× | 1.37× | 1.09× |
| Device memory peak / heap, MB | 156 / 190 | 270 / 640 | 690 / 809 | 1220 / 1259 |

Decode gains from SwiGLU and the decode attention (the device time −9.4%,
−5.0%, −7.9%, −4.4%; Llama, with the largest weights, is the most bound by
the GEMV's read rate). Prefill gains mostly from the chunk kernel.

## 3. Quality

Unchanged from v2.2 ([`results_v2.2.md`](results_v2.2.md) §3): the fused builds
compute the same values (logits bit-exact against the unfused builds on the sim,
and against the sim on the board).

## 4. Long context

Decode and prefill at attention length T (`longctx --sweep`; decode wall ms per
step, prefill device ms per chunk of 16); v2.2 in parentheses:

| T | SmolLM2 decode | prefill | Gemma decode | prefill | Qwen3 decode | prefill | Llama decode | prefill |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 37.2 (42.5) | 217.5 (276.7) | 57.0 (60.8) | 202.9 (233.7) | 126.2 (138.9) | 617.7 (843.5) | 215.7 (226.7) | 953.4 (1041.7) |
| 256 | 41.6 (47.2) | 243.6 (314.1) | 59.3 (63.2) | 206.6 (246.0) | 139.0 (152.5) | 669.2 (950.2) | 223.0 (234.6) | 993.1 (1096.0) |
| 512 | 48.8 (54.6) | 277.9 (373.3) | 63.1 (67.2) | 220.0 (277.7) | 161.0 (176.0) | 742.0 (1138.7) | 234.5 (247.0) | 1055.8 (1193.4) |
| 1024 | 62.9 (69.5) | 346.5 (491.8) | | | 205.0 (223.3) | 888.3 (1515.3) | 257.5 (271.9) | 1182.7 (1390.1) |
| 2048 | 91.2 (99.3) | 483.6 (729.0) | | | 293.1 (317.8) | 1181.3 (2269.1) | 303.7 (321.5) | 1437.4 (1785.1) |

A prefill chunk at T = 2048 costs 0.66× / 0.52× / 0.81× of v2.2's (SmolLM2 /
Qwen3 / Llama): the gain grows with T, since v2.2 read and transposed each
head's cache once per row. A whole prompt that fills the context (`longctx`):

| | SmolLM2 (2032) | Gemma (496) | Qwen3 (2032) | Llama (2032) |
|---|---:|---:|---:|---:|
| Prefill, s | 62.9 | 7.19 | 151.2 | 182.9 |
| Prefill, tok/s | **32.3** | **69.0** | **13.4** (v2.2: 7.0) | **11.1** (8.9) |

## 5. Decode at T = 2048 and the device time

| | SmolLM2 | Qwen3 | Llama |
|---|---:|---:|---:|
| Decode at T = 2048, tok/s (v2.2) | 11.0 (10.1) | 3.41 (3.15) | 3.29 (3.11) |
| Device cycles per decode step (`profile --batch`) | 3.29M | 12.16M | 21.31M |

Decode is still bound by the weight read rate (v2.2 §5: the GEMV's LD busy
70–91%); the fusions removed the round trips through DDR between the small
dispatches around the GEMVs, not the weight bytes.

## 6. RISC-V backend (C7)

Decode ms per step (16 steps after a 40-token prompt; and over 256 tokens), all
tokens identical and the last step's logits bit-exact against the ARM's loop:

| | SmolLM2 | Gemma | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| ARM loop, descriptor lists (`llm`) | 37.3 | 57.0 | 126.1 | 215.7 |
| Plan B: loop on the PicoRV32 (`rvsched`) | **34.2** | 58.3 | 125.8 | 215.7 |
| Plan A: every dispatch as rv32 code (`rvcode --embedded`) | 65.9 | 73.9 | 158.3 | 241.9 |
| Plan A inside plan B (`rvsched --rv --embedded`) | 56.6 | 70.4 | 151.9 | 238.5 |
| Over 256 tokens: ARM / plan B | 40.5 / 37.2 | 58.6 / 59.8 | 135.7 / 135.2 | 220.9 / 220.7 |

Plan A's gap to the lists narrowed with the fusions (Qwen3 179.1 → 158.3 ms):
fewer, larger dispatches mean fewer commands issued from DDR-resident code.

## 7. Provenance

| Bundle | Built at | Logs (copied back) | Results |
|---|---|---|---|
| `build/deploy_hfgen_smollm2c_pcpi` | `f767c91` | `build/board_results/board_tests_pcpi_smollm2c/smollm2c.*.log` | `results/5d1a8b5/` |
| `build/deploy_hfgen_gemmac_pcpi` | `f767c91` + the region-pass change committed as `d72a6a1` (`VERSION` says `f767c91`) | `build/board_results/board_tests_pcpi_gemmac/gemmac.*.log` | `results/6a0640a/` |
| `build/deploy_hfgen_qwen3c_pcpi` | `f767c91` | `build/board_results/board_tests_pcpi_qwen3c/qwen3c.*.log` | `results/6a0640a/` |
| `build/deploy_hfgen_llamac_pcpi` | `f767c91` | `build/board_results/board_tests_pcpi_llamac/llamac.*.log` | `results/6a0640a/` |

(The results directories are named after the commit of the bundle the `c` bundle
was copied from, as for v2.2; `VERSION` lists the steps.) Each bundle is the v2.2
`g` bundle (SmolLM2: `smollm2_lc2048`) with `sa.vmfb` and `sa_packed.irpa`
replaced (`build/chunk/bundles_chunk.sh`); `qllama.irpa` of the new exports is
byte-identical to v2.2's, so the bundles' simulator references stay valid. The
commits after them change no output for these exports: `d72a6a1`'s region-pass
change leaves Qwen3's chunk region as it was (176 regions, checked on the 2-layer
build) and was in Gemma's build; `d9e6aa9`'s guard only stops chunks of an earlier
export. v2.2 numbers: [`results_v2.2.md`](results_v2.2.md), and the `g` / `smollm2`
logs in the same copied-back directories. Board tests:
`compiler/scripts/run_board_tests.sh smollm2c gemmac qwen3c llamac`.

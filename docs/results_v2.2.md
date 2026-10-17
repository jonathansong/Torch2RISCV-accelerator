# KV260 results v2.2 (tag `v2.2-kv260`)

The consolidated board numbers behind the talk: four unmodified HuggingFace
models compiled whole by the IREE `sa` backend, long context, and the RISC-V
backend paths (C7). Every number below was measured on the board on
2026-10-16/17 and comes from a log or a results directory named in §7.

Since [`v2.0-kv260`](generality_implementation_plan.md) (context 256) and
`v2.1-kv260-longctx` (Qwen3 / Llama at 2048, Gemma at 512):

- SmolLM2-135M compiled for 2048 positions too;
- one executable per kernel instead of one per layer, on by default
  ([`riscv_backend_design.md`](riscv_backend_design.md) §9.11, IREE patch 0003,
  upstream as iree-org/iree#25059);
- the RISC-V backend paths on all four models: plan B (the decode loop on the
  PicoRV32), plan A (every dispatch as generated rv32 code) and R5 (a kernel
  from IREE's llvm-cpu on the core).

## 1. Setup

| | |
|---|---|
| Board | AMD Kria KV260, Ubuntu, runtime aarch64 |
| Overlay | `d16_100mhz_w128_p6_gemv_np4_pcpi`: 16 × 16 array, 100 MHz, 128-bit DMA, batched SFU, four-port streaming GEMV (also driven over PCPI); overlay commit `a6974fc`, WNS +0.286 ns |
| Compiler | IREE `e4a3b0405d` + `compiler/patches/iree` 0001–0003, `compile_sa.sh` defaults (generalization on) |
| Models | public W8A8 checkpoints (Llama: RedHatAI; SmolLM2 / Qwen3: llm-compressor; Gemma: the project's own W8A8), int8 KV cache with K per channel, KV scales calibrated on 2048 WikiText-2 train tokens |
| Context | compiled for 2048 positions (Gemma 512: its sliding window) |

All four models: every dispatch on the accelerator (0 on the host), board
logits bit-exact with the functional simulator in 17/17 steps, prefill vs
decode-only bit-exact over a prompt that fills the context, and no per-step
host copies of the KV cache.

## 2. Main table

Decode is one token per step after a 32-token prompt (`profile --batch`, wall
time); prefill is one 16-token chunk at T = 64 (device time, `longctx --sweep`).

| | SmolLM2-135M | Gemma-3-270M | Qwen3-0.6B | Llama-3.2-1B |
|---|---:|---:|---:|---:|
| Executables (all on the accelerator) | 1127 | 739 | 1051 | 665 |
| Decode, ms per step | 42.4 | 60.7 | 138.7 | 226.5 |
| Decode, tok/s | **23.6** | **16.5** | **7.21** | **4.42** |
| 　of which device / rest of the runtime, ms | 36.3 / 6.1 | 56.3 / 4.4 | 132.1 / 6.5 | 223.0 / 3.5 |
| Prefill, ms per chunk of 16 | 276.7 | 233.7 | 843.5 | 1041.7 |
| Prefill, tok/s | **57.8** | **68.5** | **19.0** | **15.4** |
| Device memory peak / heap, MB | 155 / 190 | 270 / 640 | 689 / 809 | 1219 / 1259 |

Compared with `v2.0-kv260` (compiled for 256): decode is 1.07×, 1.10×, 1.04×
and 1.09× faster; prefill changes by −4%, +9%, −9% and +2%. The bundles
compiled for 2048 pay a little at short T in prefill's attention
([`iree_compiler_plan.md`](iree_compiler_plan.md) §8.22).

## 3. Quality

**Teacher forcing on the board** (`board_run.py teacher`, WikiText-2 test, two
windows of the context length, scored by `board_quality.py score` against the
fp32 HuggingFace model):

| | SmolLM2 | Gemma | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| Windows | 2 × 2048 | 2 × 512 | 2 × 2048 | 2 × 2048 |
| PPL (fp32) | 13.79 (13.48) | 43.59 (43.98) | 18.58 (18.77) | 13.56 (13.21) |
| top-1 vs fp32 | 90.7% | 76.9% | 88.4% | 91.9% |
| KL(fp32 ‖ board) over the top 64 | 0.025 | 0.213 | 0.055 | 0.037 |

Gemma, Qwen3 and Llama score exactly as at `v2.1-kv260-longctx`: one
executable per kernel changes no arithmetic. Gemma's lower top-1 was traced to its int8
attention probabilities
([`generality_implementation_plan.md`](generality_implementation_plan.md) §12.4).

**One long prompt** (`longctx`, the first 2031 tokens of the test set, 495 for
Gemma; the board's own prefill, then the per-position next-token loss):

| | SmolLM2 | Gemma (495) | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| PPL (fp32) | 11.29 (11.09) | 35.20 (31.17) | 14.17 (14.17) | 10.06 (9.81) |
| top-1 vs fp32 | 90.7% | 83.5% | 89.2% | 93.3% |

## 4. Long context

Decode, wall ms per step at attention length T (`longctx --sweep`):

| T | SmolLM2 | Gemma | Qwen3 | Llama |
|---:|---:|---:|---:|---:|
| 64 | 42.5 | 60.8 | 138.9 | 226.7 |
| 256 | 47.2 | 63.2 | 152.5 | 234.6 |
| 512 | 54.6 | 67.2 | 176.0 | 247.0 |
| 1024 | 69.5 | | 223.3 | 271.9 |
| 2048 | 99.3 | | 317.8 | 321.5 |

At T = 2048 that is 10.1, 3.15 and 3.11 tok/s; Gemma at its 512 window 14.9.
A 2032-token prompt prefills in 290 s (Qwen3, 7.0 tok/s) and 228 s (Llama,
8.9 tok/s).

## 5. Where decode time goes

The accelerator's event counters over a batched decode step (`profile
--batch`, `perf.csv`; bytes = 8 × `PC_LD_BEATS`; the four-port GEMV peak is
57.95 B/cycle):

| | SmolLM2 | Gemma | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| Device cycles per step | 3.62M | 5.63M | 13.21M | 22.30M |
| Bytes loaded per step | 139 MB | 273 MB | 610 MB | 1245 MB |
| Average read rate | 38.4 B/cycle (3.8 GB/s) | 48.6 (4.9 GB/s) | 46.2 (4.6 GB/s) | 55.9 (5.6 GB/s) |
| 　of the four-port peak | 66% | 84% | 80% | 96% |
| LD busy | 70% | 84% | 79% | 91% |
| VE / SFU active | 24% | 15% | 17% | 8% |
| All engines idle (fetch, fences) | 2.3% | 1.0% | 0.7% | 0.2% |

Decode is bound by the weight read rate, more so the larger the model; the
smaller models lose a part to vector work and to the ARM's per-step runtime
(6 ms, 14% of SmolLM2's step). A prefill chunk is compute bound instead (EX
useful 29–59%, read rate 7–13 B/cycle).

## 6. RISC-V backend (C7)

Decode ms per step, 40-token prompt, 16 steps, all tokens identical and the
last step's logits bit-exact against the ARM's loop
([`riscv_backend_design.md`](riscv_backend_design.md)):

| | SmolLM2 | Gemma | Qwen3 | Llama |
|---|---:|---:|---:|---:|
| ARM loop, descriptor lists (`llm`) | 42.6 | 60.7 | 138.8 | 226.6 |
| Plan B: loop on the PicoRV32, lists (`rvsched`) | **37.8** | 61.2 | 136.7 | 225.7 |
| Plan B + R5 argmax kernel (`rvsched --r5`) | 39.2 | 68.1 | 144.5 | 229.4 |
| Plan A: ARM loop, every dispatch as rv32 code (`rvcode --embedded`) | 81.9 | 82.7 | 179.1 | 259.7 |
| Plan A inside plan B (`rvsched --rv --embedded`) | 67.0 | 76.4 | 166.7 | 253.5 |
| Over 256 tokens: ARM / plan B | 45.9 / 40.9 | 62.4 / 62.8 | 149.0 / 146.5 | 232.3 / 231.1 |
| Decode rv32 working set (plan A code, `rv_codesize.py`) | 24 KB | 20 KB | 34 KB | 41 KB |

- Plan B matches the ARM on the three larger models and beats it on SmolLM2,
  where the ARM's per-step runtime is a larger share.
- Plan A's gap to the lists is instruction fetch from DDR (§9.10: about 27
  cycles per instruction, 9–11 instructions per command); the decode working
  set now fits a 32–64 KB instruction memory, which is the hardware step that
  would close it.
- R5's block argmax (362–368 bytes of rv32 from IREE llvm-cpu) is correct but
  1–8 ms slower per step than rt_fw's hand-written loop.

## 7. Provenance

| Bundle | Model commit | Logs | Results |
|---|---|---|---|
| `build/deploy_hfgen_smollm2_lc2048_pcpi` | `5d1a8b5` (clean; rvsched / rvembed / R5 at `5d1a8b5`) | `build/bundles_pcpi/board_tests_s3/smollm2.*.log` | `results/5d1a8b5/` |
| `build/deploy_hfgen_gemmag_pcpi` | `6a0640a`; sa.vmfb regenerated generalized at `a5aef47` | `gemmag.*.log` | `results/6a0640a/` |
| `build/deploy_hfgen_qwen3g_pcpi` | `6a0640a`; sa.vmfb generalized at `c0af1ec` | `qwen3g.*.log` | `results/6a0640a/` |
| `build/deploy_hfgen_llamag_pcpi` | `6a0640a`; sa.vmfb generalized at `a5aef47` | `llamag.*.log` | `results/6a0640a/` |

The three `g` bundles were assembled in steps (`VERSION` lists each, some from
a dirty tree). Between `6a0640a` and the tag the export, frontend and runtime
changes are RISC-V additions only. Recompiling the three models with the tag's
`compile_sa.sh` defaults gives the same dispatch sources and packed parameters
as the board-accepted generalized builds, and the one compiler fix after them
(`SahlExpandKernels` `firstToEmit`) leaves every per-dispatch binary unchanged
(`--iree-hal-dump-executable-binaries-to`: Qwen3 / Gemma / Llama 2102 / 1478 /
1330 files, 0 different; §9.11). Board tests: `compiler/scripts/run_board_tests.sh qwen3g
gemmag llamag smollm2`; summary in `build/bundles_pcpi/board_tests_s3/summary.txt`.

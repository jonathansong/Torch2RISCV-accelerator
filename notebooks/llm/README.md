# LLM inference — step 1: measure before changing the hardware

Two measurements on the unchanged m5 overlay, to decide the LLM hardware work
(int4 weights on the LD path, VE nonlinear/reduction ops, per-token
descriptor lists):

1. **ARM baseline** (`arm_baseline.sh`): llama2.c on the Cortex-A9. It runs
   fp32 (`run.c`) and Q8_0 int8 (`runq.c`), on 1 and 2 threads, and reports
   tokens/s plus a gprof share of the matmul time.
2. **Accelerator GEMVs** (`llm_gemv_bench.py`): every decode GEMV shape of
   stories15M/42M/110M, run as `C = W @ X` (weights streamed as A strips).
   For each shape it reports:
   - cycles for the PCPI path and the descriptor-list path;
   - weight bytes per cycle;
   - the counter breakdown;
   - per model, the GEMV time per token and the tokens/s bound (batch 1 and
     batch D).

| File | Runs on | What |
|---|---|---|
| `prepare_llm.sh` | host | fetches llama2.c (pinned commit), tokenizer and TinyStories checkpoints; converts to Q8_0; stages `build/deploy_llm/` |
| `quantize_q80.py` | host | NumPy port of llama2.c `export.py --version 2` (legacy fp32 `.bin` → Q8_0 for `runq.c`), no PyTorch needed |
| `arm_baseline.sh` | board | builds and runs llama2.c, writes `arm_baseline.txt` |
| `llm_gemv_bench.py` | board | the accelerator GEMV benchmark; reads `arm_baseline.txt` if present |

```sh
notebooks/llm/prepare_llm.sh                       # host, PYTHON=<python with NumPy>
scp -r build/deploy_llm xilinx@<board>:/home/xilinx/llm
# on the board
cd /home/xilinx/llm && bash arm_baseline.sh
sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh \
              && cd /home/xilinx/llm && python3 llm_gemv_bench.py'
```

stories110M is staged only as Q8_0: the fp32 file (438 MB) does not fit in
the board's 512 MB.

## Results (board, 2026-09-25, m5 overlay, D = 16, 50 MHz)

ARM Cortex-A9 running llama2.c, 256 tokens, greedy decoding. The table gives
tokens/s with 1 thread and with 2 threads.

| Model | fp32 `run` | Q8_0 `runq` | Q8_0 `runq_gs32` |
|---|---|---|---|
| stories15M | 5.5 / 10.5 | 9.9 / 19.1 | 13.1 / 22.9 |
| stories42M | 2.4 / 3.9 | 4.0 / 7.3 | 4.5 / 8.7 |
| stories110M | — | 1.5 / 2.8 | 1.9 / 3.3 |

- **gprof.** matmul takes 91–96% of the time.
- **Why stock `runq` is slow.** The Cortex-A9 has no hardware integer
  divide, and stock `runq` divides by the run-time GS in its inner loop, so
  `__divsi3` takes about 28% of the time.
- **`runq_gs32` removes that.** It fixes GS = 32 at compile time.
- **Implied weight rate.** At 2 threads, `runq_gs32` reads weights plus
  scales at about 375–390 MB/s.

Accelerator decode GEMVs (`llm_gemv_bench.py`): the result matches NumPy for
all 12 shapes.

- **LD is the bottleneck.** LD is busy 91–99.7% of the time and moves
  7.7–7.9 B per busy cycle. EX does useful work only 41–49% of the time, so
  it waits on LD about half the time. Weights arrive at 6.7–7.9 B/cycle
  overall.
- **Descriptor lists are 0.3–4% slower than PCPI here.** The descriptor
  fetch shares the port-0 read channel with LD, so it takes some of the
  weight bandwidth.
- **The classifier (32000 × dim) dominates small models.** It takes 58.5%
  of the GEMV time per token for stories15M, 38.4% for 42M and 22.0% for
  110M.

| Model | GEMV ms/token | tok/s, GEMVs only | batch-16 tok/s | best ARM (2 threads) |
|---|---|---|---|---|
| stories15M | 40.8 | 24.5 | 392 | 22.9 |
| stories42M | 109.1 | 9.2 | 147 | 8.7 |
| stories110M | 283.2 | 3.5 | 56.5 | 3.3 |

**Conclusion.** At batch 1 the accelerator is **no faster than the ARM**.
Its GEMVs alone are only 6–7% faster than all of the ARM's decoding, and the
accelerator numbers leave out attention, norms and the other non-GEMV
operations. Both sides move about 390 MB/s of weights.

The accelerator's advantage is batching: with the same weight traffic,
batch 16 is 17× the ARM.

**What this means for the hardware work.**
- Single-stream decode needs more weights per cycle. int4 weights unpacked
  on the LD path give 16 weights per cycle on one port, about 2×; this is an
  estimate.
- The classifier has to be covered as well as the layers.
- Descriptor fetch must not take LD read bandwidth for weight-streaming
  lists.

# L0: D = 8 baseline for the LLM work (prebuilt)

This is the m5 RTL (performance counters and the descriptor fetch unit)
built at **D = 8**. It is the resource and timing baseline for the LLM
inference plan (`docs/llm_inference_plan.md` §4.1). Built with Vivado 2024.1
(`scripts/build_bitstream.sh -jobs 4 -sa_d 8`).

| | L0 (D = 8) | m5 (D = 16) |
|---|---|---|
| LUT | 23452 (44.1 %) | 48928 (92.0 %) |
| FF | 19193 | 36350 |
| DSP | 96 | 184 |
| BRAM36 | 130 | 130 |
| Setup WNS at 50 MHz | +1.555 ns | +2.460 ns |

**Worst setup path (18.1 ns, 18 logic levels).** It is D-independent and
runs through the LD burst generator in `sa_ld.v`:

1. `off` feeds the address add.
2. The burst length is the minimum of the row remainder, the distance to the
   4 KB boundary, and 16 beats.
3. `words_adv * step` goes through a DSP multiply.
4. The result reaches `cur_word`.

It meets 50 MHz. It has to be pipelined for 75 MHz (L5.5).

Verified on the board (all PASS):

- **`notebooks/m4_demo.py`**:
  - int32 GEMM at 256³: 59.0 MAC/cycle (92.1 % of the 64 MAC/cycle peak);
    256×128×1024: 61.8 MAC/cycle (96.6 %);
  - fused int8 GEMM, vector operations and DMA bandwidth (7.94 B/cycle,
    LD+ST 15.76 B/cycle);
  - legacy firmware: 1024/1024.
- **`notebooks/m4_perf.py`**: 69/69 counter checks.
- **`notebooks/m5_desc_demo.py`**: every PCPI and descriptor-list result is
  bit-exact. List speedups are 8³ 7.31×, 64³ 1.11×, int8 64³ 1.18×, and
  vector 1.04–2.31×. One list reused through BASE relocation 3 times.

Needs `firmware/{gemm,vector,bwtest,matmul,matmul_insn,desc_run}/*.bin` and
`driver/pynq_matmul.py` from the same commit. These binaries are unchanged
from m5.

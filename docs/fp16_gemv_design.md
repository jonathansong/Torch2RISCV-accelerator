# FP16 GEMV on the KV260: design note

Status (2026-11-10): design, not implemented.

A prototype FP16 datapath for decode on the KV260, as a variant of the K2b
streaming GEMV unit ([`k2b_gemv_design.md`](k2b_gemv_design.md)). Weights and
the activation vector are IEEE FP16. Products are exact. Accumulation is FP32,
in an order fixed by this note, so the functional simulator reproduces the
hardware bit for bit. The target workload is decode of an unquantized FP16
checkpoint, Qwen2.5-0.5B, with the rest of the model on the ARM cores at
first.

## 1. Goal and scope

**Goal.** Run FP16 linear layers on the KV260 at the weight-streaming rate,
bit-exact with a functional model, and run an FP16 model's decode end to end
with a hand-written runtime.

**In scope**
- A FP16 mode of LD's GEMV command, and FP16 lanes in `sa_gemv.v`.
- Deterministic burst-to-port assignment in `sa_ld` for that mode (§5).
- The functional simulator, unit and system testbenches.
- A C runtime on the ARM that runs Qwen2.5-0.5B decode, with every linear
  layer on the accelerator (§8, stage P1).
- The overlay at **50 MHz** (`pl_clk0`), which leaves timing margin for the
  FP32 adders and the wider lanes.

**Not in scope (v1)**
- FP16 prefill at speed: the EX array stays int8, and prefill rows go through
  the GEMV one at a time.
- The IREE compiler's FP16 path (f16 kernels, unquantized export).
- Attention and the vector epilogues on the device; they are stage P2 (§8).

**Why Qwen2.5-0.5B.** In FP16 it is about 0.99 GB of weights, which fits the
1536 MiB u-dma-buf pool. Qwen2.5-1.5B (about 3.1 GB) does not fit the board's
4 GB next to Linux. The 0.5B and 1.5B models share the architecture: GQA, q/k/v
biases, RoPE θ = 1e6, RMSNorm, SwiGLU, tied embedding. Expected configuration
(to check against `config.json`): 24 layers, hidden 896, FFN 4864, 14 query
heads and 2 KV heads of 64, vocabulary 151936.

## 2. Numeric contract

| Step | Format and rule |
|---|---|
| Weights W | FP16 in DDR, packed `[N/16, K, 16]` as the int8 GEMV (16 output columns per strip, one k row = 32 bytes) |
| Vector x | FP16 in SPAD_A, 8 elements per 16-byte word, contiguous |
| Product x[k] · W[k][j] | **exact**, as FP32. An FP16 × FP16 product has a 22-bit significand and a magnitude in [2^-48, 2^32), so it is representable in FP32 without rounding, FP16 subnormal inputs included |
| Accumulation | FP32 adds, `sa_fp32_add` semantics (round to nearest even, flush to zero below the FP32 normal range, canonical NaN), in the order of §4 |
| Result | FP32 in ACC, one 16-column word per strip, as the int8 GEMV |
| Specials | an FP16 Inf or NaN input gives the product's IEEE value (Inf, or the canonical NaN for NaN and 0 × Inf); the FP32 adds then follow `sa_fp32_add` |
| FP32 → FP16 (producing x) | round to nearest even, FP16 subnormals kept, overflow to ±Inf, as PyTorch's `.half()` |

Everything outside the GEMV stays FP32 (on the ARM in P1, on the VE in P2).
The only roundings are therefore: x to FP16, each FP32 add, and the final
results. This is closer to an FP32 reference than an all-FP16 pipeline.

## 3. Command

LD's GEMV mode (mode 2, [`k2b_gemv_design.md`](k2b_gemv_design.md) §4) with one
new flag:

| Field | Position | FP16 meaning |
|---|---|---|
| `f16` | w3[59] (new) | 1: FP16 lanes, the semantics below |
| `row_bytes` | w2[63:48] | K × 32 (bytes per strip); hence **K ≤ 2047** |
| `ddr`, `pitch` | w1, w3[31:0] | multiples of **256** bytes (one burst, §5) |
| `rows`, `xword`, `cstep`, `acc` | as int8 | S ≤ 8 strips, x from SPAD_A word `xword` (K / 8 words), result at C + s·cstep, `acc`: add the old ACC word |

Checks (`XERR_SHAPE` otherwise):
- K a multiple of 8, so a strip is a whole number of 256-byte bursts;
- `ddr` and `pitch` 256-byte aligned;
- `f16` set only when the unit reports FP16 lanes.

The internal LD packet gains the flag bit. A capability bit (CAPS bit 27,
proposed; to check that it is free) reports FP16 lanes.

A longer K is split as the int8 GEMV splits it (WP6, `SahlExpandKernels.cpp`),
in blocks of at most 2040 with `acc = 1`. Qwen2.5-0.5B's down projection
(K = 4864) takes three blocks: 2040, 2040, 784. The x buffer stays 4 KB per
lane (256 words), which holds 2048 FP16 values, so no URAM is needed.

**Semantics** (for s < S, j < 16):

```
ACC[C + s*cstep][j] = order_§4( (acc ? ACC_old : absent), products x[k] * W_s[k][j], k < K )
```

## 4. Summation order (normative)

Integer accumulation was associative, so the int8 GEMV never had to fix an
order. FP32 accumulation is not, so the order is part of the semantics. The
simulator implements exactly this order.

1. **Bursts.** Strip s is `K / 8` bursts of 16 beats (256 bytes, 8 k rows).
   Number the bursts of the command globally, in issue order: g = s·(K/8) + b
   for burst b of strip s. **Burst g goes to port and lane p = g mod P**
   (P = number of read ports).
2. **Beats.** Within a burst the beats are (k, half) = (k0, lo), (k0, hi),
   (k0+1, lo), … for 8 consecutive k. Half lo carries columns 0–7, half hi
   columns 8–15.
3. **Lane partial sums.** Each lane keeps, per strip s, column j and parity
   q = k mod 2, one FP32 partial sum `A[p][s][q][j]`. The contributions of k
   rows go into it in increasing k. The first contribution is stored, not
   added to +0, which keeps the sign of a zero exactly as `sa_fp32_add` would.
4. **Lane value.** `L[p][s][j] = A[p][s][0][j] + A[p][s][1][j]`, or the one
   that exists.
5. **Across lanes**, as a pairwise tree in lane order:
   - P = 2: `L0 + L1`;
   - P = 4: `(L0 + L1) + (L2 + L3)`.
   A lane with no contribution to the strip is absent, and the other operand
   passes through unchanged.
6. **ACC.** With `acc = 1`, `ACC_old + V`. FP32 addition is commutative, so only
   the grouping matters.

**Why parity.** `sa_fp32_add` has a latency of 4. A lane sees beats of the same
(half, parity) every 4 cycles: (k, lo), (k, hi), (k+1, lo), (k+1, hi). Two
partial-sum sets therefore keep the adders fully busy without a read-after-
write hazard. The order stays fixed whatever the adder's latency.

**Note.** The result depends on P. Bit-exactness is per configuration, as the
golden corpus already is (`gemv_ports` in the target config).

## 5. `sa_ld`: deterministic burst assignment

Today `ar_rr` carries over from one command to the next, and a port whose
AR channel is busy or whose queue is full is skipped (`sa_ld.v`, "try the
next port"). Which port gets a burst therefore depends on AXI timing. That is
harmless for integer sums, but it would make FP16 results nondeterministic.

In FP16 mode:
- `ar_rr` restarts at port 0 when the command is accepted;
- a burst waits for its port (g mod P) instead of skipping to the next one.

Everything else is unchanged:
- `q_k` already counts 16-byte beats from the start of the strip (`off[15:4]`),
  which `sa_gemv` reads as k = `g_k` >> 1 and half = `g_k[0]`;
- the 256-byte alignment means a burst never crosses 4 KB or a strip boundary.

The cost of waiting: a burst waits behind its own port's queue (8 bursts
deep) rather than taking a free port. With 4 KB of data in flight per port,
the throughput loss should be small. `bwtest` measures it (stage F3).

## 6. `sa_gemv.v`: FP16 lanes

Per lane (one per read port), selected at build time by a parameter
(`GEMV_F16`). The int8 lanes are replaced, not duplicated (§7).

```
 beat (s, k, half, 8 x FP16 W)
   │ stage 1  x read: x buffer word k/8, element k%8 (FP16)
   │ stage 2  8 multipliers FP16 x FP16 -> FP32, exact (one DSP48E2 each for the 11 x 11 significand product;
   │          LUTs for unpacking, subnormal normalization, exponent sum, specials)
   │ stage 3  8 x sa_fp32_add (LAT 4): A[s][k%2][half*8 + i] += product i, or store on first touch
   ▼
 partial sums A: S_MAX x 2 x 16 x 32 bits per lane (8 Kbit), LUTRAM, one read and one write per cycle
```

- **Writeback** (after the last beat): the lanes' adders are idle, so they
  perform the reductions of §4. Each step is 2 issues of 8 columns, pipelined
  over strips: about 20 cycles of fill plus about 6 per strip, then one ACC
  word write per strip. That is about 70 cycles for 8 strips, against
  7,168 cycles of streaming (K = 896, 8 strips, 229 KB at 32 B per cycle,
  P = 2), or about 1%.
- **`touched`.** Becomes per (s, parity) and per (s, half) in the lane, and per
  lane for step 5 of §4.
- **x buffer, residency flag, `xready` back-pressure:** unchanged.
  K / 8 words are loaded instead of K / 16.

## 7. Resources and timing at 50 MHz

The current overlay `d16_100mhz_w128_p6_gemv_np4_pcpi` uses 101,995 LUTs of
117,120 (87.1%), 138 of 144 BRAM tiles, 349 of 1,248 DSPs and no URAM.
Measured on the board's VE (`build/p6/ooc_nb8`), one `sa_fp32_add` is about
**430 LUTs and 269 FFs**. The other figures below are estimates.

| Item | Per lane | P = 4 | P = 2 |
|---|---|---|---|
| FP32 adders (8) | 3.4K LUT, 2.2K FF | 13.8K LUT | 6.9K LUT |
| FP16 multipliers (8) | 8 DSP, about 0.5K LUT | 32 DSP, 2K LUT | 16 DSP, 1K LUT |
| Partial sums, writeback muxing, control | about 0.5–1K LUT | 2–4K LUT | 1–2K LUT |
| int8 lanes removed | — | about −4 to −6K LUT | (int8 lanes on all 4 ports removed) about −4 to −6K LUT |
| **Net LUT** | | **about +12–14K → 97–99%** | **about +4–5K → 90–91%** |

**Decision:** build **P = 2 first**, in an FP16 variant overlay. It is the
configuration that fits, and the one to judge timing and the order of §4 on.
P = 4 only fits if the per-lane cost comes in well below the estimate, or if
LUTs are freed elsewhere (for example a variant without the P6 SFU
interleave). The out-of-context synthesis of one lane (stage F2) decides.

**50 MHz.**
- The whole overlay runs on `pl_clk0`. The 50 MHz build `d8_50mhz` exists
  (`build_bitstream.sh -sa_mhz 50`).
- The FP32 adder has 4 stages: at 20 ns it has ample slack, so the
  congestion of a 90% LUT design is the risk, not the logic depth.
- Bandwidth: 16 B per port per cycle, 0.8 GB/s per port, 1.6 GB/s at P = 2
  (3.2 GB/s at P = 4), before the efficiency of about 90% measured for the
  int8 GEMV.

## 8. Runtime prototype and expected speed

**P1, linear layers on the device.** A C program on the ARM, built on the
existing driver (`compiler/runtime/sa`, u-dma-buf pool, descriptor lists). For
each linear layer it:
1. writes x as FP16 into DDR;
2. submits one list: LD x into SPAD_A, the GEMV commands (≤ 8 strips each,
   K split with `acc`), ST of the ACC words;
3. reads y as FP32.

The ARM does everything else in FP32:
- embedding lookup;
- RMSNorm, the q/k/v biases, RoPE;
- attention over an FP16 KV cache;
- SiLU · up, the residuals;
- greedy argmax.

Hand-offs per token: q/k/v (one list), o, gate/up, down per layer, plus the
classifier: 97 lists per token.

The weights come from an offline packing script: `[N/16, K, 16]` FP16, 256-byte
aligned, biases and norm weights converted to FP32.

**Expected decode speed of Qwen2.5-0.5B** (estimates):

| | P = 2 | P = 4 |
|---|---|---|
| Weights per token | about 0.99 GB | about 0.99 GB |
| Weight streaming at about 90% | about 690 ms | about 345 ms |
| ARM side + 97 hand-offs | about 20–40 ms | about 20–40 ms |
| **Decode** | **about 1.4 tok/s** | **about 2.6 tok/s** |

A prompt costs one decode step per token (prefill row by row).

**P2, epilogues on the device (optional).** VE conversions F32 → F16 (to
produce x in SPAD_A) and F16 → F32 (embedding rows, an FP16 KV cache). This
needs a free VE flag bit or element type; the VE's 2-bit type field is full.
With them, the bias, SiLU · up and residual epilogues run on the VE, and the
hand-offs drop from 97 lists per token to a few.

## 9. Verification

- **Functional model** (`llm/sa_funcsim.py`, `gemv(..., f16=1)`):
  - the products are computed exactly in float32 (NumPy float32 multiplication
    of FP16 values converted to float32 is exact);
  - the order of §4 is followed literally, adding through `llm/fp32.py` `add`;
  - a reference that sums in float64 bounds the rounding error (a sanity
    check, not the oracle).
- **Unit testbench** `tb_sa_gemv_f16` (iverilog or xsim), against the model,
  bit for bit, on vectors from a generator:
  - random normal values;
  - FP16 subnormals, ±0, ±Inf, NaN;
  - cancellation (equal and opposite products);
  - K = 8 … 2040;
  - S = 1 … 8;
  - `acc` on and off;
  - lanes left untouched (K / 8 < P).
- **System:**
  - `tb_sa_unit` FP16 GEMV cases through the descriptor path (with LD back-
    pressure on the ports);
  - on the board, `bwtest` FP16 GEMV throughput and a determinism check: the
    same command repeated under DDR load gives identical bytes.
- **End to end:**
  - the P1 runtime's logits against the same C code running on the
    functional simulator (byte-equal);
  - against PyTorch Qwen2.5-0.5B in FP32: cosine similarity per layer
    ≥ 0.999, and greedy top-1 tokens equal on a prompt set, near-ties
    excepted.
- **Regression:** the int8 overlays and the golden corpus are untouched; the
  FP16 variant is a separate overlay (`_f16` suffix) and target config.

## 10. Stages

| Stage | Work | Gate | Estimate |
|---|---|---|---|
| F1 | Functional model (§4), generator, Python reference | model ≡ float64 reference within the FP32 bound; order unit tests | 2–3 days |
| F2 | `sa_gemv` FP16 lanes, `sa_ld` deterministic mode, unit testbench; out-of-context synthesis of one lane (run by the user) | testbench bit-exact; lane LUTs within the §7 budget | 1–1.5 weeks |
| F3 | Overlay `…_gemv_np2_f16` at 50 MHz (build run by the user); `tb_sa_unit`; board `bwtest` | timing met; throughput about 1.4 GB/s; determinism check | 3–4 days |
| F4 | Packing script, P1 C runtime, sim run of the runtime | Qwen2.5-0.5B decode on the sim, top-1 equal to PyTorch | 1 week |
| F5 | Board run of P1 | board ≡ sim byte-equal; speed as §8 | 2–3 days |
| F6 (optional) | P = 4 if it fits; P2 VE conversions and epilogues | as above | open |

## 11. Risks and open points

- **LUT budget (high).** P = 4 does not fit by estimate; P = 2 does, at about
  90%, where placement and routing at 50 MHz should still close. The F2 lane
  synthesis gives the real figure before the overlay build.
- **Deterministic port assignment.** A port whose queue is full now stalls
  the others. To be measured (F3); with 256-byte bursts the expected loss is
  small.
- **The order depends on P.** Results are reproducible per configuration only.
  A future configuration with another P (or another device) gets its own
  bit-exact reference, and its tolerance against PyTorch is checked the same
  way.
- **K ≤ 2047 per command** (16-bit `row_bytes`). Longer K is split; the split
  points are part of the order (fixed by the compiler or runtime, at most 2040
  per block).
- **ACC words as FP32.** The int8 GEMV wrote int32 words. The FP16 mode writes
  FP32 bit patterns into the same words, so no ACC change is needed; consumers
  must read them as FP32 (the VE's F32 element type, or the ARM after ST).

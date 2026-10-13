# K2b / H4: microarchitecture of 2-port loading + a streaming GEMV unit

> English translation of [`k2b_gemv_design.md`](k2b_gemv_design.md) (Chinese original, translated 2026-10-13 at `db07c59`). The Chinese document is authoritative; if the two differ, follow the original.

(2026-10-09; **status 2026-10-10: G1–G5 all done, 4 ports accepted on the board, Qwen3 decode 6.28 token/s, SmolLM2 17.04**, see §8 / §8.1. Basis: the performance model of K2b step 1 in `kv260_upgrade_plan.md`, `compiler/tests/perf_model.py`.)

## 1. Goal

Almost all of a decode step's time is in the linear layers' matvecs: Qwen3 reads 607.5 MB of weights per step, LD is busy 39.5M cycles and EX 38.5M (43.4M per step); both sides move 16 bytes per cycle and are **saturated at the same time**. The performance model gives:

| Option (100 MHz) | Qwen3 decode | SmolLM2 decode |
|---|---|---|
| Wider loading only (32 B / cycle) | ×1.02 | ×1.03 |
| GEMV unit only (32 B / cycle) | ×1.02 | ×1.05 |
| Loading 32 + GEMV 32 | **×1.85** (2.31 → 4.26 token/s, device time) | **×1.76** |
| Loading 64 + GEMV 64 (4 ports) | ×3.15 | ×2.74 |

The goal of the first stage is a **decode matvec with P = 2 read ports at 32 bytes per cycle**, at the same clock (100 MHz), bit-exact with the functional simulator, and structurally extensible to P = 4.

## 2. Key observation: decode's weights are used once and need not go through the SPAD

Today a decode linear layer (`expandLinear`, `SahlExpandKernels.cpp`) does per chunk:

```
LD   weights (nc strips x K x 16 B) -> SPAD_B bank (ci & 1)      64 KB per block (Qwen3)
EX   a = A strip (x replicated to 16 rows), b = SPAD_B, c = ACC, kt = K/16, rep = nc, cstep 1, crow nc
VE   epilogue: read row 0 of ACC (words c .. c+nc-1), write the fp32 result
ST   the result
```

Board trace (Qwen3 `matvec_like_64x16x2048`): `LD rows 2 rb 32768` → `EX kt 128 rep 2 bstep 2048 cstep 1 crow 2`; the epilogue only reads words c, c+1 (row 0).

- The weights are first written into SPAD_B, read once by EX, and never used again. SPAD_B's write and read ports are both fully occupied, without any reuse.
- Only row 0 of the array's 16 rows is useful (x is replicated to 16 rows); multiply-accumulate utilization is 1/16.

So H4 bypasses the SPAD: **the LD engine streams the weights directly into the GEMV unit**. Each 128-bit beat is the 16 weights of one k row (the same word layout as SPAD_B; the packed format in DDR is unchanged), multiplied by x[k] and accumulated into 16 output columns. Then:

- each read port feeds its own group of 16 multiply-accumulates, and **the read ports do not compete for local memory**: 2 ports are 32 B / cycle without SPAD sub-banks, and so without the compiler constraints of sub-bank alignment and per-sub-bank conflict checks (most of the work of the original K2b option A is deferred until prefill needs the bandwidth);
- SPAD_B is freed during decode, and so is the EX array (decode's attention still uses the array);
- integer multiply-accumulate is order independent, so the result is bit-identical to EX's row 0.

## 3. Overall structure

```
            HP1 (m0)            HP2 (m1)
               |  AR/R            |  AR/R
        +------v------------------v------+
        |  sa_ld   burst queue QD = 8/port|----> SPAD / ACC write (one port per cycle, 16 B, unchanged)
        |  GEMV mode: both ports take a   |
        |  beat in the same cycle         |
        +------+------------------+------+
               | beat + (s, k)    | beat + (s, k)
        +------v------+    +------v------+
        | lane grp 0  |    | lane grp 1  |    x buffer: one per group (K_max = 4096 bytes)
        | 16 x int8   |    | 16 x int8   |    each group reads one x[k] byte per cycle
        | MAC, acc[s] |    | MAC, acc[s] |    acc: S_MAX x 16 x int32 per group
        +------+------+    +------+------+
               +--------+---------+
                        v  write-back: per strip the two groups are added (optionally plus the old ACC value) -> one ACC word (16 x int32)
                     ACC side B (LD's request port)
```

`sa_gemv.v` is a new module next to `sa_ld`, driven by the LD engine; the scheduler treats a GEMV as an LD-engine command.

## 4. Command: LD's GEMV mode

No new engine or opcode; it uses `mode = 2` of the LD descriptor (existing: 0 LINEAR, 1 INTERLEAVE):

| Field | Position in the LD descriptor | Meaning for GEMV |
|---|---|---|
| `ddr` | w1[31:0] (relocatable, dynamic field 1) | Start of strip 0's weights (16-byte aligned) |
| `laddr` | w2[31:0] | mem = ACC, word = C (the first word of the result) |
| `rows` | w2[47:32] | S: number of strips (1 .. S_MAX = 8) |
| `row_bytes` | w2[63:48] | Bytes per strip = K × 16 (K ≤ 4095, a multiple of 16) |
| `pitch` | w3[31:0] | DDR distance between strips (packed weights = K × 16) |
| `mode` | w3[33:32] | 2 = GEMV |
| `xword` | w3[49:34] (new) | Start word of x in SPAD_A (x stored contiguously, K/16 words) |
| `cstep` | w3[57:50] (new) | Strip s's result is written to word C + s × cstep |
| `acc` | w3[58] (new) | 1: add to the old ACC value (used when K is split) |

**Semantics** (implemented this way by the functional simulator): for s < S, j < 16

```
ACC[C + s*cstep][j] = (acc ? ACC[C + s*cstep][j] : 0) + Σ_{k<K} sx8(x[k]) * sx8(W_s[k][j])     (int32 wrap-around)
x[k]    = byte k%16 of SPAD_A word xword + k/16
W_s[k][j] = DDR[ddr + s*pitch + k*16 + j]
```

Only row 0 of each strip (one ACC word) is written. The EX version also wrote rows 1..15 (the same values computed from the replicated x); the epilogue only reads row 0, so compiled programs are not affected.

**Scheduler** (`sa_sched.v` stages d1 / d2): a GEMV's read set = SPAD_A [xword, xword + K/16) (plus the ACC result words when acc = 1), write set = ACC [C, C + (S−1)·cstep], engine = LD. The existing bank scoreboard applies directly: the epilogue's VE reads ACC → waits for this GEMV to finish; the next GEMV writes the other ACC bank → can run in parallel with the epilogue (today's double buffering is unchanged). Checks: `row_bytes % 16 == 0`, `rows ≤ S_MAX`, no out-of-range access, `mem == ACC`.

## 5. 2-port loading (`sa_ld`)

The existing `sa_ld` already has an `NPORTS` parameter: bursts (≤ 16 beats, not crossing 4 KB) are distributed round robin over the ports, 8 in flight per port, and returning beats are written to the local position recorded with the burst. But **the R side accepts only one port per cycle** (`r_sel`) and there is only one local write port, so more ports do not add bandwidth. Changes:

1. **AXI**: `NPORTS = 2`, m0 → `S_AXI_HP1_FPD` (unchanged; still shares AR / R with descriptor fetch, distinguished by an owner FIFO), m1 → `S_AXI_HP2_FPD` (new protocol conversion and address mapping; `build_bitstream.sh -nports 2`). ST automatically uses both ports' write channels as well.
2. **Two more fields in the burst queue**: in GEMV mode each burst records (strip s, starting k). As beats arrive k increments by 1 per beat; at a strip boundary s increments and k resets to 0 (a burst is at most 256 bytes = 16 k rows and strip lengths are multiples of 16 bytes, so a beat never crosses a strip).
3. **R side**: in GEMV mode each port has its own `m_rready`, and in the same cycle each port hands a beat to its own lane group; outside GEMV mode the current single-port arbitration stays (writes to SPAD / ACC remain 16 B / cycle, see stage two in §8). The GEMV lane groups have 2–3 pipeline stages and `rready` is simply 1 (no back-pressure, except during result write-back).
4. **Completion**: `done` after all beats have entered the MAC pipeline and write-back has finished; the `err` rule is unchanged (SLVERR / DECERR on any beat).

The two HP ports take different paths to the DDR controller, and 3.2 GB/s is far below the KV260 DDR4 peak (about 19 GB/s), but the actual concurrent two-port bandwidth must be confirmed with a new dual-port test in `bwtest` (also comparing HP1 + HP2 against HP1 + HP3).

## 6. GEMV unit (`sa_gemv.v`)

Per lane group (one group per read port):

- **x buffer**: K_MAX = 4096 bytes, one copy per group (the two read ports each read an x[k] in the same cycle). RAMB18 (4 KB × 8 bits) or LUTRAM. At the start of a command the LD engine reads K/16 words from SPAD_A (through the LD request port on SPAD_A side A, one word per cycle; 192 cycles for K = 3072, partly overlapped with the DDR latency of the first beat). **Residency tag** (xword, K, valid): all chunks of a matvec use the same x, so from the second command on a tag hit skips the read; any LD / VE command that writes SPAD_A clears the tag (conservative; in hardware just one OR).
- **Multiply-accumulate**: one beat = 16 int8 weights × the same x[k]. The 16 multiplies share an operand, so the DSP48E2 int8 packing can be used (two weights share x, two multiplies per DSP): 8 DSPs per group; or LUT multipliers. Pipeline: beat → read x[k] (1 cycle) → multiply (1–2 cycles) → accumulate.
- **Accumulators**: acc[s][j], S_MAX × 16 × 32-bit flip-flops (4096 bits per group). All 16 beats of a burst belong to the same strip; accumulation is a 32-bit add + an S_MAX-way select, done in one cycle (no problem at 100 MHz).
- **Write-back**: after all beats are consumed, for s = 0 .. S−1: the two groups' acc[s] are added (plus the old ACC value when acc = 1, read in one beat through the LD request port on ACC side B) and written to ACC word C + s·cstep (16 × int32, all byte enables). 1–2 cycles per strip.

Throughput: 16 bytes per port per cycle → 32 B / cycle at P = 2. One 64 KB Qwen3 chunk: 65 (latency) + 2048 (data) + 2 (write-back) ≈ 2115 cycles; today LD and EX each take about 4160 cycles.

## 7. Compiler and runtime

- **Target configuration**: `TargetConfig` gains `gemv_ports` (0 = no GEMV unit; `--iree-sa-gemv-ports=N`, only D = 16 supported); the `.hwh` / driver report `NPORTS` and whether a GEMV unit is present.
- **SAHL**: a new op `sahl.gemv(x, w, acc)` (x: a contiguous int8 vector in SPAD_A; w: a subview of nc strips in DDR; acc: an nc × 16 block of ACC). When the target has GEMV, `expandLinear` no longer allocates an A strip; each chunk's `sahl.load` of weights + `sahl.mma` is replaced by `sahl.gemv` (split into several of ≤ 8 strips each when nc > 8); the two SPAD_B weight banks are no longer needed. An int8 x is used directly; an int32 x (in ACC, as in the decode of stories / SmolLM2 / Qwen3) is converted by one `sahl.strip {sa.packed}` into contiguous int8 in SPAD_A (the same VE conversion as the old strip, just not replicated to 16 rows). The epilogue, the ACC layout (`sa.word`) and the `sahl.loop` advance (ddr advances by two chunks each time) are unchanged.
- **Constraints**: K ≤ `gemv_kmax` (4095), nc ≤ S_MAX; otherwise fall back to today's LD + EX (correct but slow). The packed weight format and the irpa are unchanged.
- **SAHW / serialization**: the LD descriptor's new fields (mode 2, xword, cstep, acc); `DescList` encoding; only ddr (field 1) is used as a dynamic field.
- **Functional simulator / oracle**: `sa_funcsim.ld` implements mode 2 (the semantics of §4, bit for bit) and the sched checks; `dispatch_check` covers all decode matvecs; the new target configuration gets its own golden corpus baseline (constraint 3 of §5).
- **Performance model**: `perf_model.py` adds the cost of a GEMV-LD (latency + x read (on a miss) + bytes / (16 × P) + write-back) and re-predicts from the newly compiled trace.

## 8. Stages

| Stage | Content | Verification |
|---|---|---|
| **G1** | Functional simulator + compiler (`sahl.gemv`, target configuration switch), no hardware | decode of Qwen3 / SmolLM2 / stories bit-exact on the sim with the existing path; `dispatch_check`; new baseline |
| **G2** ✅ (board ✅) | RTL: `sa_gemv` + `sa_ld`'s GEMV mode + `sa_sched` checks and masks, NPORTS = 1 first (functionally correct, 16 B / cycle) | `tb_sa_gemv` (random K / S / cstep / acc / alignment, against funcsim); `tb_sa_ld`; `tb_sa_unit`; firmware system simulation `desc_run`; `make test / test16 / test128` |
| **G3** 🔧 | NPORTS = 2: R side receives on both ports at once, a second HP port in the BD (HP3), `bwtest` dual-port test | `tb_sa_dma GEN="NP=2 D=16 DMA_W=128"` with GEMV mode; user builds the bitstream; board bwtest |
| **G4** | Board acceptance | REGRESSION PASS, bit-exact; decode token/s compared with the model prediction (×1.76–1.85) |

**G2 implementation (2026-10-09)**: `rtl/sysarray/sa_gemv.v` + `sa_ld` / `sa_sched` / `sa_cmdfetch` / `sa_unit` (parameter `GEMV`, requires D = 16, DMA_W = 128; CAPS bit 25; `build_bitstream.sh -gemv 1`, configuration suffix `_gemv`). Differences from §5 / §6 and details:

- Number of lanes = `NPORTS`; the R-side GEMV path receives per port independently from the start (G3 only has the BD and the bandwidth test left); `tb_sa_dma` tested the multi-lane sum at NP = 1 / 2 / 3.
- Per-lane accumulators `acc[s]` (8 × 16 × int32), not cleared: a strip's first beat on that lane writes directly (touched mask), and untouched strips count as 0 at write-back.
- `rready` is held low while x is being read (AR still issues; returning beats wait in the interconnect), one word per cycle; the residency tag is cleared on any SPAD_A write (the LD write port or a VE write).
- Write-back is one cycle per strip (two when acc = 1: the old ACC value is read first).
- Packet format: [148:133] x word, [156:149] C step, [157] acc of the LD packet (`pkt_ldg`); mode 3 is always SHAPE; without a GEMV unit mode 2 is SHAPE too.
- Scoreboard: GEMV **reads** SPAD_A on the LD port, on SPAD_A's A side just like ST's reads; two readers were not counted as a conflict and could collide on the same side of the same bank. Now LD and ST, like EX and VE, conflict on any shared bank (existing commands are unaffected: a plain LD only writes). The performance model does the same.
- With `--iree-sa-gemv-ports` non-zero the compiler sets bit 25 in the required CAPS of the executable header; the runtime check is G4 item 4.
- Tests: `tb_sa_dma GEN="NP=1|2|3 D=16 DMA_W=128"` (random K / strips / C step / acc / pitch, crossing 4 KB, crossing banks, K = 4080, residency hits and invalidations, SLVERR), `tb_sa_unit GEN="D=16 DMA_W=128 GEMV=1"` (descriptor decoding, scoreboard: VE waits for the GEMV's writes, residency, 4 kinds of errors, CAPS); all added to `make test128`; firmware system simulation `firmware/desc_run make sim SIM_D=16 SIM_DMA_W=128 SIM_GEMV=1` (PicoRV32 + RTL, Python `DescList.gemv` encoding: int32 x converted to int8 by VE, 4 + 2 strips, K accumulated in two halves, against the functional simulator and NumPy).

**G3 implementation (2026-10-09, RTL and scripts; board pending)**:

- Multi-port GEMV in the RTL was already done in G2; G3 only changed: the performance counters LD_BEATS / ST_BEATS added at most 2 per cycle (`ev2`), undercounting beats of several ports in the same cycle; now `sa_ld` / `sa_st` output the number of 8-byte lanes per cycle (`perf_lanes`, summed over all ports), and `sa_perf`'s `evx` is the extra increment per counter. Unchanged with a single port.
- `tb_sa_unit`'s memory model now has one slave per port (`NP=2`); the whole unit test (GEMM, random command streams, descriptors, GEMV, counter invariants relaxed by `NP`) passes with two ports; added to `make test128`.
- BD: `build_bitstream.sh -nports 2` (suffix `_np2`); m1_axi connects to **HP3** by default: in UG1085's PS interconnect HP1 and HP2 share the DDR controller's S4 port, while HP3 goes to S5; `-m1_hp 2` (suffix `_hp2`) for comparison. The HP port width follows `-dma_w`; the address map is the same as m0 (identity map of the low 2 GB).
- bwtest: tests 6 / 7 are one 64 KB GEMV (8 strips, K = 512; 6 includes the x read, 7 has x resident = pure streaming bandwidth); the ARM writes the descriptors into DDR (`MBOX_DL_ADDR`), the firmware submits them when CAPS bit 25 is set, and the result is stored back to DDR and checked by the driver with NumPy; `m2_bw_test.py` computes the peak as `NPORTS × 16 B`. Firmware system simulation `firmware/bwtest make sim SIM_D=16 SIM_DMA_W=128 SIM_GEMV=1` passes (single-port DDR model).

**G4 steps (board acceptance)**:

1. **The existing path on the single-port bitstream (`d16_100mhz_w128_p6_gemv`)**: `SA_KV260_CONFIG=d16_100mhz_w128_p6_gemv compiler/scripts/deploy_z1_freeze.sh --reuse d16_100mhz_w128_p6_red` (the LLM tests reuse the EX-path executables; a few minutes), then `run_board.sh` on the board: REGRESSION PASS shows the added GEMV unit does not affect existing functions; bwtest tests 6 / 7 give the single-port GEMV streaming bandwidth and check the result.
   **Result (2026-10-09, `d16_100mhz_w128_p6_gemv`, WNS +0.274 ns, LUT 92.3k / 78.8% (+4.1k), BRAM 132 (+2), DSP 349)**: REGRESSION PASS (16 items), the tokens of the 6 LLM tests identical to `p6_red`, speed unchanged (Qwen3 prefill 15.41 / decode 2.22 tok/s); bwtest GEMV 64 KB: x resident 4190 cycles = **15.64 B / cycle** (97.8% of the single-port peak; plain LD 15.70), x read adds 15 cycles, result matches NumPy.
2. **GEMV path**: full staging (without `--reuse`). `board_env.sh` derives `SA_GEMV_PORTS` from the configuration name (`_gemv` → 1, `_gemv_np2` → 2), `compile_sa.sh` adds `--iree-sa-gemv-ports`, and build directories get the suffix `_gemv` (2 ports `_gemv2`: the descriptors differ and cannot be shared; hfgen skips compiling when a `sa.vmfb` exists); each LLM test first passes the per-dispatch checks and the simulation on the host, and on the board the logits are compared bit for bit with the simulation (also bit-identical to the EX path, proven on the sim in G1). Qwen3 additionally uses `deploy_z1_freeze.sh c6p_qwen3` (640 MB udmabuf).
3. **Dual-port bitstream (`_gemv_np2`)**: as 1 and 2; bwtest test 7 should be close to 32 B / cycle; decode token/s compared with the model prediction (Qwen3 ×1.85 → about 4.1–4.3 token/s device time, SmolLM2 ×1.76), and the decode profile from `board_profile.py` fed back to `perf_model.py` to calibrate the GEMV cost.
   **Timing**: the first build had WNS -0.576 ns (742 endpoints): m1's RVALID from the PS passes, in the same cycle, through R-side port arbitration, head-of-queue selection, the multiply in `lw_word` and the GEMV / ACC select to the ACC BRAM address (with one port the arbitration is a constant and optimized away). Fix (3127e6a): with NPORTS > 1 each port's R channel in `sa_ld` first passes a 2-entry skid buffer (valid / data / ready all from registers; same throughput, 1 more cycle of latency); after the rebuild WNS +0.138 ns, the worst path starts at the `q_rp` register (still through the `lw_word` multiply), LUT 96.3k / 82.2% (+4.0k over one port), BRAM 134, DSP 349.
   **Step 1 result (2026-10-10, `d16_100mhz_w128_p6_gemv_np2`, m1 on HP3, `--reuse p6_red`)**: REGRESSION PASS (16 items), the tokens of 5 LLM tests identical to the single-port build; bwtest GEMV 64 KB: x resident 2145 cycles = **30.55 B / cycle** (95.5% of the dual-port peak, 1.95× the single port's 15.64), x read 2157 cycles / 30.38, result correct, no visible contention between HP0 and HP3 on the DDR side; plain LD / ST still about 15.7 B / cycle (the R side writes only one local word per cycle), short-row LD with 1-beat bursts 3.04 → 5.48 B / cycle (the address latencies of the two ports overlap).
   **Step 2 result (2026-10-10, `_gemv_np2`, compiled for 2 GEMV ports, `build/deploy_kv260_d16_100mhz_w128_p6_gemv_np2/results`)**: REGRESSION PASS (including Qwen3, 18 items); for every LLM test the board logits are bit-identical to the simulation and the tokens identical to the EX path (`p6_red`). Decode per step (measured on the board, including host overhead):

   | Model | EX path (`p6_red`) | GEMV 2 ports | Speedup | Model prediction |
   |---|---|---|---|---|
   | Qwen3-0.6B | 451.4 ms (2.22 tok/s) | 251.3 ms (**3.98 tok/s**) | ×1.80 | ×1.85 |
   | SmolLM2-135M (c6p) | 129.6 ms (7.71) | 79.5 ms (**12.58**) | ×1.63 | ×1.76 |
   | SmolLM2 (hfgen, generic frontend) | 131.2 ms (7.62) | 78.5 ms (12.73) | ×1.67 | — |
   | stories15M | 16.2 ms (61.9) | 9.9 ms (100.8) | ×1.63 | — |

   prefill unchanged (Qwen3 15.45 → 15.76 tok/s, SmolLM2 38.4 → 38.8). SmolLM2 is somewhat below the prediction: the model is small, and the fixed per-step overhead (non-linear layers, command and dispatch overhead) is a larger share.
4. **CAPS check**: rt_fw (RT_VERSION 2) writes the unit's CAPS into mailbox `0xD0`, and the runtime checks the bits 21–25 an executable requires when loading it; a model compiled for GEMV fails immediately on a bitstream without GEMV instead of producing wrong results. Verified on the sim by emulating a unit without GEMV with `SA_SIM_CAPS=0x01E00010` (the stories15M GEMV build reports INCOMPATIBLE on load). G1's Qwen3 has also passed: 1262 dispatches, logits byte-identical to the EX path.

**Stage two**: P = 4 (§8.1); SPAD sub-banks (LD writes to SPAD / ACC also at 32 B / cycle, prefill ×1.07); LD command pipelining (hides the ~65-cycle latency of each command, +2–3%).

### 8.1 P = 4: 4 read ports (G5)

decode is still bound by read bandwidth (the GEMV unit consumes weights at the bandwidth's rate); 4 ports = 64 B / cycle = 6.4 GB/s (KV260 DDR4 peak about 19 GB/s).

**Performance model** (100 MHz, device time, relative to the EX path; in parentheses scaled by the dual-port measured / predicted ratios 0.97 (Qwen3), 0.93 (SmolLM2)):

| | Qwen3 decode | SmolLM2 decode |
|---|---|---|
| 2 ports (now) | ×1.85 (measured ×1.80, 3.98 tok/s) | ×1.76 (measured ×1.63) |
| **4 ports** | **×3.15** (about 6.5–7 tok/s; measured ×2.83, 6.28 tok/s) | **×2.74** (about 20 tok/s; measured ×2.21, 17.04) |
| 4 ports + H2 + LD command pipelining | ×3.46 | ×3.06 |

prefill is unchanged. After 4 ports, what remains of decode is the fixed latency of LD commands (about 1.6M cycles / step), attention on EX, and VE (2.65M); the PS has no more HP ports either. The main lever beyond that is int4 weights (W4A8, half the bytes), evaluated separately.

**Port assignment**: the PS's HP0 / HP1–HP2 / HP3 connect to the DDR controller's S3 / S4 / S5, and HPC0 / HPC1 go through the CCI to S1 / S2. Today HP0 = PicoRV32, HP1 = m0, HP3 = m1.

| Option | m2 | m3 | PicoRV32 | |
|---|---|---|---|---|
| **A** (first) | HP2 | HPC0 (non-coherent) | HP0 unchanged | Smallest change; HPC0 goes through the CCI with higher latency (8 bursts in flight per port should hide it), shares S1 / S2 with the APU |
| B | HP2 | HP0 | moved to HPC0 | All four DMA ports on HP; PicoRV32 only accesses the ring, very little traffic |

`build_bitstream -nports 4 [-m3_port hpc0|hp0]` (B adds the suffix `_m3hp0`); build A first, and build B for comparison only if bwtest test 7 is clearly below 60 B / cycle. The same test also confirms that HP1 and HP2 share S4.

**RTL**:
- `sa_unit`: add `m3_axi` (today m0..m2);
- `sa_ld`: supports any number of ports already (per-port queue + skid buffer); R-side arbitration becomes 4-to-1 and the path `lw_word` (multiply) → ACC address gets longer (dual-port slack +0.138 ns); if needed move `lw_word` / `lw_lane` one stage earlier (one more cycle of latency on local writes, same throughput);
- `sa_gemv` (NL = 4): one x buffer (1 RAMB36), accumulators and 16 multiplies per lane; at write-back each column adds 4 lanes + the old ACC value, with one more pipeline stage;
- Testbenches: `tb_sa_dma` NP = 4, `tb_sa_unit` memory slaves up to 4, GEMV NP = 4 in the Makefile.

**Resource estimate**: single → dual port +4.0k LUT; 4 ports about 101–103k LUT (87–88%), BRAM 136 / 144, DSP 349. LUT is the main risk (routing congestion, harder timing closure); if it does not fit, put the GEMV multiplies into DSPs (about 900 left) or let two lanes share one dual-read-port x buffer.

**Software**: the compiler already accepts `gemv_ports` 0..4; `board_env.sh` adds `_gemv_np4` → `SA_GEMV_PORTS=4` (build suffix `_gemv4`); bwtest's GEMV peak is already computed as NPORTS × 16.

**G5-1 implementation (2026-10-10)**: `sa_unit` adds `m3_axi`; with NPORTS > 2, `sa_ld` registers the local write (`lw_*`) one stage (the 4-to-1 arbitration and the `lw_word` multiply end at a register; a command's done comes in the cycle after its last beat is taken, when the write is already in the register and is written on the same clock edge); with NL > 2, `sa_gemv` first registers the sum of the lanes and then adds the old ACC value (two cycles per strip); `build_bitstream -nports 4 [-m3_port hpc0|hp0]`, `kv260_bd.tcl` configures the PS, wiring, clocks and addresses from a port list (PicoRV32's port, the ports of m0..m3); `board_env.sh`: `_gemv_np4` → 4 ports, build suffix `_gemv4`. Verification: `make test / test16 / test128` all pass (new `tb_sa_dma` NP = 4, `tb_sa_unit` GEMV NP = 4); mutations (removing the 4th lane, removing the registered lane sum, a wrong registered `lw_lane` value) are all caught; firmware bwtest / desc_run co-simulation passes; stories compiled with `gemv_ports = 4` has per-dispatch checks and simulation logits bit-identical.

**G5-2 / G5-3 results (2026-10-10, `d16_100mhz_w128_p6_gemv_np4`, m2 HP2, m3 HPC0)**: WNS **+0.168 ns** (dual port +0.138; the worst path is now the fan-out of the PicoRV32 reset to the array PEs, 97% routing; the GEMV / LD data paths are no longer critical), LUT 101,878 (87.0%), FF 67,688, BRAM 138 / 144 (one x buffer per lane), DSP 349. On the board with `--reuse p6_red`: REGRESSION PASS (14 items), the tokens of 5 LLM tests identical to `p6_red`; bwtest GEMV 64 KB: x resident 1131 cycles = **57.95 B / cycle** (90.5% of 64), x read 1164 cycles / 56.30. The ratio is below the dual port's (95.5%) because the transfer time halves while the fixed overhead of about 100 cycles per command stays (dual port 2145 = 2048 + 97, 4 ports 1131 = 1024 + 107): the steady-state bandwidth is close to 64 B / cycle, and neither HPC0 through the CCI nor HP1 / HP2 sharing S4 shows a bottleneck, so no `-m3_port hp0` build is needed for comparison. Plain LD / ST still about 15.7 B / cycle; short-row LD with 1-beat bursts 5.48 → 7.61.

**G5-4 results (2026-10-10, `_gemv_np4`, compiled for 4 GEMV ports, `build/deploy_kv260_d16_100mhz_w128_p6_gemv_np4/results`)**: on the host every LLM test's per-dispatch checks pass and the simulation logits are bit-identical (`sa.vmfb` differs from the 2-port version; hfgen recompiled into `smollm2_p16_gemv4`); on the board REGRESSION PASS (including Qwen3, 16 items), logits bit-identical to the simulation, tokens identical to the 2-port / EX path; bwtest GEMV 57.44–57.64 B / cycle. Decode per step (measured on the board, including host overhead):

| Model | EX path | GEMV 2 ports | GEMV 4 ports | 4 ports / 2 ports | 4 ports / EX | Model prediction |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 451.4 ms (2.22) | 251.3 ms (3.98) | 159.2 ms (**6.28 tok/s**) | ×1.58 | ×2.83 | ×3.15 |
| SmolLM2-135M (c6p) | 129.6 ms (7.71) | 79.5 ms (12.58) | 58.7 ms (**17.04**) | ×1.35 | ×2.21 | ×2.74 |
| SmolLM2 (hfgen) | 131.2 ms (7.62) | 78.5 ms (12.73) | 58.2 ms (17.19) | ×1.35 | ×2.25 | — |
| stories15M | 16.2 ms (61.9) | 9.9 ms (100.8) | 7.6 ms (131.65) | ×1.31 | ×2.13 | — |

prefill unchanged (Qwen3 15.90 tok/s, SmolLM2 39.0). Why it is below the prediction: model each step as a fixed part F plus a GEMV part G inversely proportional to the number of ports; solving from the 2-port / 4-port points gives Qwen3 G ≈ 184 ms (at 2 ports), **F ≈ 67 ms**, and SmolLM2 G ≈ 42 ms, **F ≈ 38 ms**; at 4 ports the fixed part is already 42% of Qwen3's step and 65% of SmolLM2's (attention on EX, VE / SFU, LD command latency, dispatch and host overhead), while GEMV itself scales nearly linearly. Improving decode further means cutting F (H2 out-of-order issue, LD command pipelining, fewer dispatches and less host overhead), not more ports; int4 weights only shrink G.

**Measured breakdown of F (2026-10-12, WP9, `generality_implementation_plan.md` §13.1)**: per-dispatch timing of Qwen3's generic path (`dispatch_stats.py`): a batched step is 175.0 ms on the device and 3.4 ms in the rest of the runtime, so host overhead is negligible; GEMV linear layers 111 ms (598 MB, 53.7 B / cycle = 93% of the 4-port peak), attention (static T = 256) 46.5 ms, normalization / RoPE / softmax / quantization 18.7 ms. Before that the v projection (int8 result written directly into the KV cache) did not use GEMV and cost 33 ms more per step (fixed by `14f5612`, 211.1 → 178.4 ms/step); with the attention over the actual length (`1860823`) it is 145.4 ms/step (6.88 tok/s), GEMV being 80% of the device time.

**4-port model calibration and next-step what-if (2026-10-12, Qwen3 generic path `deploy_hfgen_qwen3_dyn`, dynamic attention length)**: trace from `DEPLOY=build MODEL=deploy_hfgen_qwen3_dyn OUT=qwen3_dyn MB=640 compiler/scripts/perf_trace.sh`, then `perf_model.py --trace build/perfmodel/qwen3_dyn.trace --board build/deploy_hfgen_qwen3_dyn/prof3.csv --gemv-ports 4 [--whatif next]`. With VE / frontend refitted (f_desc 24 cycles / descriptor), the model versus the board is 80.46M / 78.68M cycles for a prefill chunk (**+2.3%**) and 14.38M / 13.88M for a decode step (**+3.6%**; the largest GEMV linear layers +2–10%). What-if with 4 ports as the baseline (time speedups; at 200 MHz the DDR latency is unchanged in ns, LD / ST latency cycles ×2, and 4 ports × 16 B × 200 MHz = 12.8 GB/s is still below the DDR4 peak of 19.2):

| Option | prefill tok/s | decode tok/s | prefill / decode |
|---|---|---|---|
| Baseline (4-port GEMV, 100 MHz) | 20.61 | 6.95 | ×1.00 / ×1.00 |
| 200 MHz | 38.66 | 12.48 | ×1.88 / ×1.79 |
| W4A8 (half the decode weight bytes) | 20.61 | 10.28 | ×1.00 / ×1.48 |
| 6 GEMV ports (HP0–3 + HPC0–1) | 20.61 | 8.87 | ×1.00 / ×1.28 |
| LD command pipelining | 20.92 | 7.21 | ×1.01 / ×1.04 |
| H2 out-of-order issue | 22.07 | 7.33 | ×1.07 / ×1.05 |
| D = 32 array | 21.82 | 7.07 | ×1.06 / ×1.02 |
| W4A8 + H2 + LD pipelining | 22.24 | 13.14 | ×1.08 / ×1.89 |
| 200 MHz + W4A8 | 38.66 | 17.58 | ×1.88 / ×2.53 |
| 200 MHz + W4A8 + H2 + LD pipelining | 42.92 | 25.01 | ×2.08 / ×3.60 |

The model covers device time only (not the runtime's 3–7 ms / step); URAM (SPAD capacity) changes the compiler's tiling and cannot be evaluated with a fixed trace. Conclusion in `generality_implementation_plan.md` §17 question 6.

**Steps**: G5-1 RTL / testbenches / BD parameters, `-bd_only` check after the simulations pass; G5-2 build `-nports 4`; G5-3 board regression with `--reuse p6_red` + bwtest (test 7 should be close to 64 B / cycle); G5-4 full 4-port GEMV staging, board decode compared with the prediction.

## 9. Resource and timing estimate (P = 2)

| Part | LUT | FF | DSP | BRAM |
|---|---|---|---|---|
| `sa_gemv`: 32 int8 multiplies (DSP packing) | — | — | 16 | — |
| Accumulators 2 × 8 × 16 × 32 bits + 32 32-bit adds + selects | ~2k | ~8.5k | — | — |
| Write-back: 16 32-bit adds (two groups + old ACC value) | ~0.6k | ~0.5k | — | — |
| 2 x buffers | — | — | — | 1 RAMB36 (two RAMB18) |
| `sa_ld` dual-port R side + burst tags | ~0.5k | ~0.3k | — | — |
| BD: protocol conversion for the second HP port | ~1–2k | ~2k | — | — |
| **Total** | **~4–5k** | ~11k | 16 | 1 |

Today LUT 88.25k / 117k (75%), FF about 61k / 234k, DSP 347 / 1248, BRAM 130 / 144; everything fits (LUT about 79%). The new paths are all short (one level of 32-bit adds, DSP multiplies) and will not be critical at 100 MHz.

## 10. Risks

- **DDR bandwidth with two concurrent ports**: the model assumes 16 B / cycle per port; confirmed with G3's dual-port bwtest; if insufficient, first check the assignment of HP ports to DDR controller ports.
- **m0 shared with descriptor fetch**: fetches interleave with weight bursts on m0; with round-robin assignment m0 carries a little more. GEMV bursts could prefer m1, or fetch could move to m1.
- **x residency tag**: the clearing condition is deliberately conservative (any write to SPAD_A clears it); the functional simulator implements the tag too and `dispatch_check` compares it, so a wrong tag shows up as a result mismatch.
- **Only row 0 is written**: relies on the epilogue reading only row 0. The compiler generates GEMV only in `expandLinear`, where the epilogue layout is fixed; funcsim's result comparison catches any read of rows 1..15.

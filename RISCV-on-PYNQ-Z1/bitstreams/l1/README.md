# L1: host interface and command extensions (prebuilt)

This build is the L0 D = 8 accelerator plus the L1 additions of
`docs/llm_inference_plan.md` §5:

- **Host interface**: `mat_notify` and the `notify_irq` output (a rising
  edge, connected to `irqConcat/In1` → axi_intc → IRQ_F2P), plus the
  `NOTIFY_COUNT` CSR (0xD4).
- **Command extensions**, compiler-facing, in `sa_cmdfetch.v`: BASE0–15,
  PARAM0–7, two dynamic field slots per descriptor, SETREG, LOOP_END
  (2 levels), relative JUMP, CALL/RET (depth 4), LDPARAM.
- **CAPS**: bits 22 and 24 are set.

It is used with the resident runtime firmware `firmware/rt` (`rt_fw.bin`)
and `driver/pynq_matmul.py` `Device`.

Built with Vivado 2024.1 (`scripts/build_bitstream.sh -jobs 4 -sa_d 8`):
27,136 LUT (51.0 %), 20,341 FF, 130 BRAM36, setup WNS +2.348 ns at 50 MHz.
The L0 build was 23,452 LUT with WNS +1.555 ns. The descriptor fetch unit
alone is 4.4k LUT (OOC).

Board results (all PASS):

- **`notebooks/llm/l1_ring_demo.py`**:
  1. 200 asynchronous submissions (GEMM and vector lists, ring of 64), all
     correct. Submitting them took 1.80 s and everything finished in 2.35 s,
     mostly Python-side buffer allocation.
  2. One static list runs with different parameter blocks and bases: the
     chunk count comes from PARAM2 through the LOOP_END count field, and the
     addresses advance by PARAM strides (1, 5 and 16 chunks).
  3. An index gather with LDPARAM + CALL/RET + SETREG add: 44 descriptors.
  4. Error isolation: the invalid list completes with status 0x1004103
     (fetch unit, shape error), and the next entry is correct.
  5. The notify interrupt through `pynq.Interrupt("matmul_0/notify_irq")`:
     20 entries waited for by interrupt.
  6. Submit → completion latency (median): NOP 739 µs, small list 745 µs
     (polling) and 736 µs (interrupt). This is dominated by the Python and
     PYNQ overhead on the ARM, not the hardware.
- **Regression**: `m4_demo.py`, `m4_perf.py` (69/69) and `m5_desc_demo.py`
  all pass and match L0 cycle for cycle, apart from a few cycles of DDR
  latency jitter.

Needs `firmware/{gemm,vector,bwtest,matmul,matmul_insn,desc_run,rt}/*.bin`
and `driver/pynq_matmul.py` from the same commit.

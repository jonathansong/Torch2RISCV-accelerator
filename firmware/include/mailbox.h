/*
 * ARM <-> PicoRV32 mailbox for the matmul batch firmware, in the last
 * 256 bytes of the program BRAM (RISC-V 0xC0001F00, ARM 0x40010000+0x1F00).
 * Shared by all firmware directories (matmul/, matmul_insn/, gemm/,
 * bwtest/, vector/). Keep in sync with driver/pynq_matmul.py and sim/tb_system.v.
 *
 * Job i uses A = A_BASE + 64*i, B = B_BASE + 64*i, C = C_BASE + 256*i and
 * descriptor D = DESC_BASE + 16*i = {A, B, DIM, 0} (DDR physical addresses;
 * buffers must not straddle 4 KB pages, which holds for page-aligned bases).
 */
#ifndef MAILBOX_H
#define MAILBOX_H

#define MBOX_BASE          0xC0001F00u
#define MBOX_OFFSET        0x1F00u

#define MBOX_STATUS        0x00  /* fw: STATUS_*                               */
#define MBOX_N_JOBS        0x04  /* in                                          */
#define MBOX_A_BASE        0x08  /* in                                          */
#define MBOX_B_BASE        0x0C  /* in                                          */
#define MBOX_C_BASE        0x10  /* in                                          */
#define MBOX_JOBS_DONE     0x14  /* out: jobs finished (progress)               */
#define MBOX_ERRORS        0x18  /* out: jobs that ended with STATUS.error      */
#define MBOX_FIRST_ERR     0x1C  /* out: STATUS of the first failing job        */
#define MBOX_TOTAL_CYCLES  0x20  /* out: RISC-V cycles for the whole batch      */
#define MBOX_ACCEL_CYCLES  0x24  /* out: sum of matmul CYCLES over all jobs     */
#define MBOX_UNIT_ID       0x28  /* out: matmul ID register as read by the core */
#define MBOX_DESC_BASE     0x2C  /* in:  descriptor table (mat_trigger path)    */
#define MBOX_GEMM_M        0x30  /* in:  GEMM firmware: C (M x N) = A (M x K) B  */
#define MBOX_GEMM_N        0x34  /*      (+ bias), all multiples of 8, row-major  */
#define MBOX_GEMM_K        0x38
#define MBOX_BIAS_BASE     0x3C  /* in:  int32 M x N bias, 0 = none              */
#define MBOX_EXT_STATUS    0x40  /* out: mat_fence extended status at the end    */
#define MBOX_BW_SRC        0x08  /* bwtest firmware: source buffer (= A_BASE)     */
#define MBOX_BW_DST        0x10  /*                  destination (= C_BASE)       */
#define MBOX_BW_CYCLES     0x44  /* out: bwtest cycles per test, 6 words          */
/* M3 vector engine (firmware/vector: standalone op; firmware/gemm: epilogue).
 * VE parameters, same meaning as the vec_cfg keys (sysarray_intrinsics.h): */
#define MBOX_V_LEN         0x60  /* in:  vector firmware: elements, multiple of 8 */
#define MBOX_V_OP          0x64  /* in:  op | RELU << 4 | REQUANT << 5            */
#define MBOX_V_TYPES       0x68  /* in:  vector firmware: in | out << 2           */
#define MBOX_V_PERIOD      0x6C  /* in:  vector firmware: src2 period in groups   */
#define MBOX_V_SCALE       0x70  /* in:  REQUANT: y = ((x*scale + rnd) >> shift)  */
#define MBOX_V_SHIFT       0x74  /*      + zp, clamped to [lo, hi]                */
#define MBOX_V_ZP          0x78
#define MBOX_V_LO          0x7C
#define MBOX_V_HI          0x80
#define MBOX_GEMM_Q        0x84  /* in:  GEMM firmware: 1 = int8 output through the
                                  *      VE epilogue (+ bias vector, V_OP RELU,
                                  *      REQUANT with V_SCALE..V_HI); BIAS_BASE is
                                  *      then an int32 vector of N (0 = none)      */
#define MBOX_GEMM_FLAGS    0x88  /* in:  GEMM firmware schedule switches (0 = tuned
                                  *      default): bit 0 no A prefetch, bit 1 no B
                                  *      split, bit 2 B split even for B < 16 KB */
#define GEMM_NO_PREFETCH   (1u << 0)
#define GEMM_NO_BSPLIT     (1u << 1)
#define GEMM_FORCE_BSPLIT  (1u << 2)

#define STATUS_RUNNING     0x00000001u
#define STATUS_DONE        0x600D600Du

#define ERR_NO_UNIT        0xDEAD0001u   /* FIRST_ERR: ID register mismatch  */
#define ERR_TIMEOUT        0xDEAD0002u   /* FIRST_ERR: job never set done    */

#endif

/*
 * C wrappers for the matmul custom instructions (docs/custom_isa_encoding.md).
 *
 * R-type in the custom-0 opcode space (0x0B), funct7 = group, funct3 = op.
 * Emitted with the assembler's `.insn` directive, so no compiler changes are
 * needed; the Phase 5 MLIR lowering emits the same `.insn` strings as LLVM
 * inline asm.
 *
 * funct7 = 0 is executed by rtl/matmul/matmul_pcpi.v (Phase 4) or
 * rtl/sysarray/sa_legacy.v; funct7 = 1 / 2 by rtl/sysarray (M1-M3).
 */
#ifndef SYSARRAY_INTRINSICS_H
#define SYSARRAY_INTRINSICS_H

#include <stdint.h>

/* Job descriptor read by the accelerator from DDR (16-byte aligned). */
struct mat_desc {
    uint32_t a;          /* 8x8 int8 A, row-major, 8-byte aligned  */
    uint32_t b;          /* 8x8 int8 B                              */
    uint32_t dim;        /* MAT_DIM(8, 8, 8)                        */
    uint32_t reserved;   /* 0                                       */
} __attribute__((aligned(16)));

#define MAT_DIM(m, n, k)  ((uint32_t)(m) | ((uint32_t)(n) << 10) | ((uint32_t)(k) << 20))

/* STATUS word returned by mat_status / mat_wait (same as the STATUS CSR) */
#define MAT_STATUS_DONE          (1u << 0)
#define MAT_STATUS_BUSY          (1u << 1)
#define MAT_STATUS_ERROR         (1u << 2)
#define MAT_STATUS_ERR_CODE(s)   (((s) >> 8) & 0xF)

/*
 * mat_trigger: start C = A * B for the job described at `desc`, writing C
 * (8x8 int32) to `dst`. Returns as soon as the unit accepts the job; stalls
 * only while a previous job is still running.
 */
static inline void mat_trigger(uint32_t desc, uint32_t dst)
{
    __asm__ volatile (".insn r 0x0B, 0, 0, x0, %0, %1" :: "r"(desc), "r"(dst) : "memory");
}

/* mat_status: current STATUS, never stalls. */
static inline uint32_t mat_status(void)
{
    uint32_t rd;
    __asm__ volatile (".insn r 0x0B, 1, 0, %0, x0, x0" : "=r"(rd));
    return rd;
}

/* mat_reset: clear done/error/IRQ status (waits for a running job to end). */
static inline void mat_reset(void)
{
    __asm__ volatile (".insn r 0x0B, 2, 0, x0, x0, x0" ::: "memory");
}

/* mat_wait: stall until the unit is idle, return STATUS. */
static inline uint32_t mat_wait(void)
{
    uint32_t rd;
    __asm__ volatile (".insn r 0x0B, 3, 0, %0, x0, x0" : "=r"(rd) :: "memory");
    return rd;
}

/* mat_cycles: accelerator cycles of the last job. */
static inline uint32_t mat_cycles(void)
{
    uint32_t rd;
    __asm__ volatile (".insn r 0x0B, 4, 0, %0, x0, x0" : "=r"(rd));
    return rd;
}

/* ------------------------------------------------------------------------
 * funct7 = 1: double-buffered accelerator (docs/double_buffer_design.md §8)
 * Commands are queued and return immediately (they stall only while the
 * command queue is full); ordering between engines is kept by the
 * hardware bank scoreboard. DDR is NOT tracked: fence between a store and
 * a later load of the same DDR bytes.
 * ---------------------------------------------------------------------- */

/*
 * Capabilities of the loaded overlay, from the CAPS CSR (RISC-V address
 * 0x80000024): [7:0] D, [15:8] VL, [19:16] DMA ports, [20] performance
 * counters. sa_init() reads it once. The array size D (8 or 16) is taken
 * from it at run time, so one firmware image runs on every build;
 * -DSA_D=<n> fixes D at compile time.
 */
#define SA_CAPS_ADDR    0x80000024u
#define SA_CAPS_PERF    (1u << 20)
#define SA_CAPS_DESC    (1u << 21)
#define SA_CAPS_NOTIFY  (1u << 22)    /* L1: mat_notify + notify_irq                  */
#define SA_CAPS_CMDX    (1u << 24)    /* L1: BASE0-15, PARAM0-7, dynamic fields,
                                       *     SETREG, LOOP_END, CALL / RET, LDPARAM   */
#define SA_CAPS_FPVE    (1u << 23)    /* L2: fp32 vector engine, SFU, TRANSPOSE       */
static uint32_t sa_caps_runtime = 0u;
#ifndef SA_D
static uint32_t sa_d_runtime = 8u;
#define SA_D            sa_d_runtime
static inline uint32_t sa_init(void)
{
    uint32_t caps = *(volatile uint32_t *)SA_CAPS_ADDR;
    uint32_t d = caps & 0xFFu;
    sa_caps_runtime = caps;
    sa_d_runtime = (d == 8u || d == 16u) ? d : 8u;
    return sa_d_runtime;
}
#else
static inline uint32_t sa_init(void)
{
    sa_caps_runtime = *(volatile uint32_t *)SA_CAPS_ADDR;
    return SA_D;
}
#endif
/* the overlay has performance counters (mat_perf would trap without them) */
static inline int sa_has_perf(void) { return (sa_caps_runtime & SA_CAPS_PERF) != 0; }
/* the overlay has the descriptor fetch unit (mat_submit would trap without it) */
static inline int sa_has_desc(void) { return (sa_caps_runtime & SA_CAPS_DESC) != 0; }
/* L1: mat_notify (host interrupt) and the command extensions */
static inline int sa_has_notify(void) { return (sa_caps_runtime & SA_CAPS_NOTIFY) != 0; }
static inline int sa_has_cmdx(void) { return (sa_caps_runtime & SA_CAPS_CMDX) != 0; }
/* L2: fp32 vector engine (docs/llm_inference_plan.md §3.2) */
static inline int sa_has_fpve(void) { return (sa_caps_runtime & SA_CAPS_FPVE) != 0; }
#define SA_SPAD_WORDS   (131072u / SA_D)            /* per SPAD, D-byte words      */
#define SA_ACC_WORDS    (262144u / (4u * SA_D))     /* D x int32 words             */
#define SA_SPAD_BANK    (SA_SPAD_WORDS / 2u)        /* first word of bank 1        */
#define SA_ACC_BANK     (SA_ACC_WORDS / 2u)

#define SA_MEM_SPAD_A   1u
#define SA_MEM_SPAD_B   2u
#define SA_MEM_ACC      3u
#define SA_LADDR(mem, word)  (((uint32_t)(mem) << 28) | (uint32_t)(word))

/* mat_cfg keys */
#define SA_CFG_LD_ROWS       0u
#define SA_CFG_LD_ROW_BYTES  1u
#define SA_CFG_LD_PITCH      2u
#define SA_CFG_LD_MODE       3u
#define SA_CFG_ST_ROWS       4u
#define SA_CFG_ST_ROW_BYTES  5u
#define SA_CFG_ST_PITCH      6u
#define SA_CFG_EX_REPEAT     7u      /* M2: C tiles per mat_exec          */
#define SA_CFG_EX_B_STEP     8u      /*     SPAD_B words between B strips */
#define SA_CFG_EX_C_STEP     9u      /*     ACC words between tiles       */
#define SA_CFG_EX_C_ROW      10u     /*     ACC words between tile rows   */
#define SA_CFG_BASE(n)       (11u + (n))  /* descriptor relocation bases BASE0..BASE3
                                           *  (BASE0..BASE15 with SA_CAPS_CMDX)   */
#define SA_CFG_PARAM(n)      (27u + (n))  /* L1: descriptor parameters PARAM0..PARAM7 */
#define SA_LD_LINEAR         0u
#define SA_LD_INTERLEAVE     1u

/* mat_fence engine mask / extended status */
#define SA_ENG_ALL           0u
#define SA_XST_IDLE          (1u << 0)
#define SA_XST_ERROR         (1u << 1)
#define SA_XST_CODE(s)       (((s) >> 8) & 0xFu)
#define SA_XST_ENGINE(s)     (((s) >> 12) & 0xFu)

static inline void mat_cfg(uint32_t key, uint32_t value)
{
    __asm__ volatile (".insn r 0x0B, 0, 1, x0, %0, %1" :: "r"(key), "r"(value));
}

/* DDR -> local memory, shape from the LD_* configuration */
static inline void mat_load(uint32_t ddr, uint32_t laddr)
{
    __asm__ volatile (".insn r 0x0B, 1, 1, x0, %0, %1" :: "r"(ddr), "r"(laddr) : "memory");
}

/* local memory -> DDR, shape from the ST_* configuration */
static inline void mat_store(uint32_t ddr, uint32_t laddr)
{
    __asm__ volatile (".insn r 0x0B, 2, 1, x0, %0, %1" :: "r"(ddr), "r"(laddr) : "memory");
}

/* C (ACC word c) (+)= A strip (SPAD_A word a) x B strip (SPAD_B word b), Kt k-tiles */
static inline void mat_exec(uint32_t a, uint32_t b, uint32_t c, uint32_t kt, uint32_t accumulate)
{
    uint32_t rs1 = (b << 16) | (a & 0xFFFFu);
    uint32_t rs2 = (accumulate ? (1u << 28) : 0u) | ((kt & 0xFFFu) << 16) | (c & 0xFFFFu);
    __asm__ volatile (".insn r 0x0B, 3, 1, x0, %0, %1" :: "r"(rs1), "r"(rs2) : "memory");
}

/* wait until the queue is empty and the engines in `mask` are idle; extended status */
static inline uint32_t mat_fence(uint32_t mask)
{
    uint32_t rd;
    __asm__ volatile (".insn r 0x0B, 4, 1, %0, %1, x0" : "=r"(rd) : "r"(mask) : "memory");
    return rd;
}

static inline void mat_cfg_load(uint32_t rows, uint32_t row_bytes, uint32_t pitch, uint32_t mode)
{
    mat_cfg(SA_CFG_LD_ROWS, rows);
    mat_cfg(SA_CFG_LD_ROW_BYTES, row_bytes);
    mat_cfg(SA_CFG_LD_PITCH, pitch);
    mat_cfg(SA_CFG_LD_MODE, mode);
}

/* mat_exec computes `repeat` C tiles: tile r uses B strip b + r*b_step and
 * writes its row i to ACC word c + r*c_step + i*c_row */
static inline void mat_cfg_exec(uint32_t repeat, uint32_t b_step, uint32_t c_step, uint32_t c_row)
{
    mat_cfg(SA_CFG_EX_REPEAT, repeat);
    mat_cfg(SA_CFG_EX_B_STEP, b_step);
    mat_cfg(SA_CFG_EX_C_STEP, c_step);
    mat_cfg(SA_CFG_EX_C_ROW, c_row);
}

static inline void mat_cfg_store(uint32_t rows, uint32_t row_bytes, uint32_t pitch)
{
    mat_cfg(SA_CFG_ST_ROWS, rows);
    mat_cfg(SA_CFG_ST_ROW_BYTES, row_bytes);
    mat_cfg(SA_CFG_ST_PITCH, pitch);
}

/* ------------------------------------------------------------------------
 * funct7 = 2: vector engine (M3, docs/double_buffer_design.md §6)
 * vec_run queues dst[g] = post(op(src1[g], src2[g'])) over LEN / SA_D
 * groups of SA_D elements; g' = g (period 0), 0 (period 1) or g mod period.
 * Types live in fixed memories: int32 in ACC (1 word per group), int8 in
 * SPAD (1 word), int16 in SPAD (2 words). Ordered against LD/ST/EX by the
 * same bank scoreboard; mat_fence waits for it too.
 * ---------------------------------------------------------------------- */
#define SA_VCFG_OP           0u      /* op | SA_V_RELU | SA_V_REQUANT              */
#define SA_VCFG_LEN          1u      /* elements, multiple of SA_D                 */
#define SA_VCFG_DST          2u      /* LADDR                                      */
#define SA_VCFG_TYPES        3u      /* in | out << 2                              */
#define SA_VCFG_SRC2_MOD     4u      /* src2 period in groups (0 = elementwise)    */
#define SA_VCFG_SCALE        5u      /* REQUANT: y = ((x*scale + 2^(shift-1))      */
#define SA_VCFG_SHIFT        6u      /*          >> shift) + zp, clamp [lo, hi]    */
#define SA_VCFG_ZP           7u
#define SA_VCFG_CLAMP_LO     8u
#define SA_VCFG_CLAMP_HI     9u
/* L2 (SA_CAPS_FPVE); only read by fp commands (FLAGS bit 0) and TRANSPOSE */
#define SA_VCFG_FLAGS       10u      /* [0] FP [3:1] FUNC [5:4] M1 [7:6] M2        */
                                     /* [9:8] REDUCE [10] SWAPNEG                  */
#define SA_VCFG_IMM         11u      /* fp32 bits: src2 immediate (M2 = IMM)       */
#define SA_VCFG_A           12u      /* fp32 bits: y = op(...) * A + B             */
#define SA_VCFG_B           13u
#define SA_VCFG_ROW         14u      /* [15:0] ROWLEN, [31:16] VALID               */
#define SA_VCFG_P1S         15u      /* [15:0] P1, [31:16] S (TRANSPOSE)           */
#define SA_VF_FP             1u
#define SA_VFUNC(f)          ((uint32_t)(f) << 1)   /* 0 none 1 EXP 2 RECIP 3 RSQRT 4 ABS */
#define SA_VM1(m)            ((uint32_t)(m) << 4)   /* 0 LIN 1 MOD 2 DIV 3 IMM             */
#define SA_VM2(m)            ((uint32_t)(m) << 6)
#define SA_VRED(r)           ((uint32_t)(r) << 8)   /* 0 none 1 SUM 2 MAX                  */
#define SA_VF_SWAPNEG        (1u << 10)

#define SA_VOP_ADD           0u      /* int32 saturating                           */
#define SA_VOP_SUB           1u
#define SA_VOP_MUL           2u      /* int8 / int16 inputs only                   */
#define SA_VOP_MAX           3u
#define SA_VOP_MIN           4u
#define SA_VOP_COPY          5u      /* unary: src2 ignored                        */
#define SA_V_RELU            (1u << 4)
#define SA_V_REQUANT         (1u << 5)
#define SA_VT_I8             0u
#define SA_VT_I16            1u
#define SA_VT_I32            2u
#define SA_VT_F32            3u      /* L2: ACC only                               */
#define SA_VT2(t)            ((uint32_t)(t) << 4)   /* L2 src2 type: 0 = as src1, 1 I8, 2 I32, 3 F32 */
#define SA_VOP_TRANSPOSE     6u      /* L2: int8 D x D blocks, S from SA_VCFG_P1S  */
#define SA_VTYPES(in, out)   ((uint32_t)(in) | ((uint32_t)(out) << 2))

static inline void vec_cfg(uint32_t key, uint32_t value)
{
    __asm__ volatile (".insn r 0x0B, 0, 2, x0, %0, %1" :: "r"(key), "r"(value));
}

/* queue one vector command reading LADDRs src1 / src2 (all other operands from vec_cfg) */
static inline void vec_run(uint32_t src1, uint32_t src2)
{
    __asm__ volatile (".insn r 0x0B, 1, 2, x0, %0, %1" :: "r"(src1), "r"(src2) : "memory");
}

/* ------------------------------------------------------------------------
 * Performance counters (funct7 = 1, funct3 = 5; rtl/sysarray/sa_perf.v,
 * docs/perf_counters_and_desc_dma_plan.md part 1). Only when sa_has_perf():
 * on an overlay without counters the instruction is not answered and the
 * core traps.
 * ---------------------------------------------------------------------- */
#define SA_PERF_CLEAR        (1u << 0)
#define SA_PERF_ENABLE       (1u << 1)
enum {
    SA_PC_CYCLES = 0, SA_PC_CMD_LD, SA_PC_CMD_ST, SA_PC_CMD_EX, SA_PC_CMD_VE,
    SA_PC_PCPI_QFULL, SA_PC_PCPI_FENCE, SA_PC_HAZ_LD, SA_PC_HAZ_ST, SA_PC_HAZ_EX,
    SA_PC_HAZ_VE, SA_PC_DISP_FULL, SA_PC_STARVE, SA_PC_ALL_IDLE, SA_PC_EX_STEP,
    SA_PC_EX_USEFUL, SA_PC_EX_SWAPWAIT, SA_PC_EX_TILES, SA_PC_LD_BUSY, SA_PC_LD_BEATS,
    SA_PC_LD_ARSTALL, SA_PC_ST_BUSY, SA_PC_ST_BEATS, SA_PC_ST_WSTALL, SA_PC_VE_ACTIVE,
    SA_PC_VE_RDBLOCK, SA_PC_VE_CREDIT, SA_PC_VE_GROUPS,
    SA_PC_NUM = 32                              /* 28..31 reserved */
};

/* control: SA_PERF_CLEAR / SA_PERF_ENABLE (0 = freeze); returns the counter count */
static inline uint32_t mat_perf_ctl(uint32_t flags)
{
    uint32_t rd;
    __asm__ volatile (".insn r 0x0B, 5, 1, %0, %1, %2" : "=r"(rd) : "r"(1u << 31), "r"(flags) : "memory");
    return rd;
}

/* read counter idx */
static inline uint32_t mat_perf_read(uint32_t idx)
{
    uint32_t rd;
    __asm__ volatile (".insn r 0x0B, 5, 1, %0, %1, x0" : "=r"(rd) : "r"(idx));
    return rd;
}

/* measurement window: clear + start (returns the counter count, 0 without counters) */
static inline uint32_t sa_perf_begin(void)
{
    return sa_has_perf() ? mat_perf_ctl(SA_PERF_CLEAR | SA_PERF_ENABLE) : 0u;
}

/* freeze and copy up to max counters to dst; returns how many were copied */
static inline uint32_t sa_perf_end(volatile uint32_t *dst, uint32_t max)
{
    if (!sa_has_perf())
        return 0u;
    uint32_t n = mat_perf_ctl(0);
    if (n > max)
        n = max;
    for (uint32_t i = 0; i < n; i++)
        dst[i] = mat_perf_read(i);
    return n;
}

/* ------------------------------------------------------------------------
 * Descriptor lists (funct7 = 1, funct3 = 6; rtl/sysarray/sa_cmdfetch.v,
 * docs/double_buffer_design.md §8.6). Only when sa_has_desc().
 * mat_submit starts the fetch unit on a 64-byte aligned list in DDR and
 * returns at once; queued PCPI commands issued afterwards wait for the list,
 * and mat_fence waits for it to finish. count 0 = run until END.
 * Status CSRs (RISC-V 0x80000000 + offset): 0xC0 last descriptor address,
 * 0xC4 lists finished, 0xC8 last END value, 0xCC failing index, 0xD0
 * descriptors decoded in the current list.
 * ---------------------------------------------------------------------- */
#define SA_CSR(off)          (*(volatile uint32_t *)(0x80000000u + (off)))
#define SA_CSR_DESC_ADDR     0xC0u
#define SA_CSR_DESC_DONE     0xC4u
#define SA_CSR_DESC_STATUS   0xC8u
#define SA_CSR_DESC_ERR_IDX  0xCCu
#define SA_CSR_DESC_EXEC     0xD0u

static inline void mat_submit(uint32_t list, uint32_t count)
{
    __asm__ volatile (".insn r 0x0B, 6, 1, x0, %0, %1" :: "r"(list), "r"(count) : "memory");
}

/* L1 (docs/llm_inference_plan.md §5.1): mat_notify = funct7 1, funct3 7.
 * One pulse on the unit's notify_irq (host interrupt) and NOTIFY_COUNT + 1
 * (CSR 0xD4). Only when sa_has_notify(). */
#define SA_CSR_NOTIFY_COUNT  0xD4u
static inline void mat_notify(void)
{
    __asm__ volatile (".insn r 0x0B, 7, 1, x0, x0, x0" ::: "memory");
}

/* REQUANT parameters and output clamp window */
static inline void vec_cfg_requant(int32_t scale, uint32_t shift, int32_t zp, int32_t lo, int32_t hi)
{
    vec_cfg(SA_VCFG_SCALE, (uint32_t)scale);
    vec_cfg(SA_VCFG_SHIFT, shift);
    vec_cfg(SA_VCFG_ZP, (uint32_t)zp);
    vec_cfg(SA_VCFG_CLAMP_LO, (uint32_t)lo);
    vec_cfg(SA_VCFG_CLAMP_HI, (uint32_t)hi);
}

#endif

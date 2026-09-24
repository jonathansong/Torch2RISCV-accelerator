/*
 * C wrappers for the matmul custom instructions (docs/custom_isa_encoding.md).
 *
 * R-type in the custom-0 opcode space (0x0B), funct7 = 0, funct3 = op.
 * Emitted with the assembler's `.insn` directive, so no compiler changes are
 * needed; the Phase 5 MLIR lowering emits the same `.insn` strings as LLVM
 * inline asm.
 *
 * Executed by rtl/matmul/matmul_pcpi.v through the PicoRV32 PCPI port.
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

#ifndef SA_D
#define SA_D            8u                          /* array size of the build     */
#endif
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

static inline void mat_cfg_store(uint32_t rows, uint32_t row_bytes, uint32_t pitch)
{
    mat_cfg(SA_CFG_ST_ROWS, rows);
    mat_cfg(SA_CFG_ST_ROW_BYTES, row_bytes);
    mat_cfg(SA_CFG_ST_PITCH, pitch);
}

#endif

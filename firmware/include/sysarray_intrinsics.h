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

#endif

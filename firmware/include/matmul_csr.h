/*
 * matmul_unit CSRs as seen from the PicoRV32 (rtl/matmul/README.md).
 * The block design maps them at 0x80000000 on the RISC-V bus only.
 */
#ifndef MATMUL_CSR_H
#define MATMUL_CSR_H

#include <stdint.h>

#define MATMUL_BASE        0x80000000u

#define MATMUL_CTRL        0x00
#define MATMUL_STATUS      0x04
#define MATMUL_SRC_A       0x08
#define MATMUL_SRC_B       0x0C
#define MATMUL_DST         0x10
#define MATMUL_DIM         0x14
#define MATMUL_IRQ_STATUS  0x18
#define MATMUL_CYCLES      0x1C
#define MATMUL_ID          0x20

#define MATMUL_CTRL_START       (1u << 0)
#define MATMUL_CTRL_SOFT_RESET  (1u << 1)
#define MATMUL_CTRL_IRQ_EN      (1u << 2)

#define MATMUL_STATUS_DONE      (1u << 0)
#define MATMUL_STATUS_BUSY      (1u << 1)
#define MATMUL_STATUS_ERROR     (1u << 2)
#define MATMUL_STATUS_ERR_CODE(s) (((s) >> 8) & 0xF)

#define MATMUL_ID_VALUE    0x4D4D3038u   /* "MM08" */

/* 8x8x8 tile: A, B are 8x8 int8 (64 B), C is 8x8 int32 (256 B) */
#define MATMUL_A_BYTES     64u
#define MATMUL_B_BYTES     64u
#define MATMUL_C_BYTES     256u

static inline void matmul_write(uint32_t off, uint32_t v)
{
    *(volatile uint32_t *)(MATMUL_BASE + off) = v;
}

static inline uint32_t matmul_read(uint32_t off)
{
    return *(volatile uint32_t *)(MATMUL_BASE + off);
}

#endif

/*
 * PicoRV32 DDR access test (runs from BRAM, touches DDR through S_AXI_HP0).
 *
 *   1. read   : sum all src words                    -> MBOX_SRC_SUM
 *   2. write  : dst[i] = src[i] * 3 + i               (32-bit stores)
 *   3. sub-word stores after the words (checks WSTRB through the
 *      32->64 bit width converter):
 *        bytes  b[i] = i ^ 0x5A        (SUB_BYTES of them)
 *        halves h[i] = 0xA000 + 3*i    (SUB_HALVES of them)
 *   4. read everything back and count mismatches     -> MBOX_ERRORS
 *
 * The ARM side checks the same values independently after
 * invalidating its view of the buffers.
 */
#include <stdint.h>
#include "mailbox.h"

#define MBOX(off) (*(volatile uint32_t *)(MBOX_BASE + (off)))

static inline uint32_t rdcycle(void)
{
    uint32_t c;
    __asm__ volatile ("rdcycle %0" : "=r"(c));
    return c;
}

int main(void)
{
    MBOX(MBOX_STATUS) = STATUS_RUNNING;

    volatile uint32_t *src = (volatile uint32_t *)MBOX(MBOX_SRC_ADDR);
    volatile uint32_t *dst = (volatile uint32_t *)MBOX(MBOX_DST_ADDR);
    uint32_t n = MBOX(MBOX_N_WORDS);
    volatile uint8_t  *b = (volatile uint8_t  *)(dst + n);
    volatile uint16_t *h = (volatile uint16_t *)(b + SUB_BYTES);

    uint32_t t0 = rdcycle();

    uint32_t sum = 0;
    for (uint32_t i = 0; i < n; i++)
        sum += src[i];

    for (uint32_t i = 0; i < n; i++)
        dst[i] = src[i] * 3 + i;
    for (uint32_t i = 0; i < SUB_BYTES; i++)
        b[i] = (uint8_t)(i ^ 0x5A);
    for (uint32_t i = 0; i < SUB_HALVES; i++)
        h[i] = (uint16_t)(0xA000 + 3 * i);

    uint32_t errors = 0;
    for (uint32_t i = 0; i < n; i++)
        errors += dst[i] != src[i] * 3 + i;
    for (uint32_t i = 0; i < SUB_BYTES; i++)
        errors += b[i] != (uint8_t)(i ^ 0x5A);
    for (uint32_t i = 0; i < SUB_HALVES; i++)
        errors += h[i] != (uint16_t)(0xA000 + 3 * i);

    MBOX(MBOX_CYCLES)  = rdcycle() - t0;
    MBOX(MBOX_SRC_SUM) = sum;
    MBOX(MBOX_ERRORS)  = errors;
    MBOX(MBOX_STATUS)  = STATUS_DONE;
    return 0;
}

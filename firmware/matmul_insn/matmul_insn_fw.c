/*
 * Phase 4 firmware: run a batch of 8x8x8 int8 matmuls through the custom
 * matrix instructions (PCPI), without touching the matmul CSRs.
 *
 * Per job: mat_trigger(descriptor, C) then mat_wait(); the descriptors
 * {A, B, DIM, 0} are prepared in DDR by the ARM (driver/pynq_matmul.py).
 * Same mailbox and reporting as the CSR firmware (firmware/matmul), so the
 * two paths can be compared directly.
 */
#include <stdint.h>
#include "mailbox.h"
#include "sysarray_intrinsics.h"

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

    uint32_t n         = MBOX(MBOX_N_JOBS);
    uint32_t desc_base = MBOX(MBOX_DESC_BASE);
    uint32_t c_base    = MBOX(MBOX_C_BASE);
    uint32_t errors = 0, accel_cycles = 0, done = 0;

    mat_reset();

    uint32_t t0 = rdcycle();
    for (uint32_t i = 0; i < n; i++) {
        mat_trigger(desc_base + i * sizeof(struct mat_desc), c_base + i * 256u);
        uint32_t status = mat_wait();
        if (status & MAT_STATUS_ERROR) {
            if (errors == 0)
                MBOX(MBOX_FIRST_ERR) = status;
            errors++;
        }
        accel_cycles += mat_cycles();
        MBOX(MBOX_JOBS_DONE) = ++done;
    }
    uint32_t t1 = rdcycle();

    MBOX(MBOX_TOTAL_CYCLES) = t1 - t0;
    MBOX(MBOX_ACCEL_CYCLES) = accel_cycles;
    MBOX(MBOX_ERRORS)       = errors;
    MBOX(MBOX_STATUS)       = STATUS_DONE;
    return 0;
}

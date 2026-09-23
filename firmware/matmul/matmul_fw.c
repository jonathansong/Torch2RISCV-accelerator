/*
 * Phase 3 firmware: run a batch of 8x8x8 int8 matmuls on matmul_unit
 * through its CSRs (loosely coupled path).
 *
 * Per job: program SRC_A / SRC_B / DST, set CTRL.start, poll STATUS.done.
 * Results and timing are reported through the mailbox (mailbox.h).
 */
#include <stdint.h>
#include "mailbox.h"
#include "matmul_csr.h"

#define MBOX(off) (*(volatile uint32_t *)(MBOX_BASE + (off)))

#define POLL_LIMIT 100000u   /* one job takes ~200 cycles; this is ~ms */

static inline uint32_t rdcycle(void)
{
    uint32_t c;
    __asm__ volatile ("rdcycle %0" : "=r"(c));
    return c;
}

static void report_error(uint32_t *errors, uint32_t status)
{
    if (*errors == 0)
        MBOX(MBOX_FIRST_ERR) = status;
    *errors += 1;
}

int main(void)
{
    MBOX(MBOX_STATUS) = STATUS_RUNNING;

    uint32_t n      = MBOX(MBOX_N_JOBS);
    uint32_t a_base = MBOX(MBOX_A_BASE);
    uint32_t b_base = MBOX(MBOX_B_BASE);
    uint32_t c_base = MBOX(MBOX_C_BASE);
    uint32_t errors = 0, accel_cycles = 0, done = 0;

    uint32_t id = matmul_read(MATMUL_ID);
    MBOX(MBOX_UNIT_ID) = id;
    if (id != MATMUL_ID_VALUE) {
        report_error(&errors, ERR_NO_UNIT);
        n = 0;
    }

    matmul_write(MATMUL_CTRL, MATMUL_CTRL_SOFT_RESET);
    matmul_write(MATMUL_IRQ_STATUS, 1);

    uint32_t t0 = rdcycle();
    for (uint32_t i = 0; i < n; i++) {
        matmul_write(MATMUL_SRC_A, a_base + i * MATMUL_A_BYTES);
        matmul_write(MATMUL_SRC_B, b_base + i * MATMUL_B_BYTES);
        matmul_write(MATMUL_DST,   c_base + i * MATMUL_C_BYTES);
        matmul_write(MATMUL_CTRL,  MATMUL_CTRL_START);

        uint32_t status, polls = 0;
        do {
            status = matmul_read(MATMUL_STATUS);
        } while (!(status & MATMUL_STATUS_DONE) && ++polls < POLL_LIMIT);

        if (!(status & MATMUL_STATUS_DONE)) {
            report_error(&errors, ERR_TIMEOUT);
            break;
        }
        if (status & MATMUL_STATUS_ERROR)
            report_error(&errors, status);
        accel_cycles += matmul_read(MATMUL_CYCLES);
        MBOX(MBOX_JOBS_DONE) = ++done;
    }
    uint32_t t1 = rdcycle();

    MBOX(MBOX_TOTAL_CYCLES) = t1 - t0;
    MBOX(MBOX_ACCEL_CYCLES) = accel_cycles;
    MBOX(MBOX_ERRORS)       = errors;
    MBOX(MBOX_STATUS)       = STATUS_DONE;
    return 0;
}

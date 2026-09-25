/*
 * Descriptor-list runner (docs/double_buffer_design.md §8.6).
 *
 * The ARM builds a list of 64-byte descriptors in DDR (driver/pynq_matmul.py
 * DescList) and passes its address, a descriptor count (0 = until END) and
 * the relocation bases BASE0..BASE3 through the mailbox. This firmware only
 * loads the bases, submits the list and waits for it: one mat_submit
 * instead of one PCPI instruction per command and configuration value.
 * Results: TOTAL_CYCLES (submit .. fence), EXT_STATUS, DL_STATUS (last END
 * value), DL_EXEC (descriptors decoded), performance counters (if present).
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
    sa_init();
    if (!sa_has_desc()) {                          /* mat_submit would trap */
        MBOX(MBOX_ERRORS)    = 1;
        MBOX(MBOX_FIRST_ERR) = ERR_NO_DESC;
        MBOX(MBOX_STATUS)    = STATUS_DONE;
        return 0;
    }
    uint32_t list = MBOX(MBOX_DL_ADDR), count = MBOX(MBOX_DL_COUNT);

    mat_reset();                                   /* clear any sticky error */
    for (uint32_t i = 0; i < 4; i++)
        mat_cfg(SA_CFG_BASE(i), MBOX(MBOX_DL_BASE0 + 4 * i));

    sa_perf_begin();
    uint32_t t0 = rdcycle();
    mat_submit(list, count);
    uint32_t st = mat_fence(SA_ENG_ALL);
    uint32_t t1 = rdcycle();
    MBOX(MBOX_PERF_COUNT) = sa_perf_end(PERF_AREA, PERF_AREA_WORDS);

    MBOX(MBOX_TOTAL_CYCLES) = t1 - t0;
    MBOX(MBOX_EXT_STATUS)   = st;
    MBOX(MBOX_DL_STATUS)    = SA_CSR(SA_CSR_DESC_STATUS);
    MBOX(MBOX_DL_EXEC)      = SA_CSR(SA_CSR_DESC_EXEC);
    MBOX(MBOX_ERRORS)       = (st & SA_XST_ERROR) ? 1u : 0u;
    MBOX(MBOX_FIRST_ERR)    = st;
    MBOX(MBOX_STATUS)       = STATUS_DONE;
    return 0;
}

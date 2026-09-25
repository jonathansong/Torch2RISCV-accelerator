/*
 * Resident runtime firmware: the command processor of the LLM inference
 * system (docs/llm_inference_plan.md §5.1-5.2).
 *
 * Loaded once by the ARM, it serves a submission ring in DDR (read over
 * S_AXI_HP0): the ARM writes 64-byte entries, flushes them and rings the
 * doorbell (mailbox RING_TAIL); for each entry this firmware loads BASE0-3
 * and, from a parameter block, PARAM0-7, runs the descriptor list with one
 * mat_submit, writes a 32-byte completion record at the same index of the
 * completion ring, advances RING_HEAD and, if asked, raises the host
 * interrupt with mat_notify. A failing list is recorded (status =
 * extended status) and the unit is reset; later entries run normally.
 *
 * Entry (8 x u64): w0 [7:0] type, [8] IRQ, [9] PERF, [63:32] seq;
 *   w1 list address; w2 [31:0] count (0 = until END); w3..w6 BASE0..3;
 *   w7 [31:0] parameter block address (8 x u32, 0 = keep PARAM0..7).
 * Completion (8 x u32): seq, status (0 or extended status), cycles
 *   (mat_submit .. fence), descriptors decoded, END value, 0, 0, 0.
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
    MBOX(MBOX_FW_VERSION) = RT_VERSION;
    MBOX(MBOX_RING_HEAD) = 0;
    sa_init();
    uint32_t size = MBOX(MBOX_RING_SIZE);
    if (!sa_has_desc() || !sa_has_notify() || !sa_has_cmdx() || size == 0 || (size & (size - 1))) {
        MBOX(MBOX_FW_STATE) = (size == 0 || (size & (size - 1))) && sa_has_cmdx() ? ERR_BAD_RING : ERR_NO_L1;
        MBOX(MBOX_STATUS) = STATUS_DONE;
        return 0;
    }
    uint32_t ring = MBOX(MBOX_RING_BASE), cpl = MBOX(MBOX_CPL_BASE);
    uint32_t head = 0, beat = 0;
    mat_reset();                                        /* clear any sticky error */
    MBOX(MBOX_FW_STATE) = RT_READY;

    for (;;) {
        if (MBOX(MBOX_RING_TAIL) == head) {             /* doorbell polled from the BRAM */
            MBOX(MBOX_HEARTBEAT) = ++beat;
            continue;
        }
        uint32_t slot = head & (size - 1);
        volatile uint32_t *e = (volatile uint32_t *)(ring + 64u * slot);
        volatile uint32_t *c = (volatile uint32_t *)(cpl + 32u * slot);
        uint32_t type = e[0] & 0xFFu, flags = e[0], seq = e[1];
        uint32_t status = 0, cycles = 0, exec = 0, dstat = 0;

        if (type == RT_RUN_LIST) {
            for (uint32_t i = 0; i < 4; i++)
                mat_cfg(SA_CFG_BASE(i), e[6 + 2 * i]);
            uint32_t pb = e[14];
            if (pb)
                for (uint32_t k = 0; k < 8; k++)
                    mat_cfg(SA_CFG_PARAM(k), ((volatile uint32_t *)pb)[k]);
            if (flags & RT_F_PERF)
                sa_perf_begin();
            uint32_t t0 = rdcycle();
            mat_submit(e[2], e[4]);
            uint32_t st = mat_fence(SA_ENG_ALL);
            cycles = rdcycle() - t0;
            if (flags & RT_F_PERF)
                MBOX(MBOX_PERF_COUNT) = sa_perf_end(PERF_AREA, PERF_AREA_WORDS);
            exec  = SA_CSR(SA_CSR_DESC_EXEC);
            dstat = SA_CSR(SA_CSR_DESC_STATUS);
            if (st & SA_XST_ERROR) {
                status = st;
                mat_reset();
            }
        } else if (type == RT_RESET) {
            mat_reset();
        } else if (type != RT_NOP && type != RT_EXIT) {
            status = 0xFFFFFFFFu;                       /* unknown entry type */
        }

        c[1] = status; c[2] = cycles; c[3] = exec; c[4] = dstat;
        c[5] = 0; c[6] = 0; c[7] = 0;
        c[0] = seq;                                     /* stores complete in order (B response) */
        MBOX(MBOX_RING_HEAD) = ++head;
        if (flags & RT_F_IRQ)
            mat_notify();
        if (type == RT_EXIT)
            break;
    }
    MBOX(MBOX_STATUS) = STATUS_DONE;
    return 0;
}

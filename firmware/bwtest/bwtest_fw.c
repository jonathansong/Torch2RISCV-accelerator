/*
 * DMA bandwidth self-test (docs/double_buffer_design.md, M2).
 *
 * Mailbox: BW_SRC = 64 KB source, BW_DST = 192 KB destination (ARM-allocated).
 * Each test: rdcycle, queue the transfer(s), mat_fence, rdcycle.
 *   0  LD  src      -> SPAD_A    64 KB contiguous (long bursts)
 *   1  ST  SPAD_A   -> dst+0     64 KB            (dst[0..64K] == src)
 *   2  LD  src      -> ACC       64 KB contiguous
 *   3  ST  ACC      -> dst+64K   64 KB            (dst[64K..128K] == src)
 *   4  LD  src      -> SPAD_B    8192 rows of 8 B, pitch 16 (single-beat bursts)
 *   5  LD  src      -> SPAD_B 64 KB  and  ST SPAD_A -> dst+128K 64 KB, concurrently
 *   6  GEMV         64 KB of weights streamed (all read ports), x loaded first
 *   7  GEMV         the same, x resident: the streaming rate
 * Tests 6 / 7 (K2b) run when the unit has GEMV (CAPS bit 25) and the ARM put
 * a one-descriptor list in MBOX_DL_ADDR (x = SPAD_A words 0.., which hold
 * src after test 0; results ACC words 0..; then ST to dst+192K, 512 bytes).
 * Cycles go to MBOX_BW_CYCLES[i]; the ARM checks the destination data.
 */
#include <stdint.h>
#include "mailbox.h"
#include "sysarray_intrinsics.h"

#define MBOX(off) (*(volatile uint32_t *)(MBOX_BASE + (off)))
#define KB64      65536u

static inline uint32_t rdcycle(void)
{
    uint32_t c;
    __asm__ volatile ("rdcycle %0" : "=r"(c));
    return c;
}

static uint32_t errors;

static void record(int i, uint32_t t0)
{
    uint32_t st = mat_fence(SA_ENG_ALL);
    MBOX(MBOX_BW_CYCLES + 4 * i) = rdcycle() - t0;
    if (st & SA_XST_ERROR) {
        if (!errors) MBOX(MBOX_FIRST_ERR) = st;
        errors++;
        mat_reset();
    }
}

int main(void)
{
    MBOX(MBOX_STATUS) = STATUS_RUNNING;
    uint32_t src = MBOX(MBOX_BW_SRC), dst = MBOX(MBOX_BW_DST), t0;

    sa_init();
    mat_reset();
    /* contiguous 64 KB: 64 rows of 1 KB (merged into one long row by the DMA) */
    mat_cfg_load(64, 1024, 1024, SA_LD_LINEAR);
    mat_cfg_store(64, 1024, 1024);

    t0 = rdcycle(); mat_load(src, SA_LADDR(SA_MEM_SPAD_A, 0));        record(0, t0);
    t0 = rdcycle(); mat_store(dst, SA_LADDR(SA_MEM_SPAD_A, 0));       record(1, t0);
    t0 = rdcycle(); mat_load(src, SA_LADDR(SA_MEM_ACC, 0));           record(2, t0);
    t0 = rdcycle(); mat_store(dst + KB64, SA_LADDR(SA_MEM_ACC, 0));   record(3, t0);

    mat_cfg_load(8192, 8, 16, SA_LD_LINEAR);                           /* 8-byte rows */
    t0 = rdcycle(); mat_load(src, SA_LADDR(SA_MEM_SPAD_B, 0));        record(4, t0);

    mat_cfg_load(64, 1024, 1024, SA_LD_LINEAR);
    t0 = rdcycle();
    mat_load(src, SA_LADDR(SA_MEM_SPAD_B, 0));                         /* LD and ST engines */
    mat_store(dst + 2 * KB64, SA_LADDR(SA_MEM_SPAD_A, 0));             /* run concurrently  */
    record(5, t0);

    uint32_t list = MBOX(MBOX_DL_ADDR);
    if (list && sa_has_gemv() && sa_has_desc()) {
        t0 = rdcycle(); mat_submit(list, 1);                           record(6, t0);
        t0 = rdcycle(); mat_submit(list, 1);                           record(7, t0);
        mat_cfg_store(1, 512, 512);
        mat_store(dst + 3 * KB64, SA_LADDR(SA_MEM_ACC, 0));
        if (mat_fence(SA_ENG_ALL) & SA_XST_ERROR) {
            if (!errors) MBOX(MBOX_FIRST_ERR) = mat_fence(SA_ENG_ALL);
            errors++;
            mat_reset();
        }
    }

    MBOX(MBOX_ERRORS) = errors;
    MBOX(MBOX_STATUS) = STATUS_DONE;
    return 0;
}

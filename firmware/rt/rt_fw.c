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
 *   (mat_submit .. fence), descriptors decoded, END value, the core's
 *   instructions retired over the same window (version 7), 0, 0.
 * PERF (w0 bit 9): the unit's event counters over the entry into the perf
 *   area (RT_RUN_LIST; from version 7 also RT_CALL and RT_RUN_SCHED, the
 *   latter over the whole run).
 * RT_RUN_SCHED (version 3): w1 is a decode schedule (rvsched.h,
 *   docs/riscv_backend_design.md §7.1): the core runs the decode loop itself,
 *   the lists one after another; status = the schedule's, cycles = the whole run.
 * RT_RUN_SCHED tables (version 5: RVS2) may hold calls of generated code too.
 * RT_CALL (version 4): w1 is generated RISC-V code (docs/riscv_backend_design.md
 *   §9, plan A), called as f(args, args + 64) with w2 = args: the dispatch's
 *   16 binding addresses, then its push constants; it issues the dispatch's
 *   commands over PCPI. Then all engines idle; status = the extended status
 *   of an error, cycles = call .. idle.
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

static inline uint32_t rdinstret(void)
{
    uint32_t c;
    __asm__ volatile ("rdinstret %0" : "=r"(c));
    return c;
}

/* one list of a schedule: to its END, then all engines idle; 0 or the error status */
static uint32_t rt_submit(uint32_t list)
{
    mat_submit(list, 0);
    uint32_t st = mat_fence(SA_ENG_ALL);
    if (st & SA_XST_ERROR) {
        mat_reset();
        return st;
    }
    return 0;
}

/* generated RISC-V code (plan A): f(args, args + 16 words), then all engines idle; 0 or the error status */
static uint32_t rt_call(uint32_t fn, uint32_t args)
{
    ((void (*)(const uint32_t *, const uint32_t *))fn)((const uint32_t *)args, (const uint32_t *)(args + 64u));
    uint32_t st = mat_fence(SA_ENG_ALL);
    if (st & SA_XST_ERROR) {
        mat_reset();
        return st;
    }
    return 0;
}

#define RVS_CALL(c, a) rt_call(c, a)
#define RVS_RD(a)     (*(volatile uint32_t *)(a))
#define RVS_WR(a, v)  (*(volatile uint32_t *)(a) = (v))
#define RVS_SUBMIT(l) rt_submit(l)
#define RVS_CYCLES()  rdcycle()
#include "rvsched.h"

int main(void)
{
    MBOX(MBOX_STATUS) = STATUS_RUNNING;
    MBOX(MBOX_FW_VERSION) = RT_VERSION;
    MBOX(MBOX_RING_HEAD) = 0;
    sa_init();
    MBOX(MBOX_FW_CAPS) = sa_caps_runtime;          /* the host checks executables' required CAPS */
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
        uint32_t status = 0, cycles = 0, exec = 0, dstat = 0, instret = 0;

        if (type == RT_RUN_LIST) {
            for (uint32_t i = 0; i < 4; i++)
                mat_cfg(SA_CFG_BASE(i), e[6 + 2 * i]);
            uint32_t pb = e[14];
            if (pb)
                for (uint32_t k = 0; k < 8; k++)
                    mat_cfg(SA_CFG_PARAM(k), ((volatile uint32_t *)pb)[k]);
            if (flags & RT_F_PERF)
                sa_perf_begin();
            uint32_t i0 = rdinstret(), t0 = rdcycle();
            mat_submit(e[2], e[4]);
            uint32_t st = mat_fence(SA_ENG_ALL);
            cycles = rdcycle() - t0;
            instret = rdinstret() - i0;
            if (flags & RT_F_PERF)
                MBOX(MBOX_PERF_COUNT) = sa_perf_end(PERF_AREA, PERF_AREA_WORDS);
            exec  = SA_CSR(SA_CSR_DESC_EXEC);
            dstat = SA_CSR(SA_CSR_DESC_STATUS);
            if (st & SA_XST_ERROR) {
                status = st;
                mat_reset();
            }
        } else if (type == RT_RUN_SCHED || type == RT_CALL) {
            if (flags & RT_F_PERF)
                sa_perf_begin();
            uint32_t i0 = rdinstret(), t0 = rdcycle();
            status = type == RT_CALL ? rt_call(e[2], e[4]) : rvs_run(e[2]);
            cycles = rdcycle() - t0;
            instret = rdinstret() - i0;
            if (flags & RT_F_PERF)
                MBOX(MBOX_PERF_COUNT) = sa_perf_end(PERF_AREA, PERF_AREA_WORDS);
        } else if (type == RT_RESET) {
            mat_reset();
        } else if (type != RT_NOP && type != RT_EXIT) {
            status = 0xFFFFFFFFu;                       /* unknown entry type */
        }

        c[1] = status; c[2] = cycles; c[3] = exec; c[4] = dstat;
        c[5] = instret; c[6] = 0; c[7] = 0;
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

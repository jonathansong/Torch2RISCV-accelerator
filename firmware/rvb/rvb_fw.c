/*
 * RISC-V backend test (docs/riscv_backend_design.md §5, R2 / R3 first form):
 * one compiled dispatch (compiler/tests/rv_dispatch.py) run twice on the
 * unit, timed with rdcycle:
 *   phase 1  its descriptor list, as the driver submits it (mat_submit:
 *            SETREG of the registers, CALL of the template, END);
 *   phase 2  the same commands issued over PCPI by rvb_gen() (generated C,
 *            in the program BRAM or, RVB_CODE=ddr, in DDR).
 * Between the phases the testbench (firmware/sim/tb_system.v RVB_TEST)
 * compares the bindings with the functional simulator's result and restores
 * them; the firmware waits for it in the mailbox.
 */
#include <stdint.h>
#include "sysarray_intrinsics.h"
#include "rvb_args.h"

#define RV_MBOX   ((volatile uint32_t *)0xC0001FE0u)  /* mailbox 0xE0..0xFC (free after MBOX_FW_CAPS) */
enum { RV_PHASE, RV_GO, RV_CYC_DESC, RV_CYC_PCPI, RV_ST_DESC, RV_ST_PCPI, RV_MAGIC };

void rvb_gen(void);

static inline uint32_t rdcycle(void)
{
    uint32_t c;
    __asm__ volatile ("rdcycle %0" : "=r"(c));
    return c;
}

int main(void)
{
    uint32_t t0, t1;
    RV_MBOX[RV_MAGIC] = 0x52564230u;               /* "RVB0" */
    t0 = rdcycle();
    mat_submit(RVB_LIST, 0);
    RV_MBOX[RV_ST_DESC] = mat_fence(0);
    t1 = rdcycle();
    RV_MBOX[RV_CYC_DESC] = t1 - t0;
    RV_MBOX[RV_PHASE] = 1;
    while (RV_MBOX[RV_GO] != 2)
        ;
    t0 = rdcycle();
    rvb_gen();                                      /* (ends with mat_fence(0)) */
    RV_MBOX[RV_ST_PCPI] = mat_fence(0);
    t1 = rdcycle();
    RV_MBOX[RV_CYC_PCPI] = t1 - t0;
    RV_MBOX[RV_PHASE] = 2;
    return 0;
}

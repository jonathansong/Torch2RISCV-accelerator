/*
 * ARM <-> PicoRV32 mailbox for the matmul batch firmware, in the last
 * 256 bytes of the program BRAM (RISC-V 0xC0001F00, ARM 0x40010000+0x1F00).
 * Shared by the CSR firmware (matmul/) and the custom-instruction firmware
 * (matmul_insn/). Keep in sync with driver/pynq_matmul.py and sim/tb_system.v.
 *
 * Job i uses A = A_BASE + 64*i, B = B_BASE + 64*i, C = C_BASE + 256*i and
 * descriptor D = DESC_BASE + 16*i = {A, B, DIM, 0} (DDR physical addresses;
 * buffers must not straddle 4 KB pages, which holds for page-aligned bases).
 */
#ifndef MAILBOX_H
#define MAILBOX_H

#define MBOX_BASE          0xC0001F00u
#define MBOX_OFFSET        0x1F00u

#define MBOX_STATUS        0x00  /* fw: STATUS_*                               */
#define MBOX_N_JOBS        0x04  /* in                                          */
#define MBOX_A_BASE        0x08  /* in                                          */
#define MBOX_B_BASE        0x0C  /* in                                          */
#define MBOX_C_BASE        0x10  /* in                                          */
#define MBOX_JOBS_DONE     0x14  /* out: jobs finished (progress)               */
#define MBOX_ERRORS        0x18  /* out: jobs that ended with STATUS.error      */
#define MBOX_FIRST_ERR     0x1C  /* out: STATUS of the first failing job        */
#define MBOX_TOTAL_CYCLES  0x20  /* out: RISC-V cycles for the whole batch      */
#define MBOX_ACCEL_CYCLES  0x24  /* out: sum of matmul CYCLES over all jobs     */
#define MBOX_UNIT_ID       0x28  /* out: matmul ID register as read by the core */
#define MBOX_DESC_BASE     0x2C  /* in:  descriptor table (mat_trigger path)    */

#define STATUS_RUNNING     0x00000001u
#define STATUS_DONE        0x600D600Du

#define ERR_NO_UNIT        0xDEAD0001u   /* FIRST_ERR: ID register mismatch  */
#define ERR_TIMEOUT        0xDEAD0002u   /* FIRST_ERR: job never set done    */

#endif

/*
 * ARM <-> PicoRV32 mailbox, shared by ddr_test.c, sim/tb_ddr_test.v and
 * ddr_test.py. It sits in the last 256 bytes of the 8 KB program BRAM:
 *   RISC-V view: 0xC0001F00      ARM view: 0x40010000 + 0x1F00
 * All fields are 32-bit words; offsets are bytes from MBOX_BASE.
 */
#ifndef MAILBOX_H
#define MAILBOX_H

#define MBOX_BASE        0xC0001F00u
#define MBOX_OFFSET      0x1F00u      /* offset inside the BRAM */

#define MBOX_STATUS      0x00  /* RISC-V writes STATUS_*             */
#define MBOX_SRC_ADDR    0x04  /* in:  DDR phys addr of src buffer    */
#define MBOX_DST_ADDR    0x08  /* in:  DDR phys addr of dst buffer    */
#define MBOX_N_WORDS     0x0C  /* in:  number of 32-bit src words     */
#define MBOX_SRC_SUM     0x10  /* out: sum of src words (read test)   */
#define MBOX_ERRORS      0x14  /* out: self-check mismatches          */
#define MBOX_CYCLES      0x18  /* out: cycles spent in the test       */

#define STATUS_IDLE      0x00000000u
#define STATUS_RUNNING   0x00000001u
#define STATUS_DONE      0x600D600Du

/* dst layout: n_words words, then SUB_BYTES bytes, then SUB_HALVES halfwords */
#define SUB_BYTES        64
#define SUB_HALVES       32

#endif

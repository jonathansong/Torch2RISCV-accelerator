/*
 * M3 firmware: one vector-engine operation over DDR arrays, without matmul.
 *
 *   dst[i] = post(op(src1[i], src2[i']))       i = 0 .. LEN-1
 *   post   = RELU, REQUANT ((x*scale + 2^(shift-1)) >> shift) + zp,
 *            clamp [lo, hi], saturate to the output type
 *   i'     = i (period 0), i mod 8 (period 1: one broadcast group) or
 *            i mod 8*period (src2 holds `period` groups)
 *
 * src1 / src2: LEN int8, int16 or int32 (V_TYPES in), dst: LEN of the output
 * type, all in DDR (A_BASE, B_BASE, C_BASE, 8-byte aligned); LEN a multiple
 * of 8. Parameters and results: mailbox.h.
 *
 * The arrays stream through the local memories in chunks of up to 512
 * groups, alternating banks: while the VE works on chunk n in one bank, the
 * DMA loads chunk n+1 into the other and stores chunk n-1 (hardware bank
 * scoreboard, as in the GEMM firmware). Placement per bank:
 *   int8 / int16 in:  src1 SPAD_A word 0, src2 SPAD_B word 0
 *   int32 in:         src1 ACC word 0,    src2 ACC word 1024
 *   int8 / int16 out: SPAD_B word SA_SPAD_BANK/2;  int32 out: ACC word 2048
 */
#include <stdint.h>
#include "mailbox.h"
#include "sysarray_intrinsics.h"

#define MBOX(off) (*(volatile uint32_t *)(MBOX_BASE + (off)))

#define CHUNK_GROUPS  512u
#define ROW_BYTES     256u      /* DMA row size of the bulk of a transfer */

static inline uint32_t rdcycle(void)
{
    uint32_t c;
    __asm__ volatile ("rdcycle %0" : "=r"(c));
    return c;
}

static inline uint32_t esize(uint32_t t)     { return t == SA_VT_I8 ? 1u : t == SA_VT_I16 ? 2u : 4u; }
static inline uint32_t tmem_in(uint32_t t)   { return t == SA_VT_I32 ? SA_MEM_ACC : SA_MEM_SPAD_A; }
static inline uint32_t wbytes(uint32_t mem)  { return mem == SA_MEM_ACC ? 4u * SA_D : SA_D; }
static inline uint32_t bank(uint32_t mem, uint32_t b)
{
    return b * (mem == SA_MEM_ACC ? SA_ACC_BANK : SA_SPAD_BANK);
}

/* contiguous DDR <-> local words: 256-byte rows, then one tail row */
static void load(uint32_t ddr, uint32_t mem, uint32_t word, uint32_t bytes)
{
    uint32_t rows = bytes / ROW_BYTES, tail = bytes % ROW_BYTES;
    if (rows) {
        mat_cfg_load(rows, ROW_BYTES, ROW_BYTES, SA_LD_LINEAR);
        mat_load(ddr, SA_LADDR(mem, word));
    }
    if (tail) {
        mat_cfg_load(1, tail, tail, SA_LD_LINEAR);
        mat_load(ddr + rows * ROW_BYTES, SA_LADDR(mem, word + rows * ROW_BYTES / wbytes(mem)));
    }
}

static void store(uint32_t ddr, uint32_t mem, uint32_t word, uint32_t bytes)
{
    uint32_t rows = bytes / ROW_BYTES, tail = bytes % ROW_BYTES;
    if (rows) {
        mat_cfg_store(rows, ROW_BYTES, ROW_BYTES);
        mat_store(ddr, SA_LADDR(mem, word));
    }
    if (tail) {
        mat_cfg_store(1, tail, tail);
        mat_store(ddr + rows * ROW_BYTES, SA_LADDR(mem, word + rows * ROW_BYTES / wbytes(mem)));
    }
}

int main(void)
{
    MBOX(MBOX_STATUS) = STATUS_RUNNING;

    uint32_t src1 = MBOX(MBOX_A_BASE), src2 = MBOX(MBOX_B_BASE), dst = MBOX(MBOX_C_BASE);
    uint32_t len = MBOX(MBOX_V_LEN), op = MBOX(MBOX_V_OP), types = MBOX(MBOX_V_TYPES);
    uint32_t period = MBOX(MBOX_V_PERIOD);
    uint32_t it = types & 3u, ot = (types >> 2) & 3u;
    uint32_t unary = (op & 7u) == SA_VOP_COPY;

    uint32_t m1 = tmem_in(it);
    uint32_t m2 = it == SA_VT_I32 ? SA_MEM_ACC : SA_MEM_SPAD_B;
    uint32_t w2 = it == SA_VT_I32 ? 1024u : 0u;
    uint32_t md = ot == SA_VT_I32 ? SA_MEM_ACC : SA_MEM_SPAD_B;
    uint32_t wd = ot == SA_VT_I32 ? 2048u : SA_SPAD_BANK / 2u;
    uint32_t gin = SA_D * esize(it), gout = SA_D * esize(ot);   /* bytes per group */

    /* a chunk restarts the src2 period, so it holds whole periods */
    uint32_t cg = period > 1 ? (CHUNK_GROUPS / period) * period : CHUNK_GROUPS;
    uint32_t groups = len / SA_D, chunks = 0;

    mat_reset();
    uint32_t t0 = rdcycle();

    vec_cfg(SA_VCFG_OP, op);
    vec_cfg(SA_VCFG_TYPES, types);
    vec_cfg(SA_VCFG_SRC2_MOD, period);
    vec_cfg_requant((int32_t)MBOX(MBOX_V_SCALE), MBOX(MBOX_V_SHIFT), (int32_t)MBOX(MBOX_V_ZP),
                    (int32_t)MBOX(MBOX_V_LO), (int32_t)MBOX(MBOX_V_HI));
    if (!unary && period)                          /* periodic src2: resident in both banks */
        for (uint32_t b = 0; b < 2; b++)
            load(src2, m2, bank(m2, b) + w2, period * gin);

    for (uint32_t g = 0; g < groups; g += cg) {
        uint32_t n = groups - g < cg ? groups - g : cg;
        uint32_t b = chunks & 1u;
        load(src1 + g * gin, m1, bank(m1, b), n * gin);
        if (!unary && !period)
            load(src2 + g * gin, m2, bank(m2, b) + w2, n * gin);
        vec_cfg(SA_VCFG_LEN, n * SA_D);
        vec_cfg(SA_VCFG_DST, SA_LADDR(md, bank(md, b) + wd));
        vec_run(SA_LADDR(m1, bank(m1, b)), SA_LADDR(m2, bank(m2, b) + w2));
        store(dst + g * gout, md, bank(md, b) + wd, n * gout);
        chunks++;
    }
    uint32_t st = mat_fence(SA_ENG_ALL);
    uint32_t t1 = rdcycle();

    MBOX(MBOX_TOTAL_CYCLES) = t1 - t0;
    MBOX(MBOX_JOBS_DONE)    = chunks;
    MBOX(MBOX_EXT_STATUS)   = st;
    MBOX(MBOX_ERRORS)       = (st & SA_XST_ERROR) ? 1u : 0u;
    MBOX(MBOX_FIRST_ERR)    = st;
    MBOX(MBOX_STATUS)       = STATUS_DONE;
    return 0;
}

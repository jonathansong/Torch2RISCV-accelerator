/*
 * M1-M4 firmware: C = A x B (+ bias) on the double-buffered accelerator.
 *
 *   A: M x K int8, B: K x N int8, C / bias: M x N int32, all row-major in
 *   DDR; M, N, K multiples of D (8 or 16, read from CAPS). Parameters and
 *   results: mailbox.h.
 *
 * Schedule (docs/double_buffer_design.md §7):
 *   - B resident: one INTERLEAVE load of the whole K x N matrix puts
 *     column strip j at SPAD_B word j*K (word k = B[k][Dj .. Dj+D-1]).
 *   - Per D-row strip of A (alternating SPAD_A / ACC banks), M2 commands:
 *     load the A strip, one mat_exec with repeat N/D (B step K) that lays
 *     the strip's C tiles out row-major in ACC (tile j row i at word
 *     j + i*N/D), one mat_store of the whole D x N strip. With a bias, the
 *     bias strip is loaded into ACC in the same layout first and the exec
 *     accumulates onto it.
 *   - Commands are only queued here; the hardware scoreboard overlaps the
 *     next A load / C store with the current exec and orders the rest.
 *
 * M4 schedule tuning (MBOX_GEMM_FLAGS switches each off):
 *   - A prefetch: the scheduler dispatches in order, so the strip loop queues
 *     EX(i), LD A(i+1), ST(i) - the next A strip loads during EX(i) instead
 *     of waiting behind ST(i), which itself waits for EX(i).
 *   - B split: the scoreboard tracks whole banks, so the first exec would
 *     wait for all of B. B's column strips are split between the two SPAD_B
 *     banks (loaded half 0, A(0), half 1) and each strip runs two execs, one
 *     per half, so the first exec starts after half of B has arrived.
 *     Only for B >= 16 KB (board, D = 16: +3 % at 128^3 and 256^3 on top of
 *     the prefetch; -25 % at 64^3, where two short execs per strip cost more
 *     than the half B load saves).
 *
 * M3 int8 epilogue (MBOX_GEMM_Q = 1): C (M x N int8) =
 *   sat8(clamp(requant(relu(A x B + bias)))), bias an int32 vector of N.
 *   After each exec one vec_run turns the strip's int32 tiles into int8
 *   (+ bias with src2 period N/D, RELU, REQUANT) in the upper half of the
 *   strip's SPAD_A bank, and the store moves N bytes per row. A copy of the
 *   bias vector sits at the end of each ACC bank, so a strip's epilogue only
 *   touches its own banks and overlaps the other strip's exec.
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

/* A strip i (rows i..i+D-1) into SPAD_A word abank; with bias_strip also
 * the matching int32 bias rows into ACC word cbank (C layout) */
static void load_strip(uint32_t a, uint32_t bias, uint32_t i, uint32_t K, uint32_t N, uint32_t d,
                       uint32_t abank, uint32_t cbank, uint32_t bias_strip)
{
    mat_cfg_load(d, K, K, SA_LD_INTERLEAVE);
    mat_load(a + i * K, SA_LADDR(SA_MEM_SPAD_A, abank));
    if (bias_strip) {
        mat_cfg_load(d, 4 * N, 4 * N, SA_LD_LINEAR);
        mat_load(bias + i * 4 * N, SA_LADDR(SA_MEM_ACC, cbank));
    }
}

int main(void)
{
    MBOX(MBOX_STATUS) = STATUS_RUNNING;
    sa_init();                                     /* D of the overlay (CAPS) */

    uint32_t M = MBOX(MBOX_GEMM_M), N = MBOX(MBOX_GEMM_N), K = MBOX(MBOX_GEMM_K);
    uint32_t a = MBOX(MBOX_A_BASE), b = MBOX(MBOX_B_BASE), c = MBOX(MBOX_C_BASE);
    uint32_t bias = MBOX(MBOX_BIAS_BASE);
    uint32_t quant = MBOX(MBOX_GEMM_Q) & 1u;
    uint32_t flags = MBOX(MBOX_GEMM_FLAGS);
    /* everything that divides by the run-time D, once, before timing starts */
    uint32_t d = SA_D, nt = N / d, kt = K / d, tiles = 0;
    uint32_t sbank = SA_SPAD_BANK, cbank_w = SA_ACC_BANK;
    uint32_t cbytes = quant ? N : 4 * N;           /* bytes per row of C */
    uint32_t vbias = cbank_w - nt;                 /* bias vector copy, end of each ACC bank */
    uint32_t qout  = sbank / 2;                    /* int8 strip, upper half of the SPAD_A bank */
    uint32_t prefetch = !(flags & GEMM_NO_PREFETCH);
    /* B split: strips 0..nt0-1 at SPAD_B word j*K (bank 0), the rest at
     * sbank + (j-nt0)*K (bank 1), if both halves fit their bank */
    uint32_t nt0 = (nt + 1) / 2, nt1 = nt - nt0;
    uint32_t bsplit = !(flags & GEMM_NO_BSPLIT) && nt1 != 0 && nt0 * K <= sbank &&
                      (K * N >= 16384u || (flags & GEMM_FORCE_BSPLIT));
    uint32_t bias_strip = bias && !quant;          /* int32 output: bias rows loaded per strip */

    mat_reset();                                   /* clear any sticky error */
    uint32_t t0 = rdcycle();

    /* resident B, then the first A strip (split B: half 0, A(0), half 1) */
    if (bsplit) {
        mat_cfg_load(K, nt0 * d, N, SA_LD_INTERLEAVE);
        mat_load(b, SA_LADDR(SA_MEM_SPAD_B, 0));
    } else {
        mat_cfg_load(K, N, N, SA_LD_INTERLEAVE);
        mat_load(b, SA_LADDR(SA_MEM_SPAD_B, 0));
    }
    load_strip(a, bias, 0, K, N, d, 0, 0, bias_strip);
    if (bsplit) {
        mat_cfg_load(K, nt1 * d, N, SA_LD_INTERLEAVE);
        mat_load(b + nt0 * d, SA_LADDR(SA_MEM_SPAD_B, sbank));
    }
    if (quant) {
        uint32_t op = MBOX(MBOX_V_OP) & (SA_V_RELU | SA_V_REQUANT);
        if (bias) {
            mat_cfg_load(1, 4 * N, 4 * N, SA_LD_LINEAR);
            mat_load(bias, SA_LADDR(SA_MEM_ACC, vbias));
            mat_load(bias, SA_LADDR(SA_MEM_ACC, cbank_w + vbias));
        }
        vec_cfg(SA_VCFG_OP, op | (bias ? SA_VOP_ADD : SA_VOP_COPY));
        vec_cfg(SA_VCFG_LEN, d * N);               /* the whole D x N strip */
        vec_cfg(SA_VCFG_TYPES, SA_VTYPES(SA_VT_I32, SA_VT_I8));
        vec_cfg(SA_VCFG_SRC2_MOD, nt);             /* bias repeats every row */
        vec_cfg_requant((int32_t)MBOX(MBOX_V_SCALE), MBOX(MBOX_V_SHIFT), (int32_t)MBOX(MBOX_V_ZP),
                        (int32_t)MBOX(MBOX_V_LO), (int32_t)MBOX(MBOX_V_HI));
    }
    if (!bsplit)
        mat_cfg_exec(nt, K, 1, nt);                /* a row of C tiles per exec, row-major */
    mat_cfg_store(d, cbytes, cbytes);              /* a whole D x N strip of C */

    uint32_t odd = 0;                              /* strips alternate banks */
    for (uint32_t i = 0; i < M; i += d, odd ^= 1u) {
        uint32_t abank = odd ? sbank : 0u;
        uint32_t cbank = odd ? cbank_w : 0u;
        if (i != 0 && !prefetch)
            load_strip(a, bias, i, K, N, d, abank, cbank, bias_strip);
        if (bsplit) {                              /* tiles 0..nt0-1, then nt0..nt-1 */
            mat_cfg_exec(nt0, K, 1, nt);
            mat_exec(abank, 0, cbank, kt, bias_strip);
            mat_cfg_exec(nt1, K, 1, nt);
            mat_exec(abank, sbank, cbank + nt0, kt, bias_strip);
        } else {
            mat_exec(abank, 0, cbank, kt, bias_strip);
        }
        if (quant) {
            vec_cfg(SA_VCFG_DST, SA_LADDR(SA_MEM_SPAD_A, abank + qout));
            vec_run(SA_LADDR(SA_MEM_ACC, cbank), SA_LADDR(SA_MEM_ACC, cbank + vbias));
        }
        if (prefetch && i + d < M)                 /* next strip, before this strip's store */
            load_strip(a, bias, i + d, K, N, d, odd ? 0u : sbank, odd ? 0u : cbank_w, bias_strip);
        if (quant) {
            mat_store(c + i * N, SA_LADDR(SA_MEM_SPAD_A, abank + qout));
        } else {
            mat_store(c + i * 4 * N, SA_LADDR(SA_MEM_ACC, cbank));
        }
        tiles += nt;
    }
    uint32_t st = mat_fence(SA_ENG_ALL);
    uint32_t t1 = rdcycle();

    MBOX(MBOX_TOTAL_CYCLES) = t1 - t0;
    MBOX(MBOX_JOBS_DONE)    = tiles;
    MBOX(MBOX_EXT_STATUS)   = st;
    MBOX(MBOX_ERRORS)       = (st & SA_XST_ERROR) ? 1u : 0u;
    MBOX(MBOX_FIRST_ERR)    = st;
    MBOX(MBOX_STATUS)       = STATUS_DONE;
    return 0;
}

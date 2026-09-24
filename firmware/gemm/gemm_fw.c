/*
 * M1-M3 firmware: C = A x B (+ bias) on the double-buffered accelerator.
 *
 *   A: M x K int8, B: K x N int8, C / bias: M x N int32, all row-major in
 *   DDR; M, N, K multiples of 8 (D). Parameters and results: mailbox.h.
 *
 * Schedule (docs/double_buffer_design.md §7):
 *   - B resident: one INTERLEAVE load of the whole K x N matrix puts
 *     column strip j at SPAD_B word j*K (word k = B[k][8j .. 8j+7]).
 *   - Per 8-row strip of A (alternating SPAD_A / ACC banks), M2 commands:
 *     load the A strip, one mat_exec with repeat N/8 (B step K) that lays
 *     the strip's C tiles out row-major in ACC (tile j row i at word
 *     j + i*N/8), one mat_store of the whole 8 x N strip. With a bias, the
 *     bias strip is loaded into ACC in the same layout first and the exec
 *     accumulates onto it.
 *   - Commands are only queued here; the hardware scoreboard overlaps the
 *     next A load / C store with the current exec and orders the rest.
 *
 * M3 int8 epilogue (MBOX_GEMM_Q = 1): C (M x N int8) =
 *   sat8(clamp(requant(relu(A x B + bias)))), bias an int32 vector of N.
 *   After each exec one vec_run turns the strip's int32 tiles into int8
 *   (+ bias with src2 period N/8, RELU, REQUANT) in the upper half of the
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

int main(void)
{
    MBOX(MBOX_STATUS) = STATUS_RUNNING;

    uint32_t M = MBOX(MBOX_GEMM_M), N = MBOX(MBOX_GEMM_N), K = MBOX(MBOX_GEMM_K);
    uint32_t a = MBOX(MBOX_A_BASE), b = MBOX(MBOX_B_BASE), c = MBOX(MBOX_C_BASE);
    uint32_t bias = MBOX(MBOX_BIAS_BASE);
    uint32_t quant = MBOX(MBOX_GEMM_Q) & 1u;
    uint32_t kt = K / SA_D, tiles = 0;
    uint32_t cbytes = quant ? N : 4 * N;           /* bytes per row of C */
    uint32_t vbias = SA_ACC_BANK - N / SA_D;       /* bias vector copy, end of each ACC bank */
    uint32_t qout  = SA_SPAD_BANK / 2;             /* int8 strip, upper half of the SPAD_A bank */

    mat_reset();                                   /* clear any sticky error */
    uint32_t t0 = rdcycle();

    mat_cfg_load(K, N, N, SA_LD_INTERLEAVE);       /* whole B, resident in SPAD_B */
    mat_load(b, SA_LADDR(SA_MEM_SPAD_B, 0));
    if (quant) {
        uint32_t op = MBOX(MBOX_V_OP) & (SA_V_RELU | SA_V_REQUANT);
        if (bias) {
            mat_cfg_load(1, 4 * N, 4 * N, SA_LD_LINEAR);
            mat_load(bias, SA_LADDR(SA_MEM_ACC, vbias));
            mat_load(bias, SA_LADDR(SA_MEM_ACC, SA_ACC_BANK + vbias));
        }
        vec_cfg(SA_VCFG_OP, op | (bias ? SA_VOP_ADD : SA_VOP_COPY));
        vec_cfg(SA_VCFG_LEN, SA_D * N);            /* the whole 8 x N strip */
        vec_cfg(SA_VCFG_TYPES, SA_VTYPES(SA_VT_I32, SA_VT_I8));
        vec_cfg(SA_VCFG_SRC2_MOD, N / SA_D);       /* bias repeats every row */
        vec_cfg_requant((int32_t)MBOX(MBOX_V_SCALE), MBOX(MBOX_V_SHIFT), (int32_t)MBOX(MBOX_V_ZP),
                        (int32_t)MBOX(MBOX_V_LO), (int32_t)MBOX(MBOX_V_HI));
    }
    mat_cfg_exec(N / SA_D, K, 1, N / SA_D);        /* a row of C tiles per exec, row-major */
    mat_cfg_store(SA_D, cbytes, cbytes);           /* a whole 8 x N strip of C */

    for (uint32_t i = 0; i < M; i += SA_D) {
        uint32_t odd   = (i / SA_D) & 1u;
        uint32_t abank = odd * SA_SPAD_BANK;
        uint32_t cbank = odd * SA_ACC_BANK;
        mat_cfg_load(SA_D, K, K, SA_LD_INTERLEAVE);
        mat_load(a + i * K, SA_LADDR(SA_MEM_SPAD_A, abank));
        if (bias && !quant) {
            mat_cfg_load(SA_D, 4 * N, 4 * N, SA_LD_LINEAR);
            mat_load(bias + i * 4 * N, SA_LADDR(SA_MEM_ACC, cbank));
        }
        mat_exec(abank, 0, cbank, kt, bias != 0 && !quant);
        if (quant) {
            vec_cfg(SA_VCFG_DST, SA_LADDR(SA_MEM_SPAD_A, abank + qout));
            vec_run(SA_LADDR(SA_MEM_ACC, cbank), SA_LADDR(SA_MEM_ACC, cbank + vbias));
            mat_store(c + i * N, SA_LADDR(SA_MEM_SPAD_A, abank + qout));
        } else {
            mat_store(c + i * 4 * N, SA_LADDR(SA_MEM_ACC, cbank));
        }
        tiles += N / SA_D;
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

/*
 * Plan B of the RISC-V backend (docs/riscv_backend_design.md §7.1): the
 * PicoRV32 runs the decode loop. The ARM records each attention-length
 * bucket's descriptor lists once (the sa driver's dry run,
 * compiler/runtime/sa/sa_rvsched.c) and leaves a step table in DDR; the core
 * then, per step, writes the inputs (token, position, the valid mask),
 * patches the lists' words that differ between buckets when the bucket
 * changes, submits the lists (the model's step, then a block-max reduction of
 * the logits), picks the argmax from the block maxima and the winning block,
 * and feeds the token back. The ARM is not involved until the run ends.
 *
 * Shared by rt_fw (firmware/rt/rt_fw.c: ring entry RT_RUN_SCHED) and the sim
 * transport's emulation of it (compiler/runtime/sa/sa_transport_sim.c). The
 * includer defines, before including this header:
 *   RVS_RD(a), RVS_WR(a, v)  32-bit access at device (physical) address a
 *   RVS_SUBMIT(list)         runs the descriptor list at list to its END and
 *                            waits; evaluates to 0 or the unit's error status
 *   RVS_CALL(code, args)     calls generated RISC-V code (plan A, §9) as
 *                            code(args, args + 16 words), waits for the engines;
 *                            0 or the unit's error status
 *   RVS_CYCLES()             a cycle counter (uint32_t)
 *
 * Table (u32 words at a 64-byte aligned address; addresses are physical):
 */
#ifndef RVSCHED_H
#define RVSCHED_H

#include <stdint.h>

#define RVS_MAGIC 0x32535652u            /* "RVS2": items (lists or calls of generated code) */
enum {
  RVS_T_MAGIC = 0,
  RVS_T_NITEMS,      /* items per step, in order (the step's dispatches, then the block maxima's) */
  RVS_T_ITEMS,       /*   two words each: an address and a code address (0: a descriptor list at the
                      *   address, submitted; else generated code, called with the 64-word argument
                      *   block at the address: 16 binding addresses, the push constants) */
  RVS_T_NWORDS,      /* list / argument words that differ between buckets */
  RVS_T_WORDS,       /*   their addresses (nwords words) */
  RVS_T_PAD,         /* positions per bucket: bucket b holds positions [b pad, (b + 1) pad), valid length (b + 1) pad */
  RVS_T_BUCKET0,     /* first bucket in the table */
  RVS_T_NBUCKETS,
  RVS_T_BUCKETS,     /*   RVS_B_* words each */
  RVS_T_VALUES,      /*   nwords values each, the words' values in that bucket */
  RVS_T_NBLOCKS,     /* argmax: block maxima (f32) */
  RVS_T_BLOCK,       /*   positions per block */
  RVS_T_POS,         /* run: first position */
  RVS_T_STEPS,       /*      steps (at most) */
  RVS_T_TOKEN,       /*      first input token */
  RVS_T_STOP,        /*      stop after generating this token (0xFFFFFFFF: none) */
  RVS_T_OUT,         /*      RVS_O_* words per step */
  RVS_T_DONE,        /* out: steps completed */
  RVS_T_STATUS,      /* out: 0, RVS_E_*, or the unit's status of a failing list */
  RVS_T_WORDS_ = 24
};
enum { RVS_B_TOKEN = 0, RVS_B_POS, RVS_B_VALID, RVS_B_LOGITS, RVS_B_BMAX, RVS_B_WORDS_ = 8 };
enum { RVS_O_TOKEN = 0, RVS_O_CYCLES, RVS_O_DEVICE, RVS_O_LIST_ERR, RVS_O_WORDS_ = 4 };
#define RVS_E_MAGIC 0xE0000001u
#define RVS_E_RANGE 0xE0000002u          /* a position outside the table's buckets */
#define RVS_F32_ONE 0x3F800000u

#ifdef RVS_RD
/* f32 bits -> an unsigned key with the floats' order */
static inline uint32_t rvs_key(uint32_t f) { return (f & 0x80000000u) ? ~f : (f | 0x80000000u); }

/* Runs the table's steps; returns RVS_T_STATUS (also written there). */
static uint32_t rvs_run(uint32_t tab)
{
#define RVS_T(i) RVS_RD(tab + 4u * (uint32_t)(i))
    uint32_t status = 0;
    if (RVS_T(RVS_T_MAGIC) != RVS_MAGIC) {
        RVS_WR(tab + 4u * RVS_T_STATUS, RVS_E_MAGIC);
        return RVS_E_MAGIC;
    }
    uint32_t nitems = RVS_T(RVS_T_NITEMS), items = RVS_T(RVS_T_ITEMS);
    uint32_t nwords = RVS_T(RVS_T_NWORDS), words = RVS_T(RVS_T_WORDS);
    uint32_t pad = RVS_T(RVS_T_PAD), b0 = RVS_T(RVS_T_BUCKET0), nb = RVS_T(RVS_T_NBUCKETS);
    uint32_t buckets = RVS_T(RVS_T_BUCKETS), values = RVS_T(RVS_T_VALUES);
    uint32_t nblocks = RVS_T(RVS_T_NBLOCKS), block = RVS_T(RVS_T_BLOCK);
    uint32_t pos = RVS_T(RVS_T_POS), steps = RVS_T(RVS_T_STEPS), token = RVS_T(RVS_T_TOKEN);
    uint32_t stop = RVS_T(RVS_T_STOP), out = RVS_T(RVS_T_OUT);
    uint32_t cur = 0xFFFFFFFFu, bk = 0, s;
    RVS_WR(tab + 4u * RVS_T_DONE, 0u);
    for (s = 0; s < steps; ++s, ++pos) {
        uint32_t t0 = RVS_CYCLES();
        uint32_t b = pos / pad;
        if (b < b0 || b - b0 >= nb) {
            status = RVS_E_RANGE;
            break;
        }
        bk = buckets + 4u * RVS_B_WORDS_ * (b - b0);
        uint32_t valid = RVS_RD(bk + 4u * RVS_B_VALID);
        if (b != cur) {                  /* the bucket's lists and its whole valid mask */
            uint32_t v = values + 4u * nwords * (b - b0);
            for (uint32_t i = 0; i < nwords; ++i)
                RVS_WR(RVS_RD(words + 4u * i), RVS_RD(v + 4u * i));
            uint32_t vlen = (b + 1u) * pad;
            for (uint32_t i = 0; i < vlen; ++i)
                RVS_WR(valid + 4u * i, i <= pos ? RVS_F32_ONE : 0u);
            cur = b;
        } else {
            RVS_WR(valid + 4u * pos, RVS_F32_ONE);
        }
        uint32_t ta = RVS_RD(bk + 4u * RVS_B_TOKEN), pa = RVS_RD(bk + 4u * RVS_B_POS);
        RVS_WR(ta, token);               /* i64 inputs */
        RVS_WR(ta + 4u, 0u);
        RVS_WR(pa, pos);
        RVS_WR(pa + 4u, 0u);
        uint32_t t1 = RVS_CYCLES(), l;
        for (l = 0; l < nitems; ++l) {
            uint32_t a = RVS_RD(items + 8u * l), code = RVS_RD(items + 8u * l + 4u);
            uint32_t st = code ? RVS_CALL(code, a) : RVS_SUBMIT(a);
            if (st) {
                status = st;
                break;
            }
        }
        uint32_t t2 = RVS_CYCLES();
        if (status) {
            RVS_WR(out + 4u * (RVS_O_WORDS_ * s + RVS_O_LIST_ERR), l);
            break;
        }
        /* argmax: the first block with the largest maximum, the first position in it holding that value */
        uint32_t bm = RVS_RD(bk + 4u * RVS_B_BMAX), best = 0, bkey = rvs_key(RVS_RD(bm));
        for (uint32_t j = 1; j < nblocks; ++j) {
            uint32_t k = rvs_key(RVS_RD(bm + 4u * j));
            if (k > bkey) {
                bkey = k;
                best = j;
            }
        }
        uint32_t m = RVS_RD(bm + 4u * best), lg = RVS_RD(bk + 4u * RVS_B_LOGITS) + 4u * block * best, idx = 0;
        for (uint32_t i = 0; i < block; ++i) {
            uint32_t f = RVS_RD(lg + 4u * i);
            if (f == m || ((f | m) << 1) == 0u) {   /* (+0 == -0) */
                idx = i;
                break;
            }
        }
        token = best * block + idx;
        uint32_t o = out + 4u * RVS_O_WORDS_ * s;
        RVS_WR(o + 4u * RVS_O_TOKEN, token);
        RVS_WR(o + 4u * RVS_O_DEVICE, t2 - t1);
        RVS_WR(o + 4u * RVS_O_CYCLES, RVS_CYCLES() - t0);
        RVS_WR(tab + 4u * RVS_T_DONE, s + 1u);
        if (token == stop)
            break;
    }
    RVS_WR(tab + 4u * RVS_T_STATUS, status);
    return status;
#undef RVS_T
}
#endif /* RVS_RD */

#endif /* RVSCHED_H */

/* Runtime of the generated RISC-V code: sa_rv_rt.h. */
#include "sa_rv_rt.h"

#ifdef SA_RV_HOST
#define MAT_CFG(k, v)            sa_rv_host_pcpi("mat_cfg", 2, (k), (v), 0, 0, 0)
#define VEC_CFG(k, v)            sa_rv_host_pcpi("vec_cfg", 2, (k), (v), 0, 0, 0)
#define GEMV_CFG(k, v)           sa_rv_host_pcpi("gemv_cfg", 2, (k), (v), 0, 0, 0)
#define MAT_LOAD(d, l)           sa_rv_host_pcpi("mat_load", 2, (d), (l), 0, 0, 0)
#define MAT_STORE(d, l)          sa_rv_host_pcpi("mat_store", 2, (d), (l), 0, 0, 0)
#define MAT_EXEC(a, b, c, k, x)  sa_rv_host_pcpi("mat_exec", 5, (a), (b), (c), (k), (x))
#define VEC_RUN(s1, s2)          sa_rv_host_pcpi("vec_run", 2, (s1), (s2), 0, 0, 0)
#define GEMV_RUN(w, c, s, x)     sa_rv_host_pcpi("gemv_run", 4, (w), (c), (s), (x), 0)
#define MAT_FENCE(m)             sa_rv_host_pcpi("mat_fence", 1, (m), 0, 0, 0, 0)
#define LOAD32(a)                sa_rv_host_load32(a)
#define NOTE(w, a, b)            sa_rv_host_note((w), (a), (b))
#else
#include "sysarray_intrinsics.h"
#define MAT_CFG(k, v)            mat_cfg((k), (v))
#define VEC_CFG(k, v)            vec_cfg((k), (v))
#define GEMV_CFG(k, v)           gemv_cfg((k), (v))
#define MAT_LOAD(d, l)           mat_load((d), (l))
#define MAT_STORE(d, l)          mat_store((d), (l))
#define MAT_EXEC(a, b, c, k, x)  mat_exec((a), (b), (c), (k), (x))
#define VEC_RUN(s1, s2)          vec_run((s1), (s2))
#define GEMV_RUN(w, c, s, x)     gemv_run((w), (c), (s), (x))
#define MAT_FENCE(m)             ((void)mat_fence(m))
#define LOAD32(a)                (*(volatile uint32_t *)(uintptr_t)(a))
#define NOTE(w, a, b)            ((void)0)
#endif

enum { OP_LD = 0x01, OP_ST = 0x02, OP_EX = 0x03, OP_VE = 0x04, OP_FENCE = 0x10 };
#define MODE_GEMV 2u

/* dynamic fields: field -> (word, lsb, width) per opcode (llm/sa_funcsim.py DYN_FIELDS) */
typedef struct { uint8_t word, lsb, width; } field_t;
static const field_t F_DMA[6] = {{0, 0, 0}, {1, 0, 32}, {2, 0, 16}, {2, 32, 16}, {2, 48, 16}, {3, 0, 32}};
static const field_t F_EX[8] = {{0, 0, 0}, {1, 0, 16}, {1, 16, 16}, {1, 32, 16}, {1, 48, 12},
                                {2, 0, 12}, {2, 16, 16}, {2, 32, 16}};
static const field_t F_VE[12] = {{0, 0, 0}, {1, 0, 16}, {1, 32, 16}, {2, 0, 16}, {2, 32, 32}, {7, 16, 16},
                                 {7, 0, 16}, {7, 32, 16}, {3, 16, 16}, {6, 0, 32}, {6, 32, 32}, {5, 32, 32}};

static inline uint32_t fld(uint64_t x, unsigned lo, unsigned n)
{
    return (uint32_t)((x >> lo) & (n >= 64 ? ~0ull : ((1ull << n) - 1)));
}

void sa_rv_cmd(const uint32_t *desc, uint32_t base, uint32_t p0, uint32_t p1)
{
    uint64_t w[8];
    for (int i = 0; i < 8; ++i)
        w[i] = (uint64_t)desc[2 * i] | (uint64_t)desc[2 * i + 1] << 32;
    uint32_t hdr = (uint32_t)w[0], op = hdr & 0xFF;
    const field_t *tab = op == OP_LD || op == OP_ST ? F_DMA : op == OP_EX ? F_EX : op == OP_VE ? F_VE : 0;
    unsigned ntab = op == OP_LD || op == OP_ST ? 6 : op == OP_EX ? 8 : op == OP_VE ? 12 : 0;
    for (unsigned s = 0; s < 2; ++s) {
        uint32_t slot = (hdr >> (16 + 8 * s)) & 0xFF, f = slot & 0xF;
        if (!f || f >= ntab)
            continue;
        field_t t = tab[f];
        uint64_t mask = t.width == 32 ? 0xFFFFFFFFull : ((1ull << t.width) - 1);
        uint64_t p = s ? p1 : p0, old = (w[t.word] >> t.lsb) & mask;
        uint64_t nv = ((slot & 0x80) ? old + p : p) & mask;
        w[t.word] = (w[t.word] & ~(mask << t.lsb)) | (nv << t.lsb);
    }
    if (op != OP_FENCE && ((hdr >> 11) & 1))           /* FENCE_BEFORE */
        MAT_FENCE(0);
    uint32_t ddr = fld(w[1], 0, 32) + (((hdr >> 8) & 1) ? base : 0);
    if (op == OP_LD && fld(w[3], 32, 2) == MODE_GEMV) {
        GEMV_CFG(0, fld(w[2], 48, 16));                    /* SA_GCFG_KBYTES */
        GEMV_CFG(1, fld(w[3], 0, 32));                     /* SA_GCFG_PITCH */
        GEMV_CFG(2, fld(w[3], 34, 16));                    /* SA_GCFG_XWORD */
        GEMV_CFG(3, fld(w[3], 50, 8));                     /* SA_GCFG_CSTEP */
        GEMV_RUN(ddr, fld(w[2], 0, 32) & 0xFFFF, fld(w[2], 32, 16), fld(w[3], 58, 1));
    } else if (op == OP_LD) {
        MAT_CFG(0, fld(w[2], 32, 16));                     /* SA_CFG_LD_ROWS */
        MAT_CFG(1, fld(w[2], 48, 16));                     /* SA_CFG_LD_ROW_BYTES */
        MAT_CFG(2, fld(w[3], 0, 32));                      /* SA_CFG_LD_PITCH */
        MAT_CFG(3, fld(w[3], 32, 2));                      /* SA_CFG_LD_MODE */
        MAT_LOAD(ddr, fld(w[2], 0, 32));
    } else if (op == OP_ST) {
        MAT_CFG(4, fld(w[2], 32, 16));                     /* SA_CFG_ST_ROWS */
        MAT_CFG(5, fld(w[2], 48, 16));                     /* SA_CFG_ST_ROW_BYTES */
        MAT_CFG(6, fld(w[3], 0, 32));                      /* SA_CFG_ST_PITCH */
        MAT_STORE(ddr, fld(w[2], 0, 32));
    } else if (op == OP_EX) {
        MAT_CFG(7, fld(w[2], 0, 12));                      /* SA_CFG_EX_REPEAT */
        MAT_CFG(8, fld(w[2], 16, 16));                     /* SA_CFG_EX_B_STEP */
        MAT_CFG(9, fld(w[2], 32, 16));                     /* SA_CFG_EX_C_STEP */
        MAT_CFG(10, fld(w[2], 48, 16));                    /* SA_CFG_EX_C_ROW */
        MAT_EXEC(fld(w[1], 0, 16), fld(w[1], 16, 16), fld(w[1], 32, 16), fld(w[1], 48, 12), fld(w[1], 60, 1));
    } else if (op == OP_VE) {
        VEC_CFG(0, fld(w[3], 0, 8));                       /* SA_VCFG_OP */
        VEC_CFG(1, fld(w[2], 32, 32));                     /* SA_VCFG_LEN */
        VEC_CFG(2, fld(w[2], 0, 32));                      /* SA_VCFG_DST */
        VEC_CFG(3, fld(w[3], 8, 6));                       /* SA_VCFG_TYPES */
        VEC_CFG(4, fld(w[3], 16, 16));                     /* SA_VCFG_SRC2_MOD */
        VEC_CFG(5, fld(w[3], 32, 16));                     /* SA_VCFG_SCALE */
        VEC_CFG(6, fld(w[3], 48, 5));                      /* SA_VCFG_SHIFT */
        VEC_CFG(7, fld(w[4], 0, 32));                      /* SA_VCFG_ZP */
        VEC_CFG(8, fld(w[4], 32, 32));                     /* SA_VCFG_CLAMP_LO */
        VEC_CFG(9, fld(w[5], 0, 32));                      /* SA_VCFG_CLAMP_HI */
        VEC_CFG(10, fld(w[3], 53, 11));                    /* SA_VCFG_FLAGS */
        VEC_CFG(11, fld(w[5], 32, 32));                    /* SA_VCFG_IMM */
        VEC_CFG(12, fld(w[6], 0, 32));                     /* SA_VCFG_A */
        VEC_CFG(13, fld(w[6], 32, 32));                    /* SA_VCFG_B */
        VEC_CFG(14, fld(w[7], 0, 32));                     /* SA_VCFG_ROW */
        VEC_CFG(15, fld(w[7], 32, 32));                    /* SA_VCFG_P1S */
        VEC_RUN(fld(w[1], 0, 32), fld(w[1], 32, 32));
    } else if (op == OP_FENCE) {
        MAT_FENCE(fld(w[1], 0, 4));
    }
}

uint32_t sa_rv_ldparam(uint32_t addr)
{
#ifdef SA_RV_HOST
    NOTE("fence_for_ldparam", 0, 0);           /* (not a command of the descriptor path) */
#else
    MAT_FENCE(0);
#endif
    uint32_t v = LOAD32(addr);
    NOTE("ldparam", addr, v);
    return v;
}

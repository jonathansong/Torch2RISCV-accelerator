/* Runtime of the generated RISC-V code: sa_rv_rt.h. */
#include "sa_rv_rt.h"

#if defined(SA_RV_SIM)
/* ---------------------------------------------------------------- the sim transport
 * (compiler/runtime/sa/sa_transport_sim.c): the PCPI configuration kept here,
 * each command rebuilt as the descriptor it stands for (the inverse of the
 * mapping in SahwToRiscv.cpp) and handed over to run on the functional
 * simulator; loads from the shared window */
static void (*sim_cmd)(const uint64_t *desc);
static uint32_t (*sim_load32)(uint32_t addr);
static void (*sim_store32)(uint32_t addr, uint32_t value);
static uint32_t mat[11], vec[16], gemv[4];
void sa_rv_sim_init(void (*cmd)(const uint64_t *desc), uint32_t (*load32)(uint32_t addr))
{
    sim_cmd = cmd;
    sim_load32 = load32;
}
/* (separate: a transport without it still runs code that stores nothing) */
void sa_rv_sim_init_store(void (*store32)(uint32_t addr, uint32_t value)) { sim_store32 = store32; }
static void run(uint64_t w0, uint64_t w1, uint64_t w2, uint64_t w3, uint64_t w4, uint64_t w5, uint64_t w6,
                uint64_t w7)
{
    uint64_t w[8] = {w0, w1, w2, w3, w4, w5, w6, w7};
    sim_cmd(w);
}
void sa_rv_mat_cfg(uint32_t key, uint32_t value) { if (key < 11) mat[key] = value; }
void sa_rv_vec_cfg(uint32_t key, uint32_t value) { if (key < 16) vec[key] = value; }
void sa_rv_gemv_cfg(uint32_t key, uint32_t value) { if (key < 4) gemv[key] = value; }
void sa_rv_mat_load(uint32_t ddr, uint32_t laddr)
{
    run(0x01, ddr, laddr | (uint64_t)mat[0] << 32 | (uint64_t)mat[1] << 48, mat[2] | (uint64_t)(mat[3] & 3) << 32,
        0, 0, 0, 0);
}
void sa_rv_mat_store(uint32_t ddr, uint32_t laddr)
{
    run(0x02, ddr, laddr | (uint64_t)mat[4] << 32 | (uint64_t)mat[5] << 48, mat[6], 0, 0, 0, 0);
}
void sa_rv_mat_exec(uint32_t a, uint32_t b, uint32_t c, uint32_t kt, uint32_t accumulate)
{
    run(0x03, a | (uint64_t)b << 16 | (uint64_t)c << 32 | (uint64_t)kt << 48 | (uint64_t)accumulate << 60,
        mat[7] | (uint64_t)mat[8] << 16 | (uint64_t)mat[9] << 32 | (uint64_t)mat[10] << 48, 0, 0, 0, 0, 0);
}
void sa_rv_vec_run(uint32_t src1, uint32_t src2)
{
    run(0x04, src1 | (uint64_t)src2 << 32, vec[2] | (uint64_t)vec[1] << 32,
        vec[0] | (uint64_t)vec[3] << 8 | (uint64_t)vec[4] << 16 | (uint64_t)vec[5] << 32 | (uint64_t)vec[6] << 48 |
            (uint64_t)vec[10] << 53,
        vec[7] | (uint64_t)vec[8] << 32, vec[9] | (uint64_t)vec[11] << 32, vec[12] | (uint64_t)vec[13] << 32,
        vec[14] | (uint64_t)vec[15] << 32);
}
void sa_rv_gemv_run(uint32_t w, uint32_t c, uint32_t strips, uint32_t accumulate)
{
    run(0x01, w, (0x30000000u | c) | (uint64_t)strips << 32 | (uint64_t)gemv[0] << 48,
        gemv[1] | 2ull << 32 | (uint64_t)gemv[2] << 34 | (uint64_t)gemv[3] << 50 | (uint64_t)accumulate << 58,
        0, 0, 0, 0);
}
void sa_rv_mat_fence(uint32_t mask) { run(0x10, mask, 0, 0, 0, 0, 0, 0); }
uint32_t sa_rv_ldparam(uint32_t addr) { return sim_load32(addr); }   /* (each command ran to its end) */
void sa_rv_store32(uint32_t addr, uint32_t value) { sim_store32(addr, value); }

#elif defined(SA_RV_HOST)
/* ---------------------------------------------------------------- the host's recorder */
void sa_rv_mat_cfg(uint32_t k, uint32_t v) { sa_rv_host_pcpi("mat_cfg", 2, k, v, 0, 0, 0); }
void sa_rv_vec_cfg(uint32_t k, uint32_t v) { sa_rv_host_pcpi("vec_cfg", 2, k, v, 0, 0, 0); }
void sa_rv_gemv_cfg(uint32_t k, uint32_t v) { sa_rv_host_pcpi("gemv_cfg", 2, k, v, 0, 0, 0); }
void sa_rv_mat_load(uint32_t d, uint32_t l) { sa_rv_host_pcpi("mat_load", 2, d, l, 0, 0, 0); }
void sa_rv_mat_store(uint32_t d, uint32_t l) { sa_rv_host_pcpi("mat_store", 2, d, l, 0, 0, 0); }
void sa_rv_mat_exec(uint32_t a, uint32_t b, uint32_t c, uint32_t kt, uint32_t x)
{
    sa_rv_host_pcpi("mat_exec", 5, a, b, c, kt, x);
}
void sa_rv_vec_run(uint32_t s1, uint32_t s2) { sa_rv_host_pcpi("vec_run", 2, s1, s2, 0, 0, 0); }
void sa_rv_gemv_run(uint32_t w, uint32_t c, uint32_t s, uint32_t x) { sa_rv_host_pcpi("gemv_run", 4, w, c, s, x, 0); }
void sa_rv_mat_fence(uint32_t m) { sa_rv_host_pcpi("mat_fence", 1, m, 0, 0, 0, 0); }
uint32_t sa_rv_ldparam(uint32_t addr)
{
    sa_rv_host_note("fence_for_ldparam", 0, 0);       /* (not a command of the descriptor path) */
    uint32_t v = sa_rv_host_load32(addr);
    sa_rv_host_note("ldparam", addr, v);
    return v;
}
void sa_rv_store32(uint32_t addr, uint32_t value) { sa_rv_host_note("store32", addr, value); }

#else
/* ---------------------------------------------------------------- the PicoRV32 */
#include "sysarray_intrinsics.h"
void sa_rv_mat_cfg(uint32_t k, uint32_t v) { mat_cfg(k, v); }
void sa_rv_vec_cfg(uint32_t k, uint32_t v) { vec_cfg(k, v); }
void sa_rv_gemv_cfg(uint32_t k, uint32_t v) { gemv_cfg(k, v); }
void sa_rv_mat_load(uint32_t d, uint32_t l) { mat_load(d, l); }
void sa_rv_mat_store(uint32_t d, uint32_t l) { mat_store(d, l); }
void sa_rv_mat_exec(uint32_t a, uint32_t b, uint32_t c, uint32_t kt, uint32_t x) { mat_exec(a, b, c, kt, x); }
void sa_rv_vec_run(uint32_t s1, uint32_t s2) { vec_run(s1, s2); }
void sa_rv_gemv_run(uint32_t w, uint32_t c, uint32_t s, uint32_t x) { gemv_run(w, c, s, x); }
void sa_rv_mat_fence(uint32_t m) { (void)mat_fence(m); }
uint32_t sa_rv_ldparam(uint32_t addr)
{
    (void)mat_fence(0);
    return *(volatile uint32_t *)(uintptr_t)addr;
}
void sa_rv_store32(uint32_t addr, uint32_t value) { *(volatile uint32_t *)(uintptr_t)addr = value; }
#endif

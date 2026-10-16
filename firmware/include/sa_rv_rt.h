/*
 * Runtime of the generated RISC-V code (docs/riscv_backend_design.md §9, the
 * sahw -> LLVM -> riscv32 backend). The compiler (SahwToRiscv.cpp) emits per
 * export a function
 *     void <export>(const uint32_t *bindings, const uint32_t *push_constants)
 * whose commands are specialized at compile time: each is a few calls of the
 * primitives below, one PCPI instruction each (sysarray_intrinsics.h), the
 * configuration keys the command uses (only those whose value changed, when
 * the compiler can tell), then the command.
 *
 * Implementation: firmware/rv/sa_rv_rt.c. With SA_RV_HOST it is compiled for
 * the host, the PCPI operations recorded (compiler/tests/rv_equiv.py); with
 * SA_RV_SIM for the sa driver's sim transport (each command rebuilt as a
 * descriptor from the configuration and run on the functional simulator).
 */
#ifndef SA_RV_RT_H
#define SA_RV_RT_H

#include <stdint.h>

void sa_rv_mat_cfg(uint32_t key, uint32_t value);
void sa_rv_vec_cfg(uint32_t key, uint32_t value);
void sa_rv_gemv_cfg(uint32_t key, uint32_t value);
void sa_rv_mat_load(uint32_t ddr, uint32_t laddr);
void sa_rv_mat_store(uint32_t ddr, uint32_t laddr);
void sa_rv_mat_exec(uint32_t a, uint32_t b, uint32_t c, uint32_t kt, uint32_t accumulate);
void sa_rv_vec_run(uint32_t src1, uint32_t src2);
void sa_rv_gemv_run(uint32_t w, uint32_t c, uint32_t strips, uint32_t accumulate);
void sa_rv_mat_fence(uint32_t mask);
/* LDPARAM: the 32-bit word at a DDR address, after the engines are idle (it may
 * have just been stored) */
uint32_t sa_rv_ldparam(uint32_t addr);

#ifdef SA_RV_HOST
/* the host's recorder (rv_equiv.py's harness): one line per PCPI operation */
void sa_rv_host_pcpi(const char *op, int n, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e);
void sa_rv_host_note(const char *what, uint32_t a, uint32_t b);
uint32_t sa_rv_host_load32(uint32_t addr);
#endif

#endif /* SA_RV_RT_H */

/*
 * Runtime of the generated RISC-V code (docs/riscv_backend_design.md §9, the
 * sahw -> LLVM -> riscv32 backend). The compiler (SahwToRiscv.cpp) emits per
 * export a function
 *     void <export>(const uint32_t *bindings, const uint32_t *push_constants)
 * whose commands are calls to sa_rv_cmd: the command's 64-byte descriptor as
 * the descriptor backend encodes it, the value of its BASE register and of its
 * dynamic-field PARAMs (slot 0, slot 1). sa_rv_cmd applies them as the fetch
 * unit does (fields replaced or added, the DDR address relocated,
 * FENCE_BEFORE) and issues the command over PCPI (sysarray_intrinsics.h; the
 * mapping of compiler/tests/rv_dispatch.py, checked on the board), every
 * configuration key written for every command (speed is not the point).
 *
 * Implementation: firmware/rv/sa_rv_rt.c. With SA_RV_HOST it is compiled for
 * the host, the PCPI operations recorded instead (compiler/tests/rv_equiv.py).
 */
#ifndef SA_RV_RT_H
#define SA_RV_RT_H

#include <stdint.h>

void sa_rv_cmd(const uint32_t *desc, uint32_t base, uint32_t p0, uint32_t p1);
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

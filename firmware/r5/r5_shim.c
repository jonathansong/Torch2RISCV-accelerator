/*
 * R5 of the RISC-V backend (docs/riscv_backend_design.md §5, model C): a
 * kernel compiled by IREE's llvm-cpu backend for the PicoRV32 (riscv32,
 * --iree-llvmcpu-link-static), called by the core like plan A's generated code:
 * r5_entry(args, constants), args the 16 binding addresses. This shim is the
 * part of IREE's local executable ABI the kernel needs
 * (iree/hal/local/executable_library.h, v0): the environment, the dispatch
 * state (bindings, push constants, workgroup count and size) and one workgroup
 * state per workgroup, run one after another. No imports, no library query
 * (the linker drops it).
 *
 * compiler/tests/r5_kernel.py compiles it with
 *   R5_KERNEL       the export's function
 *   R5_BINDINGS     its binding count
 *   R5_CONSTANTS    its push-constant count
 *   R5_COUNT_X/Y/Z  its workgroup count (static), R5_SIZE_X/Y/Z its workgroup size
 * and, for the sim's host build (R5_HOST), the binding addresses translated by
 * the callback r5_host_init sets.
 */
#include <stddef.h>
#include <stdint.h>

#include "iree/hal/local/executable_library.h"

int R5_KERNEL(const iree_hal_executable_environment_v0_t* environment,
              const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
              const iree_hal_executable_workgroup_state_v0_t* workgroup_state);

#ifdef R5_HOST
static void* (*r5_ptr)(uint32_t phys);
void r5_host_init(void* (*ptr)(uint32_t phys)) { r5_ptr = ptr; }
#define R5_PTR(a) r5_ptr(a)
#else
#define R5_PTR(a) ((void*)(uintptr_t)(a))
/* (the compiler's zeroing of the states) */
void* memset(void* d, int c, size_t n)
{
    unsigned char* p = (unsigned char*)d;
    while (n--) *p++ = (unsigned char)c;
    return d;
}
#endif

void r5_entry(const uint32_t* args, const uint32_t* constants)
{
    void* ptrs[R5_BINDINGS];
    size_t lengths[R5_BINDINGS];
    for (int i = 0; i < R5_BINDINGS; ++i) {
        ptrs[i] = R5_PTR(args[i]);
        lengths[i] = 0x7FFFFFFF;         /* (unknown; static shapes do not read them) */
    }
    iree_hal_executable_environment_v0_t env = {0};
    iree_hal_executable_dispatch_state_v0_t ds = {0};
    ds.workgroup_size_x = R5_SIZE_X;
    ds.workgroup_size_y = R5_SIZE_Y;
    ds.workgroup_size_z = R5_SIZE_Z;
    ds.constant_count = R5_CONSTANTS;
    ds.workgroup_count_x = R5_COUNT_X;
    ds.workgroup_count_y = R5_COUNT_Y;
    ds.workgroup_count_z = R5_COUNT_Z;
    ds.max_concurrency = 1;
    ds.binding_count = R5_BINDINGS;
    ds.constants = constants;
    ds.binding_ptrs = ptrs;
    ds.binding_lengths = lengths;
    iree_hal_executable_workgroup_state_v0_t wg = {0};
    for (uint32_t z = 0; z < R5_COUNT_Z; ++z)
        for (uint32_t y = 0; y < R5_COUNT_Y; ++y)
            for (uint32_t x = 0; x < R5_COUNT_X; ++x) {
                wg.workgroup_id_x = x;
                wg.workgroup_id_y = y;
                wg.workgroup_id_z = (uint16_t)z;
                R5_KERNEL(&env, &ds, &wg);
            }
}

// sa HAL driver: the sa-desc-v1 executable loader (sa_loader.c).
#ifndef SA_LOADER_H_
#define SA_LOADER_H_

#include "iree/hal/local/executable_loader.h"
#include "sa_context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Appends one dispatch of an sa-desc executable to the context's batch list
// (sa_command_buffer.c); with SA_PROFILE or SA_NO_BATCH it runs at once.
iree_status_t sa_executable_append(iree_hal_executable_t* executable, uint32_t ordinal, uint32_t binding_count,
                                   void* const* binding_ptrs, uint32_t constant_count, const uint32_t* constants);

iree_status_t sa_loader_create(sa_context_t* context, iree_allocator_t host_allocator,
                               iree_hal_executable_loader_t** out_loader);

#ifdef __cplusplus
}
#endif

#endif  // SA_LOADER_H_

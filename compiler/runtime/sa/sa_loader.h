// sa HAL driver: the sa-desc-v1 executable loader (sa_loader.c).
#ifndef SA_LOADER_H_
#define SA_LOADER_H_

#include "iree/hal/local/executable_loader.h"
#include "sa_context.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t sa_loader_create(sa_context_t* context, iree_allocator_t host_allocator,
                               iree_hal_executable_loader_t** out_loader);

#ifdef __cplusplus
}
#endif

#endif  // SA_LOADER_H_

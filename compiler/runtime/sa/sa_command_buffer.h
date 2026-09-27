// sa HAL driver: the batching command buffer (sa_command_buffer.c).
#ifndef SA_COMMAND_BUFFER_H_
#define SA_COMMAND_BUFFER_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_host_size_t sa_command_buffer_size(iree_hal_command_buffer_mode_t mode, iree_host_size_t binding_capacity);
iree_status_t sa_command_buffer_initialize(iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
                                           iree_hal_command_category_t categories,
                                           iree_hal_queue_affinity_t queue_affinity, iree_allocator_t host_allocator,
                                           iree_byte_span_t storage, iree_hal_command_buffer_t** out);
void sa_command_buffer_deinitialize(iree_hal_command_buffer_t* command_buffer);
iree_status_t sa_command_buffer_create(iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
                                       iree_hal_command_category_t categories,
                                       iree_hal_queue_affinity_t queue_affinity, iree_allocator_t host_allocator,
                                       iree_hal_command_buffer_t** out);
bool sa_command_buffer_isa(iree_hal_command_buffer_t* command_buffer);

#ifdef __cplusplus
}
#endif

#endif  // SA_COMMAND_BUFFER_H_

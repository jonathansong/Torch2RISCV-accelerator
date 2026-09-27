// sa HAL driver: the command buffer (docs/iree_compiler_plan.md §5.2, C4).
//
// Structured as IREE's inline command buffer (hal/local/inline_command_buffer.c),
// but a dispatch is not run when it is recorded: it is appended to the
// context's batch list (SETREG + CALL of its template), an execution barrier
// becomes a FENCE before the next dispatch, and the list is submitted once,
// at the end of the command buffer. Host-side commands (fill / update / copy)
// first submit what is queued, so they see the results of earlier dispatches.
// With SA_PROFILE or SA_NO_BATCH every dispatch runs at once (sa_loader.c).
#include "sa_command_buffer.h"

#include <string.h>

#include "iree/hal/local/local_executable.h"
#include "sa_context.h"
#include "sa_loader.h"

typedef struct sa_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  sa_context_t* context;
} sa_command_buffer_t;

static const iree_hal_command_buffer_vtable_t sa_command_buffer_vtable;

static sa_command_buffer_t* sa_command_buffer_cast(iree_hal_command_buffer_t* base) {
  IREE_HAL_ASSERT_TYPE(base, &sa_command_buffer_vtable);
  return (sa_command_buffer_t*)base;
}

iree_host_size_t sa_command_buffer_size(iree_hal_command_buffer_mode_t mode, iree_host_size_t binding_capacity) {
  return sizeof(sa_command_buffer_t) + iree_hal_command_buffer_validation_state_size(mode, binding_capacity);
}

iree_status_t sa_command_buffer_initialize(iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
                                           iree_hal_command_category_t categories,
                                           iree_hal_queue_affinity_t queue_affinity, iree_allocator_t host_allocator,
                                           iree_byte_span_t storage, iree_hal_command_buffer_t** out) {
  *out = NULL;
  if (storage.data_length < sa_command_buffer_size(mode, 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "sa command buffer: storage too small");
  }
  sa_command_buffer_t* cb = (sa_command_buffer_t*)storage.data;
  memset(cb, 0, sizeof(*cb));
  iree_hal_command_buffer_initialize(device_allocator, mode, categories, queue_affinity, 0,
                                     (uint8_t*)cb + sizeof(*cb), &sa_command_buffer_vtable, &cb->base);
  cb->host_allocator = host_allocator;
  IREE_RETURN_IF_ERROR(sa_context_get(&cb->context));
  *out = &cb->base;
  return iree_ok_status();
}

void sa_command_buffer_deinitialize(iree_hal_command_buffer_t* base) { (void)base; }

iree_status_t sa_command_buffer_create(iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
                                       iree_hal_command_category_t categories,
                                       iree_hal_queue_affinity_t queue_affinity, iree_allocator_t host_allocator,
                                       iree_hal_command_buffer_t** out) {
  *out = NULL;
  iree_host_size_t size = sa_command_buffer_size(mode, 0);
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, size, (void**)&storage));
  iree_status_t status = sa_command_buffer_initialize(device_allocator, mode, categories, queue_affinity,
                                                      host_allocator, iree_make_byte_span(storage, size), out);
  if (!iree_status_is_ok(status)) iree_allocator_free(host_allocator, storage);
  return status;
}

static void sa_command_buffer_destroy(iree_hal_command_buffer_t* base) {
  sa_command_buffer_t* cb = sa_command_buffer_cast(base);
  iree_allocator_free(cb->host_allocator, cb);
}

bool sa_command_buffer_isa(iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource, &sa_command_buffer_vtable);
}

static iree_status_t sa_command_buffer_begin(iree_hal_command_buffer_t* base) {
  return sa_context_batch_flush(sa_command_buffer_cast(base)->context);
}

static iree_status_t sa_command_buffer_end(iree_hal_command_buffer_t* base) {
  return sa_context_batch_flush(sa_command_buffer_cast(base)->context);
}

static iree_status_t sa_command_buffer_begin_debug_group(iree_hal_command_buffer_t* base, iree_string_view_t label,
                                                         iree_hal_label_color_t label_color,
                                                         const iree_hal_label_location_t* location) {
  return iree_ok_status();
}

static iree_status_t sa_command_buffer_end_debug_group(iree_hal_command_buffer_t* base) { return iree_ok_status(); }

static iree_status_t sa_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask, iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count, const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count, const iree_hal_buffer_barrier_t* buffer_barriers) {
  // later dispatches may read what earlier ones wrote to DDR: the device does
  // not track DDR, so a FENCE (all engines idle) goes before the next dispatch
  sa_context_batch_barrier(sa_command_buffer_cast(base)->context);
  return iree_ok_status();
}

static iree_status_t sa_command_buffer_signal_event(iree_hal_command_buffer_t* base, iree_hal_event_t* event,
                                                    iree_hal_execution_stage_t source_stage_mask) {
  return iree_ok_status();
}

static iree_status_t sa_command_buffer_reset_event(iree_hal_command_buffer_t* base, iree_hal_event_t* event,
                                                   iree_hal_execution_stage_t source_stage_mask) {
  return iree_ok_status();
}

static iree_status_t sa_command_buffer_wait_events(
    iree_hal_command_buffer_t* base, iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask, iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count, const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count, const iree_hal_buffer_barrier_t* buffer_barriers) {
  sa_context_batch_barrier(sa_command_buffer_cast(base)->context);
  return iree_ok_status();
}

static iree_status_t sa_command_buffer_advise_buffer(iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t buffer_ref,
                                                     iree_hal_memory_advise_flags_t flags, uint64_t arg0,
                                                     uint64_t arg1) {
  return iree_ok_status();
}

// host-side commands: the queued dispatches run first
static iree_status_t sa_command_buffer_fill_buffer(iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t target_ref,
                                                   const void* pattern, iree_host_size_t pattern_length,
                                                   iree_hal_fill_flags_t flags) {
  IREE_RETURN_IF_ERROR(sa_context_batch_flush(sa_command_buffer_cast(base)->context));
  return iree_hal_buffer_map_fill(target_ref.buffer, target_ref.offset, target_ref.length, pattern, pattern_length);
}

static iree_status_t sa_command_buffer_update_buffer(iree_hal_command_buffer_t* base, const void* source_buffer,
                                                     iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
                                                     iree_hal_update_flags_t flags) {
  IREE_RETURN_IF_ERROR(sa_context_batch_flush(sa_command_buffer_cast(base)->context));
  return iree_hal_buffer_map_write(target_ref.buffer, target_ref.offset,
                                   (const uint8_t*)source_buffer + source_offset, target_ref.length);
}

static iree_status_t sa_command_buffer_copy_buffer(iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t source_ref,
                                                   iree_hal_buffer_ref_t target_ref, iree_hal_copy_flags_t flags) {
  IREE_RETURN_IF_ERROR(sa_context_batch_flush(sa_command_buffer_cast(base)->context));
  return iree_hal_buffer_map_copy(source_ref.buffer, source_ref.offset, target_ref.buffer, target_ref.offset,
                                  target_ref.length);
}

static iree_status_t sa_command_buffer_collective(iree_hal_command_buffer_t* base, iree_hal_channel_t* channel,
                                                  iree_hal_collective_op_t op, uint32_t param,
                                                  iree_hal_buffer_ref_t send_ref, iree_hal_buffer_ref_t recv_ref,
                                                  iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "sa: collectives are not supported");
}

static iree_status_t sa_command_buffer_dispatch(iree_hal_command_buffer_t* base, iree_hal_executable_t* executable,
                                                iree_hal_executable_export_ordinal_t export_ordinal,
                                                const iree_hal_dispatch_config_t config,
                                                iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
                                                iree_hal_dispatch_flags_t flags) {
  if (iree_hal_dispatch_uses_custom_arguments(flags)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "sa: direct / indirect dispatch arguments");
  }
  if (constants.data_length % 4) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "constants not 4-byte");
  if (bindings.count > 16) return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "sa: more than 16 bindings");
  void* ptrs[16];
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    if (!bindings.values[i].buffer) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "binding %u is NULL", (unsigned)i);
    }
    iree_hal_buffer_mapping_t mapping = {{0}};
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(bindings.values[i].buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
                                                   IREE_HAL_MEMORY_ACCESS_ANY, bindings.values[i].offset,
                                                   bindings.values[i].length, &mapping));
    ptrs[i] = mapping.contents.data;
  }
  return sa_executable_append(executable, export_ordinal, (uint32_t)bindings.count, ptrs,
                              (uint32_t)(constants.data_length / 4), (const uint32_t*)constants.data);
}

static const iree_hal_command_buffer_vtable_t sa_command_buffer_vtable = {
    .destroy = sa_command_buffer_destroy,
    .begin = sa_command_buffer_begin,
    .end = sa_command_buffer_end,
    .begin_debug_group = sa_command_buffer_begin_debug_group,
    .end_debug_group = sa_command_buffer_end_debug_group,
    .execution_barrier = sa_command_buffer_execution_barrier,
    .signal_event = sa_command_buffer_signal_event,
    .reset_event = sa_command_buffer_reset_event,
    .wait_events = sa_command_buffer_wait_events,
    .advise_buffer = sa_command_buffer_advise_buffer,
    .fill_buffer = sa_command_buffer_fill_buffer,
    .update_buffer = sa_command_buffer_update_buffer,
    .copy_buffer = sa_command_buffer_copy_buffer,
    .collective = sa_command_buffer_collective,
    .dispatch = sa_command_buffer_dispatch,
};

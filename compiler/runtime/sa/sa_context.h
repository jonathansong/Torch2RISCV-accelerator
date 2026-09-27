// sa HAL driver: device context (docs/iree_compiler_plan.md §5).
//
// The sa device is IREE's synchronous local device with
//   - a heap allocator whose data comes from the context's arena: a window of
//     device-visible memory (the board's CMA, or the simulator's shared-memory
//     DDR), so every buffer has a physical address the accelerator can use;
//   - the sa-desc-v1 executable loader (sa_loader.c): one dispatch = one
//     descriptor list (SETREG bindings / constants, CALL the template, END)
//     submitted through the transport and waited for.
#ifndef SA_CONTEXT_H_
#define SA_CONTEXT_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/driver_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Completion record of one descriptor list (as rt_fw's, llm_inference_plan.md §5.1).
typedef struct sa_completion_t {
  uint32_t status;       // 0 = success, else the unit's extended status
  uint32_t cycles;       // device cycles (0 in the simulator)
  uint32_t descriptors;  // descriptors decoded (on failure: the failing one's index)
  uint32_t end;          // the END value
} sa_completion_t;

typedef struct sa_transport_t sa_transport_t;
struct sa_transport_t {
  const char* name;       // "sim" or "board"
  uint8_t* mem;           // the device-visible window, mapped into this process
  uint32_t mem_phys;      // its physical (device) address
  uint32_t mem_size;
  uint32_t heap_offset;   // first byte the arena may use (after transport-private data)
  uint32_t d;             // array size of the device
  // Runs the descriptor list at |list_phys| and waits for its completion.
  iree_status_t (*run)(sa_transport_t* t, uint32_t list_phys, sa_completion_t* out);
  void (*close)(sa_transport_t* t);
  void* impl;
};

// Transports (sa_transport_sim.c, sa_transport_board.c).
iree_status_t sa_transport_sim_open(iree_allocator_t host_allocator, sa_transport_t** out);
iree_status_t sa_transport_board_open(iree_allocator_t host_allocator, sa_transport_t** out);

// Board: memory and mailbox already mapped by the caller (for example a PYNQ
// launcher that loaded this library with ctypes), with rt_fw running and its
// ring at the start of |mem| (see sa_transport_board.c).
void sa_board_attach(void* mem, uint32_t mem_phys, uint32_t mem_size, void* mailbox, uint32_t ring_entries,
                     uint32_t d);

typedef struct sa_context_t {
  sa_transport_t* transport;
  iree_slim_mutex_t mutex;         // the arena and the list scratch
  uint8_t* heap_begin;
  uint8_t* heap_end;
  uint64_t* list;                  // scratch for dispatch lists (64-byte aligned)
  uint32_t list_phys;
  uint32_t list_capacity;          // descriptors
  uint64_t dispatches;             // statistics
  // the batch list (sa_context_batch_*)
  uint64_t* batch;
  uint32_t batch_phys, batch_capacity, batch_count, batch_dispatches;
  bool batch_fence;                // a FENCE before the next dispatch
  struct {
    uint32_t first;                // index of its first descriptor
    const char* name;
    uint32_t name_len;
  } batch_names[1024];
  uint64_t lists;                  // lists submitted (batches or single dispatches)
} sa_context_t;

// The process-wide context: the transport comes from SA_TRANSPORT (sim /
// board; default sim) or from sa_board_attach(). Created on first use and
// kept for the lifetime of the process (buffers may outlive devices).
iree_status_t sa_context_get(sa_context_t** out_context);

// iree_allocator_t over the arena (MALLOC / CALLOC / REALLOC / FREE).
iree_allocator_t sa_context_data_allocator(sa_context_t* context);

// Physical address of |ptr| inside the device-visible window.
iree_status_t sa_context_phys(sa_context_t* context, const void* ptr, uint32_t* out_phys);

// SETREG register numbers: BASE n = n, PARAM n = SA_REG_PARAM + n.
#define SA_REG_PARAM 16

// Runs one dispatch: builds SETREG (regs[i] = values[i], at most 24) + CALL
// entry_phys + END in the scratch list and runs it.
iree_status_t sa_context_dispatch(sa_context_t* context, uint32_t entry_phys, uint32_t count,
                                  const uint32_t* regs, const uint32_t* values, sa_completion_t* out);

// Batches (docs/iree_compiler_plan.md §5.2, C4): the dispatches of one
// command buffer go into one descriptor list, submitted once:
//   add (SETREG + CALL of one dispatch; `name` for error messages),
//   barrier (a FENCE before the next dispatch: its inputs are in DDR),
//   flush (END, run the list, wait; on a device error the failing dispatch
//   is named). A full list is flushed by add.
iree_status_t sa_context_batch_add(sa_context_t* context, uint32_t entry_phys, uint32_t count, const uint32_t* regs,
                                   const uint32_t* values, iree_string_view_t name);
void sa_context_batch_barrier(sa_context_t* context);
iree_status_t sa_context_batch_flush(sa_context_t* context);

// Profiling (SA_PROFILE=1): per export, the dispatch count, the device cycles
// (rt_fw's completion records) and the host time of the submission (list
// build, ring, wait).
void sa_context_profile_record(sa_context_t* context, iree_string_view_t name, uint32_t cycles, uint64_t host_ns);
// Prints the profile (sorted by device cycles) to |f| (a FILE*); per_step
// divides the totals (e.g. by the number of tokens). No-op without SA_PROFILE.
void sa_context_profile_report(void* f, double per_step);
// Totals since the last reset: dispatches, device cycles, host ns.
void sa_context_profile_totals(uint64_t* dispatches, uint64_t* cycles, uint64_t* host_ns);
void sa_context_profile_reset(void);

// Registers the "sa" HAL driver (sa_driver_module.c).
iree_status_t iree_hal_sa_driver_module_register(iree_hal_driver_registry_t* registry);

#ifdef __cplusplus
}
#endif

#endif  // SA_CONTEXT_H_

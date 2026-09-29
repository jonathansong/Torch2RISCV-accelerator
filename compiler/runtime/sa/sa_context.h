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

// A byte range [begin, end) of device-visible memory (host addresses).
typedef struct sa_range_t {
  uintptr_t begin, end;
} sa_range_t;

struct sa_batch_entry_t;

typedef struct sa_context_t {
  sa_transport_t* transport;
  iree_slim_mutex_t mutex;         // the arena and the list scratch
  uint8_t* heap_begin;
  uint8_t* heap_end;
  uint64_t heap_used, heap_peak;   // arena bytes in use (with block headers), its maximum
  uint64_t* list;                  // scratch for dispatch lists (64-byte aligned)
  uint32_t list_phys;
  uint32_t list_capacity;          // descriptors
  uint64_t dispatches;             // statistics
  // the batch (sa_context_batch_*): dispatches recorded, the list built at flush
  uint64_t* batch;
  uint32_t batch_phys, batch_capacity;
  struct sa_batch_entry_t* entries;
  uint32_t entry_count, entry_capacity, entry_descriptors;
  bool batch_fence;                // a barrier before the next dispatch
  struct {
    uint32_t first;                // index of its first descriptor
    const char* name;
    uint32_t name_len;
  } batch_names[1024];
  uint64_t lists;                  // lists submitted (batches or single dispatches)
  uint64_t host_dispatches;        // dispatches run on the host (the VMVX fallback)
  // statistics: prefixes run inside an earlier dispatch / before their FENCE,
  // barriers, FENCEs emitted
  uint64_t batch_hosted, batch_hoisted, batch_barriers, batch_fences;
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

// One dispatch of an sa-desc export (sa_loader.c): its entry points, register
// values and the memory it reads and writes (sa-desc version 3).
typedef struct sa_dispatch_t {
  uint32_t prefix_phys;            // loads it may run early (0: none; through BASE15)
  uint32_t head_phys;              // the body's leading loads (0: none, body_phys is the whole body)
  uint32_t body_phys;              // the body (after the head, if any)
  uint32_t prefix_base;            // BASE15 for the prefix
  uint32_t count;                  // register values (at most 24)
  const uint32_t* regs;
  const uint32_t* values;
  bool spad_b;                     // it touches SPAD_B (where the prefixes load)
  const sa_range_t* reads;         // DDR it loads from (the prefix included)
  uint32_t read_count;
  const sa_range_t* writes;        // DDR it stores to
  uint32_t write_count;
  const sa_range_t* prefix_reads;  // DDR the prefix loads from
  uint32_t prefix_read_count;
  iree_string_view_t name;
} sa_dispatch_t;

// Runs one dispatch: SETREG of its registers (+ BASE15), CALL prefix, head,
// body, END in the scratch list; runs it and waits.
iree_status_t sa_context_dispatch(sa_context_t* context, const sa_dispatch_t* dispatch, sa_completion_t* out);

// Batches (docs/iree_compiler_plan.md §5.2, §7): the dispatches of one
// command buffer go into one descriptor list, submitted once:
//   add (records a dispatch; the arrays are copied),
//   barrier (the next dispatch depends on the previous ones through DDR),
//   flush (builds the list, END, runs it, waits; on a device error the
//   failing dispatch is named). A full batch is flushed by add.
// The list: at a barrier a FENCE only when the DDR ranges conflict (reads
// after writes: ST idle; writes after reads: LD idle; none: no FENCE). A
// prefix whose ranges no dispatch of the list writes runs inside the earliest
// dispatch after the last one that touches SPAD_B, between its head and the
// rest (so its loads overlap that dispatch's compute), else before its own
// dispatch's FENCE when possible.
iree_status_t sa_context_batch_add(sa_context_t* context, const sa_dispatch_t* dispatch);
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

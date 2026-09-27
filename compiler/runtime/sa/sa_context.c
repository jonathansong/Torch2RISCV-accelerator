// sa HAL driver: the context, its arena allocator and the dispatch list (sa_context.h).
#include "sa_context.h"

#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------- descriptors
// Opcodes and encodings of docs/llm_inference_plan.md §5.3 (driver/pynq_matmul.py DescList).
enum { SA_OP_END = 0x12, SA_OP_SETREG = 0x14, SA_OP_CALL = 0x15 };
enum { SA_REG_PARAM = 16 };

static void sa_desc_clear(uint64_t* w) { memset(w, 0, 64); }

// SETREG of up to three (register, value) pairs, replace mode.
static void sa_desc_setreg(uint64_t* w, uint32_t n, const uint32_t* regs, const uint32_t* vals) {
  sa_desc_clear(w);
  w[0] = SA_OP_SETREG;
  uint64_t sel = 0;
  for (uint32_t i = 0; i < n; ++i) {
    sel |= (uint64_t)(0x40u | (regs[i] & 0x3Fu)) << (8 * i);
    w[2 + i] = vals[i];
  }
  w[1] = sel;
}

// ---------------------------------------------------------------- arena
// First-fit allocator over [heap_begin, heap_end): 16-byte block headers,
// blocks are contiguous; free merges forward, allocation merges free runs.
typedef struct sa_block_t {
  uint64_t size;  // bytes including this header, multiple of 16
  uint64_t used;
} sa_block_t;

static sa_block_t* sa_next(sa_context_t* c, sa_block_t* b) {
  uint8_t* n = (uint8_t*)b + b->size;
  return n < c->heap_end ? (sa_block_t*)n : NULL;
}

static void* sa_arena_alloc(sa_context_t* c, uint64_t bytes) {
  uint64_t need = ((bytes + sizeof(sa_block_t)) + 15) & ~(uint64_t)15;
  for (sa_block_t* b = (sa_block_t*)c->heap_begin; b; b = sa_next(c, b)) {
    if (b->used) continue;
    for (sa_block_t* n = sa_next(c, b); n && !n->used; n = sa_next(c, b)) b->size += n->size;
    if (b->size < need) continue;
    if (b->size - need >= 64) {
      sa_block_t* rest = (sa_block_t*)((uint8_t*)b + need);
      rest->size = b->size - need;
      rest->used = 0;
      b->size = need;
    }
    b->used = 1;
    return b + 1;
  }
  return NULL;
}

// The block holding |p|. Callers may pass any pointer inside a block: IREE's
// heap buffers allocate with iree_allocator_malloc_aligned but free the
// aligned pointer with iree_allocator_free (hal/buffer_heap.c, split mode).
static sa_block_t* sa_arena_block(sa_context_t* c, void* p) {
  for (sa_block_t* b = (sa_block_t*)c->heap_begin; b; b = sa_next(c, b)) {
    if ((uint8_t*)p > (uint8_t*)b && (uint8_t*)p < (uint8_t*)b + b->size) return b->used ? b : NULL;
  }
  return NULL;
}

static void sa_arena_free(sa_context_t* c, void* p) {
  if (!p) return;
  sa_block_t* b = sa_arena_block(c, p);
  if (!b) return;  // not ours (or already free): ignore
  b->used = 0;
  for (sa_block_t* n = sa_next(c, b); n && !n->used; n = sa_next(c, b)) b->size += n->size;
}

static iree_status_t sa_arena_ctl(void* self, iree_allocator_command_t command, const void* params,
                                  void** inout_ptr) {
  sa_context_t* c = (sa_context_t*)self;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&c->mutex);
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
    case IREE_ALLOCATOR_COMMAND_CALLOC: {
      uint64_t n = ((const iree_allocator_alloc_params_t*)params)->byte_length;
      void* p = sa_arena_alloc(c, n);
      if (!p) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "sa arena: out of device memory allocating %llu bytes",
                                  (unsigned long long)n);
      } else if (command == IREE_ALLOCATOR_COMMAND_CALLOC) {
        memset(p, 0, n);
      }
      *inout_ptr = p;
      break;
    }
    case IREE_ALLOCATOR_COMMAND_REALLOC: {
      uint64_t n = ((const iree_allocator_alloc_params_t*)params)->byte_length;
      void* old = *inout_ptr;
      sa_block_t* ob = old ? sa_arena_block(c, old) : NULL;
      if (old && !ob) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "sa arena: realloc of a foreign pointer");
        break;
      }
      uint64_t have = ob ? (uint64_t)((uint8_t*)ob + ob->size - (uint8_t*)old) : 0;
      if (old && have >= n) break;
      void* p = sa_arena_alloc(c, n);
      if (!p) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "sa arena: out of device memory");
        break;
      }
      if (old) {
        memcpy(p, old, have);
        sa_arena_free(c, old);
      }
      *inout_ptr = p;
      break;
    }
    case IREE_ALLOCATOR_COMMAND_FREE:
      sa_arena_free(c, *inout_ptr);
      *inout_ptr = NULL;
      break;
    default:
      status = iree_make_status(IREE_STATUS_UNIMPLEMENTED, "sa arena: allocator command %d", (int)command);
  }
  iree_slim_mutex_unlock(&c->mutex);
  return status;
}

iree_allocator_t sa_context_data_allocator(sa_context_t* context) {
  iree_allocator_t a = {context, sa_arena_ctl};
  return a;
}

iree_status_t sa_context_phys(sa_context_t* c, const void* ptr, uint32_t* out_phys) {
  const uint8_t* p = (const uint8_t*)ptr;
  sa_transport_t* t = c->transport;
  if (p < t->mem || p >= t->mem + t->mem_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sa: buffer %p is not in the device-visible window (%p + %u); "
                            "only buffers of the sa device allocator can be bound",
                            ptr, t->mem, t->mem_size);
  }
  *out_phys = t->mem_phys + (uint32_t)(p - t->mem);
  return iree_ok_status();
}

// ---------------------------------------------------------------- context
static sa_context_t* sa_global_context = NULL;

iree_status_t sa_context_get(sa_context_t** out_context) {
  if (sa_global_context) {
    *out_context = sa_global_context;
    return iree_ok_status();
  }
  iree_allocator_t host = iree_allocator_system();
  sa_transport_t* t = NULL;
  const char* which = getenv("SA_TRANSPORT");
  iree_status_t status;
  if (which && strcmp(which, "board") == 0) {
    status = sa_transport_board_open(host, &t);
  } else if (!which || strcmp(which, "sim") == 0) {
    status = sa_transport_sim_open(host, &t);
    // an attached board wins over the default
    if (!which && !iree_status_is_ok(status)) {
      iree_status_t board = sa_transport_board_open(host, &t);
      if (iree_status_is_ok(board)) {
        iree_status_ignore(status);
        status = board;
      } else {
        iree_status_ignore(board);
      }
    }
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "SA_TRANSPORT=%s: expected sim or board", which);
  }
  IREE_RETURN_IF_ERROR(status);

  sa_context_t* c = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host, sizeof(*c), (void**)&c));
  memset(c, 0, sizeof(*c));
  c->transport = t;
  iree_slim_mutex_initialize(&c->mutex);
  c->heap_begin = t->mem + ((t->heap_offset + 63) & ~63u);
  c->heap_end = t->mem + (t->mem_size & ~15u);
  sa_block_t* first = (sa_block_t*)c->heap_begin;
  first->size = (uint64_t)(c->heap_end - c->heap_begin);
  first->used = 0;

  // the dispatch list scratch: 64-byte aligned
  c->list_capacity = 32;
  void* raw = sa_arena_alloc(c, 64 * c->list_capacity + 64);
  if (!raw) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "sa: no room for the list scratch");
  c->list = (uint64_t*)(((uintptr_t)raw + 63) & ~(uintptr_t)63);
  IREE_RETURN_IF_ERROR(sa_context_phys(c, c->list, &c->list_phys));
  sa_global_context = c;
  *out_context = c;
  return iree_ok_status();
}

iree_status_t sa_context_dispatch(sa_context_t* c, uint32_t entry_phys, uint32_t binding_count,
                                  const uint32_t* binding_phys, uint32_t constant_count,
                                  const uint32_t* constants, sa_completion_t* out) {
  if (binding_count > 16 || constant_count > 6) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sa: %u bindings / %u constants (at most 16 / 6)", binding_count, constant_count);
  }
  uint32_t regs[22], vals[22], n = 0;
  for (uint32_t i = 0; i < binding_count; ++i) {
    regs[n] = i;
    vals[n++] = binding_phys[i];
  }
  for (uint32_t i = 0; i < constant_count; ++i) {
    regs[n] = SA_REG_PARAM + i;
    vals[n++] = constants[i];
  }
  iree_slim_mutex_lock(&c->mutex);
  uint64_t* w = c->list;
  uint32_t k = 0;
  for (uint32_t i = 0; i < n; i += 3, ++k) {
    sa_desc_setreg(w + 8 * k, n - i < 3 ? n - i : 3, regs + i, vals + i);
  }
  sa_desc_clear(w + 8 * k);
  w[8 * k] = SA_OP_CALL;
  w[8 * k + 1] = entry_phys;
  ++k;
  sa_desc_clear(w + 8 * k);
  w[8 * k] = SA_OP_END;
  w[8 * k + 1] = 0x5A;
  __sync_synchronize();
  iree_status_t status = c->transport->run(c->transport, c->list_phys, out);
  c->dispatches++;
  iree_slim_mutex_unlock(&c->mutex);
  return status;
}

// sa HAL driver: the context, its arena allocator and the dispatch list (sa_context.h).
#include "sa_context.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------- descriptors
// Opcodes and encodings of docs/llm_inference_plan.md §5.3 (driver/pynq_matmul.py DescList).
enum { SA_OP_FENCE = 0x10, SA_OP_END = 0x12, SA_OP_SETREG = 0x14, SA_OP_CALL = 0x15 };

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
    c->heap_used += b->size;
    if (c->heap_used > c->heap_peak) c->heap_peak = c->heap_used;
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
  c->heap_used -= b->size;
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
                                  "sa arena: out of device memory allocating %llu bytes (%.1f of %.1f MB in use)",
                                  (unsigned long long)n, c->heap_used / 1048576.0,
                                  (double)(c->heap_end - c->heap_begin) / 1048576.0);
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
  // the batch list: 8192 descriptors (512 KB)
  c->batch_capacity = 8192;
  void* braw = sa_arena_alloc(c, 64ull * c->batch_capacity + 64);
  if (!braw) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "sa: no room for the batch list");
  c->batch = (uint64_t*)(((uintptr_t)braw + 63) & ~(uintptr_t)63);
  IREE_RETURN_IF_ERROR(sa_context_phys(c, c->batch, &c->batch_phys));
  sa_global_context = c;
  *out_context = c;
  return iree_ok_status();
}

// SETREGs of n registers at w; returns the descriptors written.
static uint32_t sa_emit_setregs(uint64_t* w, uint32_t n, const uint32_t* regs, const uint32_t* vals) {
  uint32_t k = 0;
  for (uint32_t i = 0; i < n; i += 3, ++k) sa_desc_setreg(w + 8 * k, n - i < 3 ? n - i : 3, regs + i, vals + i);
  return k;
}

static void sa_emit(uint64_t* w, uint32_t* k, uint32_t op, uint64_t w1) {
  uint64_t* d = w + 8 * *k;
  sa_desc_clear(d);
  d[0] = op;
  d[1] = w1;
  ++*k;
}

iree_status_t sa_context_dispatch(sa_context_t* c, const sa_dispatch_t* x, sa_completion_t* out) {
  uint32_t n = x->count;
  if (n > 24 || (n + 3) / 3 + 4 > c->list_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "sa: %u register values (at most 24)", n);
  }
  iree_slim_mutex_lock(&c->mutex);
  uint64_t* w = c->list;
  uint32_t k = sa_emit_setregs(w, n, x->regs, x->values);
  if (x->prefix_phys) {
    uint32_t r = 15;
    k += sa_emit_setregs(w + 8 * k, 1, &r, &x->prefix_base);
    sa_emit(w, &k, SA_OP_CALL, x->prefix_phys);
  }
  if (x->head_phys) sa_emit(w, &k, SA_OP_CALL, x->head_phys);
  sa_emit(w, &k, SA_OP_CALL, x->body_phys);
  sa_emit(w, &k, SA_OP_END, 0x5A);
  __sync_synchronize();
  iree_status_t status = c->transport->run(c->transport, c->list_phys, out);
  c->dispatches++;
  c->lists++;
  iree_slim_mutex_unlock(&c->mutex);
  return status;
}

// ---------------------------------------------------------------- batches
static const char* sa_engine_names[] = {"LD", "ST", "EX", "VE", "FETCH"};
static const char* sa_error_names[] = {"none", "shape", "range", "read response", "write response"};

// A recorded dispatch (sa_context_batch_add).
typedef struct sa_batch_entry_t {
  sa_dispatch_t d;                 // its arrays point into this entry
  uint32_t regs[24], vals[24];
  sa_range_t ranges[48];           // reads, writes, prefix reads
  bool barrier;                    // a barrier before it
  int host;                        // the dispatch whose prefix runs inside it (-1: none)
  bool hosted;                     // its prefix runs inside an earlier dispatch
} sa_batch_entry_t;

// A set of DDR ranges (overflow: everything).
typedef struct sa_range_set_t {
  sa_range_t r[256];
  uint32_t n;
  bool all;
} sa_range_set_t;

static void sa_set_add(sa_range_set_t* s, const sa_range_t* r, uint32_t n) {
  for (uint32_t i = 0; i < n && !s->all; ++i) {
    if (s->n == IREE_ARRAYSIZE(s->r)) s->all = true;
    else s->r[s->n++] = r[i];
  }
}

static bool sa_set_overlaps(const sa_range_set_t* s, const sa_range_t* r, uint32_t n) {
  if (n == 0) return false;
  if (s->all) return true;
  for (uint32_t i = 0; i < n; ++i)
    for (uint32_t j = 0; j < s->n; ++j)
      if (r[i].begin < s->r[j].end && s->r[j].begin < r[i].end) return true;
  return false;
}

static uint32_t sa_entry_descriptors(const sa_dispatch_t* x) {
  return (x->count + 3) / 3 + 1 /* FENCE */ + 3 /* CALLs */ + 2 /* a hosted prefix: SETREG, CALL */;
}

iree_status_t sa_context_batch_add(sa_context_t* c, const sa_dispatch_t* x) {
  if (x->count > 24 || x->read_count > 16 || x->write_count > 16 || x->prefix_read_count > 16) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "sa: dispatch '%.*s' too large for a batch",
                            (int)x->name.size, x->name.data);
  }
  if (!c->entries) {
    c->entry_capacity = IREE_ARRAYSIZE(c->batch_names);
    c->entries = (sa_batch_entry_t*)calloc(c->entry_capacity, sizeof(sa_batch_entry_t));
    if (!c->entries) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "sa: no memory for the batch");
  }
  uint32_t need = sa_entry_descriptors(x);
  if (c->entry_count == c->entry_capacity || c->entry_descriptors + need + 1 > c->batch_capacity) {
    IREE_RETURN_IF_ERROR(sa_context_batch_flush(c));
  }
  iree_slim_mutex_lock(&c->mutex);
  sa_batch_entry_t* e = &c->entries[c->entry_count++];
  e->d = *x;
  memcpy(e->regs, x->regs, x->count * sizeof(uint32_t));
  memcpy(e->vals, x->values, x->count * sizeof(uint32_t));
  e->d.regs = e->regs;
  e->d.values = e->vals;
  sa_range_t* r = e->ranges;
  memcpy(r, x->reads, x->read_count * sizeof(*r));
  e->d.reads = r;
  r += x->read_count;
  memcpy(r, x->writes, x->write_count * sizeof(*r));
  e->d.writes = r;
  r += x->write_count;
  memcpy(r, x->prefix_reads, x->prefix_read_count * sizeof(*r));
  e->d.prefix_reads = r;
  e->barrier = c->batch_fence;
  e->host = -1;
  e->hosted = false;
  c->batch_fence = false;
  c->entry_descriptors += need;
  c->dispatches++;
  iree_slim_mutex_unlock(&c->mutex);
  return iree_ok_status();
}

void sa_context_batch_barrier(sa_context_t* c) { c->batch_fence = true; }

// Builds the list of the recorded dispatches (sa_context_batch_add); returns
// the descriptor count (END included).
static uint32_t sa_batch_build(sa_context_t* c) {
  sa_batch_entry_t* e = c->entries;
  uint32_t m = c->entry_count;
  static sa_range_set_t written_all, reads, writes;     // under c->mutex
  // where the prefixes run: inside the earliest dispatch with a head after
  // the last one touching SPAD_B, if nothing in the list writes what they read
  written_all.n = 0;
  written_all.all = false;
  for (uint32_t i = 0; i < m; ++i) sa_set_add(&written_all, e[i].d.writes, e[i].d.write_count);
  for (uint32_t k = 0; k < m; ++k) {
    if (!e[k].d.prefix_phys || sa_set_overlaps(&written_all, e[k].d.prefix_reads, e[k].d.prefix_read_count))
      continue;
    int best = -1;
    for (int j = (int)k - 1; j >= 0 && !e[j].d.spad_b; --j)
      if (e[j].d.head_phys && e[j].host < 0) best = j;
    if (best >= 0) {
      e[best].host = (int)k;
      e[k].hosted = true;
    }
  }
  uint64_t* w = c->batch;
  uint32_t k = 0, r15 = 15;
  reads.n = writes.n = 0;
  reads.all = writes.all = false;
  for (uint32_t i = 0; i < m; ++i) {
    const sa_dispatch_t* x = &e[i].d;
    bool early = false;                 // the prefix before the FENCE
    if (e[i].barrier && i > 0) {
      c->batch_barriers++;
      uint32_t mask = 0;                // FENCE engines: bit 0 LD, bit 1 ST
      if (sa_set_overlaps(&writes, x->reads, x->read_count) || sa_set_overlaps(&writes, x->writes, x->write_count))
        mask |= 2;
      if (sa_set_overlaps(&reads, x->writes, x->write_count)) mask |= 1;
      if (mask && x->prefix_phys && !e[i].hosted &&
          !sa_set_overlaps(&writes, x->prefix_reads, x->prefix_read_count)) {
        early = true;
        k += sa_emit_setregs(w + 8 * k, 1, &r15, &x->prefix_base);
        sa_emit(w, &k, SA_OP_CALL, x->prefix_phys);
        sa_set_add(&reads, x->prefix_reads, x->prefix_read_count);
        c->batch_hoisted++;
      }
      if (mask) {
        sa_emit(w, &k, SA_OP_FENCE, mask);
        c->batch_fences++;
        if (mask & 2) writes.n = 0, writes.all = false;
        if (mask & 1) reads.n = 0, reads.all = false;
      }
    }
    if (i < IREE_ARRAYSIZE(c->batch_names)) {
      c->batch_names[i].first = k;
      c->batch_names[i].name = x->name.data;
      c->batch_names[i].name_len = (uint32_t)x->name.size;
    }
    k += sa_emit_setregs(w + 8 * k, x->count, x->regs, x->values);
    if (x->prefix_phys && !e[i].hosted && !early) {
      k += sa_emit_setregs(w + 8 * k, 1, &r15, &x->prefix_base);
      sa_emit(w, &k, SA_OP_CALL, x->prefix_phys);
    }
    if (x->head_phys) {
      sa_emit(w, &k, SA_OP_CALL, x->head_phys);
      if (e[i].host >= 0) {             // another dispatch's prefix, during this one's compute
        const sa_dispatch_t* p = &e[e[i].host].d;
        k += sa_emit_setregs(w + 8 * k, 1, &r15, &p->prefix_base);
        sa_emit(w, &k, SA_OP_CALL, p->prefix_phys);
        sa_set_add(&reads, p->prefix_reads, p->prefix_read_count);
        c->batch_hosted++;
      }
    }
    sa_emit(w, &k, SA_OP_CALL, x->body_phys);
    sa_set_add(&reads, x->reads, x->read_count);
    sa_set_add(&writes, x->writes, x->write_count);
  }
  sa_emit(w, &k, SA_OP_END, 0x5A);
  return k;
}

iree_status_t sa_context_batch_flush(sa_context_t* c) {
  iree_slim_mutex_lock(&c->mutex);
  if (c->entry_count == 0) {
    c->batch_fence = false;
    iree_slim_mutex_unlock(&c->mutex);
    return iree_ok_status();
  }
  sa_batch_build(c);
  __sync_synchronize();
  sa_completion_t done = {0};
  uint64_t t0 = iree_time_now();
  iree_status_t status = c->transport->run(c->transport, c->batch_phys, &done);
  uint64_t host_ns = (uint64_t)(iree_time_now() - t0);
  c->lists++;
  uint32_t ndisp = c->entry_count;
  if (ndisp > IREE_ARRAYSIZE(c->batch_names)) ndisp = IREE_ARRAYSIZE(c->batch_names);
  c->entry_count = 0;
  c->entry_descriptors = 0;
  c->batch_fence = false;
  if (iree_status_is_ok(status) && done.status != 0) {
    // the dispatch holding the failing descriptor (descriptors decoded count
    // the top-level list and the CALLed templates; report the last dispatch
    // whose CALL was reached)
    uint32_t eng = (done.status >> 12) & 0xF, code = (done.status >> 8) & 0xF;
    const char* en = eng < 5 ? sa_engine_names[eng] : "?";
    const char* cn = code < 5 ? sa_error_names[code] : "?";
    if (ndisp == 1) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "sa: dispatch '%.*s' failed on the %s device: %s %s error (status %#x, %u "
                                "descriptors decoded)",
                                (int)c->batch_names[0].name_len, c->batch_names[0].name, c->transport->name, en, cn,
                                done.status, done.descriptors);
    } else {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "sa: one of %u dispatches ('%.*s' .. '%.*s') failed on the %s device: %s %s error (status %#x, %u "
          "descriptors decoded; SA_NO_BATCH=1 runs them one by one to find it)",
          ndisp, (int)c->batch_names[0].name_len, c->batch_names[0].name, (int)c->batch_names[ndisp - 1].name_len,
          c->batch_names[ndisp - 1].name, c->transport->name, en, cn, done.status, done.descriptors);
    }
  } else if (iree_status_is_ok(status) && done.end != 0x5A) {
    status = iree_make_status(IREE_STATUS_INTERNAL, "sa: batch ended with %#x (expected 0x5a)", done.end);
  }
  iree_slim_mutex_unlock(&c->mutex);
  if (iree_status_is_ok(status)) sa_context_profile_record(c, IREE_SV("(list of dispatches)"), done.cycles, host_ns, &done);
  return status;
}

// ---------------------------------------------------------------- profiling
typedef struct sa_profile_entry_t {
  char name[96];
  uint64_t count, cycles, host_ns;
  uint64_t perf[32];                       // event counter sums (SA_PROFILE_PERF)
} sa_profile_entry_t;
static uint32_t sa_profile_perf_n;         // counters seen (0: none)
static int sa_profile_section;
static sa_profile_entry_t sa_profile[512];
static int sa_profile_n = -1;              // -1: not initialized, -2: disabled

static int sa_profile_enabled(void) {
  if (sa_profile_n == -1) sa_profile_n = getenv("SA_PROFILE") ? 0 : -2;
  return sa_profile_n >= 0;
}

void sa_context_profile_record(sa_context_t* context, iree_string_view_t name, uint32_t cycles, uint64_t host_ns,
                               const sa_completion_t* done) {
  (void)context;
  if (!sa_profile_enabled()) return;
  int i = 0;
  for (; i < sa_profile_n; ++i)
    if (strlen(sa_profile[i].name) == name.size && !memcmp(sa_profile[i].name, name.data, name.size)) break;
  if (i == sa_profile_n) {
    if (sa_profile_n == 512) return;
    size_t n = name.size < 95 ? name.size : 95;
    memcpy(sa_profile[i].name, name.data, n);
    sa_profile[i].name[n] = 0;
    ++sa_profile_n;
  }
  sa_profile[i].count++;
  sa_profile[i].cycles += cycles;
  sa_profile[i].host_ns += host_ns;
  uint32_t pn = done && done->perf_n <= 32 ? done->perf_n : 0;
  for (uint32_t k = 0; k < pn; ++k) sa_profile[i].perf[k] += done->perf[k];
  if (pn > sa_profile_perf_n) sa_profile_perf_n = pn;
}

void sa_context_profile_totals(uint64_t* dispatches, uint64_t* cycles, uint64_t* host_ns) {
  *dispatches = *cycles = *host_ns = 0;
  for (int i = 0; i < sa_profile_n; ++i) {
    *dispatches += sa_profile[i].count;
    *cycles += sa_profile[i].cycles;
    *host_ns += sa_profile[i].host_ns;
  }
}

void sa_context_profile_reset(void) {
  if (sa_profile_n > 0) {
    memset(sa_profile, 0, sizeof(sa_profile));
    sa_profile_n = 0;
  }
}

static int sa_profile_cmp(const void* a, const void* b) {
  const sa_profile_entry_t *x = a, *y = b;
  return x->cycles < y->cycles ? 1 : x->cycles > y->cycles ? -1 : 0;
}

void sa_context_profile_report(void* fp, double per_step) {
  if (!sa_profile_enabled() || sa_profile_n <= 0) return;
  FILE* f = (FILE*)fp;
  qsort(sa_profile, (size_t)sa_profile_n, sizeof(sa_profile[0]), sa_profile_cmp);
  uint64_t n, cyc, ns;
  sa_context_profile_totals(&n, &cyc, &ns);
  fprintf(f, "%-58s %8s %12s %8s %10s\n", "export", "calls", "cycles", "cyc %", "host us");
  for (int i = 0; i < sa_profile_n; ++i) {
    const sa_profile_entry_t* e = &sa_profile[i];
    fprintf(f, "%-58.58s %8.1f %12.0f %7.1f%% %10.1f\n", e->name, e->count / per_step, e->cycles / per_step,
            100.0 * (double)e->cycles / (double)(cyc ? cyc : 1), e->host_ns / per_step / 1e3);
  }
  fprintf(f, "%-58s %8.1f %12.0f %8s %10.1f   (per step)\n", "total", n / per_step, cyc / per_step, "",
          ns / per_step / 1e3);
  // SA_PROFILE_PERF=<file>: the raw sums (not divided) with the event counters,
  // one section per report (prefill chunks, then decode steps)
  const char* path = getenv("SA_PROFILE_PERF");
  if (path && sa_profile_perf_n) {
    FILE* c = fopen(path, sa_profile_section++ ? "a" : "w");
    if (c) {
      fprintf(c, "# section %d per_step %.3f\nexport,calls,cycles,host_ns", sa_profile_section - 1, per_step);
      for (uint32_t k = 0; k < sa_profile_perf_n; ++k) fprintf(c, ",c%u", k);
      fprintf(c, "\n");
      for (int i = 0; i < sa_profile_n; ++i) {
        const sa_profile_entry_t* e = &sa_profile[i];
        fprintf(c, "%s,%llu,%llu,%llu", e->name, (unsigned long long)e->count, (unsigned long long)e->cycles,
                (unsigned long long)e->host_ns);
        for (uint32_t k = 0; k < sa_profile_perf_n; ++k) fprintf(c, ",%llu", (unsigned long long)e->perf[k]);
        fprintf(c, "\n");
      }
      fclose(c);
    }
  }
}

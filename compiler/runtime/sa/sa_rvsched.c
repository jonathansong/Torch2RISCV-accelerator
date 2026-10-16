// sa HAL driver: the step table of the decode loop on the RISC-V core (sa_rvsched.h).
// A step is a sequence of items (sa_record_t): descriptor lists, or with
// SA_RV calls of the dispatches' generated RISC-V code (plan A inside plan B,
// docs/riscv_backend_design.md §0, S3). Bucket 0's items are copied into the
// table once (a list's descriptors, a call's 64-word argument block); the
// words that differ between buckets are patched by the core at a bucket change.
// SA_R5=<dir> (R5, docs/riscv_backend_design.md §5): the core picks the block
// of the largest maximum with a kernel IREE's llvm-cpu compiled for it
// (compiler/tests/r5_kernel.py: r5.bin, r5.txt; the sim calls r5_host.so).
#include "sa_rvsched.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../firmware/include/rvsched.h"  // the decode schedule shared with rt_fw

typedef struct sa_rvs_bucket_t {
  sa_record_t rec;
  sa_rvsched_io_t io;
  bool have;
} sa_rvs_bucket_t;

// a patched word: item, word in it
typedef struct sa_rvs_word_t {
  uint32_t item, word;
} sa_rvs_word_t;

struct sa_rvsched_t {
  sa_context_t* c;
  uint32_t pad, b0, nb, max_steps, nblocks, block;
  int pass;                          // 1, 2; 3 after the second (record_end then checks)
  sa_rvs_bucket_t* buckets;          // this pass's recordings
  sa_record_t rec;                   // the recording in progress
  // from pass 1: the items' shape and the room for the patched words
  uint32_t nitems, cap;
  uint32_t* item_code;               // per item: 0 (a list) or the function
  uint32_t* item_words;              // per item: its words
  uint32_t* off_copy;                // per item: its copy in the table
  // the table in the device window
  void* raw;
  uint8_t* table;
  uint32_t table_phys, table_bytes;
  uint32_t off_items, off_words, off_buckets, off_values, off_out;
  uint32_t nwords;
  uint32_t last_pos;                 // position of the last step run
  // R5: the block argmax kernel (code, then its argument block) in the device window
  void* r5_raw;
  uint32_t r5_bytes, r5_n, r5_phys, r5_args_phys;
  char r5_symbol[160];
};

static void sa_rvs_clear(sa_rvsched_t* s) {
  for (uint32_t b = 0; b < s->nb; ++b) {
    sa_record_free(&s->buckets[b].rec);
    s->buckets[b].have = false;
  }
}

iree_status_t sa_rvsched_create(sa_context_t* context, uint32_t pad, uint32_t bucket0, uint32_t nbuckets,
                                uint32_t max_steps, sa_rvsched_t** out) {
  if (!context->transport->run_sched) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "sa: the %s transport cannot run a decode schedule",
                            context->transport->name);
  }
  if (pad == 0 || nbuckets == 0 || max_steps == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "sa rvsched: empty schedule");
  }
  sa_rvsched_t* s = (sa_rvsched_t*)calloc(1, sizeof(*s));
  if (!s) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "sa rvsched");
  s->buckets = (sa_rvs_bucket_t*)calloc(nbuckets, sizeof(*s->buckets));
  if (!s->buckets) {
    free(s);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "sa rvsched");
  }
  s->c = context;
  s->pad = pad;
  s->b0 = bucket0;
  s->nb = nbuckets;
  s->max_steps = max_steps;
  s->pass = 1;
  *out = s;
  return iree_ok_status();
}

void sa_rvsched_record_begin(sa_rvsched_t* s) {
  sa_record_free(&s->rec);
  s->rec.dry = true;
  s->c->record = &s->rec;
}

static bool sa_rvs_same(const sa_record_t* a, const sa_record_t* b) {
  if (a->count != b->count) return false;
  for (uint32_t i = 0; i < a->count; ++i) {
    const sa_record_item_t *x = &a->items[i], *y = &b->items[i];
    if (x->code_phys != y->code_phys || x->nwords != y->nwords || memcmp(x->words, y->words, 4ull * x->nwords))
      return false;
  }
  return true;
}

iree_status_t sa_rvsched_record_end(sa_rvsched_t* s, uint32_t bucket, const sa_rvsched_io_t* io) {
  s->c->record = NULL;
  if (bucket < s->b0 || bucket - s->b0 >= s->nb) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "sa rvsched: bucket %u outside the table", bucket);
  }
  if (s->rec.dropped || s->rec.count == 0) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "sa rvsched: a step of %u items, %u not recorded",
                            s->rec.count, s->rec.dropped);
  }
  sa_rvs_bucket_t* e = &s->buckets[bucket - s->b0];
  if (s->pass == 3) {                // a check: the same items as the bucket's recording
    bool same = e->have && sa_rvs_same(&e->rec, &s->rec) && !memcmp(&e->io, io, sizeof(*io));
    sa_record_free(&s->rec);
    if (!same) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "sa rvsched: a step of bucket %u differs from its recording", bucket);
    }
    return iree_ok_status();
  }
  sa_record_free(&e->rec);
  e->rec = s->rec;                   // (takes the copies)
  memset(&s->rec, 0, sizeof(s->rec));
  e->io = *io;
  e->have = true;
  return iree_ok_status();
}

// The items' words that differ from bucket 0's in any bucket: their count,
// the first cap of them in idx.
static iree_status_t sa_rvs_diff(sa_rvsched_t* s, sa_rvs_word_t* idx, uint32_t cap, uint32_t* out_n) {
  const sa_record_t* r0 = &s->buckets[0].rec;
  for (uint32_t b = 0; b < s->nb; ++b) {
    const sa_rvs_bucket_t* e = &s->buckets[b];
    if (!e->have) return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "sa rvsched: bucket %u not recorded", s->b0 + b);
    if (e->rec.count != r0->count) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "sa rvsched: bucket %u has %u items, bucket %u %u",
                              s->b0 + b, e->rec.count, s->b0, r0->count);
    }
    for (uint32_t i = 0; i < r0->count; ++i)
      if (e->rec.items[i].nwords != r0->items[i].nwords || e->rec.items[i].code_phys != r0->items[i].code_phys) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "sa rvsched: item %u differs in shape between buckets %u and %u", i, s->b0 + b, s->b0);
      }
  }
  uint32_t n = 0;
  for (uint32_t i = 0; i < r0->count; ++i) {
    const uint32_t* w0 = r0->items[i].words;
    for (uint32_t w = 0; w < r0->items[i].nwords; ++w) {
      for (uint32_t b = 1; b < s->nb; ++b) {
        if (s->buckets[b].rec.items[i].words[w] != w0[w]) {
          if (idx && n < cap) idx[n] = (sa_rvs_word_t){i, w};
          ++n;
          break;
        }
      }
    }
  }
  *out_n = n;
  return iree_ok_status();
}

static uint32_t sa_align64(uint32_t x) { return (x + 63u) & ~63u; }

static sa_transport_t* sa_rvs_r5_t;     // (the host build's binding addresses)
static void* sa_rvs_r5_ptr(uint32_t phys) { return sa_rvs_r5_t->mem + (phys - sa_rvs_r5_t->mem_phys); }

// R5: loads SA_R5's kernel into the device window (after pass 1, with the
// table: the second pass's recordings see the same allocations).
static iree_status_t sa_rvs_r5_load(sa_rvsched_t* s, const char* dir) {
  sa_context_t* c = s->c;
  sa_transport_t* t = c->transport;
  if (t->fw_version < 6) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "sa r5: the %s transport's rt_fw is version %u, R5 needs 6",
                            t->name, t->fw_version);
  }
  char path[1024];
  unsigned n = 0, bytes = 0;
  snprintf(path, sizeof(path), "%s/r5.txt", dir);
  FILE* f = fopen(path, "r");
  if (!f) return iree_make_status(IREE_STATUS_NOT_FOUND, "sa r5: cannot open %s", path);
  int got = fscanf(f, "%159s %u %u", s->r5_symbol, &n, &bytes);
  fclose(f);
  if (got != 3) return iree_make_status(IREE_STATUS_DATA_LOSS, "sa r5: %s is not '<symbol> <n> <bytes>'", path);
  s->r5_n = n;
  uint32_t args_off = sa_align64(bytes);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(sa_context_data_allocator(c), args_off + 68u * 4u + 64u, &s->r5_raw));
  uint8_t* code = (uint8_t*)(((uintptr_t)s->r5_raw + 63) & ~(uintptr_t)63);
  snprintf(path, sizeof(path), "%s/r5.bin", dir);
  f = fopen(path, "rb");
  if (!f) return iree_make_status(IREE_STATUS_NOT_FOUND, "sa r5: cannot open %s", path);
  size_t rd = fread(code, 1, bytes, f);
  fclose(f);
  if (rd != bytes) return iree_make_status(IREE_STATUS_DATA_LOSS, "sa r5: %s is not %u bytes", path, bytes);
  memset(code + args_off, 0, 68u * 4u);
  uint32_t code_phys = 0;
  IREE_RETURN_IF_ERROR(sa_context_phys(c, code, &code_phys));
  if (strcmp(t->name, "sim") == 0) {     // the sim runs the host build
    snprintf(path, sizeof(path), "%s/r5_host.so", dir);
    void* so = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!so) return iree_make_status(IREE_STATUS_NOT_FOUND, "sa r5: %s", dlerror());
    void (*init)(void* (*)(uint32_t)) = (void (*)(void* (*)(uint32_t)))dlsym(so, "r5_host_init");
    sa_rv_host_fn_t fn = (sa_rv_host_fn_t)dlsym(so, "r5_entry");
    if (!init || !fn) return iree_make_status(IREE_STATUS_NOT_FOUND, "sa r5: no r5_host_init / r5_entry in %s", path);
    sa_rvs_r5_t = t;
    init(sa_rvs_r5_ptr);
    sa_rv_set_host_kernel(code_phys, fn);
  }
  s->r5_bytes = bytes;
  s->r5_phys = code_phys;
  s->r5_args_phys = code_phys + args_off;
  fprintf(stderr, "sa r5: the block argmax by %s (IREE llvm-cpu, %u bytes of rv32) on the core\n", s->r5_symbol,
          bytes);
  return iree_ok_status();
}

iree_status_t sa_rvsched_end_pass(sa_rvsched_t* s) {
  sa_context_t* c = s->c;
  if (s->pass == 1) {
    uint32_t n = 0;
    IREE_RETURN_IF_ERROR(sa_rvs_diff(s, NULL, 0, &n));
    const sa_record_t* r0 = &s->buckets[0].rec;
    s->nitems = r0->count;
    s->cap = n + n / 4 + 64;         // the second pass's addresses differ: room for a few more
    s->item_code = (uint32_t*)calloc(s->nitems, 4);
    s->item_words = (uint32_t*)calloc(s->nitems, 4);
    s->off_copy = (uint32_t*)calloc(s->nitems, 4);
    if (!s->item_code || !s->item_words || !s->off_copy) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "sa rvsched");
    uint32_t o = sa_align64(RVS_T_WORDS_ * 4u);
    for (uint32_t i = 0; i < s->nitems; ++i) {
      s->item_code[i] = r0->items[i].code_phys;
      s->item_words[i] = r0->items[i].nwords;
      s->off_copy[i] = o;
      o = sa_align64(o + 4u * r0->items[i].nwords);
    }
    s->off_items = o;
    o = sa_align64(o + 8u * s->nitems);
    s->off_words = o;
    o = sa_align64(o + 4u * s->cap);
    s->off_buckets = o;
    o = sa_align64(o + 4u * RVS_B_WORDS_ * s->nb);
    s->off_values = o;
    o = sa_align64(o + 4u * s->cap * s->nb);
    s->off_out = o;
    o = sa_align64(o + 4u * RVS_O_WORDS_ * s->max_steps);
    s->table_bytes = o;
    iree_allocator_t a = sa_context_data_allocator(c);
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(a, (iree_host_size_t)o + 64, &s->raw));
    s->table = (uint8_t*)(((uintptr_t)s->raw + 63) & ~(uintptr_t)63);
    IREE_RETURN_IF_ERROR(sa_context_phys(c, s->table, &s->table_phys));
    const char* r5 = getenv("SA_R5");
    if (r5 && *r5) {
      IREE_RETURN_IF_ERROR(sa_rvs_r5_load(s, r5));
    }
    sa_rvs_clear(s);
    s->pass = 2;
    return iree_ok_status();
  }
  if (s->pass != 2) return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "sa rvsched: both passes done");
  const sa_record_t* r0 = &s->buckets[0].rec;
  if (r0->count != s->nitems) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "sa rvsched: %u items in pass 2, %u in pass 1",
                            r0->count, s->nitems);
  }
  for (uint32_t i = 0; i < s->nitems; ++i)
    if (r0->items[i].nwords != s->item_words[i] || r0->items[i].code_phys != s->item_code[i]) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "sa rvsched: item %u changed between the passes", i);
    }
  sa_rvs_word_t* idx = (sa_rvs_word_t*)malloc(sizeof(*idx) * s->cap);
  if (!idx) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "sa rvsched");
  uint32_t n = 0;
  iree_status_t status = sa_rvs_diff(s, idx, s->cap, &n);
  if (iree_status_is_ok(status) && n > s->cap) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "sa rvsched: %u patched words, room for %u", n, s->cap);
  }
  const sa_rvsched_io_t* io0 = &s->buckets[0].io;
  for (uint32_t b = 1; iree_status_is_ok(status) && b < s->nb; ++b)
    if (s->buckets[b].io.vocab != io0->vocab || s->buckets[b].io.nblocks != io0->nblocks) status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION, "sa rvsched: the logits' shape differs between buckets");
  if (iree_status_is_ok(status) && (io0->nblocks == 0 || io0->vocab % io0->nblocks)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "sa rvsched: %u block maxima of %u logits", io0->nblocks,
                              io0->vocab);
  }
  if (!iree_status_is_ok(status)) {
    free(idx);
    return status;
  }
  s->nblocks = io0->nblocks;
  s->block = io0->vocab / io0->nblocks;
  s->nwords = n;
  uint8_t* t = s->table;
  uint32_t P = s->table_phys;
  uint32_t* hdr = (uint32_t*)t;
  memset(hdr, 0, RVS_T_WORDS_ * 4u);
  hdr[RVS_T_MAGIC] = RVS_MAGIC;
  hdr[RVS_T_NITEMS] = s->nitems;
  hdr[RVS_T_ITEMS] = P + s->off_items;
  hdr[RVS_T_NWORDS] = n;
  hdr[RVS_T_WORDS] = P + s->off_words;
  hdr[RVS_T_PAD] = s->pad;
  hdr[RVS_T_BUCKET0] = s->b0;
  hdr[RVS_T_NBUCKETS] = s->nb;
  hdr[RVS_T_BUCKETS] = P + s->off_buckets;
  hdr[RVS_T_VALUES] = P + s->off_values;
  hdr[RVS_T_NBLOCKS] = s->nblocks;
  hdr[RVS_T_BLOCK] = s->block;
  hdr[RVS_T_OUT] = P + s->off_out;
  uint32_t* items = (uint32_t*)(t + s->off_items);
  for (uint32_t i = 0; i < s->nitems; ++i) {
    memcpy(t + s->off_copy[i], r0->items[i].words, 4ull * s->item_words[i]);
    items[2 * i] = P + s->off_copy[i];
    items[2 * i + 1] = s->item_code[i];
  }
  uint32_t* words = (uint32_t*)(t + s->off_words);
  for (uint32_t i = 0; i < n; ++i) words[i] = P + s->off_copy[idx[i].item] + 4u * idx[i].word;
  for (uint32_t b = 0; b < s->nb; ++b) {
    const sa_rvs_bucket_t* e = &s->buckets[b];
    uint32_t* bw = (uint32_t*)(t + s->off_buckets) + RVS_B_WORDS_ * b;
    memset(bw, 0, 4u * RVS_B_WORDS_);
    bw[RVS_B_TOKEN] = e->io.token;
    bw[RVS_B_POS] = e->io.pos;
    bw[RVS_B_VALID] = e->io.valid;
    bw[RVS_B_LOGITS] = e->io.logits;
    bw[RVS_B_BMAX] = e->io.bmax;
    uint32_t* v = (uint32_t*)(t + s->off_values) + (size_t)n * b;
    for (uint32_t i = 0; i < n; ++i) v[i] = e->rec.items[idx[i].item].words[idx[i].word];
  }
  free(idx);
  if (s->r5_bytes) {
    if (s->r5_n != s->nblocks) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "sa r5: the kernel takes %u block maxima, the logits have %u", s->r5_n, s->nblocks);
    }
    hdr[RVS_T_ARGMAX] = s->r5_phys;
    hdr[RVS_T_ARGMAX_ARGS] = s->r5_args_phys;
  }
  __sync_synchronize();
  s->pass = 3;
  return iree_ok_status();
}

void sa_rvsched_info(const sa_rvsched_t* s, sa_rvsched_info_t* out) {
  memset(out, 0, sizeof(*out));
  for (uint32_t i = 0; i < s->nitems; ++i) {
    if (s->item_code[i]) {
      out->calls++;
    } else {
      out->lists++;
      out->descriptors += s->item_words[i] / 16;
    }
  }
  out->words = s->nwords;
  out->table_bytes = s->table_bytes;
  out->table_phys = s->table_phys;
  out->r5_bytes = s->r5_bytes;
  out->r5_symbol = s->r5_bytes ? s->r5_symbol : NULL;
}

iree_status_t sa_rvsched_run(sa_rvsched_t* s, uint32_t pos, uint32_t steps, uint32_t token, int32_t stop,
                             uint32_t* out_tokens, uint32_t* out_cycles, uint32_t* out_device, uint32_t* out_done) {
  *out_done = 0;
  if (s->pass != 3) return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "sa rvsched: the table is not written");
  if (steps > s->max_steps) steps = s->max_steps;
  volatile uint32_t* hdr = (volatile uint32_t*)s->table;
  hdr[RVS_T_POS] = pos;
  hdr[RVS_T_STEPS] = steps;
  hdr[RVS_T_TOKEN] = token;
  hdr[RVS_T_STOP] = stop < 0 ? 0xFFFFFFFFu : (uint32_t)stop;
  hdr[RVS_T_DONE] = 0;
  hdr[RVS_T_STATUS] = 0;
  __sync_synchronize();
  uint32_t st = 0;
  sa_transport_t* t = s->c->transport;
  iree_status_t status = t->run_sched(t, s->table_phys, 10.0 + 10.0 * steps, &st);
  __sync_synchronize();
  uint32_t done = hdr[RVS_T_DONE];
  if (done > steps) done = steps;
  const volatile uint32_t* o = (const volatile uint32_t*)(s->table + s->off_out);
  for (uint32_t i = 0; i < done; ++i) {
    out_tokens[i] = o[RVS_O_WORDS_ * i + RVS_O_TOKEN];
    out_cycles[i] = o[RVS_O_WORDS_ * i + RVS_O_CYCLES];
    out_device[i] = o[RVS_O_WORDS_ * i + RVS_O_DEVICE];
  }
  *out_done = done;
  if (done) s->last_pos = pos + done - 1;
  if (iree_status_is_ok(status) && st) {
    status = iree_make_status(IREE_STATUS_INTERNAL, "sa rvsched: the core stopped after %u steps with status %#x%s",
                              done, st,
                              st == RVS_E_RANGE   ? " (a position outside the table)"
                              : st == RVS_E_MAGIC ? " (no table)"
                              : st == RVS_E_ARGMAX ? " (the R5 kernel wrote no block index)"
                                                  : " (an item failed)");
  }
  return status;
}

const float* sa_rvsched_logits(const sa_rvsched_t* s, uint32_t pos) {
  uint32_t b = pos / s->pad;
  if (b < s->b0 || b - s->b0 >= s->nb || s->pass != 3) return NULL;
  const uint32_t* bw = (const uint32_t*)(s->table + s->off_buckets) + RVS_B_WORDS_ * (b - s->b0);
  sa_transport_t* t = s->c->transport;
  return (const float*)(t->mem + (bw[RVS_B_LOGITS] - t->mem_phys));
}

void sa_rvsched_destroy(sa_rvsched_t* s) {
  if (!s) return;
  if (s->c->record == &s->rec) s->c->record = NULL;
  sa_record_free(&s->rec);
  sa_rvs_clear(s);
  free(s->buckets);
  free(s->item_code);
  free(s->item_words);
  free(s->off_copy);
  if (s->raw) iree_allocator_free(sa_context_data_allocator(s->c), s->raw);
  if (s->r5_raw) iree_allocator_free(sa_context_data_allocator(s->c), s->r5_raw);
  if (s->r5_bytes) sa_rv_set_host_kernel(0, NULL);
  free(s);
}

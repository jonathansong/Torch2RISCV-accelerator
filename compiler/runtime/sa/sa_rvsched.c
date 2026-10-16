// sa HAL driver: the step table of the decode loop on the RISC-V core (sa_rvsched.h).
#include "sa_rvsched.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../firmware/include/rvsched.h"  // the decode schedule shared with rt_fw

typedef struct sa_rvs_bucket_t {
  sa_record_t rec;
  sa_rvsched_io_t io;
  bool have;
} sa_rvs_bucket_t;

struct sa_rvsched_t {
  sa_context_t* c;
  uint32_t pad, b0, nb, max_steps, nblocks, block;
  int pass;                          // 1, 2; 3 after the second (record_end then checks)
  sa_rvs_bucket_t* buckets;          // this pass's recordings
  sa_record_t rec;                   // the recording in progress
  // from pass 1: the lists' shape and the room for the patched words
  uint32_t nlists, ndesc[SA_RECORD_MAX_LISTS], cap;
  // the table in the device window
  void* raw;
  uint8_t* table;
  uint32_t table_phys, table_bytes;
  uint32_t off_copy[SA_RECORD_MAX_LISTS], off_lists, off_words, off_buckets, off_values, off_out;
  uint32_t nwords;
  uint32_t last_pos;                 // position of the last step run
};

static void sa_record_free(sa_record_t* r) {
  for (uint32_t i = 0; i < r->count; ++i) free(r->lists[i]);
  memset(r, 0, sizeof(*r));
}

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
  for (uint32_t l = 0; l < a->count; ++l)
    if (a->descriptors[l] != b->descriptors[l] || memcmp(a->lists[l], b->lists[l], 64ull * a->descriptors[l]))
      return false;
  return true;
}

iree_status_t sa_rvsched_record_end(sa_rvsched_t* s, uint32_t bucket, const sa_rvsched_io_t* io) {
  s->c->record = NULL;
  if (bucket < s->b0 || bucket - s->b0 >= s->nb) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "sa rvsched: bucket %u outside the table", bucket);
  }
  if (s->rec.dropped || s->rec.count == 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "sa rvsched: %u lists in a step (at most %d)",
                            s->rec.count + s->rec.dropped, SA_RECORD_MAX_LISTS);
  }
  sa_rvs_bucket_t* e = &s->buckets[bucket - s->b0];
  if (s->pass == 3) {                // a check: the same lists as the bucket's recording
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

// The lists' 32-bit words that differ from bucket 0's in any bucket; returns
// their count, fills idx (list << 24 | word) up to cap.
static iree_status_t sa_rvs_diff(sa_rvsched_t* s, uint32_t* idx, uint32_t cap, uint32_t* out_n) {
  const sa_record_t* r0 = &s->buckets[0].rec;
  for (uint32_t b = 0; b < s->nb; ++b) {
    const sa_rvs_bucket_t* e = &s->buckets[b];
    if (!e->have) return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "sa rvsched: bucket %u not recorded", s->b0 + b);
    if (e->rec.count != r0->count) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "sa rvsched: bucket %u has %u lists, bucket %u %u",
                              s->b0 + b, e->rec.count, s->b0, r0->count);
    }
    for (uint32_t l = 0; l < r0->count; ++l)
      if (e->rec.descriptors[l] != r0->descriptors[l]) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "sa rvsched: list %u has %u descriptors in bucket %u, %u in bucket %u", l,
                                e->rec.descriptors[l], s->b0 + b, r0->descriptors[l], s->b0);
      }
  }
  uint32_t n = 0;
  for (uint32_t l = 0; l < r0->count; ++l) {
    const uint32_t* w0 = (const uint32_t*)r0->lists[l];
    uint32_t nw = r0->descriptors[l] * 16;
    if (nw >= (1u << 24)) return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "sa rvsched: list too long");
    for (uint32_t w = 0; w < nw; ++w) {
      for (uint32_t b = 1; b < s->nb; ++b) {
        if (((const uint32_t*)s->buckets[b].rec.lists[l])[w] != w0[w]) {
          if (idx && n < cap) idx[n] = l << 24 | w;
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

iree_status_t sa_rvsched_end_pass(sa_rvsched_t* s) {
  sa_context_t* c = s->c;
  if (s->pass == 1) {
    uint32_t n = 0;
    IREE_RETURN_IF_ERROR(sa_rvs_diff(s, NULL, 0, &n));
    const sa_record_t* r0 = &s->buckets[0].rec;
    s->nlists = r0->count;
    s->cap = n + n / 4 + 64;         // the second pass's addresses differ: room for a few more
    uint32_t o = sa_align64(RVS_T_WORDS_ * 4u);
    for (uint32_t l = 0; l < s->nlists; ++l) {
      s->ndesc[l] = r0->descriptors[l];
      s->off_copy[l] = o;
      o += 64u * s->ndesc[l];
    }
    s->off_lists = o;
    o = sa_align64(o + 4u * s->nlists);
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
    sa_rvs_clear(s);
    s->pass = 2;
    return iree_ok_status();
  }
  if (s->pass != 2) return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "sa rvsched: both passes done");
  const sa_record_t* r0 = &s->buckets[0].rec;
  if (r0->count != s->nlists) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "sa rvsched: %u lists in pass 2, %u in pass 1",
                            r0->count, s->nlists);
  }
  for (uint32_t l = 0; l < s->nlists; ++l)
    if (r0->descriptors[l] != s->ndesc[l]) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "sa rvsched: list %u changed between the passes", l);
    }
  uint32_t* idx = (uint32_t*)malloc(4ull * s->cap);
  if (!idx) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "sa rvsched");
  uint32_t n = 0;
  iree_status_t status = sa_rvs_diff(s, idx, s->cap, &n);
  if (iree_status_is_ok(status) && n > s->cap) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "sa rvsched: %u patched words, room for %u", n, s->cap);
  }
  if (!iree_status_is_ok(status)) {
    free(idx);
    return status;
  }
  const sa_rvsched_io_t* io0 = &s->buckets[0].io;
  for (uint32_t b = 1; b < s->nb; ++b)
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
  hdr[RVS_T_NLISTS] = s->nlists;
  hdr[RVS_T_LISTS] = P + s->off_lists;
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
  uint32_t* lists = (uint32_t*)(t + s->off_lists);
  for (uint32_t l = 0; l < s->nlists; ++l) {
    memcpy(t + s->off_copy[l], r0->lists[l], 64ull * s->ndesc[l]);
    lists[l] = P + s->off_copy[l];
  }
  uint32_t* words = (uint32_t*)(t + s->off_words);
  for (uint32_t i = 0; i < n; ++i) words[i] = P + s->off_copy[idx[i] >> 24] + 4u * (idx[i] & 0xFFFFFFu);
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
    for (uint32_t i = 0; i < n; ++i) v[i] = ((const uint32_t*)e->rec.lists[idx[i] >> 24])[idx[i] & 0xFFFFFFu];
  }
  free(idx);
  __sync_synchronize();
  s->pass = 3;
  return iree_ok_status();
}

void sa_rvsched_info(const sa_rvsched_t* s, sa_rvsched_info_t* out) {
  memset(out, 0, sizeof(*out));
  out->lists = s->nlists;
  for (uint32_t l = 0; l < s->nlists; ++l) out->descriptors += s->ndesc[l];
  out->words = s->nwords;
  out->table_bytes = s->table_bytes;
  out->table_phys = s->table_phys;
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
  iree_status_t status = t->run_sched(t, s->table_phys, 10.0 + 2.0 * steps, &st);
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
                                                  : " (a list failed)");
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
  if (s->raw) iree_allocator_free(sa_context_data_allocator(s->c), s->raw);
  free(s);
}

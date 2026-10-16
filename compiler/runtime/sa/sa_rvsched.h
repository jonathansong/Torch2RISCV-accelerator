// sa HAL driver: the decode loop on the device's RISC-V core (plan B,
// docs/riscv_backend_design.md §7.1; the table and the core's loop:
// firmware/include/rvsched.h).
//
// The host records one decode step per attention-length bucket in the
// driver's dry mode (the lists are built, not run) and this builder turns the
// recordings into the step table: bucket 0's lists copied once, the words that
// differ between buckets with their value in each bucket, each bucket's input
// and output addresses. Two passes, because the table lives in the device
// window too: the first sizes it, the table is allocated, the second records
// the addresses the steps will use with the table in place.
//   sa_rvsched_create(context, pad, bucket0, nbuckets, max_steps, &s)
//   for pass 1, 2: for each bucket: sa_rvsched_record_begin(s);
//                  <the host runs the step and the block maxima dry>;
//                  sa_rvsched_record_end(s, bucket, io);
//                  sa_rvsched_end_pass(s)
//   sa_rvsched_run(s, ...)            the core runs the steps
//   sa_rvsched_destroy(s)
#ifndef SA_RVSCHED_H_
#define SA_RVSCHED_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "sa_context.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sa_rvsched_t sa_rvsched_t;

// The physical addresses of one recorded step's inputs and outputs.
typedef struct sa_rvsched_io_t {
  uint32_t token, pos, valid;   // i64 [1], i64 [1], f32 [valid length]
  uint32_t logits, bmax;        // f32 [vocab], f32 [nblocks] (the block maxima)
  uint32_t vocab, nblocks;
} sa_rvsched_io_t;

// Statistics of the table (after pass 2).
typedef struct sa_rvsched_info_t {
  uint32_t lists, descriptors;  // lists per step, their descriptors
  uint32_t calls;               // calls of generated RISC-V code per step (SA_RV: plan A inside plan B)
  uint32_t words;               // list words patched between buckets
  uint32_t table_bytes;
  uint32_t table_phys;
  uint32_t r5_bytes;            // R5 (SA_R5): the block argmax kernel's rv32 code, 0: the core's own scan
  const char* r5_symbol;
} sa_rvsched_info_t;

iree_status_t sa_rvsched_create(sa_context_t* context, uint32_t pad, uint32_t bucket0, uint32_t nbuckets,
                                uint32_t max_steps, sa_rvsched_t** out);
// Starts recording a step (dry: its lists are built, not run).
void sa_rvsched_record_begin(sa_rvsched_t* s);
// Ends it: the lists recorded since begin are bucket |bucket|'s.
iree_status_t sa_rvsched_record_end(sa_rvsched_t* s, uint32_t bucket, const sa_rvsched_io_t* io);
// Ends a pass: after the first the table is allocated, after the second written.
iree_status_t sa_rvsched_end_pass(sa_rvsched_t* s);
void sa_rvsched_info(const sa_rvsched_t* s, sa_rvsched_info_t* out);
// Runs |steps| steps from |pos| with input |token| on the core (stopping after
// generating |stop|, -1: none); out_tokens[steps], out_cycles[steps] (the
// step's core cycles) and out_device[steps] (its lists' cycles) are filled for
// the steps done (*out_done).
iree_status_t sa_rvsched_run(sa_rvsched_t* s, uint32_t pos, uint32_t steps, uint32_t token, int32_t stop,
                             uint32_t* out_tokens, uint32_t* out_cycles, uint32_t* out_device, uint32_t* out_done);
// The logits of the last step run (f32 [vocab]) in the device window.
const float* sa_rvsched_logits(const sa_rvsched_t* s, uint32_t pos);
void sa_rvsched_destroy(sa_rvsched_t* s);

#ifdef __cplusplus
}
#endif

#endif  // SA_RVSCHED_H_

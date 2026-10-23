/* Device-side MoE top-1 router for PicoRV32.
 *
 * ABI: args[0] is f32 scores[E], args[1] is the 32-byte sa_moe_route v1
 * record; constants[0] is E and constants[1] is expert_stride. Ties keep the
 * first expert. NaNs do not replace the current maximum.
 */
#include <stdint.h>

#ifdef MOE_TOP1_HOST
static void* (*host_ptr)(uint32_t);
void moe_top1_host_init(void* (*fn)(uint32_t)) { host_ptr = fn; }
#define PTR(a) host_ptr(a)
#else
#define PTR(a) ((void*)(uintptr_t)(a))
#endif

void r5_entry(const uint32_t* args, const uint32_t* constants) {
  const float* scores = (const float*)PTR(args[0]);
  uint32_t* route = (uint32_t*)PTR(args[1]);
  uint32_t n = constants[0];
  uint32_t stride = constants[1];
  union {
    uint32_t u;
    float f;
  } best = {0xFF800000u};
  uint32_t best_id = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (scores[i] > best.f) {
      best.f = scores[i];
      best_id = i;
    }
  }
  for (uint32_t i = 0; i < 8; ++i) route[i] = 0;
  route[0] = 1;                         /* count */
  route[1] = best_id;                   /* expert_id[0] */
  route[3] = best_id * stride;          /* expert_offset[0] */
  route[5] = 0x3F800000u;               /* weight[0] = 1.0f */
}

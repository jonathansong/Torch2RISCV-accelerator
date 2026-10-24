/* Raw batch-1 Top-2 for an IREE dispatch. The input scores must be finite.
 * Equal scores keep the lower expert ID. This is deliberately separate from
 * moe_top2.c, which produces normalized weights and parameter offsets. */
#include <stdint.h>

#ifdef MOE_TOP2_RAW_HOST
static void* (*host_ptr)(uint32_t);
void moe_top2_raw_host_init(void* (*fn)(uint32_t)) { host_ptr = fn; }
#define PTR(a) host_ptr(a)
#else
#define PTR(a) ((void*)(uintptr_t)(a))
#endif

void r5_entry(const uint32_t* args, const uint32_t* constants) {
  const float* scores = (const float*)PTR(args[0]);
  float* values = (float*)PTR(args[1]);
  uint32_t* indices = (uint32_t*)PTR(args[2]);
  const uint32_t n = constants[0];
  if (n < 2) return;

  float first = scores[0], second = scores[1];
  uint32_t first_id = 0, second_id = 1;
  if (second > first) {
    float tmp = first;
    first = second;
    second = tmp;
    first_id = 1;
    second_id = 0;
  }
  for (uint32_t i = 2; i < n; ++i) {
    const float value = scores[i];
    if (value > first) {
      second = first;
      second_id = first_id;
      first = value;
      first_id = i;
    } else if (value > second) {
      second = value;
      second_id = i;
    }
  }
  values[0] = first;
  values[1] = second;
  indices[0] = first_id;
  indices[1] = second_id;
}

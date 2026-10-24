/* Raw batch-1 Top-2 for an IREE dispatch. Match TopkOp's scalar insertion
 * with output values initialized to -inf and indices to INT32_MAX. In
 * particular, NaNs make both ordered comparisons false; value and ID may
 * then be selected independently. This is separate from moe_top2.c, which
 * produces normalized weights and parameter offsets. */
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

  union { uint32_t bits; float value; } neg_inf = {0xFF800000u};
  float selected[2] = {neg_inf.value, neg_inf.value};
  uint32_t selected_id[2] = {0x7FFFFFFFu, 0x7FFFFFFFu};
  for (uint32_t i = 0; i < n; ++i) {
    float carry = scores[i];
    uint32_t carry_id = i;
    for (uint32_t k = 0; k < 2; ++k) {
      const float old = selected[k];
      const uint32_t old_id = selected_id[k];
      const int forward = carry > old;
      const int reverse = old > carry;
      const int take_id = forward || (forward == reverse && (int32_t)carry_id < (int32_t)old_id);
      selected[k] = forward ? carry : old;
      selected_id[k] = take_id ? carry_id : old_id;
      carry = forward ? old : carry;
      carry_id = take_id ? old_id : carry_id;
    }
  }
  values[0] = selected[0];
  values[1] = selected[1];
  indices[0] = selected_id[0];
  indices[1] = selected_id[1];
}

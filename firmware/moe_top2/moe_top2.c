/* Device-side MoE top-2 router. Writes sa_moe_route v1 with a softmax over
 * the two selected scores. Ties keep lower expert IDs; NaNs are ignored. */
#include <stdint.h>

#ifdef MOE_TOP2_HOST
static void* (*host_ptr)(uint32_t);
void moe_top2_host_init(void* (*fn)(uint32_t)) { host_ptr = fn; }
#define PTR(a) host_ptr(a)
#else
#define PTR(a) ((void*)(uintptr_t)(a))
#endif

static float exp_approx(float x) {
  if (x < -16.0f) return 0.0f;
  float z = x * 1.4426950409f;
  int n = (int)z;
  if ((float)n > z) --n;
  float r = x - (float)n * 0.6931471806f;
  float r2 = r * r;
  float p = 1.0f + r + r2 * (0.5f + r * (0.1666666716f + r * (0.0416666679f + r * 0.0083333338f)));
  union { uint32_t u; float f; } scale = {(uint32_t)(n + 127) << 23};
  return p * scale.f;
}

void r5_entry(const uint32_t* args, const uint32_t* constants) {
  const float* scores = (const float*)PTR(args[0]);
  uint32_t* route = (uint32_t*)PTR(args[1]);
  uint32_t n = constants[0], stride = constants[1];
  union { uint32_t u; float f; } first = {0xFF800000u}, second = {0xFF800000u};
  uint32_t first_id = 0, second_id = n > 1 ? 1 : 0, found = 0;
  for (uint32_t i = 0; i < n; ++i) {
    float value = scores[i];
    if (value > first.f) {
      second.f = first.f; second_id = first_id;
      first.f = value; first_id = i;
      if (found < 2) ++found;
    } else if (value > second.f) {
      second.f = value; second_id = i;
      if (found < 2) ++found;
    }
  }
  float w0 = 1.0f, w1 = 0.0f;
  if (found >= 2) {
    float e = exp_approx(second.f - first.f);
    w0 = 1.0f / (1.0f + e);
    w1 = e / (1.0f + e);
  } else if (found == 0) {
    first_id = 0; second_id = n > 1 ? 1 : 0;
    w0 = w1 = 0.5f;
  }
  union { float f; uint32_t u; } b0 = {w0}, b1 = {w1};
  route[0] = 2;
  route[1] = first_id; route[2] = second_id;
  route[3] = first_id * stride; route[4] = second_id * stride;
  route[5] = b0.u; route[6] = b1.u;
  route[7] = 0;
}

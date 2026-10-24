/* Converts a native IREE Top-2 result to the 32-byte sa_moe_route v1 ABI.
 * Bindings: selected f32 values, selected i32 IDs, route record.
 * Push constants: expert count, expert parameter stride in bytes.
 * Inputs must be finite router scores with distinct IDs in [0, count).
 */
#include <stdint.h>

#ifdef MOE_ROUTE_FROM_TOP2_HOST
static void* (*host_ptr)(uint32_t);
void moe_route_from_top2_host_init(void* (*fn)(uint32_t)) { host_ptr = fn; }
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
  const float* values = (const float*)PTR(args[0]);
  const uint32_t* ids = (const uint32_t*)PTR(args[1]);
  uint32_t* route = (uint32_t*)PTR(args[2]);
  const uint32_t count = constants[0], stride = constants[1];
  const uint32_t first = ids[0], second = ids[1];
  union { float f; uint32_t u; } first_value = {values[0]}, second_value = {values[1]};
  union { float f; uint32_t u; } b0, b1;

  /* A zero count is an invalid record, so a caller must not dispatch the FFN. */
  if (count < 2 || first >= count || second >= count || first == second ||
      stride == 0 || (uint64_t)(count - 1) * stride > UINT32_MAX ||
      (first_value.u & 0x7F800000u) == 0x7F800000u ||
      (second_value.u & 0x7F800000u) == 0x7F800000u ||
      values[0] < values[1]) {
    route[0] = 0;
    return;
  }
  float e = exp_approx(values[1] - values[0]);
  b0.f = 1.0f / (1.0f + e);
  b1.f = e / (1.0f + e);
  route[0] = 2;
  route[1] = first; route[2] = second;
  route[3] = first * stride; route[4] = second * stride;
  route[5] = b0.u; route[6] = b1.u;
  route[7] = 0;
}

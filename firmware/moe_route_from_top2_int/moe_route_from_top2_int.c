/* Integer-only Top-2 to sa_moe_route v1 bridge for an embedded rv32imc export.
 * Bindings: selected f32 values, selected i32 IDs, 32-byte route record.
 * Push constants: expert count and packed parameter stride in bytes.
 * Finite scores with magnitude below 512 are supported. An invalid input
 * writes count zero; the caller must reject that record before launching FFN.
 */
#include <stdint.h>

#ifdef MOE_ROUTE_INT_HOST
static void* (*host_ptr)(uint32_t);
void moe_route_int_host_init(void* (*fn)(uint32_t)) { host_ptr = fn; }
#define PTR(a) host_ptr(a)
#else
#define PTR(a) ((void*)(uintptr_t)(a))
#endif

static int32_t score_q16(uint32_t bits) {
  uint32_t exp = (bits >> 23) & 255u;
  uint32_t mantissa = (bits & 0x7FFFFFu) | 0x800000u;
  if (exp == 0) return 0;
  int shift = (int)exp - 134;
  uint32_t magnitude = shift >= 0 ? mantissa << shift :
                       shift > -24 ? mantissa >> -shift : 0;
  return bits >> 31 ? -(int32_t)magnitude : (int32_t)magnitude;
}

/* e^x ~= (1 + x/32768)^32768; fifteen Q31 squarings. The host differential
 * corpus measures <5e-5 absolute error in the resulting softmax weights. */
static uint32_t exp_q31(int32_t x_q16) {
  if (x_q16 <= -16 * 65536) return 0;
  uint32_t value = (uint32_t)((1u << 31) + x_q16);
  for (int i = 0; i < 15; ++i)
    value = (uint32_t)(((uint64_t)value * value + (1u << 30)) >> 31);
  return value;
}

static uint32_t q15_to_float_bits(uint32_t value) {
  if (!value) return 0;
  uint32_t p = 31u - (uint32_t)__builtin_clz(value);
  return ((p + 112u) << 23) | ((value << (23u - p)) & 0x7FFFFFu);
}

void r5_entry(const uint32_t* args, const uint32_t* constants) {
  const uint32_t* values = (const uint32_t*)PTR(args[0]);
  const uint32_t* ids = (const uint32_t*)PTR(args[1]);
  uint32_t* route = (uint32_t*)PTR(args[2]);
  uint32_t count = constants[0], stride = constants[1];
  uint32_t first = ids[0], second = ids[1];
  uint32_t first_exp = (values[0] >> 23) & 255u;
  uint32_t second_exp = (values[1] >> 23) & 255u;
  if (count < 2 || first >= count || second >= count || first == second ||
      stride == 0 || stride > UINT32_MAX / (count - 1) ||
      first_exp == 255 || second_exp == 255 ||
      first_exp > 135 || second_exp > 135) {
    route[0] = 0;
    return;
  }
  int32_t first_q16 = score_q16(values[0]);
  int32_t second_q16 = score_q16(values[1]);
  if (first_q16 < second_q16) {
    route[0] = 0;
    return;
  }
  uint32_t e_q31 = exp_q31(second_q16 - first_q16);
  uint32_t e_q15 = (e_q31 + (1u << 15)) >> 16;
  uint32_t w1_q15 = (e_q15 << 15) / ((1u << 15) + e_q15);
  route[0] = 2;
  route[1] = first; route[2] = second;
  route[3] = first * stride; route[4] = second * stride;
  route[5] = q15_to_float_bits((1u << 15) - w1_q15);
  route[6] = q15_to_float_bits(w1_q15);
  route[7] = 0;
}

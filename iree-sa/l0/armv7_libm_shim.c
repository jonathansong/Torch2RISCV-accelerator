/* Symbols that LLVM's ARMv7 code references but IREE's embedded ELF
 * executables do not provide (the platform-agnostic loader rejects any
 * import). Linked into every armv7 embedded executable by embedded_ld.sh.
 * Hidden visibility: calls bind at link time (no PLT, no dynamic symbol).
 * Freestanding, position-independent, hard-float ABI. */
#define HIDDEN __attribute__((visibility("hidden")))

/* Cortex-A9 (VFPv3 + NEON) has no IEEE maxNum/minNum instruction, so LLVM
 * calls these for max/min (softmax, ReLU, ...). IEEE 754 maxNum/minNum: a NaN
 * operand yields the other operand; -0 < +0. */
HIDDEN float fmaxf(float a, float b) { return a != a ? b : b != b ? a : a > b ? a : a < b ? b : __builtin_signbit(a) ? b : a; }
HIDDEN float fminf(float a, float b) { return a != a ? b : b != b ? a : a < b ? a : a > b ? b : __builtin_signbit(a) ? a : b; }
HIDDEN double fmax(double a, double b) { return a != a ? b : b != b ? a : a > b ? a : a < b ? b : __builtin_signbit(a) ? b : a; }
HIDDEN double fmin(double a, double b) { return a != a ? b : b != b ? a : a < b ? a : a > b ? b : __builtin_signbit(a) ? a : b; }

/* Correctly rounded fused multiply-add (round to nearest even). LLVM calls
 * fmaf for llvm.fma (IREE's exp/tanh/... polynomial approximations) on
 * targets without VFPv4, and the fmaf in IREE's bundled musl bitcode is an
 * empty `unreachable` body; embedded_ld.sh makes that one weak so this one
 * is used. musl's generic algorithm without the fenv parts: the double
 * product is exact, one double addition, then fix the rare double-rounding
 * case where the double sum lies exactly halfway between two floats. */
HIDDEN float fmaf(float x, float y, float z)
{
    union { double f; unsigned long long i; } u;
    double xy = (double)x * y, result = xy + z;
    u.f = result;
    int e = (int)(u.i >> 52 & 0x7ff);
    if ((u.i & 0x1fffffff) != 0x10000000 || e == 0x7ff || (result - xy == z && result - z == xy))
        return (float)result;
    int neg = (int)(u.i >> 63);
    double err = neg == (z > xy) ? xy - result + z : z - result + xy;
    if (neg == (err < 0))
        u.i++;
    else
        u.i--;
    return (float)u.f;
}

/* ARM EHABI personality routines, referenced by .ARM.exidx unwind entries.
 * Executables never throw or unwind; trap if one ever does. */
HIDDEN void __aeabi_unwind_cpp_pr0(void) { __builtin_trap(); }
HIDDEN void __aeabi_unwind_cpp_pr1(void) { __builtin_trap(); }
HIDDEN void __aeabi_unwind_cpp_pr2(void) { __builtin_trap(); }

/* ARM EABI integer division helpers. The Cortex-A9 (ARMv7-A without the
 * integer divide extension) has no SDIV / UDIV, so LLVM calls these for every
 * integer division, e.g. index arithmetic on dynamic shapes (found in stage
 * C0: the llama's attention length T is dynamic). Binary long division; a
 * division by zero returns 0 (the dispatches never divide by zero).
 * *divmod return {quotient, remainder} in r0:r1, i.e. as the low / high
 * halves of a 64-bit value. */
static unsigned udiv32(unsigned n, unsigned d, unsigned *rem)
{
    unsigned q = 0, r = 0;
    if (d == 0) {
        *rem = 0;
        return 0;
    }
    for (int i = 31; i >= 0; i--) {
        r = (r << 1) | ((n >> i) & 1u);
        if (r >= d) {
            r -= d;
            q |= 1u << i;
        }
    }
    *rem = r;
    return q;
}

HIDDEN unsigned __aeabi_uidiv(unsigned n, unsigned d)
{
    unsigned r;
    return udiv32(n, d, &r);
}

HIDDEN unsigned long long __aeabi_uidivmod(unsigned n, unsigned d)
{
    unsigned r, q = udiv32(n, d, &r);
    return (unsigned long long)r << 32 | q;
}

HIDDEN int __aeabi_idiv(int n, int d)
{
    unsigned r, q = udiv32(n < 0 ? 0u - (unsigned)n : (unsigned)n, d < 0 ? 0u - (unsigned)d : (unsigned)d, &r);
    return (n < 0) != (d < 0) ? (int)(0u - q) : (int)q;          /* truncation toward zero */
}

HIDDEN unsigned long long __aeabi_idivmod(int n, int d)
{
    unsigned r, q = udiv32(n < 0 ? 0u - (unsigned)n : (unsigned)n, d < 0 ? 0u - (unsigned)d : (unsigned)d, &r);
    int qs = (n < 0) != (d < 0) ? (int)(0u - q) : (int)q;
    int rs = n < 0 ? (int)(0u - r) : (int)r;                        /* the remainder has the dividend's sign */
    return (unsigned long long)(unsigned)rs << 32 | (unsigned)qs;
}

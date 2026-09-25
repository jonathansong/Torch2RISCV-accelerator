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

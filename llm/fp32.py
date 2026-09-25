"""Bit-exact model of the L2 fp32 arithmetic (docs/llm_inference_plan.md §3.2, §6.5).

Every function takes and returns fp32 bit patterns (uint32 NumPy arrays) and
computes exactly what the RTL computes (rtl/sysarray/sa_fp32_*.v, the SFU
microcode), with integer mantissa arithmetic:

- IEEE-754 binary32, round to nearest even;
- flush to zero: a subnormal input is a zero of the same sign; a result
  whose exponent, after rounding with an unbounded exponent range, is below
  the normal range becomes a zero of the same sign;
- overflow gives +-inf; any NaN input gives the canonical NaN 0x7FC00000;
  inf - inf and 0 * inf give it too;
- x + (-x) = +0 (round to nearest), -0 + -0 = -0.

On inputs and results in the normal range this is IEEE arithmetic, i.e.
NumPy float32 (checked by llm/test_fp32.py). The special functions are
fixed sequences of these operations plus table lookups (llm/sfu_tables.py),
so they are bit-exact as well; their accuracy against float64 is measured
separately.
"""
import numpy as np

QNAN = np.uint32(0x7FC00000)
PINF, NINF = np.uint32(0x7F800000), np.uint32(0xFF800000)
ONE = np.uint32(0x3F800000)
_i64 = np.int64


def u32(x):
    return np.asarray(x, dtype=np.uint32)


def from_f(x):
    """float values -> fp32 bits (NumPy rounding; for constants and tests)."""
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def to_f(b):
    return u32(b).view(np.float32)


def _unpack(b):
    b = u32(b).astype(_i64)
    s = b >> 31
    e = (b >> 23) & 0xFF
    m = b & 0x7FFFFF
    nan = (e == 255) & (m != 0)
    inf = (e == 255) & (m == 0)
    zero = e == 0                                   # FTZ: subnormals are zeros
    mant = np.where(zero, 0, m | 0x800000)          # 24-bit significand
    return s, e, mant, nan, inf, zero


def _pack(s, e, mant):
    """s, biased exponent (unbounded), 24-bit significand [2^23, 2^24) -> bits
    with FTZ (e < 1) and overflow (e > 254)."""
    out = (s << 31) | (np.clip(e, 0, 255) << 23) | (mant & 0x7FFFFF)
    out = np.where(e < 1, s << 31, out)
    out = np.where(e > 254, (s << 31) | (255 << 23), out)
    return out.astype(np.uint32)


def _round_shift(v, sh):
    """v >> sh rounded to nearest even (v >= 0, sh >= 0, elementwise)."""
    sh = np.asarray(sh, _i64)
    q = v >> sh
    rem = v - (q << sh)
    half = np.where(sh > 0, _i64(1) << np.maximum(sh - 1, 0), 0)
    up = (sh > 0) & ((rem > half) | ((rem == half) & ((q & 1) == 1)))
    return q + up


def _nlz28(v):
    """position of the top set bit of v (v > 0, < 2^62)."""
    return np.floor(np.log2(np.maximum(v, 1).astype(np.float64))).astype(_i64)


def add(a, b):
    sa, ea, ma, na, ia, za = _unpack(a)
    sb, eb, mb, nb, ib, zb = _unpack(b)
    # order by magnitude: |x| >= |y|
    swap = (eb > ea) | ((eb == ea) & (mb > ma))
    sx, ex, mx = np.where(swap, sb, sa), np.where(swap, eb, ea), np.where(swap, mb, ma)
    sy, ey, my = np.where(swap, sa, sb), np.where(swap, ea, eb), np.where(swap, ma, mb)
    d = np.minimum(ex - ey, 60)
    X = mx << 3                                     # 27 bits: significand + G R S
    Yfull = my << 3
    Y = Yfull >> d
    sticky = (Y << d) != Yfull
    Y = Y | sticky
    same = sx == sy
    S = np.where(same, X + Y, X - Y)
    # normalize to [2^26, 2^27)
    top = _nlz28(S)
    e = ex + (top - 26)
    rshift = np.maximum(top - 26, 0)
    lshift = np.maximum(26 - top, 0)
    lost = (S & ((_i64(1) << rshift) - 1)) != 0
    S2 = np.where(top > 26, (S >> rshift) | lost, S << lshift)
    # round G R S away (3 bits)
    mant = _round_shift(S2, 3)
    ovf = mant >= (1 << 24)
    mant = np.where(ovf, mant >> 1, mant)
    e = e + ovf
    out = _pack(sx, e, mant)
    out = np.where(S == 0, np.uint32(0), out)                       # exact cancellation: +0
    # zero operands (after FTZ): the other operand; -0 + -0 = -0
    bz = _pack(sa, ea, ma)
    az = _pack(sb, eb, mb)
    out = np.where(zb & ~za, bz, out)
    out = np.where(za & ~zb, az, out)
    out = np.where(za & zb, ((sa & sb) << 31).astype(np.uint32), out)
    # inf, NaN
    out = np.where(ia & ~ib, u32(a) & 0xFF800000, out)
    out = np.where(ib & ~ia, u32(b) & 0xFF800000, out)
    out = np.where(ia & ib, np.where(sa == sb, u32(a), QNAN), out)
    out = np.where(na | nb, QNAN, out)
    return u32(out)


def neg(a):
    return u32(a) ^ np.uint32(0x80000000)


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    sa, ea, ma, na, ia, za = _unpack(a)
    sb, eb, mb, nb, ib, zb = _unpack(b)
    s = sa ^ sb
    P = ma * mb                                     # [2^46, 2^48)
    hi = P >= (_i64(1) << 47)
    mant = _round_shift(P, np.where(hi, 24, 23))
    e = ea + eb - 127 + hi
    ovf = mant >= (1 << 24)
    mant = np.where(ovf, mant >> 1, mant)
    e = e + ovf
    out = _pack(s, e, mant)
    out = np.where(za | zb, (s << 31).astype(np.uint32), out)
    out = np.where(ia | ib, np.where(za | zb, QNAN, ((s << 31) | (255 << 23)).astype(np.uint32)), out)
    out = np.where(na | nb, QNAN, out)
    return u32(out)


def ldexp(a, n):
    """a * 2^n exactly (exponent add, FTZ / overflow; a normal, zero, inf or NaN)."""
    s, e, mant, na, ia, za = _unpack(a)
    out = _pack(s, e + np.asarray(n, _i64), mant)
    return u32(np.where(za | ia | na, np.where(na, QNAN, u32(a) & np.where(za, 0x80000000, 0xFFFFFFFF)), out))


def from_int(i):
    """int32 -> fp32, round to nearest even."""
    i = np.asarray(i, _i64)
    s = (i < 0).astype(_i64)
    v = np.abs(i)
    top = _nlz28(v)
    sh = np.maximum(top - 23, 0)
    mant = np.where(top > 23, _round_shift(v, sh), v << np.maximum(23 - top, 0))
    e = 127 + top
    ovf = mant >= (1 << 24)
    mant = np.where(ovf, mant >> 1, mant)
    e = e + ovf
    return u32(np.where(v == 0, np.uint32(0), _pack(s, e, mant)))


def to_int(a, lo, hi, mode="rne"):
    """fp32 -> integer: round to nearest even (mode "rne") or floor, then
    saturate to [lo, hi]; NaN -> 0."""
    s, e, mant, na, ia, za = _unpack(a)
    ex = e - 150                                    # value = mant * 2^ex
    big = ex >= 8                                   # |v| >= 2^31: saturates anyway
    left = np.where(ex >= 0, mant << np.clip(ex, 0, 7), 0)
    sh = np.clip(-ex, 0, 62)
    if mode == "rne":
        right = _round_shift(mant, sh)
        mag = np.where(ex >= 0, left, right)
        v = np.where(s == 1, -mag, mag)
    else:                                           # floor
        q = mant >> sh
        exact = (q << sh) == mant
        mag = np.where(ex >= 0, left, q)
        v = np.where(s == 1, -mag - np.where((ex < 0) & ~exact, 1, 0), mag)
    v = np.where(big | ia, np.where(s == 1, lo, hi), v)
    v = np.where(za, 0, v)
    v = np.clip(v, lo, hi)
    return np.where(na, 0, v).astype(_i64)


def fmax(a, b):
    """NaN if either is NaN; equal values (incl. +-0): +0 wins."""
    fa, fb = to_f(a).astype(np.float64), to_f(b).astype(np.float64)
    out = np.where(fa > fb, u32(a), np.where(fb > fa, u32(b), u32(a) & u32(b)))
    return u32(np.where(np.isnan(fa) | np.isnan(fb), QNAN, out))


def fmin(a, b):
    """NaN if either is NaN; equal values (incl. +-0): -0 wins."""
    fa, fb = to_f(a).astype(np.float64), to_f(b).astype(np.float64)
    out = np.where(fa < fb, u32(a), np.where(fb < fa, u32(b), u32(a) | u32(b)))
    return u32(np.where(np.isnan(fa) | np.isnan(fb), QNAN, out))


def ftz(a):
    """subnormal -> signed zero (what every unit does to its inputs)."""
    a = u32(a)
    return u32(np.where(((a >> 23) & 0xFF) == 0, a & 0x80000000, a))

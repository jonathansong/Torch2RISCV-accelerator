"""Special functions of the L2 vector engine (docs/llm_inference_plan.md §6.5),
bit-exact: fixed sequences of the fp32 operations of llm/fp32.py (the lane's
multiplier and adder) plus table lookups and exponent / field operations
(integer logic in the SFU). The RTL microcode (rtl/sysarray/sa_sfu_seq.v)
runs the same steps in the same order.

EXP(x)    x >= 89 -> +inf, x <= -104 -> +0, NaN -> NaN; otherwise
          t = x * log2(e); n = floor(t); f = t - n              (exact)
          u = f * 64; i = floor(u); r = u - i                    (exact)
          y = T_EXP[i] + r * S_EXP[i]                            (2^f)
          result = y * 2^n                                       (exponent add, FTZ)
RECIP(x)  +-0 -> +-inf, +-inf -> +-0, NaN -> NaN; x = +-m * 2^E, m in [1, 2):
          i = top 6 fraction bits, r = remaining 17 bits / 2^17
          y0 = T_REC[i] + r * S_REC[i]; y1 = y0 * (2 - m * y0)   (one Newton step)
          result = +-y1 * 2^-E
RSQRT(x)  NaN / x < 0 -> NaN, +-0 -> +inf, +inf -> +0; x = m * 2^E:
          E odd: m' = 2m, E' = E - 1 (m' in [1, 4)); j = 64 * (E odd) + top 6 fraction bits
          y0 = T_RSQ[j] + r * S_RSQ[j]
          y1 = y0 * (1.5 - (m' * (y0 * y0)) * 0.5)               (one Newton step)
          result = y1 * 2^(-E'/2)
ABS(x)    clear the sign bit.

Tables: fp32 values T (segment start) and S (segment slope), 64 entries for
EXP and RECIP, 128 for RSQRT; tables() returns them as uint32 bit patterns
(the same values go into the RTL, rtl/sysarray/sa_sfu_tables.vh).
"""
import functools

import numpy as np

import fp32
from fp32 import add, from_f, from_int, ldexp, mul, sub, to_int, u32

LOG2E = from_f(np.float32(1.4426950408889634))
C64 = from_f(np.float32(64.0))
C2 = from_f(np.float32(2.0))
C1_5 = from_f(np.float32(1.5))
C0_5 = from_f(np.float32(0.5))
R17 = 2.0 ** -17


@functools.lru_cache()
def tables():
    """{'exp': (T, S), 'recip': (T, S), 'rsqrt': (T, S)} as uint32 arrays."""
    k = np.arange(64, dtype=np.float64)
    t_exp = 2.0 ** (k / 64)
    s_exp = 2.0 ** ((k + 1) / 64) - t_exp
    m = 1 + k / 64
    t_rec = 1 / m
    s_rec = 1 / (m + 1 / 64) - t_rec
    m2 = np.concatenate([1 + k / 64, 2 * (1 + k / 64)])
    step = np.concatenate([np.full(64, 1 / 64), np.full(64, 2 / 64)])
    t_rsq = 1 / np.sqrt(m2)
    s_rsq = 1 / np.sqrt(m2 + step) - t_rsq
    return {name: (from_f(t.astype(np.float32)), from_f(s.astype(np.float32)))
            for name, (t, s) in {"exp": (t_exp, s_exp), "recip": (t_rec, s_rec), "rsqrt": (t_rsq, s_rsq)}.items()}


def _parts(x):
    x = u32(x).astype(np.int64)
    return x >> 31, (x >> 23) & 0xFF, x & 0x7FFFFF


def _r17(frac):
    """(fraction & 0x1FFFF) / 2^17 as fp32 (exact)."""
    return ldexp(from_int(frac & 0x1FFFF), -17)


def exp(x):
    x = fp32.ftz(x)
    T, S = tables()["exp"]
    xf = fp32.to_f(x).astype(np.float64)
    t = mul(x, LOG2E)
    n = to_int(t, -2**31, 2**31 - 1, "floor")
    f = sub(t, from_int(n))
    u = mul(f, C64)
    i = np.clip(to_int(u, 0, 63, "floor"), 0, 63)
    r = sub(u, from_int(i))
    y = add(T[i], mul(r, S[i]))
    res = ldexp(y, np.clip(n, -300, 300))
    res = np.where(xf >= 89, fp32.PINF, np.where(xf <= -104, np.uint32(0), res))
    return u32(np.where(np.isnan(xf), fp32.QNAN, res))


def recip(x):
    x = fp32.ftz(x)
    T, S = tables()["recip"]
    s, e, frac = _parts(x)
    i = frac >> 17
    m = u32((frac | 0x3F800000))
    y0 = add(T[i], mul(_r17(frac), S[i]))
    y1 = mul(y0, sub(C2, mul(m, y0)))
    res = ldexp(y1, 127 - e) | u32(s << 31)
    res = np.where(e == 0, u32((s << 31) | 0x7F800000), res)            # +-0 -> +-inf
    res = np.where((e == 255) & (frac == 0), u32(s << 31), res)          # +-inf -> +-0
    return u32(np.where((e == 255) & (frac != 0), fp32.QNAN, res))


def rsqrt(x):
    x = fp32.ftz(x)
    T, S = tables()["rsqrt"]
    s, e, frac = _parts(x)
    E = e - 127
    odd = (E & 1).astype(np.int64)
    j = odd * 64 + (frac >> 17)
    m2 = u32(frac | ((127 + odd) << 23))                                 # m or 2m
    y0 = add(T[j], mul(_r17(frac), S[j]))
    d = sub(C1_5, mul(mul(m2, mul(y0, y0)), C0_5))
    y1 = mul(y0, d)
    res = ldexp(y1, -((E - odd) // 2))
    res = np.where(e == 0, fp32.PINF, res)                               # +-0 -> +inf
    res = np.where((s == 1) & (e != 0), fp32.QNAN, res)                  # x < 0 (incl. -inf)
    res = np.where((s == 0) & (e == 255) & (frac == 0), np.uint32(0), res)   # +inf -> +0
    return u32(np.where((e == 255) & (frac != 0), fp32.QNAN, res))


def fabs(x):
    return u32(x) & np.uint32(0x7FFFFFFF)

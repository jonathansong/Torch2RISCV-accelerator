#!/usr/bin/env python3
"""Checks of the bit-exact fp32 model (llm/fp32.py) and the SFU (llm/sfu.py).

1. add / sub / mul against NumPy float32 (IEEE) on random operands over
   wide exponent ranges, near-cancellations and special values, wherever
   IEEE and flush-to-zero agree (results in the normal range, infinities);
   FTZ cases against the rule itself.
2. int32 -> fp32 and fp32 -> int (round to nearest even / floor, saturation).
3. EXP / RECIP / RSQRT accuracy against float64 over their input ranges,
   and their special cases.

    python3 llm/test_fp32.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp32  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        fails.append(name)


def rand_bits(rng, n, emin=1, emax=254):
    s = rng.integers(0, 2, n, dtype=np.int64)
    e = rng.integers(emin, emax + 1, n, dtype=np.int64)
    m = rng.integers(0, 1 << 23, n, dtype=np.int64)
    return ((s << 31) | (e << 23) | m).astype(np.uint32)


def ieee_ref(op, a, b):
    fa, fb = fp32.to_f(a), fp32.to_f(b)
    with np.errstate(all="ignore"):
        r = {"add": fa + fb, "sub": fa - fb, "mul": fa * fb}[op].astype(np.float32)
    return r.view(np.uint32)


def comparable(r):
    """results where IEEE and FTZ agree: normal, inf, or exact zero from normal operands"""
    e = (r >> 23) & 0xFF
    return (e != 0) | ((r & 0x7FFFFFFF) == 0)


def binops(rng):
    n = 400_000
    ops = {"add": fp32.add, "sub": fp32.sub, "mul": fp32.mul}
    for name, fn in ops.items():
        for label in ("wide exponents", "close exponents", "near cancellation"):
            if label == "wide exponents":
                a, b = rand_bits(rng, n), rand_bits(rng, n)
            elif label == "close exponents":
                a, b = rand_bits(rng, n, 100, 110), rand_bits(rng, n, 100, 110)
            else:
                a = rand_bits(rng, n, 120, 130)
                b = (a ^ np.uint32(0x80000000)) + rng.integers(-3, 4, n).astype(np.uint32)   # |b| ~ |a|, opposite sign
                if name == "sub":
                    b = b ^ np.uint32(0x80000000)
            ref, got = ieee_ref(name, a, b), fn(a, b)
            m = comparable(ref)
            bad = np.count_nonzero(ref[m] != got[m])
            check(f"{name} {label}: {m.sum()} comparable of {n}", bad == 0, f"{bad} differ")
    # specials
    sp = fp32.from_f([0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, 3.5e38, -3.5e38, 1.2e-38])
    A, B = np.meshgrid(sp, sp)
    A, B = A.ravel(), B.ravel()
    for name, fn in ops.items():
        ref, got = ieee_ref(name, A, B), fn(A, B)
        refn = np.where(np.isnan(fp32.to_f(ref)), fp32.QNAN, ref)
        m = comparable(ref) & (((A >> 23) & 0xFF) != 0) & (((B >> 23) & 0xFF) != 0) | ((A & 0x7FFFFFFF) == 0) & ((B & 0x7FFFFFFF) == 0)
        check(f"{name} special values", np.array_equal(refn[m], got[m]))
    # FTZ: results below the normal range flush to a signed zero; subnormal inputs are zeros
    tiny = fp32.from_f(np.float32(2.0 ** -70))
    r = fp32.mul(np.repeat(tiny, 2), fp32.from_f([np.float32(2.0 ** -70), -np.float32(2.0 ** -70)]))
    check("mul underflow flushes to +-0", list(r) == [0, 0x80000000])
    sub = np.array([0x00000001, 0x807FFFFF], np.uint32)
    r = fp32.add(sub, fp32.from_f([1.0, 1.0]))
    check("subnormal input is zero", list(r) == [0x3F800000, 0x3F800000])
    r = fp32.sub(fp32.from_f([np.float32(1.5 * 2.0 ** -126)]), fp32.from_f([np.float32(2.0 ** -126)]))
    check("sub result 2^-127 flushes to +0", list(r) == [0])
    check("x - x = +0, -0 + -0 = -0", list(fp32.add(fp32.from_f([3.0, -0.0]), fp32.from_f([-3.0, -0.0])))
          == [0, 0x80000000])


def conversions(rng):
    i = np.concatenate([rng.integers(-2**31, 2**31, 300_000, dtype=np.int64),
                        rng.integers(-2**25, 2**25, 100_000, dtype=np.int64),
                        np.array([0, 1, -1, 2**31 - 1, -2**31, 2**24 + 1, 2**24 + 3, -(2**24 + 1)])])
    ref = i.astype(np.float32).view(np.uint32)
    check("int32 -> fp32 (round to nearest even)", np.array_equal(fp32.from_int(i), ref))
    a = np.concatenate([rand_bits(rng, 200_000, 100, 170), fp32.from_f(np.arange(-300, 300) * 0.5)])
    f = fp32.to_f(a).astype(np.float64)
    for lo, hi, name in ((-127, 127, "int8 (+-127)"), (-2**31, 2**31 - 1, "int32")):
        ref = np.clip(np.rint(f), lo, hi).astype(np.int64)
        check(f"fp32 -> {name}, round to nearest even, saturate", np.array_equal(fp32.to_int(a, lo, hi), ref))
    ref = np.clip(np.floor(f), -2**31, 2**31 - 1).astype(np.int64)
    check("fp32 -> int32 floor", np.array_equal(fp32.to_int(a, -2**31, 2**31 - 1, "floor"), ref))
    check("fp32 -> int: NaN -> 0, +-inf saturate",
          list(fp32.to_int(fp32.from_f([np.nan, np.inf, -np.inf]), -127, 127)) == [0, 127, -127])


def sfu(rng):
    import sfu as S
    xs = {"exp": np.concatenate([rng.uniform(-104, 89, 400_000), rng.uniform(-20, 20, 400_000)]),
          "recip": np.concatenate([np.exp(rng.uniform(-85, 85, 400_000)), -np.exp(rng.uniform(-85, 85, 400_000))]),
          "rsqrt": np.exp(rng.uniform(-85, 85, 800_000))}
    ref = {"exp": np.exp, "recip": lambda x: 1 / x, "rsqrt": lambda x: 1 / np.sqrt(x)}
    target = {"exp": 3e-5, "recip": 2e-7, "rsqrt": 5e-7}
    for name, x in xs.items():
        b = fp32.from_f(x)
        xf = fp32.to_f(b).astype(np.float64)
        got = fp32.to_f(getattr(S, name)(b)).astype(np.float64)
        want = ref[name](xf)
        m = (np.abs(want) >= 2.0 ** -125) & (np.abs(want) < 2.0 ** 127)
        rel = np.abs(got[m] - want[m]) / np.abs(want[m])
        check(f"{name.upper()} max relative error {rel.max():.2e} (target {target[name]:.0e}), "
              f"{m.sum()} inputs", rel.max() <= target[name])
    sp = fp32.from_f([np.nan, np.inf, -np.inf, 0.0, -0.0, 89.0, -104.0, -1.0])
    e, r, q = S.exp(sp), S.recip(sp), S.rsqrt(sp)
    check("EXP specials: NaN, +inf, 0, 0, 1, 1, >=89 -> inf, <=-104 -> 0",
          [int(v) for v in e] == [fp32.QNAN, fp32.PINF, 0, fp32.ONE, fp32.ONE, fp32.PINF, 0, int(e[7])]
          and abs(float(fp32.to_f(e[7])) - np.exp(-1)) < 1e-5)
    check("RECIP specials: NaN, +0, -0, +inf, -inf",
          [int(v) for v in r[:5]] == [fp32.QNAN, 0, 0x80000000, fp32.PINF, fp32.NINF])
    check("RSQRT specials: NaN, 0, NaN(-inf), +inf(+0), +inf(-0), NaN(-1)",
          [int(v) for v in q[:5]] + [int(q[7])] == [fp32.QNAN, 0, fp32.QNAN, fp32.PINF, fp32.PINF, fp32.QNAN])


def main():
    rng = np.random.default_rng(3)
    print("== add / sub / mul");  binops(rng)
    print("== conversions");      conversions(rng)
    print("== special functions"); sfu(rng)
    print("PASS" if not fails else f"FAIL: {len(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())

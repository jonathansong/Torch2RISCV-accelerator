#!/usr/bin/env python3
"""Test vectors for the L2 fp32 units (sa_fp32_add / mul / cvt) from the
bit-exact model llm/fp32.py. Writes into --out:
  fp_add.hex  a b sub expected          (hex words per line)
  fp_mul.hex  a b expected
  fp_i2f.hex  x expected
  fp_f2i.hex  x floor lo hi expected
and fp_counts.vh with the counts."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "llm"))
import fp32  # noqa: E402


def bits(rng, n, emin=0, emax=255):
    s = rng.integers(0, 2, n, dtype=np.int64)
    e = rng.integers(emin, emax + 1, n, dtype=np.int64)
    m = rng.integers(0, 1 << 23, n, dtype=np.int64)
    return ((s << 31) | (e << 23) | m).astype(np.uint32)


def specials():
    v = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFFFFFFF,
         0x00000001, 0x807FFFFF, 0x00800000, 0x80800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF,
         0x00800001, 0x01000000, 0x3F800001, 0x4B000000, 0xCB000000, 0x4F000000, 0xCF000000]
    return np.array(v, np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--n", type=int, default=40000)
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    n = args.n
    sp = specials()
    A, B = np.meshgrid(sp, sp)
    A, B = A.ravel(), B.ravel()

    def mix():
        a = np.concatenate([bits(rng, n), bits(rng, n, 100, 110), bits(rng, n, 1, 30), bits(rng, n, 220, 254), A])
        b = np.concatenate([bits(rng, n), bits(rng, n, 100, 110), bits(rng, n, 1, 30), bits(rng, n, 220, 254), B])
        # near cancellations
        c = bits(rng, n, 120, 130)
        d = (c ^ np.uint32(0x80000000)) + rng.integers(-3, 4, n).astype(np.uint32)
        return np.concatenate([a, c]), np.concatenate([b, d])

    os.makedirs(args.out, exist_ok=True)
    a, b = mix()
    sub = rng.integers(0, 2, a.size).astype(np.uint32)
    y = np.where(sub == 1, fp32.sub(a, b), fp32.add(a, b))
    lines = [f"{x:08x} {z:08x} {s} {w:08x}" for x, z, s, w in zip(a, b, sub, y)]
    open(os.path.join(args.out, "fp_add.hex"), "w").write("\n".join(lines) + "\n")
    a, b = mix()
    # ties: significands with only their top 13 bits random -> products with a
    # few dropped bits, often exactly one half (round to even decides)
    def short(k):
        m = (rng.integers(0, 1 << 12, k) << 11) | (1 << 11) * rng.integers(0, 2, k)
        e = rng.integers(100, 150, k)
        return ((rng.integers(0, 2, k) << 31) | (e << 23) | m).astype(np.uint32)
    a, b = np.concatenate([a, short(n)]), np.concatenate([b, short(n)])
    y = fp32.mul(a, b)
    open(os.path.join(args.out, "fp_mul.hex"), "w").write(
        "\n".join(f"{x:08x} {z:08x} {w:08x}" for x, z, w in zip(a, b, y)) + "\n")
    i = np.concatenate([rng.integers(-2**31, 2**31, n), rng.integers(-2**25, 2**25, n),
                        np.array([0, 1, -1, 2**31 - 1, -2**31, 2**24 + 1, 2**24 + 3, -(2**24 + 1), 2**25 + 2])])
    y = fp32.from_int(i)
    open(os.path.join(args.out, "fp_i2f.hex"), "w").write(
        "\n".join(f"{x & 0xFFFFFFFF:08x} {w:08x}" for x, w in zip(i, y)) + "\n")
    x = np.concatenate([bits(rng, n, 100, 170), bits(rng, n), sp, fp32.from_f(np.arange(-300, 300) * 0.25)])
    bounds = [(-2**31, 2**31 - 1), (-127, 127), (0, 63), (-5, -1)]
    sel = rng.integers(0, len(bounds), x.size)
    fl = rng.integers(0, 2, x.size)
    lo = np.array([bounds[k][0] for k in sel])
    hi = np.array([bounds[k][1] for k in sel])
    y = np.array([0] * x.size, np.int64)
    for k, (l, h) in enumerate(bounds):
        for f in (0, 1):
            m = (sel == k) & (fl == f)
            y[m] = fp32.to_int(x[m], l, h, "floor" if f else "rne")
    open(os.path.join(args.out, "fp_f2i.hex"), "w").write("\n".join(
        f"{xx:08x} {f} {l & 0xFFFFFFFF:08x} {h & 0xFFFFFFFF:08x} {w & 0xFFFFFFFF:08x}"
        for xx, f, l, h, w in zip(x, fl, lo, hi, y)) + "\n")
    counts = {name: sum(1 for _ in open(os.path.join(args.out, f"fp_{name}.hex"))) for name in ("add", "mul", "i2f", "f2i")}
    with open(os.path.join(args.out, "fp_counts.vh"), "w") as f:
        for name, c in counts.items():
            f.write(f"localparam integer N_{name.upper()} = {c};\n")
    print(counts)


if __name__ == "__main__":
    main()

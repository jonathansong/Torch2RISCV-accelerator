#!/usr/bin/env python3
"""Generate 8x8x8 int8 test vectors and the NumPy golden result for tb_matmul_unit.

Writes into OUT_DIR (default: vectors/):
  a.hex, b.hex   one 64-bit little-endian word per matrix row (8 rows/case)
  c.hex          one int32 per C element, row-major (64 per case)
  n_cases.vh     `localparam integer NC = <cases>;`
"""
import argparse
import os

import numpy as np

N = 8


def cases(rng, n_random):
    i8 = np.int8
    full = lambda v: np.full((N, N), v, dtype=i8)
    eye = np.eye(N, dtype=i8)
    yield "zeros", full(0), full(0)
    yield "identity_left", eye, rng.integers(-128, 128, (N, N), dtype=i8)
    yield "identity_right", rng.integers(-128, 128, (N, N), dtype=i8), eye
    yield "max_positive", full(-128), full(-128)      # 8 * 16384 = 131072
    yield "max_negative", full(-128), full(127)
    yield "all_127", full(127), full(127)
    yield "sign_mix", np.tile(np.array([1, -1], dtype=i8), (N, N // 2)), full(-128)
    yield "index_pattern", (np.arange(N * N) - 32).astype(i8).reshape(N, N), \
        (np.arange(N * N)[::-1] - 32).astype(i8).reshape(N, N)
    for k in range(n_random):
        yield f"random_{k}", rng.integers(-128, 128, (N, N), dtype=i8), \
            rng.integers(-128, 128, (N, N), dtype=i8)


def row_words(m):
    """One 64-bit word per row; element k in bits [8k+7:8k]."""
    return [int.from_bytes(row.astype(np.int8).tobytes(), "little") for row in m]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="vectors")
    ap.add_argument("--random", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    a_lines, b_lines, c_lines, names = [], [], [], []
    for name, a, b in cases(rng, args.random):
        c = a.astype(np.int32) @ b.astype(np.int32)       # golden model
        a_lines += [f"{w:016x}" for w in row_words(a)]
        b_lines += [f"{w:016x}" for w in row_words(b)]
        c_lines += [f"{int(v) & 0xFFFFFFFF:08x}" for v in c.flatten()]
        names.append(name)

    for fname, lines in (("a.hex", a_lines), ("b.hex", b_lines), ("c.hex", c_lines)):
        with open(os.path.join(args.out, fname), "w") as f:
            f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "n_cases.vh"), "w") as f:
        f.write(f"localparam integer NC = {len(names)};\n")
    print(f"{len(names)} cases -> {args.out}/: {', '.join(names[:8])}, ...")


if __name__ == "__main__":
    main()

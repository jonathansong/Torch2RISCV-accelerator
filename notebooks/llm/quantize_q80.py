#!/usr/bin/env python3
"""llama2.c fp32 checkpoint (legacy .bin, version 0) -> Q8_0 checkpoint for runq.c.

A NumPy port of llama2.c export.py version2_export (no PyTorch needed): every
matmul weight and the token embedding become symmetric int8 in [-127, 127]
with one fp32 scale per group of GS elements along the input dimension; the
RMSNorm weights stay fp32. GS starts at 64 and is halved until it divides
both dim and hidden_dim (runq.c quantizes the activations of both in groups
of GS). --gs sets the starting size; prepare_llm.sh uses 32 for every model
so that arm_baseline.sh can also build runq.c with GS as a compile-time
constant.

    python3 quantize_q80.py [--gs 32] stories15M.bin stories15M_q80.bin
"""
import argparse
import struct
import sys

import numpy as np


def read_legacy(path):
    raw = np.memmap(path, dtype=np.float32, mode="r")
    dim, hidden, layers, heads, kv_heads, vocab, seq = struct.unpack("7i", raw[:7].tobytes())
    shared = vocab > 0
    vocab = abs(vocab)
    head = dim // heads
    kv = kv_heads * head
    shapes = [("tok", (vocab, dim)), ("rms_att", (layers, dim)), ("wq", (layers, dim, dim)),
              ("wk", (layers, kv, dim)), ("wv", (layers, kv, dim)), ("wo", (layers, dim, dim)),
              ("rms_ffn", (layers, dim)), ("w1", (layers, hidden, dim)), ("w2", (layers, dim, hidden)),
              ("w3", (layers, hidden, dim)), ("rms_final", (dim,)), ("freq", (seq * head,))]
    if not shared:
        shapes.append(("wcls", (vocab, dim)))
    w, off = {}, 7
    for name, shape in shapes:
        n = int(np.prod(shape))
        w[name] = raw[off:off + n].reshape(shape)
        off += n
    if off != raw.size:
        sys.exit(f"{path}: {raw.size} floats, expected {off}; not a llama2.c legacy checkpoint?")
    cfg = (dim, hidden, layers, heads, kv_heads, vocab, seq)
    return cfg, shared, w


def quantize(w, gs):
    """(int8 values, fp32 scales) of w in groups of gs (llama2.c quantize_q80)."""
    g = np.asarray(w, dtype=np.float32).reshape(-1, gs)
    scale = np.abs(g).max(axis=1) / 127.0
    q = np.round(g / np.where(scale == 0, 1, scale)[:, None]).astype(np.int8)
    err = np.abs(q.astype(np.float32) * scale[:, None] - g).max()
    return q, scale.astype(np.float32), err


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--gs", type=int, default=64, help="largest group size (power of 2)")
    args = ap.parse_args()
    cfg, shared, w = read_legacy(args.src)
    dim, hidden, layers = cfg[:3]
    gs = args.gs
    while dim % gs or hidden % gs:
        gs //= 2
    with open(args.dst, "wb") as f:
        f.write(struct.pack("Ii", 0x616B3432, 2) + struct.pack("7i", *cfg))
        f.write(struct.pack("B", int(shared)) + struct.pack("i", gs))
        f.write(b"\0" * (256 - f.tell()))
        for name in ("rms_att", "rms_ffn", "rms_final"):
            f.write(np.ascontiguousarray(w[name], dtype=np.float32).tobytes())
        tensors = [w["tok"]] + [w[n][l] for n in ("wq", "wk", "wv", "wo", "w1", "w2", "w3")
                                for l in range(layers)]
        if not shared:
            tensors.append(w["wcls"])
        worst = 0.0
        for t in tensors:
            q, s, err = quantize(t, gs)
            f.write(q.tobytes())
            f.write(s.tobytes())
            worst = max(worst, float(err))
    print(f"{args.dst}: dim {dim}, hidden {hidden}, {layers} layers, group size {gs}, "
          f"max group error {worst:.5f}")


if __name__ == "__main__":
    main()

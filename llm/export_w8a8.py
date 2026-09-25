#!/usr/bin/env python3
"""Export a llama2.c checkpoint in the device format `*.w8a8` (docs/llm_inference_plan.md §7.1).

Numerics as DeviceModel (plan §3.1): int8 weights with one fp32 scale per
output channel, an int8 embedding table with one fp32 scale per row,
static per-layer KV scales, fp32 norm weights and RoPE tables.

Linear weights are stored pre-packed for the matrix engine's B operand
(y = x W^T; B = W^T, in x out): tile t holds output channels
[t*D, (t+1)*D) as an (in x D) row-major int8 block, i.e. SPAD_B words
b + kk = W[t*D:(t+1)*D, kk]; the tiles follow each other, so any run of
consecutive tiles is one contiguous LD (LINEAR) into SPAD_B. The packing
depends on D, which the header records.

File layout (little-endian): a 64-byte header, a table of 32-byte
entries, then the data, every tensor 64-byte aligned. The layers share
one layout at a fixed stride, so a list can address layer l as
layer_base + l * layer_stride (a PARAM stepped by LOOP_END).

    header   magic "W8A8LLM\\0", u32 version, D, dim, hidden, layers, heads,
             kv_heads, vocab, seq_len, n_entries, layer_base, layer_stride
    entry    name[16], u32 offset, u32 bytes, u32 per_layer, u32 reserved
             (per_layer = 1: offset is relative to the layer's base)

Global tensors: emb_q (vocab x dim int8), emb_s (vocab f32), rms_final,
wcls (packed) + s_wcls, rope_cos / rope_sin (seq_len x dim f32, each value
stored for both elements of its rotation pair, as SWAPNEG pairs them).
Per layer: rms_att, rms_ffn, wqkv (q | k | v rows) + s_wqkv, wo + s_wo,
w13 (w1 | w3 rows) + s_w13, w2 + s_w2, kvp (8 f32: s_k, s_v, 1/s_k, 1/s_v,
s_k/sqrt(head_size), s_v/127, 0, 0).

    python3 llm/export_w8a8.py stories15M.bin --kv stories15M_kv_p99.99.npy --d 8 -o stories15M_d8.w8a8
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkpoint  # noqa: E402
from ref_model import quantize_rows, rope_tables  # noqa: E402

F = np.float32
MAGIC = b"W8A8LLM\0"
VERSION = 1
HDR = struct.Struct("<8s12I")                 # magic + 12 u32 = 56 bytes (padded to 64)
ENT = struct.Struct("<16s4I")                 # 32 bytes
ALIGN = 64
LAYER_TENSORS = ("rms_att", "rms_ffn", "wqkv", "s_wqkv", "wo", "s_wo", "w13", "s_w13", "w2", "s_w2", "kvp")


def pack_b(q, d):
    """int8 W (out x in) -> B tiles (out/d, in, d), contiguous: tile t = W[t*d:(t+1)*d, :]^T."""
    out, inp = q.shape
    if out % d or inp % d:
        raise ValueError(f"linear {out} x {inp} not a multiple of D = {d}")
    return np.ascontiguousarray(q.reshape(out // d, d, inp).transpose(0, 2, 1))


def unpack_b(tiles):
    """Inverse of pack_b: (nt, in, d) -> W (nt*d, in)."""
    nt, inp, d = tiles.shape
    return np.ascontiguousarray(tiles.transpose(0, 2, 1).reshape(nt * d, inp))


def kv_params(cfg, s_k, s_v):
    """The per-layer KV parameter block, computed as DeviceModel does."""
    s_k, s_v = F(s_k), F(s_v)
    return np.array([s_k, s_v, F(1.0 / s_k), F(1.0 / s_v), F(float(s_k) / np.sqrt(cfg.head_size)),
                     F(float(s_v) / 127.0), 0, 0], F)


def device_tensors(cfg, w, kv_scales, d):
    """({name: array} global, [{name: array}] per layer) in file order."""
    lin = lambda wt: quantize_rows(wt)
    glob = {}
    glob["emb_q"], glob["emb_s"] = quantize_rows(w["tok"])
    glob["rms_final"] = np.asarray(w["rms_final"], F)
    q, s = lin(w["wcls"])
    glob["wcls"], glob["s_wcls"] = pack_b(q, d), s
    cos, sin = rope_tables(cfg)
    glob["rope_cos"] = np.repeat(cos, 2, axis=1).astype(F)       # (seq, dim): pair (2i, 2i+1) share cos[i]
    glob["rope_sin"] = np.repeat(sin, 2, axis=1).astype(F)
    layers = []
    for l in range(cfg.layers):
        t = {"rms_att": np.asarray(w["rms_att"][l], F), "rms_ffn": np.asarray(w["rms_ffn"][l], F)}
        for name, rows in (("wqkv", (w["wq"][l], w["wk"][l], w["wv"][l])), ("wo", (w["wo"][l],)),
                           ("w13", (w["w1"][l], w["w3"][l])), ("w2", (w["w2"][l],))):
            q, s = lin(np.concatenate(rows, axis=0))
            t[name], t["s_" + name] = pack_b(q, d), s
        t["kvp"] = kv_params(cfg, *kv_scales[l])
        layers.append(t)
    return glob, layers


def _align(n):
    return -(-n // ALIGN) * ALIGN


def export(cfg, w, kv_scales, d, path):
    glob, layers = device_tensors(cfg, w, kv_scales, d)
    names = list(glob) + list(LAYER_TENSORS)
    hdr_bytes = _align(64 + ENT.size * len(names))
    entries, off = [], hdr_bytes
    for n in glob:
        entries.append((n, off, glob[n].nbytes, 0))
        off = _align(off + glob[n].nbytes)
    layer_base, rel = off, 0
    for n in LAYER_TENSORS:
        entries.append((n, rel, layers[0][n].nbytes, 1))
        rel = _align(rel + layers[0][n].nbytes)
    stride = rel
    total = layer_base + stride * cfg.layers
    buf = np.zeros(total, np.uint8)
    head = HDR.pack(MAGIC, VERSION, d, cfg.dim, cfg.hidden, cfg.layers, cfg.heads, cfg.kv_heads, cfg.vocab,
                    cfg.seq_len, len(entries), layer_base, stride)
    buf[:len(head)] = np.frombuffer(head, np.uint8)
    for i, (n, o, nb, per) in enumerate(entries):
        e = ENT.pack(n.encode().ljust(16, b"\0"), o, nb, per, 0)
        buf[64 + i * ENT.size:64 + (i + 1) * ENT.size] = np.frombuffer(e, np.uint8)
        if per:
            for l in range(cfg.layers):
                a = layer_base + l * stride + o
                buf[a:a + nb] = np.frombuffer(layers[l][n].tobytes(), np.uint8)
        else:
            buf[o:o + nb] = np.frombuffer(glob[n].tobytes(), np.uint8)
    buf.tofile(path)
    return total


class W8A8:
    """Reader of a .w8a8 file (or bytes / uint8 array already in memory)."""
    DTYPES = {"emb_q": np.int8, "wcls": np.int8, "wqkv": np.int8, "wo": np.int8, "w13": np.int8, "w2": np.int8}

    def __init__(self, src):
        self.raw = np.fromfile(src, np.uint8) if isinstance(src, str) else np.frombuffer(src, np.uint8)
        f = HDR.unpack(self.raw[:HDR.size].tobytes())
        if f[0] != MAGIC or f[1] != VERSION:
            raise ValueError("not a w8a8 v1 file")
        (self.d, dim, hidden, layers, heads, kv_heads, vocab, seq, n, self.layer_base,
         self.layer_stride) = f[2:]
        self.cfg = checkpoint.Config(dim, hidden, layers, heads, kv_heads, vocab, seq)
        self.entries = {}
        for i in range(n):
            name, off, nb, per, _ = ENT.unpack(self.raw[64 + i * ENT.size:64 + (i + 1) * ENT.size].tobytes())
            self.entries[name.rstrip(b"\0").decode()] = (off, nb, bool(per))

    def offset(self, name, layer=None):
        """Byte offset of a tensor in the file (per-layer tensors need `layer`)."""
        off, _, per = self.entries[name]
        if per:
            if layer is None:
                raise ValueError(f"{name} is per layer")
            return self.layer_base + layer * self.layer_stride + off
        return off

    def get(self, name, layer=None):
        """The tensor as a NumPy view with its natural shape."""
        c, d = self.cfg, self.d
        off, nb = self.offset(name, layer), self.entries[name][1]
        a = self.raw[off:off + nb].view(self.DTYPES.get(name, F))
        shapes = {"emb_q": (c.vocab, c.dim), "rope_cos": (c.seq_len, c.dim), "rope_sin": (c.seq_len, c.dim),
                  "wcls": (c.vocab // d, c.dim, d), "wqkv": (-1, c.dim, d), "wo": (c.dim // d, c.dim, d),
                  "w13": (2 * c.hidden // d, c.dim, d), "w2": (c.dim // d, c.hidden, d)}
        return a.reshape(shapes[name]) if name in shapes else a


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--kv", required=True, help=".npy of (layers, 2) static K / V scales (eval_quant.py --save-kv)")
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    cfg, w = checkpoint.load(args.checkpoint)
    n = export(cfg, w, np.load(args.kv), args.d, args.out)
    print(f"{args.out}: {n / 2**20:.1f} MB, D = {args.d}, {cfg}")


if __name__ == "__main__":
    main()

"""llama2.c fp32 checkpoints (legacy .bin, version 0).

Layout (llama2.c run.c memory_map_weights): a 7 x int32 header
(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len; a
negative vocab_size means a separate classifier), then fp32 tensors in the
order below; PyTorch orientation, i.e. linear weights are (out, in).
"""
import struct
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    dim: int
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    vocab: int
    seq_len: int

    @property
    def head_size(self):
        return self.dim // self.heads

    @property
    def kv_dim(self):
        return self.kv_heads * self.head_size


def load(path):
    """(Config, {name: float32 array}) of a legacy checkpoint (memory-mapped).
    Names: tok (vocab, dim), rms_att / rms_ffn (layers, dim), wq wk wv wo
    (layers, out, in), w1 w3 (layers, hidden, dim), w2 (layers, dim, hidden),
    rms_final (dim,), wcls (vocab, dim; the embedding when shared)."""
    raw = np.memmap(path, dtype=np.float32, mode="r")
    dim, hidden, layers, heads, kv_heads, vocab, seq = struct.unpack("7i", raw[:7].tobytes())
    shared = vocab > 0
    cfg = Config(dim, hidden, layers, heads, kv_heads, abs(vocab), seq)
    hs, kv = cfg.head_size, cfg.kv_dim
    shapes = [("tok", (cfg.vocab, dim)), ("rms_att", (layers, dim)), ("wq", (layers, dim, dim)),
              ("wk", (layers, kv, dim)), ("wv", (layers, kv, dim)), ("wo", (layers, dim, dim)),
              ("rms_ffn", (layers, dim)), ("w1", (layers, hidden, dim)), ("w2", (layers, dim, hidden)),
              ("w3", (layers, hidden, dim)), ("rms_final", (dim,)), ("freq", (seq * hs,))]
    if not shared:
        shapes.append(("wcls", (cfg.vocab, dim)))
    w, off = {}, 7
    for name, shape in shapes:
        n = int(np.prod(shape))
        w[name] = raw[off:off + n].reshape(shape)
        off += n
    if off != raw.size:
        raise ValueError(f"{path}: {raw.size} floats, expected {off}; not a llama2.c legacy checkpoint")
    if shared:
        w["wcls"] = w["tok"]
    del w["freq"]                     # unused RoPE tables (run.c recomputes them)
    return cfg, w

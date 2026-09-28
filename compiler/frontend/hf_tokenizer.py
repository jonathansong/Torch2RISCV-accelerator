"""Byte-level BPE tokenizer of a HuggingFace tokenizer.json (GPT-2 style: SmolLM2,
Qwen, Llama 3), with no dependencies beyond the standard library, so it also
runs on the board (compiler/tests/board_generate.py)."""
import json
import os


class BpeTokenizer:
    """Byte-level BPE from a HuggingFace tokenizer.json (GPT-2 style). Decoding is
    exact; encoding splits on spaces / punctuation before BPE (close to, not
    always equal to, the regex pre-tokenizer): enough for tests and demos."""

    def __init__(self, path):
        t = json.load(open(os.path.join(path, "tokenizer.json")))
        m = t["model"]
        self.vocab = m["vocab"]
        self.inv = {i: s for s, i in self.vocab.items()}
        for a in t.get("added_tokens", []):
            self.inv[a["id"]] = a["content"]
        merges = [tuple(x.split(" ")) if isinstance(x, str) else tuple(x) for x in m["merges"]]
        self.ranks = {p: i for i, p in enumerate(merges)}
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        self.b2u = {b: chr(c) for b, c in zip(bs, cs)}
        self.u2b = {chr(c): b for b, c in zip(bs, cs)}

    def _bpe(self, word):
        parts = list(word)
        while len(parts) > 1:
            pairs = [(self.ranks.get((parts[i], parts[i + 1]), 1 << 30), i) for i in range(len(parts) - 1)]
            r, i = min(pairs)
            if r == 1 << 30:
                break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        return parts

    def encode(self, text):
        import re
        ids = []
        for piece in re.findall(r" ?[A-Za-z]+| ?[0-9]| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+", text):
            u = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            ids += [self.vocab[p] for p in self._bpe(u)]
        return ids

    def decode(self, ids):
        s = "".join(self.inv.get(i, "") for i in ids)
        return bytes(self.u2b.get(ch, 32) for ch in s).decode("utf-8", errors="replace")

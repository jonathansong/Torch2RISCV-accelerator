"""The llama2.c tokenizer (tokenizer.bin, SentencePiece-style BPE), ported
from run.c: encode() with byte fallback and score-ordered merges, decode()
with the leading-space strip after BOS and <0xXX> byte pieces."""
import re
import struct

BOS, EOS = 1, 2


class Tokenizer:
    def __init__(self, path, vocab_size=32000):
        with open(path, "rb") as f:
            self.max_token_length = struct.unpack("i", f.read(4))[0]
            self.vocab, self.scores = [], []
            for _ in range(vocab_size):
                score, n = struct.unpack("fi", f.read(8))
                self.scores.append(score)
                self.vocab.append(f.read(n))
        self.lookup = {}
        for i, s in enumerate(self.vocab):
            self.lookup.setdefault(s, i)      # first id wins, like bsearch over a stable sort
        self._byte = re.compile(rb"^<0x([0-9A-Fa-f]{2})>$")

    def encode(self, text, bos=True, eos=False):
        tokens = [BOS] if bos else []
        data = text.encode("utf-8")
        if data:
            tokens.append(self.lookup[b" "])  # add_dummy_prefix
        i = 0
        while i < len(data):                  # one UTF-8 code point at a time (<= 4 bytes)
            j = i + 1
            while j < len(data) and (data[j] & 0xC0) == 0x80 and j - i < 4:
                j += 1
            cp = data[i:j]
            if cp in self.lookup:
                tokens.append(self.lookup[cp])
            else:
                tokens.extend(b + 3 for b in cp)   # byte fallback
            i = j
        while True:                           # merge the best-scoring adjacent pair
            best, best_id, best_idx = -1e10, -1, -1
            for k in range(len(tokens) - 1):
                tid = self.lookup.get(self.vocab[tokens[k]] + self.vocab[tokens[k + 1]], -1)
                if tid != -1 and self.scores[tid] > best:
                    best, best_id, best_idx = self.scores[tid], tid, k
            if best_idx == -1:
                break
            tokens[best_idx:best_idx + 2] = [best_id]
        if eos:
            tokens.append(EOS)
        return tokens

    def piece(self, prev, token):
        """Bytes printed for `token` after `prev` (run.c decode + safe_printf)."""
        p = self.vocab[token]
        if prev == BOS and p[:1] == b" ":
            p = p[1:]
        m = self._byte.match(p)
        if m:
            p = bytes([int(m.group(1), 16)])
        if len(p) == 1 and not (32 <= p[0] < 127 or p[0] in b" \t\n\r\x0b\x0c"):
            return b""
        return p

    def decode(self, tokens):
        """Text of a token sequence starting with BOS, as run.c prints it."""
        return b"".join(self.piece(a, b) for a, b in zip(tokens, tokens[1:])).decode("utf-8", "replace")

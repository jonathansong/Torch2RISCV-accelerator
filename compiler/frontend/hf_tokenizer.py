"""Byte-level BPE tokenizer of a HuggingFace tokenizer.json (GPT-2 style: SmolLM2,
Qwen, Llama 3), with no dependencies beyond the standard library, so it also
runs on the board (compiler/tests/board_generate.py)."""
import json
import os


GPT2_SPLIT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BpeTokenizer:
    r"""Byte-level BPE from a HuggingFace tokenizer.json (GPT-2 style). Decoding is
    exact. Encoding follows the file's normalizer (NFC) and pre-tokenizers
    (Split with a regex, Digits, ByteLevel's own GPT-2 split) and the model's
    ignore_merges, with the added (special) tokens matched first: identical to
    `tokenizers` when the `regex` module is there (Unicode classes \p{L} /
    \p{N}); without it (the board's bare Python) a close approximation."""

    def __init__(self, path):
        t = json.load(open(os.path.join(path, "tokenizer.json")))
        m = t["model"]
        self.vocab = m["vocab"]
        self.inv = {i: s for s, i in self.vocab.items()}
        self.added = {}
        for a in t.get("added_tokens", []):
            self.inv[a["id"]] = a["content"]
            self.added[a["content"]] = a["id"]
        self.ignore_merges = bool(m.get("ignore_merges"))
        norm = t.get("normalizer") or {}
        self.nfc = norm.get("type") == "NFC" or any(n.get("type") == "NFC" for n in norm.get("normalizers", []))
        pre = t.get("pre_tokenizer") or {}
        self.steps = pre.get("pretokenizers", [pre] if pre else [])
        try:
            import regex
            self.re = regex
        except ImportError:
            self.re = None
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

    def _split(self, pieces, pattern):
        """Split, behavior Isolated: each match and each stretch between matches a piece."""
        out = []
        for s in pieces:
            pos = 0
            for mt in self.re.finditer(pattern, s):
                if mt.start() > pos:
                    out.append(s[pos:mt.start()])
                if mt.end() > mt.start():
                    out.append(mt.group(0))
                pos = mt.end()
            if pos < len(s):
                out.append(s[pos:])
        return out

    def _pieces(self, text):
        if self.re is None:                     # (no Unicode classes: the old approximation)
            import re
            return re.findall(r" ?[A-Za-z]+| ?[0-9]| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+", text)
        pieces = [text]
        for st in self.steps:
            ty = st.get("type")
            if ty == "Split":
                pat = st["pattern"].get("Regex") or self.re.escape(st["pattern"]["String"])
                if st.get("behavior", "Isolated") != "Isolated" or st.get("invert"):
                    raise NotImplementedError(f"Split behavior {st.get('behavior')} / invert")
                pieces = self._split(pieces, pat)
            elif ty == "Digits":
                pieces = self._split(pieces, r"\p{N}" if st.get("individual_digits") else r"\p{N}+")
            elif ty == "ByteLevel":
                if st.get("add_prefix_space") and pieces and not pieces[0].startswith(" "):
                    pieces[0] = " " + pieces[0]
                if st.get("use_regex", True):
                    pieces = [p for s in pieces for p in self.re.findall(GPT2_SPLIT, s)]
            else:
                raise NotImplementedError(f"pre-tokenizer {ty}")
        return pieces

    def encode(self, text):
        import unicodedata
        ids = []
        # the added tokens (<|begin_of_text|>, ...) first: matched as they are
        chunks = [text]
        for tok in sorted(self.added, key=len, reverse=True):
            nxt = []
            for c in chunks:
                if isinstance(c, int):
                    nxt.append(c)
                    continue
                parts = c.split(tok)
                for i, p in enumerate(parts):
                    if i:
                        nxt.append(self.added[tok])
                    if p:
                        nxt.append(p)
            chunks = nxt
        for c in chunks:
            if isinstance(c, int):
                ids.append(c)
                continue
            if self.nfc:
                c = unicodedata.normalize("NFC", c)
            for piece in self._pieces(c):
                u = "".join(self.b2u[b] for b in piece.encode("utf-8"))
                if self.ignore_merges and u in self.vocab:
                    ids.append(self.vocab[u])
                    continue
                ids += [self.vocab[p] for p in self._bpe(u)]
        return ids

    def decode(self, ids):
        s = "".join(self.inv.get(i, "") for i in ids)
        return bytes(self.u2b.get(ch, 32) for ch in s).decode("utf-8", errors="replace")

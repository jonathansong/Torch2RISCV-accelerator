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
        self.norms = norm.get("normalizers", [norm] if norm else [])
        for n in self.norms:
            if n.get("type") not in ("NFC", "Replace", "Prepend"):
                raise NotImplementedError(f"normalizer {n.get('type')}")
        pre = t.get("pre_tokenizer") or {}
        self.steps = pre.get("pretokenizers", [pre] if pre else [])
        dec = t.get("decoder") or {}
        # byte-level (GPT-2: SmolLM2, Qwen, Llama 3) or SentencePiece-style BPE on
        # the characters with byte fallback (<0xNN>; Gemma)
        self.byte_level = any(st.get("type") == "ByteLevel" for st in self.steps) or dec.get("type") == "ByteLevel" or \
            any(x.get("type") == "ByteLevel" for x in dec.get("decoders", []))
        self.byte_fallback = bool(m.get("byte_fallback"))
        self.unk = self.vocab.get(m.get("unk_token")) if m.get("unk_token") else None
        try:
            import regex
            self.re, self.exact = regex, True
        except ImportError:                      # (the standard re: no \p{L} / \p{N})
            import re
            self.re, self.exact = re, False
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
        """The merges by rank (lowest first; equal ranks left to right), with a
        heap over the adjacent pairs and a linked list of the symbols: n log n."""
        import heapq
        sym = list(word)
        n = len(sym)
        if n < 2:
            return sym
        nxt, prv, alive = list(range(1, n + 1)), list(range(-1, n - 1)), [True] * n
        heap = []
        for i in range(n - 1):
            r = self.ranks.get((sym[i], sym[i + 1]))
            if r is not None:
                heap.append((r, i, sym[i], sym[i + 1]))
        heapq.heapify(heap)
        while heap:
            r, i, a, b = heapq.heappop(heap)
            j = nxt[i] if i < n and alive[i] else n
            if not alive[i] or j >= n or sym[i] != a or sym[j] != b:
                continue                             # (stale: one side merged since)
            sym[i] = a + b
            alive[j] = False
            nxt[i] = nxt[j]
            if nxt[i] < n:
                prv[nxt[i]] = i
            p, q = prv[i], nxt[i]
            if p >= 0:
                rr = self.ranks.get((sym[p], sym[i]))
                if rr is not None:
                    heapq.heappush(heap, (rr, p, sym[p], sym[i]))
            if q < n:
                rr = self.ranks.get((sym[i], sym[q]))
                if rr is not None:
                    heapq.heappush(heap, (rr, i, sym[i], sym[q]))
        return [sym[i] for i in range(n) if alive[i]]

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
        if not self.exact and self.byte_level:  # (no Unicode classes: the old approximation)
            import re
            return re.findall(r" ?[A-Za-z]+| ?[0-9]| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+", text)
        pieces = [text]
        for st in self.steps:
            ty = st.get("type")
            if ty == "Split":
                pat = st["pattern"].get("Regex") or self.re.escape(st["pattern"]["String"])
                beh = st.get("behavior", "Isolated")
                if beh not in ("Isolated", "MergedWithPrevious") or st.get("invert"):
                    raise NotImplementedError(f"Split behavior {beh} / invert")
                if beh == "Isolated":
                    pieces = self._split(pieces, pat)
                else:                               # each match ends the piece before it
                    out = []
                    for s_ in pieces:
                        pos = 0
                        for mt in self.re.finditer(pat, s_):
                            if mt.end() > pos:
                                out.append(s_[pos:mt.end()])
                                pos = mt.end()
                        if pos < len(s_):
                            out.append(s_[pos:])
                    pieces = out
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
            for n in self.norms:
                ty = n.get("type")
                if ty == "NFC":
                    c = unicodedata.normalize("NFC", c)
                elif ty == "Replace":
                    pat = n["pattern"]
                    c = c.replace(pat["String"], n["content"]) if "String" in pat else self.re.sub(pat["Regex"], n["content"], c)
                elif ty == "Prepend":
                    c = n["prepend"] + c
            for piece in self._pieces(c):
                u = "".join(self.b2u[b] for b in piece.encode("utf-8")) if self.byte_level else piece
                if self.ignore_merges and u in self.vocab:
                    ids.append(self.vocab[u])
                    continue
                for p in self._bpe(u):
                    if p in self.vocab:
                        ids.append(self.vocab[p])
                    elif self.byte_fallback:
                        ids += [self.vocab[f"<0x{b:02X}>"] for b in p.encode("utf-8")]
                    else:
                        ids.append(self.unk)
        return ids

    def decode(self, ids):
        if not self.byte_level:                  # ▁ -> space, <0xNN> -> its byte
            out = bytearray()
            for i in ids:
                t = self.inv.get(i, "")
                if len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
                    out.append(int(t[3:5], 16))
                else:
                    out += t.replace("\u2581", " ").encode("utf-8")
            return out.decode("utf-8", errors="replace")
        s = "".join(self.inv.get(i, "") for i in ids)
        return bytes(self.u2b.get(ch, 32) for ch in s).decode("utf-8", errors="replace")

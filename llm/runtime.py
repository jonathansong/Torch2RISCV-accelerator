"""L5: the ARM-side runtime, LlamaDevice (docs/llm_inference_plan.md §9).

    dev = LlamaDevice("stories15M_d8.w8a8", "tokenizer.bin")
    for tok, text in dev.generate("Once upon a time", steps=128, temperature=0.8, topp=0.9):
        print(text, end="", flush=True)
    dev.close()

Load: the .w8a8 file, the io area (argument block, logits), the int8 KV
cache and the static token list (compile_model.ModelCompiler) share one
CMA buffer; the list is built and written once. Per token: write the
argument block (pos, token), submit the list with the token's PARAM
block through the rt_fw ring, wait for the notify interrupt, read the
logits, sample on the ARM (NumPy). The prompt goes through the same decode
path (no sampling); a real prefill is L6.

The sampler is run.c's: temperature 0 = argmax; otherwise softmax(logits
/ T), then top-p (nucleus) or plain multinomial, with run.c's xorshift
random numbers.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "driver"))
import compile_model as CM  # noqa: E402
from export_w8a8 import W8A8  # noqa: E402
from tokenizer import BOS, Tokenizer  # noqa: E402

F = np.float32
IO_BYTES = 0x40000


class Sampler:
    """run.c's sampler (temperature, top-p, xorshift RNG)."""
    MASK = (1 << 64) - 1

    def __init__(self, temperature=0.0, topp=0.9, seed=1):
        self.t, self.topp, self.state = temperature, topp, seed & self.MASK or 1

    def _u32(self):
        s = self.state
        s ^= s >> 12
        s ^= (s << 25) & self.MASK
        s ^= s >> 27
        self.state = s
        return ((s * 0x2545F4914F6CDD1D) & self.MASK) >> 32

    def _f32(self):
        return (self._u32() >> 8) / 16777216.0

    def __call__(self, logits):
        if self.t == 0:
            return int(np.argmax(logits))
        x = np.asarray(logits, F) / F(self.t)
        p = np.exp((x - x.max()).astype(F)).astype(F)
        p /= p.sum(dtype=F)
        coin = self._f32()
        if self.topp <= 0 or self.topp >= 1:
            c = np.cumsum(p, dtype=F)
            return int(min(np.searchsorted(c, coin, side="right"), p.size - 1))
        cutoff = (1.0 - self.topp) / (p.size - 1)
        idx = np.flatnonzero(p >= cutoff)
        idx = idx[np.argsort(-p[idx], kind="stable")]
        c = np.cumsum(p[idx], dtype=F)
        last = int(min(np.searchsorted(c, F(self.topp), side="right"), idx.size - 1))
        r = coin * float(c[last])
        k = int(min(np.searchsorted(c[:last + 1], r, side="right"), last))
        return int(idx[k])


class LlamaDevice:
    def __init__(self, model_path, tokenizer_path=None, bitfile=None, firmware=None, use_irq=True, device=None):
        from pynq_matmul import Device, allocate
        self.dev = device or Device(bitfile, firmware, ring_size=16, use_irq=use_irq)
        self.m = W8A8(model_path)
        c = self.cfg = self.m.cfg
        if self.m.d != self.dev.d:
            raise ValueError(f"{model_path} is packed for D = {self.m.d}, the overlay has D = {self.dev.d}")
        self.tok = Tokenizer(tokenizer_path or os.path.join(os.path.dirname(model_path), "tokenizer.bin"), c.vocab)
        self.dl = CM.ModelCompiler(self.m).build()
        rows = self.dl.array()
        self.io = -(-self.m.raw.size // 4096) * 4096
        self.kv = self.io + IO_BYTES
        self.kvb = CM.kv_bytes(c)
        self.lst = self.kv + -(-self.kvb // 4096) * 4096
        self.buf = allocate(shape=(self.lst + rows.nbytes,), dtype=np.uint8)
        self.buf[:self.m.raw.size] = self.m.raw
        self.buf[self.io:self.lst] = 0
        self.buf[self.lst:] = np.frombuffer(rows.tobytes(), np.uint8)
        self.buf.flush()
        base = self.buf.physical_address
        self.bases = [base, base + self.io, base + self.kv]
        self.list_addr = base + self.lst
        self.pos = 0
        self.stats = []                         # per token: (pos, cycles, wall seconds)

    def reset(self):
        """Start a new sequence: empty KV cache, pos 0."""
        self.buf[self.kv:self.kv + self.kvb] = 0
        self.buf.flush()
        self.pos = 0

    def forward(self, token, perf=False):
        """Logits (float32 copy) of token at the current position; advances pos."""
        c = self.cfg
        if self.pos >= c.seq_len:
            raise ValueError(f"sequence full ({c.seq_len} positions)")
        self.buf[self.io:self.io + 8] = np.frombuffer(CM.arg_block(self.pos, token), np.uint8)
        self.buf.flush()
        t0 = time.perf_counter()
        rec = self.dev.wait(self.dev.submit(self.list_addr, bases=self.bases, perf=perf,
                                            params=CM.token_params(self.pos, self.m.d, c.head_size)), timeout=10.0)
        wall = time.perf_counter() - t0
        if rec["status"]:
            raise RuntimeError(f"pos {self.pos}: list failed, status {rec['status']:#x}")
        self.buf.invalidate()
        lo = self.io + CM.IO_LOGITS
        logits = np.array(self.buf[lo:lo + 4 * c.vocab]).view(F).copy()
        self.stats.append((self.pos, rec["cycles"], wall))
        self.pos += 1
        return logits

    def generate(self, prompt, steps=256, temperature=0.0, topp=0.9, seed=1, stop_at_bos=True, forced=None):
        """Yield (token, text piece) per generated token, run.c style: the prompt
        tokens are fed first, then sampled tokens until `steps` positions (or BOS).
        forced: a token sequence to feed instead (teacher forcing; yields the
        device's argmax at each position instead of text)."""
        self.reset()
        toks = forced if forced is not None else self.tok.encode(prompt)
        sample = Sampler(temperature, topp, seed)
        token = toks[0]
        for pos in range(min(steps, self.cfg.seq_len)):
            logits = self.forward(token)
            if forced is not None:
                yield int(np.argmax(logits)), logits
                if pos + 1 >= len(forced):
                    return
                token = forced[pos + 1]
                continue
            nxt = toks[pos + 1] if pos + 1 < len(toks) else sample(logits)
            if nxt == BOS and stop_at_bos:
                return
            yield nxt, self.tok.decode([token, nxt])[len(self.tok.decode([token])):]
            token = nxt

    def close(self):
        self.buf.freebuffer()
        self.dev.close()


class BatchLlamaDevice:
    """L5b: D sequences per step (compile_batch.BatchCompiler; plan §11.1).
    Slot b has its own KV cache and position; step() runs one token of every
    slot. A slot without a sequence runs a dummy token at pos 0 (its KV row 0
    is rewritten when a sequence starts in it)."""
    DUMMY = BOS

    def __init__(self, model_path, tokenizer_path=None, bitfile=None, firmware=None, use_irq=True, device=None):
        import compile_batch as CB
        from pynq_matmul import Device, allocate
        self.CB = CB
        self.dev = device or Device(bitfile, firmware, ring_size=16, use_irq=use_irq)
        self.m = W8A8(model_path)
        c, d = self.cfg, self.d = self.m.cfg, self.m.d
        if d != self.dev.d:
            raise ValueError(f"{model_path} is packed for D = {d}, the overlay has D = {self.dev.d}")
        self.tok = Tokenizer(tokenizer_path or os.path.join(os.path.dirname(model_path), "tokenizer.bin"), c.vocab)
        self.dl = CB.BatchCompiler(self.m).build()
        rows = self.dl.array()
        self.io = -(-self.m.raw.size // 4096) * 4096
        self.kv = self.io + -(-CB.io_bytes(c, d) // 4096) * 4096
        self.kvb = CM.kv_bytes(c)
        self.lst = self.kv + d * self.kvb
        self.buf = allocate(shape=(self.lst + rows.nbytes,), dtype=np.uint8)
        self.buf[:self.m.raw.size] = self.m.raw
        self.buf[self.io:self.lst] = 0
        self.buf[self.lst:] = np.frombuffer(rows.tobytes(), np.uint8)
        self.buf.flush()
        base = self.buf.physical_address
        self.bases = [base, base + self.io, base + self.kv]
        self.list_addr = base + self.lst
        self.stats = []                         # per step: (cycles, wall seconds)

    def clear_slot(self, b):
        a = self.kv + b * self.kvb
        self.buf[a:a + self.kvb] = 0
        self.buf.flush()

    def step(self, poses, tokens, perf=False):
        """One token for each of the D slots; returns the D x vocab logits (copy)."""
        c, d = self.cfg, self.d
        t = self.CB.arg_table(poses, tokens, d, c.head_size)
        self.buf[self.io:self.io + len(t)] = np.frombuffer(t, np.uint8)
        self.buf.flush()
        t0 = time.perf_counter()
        rec = self.dev.wait(self.dev.submit(self.list_addr, bases=self.bases, perf=perf), timeout=20.0)
        wall = time.perf_counter() - t0
        if rec["status"]:
            raise RuntimeError(f"step failed, status {rec['status']:#x}")
        self.buf.invalidate()
        lo = self.io + self.CB.IO_LOGITS
        logits = np.array(self.buf[lo:lo + 4 * d * c.vocab]).view(F).reshape(d, c.vocab).copy()
        self.stats.append((rec["cycles"], wall))
        return logits

    def generate(self, prompts, steps, stagger=0, temperature=0.0, topp=0.9, seed=1, on_logits=None):
        """Generate for len(prompts) <= D sequences at once; sequence b starts at
        step b * stagger and runs `steps` positions (prompt tokens fed first,
        then sampled; BOS does not stop it). on_logits(b, pos, logits) sees every
        position. Returns the token lists."""
        d = self.d
        n = len(prompts)
        if n > d:
            raise ValueError(f"at most {d} sequences")
        toks = [self.tok.encode(p) for p in prompts]
        samplers = [Sampler(temperature, topp, seed + b) for b in range(n)]
        out = [[t[0]] for t in toks]
        start = [b * stagger for b in range(n)]
        for b in range(d):
            self.clear_slot(b)
        total = max(start) + steps if n else 0
        for s in range(total):
            poses, cur, live = [0] * d, [self.DUMMY] * d, [False] * d
            for b in range(n):
                p = s - start[b]
                if 0 <= p < steps:
                    poses[b], cur[b], live[b] = p, out[b][p], True
            logits = self.step(poses, cur)
            for b in range(n):
                if not live[b]:
                    continue
                p = poses[b]
                if on_logits:
                    on_logits(b, p, logits[b])
                if p + 1 < steps:
                    out[b].append(toks[b][p + 1] if p + 1 < len(toks[b]) else samplers[b](logits[b]))
        return out

    def close(self):
        self.buf.freebuffer()
        self.dev.close()

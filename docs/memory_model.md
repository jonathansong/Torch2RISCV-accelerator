# Memory model (overlay)

Three bus masters share the PS DDR; only the ARM goes through the CPU caches.
This page was written for Phase 3 (`matmul_unit`). It was then extended for
the double-buffered accelerator `sa_unit` (M1–M5: `rtl/sysarray`,
`docs/double_buffer_design.md`), which keeps the same address map, and for
the LLM levels L1–L5b (`docs/llm_inference_plan.md`): the command ring, the
notify interrupt, and the runtime's buffers.

## Address maps

| Region | ARM (PS7 M_AXI_GP0) | PicoRV32 (`mem_axi`) | accelerator DMA (`m0_axi`) |
|---|---|---|---|
| PS DDR, 512 MB | `0x0000_0000`–`0x1FFF_FFFF` (Linux, cached) | `0x0000_0000`–`0x1FFF_FFFF` via S_AXI_HP0 | `0x0000_0000`–`0x1FFF_FFFF` via S_AXI_HP2 |
| Program BRAM, 8 KB used | `0x4001_0000` (64 KB window) | `0xC000_0000` (reset vector) | — |
| Performance counter area (256 B below the mailbox) | `0x4001_1E00` | `0xC000_1E00` | — |
| Mailbox (last 256 B of BRAM) | `0x4001_1F00` | `0xC000_1F00` | — |
| accelerator CSRs | — | `0x8000_0000` (4 KB): Phase 2 CSR map, CAPS 0x24, EXT_STATUS 0x28, PERF_CTRL 0x3C, counters 0x40–0xBC, descriptor status 0xC0–0xD0, NOTIFY_COUNT 0xD4 (L1) | — |
| notify interrupt (L1) | `matmul_0/notify_irq` → `irqConcat/In1` → AXI interrupt controller → IRQ_F2P (PYNQ `Interrupt("matmul_0/notify_irq")`) | raised by `mat_notify` | — |
| RISC-V clock (clk_wiz DRP) | `0x4000_1000` | — | — |
| AXI interrupt controller | `0x4002_0000` | — | — |
| RISC-V reset | PS GPIO EMIO[0], 1 = hold | — | — |

- DDR is **identity-mapped** for all three masters: a `pynq.allocate` buffer's
  `physical_address` is usable unchanged by the firmware and by the accelerator
  DMA (LD / ST engines and the descriptor fetch unit, all through HP2).
- The RISC-V can reach all of DDR, including memory Linux owns. Firmware must
  only touch buffers the ARM allocated and passed through the mailbox.
- PicoRV32, its interconnect, the accelerator and both HP ports all run on
  `riscv_clk` (clk_wiz, 50 MHz): no clock-domain crossings on these paths.
  The accelerator shares the RISC-V peripheral reset, so holding the RISC-V
  in reset also resets it.
- Firmware (code, data and stack) must stay below the performance counter
  area: < 0x1E00 bytes (`firmware/common/link.ld`, checked by
  `load_firmware()`). After its measurement window the firmware copies the
  counters to 0x1E00-0x1EFF and their number to `MBOX_PERF_COUNT`.
- The ARM cannot reach the accelerator CSRs; only PicoRV32 maps them. That
  is why the host notification is an edge interrupt (`mat_notify`) and not a
  status bit the ARM clears.

## Coherency

HP0/HP2 are not coherent with the ARM L1/L2 caches. The driver
(`driver/pynq_matmul.py`) follows the conservative protocol from the plan:

1. ARM writes A/B into `pynq.allocate` buffers, then `flush()` every buffer
   the PL will read **or write** (C is flushed too, so no dirty line can be
   evicted over the accelerator's results later).
2. ARM releases the RISC-V; the PL reads A/B and writes C.
3. After the firmware reports done, ARM `invalidate()`s C before reading it.

The BRAM mailbox is device memory on the ARM side (MMIO), so it needs no
cache maintenance.

Descriptor lists (M5) are data the PL reads: the driver writes them into a
`pynq.allocate` buffer and `flush()`es it before the firmware submits the
list, like A and B.

### Command ring (L1)

The resident runtime firmware `firmware/rt` serves a submission ring in DDR.
The mailbox holds its fields: `RING_BASE` / `RING_SIZE` / `RING_TAIL` /
`RING_HEAD` at 0xB0–0xBC, `CPL_BASE` 0xC0, and `FW_STATE` / `FW_VERSION` /
`HEARTBEAT` at 0xC4–0xCC. The driver's `Device` holds three `pynq.allocate`
buffers:

| Buffer | Content | Cache maintenance |
|---|---|---|
| ring | 64-byte entries: {type, flags, seq}, list address, count, BASE0–3, parameter block address | `flush()` after writing an entry, then the doorbell (`RING_TAIL`, mailbox MMIO) |
| parameter blocks | PARAM0–7 of an entry, 32 bytes per slot | `flush()` with the entry |
| completion records | 32 bytes per entry: seq, status, cycles, descriptors decoded, END value | `invalidate()` before reading, once `RING_HEAD` has passed the entry |

A slot (entry plus completion record) is reused only after the ARM has read
that slot's record. `submit()` collects finished records while the ring is
full. Counting `RING_HEAD` alone would let the firmware overwrite records
the ARM has not read yet; a board test found exactly this bug.

### Relocation and the LLM runtime's buffers (L3–L5b)

The DDR fields of LD / ST / LDPARAM descriptors can be relocated by a BASE
register: address = field + BASE[n]. The ring entry sets BASE0–3, so a list
built once is position-independent. `llm/runtime.py` (`LlamaDevice`,
`BatchLlamaDevice`) uses one CMA buffer per model:

| Region (BASE) | Content | Written by |
|---|---|---|
| model (BASE0) | the `.w8a8` file (`llm/export_w8a8.py`) | ARM once, then read-only |
| io (BASE1) | argument block (pos, token), or the per-sequence argument table (L5b); logits; the L5b xq staging area | ARM (arguments), device (the rest) |
| kv (BASE2) | int8 KV cache: layer l at l · 2 · seq_len · kv_dim, K then V; L5b: one cache per sequence | device; the ARM clears it when a sequence starts |
| list | the static token list (`llm/compile_model.py` / `compile_batch.py`) | ARM once |

Per token the ARM writes the argument block and `flush()`es it, submits the
list, waits for the interrupt, then `invalidate()`s and reads the logits.
Rows of the KV cache beyond the current position must be zero, which is why
the cache is cleared at the start of a sequence.

## Ordering inside the accelerator

The accelerator's scoreboard orders commands only through the on-chip
memories (SPAD / ACC banks). **DDR is not tracked**: a store and a later load
of the same DDR bytes can overlap. Software separates them with `mat_fence`
(PCPI path) or a FENCE descriptor / the FENCE_BEFORE flag (descriptor
lists). The GEMM and vector schedules never reload what they stored, so
they need none. The LLM lists do reload, and they fence in two places:

- after appending this token's K / V rows (ST) and before attention reads
  the cache back (LD);
- in L5b, between storing the quantized activation matrix and reloading it
  with LD INTERLEAVE as the A strip.

LDPARAM also reads DDR, from the fetch unit and at fetch time. It only reads
data the ARM wrote before submitting (argument blocks, model constants), so
it needs no fence.

## Alignment rules

**Legacy 8×8×8 jobs** (CSRs, funct7 = 0): A/B/C addresses must be 8-byte
aligned and each 64 B (A, B) or 256 B (C) tile must not cross a 4 KB page.
The batch firmware places job *i* at `base + 64·i` / `base + 256·i`, so a
256 B aligned base (always true for `pynq.allocate`, which is page-aligned)
satisfies this; the driver checks it.

**New ISA** (funct7 = 1 / 2): DDR addresses, row bytes and pitches of
`mat_load` / `mat_store` must be multiples of 8; the DMA splits bursts at
4 KB boundaries itself, so transfers may cross pages. Descriptor lists and
JUMP targets must be 64-byte aligned (a descriptor never crosses 4 KB).
Local addresses and ranges are checked by the scheduler (error codes 1 / 2).

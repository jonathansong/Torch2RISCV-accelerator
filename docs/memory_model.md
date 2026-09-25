# Memory model (overlay)

Three bus masters share the PS DDR; only the ARM goes through the CPU caches.
Written for Phase 3 (`matmul_unit`), extended for the double-buffered
accelerator `sa_unit` (M1–M5: `rtl/sysarray`, `docs/double_buffer_design.md`),
which keeps the same address map.

## Address maps

| Region | ARM (PS7 M_AXI_GP0) | PicoRV32 (`mem_axi`) | accelerator DMA (`m0_axi`) |
|---|---|---|---|
| PS DDR, 512 MB | `0x0000_0000`–`0x1FFF_FFFF` (Linux, cached) | `0x0000_0000`–`0x1FFF_FFFF` via S_AXI_HP0 | `0x0000_0000`–`0x1FFF_FFFF` via S_AXI_HP2 |
| Program BRAM, 8 KB used | `0x4001_0000` (64 KB window) | `0xC000_0000` (reset vector) | — |
| Performance counter area (256 B below the mailbox) | `0x4001_1E00` | `0xC000_1E00` | — |
| Mailbox (last 256 B of BRAM) | `0x4001_1F00` | `0xC000_1F00` | — |
| accelerator CSRs | — | `0x8000_0000` (4 KB): Phase 2 CSR map, CAPS 0x24, EXT_STATUS 0x28, PERF_CTRL 0x3C, counters 0x40–0xBC, descriptor status 0xC0–0xD0 | — |
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

## Ordering inside the accelerator

The accelerator's scoreboard orders commands only through the on-chip
memories (SPAD / ACC banks). **DDR is not tracked**: a store and a later load
of the same DDR bytes can overlap. Software separates them with `mat_fence`
(PCPI path) or a FENCE descriptor / the FENCE_BEFORE flag (descriptor
lists); the GEMM and vector schedules never reload what they stored, so
they need none.

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

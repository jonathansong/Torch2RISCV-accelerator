# Memory model (Phase 3 overlay)

Three bus masters share the PS DDR; only the ARM goes through the CPU caches.

## Address maps

| Region | ARM (PS7 M_AXI_GP0) | PicoRV32 (`mem_axi`) | matmul_unit (`m_axi`) |
|---|---|---|---|
| PS DDR, 512 MB | `0x0000_0000`–`0x1FFF_FFFF` (Linux, cached) | `0x0000_0000`–`0x1FFF_FFFF` via S_AXI_HP0 | `0x0000_0000`–`0x1FFF_FFFF` via S_AXI_HP2 |
| Program BRAM, 8 KB used | `0x4001_0000` (64 KB window) | `0xC000_0000` (reset vector) | — |
| Performance counter area (256 B below the mailbox) | `0x4001_1E00` | `0xC000_1E00` | — |
| Mailbox (last 256 B of BRAM) | `0x4001_1F00` | `0xC000_1F00` | — |
| matmul CSRs | — | `0x8000_0000` (4 KB) | — |
| RISC-V clock (clk_wiz DRP) | `0x4000_1000` | — | — |
| AXI interrupt controller | `0x4002_0000` | — | — |
| RISC-V reset | PS GPIO EMIO[0], 1 = hold | — | — |

- DDR is **identity-mapped** for all three masters: a `pynq.allocate` buffer's
  `physical_address` is usable unchanged by the firmware and by the matmul DMA.
- The RISC-V can reach all of DDR, including memory Linux owns. Firmware must
  only touch buffers the ARM allocated and passed through the mailbox.
- PicoRV32, its interconnect, matmul_unit and both HP ports all run on
  `riscv_clk` (clk_wiz, 50 MHz): no clock-domain crossings on these paths.
  matmul_unit shares the RISC-V peripheral reset, so holding the RISC-V in
  reset also resets the accelerator.
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

## Alignment rules (matmul_unit)

A/B/C addresses must be 8-byte aligned and each 64 B (A, B) or 256 B (C)
tile must not cross a 4 KB page. The batch firmware places job *i* at
`base + 64·i` / `base + 256·i`, so a 256 B aligned base (always true for
`pynq.allocate`, which is page-aligned) satisfies this; the driver checks it.

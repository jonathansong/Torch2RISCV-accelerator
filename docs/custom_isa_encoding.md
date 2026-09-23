# Custom matrix instructions (Phase 4)

The PicoRV32 hands every instruction it does not implement to its
co-processor port (PCPI). `rtl/matmul/matmul_pcpi.v` claims a small set of
R-type instructions in the RISC-V **custom-0** opcode space and drives the
matmul unit directly, bypassing its AXI4-Lite CSRs.

## Encoding

```
 31      25 24   20 19   15 14  12 11    7 6       0
+----------+-------+-------+------+-------+---------+
| funct7=0 |  rs2  |  rs1  |funct3|  rd   | 0001011 |   custom-0 (0x0B)
+----------+-------+-------+------+-------+---------+
```

| funct3 | Mnemonic | Operands | Semantics | Stalls the core |
|---|---|---|---|---|
| 0 | `mat_trigger` | rs1 = descriptor, rs2 = C | Start one 8×8×8 job; returns once the unit accepts it (fire-and-forget) | only while a previous job is still running |
| 1 | `mat_status` | rd | STATUS word | no |
| 2 | `mat_reset` | — | Clear done / error / IRQ status | until the unit is idle |
| 3 | `mat_wait` | rd | Wait for completion, rd = STATUS | until the unit is idle |
| 4 | `mat_cycles` | rd | Accelerator cycles of the last job (CYCLES) | no |

- `funct7` must be 0 and `funct3` ≤ 4. Anything else in custom-0 is not
  claimed, so the core's PCPI timeout turns it into an illegal-instruction
  trap (reserved for future extensions such as vector ops, via `funct7`).
- STATUS is the same word as the STATUS CSR: bit0 done, bit1 busy, bit2
  error, bits[11:8] error code (1 DIM, 2 address, 3 read response,
  4 write response).
- `mat_trigger`, `mat_reset` do not write `rd` (encode `rd = x0`).
- Instructions of the plan (§7.1) kept as specified: 0 trigger, 1 status,
  2 reset. `mat_wait` and `mat_cycles` were added so a kernel launch and its
  completion need no CSR access and no polling loop.

## Descriptor

PCPI supplies at most two source registers, but a job needs three
addresses. `rs1` therefore points at a 16-byte descriptor in DDR, which the
unit fetches itself (one 2-beat burst) — the same model as an NPU command
buffer prepared by the host:

| Offset | Field | |
|---|---|---|
| 0x0 | A | DDR address of A (8×8 int8, row-major, 8-byte aligned) |
| 0x4 | B | DDR address of B (8×8 int8) |
| 0x8 | DIM | `M \| N << 10 \| K << 20`; only 8×8×8 is accepted |
| 0xC | reserved | 0 |

The descriptor must be 16-byte aligned (otherwise error code 2). `rs2` is
the C address (8×8 int32, must not cross a 4 KB page). The descriptor's
values are loaded into SRC_A / SRC_B / DIM and rs2 into DST, so the CSRs
still show the last job for debugging.

## Execution model

- Queue depth 1: `mat_trigger` returns after acceptance, so the core can do
  other work while the unit runs. A second `mat_trigger` stalls (PCPI
  `pcpi_wait`) until the first job finishes.
- The CSR path (`CTRL.start`) and the instruction path share the datapath
  and checks; if both start in the same cycle, the CSR start wins and
  `mat_trigger` waits.
- Stalls are unbounded: a unit that never finishes (e.g. a hung bus) hangs
  the core in `mat_wait`. Recovery is the ARM holding the RISC-V in reset,
  which also resets the unit.

## Software

`firmware/include/sysarray_intrinsics.h` wraps each instruction in inline
assembly using the standard `.insn` directive (no compiler changes):

```c
mat_trigger((uint32_t)&desc[i], c_addr + 256 * i);   // .insn r 0x0B, 0, 0, x0, rs1, rs2
uint32_t st = mat_wait();                            // .insn r 0x0B, 3, 0, rd, x0, x0
```

`firmware/matmul_insn/matmul_insn_fw.c` is the Phase 4 test firmware; it
reports through the same mailbox as the CSR firmware (`firmware/matmul`),
so `driver/pynq_matmul.py` runs either one and the two paths can be
compared on identical data.

## Verification

- `rtl/matmul` `make sim`: all 32 golden cases through both the CSR and the
  PCPI path, `mat_status` while busy, trigger-while-busy stall, descriptor
  alignment / DIM errors, `mat_reset`, unclaimed encodings (funct3 = 5,
  funct7 ≠ 0, standard `mul`), CSR/PCPI interleaving; the PCPI driver
  checks that `pcpi_wait` keeps the core from timing out.
- `firmware/matmul_insn` `make sim`: the firmware on the real PicoRV32 RTL
  against the real unit — 32/32 match, **0 CPU accesses to the CSRs**.

# Local memory planning by lifetime (design, deferred)

Status: written down 2026-10-20, **not started, deferred** (§7). This is item 3
of [`iree_compiler_plan.md`](iree_compiler_plan.md) §8.17 ("lifetime-based
memory planning"), split into 3a (framework) and 3b (optimization) at the Z1
freeze and moved to the KV260's K2b compiler work items
([`kv260_upgrade_plan.md`](kv260_upgrade_plan.md)). It concerns how one
dispatch's buffers are placed in SPAD_A / SPAD_B / ACC. It does not concern the
traffic between dispatches, which is the cross-operator fusion's subject
([`fusion_design.md`](fusion_design.md)).

## 1. How local memory is allocated today

`sahl-to-sahw` (`SahlToSahw.cpp`, `SahlLowerGeneric.cpp`, the allocator in
`SahlLocalMemory.h`) places every local buffer while it lowers:

- **Bump allocation at first use.** A buffer gets the next free words of SPAD_A or
  of an ACC bank when the lowering first touches it (an ACC bank first if the op
  says `sa.prefer_bank`). Nothing is freed at a buffer's last use.
- **Release by scope.** A `sahl.scope` marks the allocator; at its end everything
  allocated inside is released (`keep`: the outer buffers stay visible; a piece of
  sahl-tile: only the reserved ones). `sahl.reserve` allocates a buffer before the
  pieces that share it. The generics before a fused kernel can be marked
  `sa.release_temps` (their results kept, their temps released after each).
- **Placement by the kernels.** A micro-kernel places its main buffers itself:
  `sa.word` (SPAD_A words), `sa.bank` (SPAD_B banks), `sahl.claim_spad` (SPAD_A
  words reserved, `sa.relative`: after what is already there).
- **When it does not fit**: sahl-tile splits a dispatch too large for ACC into
  pieces by an estimate of the lowering's needs (`pieceWords`, an upper bound: every
  buffer, a temp per fp32 operation); `SaCodegen` retries a dispatch that fails with
  "ACC full" with smaller tiles (`tile_shrink`); the attention kernels fall back to
  an in-order form.
- `sahl-plan-memory` only chooses each buffer's memory and layout (`sa.mem`,
  `sa.layout`: packed, broadcast, rows), not its address.

This is simple and correct: addresses never overlap while a buffer is live within
its scope. Its cost is space. A buffer's words stay taken until its scope ends,
however short its life.

## 2. Where it got in the way

- **The prefill chunk kernel** (fusion_design.md §6.7). The 16 rows' q paths in one
  dispatch kept every row's temporaries (RMSNorm, RoPE, quantization: several
  `[16, 128]` fp32 buffers a row), because they are allocated at the function's top
  level. The kernel works around it by re-emitting each row's q path in a scope of
  its own. A buffer read across rows (a filled reduction init) had to be cloned per
  row, and a `sahl.bcast`'s result, allocated where it is lowered, vanished at the
  end of the row's scope (the s_q of each row first copied out to a reserved
  buffer).
- **The decode attention kernel** (§6.4.6): the q path's generics had to be marked
  `sa.release_temps`. The RoPE gather's per-element broadcasts alone took about 2048
  ACC words.
- **Llama's tiled attention form** (§6.4.8) falls back to the in-order schedule via
  the ACC-full retry: two sets of G = 4 rows' buffers do not fit as allocated.
- **The generic contraction** replays the allocator to learn where x landed in ACC
  (`expandContraction`, E3a): it works only when the kernel is the dispatch's first
  command-issuing op. Planning in the IR removes that coupling.
- **sahl-tile's estimate** counts `sahl.to_i8` as the 12 operations it replaced, kept
  only so that the golden corpus stays byte-identical (§8.14 R3).

None is a correctness problem today; each is handled by a scope, a reservation, a
retry or a fallback, with some effort per kernel and some lost overlap.

## 3. The plan

**3a, framework (output unchanged).** A planner pass after `sahl-expand-kernels`:
- a lifetime analysis over the expanded `sahl` IR (first and last use of every
  local buffer, through views, nested scopes and loops; a buffer used inside a
  `sahl.loop` lives through the loop);
- it produces the placement: `sa.word` / `sa.bank` / ACC words, and the scopes;
- the first version reproduces today's allocation exactly (golden corpus
  identical), so the switch can be verified byte for byte;
- the constraints as a replaceable layer read from `TargetConfig`: SPAD / ACC
  sizes, banks, alignment, read ports, the chunk scratch at the start of each ACC
  bank;
- `expandContraction`'s replay of the allocator removed.

**3b, optimization (output changes).**
- reuse by lifetime (a buffer's words free after its last use);
- bank assignment against the scoreboard's conflicts (EX and VE conflict on any
  common SPAD_B / ACC bank, `double_buffer_design.md` §7; fusion_design.md §6.5);
- tiling decided by the plan when something does not fit, replacing sahl-tile's
  estimate and the ACC-full retry;
- accepted by `dispatch_check` (`--dirty`), the end-to-end bit-exact runs, and the
  board's per-dispatch cycles (addresses and banks change the scoreboard's overlap:
  faster or slower, to be measured).

## 4. The prerequisite: temporaries the IR does not show

A planner sees only the buffers in the IR (`memref.alloc`). The lowering also
allocates words of its own while it emits the VE commands (`newLocal()`), which no
IR value names. These must become IR buffers first, or the planner must reserve
them by a rule. As of 2026-10-20, 28 lines that call it (some allocate several
buffers), besides `bufOf`'s allocation of IR buffers at first use:

| Where | What | How many sites |
|---|---|---|
| `SahlLowerGeneric.cpp` `generic` | a temp per arithmetic operation of a body (an fp32 vector of the generic's length) | 2 (the general case, and the `select` of a mask) |
| `generic` | a reduction's temp words for `sahl.accumulate` (one a row) | 1 |
| `generic` | scalar inputs: a broadcast word of ones and the scalar's word (push constants, an i64 scalar used as a value) | 4 |
| `generic` | a gathered row (`sahl.gather` "row"), a broadcast-layout temp, the int8 output's staging, a fresh result buffer | 4 |
| `packedRowGather` | the packed gather of an embedding row: the token's index words, one-hot prefixes, the tile, the reduction, the result row | 8 |
| `intScalar`, `scalarI64` | an i64 scalar's computation and its 8 bytes loaded into a word | 2 |
| `SahlToSahw.cpp` `scalarByParam`, `scalarBcast` | a scalar broadcast over a word | 2 |
| `perElementBcast` | a per-row operand in the broadcast layout: the words replicated D times, then transposed | 2 |
| `packBcast` | a broadcast-layout buffer packed back (D elements a word) | 1 |
| `materialize` | a DDR buffer read inside a body, loaded whole | 1 |
| `sahl.iota` | its lane counts and prefixes (3 words) | 1 |

The broadcasts and packs are also cached across ops (`bcastOf`, `locals`), so their
life is longer than one op; and a few depend on the generic's shape and are only
known while it is lowered. Two ways in:
- as IR: the lowering's preparatory steps become `sahl` ops before planning
  (e.g. `sahl.bcast` for every per-row operand, a scratch operand on the ops that
  need temps), so their buffers are allocations the planner places;
- as a rule: the planner reserves, per op, an upper bound of its temps (what
  sahl-tile's estimate does today), placed by the plan and released after the op.
The first is cleaner and larger; the second keeps the lowering and wastes some words.

## 5. Effort, risk, gain

- **Effort**: 3a about 2–3 weeks (the temps of §4, the lifetime analysis, the
  constraints layer, the byte-for-byte reproduction of today's placement over the
  golden corpus of 8562 dispatches). 3b another 1–2 weeks plus board measurements.
- **Risk**: 3b changes addresses and banks everywhere, so every model's bundles
  are to be rebuilt and measured; the scoreboard's overlap may get worse in places
  (a bank split differently). 3a alone changes nothing.
- **Gain**: mostly structural. Kernels stop needing hand-placed scopes,
  reservations and retries; larger fused kernels fit (the fallbacks of §2 go away);
  the cost model and the schedule can see the memory. Direct speed for today's
  models is expected to be small: decode is bound by the GEMV's weight streaming,
  and the fallbacks of §2 occur in a few kernels only (not measured separately).

## 6. Verification, when it is done

- 3a: lit for the planner; the golden corpus byte-identical; `golden.py` reports
  any change of address.
- 3b: `dispatch_check.py --dirty` (local memories start with garbage: a planner that
  reuses words must not read unwritten ones) at the T of the block boundaries and
  `--pos 0`; logits end to end on the sim bit-exact with today's builds; the board's
  `run_board_tests.sh` and per-dispatch cycles (`profile`) against the current
  bundles.

## 7. Decision (2026-10-20)

Deferred. The compiler is complete and bit-exact for the models it supports; the
allocation's limits have workarounds in every kernel that met them, and the
expected speed gain is small next to the larger levers (W4A8, the hardware side;
fusion_design.md §7). To be taken up when a new kernel or model needs memory the
current scheme cannot give it, or before a larger fusion.

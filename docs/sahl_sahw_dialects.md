# SAHL and SAHW Dialects

This document explains the two target-specific MLIR dialects used by the SA
compiler backend:

- **SAHL** (`sahl`): tile-level operations with explicit transfers between
  DDR and local memories.
- **SAHW** (`sahw`): command-level operations describing accelerator
  registers and commands.

The authoritative operation definitions are:

- `compiler/plugins/sa/dialect/SahlOps.td`
- `compiler/plugins/sa/dialect/SahwOps.td`

## 1. Position in the compiler pipeline

After IREE forms and bufferizes a dispatch, the SA backend lowers it through
the following layers:

```text
IREE Flow/HAL dispatch
        |
        v
linalg + arith + memref
        |
        | sa-to-sahl
        v
SAHL
  - explicit DDR <-> local transfers
  - tile and micro-kernel structure
  - local-buffer lifetime and layout
        |
        | tiling, memory planning, scheduling,
        | kernel expansion, sahl-to-sahw
        v
SAHW
  - BASE and PARAM registers
  - concrete local addresses
  - LD, ST, EX, VE, LOOP_END, ...
        |
        +---------------------------+
        |                           |
        v                           v
descriptor serialization       LLVM IR generation
        |                           |
        v                           v
command-fetch unit             RV32IMC code
                                    |
                                    v
                              PicoRV32 + PCPI
```

SAHL retains enough structure for compiler decisions such as tiling, fusion,
scheduling, and local-memory allocation. SAHW is close to the hardware: with
the exception of structural and register-definition operations, each SAHW
operation represents one accelerator command.

## 2. Memory model

The accelerator has three principal local-memory spaces:

| Memory | Purpose |
|---|---|
| SPAD_A | Activations, A strips, packed vectors, and local `i8` data |
| SPAD_B | Packed B/weight tiles used by the systolic array |
| ACC | `i32` accumulators and local `f32`/integer vector data |

SAHL refers to local storage through `memref` values. Memory-planning passes
decide where each local allocation resides.

SAHW uses concrete 32-bit local word addresses:

```text
bits [31:28]  memory kind (SPAD_A, SPAD_B, or ACC)
bits [27:0]   word address within that memory
```

DDR objects originate from IREE binding subspans and views derived from them.
At SAHW level, a DDR address is either absolute or relative to a `!sahw.base`
value.

# Part I: SAHL

## 3. SAHL overview

SAHL is the tile level of the backend. After bufferization, ordinary linalg
operations can access binding-backed DDR memrefs directly. `sa-to-sahl` makes
that traffic explicit:

```text
DDR input --sahl.load--> local buffer
                              |
                              v
                       local computation
                              |
                              v
DDR output <--sahl.store-- local buffer
```

Element-wise computation generally remains as `linalg.generic`. During
SAHL-to-SAHW lowering, the arithmetic inside the generic is translated to
vector-engine commands.

## 4. SAHL data-transfer operations

### 4.1 `sahl.load`

```mlir
sahl.load %src, %dst : memref<...>, memref<...>
```

Copies the whole source view from DDR into a local buffer. Depending on the
planned layout, it normally becomes a linear or interleaved `sahw.ld`.

### 4.2 `sahl.store`

```mlir
sahl.store %src, %dst : memref<...>, memref<...>
```

Copies a complete local view back to DDR. It normally becomes `sahw.st`.

### 4.3 `sahl.gather`

```mlir
%value = sahl.gather "row" %table[%i, %j]
    : memref<...> -> f32
```

Represents a data-dependent read inside a `linalg` body. The indices are
computed from the body's iteration indices and, where applicable, an `i64`
scalar input. The `kind` records the implementation form selected by
`sa-to-sahl`:

| Kind | Meaning |
|---|---|
| `row` | `index = scalar * C + point`; load one dynamically selected row |
| `scalar` | `index = scalar * C`; read one element and broadcast it |
| `packed` | Read row `r` from packed weights `Wp[r / D, k, r % D]` |
| `swap` | Read `point ^ 1`, used for the pair swap in RoPE |
| `none` | No supported form was found; lowering reports the reason |

Depending on the form, lowering uses dynamic DMA, `LDPARAM`, local
materialization, or vector-engine indexing.

### 4.4 `sahl.scatter`

```mlir
sahl.scatter %updates, %indices, %dest
    : memref<...>, memref<...>, memref<...>
```

Writes one or more rows into a dynamically selected DDR location:

```text
dest[index + r * row_stride, ...] = updates[r, ...]
```

Its main use is updating the K/V cache at the current token position. It
lowers to stores with dynamically computed DDR addresses.

## 5. SAHL conversion and vector-layout operations

### 5.1 `sahl.to_i8`

```mlir
%result = sahl.to_i8 %x
```

Represents the quantization sequence:

```text
f32 x
  -> NaN to zero and infinities to finite limits
  -> round to nearest even
  -> clamp to [-127, 127]
  -> convert to i8
```

`sa-to-sahl` replaces the matched arithmetic chain with this operation inside
a linalg body. The vector engine performs the conversion as part of its output
stage.

### 5.2 `sahl.bcast`

```mlir
sahl.bcast %src, %dst : memref<...>, memref<...>
```

Converts a scalar or vector to broadcast layout. In packed layout, a word
contains `D` distinct elements. In broadcast layout, each word repeats one
element across all lanes:

```text
word 0 = [x0, x0, ..., x0]
word 1 = [x1, x1, ..., x1]
```

This layout is used by row-wise scales and other operands consumed with the
vector engine's divide-index mode.

### 5.3 `sahl.iota`

```mlir
sahl.iota %dst : memref<...xf32>
```

Creates the local FP32 vector `dst[i] = i`. It is used to construct masks
arithmetically, particularly causal attention masks during prefill.

### 5.4 `sahl.splat`

```mlir
sahl.splat %dst {value = 0.0 : f32} : memref<...>
```

Sets every element of a local SPAD or ACC buffer to a constant FP32 value.
Typical uses include accumulator initialization and reduction identities. It
lowers to vector-engine copy/fill commands.

### 5.5 `sahl.scalar_f32`

```mlir
sahl.scalar_f32 %src, %dst : memref<i64>, memref<...xf32>
```

Reads an `i64` scalar from DDR, converts it to FP32, and places the result in
one broadcast word. Examples include the current token position and sequence
length.

### 5.6 `sahl.transpose`

```mlir
sahl.transpose %rows, %tiles : memref<...>, memref<...>
```

Transforms row-major `R x C` data in SPAD_A into the tiled SPAD_B layout
expected by the systolic array. An important use is converting dynamic K-cache
rows into B tiles for attention score computation.

## 6. SAHL matrix operations

### 6.1 `sahl.strip`

```mlir
sahl.strip %src, %dst : memref<...>, memref<...>
```

Creates the A operand layout used by `sahl.mma`.

For prefill rows `x[D, K]`, interleaving produces:

```text
SPAD_A[word = k][lane = r] = x[r, k]
```

For decode input `x[K]`, each element is replicated over the active array
rows. With `sa.packed`, only the packed vector required by the streaming GEMV
unit is prepared.

### 6.2 `sahl.mma`

```mlir
sahl.mma %a, %b, %acc : memref<...>, memref<...>, memref<...>
```

Runs the systolic matrix engine:

```text
acc[D, nc, D] = A[D, K] x B[nc, K, D]
```

`a` is an A strip in SPAD_A, `b` contains packed tiles in SPAD_B, and `acc`
is `i32` data in ACC. The `accumulate` attribute selects `acc = A x B` versus
`acc += A x B`. It lowers to `sahw.ex`.

### 6.3 `sahl.gemv`

```mlir
sahl.gemv %x, %weights, %acc
    : memref<...>, memref<...>, memref<...>
```

Runs the streaming decode GEMV:

```text
acc[nc, D] = x[K] . weights[nc, K, D]
```

The packed input vector resides in SPAD_A. Weight strips stream directly from
DDR into the GEMV unit instead of first occupying SPAD_B. The result is written
to ACC. `accumulate` selects overwrite or accumulation. It lowers to
`sahw.ld` in GEMV mode.

## 7. SAHL structure and memory-lifetime operations

### 7.1 `sahl.kernel`

```mlir
sahl.kernel "linear" {
  ...
}
```

Groups operations that must be planned and lowered together. Supported kinds
include:

| Kind | Purpose |
|---|---|
| `linear` | Linear-layer micro-kernel |
| `attention` | Attention score or P-times-V micro-kernel |
| `contraction` | Generic contraction lowering |

The operation carrying `sahl.anchor` is the principal contraction. Loads,
extensions, fill, epilogue, and store operations remain in original order
around it. `sahl-expand-kernels` replaces the container with concrete SAHL
operations. A `sahl.kernel` that survives until `sahl-to-sahw` is unsupported
and causes that dispatch to use fallback handling.

### 7.2 `sahl.scope`

```mlir
sahl.scope {
  ...
}
```

Defines a local-memory lifetime. Buffers allocated inside the scope are
released at its end, allowing several tiled pieces to reuse the same SPAD/ACC
space.

- `keep` preserves buffers allocated before the scope.
- `spad_from` selects the starting SPAD_A word for allocations inside it.

### 7.3 `sahl.reserve`

```mlir
sahl.reserve %buffer : memref<...>
```

Allocates a local buffer before subsequent scopes so those scopes can share
it. The `bcast` attribute requests broadcast layout.

### 7.4 `sahl.loop`

```mlir
sahl.loop {count = 8, strides = [256, 16]} {
  ...
}
```

Describes a hardware loop. Operations carrying `sa.advance = j` advance their
DDR or local address by `strides[j]` each iteration. An operation after the
loop carrying `sa.advance_final = j` receives the final
`count * strides[j]` displacement.

An optional `dyn_count` obtains the runtime loop count from a push-constant
derived value; the static count remains the supported bound. It lowers to
`sahw.loop`, represented as `LOOP_END` in a descriptor list or normal control
flow in generated RISC-V code.

### 7.5 `sahl.claim_spad`

```mlir
sahl.claim_spad {words = 128}
```

Reserves the SPAD_A range `[0, words)` for fixed-placement buffers such as an
A strip or raw matrix rows. Automatic allocation begins after the claimed
range. `sa.relative` makes the claim relative to existing SPAD use in a fused
scope.

### 7.6 `sahl.cursor_reset`

```mlir
sahl.cursor_reset
```

Resets the device-side DDR row cursor. Dynamic row sizes such as `f(T)` cannot
always be represented as static view offsets, so cursor-addressed loads and
stores use:

```text
effective DDR address = base + cursor
cursor += current row size
```

`sa.skip` optionally advances the reset cursor by a number of rows.

# Part II: SAHW

## 8. SAHW overview

SAHW is the command level. It replaces memrefs and high-level computation with:

- explicit BASE and PARAM registers;
- concrete SPAD/ACC word addresses;
- accelerator command fields;
- optional dynamic command fields sourced from PARAM registers.

SAHW defines two SSA types:

```mlir
!sahw.base   // binding address plus an optional dynamic offset
!sahw.param  // a dynamic scalar/register value
```

These are SSA values until `sahw-assign-registers` assigns their physical
`reg` attributes.

## 9. SAHW template sections

### 9.1 `sahw.template`

```mlir
sahw.template "export_name"
    {bindings = 3, constants = 4, est_cycles = 1000} {
  ...
}
```

Represents one executable export. It contains register definitions and the
prefix/head/body sections.

| Attribute | Meaning |
|---|---|
| `export_name` | IREE export name |
| `bindings` | Number of buffer bindings |
| `constants` | Number of push constants |
| `est_cycles` | Estimated accelerator cycles |
| `unsupported` | Marks an export that cannot run on this target |

### 9.2 `sahw.prefix`

Contains loads that may be issued early, including during an earlier
dispatch. Stores are not allowed in a prefix.

### 9.3 `sahw.head`

Contains the body's leading loads. It can be called independently so that
another export's prefix can be scheduled between the head and the rest of the
body.

### 9.4 `sahw.body`

Contains the remaining commands and is required for every template.

## 10. SAHW register operations

### 10.1 `sahw.base`

```mlir
%base = sahw.base binding(0) {
  constant = 1,
  mul = 128,
  shift = 0,
  add = 64
} : !sahw.base
```

Computes:

```text
BASE = binding_address
     + ((push_constant[constant] * mul) >> shift)
     + add
```

When `constant = -1`, only the fixed `add` term is used. BASE values relocate
DDR offsets in LD, ST, and LDPARAM commands.

### 10.2 `sahw.param`

```mlir
%p = sahw.param {
  constant = 2,
  mul = 16,
  shift = 1,
  add = 4
} : !sahw.param
```

Computes a dynamic register value from a push constant:

```text
PARAM = ((push_constant[constant] * mul) >> shift) + add
```

PARAM values supply dynamic addresses, lengths, loop counts, strides, and
vector coefficients.

### 10.3 `sahw.private`

```mlir
%cursor = sahw.private : !sahw.param
```

Allocates a template-private PARAM register for loop offsets, row cursors,
LDPARAM results, or other temporary state. The RISC-V backend represents it as
a local slot initialized to zero.

## 11. Dynamic command fields

Most SAHW commands accept:

```mlir
dyn(%p0, %p1) {
  dyn_fields = [...],
  dyn_add = [...]
}
```

`dyn_fields[i]` chooses the command field controlled by PARAM `dyn[i]`.
`dyn_add[i]` chooses between:

```text
false: actual field = PARAM
true:  actual field = static field + PARAM
```

The descriptor format supports at most two dynamic fields per command. Typical
fields include DDR/local addresses, DMA sizes and pitch, EX repeat/steps, VE
length and coefficients, and loop count. `fence_before` requests synchronization
before the command.

## 12. SAHW command operations

### 12.1 `sahw.ld`

```mlir
sahw.ld %base dyn(%offset) {
  ddr = 0,
  laddr = 0x10000000,
  rows = 16,
  row_bytes = 128,
  pitch = 256,
  mode = 0
}
```

Transfers DDR data to local memory. The normal two-dimensional DMA form is:

```text
for r in [0, rows):
  copy row_bytes from DDR + ddr + r * pitch
                 to local + r * row_bytes
```

The modes are:

| Mode | Meaning |
|---|---|
| LINEAR | Ordinary linear/strided DMA |
| INTERLEAVE | Interleave DDR rows into SPAD_A lanes |
| GEMV | Stream weight strips from DDR directly into the GEMV unit |

GEMV mode additionally uses `xword`, `cstep`, and `gemv_acc` to identify the
SPAD_A vector, ACC result spacing, and accumulation behavior.

### 12.2 `sahw.st`

```mlir
sahw.st %base {
  ddr = 0,
  laddr = 0x30000000,
  rows = 1,
  row_bytes = 256,
  pitch = 256
}
```

Transfers local data to DDR using the same rows/row-bytes/pitch model as LD.

### 12.3 `sahw.ex`

```mlir
sahw.ex {
  a = 0,
  b = 0,
  c = 0,
  kt = 4,
  repeat = 8,
  bstep = 4,
  cstep = 1,
  crow = 0
}
```

Runs the systolic execution engine:

```text
int8 A strip x int8 B tiles -> int32 ACC
```

`accumulate` selects assignment versus accumulation. `repeat` performs several
neighboring tile computations, while `bstep` and `cstep` advance the B and C
addresses between repeats. `crow` selects or advances the ACC row.

### 12.4 `sahw.ve`

The vector-engine command implements the general form:

```text
y = FUNC(OP(s1', s2) * A + B)
```

It also supports indexed reads, reduction, validity masks, data conversion,
quantization, immediates, and clamping.

Arithmetic operations:

| Operation | Meaning |
|---|---|
| `VOP_ADD` | Addition |
| `VOP_SUB` | Subtraction |
| `VOP_MUL` | Multiplication |
| `VOP_MAX` | Maximum |
| `VOP_MIN` | Minimum |
| `VOP_COPY` | Copy or affine transform |

Functions:

| Function | Meaning |
|---|---|
| `FUNC_NONE` | No function |
| `FUNC_EXP` | Exponential |
| `FUNC_RECIP` | Reciprocal |
| `FUNC_RSQRT` | Reciprocal square root |
| `FUNC_ABS` | Absolute value |

Source index modes:

| Mode | Meaning |
|---|---|
| `IDX_LIN` | Linear access |
| `IDX_MOD` | Periodic/modulo access |
| `IDX_DIV` | Broadcast one value over a group of elements |
| `IDX_IMM` | Use an immediate operand |

Reduction modes are `RED_NONE`, `RED_SUM`, and `RED_MAX`. The command also
contains `scale`, `shift`, `zp`, `lo`, and `hi` fields for integer conversion
and quantization. Important shape/control fields include `length`, `period`,
`rowlen`, `valid`, and `p1`.

### 12.5 `sahw.transpose`

```mlir
sahw.transpose {
  src = ...,
  dst = ...,
  length = ...,
  types = ...,
  stride = ...
}
```

Transposes `D x D` blocks into the layout expected by the matrix engine.
Although represented as a distinct MLIR operation, it is serialized using the
vector descriptor with the TRANSPOSE operation code.

### 12.6 `sahw.ldparam`

```mlir
sahw.ldparam %target, %base {
  addr = 0,
  mul = 128,
  add = 16
}
```

Loads a 32-bit DDR value into a PARAM register:

```text
target = mem32[base + addr] * mul + add
```

It is used for memory-resident sequence lengths, indices, cursors, and other
runtime values.

### 12.7 `sahw.fence`

```mlir
sahw.fence {mask = 0}
```

Waits until the scheduler queue is empty and the engines selected by `mask`
are idle. `mask = 0` waits for all engines.

### 12.8 `sahw.setreg`

```mlir
sahw.setreg %p0, %p1 {
  values = [0, 128],
  add_mask = 2
}
```

Assigns up to three PARAM registers. Normally:

```text
target[i] = value[i]
```

If bit `i` of `add_mask` is set:

```text
target[i] += value[i]
```

Dynamic fields can source the assigned values from other PARAM registers.

### 12.9 `sahw.loop`

```mlir
sahw.loop 8, %p0 step 256, %p1 step 16 {
  ...
}
```

Executes the region `count` times and increments two PARAM registers after
each iteration. The descriptor backend emits the body followed by a
`LOOP_END` command. The RISC-V backend emits ordinary LLVM basic blocks,
counter updates, and a conditional branch.

## 13. SAHL-to-SAHW correspondence

| SAHL/input operation | Typical SAHW lowering |
|---|---|
| `sahl.load` | `sahw.ld` |
| `sahl.store` | `sahw.st` |
| `sahl.strip` | Interleaved `sahw.ld` or `sahw.ve` copy |
| `sahl.mma` | `sahw.ex` |
| `sahl.gemv` | `sahw.ld` in GEMV mode |
| `sahl.bcast` | `sahw.ve` |
| `sahl.iota` | A sequence of `sahw.ve` commands |
| `sahl.splat` | `sahw.ve` using `VOP_COPY` |
| `sahl.scalar_f32` | Load plus reduction-style `sahw.ve` conversion |
| `sahl.transpose` | `sahw.transpose` |
| `sahl.gather` | Dynamic LD, `sahw.ldparam`, or VE indexing |
| `sahl.scatter` | Dynamically addressed `sahw.st` |
| `sahl.loop` | `sahw.loop` / `LOOP_END` |
| `sahl.cursor_reset` | `sahw.setreg` |
| `linalg.generic` | One or more `sahw.ve` commands |
| local `memref.alloc` | A concrete SPAD/ACC word range |

## 14. Example lowering

A simplified decode linear layer can be viewed as:

```text
SAHL
  load x
  strip x
  gemv x, weights -> accumulator
  linalg.generic epilogue
  store result

SAHW
  BASE definitions
  LD x -> SPAD_A
  LD mode=GEMV weights -> ACC
  VE epilogue
  ST ACC/local result -> DDR
  FENCE
```

The same SAHW template has two equivalent backend representations.

Descriptor path:

```text
register setup
LD descriptor
LD-GEMV descriptor
VE descriptor
ST descriptor
FENCE descriptor
RET
```

RISC-V path:

```text
SAHW template
  -> LLVM function
  -> RV32IMC machine code
  -> PicoRV32
  -> PCPI matrix/vector/DMA commands
```

The descriptor path is compact and lets the command-fetch unit issue commands
efficiently. The RISC-V path provides normal control flow and more flexible
dynamic calculations. A recorded decode schedule may contain both generated
RISC-V calls and descriptor lists.

## 15. Summary

SAHL answers:

> Which tiles should be moved or computed, where should they live locally, and
> which operations belong to one scheduled kernel?

SAHW answers:

> What are the concrete registers, local addresses, command fields, loops, and
> synchronization operations sent to the accelerator?

Keeping these layers separate allows the compiler to perform high-level
tiling, fusion, and memory planning before committing to the exact hardware
command format. It also makes the same scheduled program usable by both the
descriptor-list backend and the LLVM IR/RISC-V backend.

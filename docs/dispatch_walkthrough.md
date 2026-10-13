# From an IREE dispatch to an sa-desc descriptor template

One real dispatch followed through the sa backend: what IREE hands over, every pass of the sa code generator, the sa-desc file it produces, and what the driver and the accelerator do with it at run time. Companion to [`iree_compiler_plan.md`](iree_compiler_plan.md) (§4 the sa-desc format, §8.3–§8.4 the two dialects and the pipeline, §8.20 the micro-kernel expansion) and [`execution_walkthrough.md`](execution_walkthrough.md) (the hardware side of the same descriptors).

The example is the attention scores `q · Kᵀ` of decode in SmolLM2-135M through the generic frontend (truncated to 2 layers, qsum build `build/hfgen/smol_l2qs`, D = 16, 4 GEMV ports):

```
build/hfgen/smol_l2qs/sa_sources/module_main$async_dispatch_19.mlir
```

All IR below is the compiler's real output (trimmed: repeated `layout(...)` annotations and constant-folding noise removed). [How to reproduce](#9-reproducing) is at the end.

---

## 1. What a dispatch is

IREE compiles a model into two programs:

- **device code**: the graph is cut into *dispatches*, each a group of fused operations executed as one kernel on a device. Each dispatch becomes an *executable*, compiled by a target backend: here the sa backend, which turns it into a descriptor-list template.
- **host code**: VM bytecode that runs on the ARM, allocates buffers and calls the dispatches in order (`hal.command_buffer.dispatch(executable, ordinal, constants, bindings)`), inserting barriers between them.

A SmolLM2 decode step calls about 1000 dispatches; the 30 layers share about 200 distinct executables (the same executable serves every layer, with per-layer offsets passed as constants).

## 2. The dispatch IREE hands over

### 2.1 The name

```
main$async_dispatch_19_batch_matmul_3x3xDx64_i32xi32xi64
```

| Part | Meaning |
|---|---|
| `main` | The exported function: decode (`prefill`, `prefill_kv` are the others) |
| `dispatch_19` | IREE's number |
| `batch_matmul` | The root operation |
| `3x3xDx64` | Its loop ranges, batch × M × N × K: 3 KV heads × 3 query heads per group × T × head_dim 64; `D` is a dynamic range (the attention length T) |
| `i32xi32xi64` | Element types: lhs, rhs, accumulator |

So the dispatch does 3 · 3 · T · 64 MACs (what `dispatch_stats.py` counts, §13.1 of the generality plan).

### 2.2 The file

```mlir
hal.executable public @main$async_dispatch_19 {
  hal.executable.variant public @sa_desc_v1 target(<"sa", "sa-desc-v1", {acc_bytes = 262144, bases = 16, d = 16,
      dyn_fields = 2, gemv_ports = 4, max_dynamic = 256, params = 8, spad_bytes = 131072}>) {
    hal.executable.export public @main$async_dispatch_19_batch_matmul_3x3xDx64_i32xi32xi64 ordinal(0)
        layout(#hal.pipeline.layout<constants = 5, bindings = [
            #hal.pipeline.binding<storage_buffer, ReadOnly>,
            #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
            #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>)
        count(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module { func.func @main$async_dispatch_19_batch_matmul_3x3xDx64_i32xi32xi64() { ... } }
  }
  hal.executable.variant public @vmvx_bytecode_fb target(<"vmvx", "vmvx-bytecode-fb", ...>) { ...the same body... }
}
```

| Level | Content |
|---|---|
| `hal.executable` | One executable per dispatch |
| `variant` | One per target format: `sa-desc-v1` for the accelerator, `vmvx-bytecode-fb` for the host fallback (interpreted on the ARM when sa cannot compile it, iree_compiler_plan §8.15) |
| `target` config | The hardware parameters, the compiler's `TargetConfig`: array size D, SPAD / ACC bytes, GEMV ports, dynamic fields per descriptor, number of BASE / PARAM registers, upper bound of dynamic sizes |
| `export` layout | The calling interface: 5 push constants (i32), 3 bindings (storage buffers) |
| `count` | The workgroup count; the sa backend sets it to (1, 1, 1): one dispatch = one template, executed once |
| `func.func` | The computation: linalg / tensor / arith only, no kernel implementation |

### 2.3 The body

```mlir
// push constants: three byte offsets into the bindings, and T as two i32 halves
%0..%4 = hal.interface.constant.load ordinal(0..4) : i32
%12 = index(%3 | %4 << 32)                                         // T
%13:4 = util.assume.int %5[<umin = 11136, umax = 11136>, <umin = 76800, ...>, ...], ..., %12<umin = 0, ...>
// views of the bindings
%14 = binding(0) offset(0)       : tensor<512x3x64xi8>             // the K cache: [2 layers x 256 rows, 3 KV heads, 64], int8
%15 = binding(1) offset(%13#0)   : tensor<3x3x64xi32>              // q, quantized (int8 values in i32)
%16 = binding(1) offset(%13#1)   : tensor<3x3xf32>                 // per-head scale (s_q * a_k)
%18 = binding(2) offset(%13#2)   : tensor<3x3x?xf32>{T}            // the scores
// the computation
%24 = tensor.load %14[0..T, 3, 64]                                 // the first T rows of the cache
%26 = linalg.generic { extsi i8 -> i32; (t, h, j) -> (h, j, t) }   // K^T
%27 = linalg.batch_matmul(q [3x3x64], K^T [3x64xT]) -> i64 [3x3xT]
%28 = linalg.generic { trunci -> i32, sitofp, * scale[h, g], * 0.125 }   // dequantize, * 1/sqrt(64)
tensor.store %28 -> %18
```

- `util.assume.int` lists, for each call site of this executable (one per layer and function), the values the offsets take there; the sa backend's offset analysis (`Lin`) uses them.
- One binding can hold several tensors at different offsets (`binding(1)` holds q and the scales).
- This is `hf_generic.sa_attention`'s scores step; IREE fused the sign extension, the transpose and the scaling into the same dispatch.

## 3. The sa pipeline

`buildTranslationPassPipeline` (`compiler/plugins/sa/target/SATarget.cpp`) runs, per executable:

| # | Pass | Level | What it does to this dispatch |
|---|---|---|---|
| 0 | `iree-sa-lower-workgroup-count` | HAL | Workgroup count (1, 1, 1) |
| 1 | IREE comprehensive bufferize | linalg | tensors → memrefs; local `memref.alloc`s for the intermediates |
| 2 | `iree-sa-to-sahl` | → sahl | DDR traffic made explicit (`sahl.load` / `sahl.store`); the contraction and its tail grouped into a `sahl.kernel` |
| 3 | `iree-sahl-tile` | sahl | Pieces for element-wise dispatches too large for ACC (none here) |
| 4 | `iree-sahl-plan-memory` | sahl | Placement (`sa.mem`) and layout (`sa.layout`) of each local buffer |
| 5 | `iree-sahl-schedule` | sahl | Chunking of linear micro-kernels (none here) |
| 6 | `iree-sahl-expand-kernels` | sahl | The kernel's schedule written out as explicit sahl operations |
| 7 | `iree-sahl-to-sahw` | → sahw | One sahw operation per accelerator command; one single-stage VE per arith operation |
| 8 | `iree-sahw-fuse-ve` | sahw | Single-stage VEs merged into multi-stage ones |
| 9 | `iree-sahw-split-head` | sahw | Leading loads into a head (none here: the export touches SPAD_B) |
| 10 | `iree-sahw-assign-registers` | sahw | BASE / PARAM numbering and the load table |
| 11 | serialization | bytes | `sahw.template` → sa-desc v3 |

Passes 1–8 run inside `iree-sa-codegen` on a copy of the function (if any of them fails, the dispatch goes to the host fallback). Each pass can be run alone with `iree-opt` and has lit tests in `compiler/plugins/sa/test/`.

## 4. Stage by stage

### 4.1 Bufferize: tensors become memrefs

The tensors become views of the bindings and three local allocations for the intermediates, still of dynamic size T:

```mlir
%alloc   = memref.alloc(%T) : memref<3x3x?xf32>     // the result before the store
%alloc_0 = memref.alloc(%T) : memref<3x3x?xi64>     // the accumulator
%alloc_1 = memref.alloc(%T) : memref<3x64x?xi32>    // K^T, sign-extended
%subview = memref.subview %K[0, 0, 0] [%T, 3, 64]   // the first T rows of the cache
linalg.fill(0) -> %alloc_0
linalg.generic { extsi; transpose } %subview -> %alloc_1
linalg.batch_matmul(%q, %alloc_1) -> %alloc_0
linalg.generic { trunci, sitofp, * scale, * 0.125 } (%alloc_0, %scale) -> %alloc
linalg.generic { copy } %alloc -> %out
```

### 4.2 `sa-to-sahl`: explicit DDR traffic and kernel grouping

Computation may only read and write on-chip buffers, so reads of a binding become `sahl.load` into a local buffer and the final copy becomes `sahl.store`. The matchers (`SahlKernels.cpp`) recognise the contraction and move it, with the operations that belong to it, into a `sahl.kernel`, with the `batch_matmul` marked as the anchor:

```mlir
sahl.kernel "contraction" {
  linalg.fill ins(%c0_i64) outs(%alloc_0)
  linalg.generic { extsi; (d0, d1, d2) -> (d1, d2, d0) } ins(%subview) outs(%alloc_1)
  linalg.batch_matmul {sahl.anchor} ins(%q, %alloc_1) outs(%alloc_0)
  sahl.load %scale, %alloc_2 : memref<3x3xf32, ...>, memref<3x3xf32>
  linalg.generic { trunci, sitofp, mulf %scale, mulf 0.125 } ins(%alloc_0, %alloc_2) outs(%alloc)
  sahl.store %alloc, %out
}
```

The kernel kind decides who expands it: `"linear"` (int8 linear layers: chunking, double buffering, GEMV, prefix prefetch), `"attention"` (the hand-written frontend's attention form), or `"contraction"` (the generic contraction: any batch / M / N / K layout, static or dynamic). This generic-frontend attention is matched as a **generic contraction** with a dynamic N = T.

### 4.3 `sahl-tile`, `sahl-plan-memory`, `sahl-schedule`: attributes

Nothing to tile (no element-wise piece too large for ACC) and nothing to chunk (no linear micro-kernel). `sahl-plan-memory` places every local buffer and fixes its layout:

```mlir
%alloc   = memref.alloc(%T) {sa.layout = "rows",   sa.mem = "acc"} : memref<3x3x?xf32>
%alloc_0 = memref.alloc(%T) {sa.layout = "rows",   sa.mem = "acc"} : memref<3x3x?xi64>
%alloc_1 = memref.alloc(%T) {sa.layout = "rows",   sa.mem = "acc"} : memref<3x64x?xi32>
%alloc_2 = memref.alloc()   {sa.layout = "packed", sa.mem = "acc"} : memref<3x3xf32>
```

`rows`: one row per outer index, each row starting on a word boundary (rows of dynamic length T); `packed`: D elements per word; `bcast`: one value per word, broadcast to all lanes (used later for the scales).

### 4.4 `sahl-expand-kernels`: the micro-kernel's schedule, written out

The kernel is replaced by its schedule in explicit sahl operations (`SahlExpandKernels.cpp`, iree_compiler_plan §8.20). The derived plan: batch = 3 KV heads, M = 3 query rows per head, N = T (dynamic), K = 64; the matrix `K^T` is "K-contiguous rows", so it is loaded row by row and transposed into B blocks; the 3 query rows of a head are issued one by one, each replicated into an A strip.

Before the loops, once:

```mlir
sahl.claim_spad {words = 8192}                              // SPAD_A [0, 8192): fixed-position buffers below
%alloc   = memref.alloc() {sa.layout = "packed", sa.mem = "acc"} : memref<3x3xf32>
sahl.load %scale, %alloc                                    // the 9 per-head scales
%alloc_0 = memref.alloc() {sa.layout = "bcast", sa.mem = "acc"} : memref<3x3xf32>
sahl.bcast %alloc, %alloc_0                                 // one scale per word (row broadcast in the tail)
sahl.cursor_reset                                           // the row cursor: where the next output row goes
%nb = arith.shrui %T, 4                                     // T / D blocks
```

Per KV head h (3 times; head 0 shown):

```mlir
%alloc_2 = memref.alloc(%nb) {sa.bank = 0, sa.mem = "spad_b"} : memref<?x64x16xi8>      // K^T as B blocks
%alloc_3 = memref.alloc(%T) {sa.mem = "spad_a", sa.word = 4096} : memref<?x64xi8>      // K rows
sahl.load %K[0..T, h, 0..64], %alloc_3 {sa.rows}            // T rows of 64 bytes, pitch 3 x 64
sahl.transpose %alloc_3, %alloc_2                           // -> T / 16 blocks of [64 x 16]
```

Per query row g of that head (9 times in all; (h, g) = (0, 0) shown):

```mlir
sahl.load %q[h, g, 0..64], %alloc_1                         // 64 x i32 into ACC
%alloc_7 = memref.alloc() {sa.mem = "spad_a", sa.word = 0} : memref<16x64xi8>
sahl.strip %alloc_1, %alloc_7                               // int32 -> int8, replicated to 16 rows: the A strip
sahl.scope attributes {keep} {
  %acc = memref.alloc() {sa.accumulator, sa.layout = "packed", sa.mem = "acc"} : memref<256xi64>
  sahl.mma %alloc_7, %alloc_2, %acc                         // EX: [16 x 64] x [64 x T]
  %res = memref.alloc() {sa.layout = "packed", sa.mem = "acc"} : memref<256xf32>
  linalg.generic { trunci, sitofp, mulf %scale_bcast[h, g], mulf 0.125 } ins(%acc[0..T], ...) outs(%res[0..T])
  sahl.store %res[0..T], %out[0, 0, 0..T] {sa.cursor, sa.cursor_step}   // the output row at the cursor; cursor += row
}
```

What the expansion decided, in IR form:
- **positions**: the A strip at SPAD_A word 0, the K rows at word 4096 (the other bank), the B blocks in SPAD_B bank 0;
- **layout conversions**: `sahl.transpose` (K rows → B blocks), `sahl.strip` (vector → replicated rows), `sahl.bcast` (scales → broadcast words);
- **the dynamic dimension**: buffers of size `?` with SSA sizes `T` and `T >> 4`;
- **the output rows**: the scores of row (h, g) are row 3h + g of `[3, 3, T]`, with a dynamic pitch T: written at a device-side row cursor that advances by one row after each store;
- **lifetimes**: `sahl.scope` frees the accumulator and the result after each row; `keep` keeps the outer buffers.

### 4.5 `sahl-to-sahw`: one operation per command

Every sahl operation becomes accelerator commands, every arith operation of a body a **single-stage** VE (`op`, then `× A + B`). For row (0, 0):

```mlir
sahw.ld %q_base {ddr = 0, laddr = 0x30000221, rows = 1, row_bytes = 256, pitch = 256}
sahw.ve {op = 5 /*copy*/, m1 = 2 /*DIV*/, p1 = 16, types = 2 /*I32 -> I8*/, src1 = 0x30000221, dst = 0x10000000, length = 1024}
sahw.ex dyn(%T_div_16) {a = 0, b = 0, c = 549, kt = 4, repeat = 16, bstep = 64, cstep = 1, crow = 16, dyn_fields = [5]}
sahw.ve dyn(%T) {op = 2 /*mul*/, m2 = 2, types = 62 /*I32, F32 -> F32*/, src1 = acc, src2 = scale_bcast, a = 1.0, dyn_fields = [4]}
sahw.ve dyn(%T) {op = 5 /*copy*/, a = 0.125, types = 15 /*F32*/}
sahw.ve dyn(%T) {op = 5 /*copy*/, a = 1.0, types = 15}
sahw.st %out_base dyn(%cursor, %row_bytes) {laddr = 0x30000235, rows = 1, row_bytes = 1024, dyn_fields = [1, 4]}
sahw.setreg %cursor dyn(%row_bytes) {add_mask = 1}
```

- **Local addresses** are `mem << 28 | word`: `0x1...` SPAD_A, `0x2...` SPAD_B, `0x3...` ACC (`0x30000221` = ACC word 0x221).
- **Dynamic values** become `dyn(...)` operands: SSA values that will be PARAM registers, placed into one of the descriptor's two dynamic slots (`dyn_fields`: which field they replace). The static field holds the upper bound (`repeat = 16` = 256 / 16, `row_bytes = 1024` = 256 × 4).
- **`-0.0` as the identity B** (`y + (-0) = y` for every y, keeping the sign of zero).

### 4.6 `sahw-fuse-ve`: multi-stage VEs

A VE does `FUNC(OP(s1', s2) · A + B)` followed by an output conversion in one pass. The three single-stage VEs of each tail (multiply by the scale, multiply by 0.125, copy) fold into one: `op = mul` with the broadcast scale, `A = 0.125`. Without rounding changes only (each fp32 step rounds once, as before). For this dispatch: **37 VEs → 19** (9 tails × 3 → 9 × 1, plus 9 strips and 1 broadcast).

### 4.7 `sahw-split-head`, `sahw-assign-registers`: the template

No head (the export uses SPAD_B, so another dispatch's prefix may not run inside it). The registers are numbered and every one gets a setup expression, which becomes the load table:

```mlir
sahw.template "main$async_dispatch_19_batch_matmul_3x3xDx64_i32xi32xi64" attributes {bindings = 3, constants = 5, sa.d = 16} {
  %0 = sahw.base binding(1) {constant = 1, reg = 0}        // BASE0 = binding 1 + c1   (the scales)
  %1 = sahw.private {reg = 7}                              // PARAM7: the template's own (the row cursor)
  %2 = sahw.base binding(0) {reg = 1}                      // BASE1 = binding 0        (the K cache)
  %3 = sahw.param {constant = 3, reg = 0}                  // PARAM0 = c3              (T)
  %4 = sahw.param {constant = 3, mul = 64, reg = 1}        // PARAM1 = 64 T            (transpose length)
  %5 = sahw.base binding(1) {constant = 0, reg = 2}        // BASE2 = binding 1 + c0   (q)
  %6 = sahw.param {constant = 3, reg = 2, shift = 4}       // PARAM2 = T / 16          (EX repeat)
  %7 = sahw.base binding(2) {constant = 2, reg = 3}        // BASE3 = binding 2 + c2   (the scores)
  %8 = sahw.param {constant = 3, mul = 4, reg = 3}         // PARAM3 = 4 T             (bytes of one output row)
  sahw.body { ...64 commands... }
}
```

(`c0`–`c3` are push constants; `c3` is T, the low half of the 64-bit value: the constant analysis proved the high half is 0.)

## 5. The sa-desc file

The serializer writes the template as sa-desc version 3 (`compiler/runtime/tools/sadesc.py` defines the format; the C loader is `compiler/runtime/sa/sa_loader.c`). This executable is 4448 bytes:

| Part | Offset | Content (this file) |
|---|---|---|
| Header (64 B) | 0 | magic `SADESC1\0`, version 3, D = 16, required CAPS `0x3A00000` (bits 21 DESC, 23 FP VE, 24 CMDX, 25 GEMV), 1 export, offsets of the parts below |
| Export table (32 B per export) | 64 | name, template offset, **64 descriptors**, 3 bindings, 5 constants, estimated cycles (0: no cost model yet), offset of the setup table |
| Templates (64 B per descriptor) | 128 | 64 × 64 = 4096 bytes |
| Strings | 4224 | the export names |
| Register setup table (16 B per entry) | after the strings | the 8 entries of §4.7: `kind, register, binding, push-constant ordinal, mul, div (power of two), add` |
| Export extension (32 B per export) | 4416 | prefix 0, head 0, bindings read `0x3` (0, 1), written `0x4` (2), touches SPAD_B |

The driver evaluates the setup table at each dispatch: `value = ((constant × mul) >> log2(div)) + add`, plus the binding's physical address for a BASE.

### 5.1 Three descriptors, bit by bit

Every descriptor is eight little-endian 64-bit words. w0 is the common header: `[7:0]` opcode, `[8]` RELOC (DDR addresses relative to a BASE), `{[14:13], [10:9]}` BASESEL, `[31:16]` two 8-bit dynamic slots (`[3:0]` field, `[6:4]` PARAM, `[7]` add instead of replace), `[63:32]` tag (here the descriptor's index).

**#4, the LD of head 0's K rows**

```
w0 = 0000000400030301   opcode 0x01 LD, RELOC, BASESEL 1 (BASE1 = the K cache), slot 0x03: field 3 (rows) <- PARAM0 (T), tag 4
w1 = 0000000000000000   DDR offset 0 (from BASE1)
w2 = 0040010010001000   laddr 0x10001000 (SPAD_A word 4096), rows 0x100 = 256 (replaced by T), row_bytes 0x40 = 64
w3 = 00000000000000c0   pitch 0xC0 = 192 (3 heads x 64 bytes)
```

**#8, the EX of row (0, 0)**

```
w0 = 0000000800250003   opcode 0x03 EX, slot 0x25: field 5 (repeat) <- PARAM2 (T / 16), tag 8
w1 = 0004022500000000   A = 0 (SPAD_A word 0), B = 0 (SPAD_B word 0), C = 0x225 (ACC word), Kt = 4 (64 / 16)
w2 = 0010000100400010   repeat 16 (replaced by T / 16), bstep 0x40 = 64, cstep 1, crow 0x10 = 16
```

**#10, the ST of the scores row**

```
w0 = 0000000a34f10702   opcode 0x02 ST, RELOC, BASESEL 3 (the scores), tag 10
                        slot 0xF1: field 1 (DDR address) + PARAM7 (the row cursor, add)
                        slot 0x34: field 4 (row_bytes) <- PARAM3 (4 T)
w2 = 0400000130000235   laddr 0x30000235 (ACC), rows 1, row_bytes 0x400 = 1024 (replaced by 4 T)
w3 = 0000000000000400   pitch 1024
```

### 5.2 The whole template

`compiler/runtime/tools/sadis.py` disassembles it. The first 23 of the 64 descriptors:

```
setup BASE0 = binding 1 + c1          setup PARAM0 = c3          setup BASE2 = binding 1 + c0     setup BASE3 = binding 2 + c2
setup BASE1 = binding 0               setup PARAM1 = c3 * 64     setup PARAM2 = c3 / 16           setup PARAM3 = c3 * 4
   0 LD       ddr=0x0 la=0x30000200 rows=1 rb=40 pitch=40 base=0                        the 9 scales (36 bytes -> 40) into ACC
   1 VE       op=5 s1=0x30000200 dst=0x30000201 len=256 types=0xf m1=2 p1=16            \  one scale per word
   2 VE       TRANSPOSE src=0x30000201 dst=0x30000211 len=256 types=0xf stride=1        /  (bcast layout)
   3 SETREG   r23=0                                                                    the row cursor PARAM7 = 0
   4 LD       ddr=0x0 la=0x10001000 rows=256 rb=64 pitch=192 base=1 dyn(f3,p0)          head 0: T rows of K
   5 VE       TRANSPOSE src=0x10001000 dst=0x20000000 len=16384 stride=4 dyn(f4,p1)    -> K^T blocks in SPAD_B
   6 LD       ddr=0x0 la=0x30000221 rows=1 rb=256 pitch=256 base=2                      q[0, 0] (64 x i32)
   7 VE       op=5 s1=0x30000221 dst=0x10000000 len=1024 types=0x2 m1=2 p1=16           int8, replicated to 16 rows
   8 EX       a=0x0 b=0x0 c=0x225 kt=4 rep=16 bstep=64 cstep=1 crow=16 dyn(f5,p2)       [16 x 64] x [64 x T]
   9 VE       op=2 s1=0x30000225 s2=0x30000211 dst=0x30000235 len=256 types=0x3e        f32(acc) * scale[0, 0] * 0.125
              m2=2 A=0.125 B=-0.0 dyn(f4,p0)
  10 ST       la=0x30000235 rows=1 rb=1024 pitch=1024 base=3 dyn(f1,p7+) dyn(f4,p3)     the row at the cursor
  11 SETREG   r23=0 add=1 dyn(f1,p3)                                                    cursor += 4 T
  12 LD       ddr=0x100 la=0x30000221 ...                                               q[0, 1] ... (6 descriptors per row)
  ...
  22 ST       ...
```

64 = 4 (setup) + 3 heads × 2 (K rows, transpose) + 9 rows × 6 (load q, strip, EX, tail, store, cursor).

## 6. At run time

**Load (once)**: the driver copies the template area into the device window and records the export's physical address (`sa_loader.c`).

**Per call**, e.g. decode at position 33 with `--pad=16`, so T = 48:
1. The VM calls `hal.command_buffer.dispatch(executable, 0, constants = [c0, c1, c2, T_lo, T_hi], bindings = [KV cache, activations, scores])`.
2. The sa command buffer evaluates the setup table: BASE0 = phys(binding 1) + c1, BASE1 = phys(binding 0), BASE2 = phys(binding 1) + c0, BASE3 = phys(binding 2) + c2, **PARAM0 = 48, PARAM1 = 3072, PARAM2 = 3, PARAM3 = 192**; and appends `SETREG` descriptors writing them plus a `CALL` of the template to the step's list (at flush the driver also places the FENCEs a barrier needs, from the bindings' read / write masks).
3. The step's list goes into a command-ring entry; `rt_fw` on PicoRV32 starts it (`mat_submit`).
4. The fetch unit (`sa_cmdfetch.v`) decodes each descriptor and substitutes the dynamic slots: #4 loads `rows = 48`, #5 transposes 3072 bytes, #8 runs `repeat = 3` tiles, #9 processes 48 elements, #10 stores 192 bytes at BASE3 + PARAM7, and #11 advances PARAM7 by 192. The scheduler and the engines execute the packets (execution_walkthrough §4–§5).

## 7. What the result says about efficiency

Per query row the EX computes a `[16 × 64] × [64 × T]` product, of which only row 0 is wanted: the strip replicates one query row to the 16 rows of the array (§4.4, `sahl.strip`), because the array is 16 × 16 and decode has a single query row per head. So 15/16 of the multiply-accumulates are thrown away, and each of the 9 rows also pays an LD, a strip, a tail VE and an ST. This is why decode attention reaches only about 0.45 MAC/cycle on the board (generality plan §13.1) while the linear layers stream weights at 82–94% of the GEMV peak. Batching the query rows of a head (3 here, 2–4 in the other models) into one strip, or the rows of several decode sequences, would use more of the array.

## 8. The other micro-kernels

The same pipeline handles the other kinds; the difference is in `sahl-expand-kernels`:
- **`linear`** (the int8 linear layers, about 80% of decode time): `sahl-schedule` chooses the chunk size; the expansion writes a chunk loop (`sahl.loop` + `sa.advance`, LOOP_END on the device), alternating SPAD_B banks, chunk 0's weights in the template's **prefix** (the driver can run it inside an earlier dispatch, iree_compiler_plan §7.1 step 5), and with 4 GEMV ports `sahl.gemv` instead of LD + EX: the weights stream straight from DDR into the GEMV unit (`k2b_gemv_design.md`), split in K blocks of 4080 when K is larger (generality plan §10).
- **`attention`**: the hand-written frontend's attention form (iree_compiler_plan §8.20 E4).
- **No kernel** (element-wise operations, reductions, gathers): `sahl-to-sahw` lowers the `linalg.generic` bodies directly into VE commands (`SahlLowerGeneric.cpp`), with `sahl-tile` pieces when they do not fit ACC.

`--iree-sa-ukernels=none` turns the `linear` and `attention` kernels off; everything still compiles through the generic contraction (golden corpus cases `*_none`).

## 9. Reproducing

```sh
cd Torch2RISCV-accelerator
# the dispatch without its host-fallback variant
build/iree/venv/bin/python -c "import sys; sys.path.insert(0, 'compiler/tests'); import golden; \
  open('/tmp/d19.mlir', 'w').write(golden.sa_only(open('build/hfgen/smol_l2qs/sa_sources/module_main\$async_dispatch_19.mlir').read()))"
# compile it alone, printing the IR after every pass (the sa passes included)
build/iree/build-compiler/tools/iree-compile /tmp/d19.mlir --compile-mode=hal-executable --iree-hal-target-device=sa \
  --iree-sa-d=16 --iree-sa-gemv-ports=4 --mlir-disable-threading --mlir-print-ir-after-all -o /tmp/d19.sadesc 2> /tmp/d19.ir
grep -n "IR Dump After" /tmp/d19.ir          # one section per pass
# the descriptors
build/iree/venv/bin/python compiler/runtime/tools/sadis.py /tmp/d19.sadesc
```

Every model's dispatches are dumped this way by `compile_sa.sh` (`--iree-hal-dump-executable-sources-to=<dir>/sa_sources`). A single pass can be run on IR saved from the dump with `iree-opt`, e.g. `iree-opt --iree-sahl-expand-kernels`; the correctness of every dispatch is checked by `compiler/tests/dispatch_check.py`, which evaluates the dispatch's IR directly (the oracle) and compares it bit for bit with the functional simulator running the descriptors.

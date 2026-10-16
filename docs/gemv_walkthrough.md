# One GEMV from PyTorch to the accelerator: Qwen3-0.6B's q_proj, stage by stage

This document follows a single decode dispatch of Qwen3-0.6B through the whole
stack: the PyTorch module, the exported MLIR, IREE's dispatch formation, every
pass of the sa backend, the binary it serializes, how the runtime turns the
binary into a list the hardware runs, and what the hardware does with it.
Every listing below is real output of the long-context bundle that was accepted
on the board on 2026-10-16 (`build/hfgen/qwen3_lc2048_pcpi`, overlay
`d16_100mhz_w128_p6_gemv_np4_pcpi`: D = 16, four GEMV ports), trimmed only
where marked.

**Reproduce** (about 10 seconds; writes every stage to `build/walk`):

```
SA_D=16 SA_GEMV_PORTS=4 compiler/scripts/walk_dispatch.sh build/hfgen/qwen3_lc2048_pcpi 7 build/walk
```

The script recompiles the dispatch's executable source alone with the IR
printed after each sa pass, checks that the binary is byte for byte the one
inside `sa.vmfb`, and disassembles it. Any other dispatch number works the same
way.

Related documents: [`iree_compiler_plan.md`](iree_compiler_plan.md) (the
compiler's design, §4 the binary format, §8 the dialects and passes),
[`k2b_gemv_design.md`](k2b_gemv_design.md) (the GEMV unit),
[`dispatch_walkthrough.md`](dispatch_walkthrough.md) and
[`execution_walkthrough.md`](execution_walkthrough.md) (the runtime and the
hardware side in general).

---

## 0. The dispatch

`main$async_dispatch_7_matvec_like_128x16x1024_i8xi8xi32`: the query
projection of a decoder layer. It multiplies the int8 activation `x` (1024
values, the RMS-normalized hidden state quantized per token) by the int8
weight `Wq` (2048 × 1024) into 2048 int32 sums. The name is IREE's: a
matrix-vector product of shape 128 × 16 × 1024. 128 × 16 is the 2048 outputs
in strips of 16 (the array size D), and 1024 is K.

It is one of 28 call sites of the same executable: all 28 layers' q_proj have
the same shape, so IREE compiles one executable and calls it 28 times per
decode step with different buffers.

| Quantity | Value |
|---|---|
| Weights read | 2048 × 1024 int8 = 2 MiB |
| Activation | 1024 int8 = 1 KiB |
| Result | 2048 int32 = 8 KiB |
| Template | 17 descriptors (1088 bytes), sa-desc file 1360 bytes |
| Hardware commands per call | about 30 (with the loop's 3 iterations unrolled) |
| Time per call (estimate) | about 36k cycles = 0.36 ms at 100 MHz, from the GEMV unit's measured 57.95 B/cycle on four ports (2 MiB / 57.95) |

## 1. PyTorch: the quantized linear layer

The generic frontend (`compiler/frontend/hf_generic.py`) replaces each
`nn.Linear` of the HuggingFace model with a W8A8 layer:

```python
class QLinear(torch.nn.Module):
    def forward(self, x):
        xq, s_x = _quant_act(x)                       # per-token int8 activation and its scale
        rows = xq.reshape(-1, xq.shape[-1])
        acc = torch.matmul(rows.to(torch.int32), self.wq.to(torch.int32).t())   # int32 product
        acc = acc.reshape(tuple(xq.shape[:-1]) + (acc.shape[-1],))
        y = (acc.to(torch.float32) * self.s_w) * s_x  # dequantize: per-row weight scale, token scale
        return y if self.bias is None else y + self.bias
```

Only the int32 `matmul` becomes our dispatch. The activation quantization
before it and the dequantization after it become dispatches of their own (§3).

## 2. The exported module (torch dialect)

iree-turbine exports the model to `qllama.mlir` (45 MB of text; the weights are
not in it, they are named parameters in `qllama.irpa`). The weight is a global
bound to a parameter:

```mlir
util.global private @__auto.model.model.layers.0.self_attn.q_proj.wq =
    #flow.parameter.named<"model"::"model.model.layers.0.self_attn.q_proj.wq"> : tensor<2048x1024xi8>
```

and in the decode function `main`, layer 0's product reads:

```mlir
%163 = torch.prims.convert_element_type %162, %int1 : !torch.vtensor<[1,1,1024],f32> -> !torch.vtensor<[1,1,1024],si8>
%165 = torch.aten.view %163, ... -> !torch.vtensor<[1,1024],si8>
%166 = torch.prims.convert_element_type %165, %int3 : ... -> !torch.vtensor<[1,1024],si32>
%167 = torch_c.from_builtin_tensor %__auto.model.model.layers.0.self_attn.q_proj.wq : tensor<2048x1024xi8> -> ...
%168 = torch.prims.convert_element_type %167, %int3 : ... -> !torch.vtensor<[2048,1024],si32>
%169 = torch.aten.transpose.int %168, %int0, %int1 : ... -> !torch.vtensor<[1024,2048],si32>
%170 = torch.aten.matmul %166, %169 : !torch.vtensor<[1,1024],si32>, !torch.vtensor<[1024,2048],si32> -> !torch.vtensor<[1,2048],si32>
...
%173 = torch.prims.convert_element_type %172, %int6 : ... -> !torch.vtensor<[1,1,2048],f32>
%175 = torch.aten.mul.Tensor %173, %174   // × s_w
%176 = torch.aten.mul.Tensor %175, %157   // × s_x
```

## 3. IREE: weight packing and dispatch formation

`iree-compile --iree-hal-target-device=sa` lowers torch to linalg and runs
IREE's global optimizations. Two things happen to this product before it
reaches the sa backend.

**The weight is packed** (`compiler/plugins/sa/target/PackLinearWeights.cpp`).
The accelerator reads a linear layer's weights as strips: strip t is rows
16t … 16t + 15 of W, transposed, so one K step reads 16 consecutive bytes for 16
outputs. The pass rewrites `vecmat(x, transpose(extsi(W)))` into a pack
`W[2048, 1024] → Wp[128, 1024, 16]` plus a generic over it. The pack depends
only on a parameter, so IREE evaluates it at compile time. The packed weights
are stored in `sa_packed.irpa`, the parameter archive the bundle loads, and
nothing is repacked at run time.

**Dispatch formation** cuts `main` into dispatches. Around this product they
are:

| Dispatch | What |
|---|---|
| 3 `reduction_1024_f32` | the RMSNorm's mean of squares |
| 4 `elementwise_1024_f32` | normalize × weight |
| 5 `reduction_1024_f32` | the activation's absolute maximum (its int8 scale) |
| 6 `elementwise_1024_f32xf32xi8` | quantize to int8: `x` |
| **7 `matvec_like_128x16x1024_i8xi8xi32`** | **the product** |
| 8 `elementwise_2048_i32xf32xf32xf32` | dequantize: acc × s_w × s_x |
| 9 `reduction_16x128_f32` | Qwen3's q-norm (per head) |

Dispatch 7's executable source, as IREE hands it to the backend
(`sa_sources/module_main$async_dispatch_7.mlir`, layouts shortened to `L`):

```mlir
hal.executable.variant public @sa_desc_v1 target(<"sa", "sa-desc-v1", {acc_bytes = 262144, bases = 16, d = 16,
    dyn_fields = 2, gemv_ports = 4, max_dynamic = 2048, params = 8, spad_bytes = 131072}>) {
  func.func @main$async_dispatch_7_matvec_like_128x16x1024_i8xi8xi32() {
    %0 = hal.interface.constant.load layout(L) ordinal(0) : i32        // byte offset of x in its buffer
    %1 = hal.interface.constant.load layout(L) ordinal(1) : i32        // byte offset of the result
    %4:2 = util.assume.int %2[<umin = 9280, umax = 9280>, <umin = 0, umax = 0>, <umin = 64, ...>, ... 28 entries],
                           %3[<umin = 10304, ...>, <umin = 13376, ...>, <umin = 13312, ...>, ... 28 entries]
    %5 = hal.interface.binding.subspan layout(L) binding(0) offset(%4#0) : tensor<1024xi8>          // x
    %6 = hal.interface.binding.subspan layout(L) binding(1) offset(%c0)  : tensor<128x1024x16xi8>   // Wp
    %7 = hal.interface.binding.subspan layout(L) binding(2) offset(%4#1) : tensor<128x16xi32>       // result
    ...
    %12 = linalg.generic {indexing_maps = [(d0, d1, d2) -> (d2), (d0, d1, d2) -> (d0, d2, d1), (d0, d1, d2) -> (d0, d1)],
                          iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%8, %9 : tensor<1024xi8>, tensor<128x1024x16xi8>) outs(%11 : tensor<128x16xi32>) {
      ^bb0(%in: i8, %in_0: i8, %out: i32):
        %13 = arith.extsi %in : i8 to i32
        %14 = arith.extsi %in_0 : i8 to i32
        %15 = arith.muli %13, %14 : i32
        %16 = arith.addi %out, %15 : i32
        linalg.yield %16 : i32
    } -> tensor<128x16xi32>
    iree_tensor_ext.dispatch.tensor.store %12, %7 ...
```

The target attributes are the hardware the backend compiles for: D = 16, SPAD
128 KB, ACC 256 KB, 16 BASE and 8 PARAM registers, two dynamic fields per
descriptor, four GEMV ports. The two push constants are byte offsets. IREE
packs small transient values into shared buffers, so x and the result sit at
an offset that depends on the call site. `util.assume.int` lists the value at
each of the 28 call sites: x at 9280, 0 or 64, the result at 10304, 13376 or
13312. The source also holds a VMVX variant (the host fallback); it is unused
here, because the sa variant compiles.

## 4. The sa backend, pass by pass

`SaCodegen` (`compiler/plugins/sa/transforms/SaCodegen.cpp`) runs this
pipeline on each dispatch function (`buildSahlPipeline`, `SahlPipeline.cpp`):

```
bufferize → sa-to-sahl → sahl-tile → sahl-plan-memory → sahl-schedule → sahl-expand-kernels
          → sahl-to-sahw → sahw-fuse-ve → serialization
```

`sahl` is the tile-level dialect: buffers in DDR and in the local memories,
loads, stores, linalg on local buffers, micro-kernels. `sahw` is the command
level: one op per hardware command. Listings below omit the push-constant and
binding boilerplate shown above.

### 4.1 sa-to-sahl: recognize the micro-kernel

After bufferization the tensors are memrefs. `sa-to-sahl` recognizes the
pattern "fill, load x, load packed W, int8 × int8 → int32 contraction, store"
as the `linear` micro-kernel and wraps it:

```mlir
%alloc   = memref.alloc() : memref<128x16xi32>
%alloc_0 = memref.alloc() : memref<1024xi8>
%alloc_1 = memref.alloc() : memref<128x1024x16xi8>
sahl.kernel "linear" {
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<128x16xi32>)
  sahl.load %5, %alloc_0 : memref<1024xi8, ...ddr>, memref<1024xi8>
  sahl.load %6, %alloc_1 : memref<128x1024x16xi8, ddr>, memref<128x1024x16xi8>
  linalg.generic {...} ins(%alloc_0, %alloc_1 ...) outs(%alloc ...) attrs = {sahl.anchor} {...}
  sahl.store %alloc, %7 : memref<128x16xi32>, memref<128x16xi32, ...ddr>
}
```

The local buffers are still the whole operands: 2 MiB of weights "in local
memory" is not possible, and deciding how to split them is the next passes' job.

### 4.2 sahl-tile and sahl-plan-memory

`sahl-tile` splits element-wise dispatches that do not fit ACC. A
`sahl.kernel` tiles itself, so here it changes nothing.
`sahl-plan-memory` annotates each local buffer with its memory and layout:
x `spad_a` packed, the weights `spad_a` packed (a placeholder: the GEMV path
never stores them locally), the result `acc` broadcast.

### 4.3 sahl-schedule: the chunk

```mlir
sahl.kernel "linear" attributes {chunk_tiles = 16 : i64, loop = true} { ... }
```

The schedule (`linearSchedule`, `SahlKernels.cpp`) picks how many output strips
one chunk computes. With the GEMV unit the weights stream from DDR straight
into the unit (no SPAD_B tile), so the limit is the ACC scratch area that holds
a chunk's results:

- each ACC bank (2048 words of 64 bytes) keeps a quarter as scratch, 512 words;
- a chunk needs D + 1 + 1–2 words per strip, so at most 512 / 18 ≈ 28 strips;
- the largest divisor of the 128 strips under that is **16 strips (256 outputs) per chunk**.

`loop = true`: with 128 / 16 = 8 chunks (more than 4) the chunks become a
hardware loop (`LOOP_END`) instead of straight-line code, so the template does
not grow with N.

### 4.4 sahl-expand-kernels: chunks, ping-pong, loop

The kernel expands into explicit operations (trimmed to the loop's first chunk
and the shape of the rest):

```mlir
%alloc = memref.alloc() {sa.mem = "spad_a"} : memref<1024xi8>                    // x, loaded once
sahl.load %5, %alloc
%alloc_0 = memref.alloc() {sa.mem = "acc", sa.word = 0 : i64} : memref<16x16xi32>    // chunk results, ACC bank 0
%alloc_2 = memref.alloc() {sa.mem = "acc", sa.word = 2048 : i64} : memref<16x16xi32> // chunk results, ACC bank 1
sahl.loop attributes {count = 3 : i64, strides = array<i64: 524288, 2048>} {
  // chunk A: strips 0-15 into bank 0, as two GEMVs of 8 strips
  sahl.gemv %alloc, %6[0, 0, 0] [8, 1024, 16], %alloc_0[0, 0] [8, 16]
  sahl.gemv %alloc, %6[8, 0, 0] [8, 1024, 16], %alloc_0[8, 0] [8, 16]
  sahl.store %alloc_0, %7[0, 0] [16, 16]
  // chunk B: strips 16-31 into bank 1
  sahl.gemv %alloc, %6[16, 0, 0] [8, 1024, 16], %alloc_2[0, 0] [8, 16]
  sahl.gemv %alloc, %6[24, 0, 0] [8, 1024, 16], %alloc_2[8, 0] [8, 16]
  sahl.store %alloc_2, %7[16, 0] [16, 16]
}
// after the loop: strips 96-111 (bank 0) and 112-127 (bank 1), the same four GEMVs and two stores
```

What the expansion decided:

- **x is loaded once** into SPAD_A and reused by all 128 strips.
- **Two GEMVs of 8 strips per chunk**: the GEMV unit has 8 accumulators of
  16 × int32 per lane group, so one command covers at most 8 strips.
- **Ping-pong between the two ACC banks**: chunk A writes bank 0 while the
  store of the previous chunk B drains bank 1, and the other way round. The
  hardware scoreboard sees the banks as separate resources, so a store and the
  next GEMV run at the same time.
- **A loop over pairs of chunks**: 3 iterations of (A, B) cover strips 0–95.
  Each iteration moves the weights by 524288 bytes (32 strips × 16 KiB) and the
  result by 2048 bytes (512 int32). The last pair is peeled out of the loop.

### 4.5 sahl-to-sahw: commands

Now each operation becomes one command, and DDR addresses become BASE / PARAM
register references:

```mlir
sahw.template "main$async_dispatch_7_matvec_like_128x16x1024_i8xi8xi32" attributes {sa.d = 16 : i64} {
  %0 = sahw.base binding(0) {constant = 0 : i64}   // BASE0 = x's buffer + push constant 0
  %1 = sahw.private                                 // a PARAM register of the template's own (PARAM7)
  %2 = sahw.private                                 // (PARAM6)
  %3 = sahw.base binding(1)                         // BASE1 = the packed weights
  %4 = sahw.base binding(2) {constant = 1 : i64}   // BASE2 = the result's buffer + push constant 1
  sahw.body {
    sahw.ld %0 {ddr = 0, laddr = 268435456, rows = 1, row_bytes = 1024, pitch = 1024}       // x -> SPAD_A word 0
    sahw.setreg %1, %2 {values = array<i64: 0, 0>}                                           // loop offsets = 0
    sahw.loop 3, %1 step 524288, %2 step 2048 {
      sahw.ld %3 dyn(%1) {mode = 2, ddr = 0,      laddr = 805306368, rows = 8, row_bytes = 16384, pitch = 16384, cstep = 1, dyn_fields = [1], dyn_add = [true]}
      sahw.ld %3 dyn(%1) {mode = 2, ddr = 131072, laddr = 805306376, rows = 8, row_bytes = 16384, pitch = 16384, cstep = 1, ...}
      sahw.st %4 dyn(%2) {ddr = 0,    laddr = 805306368, rows = 1, row_bytes = 1024, pitch = 1024, dyn_fields = [1], dyn_add = [true]}
      sahw.ld %3 dyn(%1) {mode = 2, ddr = 262144, laddr = 805308416, ...}
      sahw.ld %3 dyn(%1) {mode = 2, ddr = 393216, laddr = 805308424, ...}
      sahw.st %4 dyn(%2) {ddr = 1024, laddr = 805308416, ...}
    }
    sahw.ld %3 {mode = 2, ddr = 1572864, laddr = 805306368, rows = 8, ...}   // the peeled last pair
    ... (3 more LDs, 2 STs)
  }
}
```

There is **no EX (array) command**. `mode = 2` is the LD engine's GEMV mode:
the load itself streams each strip's 16 KiB through the GEMV unit, which
multiplies it by x and writes 16 int32 sums per strip into ACC. The systolic
array is not used for decode matrix-vector products: it would need the weights
staged through SPAD_B first, and decode is bandwidth-bound
([`k2b_gemv_design.md`](k2b_gemv_design.md)).

Local addresses: the top nibble is the memory (`0x1` SPAD_A, `0x3` ACC) and the
rest a word index. `805306368` = `0x30000000` is ACC word 0, `805306376` =
`0x30000008` word 8 (the second GEMV's 8 strips), and `805308416` = `0x30000800`
word 2048, ACC bank 1.

`dyn(%1)` with `dyn_add`: the hardware adds PARAM7 to the descriptor's DDR
field when it issues the command. That is how one loop body walks through the
weights: `LOOP_END` increments PARAM7 by 524288 and PARAM6 by 2048 per
iteration.

`sahw-fuse-ve` merges adjacent vector-engine commands. There are none here,
so it leaves the template unchanged.

## 5. The binary: sa-desc v3

The serializer (`SahwSerialize.cpp`, using the encoder `target/DescList.cpp`,
the C++ twin of `driver/pynq_matmul.py`'s `DescList`, byte for byte the same)
writes each template as 64-byte descriptors, plus the tables the runtime needs.
The executable's file (`compiler/runtime/tools/sadesc.py` defines the layout):

| Offset | Content |
|---|---|
| 0 | header: magic `SADESC1\0`, version 3, D = 16, required CAPS bits 21, 23, 24, 25 (descriptor fetch, fp32 VE, BASE / PARAM / dynamic fields, GEMV), 1 export |
| 64 | export table: name, 17 descriptors at template offset 0, 3 bindings, 2 constants, 3 register setup entries |
| 128 | the template: 17 × 64 bytes |
| 1216 | strings: the export's name |
| 1280 | register setup table: `BASE0 = binding 0 + c0`, `BASE1 = binding 1`, `BASE2 = binding 2 + c1` |
| 1328 | export extension: head of 2 descriptors, reads bindings 0 and 1, writes binding 2, no SPAD_B |

The disassembly (`compiler/runtime/tools/sadis.py`):

```
main$async_dispatch_7_matvec_like_128x16x1024_i8xi8xi32: 17 descriptors, 3 bindings, 2 constants; prefix 0, head 2, reads 0x3, writes 0x4, SPAD_B False
     setup BASE0 = binding 0 + ((c0 * 1) / 1) + 0
     setup BASE1 = binding 1 + ((c-1 * 1) / 1) + 0
     setup BASE2 = binding 2 + ((c1 * 1) / 1) + 0
   0 LD       ddr=0x0 la=0x10000000 rows=1 rb=1024 pitch=1024 base=0
   1 RET
   2 SETREG   r23=0 r22=0 add=0
   3 LD       ddr=0x0 la=0x30000000 rows=8 rb=16384 pitch=16384 base=1 dyn(f1,p7+)
   4 LD       ddr=0x20000 la=0x30000008 rows=8 rb=16384 pitch=16384 base=1 dyn(f1,p7+)
   5 ST       ddr=0x0 la=0x30000000 rows=1 rb=1024 pitch=1024 base=2 dyn(f1,p6+)
   6 LD       ddr=0x40000 la=0x30000800 rows=8 rb=16384 pitch=16384 base=1 dyn(f1,p7+)
   7 LD       ddr=0x60000 la=0x30000808 rows=8 rb=16384 pitch=16384 base=1 dyn(f1,p7+)
   8 ST       ddr=0x400 la=0x30000800 rows=1 rb=1024 pitch=1024 base=2 dyn(f1,p6+)
   9 LOOP_END offset=-6 count=3 k1=7 s1=524288 k2=6 s2=2048
  10 LD       ddr=0x180000 la=0x30000000 rows=8 rb=16384 pitch=16384 base=1
  11 LD       ddr=0x1a0000 la=0x30000008 rows=8 rb=16384 pitch=16384 base=1
  12 ST       ddr=0x1800 la=0x30000000 rows=1 rb=1024 pitch=1024 base=2
  13 LD       ddr=0x1c0000 la=0x30000800 rows=8 rb=16384 pitch=16384 base=1
  14 LD       ddr=0x1e0000 la=0x30000808 rows=8 rb=16384 pitch=16384 base=1
  15 ST       ddr=0x1c00 la=0x30000800 rows=1 rb=1024 pitch=1024 base=2
  16 RET
```

The template is split in two parts. The **head** (descriptors 0–1) is the
body's leading load, x, ending in RET. The **body** (2–16) is the rest. The
driver may run another dispatch's prefix between them, so that its loads
overlap this dispatch's compute (`sa_context_batch_add`). `r23` / `r22` are
PARAM7 / PARAM6: register numbers 16 and up are PARAM registers.

### 5.1 One descriptor, bit by bit

Descriptor 3, the first GEMV load, as eight little-endian 64-bit words:

```
w0 0000000200f10301   w1 0000000000000000   w2 4000000830000000   w3 0004000200004000   w4..w7 0
```

| Field | Bits | Value | Meaning |
|---|---|---|---|
| opcode | w0[7:0] | `0x01` | LD |
| relocate | w0[8] | 1 | the DDR address is relative to a BASE register |
| base | w0[10:9], w0[14:13] | 1 | BASE1: the packed weights |
| fence before | w0[11] | 0 | |
| dynamic slot 0 | w0[23:16] | `0xf1` | field 1 (the DDR address) += PARAM7 (bit 7: add; bits 6:4 = 7) |
| dynamic slot 1 | w0[31:24] | 0 | none |
| tag | w0[63:32] | 2 | the descriptor's index in the body (error reports) |
| ddr | w1[31:0] | 0 | offset from BASE1 (+ PARAM7) |
| local address | w2[31:0] | `0x30000000` | ACC word 0 |
| rows | w2[47:32] | 8 | 8 strips |
| row bytes | w2[63:48] | `0x4000` | 16384 = K × 16 bytes per strip |
| pitch | w3[31:0] | `0x4000` | strips are contiguous |
| mode | w3[33:32] | 2 | GEMV |
| x word | w3[49:34] | 0 | x starts at SPAD_A word 0 (where descriptor 0 put it) |
| C step | w3[57:50] | 1 | strip s's 16 sums go to ACC word 0 + s |
| accumulate | w3[58] | 0 | overwrite, not add |

Descriptor 5, the chunk's store: `w0 0000000400e10502`, so opcode `0x02` (ST),
BASE2, dynamic slot `0xe1` = DDR address += PARAM6. `w2 0400000130000000`
means ACC word 0, 1 row of `0x400` = 1024 bytes: 16 ACC words, the 256 int32
of 16 strips.

Descriptor 9, `LOOP_END`: the body from descriptor 3 (6 back) runs 3 times in
all; after each pass PARAM7 grows by 524288 and PARAM6 by 2048, so the second
and third passes read strips 32–63 and 64–95 and write their results 2 and 4 KiB
further.

## 6. Into the module, and onto the board

The sa-desc file is embedded in `sa.vmfb` as an executable of format
`application/x-sa-desc` (`SATarget.cpp::serializeExecutable`). The rest of the
vmfb is IREE VM bytecode: the host program that, for each decode step,
allocates the transients, records a command buffer with the step's dispatches,
about a thousand (dispatch 7 among them, 28 times, each with its bindings and its two push
constants) and submits it. The same vmfb runs on the board's ARM (the aarch64
`sa-llm-run`) and against the functional simulator on the PC.

At run time (`compiler/runtime/sa/`):

1. **Load** (`sa_loader.c`): the template's 1088 bytes are copied into the
   device window (the udmabuf the accelerator reads), so the export has a
   physical address. Once per process.
2. **Dispatch** (`sa_command_buffer.c` → `sa_context_batch_add`): for each of
   the 28 calls, the setup table turns the binding addresses and push constants
   into register values (BASE0 = x's buffer + c0, BASE1 = this layer's packed
   weights, BASE2 = the result's buffer + c1). The batch builder then appends
   to the step's list:

   ```
   SETREG  BASE0=…  BASE1=…  BASE2=…
   CALL    template + 0      (the head: load x)
   CALL    template + 128    (the body)
   ```

   plus a `FENCE` before it only if it reads something an earlier dispatch in
   the list writes (here x, written by dispatch 6, so a FENCE waits for the ST
   engine). The whole decode step is one or two such lists (3926 descriptors
   for Qwen3), ended by `END`.
3. **Submit**: the ARM writes one ring entry; `rt_fw` on the PicoRV32 issues
   `mat_submit` and waits (`mat_fence`). With
   [plan B](riscv_backend_design.md#71-plan-b-the-core-schedules-the-descriptor-lists-existing-bitstream)
   (`sa-llm-run --rvsched`) the PicoRV32 runs the whole decode loop and
   resubmits the same recorded lists itself. With
   [plan A](riscv_backend_design.md#9-the-sahw--llvm--riscv32-backend-plan-as-code-generator-plan-2026-10-16)
   (`SA_RV=embedded`) there is no list at all: iree-compile also emits each
   dispatch as rv32 code (`gemv_cfg` / `gemv_run` and the other commands as
   PCPI instructions), and the core calls it.

## 7. What the hardware does

The descriptor fetch unit (`rtl/sysarray/sa_cmdfetch.v`) reads the list from
DDR, executes `SETREG` / `CALL` / `RET` / `LOOP_END` itself, applies BASE and
the dynamic fields to each command, and pushes LD / ST commands into the
in-order scheduler (`sa_sched.v`). The scheduler issues a command once the
engine it needs is free and its local-memory banks do not conflict with
commands in flight.

For this dispatch:

1. **LD x** (descriptor 0): 1024 bytes from DDR into SPAD_A words 0–63.
2. **GEMV LDs** (3, 4, …): the LD engine reads 8 strips × 16 KiB over the four
   HP read ports (`sa_ld.v`), one lane group per port. Each lane group holds a
   copy of x (read from SPAD_A at the start of the command) and 8 × 16 int32
   accumulators. Every 16-byte beat of weights is one k step of one strip:
   16 multiply-adds with x[k] (`sa_gemv.v`). At the end, the lane groups' sums
   are added and written to ACC words C + s × cstep. This is the formula
   checked against the functional simulator:
   `ACC[C + s·cstep][j] = Σ_k sx8(x[k]) · sx8(W_s[k][j])`.
3. **ST** (5, 8, …): 1024 bytes from ACC to DDR. Because the next chunk's GEMVs
   write the other ACC bank, the scoreboard lets the store run during them.

The weight stream is the cost: 2 MiB per call at the four ports' measured
57.95 bytes per cycle is about 36k cycles (0.36 ms at 100 MHz), about 10 ms of
a decode step for the 28 layers' q_proj. All the GEMVs together take about 80%
of Qwen3's decode time (generality plan §13.2), which is why decode speed
follows DDR bandwidth, and why W4 weights are the next hardware step.

## 8. How the result is checked

- **Per dispatch**: `dispatch_check` runs every decode matvec on the
  functional simulator against a numpy oracle.
- **Golden corpus** (`compiler/scripts/run_tests.sh golden`): every unique
  dispatch of the reference exports (7708) must compile to identical
  descriptor bytes, so any change to the passes above that alters a binary is
  caught.
- **End to end**: the whole model on the simulator, and on the board, where
  this bundle's logits are bit-exact with the simulator in 17/17 steps.

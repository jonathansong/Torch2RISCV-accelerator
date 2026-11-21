# Two operators from PyTorch to the RTL, every intermediate form

This document follows two decode operators of SmolLM2-135M through every layer
of the stack:
- the PyTorch module and the exported MLIR;
- IREE's phases from GlobalOpt to the VM bytecode;
- every pass of the `sa` backend, SAHL to SAHW, and the sa-desc binary;
- the ARM runtime's command buffer and the ring entry it submits;
- the commands the device executes;
- the PicoRV32 firmware that serves the ring, and the two other ways the
  control core can issue the same commands;
- the RTL, clock by clock.

Every listing is real output, produced on 2026-11-21 with the current tree,
and trimmed only where marked. [§11](#11-reproducing) gives the commands.

The two operators:

| | **A: RMSNorm** | **B: q_proj** |
|---|---|---|
| Dispatch | `main$async_dispatch_4_elementwise_576_f32` | `main$async_dispatch_7_matvec_like_36x16x576_i8xi8xi32` |
| Computes | `y = x · rsqrt(Σx² / 576 + ε) · w` (the normalization; Σx² is dispatch 3) | `acc[n] = Σₖ x_q[k] · W_q[n, k]`, int8 × int8 → int32, 576 × 576 |
| Engine | vector engine (VE), fp32, with the SFU for rsqrt | the 4-port streaming GEMV unit |
| Called per decode step | 5 times (2 layers × 2 norms + the final norm), one executable | 2 times (one per layer), one executable |

They cover the two paths that most decode time goes through: vector work on
the VE, and weight streaming on the GEMV unit.

Related documents:
- [`dispatch_walkthrough.md`](dispatch_walkthrough.md) (attention scores) and
  [`gemv_walkthrough.md`](gemv_walkthrough.md) (Qwen3's q_proj, with a
  bit-by-bit descriptor table) go deeper into the compiler.
- [`execution_walkthrough.md`](execution_walkthrough.md) covers the PYNQ-Z1
  runtime and RTL in general.
- This document is the one that links all layers for the same two dispatches.

## Contents

0. [The bundle and the map of artefacts](#0-the-bundle-and-the-map-of-artefacts)
1. [PyTorch](#1-pytorch)
2. [The exported module (torch dialect)](#2-the-exported-module-torch-dialect)
3. [IREE: from linalg to the VM](#3-iree-from-linalg-to-the-vm)
4. [The sa backend, SAHL to SAHW](#4-the-sa-backend-sahl-to-sahw)
5. [The binary](#5-the-binary)
6. [The ARM runtime: command buffer, list, ring entry](#6-the-arm-runtime-command-buffer-list-ring-entry)
7. [What the device executes](#7-what-the-device-executes)
8. [The PicoRV32: the ring, PCPI, generated code](#8-the-picorv32-the-ring-pcpi-generated-code)
9. [The RTL, clock by clock](#9-the-rtl-clock-by-clock)
10. [What is checked at each layer](#10-what-is-checked-at-each-layer)
11. [Reproducing](#11-reproducing)

---

## 0. The bundle and the map of artefacts

**The bundle.** SmolLM2-135M, truncated to 2 layers, exported by
`hf_generic`:
- W8A8, int8 KV cache with K per channel, RoPE tables;
- decode plus a 16-row prefill;
- compiled with `compile_sa.sh` for the board's configuration (D = 16, 4 GEMV
  ports) with the default fusions and executable generalization.

All 146 executables compile for the accelerator, none for the host.
Everything is under `build/walk_e2e/`:

| Layer | Artefact | Produced by |
|---|---|---|
| PyTorch | `compiler/frontend/hf_generic.py` (`SaRMSNorm`, `QLinear`) | — |
| torch dialect | `smol/qllama.mlir`, `smol/qllama.irpa` | `test_hf_generic.py --export-only` |
| IREE phases | `phases/module.{1..12}.*.mlir` (input … flow, stream, hal, vm) | `iree-compile --dump-compilation-phases-to` |
| Dispatch sources | `smol/sa_sources/module_main$async_dispatch_{4,7}.mlir` | `--iree-hal-dump-executable-sources-to` |
| sa passes | `sa_d{4,7}/after_<pass>.mlir`, `d{4,7}_passes.txt` | `compiler/scripts/walk_dispatch.sh` |
| Binary | `sa_d{4,7}/d{4,7}.sadesc`, `.dis.txt` | the same (binary checked against `sa.vmfb`) |
| Module | `smol/sa.vmfb`, `smol/sa_packed.irpa` | `compile_sa.sh` |
| rv32 code (plan A) | `smol/rv_ll/*.ll`, `smol/rv/rvcode.bin` | `--iree-sa-emit-riscv`, `rv_image.py` |
| Ring entries and lists | `ring/ring_<n>.{entry,list,trace}.*` | `compiler/sim/sa_board_emu.py --dump` |
| PicoRV32 programs | `rvb/d{4,7}_{trace,compiled}/rvb_gen.{c,ll}`, `sim_bram/rvb_fw.elf` | `compiler/tests/rv_dispatch.py`, `firmware/rvb` |
| RTL waveforms | `rvb/d{4,7}_trace/sim_bram/wave.vcd` | xsim (`tb_system`, `RVB_TEST`) + `firmware/rvb/vcd.tcl` |

## 1. PyTorch

`hf_generic` replaces two kinds of modules in the Hugging Face model by
versions that fix the device's numerics.

**RMSNorm** (`SaRMSNorm`). The mean becomes a sum times the constant 1/n, so
the device needs no division:

```python
def forward(self, x):
    ss = torch.sum(x * x, dim=-1, keepdim=True)
    return (x * torch.rsqrt(ss * _f32c(1.0 / x.shape[-1]) + self.eps)) * self.weight
```

**Linear layers** (`QLinear`, `_qlinear`):
- int8 weights per output row with an fp32 scale;
- the activation quantized per token to int8 (`_quant_act`);
- an int32 product, then the dequantization.

The product is what dispatch B computes:

```python
xq, s_x = _quant_act(x)
acc = torch.matmul(rows.to(torch.int32), wq.to(torch.int32).t())
return (acc.to(torch.float32) * s_w) * s_x
```

## 2. The exported module (torch dialect)

`qllama.mlir` has three functions: `main` (decode), `prefill` and `prefill_kv`.

The first RMSNorm of `main` (layer 0, `input_layernorm`) follows the PyTorch
code operation for operation (lines 151–160):

```mlir
%43 = torch.aten.sum.dim_IntList %41, %42, %true, %none_47 : !torch.vtensor<[1,1,576],f32>, ... -> !torch.vtensor<[1,1,1],f32>
%44 = torch.aten.mul.Scalar %43, %float1.736110e-03 : ...           // × 1/576
%45 = torch.aten.add.Scalar %44, %float9.999990e-06, %int1_48 : ...  // + ε
%46 = torch.aten.rsqrt %45 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
%47 = torch.aten.mul.Tensor %24, %46 : !torch.vtensor<[1,1,576],f32>, !torch.vtensor<[1,1,1],f32> -> ...
%__auto.model.model.layers.0.input_layernorm.weight = util.global.load @__auto.model.model.layers.0.input_layernorm.weight : tensor<576xf32>
```

The q_proj's integer product (lines 199–217, trimmed): the int8 activation and
the int8 weight global, both extended to int32, the weight transposed, then a
matmul:

```mlir
%62 = torch.prims.convert_element_type %61, %int3 : !torch.vtensor<[1,576],si8>, !torch.int -> !torch.vtensor<[1,576],si32>
%__auto.model.model.layers.0.self_attn.q_proj.wq = util.global.load @__auto.model.model.layers.0.self_attn.q_proj.wq : tensor<576x576xi8>
%64 = torch.prims.convert_element_type %63, %int3_71 : !torch.vtensor<[576,576],si8>, !torch.int -> !torch.vtensor<[576,576],si32>
%65 = torch.aten.transpose.int %64, %int0_72, %int1_73 : ... -> !torch.vtensor<[576,576],si32>
```

The weights are not in the MLIR: `util.global.load` refers to `qllama.irpa`
(64.5 MB), the parameter archive.

## 3. IREE: from linalg to the VM

`--dump-compilation-phases-to` writes the whole module after each of IREE's 12
phases. The two operators, phase by phase:

**GlobalOpt** (`module.4.global-optimization.mlir`). The torch ops are linalg
generics now. The sum of squares is its own reduction; the normalization is
an elementwise generic over the 576 values:

```mlir
%16 = linalg.generic {iterator_types = ["parallel"]} ins(%9, %15 : tensor<576xf32>, tensor<f32>) outs(%8 : tensor<576xf32>) {
^bb0(%in: f32, %in_105: f32, %out: f32):
  %438 = arith.mulf %in_105, %cst_30 : f32      // Σx² × 1/576
  %439 = arith.addf %438, %cst_29 : f32         // + ε
  %440 = math.rsqrt %439 : f32
  %441 = arith.mulf %in, %440 : f32
  linalg.yield %441 : f32
} -> tensor<576xf32>
```

**Weight packing** (the plugin's preprocessing, `PackLinearWeights`). The
q_proj weight `576 × 576` becomes `[N/D, K, D]` = `36 × 576 × 16`: 36 strips of
16 output rows, each strip's k rows contiguous, which is the order the GEMV
unit streams them in. IREE evaluates the `linalg.pack` at compile time and
exports the result into `sa_packed.irpa` as
`layers.0.self_attn.q_proj.wq$packed`.

**DispatchCreation** (`module.5.dispatch-creation.mlir`). The normalization,
fused with the gain `w`, is one dispatch region:

```mlir
%26 = flow.dispatch.workgroups(%21, %25, %cst_5) : (tensor<576xf32>, tensor<f32>, tensor<576xf32>) -> tensor<576xf32> =
    (%arg5: ...readonly:tensor<576xf32>, %arg6: ...readonly:tensor<f32>, %arg7: ...readonly:tensor<576xf32>, %arg8: ...) {
  %106 = linalg.generic {iterator_types = ["parallel"]} ins(%102, %103, %104 : ...) outs(%105 : tensor<576xf32>) {
  ^bb0(%in: f32, %in_28: f32, %in_29: f32, %out: f32):
    %107 = arith.mulf %in_28, %cst_26 : f32
    %108 = arith.addf %107, %cst_27 : f32
    %109 = math.rsqrt %108 : f32
    %110 = arith.mulf %in, %109 : f32
    %111 = arith.mulf %110, %in_29 : f32     // × w (the norm's gain, fused in)
    linalg.yield %111 : f32
  } -> tensor<576xf32>
  ...
```

**Flow** (`module.6.flow.mlir`). Regions become executables, and identical
executables are merged.
- With executable generalization (IREE patch 0003), the five RMSNorms of a
  decode step share `@main$async_dispatch_4`, and the two layers' q_proj share
  `@main$async_dispatch_7`.
- The call sites differ only in their operands:

```mlir
%11 = flow.dispatch @main$async_dispatch_4::@main$async_dispatch_4_elementwise_576_f32(%6, %10, %__constant_tensor_576xf32_5) : (tensor<576xf32>, tensor<f32>, tensor<576xf32>) -> tensor<576xf32>
%14 = flow.dispatch @main$async_dispatch_7::@main$async_dispatch_7_matvec_like_36x16x576_i8xi8xi32(%13, %__parameter_model___auto_model_model_layers_0_self_attn_q_proj_wq_packed_tensor_36x576x16xi8) : (tensor<576xi8>, tensor<36x576x16xi8>) -> tensor<36x16xi32>
```

The q_proj executable's body is the packed matvec:

```mlir
%4 = linalg.generic {iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<576xi8>, tensor<36x576x16xi8>) outs(%3 : tensor<36x16xi32>) {
^bb0(%in: i8, %in_0: i8, %out: i32):
  %5 = arith.extsi %in : i8 to i32
  %6 = arith.extsi %in_0 : i8 to i32
  %7 = arith.muli %5, %6 : i32
  %8 = arith.addi %out, %7 : i32
  linalg.yield %8 : i32
} -> tensor<36x16xi32>
```

**Stream** (`module.7.stream.mlir`). Tensors become resources, and the
per-call offsets into them become push constants. All activations of a step
live in one 7424-byte transient resource. The first RMSNorm call:

```mlir
stream.cmd.dispatch @main$async_dispatch_4::@main$async_dispatch_4_elementwise_576_f32(%c0_i32, %c2816_i32, %c13824_i32, %c2880_i32 : i32, i32, i32, i32) {
  ro %arg25[%c0 for %c7424] : !stream.resource<transient>{%c7424},     // x at +0, Σx² at +2816
  ro %arg5[%c0 for %c612608] : !stream.resource<constant>{%c612608},   // w at +13824 (all norm gains in one constant)
  wo %arg25[%c0 for %c7424] : !stream.resource<transient>{%c7424}      // y at +2880
}
```

**Executable sources** (`module.8`, also dumped per dispatch into
`sa_sources/`). What the sa backend receives: the same linalg on
`hal.interface.binding.subspan`s at offsets taken from the push constants.
`util.assume.int` lists the values each constant takes at the five call sites
(offsets 0, 2816, 5120, 704, 4544 for x, and so on); the backend uses it to
check alignment.

**HAL** (`module.11.hal.mlir`). Each call becomes a
`hal.command_buffer.dispatch` with three bindings and the constants:

```mlir
hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%__device_0_executable_3_main$async_dispatch_4 : !hal.executable)[%c0]
    workgroups([%31, %c1, %c1]) constants([%c0_i32, %c2816_i32, %c13824_i32, %c2880_i32]) bindings([
  (%transient_buffer_3 : !hal.buffer)[%c0, %c7424],
  (%__constant_tensor_576xf32 : !hal.buffer)[%c0, %c612608],
  (%transient_buffer_3 : !hal.buffer)[%c0, %c7424]
]) flags("None")
hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%__device_0_executable_6_main$async_dispatch_7 : !hal.executable)[%c0]
    workgroups([%c1, %c1, %c1]) constants([%c5184_i32, %c2880_i32]) bindings([
  (%transient_buffer_3 : !hal.buffer)[%c0, %c7424],
  (%__parameter_model___auto_model_model_layers_0_self_attn_q_proj_wq_packed_tensor_36x576x16xi8 : !hal.buffer)[%c0, %c331776],
  (%transient_buffer_3 : !hal.buffer)[%c0, %c7424]
]) flags("None")
```

Each executable has two variants: `sa_desc_v1` and the host fallback
`vmvx_bytecode_fb`. `hal.executable.create` picks the first one the device
supports.

**VM** (`module.12.vm.mlir`). Calls into the HAL module. The sa-desc binary
is a `vm.rodata` blob in the module:

```mlir
vm.rodata private @main_async_dispatch_4_sa_desc_v1 {alignment = 16 : i64, mime_type = "application/x-sa-desc"} dense<"0x534144455343310003000000...">
%ref_14 = vm.call @hal.command_buffer.create(%__device_0, %c1, %c3, %c-1, %zero_1) : ...
vm.call.variadic @hal.command_buffer.dispatch(%ref_14, %__device_0_executable_3_main$async_dispatch_4, %zero_1, %24, %c1, %c1, %zero,
    [%zero_1, %c2816, %c13824, %c2880],
    [(%zero_1, %zero_1, %ref_12, %zero, %c7424), (%zero_1, %zero_1, %__constant_tensor_576xf32, %zero, %c612608), (%zero_1, %zero_1, %ref_12, %zero, %c7424)]) : ...
...
vm.call @hal.device.queue.execute(%__device_0, %c-1, %ref_13, %ref_15, %ref_14, %zero) : ...
```

A decode step records all its dispatches (46 call sites in `main`, some under
branches) into one command buffer, executed by one `queue.execute`.

## 4. The sa backend, SAHL to SAHW

`walk_dispatch.sh` recompiles each dispatch alone, printing the IR after every
sa pass, and checks that the binary is byte for byte the one in `sa.vmfb`. The
pipeline: bufferize → `sa-to-sahl` → `sahl-tile` → `sahl-plan-memory` →
`sahl-schedule` → `sahl-expand-kernels` → `sahl-to-sahw` → `sahw-fuse-ve` →
`sahw-split-head` → `sahw-assign-registers` → serialize.

### 4.1 Dispatch A (RMSNorm)

**`sa-to-sahl`.** The binding loads and stores become `sahl.load` /
`sahl.store` into local buffers. The generic stays as it is: the VE lowering
handles it later.

```mlir
%alloc_1 = memref.alloc() : memref<576xf32>
sahl.load %9, %alloc_1 : memref<576xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<576xf32>   // x
%alloc_2 = memref.alloc() : memref<f32>
sahl.load %10, %alloc_2 : memref<f32, strided<[], offset: ?>, ...>, memref<f32>                                            // Σx²
%alloc_3 = memref.alloc() : memref<576xf32>
sahl.load %11, %alloc_3 : memref<576xf32, strided<[1], offset: ?>, ...>, memref<576xf32>                                  // w
linalg.generic {...} ins(%alloc_1, %alloc_2, %alloc_3 : ...) outs(%alloc : memref<576xf32>) { ... math.rsqrt ... }
sahl.store %alloc, %12 : memref<576xf32>, memref<576xf32, strided<[1], offset: ?>, ...>                                  // y
```

**`sahl-schedule`.** It assigns each buffer a memory and a layout: here all
are fp32 vectors in ACC, "packed" (16 values per 64-byte word):
`memref.alloc() {sa.layout = "packed", sa.mem = "acc"}`. Tiling and kernel
expansion leave this dispatch unchanged: 576 values fit, and there is no
micro-kernel to expand.

**`sahl-to-sahw`.** One `sahw.template` with one hardware command per op:
- three loads;
- seven VE operations: the scalar Σx² broadcast, × 1/576, + ε, rsqrt, × x,
  × w, a copy;
- one store.

Local addresses are `0x3000_0000 + ACC word`:

```mlir
sahw.template "main$async_dispatch_4_elementwise_576_f32" attributes {sa.d = 16 : i64} {
  %0 = sahw.base binding(0) {constant = 0 : i64}     // BASE0 = binding 0 + c0  (x)
  %1 = sahw.base binding(0) {constant = 1 : i64}     // BASE1 = binding 0 + c1  (Σx²)
  %2 = sahw.base binding(1) {constant = 2 : i64}     // BASE2 = binding 1 + c2  (w)
  %3 = sahw.base binding(2) {constant = 3 : i64}     // BASE3 = binding 2 + c3  (y)
  sahw.body {
    sahw.ld %0 {ddr = 0, laddr = 805306880 /*ACC 0x200*/, row_bytes = 2304, rows = 1, pitch = 2304}
    sahw.ld %1 {ddr = 0, laddr = 805306916 /*ACC 0x224*/, row_bytes = 8, rows = 1, pitch = 8}
    sahw.ld %2 {ddr = 0, laddr = 805306917 /*ACC 0x225*/, row_bytes = 2304, rows = 1, pitch = 2304}
    sahw.ve {op = 5, src1 = 0x224, dst = 0x249, length = 16, reduce = 2, fp = true, ...}           // Σx² to all 16 lanes
    sahw.ve {op = 5, src1 = 0x249, dst = 0x24a, length = 16, a = 0.00173611112, fp = true, ...}    // × 1/576
    sahw.ve {op = 5, src1 = 0x24a, dst = 0x24b, length = 16, b = 9.99999974E-6, fp = true, ...}    // + ε
    sahw.ve {op = 5, src1 = 0x24b, dst = 0x24c, length = 16, func = 3, fp = true, ...}             // rsqrt (SFU)
    sahw.ve {op = 2, src1 = 0x200, src2 = 0x24c, dst = 0x24d, length = 576, m2 = 2, period = 65535} // x × r (r broadcast)
    sahw.ve {op = 2, src1 = 0x24d, src2 = 0x225, dst = 0x271, length = 576, ...}                   // × w
    sahw.ve {op = 5, src1 = 0x271, dst = 0x295, length = 576, ...}                                 // copy to the output
    sahw.st %3 {ddr = 0, laddr = 805307029 /*ACC 0x295*/, row_bytes = 2304, rows = 1, pitch = 2304}
  }
}
```

(Addresses written in hex and attributes trimmed for reading. The real
attributes are decimal; see `sa_d4/after_iree-sahl-to-sahw.mlir`.)

**`sahw-fuse-ve`.** A VE command computes `func(A · src1 + B)` in one pass, so
the peephole merges chains of single-stage operations:
- × 1/576, + ε and rsqrt become one command with `A = 1/576`, `B = ε`,
  `func = rsqrt`;
- the final copy disappears: × w writes the output word directly.

Seven VE commands become four:

```mlir
sahw.ve {op = 5, src1 = 0x224, dst = 0x249, length = 16, reduce = 2, ...}
sahw.ve {op = 5, src1 = 0x249, dst = 0x24c, length = 16, a = 0.00173611112, b = 9.99999974E-6, func = 3, ...}
sahw.ve {op = 2, src1 = 0x200, src2 = 0x24c, dst = 0x24d, length = 576, m2 = 2, period = 65535, ...}
sahw.ve {op = 2, src1 = 0x24d, src2 = 0x225, dst = 0x295, length = 576, ...}
```

**`sahw-split-head`.** The leading loads become the *head*, ending in RET; the
driver may issue them early, overlapping the previous dispatch (§6). The
disassembly of the final binary:

```
main$async_dispatch_4_elementwise_576_f32: 10 descriptors, 3 bindings, 4 constants; prefix 0, head 4, reads 0x3, writes 0x4, SPAD_B False
     setup BASE0 = binding 0 + ((c0 * 1) / 1) + 0
     setup BASE1 = binding 0 + ((c1 * 1) / 1) + 0
     setup BASE2 = binding 1 + ((c2 * 1) / 1) + 0
     setup BASE3 = binding 2 + ((c3 * 1) / 1) + 0
   0 LD       ddr=0x0 la=0x30000200 rows=1 rb=2304 pitch=2304 base=0
   1 LD       ddr=0x0 la=0x30000224 rows=1 rb=8 pitch=8 base=1
   2 LD       ddr=0x0 la=0x30000225 rows=1 rb=2304 pitch=2304 base=2
   3 RET
   4 VE       op=5 s1=0x30000224 s2=0x0 dst=0x30000249 len=16 types=0xf func=0 m1=0 m2=0 red=2 A=1.0 B=-0.0 ...
   5 VE       op=5 s1=0x30000249 s2=0x0 dst=0x3000024c len=16 types=0xf func=3 m1=0 m2=0 red=0 A=0.0017361111240461469 B=9.999999747378752e-06 ...
   6 VE       op=2 s1=0x30000200 s2=0x3000024c dst=0x3000024d len=576 types=0xf func=0 m1=0 m2=2 red=0 A=1.0 B=-0.0 ... period=65535 ...
   7 VE       op=2 s1=0x3000024d s2=0x30000225 dst=0x30000295 len=576 types=0xf func=0 m1=0 m2=0 red=0 A=1.0 B=-0.0 ...
   8 ST       ddr=0x0 la=0x30000295 rows=1 rb=2304 pitch=2304 base=3
   9 RET
```

### 4.2 Dispatch B (q_proj)

**`sa-to-sahl`.** It recognizes the packed matvec as the `linear`
micro-kernel:

```mlir
sahl.kernel "linear" {
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<36x16xi32>)
  sahl.load %5, %alloc_0 : memref<576xi8, ...>, memref<576xi8>                     // x_q
  sahl.load %6, %alloc_1 : memref<36x576x16xi8, ...>, memref<36x576x16xi8>         // the packed weights
  linalg.generic {iterator_types = ["parallel", "parallel", "reduction"]} ins(...) outs(%alloc : memref<36x16xi32>) { ... }
  sahl.store %alloc, %7 : memref<36x16xi32>, memref<36x16xi32, strided<[16, 1], offset: ?>, ...>
}
```

**`sahl-expand-kernels`.** On a target with the GEMV unit
(`--iree-sa-gemv-ports=4`), the kernel expands into GEMV commands:
- x is loaded once;
- each `sahl.gemv` streams 6 strips (6 × 576 × 16 = 55,296 bytes) from DDR
  through the GEMV unit, leaving 6 ACC words of 16 int32 sums;
- the 36 strips are written back in two halves of 18 words, so the second
  half's GEMVs can run while the first half is stored.

```mlir
sahl.load %5, %alloc : memref<576xi8, ...>, memref<576xi8>
sahl.gemv %alloc, %subview,   %subview_4 : memref<576xi8>, memref<6x576x16xi8, strided<[9216, 16, 1]>, ...>, memref<6x16xi32, ...>
sahl.gemv %alloc, %subview_5, %subview_6 : memref<576xi8>, memref<6x576x16xi8, strided<[9216, 16, 1], offset: 55296>, ...>, ...
sahl.gemv %alloc, %subview_7, %subview_8 : ..., offset: 110592>, ...
sahl.store %alloc_0, %subview_9 : memref<18x16xi32>, ...
sahl.gemv ... offset: 165888 ...
sahl.gemv ... offset: 221184 ...
sahl.gemv ... offset: 276480 ...
sahl.store %alloc_2, %subview_16 : memref<18x16xi32>, ...
```

**`sahl-to-sahw`.** Each `sahl.gemv` is an LD in mode 2 (GEMV); the x load
targets SPAD_A (`0x1000_0000`):

```mlir
sahw.ld %0 {ddr = 0, laddr = 268435456 /*SPAD_A 0*/, row_bytes = 576, rows = 1, pitch = 576}
sahw.ld %1 {mode = 2, ddr = 0,      laddr = 805306368 /*ACC 0*/, rows = 6, row_bytes = 9216, pitch = 9216, cstep = 1}
sahw.ld %1 {mode = 2, ddr = 55296,  laddr = 805306374 /*ACC 6*/, rows = 6, row_bytes = 9216, pitch = 9216, cstep = 1}
sahw.ld %1 {mode = 2, ddr = 110592, laddr = 805306380 /*ACC 12*/, ...}
sahw.st %2 {ddr = 0, laddr = 805306368, row_bytes = 1152, rows = 1, pitch = 1152}     // 18 words × 64 B
sahw.ld %1 {mode = 2, ddr = 165888, laddr = 805308416 /*ACC 0x800, the other bank*/, ...}
...
sahw.st %2 {ddr = 1152, laddr = 805308416, row_bytes = 1152, rows = 1, pitch = 1152}
```

The second half's results go to the other ACC bank (word 0x800). The scheduler
can therefore run its GEMVs while the store reads the first bank.

Disassembly (the head is the x load, plus the first half's three GEMVs):

```
main$async_dispatch_7_matvec_like_36x16x576_i8xi8xi32: 11 descriptors, 3 bindings, 2 constants; prefix 0, head 5, reads 0x3, writes 0x4, SPAD_B False
     setup BASE0 = binding 0 + ((c0 * 1) / 1) + 0
     setup BASE1 = binding 1 + ((c-1 * 1) / 1) + 0
     setup BASE2 = binding 2 + ((c1 * 1) / 1) + 0
   0 LD       ddr=0x0 la=0x10000000 rows=1 rb=576 pitch=576 base=0
   1 LD       ddr=0x0 la=0x30000000 rows=6 rb=9216 pitch=9216 base=1
   2 LD       ddr=0xd800 la=0x30000006 rows=6 rb=9216 pitch=9216 base=1
   3 LD       ddr=0x1b000 la=0x3000000c rows=6 rb=9216 pitch=9216 base=1
   4 RET
   5 ST       ddr=0x0 la=0x30000000 rows=1 rb=1152 pitch=1152 base=2
   6 LD       ddr=0x28800 la=0x30000800 rows=6 rb=9216 pitch=9216 base=1
   7 LD       ddr=0x36000 la=0x30000806 rows=6 rb=9216 pitch=9216 base=1
   8 LD       ddr=0x43800 la=0x3000080c rows=6 rb=9216 pitch=9216 base=1
   9 ST       ddr=0x480 la=0x30000800 rows=1 rb=1152 pitch=1152 base=2
  10 RET
```

(`c-1` means no constant: the weights are always at the start of their
binding. `sadis.py` shows mode-2 LDs like plain LDs; the raw bits are in §5.)

## 5. The binary

`d4.sadesc` is 912 bytes, `d7.sadesc` 976. The layout (header, export table,
templates of 64-byte descriptors, strings, register setup table, export
extension) is described field by field in
[`gemv_walkthrough.md`](gemv_walkthrough.md) §5. The header of `d4.sadesc`:

```
00000000: 5341 4445 5343 3100 0300 0000 1000 0000  SADESC1.  version 3, D = 16
00000010: 0000 a003 0100 0000 4000 0000 8000 0000  required CAPS 0x03a00000 (bits 21, 23, 24, 25), 1 export, ...
```

Three descriptors of dispatch A, as eight little-endian 64-bit words:

```
0 (LD x)       0000000000000101 0000000000000000 0900000130000200 0000000000000900 0 0 0 0
5 (VE rsqrt)   0000000400000004 0000000030000249 000000103000024c 00e0000100000f05 8000000000000000 000000007fffffff 3727c5ac3ae38e39 0
6 (VE x × r)   0000000500000004 3000024c30000200 000002403000024d 10200001ffff0f02 8000000000000000 000000007fffffff 800000003f800000 0
```

- **Descriptor 0**: opcode `0x01` (LD), w0[8] = 1 (the DDR address is relative
  to a BASE register), BASE0. w2 = local address `0x3000_0200` (ACC word
  0x200), 1 row of `0x900` = 2304 bytes.
- **Descriptor 5**, the fused VE: opcode `0x04`, tag 4. w1 = src1, w2 = dst
  and length 16. w3[7:0] = 5 (the operation), w3[11:8] = 0xf (all four types
  fp32), flags w3[63:53] = 0x7 (fp, func 3 = rsqrt). w4 / w5 = the clamp range.
  w6 = **A = `0x3ae38e39` (1/576), B = `0x3727c5ac` (1e-5)**.
- **Descriptor 6**: opcode `0x04`, operation 2 (multiply), w3 = `…ffff…`:
  src2 repeats with period 65535, so the one value r is broadcast to all 576
  lanes.

The first GEMV of dispatch B, at offset `0xc0`:

```
000000c0: 0103 0000 0100 0000  0000 0000 0000 0000   w0 = 0x0000000100000301: LD, BASE1, tag 1;  w1 = ddr 0
000000d0: 0000 0030 0600 0024  0024 0000 0200 0400   w2 = ACC 0, 6 rows of 0x2400 bytes;  w3 = pitch 0x2400, mode 2, x word 0, C step 1
```

Every number in these words comes from the SAHW attributes above.

## 6. The ARM runtime: command buffer, list, ring entry

On the board, `sa-llm-run` runs the IREE VM on the ARM.
- The VM's `hal.command_buffer.dispatch` calls reach the sa HAL driver
  (`compiler/runtime/sa`).
- The driver's loader (`sa_loader.c`) has copied each executable's templates
  into the device window, a u-dma-buf region shared with the accelerator.
- For a dispatch, the command buffer (`sa_command_buffer.c`, batched by
  `sa_context`) appends to **one descriptor list per command buffer**:
  1. SETREG descriptors loading the BASE registers, each = binding address +
     push constant, as the setup table says;
  2. a CALL of the template's head and a CALL of its body;
  3. a FENCE where the next dispatch reads what an earlier one wrote.
- `queue.execute` writes the list's address into a 64-byte entry of the
  submission ring and rings the doorbell. `rt_fw` on the PicoRV32 runs it (§8).

**How it was captured.** `compiler/sim/sa_board_emu.py` runs the unmodified
board transport (`sa_transport_board.c`) on the host, with rt_fw's side of the
ring served by the functional simulator. Its `--dump DIR` option, added for
this document, saves each ring entry, its list and the commands executed. Three
decode steps (tokens 1, 504, then one generated) gave three ring entries, all
with status 0.

**Ring entry 1** (decode step 2), as eight 64-bit words:

```
0000000100000001 0000000010010880 0000000000000000 0 0 0 0 0
```

- w0: type 1 = `RT_RUN_LIST`, sequence number 1;
- w1: the list at `0x1001_0880`;
- w2 = 0: count 0, run to the END descriptor;
- w3..w6: BASE0..3 = 0;
- w7 = 0: no parameter block. The dispatches set their own registers.

**The list.** 192 descriptors for the whole decode step: 73 SETREG, 83 CALL,
35 FENCE and one END. Dispatch A's first call (layer 0's input_layernorm) and
dispatch B's first call (layer 0's q_proj):

```
12 SETREG 0000000000000014 0000000000424140 000000001238a000 000000001238ab00 00000000100d5cc0
13 SETREG 0000000000000014 0000000000000043 000000001238ab40 0 0
14 CALL   0000000000000015 0000000010091b40                       head of dispatch 4 (descriptors 0-3)
15 CALL   0000000000000015 0000000010091c40                       body of dispatch 4 (+0x100 = descriptor 4)
16 FENCE  0000000000000010 0000000000000003
 ...
25 SETREG 0000000000000014 0000000000424140 000000001238b440 0000000011c68080 000000001238ab40
26 CALL   0000000000000015 0000000010092280                       head of dispatch 7 (descriptors 0-4)
27 CALL   0000000000000015 00000000100923c0                       body of dispatch 7 (+0x140 = descriptor 5)
 ...
191 END   0000000000000012 000000000000005a
```

**Reading the SETREG.** w1 = `0x424140` holds three register selectors:
`0x40` | BASE0, `0x41` | BASE1, `0x42` | BASE2 (bit 6 = valid). w2..w4 are the
values. For dispatch A, check them against the push constants (0, 2816,
13824, 2880):

| Register | Value | = binding + constant |
|---|---|---|
| BASE0 (x) | `0x1238_a000` | transient buffer + 0 |
| BASE1 (Σx²) | `0x1238_ab00` | transient buffer + 2816 (0xb00) |
| BASE2 (w) | `0x100d_5cc0` | norm-gain constant + 13824 (0x3600) |
| BASE3 (y), entry 13 | `0x1238_ab40` | transient buffer + 2880 (0xb40) |

For dispatch B (constants 5184, 2880): BASE0 = `0x1238_b440` = transient +
0x1440 (x_q). BASE1 = `0x11c6_8080` = layer 0's packed q_proj weights, a
binding at constant offset 0. BASE2 = `0x1238_ab40` = transient + 0xb40.

**The FENCE** (`w1 = 3`, an engine mask: bit 0 LD, bit 1 ST) waits for the LD
and ST engines, because the next dispatch reads what this one stored. **The two CALLs**
are the head/body split of §4: when consecutive dispatches have no
dependency, the batcher issues the next dispatch's head before this
dispatch's body, so its loads overlap the current work.

## 7. What the device executes

The emulator's trace records each command after the fetch unit has applied
BASE registers and dynamic fields. These are the real DDR addresses the
engines see. From `ring_1.trace.npy` (dispatch A, then dispatch B; `dl` = the
descriptor counter of the list):

```
  dl  cmd    hdr        ddr        w2 (local, rows, bytes)  w3
  89  LD     00000101   1238a000   0900000130000200         0000000000000900     x       -> ACC 0x200
  90  LD     00000301   1238ab00   0008000130000224         0000000000000008     Σx²     -> ACC 0x224
  91  LD     00000501   100d5cc0   0900000130000225         0000000000000900     w       -> ACC 0x225
  94  VE     00000004              0000001030000249         4020000100000f05     Σx² broadcast
  95  VE     00000004              000000103000024c         00e0000100000f05     rsqrt(Σx² / 576 + ε)
  96  VE     00000004              000002403000024d         10200001ffff0f02     x × r
  97  VE     00000004              0000024030000295         0020000100000f02     × w
  98  ST     00000702   1238ab40   0900000130000295         0000000000000900     y       -> DDR
 100  FENCE
 ...
 126  LD     00000101   1238b440   0240000110000000         0000000000000240     x_q     -> SPAD_A 0
 127  LD     00000301   11c68080   2400000630000000         0004000200002400     GEMV strips 0-5   -> ACC 0..5
 128  LD     00000301   11c75880   2400000630000006         0004000200002400     GEMV strips 6-11  -> ACC 6..11
 129  LD     00000301   11c83080   240000063000000c         0004000200002400     GEMV strips 12-17 -> ACC 12..17
 132  ST     00000502   1238ab40   0480000130000000         0000000000000480     18 words -> DDR
 133  LD     00000301   11c90880   2400000630000800         0004000200002400     GEMV strips 18-23 -> ACC 0x800..
 134  LD     00000301   11c9e080   2400000630000806         0004000200002400
 135  LD     00000301   11cab880   240000063000080c         0004000200002400
 136  ST     00000502   1238afc0   0480000130000800         0000000000000480     18 words -> DDR + 0x480
```

The GEMV loads stream 55,296 bytes each, contiguous from the packed table
(`0x11c68080`, `+0xd800`, …). The functional simulator computes exactly what
the RTL does (§9), so this trace is also the reference the board is compared
against.

## 8. The PicoRV32: the ring, PCPI, generated code

The control core can issue a dispatch's commands three ways. All three were
run on the RTL for both dispatches (§9).

### 8.1 rt_fw: serving the ring (the default)

`firmware/rt/rt_fw.c` (161 lines) is loaded once by the ARM and loops forever:

```c
for (;;) {
    if (MBOX(MBOX_RING_TAIL) == head) {             /* doorbell polled from the BRAM */
        MBOX(MBOX_HEARTBEAT) = ++beat;
        continue;
    }
    uint32_t slot = head & (size - 1);
    volatile uint32_t *e = (volatile uint32_t *)(ring + 64u * slot);
    ...
    if (type == RT_RUN_LIST) {
        for (uint32_t i = 0; i < 4; i++)
            mat_cfg(SA_CFG_BASE(i), e[6 + 2 * i]);       /* BASE0..3 from the entry */
        uint32_t pb = e[14];
        if (pb)
            for (uint32_t k = 0; k < 8; k++)
                mat_cfg(SA_CFG_PARAM(k), ((volatile uint32_t *)pb)[k]);
        uint32_t i0 = rdinstret(), t0 = rdcycle();
        mat_submit(e[2], e[4]);                         /* the fetch unit runs the list */
        uint32_t st = mat_fence(SA_ENG_ALL);            /* wait for every engine */
        cycles = rdcycle() - t0;
        ...
    }
    c[1] = status; c[2] = cycles; c[3] = exec; c[4] = dstat; ...
    c[0] = seq;
    MBOX(MBOX_RING_HEAD) = ++head;
}
```

The same code as rv32 machine code (`llvm-objdump` of `rt_fw.elf`; custom-0
instructions print as `<unknown>`):

```
c0000116: 0c 4d        lw   a1, 0x18(a0)        # e[6]: BASE0
c0000118: 0b 80 b8 02  <unknown>                # 0x02b8800b: custom-0, funct7 1, funct3 0 = mat_cfg(key, a1) (key preloaded in a7)
c000011c: 0c 51        lw   a1, 0x20(a0)        # e[8]: BASE1
c000011e: 0b 80 b2 02  <unknown>                # mat_cfg ...
 ...
c000018c: 73 28 20 c0  rdinstret a6
c0000190: f3 24 00 c0  rdcycle s1
c0000194: 10 45        lw   a2, 0x8(a0)         # e[2]: the list address
c0000196: 08 49        lw   a0, 0x10(a0)        # e[4]: the count
c000019a: 0b 60 a6 02  <unknown>                # 0x02a6600b: custom-0, funct7 1, funct3 6 = mat_submit(a2, a0)
c000019e: 0b 46 07 02  <unknown>                # funct3 4 = mat_fence: rd = the unit's status
```

The custom instructions reach `sa_unit` through PicoRV32's PCPI port. In
`sa_pcpi.v`, `mat_submit` hands the list address to `sa_cmdfetch.v`, the
descriptor fetch unit. From there the hardware alone fetches the 192
descriptors over AXI, executes SETREG / CALL / RET / FENCE itself, and passes
LD / ST / EX / VE commands to the scheduler (`sa_sched.v`). The core only waits
in `mat_fence`.

### 8.2 PCPI: the core issues every command

`rv_dispatch.py --code trace` writes the dispatch's commands as straight-line
C over the same custom instructions (`rvb_gen.c`): one `mat_cfg` / `vec_cfg`
per changed configuration key, then `mat_load` / `vec_run` / `mat_store`.
Dispatch A is 50 PCPI instructions:

```c
mat_cfg(SA_CFG_LD_ROWS, 0x1u);
mat_cfg(SA_CFG_LD_ROW_BYTES, 0x900u);
mat_cfg(SA_CFG_LD_PITCH, 0x900u);
mat_cfg(SA_CFG_LD_MODE, 0x0u);
mat_load(W + 0x1000u, 0x30000200u);              /* x */
...
vec_cfg(SA_VCFG_DST, 0x3000024cu);
vec_cfg(SA_VCFG_FLAGS, 0x7u);                     /* fp, func = rsqrt */
vec_cfg(SA_VCFG_A, 0x3ae38e39u);                  /* 1/576 */
vec_cfg(SA_VCFG_B, 0x3727c5acu);                  /* 1e-5 */
vec_cfg(SA_VCFG_ROW, 0x0u);
vec_run(0x30000249u, 0x0u);                       /* the fused VE of descriptor 5 */
...
mat_store(W + 0x6b40u, 0x30000295u);
mat_fence(0);
```

Dispatch B is 20 PCPI instructions: the x load, then for each GEMV the mode,
C step and address.

### 8.3 Plan A: the compiler's own rv32 code

With `--iree-sa-emit-riscv`, the backend also lowers each SAHW template to
LLVM IR (`SahwToRiscv.cpp`): a function of the dispatch's bindings and push
constants that computes the BASE values and issues the commands through
`sa_rv_*` runtime calls. `smol/rv_ll/` holds one `.ll` per export, and
`rv_image.py` links them into `smol/rv/rvcode.bin` (220.6 KB for 146 exports).
From dispatch A's (`rvb/d4_compiled/rvb_gen.ll`, 127 lines):

```llvm
define void @main_async_dispatch_4_elementwise_576_f32(ptr %bindings, ptr %push_constants) #0 {
  %1 = load i32, ptr %0, align 4                 ; binding 0
  %3 = load i32, ptr %2, align 4                 ; constant 0
  ...
  %9 = add i32 %1, %8                            ; BASE0 = binding 0 + c0
  store i32 %9, ptr %base, align 4
  ...
  call void @sa_rv_mat_cfg(i32 1, i32 2304)
  call void @sa_rv_mat_load(i32 %41, i32 805306880)
  ...
  call void @sa_rv_vec_cfg(i32 12, i32 987991609)   ; A = 0x3ae38e39
  call void @sa_rv_vec_cfg(i32 13, i32 925353388)   ; B = 0x3727c5ac
  call void @sa_rv_vec_run(i32 805306953, i32 0)
```

Compiled for rv32imc (`rvb_fw.elf`):

```
c00000ac <main_async_dispatch_4_elementwise_576_f32>:
c00000c4: 04 41        lw   s1, 0x0(a0)        # binding 0
c00000c6: 80 41        lw   s0, 0x0(a1)        # constant 0
 ...
c00000d6: 26 94        add  s0, s0, s1         # BASE0
c00000de: 85 45        li   a1, 0x1
c00000e0: 01 45        li   a0, 0x0
c00000e2: 97 00 00 00  auipc ra, 0x0
c00000e6: e7 80 60 2b  jalr 0x2b6(ra) <sa_rv_mat_cfg>
 ...
c00003aa <sa_rv_mat_load>:
c00003aa: 0b 10 b5 02  <unknown>               # custom-0, funct3 1 = mat_load(a0, a1)
c00003ae: 82 80        ret
```

On the board, an export marked `rv_only` (the MoE Top-k) always runs this
way, through an `RT_CALL` ring entry. Any export can, with `SA_RV=embedded`.

## 9. The RTL, clock by clock

`firmware/rvb` runs one dispatch on the system testbench `tb_system`
(`RVB_TEST`):
- the PicoRV32 RTL and `sa_unit` (D = 16, 128-bit DMA, the GEMV unit);
- a DDR model loaded with the dispatch's inputs (`rv_dispatch.py` lays them out
  as `dispatch_check.py` does);
- xsim.

It runs the dispatch twice: as its descriptor list (`mat_submit`, as rt_fw
does) and over PCPI (§8.2 or §8.3). After each run it compares the output
bytes with the functional simulator's.

| Dispatch | Program | Descriptor list | PCPI / generated code | Bytes vs functional simulator |
|---|---|---|---|---|
| A (RMSNorm) | straight-line PCPI (§8.2), 554-byte image | 2741 cycles | 2517 | **PASS**, both runs |
| A | plan A code (§8.3), 1100-byte image | 2741 | 2863 | **PASS** |
| B (q_proj) | straight-line PCPI, 392-byte image | 87142 | 86815 | **PASS** |
| B | plan A code, 712-byte image | 87142 | 87320 | **PASS** |

For dispatch A, PCPI is faster than the list: the code sits in the BRAM, while
the fetch unit reads each descriptor from the DDR model. Plan A's function
calls add about 350 cycles. For dispatch B the three are within 0.6%: all the
time is weight streaming.

**Waveforms.** Elaborating the same testbench with `-debug typical` and
logging the fetch / scheduler / engine handshakes (`firmware/rvb/vcd.tcl`)
gives a VCD. `compiler/tests/vcd_timeline.py` turns it into events, in cycles
from the fetch unit's start.

Dispatch A, descriptor list:

```
    117  LD issued                 x (2304 B)
    889  LD issued / LD done       Σx² (8 B); x done
    933  LD issued / LD done       w (2304 B)
   1121  fetch idle (END reached)  the fetch unit has read the whole list
   1588  LD done
   1590  VE issued                 Σx² broadcast   (waits for the LDs: the scoreboard on ACC words)
   1624  VE issued / VE done       rsqrt(Σx² / 576 + ε)
   1744  VE issued / VE done       x × r   (576 values, 36 words)
   1880  VE issued / VE done       × w
   2016  VE done
   2018  ST issued                 y (2304 B)
   2713  ST done
   2714  unit idle: list done
counts {'rbeat': 373, 'ld': 3, 've': 4, 'st': 1}
```

The fetch unit runs ahead: it has read the whole list by cycle 1121, and the
scheduler holds the VEs until their inputs have landed. The 4 VE commands
take about 430 cycles; the rest is DDR traffic.

Dispatch B, descriptor list:

```
     84  LD issued                 x_q (576 B) -> SPAD_A
    282  LD issued / LD done       GEMV strips 0-5
   1252  fetch idle (END reached)
  14851  LD issued / LD done       GEMV strips 6-11
  29315  LD issued / LD done       GEMV strips 12-17
  43537  LD done
  43539  ST issued                 first half (18 words)
  43540  LD issued                 GEMV strips 18-23, into the other ACC bank, overlapping the store
  43852  ST done
  58166  LD issued / LD done
  72493  LD issued / LD done
  86751  LD done
  86753  ST issued                 second half
  87109  ST done
  87110  unit idle: list done
counts {'rbeat': 20852, 'ld': 7, 'st': 2}
```

- **Data volume.** 20,852 read beats of 16 bytes = 333,632 bytes: 331,776
  bytes of weights, plus x and the list.
- **Overlap.** The GEMV of strips 18–23 is issued one cycle after the first
  store, so the second ACC bank lets the store and the next weight stream
  overlap, as §4.2 intended.
- **Not board numbers.** The testbench has **one** DMA port and a simple DDR
  model: each 55 KB GEMV takes about 14,400 cycles here, 3.8 B/cycle. The board
  has 4 ports, and `bwtest` measured 57.95 B/cycle. At that rate, dispatch B's
  331,776 bytes stream in about 5,700 cycles, 57 µs at 100 MHz.

## 10. What is checked at each layer

| Layer | Check | Result here |
|---|---|---|
| sa backend | `walk_dispatch.sh`: the dispatch recompiled alone gives the same binary as in `sa.vmfb` | identical (both) |
| Whole compile | recompiling the model with the phase dump gives the same `sa.vmfb` | byte-identical |
| Codegen | `golden.py`: the recorded corpus compiles to the same bytes | (unchanged by this work) |
| Descriptors | `rv_dispatch.py`: the functional simulator against the oracle that evaluates the dispatch's IR | equal (both) |
| Runtime | the board transport through the ring emulator, 3 decode steps | all entries status 0 |
| RTL | `tb_system` `RVB_TEST`: output bytes against the functional simulator, list and PCPI and plan A | PASS (4 runs) |
| Board | `run_board_tests.sh`: logits against the simulator, bit for bit | the release bundles (`results_v2.4.md`) |

## 11. Reproducing

Run from the repository root with `source compiler/env.sh`. Big jobs go under
`systemd-run --user -p MemoryMax=10G` (CLAUDE.md). The scripts used here are
kept in `build/walk_e2e/*.sh`.

```sh
export SA_D=16 SA_GEMV_PORTS=4
W=build/walk_e2e; T=$W/smol
# 1. export (about 1.5 min)
$SA_PY compiler/tests/test_hf_generic.py --model build/llm_cache/SmolLM2-135M --out $T \
  --quant --rope --attn --kv-channel k --layers 2 --prefill 16 --export-only
# 2. compile: binaries, rv32 LLVM IR, every IREE phase (about 30 s)
mkdir -p $T/sa_binaries $T/rv_ll $W/phases
SA_COMPILE_FLAGS="--iree-sa-codegen-report --iree-sa-emit-riscv=$T/rv_ll \
  --iree-hal-dump-executable-binaries-to=$T/sa_binaries --dump-compilation-phases-to=$W/phases" \
  compiler/scripts/compile_sa.sh $T
$SA_PY compiler/tests/rv_image.py $T/rv_ll --out $T/rv
# 3. the sa passes of each dispatch
compiler/scripts/walk_dispatch.sh $T 4 $W/sa_d4
compiler/scripts/walk_dispatch.sh $T 7 $W/sa_d7
# 4. the runtime: ring entries, lists, executed commands
$SA_PY compiler/sim/sa_board_emu.py --d 16 --mb 128 --dump $W/ring -- \
  build/iree/build-sa-host/runtime/plugins/hal/drivers/sa/sa-llm-run --device=sa --module=$T/sa.vmfb \
  --parameters=model=$T/sa_packed.irpa --tokens=1,504 --generate=1 --abi=hfv --pad=16
# 5. the PicoRV32 programs and the RTL (xsim, about 35 s per run)
for n in 4 7; do for c in trace compiled; do
  $SA_PY compiler/tests/rv_dispatch.py "$T/sa_sources/module_main\$async_dispatch_$n.mlir" --t 16 --code $c --out $W/rvb/d${n}_$c
  make -C firmware/rvb sim RVB=$PWD/$W/rvb/d${n}_$c PYTHON=$SA_PY
done; done
# 6. waveforms of the descriptor-list run
cd $W/rvb/d4_trace/sim_bram && source <Vivado>/settings64.sh && \
  xelab -debug typical tb_system -s tbdbg && xsim tbdbg -tclbatch ../../../../../firmware/rvb/vcd.tcl && cd -
$SA_PY compiler/tests/vcd_timeline.py $W/rvb/d4_trace/sim_bram/wave.vcd
```

Any other dispatch works the same way: pick its number in `sa_sources/`, and
find its call in the ring list by its BASE values (binding address + push
constants from the HAL phase).

// sahl-expand-kernels (docs/iree_compiler_plan.md §8.22, C6.5 steps b and c):
// decode's q . K^T over a cache longer than one SPAD_B bank holds (Gemma-3,
// 1 KV head, G = 4 query rows, head_dim 256, compiled for 512 positions).
// Step b: the G rows of q (int32) into ACC, one VE copy to int8, two
// transposes into one A tile, one EX per block for all rows. Step c: the T
// loop rotated (block i's LD, block i - 1's EX and copies, block i's
// transpose), the last block's EX and copies after the loop at the loop
// PARAM's final value (sa.advance_final), the row buffers one block longer
// in front (iteration 0's copies). Each iteration's block of 128 rows as two
// halves of 64 whose tiles alternate SPAD_B banks (docs/fusion_design.md
// §6.2.2): per half its LD, the previous half's EX (the other bank), its
// transpose, then the previous half's copies.
// RUN: iree-opt --iree-sahl-expand-kernels="d=16 max-dynamic=512" %s | FileCheck %s
// RUN: iree-opt --iree-sahl-expand-kernels="d=16 max-dynamic=512" --iree-sahl-to-sahw="d=16 max-dynamic=512" %s | FileCheck %s --check-prefix=LOW

// CHECK-LABEL: func.func @scores_tiled
// CHECK: %[[XA:.+]] = memref.alloc() {sa.layout = "packed", sa.mem = "acc"} : memref<4x256xi32>
// CHECK-NEXT: sahl.load %{{.+}}, %[[XA]]
// CHECK-NEXT: %[[XR:.+]] = memref.alloc() {sa.mem = "spad_a", sa.word = 4096 : i64} : memref<4x256xi8>
// CHECK-NEXT: sahl.strip %[[XA]], %[[XR]] {sa.packed}
// CHECK-NEXT: %[[COLS:.+]] = memref.alloc() {sa.mem = "spad_a", sa.word = 4352 : i64} : memref<256x16xi8>
// CHECK-NEXT: %[[ROWS:.+]] = memref.alloc() {sa.mem = "spad_a", sa.word = 4096 : i64} : memref<16x256xi8>
// CHECK-NEXT: sahl.transpose %[[ROWS]], %[[COLS]]
// CHECK-NEXT: %[[A:.+]] = memref.alloc() {sa.mem = "spad_a", sa.word = 0 : i64} : memref<16x256xi8>
// CHECK-NEXT: sahl.transpose %[[COLS]], %[[A]]
// the scratch: D slots of Tb = 128; the row buffers: guard block + 5 blocks
// CHECK: memref.alloc() {sa.accumulator, sa.layout = "packed", sa.mem = "acc"} : memref<2048xi64>
// CHECK-COUNT-4: memref.alloc() {sa.accumulator, sa.layout = "packed", sa.mem = "acc"} : memref<768xi64>
// CHECK: sahl.loop count(%{{.+}}) attributes {count = 5 : i64, strides = array<i64: 32768, 8>}
// CHECK: sahl.load %{{.+}}, %{{.+}} {sa.advance = 0 : i64, sa.rows} : memref<64x256xi8
// CHECK: memref.alloc() {sa.bank = 1 : i64, sa.mem = "spad_b"} : memref<4x256x16xi8>
// CHECK: sahl.mma
// CHECK-NEXT: memref.alloc() {sa.bank = 0 : i64, sa.mem = "spad_b"} : memref<4x256x16xi8>
// CHECK-NEXT: sahl.transpose
// CHECK: memref.subview %{{.+}}[64] [64] [1] : memref<768xi64>
// CHECK-NEXT: memref.copy %{{.+}}, %{{.+}} {sa.advance = 1 : i64}
// CHECK: sahl.load %{{.+}}, %{{.+}} {sa.advance = 0 : i64, sa.rows} : memref<64x256xi8, strided<[256, 1], offset: 16384>
// CHECK: memref.alloc() {sa.bank = 0 : i64, sa.mem = "spad_b"} : memref<4x256x16xi8>
// CHECK: sahl.mma
// CHECK-NEXT: memref.alloc() {sa.bank = 1 : i64, sa.mem = "spad_b"} : memref<4x256x16xi8>
// CHECK-NEXT: sahl.transpose
// CHECK: memref.subview %{{.+}}[128] [64] [1] : memref<768xi64>
// CHECK-NEXT: memref.copy %{{.+}}, %{{.+}} {sa.advance = 1 : i64}
// CHECK: {{^ +}$}}
// CHECK-NEXT: memref.alloc() {sa.bank = 1 : i64, sa.mem = "spad_b"} : memref<4x256x16xi8>
// CHECK: sahl.mma
// CHECK-NOT: sahl.mma
// CHECK-COUNT-4: memref.copy %{{.+}}, %{{.+}} {sa.advance_final = 1 : i64}
// CHECK-COUNT-4: sahl.scope

// LOW-LABEL: sahw.template "scores_tiled"
// LOW: sahw.transpose {{.*}}stride = 16
// LOW-NEXT: sahw.transpose {dst = 268435456 : i64, {{.*}}stride = 1
// LOW: sahw.setreg %[[P0:.+]], %[[P1:.+]] {values = array<i64: 0, 0>}
// LOW-NEXT: sahw.loop 5, %[[P0]] step 32768, %[[P1]] step 8 dyn(
// LOW-NEXT: sahw.ld %{{.+}} dyn(%[[P0]]) {ddr = 0 : i64
// LOW-NEXT: sahw.ex {a = 0 : i64, b = 4096 : i64
// LOW-NEXT: sahw.transpose {dst = 536870912 : i64
// LOW-COUNT-4: sahw.ve dyn(%[[P1]]) {{.*}}dyn_add = array<i1: true>, dyn_fields = array<i32: 3>
// LOW-NEXT: sahw.ld %{{.+}} dyn(%[[P0]]) {ddr = 16384 : i64
// LOW-NEXT: sahw.ex {a = 0 : i64, b = 0 : i64
// LOW-NEXT: sahw.transpose {dst = 536875008 : i64
// LOW-COUNT-4: sahw.ve dyn(%[[P1]]) {{.*}}dyn_add = array<i1: true>, dyn_fields = array<i32: 3>
// LOW-NEXT: {{^ +}$}}
// LOW-NEXT: sahw.ex {a = 0 : i64, b = 4096 : i64
// LOW-COUNT-4: sahw.ve dyn(%[[P1]]) {{.*}}dyn_add = array<i1: true>, dyn_fields = array<i32: 3>
func.func @scores_tiled() {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 6.250000e-02 : f32
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(<constants = 5, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(<constants = 5, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(<constants = 5, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(<constants = 5, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(3) : i32
  %4 = hal.interface.constant.load layout(<constants = 5, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(4) : i32
  %5 = arith.index_castui %0 : i32 to index
  %6 = arith.index_castui %1 : i32 to index
  %7 = arith.index_castui %2 : i32 to index
  %8 = arith.extui %3 : i32 to i64
  %9 = arith.extui %4 : i32 to i64
  %10 = arith.shli %9, %c32_i64 : i64
  %11 = arith.ori %8, %10 : i64
  %12 = arith.index_castui %11 : i64 to index
  %13:4 = util.assume.int 
      %5<umin = 0, umax = 1048576, udiv = 64>,
      %6<umin = 0, umax = 1048576, udiv = 64>,
      %7<umin = 0, umax = 1048576, udiv = 64>,
      %12<umin = 0, umax = 9007199254740991>
    : index, index, index, index
  %14 = hal.interface.binding.subspan layout(<constants = 5, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<1024x1x256xi8, #hal.descriptor_type<storage_buffer>>
  %15 = hal.interface.binding.subspan layout(<constants = 5, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%13#0) flags("ReadOnly|Indirect") : memref<1x4x256xi32, strided<[1024, 256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %16 = hal.interface.binding.subspan layout(<constants = 5, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%13#1) flags("ReadOnly|Indirect") : memref<1x4xf32, strided<[4, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %17 = iree_tensor_ext.dispatch.workload.ordinal %13#3, 0 : index
  %18 = hal.interface.binding.subspan layout(<constants = 5, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%13#2) flags(Indirect) : memref<1x4x?xf32, strided<[?, ?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>{%17}
  %alloc = memref.alloc(%17) {sa.layout = "rows", sa.mem = "acc"} : memref<1x4x?xf32>
  %alloc_0 = memref.alloc(%17) {sa.layout = "rows", sa.mem = "acc"} : memref<1x4x?xi64>
  %alloc_1 = memref.alloc(%17) {sa.layout = "rows", sa.mem = "acc"} : memref<1x256x?xi32>
  %subview = memref.subview %14[0, 0, 0] [%17, 1, 256] [1, 1, 1] : memref<1024x1x256xi8, #hal.descriptor_type<storage_buffer>> to memref<?x1x256xi8, strided<[256, 256, 1]>, #hal.descriptor_type<storage_buffer>>
  %alloc_2 = memref.alloc() {sa.layout = "packed", sa.mem = "acc"} : memref<1x4xf32>
  sahl.kernel "contraction" {
    linalg.fill ins(%c0_i64 : i64) outs(%alloc_0 : memref<1x4x?xi64>)
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview : memref<?x1x256xi8, strided<[256, 256, 1]>, #hal.descriptor_type<storage_buffer>>) outs(%alloc_1 : memref<1x256x?xi32>) {
    ^bb0(%in: i8, %out: i32):
      %19 = arith.extsi %in : i8 to i32
      linalg.yield %19 : i32
    }
    linalg.batch_matmul {sahl.anchor} ins(%15, %alloc_1 : memref<1x4x256xi32, strided<[1024, 256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<1x256x?xi32>) outs(%alloc_0 : memref<1x4x?xi64>)
    sahl.load %16, %alloc_2 : memref<1x4xf32, strided<[4, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<1x4xf32>
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_0, %alloc_2 : memref<1x4x?xi64>, memref<1x4xf32>) outs(%alloc : memref<1x4x?xf32>) {
    ^bb0(%in: i64, %in_3: f32, %out: f32):
      %19 = arith.trunci %in : i64 to i32
      %20 = arith.sitofp %19 : i32 to f32
      %21 = arith.mulf %20, %in_3 : f32
      %22 = arith.mulf %21, %cst : f32
      linalg.yield %22 : f32
    }
    sahl.store %alloc, %18 : memref<1x4x?xf32>, memref<1x4x?xf32, strided<[?, ?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  }
  return
}


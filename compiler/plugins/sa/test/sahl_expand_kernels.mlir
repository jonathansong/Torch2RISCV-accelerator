// sahl-expand-kernels (docs/iree_compiler_plan.md §8.17 item 2, §8.20): a
// linear micro-kernel's schedule written out in sahl. Input: stories15M's
// prefill QKV projection (M = 8 rows, 108 output tiles; dispatch prefill 25,
// before sa-to-sahl). sahl-schedule chooses 27 tiles per chunk: 4 chunks.
// RUN: iree-opt --iree-sa-to-sahl --iree-sahl-schedule --iree-sahl-expand-kernels %s | FileCheck %s
// RUN: iree-opt --iree-sa-to-sahl --iree-sahl-schedule --iree-sahl-expand-kernels --iree-sahl-to-sahw %s | FileCheck %s --check-prefix=LOW

// CHECK-LABEL: func.func @prefill$async_dispatch_25
// CHECK-NOT: sahl.kernel
// the A strip of the rows, the per-row scale in the broadcast layout
// CHECK: %[[STRIP:.+]] = memref.alloc() {sa.layout = "strip", sa.mem = "spad_a"} : memref<8x288xi8>
// CHECK: sahl.strip %{{.+}}, %[[STRIP]]
// CHECK: sahl.bcast %{{.+}}, %[[SX:.+]] : memref<8xf32>, memref<8xf32>
// two sets of per-column buffers, two SPAD_B banks; chunk 0 loaded
// CHECK: sahl.reserve %[[C0:.+]] : memref<27x8xf32>
// CHECK: sahl.reserve %[[C1:.+]] : memref<27x8xf32>
// CHECK: %[[B0:.+]] = memref.alloc() {sa.bank = 0 : i64, sa.mem = "spad_b"} : memref<27x288x8xi8>
// CHECK: %[[B1:.+]] = memref.alloc() {sa.bank = 1 : i64, sa.mem = "spad_b"} : memref<27x288x8xi8>
// CHECK: sahl.load %{{.+}}, %[[B0]]
// CHECK: sahl.load %{{.+}}, %[[C0]]
// chunk 0: EX on bank 0, then the prefetch of chunk 1 into bank 1, the epilogue, the store
// CHECK: sahl.scope attributes {keep} {
// CHECK: sahl.mma %[[STRIP]], %[[B0]], %[[ACC:.+]] :
// CHECK: sahl.load %{{.+}}, %[[B1]]
// CHECK: sahl.load %{{.+}}, %[[C1]]
// CHECK: linalg.generic {{.*}} ins(%[[ACC]], %[[C0]], %{{.+}} :
// CHECK: sahl.store
// chunk 1: bank 1, prefetch into bank 0
// CHECK: sahl.scope attributes {keep} {
// CHECK: sahl.mma %[[STRIP]], %[[B1]], %{{.+}} :
// CHECK: sahl.load %{{.+}}, %[[B0]]
// CHECK-COUNT-2: sahl.scope attributes {keep}
// CHECK-NOT: sahl.scope
// CHECK: return

// LOW: sahw.template
// LOW-COUNT-4: sahw.ex
// LOW-NOT: sahw.ex
func.func @prefill$async_dispatch_25_matmul_like_8x108x8x288_i8xi8xi32() {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(3) : i32
  %4 = arith.index_castui %0 : i32 to index
  %5 = arith.index_castui %1 : i32 to index
  %6 = arith.index_castui %2 : i32 to index
  %7 = arith.index_castui %3 : i32 to index
  %8:4 = util.assume.int 
      %4[<umin = 27712, umax = 27712, udiv = 27712>, <umin = 9280, umax = 9280, udiv = 9280>, <umin = 9280, umax = 9280, udiv = 9280>, <umin = 9280, umax = 9280, udiv = 9280>, <umin = 9280, umax = 9280, udiv = 9280>, <umin = 9280, umax = 9280, udiv = 9280>, <umin = 27712, umax = 27712, udiv = 27712>, <umin = 9280, umax = 9280, udiv = 9280>, <umin = 9280, umax = 9280, udiv = 9280>, <umin = 9280, umax = 9280, udiv = 9280>, <umin = 9280, umax = 9280, udiv = 9280>, <umin = 9280, umax = 9280, udiv = 9280>], 
      %5[<umin = 27648, umax = 27648, udiv = 27648>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 27648, umax = 27648, udiv = 27648>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>], 
      %6[<umin = 204032, umax = 204032, udiv = 204032>, <umin = 185216, umax = 185216, udiv = 185216>, <umin = 173312, umax = 173312, udiv = 173312>, <umin = 161408, umax = 161408, udiv = 161408>, <umin = 149504, umax = 149504, udiv = 149504>, <umin = 137600, umax = 137600, udiv = 137600>, <umin = 204032, umax = 204032, udiv = 204032>, <umin = 185216, umax = 185216, udiv = 185216>, <umin = 173312, umax = 173312, udiv = 173312>, <umin = 161408, umax = 161408, udiv = 161408>, <umin = 149504, umax = 149504, udiv = 149504>, <umin = 137600, umax = 137600, udiv = 137600>], 
      %7[<umin = 30016, umax = 30016, udiv = 30016>, <umin = 11584, umax = 11584, udiv = 11584>, <umin = 11584, umax = 11584, udiv = 11584>, <umin = 11584, umax = 11584, udiv = 11584>, <umin = 11584, umax = 11584, udiv = 11584>, <umin = 11584, umax = 11584, udiv = 11584>, <umin = 30016, umax = 30016, udiv = 30016>, <umin = 11584, umax = 11584, udiv = 11584>, <umin = 11584, umax = 11584, udiv = 11584>, <umin = 11584, umax = 11584, udiv = 11584>, <umin = 11584, umax = 11584, udiv = 11584>, <umin = 11584, umax = 11584, udiv = 11584>]
    : index, index, index, index
  %9 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%8#0) flags("ReadOnly|Indirect") : memref<8x288xi8, strided<[288, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %10 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<108x288x8xi8, #hal.descriptor_type<storage_buffer>>
  %11 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%8#2) flags(ReadOnly) : memref<108x8xf32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %12 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%8#1) flags("ReadOnly|Indirect") : memref<8xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %13 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%8#3) flags(Indirect) : memref<8x108x8xf32, strided<[864, 8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %alloc = memref.alloc() : memref<8x108x8xf32>
  %alloc_0 = memref.alloc() : memref<8x108x8xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_0 : memref<8x108x8xi32>)
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%9, %10 : memref<8x288xi8, strided<[288, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<108x288x8xi8, #hal.descriptor_type<storage_buffer>>) outs(%alloc_0 : memref<8x108x8xi32>) {
  ^bb0(%in: i8, %in_1: i8, %out: i32):
    %14 = arith.extsi %in : i8 to i32
    %15 = arith.extsi %in_1 : i8 to i32
    %16 = arith.muli %14, %15 : i32
    %17 = arith.addi %out, %16 : i32
    linalg.yield %17 : i32
  }
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_0, %11, %12 : memref<8x108x8xi32>, memref<108x8xf32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<8xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%alloc : memref<8x108x8xf32>) {
  ^bb0(%in: i32, %in_1: f32, %in_2: f32, %out: f32):
    %14 = arith.sitofp %in : i32 to f32
    %15 = arith.mulf %14, %in_1 : f32
    %16 = arith.mulf %15, %in_2 : f32
    linalg.yield %16 : f32
  }
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<8x108x8xf32>) outs(%13 : memref<8x108x8xf32, strided<[864, 8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  }
  return
}

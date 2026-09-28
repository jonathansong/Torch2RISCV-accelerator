// sa-to-sahl's kernel grouping (docs/iree_compiler_plan.md §8.14, C8 R1): the
// operations of a match move into a sahl.kernel at the contraction's position,
// the allocations they use before it. Input: stories15M's W13 (decode,
// dispatch main 28, before sa-to-sahl).
// RUN: iree-opt --iree-sa-to-sahl %s | FileCheck %s --check-prefix=UK
// RUN: iree-opt --iree-sa-to-sahl="ukernels=none" %s | FileCheck %s --check-prefix=GEN
// RUN: iree-opt --iree-sa-to-sahl --iree-sahl-to-sahw %s | FileCheck %s --check-prefix=LOW

// UK-LABEL: func.func @main$async_dispatch_28
// UK: memref.alloc() : memref<192x8xf32>
// UK: memref.alloc() : memref<f32>
// UK-NEXT: sahl.kernel "linear" {
// UK-NEXT: linalg.fill
// UK-NEXT: sahl.load
// UK-NEXT: sahl.load
// UK-NEXT: linalg.generic
// UK-SAME: sahl.anchor
// UK: sahl.load
// UK-NEXT: sahl.load
// UK-NEXT: linalg.generic
// UK: sahl.store
// UK-NEXT: }
// UK-NEXT: return
// GEN: sahl.kernel "contraction" {
// GEN-NOT: sahl.kernel
// The linear micro-kernel: chunk 0's weights, then the chunk loop with EX.
// LOW: sahw.template "main$async_dispatch_28
// LOW: sahw.ld
// LOW: sahw.ex
// LOW: sahw.st

func.func @main$async_dispatch_28_matvec_like_192x8x288_i32xi8xi64() {
  %c0_i64 = arith.constant 0 : i64
  %c2432 = arith.constant 2432 : index
  %c2368 = arith.constant 2368 : index
  %c0 = arith.constant 0 : index
  %c3584 = arith.constant 3584 : index
  %0 = hal.interface.constant.load layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
  %1 = arith.index_castui %0 : i32 to index
  %2 = util.assume.int %1[<umin = 189824, umax = 189824, udiv = 189824>, <umin = 177920, umax = 177920, udiv = 177920>, <umin = 166016, umax = 166016, udiv = 166016>, <umin = 154112, umax = 154112, udiv = 154112>, <umin = 142208, umax = 142208, udiv = 142208>, <umin = 130304, umax = 130304, udiv = 130304>] : index
  %3 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c2432) flags("ReadOnly|Indirect") : memref<288xi32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %4 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<192x288x8xi8, #hal.descriptor_type<storage_buffer>>
  %5 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%2) flags(ReadOnly) : memref<192x8xf32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %6 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c2368) flags("ReadOnly|Indirect") : memref<f32, strided<[], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %7 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c3584) flags(Indirect) : memref<192x8xf32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %alloc = memref.alloc() : memref<192x8xf32>
  %alloc_0 = memref.alloc() : memref<192x8xi64>
  linalg.fill ins(%c0_i64 : i64) outs(%alloc_0 : memref<192x8xi64>)
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : memref<288xi32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<192x288x8xi8, #hal.descriptor_type<storage_buffer>>) outs(%alloc_0 : memref<192x8xi64>) {
  ^bb0(%in: i32, %in_1: i8, %out: i64):
    %8 = arith.extsi %in : i32 to i64
    %9 = arith.extsi %in_1 : i8 to i64
    %10 = arith.muli %8, %9 : i64
    %11 = arith.addi %out, %10 : i64
    linalg.yield %11 : i64
  }
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_0, %5, %6 : memref<192x8xi64>, memref<192x8xf32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<f32, strided<[], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%alloc : memref<192x8xf32>) {
  ^bb0(%in: i64, %in_1: f32, %in_2: f32, %out: f32):
    %8 = arith.trunci %in : i64 to i32
    %9 = arith.sitofp %8 : i32 to f32
    %10 = arith.mulf %9, %in_1 : f32
    %11 = arith.mulf %10, %in_2 : f32
    linalg.yield %11 : f32
  }
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc : memref<192x8xf32>) outs(%7 : memref<192x8xf32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  }
  return
}


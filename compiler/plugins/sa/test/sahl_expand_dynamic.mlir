// sahl-expand-kernels (docs/iree_compiler_plan.md §8.20): the generic
// contraction with a dynamic K or N, written out in sahl. Inputs: prefill
// attention batch_matmuls of a 3-head model (dispatches before
// sahl-expand-kernels): the scores q . K^T (N = T: the K rows transposed,
// T / D tiles; each row's T results stored at the row cursor) and p . V
// (K = T: each row of p, its rows a dynamic stride apart, loaded at the row
// cursor; the V rows interleaved).
// RUN: iree-opt --split-input-file --iree-sahl-expand-kernels %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-sahl-expand-kernels --iree-sahl-to-sahw %s | FileCheck %s --check-prefix=LOW

// CHECK-LABEL: func.func @scores
// CHECK-NOT: sahl.kernel
// CHECK: sahl.cursor_reset
// CHECK: %[[NT:.+]] = arith.shrui %{{.+}}, %{{.+}} : index
// CHECK: memref.alloc(%[[NT]]) {sa.bank = 0 : i64, sa.mem = "spad_b"} : memref<?x64x8xi8>
// CHECK: sahl.load %{{.+}}, %{{.+}} {sa.rows}
// CHECK: sahl.transpose
// CHECK-COUNT-8: sahl.store %{{.+}}, %{{.+}} {sa.cursor, sa.cursor_step}
// CHECK: sahl.store %{{.+}}, %{{.+}} {sa.cursor} :
// CHECK-NOT: sahl.store
// LOW-LABEL: sahw.template "scores"
// LOW: %[[CUR:.+]] = sahw.private
// LOW: sahw.body
// LOW-NEXT: sahw.setreg %[[CUR]] {values = array<i64: 0>}
// LOW: sahw.transpose dyn(
// LOW: sahw.ex dyn(
// LOW: sahw.st %{{.+}} dyn(%[[CUR]], %[[B:.+]]) {{.*}}dyn_add = array<i1: true, false>
// LOW-NEXT: sahw.setreg %[[CUR]] dyn(%[[B]]) {add_mask = 1

func.func @scores() {
  %c32_i64 = arith.constant 32 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(3) : i32
  %4 = arith.index_castui %0 : i32 to index
  %5 = arith.index_castui %1 : i32 to index
  %6 = arith.extui %2 : i32 to i64
  %7 = arith.extui %3 : i32 to i64
  %8 = arith.shli %7, %c32_i64 : i64
  %9 = arith.ori %6, %8 : i64
  %10 = arith.index_castui %9 : i64 to index
  %11:3 = util.assume.int 
      %4<umin = 0, umax = 1048576, udiv = 64>, 
      %5<umin = 0, umax = 1048576, udiv = 64>, 
      %10<umin = 0, umax = 9007199254740991>
    : index, index, index
  %12 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<7680x3x64xi8, #hal.descriptor_type<storage_buffer>>
  %13 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%11#0) flags("ReadOnly|Indirect") : memref<3x3x64xi32, strided<[192, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %14 = iree_tensor_ext.dispatch.workload.ordinal %11#2, 0 : index
  %15 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%11#1) flags(Indirect) : memref<3x3x?xf32, strided<[?, ?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>{%14}
  %alloc = memref.alloc(%14) {sa.layout = "rows", sa.mem = "acc"} : memref<3x3x?xf32>
  %alloc_0 = memref.alloc(%14) {sa.layout = "rows", sa.mem = "acc"} : memref<3x3x?xi64>
  %alloc_1 = memref.alloc(%14) {sa.layout = "rows", sa.mem = "acc"} : memref<3x64x?xi32>
  %subview = memref.subview %12[3840, 0, 0] [%14, 3, 64] [1, 1, 1] : memref<7680x3x64xi8, #hal.descriptor_type<storage_buffer>> to memref<?x3x64xi8, strided<[192, 64, 1], offset: 737280>, #hal.descriptor_type<storage_buffer>>
  sahl.kernel "contraction" {
    linalg.fill ins(%c0_i64 : i64) outs(%alloc_0 : memref<3x3x?xi64>)
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview : memref<?x3x64xi8, strided<[192, 64, 1], offset: 737280>, #hal.descriptor_type<storage_buffer>>) outs(%alloc_1 : memref<3x64x?xi32>) {
    ^bb0(%in: i8, %out: i32):
      %16 = arith.extsi %in : i8 to i32
      linalg.yield %16 : i32
    }
    linalg.batch_matmul {sahl.anchor} ins(%13, %alloc_1 : memref<3x3x64xi32, strided<[192, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<3x64x?xi32>) outs(%alloc_0 : memref<3x3x?xi64>)
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_0 : memref<3x3x?xi64>) outs(%alloc : memref<3x3x?xf32>) {
    ^bb0(%in: i64, %out: f32):
      %16 = arith.trunci %in : i64 to i32
      %17 = arith.sitofp %16 : i32 to f32
      linalg.yield %17 : f32
    }
    sahl.store %alloc, %15 : memref<3x3x?xf32>, memref<3x3x?xf32, strided<[?, ?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @pv
// CHECK-NOT: sahl.kernel
// CHECK: sahl.cursor_reset
// CHECK: memref.alloc(%{{.+}}) {sa.bank = 0 : i64, sa.mem = "spad_b"} : memref<8x?x8xi8>
// CHECK: sahl.load %{{.+}}, %{{.+}} {sa.interleave}
// CHECK-COUNT-9: sahl.load %{{.+}}, %{{.+}} {sa.cursor, sa.cursor_step}
// LOW-LABEL: sahw.template "pv"
// LOW: %[[CUR:.+]] = sahw.private
// LOW: sahw.body
// LOW-NEXT: sahw.setreg %[[CUR]] {values = array<i64: 0>}
// LOW: sahw.ld %{{.+}} dyn(%[[CUR]], %[[B:.+]]) {{.*}}dyn_add = array<i1: true, false>
// LOW-NEXT: sahw.setreg %[[CUR]] dyn(%[[B]]) {add_mask = 1

func.func @pv() {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 5.68427728E-4 : f32
  %c0_i64 = arith.constant 0 : i64
  %c1474560 = arith.constant 1474560 : index
  %0 = hal.interface.constant.load layout(<constants = 6, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(<constants = 6, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(<constants = 6, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(<constants = 6, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(3) : i32
  %4 = hal.interface.constant.load layout(<constants = 6, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(4) : i32
  %5 = hal.interface.constant.load layout(<constants = 6, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(5) : i32
  %6 = arith.index_castui %0 : i32 to index
  %7 = arith.index_castui %1 : i32 to index
  %8 = arith.extui %2 : i32 to i64
  %9 = arith.extui %3 : i32 to i64
  %10 = arith.shli %9, %c32_i64 : i64
  %11 = arith.ori %8, %10 : i64
  %12 = arith.index_castui %11 : i64 to index
  %13 = arith.extui %4 : i32 to i64
  %14 = arith.extui %5 : i32 to i64
  %15 = arith.shli %14, %c32_i64 : i64
  %16 = arith.ori %13, %15 : i64
  %17 = arith.index_castui %16 : i64 to index
  %18:4 = util.assume.int 
      %6<umin = 0, umax = 1048576, udiv = 64>, 
      %7<umin = 0, umax = 1048576, udiv = 64>, 
      %12<umin = 0, umax = 9007199254740991>, 
      %17<umin = 0, umax = 9007199254740991>
    : index, index, index, index
  %19 = hal.interface.binding.subspan layout(<constants = 6, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c1474560) flags(ReadOnly) : memref<7680x3x64xi8, strided<[192, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %20 = hal.interface.binding.subspan layout(<constants = 6, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%18#1) flags(Indirect) : memref<3x3x64xf32, strided<[192, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %21 = iree_tensor_ext.dispatch.workload.ordinal %18#3, 1 : index
  %22 = hal.interface.binding.subspan layout(<constants = 6, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%18#0) flags("ReadOnly|Indirect") : memref<3x3x?xi32, strided<[?, ?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>{%21}
  %23 = iree_tensor_ext.dispatch.workload.ordinal %18#2, 0 : index
  %alloc = memref.alloc() {sa.layout = "packed", sa.mem = "acc"} : memref<3x3x64xf32>
  %alloc_0 = memref.alloc() {sa.layout = "packed", sa.mem = "acc"} : memref<3x3x64xi64>
  %alloc_1 = memref.alloc(%23) {sa.layout = "rows", sa.mem = "acc"} : memref<3x?x64xi32>
  %subview = memref.subview %19[5888, 0, 0] [%23, 3, 64] [1, 1, 1] : memref<7680x3x64xi8, strided<[192, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<?x3x64xi8, strided<[192, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  sahl.kernel "contraction" {
    linalg.fill ins(%c0_i64 : i64) outs(%alloc_0 : memref<3x3x64xi64>)
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview : memref<?x3x64xi8, strided<[192, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%alloc_1 : memref<3x?x64xi32>) {
    ^bb0(%in: i8, %out: i32):
      %24 = arith.extsi %in : i8 to i32
      linalg.yield %24 : i32
    }
    linalg.batch_matmul {sahl.anchor} ins(%22, %alloc_1 : memref<3x3x?xi32, strided<[?, ?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<3x?x64xi32>) outs(%alloc_0 : memref<3x3x64xi64>)
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_0 : memref<3x3x64xi64>) outs(%alloc : memref<3x3x64xf32>) {
    ^bb0(%in: i64, %out: f32):
      %24 = arith.trunci %in : i64 to i32
      %25 = arith.sitofp %24 : i32 to f32
      %26 = arith.mulf %25, %cst : f32
      linalg.yield %26 : f32
    }
    sahl.store %alloc, %20 : memref<3x3x64xf32>, memref<3x3x64xf32, strided<[192, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  }
  return
}

// sahl-tile row pieces over [R, ?] (C6.5b, docs/iree_compiler_plan.md §8.22):
// the attention probabilities over the cache (qsum: the max over T, then
// i8(exp(s - max) * 127)) of Qwen3's 16 heads at max_dynamic = 2048 (decode
// dispatch 24, after sa-to-sahl). Untiled the lowering runs out of ACC; on
// SaCodegen's retry (tile-shrink = 1) the 16 rows go in pieces of 4, each a
// static row offset into the [16, ?] bindings. Without the retry: untouched
// (the dynamic-row estimate is far above what the lowering takes).
// RUN: iree-opt --iree-sahl-tile="d=16 max-dynamic=2048 tile-shrink=1" %s | FileCheck %s
// RUN: iree-opt --iree-sahl-tile="d=16 max-dynamic=2048" %s | FileCheck %s --check-prefix=FIRST
// RUN: iree-opt --iree-sahl-tile="d=16 max-dynamic=2048 tile-shrink=1" --iree-sahl-to-sahw="d=16 max-dynamic=2048" %s | FileCheck %s --check-prefix=LOW
// CHECK-LABEL: func.func @main$async_dispatch_24
// CHECK: sahl.scope {
// CHECK: memref.alloc() : memref<4xf32>
// CHECK: memref.subview %{{.+}}[0, 0] [4, %{{.+}}] [1, 1]
// CHECK: memref.alloc(%{{.+}}) : memref<4x?xf32>
// CHECK: linalg.generic {{.*}}iterator_types = ["parallel", "reduction"]
// CHECK: memref.alloc(%{{.+}}) : memref<4x?xi8>
// CHECK: sahl.store %{{.+}}, %{{.+}} : memref<4x?xi8>
// CHECK: sahl.scope {
// CHECK: memref.subview %{{.+}}[4, 0] [4, %{{.+}}] [1, 1]
// CHECK: sahl.scope {
// CHECK: memref.subview %{{.+}}[8, 0] [4, %{{.+}}] [1, 1]
// CHECK: sahl.scope {
// CHECK: memref.subview %{{.+}}[12, 0] [4, %{{.+}}] [1, 1]
// CHECK-NOT: sahl.scope
// CHECK: return
// FIRST-LABEL: func.func @main$async_dispatch_24
// FIRST-NOT: sahl.scope
// Each row a DMA at the row offset PARAM (SETREG += T * 4 bytes); the second
// piece's loads start 4 rows in.
// LOW-LABEL: sahw.template "main$async_dispatch_24
// LOW: sahw.setreg %[[ACC:.+]] {values = array<i64: 0>}
// LOW-NEXT: sahw.ld %{{.+}} dyn(%[[ACC]], %[[TB:.+]]) {{.*}}dyn_fields = array<i32: 1, 4>
// LOW-COUNT-3: sahw.setreg %[[ACC]] dyn(%[[TB]]) {add_mask = 1
// LOW: sahw.setreg %[[ACC]] {values = array<i64: 0>}
// LOW-NEXT: sahw.st
// LOW: sahw.setreg %[[ACC]] {values = array<i64: 0>}
// LOW-NEXT: sahw.setreg %[[ACC]] dyn(%[[TB]]) {add_mask = 1
// LOW-NEXT: sahw.setreg %[[ACC]] dyn(%[[TB]]) {add_mask = 1
// LOW-NEXT: sahw.setreg %[[ACC]] dyn(%[[TB]]) {add_mask = 1
// LOW-NEXT: sahw.setreg %[[ACC]] dyn(%[[TB]]) {add_mask = 1
// LOW-NEXT: sahw.ld %{{.+}} dyn(%[[ACC]], %[[TB]])

func.func @main$async_dispatch_24_reduction_16xD_i64xf32xf32() {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0xFF800000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 1.270000e+02 : f32
  %cst_2 = arith.constant 0x7F800000 : f32
  %cst_3 = arith.constant 3.40282347E+38 : f32
  %cst_4 = arith.constant -3.40282347E+38 : f32
  %cst_5 = arith.constant -1.270000e+02 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(3) : i32
  %4 = hal.interface.constant.load layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(4) : i32
  %5 = hal.interface.constant.load layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(5) : i32
  %6 = hal.interface.constant.load layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(6) : i32
  %7 = hal.interface.constant.load layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(7) : i32
  %8 = arith.extui %0 : i32 to i64
  %9 = arith.extui %1 : i32 to i64
  %10 = arith.shli %9, %c32_i64 : i64
  %11 = arith.ori %8, %10 : i64
  %12 = arith.index_castui %11 {stream.alignment = 64 : index} : i64 to index
  %13 = arith.extui %2 : i32 to i64
  %14 = arith.extui %3 : i32 to i64
  %15 = arith.shli %14, %c32_i64 : i64
  %16 = arith.ori %13, %15 : i64
  %17 = arith.index_castui %16 : i64 to index
  %18 = arith.extui %4 : i32 to i64
  %19 = arith.extui %5 : i32 to i64
  %20 = arith.shli %19, %c32_i64 : i64
  %21 = arith.ori %18, %20 : i64
  %22 = arith.index_castui %21 : i64 to index
  %23 = arith.extui %6 : i32 to i64
  %24 = arith.extui %7 : i32 to i64
  %25 = arith.shli %24, %c32_i64 : i64
  %26 = arith.ori %23, %25 : i64
  %27 = arith.index_castui %26 : i64 to index
  %28:4 = util.assume.int 
      %12[<umin = 43072, umax = 43072, udiv = 43072>, <umin = 43072, umax = 720575940379322368>], 
      %17<umin = 43072, umax = 576460752303466496>, 
      %22<umin = 0, umax = 9007199254740991>, 
      %27<umin = 0, umax = 9007199254740991>
    : index, index, index, index
  %29 = hal.interface.binding.subspan layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<i64, #hal.descriptor_type<storage_buffer>>
  %30 = iree_tensor_ext.dispatch.workload.ordinal %28#2, 0 : index
  %31 = iree_tensor_ext.dispatch.workload.ordinal %28#3, 1 : index
  %32 = hal.interface.binding.subspan layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%28#0) flags("ReadOnly|Indirect") : memref<16x?xf32, strided<[?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>{%30}
  %33 = hal.interface.binding.subspan layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%28#1) flags(Indirect) : memref<16x?xi8, strided<[?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>{%31}
  %alloc = memref.alloc(%31) : memref<16x?xi8>
  %alloc_6 = memref.alloc() : memref<16xf32>
  linalg.fill ins(%cst : f32) outs(%alloc_6 : memref<16xf32>)
  %c1 = arith.constant 1 : index
  %dim = memref.dim %32, %c1 : memref<16x?xf32, strided<[?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %alloc_7 = memref.alloc(%dim) : memref<16x?xf32>
  sahl.load %32, %alloc_7 : memref<16x?xf32, strided<[?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<16x?xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%29, %alloc_7 : memref<i64, #hal.descriptor_type<storage_buffer>>, memref<16x?xf32>) outs(%alloc_6 : memref<16xf32>) {
  ^bb0(%in: i64, %in_8: f32, %out: f32):
    %34 = linalg.index 1 : index
    %35 = arith.index_cast %34 : index to i64
    %36 = arith.cmpi sle, %35, %in : i64
    %37 = arith.select %36, %in_8, %cst : f32
    %38 = arith.maximumf %37, %out : f32
    linalg.yield %38 : f32
  }
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%29, %alloc_7, %alloc_6 : memref<i64, #hal.descriptor_type<storage_buffer>>, memref<16x?xf32>, memref<16xf32>) outs(%alloc : memref<16x?xi8>) {
  ^bb0(%in: i64, %in_8: f32, %in_9: f32, %out: i8):
    %34 = linalg.index 1 : index
    %35 = arith.subf %in_8, %in_9 : f32
    %36 = math.exp %35 : f32
    %37 = arith.index_cast %34 : index to i64
    %38 = arith.cmpi sle, %37, %in : i64
    %39 = arith.select %38, %36, %cst_0 : f32
    %40 = arith.mulf %39, %cst_1 : f32
    %41 = sahl.to_i8 %40
    linalg.yield %41 : i8
  }
  sahl.store %alloc, %33 : memref<16x?xi8>, memref<16x?xi8, strided<[?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  return
}


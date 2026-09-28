// sahl-tile (docs/iree_compiler_plan.md §8.14, C8 R3): an element-wise
// dispatch too large for ACC in flat pieces. Input: the K = 16384 synthetic
// model's SiLU + abs-max (dispatch main 30, after sa-to-sahl): 16384 elements
// in 4 pieces of 4096; the abs-max's result allocated before them
// (sahl.reserve), filled once, the later pieces' reductions sahl.accumulate.
// RUN: iree-opt --iree-sahl-tile %s | FileCheck %s
// RUN: iree-opt --iree-sahl-tile --iree-sahl-to-sahw --iree-sahw-fuse-ve %s | FileCheck %s --check-prefix=LOW
// CHECK-LABEL: func.func @main$async_dispatch_30
// CHECK: sahl.reserve %[[MAX:.+]] {bcast} : memref<f32>
// CHECK-NEXT: linalg.fill ins(%{{.+}} : f32) outs(%[[MAX]] : memref<f32>)
// CHECK-NEXT: sahl.scope {
// CHECK-NEXT: memref.subview %{{.+}}[0] [4096] [1]
// CHECK: memref.alloc() : memref<4096xf32>
// CHECK: linalg.generic {{.*}}iterator_types = ["reduction"]} ins(%{{.+}} : memref<4096xf32>) outs(%[[MAX]] : memref<f32>) {
// CHECK: sahl.scope {
// CHECK-NEXT: memref.subview %{{.+}}[4096] [4096] [1]
// CHECK: linalg.generic {{.*}}iterator_types = ["reduction"]} ins(%{{.+}} : memref<4096xf32>) outs(%[[MAX]] : memref<f32>) attrs = {sahl.accumulate}
// CHECK: sahl.scope {
// CHECK: sahl.scope {
// CHECK-NOT: sahl.scope
// CHECK: return
// Each piece stores its 4096 results.
// LOW-LABEL: sahw.template "main$async_dispatch_30
// LOW-COUNT-4: sahw.st %{{.+}} {ddr = {{[0-9]+}} : i64, laddr = {{[0-9]+}} : i64, pitch = 16384 : i64, row_bytes = 16384 : i64

func.func @main$async_dispatch_30_reduction_16384_f32() {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %c3200 = arith.constant 3200 : index
  %c134272 = arith.constant 134272 : index
  %c1024 = arith.constant 1024 : index
  %c199808 = arith.constant 199808 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c3200) flags("ReadOnly|Indirect") : memref<32768xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c134272) flags("ReadOnly|Indirect") : memref<16384xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c1024) flags(Indirect) : memref<f32, strided<[], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c199808) flags(Indirect) : memref<16384xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %alloc = memref.alloc() : memref<f32>
  %alloc_1 = memref.alloc() : memref<16384xf32>
  %subview = memref.subview %0[16384] [16384] [1] : memref<32768xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<16384xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %subview_2 = memref.subview %0[0] [16384] [1] : memref<32768xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<16384xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<f32>)
  %alloc_3 = memref.alloc() : memref<16384xf32>
  sahl.load %subview_2, %alloc_3 : memref<16384xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<16384xf32>
  %alloc_4 = memref.alloc() : memref<16384xf32>
  sahl.load %1, %alloc_4 : memref<16384xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<16384xf32>
  %alloc_5 = memref.alloc() : memref<16384xf32>
  sahl.load %subview, %alloc_5 : memref<16384xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<16384xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%alloc_3, %alloc_4, %alloc_5 : memref<16384xf32>, memref<16384xf32>, memref<16384xf32>) outs(%alloc_1 : memref<16384xf32>) {
  ^bb0(%in: f32, %in_6: f32, %in_7: f32, %out: f32):
    %4 = arith.addf %in_6, %cst : f32
    %5 = arith.divf %cst, %4 : f32
    %6 = arith.mulf %in, %5 : f32
    %7 = arith.mulf %6, %in_7 : f32
    linalg.yield %7 : f32
  }
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%alloc_1 : memref<16384xf32>) outs(%alloc : memref<f32>) {
  ^bb0(%in: f32, %out: f32):
    %4 = math.absf %in : f32
    %5 = arith.maximumf %4, %out : f32
    linalg.yield %5 : f32
  }
  sahl.store %alloc, %2 : memref<f32>, memref<f32, strided<[], offset: ?>, #hal.descriptor_type<storage_buffer>>
  sahl.store %alloc_1, %3 : memref<16384xf32>, memref<16384xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  return
}


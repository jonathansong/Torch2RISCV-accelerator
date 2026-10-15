// sa-to-sahl's scatter of strided rows (docs/iree_compiler_plan.md §8.22): the
// packed V cache's token row (SmolLM2, 2 layers, max_len 256: 12 groups of 16
// channels, 272 rows apart in the layer's [12 * 272, 16] view; hf_generic's
// index_copy, dispatch main 14 before sa-to-sahl), its row indices computed in
// the dispatch as r * 272 + the position read from DDR, becomes one
// sahl.scatter with sa.row_stride = 272, lowered as LDPARAM (position * 16)
// and one ST of 12 rows of 16 bytes, 272 * 16 bytes apart.
// RUN: iree-opt --iree-sa-to-sahl %s | FileCheck %s
// RUN: iree-opt --iree-sa-to-sahl --iree-sahl-to-sahw="d=16" %s | FileCheck %s --check-prefix=LOW
// CHECK-LABEL: func.func @main$async_dispatch_14_scatter_3264x16xi8_dispatch_tensor_store
// CHECK-NOT: iree_linalg_ext.scatter
// CHECK-NOT: linalg.index
// CHECK: sahl.scatter %{{.+}}, %{{.+}}, %{{.+}} {sa.row_stride = 272 : i64} : memref<12x16xi8
// LOW: sahw.ldparam %{{.+}}, %{{.+}} {add = 0 : i64, addr = {{[0-9]+}} : i64, mul = 16 : i64}
// LOW-NEXT: sahw.st %{{.+}} dyn(%{{.+}}) {{.*}}dyn_fields = array<i32: 1>{{.*}}pitch = 4352 : i64, row_bytes = 16 : i64, rows = 12 : i64}
func.func @main$async_dispatch_14_scatter_3264x16xi8_dispatch_tensor_store() {
  %c272_i64 = arith.constant 272 : i64
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer>], flags = Indirect>) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer>], flags = Indirect>) ordinal(1) : i32
  %2 = arith.index_castui %0 : i32 to index
  %3 = arith.index_castui %1 : i32 to index
  %4:2 = util.assume.int 
      %2[<umin = 2880, umax = 2880, udiv = 2880>, <umin = 1344, umax = 1344, udiv = 1344>], 
      %3[<umin = 98304, umax = 98304, udiv = 98304>, <umin = 150528, umax = 150528, udiv = 150528>]
    : index, index
  %5 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<i64, #hal.descriptor_type<storage_buffer>>
  %6 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer>], flags = Indirect>) binding(1) alignment(64) offset(%4#0) flags("ReadOnly|Indirect") : memref<12x16xi8, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %7 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer>], flags = Indirect>) binding(2) alignment(64) offset(%4#1) flags("None") : memref<3264x16xi8, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %alloc = memref.alloc() : memref<12xi64>
  linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%5 : memref<i64, #hal.descriptor_type<storage_buffer>>) outs(%alloc : memref<12xi64>) {
  ^bb0(%in: i64, %out: i64):
    %8 = linalg.index 0 : index
    %9 = arith.index_cast %8 : index to i64
    %10 = arith.muli %9, %c272_i64 : i64
    %11 = arith.addi %10, %in : i64
    linalg.yield %11 : i64
  }
  iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%6, %alloc : memref<12x16xi8, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<12xi64>) outs(%7 : memref<3264x16xi8, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) {
  ^bb0(%arg0: i8, %arg1: i8):
    iree_linalg_ext.yield %arg0 : i8
  }
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : memref<3264x16xi8, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%7 : memref<3264x16xi8, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) {
  ^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
  }
  return
}


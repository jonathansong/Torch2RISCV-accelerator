// sa-to-sahl's scatter (docs/iree_compiler_plan.md §8.14, C8 R1c): a one-row
// iree_linalg_ext.scatter (Qwen3's KV cache write, dispatch main 289, before
// sa-to-sahl) becomes sahl.scatter; lowered as LDPARAM (index * row bytes) and
// a ST with that DDR offset.
// RUN: iree-opt --iree-sa-to-sahl %s | FileCheck %s
// RUN: iree-opt --iree-sa-to-sahl --iree-sahl-to-sahw %s | FileCheck %s --check-prefix=LOW
// CHECK-LABEL: func.func @main$async_dispatch_289
// CHECK-NOT: iree_linalg_ext.scatter
// CHECK: sahl.scatter %{{.+}}, %{{.+}}, %{{.+}} : memref<1x8x128xi8>, memref<1xi64
// LOW: sahw.ldparam %{{.+}}, %{{.+}} {add = 0 : i64, addr = {{[0-9]+}} : i64, mul = 1024 : i64}
// LOW-NEXT: sahw.st %{{.+}} dyn(%{{.+}}) {{.*}}dyn_fields = array<i32: 1>{{.*}}row_bytes = 1024 : i64

func.func @main$async_dispatch_289_scatter_7168x8x128xi8_dispatch_tensor_store() {
  %cst = arith.constant 5.94111347 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 0x7F800000 : f32
  %cst_2 = arith.constant 3.40282347E+38 : f32
  %cst_3 = arith.constant 0xFF800000 : f32
  %cst_4 = arith.constant -3.40282347E+38 : f32
  %cst_5 = arith.constant -1.270000e+02 : f32
  %cst_6 = arith.constant 1.270000e+02 : f32
  %c20608 = arith.constant 20608 : index
  %c8192 = arith.constant 8192 : index
  %c7340032 = arith.constant 7340032 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer>], flags = Indirect>) binding(0) alignment(64) offset(%c20608) flags("ReadOnly|Indirect") : memref<1x8x128xf32, strided<[1024, 128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer>], flags = Indirect>) binding(0) alignment(64) offset(%c8192) flags("ReadOnly|Indirect") : memref<1xi64, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer>], flags = Indirect>) binding(1) alignment(64) offset(%c7340032) flags("None") : memref<7168x8x128xi8, strided<[1024, 128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %alloc = memref.alloc() : memref<1x8x128xi8>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : memref<1x8x128xf32, strided<[1024, 128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%alloc : memref<1x8x128xi8>) {
  ^bb0(%in: f32, %out: i8):
    %3 = arith.mulf %in, %cst : f32
    %4 = arith.cmpf une, %3, %3 : f32
    %5 = arith.select %4, %cst_0, %3 : f32
    %6 = arith.cmpf oeq, %5, %cst_1 : f32
    %7 = arith.select %6, %cst_2, %5 : f32
    %8 = arith.cmpf oeq, %7, %cst_3 : f32
    %9 = arith.select %8, %cst_4, %7 : f32
    %10 = math.roundeven %9 : f32
    %11 = arith.cmpf ult, %10, %cst_5 : f32
    %12 = arith.select %11, %cst_5, %10 : f32
    %13 = arith.cmpf ugt, %12, %cst_6 : f32
    %14 = arith.select %13, %cst_6, %12 : f32
    %15 = arith.fptosi %14 : f32 to i8
    linalg.yield %15 : i8
  }
  iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%alloc, %1 : memref<1x8x128xi8>, memref<1xi64, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%2 : memref<7168x8x128xi8, strided<[1024, 128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) {
  ^bb0(%arg0: i8, %arg1: i8):
    iree_linalg_ext.yield %arg0 : i8
  }
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : memref<7168x8x128xi8, strided<[1024, 128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%2 : memref<7168x8x128xi8, strided<[1024, 128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) {
  ^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
  }
  return
}


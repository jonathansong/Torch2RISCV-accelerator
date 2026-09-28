// sa-to-sahl (docs/iree_compiler_plan.md §8.4 step 5): DDR operands of a
// bufferized dispatch get local copies (sahl.load before, sahl.store after).
// RUN: iree-opt --iree-sa-to-sahl %s | FileCheck %s

// A column slice of a DDR buffer in, exp(-x), a DDR buffer out (the prefill SiLU's first step).
// CHECK-LABEL: func.func @exp_cols
// CHECK: %[[SV:.+]] = memref.subview
// CHECK: %[[L:.+]] = memref.alloc() : memref<8x768xf32>
// CHECK: sahl.load %[[SV]], %[[L]] : memref<8x768xf32, strided<[1536, 1]>
// CHECK: linalg.generic {{.*}} ins(%[[L]] : memref<8x768xf32>)
// CHECK: math.exp
// CHECK: sahl.store %{{.+}}, %{{.+}} : memref<8x768xf32>, memref<8x768xf32, #hal.descriptor_type<storage_buffer>>
// CHECK-NOT: linalg.generic
// CHECK: return
#pl = #hal.pipeline.layout<constants = 0, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
func.func @exp_cols() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant -1.0 : f32
  %0 = hal.interface.binding.subspan layout(#pl) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x1536xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(#pl) binding(1) alignment(64) offset(%c0) : memref<8x768xf32, #hal.descriptor_type<storage_buffer>>
  %alloc = memref.alloc() : memref<8x768xf32>
  %sv = memref.subview %0[0, 0] [8, 768] [1, 1] : memref<8x1536xf32, #hal.descriptor_type<storage_buffer>> to memref<8x768xf32, strided<[1536, 1]>, #hal.descriptor_type<storage_buffer>>
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%sv : memref<8x768xf32, strided<[1536, 1]>, #hal.descriptor_type<storage_buffer>>) outs(%alloc : memref<8x768xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.mulf %in, %cst : f32
    %3 = math.exp %2 : f32
    linalg.yield %3 : f32
  }
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%alloc : memref<8x768xf32>) outs(%1 : memref<8x768xf32, #hal.descriptor_type<storage_buffer>>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  }
  return
}

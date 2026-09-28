// sa-to-sahl (docs/iree_compiler_plan.md §8.4 step 5): DDR operands of a
// bufferized dispatch get local copies (sahl.load before, sahl.store after).
// RUN: iree-opt --split-input-file --iree-sa-to-sahl %s | FileCheck %s

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

// -----

// The to_i8 chain inside a body becomes one sahl.to_i8.
// CHECK-LABEL: func.func @quant
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[IN:.+]]: f32, %{{.+}}: i8):
// CHECK-NEXT: %[[Q:.+]] = sahl.to_i8 %[[IN]]
// CHECK-NEXT: linalg.yield %[[Q]] : i8
#pl2 = #hal.pipeline.layout<constants = 0, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>
#id2 = affine_map<(d0) -> (d0)>
func.func @quant() {
  %c0 = arith.constant 0 : index
  %c127 = arith.constant 1.270000e+02 : f32
  %cm127 = arith.constant -1.270000e+02 : f32
  %fmin = arith.constant -3.40282347E+38 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %fmax = arith.constant 3.40282347E+38 : f32
  %pinf = arith.constant 0x7F800000 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %x = hal.interface.binding.subspan layout(#pl2) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<64xf32, #hal.descriptor_type<storage_buffer>>
  %q = hal.interface.binding.subspan layout(#pl2) binding(1) alignment(64) offset(%c0) : memref<64xi8, #hal.descriptor_type<storage_buffer>>
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel"]} ins(%x : memref<64xf32, #hal.descriptor_type<storage_buffer>>) outs(%q : memref<64xi8, #hal.descriptor_type<storage_buffer>>) {
  ^bb0(%y: f32, %out: i8):
    %n = arith.cmpf une, %y, %y : f32
    %y1 = arith.select %n, %zero, %y : f32
    %p = arith.cmpf oeq, %y1, %pinf : f32
    %y2 = arith.select %p, %fmax, %y1 : f32
    %ni = arith.cmpf oeq, %y2, %ninf : f32
    %y3 = arith.select %ni, %fmin, %y2 : f32
    %rn = math.roundeven %y3 : f32
    %lo = arith.cmpf ult, %rn, %cm127 : f32
    %y4 = arith.select %lo, %cm127, %rn : f32
    %hi = arith.cmpf ugt, %y4, %c127 : f32
    %y5 = arith.select %hi, %c127, %y4 : f32
    %i = arith.fptosi %y5 : f32 to i8
    linalg.yield %i : i8
  }
  return
}

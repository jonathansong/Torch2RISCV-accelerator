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

// -----

// Prefill's RoPE of M rows (three loops: row, head, element; a table per row,
// a gather indexed by the row) -> one generic per row of the decode form:
// the row's DDR views and the gather source sliced at the row, a local result
// per row stored into its row of the DDR result.
// CHECK-LABEL: func.func @rows_unrolled
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel"]
// CHECK: sahl.gather "none" %{{.+}}[%{{.+}}, %{{.+}}] : memref<2x4xf32, strided<[4, 1]>, #hal.descriptor_type<storage_buffer>>
// CHECK: sahl.store %{{.+}}, %{{.+}} : memref<2x4xf32>, memref<2x4xf32, strided<[4, 1]>
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel"]
// CHECK: sahl.gather "none" %{{.+}}[%{{.+}}, %{{.+}}] : memref<2x4xf32, strided<[4, 1], offset: 8>
// CHECK: sahl.store %{{.+}}, %{{.+}} : memref<2x4xf32>, memref<2x4xf32, strided<[4, 1], offset: 8>
// CHECK-NOT: linalg.generic
// CHECK: return
#pl = #hal.pipeline.layout<constants = 0, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>
#x = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#t = affine_map<(d0, d1, d2) -> (d0, d2)>
func.func @rows_unrolled() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %0 = hal.interface.binding.subspan layout(#pl) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<2x2x4xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(#pl) binding(0) alignment(64) offset(%c64) flags(ReadOnly) : memref<2x4xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(#pl) binding(0) alignment(64) offset(%c128) flags(ReadOnly) : memref<2x2x4xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(#pl) binding(1) alignment(64) offset(%c0) : memref<2x2x4xf32, #hal.descriptor_type<storage_buffer>>
  %alloc = memref.alloc() : memref<2x2x4xf32>
  linalg.generic {indexing_maps = [#x, #t, #x], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %1 : memref<2x2x4xf32, #hal.descriptor_type<storage_buffer>>, memref<2x4xf32, #hal.descriptor_type<storage_buffer>>) outs(%alloc : memref<2x2x4xf32>) {
  ^bb0(%in: f32, %t: f32, %out: f32):
    %r = linalg.index 0 : index
    %h = linalg.index 1 : index
    %e = linalg.index 2 : index
    %g = memref.load %2[%r, %h, %e] : memref<2x2x4xf32, #hal.descriptor_type<storage_buffer>>
    %p = arith.mulf %in, %t : f32
    %s = arith.addf %p, %g : f32
    linalg.yield %s : f32
  }
  linalg.generic {indexing_maps = [#x, #x], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<2x2x4xf32>) outs(%3 : memref<2x2x4xf32, #hal.descriptor_type<storage_buffer>>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  }
  return
}

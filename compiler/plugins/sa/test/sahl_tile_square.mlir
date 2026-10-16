// sahl-tile on a square [R, R] dispatch (here 256 x 256 f32, too large for
// ACC at once): row pieces, each 1-D [R] operand's role from its use: the
// per-row scale (map (d0)) and the result are sliced with the rows, the
// per-column bias (map (d1)) is loaded whole in every piece. Before
// (2026-10-16) R == C was refused (the roles ambiguous by shape) and the
// dispatch failed with "ACC full".
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-iree-comprehensive-bufferize,iree-sa-to-sahl,iree-sahl-tile{d=16}))" %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-iree-comprehensive-bufferize,iree-sa-to-sahl,iree-sahl-tile{d=16},iree-sahl-plan-memory,iree-sahl-schedule,iree-sahl-expand-kernels),iree-sahl-to-sahw{d=16})" %s | FileCheck %s --check-prefix=LOW
// CHECK-LABEL: func.func @square_rows
// CHECK: sahl.scope {
// CHECK: memref.subview %[[X:.+]][0, 0] [48, 256] [1, 1]
// CHECK: memref.subview %[[S:.+]][0] [48] [1]
// CHECK: sahl.load %[[B:[0-9]+]], %{{.+}} : memref<256xf32, #hal.descriptor_type<storage_buffer>>, memref<256xf32>
// CHECK: memref.subview %[[O:.+]][0] [48] [1]
// CHECK: sahl.scope {
// CHECK: memref.subview %[[X]][48, 0] [48, 256] [1, 1]
// CHECK: memref.subview %[[S]][48] [48] [1]
// CHECK: sahl.load %[[B]], %{{.+}} : memref<256xf32, #hal.descriptor_type<storage_buffer>>, memref<256xf32>
// CHECK: memref.subview %[[X]][240, 0] [16, 256] [1, 1]
// CHECK: memref.subview %[[S]][240] [16] [1]
// CHECK: memref.subview %[[O]][240] [16] [1]
// LOW-LABEL: sahw.template "square_rows"

func.func @square_rows() {
  %ninf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256xf32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256xf32>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256xf32>>
  %x = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf32>> -> tensor<256x256xf32>
  %s = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [256], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
  %b = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [256], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
  %e = tensor.empty() : tensor<256xf32>
  %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<256xf32>) -> tensor<256xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%x, %s, %b : tensor<256x256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%f : tensor<256xf32>) {
  ^bb0(%a: f32, %sr: f32, %bc: f32, %o: f32):
    %m = arith.mulf %a, %sr : f32
    %p = arith.addf %m, %bc : f32
    %q = arith.maximumf %p, %o : f32
    linalg.yield %q : f32
  } -> tensor<256xf32>
  iree_tensor_ext.dispatch.tensor.store %r, %3, offsets = [0], sizes = [256], strides = [1] : tensor<256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256xf32>>
  return
}

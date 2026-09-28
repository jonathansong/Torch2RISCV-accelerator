// sa-to-sahl's gathers (docs/iree_compiler_plan.md §8.14, C8 R1b): each
// memref.load in a body becomes a sahl.gather with its form (gatherForm).
// RUN: iree-opt --split-input-file --iree-sa-to-sahl %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-sa-to-sahl --iree-sahl-to-sahw %s | FileCheck %s --check-prefix=LOW

// An embedding row: table[token, j] over j.
// CHECK-LABEL: func.func @embed_row
// CHECK: sahl.gather "row" %{{.+}}[%{{.+}}, %{{.+}}] : memref<100x64xf32
// LOW-LABEL: sahw.template "embed_row"
// LOW: sahw.ldparam
// LOW-NEXT: sahw.ld {{.*}}dyn_fields
#pl = #hal.pipeline.layout<constants = 0, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>
#v = affine_map<(d0) -> (d0)>
#s = affine_map<(d0) -> ()>
func.func @embed_row() {
  %c0 = arith.constant 0 : index
  %tab = hal.interface.binding.subspan layout(#pl) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<100x64xf32, #hal.descriptor_type<storage_buffer>>
  %tok = hal.interface.binding.subspan layout(#pl) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<i64, #hal.descriptor_type<storage_buffer>>
  %out = hal.interface.binding.subspan layout(#pl) binding(2) alignment(64) offset(%c0) : memref<64xf32, #hal.descriptor_type<storage_buffer>>
  linalg.generic {indexing_maps = [#s, #v], iterator_types = ["parallel"]} ins(%tok : memref<i64, #hal.descriptor_type<storage_buffer>>) outs(%out : memref<64xf32, #hal.descriptor_type<storage_buffer>>) {
  ^bb0(%t: i64, %o: f32):
    %r = arith.index_cast %t : i64 to index
    %j = linalg.index 0 : index
    %e = memref.load %tab[%r, %j] : memref<100x64xf32, #hal.descriptor_type<storage_buffer>>
    linalg.yield %e : f32
  }
  return
}

// -----

// One element for every point (the index depends on the scalar only): "scalar".
// CHECK-LABEL: func.func @embed_scalar
// CHECK: sahl.gather "scalar" %{{.+}}[%{{.+}}] : memref<100xf32
// LOW-LABEL: sahw.template "embed_scalar"
// LOW: sahw.ldparam
// LOW-NEXT: sahw.ldparam {{.*}}dyn_fields
#pl2 = #hal.pipeline.layout<constants = 0, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>
#v2 = affine_map<(d0) -> (d0)>
#s2 = affine_map<(d0) -> ()>
func.func @embed_scalar() {
  %c0 = arith.constant 0 : index
  %tab = hal.interface.binding.subspan layout(#pl2) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<100xf32, #hal.descriptor_type<storage_buffer>>
  %tok = hal.interface.binding.subspan layout(#pl2) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<i64, #hal.descriptor_type<storage_buffer>>
  %out = hal.interface.binding.subspan layout(#pl2) binding(2) alignment(64) offset(%c0) : memref<64xf32, #hal.descriptor_type<storage_buffer>>
  linalg.generic {indexing_maps = [#s2, #v2], iterator_types = ["parallel"]} ins(%tok : memref<i64, #hal.descriptor_type<storage_buffer>>) outs(%out : memref<64xf32, #hal.descriptor_type<storage_buffer>>) {
  ^bb0(%t: i64, %o: f32):
    %r = arith.index_cast %t : i64 to index
    %e = memref.load %tab[%r] : memref<100xf32, #hal.descriptor_type<storage_buffer>>
    linalg.yield %e : f32
  }
  return
}

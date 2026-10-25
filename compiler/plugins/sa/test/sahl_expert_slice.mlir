// sahl-to-sahw, MoE expert slices (docs/moe_native_lowering_design.md §6): a
// DDR view at a row given by an id in DDR (memref.load + index_cast) reads the
// table's BASE plus P = id * stride, P computed at the start of the body after
// a FENCE: one LDPARAM for a stride within its 16-bit factor, else id * stride
// on the VE through an output-only binding's first word; every DMA on the view
// adds P to its DDR address (a dynamic slot, add mode).
// RUN: iree-opt --split-input-file --iree-sahl-to-sahw="d=16" --verify-diagnostics %s | FileCheck %s

// A stride of 256 bytes (a scale table's row): LDPARAM of ids[1] times 256.
// CHECK-LABEL: sahw.template "expert_row_small"
// CHECK-DAG:   %[[IDS:.+]] = sahw.base binding(0)
// CHECK-DAG:   %[[T:.+]] = sahw.base binding(1)
// CHECK-DAG:   %[[P:.+]] = sahw.private
// CHECK:       sahw.body {
// CHECK-NEXT:    sahw.fence
// CHECK-NEXT:    sahw.ldparam %[[P]], %[[IDS]] {add = 0 : i64, addr = 4 : i64, mul = 256 : i64}
// CHECK-NEXT:    sahw.ld %[[T]] dyn(%[[P]]) {ddr = 0 : i64, dyn_add = array<i1: true>, dyn_fields = array<i32: 1>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>
func.func @expert_row_small() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<2xi32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<4x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : memref<64xf32, #hal.descriptor_type<storage_buffer>>
  %id = memref.load %0[%c1] : memref<2xi32, #hal.descriptor_type<storage_buffer>>
  %i = arith.index_cast %id : i32 to index
  %row = memref.subview %1[%i, 0] [1, 64] [1, 1] : memref<4x64xf32, #hal.descriptor_type<storage_buffer>> to memref<64xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %buf = memref.alloc() : memref<64xf32>
  sahl.load %row, %buf : memref<64xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<64xf32>
  sahl.store %buf, %2 : memref<64xf32>, memref<64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----

// A stride of 131072 bytes (an expert's packed weights): ids[0] (8 bytes) into
// ACC word 0, times 131072 on the VE into word 1, stored into the output's
// first word, FENCE, LDPARAM of it.
// CHECK-LABEL: sahw.template "expert_row_large"
// CHECK-DAG:   %[[IDS:.+]] = sahw.base binding(0)
// CHECK-DAG:   %[[T:.+]] = sahw.base binding(1)
// CHECK-DAG:   %[[OUT:.+]] = sahw.base binding(2)
// CHECK-DAG:   %[[P:.+]] = sahw.private
// CHECK:       sahw.body {
// CHECK-NEXT:    sahw.fence
// CHECK-NEXT:    sahw.ld %[[IDS]] {ddr = 0 : i64, laddr = 805306368 : i64, pitch = 8 : i64, row_bytes = 8 : i64, rows = 1 : i64}
// CHECK-NEXT:    sahw.ve {a = 1.310720e+05 : f32, b = 0.000000e+00 : f32, dst = 805306369 : i64, {{.*}}src1 = 805306368 : i64
// CHECK-NEXT:    sahw.st %[[OUT]] {ddr = 0 : i64, laddr = 805306369 : i64, pitch = 8 : i64, row_bytes = 8 : i64, rows = 1 : i64}
// CHECK-NEXT:    sahw.fence
// CHECK-NEXT:    sahw.ldparam %[[P]], %[[OUT]] {add = 0 : i64, addr = 0 : i64, mul = 1 : i64}
// CHECK-NEXT:    sahw.ld %[[T]] dyn(%[[P]]) {ddr = 0 : i64, dyn_add = array<i1: true>, dyn_fields = array<i32: 1>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>
func.func @expert_row_large() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<2xi32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<4x131072xi8, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : memref<64xi8, #hal.descriptor_type<storage_buffer>>
  %id = memref.load %0[%c0] : memref<2xi32, #hal.descriptor_type<storage_buffer>>
  %i = arith.index_cast %id : i32 to index
  %row = memref.subview %1[%i, 0] [1, 64] [1, 1] : memref<4x131072xi8, #hal.descriptor_type<storage_buffer>> to memref<64xi8, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %buf = memref.alloc() : memref<64xi8>
  sahl.load %row, %buf : memref<64xi8, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<64xi8>
  sahl.store %buf, %2 : memref<64xi8>, memref<64xi8, #hal.descriptor_type<storage_buffer>>
  return
}

// -----

// A stride whose multiples fp32 does not hold exactly (2^24 + 1): refused, the
// dispatch then runs on the host.
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>
// expected-error @+1 {{expert slice: id * stride not exact in fp32}}
func.func @expert_row_inexact() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<2xi32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<2x16777217xi8, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : memref<64xi8, #hal.descriptor_type<storage_buffer>>
  %id = memref.load %0[%c0] : memref<2xi32, #hal.descriptor_type<storage_buffer>>
  %i = arith.index_cast %id : i32 to index
  %row = memref.subview %1[%i, 0] [1, 64] [1, 1] : memref<2x16777217xi8, #hal.descriptor_type<storage_buffer>> to memref<64xi8, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %buf = memref.alloc() : memref<64xi8>
  sahl.load %row, %buf : memref<64xi8, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<64xi8>
  sahl.store %buf, %2 : memref<64xi8>, memref<64xi8, #hal.descriptor_type<storage_buffer>>
  return
}

// A native Top-2 route-record export uses three contiguous DDR bindings and
// emits a RISC-V-only SA template with the expert count and parameter stride.
// RUN: iree-opt --iree-sahl-to-sahw %s | FileCheck %s

// CHECK: sahw.template "route_top2"
// CHECK-SAME: sa.moe.route_top2 = array<i64: 16, 6784>
// CHECK-SAME: sa.riscv_only
// CHECK-NOT: sahw.ve

#pl = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>
func.func @route_top2() {
  %c0 = arith.constant 0 : index
  %values = hal.interface.binding.subspan layout(#pl) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<1x2xf32, #hal.descriptor_type<storage_buffer>>
  %indices = hal.interface.binding.subspan layout(#pl) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<1x2xi32, #hal.descriptor_type<storage_buffer>>
  %route = hal.interface.binding.subspan layout(#pl) binding(2) alignment(64) offset(%c0) : memref<8xi32, #hal.descriptor_type<storage_buffer>>
  sahl.route_top2 %values, %indices, %route {experts = 16 : i64, stride = 6784 : i64} : memref<1x2xf32, #hal.descriptor_type<storage_buffer>>, memref<1x2xi32, #hal.descriptor_type<storage_buffer>>, memref<8xi32, #hal.descriptor_type<storage_buffer>>
  return
}

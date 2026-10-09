// sahl.gemv -> one LD in mode GEMV (K2b streaming GEMV unit, docs/k2b_gemv_design.md §4),
// and its error on a target without the unit.
// RUN: iree-opt --iree-sahl-to-sahw="d=16 gemv-ports=2" %s | FileCheck %s --check-prefix=D16
// RUN: iree-opt --iree-sahl-to-sahw="d=16" --verify-diagnostics %s
// K2b (docs/k2b_gemv_design.md): x . 4 weight strips on the streaming GEMV unit:
// one LD in mode GEMV (strips, K x D bytes each, C step 1; x at SPAD_A word 0: xword's default).
// D16-LABEL: sahw.template "gemv_strips"
// D16: sahw.ld %{{.+}} {ddr = 0 : i64, laddr = [[X:[0-9]+]] : i64, pitch = 288 : i64, row_bytes = 288 : i64, rows = 1 : i64}
// D16-NEXT: sahw.ld %{{.+}} {cstep = 1 : i64, ddr = 0 : i64, laddr = [[C:[0-9]+]] : i64, mode = 2 : i64, pitch = 4608 : i64, row_bytes = 4608 : i64, rows = 4 : i64}
// D16-NEXT: sahw.st %{{.+}} {ddr = 0 : i64, laddr = [[C]] : i64, pitch = 256 : i64, row_bytes = 256 : i64, rows = 1 : i64}
#gemv_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>
// expected-error @below {{gemv: the target has no GEMV unit}}
func.func @gemv_strips() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#gemv_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<288xi8, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(#gemv_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<4x288x16xi8, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(#gemv_layout) binding(2) alignment(64) offset(%c0) : memref<4x16xi32, #hal.descriptor_type<storage_buffer>>
  %x = memref.alloc() {sa.layout = "packed", sa.mem = "spad_a"} : memref<288xi8>
  sahl.load %0, %x : memref<288xi8, #hal.descriptor_type<storage_buffer>>, memref<288xi8>
  %acc = memref.alloc() {sa.layout = "packed", sa.mem = "acc"} : memref<4x16xi32>
  sahl.gemv %x, %1, %acc : memref<288xi8>, memref<4x288x16xi8, #hal.descriptor_type<storage_buffer>>, memref<4x16xi32>
  sahl.store %acc, %2 : memref<4x16xi32>, memref<4x16xi32, #hal.descriptor_type<storage_buffer>>
  return
}

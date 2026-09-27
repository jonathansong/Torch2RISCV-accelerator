// sahw-split-head and sahw-assign-registers (docs/iree_compiler_plan.md §8.4).
// RUN: iree-opt --iree-sahw-split-head --iree-sahw-assign-registers %s | FileCheck %s

// Registers: BASE from 0, setup PARAMs from 0, private PARAMs from 7 down, in
// definition order. The leading loads of an export that does not touch SPAD_B
// become its head.
// CHECK-LABEL: sahw.template "elementwise"
// CHECK: sahw.base binding(0) {reg = 0 : i64}
// CHECK: sahw.param {constant = 1 : i64, reg = 0 : i64}
// CHECK: sahw.private {reg = 7 : i64}
// CHECK: sahw.private {reg = 6 : i64}
// CHECK: sahw.base binding(1) {reg = 1 : i64}
// CHECK: sahw.head {
// CHECK-NEXT: sahw.ld %{{.+}} {ddr = 0
// CHECK-NEXT: sahw.ld %{{.+}} {ddr = 1152
// CHECK: sahw.body {
// CHECK-NEXT: sahw.ve
sahw.template "elementwise" attributes {bindings = 2 : i64, constants = 2 : i64} {
  %x = sahw.base binding(0)
  %n = sahw.param {constant = 1 : i64}
  %p7 = sahw.private
  %p6 = sahw.private
  %y = sahw.base binding(1)
  sahw.body {
    sahw.ld %x {ddr = 0 : i64, laddr = 805306368 : i64, rows = 1 : i64, row_bytes = 1152 : i64, pitch = 1152 : i64}
    sahw.ld %x {ddr = 1152 : i64, laddr = 805306404 : i64, rows = 1 : i64, row_bytes = 1152 : i64, pitch = 1152 : i64}
    sahw.ve {src1 = 805306368 : i64, src2 = 805306404 : i64, dst = 805306440 : i64, length = 288 : i64, op = 0 : i64, types = 15 : i64, fp = true, imm = 0.0 : f32, a = 1.0 : f32, b = -0.0 : f32}
    sahw.st %y {ddr = 0 : i64, laddr = 805306440 : i64, rows = 1 : i64, row_bytes = 1152 : i64, pitch = 1152 : i64}
  }
}

// An export that touches SPAD_B (an EX) keeps its loads in the body.
// CHECK-LABEL: sahw.template "gemm"
// CHECK-NOT: sahw.head
// CHECK: sahw.body
sahw.template "gemm" attributes {bindings = 1 : i64, constants = 0 : i64} {
  %x = sahw.base binding(0)
  sahw.body {
    sahw.ld %x {ddr = 0 : i64, laddr = 268435456 : i64, rows = 1 : i64, row_bytes = 288 : i64, pitch = 288 : i64}
    sahw.ex {a = 0 : i64, b = 0 : i64, c = 0 : i64, kt = 36 : i64, repeat = 1 : i64, bstep = 0 : i64, cstep = 0 : i64, crow = 1 : i64}
  }
}

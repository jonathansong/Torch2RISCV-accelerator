// The sahw dialect parses, verifies and prints back (the printed form parses again).
// RUN: iree-opt %s | iree-opt | FileCheck %s

// CHECK-LABEL: sahw.template "linear"
// CHECK: %[[W:.+]] = sahw.base binding(1) {constant = 0 : i64, reg = 1 : i64}
// CHECK: %[[P:.+]] = sahw.private {reg = 7 : i64}
// CHECK: sahw.prefix {
// CHECK-NEXT: sahw.ld %[[W]] {ddr = 0 : i64
// CHECK: sahw.body {
// CHECK: sahw.setreg %[[P]] {values = array<i64: 0>}
// CHECK: sahw.loop 3, %[[P]] step 4608, %[[P]] step 64 {
// CHECK: sahw.ld %[[W]] dyn(%[[P]]) {ddr = 4608 : i64, dyn_add = array<i1: true>, dyn_fields = array<i32: 1>
// CHECK: sahw.ve {a = 2.000000e+00 : f32, b = -0.000000e+00 : f32
// CHECK: sahw.fence {mask = 3 : i64}
sahw.template "linear" attributes {bindings = 2 : i64, constants = 1 : i64} {
  %x = sahw.base binding(0) {reg = 0 : i64}
  %w = sahw.base binding(1) {constant = 0 : i64, reg = 1 : i64}
  %p = sahw.private {reg = 7 : i64}
  sahw.prefix {
    sahw.ld %w {ddr = 0 : i64, laddr = 536870912 : i64, rows = 18 : i64, row_bytes = 2304 : i64, pitch = 2304 : i64}
  }
  sahw.body {
    sahw.ld %x {ddr = 0 : i64, laddr = 805307392 : i64, rows = 1 : i64, row_bytes = 1152 : i64, pitch = 1152 : i64}
    sahw.setreg %p {values = array<i64: 0>}
    sahw.loop 3, %p step 4608, %p step 64 {
      sahw.ex {a = 0 : i64, b = 0 : i64, c = 0 : i64, kt = 36 : i64, repeat = 18 : i64, bstep = 288 : i64, cstep = 1 : i64, crow = 18 : i64}
      sahw.ld %w dyn(%p) {ddr = 4608 : i64, laddr = 536879104 : i64, rows = 18 : i64, row_bytes = 2304 : i64, pitch = 2304 : i64, dyn_fields = array<i32: 1>, dyn_add = array<i1: true>}
    }
    sahw.ve {src1 = 805306368 : i64, src2 = 0 : i64, dst = 805306530 : i64, length = 144 : i64, op = 5 : i64, types = 15 : i64, fp = true, imm = 0.0 : f32, a = 2.0 : f32, b = -0.0 : f32}
    sahw.fence {mask = 3 : i64}
  }
}

// RUN: iree-opt --split-input-file --iree-util-apply-patterns --allow-unregistered-dialect %s | FileCheck %s

// compiler/patches/iree/0001-util-merge-index-switch-same-selector.patch: two
// adjacent switches merge only when they switch on the same value. With the
// host fallback (plan §8.15), adjacent dispatches of executables with
// different variant conditions select their variants (and workgroup counts)
// with different values; merged, a dispatch run on the host took the sa
// variant's workgroup count (1 instead of 3: a third of its output).

// CHECK-LABEL: @differentSelectors
// CHECK-SAME: (%[[A:.+]]: index, %[[B:.+]]: index)
util.func @differentSelectors(%a: index, %b: index) -> (i32, i32) {
  // CHECK: scf.index_switch %[[A]] -> i32
  %r0 = scf.index_switch %a -> i32
  case 0 {
    %v = "some.sa.count"() : () -> i32
    scf.yield %v : i32
  }
  case 1 {
    %v = "some.vmvx.count"() : () -> i32
    scf.yield %v : i32
  }
  default {
    %v = "some.none"() : () -> i32
    scf.yield %v : i32
  }
  // CHECK: scf.index_switch %[[B]] -> i32
  %r1 = scf.index_switch %b -> i32
  case 0 {
    %v = "some.sa.count"() : () -> i32
    scf.yield %v : i32
  }
  case 1 {
    %v = "some.vmvx.count"() : () -> i32
    scf.yield %v : i32
  }
  default {
    %v = "some.none"() : () -> i32
    scf.yield %v : i32
  }
  util.return %r0, %r1 : i32, i32
}

// -----

// CHECK-LABEL: @sameSelector
// CHECK-SAME: (%[[A:.+]]: index)
util.func @sameSelector(%a: index) -> (i32, i32) {
  // CHECK: scf.index_switch %[[A]] -> i32, i32
  // CHECK-NOT: scf.index_switch
  %r0 = scf.index_switch %a -> i32
  case 0 {
    %v = "some.op0"() : () -> i32
    scf.yield %v : i32
  }
  case 1 {
    %v = "some.op0.case1"() : () -> i32
    scf.yield %v : i32
  }
  default {
    %v = "some.none0"() : () -> i32
    scf.yield %v : i32
  }
  %r1 = scf.index_switch %a -> i32
  case 0 {
    %v = "some.op1"() : () -> i32
    scf.yield %v : i32
  }
  case 1 {
    %v = "some.op1.case1"() : () -> i32
    scf.yield %v : i32
  }
  default {
    %v = "some.none1"() : () -> i32
    scf.yield %v : i32
  }
  util.return %r0, %r1 : i32, i32
}

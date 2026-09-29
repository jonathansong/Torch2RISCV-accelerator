// The host fallback (docs/iree_compiler_plan.md §8.15): each dispatch gets a
// VMVX variant after its sa one; a dispatch the sa backend cannot compile
// (tanh) has a false condition on its sa variant, so the runtime (and the
// dispatch site) take the VMVX one. The one it compiles (exp) keeps a plain
// sa variant, taken first.
// RUN: iree-compile %s --iree-hal-target-device=sa --iree-sa-host-fallback --iree-hal-link-executables=false --compile-to=executable-targets | FileCheck %s
// CHECK: hal.executable private @main_dispatch_0 {
// CHECK-NEXT: hal.executable.variant public @sa_desc_v1
// CHECK-NEXT: hal.executable.condition
// CHECK-NEXT: %[[F:.+]] = arith.constant false
// CHECK-NEXT: hal.return %[[F]] : i1
// CHECK: hal.executable.variant public @vmvx_bytecode_fb
// CHECK: hal.executable private @main_dispatch_1 {
// CHECK-NEXT: hal.executable.variant public @sa_desc_v1
// CHECK-NOT: hal.executable.condition
// CHECK: hal.executable.variant public @vmvx_bytecode_fb
func.func @main(%x: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %t = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%a: f32, %o: f32):
    %h = math.tanh %a : f32
    linalg.yield %h : f32
  } -> tensor<64xf32>
  %b = util.optimization_barrier %t : tensor<64xf32>
  %e2 = tensor.empty() : tensor<64xf32>
  %y = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%b : tensor<64xf32>) outs(%e2 : tensor<64xf32>) {
  ^bb0(%a: f32, %o: f32):
    %s = math.exp %a : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  return %y : tensor<64xf32>
}

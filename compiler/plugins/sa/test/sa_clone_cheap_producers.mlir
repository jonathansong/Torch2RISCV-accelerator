// sa-clone-cheap-producers (docs/iree_compiler_plan.md §6.8): a value computed
// from indices and scalars only (the attention mask: index <= pos) is
// recomputed inside each consumer, so the prefix structure stays visible to
// the sa backend (the VE's VALID count).
// RUN: iree-opt --split-input-file --iree-sa-clone-cheap-producers="force=true" %s | FileCheck %s
// CHECK-LABEL: func.func @mask
// CHECK-COUNT-2: arith.cmpi sle
func.func @mask(%pos: tensor<i64>, %a: tensor<16xf32>, %b: tensor<16xf32>) -> (tensor<16xf32>, tensor<16xf32>) {
  %e = tensor.empty() : tensor<16xi1>
  %m = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%pos : tensor<i64>) outs(%e : tensor<16xi1>) {
  ^bb0(%p: i64, %o: i1):
    %i = linalg.index 0 : index
    %ii = arith.index_cast %i : index to i64
    %c = arith.cmpi sle, %ii, %p : i64
    linalg.yield %c : i1
  } -> tensor<16xi1>
  %z = arith.constant 0.0 : f32
  %f0 = tensor.empty() : tensor<16xf32>
  %x = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%m, %a : tensor<16xi1>, tensor<16xf32>) outs(%f0 : tensor<16xf32>) {
  ^bb0(%c: i1, %v: f32, %o: f32):
    %s = arith.select %c, %v, %z : f32
    linalg.yield %s : f32
  } -> tensor<16xf32>
  %y = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%m, %b : tensor<16xi1>, tensor<16xf32>) outs(%f0 : tensor<16xf32>) {
  ^bb0(%c: i1, %v: f32, %o: f32):
    %s = arith.select %c, %v, %z : f32
    linalg.yield %s : f32
  } -> tensor<16xf32>
  return %x, %y : tensor<16xf32>, tensor<16xf32>
}

// -----

// Decode's activation scale s = absmax / 127 as a [1, 1, 1] tensor (before
// IREE folds unit dimensions; generality plan §13.4): fused into both its
// consumers (the quantization's 1 / s and a dequantization's * s), no
// generic of its own left.
// CHECK-LABEL: func.func @unit_scale
// CHECK-COUNT-2: arith.mulf %{{.+}}, %{{.+}} : f32
// CHECK-NOT: linalg.generic {{.*}} ins(%{{.+}} : tensor<1x1x1xf32>) outs
func.func @unit_scale(%amax: tensor<1x1x1xf32>, %x: tensor<1x1x8xf32>, %acc: tensor<1x1x8xf32>) -> (tensor<1x1x8xf32>, tensor<1x1x8xf32>) {
  %c = arith.constant 0.00787401571 : f32
  %one = arith.constant 1.0 : f32
  %e1 = tensor.empty() : tensor<1x1x1xf32>
  %s = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%amax : tensor<1x1x1xf32>) outs(%e1 : tensor<1x1x1xf32>) {
  ^bb0(%a: f32, %o: f32):
    %m = arith.mulf %a, %c : f32
    linalg.yield %m : f32
  } -> tensor<1x1x1xf32>
  %e8 = tensor.empty() : tensor<1x1x8xf32>
  %q = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%x, %s : tensor<1x1x8xf32>, tensor<1x1x1xf32>) outs(%e8 : tensor<1x1x8xf32>) {
  ^bb0(%v: f32, %sv: f32, %o: f32):
    %r = arith.divf %one, %sv : f32
    %y = arith.mulf %v, %r : f32
    linalg.yield %y : f32
  } -> tensor<1x1x8xf32>
  %d = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%acc, %s : tensor<1x1x8xf32>, tensor<1x1x1xf32>) outs(%e8 : tensor<1x1x8xf32>) {
  ^bb0(%v: f32, %sv: f32, %o: f32):
    %y = arith.mulf %v, %sv : f32
    linalg.yield %y : f32
  } -> tensor<1x1x8xf32>
  return %q, %d : tensor<1x1x8xf32>, tensor<1x1x8xf32>
}

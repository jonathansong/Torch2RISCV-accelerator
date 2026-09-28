// sa-clone-cheap-producers (docs/iree_compiler_plan.md §6.8): a value computed
// from indices and scalars only (the attention mask: index <= pos) is
// recomputed inside each consumer, so the prefix structure stays visible to
// the sa backend (the VE's VALID count).
// RUN: iree-opt --iree-sa-clone-cheap-producers="force=true" %s | FileCheck %s
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

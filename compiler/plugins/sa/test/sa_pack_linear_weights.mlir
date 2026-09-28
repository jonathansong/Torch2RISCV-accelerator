// sa-pack-linear-weights (docs/iree_compiler_plan.md §6.3, C6.0): a linear
// layer vecmat(x, transpose(extsi(W))) with W an immutable global becomes a
// contraction over the packed weights Wp = pack(W) [N / D, K, D], a global of
// its own set by an initializer; a gather of W reads Wp[r / D, c, r % D].
// RUN: iree-opt --iree-sa-pack-linear-weights="force=true" %s | FileCheck %s

// CHECK: util.global private @w$packed : tensor<2x8x8xi8>
// CHECK: util.initializer {
// CHECK: %[[W:.+]] = util.global.load @w : tensor<16x8xi8>
// CHECK: linalg.pack %[[W]] inner_dims_pos = [0] inner_tiles = [8]
// CHECK: util.global.store %{{.+}}, @w$packed
// CHECK-LABEL: func.func @lin
// CHECK: %[[WP:.+]] = util.global.load @w$packed : tensor<2x8x8xi8>
// CHECK: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction"]{{.*}} ins(%{{.+}}, %[[WP]] : tensor<8xi32>, tensor<2x8x8xi8>)
// CHECK: tensor.collapse_shape %{{.+}} {{\[\[}}0, 1{{\]\]}} : tensor<2x8xi32> into tensor<16xi32>
// CHECK-NOT: linalg.vecmat
// CHECK-LABEL: func.func @embed
// CHECK: tensor.extract %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] : tensor<2x8x8xi8>
util.global private @w = dense<1> : tensor<16x8xi8>
func.func @lin(%x: tensor<8xi8>) -> tensor<16xi32> {
  %c0 = arith.constant 0 : i32
  %w = util.global.load @w : tensor<16x8xi8>
  %e0 = tensor.empty() : tensor<16x8xi32>
  %we = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%w : tensor<16x8xi8>) outs(%e0 : tensor<16x8xi32>) {
  ^bb0(%a: i8, %o: i32):
    %e = arith.extsi %a : i8 to i32
    linalg.yield %e : i32
  } -> tensor<16x8xi32>
  %t0 = tensor.empty() : tensor<8x16xi32>
  %wt = linalg.transpose ins(%we : tensor<16x8xi32>) outs(%t0 : tensor<8x16xi32>) permutation = [1, 0]
  %x0 = tensor.empty() : tensor<8xi32>
  %xe = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x : tensor<8xi8>) outs(%x0 : tensor<8xi32>) {
  ^bb0(%a: i8, %o: i32):
    %e = arith.extsi %a : i8 to i32
    linalg.yield %e : i32
  } -> tensor<8xi32>
  %z = tensor.empty() : tensor<16xi32>
  %zf = linalg.fill ins(%c0 : i32) outs(%z : tensor<16xi32>) -> tensor<16xi32>
  %y = linalg.vecmat ins(%xe, %wt : tensor<8xi32>, tensor<8x16xi32>) outs(%zf : tensor<16xi32>) -> tensor<16xi32>
  return %y : tensor<16xi32>
}
func.func @embed(%r: index, %c: index) -> i8 {
  %w = util.global.load @w : tensor<16x8xi8>
  %e = tensor.extract %w[%r, %c] : tensor<16x8xi8>
  return %e : i8
}

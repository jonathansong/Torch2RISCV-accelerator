// sa-pack-linear-weights, expert tables (docs/moe_native_lowering_design.md §5.1):
// a gather of one row of a constant table at a Top-2 id (a sort's indices)
// becomes a dynamic extract_slice of the table, and a linear layer over a
// slice of an int8 table T [E, N, K] reads the slice of Tp = pack(T) along N,
// [E, N/D, K, D], at the same id. A gather at another index (an embedding's
// token id) keeps its form.
// RUN: iree-opt --split-input-file --iree-sa-pack-linear-weights="force=true" %s | FileCheck %s

// CHECK: util.global private @t$packed : tensor<2x2x8x8xi8>
// CHECK: util.initializer {
// CHECK: %[[T:.+]] = util.global.load @t : tensor<2x16x8xi8>
// CHECK: linalg.pack %[[T]] inner_dims_pos = [1] inner_tiles = [8] {{.*}} : tensor<2x16x8xi8> -> tensor<2x2x8x8xi8>
// CHECK: util.global.store %{{.+}}, @t$packed
// CHECK-LABEL: func.func @expert
// CHECK: %[[SORTED:.+]]:2 = iree_linalg_ext.sort
// CHECK: %[[TP:.+]] = util.global.load @t$packed : tensor<2x2x8x8xi8>
// CHECK: %[[ID:.+]] = tensor.extract %{{.+}}[%{{.+}}] : tensor<1xi64>
// CHECK: %[[I:.+]] = arith.index_cast %[[ID]] : i64 to index
// CHECK: %[[W:.+]] = tensor.extract_slice %[[TP]][%[[I]], 0, 0, 0] [1, 2, 8, 8] [1, 1, 1, 1] : tensor<2x2x8x8xi8> to tensor<2x8x8xi8>
// CHECK: %[[S:.+]] = tensor.extract_slice %{{.+}}[%{{.+}}, 0] [1, 16] [1, 1] : tensor<2x16xf32> to tensor<16xf32>
// CHECK: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "parallel", "reduction"]{{.*}} ins(%{{.+}}, %[[W]] : tensor<1x8xi32>, tensor<2x8x8xi8>)
// CHECK-NOT: linalg.matmul
// CHECK-NOT: tensor.extract %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] : tensor<2x16x8xi8>
util.global private @t = dense<1> : tensor<2x16x8xi8>
util.global private @s = dense<1.0> : tensor<2x16xf32>
func.func @expert(%x: tensor<1x8xi32>, %scores: tensor<1x4xf32>, %ids: tensor<1x4xi64>) -> (tensor<1x16xi64>, tensor<16xf32>) {
  %sorted:2 = iree_linalg_ext.sort dimension(1) outs(%scores, %ids : tensor<1x4xf32>, tensor<1x4xi64>) {
  ^bb0(%a: f32, %b: f32, %ia: i64, %ib: i64):
    %ge = arith.cmpf oge, %a, %b : f32
    iree_linalg_ext.yield %ge : i1
  } -> tensor<1x4xf32>, tensor<1x4xi64>
  %top = tensor.extract_slice %sorted#1[0, 0] [1, 2] [1, 1] : tensor<1x4xi64> to tensor<1x2xi64>
  %flat = tensor.collapse_shape %top [[0, 1]] : tensor<1x2xi64> into tensor<2xi64>
  %id = tensor.extract_slice %flat[0] [1] [1] : tensor<2xi64> to tensor<1xi64>
  %t = util.global.load @t : tensor<2x16x8xi8>
  %ge = tensor.empty() : tensor<1x16x8xi8>
  %g = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%id : tensor<1xi64>) outs(%ge : tensor<1x16x8xi8>) {
  ^bb0(%in: i64, %out: i8):
    %i = arith.index_cast %in : i64 to index
    %j = linalg.index 1 : index
    %k = linalg.index 2 : index
    %v = tensor.extract %t[%i, %j, %k] : tensor<2x16x8xi8>
    linalg.yield %v : i8
  } -> tensor<1x16x8xi8>
  %w = tensor.collapse_shape %g [[0, 1], [2]] : tensor<1x16x8xi8> into tensor<16x8xi8>
  %s = util.global.load @s : tensor<2x16xf32>
  %se = tensor.empty() : tensor<1x16xf32>
  %sg = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%id : tensor<1xi64>) outs(%se : tensor<1x16xf32>) {
  ^bb0(%in: i64, %out: f32):
    %i = arith.index_cast %in : i64 to index
    %j = linalg.index 1 : index
    %v = tensor.extract %s[%i, %j] : tensor<2x16xf32>
    linalg.yield %v : f32
  } -> tensor<1x16xf32>
  %scale = tensor.collapse_shape %sg [[0, 1]] : tensor<1x16xf32> into tensor<16xf32>
  %e0 = tensor.empty() : tensor<16x8xi32>
  %we = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%w : tensor<16x8xi8>) outs(%e0 : tensor<16x8xi32>) {
  ^bb0(%a: i8, %o: i32):
    %e = arith.extsi %a : i8 to i32
    linalg.yield %e : i32
  } -> tensor<16x8xi32>
  %t0 = tensor.empty() : tensor<8x16xi32>
  %wt = linalg.transpose ins(%we : tensor<16x8xi32>) outs(%t0 : tensor<8x16xi32>) permutation = [1, 0]
  %c0 = arith.constant 0 : i64
  %z = tensor.empty() : tensor<1x16xi64>
  %zf = linalg.fill ins(%c0 : i64) outs(%z : tensor<1x16xi64>) -> tensor<1x16xi64>
  %y = linalg.matmul ins(%x, %wt : tensor<1x8xi32>, tensor<8x16xi32>) outs(%zf : tensor<1x16xi64>) -> tensor<1x16xi64>
  return %y, %scale : tensor<1x16xi64>, tensor<16xf32>
}

// -----

// A row gather at an index that is not a Top-2 id (an embedding's token): unchanged.
// CHECK-LABEL: func.func @not_topk
// CHECK: linalg.generic
// CHECK: tensor.extract %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] : tensor<2x16x8xi8>
// CHECK-NOT: tensor.extract_slice
util.global private @t = dense<1> : tensor<2x16x8xi8>
func.func @not_topk(%id: tensor<1xi64>) -> tensor<1x16x8xi8> {
  %t = util.global.load @t : tensor<2x16x8xi8>
  %ge = tensor.empty() : tensor<1x16x8xi8>
  %g = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%id : tensor<1xi64>) outs(%ge : tensor<1x16x8xi8>) {
  ^bb0(%in: i64, %out: i8):
    %i = arith.index_cast %in : i64 to index
    %j = linalg.index 1 : index
    %k = linalg.index 2 : index
    %v = tensor.extract %t[%i, %j, %k] : tensor<2x16x8xi8>
    linalg.yield %v : i8
  } -> tensor<1x16x8xi8>
  return %g : tensor<1x16x8xi8>
}

// M5a: recognize a batch-1 router projection followed by descending top-2.
// RUN: iree-opt --split-input-file --iree-sa-form-moe-regions="force=true" %s | FileCheck %s

// CHECK-LABEL: func.func @route_top2
// CHECK:       linalg.matmul
// CHECK:       flow.dispatch.region
// CHECK-SAME:    sa.moe.experts = 8 : i64
// CHECK-SAME:    sa.moe.kind = "route_top2"
// CHECK-SAME:    sa.moe.top_k = 2 : i64
// CHECK:         iree_linalg_ext.topk
// CHECK-SAME:    sa.moe.experts = 8 : i64
// CHECK:         flow.return
func.func @route_top2(%x: tensor<1x4xf32>, %w: tensor<4x8xf32>) -> (tensor<1x2xf32>, tensor<1x2xi32>) {
  %e = tensor.empty() : tensor<1x8xf32>
  %scores = linalg.matmul ins(%x, %w : tensor<1x4xf32>, tensor<4x8xf32>) outs(%e : tensor<1x8xf32>) -> tensor<1x8xf32>
  %scaled_empty = tensor.empty() : tensor<1x8xf32>
  %scaled = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                            iterator_types = ["parallel", "parallel"]}
      ins(%scores : tensor<1x8xf32>) outs(%scaled_empty : tensor<1x8xf32>) {
  ^bb0(%score: f32, %old: f32):
    %two = arith.constant 2.0 : f32
    %value = arith.mulf %score, %two : f32
    linalg.yield %value : f32
  } -> tensor<1x8xf32>
  %ve = tensor.empty() : tensor<1x2xf32>
  %ie = tensor.empty() : tensor<1x2xi32>
  %ninf = arith.constant 0xFF800000 : f32
  %maxidx = arith.constant 2147483647 : i32
  %vi = linalg.fill ins(%ninf : f32) outs(%ve : tensor<1x2xf32>) -> tensor<1x2xf32>
  %ii = linalg.fill ins(%maxidx : i32) outs(%ie : tensor<1x2xi32>) -> tensor<1x2xi32>
  %v, %i = iree_linalg_ext.topk dimension(1) ins(%scaled : tensor<1x8xf32>)
      outs(%vi, %ii : tensor<1x2xf32>, tensor<1x2xi32>) {
  ^bb0(%a: f32, %b: f32):
    %gt = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %gt : i1
  } -> tensor<1x2xf32>, tensor<1x2xi32>
  return %v, %i : tensor<1x2xf32>, tensor<1x2xi32>
}

// -----

// A quantized router has an i8 contraction followed by dequantization.
#x = affine_map<(d0, d1, d2) -> (d2)>
#w = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#o = affine_map<(d0, d1, d2) -> (d0, d1)>
#id = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @quantized_router
// CHECK:       linalg.generic
// CHECK:       flow.dispatch.region
// CHECK-SAME:    sa.moe.experts = 8 : i64
// CHECK-SAME:    sa.moe.kind = "route_top2"
// CHECK:         iree_linalg_ext.topk
// CHECK-SAME:    sa.moe.experts = 8 : i64
func.func @quantized_router(%x: tensor<32xi8>, %w: tensor<1x32x8xi8>) -> (tensor<1x2xf32>, tensor<1x2xi32>) {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<1x8xi32>
  %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<1x8xi32>) -> tensor<1x8xi32>
  %raw = linalg.generic {indexing_maps = [#x, #w, #o], iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%x, %w : tensor<32xi8>, tensor<1x32x8xi8>) outs(%z : tensor<1x8xi32>) {
  ^bb0(%a: i8, %b: i8, %acc: i32):
    %ai = arith.extsi %a : i8 to i32
    %bi = arith.extsi %b : i8 to i32
    %mul = arith.muli %ai, %bi : i32
    %sum = arith.addi %acc, %mul : i32
    linalg.yield %sum : i32
  } -> tensor<1x8xi32>
  %fe = tensor.empty() : tensor<1x8xf32>
  %scores = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%raw : tensor<1x8xi32>) outs(%fe : tensor<1x8xf32>) {
  ^bb0(%raw_score: i32, %old: f32):
    %score = arith.sitofp %raw_score : i32 to f32
    linalg.yield %score : f32
  } -> tensor<1x8xf32>
  %ve = tensor.empty() : tensor<1x2xf32>
  %ie = tensor.empty() : tensor<1x2xi32>
  %ninf = arith.constant 0xFF800000 : f32
  %maxidx = arith.constant 2147483647 : i32
  %vi = linalg.fill ins(%ninf : f32) outs(%ve : tensor<1x2xf32>) -> tensor<1x2xf32>
  %ii = linalg.fill ins(%maxidx : i32) outs(%ie : tensor<1x2xi32>) -> tensor<1x2xi32>
  %v, %i = iree_linalg_ext.topk dimension(1) ins(%scores : tensor<1x8xf32>)
      outs(%vi, %ii : tensor<1x2xf32>, tensor<1x2xi32>) {
  ^bb0(%a: f32, %b: f32):
    %gt = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %gt : i1
  } -> tensor<1x2xf32>, tensor<1x2xi32>
  return %v, %i : tensor<1x2xf32>, tensor<1x2xi32>
}

// -----

// The Top-K scalar implementation reads its output initializers. Empty
// outputs cannot be replaced by the raw Top-2 kernel.
// CHECK-LABEL: func.func @uninitialized_top2
// CHECK-NOT: flow.dispatch.region
func.func @uninitialized_top2(%x: tensor<1x4xf32>, %w: tensor<4x8xf32>) -> (tensor<1x2xf32>, tensor<1x2xi32>) {
  %e = tensor.empty() : tensor<1x8xf32>
  %scores = linalg.matmul ins(%x, %w : tensor<1x4xf32>, tensor<4x8xf32>) outs(%e : tensor<1x8xf32>) -> tensor<1x8xf32>
  %ve = tensor.empty() : tensor<1x2xf32>
  %ie = tensor.empty() : tensor<1x2xi32>
  %v, %i = iree_linalg_ext.topk dimension(1) ins(%scores : tensor<1x8xf32>)
      outs(%ve, %ie : tensor<1x2xf32>, tensor<1x2xi32>) {
  ^bb0(%a: f32, %b: f32):
    %gt = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %gt : i1
  } -> tensor<1x2xf32>, tensor<1x2xi32>
  return %v, %i : tensor<1x2xf32>, tensor<1x2xi32>
}

// -----

// K=3 must remain on IREE's ordinary path.
// CHECK-LABEL: func.func @not_top2
// CHECK-NOT: flow.dispatch.region
func.func @not_top2(%x: tensor<1x4xf32>, %w: tensor<4x8xf32>) -> (tensor<1x3xf32>, tensor<1x3xi32>) {
  %e = tensor.empty() : tensor<1x8xf32>
  %scores = linalg.matmul ins(%x, %w : tensor<1x4xf32>, tensor<4x8xf32>) outs(%e : tensor<1x8xf32>) -> tensor<1x8xf32>
  %ve = tensor.empty() : tensor<1x3xf32>
  %ie = tensor.empty() : tensor<1x3xi32>
  %v, %i = iree_linalg_ext.topk dimension(1) ins(%scores : tensor<1x8xf32>)
      outs(%ve, %ie : tensor<1x3xf32>, tensor<1x3xi32>) {
  ^bb0(%a: f32, %b: f32):
    %gt = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %gt : i1
  } -> tensor<1x3xf32>, tensor<1x3xi32>
  return %v, %i : tensor<1x3xf32>, tensor<1x3xi32>
}

// -----

// A top-k without a router projection must not be classified as MoE.
// CHECK-LABEL: func.func @not_router
// CHECK-NOT: flow.dispatch.region
func.func @not_router(%scores: tensor<1x8xf32>) -> (tensor<1x2xf32>, tensor<1x2xi32>) {
  %ve = tensor.empty() : tensor<1x2xf32>
  %ie = tensor.empty() : tensor<1x2xi32>
  %v, %i = iree_linalg_ext.topk dimension(1) ins(%scores : tensor<1x8xf32>)
      outs(%ve, %ie : tensor<1x2xf32>, tensor<1x2xi32>) {
  ^bb0(%a: f32, %b: f32):
    %gt = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %gt : i1
  } -> tensor<1x2xf32>, tensor<1x2xi32>
  return %v, %i : tensor<1x2xf32>, tensor<1x2xi32>
}

// -----

// torch.topk is imported as a full descending sort and a two-element prefix.
// The pass must form the same device Top-2 region for that frontend spelling.
// CHECK-LABEL: func.func @torch_sort_prefix
// CHECK:       linalg.matmul
// CHECK:       flow.dispatch.region
// CHECK-SAME:    sa.moe.experts = 4 : i64
// CHECK:         iree_linalg_ext.topk
// CHECK-NOT:   iree_linalg_ext.sort
func.func @torch_sort_prefix(%x: tensor<1x32xf32>, %w: tensor<32x4xf32>) -> (tensor<1x2xf32>, tensor<1x2xi64>) {
  %c0 = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x4xf32>
  %init = linalg.fill ins(%c0 : f32) outs(%empty : tensor<1x4xf32>) -> tensor<1x4xf32>
  %scores = linalg.matmul ins(%x, %w : tensor<1x32xf32>, tensor<32x4xf32>) outs(%init : tensor<1x4xf32>) -> tensor<1x4xf32>
  %ids_empty = tensor.empty() : tensor<1x4xi64>
  %ids = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%ids_empty : tensor<1x4xi64>) {
  ^bb0(%old: i64):
    %index = linalg.index 1 : index
    %cast = arith.index_cast %index : index to i64
    linalg.yield %cast : i64
  } -> tensor<1x4xi64>
  %sorted:2 = iree_linalg_ext.sort dimension(1) outs(%scores, %ids : tensor<1x4xf32>, tensor<1x4xi64>) {
  ^bb0(%a: f32, %b: f32, %ia: i64, %ib: i64):
    %ge = arith.cmpf oge, %a, %b : f32
    iree_linalg_ext.yield %ge : i1
  } -> tensor<1x4xf32>, tensor<1x4xi64>
  %values = tensor.extract_slice %sorted#0[0, 0] [1, 2] [1, 1] : tensor<1x4xf32> to tensor<1x2xf32>
  %indices = tensor.extract_slice %sorted#1[0, 0] [1, 2] [1, 1] : tensor<1x4xi64> to tensor<1x2xi64>
  return %values, %indices : tensor<1x2xf32>, tensor<1x2xi64>
}

// M5a: recognize a batch-1 router projection followed by descending top-2.
// RUN: iree-opt --split-input-file --iree-sa-form-moe-regions="force=true" %s | FileCheck %s

// CHECK-LABEL: func.func @route_top2
// CHECK:       linalg.matmul
// CHECK:       flow.dispatch.region
// CHECK-SAME:    sa.moe.experts = 8 : i64
// CHECK-SAME:    sa.moe.kind = "route_top2"
// CHECK-SAME:    sa.moe.top_k = 2 : i64
// CHECK:         iree_linalg_ext.topk
// CHECK:         flow.return
func.func @route_top2(%x: tensor<1x4xf32>, %w: tensor<4x8xf32>) -> (tensor<1x2xf32>, tensor<1x2xi32>) {
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

// IsolateMoeExperts (docs/moe_native_lowering_design.md §23): with a Top-k router of k > 2, each
// expert's down-projection epilogue ends at a util.optimization_barrier before the experts' sum.
// RUN: iree-opt --split-input-file --iree-sa-isolate-moe-experts="force=true" %s | FileCheck %s

#x = affine_map<(d0, d1) -> (d1)>
#w = affine_map<(d0, d1) -> (d1, d0)>
#o = affine_map<(d0, d1) -> (d0)>
#id = affine_map<(d0) -> (d0)>
#s = affine_map<(d0) -> ()>
// CHECK-LABEL: func.func @top8
// CHECK:       %[[T0:.+]] = linalg.generic {{.*}} ins(%{{.+}}, %{{.+}} : tensor<16xf32>, tensor<f32>)
// CHECK:       %[[B0:.+]] = util.optimization_barrier %[[T0]]
// CHECK:       %[[T1:.+]] = linalg.generic {{.*}} ins(%{{.+}}, %{{.+}} : tensor<16xf32>, tensor<f32>)
// CHECK:       %[[B1:.+]] = util.optimization_barrier %[[T1]]
// CHECK:       linalg.generic {{.*}} ins(%[[B0]], %[[B1]] :
func.func @top8(%x: tensor<32xi8>, %table: tensor<4x32x16xi8>, %ids: tensor<2xi32>, %w0: tensor<f32>,
                %w1: tensor<f32>, %scores: tensor<1x4xf32>) -> (tensor<16xf32>, tensor<1x8xf32>) {
  %r = flow.dispatch.region -> (tensor<1x8xf32>) attributes {sa.moe.experts = 32 : i64, sa.moe.kind = "topk", sa.moe.top_k = 8 : i64} {
    %e = tensor.empty() : tensor<1x8xf32>
    flow.return %e : tensor<1x8xf32>
  }
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %z = arith.constant 0 : i32
  %i0 = tensor.extract %ids[%c0] : tensor<2xi32>
  %i1 = tensor.extract %ids[%c1] : tensor<2xi32>
  %x0 = arith.index_cast %i0 : i32 to index
  %x1 = arith.index_cast %i1 : i32 to index
  %t0 = tensor.extract_slice %table[%x0, 0, 0] [1, 32, 16] [1, 1, 1] : tensor<4x32x16xi8> to tensor<32x16xi8>
  %t1 = tensor.extract_slice %table[%x1, 0, 0] [1, 32, 16] [1, 1, 1] : tensor<4x32x16xi8> to tensor<32x16xi8>
  %ae = tensor.empty() : tensor<16xi32>
  %a = linalg.fill ins(%z : i32) outs(%ae : tensor<16xi32>) -> tensor<16xi32>
  %fe = tensor.empty() : tensor<16xf32>
  %acc0 = linalg.generic {indexing_maps = [#x, #w, #o], iterator_types = ["parallel", "reduction"]}
      ins(%x, %t0 : tensor<32xi8>, tensor<32x16xi8>) outs(%a : tensor<16xi32>) {
  ^bb0(%p: i8, %q: i8, %acc: i32):
    %pe = arith.extsi %p : i8 to i32
    %qe = arith.extsi %q : i8 to i32
    %m = arith.muli %pe, %qe : i32
    %s = arith.addi %m, %acc : i32
    linalg.yield %s : i32
  } -> tensor<16xi32>
  %f0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%acc0 : tensor<16xi32>) outs(%fe : tensor<16xf32>) {
  ^bb0(%p: i32, %o: f32):
    %v = arith.sitofp %p : i32 to f32
    linalg.yield %v : f32
  } -> tensor<16xf32>
  %e0 = linalg.generic {indexing_maps = [#id, #s, #id], iterator_types = ["parallel"]} ins(%f0, %w0 : tensor<16xf32>, tensor<f32>) outs(%fe : tensor<16xf32>) {
  ^bb0(%p: f32, %q: f32, %o: f32):
    %v = arith.mulf %p, %q : f32
    linalg.yield %v : f32
  } -> tensor<16xf32>
  %acc1 = linalg.generic {indexing_maps = [#x, #w, #o], iterator_types = ["parallel", "reduction"]}
      ins(%x, %t1 : tensor<32xi8>, tensor<32x16xi8>) outs(%a : tensor<16xi32>) {
  ^bb0(%p: i8, %q: i8, %acc: i32):
    %pe = arith.extsi %p : i8 to i32
    %qe = arith.extsi %q : i8 to i32
    %m = arith.muli %pe, %qe : i32
    %s = arith.addi %m, %acc : i32
    linalg.yield %s : i32
  } -> tensor<16xi32>
  %f1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%acc1 : tensor<16xi32>) outs(%fe : tensor<16xf32>) {
  ^bb0(%p: i32, %o: f32):
    %v = arith.sitofp %p : i32 to f32
    linalg.yield %v : f32
  } -> tensor<16xf32>
  %e1 = linalg.generic {indexing_maps = [#id, #s, #id], iterator_types = ["parallel"]} ins(%f1, %w1 : tensor<16xf32>, tensor<f32>) outs(%fe : tensor<16xf32>) {
  ^bb0(%p: f32, %q: f32, %o: f32):
    %v = arith.mulf %p, %q : f32
    linalg.yield %v : f32
  } -> tensor<16xf32>
  %y = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]} ins(%e0, %e1 : tensor<16xf32>, tensor<16xf32>) outs(%fe : tensor<16xf32>) {
  ^bb0(%p: f32, %q: f32, %o: f32):
    %v = arith.addf %p, %q : f32
    linalg.yield %v : f32
  } -> tensor<16xf32>
  return %y, %r : tensor<16xf32>, tensor<1x8xf32>
}

// -----

// Top-2: the dispatches of before, no barrier.
#x = affine_map<(d0, d1) -> (d1)>
#w = affine_map<(d0, d1) -> (d1, d0)>
#o = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @top2
// CHECK-NOT:   util.optimization_barrier
func.func @top2(%x: tensor<32xi8>, %table: tensor<4x32x16xi8>, %ids: tensor<2xi32>) -> (tensor<16xi32>, tensor<1x2xf32>) {
  %r = flow.dispatch.region -> (tensor<1x2xf32>) attributes {sa.moe.experts = 8 : i64, sa.moe.kind = "topk", sa.moe.top_k = 2 : i64} {
    %e = tensor.empty() : tensor<1x2xf32>
    flow.return %e : tensor<1x2xf32>
  }
  %c0 = arith.constant 0 : index
  %z = arith.constant 0 : i32
  %i0 = tensor.extract %ids[%c0] : tensor<2xi32>
  %x0 = arith.index_cast %i0 : i32 to index
  %t0 = tensor.extract_slice %table[%x0, 0, 0] [1, 32, 16] [1, 1, 1] : tensor<4x32x16xi8> to tensor<32x16xi8>
  %ae = tensor.empty() : tensor<16xi32>
  %a = linalg.fill ins(%z : i32) outs(%ae : tensor<16xi32>) -> tensor<16xi32>
  %acc0 = linalg.generic {indexing_maps = [#x, #w, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]}
      ins(%x, %t0 : tensor<32xi8>, tensor<32x16xi8>) outs(%a : tensor<16xi32>) {
  ^bb0(%p: i8, %q: i8, %acc: i32):
    %pe = arith.extsi %p : i8 to i32
    %qe = arith.extsi %q : i8 to i32
    %m = arith.muli %pe, %qe : i32
    %s = arith.addi %m, %acc : i32
    linalg.yield %s : i32
  } -> tensor<16xi32>
  %fe = tensor.empty() : tensor<16xi32>
  %y = linalg.generic {indexing_maps = [#o, #o], iterator_types = ["parallel"]} ins(%acc0 : tensor<16xi32>) outs(%fe : tensor<16xi32>) {
  ^bb0(%p: i32, %q: i32):
    %v = arith.addi %p, %p : i32
    linalg.yield %v : i32
  } -> tensor<16xi32>
  return %y, %r : tensor<16xi32>, tensor<1x2xf32>
}

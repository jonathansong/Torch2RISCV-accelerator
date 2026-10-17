// sa-form-ffn-regions (docs/fusion_design.md §4.2): decode's gated FFN, the
// two projections of one x_q, the gating, the amax and the int8 output, in one
// flow.dispatch.region; the down projection (a contraction) stays outside.
// RUN: iree-opt --split-input-file --iree-sa-form-ffn-regions="force=true" %s | FileCheck %s

#x = affine_map<(d0, d1, d2) -> (d2)>
#w = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#o = affine_map<(d0, d1, d2) -> (d0, d1)>
#id = affine_map<(d0) -> (d0)>
#sc = affine_map<(d0) -> ()>
// CHECK-LABEL: func.func @ffn
// CHECK:       %[[R:.+]]:2 = flow.dispatch.region -> (tensor<64xi8>, tensor<f32>)
// CHECK-COUNT-2: arith.muli
// CHECK:         math.exp
// CHECK:         arith.maximumf
// CHECK:         arith.fptosi
// CHECK:         flow.return
// the down projection, outside, of the region's int8 h
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%{{.+}}, %{{.+}} : tensor<64xi8>, tensor<1x64x16xi8>)
func.func @ffn(%x: tensor<32xi8>, %w1: tensor<4x32x16xi8>, %w2: tensor<4x32x16xi8>, %wd: tensor<1x64x16xi8>)
    -> (tensor<1x16xi32>, tensor<f32>) {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<4x16xi32>
  %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<4x16xi32>) -> tensor<4x16xi32>
  %a = linalg.generic {indexing_maps = [#x, #w, #o], iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%x, %w1 : tensor<32xi8>, tensor<4x32x16xi8>) outs(%z : tensor<4x16xi32>) {
  ^bb0(%i: i8, %j: i8, %acc: i32):
    %0 = arith.extsi %i : i8 to i32
    %1 = arith.extsi %j : i8 to i32
    %2 = arith.muli %0, %1 : i32
    %3 = arith.addi %acc, %2 : i32
    linalg.yield %3 : i32
  } -> tensor<4x16xi32>
  %b = linalg.generic {indexing_maps = [#x, #w, #o], iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%x, %w2 : tensor<32xi8>, tensor<4x32x16xi8>) outs(%z : tensor<4x16xi32>) {
  ^bb0(%i: i8, %j: i8, %acc: i32):
    %0 = arith.extsi %i : i8 to i32
    %1 = arith.extsi %j : i8 to i32
    %2 = arith.muli %0, %1 : i32
    %3 = arith.addi %acc, %2 : i32
    linalg.yield %3 : i32
  } -> tensor<4x16xi32>
  %af = tensor.collapse_shape %a [[0, 1]] : tensor<4x16xi32> into tensor<64xi32>
  %bf = tensor.collapse_shape %b [[0, 1]] : tensor<4x16xi32> into tensor<64xi32>
  %he = tensor.empty() : tensor<64xf32>
  %h = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}
      ins(%af, %bf : tensor<64xi32>, tensor<64xi32>) outs(%he : tensor<64xf32>) {
  ^bb0(%i: i32, %j: i32, %o: f32):
    %0 = arith.sitofp %i : i32 to f32
    %1 = arith.sitofp %j : i32 to f32
    %2 = arith.negf %0 : f32
    %3 = math.exp %2 : f32
    %4 = arith.mulf %3, %1 : f32
    linalg.yield %4 : f32
  } -> tensor<64xf32>
  %ninf = arith.constant 0xFF800000 : f32
  %me = tensor.empty() : tensor<f32>
  %mi = linalg.fill ins(%ninf : f32) outs(%me : tensor<f32>) -> tensor<f32>
  %m = linalg.generic {indexing_maps = [#id, #sc], iterator_types = ["reduction"]}
      ins(%h : tensor<64xf32>) outs(%mi : tensor<f32>) {
  ^bb0(%v: f32, %o: f32):
    %0 = math.absf %v : f32
    %1 = arith.maximumf %0, %o : f32
    linalg.yield %1 : f32
  } -> tensor<f32>
  %qe = tensor.empty() : tensor<64xi8>
  %q = linalg.generic {indexing_maps = [#id, #sc, #id], iterator_types = ["parallel"]}
      ins(%h, %m : tensor<64xf32>, tensor<f32>) outs(%qe : tensor<64xi8>) {
  ^bb0(%v: f32, %s: f32, %o: i8):
    %0 = arith.divf %v, %s : f32
    %1 = arith.fptosi %0 : f32 to i8
    linalg.yield %1 : i8
  } -> tensor<64xi8>
  %de = tensor.empty() : tensor<1x16xi32>
  %dz = linalg.fill ins(%c0 : i32) outs(%de : tensor<1x16xi32>) -> tensor<1x16xi32>
  %y = linalg.generic {indexing_maps = [#x, #w, #o], iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%q, %wd : tensor<64xi8>, tensor<1x64x16xi8>) outs(%dz : tensor<1x16xi32>) {
  ^bb0(%i: i8, %j: i8, %acc: i32):
    %0 = arith.extsi %i : i8 to i32
    %1 = arith.extsi %j : i8 to i32
    %2 = arith.muli %0, %1 : i32
    %3 = arith.addi %acc, %2 : i32
    linalg.yield %3 : i32
  } -> tensor<1x16xi32>
  return %y, %m : tensor<1x16xi32>, tensor<f32>
}

// -----

// two projections of one x with no op depending on both (K and V): left alone
#x = affine_map<(d0, d1, d2) -> (d2)>
#w = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#o = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @kv
// CHECK-NOT:   flow.dispatch.region
func.func @kv(%x: tensor<32xi8>, %w1: tensor<4x32x16xi8>, %w2: tensor<4x32x16xi8>) -> (tensor<4x16xi32>, tensor<4x16xi32>) {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<4x16xi32>
  %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<4x16xi32>) -> tensor<4x16xi32>
  %a = linalg.generic {indexing_maps = [#x, #w, #o], iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%x, %w1 : tensor<32xi8>, tensor<4x32x16xi8>) outs(%z : tensor<4x16xi32>) {
  ^bb0(%i: i8, %j: i8, %acc: i32):
    %0 = arith.extsi %i : i8 to i32
    %1 = arith.extsi %j : i8 to i32
    %2 = arith.muli %0, %1 : i32
    %3 = arith.addi %acc, %2 : i32
    linalg.yield %3 : i32
  } -> tensor<4x16xi32>
  %b = linalg.generic {indexing_maps = [#x, #w, #o], iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%x, %w2 : tensor<32xi8>, tensor<4x32x16xi8>) outs(%z : tensor<4x16xi32>) {
  ^bb0(%i: i8, %j: i8, %acc: i32):
    %0 = arith.extsi %i : i8 to i32
    %1 = arith.extsi %j : i8 to i32
    %2 = arith.muli %0, %1 : i32
    %3 = arith.addi %acc, %2 : i32
    linalg.yield %3 : i32
  } -> tensor<4x16xi32>
  return %a, %b : tensor<4x16xi32>, tensor<4x16xi32>
}

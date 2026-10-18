// sa-form-attn-regions (docs/fusion_design.md §6.4.1): the dispatch regions of
// decode's attention, as IREE forms them (q path, scores over the K cache slice
// [:T], probabilities, their sum, p . V over the V cache slice, amax, int8), in
// one flow.dispatch.region; q_proj and o_proj (static contractions) and a value
// the q path shares with the outside stay outside.
// RUN: iree-opt --split-input-file --iree-sa-form-attn-regions="force=true max-dynamic=8" %s | FileCheck %s

#v1 = affine_map<(d0, d1) -> (d1)>
#m2 = affine_map<(d0, d1) -> (d0, d1)>
#r2 = affine_map<(d0, d1) -> (d0)>
#s2 = affine_map<(d0, d1) -> ()>
#id = affine_map<(d0) -> (d0)>
#sc = affine_map<(d0) -> ()>
#qa = affine_map<(d0, d1, d2) -> (d0, d2)>
#ka = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#va = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#o3 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @attn
// CHECK:       flow.dispatch.region -> (tensor<8xi32>)
// CHECK:       %[[ROPE:.+]] = flow.dispatch.region -> (tensor<2x4xf32>)
// CHECK:       %[[A:.+]]:2 = flow.dispatch.region -> (tensor<f32>, tensor<8xi8>)
// CHECK-NEXT:    %[[T:.+]] = util.assume.int %{{.+}}<umin = 0, umax = 8> : index
// CHECK:         tensor.extract_slice %{{.+}}[0, 0, 0] [2, %[[T]], 4]
// CHECK-NOT:     flow.dispatch.region
// CHECK:         math.exp
// CHECK:         flow.return
// CHECK:       flow.dispatch.region -> (tensor<8xi32>)
// CHECK:         linalg.generic
// CHECK-SAME:    ins(%[[A]]#1, %{{.+}}, %[[A]]#0
func.func @attn(%x: tensor<8xi8>, %wq: tensor<8x8xi8>, %wo: tensor<8x8xi8>, %kc: tensor<2x8x4xi8>,
                %vc: tensor<2x8x4xi8>, %T: index, %pos: tensor<i64>) -> (tensor<8xi32>, tensor<2x4xf32>) {
  %qp = flow.dispatch.region -> (tensor<8xi32>) {
    %c0 = arith.constant 0 : i32
    %e = tensor.empty() : tensor<8xi32>
    %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<8xi32>) -> tensor<8xi32>
    %0 = linalg.generic {indexing_maps = [#v1, #m2, #r2], iterator_types = ["parallel", "reduction"]}
        ins(%x, %wq : tensor<8xi8>, tensor<8x8xi8>) outs(%z : tensor<8xi32>) {
    ^bb0(%i: i8, %j: i8, %acc: i32):
      %1 = arith.extsi %i : i8 to i32
      %2 = arith.extsi %j : i8 to i32
      %3 = arith.muli %1, %2 : i32
      %4 = arith.addi %3, %acc : i32
      linalg.yield %4 : i32
    } -> tensor<8xi32>
    flow.return %0 : tensor<8xi32>
  }
  // shared with the outside (returned): stays outside
  %rope = flow.dispatch.region -> (tensor<2x4xf32>) {
    %e = tensor.empty() : tensor<2x4xf32>
    %0 = linalg.generic {indexing_maps = [#s2, #m2], iterator_types = ["parallel", "parallel"]}
        ins(%pos : tensor<i64>) outs(%e : tensor<2x4xf32>) {
    ^bb0(%p: i64, %o: f32):
      %1 = arith.sitofp %p : i64 to f32
      linalg.yield %1 : f32
    } -> tensor<2x4xf32>
    flow.return %0 : tensor<2x4xf32>
  }
  %qe = tensor.expand_shape %qp [[0, 1]] output_shape [2, 4] : tensor<8xi32> into tensor<2x4xi32>
  // q path: dequantize with RoPE, amax per head, int8
  %qf = flow.dispatch.region -> (tensor<2x4xf32>) {
    %e = tensor.empty() : tensor<2x4xf32>
    %0 = linalg.generic {indexing_maps = [#m2, #m2, #m2], iterator_types = ["parallel", "parallel"]}
        ins(%qe, %rope : tensor<2x4xi32>, tensor<2x4xf32>) outs(%e : tensor<2x4xf32>) {
    ^bb0(%i: i32, %r: f32, %o: f32):
      %1 = arith.sitofp %i : i32 to f32
      %2 = arith.mulf %1, %r : f32
      linalg.yield %2 : f32
    } -> tensor<2x4xf32>
    flow.return %0 : tensor<2x4xf32>
  }
  %sq = flow.dispatch.region -> (tensor<2xf32>) {
    %ninf = arith.constant 0xFF800000 : f32
    %e = tensor.empty() : tensor<2xf32>
    %z = linalg.fill ins(%ninf : f32) outs(%e : tensor<2xf32>) -> tensor<2xf32>
    %0 = linalg.generic {indexing_maps = [#m2, #r2], iterator_types = ["parallel", "reduction"]}
        ins(%qf : tensor<2x4xf32>) outs(%z : tensor<2xf32>) {
    ^bb0(%i: f32, %acc: f32):
      %1 = math.absf %i : f32
      %2 = arith.maximumf %1, %acc : f32
      linalg.yield %2 : f32
    } -> tensor<2xf32>
    flow.return %0 : tensor<2xf32>
  }
  %q8 = flow.dispatch.region -> (tensor<2x4xi32>) {
    %e = tensor.empty() : tensor<2x4xi32>
    %0 = linalg.generic {indexing_maps = [#m2, #r2, #m2], iterator_types = ["parallel", "parallel"]}
        ins(%qf, %sq : tensor<2x4xf32>, tensor<2xf32>) outs(%e : tensor<2x4xi32>) {
    ^bb0(%i: f32, %s: f32, %o: i32):
      %1 = arith.divf %i, %s : f32
      %2 = arith.fptosi %1 : f32 to i32
      linalg.yield %2 : i32
    } -> tensor<2x4xi32>
    flow.return %0 : tensor<2x4xi32>
  }
  // the scores over the K cache slice [:T]
  %sc = flow.dispatch.region -> (tensor<2x?xf32>{%T}) {
    %c0 = arith.constant 0 : i32
    %k = tensor.extract_slice %kc[0, 0, 0] [2, %T, 4] [1, 1, 1] : tensor<2x8x4xi8> to tensor<2x?x4xi8>
    %e = tensor.empty(%T) : tensor<2x?xi32>
    %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<2x?xi32>) -> tensor<2x?xi32>
    %0 = linalg.generic {indexing_maps = [#qa, #ka, #o3], iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%q8, %k : tensor<2x4xi32>, tensor<2x?x4xi8>) outs(%z : tensor<2x?xi32>) {
    ^bb0(%i: i32, %j: i8, %acc: i32):
      %1 = arith.extsi %j : i8 to i32
      %2 = arith.muli %i, %1 : i32
      %3 = arith.addi %2, %acc : i32
      linalg.yield %3 : i32
    } -> tensor<2x?xi32>
    %f = tensor.empty(%T) : tensor<2x?xf32>
    %1 = linalg.generic {indexing_maps = [#m2, #r2, #m2], iterator_types = ["parallel", "parallel"]}
        ins(%0, %sq : tensor<2x?xi32>, tensor<2xf32>) outs(%f : tensor<2x?xf32>) {
    ^bb0(%i: i32, %s: f32, %o: f32):
      %2 = arith.sitofp %i : i32 to f32
      %3 = arith.mulf %2, %s : f32
      linalg.yield %3 : f32
    } -> tensor<2x?xf32>
    flow.return %1 : tensor<2x?xf32>
  }
  // the probabilities: masked max, i8(exp(s - max) * 127)
  %p = flow.dispatch.region -> (tensor<2x?xi8>{%T}) {
    %ninf = arith.constant 0xFF800000 : f32
    %c127 = arith.constant 127.0 : f32
    %e = tensor.empty() : tensor<2xf32>
    %z = linalg.fill ins(%ninf : f32) outs(%e : tensor<2xf32>) -> tensor<2xf32>
    %m = linalg.generic {indexing_maps = [#s2, #m2, #r2], iterator_types = ["parallel", "reduction"]}
        ins(%pos, %sc : tensor<i64>, tensor<2x?xf32>) outs(%z : tensor<2xf32>) {
    ^bb0(%ps: i64, %i: f32, %acc: f32):
      %1 = linalg.index 1 : index
      %2 = arith.index_cast %1 : index to i64
      %3 = arith.cmpi sle, %2, %ps : i64
      %4 = arith.select %3, %i, %ninf : f32
      %5 = arith.maximumf %4, %acc : f32
      linalg.yield %5 : f32
    } -> tensor<2xf32>
    %f = tensor.empty(%T) : tensor<2x?xi8>
    %0 = linalg.generic {indexing_maps = [#m2, #r2, #m2], iterator_types = ["parallel", "parallel"]}
        ins(%sc, %m : tensor<2x?xf32>, tensor<2xf32>) outs(%f : tensor<2x?xi8>) {
    ^bb0(%i: f32, %mx: f32, %o: i8):
      %1 = arith.subf %i, %mx : f32
      %2 = math.exp %1 : f32
      %3 = arith.mulf %2, %c127 : f32
      %4 = arith.fptosi %3 : f32 to i8
      linalg.yield %4 : i8
    } -> tensor<2x?xi8>
    flow.return %0 : tensor<2x?xi8>
  }
  %ps = flow.dispatch.region -> (tensor<2xf32>) {
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<2xf32>
    %z = linalg.fill ins(%zero : f32) outs(%e : tensor<2xf32>) -> tensor<2xf32>
    %0 = linalg.generic {indexing_maps = [#m2, #r2], iterator_types = ["parallel", "reduction"]}
        ins(%p : tensor<2x?xi8>) outs(%z : tensor<2xf32>) {
    ^bb0(%i: i8, %acc: f32):
      %1 = arith.sitofp %i : i8 to f32
      %2 = arith.addf %1, %acc : f32
      linalg.yield %2 : f32
    } -> tensor<2xf32>
    flow.return %0 : tensor<2xf32>
  }
  // p . V over the V cache slice [:T], / sum
  %pv = flow.dispatch.region -> (tensor<2x4xf32>) {
    %c0 = arith.constant 0 : i32
    %v = tensor.extract_slice %vc[0, 0, 0] [2, %T, 4] [1, 1, 1] : tensor<2x8x4xi8> to tensor<2x?x4xi8>
    %e = tensor.empty() : tensor<2x4xi32>
    %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<2x4xi32>) -> tensor<2x4xi32>
    %0 = linalg.generic {indexing_maps = [#qa, #va, #o3], iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%p, %v : tensor<2x?xi8>, tensor<2x?x4xi8>) outs(%z : tensor<2x4xi32>) {
    ^bb0(%i: i8, %j: i8, %acc: i32):
      %1 = arith.extsi %i : i8 to i32
      %2 = arith.extsi %j : i8 to i32
      %3 = arith.muli %1, %2 : i32
      %4 = arith.addi %3, %acc : i32
      linalg.yield %4 : i32
    } -> tensor<2x4xi32>
    %f = tensor.empty() : tensor<2x4xf32>
    %1 = linalg.generic {indexing_maps = [#m2, #r2, #m2], iterator_types = ["parallel", "parallel"]}
        ins(%0, %ps : tensor<2x4xi32>, tensor<2xf32>) outs(%f : tensor<2x4xf32>) {
    ^bb0(%i: i32, %s: f32, %o: f32):
      %2 = arith.sitofp %i : i32 to f32
      %3 = arith.divf %2, %s : f32
      linalg.yield %3 : f32
    } -> tensor<2x4xf32>
    flow.return %1 : tensor<2x4xf32>
  }
  %pvc = tensor.collapse_shape %pv [[0, 1]] : tensor<2x4xf32> into tensor<8xf32>
  // amax and int8
  %am = flow.dispatch.region -> (tensor<f32>) {
    %ninf = arith.constant 0xFF800000 : f32
    %e = tensor.empty() : tensor<f32>
    %z = linalg.fill ins(%ninf : f32) outs(%e : tensor<f32>) -> tensor<f32>
    %0 = linalg.generic {indexing_maps = [#id, #sc], iterator_types = ["reduction"]}
        ins(%pvc : tensor<8xf32>) outs(%z : tensor<f32>) {
    ^bb0(%i: f32, %acc: f32):
      %1 = math.absf %i : f32
      %2 = arith.maximumf %1, %acc : f32
      linalg.yield %2 : f32
    } -> tensor<f32>
    flow.return %0 : tensor<f32>
  }
  %o8 = flow.dispatch.region -> (tensor<8xi8>) {
    %e = tensor.empty() : tensor<8xi8>
    %0 = linalg.generic {indexing_maps = [#id, #sc, #id], iterator_types = ["parallel"]}
        ins(%pvc, %am : tensor<8xf32>, tensor<f32>) outs(%e : tensor<8xi8>) {
    ^bb0(%i: f32, %s: f32, %o: i8):
      %1 = arith.divf %i, %s : f32
      %2 = arith.fptosi %1 : f32 to i8
      linalg.yield %2 : i8
    } -> tensor<8xi8>
    flow.return %0 : tensor<8xi8>
  }
  // o_proj: a static contraction, outside
  %op = flow.dispatch.region -> (tensor<8xi32>) {
    %c0 = arith.constant 0 : i32
    %e = tensor.empty() : tensor<8xi32>
    %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<8xi32>) -> tensor<8xi32>
    %0 = linalg.generic {indexing_maps = [#v1, #m2, #s2, #r2], iterator_types = ["parallel", "reduction"]}
        ins(%o8, %wo, %am : tensor<8xi8>, tensor<8x8xi8>, tensor<f32>) outs(%z : tensor<8xi32>) {
    ^bb0(%i: i8, %j: i8, %s: f32, %acc: i32):
      %1 = arith.extsi %i : i8 to i32
      %2 = arith.extsi %j : i8 to i32
      %3 = arith.muli %1, %2 : i32
      %4 = arith.addi %3, %acc : i32
      linalg.yield %4 : i32
    } -> tensor<8xi32>
    flow.return %0 : tensor<8xi32>
  }
  return %op, %rope : tensor<8xi32>, tensor<2x4xf32>
}

// -----

// A near miss: scores over a cache slice, but no p . V: left alone.
#qa = affine_map<(d0, d1, d2) -> (d0, d2)>
#ka = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#o3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#m2 = affine_map<(d0, d1) -> (d0, d1)>
#r2 = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func.func @no_pv
// CHECK-COUNT-2: flow.dispatch.region
func.func @no_pv(%q8: tensor<2x4xi32>, %kc: tensor<2x8x4xi8>, %T: index) -> tensor<2xi32> {
  %sc = flow.dispatch.region -> (tensor<2x?xi32>{%T}) {
    %c0 = arith.constant 0 : i32
    %k = tensor.extract_slice %kc[0, 0, 0] [2, %T, 4] [1, 1, 1] : tensor<2x8x4xi8> to tensor<2x?x4xi8>
    %e = tensor.empty(%T) : tensor<2x?xi32>
    %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<2x?xi32>) -> tensor<2x?xi32>
    %0 = linalg.generic {indexing_maps = [#qa, #ka, #o3], iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%q8, %k : tensor<2x4xi32>, tensor<2x?x4xi8>) outs(%z : tensor<2x?xi32>) {
    ^bb0(%i: i32, %j: i8, %acc: i32):
      %1 = arith.extsi %j : i8 to i32
      %2 = arith.muli %i, %1 : i32
      %3 = arith.addi %2, %acc : i32
      linalg.yield %3 : i32
    } -> tensor<2x?xi32>
    flow.return %0 : tensor<2x?xi32>
  }
  %m = flow.dispatch.region -> (tensor<2xi32>) {
    %c0 = arith.constant 0 : i32
    %e = tensor.empty() : tensor<2xi32>
    %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<2xi32>) -> tensor<2xi32>
    %0 = linalg.generic {indexing_maps = [#m2, #r2], iterator_types = ["parallel", "reduction"]}
        ins(%sc : tensor<2x?xi32>) outs(%z : tensor<2xi32>) {
    ^bb0(%i: i32, %acc: i32):
      %1 = arith.maxsi %i, %acc : i32
      linalg.yield %1 : i32
    } -> tensor<2xi32>
    flow.return %0 : tensor<2xi32>
  }
  return %m : tensor<2xi32>
}

// M5: W8 router projection, initialized descending Top-2, and raw outputs.
#x = affine_map<(d0, d1, d2) -> (d2)>
#w = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#o = affine_map<(d0, d1, d2) -> (d0, d1)>
#id = affine_map<(d0, d1) -> (d0, d1)>
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

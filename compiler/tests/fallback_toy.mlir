// tanh (the sa backend has no tanh: the host) then exp (the accelerator's SFU)
func.func @main(%x: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %t = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%a: f32, %o: f32):
    %h = math.tanh %a : f32
    linalg.yield %h : f32
  } -> tensor<64xf32>
  %b = util.optimization_barrier %t : tensor<64xf32>
  %e2 = tensor.empty() : tensor<64xf32>
  %y = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%b : tensor<64xf32>) outs(%e2 : tensor<64xf32>) {
  ^bb0(%a: f32, %o: f32):
    %s = math.exp %a : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  return %y : tensor<64xf32>
}

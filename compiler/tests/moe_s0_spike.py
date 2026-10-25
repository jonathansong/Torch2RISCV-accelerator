#!/usr/bin/env python3
"""S0 spike of docs/moe_native_lowering_design.md: one expert's linear layer over
a packed [E, N/D, K, D] table, the expert selected by an id read on the device.

Writes, for one shape, the MLIR variants and a parameter archive:
  <out>/static.mlir     expert 1 at a constant offset (the reference: compiles on the SA)
  <out>/dyn.mlir        the id extracted inside the dispatch region
  <out>/sliceout.mlir   the id and the slices outside the region (IREE clones them in)
  <out>/noregion.mlir   no explicit region (IREE forms the dispatch)
  <out>/params.irpa     "w" int8 [E, N, K], "s" f32 [E, N] (scope "model")

Compile each with compile_sa.sh's flags plus --iree-sa-codegen-report and
--iree-parameter-import=model=<out>/params.irpa (§11 of the design lists the
results).
"""
import argparse
from pathlib import Path

import numpy as np


def module(variant, E, N, K, D):
    T = N // D
    idx = ("%e64 = tensor.extract %ids[%c0] : tensor<2xi64>\n"
           "    %e = arith.index_cast %e64 : i64 to index") if variant != "static" else \
          "%e = arith.constant 1 : index"
    slices = (f"%w = tensor.extract_slice %wp[%e, 0, 0, 0] [1, {T}, {K}, {D}] [1, 1, 1, 1] : "
              f"tensor<{E}x{T}x{K}x{D}xi8> to tensor<{T}x{K}x{D}xi8>\n"
              f"    %s = tensor.extract_slice %ws[%e, 0] [1, {N}] [1, 1] : tensor<{E}x{N}xf32> to tensor<{N}xf32>")
    body = f"""%a0 = tensor.empty() : tensor<{T}x{D}xi32>
    %a1 = linalg.fill ins(%c0_i32 : i32) outs(%a0 : tensor<{T}x{D}xi32>) -> tensor<{T}x{D}xi32>
    %acc = linalg.generic {{indexing_maps = [affine_map<(t, j, k) -> (k)>, affine_map<(t, j, k) -> (t, k, j)>, affine_map<(t, j, k) -> (t, j)>], iterator_types = ["parallel", "parallel", "reduction"]}} ins(%x, %w : tensor<{K}xi8>, tensor<{T}x{K}x{D}xi8>) outs(%a1 : tensor<{T}x{D}xi32>) {{
    ^bb0(%i: i8, %j: i8, %o: i32):
      %1 = arith.extsi %i : i8 to i32
      %2 = arith.extsi %j : i8 to i32
      %3 = arith.muli %1, %2 : i32
      %4 = arith.addi %o, %3 : i32
      linalg.yield %4 : i32
    }} -> tensor<{T}x{D}xi32>
    %se = tensor.expand_shape %s [[0, 1]] output_shape [{T}, {D}] : tensor<{N}xf32> into tensor<{T}x{D}xf32>
    %y0 = tensor.empty() : tensor<{T}x{D}xf32>
    %y = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}} ins(%acc, %se, %sx : tensor<{T}x{D}xi32>, tensor<{T}x{D}xf32>, tensor<f32>) outs(%y0 : tensor<{T}x{D}xf32>) {{
    ^bb0(%i: i32, %sc: f32, %q: f32, %o: f32):
      %1 = arith.sitofp %i : i32 to f32
      %2 = arith.mulf %1, %sc : f32
      %3 = arith.mulf %2, %q : f32
      linalg.yield %3 : f32
    }} -> tensor<{T}x{D}xf32>"""
    if variant in ("static", "dyn"):
        region = f"""%r = flow.dispatch.region -> (tensor<{T}x{D}xf32>) {{
    {idx}
    {slices}
    {body}
    flow.return %y : tensor<{T}x{D}xf32>
    }}"""
    elif variant == "sliceout":
        region = f"""{idx}
    {slices}
    %r = flow.dispatch.region -> (tensor<{T}x{D}xf32>) {{
    {body}
    flow.return %y : tensor<{T}x{D}xf32>
    }}"""
    else:
        region = f"""{idx}
    {slices}
    {body}
    %r = util.optimization_barrier %y : tensor<{T}x{D}xf32>"""
    return f"""module {{
  util.global private @w = #flow.parameter.named<"model"::"w"> : tensor<{E}x{N}x{K}xi8>
  util.global private @s = #flow.parameter.named<"model"::"s"> : tensor<{E}x{N}xf32>
  util.global private @w$packed : tensor<{E}x{T}x{K}x{D}xi8>
  util.initializer {{
    %w = util.global.load @w : tensor<{E}x{N}x{K}xi8>
    %0 = tensor.empty() : tensor<{E}x{T}x{K}x{D}xi8>
    %p = linalg.pack %w inner_dims_pos = [1] inner_tiles = [{D}] into %0 : tensor<{E}x{N}x{K}xi8> -> tensor<{E}x{T}x{K}x{D}xi8>
    util.global.store %p, @w$packed : tensor<{E}x{T}x{K}x{D}xi8>
    util.return
  }}
  util.func public @main(%x: tensor<{K}xi8>, %sx: tensor<f32>, %ids: tensor<2xi64>) -> tensor<{N}xf32> {{
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %wp = util.global.load @w$packed : tensor<{E}x{T}x{K}x{D}xi8>
    %ws = util.global.load @s : tensor<{E}x{N}xf32>
    {region}
    %rc = tensor.collapse_shape %r [[0, 1]] : tensor<{T}x{D}xf32> into tensor<{N}xf32>
    util.return %rc : tensor<{N}xf32>
  }}
}}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--n", type=int, default=64, help="output features (gate/up: I, down: H)")
    ap.add_argument("--k", type=int, default=32, help="input features")
    ap.add_argument("--d", type=int, default=16)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for v in ("static", "dyn", "sliceout", "noregion"):
        (a.out / f"{v}.mlir").write_text(module(v, a.experts, a.n, a.k, a.d))
    import iree.runtime as rt
    r = np.random.default_rng(0)
    idx = rt.ParameterIndex()
    idx.add_buffer("w", r.integers(-127, 128, (a.experts, a.n, a.k), dtype=np.int8).tobytes())
    idx.add_buffer("s", (r.random((a.experts, a.n), dtype=np.float32) * 0.01 + 0.001).tobytes())
    idx.create_archive_file(str(a.out / "params.irpa"))
    print(f"{a.out}: 4 variants, E={a.experts} N={a.n} K={a.k} D={a.d}")


if __name__ == "__main__":
    main()

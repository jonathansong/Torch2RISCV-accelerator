// Descriptor templates of the sa backend (docs/iree_compiler_plan.md §6.4).
// Each template is the C++ port of a hand-written generator
// (llm/compile_layer.py) under the sa-desc-v1 calling convention; the
// Python reference is compiler/plugins/sa/templates/reference.py, and the
// output must equal it byte for byte.
#ifndef SA_TARGET_TEMPLATES_H_
#define SA_TARGET_TEMPLATES_H_

#include <cstdint>
#include <string>

#include "DescList.h"

namespace sa {

// Local memory plan for array size d (compile_layer.Layout).
struct Layout {
  explicit Layout(uint32_t d);
  // Output tiles per chunk (Layout.chunk_tiles); 0 if one tile does not fit.
  uint32_t chunkTiles(uint32_t k, uint32_t nt, uint32_t rows = 1) const;

  uint32_t d, sbank, cbank, scr, acc0, acc1;
};

// The int8 linear dispatch (reference.qlinear):
//   y[n] = float(sum_k x[k] * W[n, k]) * s_w[n] * s_x
// with the weights packed as export_w8a8.pack_b (i8[n / D, k, D]).
struct QLinear {
  uint32_t k = 0, n = 0;
  bool xI32 = false;          // x is i32 (int8 values) instead of i8
  int bX = 0, bW = 1, bSw = 2, bSx = 3, bY = 4;  // binding of each operand
};

// Builds the template into |dl| (ending with RET). Returns an empty string on
// success, else why the shape is not supported.
std::string buildQLinear(uint32_t d, const QLinear &q, DescList &dl);

}  // namespace sa

#endif  // SA_TARGET_TEMPLATES_H_

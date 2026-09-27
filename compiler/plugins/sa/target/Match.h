// Dispatch matching of the sa backend (docs/iree_compiler_plan.md §6.4):
// recognizes the structure of a dispatch function and extracts the
// parameters of the template that implements it.
#ifndef SA_TARGET_MATCH_H_
#define SA_TARGET_MATCH_H_

#include <string>

#include "Templates.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler::sa {

// The int8 linear dispatch after PackLinearWeights:
//   x   = load binding bX  (i32[k], or i8[k] followed by extsi to i32)
//   Wp  = load binding bW  (i8[k / D ... ] = i8[n/D, k, D])
//   acc = generic (t, j, k) acc[t, j] += extsi(x[k]) * extsi(Wp[t, k, j])   (acc = fill 0)
//   y   = generic (t, j): float(trunci(acc[t, j])) * s_w[t, j] * s_x       (in this order)
//   store y -> binding bY
// Returns false with |why| set if |func| is not of this form.
bool matchQLinear(FunctionOpInterface func, int64_t d, ::sa::QLinear &out, std::string &why);

}  // namespace mlir::iree_compiler::sa

#endif  // SA_TARGET_MATCH_H_

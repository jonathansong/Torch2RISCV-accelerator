// Values derived from push constants (docs/iree_compiler_plan.md §6.8), shared
// by the C3/C4 code generator and the C5 passes.
#ifndef SA_TRANSFORMS_SALIN_H_
#define SA_TRANSFORMS_SALIN_H_

#include <cstdint>
#include <optional>
#include <tuple>

#include "mlir/IR/Value.h"

namespace mlir::iree_compiler::sa {

// ((C * mul) >> shift) + add, C = push constant `ord` (or none: add only).
struct Lin {
  int ord = -1;
  int64_t mul = 1;
  int shift = 0;
  int64_t add = 0;
  bool isConst() const { return ord < 0; }
  bool operator<(const Lin &o) const {
    return std::tie(ord, mul, shift, add) < std::tie(o.ord, o.mul, o.shift, o.add);
  }
  bool operator==(const Lin &o) const { return !(*this < o) && !(o < *this); }
};

// The Lin of an index / integer value (push constants through casts,
// util.assume.int, 64-bit lo | hi << 32, * and + by constants).
std::optional<Lin> linOf(Value v);

}  // namespace mlir::iree_compiler::sa

#endif  // SA_TRANSFORMS_SALIN_H_

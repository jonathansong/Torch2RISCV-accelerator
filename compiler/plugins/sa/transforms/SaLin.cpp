// Lin analysis (SaLin.h).
#include "SaLin.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir::iree_compiler::sa {

std::optional<Lin> linOf(Value v) {
  if (auto c = getConstantIntValue(v)) return Lin{-1, 1, 0, *c};
  Operation *op = v.getDefiningOp();
  if (!op) return std::nullopt;
  if (auto load = dyn_cast<IREE::HAL::InterfaceConstantLoadOp>(op))
    return Lin{int(load.getOrdinal().getZExtValue()), 1, 0, 0};
  if (isa<arith::IndexCastUIOp, arith::IndexCastOp, arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp>(op))
    return linOf(op->getOperand(0));
  if (auto a = dyn_cast<IREE::Util::AssumeIntOp>(op)) return linOf(a.getOperand(cast<OpResult>(v).getResultNumber()));
  if (isa<IREE::TensorExt::DispatchWorkloadOrdinalOp>(op)) return linOf(op->getOperand(0));
  if (auto o = dyn_cast<arith::OrIOp>(op)) {
    // lo | (hi << 32): a 64-bit value from two push constants; the device is
    // 32-bit, the high word is 0 (the runtime values here are offsets / lengths)
    for (int i = 0; i < 2; ++i) {
      if (auto sh = op->getOperand(1 - i).getDefiningOp<arith::ShLIOp>()) {
        auto amount = getConstantIntValue(sh.getRhs());
        if (amount && *amount == 32) return linOf(op->getOperand(i));
      }
    }
    return std::nullopt;
  }
  if (auto m = dyn_cast<arith::MulIOp>(op)) {
    for (int i = 0; i < 2; ++i)
      if (auto c = getConstantIntValue(op->getOperand(1 - i)))
        if (auto l = linOf(op->getOperand(i)); l && l->shift == 0) return Lin{l->ord, l->mul * *c, 0, l->add * *c};
    return std::nullopt;
  }
  if (isa<arith::AddIOp>(op)) {
    for (int i = 0; i < 2; ++i)
      if (auto c = getConstantIntValue(op->getOperand(1 - i)))
        if (auto l = linOf(op->getOperand(i))) return Lin{l->ord, l->mul, l->shift, l->add + *c};
    return std::nullopt;
  }
  return std::nullopt;
}


}  // namespace mlir::iree_compiler::sa

// sa translation: a dispatch is one descriptor template, run once (the sa
// HAL driver issues it at workgroup (0, 0, 0)), so each export's workgroup
// count region returns (1, 1, 1) instead of IREE's slice-derived count.
#include "SAPasses.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler::sa {
namespace {

struct LowerWorkgroupCountPass
    : public PassWrapper<LowerWorkgroupCountPass, OperationPass<IREE::HAL::ExecutableVariantOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerWorkgroupCountPass)

  StringRef getArgument() const override { return "iree-sa-lower-workgroup-count"; }
  StringRef getDescription() const override { return "Sets every sa export's workgroup count to (1, 1, 1)"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    for (auto exportOp : getOperation().getOps<IREE::HAL::ExecutableExportOp>()) {
      Block *body = exportOp.getWorkgroupCountBody();
      if (!body) continue;
      Operation *ret = body->getTerminator();
      OpBuilder b(ret);
      Value one = arith::ConstantIndexOp::create(b, ret->getLoc(), 1);
      SmallVector<Value> ones(ret->getNumOperands(), one);
      ret->setOperands(ones);
      // drop what computed the old count
      SmallVector<Operation *> dead;
      for (Operation &op : llvm::reverse(body->getOperations()))
        if (&op != ret && &op != one.getDefiningOp()) dead.push_back(&op);
      for (Operation *op : dead)
        if (op->use_empty()) op->erase();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createLowerWorkgroupCountPass() { return std::make_unique<LowerWorkgroupCountPass>(); }

void registerLowerWorkgroupCountPass() { PassRegistration<LowerWorkgroupCountPass>(); }

}  // namespace mlir::iree_compiler::sa

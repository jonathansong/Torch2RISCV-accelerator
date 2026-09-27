// sa preprocessing: clone cheap element-wise producers into their consumers
// (docs/iree_compiler_plan.md §6.8).
//
// A value computed from indices and scalars only, e.g. the attention mask
//     iota = generic { linalg.index 0 }                  : tensor<Txi64>
//     mask = generic ins(iota, pos) { cmpi sle }          : tensor<Txi1>
// is used by several consumers (the masked max and the masked exp). IREE
// then materializes it in a dispatch of its own, and each consumer only sees
// a loaded tensor: the prefix structure (index <= pos) is lost, and the
// accelerator cannot compute a lane iota. Recomputing such values inside
// every consumer keeps the structure visible to the sa backend, which maps
// "index <= scalar" to the VE's VALID count.
//
// Cheap producer: a linalg.generic with only parallel loops and identity
// output maps whose inputs are all rank-0 tensors (none is fine).
// Runs only when the module targets the sa device.
#include "SAPasses.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::sa {
namespace {

bool isCheapProducer(linalg::GenericOp op) {
  if (!op.isAllParallelLoops() || op.getNumDpsInits() != 1 || op->getNumResults() != 1) return false;
  for (OpOperand *in : op.getDpsInputOperands()) {
    auto t = dyn_cast<RankedTensorType>(in->get().getType());
    if (!t || t.getRank() != 0) return false;
  }
  return op.getMatchingIndexingMap(op.getDpsInitOperand(0)).isIdentity();
}

bool targetsSADevice(ModuleOp module) {
  bool found = false;
  module.walk([&](Operation *op) {
    for (NamedAttribute a : op->getAttrs())
      a.getValue().walk([&](IREE::HAL::DeviceTargetAttr t) {
        if (t.getDeviceID().getValue() == "sa") found = true;
      });
    return found ? WalkResult::interrupt() : WalkResult::advance();
  });
  return found;
}

struct CloneCheapProducersPass : public PassWrapper<CloneCheapProducersPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CloneCheapProducersPass)

  CloneCheapProducersPass() = default;
  CloneCheapProducersPass(const CloneCheapProducersPass &other) : PassWrapper(other) {}

  StringRef getArgument() const override { return "iree-sa-clone-cheap-producers"; }
  StringRef getDescription() const override {
    return "Recomputes index / scalar-only element-wise values inside each consumer";
  }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<linalg::LinalgDialect>(); }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!force && !targetsSADevice(module)) return;
    // MLIR's element-wise fusion, restricted to cheap producers: a producer
    // with several users is fused into each of them (and dropped when unused)
    RewritePatternSet patterns(&getContext());
    linalg::populateElementwiseOpsFusionPatterns(patterns, [](OpOperand *use) {
      auto producer = use->get().getDefiningOp<linalg::GenericOp>();
      return producer && isCheapProducer(producer);
    });
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) signalPassFailure();
  }

  Option<bool> force{*this, "force", llvm::cl::desc("Run without an sa device target (tests)"),
                     llvm::cl::init(false)};
};

}  // namespace

std::unique_ptr<Pass> createCloneCheapProducersPass() { return std::make_unique<CloneCheapProducersPass>(); }

void registerCloneCheapProducersPass() { PassRegistration<CloneCheapProducersPass>(); }

}  // namespace mlir::iree_compiler::sa

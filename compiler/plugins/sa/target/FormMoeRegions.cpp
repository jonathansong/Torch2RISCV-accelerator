// M5: recognize a batch-1 top-2 router projection before dispatch creation.
//
// This pass deliberately forms only the router/top-k region. Expert lookup,
// normalization, and FFN lowering are later M5 steps; preserving the original
// topk operation keeps this first step semantics-neutral. A region is formed
// only for a statically shaped 4..16-expert router and descending f32 topk.
#include "SAPasses.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::sa {
namespace {

std::optional<int64_t> routerExperts(IREE::LinalgExt::TopkOp topk) {
  if (topk.getIndices()) return std::nullopt;
  auto scores = dyn_cast<RankedTensorType>(topk.getValues().getType());
  auto values = dyn_cast<RankedTensorType>(topk.getOutputValues().getType());
  auto indices = dyn_cast<RankedTensorType>(topk.getOutputIndices().getType());
  if (!scores || !values || !indices || !scores.hasStaticShape() ||
      !values.hasStaticShape() || !indices.hasStaticShape() ||
      !scores.getElementType().isF32() || !values.getElementType().isF32() ||
      !indices.getElementType().isInteger(32) || scores.getRank() != 2 ||
      values.getRank() != 2 || indices.getRank() != 2 ||
      scores.getDimSize(0) != 1 || values.getDimSize(0) != 1 ||
      indices.getDimSize(0) != 1 || topk.getDimension() != 1 ||
      values.getDimSize(1) != 2 || indices.getDimSize(1) != 2)
    return std::nullopt;
  int64_t experts = scores.getDimSize(1);
  if (experts < 4 || experts > 16) return std::nullopt;
  // This must be the router projection, rather than an unrelated top-k.
  auto projection = dyn_cast_or_null<linalg::LinalgOp>(topk.getValues().getDefiningOp());
  if (!projection || projection.getNumReductionLoops() != 1) return std::nullopt;
  Block &comparator = topk.getRegion().front();
  if (comparator.getOperations().size() != 2) return std::nullopt;
  auto cmp = dyn_cast<arith::CmpFOp>(comparator.front());
  if (!cmp || cmp.getPredicate() != arith::CmpFPredicate::OGT ||
      cmp.getLhs() != comparator.getArgument(0) ||
      cmp.getRhs() != comparator.getArgument(1))
    return std::nullopt;
  return experts;
}

struct FormMoeRegionsPass : public PassWrapper<FormMoeRegionsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormMoeRegionsPass)

  FormMoeRegionsPass() = default;
  explicit FormMoeRegionsPass(bool enabled) { this->enabled = enabled; }
  FormMoeRegionsPass(const FormMoeRegionsPass &other) : PassWrapper(other) {}

  StringRef getArgument() const override { return "iree-sa-form-moe-regions"; }
  StringRef getDescription() const override {
    return "Recognizes a batch-1 top-2 MoE router and forms its dispatch region";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override {
    if (!enabled || (!force && !targetsSA(getOperation()))) return;
    SmallVector<std::pair<IREE::LinalgExt::TopkOp, int64_t>> matches;
    getOperation()->walk([&](IREE::LinalgExt::TopkOp topk) {
      if (auto experts = routerExperts(topk)) matches.emplace_back(topk, *experts);
    });
    IRRewriter rewriter(&getContext());
    for (auto [topk, experts] : matches) {
      rewriter.setInsertionPoint(topk);
      auto region = IREE::Flow::wrapOpInDispatchRegion(rewriter, topk);
      if (failed(region)) return signalPassFailure();
      (*region)->setAttr("sa.moe.experts", rewriter.getI64IntegerAttr(experts));
      (*region)->setAttr("sa.moe.top_k", rewriter.getI64IntegerAttr(2));
      (*region)->setAttr("sa.moe.kind", rewriter.getStringAttr("route_top2"));
    }
    if (report)
      llvm::errs() << "sa form-moe-regions: " << matches.size() << " top-2 router regions\n";
  }

  Option<bool> enabled{*this, "enabled", llvm::cl::desc("Form MoE router regions"), llvm::cl::init(true)};
  Option<bool> force{*this, "force", llvm::cl::desc("Run without an sa device target (tests)"), llvm::cl::init(false)};
  Option<bool> report{*this, "report", llvm::cl::desc("Print the recognized region count"), llvm::cl::init(false)};
};

} // namespace

std::unique_ptr<Pass> createFormMoeRegionsPass(bool enabled) { return std::make_unique<FormMoeRegionsPass>(enabled); }
void registerFormMoeRegionsPass() { PassRegistration<FormMoeRegionsPass>(); }

} // namespace mlir::iree_compiler::sa

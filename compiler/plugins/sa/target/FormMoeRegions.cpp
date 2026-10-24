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
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::sa {
namespace {

bool comesFromProjection(Value value, llvm::DenseSet<Operation *> &seen, unsigned depth = 0) {
  Operation *op = value.getDefiningOp();
  if (!op || depth > 8 || !seen.insert(op).second) return false;
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    if (linalgOp.getNumReductionLoops() == 1 && linalgOp.getNumDpsInputs() >= 2) return true;
    if (linalgOp.getNumReductionLoops() != 0) return false;
    for (Value input : linalgOp.getDpsInputs())
      if (comesFromProjection(input, seen, depth + 1)) return true;
    return false;
  }
  if (!isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::CastOp>(op)) return false;
  return comesFromProjection(op->getOperand(0), seen, depth + 1);
}

bool hasTopkInitialization(IREE::LinalgExt::TopkOp topk) {
  auto valuesFill = topk.getOutputValues().getDefiningOp<linalg::FillOp>();
  auto indicesFill = topk.getOutputIndices().getDefiningOp<linalg::FillOp>();
  if (!valuesFill || !indicesFill) return false;
  auto valuesConstant = valuesFill.getDpsInputs()[0].getDefiningOp<arith::ConstantOp>();
  auto indicesConstant = indicesFill.getDpsInputs()[0].getDefiningOp<arith::ConstantOp>();
  if (!valuesConstant || !indicesConstant) return false;
  auto value = dyn_cast<FloatAttr>(valuesConstant.getValue());
  auto index = dyn_cast<IntegerAttr>(indicesConstant.getValue());
  return value && value.getValue().isInfinity() && value.getValue().isNegative() &&
         index && index.getValue().getSExtValue() == 0x7fffffff;
}

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
  // Top-K's scalar implementation reads both output buffers before inserting
  // each candidate. Only the descending sentinel initialization is safe to
  // replace with a fresh Top-2 selection.
  if (!hasTopkInitialization(topk)) return std::nullopt;
  // This must be the router projection, rather than an unrelated top-k.
  llvm::DenseSet<Operation *> seen;
  if (!comesFromProjection(topk.getValues(), seen)) return std::nullopt;
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
      topk->setAttr("sa.moe.experts", rewriter.getI64IntegerAttr(experts));
      topk->setAttr("sa.moe.top_k", rewriter.getI64IntegerAttr(2));
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

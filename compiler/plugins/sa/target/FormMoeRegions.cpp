// sa preprocessing: an MoE router's Top-k in a dispatch region of its own
// (docs/moe_native_lowering_design.md §5.2, §23). A Torch sort's k-element
// prefix becomes an iree_linalg_ext.topk (i32 ids; the ids read as indices, the
// expert slices' offsets, read the i32 ids: readIdsAsI32), and a descending
// f32 Top-k (k <= kMaxK) of a statically shaped router projection of
// kMinExperts..kMaxExperts experts is wrapped in a region; sa-to-sahl makes it
// sahl.topk, which runs as RISC-V code only.
// The experts' FFNs need no regions here: their slices of the packed tables
// (PackLinearWeights) stay in the consuming dispatches.
//
// iree-sa-isolate-moe-experts (after FormFfnRegions): with k > 2, IREE's
// elementwise fusion merges the k experts' down-projection epilogues (i32 to
// f32, the row scales selected by id, s_x, the routing weight) and their sum
// into one dispatch, three or more bindings per expert: beyond the 15 BASE
// registers at k = 8. A util.optimization_barrier on each expert's term (the
// end of its epilogue, before the sum) keeps the epilogue with its contraction
// (FormDispatchRegions fuses it there as a consumer) and leaves the sum one
// f32 vector per expert (docs/moe_native_lowering_design.md §23). Top-2 keeps
// the dispatches of before. The barriers carry sa.moe.isolate, and
// iree-sa-drop-moe-isolation removes them once the dispatch regions are
// formed: left to Stream, each split the command buffer (§24).
#include "SAPasses.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <cstdint>
#include <limits>

namespace mlir::iree_compiler::sa {
namespace {

constexpr int64_t kMinExperts = 4, kMaxExperts = 64, kMaxK = 16;

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

bool isIotaIndices(Value value) {
  auto generic = value.getDefiningOp<linalg::GenericOp>();
  if (!generic)
    if (auto expand = value.getDefiningOp<tensor::ExpandShapeOp>())
      generic = expand.getSrc().getDefiningOp<linalg::GenericOp>();
  if (!generic || generic.getNumDpsInputs() || generic.getNumDpsInits() != 1 ||
      generic.getNumLoops() != generic.getNumParallelLoops() ||
      (generic.getNumLoops() != 1 && generic.getNumLoops() != 2) ||
      !generic.getIndexingMapsArray().front().isIdentity())
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(generic.getRegion().front().getTerminator());
  auto cast = yield && yield.getValues().size() == 1
                  ? yield.getValues()[0].getDefiningOp<arith::IndexCastOp>() : arith::IndexCastOp();
  auto index = cast ? cast.getIn().getDefiningOp<linalg::IndexOp>() : linalg::IndexOp();
  return cast && cast.getType().isInteger(64) &&
         index && index.getDim() == generic.getNumLoops() - 1;
}

// The sorted tensor's only use, a prefix [1, k] (or [k]) slice; k from it
tensor::ExtractSliceOp topkPrefixSlice(Value sorted, int64_t &k) {
  if (!sorted.hasOneUse()) return {};
  auto slice = dyn_cast<tensor::ExtractSliceOp>(*sorted.user_begin());
  if (!slice || slice.getStaticOffsets() != ArrayRef<int64_t>({0, 0}) ||
      slice.getStaticSizes().size() != 2 || slice.getStaticSizes()[0] != 1 ||
      slice.getStaticStrides() != ArrayRef<int64_t>({1, 1}))
    return {};
  k = slice.getStaticSizes()[1];
  auto type = dyn_cast<RankedTensorType>(slice.getType());
  return type && ((type.getRank() == 1 && type.getDimSize(0) == k) ||
                  (type.getRank() == 2 && type.getShape() == ArrayRef<int64_t>({1, k})))
             ? slice : tensor::ExtractSliceOp();
}

// The view chain from |ids64| = extsi(|ids32|) to |v| (extract_slice,
// collapse_shape, expand_shape), rebuilt on the i32 ids; null if |v| is not
// such a view.
Value rebuildOnI32(OpBuilder &b, Value v, Value ids64, Value ids32, unsigned depth = 0) {
  if (v == ids64) return ids32;
  Operation *def = v.getDefiningOp();
  if (!def || depth > 8) return {};
  auto as32 = [&](Type t) { return cast<RankedTensorType>(t).clone(b.getI32Type()); };
  if (auto sl = dyn_cast<tensor::ExtractSliceOp>(def)) {
    Value src = rebuildOnI32(b, sl.getSource(), ids64, ids32, depth + 1);
    if (!src) return {};
    b.setInsertionPointAfter(sl);
    return tensor::ExtractSliceOp::create(b, sl.getLoc(), as32(sl.getType()), src, sl.getMixedOffsets(),
                                          sl.getMixedSizes(), sl.getMixedStrides());
  }
  if (auto cs = dyn_cast<tensor::CollapseShapeOp>(def)) {
    Value src = rebuildOnI32(b, cs.getSrc(), ids64, ids32, depth + 1);
    if (!src) return {};
    b.setInsertionPointAfter(cs);
    return tensor::CollapseShapeOp::create(b, cs.getLoc(), as32(cs.getType()), src, cs.getReassociationIndices());
  }
  if (auto es = dyn_cast<tensor::ExpandShapeOp>(def)) {
    Value src = rebuildOnI32(b, es.getSrc(), ids64, ids32, depth + 1);
    if (!src || !cast<RankedTensorType>(es.getType()).hasStaticShape()) return {};
    b.setInsertionPointAfter(es);
    return tensor::ExpandShapeOp::create(b, es.getLoc(), as32(es.getType()), src, es.getReassociationIndices());
  }
  return {};
}

// The expert ids are extended back to the sort's i64 for the graph after the
// Top-k. An id read as an index (tensor.extract + index_cast: the expert
// slices, docs/moe_native_lowering_design.md §5.1) reads the i32 id instead,
// index_cast(extsi(x)) = index_cast(x): no dispatch of its own for the
// extension. Returns the number of ids so read.
unsigned readIdsAsI32(IRRewriter &rewriter, arith::ExtSIOp ext) {
  Value ids64 = ext.getResult(), ids32 = ext.getIn();
  SmallVector<tensor::ExtractOp> reads;
  SmallVector<Value> work = {ids64};
  for (unsigned i = 0; i < work.size() && i < 64; ++i)
    for (Operation *u : work[i].getUsers()) {
      if (auto ex = dyn_cast<tensor::ExtractOp>(u)) reads.push_back(ex);
      else if (isa<tensor::ExtractSliceOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp>(u))
        work.push_back(u->getResult(0));
    }
  unsigned n = 0;
  for (tensor::ExtractOp ex : reads) {
    SmallVector<arith::IndexCastOp> casts;
    for (Operation *u : ex->getUsers())
      if (auto c = dyn_cast<arith::IndexCastOp>(u)) casts.push_back(c);
    if (casts.empty()) continue;
    OpBuilder b(ex);
    Value src = rebuildOnI32(b, ex.getTensor(), ids64, ids32);
    if (!src) continue;
    b.setInsertionPoint(ex);
    Value id = tensor::ExtractOp::create(b, ex.getLoc(), src, ex.getIndices());
    for (arith::IndexCastOp c : casts) {
      rewriter.setInsertionPoint(c);
      rewriter.replaceOpWithNewOp<arith::IndexCastOp>(c, c.getType(), id);
    }
    if (ex->use_empty()) rewriter.eraseOp(ex);
    ++n;
  }
  return n;
}

bool rewriteTorchTopkSort(IREE::LinalgExt::SortOp sort, IRRewriter &rewriter) {
  if (sort.getNumResults() != 2 || sort.getDimension() != 1) return false;
  auto scoresType = dyn_cast<RankedTensorType>(sort.getOutputs()[0].getType());
  auto idsType = dyn_cast<RankedTensorType>(sort.getOutputs()[1].getType());
  if (!scoresType || !idsType || !scoresType.hasStaticShape() || !idsType.hasStaticShape() ||
      scoresType.getRank() != 2 || idsType.getRank() != 2 ||
      scoresType.getDimSize(0) != 1 || idsType.getShape() != scoresType.getShape() ||
      !scoresType.getElementType().isF32() || !idsType.getElementType().isInteger(64))
    return false;
  int64_t experts = scoresType.getDimSize(1);
  if (experts < kMinExperts || experts > kMaxExperts || !isIotaIndices(sort.getOutputs()[1])) return false;
  llvm::DenseSet<Operation *> seen;
  if (!comesFromProjection(sort.getOutputs()[0], seen)) return false;
  int64_t k = 0, kIds = 0;
  auto valueSlice = topkPrefixSlice(sort.getResult(0), k);
  auto indexSlice = topkPrefixSlice(sort.getResult(1), kIds);
  if (!valueSlice || !indexSlice || k != kIds || k < 1 || k >= experts || k > kMaxK) return false;
  Block &comparator = sort.getRegion().front();
  if (comparator.getOperations().size() != 2 || comparator.getNumArguments() != 4) return false;
  auto cmp = dyn_cast<arith::CmpFOp>(comparator.front());
  if (!cmp || cmp.getPredicate() != arith::CmpFPredicate::OGE ||
      cmp.getLhs() != comparator.getArgument(0) || cmp.getRhs() != comparator.getArgument(1))
    return false;

  rewriter.setInsertionPoint(sort);
  auto loc = sort.getLoc();
  auto f32 = rewriter.getF32Type();
  auto i32 = rewriter.getI32Type();
  auto valuesType = RankedTensorType::get({1, k}, f32);
  auto indicesType = RankedTensorType::get({1, k}, i32);
  Value negInf = arith::ConstantFloatOp::create(rewriter, loc,
      f32, APFloat(-std::numeric_limits<float>::infinity()));
  Value maxIndex = arith::ConstantIntOp::create(rewriter, loc, INT32_MAX, 32);
  Value emptyValues = tensor::EmptyOp::create(rewriter, loc, valuesType.getShape(), f32);
  Value emptyIndices = tensor::EmptyOp::create(rewriter, loc, indicesType.getShape(), i32);
  Value initValues = linalg::FillOp::create(rewriter, loc, negInf, emptyValues).getResult(0);
  Value initIndices = linalg::FillOp::create(rewriter, loc, maxIndex, emptyIndices).getResult(0);
  auto topk = IREE::LinalgExt::TopkOp::create(rewriter, loc, TypeRange{valuesType, indicesType},
      sort.getOutputs()[0], Value(), initValues, initIndices, 1);
  Block *body = rewriter.createBlock(&topk.getRegion(), {}, {f32, f32}, {loc, loc});
  rewriter.setInsertionPointToEnd(body);
  Value greater = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGT,
                                        body->getArgument(0), body->getArgument(1));
  IREE::LinalgExt::YieldOp::create(rewriter, loc, greater);
  rewriter.setInsertionPointAfter(topk);
  SmallVector<ReassociationIndices> reassociation{{0, 1}};
  Value valuesK = topk.getResult(0);
  auto indices64Type = RankedTensorType::get({1, k}, rewriter.getI64Type());
  auto ext = arith::ExtSIOp::create(rewriter, loc, indices64Type, topk.getResult(1));
  Value idsK = ext;
  if (cast<RankedTensorType>(valueSlice.getType()).getRank() == 1)
    valuesK = tensor::CollapseShapeOp::create(rewriter, loc, valueSlice.getType(),
                                               valuesK, reassociation);
  if (cast<RankedTensorType>(indexSlice.getType()).getRank() == 1)
    idsK = tensor::CollapseShapeOp::create(rewriter, loc, indexSlice.getType(),
                                            idsK, reassociation);
  rewriter.replaceOp(valueSlice, valuesK);
  rewriter.replaceOp(indexSlice, idsK);
  rewriter.eraseOp(sort);
  readIdsAsI32(rewriter, ext);
  return true;
}

// (experts, k) of a router's Top-k
std::optional<std::pair<int64_t, int64_t>> routerExperts(IREE::LinalgExt::TopkOp topk) {
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
      values.getDimSize(1) != indices.getDimSize(1))
    return std::nullopt;
  int64_t experts = scores.getDimSize(1), k = values.getDimSize(1);
  if (experts < kMinExperts || experts > kMaxExperts || k < 1 || k >= experts || k > kMaxK) return std::nullopt;
  // Top-K's scalar implementation reads both output buffers before inserting
  // each candidate. Only the descending sentinel initialization is safe to
  // replace with a fresh Top-k selection.
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
  return std::make_pair(experts, k);
}

struct FormMoeRegionsPass : public PassWrapper<FormMoeRegionsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormMoeRegionsPass)

  FormMoeRegionsPass() = default;
  explicit FormMoeRegionsPass(bool enabled) { this->enabled = enabled; }
  FormMoeRegionsPass(const FormMoeRegionsPass &other) : PassWrapper(other) {}

  StringRef getArgument() const override { return "iree-sa-form-moe-regions"; }
  StringRef getDescription() const override {
    return "Recognizes a batch-1 top-k MoE router and forms its dispatch region";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override {
    if (!enabled || (!force && !targetsSA(getOperation()))) return;
    IRRewriter rewriter(&getContext());
    SmallVector<IREE::LinalgExt::SortOp> sorts;
    getOperation()->walk([&](IREE::LinalgExt::SortOp sort) { sorts.push_back(sort); });
    unsigned convertedSorts = 0;
    for (auto sort : sorts) convertedSorts += rewriteTorchTopkSort(sort, rewriter);
    SmallVector<std::tuple<IREE::LinalgExt::TopkOp, int64_t, int64_t>> matches;
    getOperation()->walk([&](IREE::LinalgExt::TopkOp topk) {
      if (auto ek = routerExperts(topk)) matches.emplace_back(topk, ek->first, ek->second);
    });
    for (auto [topk, experts, k] : matches) {
      topk->setAttr("sa.moe.experts", rewriter.getI64IntegerAttr(experts));
      topk->setAttr("sa.moe.top_k", rewriter.getI64IntegerAttr(k));
      rewriter.setInsertionPoint(topk);
      auto region = IREE::Flow::wrapOpInDispatchRegion(rewriter, topk);
      if (failed(region)) return signalPassFailure();
      (*region)->setAttr("sa.moe.experts", rewriter.getI64IntegerAttr(experts));
      (*region)->setAttr("sa.moe.top_k", rewriter.getI64IntegerAttr(k));
      (*region)->setAttr("sa.moe.kind", rewriter.getStringAttr("topk"));
    }
    if (report)
      llvm::errs() << "sa form-moe-regions: " << matches.size() << " top-k router regions ("
                   << convertedSorts << " Torch sort prefixes)\n";
  }

  Option<bool> enabled{*this, "enabled", llvm::cl::desc("Form MoE router regions"), llvm::cl::init(true)};
  Option<bool> force{*this, "force", llvm::cl::desc("Run without an sa device target (tests)"), llvm::cl::init(false)};
  Option<bool> report{*this, "report", llvm::cl::desc("Print the recognized region count"), llvm::cl::init(false)};
};

// int8 x int8 -> int32 with one reduction loop (FormFfnRegions' contractions)
bool isInt8Contraction(Operation *op) {
  auto g = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!g || g.getNumDpsInputs() != 2 || g.getNumDpsInits() != 1 || g.getNumReductionLoops() != 1) return false;
  auto et = [](Value v) { return cast<ShapedType>(v.getType()).getElementType(); };
  return et(g.getDpsInputs()[0]).isInteger(8) && et(g.getDpsInputs()[1]).isInteger(8) &&
         et(g.getDpsInits()[0]).isInteger(32);
}

// an expert's weights: a slice of a packed table at a dynamic offset (the id)
bool isExpertSlice(Value w) {
  auto sl = w.getDefiningOp<tensor::ExtractSliceOp>();
  return sl && !sl.getOffsets().empty();
}

// v depends on an int8 contraction outside the dispatch regions (bounded
// search through the producers in the block; regions and other blocks end it)
bool dependsOnContraction(Value v, Block *blk, llvm::DenseSet<Operation *> &seen, unsigned depth = 0) {
  Operation *d = v.getDefiningOp();
  if (!d || d->getBlock() != blk || depth > 24 || isa<IREE::Flow::DispatchRegionOp>(d) || !seen.insert(d).second)
    return false;
  if (isInt8Contraction(d)) return true;
  for (Value o : d->getOperands())
    if (dependsOnContraction(o, blk, seen, depth + 1)) return true;
  return false;
}

struct IsolateMoeExpertsPass : public PassWrapper<IsolateMoeExpertsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IsolateMoeExpertsPass)

  IsolateMoeExpertsPass() = default;
  IsolateMoeExpertsPass(const IsolateMoeExpertsPass &other) : PassWrapper(other) {}

  StringRef getArgument() const override { return "iree-sa-isolate-moe-experts"; }
  StringRef getDescription() const override {
    return "Keeps each MoE expert's down-projection epilogue in its own dispatch (Top-k, k > 2)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    if (!force && !targetsSA(getOperation())) return;
    unsigned n = 0;
    getOperation()->walk([&](Block *blk) {
      bool topkOver2 = llvm::any_of(*blk, [](Operation &op) {
        auto k = op.getAttrOfType<IntegerAttr>("sa.moe.top_k");
        return isa<IREE::Flow::DispatchRegionOp>(op) && k && k.getInt() > 2;
      });
      if (!topkOver2) return;
      SmallVector<Value> ends;
      for (Operation &op : *blk) {
        if (!isInt8Contraction(&op) || !isExpertSlice(op.getOperand(1))) continue;
        Value v = op.getResult(0);
        while (v.hasOneUse()) {
          Operation *u = *v.user_begin();
          if (u->getBlock() != blk || u->getNumResults() != 1) break;
          auto g = dyn_cast<linalg::GenericOp>(u);
          if (!(g && g.getNumLoops() == g.getNumParallelLoops()) &&
              !isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(u))
            break;
          // the sum with another expert's term: the term ends before it
          bool other = llvm::any_of(u->getOperands(), [&](Value o) {
            llvm::DenseSet<Operation *> seen;
            return o != v && dependsOnContraction(o, blk, seen);
          });
          if (other) break;
          v = u->getResult(0);
        }
        if (v != op.getResult(0)) ends.push_back(v);
      }
      for (Value v : ends) {
        OpBuilder b(v.getContext());
        b.setInsertionPointAfterValue(v);
        auto barrier = IREE::Util::OptimizationBarrierOp::create(b, v.getLoc(), ValueRange{v});
        barrier->setAttr("sa.moe.isolate", b.getUnitAttr());
        v.replaceAllUsesExcept(barrier.getResult(0), barrier);
        ++n;
      }
    });
    if (report) llvm::errs() << "sa isolate-moe-experts: " << n << " expert terms\n";
  }

  Option<bool> force{*this, "force", llvm::cl::desc("Run without an sa device target (tests)"), llvm::cl::init(false)};
  Option<bool> report{*this, "report", llvm::cl::desc("Print the isolated term count"), llvm::cl::init(false)};
};

} // namespace

std::unique_ptr<Pass> createFormMoeRegionsPass(bool enabled) { return std::make_unique<FormMoeRegionsPass>(enabled); }
void registerFormMoeRegionsPass() { PassRegistration<FormMoeRegionsPass>(); }
std::unique_ptr<Pass> createIsolateMoeExpertsPass() { return std::make_unique<IsolateMoeExpertsPass>(); }
void registerIsolateMoeExpertsPass() { PassRegistration<IsolateMoeExpertsPass>(); }

namespace {
struct DropMoeIsolationPass : public PassWrapper<DropMoeIsolationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DropMoeIsolationPass)
  StringRef getArgument() const override { return "iree-sa-drop-moe-isolation"; }
  StringRef getDescription() const override {
    return "Removes iree-sa-isolate-moe-experts' barriers once the dispatch regions are formed";
  }
  void runOnOperation() override {
    SmallVector<IREE::Util::OptimizationBarrierOp> barriers;
    getOperation()->walk([&](IREE::Util::OptimizationBarrierOp op) {
      if (op->hasAttr("sa.moe.isolate")) barriers.push_back(op);
    });
    for (auto op : barriers) {
      op->replaceAllUsesWith(op->getOperands());
      op->erase();
    }
  }
};
} // namespace

std::unique_ptr<Pass> createDropMoeIsolationPass() { return std::make_unique<DropMoeIsolationPass>(); }
void registerDropMoeIsolationPass() { PassRegistration<DropMoeIsolationPass>(); }

} // namespace mlir::iree_compiler::sa

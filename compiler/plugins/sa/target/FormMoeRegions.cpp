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
#include <cstdint>
#include <limits>

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

tensor::ExtractSliceOp top2PrefixSlice(Value sorted) {
  if (!sorted.hasOneUse()) return {};
  auto slice = dyn_cast<tensor::ExtractSliceOp>(*sorted.user_begin());
  if (!slice || slice.getStaticOffsets() != ArrayRef<int64_t>({0, 0}) ||
      slice.getStaticSizes() != ArrayRef<int64_t>({1, 2}) ||
      slice.getStaticStrides() != ArrayRef<int64_t>({1, 1}))
    return {};
  auto type = dyn_cast<RankedTensorType>(slice.getType());
  return type && ((type.getRank() == 1 && type.getDimSize(0) == 2) ||
                  (type.getRank() == 2 && type.getShape() == ArrayRef<int64_t>({1, 2})))
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
// Top-2. An id read as an index (tensor.extract + index_cast: the expert
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

bool rewriteTorchTop2Sort(IREE::LinalgExt::SortOp sort, IRRewriter &rewriter) {
  if (sort.getNumResults() != 2 || sort.getDimension() != 1) return false;
  auto scoresType = dyn_cast<RankedTensorType>(sort.getOutputs()[0].getType());
  auto idsType = dyn_cast<RankedTensorType>(sort.getOutputs()[1].getType());
  if (!scoresType || !idsType || !scoresType.hasStaticShape() || !idsType.hasStaticShape() ||
      scoresType.getRank() != 2 || idsType.getRank() != 2 ||
      scoresType.getDimSize(0) != 1 || idsType.getShape() != scoresType.getShape() ||
      !scoresType.getElementType().isF32() || !idsType.getElementType().isInteger(64))
    return false;
  int64_t experts = scoresType.getDimSize(1);
  if (experts < 4 || experts > 16 || !isIotaIndices(sort.getOutputs()[1])) return false;
  llvm::DenseSet<Operation *> seen;
  if (!comesFromProjection(sort.getOutputs()[0], seen)) return false;
  auto valueSlice = top2PrefixSlice(sort.getResult(0));
  auto indexSlice = top2PrefixSlice(sort.getResult(1));
  if (!valueSlice || !indexSlice) return false;
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
  auto valuesType = RankedTensorType::get({1, 2}, f32);
  auto indicesType = RankedTensorType::get({1, 2}, i32);
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
  Value values2 = topk.getResult(0);
  auto indices64Type = RankedTensorType::get({1, 2}, rewriter.getI64Type());
  auto ext = arith::ExtSIOp::create(rewriter, loc, indices64Type, topk.getResult(1));
  Value ids2 = ext;
  if (cast<RankedTensorType>(valueSlice.getType()).getRank() == 1)
    values2 = tensor::CollapseShapeOp::create(rewriter, loc, valueSlice.getType(),
                                               values2, reassociation);
  if (cast<RankedTensorType>(indexSlice.getType()).getRank() == 1)
    ids2 = tensor::CollapseShapeOp::create(rewriter, loc, indexSlice.getType(),
                                            ids2, reassociation);
  rewriter.replaceOp(valueSlice, values2);
  rewriter.replaceOp(indexSlice, ids2);
  rewriter.eraseOp(sort);
  readIdsAsI32(rewriter, ext);
  return true;
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

bool derivesFromTop2Ids(Value value, IREE::LinalgExt::TopkOp topk, unsigned depth = 0) {
  if (value == topk.getResult(1)) return true;
  if (depth > 4) return false;
  Operation *producer = value.getDefiningOp();
  if (!producer || producer->getNumOperands() != 1 ||
      !isa<arith::ExtSIOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp,
           tensor::CastOp>(producer))
    return false;
  return derivesFromTop2Ids(producer->getOperand(0), topk, depth + 1);
}

bool derivesFrom(Value value, Value ancestor, unsigned depth = 0) {
  if (value == ancestor) return true;
  if (depth > 16) return false;
  Operation *producer = value.getDefiningOp();
  if (!producer) return false;
  for (Value operand : producer->getOperands())
    if (derivesFrom(operand, ancestor, depth + 1)) return true;
  return false;
}

// This is an annotation only. It intentionally requires the three expert
// tables to share the same dynamic Top-2 IDs and verifies both FFN dimensions;
// unsupported computation continues on IREE's ordinary fallback path.
std::optional<std::pair<int64_t, int64_t>> mixtralExpertShape(
    Operation *scope, IREE::LinalgExt::TopkOp topk, int64_t experts) {
  auto projection = topk.getValues().getDefiningOp<linalg::MatmulOp>();
  if (!projection) return std::nullopt;
  auto input = dyn_cast<RankedTensorType>(projection.getDpsInputs()[0].getType());
  if (!input || input.getRank() != 2 || input.getDimSize(0) != 1 ||
      !input.getElementType().isF32()) return std::nullopt;
  int64_t dim = input.getDimSize(1);
  if (dim <= 0) return std::nullopt;
  SmallVector<RankedTensorType> gathered;
  scope->walk([&](linalg::GenericOp generic) {
    if (generic.getNumDpsInputs() != 1 || generic->getNumResults() != 1 ||
        !derivesFromTop2Ids(generic.getDpsInputs()[0], topk))
      return;
    auto result = dyn_cast<RankedTensorType>(generic->getResult(0).getType());
    if (!result || result.getRank() != 3 || result.getDimSize(0) != 2 ||
        !result.getElementType().isF32())
      return;
    tensor::ExtractOp extract;
    generic.walk([&](tensor::ExtractOp op) { extract = op; });
    if (!extract || extract.getIndices().size() != 3) return;
    auto expertIndex = extract.getIndices()[0].getDefiningOp<arith::IndexCastOp>();
    auto row = extract.getIndices()[1].getDefiningOp<linalg::IndexOp>();
    auto column = extract.getIndices()[2].getDefiningOp<linalg::IndexOp>();
    if (!expertIndex || expertIndex.getIn() != generic.getRegion().front().getArgument(0) ||
        !row || row.getDim() != 1 || !column || column.getDim() != 2)
      return;
    auto table = dyn_cast<RankedTensorType>(extract.getTensor().getType());
    if (!table || table.getRank() != 3 || table.getDimSize(0) != experts ||
        table.getDimSize(1) != result.getDimSize(1) ||
        table.getDimSize(2) != result.getDimSize(2))
      return;
    gathered.push_back(result);
  });
  if (gathered.size() != 3) return std::nullopt;
  int64_t hidden = 0;
  unsigned inputTables = 0, outputTables = 0;
  for (auto type : gathered) {
    if (type.getDimSize(2) == dim) {
      if (!hidden) hidden = type.getDimSize(1);
      if (hidden == type.getDimSize(1)) ++inputTables;
    } else if (type.getDimSize(1) == dim) {
      if (!hidden) hidden = type.getDimSize(2);
      if (hidden == type.getDimSize(2)) ++outputTables;
    }
  }
  if (hidden <= dim || inputTables != 2 || outputTables != 1) return std::nullopt;
  unsigned inputMatmuls = 0, outputMatmuls = 0;
  Value downResult;
  scope->walk([&](linalg::BatchMatmulOp matmul) {
    auto lhs = dyn_cast<RankedTensorType>(matmul.getDpsInputs()[0].getType());
    auto rhs = dyn_cast<RankedTensorType>(matmul.getDpsInputs()[1].getType());
    if (!lhs || !rhs || lhs.getRank() != 3 || rhs.getRank() != 3 ||
        lhs.getDimSize(0) != 2 || rhs.getDimSize(0) != 2 || rhs.getDimSize(2) != 1)
      return;
    if (lhs.getDimSize(1) == hidden && lhs.getDimSize(2) == dim &&
        rhs.getDimSize(1) == dim)
      ++inputMatmuls;
    if (lhs.getDimSize(1) == dim && lhs.getDimSize(2) == hidden &&
        rhs.getDimSize(1) == hidden) {
      ++outputMatmuls;
      downResult = matmul->getResult(0);
    }
  });
  if (inputMatmuls != 2 || outputMatmuls != 1) return std::nullopt;
  Value weightedResult;
  scope->walk([&](linalg::GenericOp generic) {
    if (generic.getNumDpsInputs() != 2 || generic->getNumResults() != 1) return;
    auto type = dyn_cast<RankedTensorType>(generic->getResult(0).getType());
    if (!type || type.getShape() != ArrayRef<int64_t>({2, dim})) return;
    Value first = generic.getDpsInputs()[0], second = generic.getDpsInputs()[1];
    if (!((derivesFrom(first, downResult) && derivesFrom(second, topk.getResult(0))) ||
          (derivesFrom(second, downResult) && derivesFrom(first, topk.getResult(0)))))
      return;
    auto yield = dyn_cast<linalg::YieldOp>(generic.getRegion().front().getTerminator());
    if (yield && yield.getValues().size() == 1 &&
        yield.getValues()[0].getDefiningOp<arith::MulFOp>())
      weightedResult = generic->getResult(0);
  });
  if (!weightedResult) return std::nullopt;
  bool combined = false;
  scope->walk([&](linalg::GenericOp generic) {
    if (generic.getNumDpsInputs() != 1 || generic.getDpsInputs()[0] != weightedResult ||
        generic.getNumReductionLoops() != 1 || generic->getNumResults() != 1)
      return;
    auto type = dyn_cast<RankedTensorType>(generic->getResult(0).getType());
    auto yield = dyn_cast<linalg::YieldOp>(generic.getRegion().front().getTerminator());
    if (type && type.getShape() == ArrayRef<int64_t>({1, dim}) && yield &&
        yield.getValues().size() == 1 &&
        yield.getValues()[0].getDefiningOp<arith::AddFOp>())
      combined = true;
  });
  if (!combined) return std::nullopt;
  return std::pair<int64_t, int64_t>{dim, hidden};
}

// sa-moe-parameters-v1 stores each weight before its per-row f32 scales,
// aligning every section to 64 bytes. The route record needs this stride.
std::optional<int64_t> packedExpertStride(int64_t dim, int64_t hidden,
                                          int64_t experts) {
  if (dim <= 0 || hidden <= 0 || dim > (1 << 20) || hidden > (1 << 20))
    return std::nullopt;
  uint64_t weightBytes = uint64_t(dim) * uint64_t(hidden);
  uint64_t offset = 0;
  auto append = [&](uint64_t bytes) {
    offset = (offset + 63) & ~uint64_t(63);
    offset += bytes;
  };
  append(weightBytes);
  append(uint64_t(hidden) * 4);
  append(weightBytes);
  append(uint64_t(hidden) * 4);
  append(weightBytes);
  append(uint64_t(dim) * 4);
  uint64_t stride = (offset + 63) & ~uint64_t(63);
  if (stride == 0 || stride * uint64_t(experts - 1) > UINT32_MAX)
    return std::nullopt;
  return int64_t(stride);
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
    IRRewriter rewriter(&getContext());
    SmallVector<IREE::LinalgExt::SortOp> sorts;
    getOperation()->walk([&](IREE::LinalgExt::SortOp sort) { sorts.push_back(sort); });
    unsigned convertedSorts = 0;
    for (auto sort : sorts) convertedSorts += rewriteTorchTop2Sort(sort, rewriter);
    SmallVector<std::pair<IREE::LinalgExt::TopkOp, int64_t>> matches;
    getOperation()->walk([&](IREE::LinalgExt::TopkOp topk) {
      if (auto experts = routerExperts(topk)) matches.emplace_back(topk, *experts);
    });
    unsigned fullBlocks = 0;
    for (auto [topk, experts] : matches) {
      Operation *scope = topk->getParentOp();
      while (scope && scope->getName().getStringRef() != "util.func" &&
             scope->getName().getStringRef() != "func.func")
        scope = scope->getParentOp();
      if (scope)
        if (auto shape = mixtralExpertShape(scope, topk, experts)) {
          if (auto stride = packedExpertStride(shape->first, shape->second, experts)) {
            scope->setAttr("sa.moe.arch", rewriter.getStringAttr("mixtral_top2_ffn"));
            scope->setAttr("sa.moe.dim", rewriter.getI64IntegerAttr(shape->first));
            scope->setAttr("sa.moe.hidden", rewriter.getI64IntegerAttr(shape->second));
            scope->setAttr("sa.moe.experts", rewriter.getI64IntegerAttr(experts));
            scope->setAttr("sa.moe.expert_stride", rewriter.getI64IntegerAttr(*stride));
            ++fullBlocks;
          }
        }
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
      llvm::errs() << "sa form-moe-regions: " << matches.size() << " top-2 router regions ("
                   << convertedSorts << " Torch sort prefixes), " << fullBlocks
                   << " full expert blocks\n";
  }

  Option<bool> enabled{*this, "enabled", llvm::cl::desc("Form MoE router regions"), llvm::cl::init(true)};
  Option<bool> force{*this, "force", llvm::cl::desc("Run without an sa device target (tests)"), llvm::cl::init(false)};
  Option<bool> report{*this, "report", llvm::cl::desc("Print the recognized region count"), llvm::cl::init(false)};
};

} // namespace

std::unique_ptr<Pass> createFormMoeRegionsPass(bool enabled) { return std::make_unique<FormMoeRegionsPass>(enabled); }
void registerFormMoeRegionsPass() { PassRegistration<FormMoeRegionsPass>(); }

} // namespace mlir::iree_compiler::sa

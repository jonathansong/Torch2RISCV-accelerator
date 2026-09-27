// sa preprocessing: pack the constant weights of int8 linear layers
// (docs/iree_compiler_plan.md §6.3).
//
// The accelerator reads a linear layer's weights as B tiles: tile t is
// W[t*D:(t+1)*D, :]^T, i.e. the layout i8[N/D, K, D] of export_w8a8.pack_b.
// The frontend writes a linear layer the standard way (llm/../qllama.py),
// which after input conversion is
//     vecmat(x, transpose(extsi(W)))      W: tensor<NxKxi8>, a constant / immutable global
// and this pass rewrites it into
//     Wp  = linalg.pack W inner_dims_pos = [0] inner_tiles = [D]    -> tensor<N/D x K x D x i8>
//     acc = linalg.generic (t, j, k): out[t, j] += ext(x[k]) * ext(Wp[t, k, j])
//     y   = tensor.collapse_shape acc [[0, 1]]
// with the same arithmetic as the vecmat (inputs sign-extended to the
// accumulator type). Wp depends on constants only: IREE hoists it and, with
// the parameters imported (--iree-parameter-import), evaluates it at compile
// time, so the packed weights land in the module or in an exported parameter
// archive and the dispatch reads them directly; nothing is repacked at run time.
//
// Runs only when the module targets the sa device.
#include "SAPasses.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::sa {
namespace {

// True if |op| is linalg.generic y = extsi(x) (elementwise, one input).
static bool isExtsiGeneric(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 || !op.isAllParallelLoops()) return false;
  for (AffineMap m : op.getIndexingMapsArray())
    if (!m.isIdentity()) return false;
  Block &body = op.getRegion().front();
  if (body.getOperations().size() != 2) return false;
  auto ext = dyn_cast<arith::ExtSIOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  return ext && yield && ext.getIn() == body.getArgument(0) && yield.getOperand(0) == ext.getResult();
}

// True if |v| only depends on constants: an immutable global or a constant.
static bool isConstantWeight(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def) return false;
  if (auto load = dyn_cast<IREE::Util::GlobalLoadOpInterface>(def)) {
    // the load's own immutable flag is set later; ask the global
    if (load.isGlobalImmutable()) return true;
    auto global = SymbolTable::lookupNearestSymbolFrom<IREE::Util::GlobalOpInterface>(load, load.getGlobalAttr());
    return global && !global.isGlobalMutable();
  }
  return isa<arith::ConstantOp>(def);
}

// vecmat(x, transpose(extsi(W))) -> pack + generic + collapse_shape.
static LogicalResult rewriteVecmat(RewriterBase &rewriter, linalg::VecmatOp mm, int64_t d) {
  Value x = mm.getDpsInputs()[0], rhs = mm.getDpsInputs()[1], init = mm.getDpsInits()[0];
  auto tr = rhs.getDefiningOp<linalg::TransposeOp>();
  if (!tr || tr.getPermutation() != ArrayRef<int64_t>{1, 0}) return failure();
  auto ext = tr.getInput().getDefiningOp<linalg::GenericOp>();
  if (!ext || !isExtsiGeneric(ext)) return failure();
  Value w = ext.getDpsInputs()[0];
  auto wType = dyn_cast<RankedTensorType>(w.getType());
  auto xType = dyn_cast<RankedTensorType>(x.getType());
  auto outType = dyn_cast<RankedTensorType>(init.getType());
  if (!wType || !xType || !outType || !wType.hasStaticShape() || !outType.hasStaticShape() ||
      !wType.getElementType().isInteger(8) || !isConstantWeight(w))
    return failure();
  auto outElem = dyn_cast<IntegerType>(outType.getElementType());
  auto xElem = dyn_cast<IntegerType>(xType.getElementType());
  if (!outElem || !xElem || xElem.getWidth() > outElem.getWidth()) return failure();
  int64_t n = wType.getDimSize(0), k = wType.getDimSize(1);
  if (n % d || k % d) return failure();

  Location loc = mm.getLoc();
  rewriter.setInsertionPoint(mm);
  // Wp = pack(W): [N, K] -> [N/D, K, D]
  SmallVector<OpFoldResult> tiles = {rewriter.getIndexAttr(d)};
  Value dest = linalg::PackOp::createDestinationTensor(rewriter, loc, w, tiles, {0}, {});
  Value wp = linalg::PackOp::create(rewriter, loc, w, dest, {0}, tiles).getResult();
  // the accumulator as [N/D, D]
  auto acc2Type = RankedTensorType::get({n / d, d}, outElem);
  SmallVector<ReassociationIndices> reassoc = {{0, 1}};
  Value init2 = tensor::ExpandShapeOp::create(rewriter, loc, acc2Type, init, reassoc).getResult();
  MLIRContext *ctx = rewriter.getContext();
  AffineExpr t, j, kk;
  bindDims(ctx, t, j, kk);
  SmallVector<AffineMap> maps = {AffineMap::get(3, 0, {kk}, ctx), AffineMap::get(3, 0, {t, kk, j}, ctx),
                                 AffineMap::get(3, 0, {t, j}, ctx)};
  SmallVector<utils::IteratorType> iters = {utils::IteratorType::parallel, utils::IteratorType::parallel,
                                            utils::IteratorType::reduction};
  auto gen = linalg::GenericOp::create(
      rewriter, loc, TypeRange{acc2Type}, ValueRange{x, wp}, ValueRange{init2}, maps, iters,
      [&](OpBuilder &b, Location l, ValueRange args) {
        Value xe = args[0];
        if (xElem.getWidth() < outElem.getWidth()) xe = arith::ExtSIOp::create(b, l, outElem, xe);
        Value we = arith::ExtSIOp::create(b, l, outElem, args[1]);
        Value m = arith::MulIOp::create(b, l, xe, we);
        Value r = arith::AddIOp::create(b, l, args[2], m);
        linalg::YieldOp::create(b, l, r);
      });
  Value y = tensor::CollapseShapeOp::create(rewriter, loc, outType, gen.getResult(0), reassoc).getResult();
  rewriter.replaceOp(mm, y);
  if (tr->use_empty()) rewriter.eraseOp(tr);
  if (ext->use_empty()) rewriter.eraseOp(ext);
  return success();
}

// True if the module targets the sa device anywhere (#hal.device.target<"sa", ...>).
static bool targetsSA(ModuleOp module) {
  bool found = false;
  module.walk([&](Operation *op) {
    for (NamedAttribute a : op->getAttrs()) {
      a.getValue().walk([&](IREE::HAL::DeviceTargetAttr t) {
        if (t.getDeviceID().getValue() == "sa") found = true;
      });
    }
    return found ? WalkResult::interrupt() : WalkResult::advance();
  });
  return found;
}

struct PackLinearWeightsPass : public PassWrapper<PackLinearWeightsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PackLinearWeightsPass)

  PackLinearWeightsPass() = default;
  PackLinearWeightsPass(const PackLinearWeightsPass &other) : PassWrapper(other) {}
  explicit PackLinearWeightsPass(int64_t d) { this->d = d; }

  StringRef getArgument() const override { return "iree-sa-pack-linear-weights"; }
  StringRef getDescription() const override {
    return "Packs the constant weights of int8 linear layers into the sa B-tile layout";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!force && !targetsSA(module)) return;
    SmallVector<linalg::VecmatOp> ops;
    module.walk([&](linalg::VecmatOp op) { ops.push_back(op); });
    IRRewriter rewriter(&getContext());
    for (linalg::VecmatOp op : ops) (void)rewriteVecmat(rewriter, op, d);
  }

  Option<int64_t> d{*this, "d", llvm::cl::desc("Array size D"), llvm::cl::init(8)};
  Option<bool> force{*this, "force", llvm::cl::desc("Run without an sa device target (tests)"),
                     llvm::cl::init(false)};
};

}  // namespace

std::unique_ptr<Pass> createPackLinearWeightsPass(int64_t d) {
  return std::make_unique<PackLinearWeightsPass>(d);
}

void registerPackLinearWeightsPass() { PassRegistration<PackLinearWeightsPass>(); }

}  // namespace mlir::iree_compiler::sa

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
// accumulator type). Each weight is packed once (after its definition), and
// its other readers that gather elements (tensor.extract W[r, c], the
// embedding of a model whose classifier shares it) read the packed copy
// instead, Wp[r / D, c, r % D] (plan §8.12 C6.0): the unpacked table is then
// dead, so it is stored once.
//
// Expert tables (docs/moe_native_lowering_design.md §5.1): a gather of one row
// of a constant table T [E, ...] at an id from a Top-2 (a sort's or topk's
// indices), G[0, ...] = T[id, ...], becomes extract_slice T[id, 0, ..] (and the
// collapse dropping its unit dimension a rank-reduced slice), so the selected
// expert stays a dynamic view of T in the dispatch. A linear layer over such a
// slice of an int8 table T [E, N, K] reads the packed table Tp = pack(T) along
// N, [E, N/D, K, D], at the same id. Wp depends on constants only: IREE hoists it and, with
// the parameters imported (--iree-parameter-import), evaluates it at compile
// time, so the packed weights land in the module or in an exported parameter
// archive and the dispatch reads them directly; nothing is repacked at run time.
//
// Runs only when the module targets the sa device.
#include "SAPasses.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

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

// The packed copy of W (one per weight, right after its definition).
// The packed copy of W, one per weight. An immutable global's is a global of
// its own, W$packed, set by an initializer (IREE evaluates it at compile time):
// every function (prefill and decode) and every load of W shares it. Other
// weights are packed right after their definition.
static llvm::DenseMap<Operation *, IREE::Util::GlobalOp> packedGlobals;

// dim: the packed dimension (0: a linear layer's W [N, K]; 1: an expert table [E, N, K]).
static Value packOf(RewriterBase &rewriter, Value w, int64_t d, llvm::MapVector<Value, Value> &packs,
                    int64_t dim = 0) {
  if (auto it = packs.find(w); it != packs.end()) return it->second;
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<OpFoldResult> tiles = {rewriter.getIndexAttr(d)};
  Location loc = w.getLoc();
  if (auto load = w.getDefiningOp<IREE::Util::GlobalLoadOpInterface>()) {
    auto global = SymbolTable::lookupNearestSymbolFrom<IREE::Util::GlobalOp>(load, load.getGlobalAttr());
    if (global && !global.isGlobalMutable()) {
      IREE::Util::GlobalOp &pg = packedGlobals[global.getOperation()];
      if (!pg) {
        OpBuilder mb(global);
        mb.setInsertionPointAfter(global);
        auto wType = cast<RankedTensorType>(w.getType());
        SmallVector<int64_t> pShape(wType.getShape());
        pShape[dim] /= d;
        pShape.push_back(d);
        auto pType = RankedTensorType::get(pShape, wType.getElementType());
        auto module = global->getParentOfType<ModuleOp>();
        std::string name = (global.getSymName() + "$packed").str();
        for (int i = 1; SymbolTable::lookupSymbolIn(module, name); ++i)
          name = (global.getSymName() + "$packed" + std::to_string(i)).str();
        pg = IREE::Util::GlobalOp::create(mb, loc, name, /*isMutable=*/false, pType);
        pg.setPrivate();
        mb.setInsertionPointAfter(pg);
        auto init = IREE::Util::InitializerOp::create(mb, loc);
        OpBuilder ib = OpBuilder::atBlockBegin(init.addEntryBlock());
        Value src = global.createLoadOp(loc, ib).getLoadedGlobalValue();
        Value dest = linalg::PackOp::createDestinationTensor(ib, loc, src, tiles, {dim}, {});
        Value packed = linalg::PackOp::create(ib, loc, src, dest, {dim}, tiles).getResult();
        pg.createStoreOp(loc, packed, ib);
        IREE::Util::ReturnOp::create(ib, loc);
      }
      rewriter.setInsertionPointAfter(load);
      Value wp = pg.createLoadOp(loc, rewriter).getLoadedGlobalValue();
      packs[w] = wp;
      return wp;
    }
  }
  if (Operation *def = w.getDefiningOp()) rewriter.setInsertionPointAfter(def);
  else rewriter.setInsertionPointToStart(cast<BlockArgument>(w).getOwner());
  Value dest = linalg::PackOp::createDestinationTensor(rewriter, loc, w, tiles, {dim}, {});
  Value wp = linalg::PackOp::create(rewriter, loc, w, dest, {dim}, tiles).getResult();
  packs[w] = wp;
  return wp;
}

// True if |v| is (a reshape or slice of) the indices of a sort or a topk: a
// Top-2's expert ids (a model's other gathers, e.g. the embedding's token ids,
// are not).
static bool isTopkIndices(Value v, unsigned depth = 0) {
  if (depth > 8) return false;
  Operation *def = v.getDefiningOp();
  if (!def) return false;
  if (isa<IREE::LinalgExt::SortOp>(def)) return cast<OpResult>(v).getResultNumber() == 1;
  if (auto topk = dyn_cast<IREE::LinalgExt::TopkOp>(def)) return v == topk.getResult(1);
  if (isa<tensor::ExtractSliceOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::CastOp>(def))
    return isTopkIndices(def->getOperand(0), depth + 1);
  return false;
}

// A gather of one row of a constant table at a Top-2 id,
//     G[0, i1, .., ir] = T[index_cast(id[0]), i1, .., ir]     id: tensor<1xi64 | i32>
// -> extract_slice T[id[0], 0, ..] [1, ..] (G's type); a collapse of it dropping
// the unit dimension -> the rank-reduced slice.
static LogicalResult rewriteExpertGather(RewriterBase &rewriter, linalg::GenericOp g) {
  if (g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1 || !g.isAllParallelLoops()) return failure();
  Value id = g.getDpsInputs()[0];
  auto idType = dyn_cast<RankedTensorType>(id.getType());
  auto outType = dyn_cast<RankedTensorType>(g.getResult(0).getType());
  int64_t rank = outType ? outType.getRank() : 0;
  if (!idType || idType.getRank() != 1 || idType.getDimSize(0) != 1 || !idType.getElementType().isInteger() ||
      !outType || !outType.hasStaticShape() || rank < 2 || outType.getDimSize(0) != 1 || !isTopkIndices(id))
    return failure();
  MLIRContext *ctx = g.getContext();
  if (g.getIndexingMapsArray()[0] != AffineMap::get(rank, 0, {getAffineDimExpr(0, ctx)}, ctx) ||
      !g.getIndexingMapsArray()[1].isIdentity())
    return failure();
  Block &b = g.getRegion().front();
  auto y = cast<linalg::YieldOp>(b.getTerminator());
  auto ex = y.getOperand(0).getDefiningOp<tensor::ExtractOp>();
  if (!ex || ex->getBlock() != &b || ex.getIndices().size() != size_t(rank)) return failure();
  auto cast0 = ex.getIndices()[0].getDefiningOp<arith::IndexCastOp>();
  if (!cast0 || cast0.getIn() != b.getArgument(0)) return failure();
  for (int64_t i = 1; i < rank; ++i) {
    auto ix = ex.getIndices()[i].getDefiningOp<linalg::IndexOp>();
    if (!ix || ix.getDim() != uint64_t(i)) return failure();
  }
  // nothing else in the body: index_cast, the linalg.index ops, the extract
  if (int64_t(b.getOperations().size()) != rank + 2) return failure();
  Value table = ex.getTensor();
  auto tType = dyn_cast<RankedTensorType>(table.getType());
  if (!tType || !tType.hasStaticShape() || tType.getRank() != rank || !isConstantWeight(table) ||
      tType.getShape().drop_front() != outType.getShape().drop_front())
    return failure();

  rewriter.setInsertionPoint(g);
  Location loc = g.getLoc();
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value e = tensor::ExtractOp::create(rewriter, loc, id, ValueRange{c0});
  Value ei = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), e);
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0)), strides(rank, rewriter.getIndexAttr(1)),
      sizes;
  offsets[0] = ei;
  for (int64_t s : outType.getShape()) sizes.push_back(rewriter.getIndexAttr(s));
  Value slice = tensor::ExtractSliceOp::create(rewriter, loc, outType, table, offsets, sizes, strides);
  for (OpOperand &use : llvm::make_early_inc_range(g.getResult(0).getUses())) {
    auto cs = dyn_cast<tensor::CollapseShapeOp>(use.getOwner());
    if (!cs || cs.getReassociationIndices().empty() || cs.getReassociationIndices()[0] != ReassociationIndices{0, 1})
      continue;
    bool rest = true;
    for (auto [i, r] : llvm::enumerate(ArrayRef(cs.getReassociationIndices()).drop_front()))
      rest &= r == ReassociationIndices{int64_t(i) + 2};
    if (!rest) continue;
    rewriter.setInsertionPoint(cs);
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(cs, cs.getResultType(), table, offsets, sizes, strides);
  }
  rewriter.replaceOp(g, slice);
  if (slice.use_empty()) rewriter.eraseOp(slice.getDefiningOp());
  return success();
}

// W = extract_slice T[id, 0, 0] [1, N, K] (rank-reduced) of a constant int8
// table T [E, N, K]: (T, the offsets); null otherwise.
static std::pair<Value, SmallVector<OpFoldResult>> expertSlice(Value w) {
  auto sl = w.getDefiningOp<tensor::ExtractSliceOp>();
  if (!sl) return {};
  auto tType = dyn_cast<RankedTensorType>(sl.getSource().getType());
  auto wType = cast<RankedTensorType>(w.getType());
  if (!tType || !tType.hasStaticShape() || tType.getRank() != 3 || wType.getRank() != 2 ||
      !isConstantWeight(sl.getSource()) || tType.getShape().drop_front() != wType.getShape() ||
      !sl.hasUnitStride())
    return {};
  auto offs = sl.getMixedOffsets();
  if (!isConstantIntValue(offs[1], 0) || !isConstantIntValue(offs[2], 0) || !isConstantIntValue(sl.getMixedSizes()[0], 1))
    return {};
  return {sl.getSource(), SmallVector<OpFoldResult>(offs)};
}

// The packed weights of a linear layer over W: pack(W), or for an expert slice
// of T, the same slice of pack(T) along N: Tp[id, 0, 0, 0] [1, N/D, K, D].
static Value packedWeights(RewriterBase &rewriter, Value w, int64_t d, llvm::MapVector<Value, Value> &packs) {
  auto [table, offs] = expertSlice(w);
  if (!table) return packOf(rewriter, w, d, packs);
  Value tp = packOf(rewriter, table, d, packs, /*dim=*/1);
  auto wType = cast<RankedTensorType>(w.getType());
  int64_t n = wType.getDimSize(0), k = wType.getDimSize(1);
  auto pType = RankedTensorType::get({n / d, k, d}, wType.getElementType());
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(w.getDefiningOp());
  offs.push_back(rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(n / d), rewriter.getIndexAttr(k),
                                     rewriter.getIndexAttr(d)};
  SmallVector<OpFoldResult> strides(4, rewriter.getIndexAttr(1));
  return tensor::ExtractSliceOp::create(rewriter, w.getLoc(), pType, tp, offs, sizes, strides).getResult();
}

// The gathers of W (tensor.extract W[r, c]) -> Wp[r / D, c, r % D]. The device
// needs the tile's byte offset (r / D) * K * D as a 16-bit LDPARAM factor.
static void redirectGathers(RewriterBase &rewriter, Value w, Value wp, int64_t d) {
  auto wType = cast<RankedTensorType>(w.getType());
  if (wType.getRank() != 2 || wType.getDimSize(1) * d > 65535) return;
  SmallVector<tensor::ExtractOp> extracts;
  for (Operation *u : w.getUsers())
    if (auto ex = dyn_cast<tensor::ExtractOp>(u); ex && ex.getTensor() == w) extracts.push_back(ex);
  for (tensor::ExtractOp ex : extracts) {
    rewriter.setInsertionPoint(ex);
    Location loc = ex.getLoc();
    Value r = ex.getIndices()[0], c = ex.getIndices()[1];
    Value dv = arith::ConstantIndexOp::create(rewriter, loc, d);
    Value t = arith::DivSIOp::create(rewriter, loc, r, dv);
    Value l = arith::RemSIOp::create(rewriter, loc, r, dv);
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(ex, wp, ValueRange{t, c, l});
  }
}

// vecmat(x, transpose(extsi(W))) -> pack + generic + collapse_shape.
static LogicalResult rewriteVecmat(RewriterBase &rewriter, linalg::VecmatOp mm, int64_t d,
                                   llvm::MapVector<Value, Value> &packs) {
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
      !wType.getElementType().isInteger(8) || !(isConstantWeight(w) || expertSlice(w).first))
    return failure();
  auto outElem = dyn_cast<IntegerType>(outType.getElementType());
  auto xElem = dyn_cast<IntegerType>(xType.getElementType());
  if (!outElem || !xElem || xElem.getWidth() > outElem.getWidth()) return failure();
  int64_t n = wType.getDimSize(0), k = wType.getDimSize(1);
  if (n % d || k % d) return failure();

  Location loc = mm.getLoc();
  // Wp = pack(W): [N, K] -> [N/D, K, D]
  Value wp = packedWeights(rewriter, w, d, packs);
  rewriter.setInsertionPoint(mm);
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

// True if |v| is linalg.fill of a zero.
static bool isZeroFill(Value v) {
  auto fill = v.getDefiningOp<linalg::FillOp>();
  if (!fill) return false;
  auto c = fill.getDpsInputs()[0].getDefiningOp<arith::ConstantOp>();
  auto a = c ? dyn_cast<IntegerAttr>(c.getValue()) : IntegerAttr();
  return a && a.getValue().isZero();
}

// The prefill form (plan §8.13): matmul(X, transpose(extsi(W))) of M rows into a
// zero-filled accumulator (torch-mlir accumulates the int32 matmul in i64 and
// truncates) -> a generic over the packed weights,
//     acc[m, t, j] += ext(X[m, k]) * ext(Wp[t, k, j])     (i32)
// collapsed to [M, N] (and sign-extended back to the matmul's type: the
// consumer's trunci then folds away; an int8 x int8 sum over K < 2^17 cannot
// overflow i32, and i32 wraps as the i64 sum truncated anyway). X is taken
// before its extsi when it has one, so it stays int8 in memory (the A strip of
// D rows is then one interleaved DMA).
static LogicalResult rewriteMatmul(RewriterBase &rewriter, linalg::MatmulOp mm, int64_t d,
                                   llvm::MapVector<Value, Value> &packs) {
  if (mm.hasUserDefinedMaps()) return failure();
  Value x = mm.getDpsInputs()[0], rhs = mm.getDpsInputs()[1], init = mm.getDpsInits()[0];
  auto tr = rhs.getDefiningOp<linalg::TransposeOp>();
  if (!tr || tr.getPermutation() != ArrayRef<int64_t>{1, 0}) return failure();
  auto ext = tr.getInput().getDefiningOp<linalg::GenericOp>();
  if (!ext || !isExtsiGeneric(ext)) return failure();
  Value w = ext.getDpsInputs()[0];
  auto wType = dyn_cast<RankedTensorType>(w.getType());
  auto outType = dyn_cast<RankedTensorType>(init.getType());
  if (!wType || !outType || !wType.hasStaticShape() || !outType.hasStaticShape() ||
      !wType.getElementType().isInteger(8) || !(isConstantWeight(w) || expertSlice(w).first) || !isZeroFill(init))
    return failure();
  if (auto xe = x.getDefiningOp<linalg::GenericOp>(); xe && isExtsiGeneric(xe))
    x = xe.getDpsInputs()[0];
  auto xType = dyn_cast<RankedTensorType>(x.getType());
  auto outElem = dyn_cast<IntegerType>(outType.getElementType());
  auto xElem = xType ? dyn_cast<IntegerType>(xType.getElementType()) : IntegerType();
  if (!xType || !xType.hasStaticShape() || !outElem || !xElem || outElem.getWidth() < 32 || xElem.getWidth() > 32)
    return failure();
  int64_t m = outType.getDimSize(0), n = wType.getDimSize(0), k = wType.getDimSize(1);
  if (n % d || k % d || k >= (1 << 17)) return failure();

  Location loc = mm.getLoc();
  Value wp = packedWeights(rewriter, w, d, packs);
  rewriter.setInsertionPoint(mm);
  Type i32 = rewriter.getI32Type();
  auto acc3Type = RankedTensorType::get({m, n / d, d}, i32);
  Value zero = arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
  Value empty = tensor::EmptyOp::create(rewriter, loc, acc3Type.getShape(), i32);
  Value init3 = linalg::FillOp::create(rewriter, loc, zero, empty).getResult(0);
  MLIRContext *ctx = rewriter.getContext();
  AffineExpr r, t, j, kk;
  bindDims(ctx, r, t, j, kk);
  SmallVector<AffineMap> maps = {AffineMap::get(4, 0, {r, kk}, ctx), AffineMap::get(4, 0, {t, kk, j}, ctx),
                                 AffineMap::get(4, 0, {r, t, j}, ctx)};
  SmallVector<utils::IteratorType> iters = {utils::IteratorType::parallel, utils::IteratorType::parallel,
                                            utils::IteratorType::parallel, utils::IteratorType::reduction};
  auto gen = linalg::GenericOp::create(
      rewriter, loc, TypeRange{acc3Type}, ValueRange{x, wp}, ValueRange{init3}, maps, iters,
      [&](OpBuilder &b, Location l, ValueRange args) {
        Value xe = args[0];
        if (xElem.getWidth() < 32) xe = arith::ExtSIOp::create(b, l, i32, xe);
        Value we = arith::ExtSIOp::create(b, l, i32, args[1]);
        Value p = arith::MulIOp::create(b, l, xe, we);
        linalg::YieldOp::create(b, l, arith::AddIOp::create(b, l, args[2], p).getResult());
      });
  auto acc2Type = RankedTensorType::get({m, n}, i32);
  SmallVector<ReassociationIndices> reassoc = {{0}, {1, 2}};
  Value y = tensor::CollapseShapeOp::create(rewriter, loc, acc2Type, gen.getResult(0), reassoc).getResult();
  if (outElem.getWidth() > 32) {
    Value e = tensor::EmptyOp::create(rewriter, loc, outType.getShape(), outElem);
    SmallVector<AffineMap> id(2, rewriter.getMultiDimIdentityMap(2));
    SmallVector<utils::IteratorType> par(2, utils::IteratorType::parallel);
    y = linalg::GenericOp::create(rewriter, loc, TypeRange{outType}, ValueRange{y}, ValueRange{e}, id, par,
                                  [&](OpBuilder &b, Location l, ValueRange args) {
                                    linalg::YieldOp::create(b, l, arith::ExtSIOp::create(b, l, outElem, args[0])
                                                                      .getResult());
                                  })
            .getResult(0);
  }
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
    IRRewriter rewriter(&getContext());
    SmallVector<linalg::GenericOp> gens;
    module.walk([&](linalg::GenericOp op) { gens.push_back(op); });
    for (linalg::GenericOp op : gens) (void)rewriteExpertGather(rewriter, op);
    SmallVector<linalg::VecmatOp> ops;
    module.walk([&](linalg::VecmatOp op) { ops.push_back(op); });
    SmallVector<linalg::MatmulOp> mms;
    module.walk([&](linalg::MatmulOp op) { mms.push_back(op); });
    llvm::MapVector<Value, Value> packs;
    packedGlobals.clear();
    for (linalg::VecmatOp op : ops) (void)rewriteVecmat(rewriter, op, d, packs);
    for (linalg::MatmulOp op : mms) (void)rewriteMatmul(rewriter, op, d, packs);
    if (gathers) {
      // by global: a gather of any load of a packed global (another function's,
      // e.g. prefill and decode, or a second load) reads a packed copy of it
      llvm::DenseSet<Attribute> globals;
      for (auto &[w, wp] : packs)
        if (auto load = w.getDefiningOp<IREE::Util::GlobalLoadOpInterface>();
            load && cast<RankedTensorType>(w.getType()).getRank() == 2)
          globals.insert(load.getGlobalAttr());
      llvm::SetVector<Value> loads;
      module.walk([&](tensor::ExtractOp ex) {
        if (auto load = ex.getTensor().getDefiningOp<IREE::Util::GlobalLoadOpInterface>();
            load && globals.contains(load.getGlobalAttr()))
          loads.insert(ex.getTensor());
      });
      for (Value v : loads) packOf(rewriter, v, d, packs);
      for (auto &[w, wp] : packs) redirectGathers(rewriter, w, wp, d);
    }
  }

  Option<int64_t> d{*this, "d", llvm::cl::desc("Array size D"), llvm::cl::init(8)};
  Option<bool> gathers{*this, "gathers", llvm::cl::desc("Gathers of packed weights read the packed copy"),
                       llvm::cl::init(true)};
  Option<bool> force{*this, "force", llvm::cl::desc("Run without an sa device target (tests)"),
                     llvm::cl::init(false)};
};

}  // namespace

std::unique_ptr<Pass> createPackLinearWeightsPass(int64_t d) {
  return std::make_unique<PackLinearWeightsPass>(d);
}

void registerPackLinearWeightsPass() { PassRegistration<PackLinearWeightsPass>(); }

}  // namespace mlir::iree_compiler::sa

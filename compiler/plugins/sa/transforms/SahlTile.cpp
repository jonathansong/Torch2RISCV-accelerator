// sahl-tile (docs/iree_compiler_plan.md §8.4 step 2, §8.14 C8 R3): an
// element-wise dispatch whose local buffers do not fit ACC is split into
// pieces, each a sahl.scope working on views and local buffers of the piece
// (its local memory is released at its end):
//   flat pieces: every operation over the same N elements (identity maps,
//     contiguous views; a max reduction of the whole vector into a scalar,
//     combined over the pieces: the later pieces' reductions are marked
//     sahl.accumulate, their result buffer allocated before the pieces with
//     sahl.reserve and its fill done once);
//   row pieces: every operation row by row over [R, C] or [R, C1, C2, ...]
//     (an even number of rows per piece; two rows whose generics are lowered
//     one row at a time, sahl.row_by_row, when not even two fit).
// The ACC estimate is the lowering's (an upper bound: every buffer, a temp per
// fp32 operation); sahl-plan-memory replaces it (R4).
#include "SahlDialect.h"
#include "SahlKernels.h"
#include "SahlPasses.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::iree_compiler::sa {
namespace {

// the lowering's decisions (moved from sahl-to-sahw)
struct Decider {
  func::FuncOp f;
  int64_t d;
  // A dispatch whose operations go row by row over [R, C] (R != C) or
  // [R, C1, C2, ...] (a row's shape T = [C] or [C1, C2, ...]): generics over all
  // the dimensions with maps (r, c...), (r), (c...) or () (reductions over c...
  // only, no indices or gathers) or over [R] / [C]; loads / stores of [R, T]
  // (contiguous or a column slice), [R], T or scalars. Its rows are
  // independent: R and T.
  std::optional<std::pair<int64_t, SmallVector<int64_t>>> rowPieceable() {
    int64_t R = -1;
    SmallVector<int64_t> T;
    for (Operation &op : f.getBody().front())
      for (Value v : op.getOperands())
        if (auto mt = dyn_cast<MemRefType>(v.getType()); mt && mt.hasStaticShape() && mt.getRank() >= 2 && R < 0 &&
                                                            isa<linalg::GenericOp, sahl::LoadOp, sahl::StoreOp>(op)) {
          R = mt.getDimSize(0);
          T.assign(mt.getShape().begin() + 1, mt.getShape().end());
        }
    if (R <= 1 || T.empty() || llvm::product_of(T) <= 1 || (T.size() == 1 && R == T[0])) return std::nullopt;
    const unsigned rank = T.size() + 1;
    auto shapeOk = [&](Value v) {
      auto mt = dyn_cast<MemRefType>(v.getType());
      if (!mt || !mt.hasStaticShape()) return false;
      if (mt.getNumElements() == 1) return true;
      ArrayRef<int64_t> sh = mt.getShape();
      if (sh.size() == rank) return sh[0] == R && sh.drop_front() == ArrayRef<int64_t>(T);
      return (sh.size() == 1 && sh[0] == R) || sh == ArrayRef<int64_t>(T);
    };
    MLIRContext *ctx = f.getContext();
    SmallVector<AffineExpr> all, inner;
    for (unsigned i = 0; i < rank; ++i) (i ? inner : all).push_back(getAffineDimExpr(i, ctx));
    all.append(inner);
    AffineExpr r = getAffineDimExpr(0, ctx);
    for (Operation &op : f.getBody().front()) {
      if (auto l = dyn_cast<sahl::LoadOp>(&op)) {
        if (!shapeOk(l.getSrc()) || !shapeOk(l.getDst())) return std::nullopt;
      } else if (auto st = dyn_cast<sahl::StoreOp>(&op)) {
        if (!shapeOk(st.getSrc()) || !shapeOk(st.getDst())) return std::nullopt;
      } else if (auto fl = dyn_cast<linalg::FillOp>(&op)) {
        if (!shapeOk(fl.getDpsInits()[0])) return std::nullopt;
      } else if (auto g = dyn_cast<linalg::GenericOp>(&op)) {
        Block &b = g.getRegion().front();
        if (!b.getOps<linalg::IndexOp>().empty() || !b.getOps<sahl::GatherOp>().empty()) return std::nullopt;
        auto iters = g.getIteratorTypesArray();
        auto maps = g.getIndexingMapsArray();
        if (iters.size() == rank) {
          if (iters[0] != utils::IteratorType::parallel) return std::nullopt;
          for (auto [v, m] : llvm::zip(g->getOperands(), maps)) {
            if (!shapeOk(v)) return std::nullopt;
            if (m != AffineMap::get(rank, 0, all, ctx) && m != AffineMap::get(rank, 0, {r}, ctx) &&
                m != AffineMap::get(rank, 0, inner, ctx) && m.getNumResults() != 0)
              return std::nullopt;
          }
        } else if (iters.size() == 1) {
          for (auto [v, m] : llvm::zip(g->getOperands(), maps))
            if (!shapeOk(v) || !(m.isIdentity() || m.getNumResults() == 0)) return std::nullopt;
        } else {
          return std::nullopt;
        }
      } else if (!isa<arith::ConstantOp, memref::AllocOp, memref::DeallocOp, memref::SubViewOp, memref::CastOp,
                      IREE::HAL::InterfaceBindingSubspanOp, IREE::HAL::InterfaceConstantLoadOp, func::ReturnOp,
                      arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtUIOp, arith::ShLIOp, arith::OrIOp>(op) &&
                 !(op.getDialect() && op.getDialect()->getNamespace() == "util")) {
        return std::nullopt;
      }
    }
    return std::make_pair(R, T);
  }
  // ACC words such a dispatch may need at once: every [R, T] buffer, a temp per
  // fp32 operation of a generic over all the dimensions (with more than one
  // dimension in a row: only the operations on a whole row's values, not those
  // on a per-row scalar, e.g. RMSNorm's rsqrt of the mean, which with the other
  // buffers (T, [R]) go into a part that does not scale with the rows; [R, C] as
  // before): {words that scale with the rows, words that do not}
  std::pair<int64_t, int64_t> pieceWordsRows(int64_t R, ArrayRef<int64_t> T) {
    const unsigned rank = T.size() + 1;
    int64_t bufs = 0, fixed = 0;
    for (Operation &op : f.getBody().front()) {
      if (auto a = dyn_cast<memref::AllocOp>(&op)) {
        auto mt = cast<MemRefType>(a.getType());
        if (mt.getElementType().isInteger(8) || !mt.hasStaticShape()) continue;
        if (mt.getRank() == int64_t(rank) && mt.getDimSize(0) == R) ++bufs;
        else if (rank > 2) fixed += (mt.getNumElements() + d - 1) / d;
      } else if (auto g = dyn_cast<linalg::GenericOp>(&op); g && g.getNumLoops() == rank) {
        // the values that vary within a row: from an operand mapped to more than d0
        llvm::DenseSet<Value> wide;
        Block &blk = g.getRegion().front();
        for (auto [arg, m] : llvm::zip(blk.getArguments(), g.getIndexingMapsArray()))
          if (rank == 2 || m.getNumResults() > 1 || (m.getNumResults() == 1 && m.getResult(0) != getAffineDimExpr(0, f.getContext())))
            wide.insert(arg);
        // the operations that make a new fp32 value (casts, compares, selects and
        // the to_i8 chain are stages of the VE that consumes them)
        for (Operation &o : blk.without_terminator()) {
          if (llvm::any_of(o.getOperands(), [&](Value v) { return wide.contains(v); }))
            for (Value r : o.getResults()) wide.insert(r);
          if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp, arith::MaximumFOp, arith::MinimumFOp,
                  math::ExpOp, math::RsqrtOp, math::AbsFOp, arith::NegFOp>(o)) {
            if (rank == 2 || wide.contains(o.getResult(0))) ++bufs;
            else fixed += R;                     // (a word per row: the broadcast layout)
          }
        }
      }
    }
    return {bufs * ((R * llvm::product_of(T) + d - 1) / d), fixed};
  }

  // An element-wise dispatch (static contiguous loads / stores, generics with
  // identity maps over N elements, scalar inputs): N, else nothing.
  std::optional<int64_t> pieceable() {
    int64_t n = -1;
    auto size = [&](Value v, bool scalarOk) {
      auto mt = dyn_cast<MemRefType>(v.getType());
      if (!mt || !mt.hasStaticShape()) return false;
      // flat pieces of a contiguous view only (a column slice goes by rows: rowPieceable)
      SmallVector<int64_t> st;
      int64_t off;
      if (failed(mt.getStridesAndOffset(st, off))) return false;
      int64_t expect = 1;
      for (int i = int(mt.getRank()) - 1; i >= 0; --i) {
        if (mt.getDimSize(i) != 1 && st[i] != expect) return false;
        expect *= mt.getDimSize(i);
      }
      if (mt.getRank() == 0 || (scalarOk && mt.getNumElements() == 1)) return scalarOk;
      if (n < 0) n = mt.getNumElements();
      return mt.getNumElements() == n;
    };
    for (Operation &op : f.getBody().front()) {
      if (auto l = dyn_cast<sahl::LoadOp>(&op)) {
        if (!size(l.getSrc(), true) || !size(l.getDst(), true)) return std::nullopt;
      } else if (auto st = dyn_cast<sahl::StoreOp>(&op)) {
        if (!size(st.getSrc(), true) || !size(st.getDst(), true)) return std::nullopt;
      } else if (auto g = dyn_cast<linalg::GenericOp>(&op)) {
        if (!g.getRegion().front().getOps<linalg::IndexOp>().empty() ||
            !g.getRegion().front().getOps<sahl::GatherOp>().empty())
          return std::nullopt;
        if (g.getNumReductionLoops()) {
          // a max of the whole vector into a scalar (the pieces' maxima combine in any order)
          auto maps = g.getIndexingMapsArray();
          auto yield = cast<linalg::YieldOp>(g.getRegion().front().getTerminator());
          if (g.getNumLoops() != 1 || g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1 ||
              !maps[0].isIdentity() || maps[1].getNumResults() != 0 || !size(g.getDpsInputs()[0], false) ||
              !yield.getOperand(0).getDefiningOp<arith::MaximumFOp>())
            return std::nullopt;
          continue;
        }
        for (auto [v, m] : llvm::zip(g->getOperands(), g.getIndexingMapsArray()))
          if (!(m.isIdentity() && size(v, false)) && !(m.getNumResults() == 0 && size(v, true))) return std::nullopt;
      } else if (auto fl = dyn_cast<linalg::FillOp>(&op)) {
        if (!size(fl.getDpsInits()[0], true)) return std::nullopt;
      } else if (!isa<arith::ConstantOp, memref::AllocOp, memref::DeallocOp, memref::SubViewOp, memref::CastOp,
                      IREE::HAL::InterfaceBindingSubspanOp, IREE::HAL::InterfaceConstantLoadOp, func::ReturnOp,
                      arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtUIOp, arith::ShLIOp, arith::OrIOp>(op) &&
                 !(op.getDialect() && op.getDialect()->getNamespace() == "util")) {
        return std::nullopt;
      }
    }
    if (n <= 0) return std::nullopt;
    return n;
  }
  // ACC words an element-wise dispatch of n elements may need at once (an upper
  // bound: every N-element buffer, one temp per operation of a generic)
  int64_t pieceWords(int64_t n) {
    int64_t bufs = 0;
    for (Operation &op : f.getBody().front()) {
      if (auto a = dyn_cast<memref::AllocOp>(&op)) {
        auto mt = cast<MemRefType>(a.getType());
        if (mt.getNumElements() == n && !mt.getElementType().isInteger(8)) ++bufs;
      } else if (auto g = dyn_cast<linalg::GenericOp>(&op)) {
        // (a sahl.to_i8 counts as the 12 operations of the chain it replaced: the
        // estimate, and so the pieces, as before; sahl-plan-memory replaces this, §8.14 R4)
        for (Operation &o : g.getRegion().front().without_terminator()) bufs += isa<sahl::ToI8Op>(o) ? 12 : 1;
      }
    }
    return bufs * ((n + d - 1) / d);
  }

};

bool isWork(Operation &op) {
  return isa<sahl::LoadOp, sahl::StoreOp, linalg::FillOp, linalg::GenericOp>(op);
}

// flat pieces [off, off + len) of the N-element operands
void flatPieces(func::FuncOp f, int64_t N, int64_t len) {
  Block &blk = f.getBody().front();
  MLIRContext *ctx = f.getContext();
  SmallVector<Operation *> work;
  for (Operation &op : blk)
    if (isWork(op)) work.push_back(&op);
  Operation *term = blk.getTerminator();
  OpBuilder b(term);
  // the reductions' results: allocated first, filled once
  llvm::DenseSet<Value> keep;
  for (Operation *op : work)
    if (auto g = dyn_cast<linalg::GenericOp>(op); g && g.getNumReductionLoops()) {
      keep.insert(g.getDpsInits()[0]);
      auto r = sahl::ReserveOp::create(b, g.getLoc(), g.getDpsInits()[0]);
      r.setBcast(true);
    }
  for (Operation *op : work)
    if (auto fl = dyn_cast<linalg::FillOp>(op); fl && keep.contains(fl.getDpsInits()[0])) b.clone(*op);
  auto isN = [&](Value v) {
    auto mt = dyn_cast<MemRefType>(v.getType());
    return mt && mt.hasStaticShape() && mt.getNumElements() == N;
  };
  for (int64_t off = 0; off < N; off += len) {
    int64_t n = std::min(len, N - off);
    auto sc = sahl::ScopeOp::create(b, f.getLoc(), /*keep=*/false, /*spad_from=*/IntegerAttr());
    Block *sb = &sc.getBody().emplaceBlock();
    OpBuilder ib = OpBuilder::atBlockEnd(sb);
    IRMapping map;
    // the piece of a value: a local [n], or a view of DDR [n]
    auto pieceOf = [&](Value v) -> Value {
      if (Value m = map.lookupOrNull(v)) return m;
      auto mt = cast<MemRefType>(v.getType());
      Value r;
      if (!ddrRoot(v)) {
        r = memref::AllocOp::create(ib, v.getLoc(), MemRefType::get({n}, mt.getElementType()));
      } else {
        Value flat = v;
        if (mt.getRank() != 1) {
          SmallVector<ReassociationIndices> re(1);
          for (int64_t i = 0; i < mt.getRank(); ++i) re[0].push_back(i);
          flat = memref::CollapseShapeOp::create(ib, v.getLoc(), v, re);
        }
        r = memref::SubViewOp::create(ib, v.getLoc(), flat, ArrayRef<OpFoldResult>{ib.getIndexAttr(off)},
                                      ArrayRef<OpFoldResult>{ib.getIndexAttr(n)},
                                      ArrayRef<OpFoldResult>{ib.getIndexAttr(1)});
      }
      map.map(v, r);
      return r;
    };
    for (Operation *op : work) {
      if (auto fl = dyn_cast<linalg::FillOp>(op); fl && keep.contains(fl.getDpsInits()[0])) continue;
      for (Value v : op->getOperands())
        if (isN(v)) pieceOf(v);
      Operation *c = ib.clone(*op, map);
      if (auto g = dyn_cast<linalg::GenericOp>(c)) {
        // one loop over the piece: identity maps (d0) -> (d0), scalars (d0) -> ()
        SmallVector<AffineMap> maps;
        for (AffineMap m : g.getIndexingMapsArray())
          maps.push_back(m.getNumResults() == 0 ? AffineMap::get(1, 0, {}, ctx)
                                                : AffineMap::get(1, 0, {getAffineDimExpr(0, ctx)}, ctx));
        g.setIndexingMapsAttr(b.getAffineMapArrayAttr(maps));
        auto it = g.getNumReductionLoops() ? utils::IteratorType::reduction : utils::IteratorType::parallel;
        g.setIteratorTypesAttr(b.getArrayAttr({linalg::IteratorTypeAttr::get(ctx, it)}));
        if (g.getNumReductionLoops() && off > 0) g->setAttr("sahl.accumulate", UnitAttr::get(ctx));
      }
    }
  }
  for (Operation *op : llvm::reverse(work)) op->erase();
}

// row pieces [r0, r0 + rn) of the [R, T] and [R] operands
void rowPieces(func::FuncOp f, int64_t R, ArrayRef<int64_t> T, int64_t rn, bool rowByRow) {
  Block &blk = f.getBody().front();
  SmallVector<Operation *> work;
  for (Operation &op : blk)
    if (isWork(op)) work.push_back(&op);
  OpBuilder b(blk.getTerminator());
  auto byRow = [&](Value v) {
    auto mt = dyn_cast<MemRefType>(v.getType());
    return mt && mt.hasStaticShape() && mt.getRank() >= 1 && mt.getDimSize(0) == R &&
           (mt.getRank() == 1 || mt.getShape().drop_front() == T);
  };
  for (int64_t r0 = 0; r0 < R; r0 += rn) {
    int64_t n = std::min(rn, R - r0);
    auto sc = sahl::ScopeOp::create(b, f.getLoc(), /*keep=*/false, /*spad_from=*/IntegerAttr());
    Block *sb = &sc.getBody().emplaceBlock();
    OpBuilder ib = OpBuilder::atBlockEnd(sb);
    IRMapping map;
    for (Operation *op : work) {
      for (Value v : op->getOperands()) {
        if (!byRow(v) || map.contains(v)) continue;
        auto mt = cast<MemRefType>(v.getType());
        SmallVector<int64_t> shape(mt.getShape());
        shape[0] = n;
        Value r;
        if (!ddrRoot(v)) {
          r = memref::AllocOp::create(ib, v.getLoc(), MemRefType::get(shape, mt.getElementType()));
        } else {
          SmallVector<OpFoldResult> offs(mt.getRank(), ib.getIndexAttr(0)), sizes, strides(mt.getRank(), ib.getIndexAttr(1));
          offs[0] = ib.getIndexAttr(r0);
          for (int64_t s : shape) sizes.push_back(ib.getIndexAttr(s));
          r = memref::SubViewOp::create(ib, v.getLoc(), v, offs, sizes, strides);
        }
        map.map(v, r);
      }
      Operation *c = ib.clone(*op, map);
      if (rowByRow && isa<linalg::GenericOp>(c)) c->setAttr("sahl.row_by_row", UnitAttr::get(c->getContext()));
    }
  }
  for (Operation *op : llvm::reverse(work)) op->erase();
}

struct SahlTilePass : public PassWrapper<SahlTilePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SahlTilePass)
  SahlTilePass() = default;
  explicit SahlTilePass(const TargetConfig &c) : cfg(c), fromOptions(false) {}
  SahlTilePass(const SahlTilePass &o) : PassWrapper(o), cfg(o.cfg), fromOptions(o.fromOptions) {}
  StringRef getArgument() const override { return "iree-sahl-tile"; }
  StringRef getDescription() const override {
    return "Splits an element-wise dispatch too large for ACC into pieces (sahl.scope)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<sahl::SahlDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    if (fromOptions) {
      cfg.d = optD;
      cfg.spadBytes = optSpadKB * 1024;
      cfg.accBytes = optAccKB * 1024;
    }
    func::FuncOp f = getOperation();
    if (!f.getBody().front().getOps<sahl::KernelOp>().empty()) return;   // (kernels tile themselves)
    ::sa::Layout lay(uint32_t(cfg.d), uint32_t(cfg.spadBytes), uint32_t(cfg.accBytes));
    const int64_t cap = 2 * (lay.cbank - lay.scr), d = cfg.d;
    Decider dc{f, d};
    if (auto n = dc.pieceable(); n && dc.pieceWords(*n) > cap) {
      int64_t pieces = (dc.pieceWords(*n) + cap - 1) / cap;
      int64_t len = ((*n + pieces - 1) / pieces + d - 1) / d * d;
      flatPieces(f, *n, len);
      return;
    }
    if (auto rt = dc.rowPieceable()) {
      auto &[R, T] = *rt;
      auto [words, fixed] = dc.pieceWordsRows(R, T);
      if (words + fixed > cap) {
        // rows per piece: even (the per-row vectors' DMAs stay 8-byte aligned); if
        // not even two fit, two whose generics the lowering takes one row at a time
        // (a row's temps released after it; it reports ACC full if that does not fit)
        int64_t rn = R * std::max<int64_t>(cap - fixed, 0) / words / 2 * 2;
        bool rowByRow = rn < 2;
        if (rowByRow) rn = 2;
        if (rn >= R) return;
        rowPieces(f, R, T, rn, rowByRow);
      }
    }
  }

  TargetConfig cfg;
  bool fromOptions = true;
  Option<int64_t> optD{*this, "d", llvm::cl::desc("array size D"), llvm::cl::init(8)};
  Option<int64_t> optSpadKB{*this, "spad-kb", llvm::cl::desc("SPAD size (KB, two banks)"), llvm::cl::init(128)};
  Option<int64_t> optAccKB{*this, "acc-kb", llvm::cl::desc("ACC size (KB, two banks)"), llvm::cl::init(256)};
};

}  // namespace

std::unique_ptr<Pass> createSahlTilePass(const TargetConfig &config) { return std::make_unique<SahlTilePass>(config); }
void registerSahlTilePass() { PassRegistration<SahlTilePass>(); }

}  // namespace mlir::iree_compiler::sa

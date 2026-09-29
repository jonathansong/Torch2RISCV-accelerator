// sa-to-sahl (docs/iree_compiler_plan.md §8.4 step 5, C5.1): after
// bufferization a dispatch is linalg on memrefs that reads and writes binding
// subspans (DDR) directly. This pass makes the traffic explicit: every DDR
// operand of a linalg op is replaced by a local buffer (memref.alloc) with a
// sahl.load before (inputs, and outputs whose old value is read) and a
// sahl.store after (outputs); an identity copy into DDR becomes a sahl.store.
// Computation then only touches local buffers. Each to_i8 chain inside a
// body becomes one sahl.to_i8, a one-row scatter a sahl.scatter, each gather (a memref.load in a body) a
// sahl.gather with the form the lowering will use. Then the kernel matcher
// (SahlKernels.h) decides which operations are lowered together: each match
// (a linear layer or attention micro-kernel, a generic contraction) moves into a
// sahl.kernel at the position of its contraction (the anchor).
#include <functional>

#include "SahlDialect.h"
#include "SahlKernels.h"
#include "SahlPasses.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::iree_compiler::sa {

Value ddrRoot(Value v) {
  while (Operation *op = v.getDefiningOp()) {
    if (isa<IREE::HAL::InterfaceBindingSubspanOp>(op)) return v;
    if (auto s = dyn_cast<memref::SubViewOp>(op)) v = s.getSource();
    else if (auto c = dyn_cast<memref::CastOp>(op)) v = c.getSource();
    else if (auto c = dyn_cast<memref::CollapseShapeOp>(op)) v = c.getSrc();
    else if (auto c = dyn_cast<memref::ExpandShapeOp>(op)) v = c.getSrc();
    else return {};
  }
  return {};
}

namespace {

bool isIdentityCopy(linalg::GenericOp g) {
  if (g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1) return false;
  if (!llvm::all_of(g.getIndexingMapsArray(), [](AffineMap m) { return m.isIdentity(); })) return false;
  if (g.getNumReductionLoops()) return false;
  Block &b = g.getRegion().front();
  auto y = cast<linalg::YieldOp>(b.getTerminator());
  return b.getOperations().size() == 1 && y.getOperand(0) == b.getArgument(0);
}

// Whether the lowering takes this nest of three or more loops as it is: an
// element-wise nest with identity maps and no indices (one flat loop), or one
// whose loops merge into (row, inner) (the checks of Lowerer::generic).
bool loopsLowerable(linalg::GenericOp g) {
  int n = int(g.getNumLoops());
  auto iters = g.getIteratorTypesArray();
  Block &body = g.getRegion().front();
  bool noIndex = body.getOps<linalg::IndexOp>().empty();
  bool red = llvm::any_of(iters, [](utils::IteratorType t) { return t == utils::IteratorType::reduction; });
  auto maps = g.getIndexingMapsArray();
  if (llvm::all_of(maps, [](AffineMap m) { return m.isIdentity(); }) && !red && noIndex) return true;
  bool gathers = !body.getOps<memref::LoadOp>().empty();
  for (int split = n - 1; split >= 1; --split) {
    bool ok = !gathers;
    for (linalg::IndexOp ix : body.getOps<linalg::IndexOp>()) ok &= int(ix.getDim()) == n - 1 && split == n - 1;
    for (int L = 0; L < n; ++L)
      ok &= iters[L] == utils::IteratorType::parallel || (L == n - 1 && split == n - 1);
    for (AffineMap m : maps) {
      SmallVector<int64_t> dims;
      for (AffineExpr e : m.getResults()) {
        auto de = dyn_cast<AffineDimExpr>(e);
        if (!de) ok = false;
        else dims.push_back(de.getPosition());
      }
      auto seq = [&](int64_t from, int64_t to) {
        if (int64_t(dims.size()) != to - from) return false;
        for (int64_t i = 0; i < to - from; ++i)
          if (dims[i] != from + i) return false;
        return true;
      };
      ok &= seq(0, n) || seq(0, split) || seq(split, n) || dims.empty();
    }
    if (ok) return true;
  }
  return false;
}

// A slice of v at `at` along dimension `dim`, that dimension dropped.
Value rowSlice(OpBuilder &b, Location loc, Value v, int64_t dim, int64_t at) {
  auto mt = cast<MemRefType>(v.getType());
  SmallVector<OpFoldResult> offs(mt.getRank(), b.getIndexAttr(0)), sizes, strides(mt.getRank(), b.getIndexAttr(1));
  SmallVector<int64_t> shape;
  offs[dim] = b.getIndexAttr(at);
  for (int64_t i = 0; i < mt.getRank(); ++i) {
    sizes.push_back(b.getIndexAttr(i == dim ? 1 : mt.getDimSize(i)));
    if (i != dim) shape.push_back(mt.getDimSize(i));
  }
  auto rt = memref::SubViewOp::inferRankReducedResultType(shape, mt, offs, sizes, strides);
  return memref::SubViewOp::create(b, loc, cast<MemRefType>(rt), v, offs, sizes, strides);
}

// Prefill's per-token element-wise work over [M, ...] with more than two
// loops the lowering does not take (e.g. RoPE and the KV quantization of M
// rows: indices of the row, the pair-swap gather, tables per row) -> one
// generic per row over the other loops: each an operation of the decode
// form. Only an all-parallel nest whose outer loop is static (<= 16) and
// whose operands are DDR views (sliced per row), untouched by the row, or
// (the result) a local buffer copied whole into DDR right after.
void unrollOuterRows(func::FuncOp f) {
  SmallVector<linalg::GenericOp> todo;
  for (auto g : f.getBody().front().getOps<linalg::GenericOp>())
    if (g.getNumLoops() >= 3 && g.getNumReductionLoops() == 0 && g.getNumDpsInits() == 1 && !loopsLowerable(g))
      todo.push_back(g);
  MLIRContext *ctx = f.getContext();
  for (linalg::GenericOp g : todo) {
    const int n = int(g.getNumLoops());
    auto maps = g.getIndexingMapsArray();
    // each operand: the result position of loop 0 in its map (-1: none)
    SmallVector<int64_t> at;
    int64_t R = -1;
    bool ok = true;
    for (auto [i, opnd] : llvm::enumerate(g->getOperands())) {
      auto mt = dyn_cast<MemRefType>(opnd.getType());
      int64_t p = -1;
      for (auto [r, e] : llvm::enumerate(maps[i].getResults())) {
        if (!e.isFunctionOfDim(0)) continue;
        auto de = dyn_cast<AffineDimExpr>(e);
        if (!de || p >= 0) ok = false;
        p = int64_t(r);
      }
      if (p >= 0 && mt && mt.hasStaticShape()) R = mt.getDimSize(p);
      at.push_back(p);
    }
    if (!ok || R < 2 || R > 16) continue;
    // the result: a DDR view, or a local buffer whose only other use is its identity copy into DDR
    Value out = g.getDpsInits()[0];
    linalg::GenericOp copy;
    if (!ddrRoot(out)) {
      if (!out.getDefiningOp<memref::AllocOp>() || at.back() != 0) continue;
      for (Operation *u : out.getUsers()) {
        if (u == g.getOperation()) continue;
        auto c = dyn_cast<linalg::GenericOp>(u);
        if (!c || copy || !isIdentityCopy(c) || c.getDpsInputs()[0] != out || !ddrRoot(c.getDpsInits()[0])) ok = false;
        else copy = c;
      }
      if (!ok || !copy || !g->isBeforeInBlock(copy)) continue;
    }
    for (auto [i, opnd] : llvm::enumerate(g->getOperands()))
      if (at[i] >= 0 && Value(opnd) != out && !ddrRoot(opnd)) ok = false;   // inputs sliced per row: DDR only
    // the gathers (memref.load in the body): loop 0's index at most as a whole index, sliced with the row
    Block &body = g.getRegion().front();
    for (auto ld : body.getOps<memref::LoadOp>()) {
      for (Value ix : ld.getIndices()) {
        std::function<bool(Value)> uses0 = [&](Value v) -> bool {
          if (auto li = v.getDefiningOp<linalg::IndexOp>()) return li.getDim() == 0;
          Operation *o = v.getDefiningOp();
          return o && o->getBlock() == &body && llvm::any_of(o->getOperands(), uses0);
        };
        auto li = ix.getDefiningOp<linalg::IndexOp>();
        if (uses0(ix) && !(li && li.getDim() == 0)) ok = false;
      }
      if (!ddrRoot(ld.getMemRef())) ok = false;
    }
    if (!ok) continue;
    // the maps without loop 0 (loops 1.. renumbered)
    SmallVector<AffineExpr> down{getAffineConstantExpr(0, ctx)};   // d0 (appears only at at[i]), d_k -> d_(k-1)
    for (int k = 1; k < n; ++k) down.push_back(getAffineDimExpr(k - 1, ctx));
    SmallVector<AffineMap> maps2;
    for (auto [i, m] : llvm::enumerate(maps)) {
      SmallVector<AffineExpr> rs;
      for (auto [r, e] : llvm::enumerate(m.getResults()))
        if (int64_t(r) != at[i]) rs.push_back(e.replaceDims(down));
      maps2.push_back(AffineMap::get(n - 1, 0, rs, ctx));
    }
    SmallVector<utils::IteratorType> iters2(n - 1, utils::IteratorType::parallel);
    OpBuilder b(copy ? copy.getOperation() : g.getOperation());
    Location loc = g.getLoc();
    Value dst = copy ? copy.getDpsInits()[0] : out;
    for (int64_t r = 0; r < R; ++r) {
      SmallVector<Value> ins;
      for (auto [i, v] : llvm::enumerate(g.getDpsInputs()))
        ins.push_back(at[i] >= 0 ? rowSlice(b, loc, v, at[i], r) : v);
      Value o;
      if (copy) {
        auto mt = cast<MemRefType>(out.getType());
        o = memref::AllocOp::create(b, loc, MemRefType::get(mt.getShape().drop_front(), mt.getElementType()));
      } else {
        o = rowSlice(b, loc, out, at.back(), r);
      }
      auto g2 = linalg::GenericOp::create(b, loc, TypeRange{}, ins, ValueRange{o}, maps2, iters2);
      IRMapping map;
      g.getRegion().cloneInto(&g2.getRegion(), map);
      Block &nb = g2.getRegion().front();
      SmallVector<linalg::IndexOp> ixs(nb.getOps<linalg::IndexOp>());
      for (linalg::IndexOp ix : ixs) {
        OpBuilder ib(ix);
        if (ix.getDim() == 0) {
          Value c = arith::ConstantIndexOp::create(ib, ix.getLoc(), r);
          // a gather indexed by the row: its source sliced at this row
          SmallVector<memref::LoadOp> lds;
          for (Operation *u : ix->getUsers())
            if (auto ld = dyn_cast<memref::LoadOp>(u)) lds.push_back(ld);
          for (memref::LoadOp ld : lds) {
            SmallVector<Value> idx;
            int64_t dim = -1;
            for (auto [k, v] : llvm::enumerate(ld.getIndices())) {
              if (v == ix.getResult() && dim < 0) dim = int64_t(k);
              else idx.push_back(v);
            }
            OpBuilder ob(g2);
            Value src = rowSlice(ob, loc, ld.getMemRef(), dim, r);
            OpBuilder lb(ld);
            auto nl = memref::LoadOp::create(lb, ld.getLoc(), src, idx);
            ld.replaceAllUsesWith(nl.getResult());
            ld.erase();
          }
          ix.replaceAllUsesWith(c);
          ix.erase();
        } else {
          ix.setDim(ix.getDim() - 1);
        }
      }
      if (copy) {
        Value d = rowSlice(b, loc, dst, 0, r);
        SmallVector<AffineMap> idm(2, AffineMap::getMultiDimIdentityMap(n - 1, ctx));
        auto c2 = linalg::GenericOp::create(b, loc, TypeRange{}, ValueRange{o}, ValueRange{d}, idm, iters2,
                                            [](OpBuilder &bb, Location l, ValueRange a) {
                                              linalg::YieldOp::create(bb, l, a[0]);
                                            });
        (void)c2;
      }
    }
    if (copy) copy.erase();
    g.erase();
    if (copy && out.use_empty()) out.getDefiningOp()->erase();
  }
}

// An iree_linalg_ext.scatter that overwrites one row along dimension 0 (the
// KV cache update) -> sahl.scatter; others stay (the lowering rejects them).
void replaceScatters(func::FuncOp f) {
  SmallVector<IREE::LinalgExt::ScatterOp> scs;
  f.walk([&](IREE::LinalgExt::ScatterOp sc) { scs.push_back(sc); });
  for (auto sc : scs) {
    auto uT = dyn_cast<MemRefType>(sc.getUpdates().getType());
    auto oT = dyn_cast<MemRefType>(sc.getOriginal().getType());
    if (!uT || !oT || sc.getDimensionMap() != ArrayRef<int64_t>{0} || uT.getDimSize(0) != 1 || !uT.hasStaticShape() ||
        !oT.hasStaticShape() || uT.getElementType() != oT.getElementType())
      continue;
    Block &b = sc.getRegion().front();
    auto y = dyn_cast<IREE::LinalgExt::YieldOp>(b.getTerminator());
    if (!y || y.getOperand(0) != b.getArgument(0)) continue;
    OpBuilder bld(sc);
    sahl::ScatterOp::create(bld, sc.getLoc(), sc.getUpdates(), sc.getIndices(), sc.getOriginal());
    sc.erase();
  }
}

// Each to_i8 chain inside a linalg body -> one sahl.to_i8.
void replaceToI8(func::FuncOp f) {
  SmallVector<arith::FPToSIOp> fps;
  f.walk([&](arith::FPToSIOp fp) {
    if (fp->getParentOfType<linalg::GenericOp>()) fps.push_back(fp);
  });
  for (arith::FPToSIOp fp : fps) {
    SmallVector<Operation *> chain;
    Value x = matchToI8(fp, chain);
    if (!x) continue;
    OpBuilder b(fp);
    auto t = sahl::ToI8Op::create(b, fp.getLoc(), b.getI8Type(), x);
    fp.getResult().replaceAllUsesWith(t.getResult());
    fp.erase();
    chain.pop_back();                             // (the fptosi)
    for (Operation *o : chain)                    // consumers first
      if (o->use_empty()) o->erase();
  }
}

// Each memref.load inside a linalg body (a gather) -> sahl.gather with its form.
void replaceGathers(func::FuncOp f, const TargetConfig &cfg) {
  SmallVector<linalg::GenericOp> gs;
  f.walk([&](linalg::GenericOp g) { gs.push_back(g); });
  for (linalg::GenericOp g : gs) {
    Block &body = g.getRegion().front();
    SmallVector<memref::LoadOp> lds(body.getOps<memref::LoadOp>());
    if (lds.empty()) continue;
    int nloops = 0;
    auto dom = gatherDomain(g, cfg, nloops);
    bool hasScalar = false;
    for (Value in : g.getDpsInputs())
      if (auto mt = dyn_cast<MemRefType>(in.getType()); mt && mt.getElementType().isInteger(64)) hasScalar = true;
    for (memref::LoadOp ld : lds) {
      OpBuilder b(ld);
      auto ga = sahl::GatherOp::create(b, ld.getLoc(), ld.getType(), ld.getMemRef(), ld.getIndices(), "none");
      ld.getResult().replaceAllUsesWith(ga.getResult());
      ld.erase();
      if (!dom) continue;
      SmallVector<SmallVector<int64_t>> d0 = nloops ? *dom : SmallVector<SmallVector<int64_t>>{{}};
      GatherForm gf = gatherForm(ga, d0, nloops, hasScalar, cfg.d, g.getNumLoops());
      if (!gf.kind.empty()) ga.setKind(gf.kind);
    }
  }
}

// Moves the operations of each match into a sahl.kernel at its anchor. The
// operations after the anchor move up: the definitions they use that come
// between (allocations, dims, constants, views: no commands) move before the
// kernel. The lowering emits a kernel where its anchor was, so the commands do
// not change.
LogicalResult groupKernels(func::FuncOp f, const TargetConfig &cfg) {
  KernelMatcher km(cfg);
  km.matchAll(f);
  Block &blk = f.getBody().front();
  SmallVector<std::pair<Operation *, StringRef>> anchors;
  for (Operation &op : blk) {
    if (km.linears.count(&op)) anchors.push_back({&op, "linear"});
    else if (km.attns.count(&op)) anchors.push_back({&op, "attention"});
    else if (km.contracts.count(&op)) anchors.push_back({&op, "contraction"});
  }
  if (anchors.size() != km.covers.size()) return f.emitError("sa-to-sahl: a kernel anchor is not at the top level");
  for (auto [a, kind] : anchors) {
    SmallVector<Operation *> ops(km.covers[a]);
    llvm::sort(ops, [](Operation *x, Operation *y) { return x->isBeforeInBlock(y); });
    OpBuilder b(a);
    auto k = sahl::KernelOp::create(b, a->getLoc(), kind);
    Block *kb = &k.getBody().emplaceBlock();
    std::function<bool(Value)> hoist = [&](Value v) -> bool {
      Operation *p = v.getDefiningOp();
      if (!p || p->getBlock() != &blk || p->isBeforeInBlock(k)) return true;
      if (!isa<memref::AllocOp, memref::DimOp, memref::SubViewOp, memref::CastOp, memref::ExpandShapeOp,
               memref::CollapseShapeOp, arith::ConstantOp>(p))
        return false;
      for (Value o : p->getOperands())
        if (!hoist(o)) return false;
      p->moveBefore(k);
      return true;
    };
    for (Operation *o : ops) {
      if (o->getBlock() != &blk || o->getNumResults())
        return o->emitError("sa-to-sahl: a kernel operation not at the top level or with results");
      bool ok = true;
      o->walk([&](Operation *x) {
        for (Value v : x->getOperands()) ok &= hoist(v);
      });
      if (!ok) return o->emitError("sa-to-sahl: a kernel operation uses a value computed after the kernel");
    }
    for (Operation *o : ops) o->moveBefore(kb, kb->end());
    a->setAttr("sahl.anchor", UnitAttr::get(f.getContext()));
  }
  return success();
}

struct SaToSahlPass : public PassWrapper<SaToSahlPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SaToSahlPass)
  SaToSahlPass() = default;
  explicit SaToSahlPass(const TargetConfig &c) : cfg(c), fromOptions(false) {}
  SaToSahlPass(const SaToSahlPass &o) : PassWrapper(o), cfg(o.cfg), fromOptions(o.fromOptions) {}
  StringRef getArgument() const override { return "iree-sa-to-sahl"; }
  StringRef getDescription() const override {
    return "Makes DDR traffic explicit: linalg on local buffers, sahl.load / sahl.store to and from bindings";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<sahl::SahlDialect, memref::MemRefDialect>();
  }

  TargetConfig cfg;
  bool fromOptions = true;
  Option<int64_t> optD{*this, "d", llvm::cl::desc("array size D"), llvm::cl::init(8)};
  Option<int64_t> optSpadKB{*this, "spad-kb", llvm::cl::desc("SPAD size (KB, two banks)"), llvm::cl::init(128)};
  Option<int64_t> optAccKB{*this, "acc-kb", llvm::cl::desc("ACC size (KB, two banks)"), llvm::cl::init(256)};
  Option<std::string> optUkernels{*this, "ukernels", llvm::cl::desc("micro-kernels: all, none or a list"),
                                  llvm::cl::init("all")};

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    unrollOuterRows(f);
    llvm::DenseMap<Value, Value> loaded;          // DDR view -> its local copy
    auto localFor = [&](OpBuilder &b, Location loc, Value ddr) -> Value {
      auto mt = cast<MemRefType>(ddr.getType());
      SmallVector<Value> dyn;                    // dynamic sizes: those of the DDR view
      for (int64_t i = 0; i < mt.getRank(); ++i)
        if (mt.isDynamicDim(i)) dyn.push_back(memref::DimOp::create(b, loc, ddr, i));
      return memref::AllocOp::create(b, loc, MemRefType::get(mt.getShape(), mt.getElementType()), dyn);
    };
    SmallVector<Operation *> ops;
    for (Operation &op : f.getBody().front()) ops.push_back(&op);
    for (Operation *op : ops) {
      if (auto c = dyn_cast<memref::CopyOp>(op)) {
        OpBuilder b(c);
        bool srcDdr = bool(ddrRoot(c.getSource())), dstDdr = bool(ddrRoot(c.getTarget()));
        if (srcDdr && !dstDdr) sahl::LoadOp::create(b, c.getLoc(), c.getSource(), c.getTarget());
        else if (!srcDdr && dstDdr) sahl::StoreOp::create(b, c.getLoc(), c.getSource(), c.getTarget());
        else {
          c.emitError("sa-to-sahl: copy between two DDR or two local buffers");
          return signalPassFailure();
        }
        c.erase();
        continue;
      }
      // a copy of a buffer onto itself (the write-back of an in-place update): nothing to do
      if (auto g = dyn_cast<linalg::GenericOp>(op); g && isIdentityCopy(g) && g.getDpsInputs()[0] == g.getDpsInits()[0]) {
        g.erase();
        continue;
      }
      auto dps = dyn_cast<DestinationStyleOpInterface>(op);
      if (!dps || !isa<linalg::LinalgOp>(op)) continue;
      // attention: the batch matmul and the extension of its cache operand stay
      // as they are (lowered together, reading the cache in DDR)
      if (isa<linalg::BatchMatmulOp>(op)) continue;
      if (auto g = dyn_cast<linalg::GenericOp>(op);
          g && g.getNumDpsInits() == 1 && llvm::any_of(g.getDpsInits()[0].getUsers(), [](Operation *u) {
            return isa<linalg::BatchMatmulOp>(u);
          }))
        continue;
      if (auto g = dyn_cast<linalg::GenericOp>(op); g && isIdentityCopy(g) && ddrRoot(g.getDpsInits()[0]) &&
                                                    !ddrRoot(g.getDpsInputs()[0])) {
        OpBuilder b(g);
        sahl::StoreOp::create(b, g.getLoc(), g.getDpsInputs()[0], g.getDpsInits()[0]);
        g.erase();
        continue;
      }
      OpBuilder b(op);
      for (OpOperand *in : dps.getDpsInputOperands()) {
        Value v = in->get();
        if (!isa<MemRefType>(v.getType()) || !ddrRoot(v)) continue;
        // i64 scalars (positions) stay in DDR: the lowering reads them through LDPARAM
        if (cast<MemRefType>(v.getType()).getElementType().isInteger(64)) continue;
        Value l = loaded.lookup(v);
        if (!l) {
          l = localFor(b, op->getLoc(), v);
          if (!l) {
            op->emitError("sa-to-sahl: DDR operand with a dynamic shape");
            return signalPassFailure();
          }
          sahl::LoadOp::create(b, op->getLoc(), v, l);
          loaded[v] = l;
        }
        in->set(l);
      }
      auto lop = cast<linalg::LinalgOp>(op);
      for (OpOperand &init : dps.getDpsInitsMutable()) {
        Value v = init.get();
        if (!ddrRoot(v)) continue;
        Value l = localFor(b, op->getLoc(), v);
        if (!l) {
          op->emitError("sa-to-sahl: DDR result with a dynamic shape");
          return signalPassFailure();
        }
        if (lop.payloadUsesValueFromOperand(&init)) sahl::LoadOp::create(b, op->getLoc(), v, l);
        init.set(l);
        OpBuilder a(op->getContext());
        a.setInsertionPointAfter(op);
        sahl::StoreOp::create(a, op->getLoc(), l, v);
        loaded.erase(v);
      }
    }
    if (fromOptions) {                         // iree-opt: the target configuration from the pass options
      cfg.d = optD;
      cfg.spadBytes = optSpadKB * 1024;
      cfg.accBytes = optAccKB * 1024;
      cfg.ukernels = optUkernels;
    }
    replaceScatters(f);
    replaceToI8(f);
    replaceGathers(f, cfg);
    if (failed(groupKernels(f, cfg))) return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createSaToSahlPass(const TargetConfig &config) {
  return std::make_unique<SaToSahlPass>(config);
}
void registerSaToSahlPass() { PassRegistration<SaToSahlPass>(); }

}  // namespace mlir::iree_compiler::sa

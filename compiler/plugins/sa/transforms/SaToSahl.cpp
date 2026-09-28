// sa-to-sahl (docs/iree_compiler_plan.md §8.4 step 5, C5.1): after
// bufferization a dispatch is linalg on memrefs that reads and writes binding
// subspans (DDR) directly. This pass makes the traffic explicit: every DDR
// operand of a linalg op is replaced by a local buffer (memref.alloc) with a
// sahl.load before (inputs, and outputs whose old value is read) and a
// sahl.store after (outputs); an identity copy into DDR becomes a sahl.store.
// Computation then only touches local buffers. Then the kernel matcher
// (SahlKernels.h) decides which operations are lowered together: each match
// (a linear layer or attention micro-kernel, a generic contraction) moves into a
// sahl.kernel at the position of its contraction (the anchor).
#include <functional>

#include "SahlDialect.h"
#include "SahlKernels.h"
#include "SahlPasses.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
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
    if (failed(groupKernels(f, cfg))) return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createSaToSahlPass(const TargetConfig &config) {
  return std::make_unique<SaToSahlPass>(config);
}
void registerSaToSahlPass() { PassRegistration<SaToSahlPass>(); }

}  // namespace mlir::iree_compiler::sa

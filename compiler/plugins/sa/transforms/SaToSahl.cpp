// sa-to-sahl (docs/iree_compiler_plan.md §8.4 step 5, C5.1): after
// bufferization a dispatch is linalg on memrefs that reads and writes binding
// subspans (DDR) directly. This pass makes the traffic explicit: every DDR
// operand of a linalg op is replaced by a local buffer (memref.alloc) with a
// sahl.load before (inputs, and outputs whose old value is read) and a
// sahl.store after (outputs); an identity copy into DDR becomes a sahl.store.
// Computation then only touches local buffers.
#include "SahlDialect.h"
#include "SahlPasses.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
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

struct SaToSahlPass : public PassWrapper<SaToSahlPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SaToSahlPass)
  StringRef getArgument() const override { return "iree-sa-to-sahl"; }
  StringRef getDescription() const override {
    return "Makes DDR traffic explicit: linalg on local buffers, sahl.load / sahl.store to and from bindings";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<sahl::SahlDialect, memref::MemRefDialect>();
  }

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
      auto dps = dyn_cast<DestinationStyleOpInterface>(op);
      if (!dps || !isa<linalg::LinalgOp>(op)) continue;
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
  }
};

}  // namespace

std::unique_ptr<Pass> createSaToSahlPass() { return std::make_unique<SaToSahlPass>(); }
void registerSaToSahlPass() { PassRegistration<SaToSahlPass>(); }

}  // namespace mlir::iree_compiler::sa

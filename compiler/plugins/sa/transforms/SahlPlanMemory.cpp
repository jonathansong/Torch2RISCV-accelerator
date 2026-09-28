// sahl-plan-memory (docs/iree_compiler_plan.md §8.4 step 7, §8.14 C8 R4):
// each local buffer (memref.alloc) gets its place and layout as attributes,
// which sahl-to-sahw follows:
//   sa.mem     "spad_a" (int8: EX operands, the VE's int8 output) or "acc"
//              (fp32, int32, i64 as two int32 lanes);
//   sa.layout  "bcast" (one word per element, the value in every lane: the
//              result of a reduction, read back by DIV / LIN modes), "rows"
//              (a dynamic innermost length: rows of max_dynamic elements),
//              or "packed" (D elements per word).
// Addresses stay the lowering's: in order at first use, released after a
// piece, a chunk or a row (SahlLocalMemory.h). A planner with liveness would
// change them (and the bank conflicts the scoreboard sees): measured on the
// board first.
#include "SahlDialect.h"
#include "SahlPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::iree_compiler::sa {
namespace {

struct SahlPlanMemoryPass : public PassWrapper<SahlPlanMemoryPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SahlPlanMemoryPass)
  StringRef getArgument() const override { return "iree-sahl-plan-memory"; }
  StringRef getDescription() const override {
    return "Places each local buffer (sa.mem) and chooses its layout (sa.layout)";
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    // the results of reductions: broadcast words
    llvm::DenseSet<Value> bcast;
    f.walk([&](linalg::GenericOp g) {
      if (g.getNumReductionLoops())
        for (Value v : g.getDpsInits()) bcast.insert(v);
    });
    f.walk([&](sahl::ReserveOp r) {
      if (r.getBcast()) bcast.insert(r.getBuffer());
    });
    Builder b(f.getContext());
    f.walk([&](memref::AllocOp a) {
      auto mt = a.getType();
      a->setAttr("sa.mem", b.getStringAttr(mt.getElementType().isInteger(8) ? "spad_a" : "acc"));
      a->setAttr("sa.layout", b.getStringAttr(bcast.contains(a.getResult()) ? "bcast"
                                              : mt.hasStaticShape()          ? "packed"
                                                                             : "rows"));
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createSahlPlanMemoryPass() { return std::make_unique<SahlPlanMemoryPass>(); }
void registerSahlPlanMemoryPass() { PassRegistration<SahlPlanMemoryPass>(); }

}  // namespace mlir::iree_compiler::sa

// sahl-schedule (docs/iree_compiler_plan.md §8.4 steps 8-9, §8.14 C8 R5): the
// schedule of each linear micro-kernel as attributes of its sahl.kernel,
// which sahl-to-sahw follows (linearSchedule in SahlKernels.h):
//   chunk_tiles  output tiles per chunk (its weights in one SPAD_B bank; the
//                next chunk's are loaded into the other bank while it computes);
//   loop         decode's form: a LOOP_END loop over pairs of chunks.
// The generic contraction's chunks and K blocks depend on where the lowering
// places x (the strip's source range), so it chooses them itself.
#include "SahlDialect.h"
#include "SahlKernels.h"
#include "SahlPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::iree_compiler::sa {
namespace {

struct SahlSchedulePass : public PassWrapper<SahlSchedulePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SahlSchedulePass)
  SahlSchedulePass() = default;
  explicit SahlSchedulePass(const TargetConfig &c) : cfg(c), fromOptions(false) {}
  SahlSchedulePass(const SahlSchedulePass &o) : PassWrapper(o), cfg(o.cfg), fromOptions(o.fromOptions) {}
  StringRef getArgument() const override { return "iree-sahl-schedule"; }
  StringRef getDescription() const override { return "Chooses the schedule of each linear micro-kernel (sahl.kernel attributes)"; }

  void runOnOperation() override {
    if (fromOptions) {
      cfg.d = optD;
      cfg.spadBytes = optSpadKB * 1024;
      cfg.accBytes = optAccKB * 1024;
    }
    func::FuncOp f = getOperation();
    ::sa::Layout lay(uint32_t(cfg.d), uint32_t(cfg.spadBytes), uint32_t(cfg.accBytes));
    Builder b(f.getContext());
    f.walk([&](sahl::KernelOp k) {
      if (k.getKind() != "linear") return;
      for (Operation &o : k.getBody().front()) {
        auto g = dyn_cast<linalg::GenericOp>(&o);
        if (!g || !o.hasAttr("sahl.anchor")) continue;
        KernelMatcher km(cfg);
        if (!km.matchLinear(g)) return;
        const LinearPlan &p = km.linears[g.getOperation()];
        KernelSchedule ks = linearSchedule(p, lay, cfg.d);
        k->setAttr("chunk_tiles", b.getI64IntegerAttr(ks.chunkTiles));
        if (!p.rows) k->setAttr("loop", b.getBoolAttr(ks.loop));
      }
    });
  }

  TargetConfig cfg;
  bool fromOptions = true;
  Option<int64_t> optD{*this, "d", llvm::cl::desc("array size D"), llvm::cl::init(8)};
  Option<int64_t> optSpadKB{*this, "spad-kb", llvm::cl::desc("SPAD size (KB, two banks)"), llvm::cl::init(128)};
  Option<int64_t> optAccKB{*this, "acc-kb", llvm::cl::desc("ACC size (KB, two banks)"), llvm::cl::init(256)};
};

}  // namespace

std::unique_ptr<Pass> createSahlSchedulePass(const TargetConfig &config) {
  return std::make_unique<SahlSchedulePass>(config);
}
void registerSahlSchedulePass() { PassRegistration<SahlSchedulePass>(); }

}  // namespace mlir::iree_compiler::sa

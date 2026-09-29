// iree-sa-codegen (docs/iree_compiler_plan.md §8.4): every export of an sa
// executable variant through the C5 pipeline (SahlPasses.h: bufferize,
// sa-to-sahl, sahl-to-sahw, sahw-fuse-ve), run on a copy of its function in a
// module nested in the inner module; the resulting sahw.template moves into
// the inner module (serializeExecutable writes them as sa-desc).
#include "../target/DescList.h"
#include "SahlPasses.h"
#include "SahwPasses.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::sa {
namespace {

struct SaCodegenPass : public PassWrapper<SaCodegenPass, OperationPass<IREE::HAL::ExecutableVariantOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SaCodegenPass)
  explicit SaCodegenPass(bool allowUnsupported = false, std::string ukernels = "all", bool report = false,
                         bool hostFallback = false, std::string hostDispatches = "")
      : allowUnsupported(allowUnsupported), ukernels(std::move(ukernels)), report(report), hostFallback(hostFallback),
        hostDispatches(std::move(hostDispatches)) {}
  SaCodegenPass(const SaCodegenPass &o)
      : PassWrapper(o), allowUnsupported(o.allowUnsupported), ukernels(o.ukernels), report(o.report),
        hostFallback(o.hostFallback), hostDispatches(o.hostDispatches) {}

  StringRef getArgument() const override { return "iree-sa-codegen"; }
  StringRef getDescription() const override {
    return "Generates every export through the C5 pipeline into a sahw.template";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<sahw::SahwDialect, arith::ArithDialect>();
    OpPassManager pm(ModuleOp::getOperationName());
    buildSahlPipeline(pm, TargetConfig());
    pm.getDependentDialects(registry);
  }

  // the C5 pipeline on a copy of the function; its template moves into the inner module
  bool compile(ModuleOp inner, FunctionOpInterface func, const TargetConfig &cfg, int64_t bindings,
               int64_t constants, std::string &why) {
    OpBuilder mb = OpBuilder::atBlockEnd(inner.getBody());
    auto tmp = ModuleOp::create(mb, func.getLoc());
    OpBuilder tb = OpBuilder::atBlockEnd(tmp.getBody());
    tb.clone(*func.getOperation());
    OpPassManager pm(ModuleOp::getOperationName());
    buildSahlPipeline(pm, cfg);
    bool ok;
    {
      ScopedDiagnosticHandler h(func.getContext(), [&](Diagnostic &dg) {
        if (why.empty()) why = dg.str();
        return success();
      });
      ok = succeeded(runPipeline(pm, tmp));
    }
    sahw::TemplateOp t;
    if (ok)
      for (auto x : tmp.getOps<sahw::TemplateOp>()) t = x;
    if (ok && t) {
      t->moveBefore(tmp);
      Builder b(t.getContext());
      t.setBindingsAttr(b.getI64IntegerAttr(bindings));
      t.setConstantsAttr(b.getI64IntegerAttr(constants));
    } else if (ok) {
      why = "no template";
      ok = false;
    }
    tmp.erase();
    return ok;
  }

  void runOnOperation() override {
    auto variant = getOperation();
    auto inner = variant.getInnerModule();
    if (!inner) return;
    TargetConfig cfg = TargetConfig::fromAttr(variant.getTarget().getConfiguration());
    cfg.ukernels = ukernels;
    if (std::string why = cfg.invalid(); !why.empty()) {
      variant.emitError() << "sa: invalid target configuration: " << why;
      return signalPassFailure();
    }
    for (auto exportOp : variant.getBlock().getOps<IREE::HAL::ExecutableExportOp>()) {
      auto func = inner.lookupSymbol<FunctionOpInterface>(exportOp.getSymName());
      if (!func) {
        exportOp.emitError() << "sa: no function for this export";
        return signalPassFailure();
      }
      auto layout = exportOp.getLayout();
      int64_t bindings = int64_t(layout.getBindings().size()), constants = int64_t(layout.getConstants());
      std::string why;
      bool forced = false;
      if (hostFallback && !hostDispatches.empty()) {
        SmallVector<StringRef> pats;
        StringRef(hostDispatches).split(pats, ',', -1, false);
        for (StringRef p : pats) forced |= exportOp.getSymName().contains(p);
        if (forced) why = "--iree-sa-host-dispatches";
      }
      if (!forced && compile(inner, func, cfg, bindings, constants, why)) {
        if (report) llvm::errs() << "sa codegen: " << exportOp.getSymName() << ": ok\n";
        continue;
      }
      if (report) llvm::errs() << "sa codegen: " << exportOp.getSymName() << ": failed (" << why << ")\n";
      if (hostFallback) {
        // the runtime takes the host (VMVX) variant: this one's condition is false
        if (!variant.getConditionOp()) {
          OpBuilder cb(variant.getContext());
          variant.createConditionOp(cb);
          OpBuilder rb = OpBuilder::atBlockBegin(&variant.getConditionOp().getBody().front());
          Value no = arith::ConstantIntOp::create(rb, variant.getLoc(), 0, 1);
          IREE::HAL::ReturnOp::create(rb, variant.getLoc(), ValueRange{no});
        }
        if (report) llvm::errs() << "sa codegen: " << exportOp.getSymName() << ": on the host (" << why << ")\n";
      } else if (!allowUnsupported) {
        auto diag = func.emitError() << "sa target: cannot compile dispatch '" << exportOp.getSymName()
                                     << "': " << why;
        diag.attachNote() << "dispatch:\n" << *func.getOperation();
        return signalPassFailure();
      }
      if (!hostFallback)
        func.emitWarning() << "sa target: dispatch '" << exportOp.getSymName() << "' unsupported: " << why;
      // an export that faults if run: an LD past the end of ACC
      OpBuilder b = OpBuilder::atBlockEnd(inner.getBody());
      Location loc = func.getLoc();
      auto t = sahw::TemplateOp::create(b, loc, exportOp.getSymName(), bindings, constants, 0, true);
      OpBuilder tb = OpBuilder::atBlockEnd(&t.getRegion().emplaceBlock());
      auto body = sahw::BodyOp::create(tb, loc);
      OpBuilder bb = OpBuilder::atBlockEnd(&body.getRegion().emplaceBlock());
      sahw::LdOp::create(bb, loc, Value(), 0, int64_t(::sa::acc(0xFFFF)), 4, 64, 64, 0, ValueRange{});
    }
  }

  bool allowUnsupported;
  std::string ukernels;
  bool report;
  bool hostFallback;
  std::string hostDispatches;
};

}  // namespace

// Link time (the sa backend's linking pipeline, before the other backends'):
// the sa variants of dispatches left to the host fallback are removed.
struct SaDropHostVariantsPass : public PassWrapper<SaDropHostVariantsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SaDropHostVariantsPass)
  StringRef getArgument() const override { return "iree-sa-drop-host-variants"; }
  StringRef getDescription() const override { return "Removes the sa variants of dispatches the host runs"; }
  void runOnOperation() override {
    SmallVector<IREE::HAL::ExecutableVariantOp> drop;
    getOperation().walk([&](IREE::HAL::ExecutableVariantOp v) {
      if (v->hasAttr("sa.host_fallback")) drop.push_back(v);
    });
    for (auto v : drop) {
      auto exe = v->getParentOfType<IREE::HAL::ExecutableOp>();
      if (llvm::range_size(exe.getOps<IREE::HAL::ExecutableVariantOp>()) < 2) {
        v.emitError() << "sa: host fallback without a host variant";
        return signalPassFailure();
      }
      v.erase();
    }
  }
};

std::unique_ptr<Pass> createSaDropHostVariantsPass() { return std::make_unique<SaDropHostVariantsPass>(); }

std::unique_ptr<Pass> createSaCodegenPass(bool allowUnsupported, const std::string &ukernels, bool report,
                                          bool hostFallback, const std::string &hostDispatches) {
  return std::make_unique<SaCodegenPass>(allowUnsupported, ukernels, report, hostFallback, hostDispatches);
}
void registerSaCodegenPass() { PassRegistration<SaCodegenPass>(); }

}  // namespace mlir::iree_compiler::sa

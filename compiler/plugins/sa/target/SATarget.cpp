// sa HAL target for iree-compile (docs/iree_compiler_plan.md §6).
//
// Registers
//   #hal.device.target<"sa", ...>                     the PYNQ-Z1 accelerator
//   #hal.executable.target<"sa", "sa-desc-v1", {...}>  descriptor-list templates
// with iree-compile. Stage C2 fills in the translation (template matching and
// descriptor generation) and serializeExecutable; until then compiling a
// dispatch for sa stops with a clear error.

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

struct SAOptions {
  // Array size of the accelerator (the .w8a8 packing and the templates depend on it).
  int d = 8;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("sa HAL target (PYNQ-Z1 accelerator)");
    binder.opt<int>("iree-sa-d", d, llvm::cl::cat(category),
                    llvm::cl::desc("Array size D of the accelerator (8 or 16)."));
  }
};

static IREE::HAL::ExecutableTargetAttr
getSAExecutableTarget(MLIRContext *context, const SAOptions &options) {
  Builder b(context);
  SmallVector<NamedAttribute> config;
  config.emplace_back(b.getStringAttr("d"), b.getI64IntegerAttr(options.d));
  return b.getAttr<IREE::HAL::ExecutableTargetAttr>(
      b.getStringAttr("sa"), b.getStringAttr("sa-desc-v1"),
      b.getDictionaryAttr(config));
}

// #hal.device.target<"sa", ...>: executables for it come from the sa backend.
class SADevice final : public TargetDevice {
public:
  explicit SADevice(const SAOptions &options) : options(options) {}

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const override {
    Builder b(context);
    auto configAttr = b.getDictionaryAttr({});
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    if (auto backend = targetRegistry.getTargetBackend("sa")) {
      backend->getDefaultExecutableTargets(context, "sa", configAttr,
                                           executableTargetAttrs);
    }
    return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("sa"),
                                            configAttr, executableTargetAttrs);
  }

private:
  const SAOptions &options;
};

class SATargetBackend final : public TargetBackend {
public:
  explicit SATargetBackend(const SAOptions &options) : options(options) {}

  std::string getLegacyDefaultDeviceID() const override { return "sa"; }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {
    executableTargetAttrs.push_back(getSAExecutableTarget(context, options));
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                    OpPassManager &passManager) override {
    // Stage C2: match each dispatch to a template and generate its
    // descriptor list (docs/iree_compiler_plan.md §6.4).
  }

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    return variantOp.emitError()
           << "sa target: descriptor generation is not implemented yet "
              "(stage C2 of docs/iree_compiler_plan.md)";
  }

private:
  const SAOptions &options;
};

struct SASession final
    : PluginSession<SASession, SAOptions,
                    PluginActivationPolicy::DefaultActivated> {
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) final {
    // #hal.device.target<"sa", ...
    targets.add("sa", [=]() { return std::make_shared<SADevice>(options); });
  }
  void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) final {
    // #hal.executable.target<"sa", ...
    targets.add("sa",
                [=]() { return std::make_shared<SATargetBackend>(options); });
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL

extern "C" bool iree_register_compiler_plugin_hal_target_sa(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::iree_compiler::IREE::HAL::SASession>(
      "hal_target_sa");
  return true;
}

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::IREE::HAL::SAOptions);

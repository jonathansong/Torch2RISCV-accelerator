// sa HAL target for iree-compile (docs/iree_compiler_plan.md §6).
//
// Registers
//   #hal.device.target<"sa", ...>                     the PYNQ-Z1 accelerator
//   #hal.executable.target<"sa", "sa-desc-v1", {...}>  descriptor-list templates
// with iree-compile, and
//   - the preprocessing pass that packs linear-layer weights into the B-tile
//     layout (PackLinearWeights.cpp, §6.3);
//   - the translation: a dispatch runs as one workgroup (LowerWorkgroupCount.cpp);
//   - serialization: each export is matched to a template (Match.cpp), the
//     template is generated (Templates.cpp) and the executable written as
//     sa-desc-v1 (compiler/runtime/tools/sadesc.py has the layout). A dispatch
//     without a template is an error that shows the dispatch.
// Stage C2 supports the int8 linear dispatch (qlinear).

#include "DescList.h"
#include "Codegen.h"
#include "SAPasses.h"
#include "Templates.h"
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

#include <cstring>

namespace mlir::iree_compiler::IREE::HAL {
namespace {

struct SAOptions {
  // Array size of the accelerator (the .w8a8 packing and the templates depend on it).
  int d = 8;
  // Upper bound of dynamic dimensions (local memory planning): the sequence length.
  int maxDynamic = 256;
  // Development: a dispatch without a template becomes a warning and an export
  // that faults when run (so a partial module compiles for per-dispatch tests).
  bool allowUnsupported = false;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("sa HAL target (PYNQ-Z1 accelerator)");
    binder.opt<int>("iree-sa-d", d, llvm::cl::cat(category),
                    llvm::cl::desc("Array size D of the accelerator (8 or 16)."));
    binder.opt<int>("iree-sa-max-dynamic", maxDynamic, llvm::cl::cat(category),
                    llvm::cl::desc("Upper bound of dynamic dimensions (the sequence length)."));
    binder.opt<bool>("iree-sa-allow-unsupported", allowUnsupported, llvm::cl::cat(category),
                     llvm::cl::desc("Warn instead of failing on dispatches without a template (development)."));
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
    // Templates are matched and generated at serialization; here every
    // export becomes a single workgroup.
    passManager.addPass(sa::createLowerWorkgroupCountPass());
  }

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    int64_t d = options.d;
    if (auto config = variantOp.getTarget().getConfiguration())
      if (auto dAttr = config.getAs<IntegerAttr>("d")) d = dAttr.getInt();

    struct Export {
      std::string name;
      std::string templ;
      uint32_t bindings, constants, cycles;
      std::vector<sa::SetupEntry> setup;
    };
    SmallVector<std::pair<uint64_t, Export>> exports;
    auto innerModule = variantOp.getInnerModule();
    for (auto exportOp : variantOp.getBlock().getOps<IREE::HAL::ExecutableExportOp>()) {
      auto func = innerModule ? innerModule.lookupSymbol<FunctionOpInterface>(exportOp.getSymName())
                              : FunctionOpInterface();
      if (!func) return exportOp.emitError() << "sa target: no function for this export";
      auto layout = exportOp.getLayout();
      uint32_t bindings = layout.getBindings().size(), constants = layout.getConstants();
      if (bindings > 16) return exportOp.emitError() << "sa target: " << bindings << " bindings (at most 16)";
      sa::CodegenOptions copt;
      copt.d = d;
      copt.maxDynamic = options.maxDynamic;
      sa::Generated gen;
      std::string why;
      uint32_t cycles = 0;
      if (!sa::generateDispatch(func, copt, gen, why)) {
        if (!options.allowUnsupported) {
          auto diag = func.emitError() << "sa target: no template for dispatch '" << exportOp.getSymName()
                                       << "': " << why;
          diag.attachNote() << "dispatch:\n" << *func.getOperation();
          return failure();
        }
        func.emitWarning() << "sa target: dispatch '" << exportOp.getSymName() << "' unsupported: " << why;
        ::sa::DescList bad;                      // faults if run: an LD past the end of ACC
        bad.ld(0, ::sa::acc(0xFFFF), 4, 64, 64, -1);
        bad.ret();
        gen.templ = bad.bytes();
        gen.setup.clear();
        cycles = 0xFFFFFFFFu;                    // marks the export as unsupported
      }
      uint64_t ordinal = exportOp.getOrdinal() ? exportOp.getOrdinal()->getZExtValue() : exports.size();
      exports.push_back({ordinal, {exportOp.getSymName().str(), gen.templ, bindings, constants, cycles, gen.setup}});
    }
    llvm::sort(exports, [](auto &a, auto &b) { return a.first < b.first; });

    // sa-desc version 2 (compiler/runtime/tools/sadesc.py): header, export
    // table, templates (64-byte aligned), names, register setup tables
    constexpr uint32_t CAPS = (1u << 21) | (1u << 23) | (1u << 24);   // DESC, FPVE, CMDX
    uint32_t n = exports.size(), expOff = 64;
    uint32_t tmplOff = (expOff + 32 * n + 63) / 64 * 64;
    std::string tmpl, strings, table, setups;
    for (auto &[ord, e] : exports) {
      tmpl += e.templ;
      strings += e.name;
    }
    uint32_t strOff = tmplOff + tmpl.size();
    uint32_t setupOff = (strOff + strings.size() + 15) / 16 * 16;
    auto u32 = [](std::string &s, uint32_t v) { s.append(reinterpret_cast<const char *>(&v), 4); };
    auto u16 = [](std::string &s, uint16_t v) { s.append(reinterpret_cast<const char *>(&v), 2); };
    uint32_t tpos = 0, spos = 0;
    for (auto &[ord, e] : exports) {
      u32(table, spos);
      u32(table, e.name.size());
      u32(table, tpos);
      u32(table, e.templ.size() / 64);
      u16(table, e.bindings);
      u16(table, e.constants);
      u32(table, e.setup.size());
      u32(table, e.cycles);
      u32(table, e.setup.empty() ? 0 : setupOff + setups.size());
      for (const sa::SetupEntry &st : e.setup) {
        setups.push_back(char(uint8_t(st.kind) | uint8_t(st.shift << 4)));
        setups.push_back(char(st.reg));
        setups.push_back(char(st.binding));
        setups.push_back(0);
        u16(setups, uint16_t(st.constant));
        u16(setups, 0);
        u32(setups, uint32_t(st.mul));
        u32(setups, uint32_t(st.add));
      }
      tpos += e.templ.size();
      spos += e.name.size();
    }
    std::string blob("SADESC1\0", 8);
    for (uint32_t v : {2u, uint32_t(d), CAPS, n, expOff, tmplOff, uint32_t(tmpl.size()), strOff,
                       uint32_t(strings.size())})
      u32(blob, v);
    blob.resize(64, '\0');
    blob += table;
    blob.resize(tmplOff, '\0');
    blob += tmpl;
    blob += strings;
    if (!setups.empty()) {
      blob.resize(setupOff, '\0');
      blob += setups;
    }

    if (!serOptions.dumpBinariesPath.empty()) {
      dumpDataToPath(serOptions.dumpBinariesPath, serOptions.dumpBaseName, variantOp.getName(), ".sadesc",
                     StringRef(blob));
    }
    auto bufferAttr = DenseIntElementsAttr::get(
        VectorType::get({int64_t(blob.size())}, IntegerType::get(executableBuilder.getContext(), 8)),
        ArrayRef<char>(blob.data(), blob.size()));
    auto binaryOp = IREE::HAL::ExecutableBinaryOp::create(executableBuilder, variantOp.getLoc(),
                                                          variantOp.getSymName(),
                                                          variantOp.getTarget().getFormat(), bufferAttr);
    binaryOp.setMimeTypeAttr(executableBuilder.getStringAttr("application/x-sa-desc"));
    return success();
  }

private:
  const SAOptions &options;
};

struct SASession final
    : PluginSession<SASession, SAOptions,
                    PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    sa::registerPackLinearWeightsPass();
    sa::registerCloneCheapProducersPass();
    sa::registerLowerWorkgroupCountPass();
  }
  // Only acts on modules that target the sa device (the pass checks), so the
  // default behavior of the compiler is unchanged.
  void extendPreprocessingPassPipeline(OpPassManager &passManager) override {
    passManager.addPass(sa::createPackLinearWeightsPass(options.d));
    passManager.addPass(sa::createCloneCheapProducersPass());
  }
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

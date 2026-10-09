// sa HAL target for iree-compile (docs/iree_compiler_plan.md §6).
//
// Registers
//   #hal.device.target<"sa", ...>                     the PYNQ-Z1 accelerator
//   #hal.executable.target<"sa", "sa-desc-v1", {...}>  descriptor-list templates
// with iree-compile, and
//   - preprocessing: the packing of linear-layer weights into the B-tile layout
//     (PackLinearWeights.cpp, §6.3), index-only producers cloned into their
//     consumers (CloneCheapProducers.cpp);
//   - the translation (§8.4): a dispatch runs as one workgroup
//     (LowerWorkgroupCount.cpp); every export through the C5 pipeline
//     (transforms/: bufferize, sa-to-sahl, sahl-to-sahw with the
//     micro-kernels of --iree-sa-ukernels, sahw-fuse-ve), then the head split
//     and register assignment on sahw;
//   - serialization: the sahw templates as sa-desc v3
//     (compiler/runtime/tools/sadesc.py has the layout).

#include "DescList.h"
#include "SAPasses.h"
#include "../transforms/SahwPasses.h"
#include "../transforms/SahlPasses.h"
#include "SahlDialect.h"
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
  // Local memories (the L2 hardware: 128 KB per SPAD, 256 KB ACC).
  int spadKB = 128, accKB = 256;
  // Development: a dispatch without a template becomes a warning and an export
  // that faults when run (so a partial module compiles for per-dispatch tests).
  bool allowUnsupported = false;
  // The micro-kernels of sahl-to-sahw (§8.8): "all", "none" or a
  // comma-separated list (linear, attention).
  std::string ukernels = "all";
  // A per-dispatch report on stderr.
  bool codegenReport = false;
  // The host fallback: the device also runs VMVX executables on its ARM host;
  // each dispatch gets a VMVX variant too, and one the sa backend cannot
  // compile disables its sa variant (a false condition), so the runtime picks
  // the VMVX one.
  bool hostFallback = false;
  // Testing the fallback: dispatches whose name contains one of these
  // (comma-separated) run on the host even if the sa backend compiles them.
  std::string hostDispatches;
  // K2b: the streaming GEMV unit (docs/k2b_gemv_design.md) with this many read ports (0: none).
  int gemvPorts = 0;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("sa HAL target (PYNQ-Z1 accelerator)");
    binder.opt<int>("iree-sa-d", d, llvm::cl::cat(category),
                    llvm::cl::desc("Array size D of the accelerator (8 or 16)."));
    binder.opt<int>("iree-sa-max-dynamic", maxDynamic, llvm::cl::cat(category),
                    llvm::cl::desc("Upper bound of dynamic dimensions (the sequence length)."));
    binder.opt<int>("iree-sa-spad-kb", spadKB, llvm::cl::cat(category),
                    llvm::cl::desc("Size of each SPAD (A, B) in KB."));
    binder.opt<int>("iree-sa-acc-kb", accKB, llvm::cl::cat(category),
                    llvm::cl::desc("Size of ACC in KB."));
    binder.opt<bool>("iree-sa-allow-unsupported", allowUnsupported, llvm::cl::cat(category),
                     llvm::cl::desc("Warn instead of failing on dispatches without a template (development)."));
    binder.opt<std::string>("iree-sa-ukernels", ukernels, llvm::cl::cat(category),
                            llvm::cl::desc("Micro-kernels: all, none, or a list (linear,attention)."));
    binder.opt<bool>("iree-sa-host-fallback", hostFallback, llvm::cl::cat(category),
                     llvm::cl::desc("Dispatches the sa backend cannot compile run on the host (VMVX)."));
    binder.opt<std::string>("iree-sa-host-dispatches", hostDispatches, llvm::cl::cat(category),
                            llvm::cl::desc("With the host fallback: run these dispatches (name substrings, "
                                           "comma-separated) on the host (testing)."));
    binder.opt<int>("iree-sa-gemv-ports", gemvPorts, llvm::cl::cat(category),
                    llvm::cl::desc("Read ports of the streaming GEMV unit (K2b; 0: none): decode's linear layers use it."));
    binder.opt<bool>("iree-sa-codegen-report", codegenReport, llvm::cl::cat(category),
                     llvm::cl::desc("Print, per dispatch, whether it compiled (and why not)."));
  }
};

static IREE::HAL::ExecutableTargetAttr
getSAExecutableTarget(MLIRContext *context, const SAOptions &options) {
  Builder b(context);
  SmallVector<NamedAttribute> config;
  sa::TargetConfig tc;
  tc.d = options.d;
  tc.maxDynamic = options.maxDynamic;
  tc.spadBytes = int64_t(options.spadKB) << 10;
  tc.accBytes = int64_t(options.accKB) << 10;
  tc.gemvPorts = options.gemvPorts;
  tc.addTo(b, config);
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
    // the host fallback: VMVX variants, after the sa ones (the runtime takes
    // the first variant whose format and condition hold)
    if (options.hostFallback) {
      if (auto vmvx = targetRegistry.getTargetBackend("vmvx"))
        vmvx->getDefaultExecutableTargets(context, "local", configAttr, executableTargetAttrs);
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

  // the dialects the translation pipeline (and the C5 pipeline it runs per
  // dispatch) creates
  void getDependentDialects(DialectRegistry &registry) const override {
    OpPassManager pm(IREE::HAL::ExecutableVariantOp::getOperationName());
    pm.addPass(sa::createSaCodegenPass(false));
    pm.addPass(sa::createSahwFuseVePass());
    pm.getDependentDialects(registry);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                    OpPassManager &passManager) override {
    // Every export becomes a single workgroup; its template is generated
    // here (sahw).
    passManager.addPass(sa::createLowerWorkgroupCountPass());
    passManager.addPass(sa::createSaCodegenPass(options.allowUnsupported, options.ukernels, options.codegenReport,
                                                options.hostFallback, options.hostDispatches));
    passManager.addPass(sa::createSahwSplitHeadPass());
    passManager.addPass(sa::createSahwAssignRegistersPass());
  }

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    sa::TargetConfig tc = sa::TargetConfig::fromAttr(variantOp.getTarget().getConfiguration());
    int64_t d = tc.d;

    struct Export {
      std::string name;
      std::string templ;
      uint32_t bindings, constants, cycles;
      std::vector<sa::SetupEntry> setup;
      uint32_t prefix, prefixReads, writes, reads, head, flags, prefixReg;
    };
    SmallVector<std::pair<uint64_t, Export>> exports;
    auto innerModule = variantOp.getInnerModule();
    // the dialect path: sahw.template ops from the translation pipeline
    llvm::StringMap<sa::sahw::TemplateOp> templates;
    if (innerModule)
      for (auto t : innerModule.getOps<sa::sahw::TemplateOp>()) templates[t.getExportName()] = t;
    for (auto exportOp : variantOp.getBlock().getOps<IREE::HAL::ExecutableExportOp>()) {
      uint64_t ordinal = exportOp.getOrdinal() ? exportOp.getOrdinal()->getZExtValue() : exports.size();
      auto layout = exportOp.getLayout();
      if (layout.getBindings().size() > 16)
        return exportOp.emitError() << "sa target: " << layout.getBindings().size() << " bindings (at most 16)";
      auto it = templates.find(exportOp.getSymName());
      if (it == templates.end()) return exportOp.emitError() << "sa target: no sahw.template for this export";
      sa::SerializedExport se;
      if (failed(sa::serializeTemplate(it->second, se))) return failure();
      exports.push_back({ordinal, {exportOp.getSymName().str(), se.templ, uint32_t(layout.getBindings().size()),
                                   uint32_t(layout.getConstants()), se.cycles, se.setup, se.prefix, se.prefixReads,
                                   se.writes, se.reads, se.head, se.flags, se.prefixReg}});
    }
    llvm::sort(exports, [](auto &a, auto &b) { return a.first < b.first; });

    // sa-desc version 3 (compiler/runtime/tools/sadesc.py): header, export
    // table, templates (64-byte aligned), names, register setup tables, the
    // export extensions (prefix, head, bindings read / written, SPAD_B use)
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
    std::string ext;
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
      for (uint32_t v : {e.prefix, e.prefixReads, e.writes, e.reads, e.head, e.flags, e.prefixReg, 0u}) u32(ext, v);
      tpos += e.templ.size();
      spos += e.name.size();
    }
    std::string blob("SADESC1\0", 8);
    uint32_t extOff = (setupOff + setups.size() + 15) / 16 * 16;
    for (uint32_t v : {3u, uint32_t(d), CAPS, n, expOff, tmplOff, uint32_t(tmpl.size()), strOff,
                       uint32_t(strings.size()), extOff})
      u32(blob, v);
    blob.resize(64, '\0');
    blob += table;
    blob.resize(tmplOff, '\0');
    blob += tmpl;
    blob += strings;
    blob.resize(setupOff, '\0');
    blob += setups;
    blob.resize(extOff, '\0');
    blob += ext;

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
    sa::registerSahwPasses();
  }
  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<sa::sahl::SahlDialect, sa::sahw::SahwDialect>();
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

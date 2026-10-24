// M5: lower a recognized Mixtral graph to the SA router/Top-2/route/FFN chain.
// The four executable binaries are produced by the compiler's earlier exports;
// this pass owns the generic HAL dispatch graph and validates the packed ABI.
#include "SAPasses.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace mlir::iree_compiler::sa {
namespace {

constexpr char kChainBody[] =
#include "MoeChainTemplate.inc"
;
constexpr char kModelBody[] =
#include "MoeModelTemplate.inc"
;

std::optional<std::string> readFile(const std::filesystem::path &path) {
  auto file = llvm::MemoryBuffer::getFile(path.string());
  if (!file) return std::nullopt;
  return file.get()->getBuffer().str();
}

std::string binaryOp(StringRef name, StringRef format, StringRef data) {
  static constexpr char digits[] = "0123456789ABCDEF";
  std::string hex;
  hex.reserve(data.size() * 2);
  for (unsigned char byte : data) {
    hex.push_back(digits[byte >> 4]);
    hex.push_back(digits[byte & 15]);
  }
  std::string result = "  hal.executable @" + name.str() + " {\n    hal.executable.binary @" +
                       (format == "sa-desc-v1" ? "sa_desc_v1" : "vmvx_bytecode_fb") +
                       " attributes {data = dense<\"0x" + hex + "\"> : vector<" +
                       std::to_string(data.size()) + "xi8>, format = \"" + format.str() + "\"";
  if (format == "sa-desc-v1")
    result += ", mime_type = \"application/x-sa-desc\"";
  return result + "}\n  }\n";
}

void replaceAll(std::string &source, StringRef token, int64_t value) {
  const std::string replacement = std::to_string(value);
  size_t at = 0;
  while ((at = source.find(token.str(), at)) != std::string::npos) {
    source.replace(at, token.size(), replacement);
    at += replacement.size();
  }
}

void replaceText(std::string &source, StringRef token, StringRef replacement) {
  size_t at = 0;
  while ((at = source.find(token.str(), at)) != std::string::npos) {
    source.replace(at, token.size(), replacement.str());
    at += replacement.size();
  }
}

struct LowerMoeChainPass
    : public PassWrapper<LowerMoeChainPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMoeChainPass)

  LowerMoeChainPass() = default;
  explicit LowerMoeChainPass(std::string path, bool modelInterface) {
    bundle = std::move(path);
    modelABI = modelInterface;
  }
  LowerMoeChainPass(const LowerMoeChainPass &other) : PassWrapper(other) {}

  StringRef getArgument() const override { return "iree-sa-lower-moe-chain"; }
  StringRef getDescription() const override {
    return "Replace one recognized Mixtral MoE graph with an SA HAL chain";
  }

  void runOnOperation() override {
    if (bundle.empty()) return;
    ModuleOp module = getOperation();
    Operation *matched = nullptr;
    unsigned matches = 0;
    module.walk([&](Operation *op) {
      if (op->getAttrOfType<StringAttr>("sa.moe.arch") ==
          StringAttr::get(&getContext(), "mixtral_top2_ffn")) {
        ++matches;
        matched = op;
      }
    });
    if (matches != 1) {
      module.emitError() << "expected exactly one recognized Mixtral Top-2 FFN, got " << matches;
      return signalPassFailure();
    }
    const std::filesystem::path dir(bundle.getValue());
    auto manifestBytes = readFile(dir / "moe_manifest.json");
    if (!manifestBytes) {
      module.emitError() << "cannot read MoE bundle manifest in " << bundle;
      return signalPassFailure();
    }
    auto json = llvm::json::parse(*manifestBytes);
    auto *manifest = json ? json->getAsObject() : nullptr;
    auto *checkpoint = manifest ? manifest->getObject("source") : nullptr;
    if (!manifest || manifest->getString("format") != "sa-moe-parameters-v1" ||
        manifest->getString("layout") != "iree-sa-packed-w8" ||
        !checkpoint || checkpoint->getString("adapter") != "mixtral" ||
        checkpoint->getBoolean("normalize_topk") != true) {
      module.emitError("invalid packed MoE manifest");
      return signalPassFailure();
    }
    auto field = [&](StringRef key) -> std::optional<int64_t> {
      auto value = manifest->getInteger(key);
      if (!value) module.emitError() << "missing MoE manifest field " << key;
      return value;
    };
    auto dim = field("dim"), hidden = field("hidden"), experts = field("experts");
    auto stride = field("expert_stride"), offset = field("expert_table_offset");
    auto totalBytes = field("total_bytes");
    auto d = field("d"), topK = field("top_k");
    if (!dim || !hidden || !experts || !stride || !offset || !totalBytes || !d || !topK)
      return signalPassFailure();
    if (*d != 16 || *topK != 2 || *experts < 4 || *experts > 16 ||
        *dim <= 0 || *dim > (1 << 20) || *hidden <= *dim || *hidden > (1 << 20) ||
        *offset < 0 || *offset % 64 || *stride <= 0 ||
        *stride > UINT32_MAX / (*experts - 1) ||
        *totalBytes < *offset || *stride > (*totalBytes - *offset) / *experts) {
      module.emitError("unsupported MoE packed ABI");
      return signalPassFailure();
    }
    for (auto [key, expected] : {std::pair{"dim", *dim}, {"hidden", *hidden},
                                 {"experts", *experts}, {"expert_stride", *stride}}) {
      auto actual = matched->getAttrOfType<IntegerAttr>(("sa.moe." + StringRef(key)).str());
      if (!actual || actual.getInt() != expected) {
        matched->emitError() << "recognized sa.moe." << key << " disagrees with packed ABI";
        return signalPassFailure();
      }
    }
    std::filesystem::path routeFile;
    std::error_code directoryError;
    std::filesystem::directory_iterator it(dir / "route_binaries", directoryError), end;
    for (; !directoryError && it != end; it.increment(directoryError))
      if (it->path().extension() == ".sadesc") {
        if (!routeFile.empty()) {
          module.emitError("expected one route executable");
          return signalPassFailure();
        }
        routeFile = it->path();
      }
    if (directoryError) {
      module.emitError() << "cannot read MoE route executables: " << directoryError.message();
      return signalPassFailure();
    }
    if (routeFile.empty()) {
      module.emitError("missing route executable");
      return signalPassFailure();
    }
    auto route = readFile(routeFile), ffn = readFile(dir / "moe_ffn.sadesc");
    auto top2 = readFile(dir / "moe_top2.sadesc"), router = readFile(dir / "moe_router.vmvx");
    if (!route || !ffn || !top2 || !router) {
      module.emitError("incomplete compiled MoE executables");
      return signalPassFailure();
    }
    auto abiBytes = readFile(dir / "moe_top2_abi.json");
    if (!abiBytes) {
      module.emitError("missing Top-2 binding-offset ABI");
      return signalPassFailure();
    }
    auto abiJSON = llvm::json::parse(*abiBytes);
    auto *abi = abiJSON ? abiJSON->getAsObject() : nullptr;
    auto scoresOffset = abi ? abi->getInteger("scores_offset") : std::nullopt;
    auto valuesOffset = abi ? abi->getInteger("values_offset") : std::nullopt;
    auto idsOffset = abi ? abi->getInteger("ids_offset") : std::nullopt;
    if (!scoresOffset || !valuesOffset || !idsOffset || *scoresOffset != 0 ||
        *valuesOffset < 0 || *valuesOffset > 4096 || *valuesOffset % 64 ||
        *idsOffset < 0 || *idsOffset > 4096 || *idsOffset % 64) {
      module.emitError("unsupported or missing Top-2 binding-offset ABI");
      return signalPassFailure();
    }
    const int64_t valuesStorage = *valuesOffset + 8;
    const int64_t idsStorage = *idsOffset + 8;
    std::string source = "module {\n";
    source += binaryOp("route", "sa-desc-v1", *route);
    source += binaryOp("ffn", "sa-desc-v1", *ffn);
    source += binaryOp("top2", "sa-desc-v1", *top2);
    source += binaryOp("router", "vmvx-bytecode-fb", *router);
    std::string body = kChainBody;
    replaceAll(body, "@SCORES_BYTES@", *experts * 4);
    replaceAll(body, "@ROUTER_BYTES@", *experts * *dim * 4);
    replaceAll(body, "@PARAM_OFFSET@", *offset);
    replaceAll(body, "@ACTIVATION_BYTES@", *dim * 4);
    replaceAll(body, "@VALUES_OFFSET@", *valuesOffset);
    replaceAll(body, "@IDS_OFFSET@", *idsOffset);
    replaceAll(body, "@VALUES_STORAGE_BYTES@", valuesStorage);
    replaceAll(body, "@IDS_STORAGE_BYTES@", idsStorage);
    replaceText(body, "@MODEL_FENCE_DECL@",
                modelABI ? "    %done = hal.fence.create device(%device : !hal.device) flags(\"None\") : !hal.fence\n"
                         "    %done_timeout = arith.constant -1 : i32" : "");
    replaceText(body, "@MODEL_FENCE_SIGNAL@", modelABI ? "%done" : "%no_fence");
    replaceText(body, "@MODEL_FENCE_WAIT@",
                modelABI ? "    %done_status = hal.fence.await until([%done]) timeout_millis(%done_timeout) flags(\"None\") : i32" : "");
    if (modelABI) {
      size_t name = body.find("util.func public @main");
      if (name == std::string::npos) {
        module.emitError("missing MoE device entry in HAL template");
        return signalPassFailure();
      }
      body.replace(name, sizeof("util.func public @main") - 1,
                   "util.func private @device");
    }
    source += body;
    if (modelABI) {
      std::string wrapper = kModelBody;
      replaceAll(wrapper, "@SCORES_BYTES@", *experts * 4);
      replaceAll(wrapper, "@ROUTER_BYTES@", *experts * *dim * 4);
      replaceAll(wrapper, "@PARAM_BYTES@", *totalBytes);
      replaceAll(wrapper, "@ACTIVATION_BYTES@", *dim * 4);
      replaceAll(wrapper, "@EXPERTS@", *experts);
      replaceAll(wrapper, "@DIM@", *dim);
      replaceAll(wrapper, "@VALUES_OFFSET@", *valuesOffset);
      replaceAll(wrapper, "@IDS_OFFSET@", *idsOffset);
      replaceAll(wrapper, "@VALUES_STORAGE_BYTES@", valuesStorage);
      replaceAll(wrapper, "@IDS_STORAGE_BYTES@", idsStorage);
      source += wrapper;
    }
    source += "}\n";
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(source, "native-moe-chain"),
                                 llvm::SMLoc());
    OwningOpRef<ModuleOp> chain = parseSourceFile<ModuleOp>(sourceMgr, &getContext());
    if (!chain) return signalPassFailure();
    module.getBody()->getOperations().clear();
    module->setAttrs(chain->getOperation()->getAttrs());
    for (Operation &op : llvm::make_early_inc_range(chain->getBody()->getOperations()))
      op.moveBefore(module.getBody(), module.getBody()->end());
  }

  Option<std::string> bundle{*this, "bundle",
                             llvm::cl::desc("Directory of compiler-generated MoE executables"),
                             llvm::cl::init("")};
  Option<bool> modelABI{*this, "model-abi",
                        llvm::cl::desc("Load packed weights and expose the one-input model ABI"),
                        llvm::cl::init(false)};
};

}  // namespace

std::unique_ptr<Pass> createLowerMoeChainPass(const std::string &bundle, bool modelABI) {
  return std::make_unique<LowerMoeChainPass>(bundle, modelABI);
}
void registerLowerMoeChainPass() { PassRegistration<LowerMoeChainPass>(); }

}  // namespace mlir::iree_compiler::sa

// C5.0 (docs/iree_compiler_plan.md §8.7): the C3/C4 code generator's output
// (descriptor rows + register setup table) raised into sahw.template ops, so
// that register assignment, the head split and serialization run on the
// dialect. The rows decode field by field (the DescList encoding); loops
// become sahw.loop regions; register numbers become values.
#include <cstring>
#include <map>
#include <set>

#include "../target/DescList.h"
#include "SahlPasses.h"
#include "SahwPasses.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler::sa {
namespace {

using Row = std::array<uint64_t, 8>;
using ::sa::DescList;
constexpr uint64_t M32 = 0xFFFFFFFFull;

float f32of(uint64_t bits) {
  uint32_t u = uint32_t(bits);
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

struct Raiser {
  OpBuilder &b;
  Location loc;
  std::map<int, Value> bases, params;
  std::string err;

  bool fail(const std::string &m) {
    if (err.empty()) err = m;
    return false;
  }

  // PARAM numbers a row refers to
  static void paramsOf(const Row &w, std::set<int> &out) {
    for (int i = 0; i < 2; ++i) {
      uint64_t f = (w[0] >> (16 + 8 * i)) & 0xFF;
      if (f) out.insert(int((f >> 4) & 7));
    }
    uint32_t op = uint32_t(w[0] & 0xFF);
    if (op == DescList::LDPARAM) out.insert(int(w[2] & 7));
    if (op == DescList::LOOP_END) {
      out.insert(int((w[2] >> 16) & 7));
      out.insert(int((w[2] >> 19) & 7));
    }
    if (op == DescList::SETREG)
      for (int i = 0; i < 3; ++i) {
        uint64_t sel = (w[1] >> (8 * i)) & 0xFF;
        if (sel && (sel & 0x3F) >= DescList::REG_PARAM) out.insert(int((sel & 0x3F) - DescList::REG_PARAM));
      }
  }

  bool common(const Row &w, Value &base, SmallVector<Value> &dyn, SmallVector<int32_t> &fields,
              SmallVector<bool> &adds, bool &fenceBefore) {
    if (w[0] & (1u << 12)) return fail("IRQ bit set");
    fenceBefore = (w[0] >> 11) & 1;
    if ((w[0] >> 8) & 1) {
      int r = int(((w[0] >> 9) & 3) | ((w[0] >> 13) & 3) << 2);
      auto it = bases.find(r);
      if (it == bases.end()) return fail("BASE" + std::to_string(r) + " without a setup entry");
      base = it->second;
    }
    for (int i = 0; i < 2; ++i) {
      uint64_t f = (w[0] >> (16 + 8 * i)) & 0xFF;
      if (!f) continue;
      dyn.push_back(params.at(int((f >> 4) & 7)));
      fields.push_back(int32_t(f & 0xF));
      adds.push_back((f >> 7) & 1);
    }
    return true;
  }

  template <typename OpT>
  void setDyn(OpT op, ArrayRef<int32_t> fields, ArrayRef<bool> adds, bool fenceBefore) {
    if (!fields.empty()) {
      op.setDynFields(fields);
      op.setDynAdd(adds);
    }
    if (fenceBefore) op.setFenceBefore(true);
  }

  // rows[from, to) into the current insertion block; loops become regions
  bool raise(ArrayRef<Row> rows) {
    std::vector<std::pair<int64_t, Operation *>> top;       // (first row, op) at this level
    for (size_t i = 0; i < rows.size(); ++i) {
      const Row &w = rows[i];
      uint32_t opc = uint32_t(w[0] & 0xFF);
      Value base;
      SmallVector<Value> dyn;
      SmallVector<int32_t> fields;
      SmallVector<bool> adds;
      bool fb = false;
      Operation *made = nullptr;
      auto i64 = [&](uint64_t v) { return int64_t(v); };
      switch (opc) {
      case DescList::LD: {
        if (!common(w, base, dyn, fields, adds, fb)) return false;
        auto op = sahw::LdOp::create(b, loc, base, i64(w[1] & M32), i64(w[2] & M32), i64((w[2] >> 32) & 0xFFFF),
                                     i64(w[2] >> 48), i64(w[3] & M32), i64(w[3] >> 32), dyn);
        setDyn(op, fields, adds, fb);
        made = op;
        break;
      }
      case DescList::ST: {
        if (!common(w, base, dyn, fields, adds, fb)) return false;
        if (w[3] >> 32) return fail("ST with bits above the pitch");
        auto op = sahw::StOp::create(b, loc, base, i64(w[1] & M32), i64(w[2] & M32), i64((w[2] >> 32) & 0xFFFF),
                                     i64(w[2] >> 48), i64(w[3]), dyn);
        setDyn(op, fields, adds, fb);
        made = op;
        break;
      }
      case DescList::EX: {
        if (!common(w, base, dyn, fields, adds, fb) || base) return fail("EX header");
        if (w[1] >> 61) return fail("EX word 1 high bits");
        auto op = sahw::ExOp::create(b, loc, i64(w[1] & 0xFFFF), i64((w[1] >> 16) & 0xFFFF),
                                     i64((w[1] >> 32) & 0xFFFF), i64((w[1] >> 48) & 0xFFF), (w[1] >> 60) & 1,
                                     i64(w[2] & 0xFFFF), i64((w[2] >> 16) & 0xFFFF), i64((w[2] >> 32) & 0xFFFF),
                                     i64(w[2] >> 48), dyn);
        setDyn(op, fields, adds, fb);
        made = op;
        break;
      }
      case DescList::VE: {
        if (!common(w, base, dyn, fields, adds, fb) || base) return fail("VE header");
        uint64_t w3 = w[3], flags = w3 >> 53;
        if ((w3 & 0xFF) == 6 && !(flags & 1)) {          // TRANSPOSE
          if (w[3] >> 16 || w[4] || w[5] || w[6] || (w[7] & ((1ull << 48) - 1)))
            return fail("TRANSPOSE fields");
          auto op = sahw::TransposeOp::create(b, loc, i64(w[1]), i64(w[2] & M32), i64(w[2] >> 32),
                                              i64((w3 >> 8) & 0xFF), i64(w[7] >> 48), dyn);
          setDyn(op, fields, adds, fb);
          made = op;
          break;
        }
        auto op = sahw::VeOp::create(
            b, loc, i64(w[1] & M32), i64(w[1] >> 32), i64(w[2] & M32), i64(w[2] >> 32), i64(w3 & 0xFF),
            i64((w3 >> 8) & 0xFF), i64((w3 >> 16) & 0xFFFF), flags & 1, i64((flags >> 1) & 7),
            i64((flags >> 4) & 3), i64((flags >> 6) & 3), i64((flags >> 8) & 3), (flags >> 10) & 1,
            llvm::APFloat(f32of(w[5] >> 32)), llvm::APFloat(f32of(w[6])), llvm::APFloat(f32of(w[6] >> 32)), i64(w[7] & 0xFFFF), i64((w[7] >> 16) & 0xFFFF), i64(w[7] >> 32),
            i64((w3 >> 32) & 0xFFFF), i64((w3 >> 48) & 0x1F), int64_t(int32_t(w[4] & M32)),
            int64_t(int32_t(w[4] >> 32)), int64_t(int32_t(w[5] & M32)), dyn);
        if (flags >> 11) return fail("VE flags");
        setDyn(op, fields, adds, fb);
        made = op;
        break;
      }
      case DescList::LDPARAM: {
        if (!common(w, base, dyn, fields, adds, fb)) return false;
        auto op = sahw::LdParamOp::create(b, loc, base, params.at(int(w[2] & 7)), i64(w[1] & M32),
                                          i64(w[3] & 0xFFFF), i64(w[4] & M32), dyn);
        setDyn(op, fields, adds, fb);
        made = op;
        break;
      }
      case DescList::FENCE:
        made = sahw::FenceOp::create(b, loc, i64(w[1]));
        break;
      case DescList::SETREG: {
        if (!common(w, base, dyn, fields, adds, fb) || base) return fail("SETREG header");
        SmallVector<Value> targets;
        SmallVector<int64_t> values;
        for (int k = 0; k < 3; ++k) {
          uint64_t sel = (w[1] >> (8 * k)) & 0xFF;
          if (!sel) break;
          if ((sel & 0x3F) < DescList::REG_PARAM) return fail("SETREG of a BASE register");
          targets.push_back(params.at(int((sel & 0x3F) - DescList::REG_PARAM)));
          values.push_back(int64_t(w[2 + k] & M32));
        }
        auto op = sahw::SetRegOp::create(b, loc, targets, values, i64(w[5] & 7), dyn);
        setDyn(op, fields, adds, fb);
        made = op;
        break;
      }
      case DescList::LOOP_END: {
        int64_t start = int64_t(i) + int32_t(uint32_t(w[1] & M32));
        auto loop = sahw::LoopOp::create(b, loc, i64(w[2] & 0xFFFF), params.at(int((w[2] >> 16) & 7)),
                                         i64(w[3]), params.at(int((w[2] >> 19) & 7)), i64(w[4]));
        Block *blk = &loop.getRegion().emplaceBlock();
        size_t firstTop = top.size();
        while (firstTop > 0 && top[firstTop - 1].first >= start) --firstTop;
        if (firstTop == top.size() || top[firstTop].first != start) return fail("loop start is not a command");
        for (size_t k = firstTop; k < top.size(); ++k) top[k].second->moveBefore(blk, blk->end());
        top.resize(firstTop);
        top.push_back({start, loop});
        continue;
      }
      case DescList::RET:
        if (i + 1 != rows.size()) return fail("RET inside a template");
        continue;
      default:
        return fail("unexpected opcode " + std::to_string(opc));
      }
      top.push_back({int64_t(i), made});
    }
    return err.empty();
  }
};

struct SahwLegacyCodegenPass
    : public PassWrapper<SahwLegacyCodegenPass, OperationPass<IREE::HAL::ExecutableVariantOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SahwLegacyCodegenPass)
  explicit SahwLegacyCodegenPass(bool allowUnsupported = false, std::string newCodegen = "off", bool report = false)
      : allowUnsupported(allowUnsupported), newCodegen(std::move(newCodegen)), report(report) {}
  SahwLegacyCodegenPass(const SahwLegacyCodegenPass &o)
      : PassWrapper(o), allowUnsupported(o.allowUnsupported), newCodegen(o.newCodegen), report(o.report) {}

  StringRef getArgument() const override { return "iree-sahw-legacy-codegen"; }
  StringRef getDescription() const override {
    return "Generates every export (the C5 pipeline, or the C3/C4 code generator raised) into sahw.template";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<sahw::SahwDialect>();
    OpPassManager pm(ModuleOp::getOperationName());
    buildSahlPipeline(pm, TargetConfig());
    pm.getDependentDialects(registry);
  }

  // The C5 pipeline on a copy of the function (in a module nested in the
  // inner module); the template moves into the inner module on success.
  bool tryNew(ModuleOp inner, FunctionOpInterface func, const TargetConfig &cfg, int64_t bindings, int64_t constants,
              std::string &why) {
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
      t->setAttr("sa.codegen", b.getStringAttr("sahl"));
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
    OpBuilder b = OpBuilder::atBlockEnd(inner.getBody());
    for (auto exportOp : variant.getBlock().getOps<IREE::HAL::ExecutableExportOp>()) {
      auto func = inner.lookupSymbol<FunctionOpInterface>(exportOp.getSymName());
      if (!func) {
        exportOp.emitError() << "sa: no function for this export";
        return signalPassFailure();
      }
      auto layout = exportOp.getLayout();
      if (newCodegen != "off") {
        std::string why;
        if (tryNew(inner, func, cfg, int64_t(layout.getBindings().size()), int64_t(layout.getConstants()), why)) {
          if (report) llvm::errs() << "sa codegen: " << exportOp.getSymName() << ": sahl\n";
          continue;
        }
        if (newCodegen == "only") {
          exportOp.emitError() << "sa: the C5 pipeline failed: " << why;
          return signalPassFailure();
        }
        if (report) llvm::errs() << "sa codegen: " << exportOp.getSymName() << ": legacy (" << why << ")\n";
      }
      Generated gen;
      std::string why;
      bool ok = generateDispatch(func, cfg.codegenOptions(), gen, why);
      Location loc = func.getLoc();
      auto t = sahw::TemplateOp::create(b, loc, exportOp.getSymName(), int64_t(layout.getBindings().size()),
                                        int64_t(layout.getConstants()), 0, !ok);
      OpBuilder tb = OpBuilder::atBlockEnd(&t.getRegion().emplaceBlock());
      if (!ok) {
        if (!allowUnsupported) {
          auto diag = func.emitError() << "sa target: no template for dispatch '" << exportOp.getSymName()
                                       << "': " << why;
          diag.attachNote() << "dispatch:\n" << *func.getOperation();
          return signalPassFailure();
        }
        func.emitWarning() << "sa target: dispatch '" << exportOp.getSymName() << "' unsupported: " << why;
        // faults if run: an LD past the end of ACC
        auto body = sahw::BodyOp::create(tb, loc);
        OpBuilder bb = OpBuilder::atBlockEnd(&body.getRegion().emplaceBlock());
        sahw::LdOp::create(bb, loc, Value(), 0, int64_t(::sa::acc(0xFFFF)), 4, 64, 64, 0, ValueRange{});
        continue;
      }
      Raiser r{tb, loc};
      // registers: the setup table in order, then the private PARAMs from 7 down
      for (const SetupEntry &e : gen.setup) {
        if (e.kind == SetupEntry::BASE)
          r.bases[e.reg] = sahw::BaseOp::create(tb, loc, tb.getType<sahw::BaseType>(), int64_t(e.binding), int64_t(e.constant), int64_t(e.mul),
                                                int64_t(e.shift), int64_t(e.add), IntegerAttr());
        else
          r.params[e.reg] = sahw::ParamOp::create(tb, loc, tb.getType<sahw::ParamType>(), int64_t(e.constant), int64_t(e.mul), int64_t(e.shift),
                                                  int64_t(e.add), IntegerAttr());
      }
      if (gen.setup.empty() && !gen.bodyRows.empty() && gen.bodyRows.size() > 1) {
        func.emitError() << "sa: C5.0 raising needs a register setup table";
        return signalPassFailure();
      }
      std::set<int> used;
      for (const Row &w : gen.bodyRows) Raiser::paramsOf(w, used);
      for (const Row &w : gen.prefixRows) Raiser::paramsOf(w, used);
      for (auto it = used.rbegin(); it != used.rend(); ++it)
        if (!r.params.count(*it)) r.params[*it] = sahw::PrivateOp::create(tb, loc, tb.getType<sahw::ParamType>(), IntegerAttr());
      if (!gen.prefixRows.empty()) {
        auto pre = sahw::PrefixOp::create(tb, loc);
        OpBuilder pb = OpBuilder::atBlockEnd(&pre.getRegion().emplaceBlock());
        Raiser rp{pb, loc, r.bases, r.params};
        if (!rp.raise(gen.prefixRows)) {
          func.emitError() << "sa: raising the prefix: " << rp.err;
          return signalPassFailure();
        }
      }
      auto body = sahw::BodyOp::create(tb, loc);
      OpBuilder bb = OpBuilder::atBlockEnd(&body.getRegion().emplaceBlock());
      Raiser rb{bb, loc, r.bases, r.params};
      if (!rb.raise(gen.bodyRows)) {
        func.emitError() << "sa: raising the template: " << rb.err;
        return signalPassFailure();
      }
    }
  }

  bool allowUnsupported;
  std::string newCodegen;
  bool report;
};

}  // namespace

std::unique_ptr<Pass> createSahwLegacyCodegenPass(bool allowUnsupported, const std::string &newCodegen,
                                                  bool report) {
  return std::make_unique<SahwLegacyCodegenPass>(allowUnsupported, newCodegen, report);
}

void registerSahwLegacyCodegenPass() { PassRegistration<SahwLegacyCodegenPass>(); }

}  // namespace mlir::iree_compiler::sa

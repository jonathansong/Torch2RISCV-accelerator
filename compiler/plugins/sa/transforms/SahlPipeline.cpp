// sahw-fuse-ve and the C5 pipeline (docs/iree_compiler_plan.md §8.4).
//
// sahw-fuse-ve: sahl-to-sahw gives one single-stage VE per operation; a VE
// computes FUNC(OP(s1', s2) * A + B) (then REDUCE, then the output type
// conversion), each stage rounded like the separate operation, so a VE that
// only applies later stages to the result of an earlier VE merges into it
// (the C3 folding rules): x * c -> A, x + c -> B, exp / rsqrt / 1/x / abs ->
// FUNC, a reduction -> REDUCE, to_i8 -> int8 output. The intermediate buffer
// must have no other reader.
#include "SahlPasses.h"
#include "SahwPasses.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::iree_compiler::sa {
namespace {

uint32_t bitsOf(const APFloat &f) { return uint32_t(f.bitcastToAPInt().getZExtValue()); }
constexpr uint32_t ONE = 0x3F800000u, NEG0 = 0x80000000u;

// [start, end) words of local memory a command reads / writes (conservative: the whole length)
struct Range {
  uint32_t mem = 0, lo = 0, hi = 0;
  bool overlaps(const Range &o) const { return mem == o.mem && lo < o.hi && o.lo < hi; }
};
Range rangeOf(int64_t la, int64_t words) {
  uint64_t a = uint64_t(la) & 0xFFFFFFFFull;
  return {uint32_t(a >> 28), uint32_t(a & 0xFFFFFFF), uint32_t(a & 0xFFFFFFF) + uint32_t(std::max<int64_t>(words, 1))};
}

// words a VE source reads: LIN the whole length; DIV by p: 1 / p of it; MOD by p: p words
int64_t srcWords(int64_t words, int64_t mode, int64_t period) {
  if (mode == 2) return period ? (words + period - 1) / period : 1;
  if (mode == 1) return period ? std::min(period, words) : words;
  return words;
}

void readsOf(Operation *op, int64_t d, SmallVectorImpl<Range> &out) {
  if (auto v = dyn_cast<sahw::VeOp>(op)) {
    int64_t words = (v.getLength() + d - 1) / d;
    out.push_back(rangeOf(v.getSrc1(), srcWords(words, v.getM1(), v.getP1())));
    if (v.getM2() != 3 /*IMM*/ && v.getSrc2()) out.push_back(rangeOf(v.getSrc2(), srcWords(words, v.getM2(), v.getPeriod())));
  } else if (auto t = dyn_cast<sahw::TransposeOp>(op)) {
    out.push_back(rangeOf(t.getSrc(), (t.getLength() + d - 1) / d));
  } else if (auto s = dyn_cast<sahw::StOp>(op)) {
    // words of D elements: 4 * D bytes in ACC, D bytes in SPAD
    int64_t wordBytes = ((uint64_t(s.getLaddr()) >> 28) & 0xF) == 3 ? 4 * d : d;
    out.push_back(rangeOf(s.getLaddr(), (s.getRows() * s.getRowBytes() + wordBytes - 1) / wordBytes));
  } else if (isa<sahw::ExOp>(op)) {
    out.push_back({0xFFFFFFFFu, 0, 0xFFFFFFFFu});          // treated as reading everything
  }
}

struct SahwFuseVePass : public PassWrapper<SahwFuseVePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SahwFuseVePass)
  StringRef getArgument() const override { return "iree-sahw-fuse-ve"; }
  StringRef getDescription() const override { return "Merges single-stage VEs into multi-stage ones"; }

  static int stageOf(sahw::VeOp v) {
    if (v.getFunc() != 0) return 3;
    if (bitsOf(v.getB()) != NEG0) return 2;
    if (bitsOf(v.getA()) != ONE) return 1;
    return 0;
  }

  // dynamic fields allowed on merged VEs: LEN (4) and VALID (5), not added
  static bool dynOf(sahw::VeOp v, Value &len, Value &valid) {
    auto f = v.getDynFields();
    auto a = v.getDynAdd();
    for (size_t i = 0; i < f.size(); ++i) {
      if (a[i]) return false;
      if (f[i] == 4) len = v.getDyn()[i];
      else if (f[i] == 5) valid = v.getDyn()[i];
      else return false;
    }
    return true;
  }

  // v2 only applies stages to src1 (a COPY without index modes or SWAPNEG)
  static bool stageOnly(sahw::VeOp v) {
    Value len, valid;
    return v.getFp() && v.getOp() == 5 && v.getM1() == 0 && v.getP1() == 0 && v.getM2() == 0 && v.getSrc2() == 0 &&
           v.getValid() == 0 && !v.getSwapneg() && v.getPeriod() == 0 && dynOf(v, len, valid) &&
           (v.getTypes() & 3) == 3 && (v.getTypes() >> 4) == 0 && !v.getFenceBefore();
  }

  bool fuseOnce(Block &blk, int64_t d) {
    SmallVector<Operation *> ops;
    for (Operation &o : blk) ops.push_back(&o);
    for (size_t j = 0; j < ops.size(); ++j) {
      auto v2 = dyn_cast<sahw::VeOp>(ops[j]);
      if (!v2 || !stageOnly(v2)) continue;
      // the producer: the nearest earlier VE writing exactly src1
      Range t = rangeOf(v2.getSrc1(), (v2.getLength() + d - 1) / d);
      sahw::VeOp v1;
      size_t i = j;
      while (i-- > 0) {
        if (auto v = dyn_cast<sahw::VeOp>(ops[i]); v && v.getDst() == v2.getSrc1()) {
          v1 = v;
          break;
        }
        if (!isa<sahw::VeOp, sahw::LdOp, sahw::StOp, sahw::TransposeOp>(ops[i])) break;   // no merging across control
      }
      Value len1, valid1, len2, valid2;
      if (!v1 || !v1.getFp() || v1.getLength() != v2.getLength() || v1.getReduce() != 0 || v1.getValid() != 0 ||
          ((v1.getTypes() >> 2) & 3) != 3 || !dynOf(v1, len1, valid1))
        continue;
      dynOf(v2, len2, valid2);
      if (len1 != len2) continue;
      bool usesA = bitsOf(v2.getA()) != ONE, usesB = bitsOf(v2.getB()) != NEG0, usesF = v2.getFunc() != 0,
           usesR = v2.getReduce() != 0, usesV = bool(valid2);
      // stages: 1 A, 2 B, 3 FUNC, 4 VALID, 5 REDUCE, then the output conversion
      int need = usesA ? 1 : usesB ? 2 : usesF ? 3 : usesV ? 4 : usesR ? 5 : 6;
      int have = valid1 ? 4 : stageOf(v1);
      if (have >= need) continue;
      if (usesV && v1.getDynFields().size() >= 2) continue;       // at most two dynamic fields
      // the intermediate has no other reader; nothing between touches v2's destination
      bool ok = true;
      Range dst2 = rangeOf(v2.getDst(), (v2.getLength() + d - 1) / d);
      // (readers after the producer: earlier commands cannot see the change)
      for (size_t k = i + 1; k < ops.size() && ok; ++k)
        ops[k]->walk([&](Operation *op) {
          if (op == v2.getOperation() || !ok) return;
          SmallVector<Range> rs;
          readsOf(op, d, rs);
          for (const Range &r : rs)
            if (r.overlaps(t)) ok = false;
        });
      for (size_t k = i + 1; k < j && ok; ++k) {
        SmallVector<Range> rs;
        readsOf(ops[k], d, rs);
        for (const Range &r : rs)
          if (r.overlaps(dst2)) ok = false;
        if (auto v = dyn_cast<sahw::VeOp>(ops[k]); v && rangeOf(v.getDst(), 1).overlaps(dst2)) ok = false;
      }
      if (!ok) continue;
      if (usesA) v1.setAAttr(v2.getAAttr());
      if (usesB) v1.setBAttr(v2.getBAttr());
      if (usesF) v1.setFunc(v2.getFunc());
      if (usesR) {
        v1.setReduce(v2.getReduce());
        v1.setRowlen(v2.getRowlen());
      }
      if (usesV) {
        SmallVector<int32_t> f(v1.getDynFields());
        SmallVector<bool> a(v1.getDynAdd());
        v1.getDynMutable().append(valid2);
        f.push_back(5);
        a.push_back(false);
        v1.setDynFields(f);
        v1.setDynAdd(a);
      }
      v1.setTypes((v1.getTypes() & ~uint64_t(0xC)) | (v2.getTypes() & 0xC));
      v1.setDst(v2.getDst());
      v2.erase();
      return true;
    }
    return false;
  }

  void runOnOperation() override {
    getOperation()->walk([&](sahw::TemplateOp t) {
      int64_t d = 8;
      if (auto a = t->getAttrOfType<IntegerAttr>("sa.d")) d = a.getInt();
      t.walk([&](Block *blk) {
        while (fuseOnce(*blk, d)) {
        }
      });
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createSahwFuseVePass() { return std::make_unique<SahwFuseVePass>(); }
void registerSahwFuseVePass() { PassRegistration<SahwFuseVePass>(); }

void buildSahlPipeline(OpPassManager &pm, const TargetConfig &config) {
  pm.nest<func::FuncOp>().addPass(createIREEComprehensiveBufferizePass(std::nullopt, std::nullopt));
  pm.nest<func::FuncOp>().addPass(createSaToSahlPass(config));
  pm.addPass(createSahlToSahwPass(config));
  pm.addPass(createSahwFuseVePass());
}

}  // namespace mlir::iree_compiler::sa

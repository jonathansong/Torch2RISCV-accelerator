// sahw passes (docs/iree_compiler_plan.md §8.4) and the target configuration.
#include "SahwPasses.h"
#include "SahlPasses.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/MapVector.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler::sa {

// ------------------------------------------------------------------ target configuration
TargetConfig TargetConfig::fromAttr(DictionaryAttr config) {
  TargetConfig c;
  if (!config) return c;
  auto get = [&](StringRef name, int64_t &v) {
    if (auto a = config.getAs<IntegerAttr>(name)) v = a.getInt();
  };
  get("d", c.d);
  get("spad_bytes", c.spadBytes);
  get("acc_bytes", c.accBytes);
  get("max_dynamic", c.maxDynamic);
  get("dyn_fields", c.dynFields);
  get("bases", c.bases);
  get("params", c.params);
  get("gemv_ports", c.gemvPorts);
  get("gemv_ksplit", c.gemvKSplit);
  get("attn_block", c.attnBlock);
  get("chunk_block", c.chunkBlock);
  return c;
}

std::string TargetConfig::invalid() const {
  auto pow2 = [](int64_t v) { return v > 0 && (v & (v - 1)) == 0; };
  if (d != 8 && d != 16) return "d must be 8 or 16";
  if (!pow2(spadBytes) || spadBytes / d > (1 << 16) || spadBytes / d < 2 * maxDynamic)
    return "spad_bytes must be a power of two, at most 2^16 words of D bytes";
  if (!pow2(accBytes) || accBytes / (4 * d) > (1 << 16) || accBytes / (4 * d) < 64)
    return "acc_bytes must be a power of two, at most 2^16 words of 4 * D bytes";
  if (gemvPorts < 0 || gemvPorts > 4 || (gemvPorts && d != 16)) return "gemv_ports must be 0..4 (D = 16 only)";
  return "";
}

void TargetConfig::addTo(Builder &b, SmallVectorImpl<NamedAttribute> &attrs) const {
  auto add = [&](StringRef name, int64_t v) { attrs.emplace_back(b.getStringAttr(name), b.getI64IntegerAttr(v)); };
  add("d", d);
  add("spad_bytes", spadBytes);
  add("acc_bytes", accBytes);
  add("max_dynamic", maxDynamic);
  add("dyn_fields", dynFields);
  add("bases", bases);
  add("params", params);
  if (gemvPorts) add("gemv_ports", gemvPorts);
  if (!gemvKSplit) add("gemv_ksplit", 0);
  if (attnBlock) add("attn_block", attnBlock);
  if (chunkBlock) add("chunk_block", chunkBlock);
}

bool TargetConfig::ukernel(StringRef name) const {
  if (ukernels == "all") return true;
  if (ukernels == "none" || ukernels.empty()) return false;
  SmallVector<StringRef> names;
  StringRef(ukernels).split(names, ',');
  return llvm::is_contained(names, name);
}

namespace {

// ------------------------------------------------------------------ helpers
bool inSpadB(int64_t laddr) { return (uint64_t(laddr) & 0xFFFFFFFFull) >> 28 == 2; }

// Whether a command touches SPAD_B (an EX: its B operand; a DMA or VE operand there).
bool touchesSpadB(Operation *op) {
  if (isa<sahw::ExOp>(op)) return true;
  if (auto o = dyn_cast<sahw::LdOp>(op)) return inSpadB(o.getLaddr());
  if (auto o = dyn_cast<sahw::StOp>(op)) return inSpadB(o.getLaddr());
  if (auto o = dyn_cast<sahw::VeOp>(op)) return inSpadB(o.getSrc1()) || inSpadB(o.getSrc2()) || inSpadB(o.getDst());
  if (auto o = dyn_cast<sahw::TransposeOp>(op)) return inSpadB(o.getSrc()) || inSpadB(o.getDst());
  return false;
}

template <typename SectionT>
SectionT sectionOf(sahw::TemplateOp t) {
  for (Operation &op : t.getRegion().front())
    if (auto s = dyn_cast<SectionT>(op)) return s;
  return {};
}

// ------------------------------------------------------------------ sahw-split-head
struct SahwSplitHeadPass : public PassWrapper<SahwSplitHeadPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SahwSplitHeadPass)
  StringRef getArgument() const override { return "iree-sahw-split-head"; }
  StringRef getDescription() const override {
    return "Moves the body's leading loads into sahw.head (exports that do not touch SPAD_B)";
  }
  void runOnOperation() override {
    getOperation()->walk([&](sahw::TemplateOp t) {
      if (t.getUnsupported() || sectionOf<sahw::HeadOp>(t)) return;
      auto body = sectionOf<sahw::BodyOp>(t);
      if (!body) return;
      Block &blk = body.getRegion().front();
      bool spadB = false;
      body.walk([&](Operation *op) { spadB |= touchesSpadB(op); });
      if (spadB) return;
      SmallVector<Operation *> head;
      for (Operation &op : blk) {
        if (!isa<sahw::LdOp>(op)) break;
        head.push_back(&op);
      }
      if (head.empty()) return;
      OpBuilder b(body);
      auto h = sahw::HeadOp::create(b, body.getLoc());
      Block *hb = &h.getRegion().emplaceBlock();
      for (Operation *op : head) op->moveBefore(hb, hb->end());
    });
  }
};

// ------------------------------------------------------------------ sahw-assign-registers
struct SahwAssignRegistersPass : public PassWrapper<SahwAssignRegistersPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SahwAssignRegistersPass)
  StringRef getArgument() const override { return "iree-sahw-assign-registers"; }
  StringRef getDescription() const override {
    return "Numbers the registers: BASE from 0 (BASE15 is the prefixes'), setup PARAMs from 0, private PARAMs from 7 down";
  }
  void runOnOperation() override {
    getOperation()->walk([&](sahw::TemplateOp t) {
      Builder b(t.getContext());
      // More private PARAMs than registers (a prefill chunk's attention: positions and
      // RoPE rows of 16 rows): one written by an LDPARAM before any read is free after
      // its last use, so its register is reused; the others keep theirs (read as 0
      // first: cursors, loop strides). Otherwise the numbering of before.
      Block &tb = t.getRegion().front();
      int nParams = 0, nPrivate = 0;
      for (Operation &op : tb) nParams += isa<sahw::ParamOp>(op), nPrivate += isa<sahw::PrivateOp>(op);
      if (nParams + nPrivate > 8) {
        llvm::DenseMap<Operation *, int> pos;
        int i = 0;
        tb.walk([&](Operation *o) { pos[o] = i++; });   // (pre-order: the program's order)
        struct Live { int first = INT_MAX, last = -1; bool defFirst = false; };
        llvm::MapVector<Operation *, Live> live;
        for (Operation &op : tb)
          if (auto o = dyn_cast<sahw::PrivateOp>(op)) {
            Live l;
            for (OpOperand &u : o.getResult().getUses()) {
              int at = pos[u.getOwner()];
              if (at < l.first) {
                l.first = at;
                auto ld = dyn_cast<sahw::LdParamOp>(u.getOwner());
                l.defFirst = ld && ld.getTarget() == o.getResult();
              }
              l.last = std::max(l.last, at);
            }
            live[o] = l;
          }
        // (loops: a use inside a loop body lives to the loop's end)
        tb.walk([&](Operation *o) {
          if (o->getNumRegions() == 0 || o->getParentOp() != t.getOperation()) return;
          int end = pos[o];
          o->walk([&](Operation *x) { end = std::max(end, pos[x]); });
          for (auto &[po, l] : live)
            if (l.first > pos[o] && l.first <= end) l.last = std::max(l.last, end);
            else if (l.last > pos[o] && l.last <= end) l.last = end;
        });
        int next = nParams;
        SmallVector<int> freeRegs;
        SmallVector<std::pair<int, int>> busy;   // (last use, register)
        SmallVector<std::pair<int, Operation *>> order;
        for (auto &[po, l] : live) order.push_back({l.first, po});
        llvm::sort(order, [](auto &a, auto &b) { return a.first < b.first; });
        for (auto [first, po] : order) {
          Live &l = live[po];
          for (size_t j = 0; j < busy.size();)
            if (busy[j].first < first) freeRegs.push_back(busy[j].second), busy.erase(busy.begin() + j);
            else ++j;
          int reg;
          if (l.defFirst && !freeRegs.empty()) reg = freeRegs.pop_back_val();
          else reg = next++;
          if (reg > 7) {
            if (getenv("SA_PARAM_DEBUG"))
              for (auto &[q, lq] : live)
                llvm::errs() << "private " << lq.first << ".." << lq.last << (lq.defFirst ? " ldparam" : " read first")
                             << " " << *(*q->getUsers().begin()) << "\n";
            po->emitError("more than 8 PARAM registers");
            return signalPassFailure();
          }
          cast<sahw::PrivateOp>(po).setRegAttr(b.getI64IntegerAttr(reg));
          if (l.defFirst) busy.push_back({l.last, reg});
        }
        int pi = 0;
        for (Operation &op : tb)
          if (auto o = dyn_cast<sahw::ParamOp>(op)) o.setRegAttr(b.getI64IntegerAttr(pi++));
        int nb = 0;
        for (Operation &op : tb)
          if (auto o = dyn_cast<sahw::BaseOp>(op)) {
            if (nb >= 15) {
              o.emitError("more than 15 BASE registers (BASE15 is the prefixes')");
              return signalPassFailure();
            }
            o.setRegAttr(b.getI64IntegerAttr(nb++));
          }
        return;
      }
      int nextBase = 0, nextParam = 0, nextPrivate = 7;
      for (Operation &op : t.getRegion().front()) {
        if (auto o = dyn_cast<sahw::BaseOp>(op)) {
          if (nextBase >= 15) {
            o.emitError("more than 15 BASE registers (BASE15 is the prefixes')");
            return signalPassFailure();
          }
          o.setRegAttr(b.getI64IntegerAttr(nextBase++));
        } else if (auto o = dyn_cast<sahw::ParamOp>(op)) {
          o.setRegAttr(b.getI64IntegerAttr(nextParam++));
        } else if (auto o = dyn_cast<sahw::PrivateOp>(op)) {
          o.setRegAttr(b.getI64IntegerAttr(nextPrivate--));
        }
        if (nextParam > nextPrivate + 1) {
          op.emitError("more than 8 PARAM registers");
          return signalPassFailure();
        }
      }
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createSahwSplitHeadPass() { return std::make_unique<SahwSplitHeadPass>(); }
std::unique_ptr<Pass> createSahwAssignRegistersPass() { return std::make_unique<SahwAssignRegistersPass>(); }

void registerSaCodegenPass();
void registerSahwPasses() {
  registerSaCodegenPass();
  registerSaToSahlPass();
  registerSahlTilePass();
  registerSahlPlanMemoryPass();
  registerSahlSchedulePass();
  registerSahlExpandKernelsPass();
  registerSahlToSahwPass();
  registerSahwFuseVePass();
  PassRegistration<SahwSplitHeadPass>();
  PassRegistration<SahwAssignRegistersPass>();
}

}  // namespace mlir::iree_compiler::sa

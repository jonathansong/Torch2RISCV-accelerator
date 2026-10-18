// sa preprocessing: one dispatch for decode's gated FFN (docs/fusion_design.md §4.2).
//
// IREE forms one dispatch per contraction, plus separate ones for a reduction
// that cannot fuse with its producer. Decode's SwiGLU (GeGLU) chain
//     a = x_q . W1,  b = x_q . W2        (two contractions with the same LHS)
//     h = act(a) * b                     (elementwise, both scales, s_x)
//     amax = max |h|;  h_q = to_i8(h * 127 / amax)
// is then four dispatches, each storing its result to DDR for the next. Here
// the chain is wrapped in one flow.dispatch.region before dispatch creation
// (FormDispatchRegions leaves ops already in a region alone, and the
// intra-dispatch elementwise fusion still runs inside it), so the sa backend
// compiles it as one template with the intermediates in local memory.
//
// The chain, structurally (no names):
//   - two contraction generics (int8 x int8 -> int32, one reduction loop) with
//     the same LHS value, a vector (every dimension but the last 1: decode),
//     and the same result type;
//   - their forward slice through non-contraction generics and tensor
//     reshapes, an op joining it when all of its operands are in the slice or
//     do not depend on it (so down_proj, a contraction, and everything after
//     it stay outside);
//   - accepted when some op of the slice depends on both contractions (the
//     gating) and the slice yields an int8 tensor (the quantized h).
// The region is placed at the slice's last op: every operand from outside
// must be defined before it, every use outside must come after it.
// Runs only when the module targets the sa device and --iree-sa-fuse-ffn is set.
#include "SAPasses.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::sa {

bool targetsSA(ModuleOp module) {
  bool found = false;
  module.walk([&](Operation *op) {
    for (NamedAttribute a : op->getAttrs())
      a.getValue().walk([&](IREE::HAL::DeviceTargetAttr t) {
        if (t.getDeviceID().getValue() == "sa") found = true;
      });
    return found ? WalkResult::interrupt() : WalkResult::advance();
  });
  return found;
}

namespace {

// int8 x int8 -> int32 with one reduction loop: x[.., k] . W[.., k, ..]
bool isInt8Contraction(linalg::GenericOp g) {
  if (g.getNumDpsInputs() != 2 || g.getNumDpsInits() != 1 || g.getNumReductionLoops() != 1) return false;
  auto et = [](Value v) { return cast<ShapedType>(v.getType()).getElementType(); };
  if (!et(g.getDpsInputs()[0]).isInteger(8) || !et(g.getDpsInputs()[1]).isInteger(8) ||
      !et(g.getDpsInits()[0]).isInteger(32))
    return false;
  Block &b = g.getRegion().front();
  auto y = cast<linalg::YieldOp>(b.getTerminator());
  auto add = y.getOperand(0).getDefiningOp<arith::AddIOp>();
  return add && (add.getLhs() == b.getArgument(2) || add.getRhs() == b.getArgument(2));
}

// a vector: every dimension but the last one 1 (decode's x; prefill's has M rows)
bool isVector(Value v) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  if (!t || !t.hasStaticShape() || t.getRank() == 0) return false;
  for (int64_t i = 0; i + 1 < t.getRank(); ++i)
    if (t.getDimSize(i) != 1) return false;
  return true;
}

// a and b compute the same value: the same SSA value, or the same operation
// (regions included) on equivalent operands. Each linear layer quantizes its
// input itself, so the two projections' x_q are two identical chains here.
bool equivalent(Value a, Value b, int depth = 0) {
  if (a == b) return true;
  if (depth > 64) return false;
  auto ra = dyn_cast<OpResult>(a), rb = dyn_cast<OpResult>(b);
  if (!ra || !rb || ra.getResultNumber() != rb.getResultNumber()) return false;
  // values inside the ops' regions: matched pairwise as the comparison goes
  llvm::DenseMap<Value, Value> inner;
  return OperationEquivalence::isEquivalentTo(
      ra.getOwner(), rb.getOwner(),
      [&](Value x, Value y) {
        if (auto it = inner.find(x); it != inner.end()) return success(it->second == y);
        return success(equivalent(x, y, depth + 1));
      },
      [&](Value x, Value y) { inner[x] = y; }, OperationEquivalence::IgnoreLocations);
}

static const bool dbg = getenv("SA_FFN_DEBUG") != nullptr;

struct FormFfnRegionsPass : public PassWrapper<FormFfnRegionsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormFfnRegionsPass)

  FormFfnRegionsPass() = default;
  explicit FormFfnRegionsPass(bool enabled) { this->enabled = enabled; }
  FormFfnRegionsPass(const FormFfnRegionsPass &other) : PassWrapper(other) {}

  StringRef getArgument() const override { return "iree-sa-form-ffn-regions"; }
  StringRef getDescription() const override {
    return "Wraps decode's gated FFN (two projections, gating, amax, int8) in one dispatch region";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, linalg::LinalgDialect, tensor::TensorDialect>();
  }

  // the chain of the contractions c1, c2 (nullopt: none)
  std::optional<SmallVector<Operation *>> chainOf(linalg::GenericOp c1, linalg::GenericOp c2) {
    Block *blk = c1->getBlock();
    llvm::SetVector<Operation *> s;
    s.insert(c1);
    s.insert(c2);
    // depends on the slice (transitively, within the block)
    llvm::DenseMap<Operation *, bool> dep;
    std::function<bool(Operation *)> dependsOnSlice = [&](Operation *op) -> bool {
      if (!op || op->getBlock() != blk) return false;
      if (s.contains(op)) return true;
      if (auto it = dep.find(op); it != dep.end()) return it->second;
      dep[op] = false;
      bool r = llvm::any_of(op->getOperands(), [&](Value v) { return dependsOnSlice(v.getDefiningOp()); });
      dep[op] = r;
      return r;
    };
    auto eligible = [&](Operation *u) {
      if (u->getBlock() != blk || s.contains(u)) return false;
      if (auto g = dyn_cast<linalg::GenericOp>(u)) {
        // (no result: an assertion's check, stripped later; it goes with the values it checks)
        if (isInt8Contraction(g) || g->getNumResults() > 1) return false;
      } else if (!isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(u)) {
        return false;
      }
      for (Value v : u->getOperands()) {
        Operation *def = v.getDefiningOp();
        if (def && s.contains(def)) continue;
        dep.clear();
        if (dependsOnSlice(def)) return false;
      }
      return true;
    };
    for (bool grown = true; grown;) {
      grown = false;
      for (size_t i = 0; i < s.size(); ++i)
        for (Value r : s[i]->getResults())
          for (Operation *u : r.getUsers())
            if (eligible(u)) s.insert(u), grown = true;
    }
    // the gating: an op depending on both contractions
    auto reaches = [&](Operation *from, Operation *to) {
      SmallVector<Operation *> work{to};
      llvm::DenseSet<Operation *> seen;
      while (!work.empty()) {
        Operation *o = work.pop_back_val();
        if (o == from) return true;
        if (!seen.insert(o).second) continue;
        for (Value v : o->getOperands())
          if (Operation *d = v.getDefiningOp(); d && s.contains(d)) work.push_back(d);
      }
      return false;
    };
    bool gate = llvm::any_of(s, [&](Operation *o) { return o != c1 && o != c2 && reaches(c1, o) && reaches(c2, o); });
    if (dbg) llvm::errs() << "ffn: slice of " << s.size() << " ops, gating " << gate << "\n";
    if (!gate) return std::nullopt;
    SmallVector<Operation *> ops(s.begin(), s.end());
    llvm::sort(ops, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
    Operation *root = ops.back();
    bool i8Out = false;
    for (Operation *o : ops) {
      for (Value v : o->getOperands())
        if (Operation *d = v.getDefiningOp(); d && d->getBlock() == blk && !s.contains(d) && !d->isBeforeInBlock(root)) {
          if (dbg) llvm::errs() << "ffn: operand defined after the region: " << *d << "\n";
          return std::nullopt;                       // an operand defined after the region's place
        }
      for (Value r : o->getResults())
        for (Operation *u : r.getUsers()) {
          if (s.contains(u)) continue;
          if (u->getBlock() == blk && !root->isBeforeInBlock(u)) {
            if (dbg) llvm::errs() << "ffn: a use before the region: " << *u << "\n";
            return std::nullopt;                     // a use before it
          }
          if (cast<ShapedType>(r.getType()).getElementType().isInteger(8)) i8Out = true;
        }
    }
    if (dbg) llvm::errs() << "ffn: int8 output " << i8Out << "\n";
    if (!i8Out) return std::nullopt;
    return ops;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!enabled || (!force && !targetsSA(module))) return;
    SmallVector<SmallVector<Operation *>> chains;
    llvm::DenseSet<Operation *> taken;
    module.walk([&](Block *blk) {
      // contractions of a vector LHS; pairs with an equivalent LHS
      SmallVector<linalg::GenericOp> cs;
      for (Operation &op : *blk)
        if (auto g = dyn_cast<linalg::GenericOp>(&op); g && isInt8Contraction(g) && isVector(g.getDpsInputs()[0]))
          cs.push_back(g);
      for (size_t i = 0; i < cs.size(); ++i)
        for (size_t j = i + 1; j < cs.size(); ++j) {
          if (taken.contains(cs[i]) || taken.contains(cs[j])) continue;
          if (cs[i]->getResult(0).getType() != cs[j]->getResult(0).getType()) continue;
          Value x1 = cs[i].getDpsInputs()[0], x2 = cs[j].getDpsInputs()[0];
          bool eq = equivalent(x1, x2);
          if (dbg) llvm::errs() << "ffn: pair " << i << "," << j << " lhs equivalent " << eq << "\n";
          if (!eq) continue;
          // the second projection reads the first one's x_q (its own copy then dead)
          if (x1 != x2) {
            Operation *d1 = x1.getDefiningOp();
            if (d1 && d1->getBlock() == blk && !d1->isBeforeInBlock(cs[j])) continue;
            cs[j]->setOperand(0, x1);
          }
          auto ch = chainOf(cs[i], cs[j]);
          if (!ch || llvm::any_of(*ch, [&](Operation *o) { return taken.contains(o); })) {
            if (x1 != x2) cs[j]->setOperand(0, x2);
            continue;
          }
          // a zero-filled accumulator of its own for each (shared, bufferization copies one)
          Value init = cs[i].getDpsInits()[0];
          if (init == cs[j].getDpsInits()[0]) {
            if (auto fl = init.getDefiningOp<linalg::FillOp>()) {
              OpBuilder ob(cs[j]);
              Operation *empty = fl.getDpsInits()[0].getDefiningOp();
              IRMapping m;
              if (empty && isa<tensor::EmptyOp>(empty)) m.map(fl.getDpsInits()[0], ob.clone(*empty)->getResult(0));
              cs[j]->setOperand(cs[j].getDpsInitOperand(0)->getOperandNumber(), ob.clone(*fl, m)->getResult(0));
            }
          }
          for (Operation *o : *ch) taken.insert(o);
          chains.push_back(*ch);
        }
    });
    IRRewriter rw(&getContext());
    for (auto &ops : chains) {
      Operation *root = ops.back();
      rw.setInsertionPoint(root);
      auto region = IREE::Flow::wrapOpInDispatchRegion(rw, root);
      if (failed(region)) return signalPassFailure();
      ops.pop_back();
      if (!ops.empty() && failed(IREE::Flow::movePrecedingOpsIntoDispatchRegion(rw, ops, *region)))
        return signalPassFailure();
    }
    if (report)
      llvm::errs() << "sa form-ffn-regions: " << chains.size() << " FFN chains in one dispatch each\n";
  }

  Option<bool> enabled{*this, "enabled", llvm::cl::desc("Form the regions (the plugin's --iree-sa-fuse-ffn)"),
                       llvm::cl::init(true)};
  Option<bool> force{*this, "force", llvm::cl::desc("Run without an sa device target (tests)"), llvm::cl::init(false)};
  Option<bool> report{*this, "report", llvm::cl::desc("Print how many chains were wrapped"), llvm::cl::init(false)};
};

}  // namespace

std::unique_ptr<Pass> createFormFfnRegionsPass(bool enabled) { return std::make_unique<FormFfnRegionsPass>(enabled); }

void registerFormFfnRegionsPass() { PassRegistration<FormFfnRegionsPass>(); }

}  // namespace mlir::iree_compiler::sa

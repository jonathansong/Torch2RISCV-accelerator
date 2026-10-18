// sa dispatch creation: one dispatch for decode's attention (docs/fusion_design.md §6.4.1).
//
// IREE forms the attention of a decode step (and of each prefill row) as a
// chain of dispatches: the q path (dequantization, QK-norm, RoPE, amax per
// head, int8), the scores q . K^T over the cache, the masked max and the
// probabilities, their sum, p . V, the amax and the int8 output. Each stores
// its result to DDR for the next. Here those dispatch regions are merged into
// one once IREE has formed them (a plugin hook after region formation, IREE
// patch 0004), so each piece keeps the form the sa backend knows from today's
// dispatches, and T, the cache slices' length, is a dynamic dimension of the
// merged region as of the pieces.
//
// The chain, structurally (no names), over the flow.dispatch.region ops of a block:
//   - the anchor: a region with a contraction (a linalg op with a reduction
//     loop and two operands of rank >= 1) whose result has a dynamic dimension
//     (the scores, [heads, G, T]);
//   - backward: a region producing a value the set reads, without a
//     contraction, all of whose results are read only by the set (the q path;
//     q_proj, the K / V paths and values shared with them stay outside);
//   - forward: a region reading the set, without a static contraction (o_proj
//     stays outside), whose other operands do not depend on the set (the
//     probabilities, the sum, p . V, the amax, the int8 output);
//   - the tensor reshapes between the set's regions go inside (one dropping unit
//     dims of a dynamic tensor as a rank-reducing slice);
//   - accepted when the set holds a second contraction over a dynamic
//     dimension (p . V) and every value used outside has a static shape.
// The merged region is placed at the set's last op: every use outside must
// come after it. The dynamic sizes inside it (T and the cache slices' lengths)
// get the bound --iree-sa-max-dynamic (util.assume.int): the pieces' dynamic
// intermediates are no longer results stored to DDR but tensors inside the
// dispatch, and a host fallback (VMVX) needs a bound to allocate them. The cache writes stay outside, before it (no DDR RAW inside a
// dispatch; the scoreboard does not track DDR).
// Runs only when the module targets the sa device and --iree-sa-fuse-attn is set.
#include "SAPasses.h"

#include <functional>
#include <map>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir::iree_compiler::sa {
namespace {

using IREE::Flow::DispatchRegionOp;

static const bool dbg = getenv("SA_ATTN_DEBUG") != nullptr;

bool isReshape(Operation *op) { return isa_and_nonnull<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op); }

// a linalg op with a reduction loop and two operands of rank >= 1
bool isContraction(linalg::LinalgOp l) {
  if (l.getNumReductionLoops() == 0) return false;
  int n = 0;
  for (OpOperand *in : l.getDpsInputOperands())
    if (auto t = dyn_cast<ShapedType>(in->get().getType()); t && t.getRank() > 0) ++n;
  return n >= 2;
}

struct RegionKind {
  bool contraction = false, dynContraction = false, staticContraction = false, dynResultContraction = false;
};

RegionKind kindOf(DispatchRegionOp r) {
  RegionKind k;
  r.getBody().walk([&](linalg::LinalgOp l) {
    if (!isContraction(l)) return;
    k.contraction = true;
    auto dyn = [](Value v) { auto t = dyn_cast<ShapedType>(v.getType()); return t && !t.hasStaticShape(); };
    bool d = llvm::any_of(l->getOperands(), dyn);
    (d ? k.dynContraction : k.staticContraction) = true;
    if (llvm::any_of(l->getResults(), dyn)) k.dynResultContraction = true;
  });
  return k;
}

struct FormAttnRegionsPass : public PassWrapper<FormAttnRegionsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormAttnRegionsPass)

  FormAttnRegionsPass() = default;
  FormAttnRegionsPass(bool report, int64_t maxDynamic) {
    this->report = report;
    this->maxDynamic = maxDynamic;
  }
  FormAttnRegionsPass(const FormAttnRegionsPass &other) : PassWrapper(other) {}

  StringRef getArgument() const override { return "iree-sa-form-attn-regions"; }
  StringRef getDescription() const override {
    return "Merges the dispatch regions of decode's attention (q path, scores, probabilities, p . V, int8 output) into one";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, IREE::Util::UtilDialect, linalg::LinalgDialect, tensor::TensorDialect>();
  }

  // the region producing v, through reshapes outside regions (null: none)
  static DispatchRegionOp producer(Value v) {
    Operation *d = v.getDefiningOp();
    while (isReshape(d)) d = d->getOperand(0).getDefiningOp();
    return dyn_cast_or_null<DispatchRegionOp>(d);
  }

  // the ops of blk reading v (an op inside a region counts as its region),
  // through reshapes outside regions
  static void readers(Block *blk, Value v, SmallVectorImpl<Operation *> &out) {
    for (Operation *u : v.getUsers()) {
      Operation *a = blk->findAncestorOpInBlock(*u);
      if (!a) continue;
      if (isReshape(a) && a == u) {
        readers(blk, a->getResult(0), out);
        continue;
      }
      out.push_back(a);
    }
  }

  // the values a region reads from outside its body
  static SmallVector<Value> captured(DispatchRegionOp r) {
    llvm::SetVector<Value> vals;
    getUsedValuesDefinedAbove(r.getBody(), vals);
    for (Value d : r.getResultDims()) vals.insert(d);
    return vals.takeVector();
  }

  // the set of the anchor (empty: not an attention)
  SmallVector<Operation *> chainOf(Block *blk, DispatchRegionOp anchor) {
    llvm::SetVector<Operation *> s;
    s.insert(anchor);
    // ops of blk depending on the set (block order: operands are defined before)
    auto dependents = [&]() {
      llvm::DenseSet<Operation *> dep(s.begin(), s.end());
      Operation *first = *llvm::min_element(s, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
      for (Operation &op : llvm::make_range(first->getIterator(), blk->end())) {
        if (dep.contains(&op)) continue;
        bool d = false;
        op.walk([&](Operation *in) {
          for (Value v : in->getOperands())
            if (Operation *a = v.getDefiningOp() ? blk->findAncestorOpInBlock(*v.getDefiningOp()) : nullptr;
                a && a != &op && dep.contains(a))
              d = true;
        });
        if (d) dep.insert(&op);
      }
      return dep;
    };
    for (bool grown = true; grown;) {
      grown = false;
      // backward: producers read only by the set, without a contraction
      for (size_t i = 0; i < s.size(); ++i) {
        auto r = dyn_cast<DispatchRegionOp>(s[i]);
        for (Value v : captured(r)) {
          DispatchRegionOp p = producer(v);
          if (!p || s.contains(p) || kindOf(p).contraction) continue;
          SmallVector<Operation *> rd;
          for (Value res : p->getResults()) readers(blk, res, rd);
          if (rd.empty() || !llvm::all_of(rd, [&](Operation *o) { return s.contains(o); })) continue;
          if (dbg) llvm::errs() << "attn: backward " << p->getResultTypes() << "\n";
          s.insert(p), grown = true;
        }
      }
      // forward: readers without a static contraction whose other operands do not depend on the set
      // (what depends on the set as it grows: what depended on it before, the new members among it)
      auto dep = dependents();
      for (size_t i = 0; i < s.size(); ++i) {
        SmallVector<Operation *> rd;
        for (Value res : s[i]->getResults()) readers(blk, res, rd);
        for (Operation *o : rd) {
          auto c = dyn_cast<DispatchRegionOp>(o);
          if (!c || s.contains(c) || kindOf(c).staticContraction) continue;
          bool ok = true;
          for (Value v : captured(c)) {
            Operation *d = v.getDefiningOp();
            Operation *a = d ? blk->findAncestorOpInBlock(*d) : nullptr;
            while (a && isReshape(a) && a->getOperand(0).getDefiningOp())   // through reshapes
              a = blk->findAncestorOpInBlock(*a->getOperand(0).getDefiningOp());
            if (a && !s.contains(a) && dep.contains(a)) ok = false;
          }
          if (!ok) continue;
          if (dbg) llvm::errs() << "attn: forward " << c->getResultTypes() << "\n";
          s.insert(c), grown = true;
        }
      }
    }
    // a second contraction over a dynamic dimension (p . V)
    int dynC = llvm::count_if(s, [](Operation *o) { return kindOf(cast<DispatchRegionOp>(o)).dynContraction; });
    if (dbg) llvm::errs() << "attn: " << s.size() << " regions, " << dynC << " over T\n";
    if (dynC < 2) return {};
    // the reshapes between the set's regions, reading the set
    SmallVector<Operation *> ops(s.begin(), s.end());
    llvm::sort(ops, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
    for (Operation *op = ops.front(); op != ops.back(); op = op->getNextNode())
      if (isReshape(op))
        if (Operation *d = op->getOperand(0).getDefiningOp(); d && s.contains(d)) s.insert(op);
    ops.assign(s.begin(), s.end());
    llvm::sort(ops, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
    // uses outside: after the last op, static shapes
    Operation *last = ops.back();
    for (Operation *o : ops)
      for (Value r : o->getResults())
        for (Operation *u : r.getUsers()) {
          Operation *a = blk->findAncestorOpInBlock(*u);
          if (!a || s.contains(a)) continue;
          if (!last->isBeforeInBlock(a) || !cast<ShapedType>(r.getType()).hasStaticShape()) {
            if (dbg) llvm::errs() << "attn: a use outside before the region or dynamic: " << *a << "\n";
            return {};
          }
        }
    return ops;
  }

  // one region of the ops (block order) at the last one
  static void merge(ArrayRef<Operation *> ops, int64_t maxDynamic) {
    Operation *last = ops.back();
    llvm::DenseSet<Operation *> in(ops.begin(), ops.end());
    SmallVector<Value> escaping;
    for (Operation *o : ops)
      for (Value r : o->getResults())
        if (llvm::any_of(r.getUsers(), [&](Operation *u) {
              Operation *a = last->getBlock()->findAncestorOpInBlock(*u);
              return a && !in.contains(a);
            }))
          escaping.push_back(r);
    OpBuilder b(last->getContext());
    b.setInsertionPointAfter(last);
    auto region = DispatchRegionOp::create(b, last->getLoc(), ValueRange(escaping).getTypes(),
                                           /*dynamicDims=*/ValueRange(), /*workload=*/ValueRange());
    Block &body = region.getBody().emplaceBlock();
    b.setInsertionPointToStart(&body);
    IRMapping m;
    for (Operation *o : ops) {
      auto r = dyn_cast<DispatchRegionOp>(o);
      if (!r) {
        // a collapse dropping unit dims of a dynamic tensor: a rank-reducing slice (inside a
        // dispatch the tensor is a bounded allocation's subview; VMVX cannot collapse that)
        auto c = dyn_cast<tensor::CollapseShapeOp>(o);
        auto st = c ? c.getSrcType() : RankedTensorType();
        if (c && !st.hasStaticShape() && llvm::all_of(c.getReassociationIndices(), [&](ReassociationIndices g) {
              return llvm::count_if(g, [&](int64_t d) { return st.getDimSize(d) != 1; }) <= 1;
            })) {
          Value src = m.lookupOrDefault(c.getSrc());
          SmallVector<OpFoldResult> offs(st.getRank(), b.getIndexAttr(0)), strides(st.getRank(), b.getIndexAttr(1)),
              sizes = tensor::getMixedSizes(b, c.getLoc(), src);
          m.map(c.getResult(), tensor::ExtractSliceOp::create(b, c.getLoc(), c.getResultType(), src, offs, sizes,
                                                              strides).getResult());
          continue;
        }
        b.clone(*o, m);
        continue;
      }
      for (Operation &inner : r.getBody().front().without_terminator()) b.clone(inner, m);
      auto ret = cast<IREE::Flow::ReturnOp>(r.getBody().front().getTerminator());
      for (auto [res, v] : llvm::zip(r->getResults(), ret.getOperands())) m.map(res, m.lookupOrDefault(v));
    }
    SmallVector<Value> rets;
    for (Value v : escaping) rets.push_back(m.lookup(v));
    IREE::Flow::ReturnOp::create(b, last->getLoc(), rets);
    // the dynamic sizes from outside, bounded
    if (maxDynamic > 0) {
      llvm::SetVector<Value> sizes;
      body.walk([&](Operation *op) {
        SmallVector<Value> szs;
        if (auto e = dyn_cast<tensor::EmptyOp>(op)) szs.append(e.getDynamicSizes().begin(), e.getDynamicSizes().end());
        if (auto x = dyn_cast<tensor::ExtractSliceOp>(op)) szs.append(x.getSizes().begin(), x.getSizes().end());
        for (Value v : szs)
          if (!region.getBody().isAncestor(v.getParentRegion())) sizes.insert(v);
      });
      b.setInsertionPointToStart(&body);
      auto bound = IREE::Util::IntAssumptionAttr::get(b.getContext(), uint64_t(0), uint64_t(maxDynamic), std::nullopt);
      for (Value v : sizes) {
        Operation *as = IREE::Util::AssumeIntOp::create(b, v.getLoc(), v, bound);
        v.replaceUsesWithIf(as->getResult(0), [&](OpOperand &use) {
          return use.getOwner() != as && region.getBody().isAncestor(use.getOwner()->getParentRegion());
        });
      }
    }
    for (auto [v, nv] : llvm::zip(escaping, region->getResults()))
      v.replaceUsesWithIf(nv, [&](OpOperand &use) {
        Operation *a = last->getBlock()->findAncestorOpInBlock(*use.getOwner());
        return a && !in.contains(a);
      });
    for (Operation *o : llvm::reverse(ops)) o->erase();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!force && !targetsSA(module)) return;
    SmallVector<SmallVector<Operation *>> chains;
    llvm::DenseSet<Operation *> taken;
    SmallVector<SmallVector<Operation *>> blockChains;
    module.walk([&](Block *blk) {
      for (Operation &op : *blk) {
        auto r = dyn_cast<DispatchRegionOp>(&op);
        if (!r || taken.contains(r) || !kindOf(r).dynResultContraction) continue;
        SmallVector<Operation *> ops = chainOf(blk, r);
        if (ops.empty() || llvm::any_of(ops, [&](Operation *o) { return taken.contains(o); })) continue;
        for (Operation *o : ops) taken.insert(o);
        blockChains.push_back(std::move(ops));
      }
      // a prefill chunk's rows (rows=false: left unfused, fusion_design.md §6.6): chains that
      // read one producer read by nothing else (the chunk's q, computed for its rows). Fused,
      // each row would load and transpose the cache once a row; a decode layer's chain has
      // its own q (values shared with the k path, e.g. RoPE tables, are read outside too)
      llvm::DenseMap<Operation *, int> chainOf_;
      for (auto [i, ch] : llvm::enumerate(blockChains))
        for (Operation *o : ch) chainOf_[o] = int(i);
      llvm::DenseSet<int> rowChains;
      llvm::DenseSet<Operation *> seen;
      // (through reshapes and slices: a row's part of the chunk's q is a slice of it)
      auto isView = [](Operation *op) { return isReshape(op) || isa_and_nonnull<tensor::ExtractSliceOp>(op); };
      auto srcRegion = [&](Value v) -> DispatchRegionOp {
        Operation *d = v.getDefiningOp();
        while (isView(d)) d = d->getOperand(0).getDefiningOp();
        return dyn_cast_or_null<DispatchRegionOp>(d);
      };
      std::function<void(Value, SmallVectorImpl<Operation *> &)> viewReaders = [&](Value v, SmallVectorImpl<Operation *> &out) {
        for (Operation *u : v.getUsers()) {
          Operation *a = blk->findAncestorOpInBlock(*u);
          if (!a) continue;
          if (isView(a) && a == u) viewReaders(a->getResult(0), out);
          else out.push_back(a);
        }
      };
      for (auto &ch : blockChains)
        for (Operation *o : ch) {
          auto reg = dyn_cast<DispatchRegionOp>(o);
          if (!reg) continue;
          for (Value v : captured(reg)) {
            DispatchRegionOp p = srcRegion(v);
            if (!p || chainOf_.count(p) || !seen.insert(p).second) continue;
            SmallVector<Operation *> rd;
            for (Value res : p->getResults()) viewReaders(res, rd);
            llvm::DenseSet<int> by;
            bool onlyChains = !rd.empty();
            for (Operation *u : rd) {
              auto it = chainOf_.find(u);
              if (it == chainOf_.end()) onlyChains = false;
              else by.insert(it->second);
            }
            if (onlyChains && by.size() > 1) rowChains.insert(by.begin(), by.end());
          }
        }
      if (dbg && !rowChains.empty()) llvm::errs() << "attn: " << rowChains.size() << " chains are a prefill chunk's rows\n";
      for (auto [i, ch] : llvm::enumerate(blockChains))
        if (rows || !rowChains.contains(int(i))) chains.push_back(std::move(ch));
      blockChains.clear();
    });
    std::map<int, int> sizes;   // regions merged -> chains
    for (auto &ops : chains) {
      ++sizes[llvm::count_if(ops, [](Operation *o) { return isa<DispatchRegionOp>(o); })];
      merge(ops, maxDynamic);
    }
    if (report) {
      llvm::errs() << "sa form-attn-regions: " << chains.size() << " attention chains in one dispatch each (regions x chains:";
      for (auto [n, c] : sizes) llvm::errs() << " " << n << "x" << c;
      llvm::errs() << ")\n";
    }
  }

  Option<bool> rows{*this, "rows", llvm::cl::desc("Also merge the attention of each row of a prefill chunk"),
                    llvm::cl::init(false)};
  Option<bool> force{*this, "force", llvm::cl::desc("Run without an sa device target (tests)"), llvm::cl::init(false)};
  Option<int64_t> maxDynamic{*this, "max-dynamic",
                             llvm::cl::desc("Bound of the dynamic sizes inside a merged region (0: none)"),
                             llvm::cl::init(0)};
  Option<bool> report{*this, "report", llvm::cl::desc("Print how many chains were merged"), llvm::cl::init(false)};
};

}  // namespace

std::unique_ptr<Pass> createFormAttnRegionsPass(bool report, int64_t maxDynamic) {
  return std::make_unique<FormAttnRegionsPass>(report, maxDynamic);
}

void registerFormAttnRegionsPass() { PassRegistration<FormAttnRegionsPass>(); }

}  // namespace mlir::iree_compiler::sa

// sahl-expand-kernels (docs/iree_compiler_plan.md §8.17 item 2, §8.20): a
// micro-kernel's schedule written out in sahl, so that sahl-to-sahw only
// translates it (one command per operation) and the schedule is visible in
// the IR: its buffers and their places, the order of the loads (the
// prefetch), the scopes that release local memory.
//
// Prefill's linear layer (M rows of int8 x, sahl.kernel "linear" with
// chunk_tiles from sahl-schedule), as the micro-kernel of C6.P:
//   - the A strips of the row blocks (sahl.strip);
//   - the per-row and scalar epilogue inputs loaded once and put in the
//     broadcast layout (sahl.bcast);
//   - two sets of per-column input buffers and two SPAD_B banks of weight
//     tiles: chunk ci uses set / bank ci & 1, the next chunk's loads follow the
//     first row block's EX of this chunk (the prefetch);
//   - per chunk and row block, a sahl.scope (its local memory released after
//     it): the EX (sahl.mma), the epilogue as a generic over the chunk
//     [D rows, nc tiles, D] of the local buffers, the store.
// Decode's linear layer (one row of x), as the micro-kernel of C3 / C4:
//   - x loaded, then copied over the D rows into the A strip (sahl.strip of
//     a local vector), the scalar epilogue inputs put in the broadcast layout;
//   - chunk ci's accumulator, per-chunk inputs and result at fixed words of
//     ACC bank ci & 1 (sa.word: the C3 layout), its weights in SPAD_B bank
//     ci & 1, the epilogue's temps in the same ACC bank first
//     (sa.prefer_bank);
//   - with many chunks (sahl-schedule's `loop`) the first pair inside a
//     sahl.loop (LOOP_END): each DMA marked sa.advance moves on by two
//     chunks per iteration; the rest after it;
//   - chunk 0's weights in the template's prefix (sa.prefix) when the kernel
//     is the first operation that emits commands (they may then load during
//     an earlier dispatch).
// The order of the operations and of the first uses of the local buffers is
// the order in which the lowering emitted the commands and allocated the
// words before, so the descriptors are the same (the golden corpus).
#include "SahlDialect.h"
#include "SahlKernels.h"
#include "SahlPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::iree_compiler::sa {
namespace {

struct Expander {
  OpBuilder b;
  Location loc;
  int64_t d;

  Value sub(Value v, ArrayRef<int64_t> offs, ArrayRef<int64_t> sizes) {
    SmallVector<OpFoldResult> o, s, st;
    for (int64_t x : offs) o.push_back(b.getIndexAttr(x));
    for (int64_t x : sizes) s.push_back(b.getIndexAttr(x));
    for (size_t i = 0; i < offs.size(); ++i) st.push_back(b.getIndexAttr(1));
    return memref::SubViewOp::create(b, loc, v, o, s, st);
  }
  Value alloc(ArrayRef<int64_t> shape, Type et, StringRef mem, StringRef layout) {
    auto a = memref::AllocOp::create(b, loc, MemRefType::get(shape, et));
    a->setAttr("sa.mem", b.getStringAttr(mem));
    if (!layout.empty()) a->setAttr("sa.layout", b.getStringAttr(layout));
    return a;
  }
  // rows [r0, r0 + D) and tiles [c0, c0 + nc) of a [M, nt, D] or [M, N] view
  Value rowsTiles(Value v, int64_t r0, int64_t c0, int64_t nc) {
    auto mt = cast<MemRefType>(v.getType());
    if (mt.getRank() == 3) return sub(v, {r0, c0, 0}, {d, nc, d});
    return sub(v, {r0, c0 * d}, {d, nc * d});
  }
};

// The kernel's operations and the buffers only they used.
void eraseKernel(sahl::KernelOp k) {
  SmallVector<Operation *> allocs;
  k.walk([&](Operation *o) {
    for (Value v : o->getOperands())
      if (auto a = v.getDefiningOp<memref::AllocOp>(); a && a->getBlock() == k->getBlock()) allocs.push_back(a);
  });
  k.erase();
  for (Operation *a : allocs)
    if (a->getBlock() && a->use_empty()) a->erase();
}

// Prefill's linear layer (LinearPlan with rows): false when it has a form the
// expansion does not write out (the kernel then stays for the lowering).
bool expandLinearRows(sahl::KernelOp k, LinearPlan p, int64_t d) {
  auto nca = k->getAttrOfType<IntegerAttr>("chunk_tiles");
  if (!nca) return false;
  auto wt = cast<MemRefType>(p.w.getType());
  const int64_t K = wt.getDimSize(1), nt = wt.getDimSize(0), M = p.rows, nb = M / d, nc = nca.getInt();
  if (nc <= 0 || nt % nc) return false;
  const int64_t nch = nt / nc;
  if (!cast<MemRefType>(p.acc.getType()).getElementType().isInteger(32)) return false;
  linalg::GenericOp epi = p.epi;
  llvm::DenseMap<int, int> kindOf;                  // epilogue input -> 1 chunked, 2 column, 3 row, 4 scalar
  llvm::DenseMap<int, Value> srcOf;
  for (auto &[i, s] : p.chunked) kindOf[i] = 1, srcOf[i] = s;
  for (auto &[i, s] : p.cols) kindOf[i] = 2, srcOf[i] = s;
  for (auto &[i, s] : p.rowv) kindOf[i] = 3, srcOf[i] = s;
  for (auto &[i, s] : p.whole) {
    if (cast<MemRefType>(s.getType()).getRank() != 0) return false;
    kindOf[i] = 4, srcOf[i] = s;
  }
  Expander e{OpBuilder(k), k.getLoc(), d};
  MLIRContext *ctx = k.getContext();
  Type f32 = Float32Type::get(ctx), i8 = IntegerType::get(ctx, 8), i32 = IntegerType::get(ctx, 32);
  // the A strips of the row blocks
  SmallVector<Value> strips;
  for (int64_t rb = 0; rb < nb; ++rb) {
    Value s = e.alloc({d, K}, i8, "spad_a", "strip");
    sahl::StripOp::create(e.b, e.loc, e.sub(p.x, {rb * d, 0}, {d, K}), s);
    strips.push_back(s);
  }
  // per-row and scalar inputs: loaded once, in the broadcast layout
  llvm::DenseMap<int, Value> bc;
  for (auto &[i, s] : p.rowv) {
    Value v = e.alloc({M}, f32, "acc", "packed");
    sahl::LoadOp::create(e.b, e.loc, s, v);
    Value w = e.alloc({M}, f32, "acc", "bcast");
    sahl::BcastOp::create(e.b, e.loc, v, w);
    bc[i] = w;
  }
  for (auto &[i, s] : p.whole) {
    Value v = e.alloc({}, f32, "acc", "packed");
    sahl::LoadOp::create(e.b, e.loc, s, v);
    Value w = e.alloc({}, f32, "acc", "bcast");
    sahl::BcastOp::create(e.b, e.loc, v, w);
    bc[i] = w;
  }
  // two sets of per-column buffers, two banks of weight tiles
  SmallVector<SmallVector<Value>> cols(2);
  for (int set = 0; set < 2; ++set)
    for (size_t m = 0; m < p.cols.size(); ++m) {
      Value c = e.alloc({nc, d}, f32, "acc", "packed");
      sahl::ReserveOp::create(e.b, e.loc, c, false);
      cols[set].push_back(c);
    }
  Value tiles[2];
  for (int bank = 0; bank < 2; ++bank) {
    tiles[bank] = e.alloc({nc, K, d}, i8, "spad_b", "");
    tiles[bank].getDefiningOp()->setAttr("sa.bank", e.b.getI64IntegerAttr(bank));
  }
  auto loadChunk = [&](int64_t ci) {
    int64_t c0 = ci * nc;
    sahl::LoadOp::create(e.b, e.loc, e.sub(p.w, {c0, 0, 0}, {nc, K, d}), tiles[ci & 1]);
    for (size_t m = 0; m < p.cols.size(); ++m)
      sahl::LoadOp::create(e.b, e.loc, e.sub(p.cols[m].second, {c0, 0}, {nc, d}), cols[ci & 1][m]);
  };
  loadChunk(0);
  Value y = p.store.getDst();
  for (int64_t ci = 0; ci < nch; ++ci) {
    const int64_t c0 = ci * nc;
    for (int64_t rb = 0; rb < nb; ++rb) {
      auto scope = sahl::ScopeOp::create(e.b, e.loc, /*keep=*/true);
      OpBuilder::InsertionGuard guard(e.b);
      e.b.setInsertionPointToEnd(&scope.getBody().emplaceBlock());
      Value acc = e.alloc({d, nc, d}, i32, "acc", "packed");
      sahl::MmaOp::create(e.b, e.loc, strips[rb], tiles[ci & 1], acc, false);
      if (rb == 0 && ci + 1 < nch) loadChunk(ci + 1);   // the prefetch into the other bank
      Value ys = e.rowsTiles(y, rb * d, c0, nc);
      if (!epi) {
        sahl::StoreOp::create(e.b, e.loc, acc, ys);
        continue;
      }
      // the epilogue on the chunk: the result allocated first (as the lowering did)
      Type ot = cast<MemRefType>(epi.getDpsInits()[0].getType()).getElementType();
      Value out = e.alloc({d, nc, d}, ot, ot.isInteger(8) ? "spad_a" : "acc", "packed");
      sahl::ReserveOp::create(e.b, e.loc, out, false);
      SmallVector<Value> ins;
      int cm = 0;
      for (int in = 0; in < epi.getNumDpsInputs(); ++in) {
        if (epi.getDpsInputs()[in] == p.acc) {
          ins.push_back(acc);
          continue;
        }
        switch (kindOf.lookup(in)) {
        case 1: {
          Type et = cast<MemRefType>(epi.getDpsInputs()[in].getType()).getElementType();
          Value l = e.alloc({d, nc, d}, et, "acc", "packed");
          sahl::LoadOp::create(e.b, e.loc, e.rowsTiles(srcOf[in], rb * d, c0, nc), l);
          ins.push_back(l);
          break;
        }
        case 2: ins.push_back(cols[ci & 1][cm++]); break;
        case 3: ins.push_back(e.sub(bc[in], {rb * d}, {d})); break;
        case 4: ins.push_back(bc[in]); break;
        default: return false;
        }
      }
      auto g = linalg::GenericOp::create(e.b, e.loc, TypeRange{}, ins, ValueRange{out}, epi.getIndexingMapsArray(),
                                         epi.getIteratorTypesArray());
      IRMapping map;
      epi.getRegion().cloneInto(&g.getRegion(), map);
      sahl::StoreOp::create(e.b, e.loc, out, ys);
    }
  }
  eraseKernel(k);
  return true;
}

// Whether nothing before op in its block emits commands (the lowering's
// template body is still empty when it reaches op).
bool firstToEmit(Operation *op) {
  static const char *quiet[] = {"arith.constant", "memref.alloc", "memref.dealloc", "memref.subview", "memref.cast",
                                "memref.dim", "memref.collapse_shape", "hal.interface.binding.subspan",
                                "hal.interface.constant.load", "iree_tensor_ext.dispatch.workload.ordinal",
                                "arith.index_cast", "arith.index_castui", "arith.extui", "arith.shli", "arith.ori",
                                "linalg.fill"};
  if (op->getParentOp() && !isa<func::FuncOp>(op->getParentOp())) return false;
  for (Operation *o = op->getPrevNode(); o; o = o->getPrevNode()) {
    StringRef n = o->getName().getStringRef();
    if (o->getDialect() && o->getDialect()->getNamespace() == "util") continue;
    if (!llvm::is_contained(quiet, n)) return false;
  }
  return true;
}

// Decode's linear layer (LinearPlan without rows).
bool expandLinear(sahl::KernelOp k, LinearPlan p, const ::sa::Layout &lay, int64_t d) {
  auto nca = k->getAttrOfType<IntegerAttr>("chunk_tiles");
  if (!nca) return false;
  auto wt = cast<MemRefType>(p.w.getType());
  const int64_t K = wt.getDimSize(1), nt = wt.getDimSize(0), nc = nca.getInt();
  if (nc <= 0 || nt % nc || K % d) return false;
  const int64_t nch = nt / nc, cb = lay.cbank, oSw = d * nc, oY = d * nc + nc;
  const int64_t wbytes = nc * K * d, fbytes = 4 * nc * d;
  auto la = k->getAttrOfType<BoolAttr>("loop");
  const bool useLoop = la && la.getValue();
  linalg::GenericOp epi = p.epi;
  for (auto &[i, s] : p.whole)
    if (cast<MemRefType>(s.getType()).getRank() != 0) return false;
  if (epi)                                        // every epilogue input of a form written out below
    for (int in = 0; in < epi.getNumDpsInputs(); ++in)
      if (epi.getDpsInputs()[in] != p.acc && !llvm::any_of(p.whole, [&](auto &w) { return w.first == in; }) &&
          !llvm::any_of(p.chunked, [&](auto &c) { return c.first == in; }))
        return false;
  auto xt = cast<MemRefType>(p.x.getType());
  auto yt = cast<MemRefType>(p.store.getDst().getType());
  if (xt.getRank() != 1 || yt.getRank() != 2) return false;
  const bool hoist = firstToEmit(k);
  Expander e{OpBuilder(k), k.getLoc(), d};
  MLIRContext *ctx = k.getContext();
  Type f32 = Float32Type::get(ctx), i8 = IntegerType::get(ctx, 8);
  // x -> the A strip (the strip's words first, as the lowering allocated them)
  Value strip = e.alloc({d, K}, i8, "spad_a", "strip");
  sahl::ReserveOp::create(e.b, e.loc, strip, false);
  Type xet = xt.getElementType();
  Value xl = e.alloc({K}, xet, xet.isInteger(8) ? "spad_a" : "acc", "packed");
  sahl::LoadOp::create(e.b, e.loc, p.x, xl);
  sahl::StripOp::create(e.b, e.loc, xl, strip);
  // scalar epilogue inputs: loaded once, in the broadcast layout
  llvm::DenseMap<int, Value> bc;
  for (auto &[i, s] : p.whole) {
    Value v = e.alloc({}, f32, "acc", "packed");
    sahl::LoadOp::create(e.b, e.loc, s, v);
    Value w = e.alloc({}, f32, "acc", "bcast");
    sahl::BcastOp::create(e.b, e.loc, v, w);
    bc[i] = w;
  }
  // the chunk buffers of the two banks (C3's ACC layout) and the weight banks
  auto placed = [&](ArrayRef<int64_t> shape, Type et, int64_t word) {
    Value v = e.alloc(shape, et, "acc", "packed");
    v.getDefiningOp()->setAttr("sa.word", e.b.getI64IntegerAttr(word));
    return v;
  };
  auto slot = [&](size_t m) { return m == 0 ? oSw : oY + nc * int64_t(m); };
  Value acc[2], out[2], tiles[2];
  SmallVector<Value> cols[2];
  for (int bank = 0; bank < 2; ++bank) {
    const int64_t c = bank * cb;
    acc[bank] = placed({nc, d}, cast<MemRefType>(p.acc.getType()).getElementType(), c);
    out[bank] = placed({nc, d}, f32, c + oY);
    for (size_t m = 0; m < p.chunked.size(); ++m) {
      Type et = cast<MemRefType>(epi.getDpsInputs()[p.chunked[m].first].getType()).getElementType();
      cols[bank].push_back(placed({nc, d}, et, c + slot(m)));
    }
    tiles[bank] = e.alloc({nc, K, d}, i8, "spad_b", "");
    tiles[bank].getDefiningOp()->setAttr("sa.bank", e.b.getI64IntegerAttr(bank));
  }
  Value y = p.store.getDst();
  auto adv = [&](Operation *o, bool dyn, int j) {
    if (dyn) o->setAttr("sa.advance", e.b.getI64IntegerAttr(j));
  };
  auto loadChunk = [&](int64_t i, bool dyn) {
    auto lw = sahl::LoadOp::create(e.b, e.loc, e.sub(p.w, {i * nc, 0, 0}, {nc, K, d}), tiles[i & 1]);
    adv(lw, dyn, 0);
    if (i == 0 && hoist) lw->setAttr("sa.prefix", e.b.getUnitAttr());
    for (size_t m = 0; m < p.chunked.size(); ++m)
      adv(sahl::LoadOp::create(e.b, e.loc, e.sub(p.chunked[m].second, {i * nc, 0}, {nc, d}), cols[i & 1][m]), dyn, 1);
  };
  auto chunkAt = [&](int64_t i, bool dyn, bool prefetch) {
    const int bank = int(i & 1);
    sahl::MmaOp::create(e.b, e.loc, strip, tiles[bank], acc[bank], false);
    if (prefetch) loadChunk(i + 1, dyn);
    Value ys = e.sub(y, {i * nc, 0}, {nc, d});
    if (!epi) {
      adv(sahl::StoreOp::create(e.b, e.loc, acc[bank], ys), dyn, 1);
      return true;
    }
    SmallVector<Value> ins;
    int cm = 0;
    for (int in = 0; in < epi.getNumDpsInputs(); ++in) {
      if (epi.getDpsInputs()[in] == p.acc) ins.push_back(acc[bank]);
      else if (bc.count(in)) ins.push_back(bc[in]);
      else if (cm < int(p.chunked.size()) && p.chunked[cm].first == in) ins.push_back(cols[bank][cm++]);
      else return false;
    }
    auto g = linalg::GenericOp::create(e.b, e.loc, TypeRange{}, ins, ValueRange{out[bank]}, epi.getIndexingMapsArray(),
                                       epi.getIteratorTypesArray());
    IRMapping map;
    epi.getRegion().cloneInto(&g.getRegion(), map);
    g->setAttr("sa.prefer_bank", e.b.getI64IntegerAttr(bank));
    adv(sahl::StoreOp::create(e.b, e.loc, out[bank], ys), dyn, 1);
    return true;
  };
  loadChunk(0, false);
  int64_t j = 0;
  if (useLoop) {
    const int64_t pairs = (nch - 1) / 2;          // the prefetch of the last pair stays in range
    if (pairs) {
      auto lp = sahl::LoopOp::create(e.b, e.loc, pairs, ArrayRef<int64_t>{2 * wbytes, 2 * fbytes});
      OpBuilder::InsertionGuard guard(e.b);
      e.b.setInsertionPointToEnd(&lp.getBody().emplaceBlock());
      if (!chunkAt(0, true, true) || !chunkAt(1, true, true)) return false;
      j = 2 * pairs;
    }
  }
  for (int64_t jj = j; jj < nch; ++jj)
    if (!chunkAt(jj, false, jj + 1 < nch)) return false;
  eraseKernel(k);
  return true;
}

struct SahlExpandKernelsPass : public PassWrapper<SahlExpandKernelsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SahlExpandKernelsPass)
  SahlExpandKernelsPass() = default;
  explicit SahlExpandKernelsPass(const TargetConfig &c) : cfg(c), fromOptions(false) {}
  SahlExpandKernelsPass(const SahlExpandKernelsPass &o) : PassWrapper(o), cfg(o.cfg), fromOptions(o.fromOptions) {}
  StringRef getArgument() const override { return "iree-sahl-expand-kernels"; }
  StringRef getDescription() const override {
    return "Writes the micro-kernels' schedules out in sahl (strips, EX, banks, prefetch, scopes)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<sahl::SahlDialect, memref::MemRefDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    if (fromOptions) {
      cfg.d = optD;
      cfg.spadBytes = optSpadKB * 1024;
      cfg.accBytes = optAccKB * 1024;
    }
    func::FuncOp f = getOperation();
    SmallVector<sahl::KernelOp> ks(f.getBody().front().getOps<sahl::KernelOp>());
    for (sahl::KernelOp k : ks) {
      if (k.getKind() != "linear") continue;
      for (Operation &o : k.getBody().front()) {
        auto g = dyn_cast<linalg::GenericOp>(&o);
        if (!g || !o.hasAttr("sahl.anchor")) continue;
        KernelMatcher km(cfg);
        if (!km.matchLinear(g)) break;
        LinearPlan p = km.linears[g.getOperation()];
        ::sa::Layout lay(uint32_t(cfg.d), uint32_t(cfg.spadBytes), uint32_t(cfg.accBytes));
        if (p.rows) expandLinearRows(k, p, cfg.d);
        else expandLinear(k, p, lay, cfg.d);
        break;
      }
    }
  }

  TargetConfig cfg;
  bool fromOptions = true;
  Option<int64_t> optD{*this, "d", llvm::cl::desc("array size D"), llvm::cl::init(8)};
  Option<int64_t> optSpadKB{*this, "spad-kb", llvm::cl::desc("SPAD size (KB, two banks)"), llvm::cl::init(128)};
  Option<int64_t> optAccKB{*this, "acc-kb", llvm::cl::desc("ACC size (KB, two banks)"), llvm::cl::init(256)};
};

}  // namespace

std::unique_ptr<Pass> createSahlExpandKernelsPass(const TargetConfig &config) {
  return std::make_unique<SahlExpandKernelsPass>(config);
}
void registerSahlExpandKernelsPass() { PassRegistration<SahlExpandKernelsPass>(); }

}  // namespace mlir::iree_compiler::sa

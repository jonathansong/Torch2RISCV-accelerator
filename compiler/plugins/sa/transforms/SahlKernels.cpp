// Kernel matching (SahlKernels.h; docs/iree_compiler_plan.md §8.14, C8 R1).
#include "SahlKernels.h"

#include <cmath>
#include <cstring>
#include <functional>

#include "SahlPasses.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir::iree_compiler::sa {

std::optional<float> constF32(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto f = dyn_cast<FloatAttr>(c.getValue()); f && f.getType().isF32()) return float(f.getValueAsDouble());
  return std::nullopt;
}

// the Lin of a dynamic size (a push constant, possibly through memref.dim of a binding)
std::optional<Lin> dynLin(Value size) {
  if (auto dim = size.getDefiningOp<memref::DimOp>()) {
    Value root = ddrRoot(dim.getSource());
    auto idx = dim.getConstantIndex();
    if (!root || !idx || root != dim.getSource()) return std::nullopt;
    auto sub = root.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    auto mt = cast<MemRefType>(root.getType());
    int k = 0;
    for (int64_t i = 0; i < *idx; ++i) k += mt.isDynamicDim(i);
    return linOf(sub.getDynamicDims()[k]);
  }
  return linOf(size);
}

void KernelMatcher::matchAll(func::FuncOp f) {
  // the micro-kernels (§8.8), as --iree-sa-ukernels allows; a gated FFN before
  // the linear layers (its contractions are not linear layers on their own)
  if (cfg.ukernel("linear") && cfg.gemvPorts)
    f.walk([&](linalg::GenericOp g) {
      if (g->getParentOp() == f.getOperation() && !owned.contains(g.getOperation())) matchFfn(g);
    });
  if (cfg.ukernel("linear"))
    f.walk([&](linalg::GenericOp g) {
      if (g->getParentOp() == f.getOperation()) matchLinear(g);
    });
  // a fused dispatch's attention (on the GEMV unit's p . V): a prefill chunk's rows, or one
  if (cfg.ukernel("attention") && cfg.gemvPorts) matchAttnChunk(f.getBody().front());
  if (cfg.ukernel("attention") && cfg.gemvPorts)
    f.walk([&](linalg::BatchMatmulOp m) {
      if (owned.contains(m.getOperation())) return;
      if (m->getParentOp() == f.getOperation()) matchAttnLayer(m);
    });
  if (cfg.ukernel("attention"))
    f.walk([&](linalg::BatchMatmulOp m) {
      if (!owned.contains(m.getOperation())) matchAttention(m);
    });
  // the other contractions: the generic lowering
  f.walk([&](linalg::LinalgOp op) {
    if (op->getParentOp() == f.getOperation() && !owned.contains(op.getOperation()) &&
        (isa<linalg::BatchMatmulOp>(op) || op.getNumReductionLoops() == 1) && op.getNumDpsInputs() == 2)
      matchContraction(op);
  });
}

void KernelMatcher::record(Operation *anchor, ArrayRef<Operation *> ops) {
  for (Operation *o : ops) owned.insert(o);
  covers[anchor].assign(ops.begin(), ops.end());
}

bool isF32Const(Value v, float want) {
  auto c = constF32(v);
  if (!c) return false;
  uint32_t a, b;
  std::memcpy(&a, &*c, 4);
  std::memcpy(&b, &want, 4);
  return a == b;
}

// qllama.to_i8 = clamp(round(nan_to_num(x)), -127, 127).to(i8) ending in fptosi:
// x and the chain's operations (the VE's int8 output conversion does all of it).
Value matchToI8(arith::FPToSIOp f, SmallVectorImpl<Operation *> &chain) {
  if (!f.getType().isInteger(8)) return {};
  const float FMAX = 3.40282347e38f, INF = INFINITY;
  auto sel = [&](Value v, arith::CmpFPredicate pred, float thr, bool thrIsSelf, float repl, Value &x) -> bool {
    auto s = v.getDefiningOp<arith::SelectOp>();
    if (!s) return false;
    auto c = s.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!c || c.getPredicate() != pred || !isF32Const(s.getTrueValue(), repl)) return false;
    x = s.getFalseValue();
    if (c.getLhs() != x) return false;
    if (thrIsSelf ? c.getRhs() != x : !isF32Const(c.getRhs(), thr)) return false;
    chain.push_back(s);
    chain.push_back(c);
    return true;
  };
  Value s5, s4, s2, s1, x;
  if (!sel(f.getIn(), arith::CmpFPredicate::UGT, 127.0f, false, 127.0f, s5)) return {};
  if (!sel(s5, arith::CmpFPredicate::ULT, -127.0f, false, -127.0f, s4)) return {};
  auto round = s4.getDefiningOp<math::RoundEvenOp>();
  if (!round) return {};
  chain.push_back(round);
  if (!sel(round.getOperand(), arith::CmpFPredicate::OEQ, -INF, false, -FMAX, s2)) return {};
  if (!sel(s2, arith::CmpFPredicate::OEQ, INF, false, FMAX, s1)) return {};
  if (!sel(s1, arith::CmpFPredicate::UNE, 0, true, 0.0f, x)) return {};
  chain.push_back(f);
  return x;
}

// Integer value of an index computation inside a linalg body at iteration
// point idx, with the scalar block argument(s) = sym (C3's evalInt).
std::optional<int64_t> evalInt(Value v, ArrayRef<int64_t> idx, int64_t sym) {
  if (auto c = getConstantIntValue(v)) return *c;
  if (isa<BlockArgument>(v)) return sym;
  Operation *op = v.getDefiningOp();
  if (!op) return std::nullopt;
  if (auto ix = dyn_cast<linalg::IndexOp>(op)) return idx[ix.getDim()];
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op))
    return evalInt(op->getOperand(0), idx, sym);
  if (op->getNumOperands() != 2) return std::nullopt;
  auto a = evalInt(op->getOperand(0), idx, sym), b = evalInt(op->getOperand(1), idx, sym);
  if (!a || !b) return std::nullopt;
  if (isa<arith::AddIOp>(op)) return *a + *b;
  if (isa<arith::SubIOp>(op)) return *a - *b;
  if (isa<arith::MulIOp>(op)) return *a * *b;
  if (isa<arith::DivSIOp>(op)) return *b ? std::optional<int64_t>(*a / *b) : std::nullopt;
  if (isa<arith::RemSIOp>(op)) return *b ? std::optional<int64_t>(*a % *b) : std::nullopt;
  if (auto c = dyn_cast<arith::CmpIOp>(op)) {
    switch (c.getPredicate()) {
    case arith::CmpIPredicate::eq: return *a == *b;
    case arith::CmpIPredicate::ne: return *a != *b;
    case arith::CmpIPredicate::slt: return *a < *b;
    case arith::CmpIPredicate::sle: return *a <= *b;
    case arith::CmpIPredicate::sgt: return *a > *b;
    case arith::CmpIPredicate::sge: return *a >= *b;
    default: return std::nullopt;
    }
  }
  return std::nullopt;
}

// Every point of an iteration space (row-major).
SmallVector<SmallVector<int64_t>> points(ArrayRef<int64_t> ranges) {
  SmallVector<SmallVector<int64_t>> out;
  int64_t n = 1;
  for (int64_t r : ranges) n *= r;
  if (n > 65536) return out;
  for (int64_t e = 0; e < n; ++e) {
    SmallVector<int64_t> p(ranges.size());
    int64_t x = e;
    for (int i = int(ranges.size()) - 1; i >= 0; --i) {
      p[i] = x % ranges[i];
      x /= ranges[i];
    }
    out.push_back(p);
  }
  return out;
}

GatherForm gatherForm(sahl::GatherOp ga, ArrayRef<SmallVector<int64_t>> dom, int nloops, bool hasScalar, int64_t d,
                      int64_t genericLoops) {
  GatherForm r;
  Block &body = *ga->getBlock();
  auto mt = cast<MemRefType>(ga.getSource().getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!mt.hasStaticShape() || failed(mt.getStridesAndOffset(strides, offset))) return r.why = "gather source layout", r;
  if (dom.empty()) return r.why = "gather over a large iteration space", r;
  auto idx = ga.getIndices();
  auto flatIndex = [&](ArrayRef<int64_t> pt, int64_t sym) -> std::optional<int64_t> {
    int64_t f = 0;
    for (unsigned i = 0; i < idx.size(); ++i) {
      auto v = evalInt(idx[i], pt, sym);
      if (!v) return std::nullopt;
      f += *v * strides[i];
    }
    return f;
  };
  bool usesSym = false;
  for (Value iv : idx) {
    std::function<bool(Value)> hasArg = [&](Value v) -> bool {
      if (isa<BlockArgument>(v)) return true;
      Operation *o = v.getDefiningOp();
      if (!o || o->getBlock() != &body) return false;
      return llvm::any_of(o->getOperands(), hasArg);
    };
    usesSym |= hasArg(iv);
  }
  if (!usesSym) {
    // a fixed permutation of a small tensor: the pair swap
    for (size_t pi = 0; pi < dom.size(); ++pi) {
      auto f = flatIndex(dom[pi], 0);
      if (!f || *f != int64_t(pi ^ 1)) return r.why = "gather with a fixed index pattern other than the pair swap", r;
    }
    return r.kind = "swap", r;
  }
  if (!hasScalar) return r.why = "gather index from a non-scalar input", r;
  // rows: index = scalar * C + iteration point
  std::optional<int64_t> C;
  bool rowGather = true;
  for (size_t pi = 0; pi < dom.size() && rowGather; ++pi) {
    auto f0 = flatIndex(dom[pi], 0), f1 = flatIndex(dom[pi], 1), f7 = flatIndex(dom[pi], 7);
    if (!f0 || !f1 || !f7 || *f0 != int64_t(pi) || *f7 - *f0 != 7 * (*f1 - *f0)) rowGather = false;
    else if (!C) C = *f1 - *f0;
    else if (*C != *f1 - *f0) rowGather = false;
  }
  // one element for the whole iteration space (the index depends on the
  // scalar only): the scalar gather, its value in every element
  bool uniform = true;
  auto u0 = flatIndex(dom.front(), 0), u1 = flatIndex(dom.front(), 1);
  for (size_t pi = 0; pi < dom.size() && uniform; ++pi)
    uniform = u0 && u1 && flatIndex(dom[pi], 0) == u0 && flatIndex(dom[pi], 1) == u1;
  if (uniform && *u0 == 0 && *u1 > 0) {
    rowGather = true;
    C = *u1;
  } else {
    uniform = false;
  }
  if (rowGather && C) {
    r.kind = nloops == 0 || uniform ? "scalar" : "row";
    r.rowElems = *C;
    return r;
  }
  // a row of a packed weight: Wp[r / D, k, r % D] over the K points
  const int64_t K = int64_t(dom.size());
  bool packed = mt.getRank() == 3 && mt.getDimSize(1) == K && mt.getDimSize(2) == d && K % d == 0 &&
                mt.getElementType().isInteger(8) && genericLoops == 1;
  for (int64_t sv : {0, 1, 7, 8, 13, 21, 1000})
    for (int64_t pi = 0; pi < K && packed; ++pi) {
      auto f = flatIndex(dom[pi], sv);
      packed = f && *f == (sv / d) * K * d + pi * d + sv % d;
    }
  if (!packed) return r.why = "gather index is not row * C + iteration index", r;
  return r.kind = "packed", r;
}

std::optional<SmallVector<SmallVector<int64_t>>> gatherDomain(linalg::GenericOp g, const TargetConfig &cfg, int &nloops) {
  auto iters = g.getIteratorTypesArray();
  auto maps = g.getIndexingMapsArray();
  nloops = int(iters.size());
  SmallVector<int64_t> ranges(nloops, -1);
  bool dyn = false;
  for (int i = 0; i < int(g->getNumOperands()); ++i) {
    auto mt = dyn_cast<MemRefType>(g->getOperand(i).getType());
    if (!mt) continue;
    for (unsigned r = 0; r < maps[i].getNumResults(); ++r) {
      auto de = dyn_cast<AffineDimExpr>(maps[i].getResult(r));
      if (!de) continue;
      int L = int(de.getPosition());
      if (!mt.isDynamicDim(r)) ranges[L] = mt.getDimSize(r);
      else if (ranges[L] < 0) ranges[L] = cfg.maxDynamic, dyn = true;
    }
  }
  for (int64_t r : ranges)
    if (r < 0) return std::nullopt;
  Block &b = g.getRegion().front();
  bool noIndex = b.getOps<linalg::IndexOp>().empty();
  bool red = llvm::any_of(iters, [](utils::IteratorType t) { return t == utils::IteratorType::reduction; });
  bool allId = llvm::all_of(maps, [](AffineMap m) { return m.isIdentity(); });
  if (nloops > 2) {                               // (as the lowering: one flat loop, else none with a gather)
    if (!(allId && !red && noIndex) || dyn) return std::nullopt;
    int64_t prod = 1;
    for (int64_t r : ranges) prod *= r;
    ranges = {prod};
    nloops = 1;
  }
  if (nloops == 2 && allId && !red && noIndex && !dyn && ranges.back() % cfg.d) {
    ranges = {ranges[0] * ranges[1]};
    nloops = 1;
  }
  return points(ranges);
}

KernelSchedule linearSchedule(const LinearPlan &p, const ::sa::Layout &lay, int64_t d, int64_t gemvPorts) {
  KernelSchedule ks;
  auto wt = cast<MemRefType>(p.w.getType());
  const int64_t k = wt.getDimSize(1), nt = wt.getDimSize(0);
  uint32_t steps = 0;
  linalg::GenericOp epi = p.epi;
  if (epi)                                        // (none: the accumulator stored as it is)
    for (Operation &o : epi.getRegion().front().without_terminator())
      if (!isa<arith::TruncIOp, arith::SIToFPOp, arith::ExtSIOp>(o)) ++steps;
  if (p.rows) {
    // prefill's rows: at most 32 tiles per chunk, a divisor of the tiles, its
    // weights in one SPAD_B bank, and a row block's scope within ACC: each ACC
    // bank above its scratch (LocalMemory, a buffer never spans two banks)
    // holds buffers of D x nc words for the accumulator, the result, every
    // chunked epilogue input and every epilogue operation's temporary (the
    // generic lowering: one per operation), after the small ones (the two sets
    // of per-column buffers, the per-row / scalar inputs and their broadcasts).
    // A small K otherwise gives chunks whose epilogue does not fit: ACC full,
    // the dispatch left to the host, whose exp / division are not the SFU's
    int64_t ops = 0;
    if (epi)
      ops = int64_t(std::distance(epi.getRegion().front().begin(), epi.getRegion().front().end())) - 1;
    const int64_t bufs = 2 + int64_t(p.chunked.size()) + ops;
    const int64_t usable = int64_t(lay.cbank) - int64_t(lay.scr);
    const int64_t small = int64_t(p.rowv.size() + p.whole.size()) * (p.rows + p.rows / d + 2);
    auto fits = [&](int64_t nc) {
      const int64_t w = d * nc, colw = 2 * int64_t(p.cols.size()) * nc;
      // the small buffers in bank 0 (allocated first), the rest wherever they fit
      return (usable - small - colw) / w + usable / w >= bufs;
    };
    int64_t nc = std::min<int64_t>(std::max<int64_t>(int64_t(lay.sbank) / rowsKBlock(k, lay, d), 1), 32);
    while (nc > 1 && (nt % nc || !fits(nc))) --nc;
    ks.chunkTiles = nc;
    return ks;
  }
  // the epilogue's extra per-chunk inputs beyond s_w need scratch words (as C3: 1 or 2 per output word)
  uint32_t extra = p.chunked.size() > 1 ? uint32_t(p.chunked.size()) - 1 : 0;
  uint32_t scratchRows = std::max<uint32_t>(steps > 2 ? 2 : 1, 1 + extra);
  ks.chunkTiles = lay.chunkTiles(uint32_t(k), uint32_t(nt), scratchRows);
  if (gemvPorts) {
    // the GEMV unit streams the weights (no SPAD_B tile): the chunk the ACC
    // scratch holds, whatever K (a K beyond one bank in K blocks, expandLinear;
    // below it the SPAD_B bound gave e.g. 2 tiles at K = 2048: a chunk's fixed
    // part, the epilogue inputs' LDs, paid 32 times, generality plan §13.4)
    int64_t cap = int64_t(lay.scr) / (d + 1 + scratchRows), nc = std::min<int64_t>(cap, nt);
    while (nc > 1 && nt % nc) --nc;
    ks.chunkTiles = nc > 0 ? nc : 0;
  }
  ks.loop = ks.chunkTiles && nt / ks.chunkTiles > 4;
  return ks;
}

int64_t rowsKBlock(int64_t k, const ::sa::Layout &lay, int64_t d) {
  const int64_t kmax = std::min<int64_t>(int64_t(lay.sbank), 65535 / d / d * d);
  if (k <= kmax) return k;
  if (k % d) return 0;
  for (int64_t kc = kmax / d * d; kc >= d; kc -= d)
    if (k % kc == 0) return kc;
  return 0;
}

sahl::LoadOp loadInto(Value local) {
  for (Operation *u : local.getUsers())
    if (auto l = dyn_cast<sahl::LoadOp>(u); l && l.getDst() == local) return l;
  return {};
}

std::optional<SmallVector<Range>> KernelMatcher::loopRanges(linalg::LinalgOp op) {
  SmallVector<Range> r(op.getNumLoops());
  SmallVector<bool> seen(op.getNumLoops(), false);
  auto maps = op.getIndexingMapsArray();
  for (OpOperand &o : op->getOpOperands()) {
    auto mt = dyn_cast<MemRefType>(o.get().getType());
    if (!mt) continue;
    AffineMap m = maps[o.getOperandNumber()];
    for (unsigned i = 0; i < m.getNumResults(); ++i) {
      auto de = dyn_cast<AffineDimExpr>(m.getResult(i));
      if (!de || seen[de.getPosition()]) continue;
      if (!mt.isDynamicDim(i)) {
        r[de.getPosition()].size = mt.getDimSize(i);
      } else {
        // the size of a subview / alloc: a push constant
        Value sz;
        if (auto sv = o.get().getDefiningOp<memref::SubViewOp>()) {
          int k = 0;
          for (unsigned j = 0; j < i; ++j) k += mt.isDynamicDim(j);
          sz = sv.getSizes()[k];
        } else if (auto al = o.get().getDefiningOp<memref::AllocOp>()) {
          int k = 0;
          for (unsigned j = 0; j < i; ++j) k += mt.isDynamicDim(j);
          sz = al.getDynamicSizes()[k];
        } else if (auto sub = o.get().getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>()) {
          int k = 0;
          for (unsigned j = 0; j < i; ++j) k += mt.isDynamicDim(j);
          sz = sub.getDynamicDims()[k];
        } else {
          continue;
        }
        auto l = dynLin(sz);
        if (!l) continue;
        r[de.getPosition()].size = cfg.maxDynamic;
        r[de.getPosition()].dyn = l;
        r[de.getPosition()].val = sz;
      }
      seen[de.getPosition()] = true;
    }
  }
  for (bool s : seen)
    if (!s) return std::nullopt;
  return r;
}

// an operand of the contraction back to DDR: its element offset per loop
std::optional<Operand> KernelMatcher::operandOf(linalg::LinalgOp op, int input) {
  Operand o;
  Value v = op.getDpsInputOperand(input)->get();
  AffineMap m = op.getIndexingMapsArray()[input];     // loops -> v indices
  // through an extension generic (extsi, a permutation) into a local
  if (auto al = v.getDefiningOp<memref::AllocOp>()) {
    linalg::GenericOp ext;
    for (Operation *u : v.getUsers())
      if (auto g = dyn_cast<linalg::GenericOp>(u); g && g.getNumDpsInits() == 1 && g.getDpsInits()[0] == v &&
                                                   g.getOperation() != op.getOperation())
        ext = g;
    if (ext) {
      if (ext.getNumDpsInputs() != 1 || ext.getNumReductionLoops()) return std::nullopt;
      Block &eb = ext.getRegion().front();
      auto es = cast<linalg::YieldOp>(eb.getTerminator()).getOperand(0).getDefiningOp<arith::ExtSIOp>();
      if (!es || es.getIn() != eb.getArgument(0) || eb.getOperations().size() != 2) return std::nullopt;
      auto em = ext.getIndexingMapsArray();
      if (!em[0].isPermutation() || !em[1].isPermutation()) return std::nullopt;
      // v indices -> ext loops -> ext input indices
      m = em[0].compose(inversePermutation(em[1])).compose(m);
      o.cover.push_back(ext);
      v = ext.getDpsInputs()[0];
    }
  }
  // a local loaded whole from DDR: its source
  if (!ddrRoot(v)) {
    auto l = loadInto(v);
    if (!l) return std::nullopt;
    o.cover.push_back(l);
    v = l.getSrc();
  }
  o.view = v;
  auto mt = cast<MemRefType>(v.getType());
  o.et = mt.getElementType();
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(mt.getStridesAndOffset(strides, offset))) return std::nullopt;
  o.coef.assign(op.getNumLoops(), 0);
  for (unsigned i = 0; i < m.getNumResults(); ++i) {
    if (auto de = dyn_cast<AffineDimExpr>(m.getResult(i))) {
      // a dynamic stride (rows of a dynamic length): -1, only for a batch loop (checked by the caller)
      if (ShapedType::isDynamic(strides[i])) {
        o.coef[de.getPosition()] = -1;
        continue;
      }
      o.coef[de.getPosition()] += strides[i];
    } else if (auto c = dyn_cast<AffineConstantExpr>(m.getResult(i)); !c || c.getValue() != 0) {
      return std::nullopt;
    }
  }
  return o;
}

bool KernelMatcher::matchContraction(linalg::LinalgOp op) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 || !linalg::isaContractionOpInterface(op))
    return false;
  auto ranges = loopRanges(op);
  if (!ranges) return false;
  auto iters = op.getIteratorTypesArray();
  ContractPlan p;
  p.op = op;
  auto a = operandOf(op, 0), b = operandOf(op, 1);
  if (!a || !b) return false;
  // the matrix: the operand with the larger output extent of its own (the
  // output loops the other operand does not have); x: the other one
  AffineMap outMap = op.getIndexingMapsArray()[2];
  auto inOut = [&](int L) { return outMap.isFunctionOfDim(L); };
  auto extent = [&](const Operand &x, const Operand &y) {
    int64_t e = 1;
    for (int L = 0; L < int(iters.size()); ++L)
      if (inOut(L) && x.coef[L] && !y.coef[L]) e *= (*ranges)[L].size;
    return e;
  };
  if (extent(*b, *a) >= extent(*a, *b)) p.x = *a, p.m = *b;
  else p.x = *b, p.m = *a;
  for (int L = 0; L < int(iters.size()); ++L) {
    if ((*ranges)[L].size == 1 && !(*ranges)[L].dyn) continue;
    bool red = iters[L] == utils::IteratorType::reduction;
    if (red) {
      if (p.kLoop >= 0 || !p.x.coef[L] || !p.m.coef[L]) return false;
      p.kLoop = L;
    } else if (p.x.coef[L] && p.m.coef[L]) {
      if (p.bLoop >= 0) return false;
      p.bLoop = L;
    } else if (p.x.coef[L] && !p.m.coef[L]) {
      if (p.gLoop >= 0 || !inOut(L)) return false;
      p.gLoop = L;
    } else if (p.m.coef[L]) {
      if (p.m.coef[L] == 1 && p.nLoop >= 0 && p.m.coef[p.nLoop] != 1) p.laneLoop = L;
      else if (p.nLoop >= 0 && p.m.coef[p.nLoop] == 1) p.laneLoop = p.nLoop, p.nLoop = L;
      else if (p.nLoop < 0) p.nLoop = L;
      else return false;
    } else {
      return false;
    }
  }
  if (p.kLoop < 0 || p.nLoop < 0 || p.x.coef[p.kLoop] != 1) return false;
  for (int L = 0; L < int(iters.size()); ++L)
    if ((p.m.coef[L] < 0 || p.x.coef[L] < 0) && L != p.bLoop && L != p.gLoop &&
        ((*ranges)[L].size != 1 || (*ranges)[L].dyn))
      return false;
  if (p.bLoop >= 0 && p.m.coef[p.bLoop] < 0) return false;
  // x rows of a dynamic length K, one per (batch, row)
  bool xRows = (p.bLoop >= 0 && p.x.coef[p.bLoop] < 0) || (p.gLoop >= 0 && p.x.coef[p.gLoop] < 0);
  if (xRows && !(*ranges)[p.kLoop].dyn) return false;
  int64_t K = (*ranges)[p.kLoop].size;
  if (p.laneLoop >= 0) {
    // (a dynamic K: the blocks of the packed V cache a pitch of at least max_dynamic rows apart)
    const bool dk = bool((*ranges)[p.kLoop].dyn);
    if ((*ranges)[p.laneLoop].size != d || p.m.coef[p.kLoop] != d ||
        (dk ? p.m.coef[p.nLoop] < K * d || p.m.coef[p.nLoop] % d : p.m.coef[p.nLoop] != K * d))
      return false;
    p.layout = MatLayout::Packed;
  } else if (p.m.coef[p.kLoop] == 1) {
    p.layout = MatLayout::RowsK;
  } else if (p.m.coef[p.nLoop] == 1) {
    p.layout = MatLayout::RowsN;
  } else {
    return false;
  }
  if (!p.m.et.isInteger(8) || !(p.x.et.isInteger(8) || p.x.et.isInteger(32))) return false;
  // the accumulator: filled with 0, consumed by one element-wise epilogue that is stored
  Value acc = op.getDpsInits()[0];
  linalg::FillOp fill;
  for (Operation *u : acc.getUsers()) {
    if (u == op.getOperation()) continue;
    if (auto f = dyn_cast<linalg::FillOp>(u)) fill = f;
    else if (auto g = dyn_cast<linalg::GenericOp>(u); g && !p.epi) p.epi = g;
    else if (auto st = dyn_cast<sahl::StoreOp>(u); st && st.getSrc() == acc && !p.store) p.store = st;
    else if (!isa<memref::DeallocOp>(u)) return false;
  }
  if (p.store) {                                  // no epilogue: the int32 accumulator stored as it is
    // (the output order (batch, row, n) is checked by the lowering)
    if (!fill || p.epi || !cast<MemRefType>(acc.getType()).getElementType().isInteger(32)) return false;
    SmallVector<Operation *> cover = {op, fill, p.store};
    cover.append(p.x.cover.begin(), p.x.cover.end());
    cover.append(p.m.cover.begin(), p.m.cover.end());
    record(op, cover);
    contracts[op.getOperation()] = p;
    return true;
  }
  if (!fill || !p.epi || p.epi.getNumDpsInits() != 1 || p.epi.getNumReductionLoops() ||
      !p.epi.getIndexingMapsArray().back().isIdentity())
    return false;
  Value out = p.epi.getDpsInits()[0];
  for (Operation *u : out.getUsers())
    if (auto s = dyn_cast<sahl::StoreOp>(u); s && s.getSrc() == out) p.store = s;
  if (!p.store) return false;
  SmallVector<Operation *> cover = {op, fill, p.epi, p.store};
  cover.append(p.x.cover.begin(), p.x.cover.end());
  cover.append(p.m.cover.begin(), p.m.cover.end());
  for (int i = 0; i < p.epi.getNumDpsInputs(); ++i) {
    Value in = p.epi.getDpsInputs()[i];
    if (in == acc) continue;
    auto l = loadInto(in);
    if (!l) return false;
    p.epiLoads.push_back({i, l.getSrc()});
    cover.push_back(l);
  }
  record(op, cover);
  contracts[op.getOperation()] = p;
  return true;
}

// A view of a local buffer as a whole: a reshape, or a subview dropping unit
// dimensions (form-attn-regions' rank-reducing slice).
static bool wholeView(Operation *o) {
  if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp>(o)) return true;
  auto sv = dyn_cast<memref::SubViewOp>(o);
  return sv && !ddrRoot(sv.getSource()) && llvm::all_of(sv.getStaticOffsets(), [](int64_t x) { return x == 0; }) &&
         sv.getSourceType().getNumDynamicDims() == sv.getType().getNumDynamicDims() &&
         (!sv.getSourceType().hasStaticShape() || sv.getSourceType().getNumElements() == sv.getType().getNumElements());
}

// A local buffer under its whole views.
static Value localRoot(Value v) {
  while (Operation *o = v.getDefiningOp()) {
    if (!wholeView(o)) break;
    v = o->getOperand(0);
  }
  return v;
}

// The users of a local buffer and of its whole views (not the views themselves).
static void usersOf(Value v, SmallVectorImpl<Operation *> &out) {
  for (Operation *u : v.getUsers()) {
    if (wholeView(u) && u->getOperand(0) == v) usersOf(u->getResult(0), out);
    else out.push_back(u);
  }
}

bool KernelMatcher::matchAttnLayer(linalg::BatchMatmulOp bmm, AttnLayerPlan *planOut,
                                   SmallVector<Operation *> *coverOut) {
  static const bool dbg = getenv("SA_ATTN_DEBUG") != nullptr;
  auto no = [&](const char *why) {
    if (dbg) llvm::errs() << "attn-layer: " << why << "\n";
    return false;
  };
  AttnLayerPlan p;
  p.bmm = bmm;
  SmallVector<Operation *> cover = {bmm};
  // the readers of the value `self` writes into the local v (or a view of it):
  // the uses after `self` in its block up to the buffer's next write (bufferization
  // reuses buffers; inside the core's sahl.kernel, at the expansion, its uses there)
  auto readers = [&](Value v, Operation *self) {
    SmallVector<Operation *> us, rs;
    usersOf(localRoot(v), us);
    Block *blk = self->getBlock();
    SmallVector<std::pair<Operation *, Operation *>> at;   // (use, its op in self's block)
    for (Operation *u : us)
      if (Operation *a = blk->findAncestorOpInBlock(*u); a && a != self && !isa<memref::DeallocOp>(u)) at.push_back({u, a});
    llvm::sort(at, [](auto &x, auto &y) { return x.second->isBeforeInBlock(y.second); });
    for (auto [u, a] : at) {
      if (a->isBeforeInBlock(self)) continue;
      auto lo = dyn_cast<linalg::LinalgOp>(u);
      bool writes = isa<linalg::FillOp>(u) ||
                    (lo && llvm::any_of(lo.getDpsInits(), [&](Value i) { return localRoot(i) == localRoot(v); }));
      bool reads = lo && llvm::any_of(lo.getDpsInputs(), [&](Value i) { return localRoot(i) == localRoot(v); });
      if (reads) rs.push_back(u);
      if (writes) break;
    }
    return rs;
  };
  // a before b (ops of the function: compared in the innermost block holding both)
  auto before = [&](Operation *a, Operation *b) {
    for (Block *blk = b->getBlock(); blk; blk = blk->getParentOp() ? blk->getParentOp()->getBlock() : nullptr)
      if (Operation *x = blk->findAncestorOpInBlock(*a)) {
        Operation *y = blk->findAncestorOpInBlock(*b);
        return x != y && x->isBeforeInBlock(y);
      }
    return false;
  };
  // the last fill of the local v (or a view) before the reduction r: its value (the fill joins the core)
  auto fillOf = [&](Value v, Operation *r) -> Value {
    SmallVector<Operation *> us;
    usersOf(localRoot(v), us);
    linalg::FillOp last;
    for (Operation *u : us)
      if (auto f = dyn_cast<linalg::FillOp>(u); f && before(f, r) && (!last || before(last, f))) last = f;
    if (!last) return {};
    cover.push_back(last);
    return last.getDpsInputs()[0];
  };
  auto local = [&](Value v) { return !ddrRoot(v) && !loadInto(localRoot(v)) && localRoot(v).getDefiningOp<memref::AllocOp>(); };
  auto onlyGeneric = [&](ArrayRef<Operation *> rs, int n) {
    return int(rs.size()) == n && llvm::all_of(rs, [](Operation *o) { return isa<linalg::GenericOp>(o); });
  };
  // q [H, G, hd] i32, local; K: an extension of the cache [T, H, hd] i8 in DDR
  Value q = bmm.getDpsInputs()[0];
  auto qt = cast<MemRefType>(q.getType());
  if (!local(q) || qt.getRank() != 3 || !qt.hasStaticShape() || !qt.getElementType().isInteger(32)) return no("q");
  p.q = q, p.H = qt.getDimSize(0), p.G = qt.getDimSize(1), p.hd = qt.getDimSize(2);
  auto k = operandOf(bmm, 1);
  if (!k || !ddrRoot(k->view)) return no("K");
  auto kt = cast<MemRefType>(k->view.getType());
  auto ksv = k->view.getDefiningOp<memref::SubViewOp>();
  // [T, H, hd], or [T, hd] for one KV head
  if (!(kt.getRank() == 3 || (kt.getRank() == 2 && p.H == 1)) || !kt.isDynamicDim(0) ||
      (kt.getRank() == 3 && kt.getDimSize(1) != p.H) || kt.getDimSize(kt.getRank() - 1) != p.hd ||
      !kt.getElementType().isInteger(8) || !ksv || ksv.getSizes().size() != 1)
    return no("K view");
  p.kCache = k->view, p.tK = ksv.getSizes()[0];
  cover.append(k->cover.begin(), k->cover.end());
  // the accumulator: filled, the scores' epilogue
  Value acc = bmm.getDpsInits()[0];
  for (Operation *u : acc.getUsers()) {
    if (u == bmm.getOperation() || isa<memref::DeallocOp>(u)) continue;
    if (isa<linalg::FillOp>(u)) cover.push_back(u);
    else if (auto g = dyn_cast<linalg::GenericOp>(u); g && !p.scoresEpi) p.scoresEpi = g;
    else return no("scores accumulator");
  }
  auto &e1 = p.scoresEpi;
  if (!e1 || e1.getNumDpsInputs() != 2 || e1.getNumDpsInits() != 1 || e1.getNumReductionLoops() ||
      e1.getDpsInputs()[0] != acc || !e1.getRegion().front().getOps<linalg::IndexOp>().empty())
    return no("scores epilogue");
  p.sq = e1.getDpsInputs()[1];
  auto sqt = cast<MemRefType>(p.sq.getType());
  if (!local(p.sq) || !sqt.hasStaticShape() || sqt.getNumElements() != p.H * p.G) return no("s_q");
  Value S = e1.getDpsInits()[0];
  cover.push_back(e1);
  // the masked max and p, reading the scores; the sum, reading p
  auto rs = readers(S, e1);
  if (!onlyGeneric(rs, 2)) return no("scores readers");
  for (Operation *o : rs) {
    auto g = cast<linalg::GenericOp>(o);
    if (g.getNumReductionLoops()) p.maxR = g;
    else p.probs = g;
  }
  if (!p.maxR || !p.probs || p.maxR.getNumDpsInputs() != 2 || p.probs.getNumDpsInputs() != 3) return no("max / p");
  p.pos = p.maxR.getDpsInputs()[0];
  auto post = cast<MemRefType>(p.pos.getType());
  if (!ddrRoot(p.pos) || post.getRank() != 0 || !post.getElementType().isInteger(64) ||
      p.probs.getDpsInputs()[0] != p.pos || localRoot(p.probs.getDpsInputs()[2]) != localRoot(p.maxR.getDpsInits()[0]))
    return no("position / max");
  Value M = p.maxR.getDpsInits()[0];
  if (!onlyGeneric(readers(M, p.maxR), 1)) return no("max readers");
  p.maxInit = fillOf(M, p.maxR);
  Value P = p.probs.getDpsInits()[0];
  auto pt = cast<MemRefType>(P.getType());
  auto pa = localRoot(P).getDefiningOp<memref::AllocOp>();
  if (!pt.getElementType().isInteger(8) || !pa || pa.getDynamicSizes().size() != 1) return no("p");
  p.tP = pa.getDynamicSizes()[0];
  cover.append({p.maxR, p.probs});
  rs = readers(P, p.probs);
  linalg::GenericOp pExt;
  if (!onlyGeneric(rs, 2)) return no("p readers");
  for (Operation *o : rs) {
    auto g = cast<linalg::GenericOp>(o);
    if (g.getNumReductionLoops()) p.sumR = g;
    else pExt = g;
  }
  if (!p.sumR || !pExt || p.sumR.getNumDpsInputs() != 1 || pExt.getNumDpsInputs() != 1) return no("sum / p extension");
  Value Sum = p.sumR.getDpsInits()[0];
  p.sumInit = fillOf(Sum, p.sumR);
  if (!p.maxInit || !p.sumInit) return no("reduction fills");
  cover.append({p.sumR, pExt});
  // p . V: the contraction of p's extension with the V cache's
  rs = readers(pExt.getDpsInits()[0], pExt);
  if (!onlyGeneric(rs, 1)) return no("p . V");
  p.pv = cast<linalg::GenericOp>(rs[0]);
  if (p.pv.getNumDpsInputs() != 2 || p.pv.getNumReductionLoops() != 1 || p.pv.getDpsInputs()[0] != pExt.getDpsInits()[0])
    return no("p . V form");
  auto v = operandOf(p.pv, 1);
  if (!v || !ddrRoot(v->view)) return no("V");
  auto vt = cast<MemRefType>(v->view.getType());
  auto vsv = v->view.getDefiningOp<memref::SubViewOp>();
  // [H, hd / D, T, D], or [hd / D, T, D] for one KV head
  const int64_t vr = vt.getRank();
  if (!(vr == 4 || (vr == 3 && p.H == 1)) || (vr == 4 && vt.getDimSize(0) != p.H) || !vt.isDynamicDim(vr - 2) ||
      vt.getDimSize(vr - 1) != d || vt.getDimSize(vr - 3) * d != p.hd || !vt.getElementType().isInteger(8) || !vsv ||
      vsv.getSizes().size() != 1)
    return no("V view");
  p.vCache = v->view, p.tV = vsv.getSizes()[0];
  cover.append(v->cover.begin(), v->cover.end());
  cover.push_back(p.pv);
  Value acc2 = p.pv.getDpsInits()[0];
  for (Operation *u : acc2.getUsers()) {
    if (u == p.pv.getOperation() || isa<memref::DeallocOp>(u)) continue;
    if (isa<linalg::FillOp>(u)) cover.push_back(u);
    else if (auto g = dyn_cast<linalg::GenericOp>(u); g && !p.pvEpi) p.pvEpi = g;
    else return no("p . V accumulator");
  }
  auto &e2 = p.pvEpi;
  if (!e2 || e2.getNumDpsInputs() != 2 || e2.getNumDpsInits() != 1 || e2.getNumReductionLoops() ||
      e2.getDpsInputs()[0] != acc2 || localRoot(e2.getDpsInputs()[1]) != localRoot(Sum) ||
      !e2.getRegion().front().getOps<linalg::IndexOp>().empty())
    return no("p . V epilogue");
  p.out = e2.getDpsInits()[0];
  auto ot = cast<MemRefType>(p.out.getType());
  if (!p.out.getDefiningOp<memref::AllocOp>() || !ot.hasStaticShape() || ot.getNumElements() != p.H * p.G * p.hd ||
      !ot.getElementType().isF32())
    return no("output");
  cover.push_back(e2);
  // the sum read only by p . V's epilogue (the other intermediates' readers checked above)
  if (readers(Sum, p.sumR).size() != 1) return no("sum readers");
  if (planOut) {
    *planOut = p;
    *coverOut = cover;
    return true;
  }
  record(bmm, cover);
  attnLayers[bmm.getOperation()] = p;
  return true;
}

// The ops of blk that write the local buffer under v (through any view of it).
static void writersOf(Block &blk, Value v, SmallVectorImpl<Operation *> &out) {
  Value r = v;
  while (Operation *o = r.getDefiningOp()) {
    if (!isa<memref::SubViewOp, memref::CollapseShapeOp, memref::ExpandShapeOp>(o)) break;
    r = o->getOperand(0);
  }
  if (!r.getDefiningOp<memref::AllocOp>()) return;
  std::function<void(Value)> walk = [&](Value x) {
    for (OpOperand &use : x.getUses()) {
      Operation *u = use.getOwner();
      if (isa<memref::SubViewOp, memref::CollapseShapeOp, memref::ExpandShapeOp>(u)) {
        walk(u->getResult(0));
        continue;
      }
      Operation *a = blk.findAncestorOpInBlock(*u);
      if (!a) continue;
      bool w = false;
      if (auto l = dyn_cast<linalg::LinalgOp>(u)) w = l.isDpsInit(&use);
      else if (auto ld = dyn_cast<sahl::LoadOp>(u)) w = ld.getDst() == x;
      else if (auto c = dyn_cast<memref::CopyOp>(u)) w = c.getTarget() == x;
      if (w) out.push_back(a);
    }
  };
  walk(r);
}

// The local buffers an op reads (its inputs; a linalg op's inits its body reads).
static void readsOf(Operation *op, SmallVectorImpl<Value> &out) {
  if (auto l = dyn_cast<linalg::LinalgOp>(op)) {
    for (OpOperand *in : l.getDpsInputOperands()) out.push_back(in->get());
    for (OpOperand &init : l.getDpsInitsMutable())
      if (l.payloadUsesValueFromOperand(&init)) out.push_back(init.get());
    // (values a body reads directly: a gather's source)
    op->walk([&](sahl::GatherOp g) { out.push_back(g.getSource()); });
    op->walk([&](memref::LoadOp g) { out.push_back(g.getMemRef()); });
    return;
  }
  if (auto c = dyn_cast<memref::CopyOp>(op)) out.push_back(c.getSource());
}

bool KernelMatcher::matchAttnChunk(Block &blk) {
  static const bool dbg = getenv("SA_ATTN_DEBUG") != nullptr;
  auto no = [&](const char *why) {
    if (dbg) llvm::errs() << "attn-chunk: " << why << "\n";
    return false;
  };
  SmallVector<linalg::BatchMatmulOp> bmms;
  for (Operation &op : blk)
    if (auto b = dyn_cast<linalg::BatchMatmulOp>(&op)) bmms.push_back(b);
  if (bmms.size() < 2) return false;
  AttnChunkPlan c;
  llvm::DenseSet<Operation *> cores;
  SmallVector<Operation *> cover;
  auto same = [](Value a, Value b) {
    if (a == b) return true;
    Operation *x = a.getDefiningOp(), *y = b.getDefiningOp();
    return x && y && OperationEquivalence::isEquivalentTo(x, y, OperationEquivalence::IgnoreLocations);
  };
  for (linalg::BatchMatmulOp bm : bmms) {
    AttnLayerPlan p;
    SmallVector<Operation *> cv;
    if (!matchAttnLayer(bm, &p, &cv)) return no("a row's core");
    if (!c.rows.empty()) {
      const AttnLayerPlan &p0 = c.rows.front();
      if (p.H != p0.H || p.G != p0.G || p.hd != p0.hd || !same(p.kCache, p0.kCache) || !same(p.vCache, p0.vCache) ||
          p.tK != p0.tK)
        return no("rows of different shapes or caches");
    }
    // the output's store into the dispatch's result
    sahl::StoreOp st;
    SmallVector<Operation *> us;
    usersOf(p.out, us);
    for (Operation *u : us)
      if (auto s = dyn_cast<sahl::StoreOp>(u); s && localRoot(s.getSrc()) == p.out) {
        if (st) return no("two stores of a row's output");
        st = s;
      }
    if (!st || !ddrRoot(st.getDst())) return no("a row's output store");
    cv.push_back(st);
    c.outDdr.push_back(st.getDst());
    for (Operation *o : cv) cores.insert(o);
    cover.append(cv.begin(), cv.end());
    c.rows.push_back(p);
  }
  // each row's q path: the last writers of q and s_q before its scores, back to DDR
  for (auto [i, p] : llvm::enumerate(c.rows)) {
    llvm::SetVector<Operation *> s;
    SmallVector<std::pair<Value, Operation *>> work = {{p.q, p.bmm}, {p.sq, p.scoresEpi}};
    while (!work.empty()) {
      auto [v, at] = work.pop_back_val();
      if (ddrRoot(v)) continue;
      SmallVector<Operation *> ws;
      writersOf(blk, v, ws);
      llvm::sort(ws, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
      // the writers before `at` back to the last one writing the whole buffer
      for (Operation *w : llvm::reverse(ws)) {
        if (!w->isBeforeInBlock(at)) continue;
        if (cores.contains(w)) return no("a q path reaching a core");
        if (!s.insert(w)) break;
        SmallVector<Value> rs;
        readsOf(w, rs);
        for (Value r : rs) work.push_back({r, w});
        bool whole = isa<linalg::FillOp, sahl::LoadOp>(w);
        if (auto g = dyn_cast<linalg::GenericOp>(w))
          whole = !g.getNumReductionLoops() && llvm::all_of(g.getDpsInitsMutable(), [&](OpOperand &o) {
            return !g.payloadUsesValueFromOperand(&o);
          });
        if (whole) break;
      }
    }
    SmallVector<Operation *> q(s.begin(), s.end());
    llvm::sort(q, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
    if (dbg) llvm::errs() << "attn-chunk: row " << i << ": q path of " << q.size() << " ops\n";
    c.qpath.push_back(q);
    for (Operation *o : q)
      if (!llvm::is_contained(cover, o)) cover.push_back(o);
  }
  // nothing else of the block between the first q path op and the last core op but
  // views, allocations and constants (the kernel takes them all)
  llvm::DenseSet<Operation *> in(cover.begin(), cover.end());
  Operation *first = *llvm::min_element(cover, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  Operation *last = *llvm::max_element(cover, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  for (Operation *o = first; o != last; o = o->getNextNode())
    if (!in.contains(o) && !isa<memref::AllocOp, memref::DeallocOp, memref::SubViewOp, memref::CollapseShapeOp,
                                memref::ExpandShapeOp, memref::DimOp, memref::CastOp, arith::ConstantOp>(o) &&
        !o->hasTrait<OpTrait::ConstantLike>() && !isa<arith::ArithDialect>(o->getDialect())) {
      if (dbg) llvm::errs() << "attn-chunk: an operation of neither a q path nor a core: " << *o << "\n";
      return false;
    }
  if (dbg) llvm::errs() << "attn-chunk: " << c.rows.size() << " rows\n";
  for (Operation *o : cover) owned.insert(o);
  covers[bmms.front().getOperation()].assign(cover.begin(), cover.end());
  attnChunks[bmms.front().getOperation()] = std::move(c);
  return true;
}

bool KernelMatcher::matchAttention(linalg::BatchMatmulOp bmm) {
  MLIRContext *ctx = bmm.getContext();
  Value a = bmm.getDpsInputs()[0], b = bmm.getDpsInputs()[1];
  AttnPlan p;
  p.bmm = bmm;
  p.scores = a.getDefiningOp<memref::AllocOp>() != nullptr;
  Value ext = p.scores ? a : b;
  p.small = p.scores ? b : a;
  linalg::GenericOp eg;
  for (Operation *u : ext.getUsers())
    if (auto g = dyn_cast<linalg::GenericOp>(u); g && g.getDpsInits()[0] == ext) eg = g;
  if (!eg || eg.getNumDpsInputs() != 1) return false;
  AffineExpr t, h, j;
  bindDims(ctx, t, h, j);
  auto m = eg.getIndexingMapsArray();
  if (m[0] != AffineMap::get(3, 0, {t, h, j}, ctx) || m[1] != AffineMap::get(3, 0, {h, t, j}, ctx)) return false;
  p.cache = eg.getDpsInputs()[0];
  auto ct = cast<MemRefType>(p.cache.getType());
  if (ct.getRank() != 3 || !ct.getElementType().isInteger(8) || !ct.isDynamicDim(0) || ct.isDynamicDim(1) ||
      ct.isDynamicDim(2))
    return false;
  auto sv = p.cache.getDefiningOp<memref::SubViewOp>();
  if (!sv || sv.getSizes().size() != 1) return false;
  auto tl = linOf(sv.getSizes()[0]);
  if (!tl) return false;
  p.T = *tl;
  p.H = ct.getDimSize(1);
  p.hs = ct.getDimSize(2);
  if (!ddrRoot(p.small)) return false;
  {
    auto st = cast<MemRefType>(p.small.getType());
    if (st.getRank() != 3 || st.getDimSize(0) != p.H) return false;
    if (p.scores ? (st.getDimSize(1) != p.hs || st.getDimSize(2) != 1) : st.getDimSize(1) != 1) return false;
  }
  Value acc = bmm.getDpsInits()[0];
  linalg::FillOp fill;
  linalg::GenericOp epi;
  for (Operation *u : acc.getUsers()) {
    if (u == bmm.getOperation()) continue;
    if (auto f = dyn_cast<linalg::FillOp>(u)) fill = f;
    else if (auto g = dyn_cast<linalg::GenericOp>(u); g && !epi) epi = g;
    else if (!isa<memref::DeallocOp>(u)) return false;
  }
  if (!fill || !epi || epi.getNumDpsInits() != 1) return false;
  Block &body = epi.getRegion().front();
  SmallVector<Operation *> ops;
  for (Operation &o : body.without_terminator()) ops.push_back(&o);
  SmallVector<Operation *> cover = {bmm, eg, fill, epi};
  if (p.scores) {
    if (ops.size() != 4 || !isa<arith::TruncIOp>(ops[0]) || !isa<arith::SIToFPOp>(ops[1]) ||
        !isa<arith::MulFOp>(ops[2]) || !isa<arith::MulFOp>(ops[3]))
      return false;
    Value o2 = ops[2]->getOperand(0) == ops[1]->getResult(0) ? ops[2]->getOperand(1) : ops[2]->getOperand(0);
    Value o3 = ops[3]->getOperand(0) == ops[2]->getResult(0) ? ops[3]->getOperand(1) : ops[3]->getOperand(0);
    auto ba = dyn_cast<BlockArgument>(o2);
    auto c = constF32(o3);
    if (!ba || !c) return false;
    Value sqLocal = epi.getDpsInputs()[ba.getArgNumber()];
    auto maps = epi.getIndexingMapsArray();
    if (maps[ba.getArgNumber()].getNumResults() != 1 || maps[ba.getArgNumber()].getResult(0) != getAffineDimExpr(0, ctx))
      return false;
    auto l = loadInto(sqLocal);
    if (!l) return false;
    p.sq = l.getSrc();
    p.scale = *c;
    cover.push_back(l);
  } else {
    if (ops.size() != 3 || !isa<arith::TruncIOp>(ops[0]) || !isa<arith::SIToFPOp>(ops[1]) ||
        !isa<arith::MulFOp>(ops[2]))
      return false;
    Value o2 = ops[2]->getOperand(0) == ops[1]->getResult(0) ? ops[2]->getOperand(1) : ops[2]->getOperand(0);
    auto c = constF32(o2);
    if (!c) return false;
    p.scale = *c;
  }
  Value out = epi.getDpsInits()[0];
  for (Operation *u : out.getUsers())
    if (auto s = dyn_cast<sahl::StoreOp>(u); s && s.getSrc() == out) p.store = s;
  if (!p.store) return false;
  cover.push_back(p.store);
  record(bmm, cover);
  attns[bmm.getOperation()] = p;
  return true;
}

// decode's GEMV contraction (t, j, k): x[k] Wp[t, k, j] -> acc[t, j], int8 x
// and weights loaded from DDR, the accumulator zero-filled: (x load, w load, fill)
// a copy generic out = in (identity maps), or null
static linalg::GenericOp asCopy(Operation *op) {
  auto g = dyn_cast<linalg::GenericOp>(op);
  if (!g || g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1 || g.getNumReductionLoops() ||
      !llvm::all_of(g.getIndexingMapsArray(), [](AffineMap m) { return m.isIdentity(); }))
    return {};
  auto y = cast<linalg::YieldOp>(g.getRegion().front().getTerminator());
  return y.getOperand(0) == g.getRegion().front().getArgument(0) ? g : linalg::GenericOp();
}

// The zero initialization of an accumulator: a fill with 0, or a copy of a
// zero-filled buffer (two accumulators from one fill: bufferization copies one)
static Operation *zeroInit(Value acc) {
  for (Operation *u : acc.getUsers()) {
    if (auto f = dyn_cast<linalg::FillOp>(u); f && f.getDpsInits()[0] == acc)
      return getConstantIntValue(f.getDpsInputs()[0]) == 0 ? u : nullptr;
    if (auto c = asCopy(u); c && c.getDpsInits()[0] == acc) {
      Value src = c.getDpsInputs()[0];
      for (Operation *v : src.getUsers())
        if (auto f = dyn_cast<linalg::FillOp>(v); f && f.getDpsInits()[0] == src && getConstantIntValue(f.getDpsInputs()[0]) == 0 &&
                                                   f->isBeforeInBlock(c))
          return u;
      return nullptr;
    }
  }
  return nullptr;
}

static bool gemvContraction(linalg::GenericOp con, sahl::LoadOp &lx, sahl::LoadOp &lw, Operation *&fill) {
  MLIRContext *ctx = con.getContext();
  AffineExpr d0, d1, d2;
  bindDims(ctx, d0, d1, d2);
  auto maps = con.getIndexingMapsArray();
  auto it = con.getIteratorTypesArray();
  if (con.getNumDpsInputs() != 2 || con.getNumDpsInits() != 1 || it.size() != 3 ||
      it[2] != utils::IteratorType::reduction || maps[0] != AffineMap::get(3, 0, {d2}, ctx) ||
      maps[1] != AffineMap::get(3, 0, {d0, d2, d1}, ctx) || maps[2] != AffineMap::get(3, 0, {d0, d1}, ctx))
    return false;
  Block &b = con.getRegion().front();
  auto y = cast<linalg::YieldOp>(b.getTerminator());
  auto add = y.getOperand(0).getDefiningOp<arith::AddIOp>();
  if (!add) return false;
  Value prod = add.getLhs() == b.getArgument(2) ? add.getRhs() : add.getLhs();
  auto mul = prod.getDefiningOp<arith::MulIOp>();
  auto isExt = [&](Value v, int arg) {
    auto e = v.getDefiningOp<arith::ExtSIOp>();
    return e && e.getIn() == b.getArgument(arg);
  };
  if (!mul || (!(isExt(mul.getLhs(), 0) && isExt(mul.getRhs(), 1)) && !(isExt(mul.getLhs(), 1) && isExt(mul.getRhs(), 0))))
    return false;
  lx = loadInto(con.getDpsInputs()[0]);
  lw = loadInto(con.getDpsInputs()[1]);
  if (!lx || !lw) return false;
  auto xt = cast<MemRefType>(lx.getSrc().getType()), wt = cast<MemRefType>(lw.getSrc().getType());
  if (!xt.getElementType().isInteger(8) || xt.getRank() != 1 || !wt.getElementType().isInteger(8) || wt.getRank() != 3)
    return false;
  fill = zeroInit(con.getDpsInits()[0]);
  return fill != nullptr;
}

bool KernelMatcher::matchFfn(linalg::GenericOp con) {
  sahl::LoadOp lx, lw;
  Operation *fill;
  if (!gemvContraction(con, lx, lw, fill)) return false;
  // the accumulator: the contraction, its zero initialization, one collapse to
  // [N] (and the copy initializing the other accumulator from it)
  auto collapseOf = [](Value acc, Operation *c, Operation *fl) -> memref::CollapseShapeOp {
    memref::CollapseShapeOp cs;
    for (Operation *u : acc.getUsers()) {
      if (u == c || u == fl || isa<memref::DeallocOp>(u)) continue;
      if (auto cp = asCopy(u); cp && cp.getDpsInputs()[0] == acc && cp->isBeforeInBlock(c)) continue;
      auto x = dyn_cast<memref::CollapseShapeOp>(u);
      if (!x || cs || cast<MemRefType>(x.getType()).getRank() != 1) return {};
      cs = x;
    }
    return cs;
  };
  auto cs = collapseOf(con.getDpsInits()[0], con, fill);
  if (!cs || !cs->hasOneUse()) return false;
  auto epi = dyn_cast<linalg::GenericOp>(*cs->getUsers().begin());
  if (!epi || epi.getNumDpsInits() != 1 || epi.getNumLoops() != 1 || epi.getNumReductionLoops() ||
      !epi.getIndexingMapsArray().back().isIdentity())
    return false;
  auto ot = cast<MemRefType>(epi.getDpsInits()[0].getType());
  const int64_t N = ot.getNumElements();
  if (!ot.getElementType().isF32() || !ot.hasStaticShape()) return false;
  FfnPlan p;
  p.epi = epi;
  p.out = epi.getDpsInits()[0];
  p.x = lx.getSrc();
  SmallVector<Operation *> cover = {epi, lx};
  int nacc = 0;
  for (int i = 0; i < epi.getNumDpsInputs(); ++i) {
    Value in = epi.getDpsInputs()[i];
    AffineMap m = epi.getIndexingMapsArray()[i];
    if (auto c = in.getDefiningOp<memref::CollapseShapeOp>()) {
      // an accumulator of one of the two contractions (this one, its partner)
      linalg::GenericOp ci;
      for (Operation *u : c.getSrc().getUsers())
        if (auto g = dyn_cast<linalg::GenericOp>(u); g && g.getNumDpsInits() == 1 && g.getDpsInits()[0] == c.getSrc() &&
                                                     g != epi && g.getNumReductionLoops() == 1)
          ci = g;
      sahl::LoadOp lxi, lwi;
      Operation *fi;
      if (!ci || nacc == 2 || !m.isIdentity() || !gemvContraction(ci, lxi, lwi, fi) || lxi != lx ||
          collapseOf(c.getSrc(), ci, fi) != c || cast<MemRefType>(lwi.getSrc().getType()) != cast<MemRefType>(lw.getSrc().getType()))
        return false;
      p.con[nacc] = ci;
      p.w[nacc] = lwi.getSrc();
      p.acc[nacc] = c.getSrc();
      p.accIn[nacc] = i;
      cover.append({ci, fi, lwi});                 // (the collapse: a view, hoisted before the kernel)
      ++nacc;
      continue;
    }
    auto l = loadInto(in);
    if (!l) return false;
    for (Operation *u : in.getUsers())
      if (u != epi.getOperation() && u != l.getOperation() && !isa<memref::DeallocOp>(u)) return false;
    auto it = cast<MemRefType>(in.getType());
    if (it.getRank() == 0 && m.getNumResults() == 0 && it.getElementType().isF32()) p.whole.push_back({i, l.getSrc()});
    else if (it.getRank() == 1 && m.isIdentity() && it.getElementType().isF32() && it.getNumElements() == N)
      p.chunked.push_back({i, l.getSrc()});
    else return false;
    cover.push_back(l);
  }
  if (nacc != 2 || (p.con[0] != con && p.con[1] != con)) return false;
  // x used by the two contractions only; anchored at the one that comes first
  for (Operation *u : con.getDpsInputs()[0].getUsers())
    if (u != p.con[0].getOperation() && u != p.con[1].getOperation() && u != lx.getOperation() && !isa<memref::DeallocOp>(u))
      return false;
  auto wt = cast<MemRefType>(p.w[0].getType());
  if (wt.getDimSize(2) != d || wt.getDimSize(0) * d != N || wt.getDimSize(1) % d) return false;
  if (p.con[1]->isBeforeInBlock(p.con[0])) {
    std::swap(p.con[0], p.con[1]), std::swap(p.w[0], p.w[1]), std::swap(p.acc[0], p.acc[1]);
    std::swap(p.accIn[0], p.accIn[1]);
  }
  if (p.con[0] != con || owned.contains(p.con[1].getOperation())) return false;
  record(con, cover);
  ffns[con.getOperation()] = p;
  return true;
}

// the pattern (nothing else may use its buffers); records the operations it covers
bool KernelMatcher::matchLinear(linalg::GenericOp con) {
  MLIRContext *ctx = con.getContext();
  AffineExpr d0, d1, d2, d3;
  bindDims(ctx, d0, d1, d2, d3);
  auto maps = con.getIndexingMapsArray();
  auto it = con.getIteratorTypesArray();
  if (con.getNumDpsInputs() != 2 || con.getNumDpsInits() != 1) return false;
  // decode: (t, j, k) x[k] Wp[t, k, j]; prefill: (m, t, j, k) x[m, k] Wp[t, k, j]
  bool rows4 = it.size() == 4 && it[3] == utils::IteratorType::reduction &&
               maps[0] == AffineMap::get(4, 0, {d0, d3}, ctx) && maps[1] == AffineMap::get(4, 0, {d1, d3, d2}, ctx) &&
               maps[2] == AffineMap::get(4, 0, {d0, d1, d2}, ctx);
  if (!rows4 && (it.size() != 3 || it[2] != utils::IteratorType::reduction ||
                 maps[0] != AffineMap::get(3, 0, {d2}, ctx) || maps[1] != AffineMap::get(3, 0, {d0, d2, d1}, ctx) ||
                 maps[2] != AffineMap::get(3, 0, {d0, d1}, ctx)))
    return false;
  Block &b = con.getRegion().front();
  auto y = cast<linalg::YieldOp>(b.getTerminator());
  auto add = y.getOperand(0).getDefiningOp<arith::AddIOp>();
  if (!add) return false;
  Value prod = add.getLhs() == b.getArgument(2) ? add.getRhs() : add.getLhs();
  auto mul = prod.getDefiningOp<arith::MulIOp>();
  if (!mul) return false;
  auto isExt = [&](Value v, int arg) {
    auto e = v.getDefiningOp<arith::ExtSIOp>();
    return e && e.getIn() == b.getArgument(arg);
  };
  if (!(isExt(mul.getLhs(), 0) && isExt(mul.getRhs(), 1)) && !(isExt(mul.getLhs(), 1) && isExt(mul.getRhs(), 0)))
    return false;
  LinearPlan p;
  p.con = con;
  auto lx = loadInto(con.getDpsInputs()[0]), lw = loadInto(con.getDpsInputs()[1]);
  SmallVector<Operation *> xOps;
  if (!lx && rows4) return false;                 // the strip of D rows: int8 x in DDR
  if (!lx) {
    // x through its i8 -> i32 extension (part of the linear layer, as C3)
    Value xi = con.getDpsInputs()[0];
    linalg::GenericOp ext;
    for (Operation *u : xi.getUsers())
      if (auto g = dyn_cast<linalg::GenericOp>(u); g && g != con && g.getNumDpsInits() == 1 && g.getDpsInits()[0] == xi)
        ext = g;
    if (!ext || ext.getNumDpsInputs() != 1 ||
        !llvm::all_of(ext.getIndexingMapsArray(), [](AffineMap m) { return m.isIdentity(); }))
      return false;
    Block &eb = ext.getRegion().front();
    auto ey = cast<linalg::YieldOp>(eb.getTerminator());
    auto es = ey.getOperand(0).getDefiningOp<arith::ExtSIOp>();
    if (!es || es.getIn() != eb.getArgument(0) || eb.getOperations().size() != 2) return false;
    for (Operation *u : xi.getUsers())
      if (u != ext.getOperation() && u != con.getOperation() && !isa<memref::DeallocOp>(u)) return false;
    lx = loadInto(ext.getDpsInputs()[0]);
    if (!lx) return false;
    xOps.push_back(ext);
  }
  if (!lx || !lw) return false;
  p.x = lx.getSrc();
  p.w = lw.getSrc();
  auto wt = cast<MemRefType>(p.w.getType());
  auto xt = cast<MemRefType>(p.x.getType());
  if (!wt.getElementType().isInteger(8) || wt.getRank() != 3 || wt.getDimSize(2) != d ||
      !(xt.getElementType().isInteger(8) || xt.getElementType().isInteger(32)))
    return false;
  if (rows4) {
    p.rows = xt.getDimSize(0);
    if (!xt.getElementType().isInteger(8) || p.rows % d) return false;
  }
  p.acc = con.getDpsInits()[0];
  linalg::FillOp fill;
  for (Operation *u : p.acc.getUsers()) {
    if (u == con.getOperation()) continue;
    if (auto f = dyn_cast<linalg::FillOp>(u)) fill = f;
    else if (auto e = dyn_cast<linalg::GenericOp>(u); e && !p.epi) p.epi = e;
    else if (auto st = dyn_cast<sahl::StoreOp>(u); st && st.getSrc() == p.acc && !p.store) p.store = st;
    else if (!isa<memref::DeallocOp>(u)) return false;
  }
  if (!fill || !getConstantIntValue(fill.getDpsInputs()[0]) || *getConstantIntValue(fill.getDpsInputs()[0]) != 0)
    return false;
  if (p.store) {                                   // the accumulator stored as it is (no epilogue)
    if (p.epi) return false;
    const int64_t kk = cast<MemRefType>(p.w.getType()).getDimSize(1);
    if ((kk * d > 65535 && !rows4 && !gemvHolds(kk)) || (p.rows / d) * kk > 2 * int64_t(lay.sbank)) return false;
    if (rows4 && !rowsKBlock(kk, lay, d)) return false;
    // decode's form: the K its schedule holds (as below)
    if (!rows4 && !gemvHolds(kk) &&
        (kk > int64_t(lay.sbank) || kk + kk / d > 2 * int64_t(lay.sbank) || lay.acc0 + kk / d + 2 > lay.cbank))
      return false;
    SmallVector<Operation *> cover = {con.getOperation(), fill.getOperation(), p.store.getOperation(),
                                      lx.getOperation(), lw.getOperation()};
    cover.append(xOps.begin(), xOps.end());
    record(con, cover);
    linears[con.getOperation()] = p;
    return true;
  }
  if (!p.epi) return false;
  linalg::GenericOp e = p.epi;
  if (e.getNumDpsInits() != 1 || e.getNumReductionLoops() || e.getNumLoops() != (rows4 ? 3 : 2) ||
      !e.getIndexingMapsArray().back().isIdentity())
    return false;
  Value out = e.getDpsInits()[0];
  for (Operation *u : out.getUsers()) {
    if (u == e.getOperation()) continue;
    if (auto s = dyn_cast<sahl::StoreOp>(u); s && s.getSrc() == out && !p.store) p.store = s;
    else if (!isa<memref::DeallocOp>(u)) return false;
  }
  // the result: fp32, or int8 ending in to_i8 (the KV cache's rows: prefill's,
  // and decode's V projection quantized straight into the cache)
  Type oet = cast<MemRefType>(out.getType()).getElementType();
  if (!p.store || !(oet.isF32() || oet.isInteger(8))) return false;
  SmallVector<Operation *> cover = {con, fill, e, p.store, lx, lw};
  cover.append(xOps.begin(), xOps.end());
  for (int i = 0; i < e.getNumDpsInputs(); ++i) {
    Value in = e.getDpsInputs()[i];
    if (in == p.acc) continue;
    auto l = loadInto(in);
    if (!l) return false;
    for (Operation *u : in.getUsers())
      if (u != e.getOperation() && u != l.getOperation() && !isa<memref::DeallocOp>(u)) return false;
    AffineMap em = e.getIndexingMapsArray()[i];
    bool f32 = cast<MemRefType>(in.getType()).getElementType().isF32();
    if (em.isIdentity()) {
      // fp32, or int32 (another linear layer's accumulator: SwiGLU's up projection)
      if (!f32 && !cast<MemRefType>(in.getType()).getElementType().isInteger(32)) return false;
      p.chunked.push_back({i, l.getSrc()});
    } else if (rows4 && f32 && em == AffineMap::get(3, 0, {d1, d2}, ctx)) {
      p.cols.push_back({i, l.getSrc()});
    } else if (rows4 && f32 && em == AffineMap::get(3, 0, {d0}, ctx)) {
      p.rowv.push_back({i, l.getSrc()});
    } else {
      p.whole.push_back({i, l.getSrc()});
    }
    cover.push_back(l);
  }
  // decode: a K the micro-kernel's schedule cannot hold (one weight tile per
  // SPAD_B bank, x and its strip in SPAD_A, x in the ACC scratch): the generic
  // lowering (K blocks). Prefill's rows: the strips of all row blocks in SPAD_A,
  // a longer K in blocks (rowsKBlock, expandLinearRows)
  const int64_t k = cast<MemRefType>(p.w.getType()).getDimSize(1);
  if (!rows4 && !gemvHolds(k) && (k > int64_t(lay.sbank) || k * d > 65535 || k + k / d > 2 * int64_t(lay.sbank) ||
                                  lay.acc0 + k / d + 2 > lay.cbank))
    return false;
  if (rows4 && ((p.rows / d) * k > 2 * int64_t(lay.sbank) || !rowsKBlock(k, lay, d))) return false;
  record(con, cover);
  linears[con.getOperation()] = p;
  return true;
}

}  // namespace mlir::iree_compiler::sa

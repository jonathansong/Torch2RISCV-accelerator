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
  // the micro-kernels (§8.8), as --iree-sa-ukernels allows
  if (cfg.ukernel("linear"))
    f.walk([&](linalg::GenericOp g) {
      if (g->getParentOp() == f.getOperation()) matchLinear(g);
    });
  if (cfg.ukernel("attention")) f.walk([&](linalg::BatchMatmulOp m) { matchAttention(m); });
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

KernelSchedule linearSchedule(const LinearPlan &p, const ::sa::Layout &lay, int64_t d) {
  KernelSchedule ks;
  auto wt = cast<MemRefType>(p.w.getType());
  const int64_t k = wt.getDimSize(1), nt = wt.getDimSize(0);
  if (p.rows) {
    // prefill's rows: at most 32 tiles per chunk (D x nc words per ACC buffer), a divisor of the tiles
    int64_t nc = std::min<int64_t>(std::max<int64_t>(int64_t(lay.sbank) / k, 1), 32);
    while (nc > 1 && nt % nc) --nc;
    ks.chunkTiles = nc;
    return ks;
  }
  // the epilogue's extra per-chunk inputs beyond s_w need scratch words (as C3: 1 or 2 per output word)
  uint32_t extra = p.chunked.size() > 1 ? uint32_t(p.chunked.size()) - 1 : 0;
  uint32_t steps = 0;
  linalg::GenericOp epi = p.epi;
  for (Operation &o : epi.getRegion().front().without_terminator())
    if (!isa<arith::TruncIOp, arith::SIToFPOp, arith::ExtSIOp>(o)) ++steps;
  uint32_t scratchRows = std::max<uint32_t>(steps > 2 ? 2 : 1, 1 + extra);
  ks.chunkTiles = lay.chunkTiles(uint32_t(k), uint32_t(nt), scratchRows);
  ks.loop = ks.chunkTiles && nt / ks.chunkTiles > 4;
  return ks;
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
    if ((*ranges)[p.laneLoop].size != d || p.m.coef[p.kLoop] != d || p.m.coef[p.nLoop] != K * d) return false;
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
    else if (auto st = dyn_cast<sahl::StoreOp>(u); st && rows4 && st.getSrc() == p.acc && !p.store) p.store = st;
    else if (!isa<memref::DeallocOp>(u)) return false;
  }
  if (!fill || !getConstantIntValue(fill.getDpsInputs()[0]) || *getConstantIntValue(fill.getDpsInputs()[0]) != 0)
    return false;
  if (p.store) {                                   // rows: the accumulator stored as it is (no epilogue)
    if (p.epi) return false;
    const int64_t kk = cast<MemRefType>(p.w.getType()).getDimSize(1);
    if (kk * d > 65535 || (p.rows / d) * kk > 2 * int64_t(lay.sbank)) return false;
    record(con, {con.getOperation(), fill.getOperation(), p.store.getOperation(), lx.getOperation(),
                 lw.getOperation()});
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
  if (!p.store || !cast<MemRefType>(out.getType()).getElementType().isF32()) return false;
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
      if (!f32) return false;
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
  // a K the micro-kernel's schedule cannot hold (one weight tile per SPAD_B bank,
  // x and its strip in SPAD_A, x in the ACC scratch): the generic lowering (K blocks)
  const int64_t k = cast<MemRefType>(p.w.getType()).getDimSize(1);
  if (k > int64_t(lay.sbank) || k * d > 65535 || k + k / d > 2 * int64_t(lay.sbank) ||
      lay.acc0 + k / d + 2 > lay.cbank)
    return false;
  if (rows4 && (p.rows / d) * k > 2 * int64_t(lay.sbank)) return false;   // the strips of all row blocks
  record(con, cover);
  linears[con.getOperation()] = p;
  return true;
}

}  // namespace mlir::iree_compiler::sa

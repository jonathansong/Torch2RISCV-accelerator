// sahl-to-sahw (docs/iree_compiler_plan.md §8.4 steps 7, 10, 13, 15; C5.1):
// one sahw.template per dispatch function in sahl form (linalg on local
// buffers, sahl.load / sahl.store).
//   - local buffers: placed in ACC (fp32 / int32) or SPAD_A (int8), bump
//     allocation (a planning pass replaces this later);
//   - DDR views: binding subspan + subview offsets -> a BASE register (binding
//     + the subspan's offset, from push constants: Lin) + a static offset;
//   - linalg.generic: every arith / math operation of the body becomes one
//     single-stage VE (sahw-fuse-ve merges them); indexing maps give the
//     operand index modes (LIN; broadcast scalar: DIV with a huge period; per
//     row: DIV by the row words; per column: MOD); a reduction over the
//     innermost loop is a VE REDUCE into one word per row; the to_i8 chain is
//     the VE's int8 output conversion.
// Unsupported forms fail (the driver then uses the C3/C4 generator).
#include <cmath>
#include <functional>
#include <cstring>
#include <map>
#include <memory>

#include "../target/DescList.h"
#include "../target/Layout.h"
#include "SaLin.h"
#include "SahlDialect.h"
#include "SahlPasses.h"
#include "SahwPasses.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir::iree_compiler::sa {
namespace {

using ::sa::VType;
constexpr float NEG0 = -0.0f;

std::optional<float> constF32(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto f = dyn_cast<FloatAttr>(c.getValue()); f && f.getType().isF32()) return float(f.getValueAsDouble());
  return std::nullopt;
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
  if (n > 4096) return out;
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

std::optional<VType> vtOf(Type t) {
  if (t.isF32()) return ::sa::VT_F32;
  if (t.isInteger(32)) return ::sa::VT_I32;
  if (t.isInteger(8)) return ::sa::VT_I8;
  return std::nullopt;
}
uint32_t esize(Type t) {
  if (t.isF32() || t.isInteger(32)) return 4;
  if (t.isInteger(8)) return 1;
  if (t.isInteger(64)) return 8;
  return 0;
}

struct LocalBuf {
  uint32_t la = 0;
  VType vt = ::sa::VT_F32;
  int64_t n = 0;
  bool bcast = false;          // one word per element, the value in every lane
  // a dynamic innermost length (upper bound max_dynamic): rows of it `rowStride`
  // words apart (rows = 1: a vector)
  std::optional<Lin> dyn;
  uint32_t rowStride = 0;
  int64_t rows = 1;
};

// a VE source operand
struct Opd {
  uint32_t la = 0;
  VType vt = ::sa::VT_F32;
  ::sa::VIdx mode = ::sa::IDX_LIN;
  uint32_t period = 0;
  bool isImm = false;
  float imm = 0.0f;
};

// a value inside a generic body
struct Val {
  // Mask: innermost index <= / < an i64 scalar (`arg`, pred); Index: linalg.index `dim`; IntExpr / IdxCond: index arithmetic / conditions
  // (evaluated where used); ScalarArg: an i64 scalar input (`arg`); SwapSrc /
  // NegSwap: the pair swap of a buffer (negated)
  enum Kind { None, Const, Mem, ToI8, I32OfI8, Index, IntExpr, IdxCond, ScalarArg, SwapSrc, NegSwap, Mask } kind = None;
  arith::CmpIPredicate pred{};
  bool negInf = false;         // masked with -inf (only for a max reduction)
  float c = 0;
  int dim = -1, arg = -1;
  Opd o;
  // what the value depends on: 0 the element, 1 nothing (a scalar), 2 the row
  int uni = 0;
  Operation *producer = nullptr;   // the VE that computed it (a fresh buffer)
  bool fresh = false;              // a buffer of its own (a gathered row): a result may use it as is
  bool exactInt = false;           // integers held exactly as fp32 (a packed gather): sitofp is a no-op
  std::shared_ptr<Val> inner;
};

class Lowerer {
public:
  Lowerer(func::FuncOp f, const TargetConfig &cfg, sahw::TemplateOp t, sahw::BodyOp body)
      : f(f), cfg(cfg), d(cfg.d), lay(uint32_t(cfg.d), uint32_t(cfg.spadBytes), uint32_t(cfg.accBytes)),
        regB(body), bb(OpBuilder::atBlockEnd(&body.getRegion().front())), loc(f.getLoc()) {
    accTop[0] = lay.acc0;
    accTop[1] = lay.acc1;
  }

  bool run() {
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
    // an element-wise dispatch too large for ACC: in pieces
    if (linears.empty() && attns.empty() && contracts.empty()) {
      if (auto n = pieceable(); n && pieceWords(*n) > 2 * (lay.cbank - lay.scr)) {
        int64_t pieces = (pieceWords(*n) + 2 * (lay.cbank - lay.scr) - 1) / (2 * (lay.cbank - lay.scr));
        int64_t len = ((*n + pieces - 1) / pieces + d - 1) / d * d;
        pieceN = *n;
        // the reductions' results live across the pieces (allocated first)
        llvm::DenseMap<Value, LocalBuf> keep;
        for (auto g : f.getBody().front().getOps<linalg::GenericOp>())
          if (g.getNumReductionLoops()) {
            auto l = bufOf(g.getDpsInits()[0], /*bcast=*/true);
            if (!l) return false;
            keep[g.getDpsInits()[0]] = *l;
          }
        for (int64_t off = 0; off < *n; off += len) {
          uint32_t savedTop[2] = {accTop[0], accTop[1]};
          uint32_t savedSpad = spadTop;
          locals = keep;
          bcastOf.clear();
          piece = std::make_pair(off, std::min(len, *n - off));
          if (!lowerBody()) return false;
          accTop[0] = savedTop[0];
          accTop[1] = savedTop[1];
          spadTop = savedSpad;
        }
        piece.reset();
        return err.empty();
      }
      if (auto rc = rowPieceable()) {
        auto [R, C] = *rc;
        int64_t words = pieceWords2D(R, C), cap = 2 * (lay.cbank - lay.scr);
        if (words > cap) {
          // rows per piece: even (the per-row vectors' DMAs stay 8-byte aligned)
          int64_t rn = std::max<int64_t>(R * cap / words, 1) / 2 * 2;
          if (rn < 2) return fail("ACC full (a row pair does not fit)");
          prR = R;
          prC = C;
          for (int64_t r0 = 0; r0 < R; r0 += rn) {
            uint32_t savedTop[2] = {accTop[0], accTop[1]};
            uint32_t savedSpad = spadTop;
            locals.clear();
            bcastOf.clear();
            rowPiece = std::make_pair(r0, std::min(rn, R - r0));
            if (!lowerBody()) return false;
            accTop[0] = savedTop[0];
            accTop[1] = savedTop[1];
            spadTop = savedSpad;
          }
          rowPiece.reset();
          return err.empty();
        }
      }
    }
    return lowerBody();
  }

  // A dispatch whose operations go row by row over [R, C] (R != C): generics
  // with maps (r, c), (r), (c) or () (reductions over c only, no indices or
  // gathers) or over [R] / [C]; loads / stores of [R, C] (contiguous or a
  // column slice), [R], [C] or scalars. Its rows are independent: (R, C).
  std::optional<std::pair<int64_t, int64_t>> rowPieceable() {
    int64_t R = -1, C = -1;
    for (Operation &op : f.getBody().front())
      for (Value v : op.getOperands())
        if (auto mt = dyn_cast<MemRefType>(v.getType()); mt && mt.hasStaticShape() && mt.getRank() == 2 && R < 0 &&
                                                            isa<linalg::GenericOp, sahl::LoadOp, sahl::StoreOp>(op)) {
          R = mt.getDimSize(0);
          C = mt.getDimSize(1);
        }
    if (R <= 1 || C <= 1 || R == C) return std::nullopt;
    auto shapeOk = [&](Value v) {
      auto mt = dyn_cast<MemRefType>(v.getType());
      if (!mt || !mt.hasStaticShape()) return false;
      if (mt.getNumElements() == 1) return true;
      if (mt.getRank() == 2) return mt.getDimSize(0) == R && mt.getDimSize(1) == C;
      return mt.getRank() == 1 && (mt.getDimSize(0) == R || mt.getDimSize(0) == C);
    };
    MLIRContext *ctx = f.getContext();
    AffineExpr r = getAffineDimExpr(0, ctx), c = getAffineDimExpr(1, ctx);
    for (Operation &op : f.getBody().front()) {
      if (auto l = dyn_cast<sahl::LoadOp>(&op)) {
        if (!shapeOk(l.getSrc()) || !shapeOk(l.getDst())) return std::nullopt;
      } else if (auto st = dyn_cast<sahl::StoreOp>(&op)) {
        if (!shapeOk(st.getSrc()) || !shapeOk(st.getDst())) return std::nullopt;
      } else if (auto fl = dyn_cast<linalg::FillOp>(&op)) {
        if (!shapeOk(fl.getDpsInits()[0])) return std::nullopt;
      } else if (auto g = dyn_cast<linalg::GenericOp>(&op)) {
        Block &b = g.getRegion().front();
        if (!b.getOps<linalg::IndexOp>().empty() || !b.getOps<memref::LoadOp>().empty()) return std::nullopt;
        auto iters = g.getIteratorTypesArray();
        auto maps = g.getIndexingMapsArray();
        if (iters.size() == 2) {
          if (iters[0] != utils::IteratorType::parallel) return std::nullopt;
          for (auto [v, m] : llvm::zip(g->getOperands(), maps)) {
            if (!shapeOk(v)) return std::nullopt;
            if (m != AffineMap::get(2, 0, {r, c}, ctx) && m != AffineMap::get(2, 0, {r}, ctx) &&
                m != AffineMap::get(2, 0, {c}, ctx) && m.getNumResults() != 0)
              return std::nullopt;
          }
        } else if (iters.size() == 1) {
          for (auto [v, m] : llvm::zip(g->getOperands(), maps))
            if (!shapeOk(v) || !(m.isIdentity() || m.getNumResults() == 0)) return std::nullopt;
        } else {
          return std::nullopt;
        }
      } else if (!isa<arith::ConstantOp, memref::AllocOp, memref::DeallocOp, memref::SubViewOp, memref::CastOp,
                      IREE::HAL::InterfaceBindingSubspanOp, IREE::HAL::InterfaceConstantLoadOp, func::ReturnOp,
                      arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtUIOp, arith::ShLIOp, arith::OrIOp>(op) &&
                 !(op.getDialect() && op.getDialect()->getNamespace() == "util")) {
        return std::nullopt;
      }
    }
    return std::make_pair(R, C);
  }
  // ACC words such a dispatch may need at once: every [R, C] buffer, a temp per
  // fp32 operation of a two-loop generic
  int64_t pieceWords2D(int64_t R, int64_t C) {
    int64_t bufs = 0;
    for (Operation &op : f.getBody().front()) {
      if (auto a = dyn_cast<memref::AllocOp>(&op)) {
        auto mt = cast<MemRefType>(a.getType());
        if (mt.getRank() == 2 && !mt.getElementType().isInteger(8)) ++bufs;
      } else if (auto g = dyn_cast<linalg::GenericOp>(&op); g && g.getNumLoops() == 2) {
        // the operations that make a new fp32 value (casts, compares, selects and
        // the to_i8 chain are stages of the VE that consumes them)
        for (Operation &o : g.getRegion().front().without_terminator())
          if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp, arith::MaximumFOp, arith::MinimumFOp,
                  math::ExpOp, math::RsqrtOp, math::AbsFOp, arith::NegFOp>(o))
            ++bufs;
      }
    }
    return bufs * ((R * C + d - 1) / d);
  }

  // An element-wise dispatch (static contiguous loads / stores, generics with
  // identity maps over N elements, scalar inputs): N, else nothing.
  std::optional<int64_t> pieceable() {
    int64_t n = -1;
    auto size = [&](Value v, bool scalarOk) {
      auto mt = dyn_cast<MemRefType>(v.getType());
      if (!mt || !mt.hasStaticShape()) return false;
      if (mt.getRank() == 0 || (scalarOk && mt.getNumElements() == 1)) return scalarOk;
      if (n < 0) n = mt.getNumElements();
      return mt.getNumElements() == n;
    };
    for (Operation &op : f.getBody().front()) {
      if (auto l = dyn_cast<sahl::LoadOp>(&op)) {
        if (!size(l.getSrc(), true) || !size(l.getDst(), true)) return std::nullopt;
      } else if (auto st = dyn_cast<sahl::StoreOp>(&op)) {
        if (!size(st.getSrc(), true) || !size(st.getDst(), true)) return std::nullopt;
      } else if (auto g = dyn_cast<linalg::GenericOp>(&op)) {
        if (!g.getRegion().front().getOps<linalg::IndexOp>().empty() ||
            !g.getRegion().front().getOps<memref::LoadOp>().empty())
          return std::nullopt;
        if (g.getNumReductionLoops()) {
          // a max of the whole vector into a scalar (the pieces' maxima combine in any order)
          auto maps = g.getIndexingMapsArray();
          auto yield = cast<linalg::YieldOp>(g.getRegion().front().getTerminator());
          if (g.getNumLoops() != 1 || g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1 ||
              !maps[0].isIdentity() || maps[1].getNumResults() != 0 || !size(g.getDpsInputs()[0], false) ||
              !yield.getOperand(0).getDefiningOp<arith::MaximumFOp>())
            return std::nullopt;
          continue;
        }
        for (auto [v, m] : llvm::zip(g->getOperands(), g.getIndexingMapsArray()))
          if (!(m.isIdentity() && size(v, false)) && !(m.getNumResults() == 0 && size(v, true))) return std::nullopt;
      } else if (auto fl = dyn_cast<linalg::FillOp>(&op)) {
        if (!size(fl.getDpsInits()[0], true)) return std::nullopt;
      } else if (!isa<arith::ConstantOp, memref::AllocOp, memref::DeallocOp, memref::SubViewOp, memref::CastOp,
                      IREE::HAL::InterfaceBindingSubspanOp, IREE::HAL::InterfaceConstantLoadOp, func::ReturnOp,
                      arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtUIOp, arith::ShLIOp, arith::OrIOp>(op) &&
                 !(op.getDialect() && op.getDialect()->getNamespace() == "util")) {
        return std::nullopt;
      }
    }
    if (n <= 0) return std::nullopt;
    return n;
  }
  // ACC words an element-wise dispatch of n elements may need at once (an upper
  // bound: every N-element buffer, one temp per operation of a generic)
  int64_t pieceWords(int64_t n) {
    int64_t bufs = 0;
    for (Operation &op : f.getBody().front()) {
      if (auto a = dyn_cast<memref::AllocOp>(&op)) {
        auto mt = cast<MemRefType>(a.getType());
        if (mt.getNumElements() == n && !mt.getElementType().isInteger(8)) ++bufs;
      } else if (auto g = dyn_cast<linalg::GenericOp>(&op)) {
        bufs += int64_t(llvm::range_size(g.getRegion().front().without_terminator()));
      }
    }
    return bufs * ((n + d - 1) / d);
  }

  bool lowerBody() {
    for (Operation &opRef : f.getBody().front()) {
      Operation *op = &opRef;
      loc = op->getLoc();
      if (auto it = linears.find(op); it != linears.end()) {
        if (!linear(it->second)) return false;
        continue;
      }
      if (auto it = attns.find(op); it != attns.end()) {
        if (!attention(it->second)) return false;
        continue;
      }
      if (auto it = contracts.find(op); it != contracts.end()) {
        if (!contraction(it->second)) return false;
        continue;
      }
      if (owned.contains(op)) continue;
      if (isa<arith::ConstantOp, memref::AllocOp, memref::DeallocOp, memref::SubViewOp, memref::CastOp,
              memref::DimOp, IREE::HAL::InterfaceBindingSubspanOp, IREE::HAL::InterfaceConstantLoadOp,
              IREE::TensorExt::DispatchWorkloadOrdinalOp, func::ReturnOp>(op))
        continue;
      if (op->getDialect() && op->getDialect()->getNamespace() == "util") continue;
      if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtUIOp, arith::ShLIOp, arith::OrIOp>(op)) continue;
      if (auto l = dyn_cast<sahl::LoadOp>(op)) {
        if (!load(l)) return false;
        continue;
      }
      if (auto s = dyn_cast<sahl::StoreOp>(op)) {
        if (!store(s)) return false;
        continue;
      }
      if (auto fl = dyn_cast<linalg::FillOp>(op)) {
        auto c = constF32(fl.getDpsInputs()[0]);
        if (!c) return fail("fill with a non-constant or non-fp32 value");
        fills[fl.getDpsInits()[0]] = *c;
        continue;
      }
      if (auto g = dyn_cast<linalg::GenericOp>(op)) {
        if (!generic(g)) return false;
        continue;
      }
      if (auto sc = dyn_cast<IREE::LinalgExt::ScatterOp>(op)) {
        if (!scatter(sc)) return false;
        continue;
      }
      return fail("unsupported operation " + op->getName().getStringRef().str());
    }
    return err.empty();
  }

  std::string err;

private:
  func::FuncOp f;
  const TargetConfig &cfg;
  int64_t d;
  ::sa::Layout lay;
  OpBuilder regB, bb;
  Location loc;
  std::map<std::pair<int, Lin>, Value> bases;
  llvm::DenseMap<Value, LocalBuf> locals;
  llvm::DenseMap<Value, LocalBuf> bcastOf;
  llvm::DenseMap<Value, float> fills;
  uint32_t accTop[2];
  uint32_t spadTop = 0;
  Operation *lastVe = nullptr;
  std::map<Lin, Value> params;
  Value rowAcc;                                  // the row loops' DDR offset
  Value curLen;                                  // row mode: the dynamic LEN of full-length VEs
  int64_t curLenStatic = -1;
  std::map<std::pair<void *, int>, Value> validParams;   // (i64 scalar, predicate) -> VALID count

  bool fail(const std::string &m) {
    if (err.empty()) err = m;
    return false;
  }

  // ------------------------------------------------------------ registers
  Value baseFor(int binding, const Lin &off) {
    Lin key = off.isConst() ? Lin{-1, 1, 0, 0} : off;
    auto it = bases.find({binding, key});
    if (it != bases.end()) return it->second;
    Value v = sahw::BaseOp::create(regB, loc, regB.getType<sahw::BaseType>(), int64_t(binding), int64_t(key.ord),
                                   key.mul, int64_t(key.shift), key.add, IntegerAttr());
    bases[{binding, key}] = v;
    return v;
  }

  Value paramFor(const Lin &v) {
    if (auto it = params.find(v); it != params.end()) return it->second;
    Value p = sahw::ParamOp::create(regB, loc, regB.getType<sahw::ParamType>(), int64_t(v.ord), v.mul,
                                    int64_t(v.shift), v.add, IntegerAttr());
    params[v] = p;
    return p;
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

  // ------------------------------------------------------------ local memory
  // piece mode (an element-wise dispatch too large for ACC): N, and the piece
  // (first element, length) being lowered
  int64_t pieceN = -1;
  std::optional<std::pair<int64_t, int64_t>> piece;
  // row-piece mode (a row-by-row dispatch too large for ACC): [R, C], the rows
  // (first, count) being lowered
  int64_t prR = -1, prC = -1;
  std::optional<std::pair<int64_t, int64_t>> rowPiece;
  int preferBank = -1;                           // ACC temps of a linear chunk: its bank
  std::optional<uint32_t> allocAcc(uint32_t words) {
    if (preferBank >= 0 && accTop[preferBank] + words <= uint32_t(preferBank + 1) * lay.cbank) {
      uint32_t w = accTop[preferBank];
      accTop[preferBank] += words;
      return w;
    }
    for (int b = 0; b < 2; ++b) {
      uint32_t end = (b + 1) * lay.cbank;
      if (accTop[b] + words <= end) {
        uint32_t w = accTop[b];
        accTop[b] += words;
        return w;
      }
    }
    fail("ACC full");
    return std::nullopt;
  }
  LocalBuf newLocal(VType vt, int64_t n, bool bcast = false) {
    LocalBuf l;
    l.vt = vt;
    l.n = n;
    l.bcast = bcast;
    uint32_t words = std::max<uint32_t>(bcast ? uint32_t(n) : uint32_t((n + d - 1) / d), 1);
    if (vt == ::sa::VT_I8) {
      if (spadTop + words > 2 * lay.sbank) fail("SPAD_A full");
      l.la = ::sa::laddr(::sa::MEM_SPAD_A, spadTop);
      spadTop += words;
    } else {
      l.la = ::sa::acc(allocAcc(words).value_or(0));
    }
    return l;
  }
  // the local of a buffer (allocated at its first use)
  std::optional<LocalBuf> bufOf(Value v, bool bcast = false) {
    if (auto it = locals.find(v); it != locals.end()) return it->second;
    if (!v.getDefiningOp<memref::AllocOp>()) return fail("operand is not a local buffer"), std::nullopt;
    auto mt = cast<MemRefType>(v.getType());
    auto vt = vtOf(mt.getElementType());
    int64_t n = mt.hasStaticShape() ? mt.getNumElements() : 0;
    if (mt.getElementType().isInteger(64)) vt = ::sa::VT_I32, n *= 2;       // (low, high) in two lanes
    if (!vt) return fail("local buffer type"), std::nullopt;
    if (!mt.hasStaticShape()) {
      // [?] or [H, ?]: rows of max_dynamic elements
      auto alloc = v.getDefiningOp<memref::AllocOp>();
      if (!mt.isDynamicDim(mt.getRank() - 1) || alloc.getDynamicSizes().size() != 1)
        return fail("dynamic local buffer other than [..., ?]"), std::nullopt;
      auto dl = dynLin(alloc.getDynamicSizes()[0]);
      if (!dl) return fail("dynamic size not from a push constant"), std::nullopt;
      int64_t rows = 1;
      for (int64_t i = 0; i + 1 < mt.getRank(); ++i) rows *= mt.getDimSize(i);
      uint32_t stride = uint32_t((cfg.maxDynamic + d - 1) / d);
      LocalBuf l = newLocal(*vt, int64_t(stride) * rows * d, bcast);
      l.dyn = dl;
      l.rowStride = stride;
      l.rows = rows;
      l.n = cfg.maxDynamic * rows;
      locals[v] = l;
      return l;
    }
    if (piece && mt.getNumElements() == pieceN) n = piece->second;
    if (rowPiece && mt.getRank() == 2 && mt.getDimSize(0) == prR) n = rowPiece->second * prC;
    if (rowPiece && mt.getRank() == 1 && mt.getDimSize(0) == prR) n = rowPiece->second;
    LocalBuf l = newLocal(*vt, n, bcast);
    locals[v] = l;
    return l;
  }

  // ------------------------------------------------------------ DDR
  struct Ddr {
    Value base;
    int64_t off = 0;           // bytes
    int64_t n = 0;
    Type et;
    std::optional<Lin> dyn;    // a dynamic innermost length (rows of it, contiguous)
    int64_t rows = 1;
    int64_t pitch = 0;         // static rows `pitch` bytes apart (a column slice), 0: contiguous
  };
  std::optional<Ddr> ddrOf(Value v, bool allowPitch = false) {
    auto mt = cast<MemRefType>(v.getType());
    if (!mt.hasStaticShape()) {
      // a binding of [?] or [H, ?] (the rows contiguous)
      auto sub = v.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
      bool lastOnly = mt.isDynamicDim(mt.getRank() - 1);
      for (int64_t i = 0; i + 1 < mt.getRank(); ++i) lastOnly &= !mt.isDynamicDim(i);
      if (!sub || !lastOnly) return fail("dynamic DDR view other than a [..., ?] binding"), std::nullopt;
      Lin off{-1, 1, 0, 0};
      if (Value o = sub.getByteOffset()) {
        auto l = linOf(o);
        if (!l) return fail("binding offset is not a constant or a push constant"), std::nullopt;
        off = *l;
      }
      auto dl = linOf(sub.getDynamicDims()[0]);
      if (!dl) return fail("dynamic dimension is not a push constant"), std::nullopt;
      Ddr r;
      r.base = baseFor(int(sub.getBinding().getZExtValue()), off);
      r.et = mt.getElementType();
      if (!esize(r.et)) return fail("DDR element type"), std::nullopt;
      r.off = off.isConst() ? off.add : 0;
      r.dyn = dl;
      r.rows = 1;
      for (int64_t i = 0; i + 1 < mt.getRank(); ++i) r.rows *= mt.getDimSize(i);
      r.n = cfg.maxDynamic * r.rows;
      return r;
    }
    // contiguous row-major view
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(mt.getStridesAndOffset(strides, offset))) return fail("DDR view strides"), std::nullopt;
    // contiguous, or rows (the leading dimension) of contiguous elements a
    // larger pitch apart (a column slice: DMA rows with a pitch)
    int64_t expect = 1, pitchElems = 0;
    for (int i = int(mt.getRank()) - 1; i >= 0; --i) {
      if (mt.getDimSize(i) != 1 && strides[i] != expect) {
        if (i != 0 || strides[0] < expect) return fail("non-contiguous DDR view"), std::nullopt;
        pitchElems = strides[0];
      }
      expect *= mt.getDimSize(i);
    }
    int64_t elem = 0;
    Value cur = v;
    while (true) {
      Operation *op = cur.getDefiningOp();
      if (!op) return fail("DDR view without a subspan"), std::nullopt;
      if (auto s = dyn_cast<memref::SubViewOp>(op)) {
        auto st = s.getSourceType();
        SmallVector<int64_t> ss;
        int64_t so;
        if (failed(st.getStridesAndOffset(ss, so))) return fail("subview source strides"), std::nullopt;
        for (auto [o, str] : llvm::zip(s.getStaticOffsets(), ss)) {
          if (ShapedType::isDynamic(o) || ShapedType::isDynamic(str)) return fail("dynamic subview"), std::nullopt;
          elem += o * str;
        }
        cur = s.getSource();
        continue;
      }
      if (auto c = dyn_cast<memref::CastOp>(op)) {
        cur = c.getSource();
        continue;
      }
      auto sub = dyn_cast<IREE::HAL::InterfaceBindingSubspanOp>(op);
      if (!sub) return fail("DDR view through " + op->getName().getStringRef().str()), std::nullopt;
      Lin off{-1, 1, 0, 0};
      if (Value o = sub.getByteOffset()) {
        auto l = linOf(o);
        if (!l) return fail("binding offset is not a constant or a push constant"), std::nullopt;
        off = *l;
      }
      Ddr r;
      r.base = baseFor(int(sub.getBinding().getZExtValue()), off);
      r.et = mt.getElementType();
      uint32_t es = esize(r.et);
      if (!es) return fail("DDR element type"), std::nullopt;
      r.off = (off.isConst() ? off.add : 0) + elem * es;
      r.n = mt.getNumElements();
      if (pitchElems) {
        if (!allowPitch) return fail("non-contiguous DDR view"), std::nullopt;
        r.rows = mt.getDimSize(0);
        r.pitch = pitchElems * es;
      }
      if (rowPiece && mt.getRank() >= 1 && mt.getDimSize(0) == prR && (mt.getRank() == 1 || mt.getDimSize(1) == prC)) {
        int64_t rowBytes = mt.getRank() == 1 ? es : (r.pitch ? r.pitch : prC * es);
        r.off += rowPiece->first * rowBytes;
        r.n = r.n / prR * rowPiece->second;
        if (r.pitch) r.rows = rowPiece->second;
      }
      if (piece && r.n == pieceN) {
        r.off += piece->first * es;
        r.n = piece->second;
      }
      return r;
    }
  }

  // rows of a dynamic length T (bytes per row = f(T)): row r at DDR offset
  // r * bytes; a PARAM accumulates the offset on the device (SETREG PARAM +=
  // PARAM), so each DMA has two dynamic fields (address, bytes)
  template <typename F>
  void rowLoop(int64_t H, Value bytesParam, F dma) {
    if (!rowAcc) rowAcc = privateParam();
    sahw::SetRegOp::create(bb, loc, ValueRange{rowAcc}, ArrayRef<int64_t>{0}, 0, ValueRange{});
    for (int64_t r = 0; r < H; ++r) {
      dma(r);
      if (r + 1 < H) {
        auto s = sahw::SetRegOp::create(bb, loc, ValueRange{rowAcc}, ArrayRef<int64_t>{0}, 1, ValueRange{bytesParam});
        s.setDynFields(ArrayRef<int32_t>{::sa::DYN_SETREG_V0});
        s.setDynAdd(ArrayRef<bool>{false});
      }
    }
  }
  template <typename OpT>
  static void dynDma(OpT op, Value acc, Value bytes) {
    SmallVector<int32_t> f;
    SmallVector<bool> a;
    if (acc) f.push_back(::sa::DYN_DMA_DDR), a.push_back(true);
    f.push_back(::sa::DYN_DMA_ROW_BYTES), a.push_back(false);
    op.setDynFields(f);
    op.setDynAdd(a);
  }
  bool dynamicDma(const Ddr &r, const LocalBuf &l, bool isLoad) {
    if (!l.dyn || l.rows != r.rows) return fail("dynamic DMA between different shapes");
    uint32_t es = esize(r.et);
    Lin b = *r.dyn;
    b.mul *= es;
    b.add *= es;
    Value bp = paramFor(b);
    int64_t maxBytes = cfg.maxDynamic * es;
    if (r.off % 8) return fail("dynamic DMA not 8-byte aligned");
    auto one = [&](int64_t row, Value acc) {
      int64_t la = int64_t(l.la) + row * l.rowStride;
      SmallVector<Value> dv;
      if (acc) dv.push_back(acc);
      dv.push_back(bp);
      if (isLoad) dynDma(sahw::LdOp::create(bb, loc, r.base, r.off, la, 1, maxBytes, maxBytes, 0, dv), acc, bp);
      else dynDma(sahw::StOp::create(bb, loc, r.base, r.off, la, 1, maxBytes, maxBytes, dv), acc, bp);
    };
    if (r.rows == 1) {
      one(0, Value());
      return true;
    }
    rowLoop(r.rows, bp, [&](int64_t row) { one(row, rowAcc); });
    return true;
  }

  // A contiguous DMA of `bytes` (a multiple of 8) between DDR and local memory:
  // one row, or beyond the 16-bit row length rows of the largest multiple of
  // the local word (so the rows are contiguous there too) and a tail.
  void contiguousDma(bool isLoad, Value base, int64_t off, uint32_t la, uint32_t bytes) {
    uint32_t wb = (la >> 28) == uint32_t(::sa::MEM_ACC) ? 4 * uint32_t(d) : uint32_t(d);
    uint32_t rb = bytes <= 65535 ? bytes : 65535 / wb * wb;
    uint32_t rows = bytes / rb, rest = bytes - rows * rb;
    auto one = [&](int64_t o, uint32_t a, uint32_t n, uint32_t b) {
      if (isLoad) sahw::LdOp::create(bb, loc, base, o, int64_t(a), n, int64_t(b), int64_t(b), 0, ValueRange{});
      else sahw::StOp::create(bb, loc, base, o, int64_t(a), n, int64_t(b), int64_t(b), ValueRange{});
    };
    one(off, la, rows, rb);
    if (rest) one(off + int64_t(rows) * rb, la + rows * (rb / wb), 1, rest);
  }

  // rows of a column slice: one DMA of `rows` rows, `pitch` bytes apart in DDR,
  // contiguous (whole words) in local memory
  bool rowsDma(bool isLoad, const Ddr &r, uint32_t la) {
    int64_t rb = r.n / r.rows * esize(r.et);
    uint32_t wb = (la >> 28) == uint32_t(::sa::MEM_ACC) ? 4 * uint32_t(d) : uint32_t(d);
    if (r.off % 8 || rb % 8 || r.pitch % 8 || rb % wb || rb > 65535 || r.rows > 65535)
      return fail("column slice: rows not whole words / 8-byte aligned");
    if (isLoad) sahw::LdOp::create(bb, loc, r.base, r.off, int64_t(la), r.rows, rb, r.pitch, 0, ValueRange{});
    else sahw::StOp::create(bb, loc, r.base, r.off, int64_t(la), r.rows, rb, r.pitch, ValueRange{});
    return true;
  }

  // an fp32 scalar at a 4-byte (not 8-byte) aligned address: through a PARAM
  // (LDPARAM reads 32-bit words), a word of ones scaled by it (1.0 * v = v)
  bool scalarByParam(const Ddr &r, const LocalBuf &dst) {
    Value pv = privateParam();
    sahw::LdParamOp::create(bb, loc, r.base, pv, r.off, 1, 0, ValueRange{});
    LocalBuf ones = newLocal(::sa::VT_F32, 1, true);
    ve({ones.la, ::sa::VT_I32}, std::nullopt, ones.la, ::sa::VT_F32, d, ::sa::VOP_COPY, 0.0f, 1.0f);
    ve({ones.la}, std::nullopt, dst.la, ::sa::VT_F32, d, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE,
       ::sa::RED_NONE, 0, 0, {{::sa::DYN_VE_A, pv, false}});
    return true;
  }

  bool load(sahl::LoadOp l) {
    auto r = ddrOf(l.getSrc(), /*allowPitch=*/true);
    if (!r) return false;
    if (r->et.isInteger(64)) return fail("i64 load");
    auto dst = bufOf(l.getDst());
    if (!dst) return false;
    if (r->dyn) return dynamicDma(*r, *dst, true);
    if (r->pitch) return rowsDma(true, *r, dst->la);
    if (r->off % 8 && r->n == 1 && r->et.isF32()) return scalarByParam(*r, *dst);
    uint32_t bytes = uint32_t((r->n * esize(r->et) + 7) / 8 * 8);
    if (r->off % 8) return fail("load not 8-byte aligned");
    contiguousDma(true, r->base, r->off, dst->la, bytes);
    return true;
  }

  bool store(sahl::StoreOp s) {
    auto r = ddrOf(s.getDst(), /*allowPitch=*/true);
    if (!r) return false;
    auto it = locals.find(s.getSrc());
    if (it == locals.end()) return fail("stored buffer not computed");
    LocalBuf l = it->second;
    if (r->dyn) return dynamicDma(*r, l, false);
    if (l.bcast && l.n > 1) l = packBcast(l);
    if (r->pitch) return rowsDma(false, *r, l.la);
    uint32_t bytes = uint32_t((r->n * esize(r->et) + 7) / 8 * 8);
    if (r->off % 8) return fail("store not 8-byte aligned");
    contiguousDma(false, r->base, r->off, l.la, bytes);
    return true;
  }

  // A row gathered from a packed weight (plan §8.12 C6.0): element k of row r
  // of Wp i8[R, K, D] at Wp[r / D, k, r % D] (the embedding of a model whose
  // classifier shares the table). The tile r / D is K contiguous words; lane
  // r % D of each word is picked by a one-hot word and a REDUCE per word, the
  // K scalars packed into K / D words. r / D and r % D come from the VE
  // (exact in fp32: round(r / D - (D - 1) / 2D)), through the result's DDR
  // (overwritten by the result later) into PARAMs.
  template <typename FlatIndex>
  std::optional<Val> packedRowGather(linalg::GenericOp g, memref::LoadOp ld, MemRefType mt,
                                     ArrayRef<SmallVector<int64_t>> dom, int scalarArgNo, FlatIndex flatIndex) {
    const int64_t K = int64_t(dom.size());
    bool packed = mt.getRank() == 3 && mt.getDimSize(1) == K && mt.getDimSize(2) == d && K % d == 0 &&
                  mt.getElementType().isInteger(8) && g.getNumLoops() == 1;
    for (int64_t sv : {0, 1, 7, 8, 13, 21, 1000})
      for (int64_t pi = 0; pi < K && packed; ++pi) {
        auto f = flatIndex(dom[pi], sv);
        packed = f && *f == (sv / d) * K * d + pi * d + sv % d;
      }
    if (!packed) return fail("gather index is not row * C + iteration index"), std::nullopt;
    if (K * d > 65535) return fail("packed gather: a tile beyond one DMA row / LDPARAM factor"), std::nullopt;
    auto sym = ddrOf(g.getDpsInputs()[scalarArgNo]);
    auto src = ddrOf(ld.getMemRef());
    std::optional<Ddr> yd;
    for (Operation *u : g.getDpsInits()[0].getUsers())
      if (auto st = dyn_cast<sahl::StoreOp>(u); st && st.getSrc() == g.getDpsInits()[0]) yd = ddrOf(st.getDst());
    if (!sym || !src || !yd) return fail("packed gather: operands / result not in DDR"), std::nullopt;
    if (yd->n * esize(yd->et) < 16 || yd->off % 8 || src->off % 8 || sym->off % 8)
      return fail("packed gather: result too small or unaligned for the scratch words"), std::nullopt;
    // the token as (lo, hi) in lanes 0, 1 of a zeroed word (junk * 0 is 0: read as i32, finite)
    LocalBuf w0 = newLocal(::sa::VT_I32, d);
    ve({w0.la, ::sa::VT_I32}, std::nullopt, w0.la, ::sa::VT_I32, d, ::sa::VOP_COPY, 0.0f, 0.0f);
    sahw::LdOp::create(bb, loc, sym->base, sym->off, int64_t(w0.la), 1, 8, 8, 0, ValueRange{});
    // t = r / D, l = r - t * D in lane 0
    LocalBuf wt = newLocal(::sa::VT_I32, d), wtd = newLocal(::sa::VT_F32, d), wl = newLocal(::sa::VT_I32, d);
    ve({w0.la, ::sa::VT_I32}, std::nullopt, wt.la, ::sa::VT_I32, d, ::sa::VOP_COPY, 1.0f / float(d),
       -float(d - 1) / float(2 * d));
    ve({wt.la, ::sa::VT_I32}, std::nullopt, wtd.la, ::sa::VT_F32, d, ::sa::VOP_COPY, float(d));
    ve({w0.la, ::sa::VT_I32}, Opd{wtd.la, ::sa::VT_F32}, wl.la, ::sa::VT_I32, d, ::sa::VOP_SUB);
    // through DDR (the result's first 16 bytes) into PARAMs
    sahw::StOp::create(bb, loc, yd->base, yd->off, int64_t(wt.la), 1, 8, 8, ValueRange{});
    sahw::StOp::create(bb, loc, yd->base, yd->off + 8, int64_t(wl.la), 1, 8, 8, ValueRange{});
    sahw::FenceOp::create(bb, loc, int64_t(0));
    Value pT = privateParam(), pV1 = privateParam(), pV0 = privateParam();
    sahw::LdParamOp::create(bb, loc, yd->base, pT, yd->off, K * d, 0, ValueRange{});
    sahw::LdParamOp::create(bb, loc, yd->base, pV1, yd->off + 8, 1, d + 1, ValueRange{});
    sahw::LdParamOp::create(bb, loc, yd->base, pV0, yd->off + 8, 1, d, ValueRange{});
    // one-hot(l) = the second word of prefix(D + l + 1) - prefix(D + l) over two words of ones
    LocalBuf ones = newLocal(::sa::VT_F32, 2 * d), p1 = newLocal(::sa::VT_F32, 2 * d),
             p0 = newLocal(::sa::VT_F32, 2 * d), oh = newLocal(::sa::VT_F32, 2 * d);
    ve({ones.la, ::sa::VT_I32}, std::nullopt, ones.la, ::sa::VT_F32, 2 * d, ::sa::VOP_COPY, 0.0f, 1.0f);
    ve({ones.la}, std::nullopt, p1.la, ::sa::VT_F32, 2 * d, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE,
       ::sa::RED_NONE, 0, 1, {{::sa::DYN_VE_VALID, pV1, false}});
    ve({ones.la}, std::nullopt, p0.la, ::sa::VT_F32, 2 * d, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE,
       ::sa::RED_NONE, 0, 1, {{::sa::DYN_VE_VALID, pV0, false}});
    ve({p1.la}, Opd{p0.la}, oh.la, ::sa::VT_F32, 2 * d, ::sa::VOP_SUB);
    // the tile, lane l of each word (one nonzero lane: the sum is exact), packed
    LocalBuf tile = newLocal(::sa::VT_I8, K * d);
    auto lt = sahw::LdOp::create(bb, loc, src->base, src->off, int64_t(tile.la), 1, K * d, K * d, 0,
                                 ValueRange{pT});
    lt.setDynFields(ArrayRef<int32_t>{::sa::DYN_DMA_DDR});
    lt.setDynAdd(ArrayRef<bool>{true});
    LocalBuf red = newLocal(::sa::VT_F32, K, /*bcast=*/true);
    ve({tile.la, ::sa::VT_I8}, Opd{oh.la + 1, ::sa::VT_F32, ::sa::IDX_MOD, 1}, red.la, ::sa::VT_F32, K * d,
       ::sa::VOP_MUL, 1.0f, NEG0, ::sa::FUNC_NONE, ::sa::RED_SUM, 1);
    LocalBuf row = packBcast(red);
    Val e;
    e.kind = Val::Mem;
    e.o = {row.la, ::sa::VT_F32, ::sa::IDX_LIN, 0};
    e.exactInt = true;
    return e;
  }

  // ------------------------------------------------------------ VE
  uint32_t types(const Opd &s1, const std::optional<Opd> &s2, VType out) {
    uint32_t t = ::sa::vtypes(s1.vt, out);
    if (s2 && !s2->isImm && s2->vt != s1.vt) t |= uint32_t(s2->vt == ::sa::VT_I8 ? 1 : s2->vt) << 4;
    return t;
  }
  struct DynF {
    int32_t field;
    Value param;
    bool add;
  };
  void ve(const Opd &s1, const std::optional<Opd> &s2, uint32_t dst, VType out, int64_t n, ::sa::VOp op,
          float A = 1.0f, float B = NEG0, ::sa::VFunc fn = ::sa::FUNC_NONE, ::sa::VRed red = ::sa::RED_NONE,
          uint32_t rowlen = 0, uint32_t valid = 0, ArrayRef<DynF> dyn = {}, bool swapneg = false) {
    int64_t m2 = ::sa::IDX_LIN, period = 0;
    float imm = 0.0f;
    uint32_t src2 = 0;
    if (s2) {
      if (s2->isImm) {
        m2 = ::sa::IDX_IMM;
        imm = s2->imm;
      } else {
        m2 = s2->mode;
        src2 = s2->la;
        period = s2->period;
      }
    }
    uint32_t len = uint32_t((n + d - 1) / d * d);
    SmallVector<Value> dv;
    SmallVector<int32_t> df;
    SmallVector<bool> da;
    if (curLen && int64_t(len) == curLenStatic) {
      dv.push_back(curLen);
      df.push_back(::sa::DYN_VE_LEN);
      da.push_back(false);
    }
    for (const DynF &x : dyn) {
      dv.push_back(x.param);
      df.push_back(x.field);
      da.push_back(x.add);
    }
    auto v = sahw::VeOp::create(bb, loc, int64_t(s1.la), int64_t(src2), int64_t(dst), int64_t(len), int64_t(op),
                       int64_t(types(s1, s2, out)), period, true, int64_t(fn), int64_t(s1.mode), m2, int64_t(red),
                       swapneg, llvm::APFloat(imm), llvm::APFloat(A), llvm::APFloat(B), int64_t(rowlen),
                       int64_t(valid), int64_t(s1.period), 1, 0, 0,
                       int64_t(INT32_MIN), int64_t(INT32_MAX), dv);
    if (!df.empty()) {
      v.setDynFields(df);
      v.setDynAdd(da);
    }
    lastVe = v;
  }

  // a buffer read inside a body: its local copy (a DDR one is loaded whole)
  std::optional<LocalBuf> materialize(Value t) {
    if (auto it = locals.find(t); it != locals.end()) return it->second;
    if (!ddrRoot(t)) return bufOf(t);
    auto r = ddrOf(t);
    if (!r) return std::nullopt;
    auto vt = vtOf(r->et);
    if (!vt) return fail("element type of a gathered buffer"), std::nullopt;
    LocalBuf l = newLocal(*vt, r->n);
    uint32_t bytes = uint32_t((r->n * esize(r->et) + 7) / 8 * 8);
    if (r->off % 8 || bytes > 65535) return fail("gathered buffer alignment / size"), std::nullopt;
    sahw::LdOp::create(bb, loc, r->base, r->off, int64_t(l.la), 1, int64_t(bytes), int64_t(bytes), 0, ValueRange{});
    locals[t] = l;
    return l;
  }

  // one word with the scalar (element 0 of l) in every lane
  std::optional<LocalBuf> scalarBcast(Value v, const LocalBuf &l) {
    if (l.bcast) return l;
    if (auto it = bcastOf.find(v); it != bcastOf.end()) return it->second;
    if (l.vt != ::sa::VT_F32) return fail("broadcast of a non-fp32 scalar"), std::nullopt;
    LocalBuf b = newLocal(::sa::VT_F32, 1, true);
    ve({l.la}, std::nullopt, b.la, ::sa::VT_F32, d, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE, ::sa::RED_MAX, 0, 1);
    bcastOf[v] = b;
    return b;
  }
  // one word per element (n values, packed) : replicate each word D times, D x D transposes
  std::optional<LocalBuf> perElementBcast(Value v, const LocalBuf &l, int64_t n) {
    if (l.bcast) return l;
    if (auto it = bcastOf.find(v); it != bcastOf.end()) return it->second;
    if (l.vt != ::sa::VT_F32) return fail("per-row values must be fp32"), std::nullopt;
    uint32_t w = uint32_t((n + d - 1) / d);
    LocalBuf rep = newLocal(::sa::VT_F32, int64_t(w) * d * d);
    Opd s{l.la, ::sa::VT_F32, ::sa::IDX_DIV, uint32_t(d)};
    ve(s, std::nullopt, rep.la, ::sa::VT_F32, int64_t(w) * d * d, ::sa::VOP_COPY);
    LocalBuf b = newLocal(::sa::VT_F32, int64_t(w) * d, true);
    sahw::TransposeOp::create(bb, loc, int64_t(rep.la), int64_t(b.la), int64_t(w * d * d),
                              int64_t(::sa::vtypes(::sa::VT_F32, ::sa::VT_F32)), 1, ValueRange{});
    bcastOf[v] = b;
    return b;
  }
  // broadcast words (one per element) -> packed: a D x D TRANSPOSE of each
  // block of D words gives D equal words; block k goes to word k (in order,
  // each overwriting the previous block's extra words), so the buffer has D - 1
  // words of room at the end
  LocalBuf packBcast(const LocalBuf &b) {
    uint32_t w = uint32_t((b.n + d - 1) / d);
    LocalBuf p = newLocal(::sa::VT_F32, int64_t(w + d - 1) * d);
    p.n = b.n;
    for (uint32_t k = 0; k < w; ++k)
      sahw::TransposeOp::create(bb, loc, int64_t(b.la + k * d), int64_t(p.la + k), int64_t(d * d),
                                int64_t(::sa::vtypes(::sa::VT_F32, ::sa::VT_F32)), 1, ValueRange{});
    return p;
  }

  // ------------------------------------------------------------ scatter (C5.3)
  // one row written at a dynamic index (the KV cache update): LDPARAM of the
  // index * row bytes, a ST with that DDR offset
  bool scatter(IREE::LinalgExt::ScatterOp sc) {
    Value upd = sc.getUpdates(), idx = sc.getIndices(), orig = sc.getOriginal();
    auto uT = cast<MemRefType>(upd.getType()), oT = cast<MemRefType>(orig.getType());
    if (sc.getDimensionMap() != ArrayRef<int64_t>{0} || uT.getDimSize(0) != 1 || !uT.hasStaticShape() ||
        !oT.hasStaticShape() || uT.getElementType() != oT.getElementType())
      return fail("scatter other than one row at a dynamic index");
    Block &b = sc.getRegion().front();
    auto y = dyn_cast<IREE::LinalgExt::YieldOp>(b.getTerminator());
    if (!y || y.getOperand(0) != b.getArgument(0)) return fail("scatter that does not overwrite");
    uint32_t es = esize(uT.getElementType());
    uint32_t rowBytes = uint32_t(uT.getNumElements() * es);
    if (rowBytes % 8 || rowBytes > 0xFFFF) return fail("scatter row size");
    auto u = materialize(upd);
    auto ir = ddrOf(idx);
    auto orr = ddrOf(orig);
    if (!u || !ir || !orr) return false;
    if (orr->off % 8) return fail("scatter target not 8-byte aligned");
    Value p = privateParam();
    sahw::LdParamOp::create(bb, loc, ir->base, p, ir->off, rowBytes, 0, ValueRange{});
    auto st = sahw::StOp::create(bb, loc, orr->base, orr->off, int64_t(u->la), 1, int64_t(rowBytes), int64_t(rowBytes),
                                 ValueRange{p});
    st.setDynFields(ArrayRef<int32_t>{::sa::DYN_DMA_DDR});
    st.setDynAdd(ArrayRef<bool>{true});
    return err.empty();
  }

  // ------------------------------------------------------------ attention (C5.3)
  // scores: bmm(ext(K^T), q) -> [H, T, 1], epilogue (f32(acc) * s_q[h]) * a_k;
  // P V:    bmm(p, ext(V))   -> [H, 1, hs], epilogue f32(acc) * a_v;
  // K / V: i8 [T, H, hs], a slice of the KV cache in DDR, through an extsi
  // generic with map (t, h, j) -> (h, t, j). Per head, as C3 (compile_model.
  // attention): K rows -> TRANSPOSE -> K^T tiles, or V rows loaded interleaved;
  // q_h or p_h as a replicated int8 A strip; EX; one VE for the epilogue. T
  // is dynamic (PARAMs from push constants).
  struct AttnPlan {
    linalg::BatchMatmulOp bmm;
    bool scores = false;
    Value cache;                                 // the cache view (DDR)
    Lin T;                                       // its (dynamic) length
    Value small;                                 // q [H, hs, 1] or p [H, 1, T] (DDR)
    Value sq;                                    // scores: s_q [H] (DDR)
    float scale = 0;
    sahl::StoreOp store;
    int64_t H = 0, hs = 0;
  };
  llvm::DenseMap<Operation *, AttnPlan> attns;

  // the byte offset of a DDR view with static offsets from its binding subspan
  std::optional<std::pair<Value, int64_t>> ddrStart(Value v, int64_t es) {
    int64_t elem = 0;
    Value cur = v;
    while (auto s = cur.getDefiningOp<memref::SubViewOp>()) {
      SmallVector<int64_t> ss;
      int64_t so;
      if (failed(s.getSourceType().getStridesAndOffset(ss, so))) return std::nullopt;
      for (auto [o, str] : llvm::zip(s.getStaticOffsets(), ss)) {
        if (ShapedType::isDynamic(o) || ShapedType::isDynamic(str)) return std::nullopt;
        elem += o * str;
      }
      cur = s.getSource();
    }
    auto sub = cur.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!sub) return std::nullopt;
    Lin off{-1, 1, 0, 0};
    if (Value o = sub.getByteOffset()) {
      auto l = linOf(o);
      if (!l) return std::nullopt;
      off = *l;
    }
    return std::make_pair(baseFor(int(sub.getBinding().getZExtValue()), off), (off.isConst() ? off.add : 0) + elem * es);
  }

  bool matchAttention(linalg::BatchMatmulOp bmm) {
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
    for (Operation *o : cover) owned.insert(o);
    attns[bmm.getOperation()] = p;
    return true;
  }

  bool attention(AttnPlan &p) {
    const int64_t H = p.H, hs = p.hs, T = cfg.maxDynamic;
    if (T % d || hs % d) return fail("attention: T and head size must be multiples of D");
    if (spadTop != 0) return fail("attention after other SPAD_A buffers");
    int logd = int(std::log2(double(d)));
    auto par = [&](int64_t mul, int shift) {
      Lin l = p.T;
      l.mul *= mul;
      l.add *= mul;
      l.shift += shift;
      return paramFor(l);
    };
    auto cache = ddrStart(p.cache, 1);
    auto yd = ddrStart(p.store.getDst(), 4);
    if (!cache || !yd) return fail("attention: operands not in DDR");
    const uint32_t hw = uint32_t(hs / d), tiles = uint32_t(T / d), sb = lay.sbank;
    if (uint32_t(T) * hw > sb) return fail("attention: K rows do not fit a SPAD bank");
    // SPAD_A: the A strip at 0, the raw K rows in bank 1; SPAD_B: K^T at 0, V in bank 1 (as C3)
    const uint32_t strip = 0, kraw = sb, kt = 0, vb = sb;
    spadTop = 2 * sb;
    if (yd->second % 8 || cache->second % 8) return fail("unaligned attention operand");
    auto dyn = [](auto op, SmallVector<std::pair<int32_t, Value>> f) {
      SmallVector<Value> vs;
      SmallVector<int32_t> fs;
      SmallVector<bool> as;
      for (auto &[field, v] : f) vs.push_back(v), fs.push_back(field), as.push_back(false);
      op.getDynMutable().assign(vs);
      op.setDynFields(fs);
      op.setDynAdd(as);
    };
    auto word = [](uint32_t la) { return int64_t(la & 0xFFFFFFF); };
    if (p.scores) {
      auto q = ddrOf(p.small);
      auto sqr = ddrOf(p.sq);
      if (!q || !sqr) return false;
      LocalBuf sqRaw = newLocal(::sa::VT_F32, H);
      sahw::LdOp::create(bb, loc, sqr->base, sqr->off, int64_t(sqRaw.la), 1, int64_t((H * 4 + 7) / 8 * 8),
                         int64_t((H * 4 + 7) / 8 * 8), 0, ValueRange{});
      auto sq = perElementBcast(p.sq, sqRaw, H);
      if (!sq) return false;
      Value pT = par(1, 0), pThs = par(hs, 0), pTiles = par(1, logd);
      LocalBuf srcw = newLocal(::sa::VT_I32, hs);
      LocalBuf c = newLocal(::sa::VT_I32, int64_t(d) * tiles * d);
      LocalBuf res = newLocal(::sa::VT_F32, int64_t(tiles) * H * d);
      for (int64_t h = 0; h < H; ++h) {
        auto l = sahw::LdOp::create(bb, loc, cache->first, cache->second + h * hs,
                                    int64_t(::sa::laddr(::sa::MEM_SPAD_A, kraw)), T, hs, H * hs, 0, ValueRange{});
        dyn(l, {{::sa::DYN_DMA_ROWS, pT}});
        auto tr = sahw::TransposeOp::create(bb, loc, int64_t(::sa::laddr(::sa::MEM_SPAD_A, kraw)),
                                            int64_t(::sa::laddr(::sa::MEM_SPAD_B, kt)), T * hs,
                                            int64_t(::sa::vtypes(::sa::VT_I8, ::sa::VT_I8)), int64_t(hw), ValueRange{});
        dyn(tr, {{::sa::DYN_VE_LEN, pThs}});
        sahw::LdOp::create(bb, loc, q->base, q->off + h * hs * 4, int64_t(srcw.la), 1, hs * 4, hs * 4, 0, ValueRange{});
        ve({srcw.la, ::sa::VT_I32, ::sa::IDX_DIV, uint32_t(d)}, std::nullopt, ::sa::laddr(::sa::MEM_SPAD_A, strip),
           ::sa::VT_I8, hs * d, ::sa::VOP_COPY);
        auto ex = sahw::ExOp::create(bb, loc, int64_t(strip), int64_t(kt), word(c.la), int64_t(hw), false,
                                     int64_t(tiles), hs, 1, int64_t(tiles), ValueRange{});
        dyn(ex, {{::sa::DYN_EX_REPEAT, pTiles}});
        ve({c.la, ::sa::VT_I32}, Opd{sq->la + uint32_t(h), ::sa::VT_F32, ::sa::IDX_DIV, 0xFFFF},
           res.la + uint32_t(h) * tiles, ::sa::VT_F32, T, ::sa::VOP_MUL, p.scale, NEG0, ::sa::FUNC_NONE,
           ::sa::RED_NONE, 0, 0, {{::sa::DYN_VE_LEN, pT, false}});
      }
      // the scores of each head: a row of T
      Value bp = par(4, 0);
      rowLoop(H, bp, [&](int64_t h) {
        auto st = sahw::StOp::create(bb, loc, yd->first, yd->second, int64_t(res.la) + h * tiles, 1, T * 4, T * 4,
                                     ValueRange{});
        dyn(st, {{::sa::DYN_DMA_DDR, rowAcc}, {::sa::DYN_DMA_ROW_BYTES, bp}});
        SmallVector<bool> a = {true, false};
        st.setDynAdd(a);
      });
    } else {
      // p: [H, 1, T'] i32 in DDR (T' the same length, from its own push constant), one row per head
      auto pm = cast<MemRefType>(p.small.getType());
      auto psub = ddrRoot(p.small).getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
      auto ps = ddrStart(p.small, 4);
      if (!ps || !psub || psub.getDynamicDims().size() != 1 || pm.getRank() != 3 || pm.getDimSize(1) != 1)
        return fail("attention p layout");
      auto plen = linOf(psub.getDynamicDims()[0]);
      if (!plen) return fail("attention p length");
      Lin pb = *plen;
      pb.mul *= 4;
      pb.add *= 4;
      Value pBytes = paramFor(pb);
      uint32_t stride = uint32_t(T / d);
      LocalBuf pl = newLocal(::sa::VT_I32, int64_t(stride) * H * d);
      if (ps->second % 8) return fail("attention p not 8-byte aligned");
      rowLoop(H, pBytes, [&](int64_t h) {
        auto l = sahw::LdOp::create(bb, loc, ps->first, ps->second, int64_t(pl.la) + h * stride, 1, T * 4, T * 4, 0,
                                    ValueRange{});
        dyn(l, {{::sa::DYN_DMA_DDR, rowAcc}, {::sa::DYN_DMA_ROW_BYTES, pBytes}});
        SmallVector<bool> a = {true, false};
        l.setDynAdd(a);
      });
      Value pT = par(1, 0), pTd = par(d, 0), pTiles = par(1, logd);
      LocalBuf c = newLocal(::sa::VT_I32, int64_t(d) * hw * d);
      LocalBuf res = newLocal(::sa::VT_F32, hs);
      for (int64_t h = 0; h < H; ++h) {
        ve({pl.la + uint32_t(h) * stride, ::sa::VT_I32, ::sa::IDX_DIV, uint32_t(d)}, std::nullopt,
           ::sa::laddr(::sa::MEM_SPAD_A, strip), ::sa::VT_I8, T * d, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE,
           ::sa::RED_NONE, 0, 0, {{::sa::DYN_VE_LEN, pTd, false}});
        auto l = sahw::LdOp::create(bb, loc, cache->first, cache->second + h * hs,
                                    int64_t(::sa::laddr(::sa::MEM_SPAD_B, vb)), T, hs, H * hs, /*INTERLEAVE=*/1,
                                    ValueRange{});
        dyn(l, {{::sa::DYN_DMA_ROWS, pT}});
        auto ex = sahw::ExOp::create(bb, loc, int64_t(strip), int64_t(vb), word(c.la), int64_t(tiles), false,
                                     int64_t(hw), T, 1, int64_t(hw), ValueRange{});
        dyn(ex, {{::sa::DYN_EX_KT, pTiles}, {::sa::DYN_EX_BSTEP, pT}});
        ve({c.la, ::sa::VT_I32}, std::nullopt, res.la, ::sa::VT_F32, hs, ::sa::VOP_COPY, p.scale, NEG0);
        sahw::StOp::create(bb, loc, yd->first, yd->second + h * hs * 4, int64_t(res.la), 1, hs * 4, hs * 4,
                           ValueRange{});
      }
    }
    return err.empty();
  }

  // ------------------------------------------------------------ contractions (C5.4, generic)
  // y[b, n] = sum_k x[b, k] * M[b, n, k] over int8 values (int32 accumulator),
  // the result consumed by one element-wise epilogue that is stored. Each
  // operand is followed back to DDR (through its sahl.load and an extsi
  // generic, possibly transposing), giving its element offset as a linear
  // function of the contraction's loops; the loops then split into batch (in
  // x and M), output n (in M only) and reduction k. The matrix becomes B tiles
  // by its layout:
  //   packed [N/D, K, D] (the packed weights): loaded as is;
  //   rows of K (n stride sN, k contiguous): rows loaded, TRANSPOSE;
  //   rows of N (k stride sK, n contiguous): loaded INTERLEAVE;
  // x becomes the A strip (each element over the D rows). Per batch, per chunk
  // of output tiles: the B tiles, EX, the epilogue on the chunk (element-wise
  // lowering), the store. A plain serial schedule (the micro-kernels are the
  // tuned ones); dynamic N or K (one chunk of all tiles) from PARAMs.
  struct Operand {
    Value view;                                   // the DDR view
    Type et;                                      // its element type
    SmallVector<int64_t> coef;                    // element offset per contraction loop
    SmallVector<Operation *> cover;               // its load / extension
  };
  enum class MatLayout { Packed, RowsK, RowsN };
  struct ContractPlan {
    linalg::LinalgOp op;
    linalg::GenericOp epi;
    sahl::StoreOp store;
    Operand x, m;
    int bLoop = -1, kLoop = -1, nLoop = -1, laneLoop = -1;   // laneLoop: the packed layout's lane
    int gLoop = -1;                                          // in x and the output only: rows of x (GQA's query heads)
    MatLayout layout = MatLayout::Packed;
    SmallVector<std::pair<int, Value>> epiLoads;             // epilogue inputs loaded whole: (input, DDR)
  };
  llvm::DenseMap<Operation *, ContractPlan> contracts;

  // the loop ranges of a linalg op: static, or a Lin (dynamic)
  struct Range {
    int64_t size = 1;
    std::optional<Lin> dyn;
  };
  std::optional<SmallVector<Range>> loopRanges(linalg::LinalgOp op) {
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
  std::optional<Operand> operandOf(linalg::LinalgOp op, int input) {
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

  bool matchContraction(linalg::LinalgOp op) {
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
      else if (!isa<memref::DeallocOp>(u)) return false;
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
    for (Operation *o : cover) owned.insert(o);
    contracts[op.getOperation()] = p;
    return true;
  }

  bool contraction(ContractPlan &p) {
    auto ranges = *loopRanges(p.op);
    const Range B = p.bLoop >= 0 ? ranges[p.bLoop] : Range{};
    const Range Gr = p.gLoop >= 0 ? ranges[p.gLoop] : Range{};
    Range Kr = ranges[p.kLoop];
    Range Nr = ranges[p.nLoop];
    if (p.laneLoop >= 0) Nr.size *= d;                    // packed: tiles x lanes
    if (B.dyn || Gr.dyn) return fail("contraction: dynamic batch or rows");
    if (Kr.dyn && Nr.dyn) return fail("contraction: dynamic K and N");
    const int64_t K = Kr.size, N = Nr.size, H = B.size, G = Gr.size;
    if (K % d || N % d) return fail("contraction: K and N must be multiples of D");
    const uint32_t sb = lay.sbank, nt = uint32_t(N / d);
    int logd = int(std::log2(double(d)));
    auto parOf = [&](const Lin &l, int64_t mul, int shift) {
      Lin x = l;
      x.mul *= mul;
      x.add *= mul;
      x.shift += shift;
      return paramFor(x);
    };
    bool xRows = (p.bLoop >= 0 && p.x.coef[p.bLoop] < 0) || (p.gLoop >= 0 && p.x.coef[p.gLoop] < 0);
    if (spadTop != 0) return fail("contraction after other SPAD_A buffers");
    spadTop = 2 * sb;                                      // the A strip at 0, raw rows in bank 1
    const uint32_t strip = 0, raw = sb;
    auto xs = ddrStart(p.x.view, esize(p.x.et)), ms = ddrStart(p.m.view, 1), ys = ddrStart(p.store.getDst(), 4);
    if (!xs || !ms || !ys) return fail("contraction: operands not in DDR");
    auto dyn = [](auto op, SmallVector<std::tuple<int32_t, Value, bool>> f) {
      SmallVector<Value> vs;
      SmallVector<int32_t> fs;
      SmallVector<bool> as;
      for (auto &[field, v, add] : f) vs.push_back(v), fs.push_back(field), as.push_back(add);
      op.getDynMutable().assign(vs);
      op.setDynFields(fs);
      op.setDynAdd(as);
    };
    // the output (and the epilogue's loops) in the order (batch, row, n)
    AffineMap outMap = p.op.getIndexingMapsArray()[2];
    auto outPos = [&](int L) -> int {
      if (L < 0) return -1;
      for (unsigned i = 0; i < outMap.getNumResults(); ++i)
        if (auto de = dyn_cast<AffineDimExpr>(outMap.getResult(i)); de && int(de.getPosition()) == L) return int(i);
      return -1;
    };
    int eb = outPos(p.bLoop), eg = outPos(p.gLoop), en = outPos(p.nLoop);
    if ((eb >= 0 && eg >= 0 && eb > eg) || (eg >= 0 && eg > en) || (eb >= 0 && eb > en))
      return fail("contraction: output not in the order (batch, row, n)");
    // epilogue inputs: per output element (loaded per chunk), per (batch, row) value, or a scalar
    enum class EpiKind { PerElement, PerRow, Scalar };
    struct EpiIn {
      EpiKind kind;
      Value src;
      LocalBuf l;
      int64_t sb = 0, sg = 0;                              // PerRow: element stride of the batch / row index
    };
    llvm::DenseMap<int, EpiIn> epiIn;
    auto emaps = p.epi.getIndexingMapsArray();
    for (auto &[i, src] : p.epiLoads) {
      Value in = p.epi.getDpsInputs()[i];
      auto mt = cast<MemRefType>(in.getType());
      AffineMap em = emaps[i];
      EpiIn e;
      e.src = src;
      if (em.isIdentity()) {
        if (Nr.dyn || !mt.getElementType().isF32()) return fail("contraction epilogue input per element");
        e.kind = EpiKind::PerElement;
        epiIn[i] = e;
        continue;
      }
      SmallVector<int64_t> st;
      int64_t off;
      if (!mt.hasStaticShape() || failed(mt.getStridesAndOffset(st, off))) return fail("contraction epilogue input layout");
      bool onlyRow = true;
      for (unsigned r = 0; r < em.getNumResults(); ++r) {
        if (auto de = dyn_cast<AffineDimExpr>(em.getResult(r))) {
          int pos = int(de.getPosition());
          if (pos == eb) e.sb = st[r];
          else if (pos == eg) e.sg = st[r];
          else onlyRow = false;
        } else if (auto c = dyn_cast<AffineConstantExpr>(em.getResult(r)); !c || c.getValue() != 0) {
          onlyRow = false;
        }
      }
      if (!onlyRow) return fail("contraction epilogue input layout");
      e.kind = (e.sb || e.sg) ? EpiKind::PerRow : EpiKind::Scalar;
      auto l = materialize(src);
      if (!l) return false;
      locals[in] = *l;
      e.l = *l;
      // broadcasts now: the chunk buffers below are released after each chunk,
      // so nothing cached may be allocated there
      if (e.kind == EpiKind::Scalar && !scalarBcast(in, *l)) return false;
      if (e.kind == EpiKind::PerRow && !perElementBcast(in, *l, mt.getNumElements())) return false;
      epiIn[i] = e;
    }
    uint32_t xes = esize(p.x.et);
    bool xI32 = p.x.et.isInteger(32);
    LocalBuf xl = newLocal(xI32 ? ::sa::VT_I32 : ::sa::VT_I8, K);
    // K blocks: the A strip of a block (kc words) in one SPAD_A bank, one B tile of it
    // in one SPAD_B bank and in one DMA row (16-bit length: kc * D bytes), and the
    // strip's DIV-mode source range (the VE checks kc words from the source start,
    // no divider) inside x's memory; a longer K accumulates the blocks in ACC
    uint32_t xDepth = (xl.la >> 28) == uint32_t(::sa::MEM_ACC) ? 2 * lay.cbank : 2 * sb;
    int64_t xRoom = int64_t(xDepth) - int64_t(xl.la & 0xFFFF) - K / d;
    uint32_t kmax = std::min<uint32_t>(sb, 65535 / uint32_t(d) / uint32_t(d) * uint32_t(d));
    kmax = uint32_t(std::min<int64_t>(kmax, xRoom / d * d));
    if (kmax < uint32_t(d)) return fail("contraction: x leaves no room for a K block");
    const uint32_t kc = uint32_t(K) > kmax ? kmax : uint32_t(K);
    const uint32_t nk = uint32_t((K + kc - 1) / kc);
    if (nk > 1 && (Kr.dyn || Nr.dyn || G != 1 || xRows))
      return fail("contraction: K blocks with a dynamic K / N or several rows of x");
    // chunks of output tiles: the B tiles of a chunk (of a K block) in one SPAD_B bank
    // (a dynamic N: all tiles, one chunk)
    uint32_t nc = nt;
    if (!Nr.dyn) {
      uint32_t cap = std::max<uint32_t>(sb / kc, 1);
      for (nc = std::min(cap, nt); nc > 1 && nt % nc; --nc) {
      }
    }
    if (kc * nc > sb) return fail("contraction: one chunk of B tiles does not fit a SPAD bank");
    if (xRows && nc != nt) return fail("contraction: x rows of a dynamic length and several chunks");
    Value pK = Kr.dyn ? parOf(*Kr.dyn, 1, 0) : Value();
    Value pN = Nr.dyn ? parOf(*Nr.dyn, 1, 0) : Value();
    if (xRows || Nr.dyn) {
      if (!rowAcc) rowAcc = privateParam();
      sahw::SetRegOp::create(bb, loc, ValueRange{rowAcc}, ArrayRef<int64_t>{0}, 0, ValueRange{});
    }
    auto addRow = [&](Value bytes) {
      auto s2 = sahw::SetRegOp::create(bb, loc, ValueRange{rowAcc}, ArrayRef<int64_t>{0}, 1, ValueRange{bytes});
      s2.setDynFields(ArrayRef<int32_t>{::sa::DYN_SETREG_V0});
      s2.setDynAdd(ArrayRef<bool>{false});
    };
    // x_(b, g) -> the A strip (each element over the D rows); with one row
    // per batch (G = 1) once per batch, else per chunk and row
    auto loadX = [&](int64_t b, int64_t g) -> bool {
      int64_t xoff = xs->second;
      if (!xRows) xoff += ((p.bLoop >= 0 ? b * p.x.coef[p.bLoop] : 0) + (p.gLoop >= 0 ? g * p.x.coef[p.gLoop] : 0)) *
                          int64_t(xes);
      if (xoff % 8) return fail("contraction: unaligned x");
      if (!xRows && !Kr.dyn) {                             // static: rows of at most 64 KB
        contiguousDma(true, xs->first, xoff, xl.la, uint32_t(K * xes));
        return true;
      }
      auto lx = sahw::LdOp::create(bb, loc, xs->first, xoff, int64_t(xl.la), 1, K * xes, K * xes, 0, ValueRange{});
      if (xRows) {
        Value xb = parOf(*Kr.dyn, xes, 0);
        dyn(lx, {{::sa::DYN_DMA_DDR, rowAcc, true}, {::sa::DYN_DMA_ROW_BYTES, xb, false}});
        addRow(xb);
      } else if (Kr.dyn) {
        dyn(lx, {{::sa::DYN_DMA_ROW_BYTES, parOf(*Kr.dyn, xes, 0), false}});
      }
      return true;
    };
    // x[k0, k0 + kl) -> the A strip (each element over the D rows)
    auto replicate = [&](int64_t k0, int64_t kl) {
      ve({xl.la + uint32_t(k0 / d), xl.vt, ::sa::IDX_DIV, uint32_t(d)}, std::nullopt,
         ::sa::laddr(::sa::MEM_SPAD_A, strip), ::sa::VT_I8, kl * d, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE,
         ::sa::RED_NONE, 0, 0,
         Kr.dyn ? SmallVector<DynF>{{::sa::DYN_VE_LEN, parOf(*Kr.dyn, d, 0), false}} : SmallVector<DynF>{});
    };
    auto loadStrip = [&](int64_t b, int64_t g) -> bool {
      if (!loadX(b, g)) return false;
      replicate(0, K);
      return true;
    };
    for (int64_t b = 0; b < H; ++b) {
      int64_t moff = ms->second + (p.bLoop >= 0 ? b * p.m.coef[p.bLoop] : 0);
      if (moff % 8) return fail("contraction: unaligned matrix");
      if (G == 1 && !(nk == 1 ? loadStrip(b, 0) : loadX(b, 0))) return false;
      uint32_t step = 0;                                   // B loads so far: the SPAD_B bank alternates
      // the B tiles of chunk c0, K range [k0, k0 + kl) -> a SPAD_B bank; returns its word address
      auto loadB = [&](uint32_t c0, int64_t k0, int64_t kl) -> uint32_t {
        uint32_t bw = (step++ & 1) * sb;
        uint32_t la = ::sa::laddr(::sa::MEM_SPAD_B, bw);
        if (p.layout == MatLayout::Packed) {
          sahw::LdOp::create(bb, loc, ms->first, moff + int64_t(c0) * K * d + k0 * d, int64_t(la), nc, kl * d,
                             K * d, 0, ValueRange{});
        } else if (p.layout == MatLayout::RowsK) {
          int64_t sN = p.m.coef[p.nLoop];
          auto l = sahw::LdOp::create(bb, loc, ms->first, moff + int64_t(c0) * d * sN + k0,
                                      int64_t(::sa::laddr(::sa::MEM_SPAD_A, raw)), int64_t(nc) * d, kl, sN, 0,
                                      ValueRange{});
          auto t = sahw::TransposeOp::create(bb, loc, int64_t(::sa::laddr(::sa::MEM_SPAD_A, raw)), int64_t(la),
                                             int64_t(nc) * d * kl, int64_t(::sa::vtypes(::sa::VT_I8, ::sa::VT_I8)),
                                             kl / d, ValueRange{});
          if (Nr.dyn) {
            dyn(l, {{::sa::DYN_DMA_ROWS, pN, false}});
            dyn(t, {{::sa::DYN_VE_LEN, parOf(*Nr.dyn, K, 0), false}});
          }
        } else {
          int64_t sK = p.m.coef[p.kLoop];
          auto l = sahw::LdOp::create(bb, loc, ms->first, moff + int64_t(c0) * d + k0 * sK, int64_t(la), kl,
                                      int64_t(nc) * d, sK, /*INTERLEAVE=*/1, ValueRange{});
          if (Kr.dyn) dyn(l, {{::sa::DYN_DMA_ROWS, pK, false}});
        }
        return bw;
      };
      auto exOp = [&](uint32_t bw, uint32_t cw, int64_t kl, bool accumulate) {
        auto ex = sahw::ExOp::create(bb, loc, int64_t(strip), int64_t(bw), int64_t(cw), kl / d, accumulate,
                                     int64_t(nc), kl, 1, int64_t(nc), ValueRange{});
        if (Kr.dyn) dyn(ex, {{::sa::DYN_EX_KT, parOf(*Kr.dyn, 1, logd), false}, {::sa::DYN_EX_BSTEP, pK, false}});
        if (Nr.dyn) dyn(ex, {{::sa::DYN_EX_REPEAT, parOf(*Nr.dyn, 1, logd), false}});
      };
      for (uint32_t c0 = 0; c0 < nt; c0 += nc) {
        // one K block: the B tiles of this chunk, shared by the rows of x
        uint32_t bw = nk == 1 ? loadB(c0, 0, K) : 0;
        for (int64_t g = 0; g < G; ++g) {
          if (G > 1 && !loadStrip(b, g)) return false;
          // the chunk's accumulator and epilogue buffers: released after its store
          uint32_t savedTop[2] = {accTop[0], accTop[1]};
          LocalBuf acc = newLocal(::sa::VT_I32, int64_t(nc) * d);
          uint32_t cw = acc.la & 0xFFFFFFF;
          if (nk == 1) {
            exOp(bw, cw, K, false);
          } else {
            for (uint32_t kb = 0; kb < nk; ++kb) {
              int64_t k0 = int64_t(kb) * kc, kl = std::min<int64_t>(kc, K - k0);
              uint32_t bwk = loadB(c0, k0, kl);
              replicate(k0, kl);
              exOp(bwk, cw, kl, kb > 0);
            }
          }
          // the epilogue on this chunk: n elements at (b, g, c0)
          int64_t row = b * G + g;
          Chunk ch;
          ch.n = int64_t(nc) * d;
          LocalBuf out = newLocal(::sa::VT_F32, ch.n);
          ch.outLa = out.la;
          for (int i = 0; i < p.epi.getNumDpsInputs(); ++i) {
            Val v;
            v.kind = Val::Mem;
            if (p.epi.getDpsInputs()[i] == p.op.getDpsInits()[0]) {
              v.o = {::sa::acc(cw), ::sa::VT_I32, ::sa::IDX_LIN, 0};
              ch.inputs[i] = v;
              continue;
            }
            const EpiIn &e = epiIn.at(i);
            if (e.kind == EpiKind::PerElement) {
              auto r = ddrStart(e.src, 4);
              if (!r) return fail("contraction epilogue input not in DDR");
              LocalBuf l = newLocal(::sa::VT_F32, ch.n);
              int64_t off = r->second + (row * N + int64_t(c0) * d) * 4;
              if (off % 8) return fail("contraction epilogue input not 8-byte aligned");
              sahw::LdOp::create(bb, loc, r->first, off, int64_t(l.la), 1, ch.n * 4, ch.n * 4, 0, ValueRange{});
              v.o = {l.la, ::sa::VT_F32, ::sa::IDX_LIN, 0};
            } else if (e.kind == EpiKind::Scalar) {
              auto bc = scalarBcast(p.epi.getDpsInputs()[i], e.l);
              if (!bc) return false;
              v.o = {bc->la, ::sa::VT_F32, ::sa::IDX_DIV, 0xFFFF};
              v.uni = 1;
            } else {
              auto bc = perElementBcast(p.epi.getDpsInputs()[i], e.l, e.l.n);
              if (!bc) return false;
              v.o = {bc->la + uint32_t(b * e.sb + g * e.sg), ::sa::VT_F32, ::sa::IDX_DIV, 0xFFFF};
              v.uni = 1;
            }
            ch.inputs[i] = v;
          }
          if (Nr.dyn) {
            curLen = pN;
            curLenStatic = (N + d - 1) / d * d;
          }
          bool ok = generic(p.epi, nullptr, &ch);
          curLen = Value();
          curLenStatic = -1;
          if (!ok) return false;
          // the store: row (b, g), chunk c0 (a dynamic N: the row's N elements, rows contiguous)
          if (Nr.dyn) {
            Value bytes = parOf(*Nr.dyn, 4, 0);
            auto st = sahw::StOp::create(bb, loc, ys->first, ys->second, int64_t(out.la), 1, N * 4, N * 4,
                                         ValueRange{});
            dyn(st, {{::sa::DYN_DMA_DDR, rowAcc, true}, {::sa::DYN_DMA_ROW_BYTES, bytes, false}});
            if (row + 1 < H * G) addRow(bytes);
          } else {
            int64_t yoff = ys->second + (row * N + int64_t(c0) * d) * 4;
            if (yoff % 8) return fail("contraction: unaligned result");
            sahw::StOp::create(bb, loc, ys->first, yoff, int64_t(out.la), 1, ch.n * 4, ch.n * 4, ValueRange{});
          }
          accTop[0] = savedTop[0];
          accTop[1] = savedTop[1];
        }
      }
    }
    return err.empty();
  }

  // ------------------------------------------------------------ linear layers (C5.2)
  // y = epilogue(sum_k x[k] * W[n, k], ...): a contraction generic with the
  // packed weights (i8 [N / D, K, D], iree-sa-pack-linear-weights) and x (i8 or
  // int8 values in i32), its int32 accumulator consumed by one element-wise
  // epilogue generic [N / D, D] whose result is stored. Lowered as C3's
  // qlinear schedule (compile_layer.linear): chunks of output tiles, the
  // weights of chunk i + 1 loaded into the other SPAD_B bank while chunk i
  // computes, each chunk's epilogue through the element-wise lowering, many
  // chunks as a LOOP_END loop over pairs, chunk 0's weights in the prefix.
  struct LinearPlan {
    linalg::GenericOp con, epi;
    Value x, w;                                  // DDR sources
    Value acc;                                   // the accumulator buffer
    sahl::StoreOp store;
    SmallVector<std::pair<int, Value>> chunked;  // epilogue inputs [N / D, D]: (input, DDR source)
    SmallVector<std::pair<int, Value>> whole;    // other epilogue inputs: (input, DDR source)
    // prefill's form (plan §8.13): M rows of int8 x (0: decode's vecmat); the
    // epilogue's per-column [N / D, D] and per-row [M] inputs (chunked: [M, N])
    int64_t rows = 0;
    SmallVector<std::pair<int, Value>> cols, rowv;
  };
  llvm::DenseMap<Operation *, LinearPlan> linears;
  llvm::DenseSet<Operation *> owned;

  static sahl::LoadOp loadInto(Value local) {
    for (Operation *u : local.getUsers())
      if (auto l = dyn_cast<sahl::LoadOp>(u); l && l.getDst() == local) return l;
    return {};
  }

  // the pattern (nothing else may use its buffers); records the operations it covers
  bool matchLinear(linalg::GenericOp con) {
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
      for (Operation *o : {con.getOperation(), fill.getOperation(), p.store.getOperation(), lx.getOperation(),
                           lw.getOperation()})
        owned.insert(o);
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
    for (Operation *o : cover) owned.insert(o);
    linears[con.getOperation()] = p;
    return true;
  }

  // prefill's linear layer (plan §8.13): each block of D rows of int8 x is one
  // A strip (an interleaved DMA: block c of K, word r = row r's D elements c,
  // exactly EX's A layout), so the D x D array computes D rows at once and a
  // chunk of weight tiles serves every row block. Per chunk of output tiles:
  // its B tiles and per-column inputs, then per row block the EX (output rows
  // crow = nc words apart), the epilogue over D x nc words (per-column inputs
  // by MOD nc, per-row broadcast words by DIV nc), the store of D rows.
  // A serial schedule (no prefetch yet).
  bool linearRows(LinearPlan &p) {
    auto wt = cast<MemRefType>(p.w.getType());
    const int64_t k = wt.getDimSize(1), nt = wt.getDimSize(0), N = nt * d, M = p.rows, nb = M / d;
    const uint32_t sb = lay.sbank;
    auto xd = ddrOf(p.x), wd = ddrOf(p.w), yd = ddrOf(p.store.getDst());
    if (!xd || !wd || !yd) return false;
    if (xd->off % 8 || wd->off % 8 || yd->off % 8 || k % 8) return fail("linear rows: unaligned operand");
    SmallVector<uint32_t> strips;
    for (int64_t rb = 0; rb < nb; ++rb) {
      LocalBuf st = newLocal(::sa::VT_I8, k * d);
      sahw::LdOp::create(bb, loc, xd->base, xd->off + rb * d * k, int64_t(st.la), d, k, k, /*INTERLEAVE=*/1,
                         ValueRange{});
      strips.push_back(st.la);
    }
    std::map<int, LocalBuf> rowB;                       // per-row inputs: a broadcast word per row
    for (auto &[i, src] : p.rowv) {
      Value in = p.epi.getDpsInputs()[i];
      auto l = materialize(src);
      if (!l) return false;
      locals[in] = *l;
      auto b = perElementBcast(in, *l, M);
      if (!b) return false;
      rowB[i] = *b;
    }
    std::map<int, LocalBuf> scal;                       // other inputs: scalars
    for (auto &[i, src] : p.whole) {
      Value in = p.epi.getDpsInputs()[i];
      if (cast<MemRefType>(in.getType()).getRank() != 0) return fail("linear rows: epilogue input layout");
      auto l = materialize(src);
      if (!l) return false;
      locals[in] = *l;
      auto b = scalarBcast(in, *l);
      if (!b) return false;
      scal[i] = *b;
    }
    SmallVector<Ddr> cd, ed;
    for (auto &[i, src] : p.cols) {
      auto r = ddrOf(src);
      if (!r || r->off % 8) return fail("linear rows: per-column input");
      cd.push_back(*r);
    }
    for (auto &[i, src] : p.chunked) {
      auto r = ddrOf(src);
      if (!r || r->off % 8) return fail("linear rows: per-element input");
      ed.push_back(*r);
    }
    // chunks: the B tiles in one SPAD_B bank, D x nc words per ACC buffer
    int64_t nc = std::min<int64_t>(std::max<int64_t>(sb / k, 1), 32);
    while (nc > 1 && nt % nc) --nc;
    if (nc * k > int64_t(sb)) return fail("linear rows: one weight tile does not fit a SPAD_B bank");
    for (int64_t c0 = 0, ci = 0; c0 < nt; c0 += nc, ++ci) {
      uint32_t savedTop[2] = {accTop[0], accTop[1]};
      uint32_t bw = uint32_t(ci & 1) * sb;
      sahw::LdOp::create(bb, loc, wd->base, wd->off + c0 * k * d, int64_t(::sa::laddr(::sa::MEM_SPAD_B, bw)), nc,
                         k * d, k * d, 0, ValueRange{});
      SmallVector<LocalBuf> cl;
      for (const Ddr &r : cd) {
        LocalBuf l = newLocal(::sa::VT_F32, nc * d);
        sahw::LdOp::create(bb, loc, r.base, r.off + c0 * d * 4, int64_t(l.la), 1, nc * d * 4, nc * d * 4, 0,
                           ValueRange{});
        cl.push_back(l);
      }
      for (int64_t rb = 0; rb < nb; ++rb) {
        uint32_t savedRb[2] = {accTop[0], accTop[1]};
        const int64_t n = d * nc * d;
        LocalBuf acc = newLocal(::sa::VT_I32, n);
        sahw::ExOp::create(bb, loc, int64_t(strips[rb] & 0xFFFFFFF), int64_t(bw), int64_t(acc.la & 0xFFFFFFF),
                           k / d, false, nc, k, 1, nc, ValueRange{});
        if (!p.epi) {                                 // the accumulator itself
          sahw::StOp::create(bb, loc, yd->base, yd->off + (rb * d * N + c0 * d) * 4, int64_t(acc.la), d, nc * d * 4,
                             N * 4, ValueRange{});
          accTop[0] = savedRb[0];
          accTop[1] = savedRb[1];
          continue;
        }
        Chunk ch;
        ch.n = n;
        LocalBuf out = newLocal(::sa::VT_F32, n);
        ch.outLa = out.la;
        const int64_t off = (rb * d * N + c0 * d) * 4;
        for (int in = 0; in < p.epi.getNumDpsInputs(); ++in) {
          Val v;
          v.kind = Val::Mem;
          if (p.epi.getDpsInputs()[in] == p.acc) {
            v.o = {acc.la, ::sa::VT_I32, ::sa::IDX_LIN, 0};
            ch.inputs[in] = v;
          }
        }
        for (size_t m = 0; m < ed.size(); ++m) {
          LocalBuf l = newLocal(::sa::VT_F32, n);
          sahw::LdOp::create(bb, loc, ed[m].base, ed[m].off + off, int64_t(l.la), d, nc * d * 4, N * 4, 0,
                             ValueRange{});
          Val v;
          v.kind = Val::Mem;
          v.o = {l.la, ::sa::VT_F32, ::sa::IDX_LIN, 0};
          ch.inputs[p.chunked[m].first] = v;
        }
        for (size_t m = 0; m < cl.size(); ++m) {
          Val v;
          v.kind = Val::Mem;
          v.o = {cl[m].la, ::sa::VT_F32, ::sa::IDX_MOD, uint32_t(nc)};
          ch.inputs[p.cols[m].first] = v;
        }
        for (auto &[i, b] : rowB) {
          Val v;
          v.kind = Val::Mem;
          v.o = {b.la + uint32_t(rb * d), ::sa::VT_F32, ::sa::IDX_DIV, uint32_t(nc)};
          ch.inputs[i] = v;
        }
        for (auto &[i, b] : scal) {
          Val v;
          v.kind = Val::Mem;
          v.o = {b.la, ::sa::VT_F32, ::sa::IDX_DIV, 0xFFFF};
          v.uni = 1;
          ch.inputs[i] = v;
        }
        if (!generic(p.epi, nullptr, &ch)) return false;
        sahw::StOp::create(bb, loc, yd->base, yd->off + off, int64_t(out.la), d, nc * d * 4, N * 4, ValueRange{});
        accTop[0] = savedRb[0];
        accTop[1] = savedRb[1];
      }
      accTop[0] = savedTop[0];
      accTop[1] = savedTop[1];
    }
    return err.empty();
  }

  bool linear(LinearPlan &p) {
    if (p.rows) return linearRows(p);
    auto wt = cast<MemRefType>(p.w.getType());
    const uint32_t k = uint32_t(wt.getDimSize(1)), nt = uint32_t(wt.getDimSize(0));
    const uint32_t sb = lay.sbank, cb = lay.cbank;
    if (k % d) return fail("linear: K not a multiple of D");
    // the epilogue's extra per-chunk inputs beyond s_w need scratch words (as C3: 1 or 2 per output word)
    uint32_t extra = p.chunked.size() > 1 ? uint32_t(p.chunked.size()) - 1 : 0;
    uint32_t steps = 0;
    for (Operation &o : p.epi.getRegion().front().without_terminator())
      if (!isa<arith::TruncIOp, arith::SIToFPOp, arith::ExtSIOp>(o)) ++steps;
    uint32_t scratchRows = std::max<uint32_t>(steps > 2 ? 2 : 1, 1 + extra);
    const uint32_t nc = lay.chunkTiles(k, nt, scratchRows);
    if (nc == 0) return fail("linear: one weight tile does not fit a SPAD_B bank");
    if (k + k / d > 2 * sb || lay.acc0 + k / d + 2 > cb) return fail("linear: K too large for the local memories");
    const uint32_t nch = nt / nc, wbytes = nc * k * d, fbytes = 4 * nc * d, oSw = d * nc, oY = d * nc + nc;
    const bool useLoop = nch > 4;
    bool hoist = bb.getInsertionBlock()->empty();
    auto xd = ddrOf(p.x), wd = ddrOf(p.w), yd = ddrOf(p.store.getDst());
    if (!xd || !wd || !yd) return false;
    SmallVector<Ddr> cd;
    for (auto &[i, src] : p.chunked) {
      auto r = ddrOf(src);
      if (!r) return false;
      cd.push_back(*r);
    }
    if (yd->off % 8 || wd->off % 8) return fail("linear: unaligned operand");
    // x -> the A strip (each element over the D rows)
    LocalBuf strip = newLocal(::sa::VT_I8, int64_t(k) * d);
    bool xI32 = xd->et.isInteger(32);
    LocalBuf xl = newLocal(xI32 ? ::sa::VT_I32 : ::sa::VT_I8, k);
    uint32_t xb = xI32 ? 4 * k : k;
    sahw::LdOp::create(bb, loc, xd->base, xd->off, int64_t(xl.la), 1, int64_t(xb), int64_t(xb), 0, ValueRange{});
    ve({xl.la, xl.vt, ::sa::IDX_DIV, uint32_t(d)}, std::nullopt, strip.la, ::sa::VT_I8, int64_t(k) * d,
       ::sa::VOP_COPY);
    // other epilogue inputs, whole (the scalar s_x: its broadcast word now, as C3)
    for (auto &[i, src] : p.whole) {
      Value in = p.epi.getDpsInputs()[i];
      auto l = materialize(src);
      if (!l) return false;
      locals[in] = *l;
      if (cast<MemRefType>(in.getType()).getRank() == 0 && !scalarBcast(in, *l)) return false;
    }
    Value pw = privateParam(), pf = privateParam();
    auto dynAdd = [](auto op, bool dyn) {
      if (!dyn) return;
      op.setDynFields(ArrayRef<int32_t>{::sa::DYN_DMA_DDR});
      op.setDynAdd(ArrayRef<bool>{true});
    };
    auto slot = [&](size_t m) { return m == 0 ? oSw : oY + nc * uint32_t(m); };
    auto load = [&](uint32_t i, bool dyn) {
      uint32_t bank = i & 1;
      OpBuilder *wb = &bb;
      std::optional<OpBuilder> pb;
      if (i == 0 && hoist) {
        auto pre = sahw::PrefixOp::create(regB, loc);
        pb.emplace(OpBuilder::atBlockEnd(&pre.getRegion().emplaceBlock()));
        wb = &*pb;
      }
      dynAdd(sahw::LdOp::create(*wb, loc, wd->base, wd->off + int64_t(i) * wbytes,
                                int64_t(::sa::laddr(::sa::MEM_SPAD_B, bank * sb)), nc, int64_t(k) * d,
                                int64_t(k) * d, 0, dyn ? ValueRange{pw} : ValueRange{}),
             dyn);
      for (size_t m = 0; m < cd.size(); ++m)
        dynAdd(sahw::LdOp::create(bb, loc, cd[m].base, cd[m].off + int64_t(i) * fbytes,
                                  int64_t(::sa::acc(bank * cb + slot(m))), 1, fbytes, fbytes, 0,
                                  dyn ? ValueRange{pf} : ValueRange{}),
               dyn);
    };
    auto chunkAt = [&](uint32_t i, bool dyn, bool prefetch) -> bool {
      uint32_t bank = i & 1, c = bank * cb;
      sahw::ExOp::create(bb, loc, int64_t(strip.la & 0xFFFFFFF), int64_t(bank * sb), int64_t(c), int64_t(k / d),
                         false, int64_t(nc), int64_t(k), 1, int64_t(nc), ValueRange{});
      if (prefetch) load(i + 1, dyn);
      Chunk ch;
      ch.n = int64_t(nc) * d;
      ch.outLa = ::sa::acc(c + oY);
      for (int in = 0; in < p.epi.getNumDpsInputs(); ++in)
        if (p.epi.getDpsInputs()[in] == p.acc) {
          Val a;
          a.kind = Val::Mem;
          a.o = {::sa::acc(c), ::sa::VT_I32, ::sa::IDX_LIN, 0};
          ch.inputs[in] = a;
        }
      for (size_t m = 0; m < p.chunked.size(); ++m) {
        Val a;
        a.kind = Val::Mem;
        a.o = {::sa::acc(c + slot(m)), ::sa::VT_F32, ::sa::IDX_LIN, 0};
        ch.inputs[p.chunked[m].first] = a;
      }
      preferBank = int(bank);
      bool ok = generic(p.epi, nullptr, &ch);
      preferBank = -1;
      if (!ok) return false;
      dynAdd(sahw::StOp::create(bb, loc, yd->base, yd->off + int64_t(i) * fbytes, int64_t(ch.outLa), 1, fbytes,
                                fbytes, dyn ? ValueRange{pf} : ValueRange{}),
             dyn);
      return true;
    };
    load(0, false);
    uint32_t j = 0;
    if (useLoop) {
      uint32_t pairs = (nch - 1) / 2;            // the prefetch of the last pair stays in range
      if (pairs) {
        sahw::SetRegOp::create(bb, loc, ValueRange{pw, pf}, ArrayRef<int64_t>{0, 0}, 0, ValueRange{});
        Block *blk = bb.getInsertionBlock();
        Operation *before = blk->empty() ? nullptr : &blk->back();
        if (!chunkAt(0, true, true) || !chunkAt(1, true, true)) return false;
        auto loop = sahw::LoopOp::create(bb, loc, int64_t(pairs), pw, int64_t(2 * wbytes), pf, int64_t(2 * fbytes));
        Block *lb = &loop.getRegion().emplaceBlock();
        SmallVector<Operation *> moved;
        for (Operation *o = before ? before->getNextNode() : &blk->front(); o && o != loop.getOperation();
             o = o->getNextNode())
          moved.push_back(o);
        for (Operation *o : moved) o->moveBefore(lb, lb->end());
        j = 2 * pairs;
      }
    }
    for (uint32_t jj = j; jj < nch; ++jj)
      if (!chunkAt(jj, false, jj + 1 < nch)) return false;
    return err.empty();
  }

  // ------------------------------------------------------------ linalg.generic
  // an i64 scalar: x + c (positions), as C3: the value v in every lane, then
  // (swapneg(v) - v) * -0.5 = [v, 0, v, 0, ...], the i64 (v, 0) in lanes 0, 1
  bool intScalar(linalg::GenericOp g) {
    Block &b = g.getRegion().front();
    SmallVector<Operation *> ops;
    for (Operation &o : b.without_terminator()) ops.push_back(&o);
    if (g.getNumDpsInputs() != 1 || ops.size() != 1 || !isa<arith::AddIOp>(ops[0]))
      return fail("i64 scalar computation other than x + constant");
    Value other = ops[0]->getOperand(0) == b.getArgument(0) ? ops[0]->getOperand(1) : ops[0]->getOperand(0);
    auto c = getConstantIntValue(other);
    if (!c || std::abs(*c) >= (1 << 23)) return fail("i64 scalar: constant");
    auto src = scalarI64(g.getDpsInputs()[0]);
    if (!src) return false;
    auto out = bufOf(g.getDpsInits()[0]);
    if (!out) return false;
    LocalBuf v = newLocal(::sa::VT_F32, 1, true);
    ve({src->la, ::sa::VT_I32}, std::nullopt, v.la, ::sa::VT_F32, d, ::sa::VOP_COPY, 1.0f, float(*c), ::sa::FUNC_NONE,
       ::sa::RED_MAX, 0, 1);
    ve({v.la}, Opd{v.la}, out->la, ::sa::VT_I32, d, ::sa::VOP_SUB, -0.5f, NEG0, ::sa::FUNC_NONE, ::sa::RED_NONE, 0, 0,
       {}, /*swapneg=*/true);
    return err.empty();
  }

  // an i64 scalar in DDR, loaded as two int32 lanes
  std::optional<LocalBuf> scalarI64(Value v) {
    if (auto it = locals.find(v); it != locals.end()) return it->second;
    auto r = ddrOf(v);
    if (!r) return std::nullopt;
    if (r->off % 8) return fail("i64 scalar not 8-byte aligned"), std::nullopt;
    LocalBuf l = newLocal(::sa::VT_I32, 2);
    sahw::LdOp::create(bb, loc, r->base, r->off, int64_t(l.la), 1, 8, 8, 0, ValueRange{});
    locals[v] = l;
    return l;
  }

  // a PARAM of the template's own (from 7 down, by sahw-assign-registers)
  Value privateParam() { return sahw::PrivateOp::create(regB, loc, regB.getType<sahw::ParamType>(), IntegerAttr()); }

  // a row of an [H, T] generic with a dynamic T (or the one row of a [T] one)
  struct RowSel {
    int64_t row = 0;
    Lin len;
  };

  // a chunk of a linear layer's epilogue: the inputs given, n elements, the
  // result into outLa
  struct Chunk {
    std::map<int, Val> inputs;
    int64_t n = 0;
    uint32_t outLa = 0;
  };

  bool generic(linalg::GenericOp g, const RowSel *sel = nullptr, const Chunk *chunk = nullptr) {
    auto iters = g.getIteratorTypesArray();
    int nloops = int(iters.size());
    SmallVector<AffineMap> maps = g.getIndexingMapsArray();
    int nin = g.getNumDpsInputs();
    if (nloops == 0 && g.getNumDpsInits() == 1 &&
        cast<MemRefType>(g.getDpsInits()[0].getType()).getElementType().isInteger(64))
      return intScalar(g);
    bool collapse = false, merge = false;
    int indexShift = 0, split = 0;
    SmallVector<AffineMap> maps2;
    if (nloops > 2 && !chunk) {
      bool noIndex = g.getRegion().front().getOps<linalg::IndexOp>().empty();
      bool red = llvm::any_of(iters, [](utils::IteratorType t) { return t == utils::IteratorType::reduction; });
      if (llvm::all_of(maps, [](AffineMap m) { return m.isIdentity(); }) && !red && noIndex) {
        collapse = true;                     // an element-wise nest with identity maps only: one flat loop
      } else {
        // loops [0, split) as one "row" loop and [split, n) as one inner loop:
        // every map is the identity, the leading loops, the trailing loops or
        // nothing (split = n - 1 first; an earlier split only for an
        // element-wise nest without indices, its trailing loops static)
        MLIRContext *ctx = g.getContext();
        AffineExpr r0 = getAffineDimExpr(0, ctx), r1 = getAffineDimExpr(1, ctx);
        bool ok = false;
        for (split = nloops - 1; split >= 1 && !ok; --split) {
          ok = g.getRegion().front().getOps<memref::LoadOp>().empty();
          // indices: only of the last loop (it stays the last)
          for (linalg::IndexOp ix : g.getRegion().front().getOps<linalg::IndexOp>())
            ok &= int(ix.getDim()) == nloops - 1 && split == nloops - 1;
          for (int L = 0; L < nloops; ++L)
            ok &= iters[L] == utils::IteratorType::parallel || (L == nloops - 1 && split == nloops - 1);
          maps2.clear();
          for (AffineMap m : maps) {
            SmallVector<int64_t> dims;
            for (AffineExpr e : m.getResults()) {
              auto de = dyn_cast<AffineDimExpr>(e);
              if (!de) ok = false;
              else dims.push_back(de.getPosition());
            }
            auto seq = [&](int64_t from, int64_t to) {
              if (int64_t(dims.size()) != to - from) return false;
              for (int64_t i = 0; i < to - from; ++i)
                if (dims[i] != from + i) return false;
              return true;
            };
            if (seq(0, nloops)) maps2.push_back(AffineMap::get(2, 0, {r0, r1}, ctx));
            else if (seq(0, split)) maps2.push_back(AffineMap::get(2, 0, {r0}, ctx));
            else if (seq(split, nloops)) maps2.push_back(AffineMap::get(2, 0, {r1}, ctx));
            else if (dims.empty()) maps2.push_back(AffineMap::get(2, 0, {}, ctx));
            else ok = false;
          }
        }
        ++split;
        if (!ok) return fail("more than two loops");
        merge = true;
      }
    }
    SmallVector<int64_t> ranges(nloops, -1);
    std::optional<Lin> dynInner;
    for (int i = 0; i < int(g->getNumOperands()) && !chunk; ++i) {
      Value opnd = g->getOperand(i);
      auto mt = dyn_cast<MemRefType>(opnd.getType());
      if (!mt) continue;
      for (unsigned r = 0; r < maps[i].getNumResults(); ++r) {
        auto de = dyn_cast<AffineDimExpr>(maps[i].getResult(r));
        if (!de) continue;
        int L = int(de.getPosition());
        if (!mt.isDynamicDim(r)) {
          ranges[L] = mt.getDimSize(r);
        } else if (ranges[L] < 0) {
          ranges[L] = cfg.maxDynamic;
          if (L != nloops - 1) return fail("dynamic loop other than the innermost");
          if (!dynInner) {
            auto lb = bufOf(opnd);
            if (!lb || !lb->dyn) return fail("dynamic operand without its length");
            dynInner = lb->dyn;
          }
        }
      }
    }
    for (int64_t r : ranges)
      if (r < 0 && !chunk) return fail("loop range not given by an operand");
    if (merge) {
      int64_t rows = 1, inner = 1;
      for (int L = 0; L < split; ++L) rows *= ranges[L];
      for (int L = split; L < nloops; ++L) inner *= ranges[L];
      if (split < nloops - 1 && dynInner) return fail("more than two loops with a dynamic length");
      ranges = {rows, inner};
      iters = {iters.front(), iters.back()};
      maps = maps2;
      indexShift = nloops - 2;                            // linalg.index n-1 -> 1
      nloops = 2;
    }
    if (collapse) {
      if (dynInner) return fail("dynamic nest of more than two loops");
      int64_t prod = 1;
      for (int64_t r : ranges) prod *= r;
      ranges = {prod};
      nloops = 1;
    }
    if (chunk) {                                          // a slice of n elements, all inputs given
      ranges = {chunk->n};
      nloops = 1;
      dynInner.reset();
    }
    if (rowPiece && nloops >= 1 && ranges[0] == prR) ranges[0] = rowPiece->second;   // row pieces
    if (piece) {                                          // piece mode: identity maps, this piece flat
      ranges = {piece->second};
      nloops = 1;
    }
    // a dynamic innermost length: one row at a time (each VE then needs only a
    // dynamic LEN, and VALID for a mask)
    if (dynInner && !sel) {
      int64_t H = nloops == 2 ? ranges[0] : 1;
      if (H > 16) return fail("more than 16 rows of a dynamic length");
      curLen = paramFor(*dynInner);
      curLenStatic = (cfg.maxDynamic + d - 1) / d * d;
      for (int64_t r = 0; r < H; ++r) {
        // a row's temps are released after it, unless it allocated something
        // later rows use (a buffer's local, a broadcast: usually the first row)
        uint32_t savedTop[2] = {accTop[0], accTop[1]};
        size_t nLocals = locals.size(), nBcast = bcastOf.size();
        RowSel rs{r, *dynInner};
        if (!generic(g, &rs)) return false;
        if (locals.size() == nLocals && bcastOf.size() == nBcast) {
          accTop[0] = savedTop[0];
          accTop[1] = savedTop[1];
        }
      }
      curLen = Value();
      curLenStatic = -1;
      return err.empty();
    }
    bool reduction = false;
    for (int L = 0; L < nloops; ++L)
      if (iters[L] == utils::IteratorType::reduction) {
        if (L != nloops - 1) return fail("reduction not over the innermost loop");
        reduction = true;
      }
    int64_t n = 1;
    for (int64_t r : ranges) n *= r;
    int64_t inner = nloops ? ranges.back() : 1;
    if (sel) n = inner;                                  // one row
    // an element-wise nest with identity maps only: one flat loop; its length
    // is rounded up to D (local buffers are whole words; the store of the
    // result rounds its bytes to 8, inside IREE's 64-byte aligned allocations)
    bool allIdentity = !reduction && llvm::all_of(maps, [](AffineMap m) { return m.isIdentity(); }) &&
                       g.getRegion().front().getOps<linalg::IndexOp>().empty() && !dynInner && !chunk;
    if (nloops == 2 && inner % d && allIdentity) {
      ranges = {n};
      nloops = 1;
      inner = n;
    }
    if (nloops > 1 && inner % d) return fail("innermost size not a multiple of D");
    if (nloops == 1 && n > d && n % d && !reduction && !allIdentity) return fail("1-D size not a multiple of D");
    uint32_t wordsPerRow = uint32_t(inner / d);

    Block &body = g.getRegion().front();
    llvm::DenseMap<Value, Val> vals;
    int scalarArgNo = -1;
    // inputs
    for (int i = 0; i < nin; ++i) {
      Value in = g.getDpsInputs()[i];
      if (chunk && chunk->inputs.count(i)) {
        vals[body.getArgument(i)] = chunk->inputs.at(i);
        continue;
      }
      AffineMap m = maps[i];
      auto mt = dyn_cast<MemRefType>(in.getType());
      if (!mt) return fail("scalar generic input");
      Val v;
      if (mt.getElementType().isInteger(64)) {
        // i64 scalars: positions / indices (masks, gathers), read through LDPARAM
        if (mt.getRank() != 0 || !ddrRoot(in)) return fail("i64 input other than a scalar in DDR");
        v.kind = Val::ScalarArg;
        v.arg = i;
        scalarArgNo = i;
        vals[body.getArgument(i)] = v;
        continue;
      }
      auto lb = bufOf(in);
      if (!lb) return false;
      v.kind = Val::Mem;
      if (m.getNumResults() == 0) {
        auto b = scalarBcast(in, *lb);
        if (!b) return false;
        v.o = {b->la, ::sa::VT_F32, ::sa::IDX_DIV, 0xFFFF};
        v.uni = 1;
      } else if (m.isIdentity() || (int(m.getNumResults()) == nloops && m.isMinorIdentity())) {
        LocalBuf l = *lb;
        if (l.bcast && l.n > 1) {
          if (sel) return fail("broadcast-layout operand in a row-by-row generic");
          l = packBcast(l);
          locals[in] = l;
        }
        v.o = {l.la, l.vt, ::sa::IDX_LIN, 0};
        if (sel && nloops == 2) {
          if (!l.rowStride) return fail("row-by-row operand not in the row layout");
          v.o.la += uint32_t(sel->row) * l.rowStride;
        }
      } else if (nloops == 2 && m.getNumResults() == 1 && m.getResult(0) == getAffineDimExpr(0, g.getContext())) {
        auto b = perElementBcast(in, *lb, ranges[0]);
        if (!b) return false;
        v.o = {b->la, ::sa::VT_F32, ::sa::IDX_DIV, wordsPerRow};
        v.uni = 2;
        if (sel) {                                          // this row's value, in every element
          v.o = {b->la + uint32_t(sel->row), ::sa::VT_F32, ::sa::IDX_DIV, 0xFFFF};
          v.uni = 1;
        }
      } else if (nloops == 2 && m.getNumResults() == 1 && m.getResult(0) == getAffineDimExpr(1, g.getContext())) {
        v.o = {lb->la, lb->vt, sel ? ::sa::IDX_LIN : ::sa::IDX_MOD, sel ? 0u : wordsPerRow};
      } else {
        return fail("unsupported indexing map of a generic input");
      }
      vals[body.getArgument(i)] = v;
    }
    auto valOf = [&](Value x) -> Val {
      if (auto it = vals.find(x); it != vals.end()) return it->second;
      Val v;
      if (auto c = constF32(x)) {
        v.kind = Val::Const;
        v.c = *c;
      }
      return v;
    };
    SmallVector<Operation *> skip;

    // gathers: memref.load of a buffer captured from outside the body, the
    // index evaluated at every point of the (small) iteration space
    auto domain = points(ranges);
    if (domain.size() > 4096) domain.clear();
    for (memref::LoadOp ld : body.getOps<memref::LoadOp>()) {
      Value t = ld.getMemRef();
      auto mt = cast<MemRefType>(t.getType());
      SmallVector<int64_t> strides;
      int64_t offset;
      if (!mt.hasStaticShape() || failed(mt.getStridesAndOffset(strides, offset))) return fail("gather source layout");
      if (domain.empty() && nloops) return fail("gather over a large iteration space");
      SmallVector<SmallVector<int64_t>> dom = nloops ? domain : SmallVector<SmallVector<int64_t>>{{}};
      auto flatIndex = [&](ArrayRef<int64_t> pt, int64_t sym) -> std::optional<int64_t> {
        int64_t f = 0;
        for (unsigned i = 0; i < ld.getIndices().size(); ++i) {
          auto v = evalInt(ld.getIndices()[i], pt, sym);
          if (!v) return std::nullopt;
          f += *v * strides[i];
        }
        return f;
      };
      bool usesSym = false;
      for (Value iv : ld.getIndices()) {
        std::function<bool(Value)> hasArg = [&](Value v) -> bool {
          if (isa<BlockArgument>(v)) return true;
          Operation *o = v.getDefiningOp();
          if (!o || o->getBlock() != &body) return false;
          return llvm::any_of(o->getOperands(), hasArg);
        };
        usesSym |= hasArg(iv);
      }
      Type et = mt.getElementType();
      auto vt = vtOf(et);
      uint32_t es = esize(et);
      if (!vt) return fail("gather element type");
      Val e;
      if (usesSym) {
        if (scalarArgNo < 0) return fail("gather index from a non-scalar input");
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
        bool uniform = !dom.empty();
        auto u0 = flatIndex(dom.front(), 0), u1 = flatIndex(dom.front(), 1);
        for (size_t pi = 0; pi < dom.size() && uniform; ++pi)
          uniform = u0 && u1 && flatIndex(dom[pi], 0) == u0 && flatIndex(dom[pi], 1) == u1;
        if (uniform && *u0 == 0 && *u1 > 0) {
          rowGather = true;
          C = *u1;
        } else {
          uniform = false;
        }
        if (!rowGather || !C) {
          auto pe = packedRowGather(g, ld, mt, dom, scalarArgNo, flatIndex);
          if (!pe) return false;
          vals[ld.getResult()] = *pe;
          skip.push_back(ld);
          continue;
        }
        auto sym = ddrOf(g.getDpsInputs()[scalarArgNo]);
        auto src = ddrOf(t);
        if (!sym || !src) return fail("gather through buffers not in DDR");
        Value p = privateParam();
        if (*C * es > 0xFFFF) return fail("gather row larger than 64 KB");
        sahw::LdParamOp::create(bb, loc, sym->base, p, sym->off, *C * es, 0, ValueRange{});
        if (nloops == 0 || uniform) {
          // one element: its bits into a PARAM, then a word of it (1.0 * v + -0 = v exactly)
          if (*vt != ::sa::VT_F32) return fail("scalar gather of a non-fp32 value");
          Value p2 = privateParam();
          auto l2 = sahw::LdParamOp::create(bb, loc, src->base, p2, src->off, 1, 0, ValueRange{p});
          l2.setDynFields(ArrayRef<int32_t>{::sa::DYN_LDPARAM_ADDR});
          l2.setDynAdd(ArrayRef<bool>{true});
          LocalBuf ones = newLocal(::sa::VT_F32, 1, true);
          ve({ones.la, ::sa::VT_I32}, std::nullopt, ones.la, ::sa::VT_F32, d, ::sa::VOP_COPY, 0.0f, 1.0f);
          LocalBuf out = newLocal(::sa::VT_F32, 1, true);
          ve({ones.la}, std::nullopt, out.la, ::sa::VT_F32, d, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE,
             ::sa::RED_NONE, 0, 0, {{::sa::DYN_VE_A, p2, false}});
          e.kind = Val::Mem;
          e.o = {out.la, ::sa::VT_F32, ::sa::IDX_DIV, 0xFFFF};
          e.uni = 1;
        } else {
          LocalBuf row = newLocal(*vt, int64_t(dom.size()));
          uint32_t bytes = uint32_t((dom.size() * es + 7) / 8 * 8);
          if (src->off % 8 || (*C * es) % 8) return fail("gathered rows not 8-byte aligned");
          auto l = sahw::LdOp::create(bb, loc, src->base, src->off, int64_t(row.la), 1, int64_t(bytes),
                                      int64_t(bytes), 0, ValueRange{p});
          l.setDynFields(ArrayRef<int32_t>{::sa::DYN_DMA_DDR});
          l.setDynAdd(ArrayRef<bool>{true});
          e.kind = Val::Mem;
          e.o = {row.la, *vt, ::sa::IDX_LIN, 0};
          e.fresh = true;
        }
      } else {
        // a fixed permutation of a small tensor: the pair swap
        bool swap = !dom.empty();
        for (size_t pi = 0; pi < dom.size() && swap; ++pi) {
          auto f = flatIndex(dom[pi], 0);
          if (!f || *f != int64_t(pi ^ 1)) swap = false;
        }
        if (!swap) return fail("gather with a fixed index pattern other than the pair swap");
        auto l = materialize(t);
        if (!l) return false;
        e.kind = Val::SwapSrc;
        e.o = {l->la, l->vt, ::sa::IDX_LIN, 0};
      }
      vals[ld.getResult()] = e;
      skip.push_back(ld);
    }

    // the to_i8 chains
    llvm::DenseMap<Value, Value> toI8Src;
    g.walk([&](arith::FPToSIOp fp) {
      SmallVector<Operation *> chain;
      if (Value x = matchToI8(fp, chain)) {
        toI8Src[fp.getResult()] = x;
        for (Operation *o : chain)
          if (o != fp.getOperation()) skip.push_back(o);
      }
    });
    // a new fp32 value: of the whole iteration space, or computed once per
    // scalar (one broadcast word) / per row (a word per row) when the operands
    // only depend on those (as C3: the uniform value is computed once)
    auto temp = [&](const Val &x1, const std::optional<Val> &x2, ::sa::VOp op, float A, float B, ::sa::VFunc fn,
                    bool swapneg = false) -> Val {
      int u = x1.uni;
      if (x2) u = (u == 0 || x2->uni == 0) ? 0 : std::max(u, x2->uni);
      Opd s1 = x1.o;
      std::optional<Opd> s2;
      if (x2) s2 = x2->o;
      Val v;
      v.kind = Val::Mem;
      if (u != 0 && !swapneg && (u == 1 || nloops == 2)) {
        int64_t words = u == 1 ? 1 : ranges[0];
        auto perWord = [&](Opd &o, int ou) {
          // the per-row array itself (word r = row r); a scalar computed into
          // one word reads its word linearly
          if ((ou == 2 || (u == 1 && ou == 1)) && !o.isImm) {
            o.mode = ::sa::IDX_LIN;
            o.period = 0;
          }
        };
        perWord(s1, x1.uni);
        if (s2) perWord(*s2, x2->uni);
        LocalBuf t = newLocal(::sa::VT_F32, words, true);
        ve(s1, s2, t.la, ::sa::VT_F32, words * d, op, A, B, fn);
        v.o = u == 1 ? Opd{t.la, ::sa::VT_F32, ::sa::IDX_DIV, 0xFFFF} : Opd{t.la, ::sa::VT_F32, ::sa::IDX_DIV, wordsPerRow};
        v.uni = u;
        return v;
      }
      LocalBuf t = newLocal(::sa::VT_F32, n);
      ve(s1, s2, t.la, ::sa::VT_F32, n, op, A, B, fn, ::sa::RED_NONE, 0, 0, {}, swapneg);
      v.o = {t.la, ::sa::VT_F32, ::sa::IDX_LIN, 0};
      v.producer = lastVe;
      return v;
    };
    Value acc = reduction ? body.getArguments().back() : Value();
    for (Operation &opRef : body.without_terminator()) {
      Operation *op = &opRef;
      if (llvm::is_contained(skip, op)) continue;
      Value res = op->getNumResults() ? op->getResult(0) : Value();
      if (auto c = dyn_cast<arith::ConstantOp>(op)) {
        if (auto fc = constF32(c)) {
          Val v;
          v.kind = Val::Const;
          v.c = *fc;
          vals[res] = v;
        }
        continue;                                      // integer constants are read where used
      }
      if (auto ix = dyn_cast<linalg::IndexOp>(op)) {
        Val v;
        v.kind = Val::Index;
        v.dim = int(ix.getDim()) - indexShift;
        vals[res] = v;
        continue;
      }
      if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(op)) {
        Val x = valOf(op->getOperand(0));
        if (x.kind == Val::Index || x.kind == Val::ScalarArg) {
          vals[res] = x;
        } else {
          Val v;
          v.kind = Val::IntExpr;
          vals[res] = v;
        }
        continue;
      }
      if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp, arith::RemSIOp>(op) &&
          res.getType().isIndex()) {
        Val v;                                         // index arithmetic: evaluated where used
        v.kind = Val::IntExpr;
        vals[res] = v;
        continue;
      }
      if (auto ci = dyn_cast<arith::CmpIOp>(op)) {
        Val a = valOf(ci.getLhs()), b = valOf(ci.getRhs());
        if (a.kind == Val::IntExpr || b.kind == Val::IntExpr || (a.kind == Val::Index && b.kind != Val::ScalarArg)) {
          Val v;                                       // a condition on indices only (evaluated where used)
          v.kind = Val::IdxCond;
          vals[res] = v;
          continue;
        }
        if (a.kind != Val::Index || b.kind != Val::ScalarArg || a.dim != nloops - 1 ||
            (ci.getPredicate() != arith::CmpIPredicate::sle && ci.getPredicate() != arith::CmpIPredicate::slt))
          return fail("comparison other than a prefix mask (innermost index <= scalar)");
        Val v;
        v.kind = Val::Mask;
        v.dim = a.dim;
        v.arg = b.arg;
        v.pred = ci.getPredicate();
        vals[res] = v;
        continue;
      }
      if (auto s = dyn_cast<arith::SelectOp>(op); s && valOf(s.getCondition()).kind == Val::IdxCond) {
        // select(cond(i), -swap(x), swap(x)) with cond(i) = (i even): the VE's SWAPNEG
        Val tv = valOf(s.getTrueValue()), fv = valOf(s.getFalseValue());
        if (tv.kind != Val::NegSwap || fv.kind != Val::SwapSrc || tv.o.la != fv.o.la || nloops > 2 ||
            (nloops == 2 && ranges.back() % 2))
          return fail("select on an index condition other than the pair swap with negation");
        for (size_t pi = 0; pi < domain.size(); ++pi) {
          auto c = evalInt(s.getCondition(), domain[pi], 0);
          if (!c || bool(*c) != (pi % 2 == 0)) return fail("pair swap with a different sign pattern");
        }
        Val x = fv;
        x.kind = Val::Mem;
        vals[res] = temp(x, std::nullopt, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE, /*swapneg=*/true);
        continue;
      }
      if (auto sl = dyn_cast<arith::SelectOp>(op)) {
        // a prefix mask: VALID = pos + 1 (sle) or pos (slt), pos = the i64 scalar
        // (its low word, through LDPARAM, once per scalar and predicate)
        Val c = valOf(sl.getCondition());
        if (c.kind != Val::Mask) return fail("select other than a prefix mask");
        bool negInf = isF32Const(sl.getFalseValue(), -INFINITY);
        if (!isF32Const(sl.getFalseValue(), 0.0f) && !negInf) return fail("masked value other than 0 / -inf");
        Value scalar = g.getDpsInputs()[c.arg];
        auto key = std::make_pair(scalar.getAsOpaquePointer(), int(c.pred));
        Value p;
        if (auto vi = validParams.find(key); vi != validParams.end()) {
          p = vi->second;
        } else {
          auto r = ddrOf(scalar);
          if (!r) return false;
          p = privateParam();
          sahw::LdParamOp::create(bb, loc, r->base, p, r->off, 1, c.pred == arith::CmpIPredicate::sle ? 1 : 0,
                                  ValueRange{});
          validParams[key] = p;
        }
        Val x = valOf(sl.getTrueValue());
        if (x.kind != Val::Mem) return fail("masked value is not a vector value");
        LocalBuf t = newLocal(::sa::VT_F32, n);
        ve(x.o, std::nullopt, t.la, ::sa::VT_F32, n, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE, ::sa::RED_NONE, 0,
           0, {{::sa::DYN_VE_VALID, p, false}});
        Val v;
        v.kind = Val::Mem;
        v.o = {t.la, ::sa::VT_F32, ::sa::IDX_LIN, 0};
        v.producer = lastVe;
        v.negInf = negInf;
        vals[res] = v;
        continue;
      }
      if (auto fp = dyn_cast<arith::FPToSIOp>(op)) {
        auto it = toI8Src.find(res);
        if (it == toI8Src.end()) return fail("fptosi other than to_i8");
        Val v;
        v.kind = Val::ToI8;
        v.inner = std::make_shared<Val>(valOf(it->second));
        vals[res] = v;
        continue;
      }
      if (auto ex = dyn_cast<arith::ExtSIOp>(op)) {
        Val x = valOf(ex.getIn());
        if (x.kind != Val::ToI8 || !ex.getType().isInteger(32)) return fail("extsi other than i8 -> i32 of to_i8");
        Val v;
        v.kind = Val::I32OfI8;
        v.inner = std::make_shared<Val>(x);
        vals[res] = v;
        continue;
      }
      if (auto tr = dyn_cast<arith::TruncIOp>(op)) {
        Val x = valOf(tr.getIn());
        if (x.kind != Val::Mem || x.o.vt == ::sa::VT_F32 || !tr.getType().isInteger(32))
          return fail("trunci other than of an accumulator");
        vals[res] = x;                                   // the accumulator is int32 on the device
        continue;
      }
      if (auto sf = dyn_cast<arith::SIToFPOp>(op)) {
        Val x = valOf(sf.getIn());
        if (x.kind != Val::Mem || (x.o.vt == ::sa::VT_F32 && !x.exactInt)) return fail("sitofp of a computed value");
        vals[res] = x;                                   // the VE reads integer sources as fp32
        continue;
      }
      auto unary = [&](float A, float B, ::sa::VFunc fn) -> bool {
        Val x = valOf(op->getOperand(0));
        if (x.kind != Val::Mem) return fail("unary operation on a non-vector value");
        vals[res] = temp(x, std::nullopt, ::sa::VOP_COPY, A, B, fn);
        return true;
      };
      if (isa<arith::NegFOp>(op)) {
        Val x = valOf(op->getOperand(0));
        if (x.kind == Val::SwapSrc) {
          x.kind = Val::NegSwap;
          vals[res] = x;
          continue;
        }
        if (!unary(-1.0f, NEG0, ::sa::FUNC_NONE)) return false;
        continue;
      }
      if (isa<math::ExpOp>(op)) {
        if (!unary(1.0f, NEG0, ::sa::FUNC_EXP)) return false;
        continue;
      }
      if (isa<math::RsqrtOp>(op)) {
        if (!unary(1.0f, NEG0, ::sa::FUNC_RSQRT)) return false;
        continue;
      }
      if (isa<math::AbsFOp>(op)) {
        if (!unary(1.0f, NEG0, ::sa::FUNC_ABS)) return false;
        continue;
      }
      if (auto dv = dyn_cast<arith::DivFOp>(op)) {
        if (!isF32Const(dv.getLhs(), 1.0f)) return fail("division other than 1 / x");
        Val x = valOf(dv.getRhs());
        if (x.kind != Val::Mem) return fail("1 / constant");
        vals[res] = temp(x, std::nullopt, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_RECIP);
        continue;
      }
      if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::MaximumFOp, arith::MinimumFOp>(op)) {
        Value lhs = op->getOperand(0), rhs = op->getOperand(1);
        if (reduction && (lhs == acc || rhs == acc)) continue;          // the combiner, at the yield
        Val a = valOf(lhs), b = valOf(rhs);
        bool comm = isa<arith::AddFOp, arith::MulFOp, arith::MaximumFOp, arith::MinimumFOp>(op);
        if (a.kind == Val::Const && b.kind == Val::Const) return fail("constant folding left to the compiler");
        if (b.kind == Val::Const || (comm && a.kind == Val::Const)) {
          if (a.kind == Val::Const) std::swap(a, b);
          if (a.kind != Val::Mem) return fail("operand is not a vector value");
          float c = b.c;
          if (isa<arith::MulFOp>(op)) vals[res] = temp(a, std::nullopt, ::sa::VOP_COPY, c, NEG0, ::sa::FUNC_NONE);
          else if (isa<arith::AddFOp>(op))
            vals[res] = temp(a, std::nullopt, ::sa::VOP_COPY, 1.0f, c, ::sa::FUNC_NONE);
          else if (isa<arith::SubFOp>(op))
            vals[res] = temp(a, std::nullopt, ::sa::VOP_COPY, 1.0f, -c, ::sa::FUNC_NONE);
          else {
            Val imm;
            imm.kind = Val::Mem;
            imm.o.isImm = true;
            imm.o.imm = c;
            imm.uni = 1;
            vals[res] = temp(a, imm, isa<arith::MaximumFOp>(op) ? ::sa::VOP_MAX : ::sa::VOP_MIN, 1.0f, NEG0,
                             ::sa::FUNC_NONE);
          }
          continue;
        }
        if (a.kind == Val::Const && isa<arith::SubFOp>(op)) {              // c - x = (-x) + c
          if (b.kind != Val::Mem) return fail("operand is not a vector value");
          vals[res] = temp(b, std::nullopt, ::sa::VOP_COPY, -1.0f, a.c, ::sa::FUNC_NONE);
          continue;
        }
        if (a.kind != Val::Mem || b.kind != Val::Mem) return fail("operand is not a vector value");
        if (a.negInf || b.negInf) return fail("-inf mask outside a max reduction");
        Val s1 = a, s2 = b;
        if (s1.o.mode != ::sa::IDX_LIN && s2.o.mode == ::sa::IDX_LIN && comm) std::swap(s1, s2);
        ::sa::VOp vop = isa<arith::AddFOp>(op)       ? ::sa::VOP_ADD
                        : isa<arith::SubFOp>(op)     ? ::sa::VOP_SUB
                        : isa<arith::MulFOp>(op)     ? ::sa::VOP_MUL
                        : isa<arith::MaximumFOp>(op) ? ::sa::VOP_MAX
                                                     : ::sa::VOP_MIN;
        vals[res] = temp(s1, s2, vop, 1.0f, NEG0, ::sa::FUNC_NONE);
        continue;
      }
      return fail(std::string("unsupported body operation ") + op->getName().getStringRef().str());
    }

    auto yield = cast<linalg::YieldOp>(body.getTerminator());
    if (reduction) {
      if (g.getNumDpsInits() != 1) return fail("multi-result reduction");
      Operation *comb = yield.getOperand(0).getDefiningOp();
      if (!comb || comb->getNumOperands() != 2) return fail("reduction combiner");
      Value elem = comb->getOperand(0) == acc ? comb->getOperand(1) : comb->getOperand(0);
      ::sa::VRed rk;
      float ident;
      if (isa<arith::AddFOp>(comb)) rk = ::sa::RED_SUM, ident = 0.0f;
      else if (isa<arith::MaximumFOp>(comb)) rk = ::sa::RED_MAX, ident = -INFINITY;
      else return fail("reduction other than an fp32 sum or max");
      Value init = g.getDpsInits()[0];
      auto fit = fills.find(init);
      if (fit == fills.end()) return fail("reduction init is not filled");
      uint32_t a, b;
      std::memcpy(&a, &fit->second, 4);
      std::memcpy(&b, &ident, 4);
      if (a != b) return fail("reduction init is not the identity");
      Val e = valOf(elem);
      if (e.kind != Val::Mem) return fail("reduced value is not a vector value");
      if (e.negInf && rk != ::sa::RED_MAX) return fail("-inf mask outside a max reduction");
      auto out = bufOf(init, /*bcast=*/true);
      if (!out) return false;
      if (!out->bcast) return fail("reduction result used before");
      // piece mode: a later piece's maximum into a temp word, then into the result
      uint32_t dst = out->la;
      if (piece && piece->first > 0) {
        if (rk != ::sa::RED_MAX) return fail("a sum over pieces");
        dst = newLocal(::sa::VT_F32, 1, /*bcast=*/true).la;
      }
      auto combine = [&]() {
        if (dst != out->la)
          ve(Opd{out->la, ::sa::VT_F32, ::sa::IDX_LIN, 0}, Opd{dst, ::sa::VT_F32, ::sa::IDX_LIN, 0}, out->la,
             ::sa::VT_F32, d, ::sa::VOP_MAX);
        return err.empty();
      };
      int64_t groups = n / d;
      auto absOp = elem.getDefiningOp<math::AbsFOp>();
      if (!sel && rk == ::sa::RED_MAX && absOp && elem.hasOneUse() && e.producer && e.uni == 0 && nloops == 1 &&
          n % d == 0 && groups >= 8) {
        // an abs-max (values >= +0: any order gives the same bits): |x| in the
        // pipelined VE mode, halved in place by element-wise maxima, then a
        // REDUCE over the few groups left (REDUCE runs one group at a time)
        while (groups % 2 == 0 && groups > 4) {
          int64_t h = groups / 2;
          ve(e.o, Opd{e.o.la + uint32_t(h), ::sa::VT_F32, ::sa::IDX_LIN, 0}, e.o.la, ::sa::VT_F32, h * d,
             ::sa::VOP_MAX);
          groups = h;
        }
        ve(e.o, std::nullopt, dst, ::sa::VT_F32, groups * d, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE, rk,
           uint32_t(groups));
        return combine();
      }
      if (sel) {                                         // this row's word
        ve(e.o, std::nullopt, out->la + uint32_t(sel->row), ::sa::VT_F32, n, ::sa::VOP_COPY, 1.0f, NEG0,
           ::sa::FUNC_NONE, rk, 0);
        return err.empty();
      }
      ve(e.o, std::nullopt, dst, ::sa::VT_F32, n, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE, rk,
         uint32_t(inner / d));
      return combine();
    }
    if (chunk && (reduction || g.getNumDpsInits() != 1)) return fail("linear epilogue with a reduction or two results");
    for (unsigned r = 0; r < unsigned(g.getNumDpsInits()); ++r) {
      Value init = g.getDpsInits()[r];
      Val y = valOf(yield.getOperand(r));
      if (chunk) {
        if (y.kind != Val::Mem || y.negInf) return fail("linear epilogue result");
        ve(y.o, std::nullopt, chunk->outLa, ::sa::VT_F32, chunk->n, ::sa::VOP_COPY);
        continue;
      }
      if (y.negInf) return fail("-inf mask outside a max reduction");
      if (!sel && y.kind == Val::Mem && y.fresh && !locals.count(init) && init.getDefiningOp<memref::AllocOp>() &&
          vtOf(cast<MemRefType>(init.getType()).getElementType()) == y.o.vt) {
        LocalBuf row;                                    // the result is the gathered row itself
        row.la = y.o.la;
        row.vt = y.o.vt;
        row.n = cast<MemRefType>(init.getType()).getNumElements();
        locals[init] = row;
        continue;
      }
      auto out0 = bufOf(init);
      if (!out0) return false;
      if (out0->bcast) {
        // a buffer in the broadcast layout (a reduction's result) reused for an
        // element-wise result: a packed buffer of its own from here on (the
        // readers so far have read the broadcast words)
        if (sel) return fail("element-wise result into a broadcast buffer");
        LocalBuf nb = newLocal(out0->vt, out0->n);
        locals[init] = nb;
        bcastOf.erase(init);
        out0 = nb;
      }
      LocalBuf outRow = *out0;
      if (sel && nloops == 2) {                          // this row of the row layout
        if (!outRow.rowStride) return fail("row-by-row result not in the row layout");
        outRow.la += uint32_t(sel->row) * outRow.rowStride;
      }
      const LocalBuf *out = &outRow;
      if (y.kind == Val::Mem) {
        if (y.o.mode != ::sa::IDX_LIN && out->vt != ::sa::VT_F32) return fail("broadcast into a non-fp32 result");
        Opd o = y.o;
        if (y.uni == 1 && n <= d) o.mode = ::sa::IDX_LIN, o.period = 0;   // one word
        ve(o, std::nullopt, out->la, out->vt, n, ::sa::VOP_COPY);
      } else if (y.kind == Val::ToI8 && out->vt == ::sa::VT_I8) {
        if (y.inner->kind != Val::Mem) return fail("to_i8 of a non-vector value");
        ve(y.inner->o, std::nullopt, out->la, ::sa::VT_I8, n, ::sa::VOP_COPY);
      } else if (y.kind == Val::I32OfI8 && out->vt == ::sa::VT_I32) {
        const Val &x = *y.inner->inner;
        if (x.kind != Val::Mem) return fail("to_i8 of a non-vector value");
        LocalBuf t8 = newLocal(::sa::VT_I8, n);
        ve(x.o, std::nullopt, t8.la, ::sa::VT_I8, n, ::sa::VOP_COPY);
        ve({t8.la, ::sa::VT_I8, ::sa::IDX_LIN, 0}, std::nullopt, out->la, ::sa::VT_I32, n, ::sa::VOP_COPY);
      } else {
        return fail("unsupported generic result");
      }
    }
    return err.empty();
  }
};

struct SahlToSahwPass : public PassWrapper<SahlToSahwPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SahlToSahwPass)
  SahlToSahwPass() = default;
  explicit SahlToSahwPass(const TargetConfig &c) : cfg(c) {}
  SahlToSahwPass(const SahlToSahwPass &o) : PassWrapper(o), cfg(o.cfg) {}
  StringRef getArgument() const override { return "iree-sahl-to-sahw"; }
  StringRef getDescription() const override {
    return "Lowers each sahl function (linalg on local buffers + sahl.load / store) to a sahw.template";
  }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<sahw::SahwDialect>(); }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    SmallVector<func::FuncOp> funcs(m.getOps<func::FuncOp>());
    for (func::FuncOp f : funcs) {
      OpBuilder b = OpBuilder::atBlockEnd(m.getBody());
      auto t = sahw::TemplateOp::create(b, f.getLoc(), f.getSymName(), 0, 0, 0, false);
      t->setAttr("sa.d", b.getI64IntegerAttr(cfg.d));
      OpBuilder tb = OpBuilder::atBlockEnd(&t.getRegion().emplaceBlock());
      auto body = sahw::BodyOp::create(tb, f.getLoc());
      body.getRegion().emplaceBlock();
      Lowerer low(f, cfg, t, body);
      if (!low.run()) {
        f.emitError() << "sahl-to-sahw: " << low.err;
        t.erase();
        return signalPassFailure();
      }
    }
  }

  TargetConfig cfg;
};

}  // namespace

std::unique_ptr<Pass> createSahlToSahwPass(const TargetConfig &config) {
  return std::make_unique<SahlToSahwPass>(config);
}
void registerSahlToSahwPass() { PassRegistration<SahlToSahwPass>(); }

}  // namespace mlir::iree_compiler::sa

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
  };
  std::optional<Ddr> ddrOf(Value v) {
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
    int64_t expect = 1;
    for (int i = int(mt.getRank()) - 1; i >= 0; --i) {
      if (mt.getDimSize(i) != 1 && strides[i] != expect) return fail("non-contiguous DDR view"), std::nullopt;
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

  bool load(sahl::LoadOp l) {
    auto r = ddrOf(l.getSrc());
    if (!r) return false;
    if (r->et.isInteger(64)) return fail("i64 load");
    auto dst = bufOf(l.getDst());
    if (!dst) return false;
    if (r->dyn) return dynamicDma(*r, *dst, true);
    uint32_t bytes = uint32_t((r->n * esize(r->et) + 7) / 8 * 8);
    if (r->off % 8) return fail("load not 8-byte aligned");
    if (bytes > 65535) return fail("load larger than 64 KB");
    sahw::LdOp::create(bb, loc, r->base, r->off, int64_t(dst->la), 1, int64_t(bytes), int64_t(bytes), 0, ValueRange{});
    return true;
  }

  bool store(sahl::StoreOp s) {
    auto r = ddrOf(s.getDst());
    if (!r) return false;
    auto it = locals.find(s.getSrc());
    if (it == locals.end()) return fail("stored buffer not computed");
    LocalBuf l = it->second;
    if (r->dyn) return dynamicDma(*r, l, false);
    if (l.bcast && l.n > 1) l = packBcast(l);
    uint32_t bytes = uint32_t((r->n * esize(r->et) + 7) / 8 * 8);
    if (r->off % 8) return fail("store not 8-byte aligned");
    if (bytes > 65535) return fail("store larger than 64 KB");
    sahw::StOp::create(bb, loc, r->base, r->off, int64_t(l.la), 1, int64_t(bytes), int64_t(bytes), ValueRange{});
    return true;
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
    // chunks of output tiles: the B tiles of a chunk in one SPAD_B bank (a dynamic N: all tiles, one chunk)
    uint32_t nc = nt;
    if (!Nr.dyn) {
      uint32_t cap = std::max<uint32_t>(sb / uint32_t(K), 1);
      for (nc = std::min(cap, nt); nc > 1 && nt % nc; --nc) {
      }
    }
    if (uint32_t(K) * nc > sb) return fail("contraction: one chunk of B tiles does not fit a SPAD bank");
    bool xRows = (p.bLoop >= 0 && p.x.coef[p.bLoop] < 0) || (p.gLoop >= 0 && p.x.coef[p.gLoop] < 0);
    if (xRows && nc != nt) return fail("contraction: x rows of a dynamic length and several chunks");
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
    auto loadStrip = [&](int64_t b, int64_t g) -> bool {
      // x_(b, g) -> the A strip (each element over the D rows)
      int64_t xoff = xs->second;
      if (!xRows) xoff += ((p.bLoop >= 0 ? b * p.x.coef[p.bLoop] : 0) + (p.gLoop >= 0 ? g * p.x.coef[p.gLoop] : 0)) *
                          int64_t(xes);
      if (xoff % 8) return fail("contraction: unaligned x");
      auto lx = sahw::LdOp::create(bb, loc, xs->first, xoff, int64_t(xl.la), 1, K * xes, K * xes, 0, ValueRange{});
      if (xRows) {
        Value xb = parOf(*Kr.dyn, xes, 0);
        dyn(lx, {{::sa::DYN_DMA_DDR, rowAcc, true}, {::sa::DYN_DMA_ROW_BYTES, xb, false}});
        addRow(xb);
      } else if (Kr.dyn) {
        dyn(lx, {{::sa::DYN_DMA_ROW_BYTES, parOf(*Kr.dyn, xes, 0), false}});
      }
      ve({xl.la, xl.vt, ::sa::IDX_DIV, uint32_t(d)}, std::nullopt, ::sa::laddr(::sa::MEM_SPAD_A, strip),
         ::sa::VT_I8, K * d, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE, ::sa::RED_NONE, 0, 0,
         Kr.dyn ? SmallVector<DynF>{{::sa::DYN_VE_LEN, parOf(*Kr.dyn, d, 0), false}} : SmallVector<DynF>{});
      return true;
    };
    for (int64_t b = 0; b < H; ++b) {
      int64_t moff = ms->second + (p.bLoop >= 0 ? b * p.m.coef[p.bLoop] : 0);
      if (moff % 8) return fail("contraction: unaligned matrix");
      if (G == 1 && !loadStrip(b, 0)) return false;
      for (uint32_t c0 = 0; c0 < nt; c0 += nc) {
        uint32_t bank = (c0 / nc) & 1, bw = bank * sb;
        uint32_t la = ::sa::laddr(::sa::MEM_SPAD_B, bw);
        // the B tiles of this chunk (shared by the rows of x)
        if (p.layout == MatLayout::Packed) {
          sahw::LdOp::create(bb, loc, ms->first, moff + int64_t(c0) * K * d, int64_t(la), nc, K * d, K * d, 0,
                             ValueRange{});
        } else if (p.layout == MatLayout::RowsK) {
          int64_t sN = p.m.coef[p.nLoop];
          auto l = sahw::LdOp::create(bb, loc, ms->first, moff + int64_t(c0) * d * sN,
                                      int64_t(::sa::laddr(::sa::MEM_SPAD_A, raw)), int64_t(nc) * d, K, sN, 0,
                                      ValueRange{});
          auto t = sahw::TransposeOp::create(bb, loc, int64_t(::sa::laddr(::sa::MEM_SPAD_A, raw)), int64_t(la),
                                             int64_t(nc) * d * K, int64_t(::sa::vtypes(::sa::VT_I8, ::sa::VT_I8)),
                                             K / d, ValueRange{});
          if (Nr.dyn) {
            dyn(l, {{::sa::DYN_DMA_ROWS, pN, false}});
            dyn(t, {{::sa::DYN_VE_LEN, parOf(*Nr.dyn, K, 0), false}});
          }
        } else {
          int64_t sK = p.m.coef[p.kLoop];
          auto l = sahw::LdOp::create(bb, loc, ms->first, moff + int64_t(c0) * d, int64_t(la), K,
                                      int64_t(nc) * d, sK, /*INTERLEAVE=*/1, ValueRange{});
          if (Kr.dyn) dyn(l, {{::sa::DYN_DMA_ROWS, pK, false}});
        }
        for (int64_t g = 0; g < G; ++g) {
          if (G > 1 && !loadStrip(b, g)) return false;
          // the chunk's accumulator and epilogue buffers: released after its store
          uint32_t savedTop[2] = {accTop[0], accTop[1]};
          LocalBuf acc = newLocal(::sa::VT_I32, int64_t(nc) * d);
          uint32_t cw = acc.la & 0xFFFFFFF;
          auto ex = sahw::ExOp::create(bb, loc, int64_t(strip), int64_t(bw), int64_t(cw), K / d, false, int64_t(nc),
                                       K, 1, int64_t(nc), ValueRange{});
          if (Kr.dyn) dyn(ex, {{::sa::DYN_EX_KT, parOf(*Kr.dyn, 1, logd), false}, {::sa::DYN_EX_BSTEP, pK, false}});
          if (Nr.dyn) dyn(ex, {{::sa::DYN_EX_REPEAT, parOf(*Nr.dyn, 1, logd), false}});
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
    p.acc = con.getDpsInits()[0];
    linalg::FillOp fill;
    for (Operation *u : p.acc.getUsers()) {
      if (u == con.getOperation()) continue;
      if (auto f = dyn_cast<linalg::FillOp>(u)) fill = f;
      else if (auto e = dyn_cast<linalg::GenericOp>(u); e && !p.epi) p.epi = e;
      else if (!isa<memref::DeallocOp>(u)) return false;
    }
    if (!fill || !p.epi || !getConstantIntValue(fill.getDpsInputs()[0]) ||
        *getConstantIntValue(fill.getDpsInputs()[0]) != 0)
      return false;
    linalg::GenericOp e = p.epi;
    if (e.getNumDpsInits() != 1 || e.getNumReductionLoops() || e.getNumLoops() != 2 ||
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
      if (e.getIndexingMapsArray()[i].isIdentity()) {
        if (!cast<MemRefType>(in.getType()).getElementType().isF32()) return false;
        p.chunked.push_back({i, l.getSrc()});
      } else {
        p.whole.push_back({i, l.getSrc()});
      }
      cover.push_back(l);
    }
    for (Operation *o : cover) owned.insert(o);
    linears[con.getOperation()] = p;
    return true;
  }

  bool linear(LinearPlan &p) {
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
    int indexShift = 0;
    SmallVector<AffineMap> maps2;
    if (nloops > 2 && !chunk) {
      bool noIndex = g.getRegion().front().getOps<linalg::IndexOp>().empty();
      bool red = llvm::any_of(iters, [](utils::IteratorType t) { return t == utils::IteratorType::reduction; });
      if (llvm::all_of(maps, [](AffineMap m) { return m.isIdentity(); }) && !red && noIndex) {
        collapse = true;                     // an element-wise nest with identity maps only: one flat loop
      } else {
        // the leading loops as one "row" loop: every map is the identity, the
        // leading loops, the last loop or nothing
        MLIRContext *ctx = g.getContext();
        AffineExpr r0 = getAffineDimExpr(0, ctx), r1 = getAffineDimExpr(1, ctx);
        // indices: only of the last loop (it stays the last), and no gathers
        bool ok = true;
        for (linalg::IndexOp ix : g.getRegion().front().getOps<linalg::IndexOp>())
          ok &= int(ix.getDim()) == nloops - 1;
        ok &= g.getRegion().front().getOps<memref::LoadOp>().empty();
        for (int L = 0; L + 1 < nloops; ++L) ok &= iters[L] == utils::IteratorType::parallel;
        for (AffineMap m : maps) {
          SmallVector<int64_t> dims;
          for (AffineExpr e : m.getResults()) {
            auto de = dyn_cast<AffineDimExpr>(e);
            if (!de) ok = false;
            else dims.push_back(de.getPosition());
          }
          bool lead = int(dims.size()) == nloops - 1, all = int(dims.size()) == nloops;
          for (size_t i = 0; i < dims.size() && (lead || all); ++i)
            if (dims[i] != int64_t(i)) lead = all = false;
          if (all) maps2.push_back(AffineMap::get(2, 0, {r0, r1}, ctx));
          else if (lead) maps2.push_back(AffineMap::get(2, 0, {r0}, ctx));
          else if (dims.size() == 1 && dims[0] == nloops - 1) maps2.push_back(AffineMap::get(2, 0, {r1}, ctx));
          else if (dims.empty()) maps2.push_back(AffineMap::get(2, 0, {}, ctx));
          else ok = false;
        }
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
      int64_t rows = 1;
      for (int L = 0; L + 1 < nloops; ++L) rows *= ranges[L];
      ranges = {rows, ranges.back()};
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
    // a dynamic innermost length: one row at a time (each VE then needs only a
    // dynamic LEN, and VALID for a mask)
    if (dynInner && !sel) {
      int64_t H = nloops == 2 ? ranges[0] : 1;
      if (H > 16) return fail("more than 16 rows of a dynamic length");
      curLen = paramFor(*dynInner);
      curLenStatic = (cfg.maxDynamic + d - 1) / d * d;
      for (int64_t r = 0; r < H; ++r) {
        RowSel rs{r, *dynInner};
        if (!generic(g, &rs)) return false;
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
        if (!rowGather || !C) return fail("gather index is not row * C + iteration index");
        auto sym = ddrOf(g.getDpsInputs()[scalarArgNo]);
        auto src = ddrOf(t);
        if (!sym || !src) return fail("gather through buffers not in DDR");
        Value p = privateParam();
        if (*C * es > 0xFFFF) return fail("gather row larger than 64 KB");
        sahw::LdParamOp::create(bb, loc, sym->base, p, sym->off, *C * es, 0, ValueRange{});
        if (nloops == 0) {
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
        if (x.kind != Val::Mem || x.o.vt == ::sa::VT_F32) return fail("sitofp of a computed value");
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
        ve(e.o, std::nullopt, out->la, ::sa::VT_F32, groups * d, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE, rk,
           uint32_t(groups));
        return err.empty();
      }
      if (sel) {                                         // this row's word
        ve(e.o, std::nullopt, out->la + uint32_t(sel->row), ::sa::VT_F32, n, ::sa::VOP_COPY, 1.0f, NEG0,
           ::sa::FUNC_NONE, rk, 0);
        return err.empty();
      }
      ve(e.o, std::nullopt, out->la, ::sa::VT_F32, n, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE, rk,
         uint32_t(inner / d));
      return err.empty();
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
      if (out0->bcast) return fail("element-wise result into a broadcast buffer");
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

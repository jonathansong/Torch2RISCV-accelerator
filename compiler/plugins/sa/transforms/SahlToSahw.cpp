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
#include "../target/Templates.h"
#include "SaLin.h"
#include "SahlDialect.h"
#include "SahlPasses.h"
#include "SahwPasses.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
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
  // Index: linalg.index `dim`; IntExpr / IdxCond: index arithmetic / conditions
  // (evaluated where used); ScalarArg: an i64 scalar input (`arg`); SwapSrc /
  // NegSwap: the pair swap of a buffer (negated)
  enum Kind { None, Const, Mem, ToI8, I32OfI8, Index, IntExpr, IdxCond, ScalarArg, SwapSrc, NegSwap } kind = None;
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
    for (Operation &opRef : f.getBody().front()) {
      Operation *op = &opRef;
      loc = op->getLoc();
      if (isa<arith::ConstantOp, memref::AllocOp, memref::DeallocOp, memref::SubViewOp, memref::CastOp,
              IREE::HAL::InterfaceBindingSubspanOp, IREE::HAL::InterfaceConstantLoadOp, func::ReturnOp>(op))
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

  // ------------------------------------------------------------ local memory
  std::optional<uint32_t> allocAcc(uint32_t words) {
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
    if (!vt || !mt.hasStaticShape()) return fail("local buffer type"), std::nullopt;
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
  };
  std::optional<Ddr> ddrOf(Value v) {
    auto mt = cast<MemRefType>(v.getType());
    if (!mt.hasStaticShape()) return fail("DDR view with a dynamic shape"), std::nullopt;
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

  bool load(sahl::LoadOp l) {
    auto r = ddrOf(l.getSrc());
    if (!r) return false;
    if (r->et.isInteger(64)) return fail("i64 load");
    auto dst = bufOf(l.getDst());
    if (!dst) return false;
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
  LocalBuf packBcast(const LocalBuf &b) {
    uint32_t w = uint32_t((b.n + d - 1) / d);
    LocalBuf p = newLocal(::sa::VT_F32, b.n);
    sahw::TransposeOp::create(bb, loc, int64_t(b.la), int64_t(p.la), int64_t(w * d * d),
                              int64_t(::sa::vtypes(::sa::VT_F32, ::sa::VT_F32)), 1, ValueRange{});
    return p;
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

  bool generic(linalg::GenericOp g) {
    auto iters = g.getIteratorTypesArray();
    int nloops = int(iters.size());
    SmallVector<AffineMap> maps = g.getIndexingMapsArray();
    int nin = g.getNumDpsInputs();
    if (nloops == 0 && g.getNumDpsInits() == 1 &&
        cast<MemRefType>(g.getDpsInits()[0].getType()).getElementType().isInteger(64))
      return intScalar(g);
    if (nloops > 2) return fail("more than two loops");
    SmallVector<int64_t> ranges(nloops, -1);
    for (int i = 0; i < int(g->getNumOperands()); ++i) {
      auto mt = dyn_cast<MemRefType>(g->getOperand(i).getType());
      if (!mt) continue;
      for (unsigned r = 0; r < maps[i].getNumResults(); ++r)
        if (auto de = dyn_cast<AffineDimExpr>(maps[i].getResult(r))) ranges[de.getPosition()] = mt.getDimSize(r);
    }
    for (int64_t r : ranges)
      if (r < 0) return fail("loop range not given by an operand");
    bool reduction = false;
    for (int L = 0; L < nloops; ++L)
      if (iters[L] == utils::IteratorType::reduction) {
        if (L != nloops - 1) return fail("reduction not over the innermost loop");
        reduction = true;
      }
    int64_t n = 1;
    for (int64_t r : ranges) n *= r;
    int64_t inner = nloops ? ranges.back() : 1;
    if (nloops > 1 && inner % d) return fail("innermost size not a multiple of D");
    if (nloops == 1 && n > d && n % d && !reduction) return fail("1-D size not a multiple of D");
    uint32_t wordsPerRow = uint32_t(inner / d);

    Block &body = g.getRegion().front();
    llvm::DenseMap<Value, Val> vals;
    int scalarArgNo = -1;
    // inputs
    for (int i = 0; i < nin; ++i) {
      Value in = g.getDpsInputs()[i];
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
          l = packBcast(l);
          locals[in] = l;
        }
        v.o = {l.la, l.vt, ::sa::IDX_LIN, 0};
      } else if (nloops == 2 && m.getNumResults() == 1 && m.getResult(0) == getAffineDimExpr(0, g.getContext())) {
        auto b = perElementBcast(in, *lb, ranges[0]);
        if (!b) return false;
        v.o = {b->la, ::sa::VT_F32, ::sa::IDX_DIV, wordsPerRow};
        v.uni = 2;
      } else if (nloops == 2 && m.getNumResults() == 1 && m.getResult(0) == getAffineDimExpr(1, g.getContext())) {
        v.o = {lb->la, lb->vt, ::sa::IDX_MOD, wordsPerRow};
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
        v.dim = int(ix.getDim());
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
        return fail("comparison other than on indices");
      }
      if (auto s = dyn_cast<arith::SelectOp>(op); s && valOf(s.getCondition()).kind == Val::IdxCond) {
        // select(cond(i), -swap(x), swap(x)) with cond(i) = (i even): the VE's SWAPNEG
        Val tv = valOf(s.getTrueValue()), fv = valOf(s.getFalseValue());
        if (tv.kind != Val::NegSwap || fv.kind != Val::SwapSrc || tv.o.la != fv.o.la || nloops != 1)
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
      auto out = bufOf(init, /*bcast=*/true);
      if (!out) return false;
      if (!out->bcast) return fail("reduction result used before");
      int64_t groups = n / d;
      auto absOp = elem.getDefiningOp<math::AbsFOp>();
      if (rk == ::sa::RED_MAX && absOp && elem.hasOneUse() && e.producer && e.uni == 0 && nloops == 1 &&
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
      ve(e.o, std::nullopt, out->la, ::sa::VT_F32, n, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE, rk,
         uint32_t(inner / d));
      return err.empty();
    }
    for (unsigned r = 0; r < unsigned(g.getNumDpsInits()); ++r) {
      Value init = g.getDpsInits()[r];
      Val y = valOf(yield.getOperand(r));
      if (y.kind == Val::Mem && y.fresh && !locals.count(init) && init.getDefiningOp<memref::AllocOp>() &&
          vtOf(cast<MemRefType>(init.getType()).getElementType()) == y.o.vt) {
        LocalBuf row;                                    // the result is the gathered row itself
        row.la = y.o.la;
        row.vt = y.o.vt;
        row.n = cast<MemRefType>(init.getType()).getNumElements();
        locals[init] = row;
        continue;
      }
      auto out = bufOf(init);
      if (!out) return false;
      if (out->bcast) return fail("element-wise result into a broadcast buffer");
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

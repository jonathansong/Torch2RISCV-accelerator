// Descriptor-template generation for one dispatch (Codegen.h,
// docs/iree_compiler_plan.md §6.8).
//
// A dispatch function is a short straight-line program over tensors:
// dispatch.tensor.load / store of bindings (with offsets that are constants
// or push constants), linalg.fill, linalg.generic (element-wise or row
// reductions) and a few structured ops. The generator
//   - resolves every binding offset and dynamic size to a constant or a
//     function of one push constant (IREE passes offsets into packed
//     resources and dynamic dimensions as push constants); dynamic ones become
//     register setup entries (BASE r = binding + f(c), PARAM r = f(c)) that
//     the driver evaluates, so the template itself has static offsets;
//   - keeps tensors in the local memories: fp32 / i32 in ACC, i8 in SPAD,
//     row-major with element e in lane e % D of word e / D ("packed"), or one
//     value per word in all lanes ("broadcast": scalars, per-row values);
//   - compiles each linalg.generic body with the VE expression compiler:
//     every VE instruction computes y = FUNC(OP(src1, src2) * A + B) with
//     index modes per source (linear, MOD, DIV), a row reduction, a VALID
//     count and an output conversion (fp32, i32, int8 = round, saturate,
//     NaN -> 0); body operations are folded into these stages in IR order,
//     constants go to A / B / IMM, exp / rsqrt / 1/x to the SFU;
//   - uses dedicated templates for contractions (the int8 linear layer,
//     Templates.cpp).
// Numerics are those of the device (DeviceModel with SfuExact): fp32 ops
// round per operation as the IR's, reductions use the device's order, the
// SFU functions stand for math.exp / math.rsqrt / 1 / x.
#include "Codegen.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <optional>

#include "DescList.h"
#include "Templates.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::iree_compiler::sa {

using IREE::TensorExt::DispatchTensorLoadOp;
using IREE::TensorExt::DispatchTensorStoreOp;
using ::sa::acc;
using ::sa::laddr;
using ::sa::VType;

namespace {

constexpr float NEG0 = -0.0f;

// ---------------------------------------------------------------- scalars
// ((C * mul) >> shift) + add, C = push constant `ord` (or none: add only).
struct Lin {
  int ord = -1;
  int64_t mul = 1;
  int shift = 0;
  int64_t add = 0;
  bool isConst() const { return ord < 0; }
  bool operator<(const Lin &o) const {
    return std::tie(ord, mul, shift, add) < std::tie(o.ord, o.mul, o.shift, o.add);
  }
  bool operator==(const Lin &o) const { return !(*this < o) && !(o < *this); }
};

std::optional<Lin> linOf(Value v) {
  if (auto c = getConstantIntValue(v)) return Lin{-1, 1, 0, *c};
  Operation *op = v.getDefiningOp();
  if (!op) return std::nullopt;
  if (auto load = dyn_cast<IREE::HAL::InterfaceConstantLoadOp>(op))
    return Lin{int(load.getOrdinal().getZExtValue()), 1, 0, 0};
  if (isa<arith::IndexCastUIOp, arith::IndexCastOp, arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp>(op))
    return linOf(op->getOperand(0));
  if (auto a = dyn_cast<IREE::Util::AssumeIntOp>(op)) return linOf(a.getOperand(cast<OpResult>(v).getResultNumber()));
  if (isa<IREE::TensorExt::DispatchWorkloadOrdinalOp>(op)) return linOf(op->getOperand(0));
  if (auto o = dyn_cast<arith::OrIOp>(op)) {
    // lo | (hi << 32): a 64-bit value from two push constants; the device is
    // 32-bit, the high word is 0 (the runtime values here are offsets / lengths)
    for (int i = 0; i < 2; ++i) {
      if (auto sh = op->getOperand(1 - i).getDefiningOp<arith::ShLIOp>()) {
        auto amount = getConstantIntValue(sh.getRhs());
        if (amount && *amount == 32) return linOf(op->getOperand(i));
      }
    }
    return std::nullopt;
  }
  if (auto m = dyn_cast<arith::MulIOp>(op)) {
    for (int i = 0; i < 2; ++i)
      if (auto c = getConstantIntValue(op->getOperand(1 - i)))
        if (auto l = linOf(op->getOperand(i)); l && l->shift == 0) return Lin{l->ord, l->mul * *c, 0, l->add * *c};
    return std::nullopt;
  }
  if (isa<arith::AddIOp>(op)) {
    for (int i = 0; i < 2; ++i)
      if (auto c = getConstantIntValue(op->getOperand(1 - i)))
        if (auto l = linOf(op->getOperand(i))) return Lin{l->ord, l->mul, l->shift, l->add + *c};
    return std::nullopt;
  }
  return std::nullopt;
}

// ---------------------------------------------------------------- local tensors
enum class Layout { Packed, Bcast };

struct Local {
  ::sa::Mem mem = ::sa::MEM_ACC;
  uint32_t word = 0;
  VType vt = ::sa::VT_F32;
  Layout layout = Layout::Packed;
  uint32_t words = 0;          // allocated words
  int64_t n = 0;               // elements (upper bound if dynamic)
  uint32_t rowStride = 0;      // > 0: rows of a dynamic length at this word stride
  int64_t rows = 1;
};

// VE source operand
struct Opd {
  uint32_t la = 0;
  VType vt = ::sa::VT_F32;
  ::sa::VIdx mode = ::sa::IDX_LIN;
  uint32_t period = 0;
  int periodParam = -1;
  bool isImm = false;
  float imm = 0.0f;
};

// a VE instruction being built: y = FUNC(OP(s1', s2) * A + B)
struct Pend {
  Opd s1;
  std::optional<Opd> s2;
  ::sa::VOp op = ::sa::VOP_COPY;
  bool swapneg = false;
  float A = 1.0f, B = NEG0;
  ::sa::VFunc func = ::sa::FUNC_NONE;
  int stage = 0;               // 0 op, 1 A, 2 B, 3 func
  int validParam = -1;         // prefix mask (VALID from a PARAM)
  bool validNegInf = false;    // masked with -inf (only for a max reduction)
};

// value of an SSA value inside a linalg body
struct EV {
  enum Kind { None, Mem, Const, Pending, ToI8, I32OfI8, Index, Mask, ScalarArg, IntExpr, SwapSrc, NegSwap, IdxCond } kind = None;
  Opd opd;                     // Mem
  float c = 0;                 // Const
  int pend = -1;               // Pending (index), ToI8 / I32OfI8 (the pending / value inside)
  std::shared_ptr<EV> inner;   // ToI8, I32OfI8
  int dim = -1;                // Index, Mask
  int arg = -1;                // ScalarArg: body argument number; Mask: the scalar's argument
  arith::CmpIPredicate pred{};
};

// geometry of a generic: loops flattened row-major
struct Geo {
  int64_t n = 0;               // elements (upper bound)
  std::optional<Lin> dynN;     // dynamic element count
  int64_t inner = 0;           // innermost loop size (upper bound)
  std::optional<Lin> dynInner;
  int64_t rows = 1;            // for reductions: output rows
  bool reduction = false;
  bool singleRow = false;      // one row of a row-by-row generic: no ROWLEN
};

// A row of a [H, T] generic with a dynamic T, compiled one row at a time
struct RowSel {
  int64_t row = 0, H = 1;
};

class Gen {
public:
  Gen(FunctionOpInterface func, const CodegenOptions &o) : func(func), opt(o), d(o.d), lay(uint32_t(o.d)) {
    accTop[0] = lay.acc0;
    accTop[1] = lay.acc1;
  }

  bool run(Generated &out, std::string &why);

private:
  FunctionOpInterface func;
  CodegenOptions opt;
  int64_t d;
  ::sa::Layout lay;
  ::sa::DescList dl;
  std::vector<SetupEntry> setup;
  std::string err;
  std::vector<Operation *> handledStores, skipOps;

  // ------------------------------------------------------------ errors
  bool fail(const std::string &m) {
    if (err.empty()) err = m;
    return false;
  }

  // ------------------------------------------------------------ registers
  std::map<std::pair<int, Lin>, int> bases;     // (binding, dynamic offset or const 0) -> BASE
  std::map<Lin, int> params;                     // dynamic value -> PARAM
  // PARAMs set by the register setup table count up from 0, the template's
  // private ones (loop offsets, LDPARAM results) down from 7 (the sa-desc
  // convention: PARAM6 / 7 private first)
  int nextBase = 0, nextParam = 0, nextPrivate = 7;

  int baseFor(int binding, const Lin &off) {
    Lin key = off.isConst() ? Lin{-1, 1, 0, 0} : off;
    auto it = bases.find({binding, key});
    if (it != bases.end()) return it->second;
    if (nextBase >= 16) return fail("more than 16 base registers"), 0;
    int r = nextBase++;
    bases[{binding, key}] = r;
    setup.push_back({SetupEntry::BASE, uint8_t(r), uint8_t(binding), int16_t(key.ord), int32_t(key.mul),
                     uint8_t(key.shift), int32_t(key.add)});
    return r;
  }
  int paramFor(const Lin &v) {
    auto it = params.find(v);
    if (it != params.end()) return it->second;
    if (nextParam > nextPrivate) return fail("more than 8 PARAM registers"), 0;
    int r = nextParam++;
    params[v] = r;
    setup.push_back({SetupEntry::PARAM, uint8_t(r), 0, int16_t(v.ord), int32_t(v.mul), uint8_t(v.shift),
                     int32_t(v.add)});
    return r;
  }
  int privateParam() {
    if (nextPrivate < nextParam) return fail("more than 8 PARAM registers"), 0;
    return nextPrivate--;
  }

  // ------------------------------------------------------------ local memory
  uint32_t accTop[2];
  uint32_t spadTop = 0;
  uint32_t allocAcc(uint32_t words) {
    for (int b = 0; b < 2; ++b) {
      uint32_t end = (b + 1) * lay.cbank;
      if (accTop[b] + words <= end) {
        uint32_t w = accTop[b];
        accTop[b] += words;
        return w;
      }
    }
    fail("ACC full");
    return 0;
  }
  uint32_t allocSpad(uint32_t words) {
    if (spadTop + words > 2 * lay.sbank) return fail("SPAD_A full"), 0;
    uint32_t w = spadTop;
    spadTop += words;
    return w;
  }
  Local newLocal(VType vt, int64_t n, Layout layout = Layout::Packed) {
    Local l;
    l.vt = vt;
    l.layout = layout;
    l.n = n;
    l.words = layout == Layout::Bcast ? uint32_t(n) : uint32_t((n + d - 1) / d);
    l.words = std::max<uint32_t>(l.words, 1);
    if (vt == ::sa::VT_I8) {
      l.mem = ::sa::MEM_SPAD_A;
      l.word = allocSpad(l.words);
    } else {
      l.mem = ::sa::MEM_ACC;
      l.word = allocAcc(l.words);
    }
    return l;
  }
  uint32_t la(const Local &l) const { return laddr(l.mem, l.word); }

  // ------------------------------------------------------------ DDR regions
  struct Region {
    int base = 0;
    uint32_t off = 0;
  };
  // (region, full shape with dynamic dims as -1, dynamic dims, element type) of a subspan
  struct Span {
    Region r;
    SmallVector<int64_t> shape;
    SmallVector<Lin> dyn;
    Type et;
  };
  std::optional<Span> spanOf(Value v) {
    auto sub = v.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!sub) return fail("tensor not from a binding"), std::nullopt;
    Span s;
    int binding = int(sub.getBinding().getZExtValue());
    Lin off{-1, 1, 0, 0};
    if (Value o = sub.getByteOffset()) {
      auto l = linOf(o);
      if (!l) return fail("binding offset is not a constant or a push constant"), std::nullopt;
      off = *l;
    }
    if (off.isConst()) {
      s.r = {baseFor(binding, off), uint32_t(off.add)};
    } else {
      s.r = {baseFor(binding, off), 0};
    }
    auto dt = cast<IREE::TensorExt::DispatchTensorType>(sub.getType());
    auto rt = dt.asRankedTensorType();
    s.et = rt.getElementType();
    for (int64_t x : rt.getShape()) s.shape.push_back(x);
    for (Value dv : sub.getDynamicDims()) {
      auto l = linOf(dv);
      if (!l) return fail("dynamic dimension is not a push constant"), std::nullopt;
      s.dyn.push_back(*l);
    }
    return s;
  }

  static uint32_t esize(Type t) {
    if (t.isF32() || t.isInteger(32)) return 4;
    if (t.isInteger(8)) return 1;
    if (t.isInteger(64)) return 8;
    return 0;
  }
  static std::optional<VType> vtOf(Type t) {
    if (t.isF32()) return ::sa::VT_F32;
    if (t.isInteger(32)) return ::sa::VT_I32;
    if (t.isInteger(8)) return ::sa::VT_I8;
    return std::nullopt;
  }

  // A loaded tensor, not yet in local memory.
  struct Source {
    Region r;                  // start of the loaded slice
    RankedTensorType type;
    int64_t n = 0;             // elements (upper bound)
    std::optional<Lin> dynN;   // dynamic element count
    std::optional<Lin> dynInner;
    int64_t inner = 0;
    std::optional<Lin> dynDim;   // the dynamic dimension itself
    int64_t before = 1, after = 1; // static elements before / after it
  };
  std::map<void *, Source> sources;          // load result -> Source
  std::map<void *, Local> locals;            // tensor value -> local copy (packed)
  std::map<void *, Local> bcastLocals;       // tensor value -> broadcast copy
  std::map<void *, float> fills;             // fill results -> value

  bool analyzeLoad(DispatchTensorLoadOp load) {
    auto span = spanOf(load.getSource());
    if (!span) return false;
    auto rt = cast<RankedTensorType>(load.getType());
    Source s;
    s.type = rt;
    // contiguous slice: offsets only in leading dims with full trailing dims
    SmallVector<int64_t> offs(load.getStaticOffsets()), sizes(load.getStaticSizes()), strides(load.getStaticStrides());
    if (!load.getOffsets().empty()) return fail("dynamic load offsets");
    for (int64_t st : strides)
      if (st != 1) return fail("strided load");
    const SmallVector<int64_t> &full = span->shape;
    if (full.size() != offs.size()) return fail("load rank");
    // byte offset of the slice start: dynamic dimensions only outermost; after
    // the first dimension with an offset or a partial size, all dimensions are
    // full (a contiguous slice)
    bool whole = true;
    for (size_t i = 0; i < full.size(); ++i)
      if (offs[i] != 0 || (full[i] >= 0 && sizes[i] != full[i])) whole = false;
    if (!whole)
      for (size_t i = 1; i < full.size(); ++i)
        if (full[i] < 0) return fail("slice of a tensor with a dynamic inner dimension");
    int64_t lin = 0, mult = 1;
    for (int i = int(full.size()) - 1; i >= 0; --i) {
      lin += offs[i] * mult;
      if (i > 0) mult *= full[i] < 0 ? 0 : full[i];
    }
    bool partial = false;
    for (size_t i = 0; i < full.size(); ++i) {
      bool isFull = full[i] < 0 || sizes[i] == full[i];
      if (partial && (!isFull || offs[i] != 0)) return fail("non-contiguous load");
      if (!isFull || offs[i] != 0) partial = true;
    }
    uint32_t es = esize(rt.getElementType());
    if (!es) return fail("element type");
    s.r = {span->r.base, uint32_t(span->r.off + lin * es)};
    // sizes
    int64_t n = 1;
    std::optional<Lin> dyn;
    auto dynIt = load.getSizes().begin();
    for (size_t i = 0; i < sizes.size(); ++i) {
      int64_t sz = sizes[i];
      if (ShapedType::isDynamic(sz)) {
        auto l = linOf(*dynIt++);
        if (!l || dyn) return fail("more than one dynamic size, or not from a push constant");
        dyn = l;
        sz = opt.maxDynamic;
      }
      n *= sz;
    }
    s.n = n;
    if (dyn) {
      int64_t other = n / opt.maxDynamic;
      s.dynN = Lin{dyn->ord, dyn->mul * other, dyn->shift, dyn->add * other};
      s.dynDim = dyn;
      bool seen = false;
      for (int64_t sz : sizes) {
        if (ShapedType::isDynamic(sz)) seen = true;
        else if (!seen) s.before *= sz;
        else s.after *= sz;
      }
    }
    int64_t last = rt.getRank() ? rt.getShape().back() : 1;
    s.inner = ShapedType::isDynamic(last) ? opt.maxDynamic : last;
    if (rt.getRank() && ShapedType::isDynamic(last)) s.dynInner = dyn;
    sources[load.getResult().getAsOpaquePointer()] = s;
    return true;
  }

  // Row loops: rows of a dynamic length T (bytes per row = f(T)), row r at
  // DDR offset r * bytes: a PARAM accumulates the offset on the device (SETREG
  // PARAM += PARAM), so each DMA has two dynamic fields (address, bytes).
  int rowAcc = -1;
  int rowAccParam() {
    if (rowAcc < 0) rowAcc = privateParam();
    return rowAcc;
  }
  template <typename F>
  void rowLoop(int64_t H, int bytesParam, F dma) {
    int acc = rowAccParam();
    dl.setreg({{::sa::DescList::REG_PARAM + uint32_t(acc), 0}});
    for (int64_t r = 0; r < H; ++r) {
      dma(r, std::vector<::sa::Dyn>{{::sa::DYN_DMA_DDR, uint32_t(acc), true},
                                    {::sa::DYN_DMA_ROW_BYTES, uint32_t(bytesParam), false}});
      if (r + 1 < H)
        dl.setreg({{::sa::DescList::REG_PARAM + uint32_t(acc), 0}}, {{::sa::DYN_SETREG_V0, uint32_t(bytesParam), false}},
                  1);
    }
  }
  bool stridedSource(const Source &s) const { return s.dynDim && s.after == 1 && s.before > 1 && s.before <= 16; }

  // LD of a source into a packed local
  std::optional<Local> materialize(Value v) {
    void *key = v.getAsOpaquePointer();
    if (auto it = locals.find(key); it != locals.end()) return it->second;
    auto sit = sources.find(key);
    if (sit == sources.end()) return fail("tensor is neither loaded nor computed"), std::nullopt;
    Source &s = sit->second;
    Type et = s.type.getElementType();
    auto vt = vtOf(et);
    int64_t n = s.n;
    if (et.isInteger(64)) {
      vt = ::sa::VT_I32;                 // (low word, high word) in two lanes
      n = 2 * s.n;
    }
    if (!vt) return fail("element type of a loaded tensor"), std::nullopt;
    uint32_t es = esize(et);
    if (stridedSource(s)) {
      // H rows of T: row r at word r * stride (the largest T)
      uint32_t stride = uint32_t((opt.maxDynamic * es + 4 * d - 1) / (4 * d));
      if (*vt == ::sa::VT_I8) stride = uint32_t((opt.maxDynamic + d - 1) / d);
      Local l = newLocal(*vt, int64_t(stride) * s.before * (*vt == ::sa::VT_I8 ? d : d));
      l.rowStride = stride;
      l.rows = s.before;
      Lin b = *s.dynDim;
      b.mul *= es;
      b.add *= es;
      int bp = paramFor(b);
      uint32_t maxBytes = uint32_t(opt.maxDynamic * es);
      if (s.r.off % 8) return fail("load not 8-byte aligned"), std::nullopt;
      rowLoop(s.before, bp, [&](int64_t r, std::vector<::sa::Dyn> dy) {
        dl.ld(s.r.off, la(l) + uint32_t(r) * stride, 1, maxBytes, maxBytes, s.r.base, 0, false, dy);
      });
      locals[key] = l;
      return l;
    }
    Local l = newLocal(*vt, n);
    uint32_t bytes = uint32_t((s.n * es + 7) / 8 * 8);
    if (s.r.off % 8) return fail("load not 8-byte aligned"), std::nullopt;
    std::vector<::sa::Dyn> dyn;
    if (s.dynN) {
      Lin b = *s.dynN;
      b.mul *= es;
      b.add *= es;
      dyn.push_back({::sa::DYN_DMA_ROW_BYTES, uint32_t(paramFor(b)), false});
    }
    if (bytes > 65535) return fail("load larger than 64 KB"), std::nullopt;
    dl.ld(s.r.off, la(l), 1, bytes, bytes, s.r.base, 0, false, dyn);
    locals[key] = l;
    return l;
  }

  // One word with the scalar in all lanes (a rank-0 fp32 source or local).
  std::optional<Local> scalarBcast(Value v) {
    void *key = v.getAsOpaquePointer();
    if (auto it = bcastLocals.find(key); it != bcastLocals.end()) return it->second;
    auto l = materialize(v);
    if (!l) return std::nullopt;
    if (l->layout == Layout::Bcast) return l;
    if (l->vt != ::sa::VT_F32) return fail("broadcast of a non-fp32 scalar"), std::nullopt;
    Local b = newLocal(::sa::VT_F32, 1, Layout::Bcast);
    ::sa::VeFp fp;
    fp.reduce = ::sa::RED_MAX;             // a max over the first element: the value, in every lane
    fp.valid = 1;
    fp.B = NEG0;
    dl.veFp(la(*l), 0, la(b), uint32_t(d), ::sa::VOP_COPY, ::sa::vtypes(::sa::VT_F32, ::sa::VT_F32), 0, fp);
    bcastLocals[key] = b;
    return b;
  }

  // One word per element (all lanes) of a packed vector of n <= a few words:
  // replicate each word D times, then D x D transposes.
  std::optional<Local> perElementBcast(Value v, int64_t n) {
    void *key = v.getAsOpaquePointer();
    if (auto it = bcastLocals.find(key); it != bcastLocals.end()) return it->second;
    auto l = materialize(v);
    if (!l) return std::nullopt;
    if (l->layout == Layout::Bcast) return l;
    if (l->vt != ::sa::VT_F32) return fail("per-row values must be fp32"), std::nullopt;
    uint32_t w = uint32_t((n + d - 1) / d);
    Local rep = newLocal(::sa::VT_F32, int64_t(w) * d * d);
    ::sa::VeFp fp;
    fp.m1 = ::sa::IDX_DIV;
    fp.p1 = uint32_t(d);
    dl.veFp(la(*l), 0, la(rep), uint32_t(w * d * d), ::sa::VOP_COPY, ::sa::vtypes(::sa::VT_F32, ::sa::VT_F32), 0,
            fp);
    Local b = newLocal(::sa::VT_F32, int64_t(w) * d, Layout::Bcast);
    dl.transpose(la(rep), la(b), uint32_t(w * d * d), ::sa::vtypes(::sa::VT_F32, ::sa::VT_F32), 1);
    bcastLocals[key] = b;
    return b;
  }

  // Broadcast words (one per element) -> packed.
  Local packBcast(const Local &b, int64_t n) {
    uint32_t w = uint32_t((n + d - 1) / d);
    Local p = newLocal(::sa::VT_F32, n);
    // the transpose reads w * D words; words past n hold whatever was there
    dl.transpose(la(b), la(p), uint32_t(w * d * d), ::sa::vtypes(::sa::VT_F32, ::sa::VT_F32), 1);
    return p;
  }

  // ------------------------------------------------------------ stores
  bool store(DispatchTensorStoreOp st) {
    auto span = spanOf(st.getTarget());
    if (!span) return false;
    Value v = st.getValue();
    auto it = locals.find(v.getAsOpaquePointer());
    if (it == locals.end()) return fail("stored value not computed locally");
    Local l = it->second;
    auto rt = cast<RankedTensorType>(v.getType());
    for (int64_t o : st.getStaticOffsets())
      if (o != 0) return fail("store with an offset");
    uint32_t es = esize(rt.getElementType());
    int64_t n = 1;
    std::optional<Lin> dyn;
    auto dynIt = st.getSizes().begin();
    for (int64_t sz : st.getStaticSizes()) {
      if (ShapedType::isDynamic(sz)) {
        auto lv = linOf(*dynIt++);
        if (!lv || dyn) return fail("store with a non push-constant or second dynamic size");
        dyn = lv;
        sz = opt.maxDynamic;
      }
      n *= sz;
    }
    if (l.rowStride) {
      if (!dyn) return fail("row-strided value stored with a static size");
      Lin b = *dyn;
      b.mul *= es;
      b.add *= es;
      int bp = paramFor(b);
      uint32_t maxBytes = uint32_t(opt.maxDynamic * es);
      if (span->r.off % 8) return fail("store not 8-byte aligned");
      rowLoop(l.rows, bp, [&](int64_t r, std::vector<::sa::Dyn> dy) {
        dl.st(span->r.off, la(l) + uint32_t(r) * l.rowStride, 1, maxBytes, maxBytes, span->r.base, false, dy);
      });
      return true;
    }
    if (l.layout == Layout::Bcast && n > 1) l = packBcast(l, n);
    uint32_t bytes = uint32_t((n * es + 7) / 8 * 8);
    std::vector<::sa::Dyn> dy;
    if (dyn) {
      int64_t other = n / opt.maxDynamic;
      Lin b{dyn->ord, dyn->mul * other * es, dyn->shift, dyn->add * other * es};
      dy.push_back({::sa::DYN_DMA_ROW_BYTES, uint32_t(paramFor(b)), false});
    }
    if (span->r.off % 8) return fail("store not 8-byte aligned");
    if (bytes > 65535) return fail("store larger than 64 KB");
    dl.st(span->r.off, la(l), 1, bytes, bytes, span->r.base, false, dy);
    return true;
  }

  // ------------------------------------------------------------ VE emission
  uint32_t lenOf(const Geo &g, std::vector<::sa::Dyn> &dyn) {
    if (g.dynN) dyn.push_back({::sa::DYN_VE_LEN, uint32_t(paramFor(*g.dynN)), false});
    return uint32_t((g.n + d - 1) / d * d);
  }

  void emitPend(const Pend &p, uint32_t dst, VType out, const Geo &g, ::sa::VRed red = ::sa::RED_NONE) {
    ::sa::VeFp fp;
    fp.m1 = p.s1.mode;
    fp.p1 = p.s1.period;
    fp.swapneg = p.swapneg;
    fp.A = p.A;
    fp.B = p.B;
    fp.func = p.func;
    fp.reduce = red;
    std::vector<::sa::Dyn> dyn;
    uint32_t len = lenOf(g, dyn);
    if (p.s1.periodParam >= 0) dyn.push_back({::sa::DYN_VE_P1, uint32_t(p.s1.periodParam), false});
    uint32_t src2 = 0, period = 0;
    if (p.s2) {
      if (p.s2->isImm) {
        fp.m2 = ::sa::IDX_IMM;
        fp.imm = p.s2->imm;
      } else {
        fp.m2 = p.s2->mode;
        src2 = p.s2->la;
        period = p.s2->period;
        if (p.s2->periodParam >= 0) dyn.push_back({::sa::DYN_VE_PERIOD, uint32_t(p.s2->periodParam), false});
        if (p.s2->vt != p.s1.vt) fp.t2 = int(p.s2->vt);
      }
    }
    if (!g.singleRow && (red != ::sa::RED_NONE || p.validParam >= 0)) {
      // rows of `inner` elements: ROWLEN words per row
      if (g.dynInner) {
        Lin rl = *g.dynInner;
        rl.shift += int(std::log2(double(d)));
        dyn.push_back({::sa::DYN_VE_ROWLEN, uint32_t(paramFor(rl)), false});
      }
      fp.rowlen = uint32_t(g.inner / d);
    }
    if (p.validParam >= 0) dyn.push_back({::sa::DYN_VE_VALID, uint32_t(p.validParam), false});
    if (dyn.size() > 2) {
      fail("VE instruction needs more than two dynamic fields");
      return;
    }
    dl.veFp(p.s1.la, src2, dst, len, p.op, ::sa::vtypes(p.s1.vt, out), period, fp, false, dyn);
  }

  // ------------------------------------------------------------ linalg.generic
  bool genericOp(linalg::GenericOp g, const RowSel *sel = nullptr);
  std::map<void *, Local> rowOutputs;         // results of row-by-row generics (shared by the rows)
  std::map<std::pair<void *, int>, int> validParams;   // prefix-mask VALID counts already in a PARAM
  bool qlinear(linalg::GenericOp con, linalg::GenericOp epi);
  bool attention(linalg::BatchMatmulOp bmm);
  bool scatter(IREE::LinalgExt::ScatterOp sc);
  bool intScalar(linalg::GenericOp g);
};

// =============================================================== helpers
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

// The to_i8 chain of qllama.to_i8 (clamp(round(nan_to_num(x)), -127, 127).to(i8))
// ending in |fptosi|: returns x, and the chain's operations in |chain|.
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
// point idx, with the scalar block argument(s) = sym. std::nullopt: not an
// index computation.
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

// Every point of an iteration space (row-major), for ranges with a small product.
SmallVector<SmallVector<int64_t>> points(ArrayRef<int64_t> ranges) {
  SmallVector<SmallVector<int64_t>> out;
  int64_t n = 1;
  for (int64_t r : ranges) n *= r;
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

// =============================================================== the generic compiler
bool Gen::genericOp(linalg::GenericOp g, const RowSel *sel) {
  // iteration space
  auto iters = g.getIteratorTypesArray();
  int nloops = int(iters.size());
  SmallVector<AffineMap> maps = g.getIndexingMapsArray();
  int nin = g.getNumDpsInputs();
  // loop ranges (upper bounds) and the dynamic one
  SmallVector<int64_t> ranges(nloops, -1);
  SmallVector<std::optional<Lin>> dynRange(nloops);
  for (int i = 0; i < int(g->getNumOperands()); ++i) {
    Value v = g->getOperand(i);
    auto rt = dyn_cast<RankedTensorType>(v.getType());
    if (!rt) continue;
    AffineMap m = maps[i];
    for (unsigned r = 0; r < m.getNumResults(); ++r) {
      auto de = dyn_cast<AffineDimExpr>(m.getResult(r));
      if (!de) continue;
      int64_t sz = rt.getDimSize(r);
      int L = int(de.getPosition());
      if (!ShapedType::isDynamic(sz)) {
        ranges[L] = sz;
      } else if (ranges[L] < 0) {
        ranges[L] = opt.maxDynamic;
        // the dynamic size: from the loaded source or the tensor.empty
        auto sit = sources.find(v.getAsOpaquePointer());
        if (sit != sources.end() && sit->second.dynInner) dynRange[L] = sit->second.dynInner;
        else if (auto e = v.getDefiningOp<tensor::EmptyOp>()) {
          if (auto l = linOf(e.getDynamicSizes()[0])) dynRange[L] = l;
        } else if (auto f = v.getDefiningOp<linalg::FillOp>()) {
          if (auto e2 = f.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>())
            if (auto l = linOf(e2.getDynamicSizes()[0])) dynRange[L] = l;
        }
        if (!dynRange[L] && sit != sources.end() && sit->second.dynN) {
          // a 1-D dynamic source
          dynRange[L] = sit->second.dynN;
        }
      }
    }
  }
  for (int L = 0; L < nloops; ++L)
    if (ranges[L] < 0) return fail("loop range not given by an operand");
  SmallVector<int> red;
  for (int L = 0; L < nloops; ++L)
    if (iters[L] == utils::IteratorType::reduction) red.push_back(L);
  if (red.size() > 1 || (!red.empty() && red[0] != nloops - 1)) return fail("reduction not over the innermost loop");
  for (int L = 0; L < nloops - 1; ++L)
    if (dynRange[L]) return fail("dynamic loop other than the innermost");
  Geo geo;
  geo.n = 1;
  for (int64_t r : ranges) geo.n *= r;
  geo.inner = nloops ? ranges.back() : 1;
  if (nloops && dynRange.back()) {
    geo.dynInner = dynRange.back();
    int64_t other = geo.n / geo.inner;
    geo.dynN = Lin{geo.dynInner->ord, geo.dynInner->mul * other, geo.dynInner->shift, geo.dynInner->add * other};
  }
  geo.reduction = !red.empty();
  geo.rows = geo.reduction ? geo.n / geo.inner : 1;
  if (nloops > 1 && geo.inner % d) return fail("innermost size not a multiple of D");
  if (nloops > 0 && geo.n > 1 && nloops == 1 && geo.n % d && !geo.reduction && geo.n > d)
    return fail("1-D size not a multiple of D");
  // [H, T] with a dynamic T: one row at a time (each VE then needs only a
  // dynamic LEN, and VALID for a mask)
  if (!sel && nloops == 2 && geo.dynInner) {
    if (ranges[0] > 16) return fail("more than 16 rows of a dynamic length");
    for (int64_t r = 0; r < ranges[0]; ++r) {
      RowSel rs{r, ranges[0]};
      if (!genericOp(g, &rs)) return false;
    }
    for (unsigned i = 0; i < g.getNumResults(); ++i) {
      auto it = rowOutputs.find(g.getResult(i).getAsOpaquePointer());
      if (it == rowOutputs.end()) return fail("row-by-row generic without its result");
      locals[g.getResult(i).getAsOpaquePointer()] = it->second;
    }
    return err.empty();
  }
  if (sel) {
    geo.n = geo.inner;
    geo.dynN = geo.dynInner;
    geo.rows = 1;
    geo.singleRow = true;
  }
  uint32_t wordsPerRow = uint32_t(geo.inner / d);
  int rowParam = -1;
  if (geo.dynInner && !sel) {
    Lin rl = *geo.dynInner;
    rl.shift += int(std::log2(double(d)));
    rowParam = paramFor(rl);
  }

  // inputs -> operands
  Block &body = g.getRegion().front();
  std::map<void *, EV> ev;
  auto evOf = [&](Value v) -> EV & {
    auto it = ev.find(v.getAsOpaquePointer());
    if (it == ev.end()) {
      EV e;
      if (auto f = constF32(v)) {        // a constant defined outside the body
        e.kind = EV::Const;
        e.c = *f;
      }
      it = ev.emplace(v.getAsOpaquePointer(), e).first;
    }
    return it->second;
  };
  for (int i = 0; i < nin; ++i) {
    Value in = g.getDpsInputs()[i];
    AffineMap m = maps[i];
    auto rt = dyn_cast<RankedTensorType>(in.getType());
    EV e;
    if (!rt) return fail("scalar (non-tensor) generic input");
    Type et = rt.getElementType();
    if (et.isInteger(64)) {
      // i64 scalars: indices / positions (masks, gathers), used through LDPARAM
      if (rt.getRank() != 0) return fail("i64 tensor input");
      e.kind = EV::ScalarArg;
      e.arg = i;
      evOf(body.getArgument(i)) = e;
      continue;
    }
    if (m.getNumResults() == 0) {
      // scalar broadcast
      auto b = scalarBcast(in);
      if (!b) return false;
      e.kind = EV::Mem;
      e.opd = {la(*b), ::sa::VT_F32, ::sa::IDX_DIV, 0xFFFF};
    } else if (m.isIdentity() || (m.getNumResults() == unsigned(nloops) && m.isMinorIdentity())) {
      auto l = materialize(in);
      if (!l) return false;
      if (l->layout != Layout::Packed) {
        // a per-row result (one word per element) used element-wise: pack it
        if (sel) return fail("broadcast-layout operand in a row-by-row generic");
        void *key = in.getAsOpaquePointer();
        Local p = packBcast(*l, rt.getNumElements());
        locals[key] = p;
        bcastLocals[key] = *l;
        l = p;
      }
      e.kind = EV::Mem;
      e.opd = {la(*l), l->vt, ::sa::IDX_LIN, 0};
      if (sel) {
        if (!l->rowStride) return fail("row-by-row operand not in the row layout");
        e.opd.la += uint32_t(sel->row) * l->rowStride;
      }
    } else if (nloops == 2 && m.getNumResults() == 1 && isa<AffineDimExpr>(m.getResult(0)) &&
               cast<AffineDimExpr>(m.getResult(0)).getPosition() == 0) {
      // one value per row
      auto b = perElementBcast(in, ranges[0]);
      if (!b) return false;
      e.kind = EV::Mem;
      e.opd = {la(*b), ::sa::VT_F32, ::sa::IDX_DIV, wordsPerRow, rowParam};
      if (sel) e.opd = {la(*b) + uint32_t(sel->row), ::sa::VT_F32, ::sa::IDX_DIV, 0xFFFF};
    } else if (nloops == 2 && m.getNumResults() == 1 && isa<AffineDimExpr>(m.getResult(0)) &&
               cast<AffineDimExpr>(m.getResult(0)).getPosition() == 1) {
      // the same vector for every row
      auto l = materialize(in);
      if (!l) return false;
      e.kind = EV::Mem;
      e.opd = {la(*l), l->vt, ::sa::IDX_MOD, wordsPerRow, rowParam};
      if (sel) e.opd = {la(*l), l->vt, ::sa::IDX_LIN, 0};
    } else {
      return fail("unsupported indexing map of a generic input");
    }
    evOf(body.getArgument(i)) = e;
  }
  // outputs: reduction init must be the identity of the combiner
  Value outInit = g.getDpsInits()[0];

  SmallVector<Operation *> skip;
  // gathers: tensor.extract from a tensor defined outside the body
  auto domain = points(ranges);
  if (domain.size() > 4096) domain.clear();          // only small spaces are evaluated point by point
  auto flatIndex = [&](tensor::ExtractOp ex, ArrayRef<int64_t> pt, int64_t sym) -> std::optional<int64_t> {
    auto rt = cast<RankedTensorType>(ex.getTensor().getType());
    int64_t f = 0;
    for (unsigned i = 0; i < ex.getIndices().size(); ++i) {
      auto v = evalInt(ex.getIndices()[i], pt, sym);
      if (!v) return std::nullopt;
      f = f * rt.getDimSize(i) + *v;
    }
    return f;
  };
  int scalarArgNo = -1;
  for (int i = 0; i < nin; ++i)
    if (auto rt = dyn_cast<RankedTensorType>(g.getDpsInputs()[i].getType());
        rt && rt.getRank() == 0 && rt.getElementType().isInteger(64))
      scalarArgNo = i;
  for (tensor::ExtractOp ex : body.getOps<tensor::ExtractOp>()) {
    Value t = ex.getTensor();
    auto src = sources.find(t.getAsOpaquePointer());
    if (src == sources.end()) return fail("extract from a tensor that is not loaded");
    if (domain.empty() && nloops) return fail("gather over a large iteration space");
    SmallVector<SmallVector<int64_t>> dom = nloops ? domain : SmallVector<SmallVector<int64_t>>{{}};
    // f(point, sym) = sym * C + e(point)?
    bool usesSym = false;
    for (Value iv : ex.getIndices()) {
      std::function<bool(Value)> hasArg = [&](Value v) -> bool {
        if (isa<BlockArgument>(v)) return true;
        Operation *o = v.getDefiningOp();
        if (!o || o->getBlock() != &body) return false;
        return llvm::any_of(o->getOperands(), hasArg);
      };
      usesSym |= hasArg(iv);
    }
    Type et = src->second.type.getElementType();
    auto vt = vtOf(et);
    uint32_t es = esize(et);
    if (!vt) return fail("gather element type");
    EV e;
    if (usesSym) {
      if (scalarArgNo < 0) return fail("gather index from a non-scalar input");
      std::optional<int64_t> C;
      bool rowGather = true;
      for (size_t pi = 0; pi < dom.size() && rowGather; ++pi) {
        auto f0 = flatIndex(ex, dom[pi], 0), f1 = flatIndex(ex, dom[pi], 1), f7 = flatIndex(ex, dom[pi], 7);
        if (!f0 || !f1 || !f7 || *f0 != int64_t(pi) || *f7 - *f0 != 7 * (*f1 - *f0)) rowGather = false;
        else if (!C) C = *f1 - *f0;
        else if (*C != *f1 - *f0) rowGather = false;
      }
      if (!rowGather || !C) return fail("gather index is not row * C + iteration index");
      auto ssrc = sources.find(g.getDpsInputs()[scalarArgNo].getAsOpaquePointer());
      if (ssrc == sources.end()) return fail("gather index not loaded from a binding");
      int p = privateParam();
      if (*C * es > 0xFFFF) return fail("gather row larger than 64 KB");
      dl.ldparam(ssrc->second.r.off, uint32_t(p), uint32_t(*C * es), 0, ssrc->second.r.base);
      if (nloops == 0) {
        // one element: its bits into a PARAM, then a word of it (1.0 * v + -0 = v exactly)
        int p2 = privateParam();
        dl.ldparam(src->second.r.off, uint32_t(p2), 1, 0, src->second.r.base, false,
                   {{::sa::DYN_LDPARAM_ADDR, uint32_t(p), true}});
        if (!vt || *vt != ::sa::VT_F32) return fail("scalar gather of a non-fp32 value");
        Local ones = newLocal(::sa::VT_F32, 1, Layout::Bcast);
        ::sa::VeFp z;                      // relu(x * 0) = +0 for the (finite, integer-typed) source
        z.A = 0.0f;
        z.B = 1.0f;                        // then + 1
        dl.veFp(la(ones), 0, la(ones), uint32_t(d), ::sa::VOP_COPY, ::sa::vtypes(::sa::VT_I32, ::sa::VT_F32), 0, z);
        Local out = newLocal(::sa::VT_F32, 1, Layout::Bcast);
        ::sa::VeFp y;
        y.B = NEG0;
        dl.veFp(la(ones), 0, la(out), uint32_t(d), ::sa::VOP_COPY, ::sa::vtypes(::sa::VT_F32, ::sa::VT_F32), 0, y,
                false, {{::sa::DYN_VE_A, uint32_t(p2), false}});
        e.kind = EV::Mem;
        e.opd = {la(out), ::sa::VT_F32, ::sa::IDX_DIV, 0xFFFF};
      } else {
        Local row = newLocal(*vt, int64_t(dom.size()));
        uint32_t bytes = uint32_t((dom.size() * es + 7) / 8 * 8);
        if (src->second.r.off % 8 || (*C * es) % 8) return fail("gathered rows not 8-byte aligned");
        dl.ld(src->second.r.off, la(row), 1, bytes, bytes, src->second.r.base, 0, false,
              {{::sa::DYN_DMA_DDR, uint32_t(p), true}});
        e.kind = EV::Mem;
        e.opd = {la(row), *vt, ::sa::IDX_LIN, 0};
        locals[ex.getResult().getAsOpaquePointer()] = row;    // an extracted row may be yielded as is
      }
    } else {
      // a fixed permutation of a (small) loaded tensor: the pair swap
      bool swap = !dom.empty();
      for (size_t pi = 0; pi < dom.size() && swap; ++pi) {
        auto f = flatIndex(ex, dom[pi], 0);
        if (!f || *f != int64_t(pi ^ 1)) swap = false;
      }
      if (!swap) return fail("gather with a fixed index pattern other than the pair swap");
      auto l = materialize(t);
      if (!l) return false;
      e.kind = EV::SwapSrc;
      e.opd = {la(*l), l->vt, ::sa::IDX_LIN, 0};
    }
    evOf(ex.getResult()) = e;
    skip.push_back(ex);
  }

  // to_i8 chains
  std::map<void *, Value> toI8Src;
  g.walk([&](arith::FPToSIOp f) {
    SmallVector<Operation *> chain;
    Value x = matchToI8(f, chain);
    if (x) {
      toI8Src[f.getResult().getAsOpaquePointer()] = x;
      for (Operation *o : chain)
        if (o != f.getOperation()) skip.push_back(o);
    }
  });

  std::vector<Pend> pends;
  auto mem = [&](EV &x) -> std::optional<Opd> {
    if (x.kind == EV::Mem) return x.opd;
    if (x.kind == EV::Pending) {
      Local t = newLocal(::sa::VT_F32, geo.reduction ? geo.n : geo.n);
      emitPend(pends[x.pend], la(t), ::sa::VT_F32, geo);
      Opd o{la(t), ::sa::VT_F32, ::sa::IDX_LIN, 0};
      x.kind = EV::Mem;
      x.opd = o;
      return o;
    }
    fail("operand is not a vector value");
    return std::nullopt;
  };
  auto newPend = [&](const Pend &p) {
    pends.push_back(p);
    EV e;
    e.kind = EV::Pending;
    e.pend = int(pends.size()) - 1;
    return e;
  };
  // a pending value that can take a further stage >= s
  auto pendAt = [&](EV &x, int s) -> Pend * {
    if (x.kind == EV::Pending && pends[x.pend].stage < s && pends[x.pend].validParam < 0) return &pends[x.pend];
    return nullptr;
  };
  auto copyPend = [&](EV &x) -> std::optional<EV> {
    auto o = mem(x);
    if (!o) return std::nullopt;
    Pend p;
    p.s1 = *o;
    return newPend(p);
  };

  Operation *yield = body.getTerminator();
  for (Operation &opRef : body.without_terminator()) {
    Operation *op = &opRef;
    if (llvm::is_contained(skip, op)) continue;
    auto res = op->getNumResults() ? op->getResult(0) : Value();
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto f = constF32(c)) {
        EV e;
        e.kind = EV::Const;
        e.c = *f;
        evOf(res) = e;
      }
      continue;   // integer constants are read where used
    }
    if (auto ix = dyn_cast<linalg::IndexOp>(op)) {
      EV e;
      e.kind = EV::Index;
      e.dim = int(ix.getDim());
      evOf(res) = e;
      continue;
    }
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(op)) {
      EV &x = evOf(op->getOperand(0));
      if (x.kind == EV::Index || x.kind == EV::ScalarArg) {
        evOf(res) = x;
      } else {
        EV e;
        e.kind = EV::IntExpr;
        evOf(res) = e;
      }
      continue;
    }
    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp, arith::RemSIOp>(op) &&
        res.getType().isIndex()) {
      EV e;                                           // index arithmetic: evaluated where used
      e.kind = EV::IntExpr;
      evOf(res) = e;
      continue;
    }
    if (auto ci = dyn_cast<arith::CmpIOp>(op)) {
      EV &a = evOf(ci.getLhs()), &b = evOf(ci.getRhs());
      if (a.kind == EV::IntExpr || b.kind == EV::IntExpr || (a.kind == EV::Index && b.kind != EV::ScalarArg)) {
        EV e;                                       // a condition on indices only (evaluated where used)
        e.kind = EV::IdxCond;
        evOf(res) = e;
        continue;
      }
      if (a.kind != EV::Index || b.kind != EV::ScalarArg || a.dim != nloops - 1 ||
          (ci.getPredicate() != arith::CmpIPredicate::sle && ci.getPredicate() != arith::CmpIPredicate::slt))
        return fail("comparison other than a prefix mask (innermost index <= scalar)");
      EV e;
      e.kind = EV::Mask;
      e.dim = a.dim;
      e.arg = b.arg;
      e.pred = ci.getPredicate();
      evOf(res) = e;
      continue;
    }
    if (auto s = dyn_cast<arith::SelectOp>(op); s && evOf(s.getCondition()).kind == EV::IdxCond) {
      // select(cond(i), -swap(x), swap(x)) with cond(i) = (i even): the VE's SWAPNEG
      EV &tv = evOf(s.getTrueValue()), &fv = evOf(s.getFalseValue());
      if (tv.kind != EV::NegSwap || fv.kind != EV::SwapSrc || tv.opd.la != fv.opd.la || nloops != 1)
        return fail("select on an index condition other than the pair swap with negation");
      for (size_t pi = 0; pi < domain.size(); ++pi) {
        auto c = evalInt(s.getCondition(), domain[pi], 0);
        if (!c || bool(*c) != (pi % 2 == 0)) return fail("pair swap with a different sign pattern");
      }
      Pend p;
      p.s1 = fv.opd;
      p.swapneg = true;
      evOf(res) = newPend(p);
      continue;
    }
    if (auto s = dyn_cast<arith::SelectOp>(op)) {
      EV &c = evOf(s.getCondition());
      if (c.kind != EV::Mask) return fail("select other than a prefix mask");
      bool negInf = isF32Const(s.getFalseValue(), -INFINITY);
      if (!isF32Const(s.getFalseValue(), 0.0f) && !negInf) return fail("masked value other than 0 / -inf");
      // VALID = pos + 1 (sle) or pos (slt), pos = the scalar input (i64, low word)
      auto span = sources.find(g.getDpsInputs()[c.arg].getAsOpaquePointer());
      if (span == sources.end()) return fail("mask scalar not loaded from a binding");
      // one PARAM per (scalar, predicate), loaded once (also for the rows of a row-by-row generic)
      auto vkey = std::make_pair(span->first, int(c.pred));
      int p;
      if (auto vi = validParams.find(vkey); vi != validParams.end()) {
        p = vi->second;
      } else {
        p = privateParam();
        validParams[vkey] = p;
        dl.ldparam(span->second.r.off, uint32_t(p), 1, c.pred == arith::CmpIPredicate::sle ? 1 : 0,
                   span->second.r.base);
      }
      EV x = evOf(s.getTrueValue());
      std::optional<EV> pe;
      if (x.kind == EV::Pending && pends[x.pend].validParam < 0) pe = x;
      else pe = copyPend(x);
      if (!pe) return false;
      pends[pe->pend].validParam = p;
      pends[pe->pend].validNegInf = negInf;
      evOf(res) = *pe;
      continue;
    }
    if (auto f = dyn_cast<arith::FPToSIOp>(op)) {
      auto it = toI8Src.find(res.getAsOpaquePointer());
      if (it == toI8Src.end()) return fail("fptosi other than to_i8");
      EV e;
      e.kind = EV::ToI8;
      e.inner = std::make_shared<EV>(evOf(it->second));
      evOf(res) = e;
      continue;
    }
    if (auto ex = dyn_cast<arith::ExtSIOp>(op)) {
      EV &x = evOf(ex.getIn());
      if (x.kind != EV::ToI8 || !ex.getType().isInteger(32)) return fail("extsi other than i8 -> i32 of to_i8");
      EV e;
      e.kind = EV::I32OfI8;
      e.inner = std::make_shared<EV>(x);
      evOf(res) = e;
      continue;
    }
    if (auto sf = dyn_cast<arith::SIToFPOp>(op)) {
      EV &x = evOf(sf.getIn());
      if (x.kind != EV::Mem || x.opd.vt == ::sa::VT_F32) return fail("sitofp of a computed value");
      evOf(res) = x;   // the VE reads integer sources as fp32
      continue;
    }
    // unary fp: negf, absf, exp, rsqrt
    auto unaryFunc = [&](::sa::VFunc fn) -> bool {
      EV x = evOf(op->getOperand(0));
      if (Pend *p = pendAt(x, 3)) {
        p->func = fn;
        p->stage = 3;
        evOf(res) = x;
        return true;
      }
      auto pe = copyPend(x);
      if (!pe) return false;
      pends[pe->pend].func = fn;
      pends[pe->pend].stage = 3;
      evOf(res) = *pe;
      return true;
    };
    if (isa<math::ExpOp>(op)) {
      if (!unaryFunc(::sa::FUNC_EXP)) return false;
      continue;
    }
    if (isa<math::RsqrtOp>(op)) {
      if (!unaryFunc(::sa::FUNC_RSQRT)) return false;
      continue;
    }
    if (isa<math::AbsFOp>(op)) {
      if (!unaryFunc(::sa::FUNC_ABS)) return false;
      continue;
    }
    // multiply by a constant: A (exact for -1: negation)
    auto mulConst = [&](EV x, float c) -> bool {
      if (Pend *p = pendAt(x, 1)) {
        p->A = c;
        p->stage = 1;
        evOf(res) = x;
        return true;
      }
      auto pe = copyPend(x);
      if (!pe) return false;
      pends[pe->pend].A = c;
      pends[pe->pend].stage = 1;
      evOf(res) = *pe;
      return true;
    };
    auto addConst = [&](EV x, float c) -> bool {
      if (Pend *p = pendAt(x, 2)) {
        p->B = c;
        p->stage = 2;
        evOf(res) = x;
        return true;
      }
      auto pe = copyPend(x);
      if (!pe) return false;
      pends[pe->pend].B = c;
      pends[pe->pend].stage = 2;
      evOf(res) = *pe;
      return true;
    };
    if (isa<arith::NegFOp>(op)) {
      if (evOf(op->getOperand(0)).kind == EV::SwapSrc) {
        EV e = evOf(op->getOperand(0));
        e.kind = EV::NegSwap;
        evOf(res) = e;
        continue;
      }
      if (!mulConst(evOf(op->getOperand(0)), -1.0f)) return false;
      continue;
    }
    if (auto dv = dyn_cast<arith::DivFOp>(op)) {
      if (!isF32Const(dv.getLhs(), 1.0f)) return fail("division other than 1 / x");
      Value x = dv.getRhs();
      EV xv = evOf(x);
      if (Pend *p = pendAt(xv, 3)) {
        p->func = ::sa::FUNC_RECIP;
        p->stage = 3;
        evOf(res) = xv;
        continue;
      }
      auto pe = copyPend(xv);
      if (!pe) return false;
      pends[pe->pend].func = ::sa::FUNC_RECIP;
      pends[pe->pend].stage = 3;
      evOf(res) = *pe;
      continue;
    }
    if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::MaximumFOp, arith::MinimumFOp>(op)) {
      Value lhs = op->getOperand(0), rhs = op->getOperand(1);
      // the reduction combiner is handled at the yield
      if (geo.reduction && (lhs == body.getArguments().back() || rhs == body.getArguments().back())) continue;
      EV a = evOf(lhs), b = evOf(rhs);
      bool comm = isa<arith::AddFOp, arith::MulFOp, arith::MaximumFOp, arith::MinimumFOp>(op);
      if (a.kind == EV::Const && b.kind == EV::Const) return fail("constant folding left to the compiler");
      if (b.kind == EV::Const || (comm && a.kind == EV::Const)) {
        if (a.kind == EV::Const) std::swap(a, b);
        float c = b.c;
        if (isa<arith::MulFOp>(op)) {
          if (!mulConst(a, c)) return false;
        } else if (isa<arith::AddFOp>(op)) {
          if (!addConst(a, c)) return false;
        } else if (isa<arith::SubFOp>(op)) {
          if (!addConst(a, -c)) return false;
        } else {
          auto o = mem(a);
          if (!o) return false;
          Pend p;
          p.s1 = *o;
          p.op = isa<arith::MaximumFOp>(op) ? ::sa::VOP_MAX : ::sa::VOP_MIN;
          p.s2 = Opd{};
          p.s2->isImm = true;
          p.s2->imm = c;
          evOf(res) = newPend(p);
        }
        continue;
      }
      if (a.kind == EV::Const && isa<arith::SubFOp>(op)) {
        // c - x = (-x) + c
        EV x = b;
        auto pe = copyPend(x);
        if (!pe) return false;
        pends[pe->pend].A = -1.0f;
        pends[pe->pend].B = a.c;
        pends[pe->pend].stage = 2;
        evOf(res) = *pe;
        continue;
      }
      auto oa = mem(a);
      auto ob = mem(b);
      if (!oa || !ob) return false;
      Pend p;
      p.s1 = *oa;
      p.s2 = *ob;
      if (p.s1.mode != ::sa::IDX_LIN && p.s2->mode == ::sa::IDX_LIN && comm) std::swap(p.s1, *p.s2);
      p.op = isa<arith::AddFOp>(op)   ? ::sa::VOP_ADD
             : isa<arith::SubFOp>(op) ? ::sa::VOP_SUB
             : isa<arith::MulFOp>(op) ? ::sa::VOP_MUL
             : isa<arith::MaximumFOp>(op) ? ::sa::VOP_MAX
                                          : ::sa::VOP_MIN;
      evOf(res) = newPend(p);
      continue;
    }
    return fail(std::string("unsupported body operation ") + op->getName().getStringRef().str());
  }

  // ---- results
  auto yieldOp = cast<linalg::YieldOp>(yield);
  if (geo.reduction) {
    if (g.getNumDpsInits() != 1) return fail("multi-result reduction");
    Value y = yieldOp.getOperand(0);
    Operation *comb = y.getDefiningOp();
    if (!comb || comb->getNumOperands() != 2) return fail("reduction combiner");
    Value acc = body.getArguments().back();
    Value elem = comb->getOperand(0) == acc ? comb->getOperand(1) : comb->getOperand(0);
    ::sa::VRed rk;
    float ident;
    if (isa<arith::AddFOp>(comb)) rk = ::sa::RED_SUM, ident = 0.0f;
    else if (isa<arith::MaximumFOp>(comb)) rk = ::sa::RED_MAX, ident = -INFINITY;
    else return fail("reduction other than an fp32 sum or max");
    auto fill = outInit.getDefiningOp<linalg::FillOp>();
    if (!fill || !isF32Const(fill.getDpsInputs()[0], ident)) return fail("reduction init is not the identity");
    EV e = evOf(elem);
    std::optional<EV> pe;
    if (e.kind == EV::Pending) pe = e;
    else pe = copyPend(e);
    if (!pe) return false;
    if (pends[pe->pend].validNegInf && rk != ::sa::RED_MAX) return fail("-inf mask in a sum");
    if (sel) {
      void *key = g.getResult(0).getAsOpaquePointer();
      if (!rowOutputs.count(key)) rowOutputs[key] = newLocal(::sa::VT_F32, sel->H, Layout::Bcast);
      emitPend(pends[pe->pend], la(rowOutputs[key]) + uint32_t(sel->row), ::sa::VT_F32, geo, rk);
      return err.empty();
    }
    Local out = newLocal(::sa::VT_F32, geo.rows, Layout::Bcast);
    emitPend(pends[pe->pend], la(out), ::sa::VT_F32, geo, rk);
    locals[g.getResult(0).getAsOpaquePointer()] = out;
    return err.empty();
  }
  for (unsigned r = 0; r < g.getNumResults(); ++r) {
    Value y = yieldOp.getOperand(r);
    Type ot = cast<RankedTensorType>(g.getResult(r).getType()).getElementType();
    if (auto it = locals.find(y.getAsOpaquePointer()); it != locals.end()) {
      locals[g.getResult(r).getAsOpaquePointer()] = it->second;   // a gathered row, as is
      continue;
    }
    EV e = evOf(y);
    // the output: in row mode one row of a shared row-strided local
    auto outLocal = [&](VType vt) -> std::pair<Local, uint32_t> {
      if (!sel) {
        Local out = newLocal(vt, geo.n);
        return {out, la(out)};
      }
      void *key = g.getResult(r).getAsOpaquePointer();
      if (!rowOutputs.count(key)) {
        uint32_t stride = uint32_t((geo.inner + d - 1) / d);
        Local out = newLocal(vt, int64_t(stride) * sel->H * d);
        out.rowStride = stride;
        out.rows = sel->H;
        rowOutputs[key] = out;
      }
      Local out = rowOutputs[key];
      return {out, la(out) + uint32_t(sel->row) * out.rowStride};
    };
    auto emitTo = [&](EV x, VType vt, bool final) -> std::optional<Local> {
      std::optional<EV> pe;
      if (x.kind == EV::Pending) pe = x;
      else pe = copyPend(x);
      if (!pe) return std::nullopt;
      if (pends[pe->pend].validNegInf) return fail("-inf mask outside a max reduction"), std::nullopt;
      if (final) {
        auto [out, dst] = outLocal(vt);
        emitPend(pends[pe->pend], dst, vt, geo);
        Local row = out;
        row.word = dst & 0xFFFF;
        return row;
      }
      Local out = newLocal(vt, geo.n);
      emitPend(pends[pe->pend], la(out), vt, geo);
      return out;
    };
    std::optional<Local> out;
    if (ot.isF32()) {
      out = emitTo(e, ::sa::VT_F32, true);
    } else if (ot.isInteger(8) && e.kind == EV::ToI8) {
      out = emitTo(*e.inner, ::sa::VT_I8, true);
    } else if (ot.isInteger(32) && e.kind == EV::I32OfI8) {
      auto i8 = emitTo(*e.inner->inner, ::sa::VT_I8, false);
      if (!i8) return false;
      auto [o32, dst] = outLocal(::sa::VT_I32);
      Pend p;
      p.s1 = {la(*i8), ::sa::VT_I8, ::sa::IDX_LIN, 0};
      emitPend(p, dst, ::sa::VT_I32, geo);
      out = o32;
    } else {
      return fail("unsupported result type of a generic");
    }
    if (!out) return false;
    if (!sel) locals[g.getResult(r).getAsOpaquePointer()] = *out;
  }
  return err.empty();
}

// =============================================================== the int8 linear layer
// con: acc[t, j] += ext(x[k]) * ext(Wp[t, k, j]); epi: y = (f32(i32(acc)) * s_w) * s_x [op c ...]
struct EpilogueStep {
  ::sa::VOp op;
  bool withConst;
  float c;
  int binding = -1;            // operand region
  uint32_t off = 0;
  int base = 0;
};
struct EpilogueCtx {
  std::vector<EpilogueStep> steps;
};

void epilogueHook(void *ctxp, ::sa::DescList &dl, uint32_t yWord, uint32_t freeWord, uint32_t ddrOff, bool dyn,
                  uint32_t pf, uint32_t words) {
  auto *ctx = static_cast<EpilogueCtx *>(ctxp);
  uint32_t d8 = 0;
  (void)d8;
  for (const EpilogueStep &s : ctx->steps) {
    ::sa::VeFp fp;
    uint32_t elems = 0;
    (void)elems;
    // chunk length in elements: words * D; recovered from the caller's word count
    // (the hook does not know D; the caller stores it in the step's base when needed)
    if (!s.withConst) {
      std::vector<::sa::Dyn> dy;
      if (dyn) dy.push_back({::sa::DYN_DMA_DDR, pf, true});
      uint32_t bytes = words * uint32_t(s.binding);   // binding field reused: bytes per word
      dl.ld(s.off + ddrOff, acc(freeWord), 1, bytes, bytes, s.base, 0, false, dy);
      fp.B = NEG0;
      dl.veFp(acc(yWord), acc(freeWord), acc(yWord), bytes / 4, s.op, ::sa::vtypes(::sa::VT_F32, ::sa::VT_F32), 0,
              fp);
    }
  }
}

bool Gen::qlinear(linalg::GenericOp con, linalg::GenericOp epi) {
  // --- contraction
  auto cmaps = con.getIndexingMapsArray();
  MLIRContext *ctx = func->getContext();
  AffineExpr t, j, k;
  bindDims(ctx, t, j, k);
  if (cmaps[0] != AffineMap::get(3, 0, {k}, ctx) || cmaps[1] != AffineMap::get(3, 0, {t, k, j}, ctx) ||
      cmaps[2] != AffineMap::get(3, 0, {t, j}, ctx))
    return fail("contraction indexing is not x[k], Wp[t, k, j] -> acc[t, j]");
  // x: a load (i32) or extsi(load i8)
  Value xv = con.getDpsInputs()[0];
  bool xI32 = true;
  if (auto ext = xv.getDefiningOp<linalg::GenericOp>()) {
    xv = ext.getDpsInputs()[0];
    xI32 = false;
  }
  auto xs = sources.find(xv.getAsOpaquePointer());
  auto ws = sources.find(con.getDpsInputs()[1].getAsOpaquePointer());
  if (xs == sources.end() || ws == sources.end()) return fail("linear operands are not loads");
  auto wType = ws->second.type;
  if (wType.getRank() != 3 || wType.getDimSize(2) != d) return fail("packed weights are not [n/D, k, D]");
  int64_t kk = wType.getDimSize(1), nt = wType.getDimSize(0), n = nt * d;
  if (xI32 != xs->second.type.getElementType().isInteger(32)) return fail("x type");
  // --- epilogue prefix: trunci, sitofp, mulf s_w, mulf s_x
  Block &b = epi.getRegion().front();
  SmallVector<Operation *> ops;
  for (Operation &o : b.without_terminator()) ops.push_back(&o);
  auto emaps = epi.getIndexingMapsArray();
  if (ops.size() < 4 || !isa<arith::TruncIOp>(ops[0]) || !isa<arith::SIToFPOp>(ops[1]) ||
      !isa<arith::MulFOp>(ops[2]) || !isa<arith::MulFOp>(ops[3]))
    return fail("linear epilogue does not start with (f32(i32(acc)) * s_w) * s_x");
  auto argNo = [&](Value v) -> int {
    auto ba = dyn_cast<BlockArgument>(v);
    return ba ? int(ba.getArgNumber()) : -1;
  };
  int accArg = -1;
  for (int i = 0; i < int(epi.getNumDpsInputs()); ++i)
    if (epi.getDpsInputs()[i] == con.getResult(0)) accArg = i;
  if (accArg < 0 || !emaps[accArg].isIdentity() || argNo(ops[0]->getOperand(0)) != accArg ||
      ops[1]->getOperand(0) != ops[0]->getResult(0))
    return fail("linear epilogue: accumulator conversion");
  auto other = [&](Operation *m, Value v) { return m->getOperand(0) == v ? m->getOperand(1) : m->getOperand(0); };
  int swArg = argNo(other(ops[2], ops[1]->getResult(0)));
  int sxArg = argNo(other(ops[3], ops[2]->getResult(0)));
  if (swArg < 0 || sxArg < 0 || swArg == accArg || sxArg == accArg || emaps[sxArg].getNumResults() != 0 ||
      !emaps[swArg].isIdentity())
    return fail("linear epilogue: s_w / s_x operands");
  auto sws = sources.find(epi.getDpsInputs()[swArg].getAsOpaquePointer());
  auto sxs = sources.find(epi.getDpsInputs()[sxArg].getAsOpaquePointer());
  if (sws == sources.end() || sxs == sources.end()) return fail("s_w / s_x are not loads");
  // further steps: (prev op operand)
  EpilogueCtx ectx;
  Value prev = ops[3]->getResult(0);
  for (size_t i = 4; i < ops.size(); ++i) {
    Operation *o = ops[i];
    if (!isa<arith::AddFOp, arith::MulFOp>(o) || (o->getOperand(0) != prev && o->getOperand(1) != prev))
      return fail("linear epilogue: unsupported operation after * s_x");
    Value opnd = other(o, prev);
    int a = argNo(opnd);
    if (a < 0 || a == accArg || !emaps[a].isIdentity())
      return fail("linear epilogue: operand is not an identity-mapped input");
    auto os = sources.find(epi.getDpsInputs()[a].getAsOpaquePointer());
    if (os == sources.end() || !os->second.type.getElementType().isF32()) return fail("linear epilogue operand");
    EpilogueStep st;
    st.op = isa<arith::AddFOp>(o) ? ::sa::VOP_ADD : ::sa::VOP_MUL;
    st.withConst = false;
    st.base = os->second.r.base;
    st.off = os->second.r.off;
    st.binding = int(4 * d);            // bytes per word
    ectx.steps.push_back(st);
    prev = o->getResult(0);
  }
  auto yieldOp = cast<linalg::YieldOp>(b.getTerminator());
  if (yieldOp.getOperand(0) != prev) return fail("linear epilogue result");
  // --- the store of the result
  DispatchTensorStoreOp store;
  for (Operation *u : epi.getResult(0).getUsers())
    if (auto s = dyn_cast<DispatchTensorStoreOp>(u)) store = s;
  if (!store) return fail("linear result is not stored");
  auto yspan = spanOf(store.getTarget());
  if (!yspan) return false;
  // --- prologue (reference.qlinear with regions)
  if (kk % d || n % d) return fail("k, n not multiples of D");
  uint32_t sx;
  if (xI32) {
    uint32_t xa = lay.acc0;
    dl.ld(xs->second.r.off, acc(xa), 1, uint32_t(4 * kk), uint32_t(4 * kk), xs->second.r.base);
    ::sa::VeFp rep;
    rep.m1 = ::sa::IDX_DIV;
    rep.p1 = uint32_t(d);
    dl.veFp(acc(xa), 0, laddr(::sa::MEM_SPAD_A, 0), uint32_t(kk * d), ::sa::VOP_COPY,
            ::sa::vtypes(::sa::VT_I32, ::sa::VT_I8), 0, rep);
    sx = xa + uint32_t(kk / d);
  } else {
    dl.ld(xs->second.r.off, laddr(::sa::MEM_SPAD_A, uint32_t(kk)), 1, uint32_t(kk), uint32_t(kk), xs->second.r.base);
    ::sa::VeFp rep;
    rep.m1 = ::sa::IDX_DIV;
    rep.p1 = uint32_t(d);
    dl.veFp(laddr(::sa::MEM_SPAD_A, uint32_t(kk)), 0, laddr(::sa::MEM_SPAD_A, 0), uint32_t(kk * d),
            ::sa::VOP_COPY, ::sa::vtypes(::sa::VT_I8, ::sa::VT_I8), 0, rep);
    sx = lay.acc0;
  }
  dl.ld(sxs->second.r.off, acc(sx), 1, 8, 8, sxs->second.r.base);
  ::sa::VeFp bc;
  bc.reduce = ::sa::RED_MAX;
  bc.valid = 1;
  dl.veFp(acc(sx), 0, acc(sx + 1), uint32_t(d), ::sa::VOP_COPY, ::sa::vtypes(::sa::VT_F32, ::sa::VT_F32), 0, bc);
  if (sws->second.r.off % 8 || yspan->r.off % 8 || ws->second.r.off % 8) return fail("unaligned linear operand");
  ::sa::LinearPlace place{ws->second.r.base, sws->second.r.base, yspan->r.base, ws->second.r.off,
                          sws->second.r.off, yspan->r.off};
  int pf = privateParam(), pw = privateParam();      // PARAM7, PARAM6 (reference.qlinear: w = 6, f = 7)
  uint32_t nc = ::sa::emitLinear(dl, lay, uint32_t(kk), uint32_t(n), sx + 1, place, uint32_t(pw), uint32_t(pf),
                                 ectx.steps.empty() ? 1 : 2, ectx.steps.empty() ? nullptr : epilogueHook, &ectx);
  if (nc == 0) return fail("linear layer does not fit the local memories");
  handledStores.push_back(store.getOperation());
  return err.empty();
}

// =============================================================== attention (batch matmuls)
// scores: bmm(ext(K^T), q) -> [H, T, 1], epilogue (f32(acc) * s_q[h]) * a_k
// P V:    bmm(p, ext(V))   -> [H, 1, hs], epilogue f32(acc) * a_v
// with K / V : i8[T, H, hs] (a slice of the KV cache) through an extsi generic
// with map (t, h, j) -> (h, t, j). Per head, as compile_model.attention: K rows
// -> TRANSPOSE -> K^T tiles; q_h or p_h as a replicated int8 A strip; EX; VE.
bool Gen::attention(linalg::BatchMatmulOp bmm) {
  Value a = bmm.getDpsInputs()[0], b = bmm.getDpsInputs()[1];
  // which operand is the extended cache slice
  bool scores = a.getDefiningOp<linalg::GenericOp>() != nullptr;
  Value cacheExt = scores ? a : b, small = scores ? b : a;
  auto ext = cacheExt.getDefiningOp<linalg::GenericOp>();
  if (!ext || ext.getNumDpsInputs() != 1) return fail("attention: cache operand is not an extsi generic");
  {
    MLIRContext *ctx = func->getContext();
    AffineExpr t, h, j;
    bindDims(ctx, t, h, j);
    auto m = ext.getIndexingMapsArray();
    if (m[0] != AffineMap::get(3, 0, {t, h, j}, ctx) || m[1] != AffineMap::get(3, 0, {h, t, j}, ctx))
      return fail("attention: cache transpose map");
  }
  auto cache = sources.find(ext.getDpsInputs()[0].getAsOpaquePointer());
  auto sm = sources.find(small.getAsOpaquePointer());
  if (cache == sources.end() || sm == sources.end()) return fail("attention operands are not loads");
  auto cT = cache->second.type;   // [T, H, hs] i8
  if (cT.getRank() != 3 || !cT.getElementType().isInteger(8)) return fail("attention cache type");
  int64_t H = cT.getDimSize(1), hs = cT.getDimSize(2);
  bool dynT = ShapedType::isDynamic(cT.getDimSize(0));
  int64_t T = dynT ? opt.maxDynamic : cT.getDimSize(0);
  if (ShapedType::isDynamic(H) || ShapedType::isDynamic(hs)) return fail("attention: dynamic heads");
  if (T % d || hs % d) return fail("attention: T and head size must be multiples of D");
  if (!cast<RankedTensorType>(small.getType()).getElementType().isInteger(32)) return fail("attention q / p type");
  // dynamic T: its PARAMs
  std::optional<Lin> tl = dynT ? cache->second.dynDim : std::nullopt;
  if (dynT && !tl) return fail("attention: dynamic length not from a push constant");
  auto par = [&](int64_t mul, int shift) {
    Lin l = *tl;
    l.mul *= mul;
    l.add *= mul;
    l.shift += shift;
    return uint32_t(paramFor(l));
  };
  int logd = int(std::log2(double(d)));
  // the epilogue
  linalg::GenericOp epi;
  for (Operation *u : bmm->getResult(0).getUsers()) epi = dyn_cast<linalg::GenericOp>(u);
  if (!epi) return fail("attention without an epilogue");
  Block &body = epi.getRegion().front();
  SmallVector<Operation *> ops;
  for (Operation &o : body.without_terminator()) ops.push_back(&o);
  auto maps = epi.getIndexingMapsArray();
  float scale = 0;
  int sqArg = -1;
  if (scores) {
    if (ops.size() != 4 || !isa<arith::TruncIOp>(ops[0]) || !isa<arith::SIToFPOp>(ops[1]) ||
        !isa<arith::MulFOp>(ops[2]) || !isa<arith::MulFOp>(ops[3]))
      return fail("attention scores epilogue is not (f32(acc) * s_q) * a_k");
    Value o2 = ops[2]->getOperand(0) == ops[1]->getResult(0) ? ops[2]->getOperand(1) : ops[2]->getOperand(0);
    auto ba = dyn_cast<BlockArgument>(o2);
    Value o3 = ops[3]->getOperand(0) == ops[2]->getResult(0) ? ops[3]->getOperand(1) : ops[3]->getOperand(0);
    auto c = constF32(o3);
    if (!ba || !c) return fail("attention scores epilogue operands");
    sqArg = int(ba.getArgNumber());
    scale = *c;
    if (maps[sqArg].getNumResults() != 1 || maps[sqArg].getResult(0) != getAffineDimExpr(0, func->getContext()))
      return fail("s_q is not per head");
  } else {
    if (ops.size() != 3 || !isa<arith::TruncIOp>(ops[0]) || !isa<arith::SIToFPOp>(ops[1]) ||
        !isa<arith::MulFOp>(ops[2]))
      return fail("attention P V epilogue is not f32(acc) * a_v");
    Value o2 = ops[2]->getOperand(0) == ops[1]->getResult(0) ? ops[2]->getOperand(1) : ops[2]->getOperand(0);
    auto c = constF32(o2);
    if (!c) return fail("attention P V scale");
    scale = *c;
  }
  DispatchTensorStoreOp store;
  for (Operation *u : epi->getResult(0).getUsers()) store = dyn_cast<DispatchTensorStoreOp>(u);
  if (!store) return fail("attention result is not stored");
  auto out = spanOf(store.getTarget());
  if (!out) return false;
  if (out->r.off % 8 || cache->second.r.off % 8 || sm->second.r.off % 8) return fail("unaligned attention operand");
  std::optional<Local> sq;
  if (scores) {
    sq = perElementBcast(epi.getDpsInputs()[sqArg], H);
    if (!sq) return false;
  }
  const uint32_t hw = uint32_t(hs / d), tiles = uint32_t(T / d), sb = lay.sbank;
  const uint32_t kraw = sb, kt = 0, vb = sb;
  if (uint32_t(T) * hw > sb) return fail("attention: K rows do not fit a SPAD bank");
  using DV = std::vector<::sa::Dyn>;
  auto dyn1 = [&](uint32_t field, uint32_t p) { return dynT ? DV{{field, p, false}} : DV{}; };
  if (scores) {
    uint32_t pT = dynT ? par(1, 0) : 0, pThs = dynT ? par(hs, 0) : 0, pTiles = dynT ? par(1, logd) : 0;
    Local srcw = newLocal(::sa::VT_I32, hs);
    Local c = newLocal(::sa::VT_I32, int64_t(d) * tiles * d);
    // the scores of all heads, one row (up to T) per head
    Local res = newLocal(::sa::VT_F32, int64_t(tiles) * H * d);
    res.rowStride = tiles;
    res.rows = H;
    for (int64_t h = 0; h < H; ++h) {
      dl.ld(cache->second.r.off + uint32_t(h * hs), laddr(::sa::MEM_SPAD_A, kraw), uint32_t(T), uint32_t(hs),
            uint32_t(H * hs), cache->second.r.base, 0, false, dyn1(::sa::DYN_DMA_ROWS, pT));
      dl.transpose(laddr(::sa::MEM_SPAD_A, kraw), laddr(::sa::MEM_SPAD_B, kt), uint32_t(T * hs),
                   ::sa::vtypes(::sa::VT_I8, ::sa::VT_I8), hw, false, dyn1(::sa::DYN_VE_LEN, pThs));
      dl.ld(sm->second.r.off + uint32_t(h * hs * 4), la(srcw), 1, uint32_t(hs * 4), uint32_t(hs * 4),
            sm->second.r.base);
      ::sa::VeFp rep;
      rep.m1 = ::sa::IDX_DIV;
      rep.p1 = uint32_t(d);
      dl.veFp(la(srcw), 0, laddr(::sa::MEM_SPAD_A, 0), uint32_t(hs * d), ::sa::VOP_COPY,
              ::sa::vtypes(::sa::VT_I32, ::sa::VT_I8), 0, rep);
      dl.ex(0, kt, c.word, hw, false, tiles, uint32_t(hs), 1, tiles, false, dyn1(::sa::DYN_EX_REPEAT, pTiles));
      ::sa::VeFp fp;
      fp.m2 = ::sa::IDX_DIV;
      fp.t2 = int(::sa::VT_F32);
      fp.A = scale;
      fp.B = NEG0;
      dl.veFp(acc(c.word), la(*sq) + uint32_t(h), la(res) + uint32_t(h) * tiles, uint32_t(T), ::sa::VOP_MUL,
              ::sa::vtypes(::sa::VT_I32, ::sa::VT_F32), 0xFFFF, fp, false, dyn1(::sa::DYN_VE_LEN, pT));
    }
    if (dynT) {
      int bp = int(par(4, 0));
      rowLoop(H, bp, [&](int64_t h, std::vector<::sa::Dyn> dy) {
        dl.st(out->r.off, la(res) + uint32_t(h) * tiles, 1, uint32_t(T * 4), uint32_t(T * 4), out->r.base, false, dy);
      });
    } else {
      for (int64_t h = 0; h < H; ++h)
        dl.st(out->r.off + uint32_t(h * T * 4), la(res) + uint32_t(h) * tiles, 1, uint32_t(T * 4), uint32_t(T * 4),
              out->r.base);
    }
  } else {
    // p: [H, 1, T] i32, one row per head
    std::optional<Local> p;
    if (dynT) {
      p = materialize(small);
      if (!p) return false;
      if (!p->rowStride) return fail("attention p not in the row layout");
    } else {
      p = newLocal(::sa::VT_I32, T * H);
      p->rowStride = uint32_t(T / d);
      dl.ld(sm->second.r.off, la(*p), 1, uint32_t(T * H * 4), uint32_t(T * H * 4), sm->second.r.base);
    }
    uint32_t pT = dynT ? par(1, 0) : 0, pTd = dynT ? par(d, 0) : 0, pTiles = dynT ? par(1, logd) : 0;
    Local c = newLocal(::sa::VT_I32, int64_t(d) * hw * d);
    Local res = newLocal(::sa::VT_F32, hs);
    for (int64_t h = 0; h < H; ++h) {
      ::sa::VeFp rep;
      rep.m1 = ::sa::IDX_DIV;
      rep.p1 = uint32_t(d);
      dl.veFp(la(*p) + uint32_t(h) * p->rowStride, 0, laddr(::sa::MEM_SPAD_A, 0), uint32_t(T * d), ::sa::VOP_COPY,
              ::sa::vtypes(::sa::VT_I32, ::sa::VT_I8), 0, rep, false, dyn1(::sa::DYN_VE_LEN, pTd));
      dl.ld(cache->second.r.off + uint32_t(h * hs), laddr(::sa::MEM_SPAD_B, vb), uint32_t(T), uint32_t(hs),
            uint32_t(H * hs), cache->second.r.base, /*LD_INTERLEAVE=*/1, false, dyn1(::sa::DYN_DMA_ROWS, pT));
      DV exd;
      if (dynT) exd = {{::sa::DYN_EX_KT, pTiles, false}, {::sa::DYN_EX_BSTEP, pT, false}};
      dl.ex(0, vb, c.word, tiles, false, hw, uint32_t(T), 1, hw, false, exd);
      ::sa::VeFp fp;
      fp.A = scale;
      fp.B = NEG0;
      dl.veFp(acc(c.word), 0, la(res), uint32_t(hs), ::sa::VOP_COPY, ::sa::vtypes(::sa::VT_I32, ::sa::VT_F32), 0,
              fp);
      dl.st(out->r.off + uint32_t(h * hs * 4), la(res), 1, uint32_t(hs * 4), uint32_t(hs * 4), out->r.base);
    }
  }
  skipOps.push_back(epi.getOperation());
  handledStores.push_back(store.getOperation());
  return err.empty();
}

// =============================================================== scatter: one row into a cache
bool Gen::scatter(IREE::LinalgExt::ScatterOp sc) {
  Value upd = sc.getUpdates(), idx = sc.getIndices(), orig = sc.getOriginal();
  auto is = sources.find(idx.getAsOpaquePointer()), os = sources.find(orig.getAsOpaquePointer());
  if (is == sources.end() || os == sources.end()) return fail("scatter index / target are not loads");
  if (!sources.count(upd.getAsOpaquePointer()) && !locals.count(upd.getAsOpaquePointer()))
    return fail("scatter update neither loaded nor computed");
  auto uT = cast<RankedTensorType>(upd.getType()), oT = cast<RankedTensorType>(orig.getType());
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
  if (!u) return false;
  int p = privateParam();
  dl.ldparam(is->second.r.off, uint32_t(p), rowBytes, 0, is->second.r.base);
  dl.st(os->second.r.off, la(*u), 1, rowBytes, rowBytes, os->second.r.base, false,
        {{::sa::DYN_DMA_DDR, uint32_t(p), true}});
  // the whole-tensor store of the result writes back the same binding: done
  for (Operation *us2 : sc->getResult(0).getUsers())
    if (auto st = dyn_cast<DispatchTensorStoreOp>(us2)) {
      auto a = st.getTarget().getDefiningOp(), bOp = orig.getDefiningOp<DispatchTensorLoadOp>().getSource().getDefiningOp();
      if (a != bOp) return fail("scatter result stored elsewhere");
      handledStores.push_back(st.getOperation());
    }
  return err.empty();
}

// =============================================================== i64 scalar + constant
// (row indices, e.g. pos + l * S): computed in fp32 (exact below 2^24) from the
// low word; the stored i64 has the high word 0.
bool Gen::intScalar(linalg::GenericOp g) {
  Block &b = g.getRegion().front();
  SmallVector<Operation *> ops;
  for (Operation &o : b.without_terminator()) ops.push_back(&o);
  if (g.getNumDpsInputs() != 1 || ops.size() != 1 || !isa<arith::AddIOp>(ops[0]))
    return fail("i64 scalar computation other than x + constant");
  Value other = ops[0]->getOperand(0) == b.getArgument(0) ? ops[0]->getOperand(1) : ops[0]->getOperand(0);
  auto c = getConstantIntValue(other);
  if (!c || std::abs(*c) >= (1 << 23)) return fail("i64 scalar: constant");
  auto src = materialize(g.getDpsInputs()[0]);
  if (!src) return false;
  // v = lo + c in every lane (a max over the first lane)
  Local v = newLocal(::sa::VT_F32, 1, Layout::Bcast);
  ::sa::VeFp f1;
  f1.B = float(*c);
  f1.reduce = ::sa::RED_MAX;
  f1.valid = 1;
  dl.veFp(la(*src), 0, la(v), uint32_t(d), ::sa::VOP_COPY, ::sa::vtypes(::sa::VT_I32, ::sa::VT_F32), 0, f1);
  // (swapneg(v) - v) * -0.5 = [v, 0, v, 0, ...]: the i64 (v, 0) in lanes 0, 1
  Local w = newLocal(::sa::VT_I32, int64_t(d));
  ::sa::VeFp f2;
  f2.swapneg = true;
  f2.A = -0.5f;
  dl.veFp(la(v), la(v), la(w), uint32_t(d), ::sa::VOP_SUB, ::sa::vtypes(::sa::VT_F32, ::sa::VT_I32), 0, f2);
  locals[g.getResult(0).getAsOpaquePointer()] = w;
  return err.empty();
}

}  // namespace

// ================================================================= driver
bool Gen::run(Generated &out, std::string &why) {
  Block &blk = func.getFunctionBody().front();
  // loads first (they only describe; the LD is emitted when a value is needed)
  for (Operation &op : blk)
    if (auto l = dyn_cast<DispatchTensorLoadOp>(op))
      if (!analyzeLoad(l)) return why = err, false;
  for (Operation &opRef : blk) {
    Operation *op = &opRef;
    if (isa<DispatchTensorLoadOp, tensor::EmptyOp, arith::ConstantOp, IREE::HAL::InterfaceConstantLoadOp,
            IREE::HAL::InterfaceBindingSubspanOp, IREE::Util::AssumeIntOp, IREE::TensorExt::DispatchWorkloadOrdinalOp,
            arith::IndexCastUIOp, arith::IndexCastOp, arith::ExtUIOp, arith::ShLIOp, arith::OrIOp, func::ReturnOp>(op))
      continue;
    if (auto f = dyn_cast<linalg::FillOp>(op)) {
      continue;   // read by the reduction / contraction that uses it
    }
    if (auto bmm = dyn_cast<linalg::BatchMatmulOp>(op)) {
      if (!attention(bmm)) return why = err, false;
      continue;
    }
    if (auto sc = dyn_cast<IREE::LinalgExt::ScatterOp>(op)) {
      if (!scatter(sc)) return why = err, false;
      continue;
    }
    if (auto g = dyn_cast<linalg::GenericOp>(op)) {
      if (llvm::is_contained(skipOps, op)) continue;
      // the extension of an attention operand is part of the attention template
      if (llvm::any_of(g->getUsers(), [](Operation *u) { return isa<linalg::BatchMatmulOp>(u); })) continue;
      if (auto rt = dyn_cast<RankedTensorType>(g.getResult(0).getType());
          rt && rt.getRank() == 0 && rt.getElementType().isInteger(64)) {
        if (!intScalar(g)) return why = err, false;
        continue;
      }
      // an int8 linear layer: contraction generic followed by its epilogue
      if (g.getNumReductionLoops() == 1 && g.getNumLoops() == 3 && g.getNumDpsInputs() == 2) {
        linalg::GenericOp epi;
        for (Operation *u : g.getResult(0).getUsers()) epi = dyn_cast<linalg::GenericOp>(u);
        if (!epi) return why = "contraction without an epilogue", false;
        if (!qlinear(g, epi)) return why = err, false;
        skipOps.push_back(epi.getOperation());
        continue;
      }
      if (llvm::is_contained(skipOps, op)) continue;
      // the i8 -> i32 sign extension of a linear's x is part of the linear
      bool feedsLinear = false;
      for (Operation *u : g->getUsers())
        if (auto c = dyn_cast<linalg::GenericOp>(u); c && c.getNumReductionLoops() == 1 && c.getNumLoops() == 3)
          feedsLinear = true;
      if (feedsLinear) continue;
      if (!genericOp(g)) return why = err, false;
      continue;
    }
    if (auto s = dyn_cast<DispatchTensorStoreOp>(op)) {
      if (llvm::is_contained(handledStores, op)) continue;
      if (!store(s)) return why = err, false;
      continue;
    }
    return why = "unsupported operation " + op->getName().getStringRef().str(), false;
  }
  if (!err.empty()) return why = err, false;
  dl.ret();
  out.templ = dl.bytes();
  out.setup = setup;
  return true;
}

bool generateDispatch(FunctionOpInterface func, const CodegenOptions &options, Generated &out, std::string &why) {
  Gen g(func, options);
  return g.run(out, why);
}

}  // namespace mlir::iree_compiler::sa

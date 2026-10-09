// The lowering of sahl-to-sahw (docs/iree_compiler_plan.md §8.4 step 10,
// §8.14 C8 R6): one sahw.template per dispatch function, translating the
// decisions the sahl passes made (gathers, pieces, memory places, the
// micro-kernels written out by sahl-expand-kernels) into commands.
// Definitions: SahlToSahw.cpp (the pass, the block walk, registers, local
// memory, DDR / DMA, the micro-kernel ops, VE helpers, scatter),
// SahlLowerGeneric.cpp (linalg.generic: element-wise, reductions, gathers).
#ifndef SA_TRANSFORMS_SAHLLOWER_H_
#define SA_TRANSFORMS_SAHLLOWER_H_

#include <cmath>
#include <functional>
#include <cstring>
#include <map>
#include <memory>

#include "../target/DescList.h"
#include "../target/Layout.h"
#include "SaLin.h"
#include "SahlDialect.h"
#include "SahlKernels.h"
#include "SahlLocalMemory.h"
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

namespace mlir::iree_compiler::sa::lower {


using ::sa::VType;
constexpr float NEG0 = -0.0f;

inline std::optional<VType> vtOf(Type t) {
  if (t.isF32()) return ::sa::VT_F32;
  if (t.isInteger(32)) return ::sa::VT_I32;
  if (t.isInteger(8)) return ::sa::VT_I8;
  return std::nullopt;
}
inline uint32_t esize(Type t) {
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
        regB(body), bb(OpBuilder::atBlockEnd(&body.getRegion().front())), loc(f.getLoc()), mem(lay) {
  }

  bool run();

  bool lowerBlock(Block &block);

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
  LocalMemory mem;                               // SPAD_A / ACC words of this template (SahlLocalMemory.h)
  Operation *lastVe = nullptr;
  std::map<Lin, Value> params;
  Value rowAcc;                                  // the row loops' DDR offset
  Value curLen;                                  // row mode: the dynamic LEN of full-length VEs
  int64_t curLenStatic = -1;
  // A VE of a fixed size inside row mode (the w * D * D block of a per-row
  // broadcast, a scalar's broadcast word) keeps its static LEN even when that
  // equals a full row's (w * D * D = maxDynamic at D = 16, w = 1)
  struct StaticLen {
    Lowerer &l;
    Value saved;
    explicit StaticLen(Lowerer &l) : l(l), saved(l.curLen) { l.curLen = Value(); }
    ~StaticLen() { l.curLen = saved; }
  };
  std::map<std::pair<void *, int>, Value> validParams;   // (i64 scalar, predicate) -> VALID count
  // its schedule: as sahl-schedule chose (the attributes), else chosen here

  bool fail(const std::string &m);

  // ------------------------------------------------------------ registers
  Value baseFor(int binding, const Lin &off);

  Value paramFor(const Lin &v);

  // ------------------------------------------------------------ local memory
  llvm::DenseMap<Value, LocalBuf> reserved;      // sahl.reserve: shared by the pieces (sahl.scope)
  LocalBuf newLocal(VType vt, int64_t n, bool bcast = false);
  // the local of a buffer (allocated at its first use)
  std::optional<LocalBuf> bufOf(Value v, bool bcast = false);
  // an expanded kernel's steps (sahl-expand-kernels)
  bool strip(sahl::StripOp st);
  bool mma(sahl::MmaOp m);
  bool gemv(sahl::GemvOp g);
  bool loop(sahl::LoopOp lp);
  int64_t leadingPitch(Value v, int64_t rb);
  int64_t placedElems(memref::AllocOp alloc, LocalBuf &l);
  // inside a sahl.loop: the PARAM a DMA's sa.advance names (none: Value()), and marking the DMA dynamic
  Value advanceOf(Operation *op);
  void advanced(Operation *op, Value adv);
  SmallVector<Value> loopParams;

  // ------------------------------------------------------------ DDR
  struct Ddr {
    Value base;
    int64_t off = 0;           // bytes
    int64_t n = 0;
    Type et;
    std::optional<Lin> dyn;    // a dynamic innermost length (rows of it, contiguous)
    int64_t rows = 1;
    int64_t pitch = 0;         // static rows `pitch` bytes apart (a column slice), 0: contiguous
    std::optional<Lin> dynRows;  // a view of a dynamic number of rows (the fields at max_dynamic rows)
  };
  std::optional<Ddr> ddrOf(Value v, bool allowPitch = false);

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
  bool dynamicDma(const Ddr &r, const LocalBuf &l, bool isLoad);
  // a row of a dynamic length at the row cursor (sa.cursor; sa.cursor_step: then the cursor on)
  bool cursorDma(Operation *op, bool isLoad, const Ddr &r, uint32_t la);

  // A contiguous DMA of `bytes` (a multiple of 8) between DDR and local memory:
  // one row, or beyond the 16-bit row length rows of the largest multiple of
  // the local word (so the rows are contiguous there too) and a tail.
  void contiguousDma(bool isLoad, Value base, int64_t off, uint32_t la, uint32_t bytes);

  // rows of a column slice: one DMA of `rows` rows, `pitch` bytes apart in DDR,
  // contiguous (whole words) in local memory
  bool rowsDma(bool isLoad, const Ddr &r, uint32_t la);

  // an fp32 scalar at a 4-byte (not 8-byte) aligned address: through a PARAM
  // (LDPARAM reads 32-bit words), a word of ones scaled by it (1.0 * v = v)
  bool scalarByParam(const Ddr &r, const LocalBuf &dst);

  bool load(sahl::LoadOp l);

  bool store(sahl::StoreOp s);

  // A row gathered from a packed weight (plan §8.12 C6.0): element k of row r
  // of Wp i8[R, K, D] at Wp[r / D, k, r % D] (the embedding of a model whose
  // classifier shares the table). The tile r / D is K contiguous words; lane
  // r % D of each word is picked by a one-hot word and a REDUCE per word, the
  // K scalars packed into K / D words. r / D and r % D come from the VE
  // (exact in fp32: round(r / D - (D - 1) / 2D)), through the result's DDR
  // (overwritten by the result later) into PARAMs.
  // (the index pattern is checked by gatherForm)
  std::optional<Val> packedRowGather(linalg::GenericOp g, sahl::GatherOp ga, int64_t K, int scalarArgNo);

  // ------------------------------------------------------------ VE
  uint32_t types(const Opd &s1, const std::optional<Opd> &s2, VType out);
  struct DynF {
    int32_t field;
    Value param;
    bool add;
  };
  void ve(const Opd &s1, const std::optional<Opd> &s2, uint32_t dst, VType out, int64_t n, ::sa::VOp op,
          float A = 1.0f, float B = NEG0, ::sa::VFunc fn = ::sa::FUNC_NONE, ::sa::VRed red = ::sa::RED_NONE,
          uint32_t rowlen = 0, uint32_t valid = 0, ArrayRef<DynF> dyn = {}, bool swapneg = false);

  // a buffer read inside a body: its local copy (a DDR one is loaded whole)
  std::optional<LocalBuf> materialize(Value t);

  // one word with the scalar (element 0 of l) in every lane
  std::optional<LocalBuf> scalarBcast(Value v, const LocalBuf &l);
  // one word per element (n values, packed) : replicate each word D times, D x D transposes
  std::optional<LocalBuf> perElementBcast(Value v, const LocalBuf &l, int64_t n);
  // broadcast words (one per element) -> packed: a D x D TRANSPOSE of each
  // block of D words gives D equal words; block k goes to word k (in order,
  // each overwriting the previous block's extra words), so the buffer has D - 1
  // words of room at the end
  LocalBuf packBcast(const LocalBuf &b);

  // ------------------------------------------------------------ scatter (C5.3)
  // one row written at a dynamic index (the KV cache update): LDPARAM of the
  // index * row bytes, a ST with that DDR offset
  // (the form is checked by sa-to-sahl: rowScatter)
  bool scatter(sahl::ScatterOp sc);

  // ------------------------------------------------------------ linalg.generic
  // an i64 scalar: x + c (positions), as C3: the value v in every lane, then
  // (swapneg(v) - v) * -0.5 = [v, 0, v, 0, ...], the i64 (v, 0) in lanes 0, 1
  bool intScalar(linalg::GenericOp g);

  // an i64 scalar in DDR, loaded as two int32 lanes
  std::optional<LocalBuf> scalarI64(Value v);

  // a PARAM of the template's own (from 7 down, by sahw-assign-registers)
  Value privateParam() { return sahw::PrivateOp::create(regB, loc, regB.getType<sahw::ParamType>(), IntegerAttr()); }

  // a row of an [H, T] generic with a dynamic T (or the one row of a [T] one),
  // or of a static [H, C] one with a prefix mask (fixed: VALID is per VE, so
  // the rows go one by one; a row of a buffer is C / D words)
  struct RowSel {
    int64_t row = 0;
    Lin len;
    bool fixed = false;
  };

  bool generic(linalg::GenericOp g, const RowSel *sel = nullptr);
};

}  // namespace mlir::iree_compiler::sa::lower

#endif  // SA_TRANSFORMS_SAHLLOWER_H_

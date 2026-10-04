// The lowering of sahl-to-sahw (docs/iree_compiler_plan.md §8.4 step 10,
// §8.14 C8 R6): one sahw.template per dispatch function, translating the
// decisions the sahl passes made (kernels, gathers, pieces, memory places,
// schedules) into commands. Definitions: SahlToSahw.cpp (the pass, the block
// walk, registers, local memory, DDR / DMA, VE helpers, scatter),
// SahlLowerGeneric.cpp (linalg.generic: element-wise, reductions, gathers),
// SahlLowerKernels.cpp (linear layers, attention, generic contractions).
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
        regB(body), bb(OpBuilder::atBlockEnd(&body.getRegion().front())), loc(f.getLoc()), mem(lay), km(cfg) {
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
  std::map<std::pair<void *, int>, Value> validParams;   // (i64 scalar, predicate) -> VALID count
  KernelMatcher km;                              // the plans of the kernels (SahlKernels.h)
  llvm::DenseMap<Operation *, Operation *> anchorOf;   // sahl.kernel -> its anchor
  sahl::KernelOp curKernel;                      // the kernel being lowered
  // its schedule: as sahl-schedule chose (the attributes), else chosen here
  KernelSchedule scheduleOf(const LinearPlan &p);

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

  // ------------------------------------------------------------ attention (C5.3)
  // scores: bmm(ext(K^T), q) -> [H, T, 1], epilogue (f32(acc) * s_q[h]) * a_k;
  // P V:    bmm(p, ext(V))   -> [H, 1, hs], epilogue f32(acc) * a_v;
  // K / V: i8 [T, H, hs], a slice of the KV cache in DDR, through an extsi
  // generic with map (t, h, j) -> (h, t, j). Per head, as C3 (compile_model.
  // attention): K rows -> TRANSPOSE -> K^T tiles, or V rows loaded interleaved;
  // q_h or p_h as a replicated int8 A strip; EX; one VE for the epilogue. T
  // is dynamic (PARAMs from push constants).

  // the byte offset of a DDR view with static offsets from its binding subspan
  std::optional<std::pair<Value, int64_t>> ddrStart(Value v, int64_t es);

  bool attention(AttnPlan &p);

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

  bool contraction(ContractPlan &p);

  // ------------------------------------------------------------ linear layers (C5.2)
  // y = epilogue(sum_k x[k] * W[n, k], ...): a contraction generic with the
  // packed weights (i8 [N / D, K, D], iree-sa-pack-linear-weights) and x (i8 or
  // int8 values in i32), its int32 accumulator consumed by one element-wise
  // epilogue generic [N / D, D] whose result is stored. Lowered as C3's
  // qlinear schedule (compile_layer.linear): chunks of output tiles, the
  // weights of chunk i + 1 loaded into the other SPAD_B bank while chunk i
  // computes, each chunk's epilogue through the element-wise lowering, many
  // chunks as a LOOP_END loop over pairs, chunk 0's weights in the prefix.

  // prefill's linear layer (plan §8.13): each block of D rows of int8 x is one
  // A strip (an interleaved DMA: block c of K, word r = row r's D elements c,
  // exactly EX's A layout), so the D x D array computes D rows at once and a
  // chunk of weight tiles serves every row block. Per chunk of output tiles:
  // its B tiles and per-column inputs, then per row block the EX (output rows
  // crow = nc words apart), the epilogue over D x nc words (per-column inputs
  // by MOD nc, per-row broadcast words by DIV nc), the store of D rows. The
  // next chunk's weights are loaded during this chunk's EX and epilogues (the
  // other bank, right after its first EX, as decode's linear).
  bool linearRows(LinearPlan &p);

  bool linear(LinearPlan &p);

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

  // a chunk of a linear layer's epilogue: the inputs given, n elements, the
  // result into outLa (of type outVt: fp32, or int32 for an epilogue that only
  // truncates the accumulator)
  struct Chunk {
    std::map<int, Val> inputs;
    int64_t n = 0;
    uint32_t outLa = 0;
    VType outVt = ::sa::VT_F32;
  };

  bool generic(linalg::GenericOp g, const RowSel *sel = nullptr, const Chunk *chunk = nullptr);
};

}  // namespace mlir::iree_compiler::sa::lower

#endif  // SA_TRANSFORMS_SAHLLOWER_H_

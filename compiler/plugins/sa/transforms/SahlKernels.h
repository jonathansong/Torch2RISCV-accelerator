// Kernel matching of the sa code generator (docs/iree_compiler_plan.md §8.14,
// C8 R1): which operations of a sahl function form a micro-kernel (a linear
// layer, attention) or a generic contraction, and the plan each one is lowered
// by. Shared by sa-to-sahl (which groups each match into a sahl.kernel) and
// sahl-to-sahw (which lowers the plans).
#ifndef SA_TRANSFORMS_SAHLKERNELS_H_
#define SA_TRANSFORMS_SAHLKERNELS_H_

#include <optional>

#include "../target/Layout.h"
#include "SaLin.h"
#include "SahlDialect.h"
#include "SahwPasses.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::iree_compiler::sa {

// An fp32 constant.
std::optional<float> constF32(Value v);
// The Lin of a dynamic size (a push constant, possibly through memref.dim of a binding).
std::optional<Lin> dynLin(Value size);
// The sahl.load that fills a local buffer (null if none).
sahl::LoadOp loadInto(Value local);

// ------------------------------------------------------------ attention (C5.3)
// scores: bmm(ext(K^T), q) -> [H, T, 1], epilogue (f32(acc) * s_q[h]) * a_k;
// P V:    bmm(p, ext(V))   -> [H, 1, hs], epilogue f32(acc) * a_v;
// K / V: i8 [T, H, hs], a slice of the KV cache in DDR, through an extsi
// generic with map (t, h, j) -> (h, t, j).
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

// ------------------------------------------------------------ contractions (C5.4, generic)
// y[b, n] = sum_k x[b, k] * M[b, n, k] over int8 values (int32 accumulator),
// the result consumed by one element-wise epilogue that is stored; each operand
// followed back to DDR (see SahlToSahw.cpp, contraction()).
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

// the loop ranges of a linalg op: static, or a Lin (dynamic)
struct Range {
  int64_t size = 1;
  std::optional<Lin> dyn;
};

// ------------------------------------------------------------ linear layers (C5.2)
// y = epilogue(sum_k x[k] * W[n, k], ...): a contraction generic with the
// packed weights (i8 [N / D, K, D]) and x (i8 or int8 values in i32), its
// int32 accumulator consumed by one element-wise epilogue whose result is
// stored (or, prefill's rows form, the accumulator stored as it is).
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

class KernelMatcher {
 public:
  explicit KernelMatcher(const TargetConfig &cfg)
      : cfg(cfg), d(cfg.d), lay(uint32_t(cfg.d), uint32_t(cfg.spadBytes), uint32_t(cfg.accBytes)) {}

  // Every match of a function, in the lowering's order: the linear layers and
  // attention (as the target configuration's micro-kernels allow), then the
  // other contractions.
  void matchAll(func::FuncOp f);
  // Each records its plan and the operations it covers (nothing else may use its buffers).
  bool matchLinear(linalg::GenericOp con);
  bool matchAttention(linalg::BatchMatmulOp bmm);
  bool matchContraction(linalg::LinalgOp op);

  std::optional<SmallVector<Range>> loopRanges(linalg::LinalgOp op);
  // an operand of the contraction back to DDR: its element offset per loop
  std::optional<Operand> operandOf(linalg::LinalgOp op, int input);

  llvm::DenseMap<Operation *, LinearPlan> linears;
  llvm::DenseMap<Operation *, AttnPlan> attns;
  llvm::DenseMap<Operation *, ContractPlan> contracts;
  llvm::DenseSet<Operation *> owned;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> covers;   // anchor -> the operations of its match

 private:
  void record(Operation *anchor, ArrayRef<Operation *> ops);
  TargetConfig cfg;
  int64_t d;
  ::sa::Layout lay;
};

}  // namespace mlir::iree_compiler::sa

#endif  // SA_TRANSFORMS_SAHLKERNELS_H_

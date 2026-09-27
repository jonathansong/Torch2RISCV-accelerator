// Dispatch matching (Match.h).
#include "Match.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir::iree_compiler::sa {

using IREE::TensorExt::DispatchTensorLoadOp;
using IREE::TensorExt::DispatchTensorStoreOp;

namespace {

// Binding ordinal behind a dispatch.tensor.load / store of the whole tensor
// at byte offset 0; -1 otherwise.
template <typename OpTy>
int wholeBinding(OpTy op, Value source, RankedTensorType type) {
  auto subspan = source.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  if (!subspan) return -1;
  if (Value off = subspan.getByteOffset()) {
    std::optional<int64_t> c = getConstantIntValue(off);
    if (!c || *c != 0) return -1;
  }
  for (int64_t o : op.getStaticOffsets())
    if (o != 0) return -1;
  for (int64_t s : op.getStaticStrides())
    if (s != 1) return -1;
  if (op.getStaticSizes() != type.getShape()) return -1;
  return int(subspan.getBinding().getZExtValue());
}

int loadBinding(Value v, RankedTensorType &type) {
  auto load = v.getDefiningOp<DispatchTensorLoadOp>();
  if (!load) return -1;
  type = load.getType();
  return wholeBinding(load, load.getSource(), type);
}

// The operations of a linalg body, without the terminator.
SmallVector<Operation *> bodyOps(linalg::GenericOp g) {
  SmallVector<Operation *> ops;
  for (Operation &op : g.getRegion().front().without_terminator()) ops.push_back(&op);
  return ops;
}

bool isIdentity(AffineMap m) { return m.isIdentity(); }

}  // namespace

bool matchQLinear(FunctionOpInterface func, int64_t d, ::sa::QLinear &out, std::string &why) {
  // the single store
  DispatchTensorStoreOp store;
  int stores = 0;
  func.walk([&](DispatchTensorStoreOp s) {
    store = s;
    ++stores;
  });
  if (stores != 1) return why = "not exactly one result", false;
  auto yType = cast<RankedTensorType>(store.getValue().getType());
  int bY = wholeBinding(store, store.getTarget(), yType);
  if (bY < 0) return why = "the result is not stored whole at offset 0", false;

  // epilogue: y = float(trunci(acc)) * s_w * s_x
  auto epi = store.getValue().getDefiningOp<linalg::GenericOp>();
  if (!epi || !epi.isAllParallelLoops() || epi.getNumDpsInputs() != 3 || epi.getNumDpsInits() != 1)
    return why = "the result is not an element-wise epilogue with three inputs", false;
  auto epiMaps = epi.getIndexingMapsArray();
  if (!isIdentity(epiMaps[0]) || !isIdentity(epiMaps[1]) || epiMaps[2].getNumResults() != 0 ||
      !isIdentity(epiMaps[3]))
    return why = "epilogue indexing is not (acc, s_w, scalar s_x)", false;
  {
    Block &b = epi.getRegion().front();
    SmallVector<Operation *> ops = bodyOps(epi);
    if (ops.size() != 4) return why = "epilogue body is not trunci, sitofp, mulf, mulf", false;
    auto tr = dyn_cast<arith::TruncIOp>(ops[0]);
    auto cv = dyn_cast<arith::SIToFPOp>(ops[1]);
    auto m1 = dyn_cast<arith::MulFOp>(ops[2]);
    auto m2 = dyn_cast<arith::MulFOp>(ops[3]);
    auto yield = cast<linalg::YieldOp>(b.getTerminator());
    auto uses = [](arith::MulFOp m, Value a, Value c) {
      return (m.getLhs() == a && m.getRhs() == c) || (m.getLhs() == c && m.getRhs() == a);
    };
    if (!tr || !cv || !m1 || !m2 || tr.getIn() != b.getArgument(0) || !tr.getType().isInteger(32) ||
        cv.getIn() != tr.getResult() || !cv.getType().isF32() || !uses(m1, cv.getResult(), b.getArgument(1)) ||
        !uses(m2, m1.getResult(), b.getArgument(2)) || yield.getOperand(0) != m2.getResult())
      return why = "epilogue is not (f32(i32(acc)) * s_w) * s_x", false;
    if (m1.getFastmathAttr() && m1.getFastmath() != arith::FastMathFlags::none)
      return why = "epilogue uses fast-math flags", false;
  }
  RankedTensorType swType, sxType, xType, wType;
  int bSw = loadBinding(epi.getDpsInputs()[1], swType);
  int bSx = loadBinding(epi.getDpsInputs()[2], sxType);
  if (bSw < 0 || bSx < 0 || !swType.getElementType().isF32() || !sxType.getElementType().isF32() ||
      sxType.getRank() != 0)
    return why = "s_w / s_x are not whole f32 bindings", false;

  // contraction: acc[t, j] += extsi(x[k]) * extsi(Wp[t, k, j])
  auto con = epi.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!con || con.getNumDpsInputs() != 2 || con.getNumDpsInits() != 1 || con.getNumLoops() != 3)
    return why = "the epilogue input is not a two-operand contraction", false;
  auto iters = con.getIteratorTypesArray();
  if (iters[0] != utils::IteratorType::parallel || iters[1] != utils::IteratorType::parallel ||
      iters[2] != utils::IteratorType::reduction)
    return why = "contraction iterators are not (parallel, parallel, reduction)", false;
  {
    MLIRContext *ctx = func->getContext();
    AffineExpr t, j, k;
    bindDims(ctx, t, j, k);
    auto maps = con.getIndexingMapsArray();
    if (maps[0] != AffineMap::get(3, 0, {k}, ctx) || maps[1] != AffineMap::get(3, 0, {t, k, j}, ctx) ||
        maps[2] != AffineMap::get(3, 0, {t, j}, ctx))
      return why = "contraction indexing is not x[k], Wp[t, k, j] -> acc[t, j] (weights not packed?)", false;
    Block &b = con.getRegion().front();
    SmallVector<Operation *> ops = bodyOps(con);
    Value xa = b.getArgument(0), wa = b.getArgument(1), acc = b.getArgument(2);
    size_t i = 0;
    Value xe = xa, we = wa;
    if (i < ops.size())
      if (auto e = dyn_cast<arith::ExtSIOp>(ops[i]); e && e.getIn() == xa) xe = e.getResult(), ++i;
    if (i < ops.size())
      if (auto e = dyn_cast<arith::ExtSIOp>(ops[i]); e && e.getIn() == wa) we = e.getResult(), ++i;
    auto mul = i < ops.size() ? dyn_cast<arith::MulIOp>(ops[i++]) : nullptr;
    auto add = i < ops.size() ? dyn_cast<arith::AddIOp>(ops[i++]) : nullptr;
    auto yield = cast<linalg::YieldOp>(b.getTerminator());
    if (i != ops.size() || !mul || !add ||
        !((mul.getLhs() == xe && mul.getRhs() == we) || (mul.getLhs() == we && mul.getRhs() == xe)) ||
        !((add.getLhs() == acc && add.getRhs() == mul.getResult()) ||
          (add.getLhs() == mul.getResult() && add.getRhs() == acc)) ||
        yield.getOperand(0) != add.getResult())
      return why = "contraction body is not acc + ext(x) * ext(w)", false;
    if (!wa.getType().isInteger(8)) return why = "weights are not i8", false;
    // the accumulator must be at least 32 bits (int8 x int8 over k < 2^17 fits i32)
    auto accT = dyn_cast<IntegerType>(acc.getType());
    if (!accT || accT.getWidth() < 32) return why = "accumulator narrower than 32 bits", false;
  }
  auto fill = con.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill) return why = "the accumulator is not initialized by a fill", false;
  std::optional<int64_t> zero = getConstantIntValue(fill.getDpsInputs()[0]);
  if (!zero || *zero != 0) return why = "the accumulator does not start at 0", false;

  // x: i32 load, or i8 load + extsi generic
  Value xv = con.getDpsInputs()[0];
  bool xI32 = true;
  if (auto ext = xv.getDefiningOp<linalg::GenericOp>()) {
    SmallVector<Operation *> ops = bodyOps(ext);
    if (ext.getNumDpsInputs() != 1 || ops.size() != 1 || !isa<arith::ExtSIOp>(ops[0]) ||
        !isIdentity(ext.getIndexingMapsArray()[0]))
      return why = "x is not a load or a sign extension of a load", false;
    xv = ext.getDpsInputs()[0];
  }
  int bX = loadBinding(xv, xType);
  int bW = loadBinding(con.getDpsInputs()[1], wType);
  if (bX < 0 || bW < 0) return why = "x / weights are not whole bindings", false;
  if (xType.getElementType().isInteger(8))
    xI32 = false;
  else if (!xType.getElementType().isInteger(32))
    return why = "x is neither i8 nor i32", false;
  if (xType.getRank() != 1 || wType.getRank() != 3 || !wType.hasStaticShape() || !xType.hasStaticShape())
    return why = "unexpected ranks or dynamic shapes", false;
  int64_t k = xType.getDimSize(0), nt = wType.getDimSize(0);
  if (wType.getDimSize(1) != k || wType.getDimSize(2) != d)
    return why = "packed weights are not [n/D, k, D] for this D", false;
  if (yType.getNumElements() != nt * d || swType.getNumElements() != nt * d)
    return why = "output / s_w sizes do not match the weights", false;
  if (!yType.getElementType().isF32()) return why = "result is not f32", false;

  out.k = uint32_t(k);
  out.n = uint32_t(nt * d);
  out.xI32 = xI32;
  out.bX = bX;
  out.bW = bW;
  out.bSw = bSw;
  out.bSx = bSx;
  out.bY = bY;
  return true;
}

}  // namespace mlir::iree_compiler::sa

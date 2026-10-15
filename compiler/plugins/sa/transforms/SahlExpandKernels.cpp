// sahl-expand-kernels (docs/iree_compiler_plan.md §8.17 item 2, §8.20): a
// micro-kernel's schedule written out in sahl, so that sahl-to-sahw only
// translates it (one command per operation) and the schedule is visible in
// the IR: its buffers and their places, the order of the loads (the
// prefetch), the scopes that release local memory.
//
// Prefill's linear layer (M rows of int8 x, sahl.kernel "linear" with
// chunk_tiles from sahl-schedule), as the micro-kernel of C6.P:
//   - the A strips of the row blocks (sahl.strip);
//   - the per-row and scalar epilogue inputs loaded once and put in the
//     broadcast layout (sahl.bcast);
//   - two sets of per-column input buffers and two SPAD_B banks of weight
//     tiles: chunk ci uses set / bank ci & 1, the next chunk's loads follow the
//     first row block's EX of this chunk (the prefetch);
//   - per chunk and row block, a sahl.scope (its local memory released after
//     it): the EX (sahl.mma), the epilogue as a generic over the chunk
//     [D rows, nc tiles, D] of the local buffers, the store.
//   A K beyond one tile's (rowsKBlock: SPAD_B bank, K x D <= 65535; Llama's
//   down projection, K = 8192) in equal blocks (docs/generality_implementation_plan.md
//   WP6): a strip per row block and block, the tiles of each (chunk, row
//   block, block) loaded in turn into alternating banks, one EX per block
//   (the first writes the accumulator, the others add).
// Decode's linear layer (one row of x), as the micro-kernel of C3 / C4:
//   - x loaded, then copied over the D rows into the A strip (sahl.strip of
//     a local vector), the scalar epilogue inputs put in the broadcast layout;
//   - chunk ci's accumulator, per-chunk inputs and result at fixed words of
//     ACC bank ci & 1 (sa.word: the C3 layout), its weights in SPAD_B bank
//     ci & 1, the epilogue's temps in the same ACC bank first
//     (sa.prefer_bank);
//   - with many chunks (sahl-schedule's `loop`) the first pair inside a
//     sahl.loop (LOOP_END): each DMA marked sa.advance moves on by two
//     chunks per iteration; the rest after it;
//   - chunk 0's weights in the template's prefix (sa.prefix) when the kernel
//     is the first operation that emits commands (they may then load during
//     an earlier dispatch).
// A generic contraction (sahl.kernel "contraction", static sizes: the
// micro-kernels' fallback, attention without its micro-kernel, the linear
// layers with ukernels=none), as its lowering of C5.4:
//   - per-row and scalar epilogue inputs loaded once, broadcast;
//   - SPAD_A claimed (sahl.claim_spad): the A strip at word 0, x (int8) or
//     the raw rows of a row-major matrix in bank 1 (sa.word);
//   - per batch b: x's row -> the strip (per row g when there are several
//     rows of x); per chunk of output tiles its B tiles in SPAD_B, the banks
//     alternating per load (packed: one row per tile; row-major: rows
//     transposed; K-major: rows interleaved); K blocks accumulated;
//   - per chunk and row a sahl.scope {keep}: EX, the epilogue as a 1-D
//     generic over the chunk (per-row inputs: their word for (b, g)), the
//     store; an int8-x layout's SPAD temps after x (spad_from).
// Attention (sahl.kernel "attention": decode's micro-kernel of C3, one
// query row per head, the cache's length T dynamic): SPAD_A claimed, per head
// the scores q . K^T (K's T rows into bank 1, transposed into SPAD_B, the
// query copied over the D rows, EX, times s_q and the scale into the head's
// row of T scores; the H rows stored back to back) or P . V (the H rows of p
// loaded back to back, per head p copied over the D rows, V's T rows
// interleaved into SPAD_B bank 1, EX, the scale, the head's row stored).
// Dynamic sizes are views and buffers sized by T (the fields at the bound,
// the count a dynamic field).
// The cache's T beyond one SPAD_B bank (C6.5, docs/iree_compiler_plan.md
// §8.22): a contraction whose dynamic N is T (q . K^T, K's rows transposed)
// goes over T in blocks of Tb rows (128 by default, --iree-sa-attn-block; Tb x K
// within one bank) in a sahl.loop whose
// count comes from T (ceil(T / Tb), T a multiple of D: ((T (M - 1)) >> log2(M Tb))
// + 1 within the setup entry's 15-bit shift, else T >> log2 Tb, + 1: a T that is a
// multiple of Tb then computes one block of scores past T, not stored): per block
// the rows loaded and transposed, one EX per row of x (the strips of all rows
// made before the loop), the DMA moving on with the loop; then per row the
// epilogue over T and the store. An EX writes its D output rows one after the
// other (Tb / D words each): row g's EX goes to slot g of a scratch of
// G - 1 + D slots, its copies into slots g + 1, ... (each rewritten by its own
// EX later in the block), and a VE copy moves slot g into row g's buffer at the
// block's place (its destination moving on with the loop).
// Only integer accumulations are split, so the scores are those of the
// untiled form, bit for bit. A dynamic K (p . V) with several chunks of
// output tiles walks the rows of x again per chunk (the cursor repositioned).
// The order of the operations and of the first uses of the local buffers is
// the order in which the lowering emitted the commands and allocated the
// words before, so the descriptors are the same (the golden corpus).
#include <cstdlib>
#include "SahlDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "SahlKernels.h"
#include "SahlLocalMemory.h"
#include "SahlPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::iree_compiler::sa {
namespace {

struct Expander {
  OpBuilder b;
  Location loc;
  int64_t d;

  Value sub(Value v, ArrayRef<int64_t> offs, ArrayRef<int64_t> sizes) {
    SmallVector<OpFoldResult> o, s, st;
    for (int64_t x : offs) o.push_back(b.getIndexAttr(x));
    for (int64_t x : sizes) s.push_back(b.getIndexAttr(x));
    for (size_t i = 0; i < offs.size(); ++i) st.push_back(b.getIndexAttr(1));
    return memref::SubViewOp::create(b, loc, v, o, s, st);
  }
  Value alloc(ArrayRef<int64_t> shape, Type et, StringRef mem, StringRef layout) {
    auto a = memref::AllocOp::create(b, loc, MemRefType::get(shape, et));
    a->setAttr("sa.mem", b.getStringAttr(mem));
    if (!layout.empty()) a->setAttr("sa.layout", b.getStringAttr(layout));
    return a;
  }
  // a view of v whose dimensions follow loops (map: dimension -> loop):
  // per loop an offset and size (a loop in `dyn`: from 0, its size that
  // value), others whole; dimensions of the loops in `drop` and constant ones removed
  Value view(Value v, AffineMap map, const llvm::DenseMap<int, std::pair<int64_t, int64_t>> &at,
             ArrayRef<int> drop, const llvm::DenseMap<int, Value> &dyn = llvm::DenseMap<int, Value>()) {
    auto mt = cast<MemRefType>(v.getType());
    SmallVector<OpFoldResult> o, sz, st;
    SmallVector<int64_t> shape;
    for (unsigned r = 0; r < map.getNumResults(); ++r) {
      int64_t off = 0, size = mt.getDimSize(r);
      bool keep = true;
      if (auto de = dyn_cast<AffineDimExpr>(map.getResult(r))) {
        int L = int(de.getPosition());
        if (auto it = at.find(L); it != at.end()) off = it->second.first, size = it->second.second;
        keep = !llvm::is_contained(drop, L);
        if (auto it = dyn.find(L); it != dyn.end()) {
          o.push_back(b.getIndexAttr(0));
          sz.push_back(it->second);
          st.push_back(b.getIndexAttr(1));
          if (keep) shape.push_back(ShapedType::kDynamic);
          continue;
        }
      } else {
        keep = false, size = 1;
      }
      o.push_back(b.getIndexAttr(off));
      sz.push_back(b.getIndexAttr(size));
      st.push_back(b.getIndexAttr(1));
      if (keep) shape.push_back(size);
    }
    auto rt = memref::SubViewOp::inferRankReducedResultType(shape, mt, o, sz, st);
    return memref::SubViewOp::create(b, loc, cast<MemRefType>(rt), v, o, sz, st);
  }
  // the dimension of rows of a dynamic length: the one dynamic dimension,
  // only unit dimensions after it (-1: another form)
  static int64_t rowDim(MemRefType mt) {
    int64_t dd = -1;
    for (int64_t r = 0; r < mt.getRank(); ++r) {
      if (mt.isDynamicDim(r)) {
        if (dd >= 0) return -1;
        dd = r;
      } else if (dd >= 0 && mt.getDimSize(r) != 1) {
        return -1;
      }
    }
    return dd;
  }
  // row 0 of rows of a dynamic length (rowDim, `len` long) a dynamic stride
  // apart: [?] (the row cursor's view)
  Value row0(Value v, Value len) {
    auto mt = cast<MemRefType>(v.getType());
    int64_t dd = rowDim(mt);
    SmallVector<OpFoldResult> o, sz, st;
    for (int64_t r = 0; r < mt.getRank(); ++r) {
      o.push_back(b.getIndexAttr(0));
      sz.push_back(r != dd ? OpFoldResult(b.getIndexAttr(1)) : OpFoldResult(len));
      st.push_back(b.getIndexAttr(1));
    }
    auto rt = memref::SubViewOp::inferRankReducedResultType({ShapedType::kDynamic}, mt, o, sz, st);
    return memref::SubViewOp::create(b, loc, cast<MemRefType>(rt), v, o, sz, st);
  }
  // [0, len) of a local 1-D buffer
  Value prefix(Value v, Value len) {
    SmallVector<OpFoldResult> o{b.getIndexAttr(0)}, sz{len}, st{b.getIndexAttr(1)};
    auto rt = memref::SubViewOp::inferRankReducedResultType({ShapedType::kDynamic}, cast<MemRefType>(v.getType()), o, sz, st);
    return memref::SubViewOp::create(b, loc, cast<MemRefType>(rt), v, o, sz, st);
  }
  // rows [r0, r0 + D) and tiles [c0, c0 + nc) of a [M, nt, D] or [M, N] view
  Value rowsTiles(Value v, int64_t r0, int64_t c0, int64_t nc) {
    auto mt = cast<MemRefType>(v.getType());
    if (mt.getRank() == 3) return sub(v, {r0, c0, 0}, {d, nc, d});
    return sub(v, {r0, c0 * d}, {d, nc * d});
  }
};

// The kernel's operations and the buffers only they used.
void eraseKernel(sahl::KernelOp k) {
  SmallVector<Operation *> allocs;
  k.walk([&](Operation *o) {
    for (Value v : o->getOperands())
      if (auto a = v.getDefiningOp<memref::AllocOp>(); a && a->getBlock() == k->getBlock()) allocs.push_back(a);
  });
  k.erase();
  for (Operation *a : allocs)
    if (a->getBlock() && a->use_empty()) a->erase();
}

// Prefill's linear layer (LinearPlan with rows): false when it has a form the
// expansion does not write out (the kernel then stays for the lowering).
bool expandLinearRows(sahl::KernelOp k, LinearPlan p, const ::sa::Layout &lay, int64_t d) {
  auto nca = k->getAttrOfType<IntegerAttr>("chunk_tiles");
  if (!nca) return false;
  auto wt = cast<MemRefType>(p.w.getType());
  const int64_t K = wt.getDimSize(1), nt = wt.getDimSize(0), M = p.rows, nb = M / d, nc = nca.getInt();
  const int64_t kc = rowsKBlock(K, lay, d);
  if (nc <= 0 || nt % nc || !kc) return false;
  const int64_t nch = nt / nc, nk = K / kc;
  if (!cast<MemRefType>(p.acc.getType()).getElementType().isInteger(32)) return false;
  linalg::GenericOp epi = p.epi;
  llvm::DenseMap<int, int> kindOf;                  // epilogue input -> 1 chunked, 2 column, 3 row, 4 scalar
  llvm::DenseMap<int, Value> srcOf;
  for (auto &[i, s] : p.chunked) kindOf[i] = 1, srcOf[i] = s;
  for (auto &[i, s] : p.cols) kindOf[i] = 2, srcOf[i] = s;
  for (auto &[i, s] : p.rowv) kindOf[i] = 3, srcOf[i] = s;
  for (auto &[i, s] : p.whole) {
    if (cast<MemRefType>(s.getType()).getRank() != 0) return false;
    kindOf[i] = 4, srcOf[i] = s;
  }
  Expander e{OpBuilder(k), k.getLoc(), d};
  MLIRContext *ctx = k.getContext();
  Type f32 = Float32Type::get(ctx), i8 = IntegerType::get(ctx, 8), i32 = IntegerType::get(ctx, 32);
  // the A strips of the row blocks (per K block)
  SmallVector<SmallVector<Value>> strips(nb);
  for (int64_t rb = 0; rb < nb; ++rb)
    for (int64_t kb = 0; kb < nk; ++kb) {
      Value s = e.alloc({d, kc}, i8, "spad_a", "strip");
      sahl::StripOp::create(e.b, e.loc, e.sub(p.x, {rb * d, kb * kc}, {d, kc}), s);
      strips[rb].push_back(s);
    }
  // per-row and scalar inputs: loaded once, in the broadcast layout
  llvm::DenseMap<int, Value> bc;
  for (auto &[i, s] : p.rowv) {
    Value v = e.alloc({M}, f32, "acc", "packed");
    sahl::LoadOp::create(e.b, e.loc, s, v);
    Value w = e.alloc({M}, f32, "acc", "bcast");
    sahl::BcastOp::create(e.b, e.loc, v, w);
    bc[i] = w;
  }
  for (auto &[i, s] : p.whole) {
    Value v = e.alloc({}, f32, "acc", "packed");
    sahl::LoadOp::create(e.b, e.loc, s, v);
    Value w = e.alloc({}, f32, "acc", "bcast");
    sahl::BcastOp::create(e.b, e.loc, v, w);
    bc[i] = w;
  }
  // two sets of per-column buffers, two banks of weight tiles
  SmallVector<SmallVector<Value>> cols(2);
  for (int set = 0; set < 2; ++set)
    for (size_t m = 0; m < p.cols.size(); ++m) {
      Value c = e.alloc({nc, d}, f32, "acc", "packed");
      sahl::ReserveOp::create(e.b, e.loc, c, false);
      cols[set].push_back(c);
    }
  Value tiles[2];
  for (int bank = 0; bank < 2; ++bank) {
    tiles[bank] = e.alloc({nc, kc, d}, i8, "spad_b", "");
    tiles[bank].getDefiningOp()->setAttr("sa.bank", e.b.getI64IntegerAttr(bank));
  }
  // K block kb of chunk ci's tiles into a bank; a chunk's first block with its per-column inputs
  auto loadTiles = [&](int64_t ci, int64_t kb, int bank, bool withCols = true) {
    int64_t c0 = ci * nc;
    sahl::LoadOp::create(e.b, e.loc, e.sub(p.w, {c0, kb * kc, 0}, {nc, kc, d}), tiles[bank]);
    if (kb == 0 && withCols)
      for (size_t m = 0; m < p.cols.size(); ++m)
        sahl::LoadOp::create(e.b, e.loc, e.sub(p.cols[m].second, {c0, 0}, {nc, d}), cols[ci & 1][m]);
  };
  // one K block: the tiles of chunk ci serve all its row blocks (bank ci & 1).
  // K blocks: the tiles of each (chunk, row block, block) in turn, the banks
  // alternating per load (the next load after the current EX), the blocks
  // accumulated (int32, |sum| <= K x 2^14)
  int bank = 0;
  loadTiles(0, 0, 0);
  Value y = p.store.getDst();
  for (int64_t ci = 0; ci < nch; ++ci) {
    const int64_t c0 = ci * nc;
    for (int64_t rb = 0; rb < nb; ++rb) {
      auto scope = sahl::ScopeOp::create(e.b, e.loc, /*keep=*/true, IntegerAttr());
      OpBuilder::InsertionGuard guard(e.b);
      e.b.setInsertionPointToEnd(&scope.getBody().emplaceBlock());
      Value acc = e.alloc({d, nc, d}, i32, "acc", "packed");
      if (nk == 1) {
        sahl::MmaOp::create(e.b, e.loc, strips[rb][0], tiles[ci & 1], acc, false);
        if (rb == 0 && ci + 1 < nch) loadTiles(ci + 1, 0, int((ci + 1) & 1));   // the prefetch into the other bank
      } else {
        for (int64_t kb = 0; kb < nk; ++kb) {
          sahl::MmaOp::create(e.b, e.loc, strips[rb][kb], tiles[bank], acc, kb > 0);
          bank ^= 1;
          if (kb + 1 < nk) loadTiles(ci, kb + 1, bank);
          else if (rb + 1 < nb) loadTiles(ci, 0, bank, /*withCols=*/false);   // (this chunk's, still in use)
          else if (ci + 1 < nch) loadTiles(ci + 1, 0, bank);
        }
      }
      Value ys = e.rowsTiles(y, rb * d, c0, nc);
      if (!epi) {
        sahl::StoreOp::create(e.b, e.loc, acc, ys);
        continue;
      }
      // the epilogue on the chunk: the result allocated first (as the lowering did)
      Type ot = cast<MemRefType>(epi.getDpsInits()[0].getType()).getElementType();
      Value out = e.alloc({d, nc, d}, ot, ot.isInteger(8) ? "spad_a" : "acc", "packed");
      sahl::ReserveOp::create(e.b, e.loc, out, false);
      SmallVector<Value> ins;
      int cm = 0;
      for (int in = 0; in < epi.getNumDpsInputs(); ++in) {
        if (epi.getDpsInputs()[in] == p.acc) {
          ins.push_back(acc);
          continue;
        }
        switch (kindOf.lookup(in)) {
        case 1: {
          Type et = cast<MemRefType>(epi.getDpsInputs()[in].getType()).getElementType();
          Value l = e.alloc({d, nc, d}, et, "acc", "packed");
          sahl::LoadOp::create(e.b, e.loc, e.rowsTiles(srcOf[in], rb * d, c0, nc), l);
          ins.push_back(l);
          break;
        }
        case 2: ins.push_back(cols[ci & 1][cm++]); break;
        case 3: ins.push_back(e.sub(bc[in], {rb * d}, {d})); break;
        case 4: ins.push_back(bc[in]); break;
        default: return false;
        }
      }
      auto g = linalg::GenericOp::create(e.b, e.loc, TypeRange{}, ins, ValueRange{out}, epi.getIndexingMapsArray(),
                                         epi.getIteratorTypesArray());
      IRMapping map;
      epi.getRegion().cloneInto(&g.getRegion(), map);
      sahl::StoreOp::create(e.b, e.loc, out, ys);
    }
  }
  eraseKernel(k);
  return true;
}

// Whether nothing before op in its block emits commands (the lowering's
// template body is still empty when it reaches op).
bool firstToEmit(Operation *op) {
  static const char *quiet[] = {"arith.constant", "memref.alloc", "memref.dealloc", "memref.subview", "memref.cast",
                                "memref.dim", "memref.collapse_shape", "hal.interface.binding.subspan",
                                "hal.interface.constant.load", "iree_tensor_ext.dispatch.workload.ordinal",
                                "arith.index_cast", "arith.index_castui", "arith.extui", "arith.shli", "arith.ori",
                                "linalg.fill"};
  if (op->getParentOp() && !isa<func::FuncOp>(op->getParentOp())) return false;
  for (Operation *o = op->getPrevNode(); o; o = o->getPrevNode()) {
    StringRef n = o->getName().getStringRef();
    if (o->getDialect() && o->getDialect()->getNamespace() == "util") continue;
    if (!llvm::is_contained(quiet, n)) return false;
  }
  return true;
}

// Decode's linear layer (LinearPlan without rows).
bool expandLinear(sahl::KernelOp k, LinearPlan p, const ::sa::Layout &lay, int64_t d, int64_t gemvPorts) {
  auto nca = k->getAttrOfType<IntegerAttr>("chunk_tiles");
  if (!nca) return false;
  auto wt = cast<MemRefType>(p.w.getType());
  const int64_t K = wt.getDimSize(1), nt = wt.getDimSize(0), nc = nca.getInt();
  if (nc <= 0 || nt % nc || K % d) return false;
  const int64_t nch = nt / nc, cb = lay.cbank, oSw = d * nc, oY = d * nc + nc;
  const int64_t wbytes = nc * K * d, fbytes = 4 * nc * d;
  auto la = k->getAttrOfType<BoolAttr>("loop");
  linalg::GenericOp epi = p.epi;
  // an int8 result (to_i8 into the KV cache) in SPAD_A: its DDR stride per
  // chunk (nc x D bytes) is a third one beside the weights' and the fp32
  // inputs' (LOOP_END has two), so its chunks are written out
  const bool i8Out = epi && cast<MemRefType>(epi.getDpsInits()[0].getType()).getElementType().isInteger(8);
  const bool useLoop = la && la.getValue() && !i8Out;
  for (auto &[i, s] : p.whole)
    if (cast<MemRefType>(s.getType()).getRank() != 0) return false;
  if (epi)                                        // every epilogue input of a form written out below
    for (int in = 0; in < epi.getNumDpsInputs(); ++in)
      if (epi.getDpsInputs()[in] != p.acc && !llvm::any_of(p.whole, [&](auto &w) { return w.first == in; }) &&
          !llvm::any_of(p.chunked, [&](auto &c) { return c.first == in; }))
        return false;
  auto xt = cast<MemRefType>(p.x.getType());
  auto yt = cast<MemRefType>(p.store.getDst().getType());
  if (xt.getRank() != 1 || yt.getRank() != 2) return false;
  Type xet = xt.getElementType();
  // K2b: the streaming GEMV unit (int8 x): no A strip, no SPAD_B tiles; each
  // chunk's weights stream into the unit (groups of <= 8 strips). A strip is at
  // most 65535 bytes (row_bytes, K x D): a longer K in blocks of at most 4080
  // (a multiple of D), the first writing the accumulator, the others adding to
  // it (acc; int32, |sum| <= K x 2^14, no overflow below K = 2^17); the
  // epilogue reads it after the last (docs/generality_implementation_plan.md WP6)
  const bool useGemv = gemvPorts != 0;
  const int64_t kBlock = std::min<int64_t>(K, 65535 / d / d * d);
  int64_t ncg = std::min<int64_t>(nc, 8);
  while (nc % ncg) --ncg;
  const bool hoist = firstToEmit(k) && !useGemv;
  Expander e{OpBuilder(k), k.getLoc(), d};
  MLIRContext *ctx = k.getContext();
  Type f32 = Float32Type::get(ctx), i8 = IntegerType::get(ctx, 8);
  // x -> the A strip (the strip's words first, as the lowering allocated them)
  Value strip;
  if (!useGemv) {
    strip = e.alloc({d, K}, i8, "spad_a", "strip");
    sahl::ReserveOp::create(e.b, e.loc, strip, false);
  }
  Value xl = e.alloc({K}, xet, xet.isInteger(8) ? "spad_a" : "acc", "packed");
  sahl::LoadOp::create(e.b, e.loc, p.x, xl);
  if (!useGemv) {
    sahl::StripOp::create(e.b, e.loc, xl, strip);
  } else if (!xet.isInteger(8)) {
    // int32 x (in ACC) -> int8 packed in SPAD_A for the unit (the strip's conversion, one row)
    Value xs = e.alloc({K}, i8, "spad_a", "packed");
    sahl::StripOp::create(e.b, e.loc, xl, xs)->setAttr("sa.packed", e.b.getUnitAttr());
    xl = xs;
  }
  // scalar epilogue inputs: loaded once, in the broadcast layout
  llvm::DenseMap<int, Value> bc;
  for (auto &[i, s] : p.whole) {
    Value v = e.alloc({}, f32, "acc", "packed");
    sahl::LoadOp::create(e.b, e.loc, s, v);
    Value w = e.alloc({}, f32, "acc", "bcast");
    sahl::BcastOp::create(e.b, e.loc, v, w);
    bc[i] = w;
  }
  // the chunk buffers of the two banks (C3's ACC layout) and the weight banks
  auto placed = [&](ArrayRef<int64_t> shape, Type et, int64_t word) {
    Value v = e.alloc(shape, et, "acc", "packed");
    v.getDefiningOp()->setAttr("sa.word", e.b.getI64IntegerAttr(word));
    return v;
  };
  auto slot = [&](size_t m) { return m == 0 ? oSw : oY + nc * int64_t(m); };
  Value acc[2], out[2], tiles[2];
  SmallVector<Value> cols[2];
  for (int bank = 0; bank < 2; ++bank) {
    const int64_t c = bank * cb;
    acc[bank] = placed({nc, d}, cast<MemRefType>(p.acc.getType()).getElementType(), c);
    out[bank] = i8Out ? e.alloc({nc, d}, i8, "spad_a", "packed") : placed({nc, d}, f32, c + oY);
    for (size_t m = 0; m < p.chunked.size(); ++m) {
      Type et = cast<MemRefType>(epi.getDpsInputs()[p.chunked[m].first].getType()).getElementType();
      cols[bank].push_back(placed({nc, d}, et, c + slot(m)));
    }
    if (useGemv) continue;
    tiles[bank] = e.alloc({nc, K, d}, i8, "spad_b", "");
    tiles[bank].getDefiningOp()->setAttr("sa.bank", e.b.getI64IntegerAttr(bank));
  }
  Value y = p.store.getDst();
  auto adv = [&](Operation *o, bool dyn, int j) {
    if (dyn) o->setAttr("sa.advance", e.b.getI64IntegerAttr(j));
  };
  auto loadChunk = [&](int64_t i, bool dyn) {
    if (!useGemv) {
      auto lw = sahl::LoadOp::create(e.b, e.loc, e.sub(p.w, {i * nc, 0, 0}, {nc, K, d}), tiles[i & 1]);
      adv(lw, dyn, 0);
      if (i == 0 && hoist) lw->setAttr("sa.prefix", e.b.getUnitAttr());
    }
    for (size_t m = 0; m < p.chunked.size(); ++m)
      adv(sahl::LoadOp::create(e.b, e.loc, e.sub(p.chunked[m].second, {i * nc, 0}, {nc, d}), cols[i & 1][m]), dyn, 1);
  };
  auto chunkAt = [&](int64_t i, bool dyn, bool prefetch) {
    const int bank = int(i & 1);
    if (useGemv)
      for (int64_t g = 0; g < nc; g += ncg)
        for (int64_t k0 = 0; k0 < K; k0 += kBlock) {
          const int64_t kb = std::min(kBlock, K - k0);
          Value xs = kb == K ? xl : e.sub(xl, {k0}, {kb});
          adv(sahl::GemvOp::create(e.b, e.loc, xs, e.sub(p.w, {i * nc + g, k0, 0}, {ncg, kb, d}),
                                   e.sub(acc[bank], {g, 0}, {ncg, d}), k0 > 0),
              dyn, 0);
        }
    else
      sahl::MmaOp::create(e.b, e.loc, strip, tiles[bank], acc[bank], false);
    if (prefetch) loadChunk(i + 1, dyn);
    Value ys = e.sub(y, {i * nc, 0}, {nc, d});
    if (!epi) {
      adv(sahl::StoreOp::create(e.b, e.loc, acc[bank], ys), dyn, 1);
      return true;
    }
    SmallVector<Value> ins;
    int cm = 0;
    for (int in = 0; in < epi.getNumDpsInputs(); ++in) {
      if (epi.getDpsInputs()[in] == p.acc) ins.push_back(acc[bank]);
      else if (bc.count(in)) ins.push_back(bc[in]);
      else if (cm < int(p.chunked.size()) && p.chunked[cm].first == in) ins.push_back(cols[bank][cm++]);
      else return false;
    }
    auto g = linalg::GenericOp::create(e.b, e.loc, TypeRange{}, ins, ValueRange{out[bank]}, epi.getIndexingMapsArray(),
                                       epi.getIteratorTypesArray());
    IRMapping map;
    epi.getRegion().cloneInto(&g.getRegion(), map);
    g->setAttr("sa.prefer_bank", e.b.getI64IntegerAttr(bank));
    adv(sahl::StoreOp::create(e.b, e.loc, out[bank], ys), dyn, 1);
    return true;
  };
  loadChunk(0, false);
  int64_t j = 0;
  if (useLoop) {
    const int64_t pairs = (nch - 1) / 2;          // the prefetch of the last pair stays in range
    if (pairs) {
      auto lp = sahl::LoopOp::create(e.b, e.loc, pairs, ArrayRef<int64_t>{2 * wbytes, 2 * fbytes}, Value());
      OpBuilder::InsertionGuard guard(e.b);
      e.b.setInsertionPointToEnd(&lp.getBody().emplaceBlock());
      if (!chunkAt(0, true, true) || !chunkAt(1, true, true)) return false;
      j = 2 * pairs;
    }
  }
  for (int64_t jj = j; jj < nch; ++jj)
    if (!chunkAt(jj, false, jj + 1 < nch)) return false;
  eraseKernel(k);
  return true;
}

// A kernel left unexpanded: the check that refused it (this file's line) as
// sa.unexpanded, which sahl-to-sahw's report prints (the fallback's reason).
static bool notExpanded(sahl::KernelOp k, int line) {
  k->setAttr("sa.unexpanded", StringAttr::get(k.getContext(), "SahlExpandKernels.cpp:" + std::to_string(line)));
  return false;
}

// A generic contraction (ContractPlan): false when it has a form not written
// out here (the lowering). A dynamic K or N (the cache's T: attention's
// batch_matmuls in prefill) has one K block and one chunk; rows of x of a
// dynamic length, or results of a dynamic length, are walked with the row cursor.
bool expandContraction(sahl::KernelOp k, ContractPlan p, KernelMatcher &km, const ::sa::Layout &lay, int64_t d,
                       int64_t attnBlock, int64_t maxDyn, int64_t gemvPorts) {
  auto rr = km.loopRanges(p.op);
  if (!rr) return notExpanded(k, __LINE__);
  auto ranges = *rr;
  Range B = p.bLoop >= 0 ? ranges[p.bLoop] : Range{}, Gr = p.gLoop >= 0 ? ranges[p.gLoop] : Range{};
  Range Kr = ranges[p.kLoop], Nr = ranges[p.nLoop];
  if (B.dyn || Gr.dyn || (Kr.dyn && Nr.dyn)) return notExpanded(k, __LINE__);
  const int64_t K = Kr.size, N = Nr.size * (p.laneLoop >= 0 ? d : 1), H = B.size, G = Gr.size;
  const bool xRows = (p.bLoop >= 0 && p.x.coef[p.bLoop] < 0) || (p.gLoop >= 0 && p.x.coef[p.gLoop] < 0);
  const bool dyn = Kr.dyn || Nr.dyn;
  if (xRows != bool(Kr.dyn)) return notExpanded(k, __LINE__);          // (x of a dynamic K: its rows a dynamic stride apart)
  // p . V over the packed V cache (a dynamic K, the matrix [n, T, D] blocks): on
  // the GEMV unit, per row of x one command per 8 groups (plan §8.22)
  const bool gemvDyn = gemvPorts && Kr.dyn && !Nr.dyn && p.layout == MatLayout::Packed && p.laneLoop >= 0 &&
                       p.x.et.isInteger(8) && K * d <= 65535;
  if (dyn && !gemvDyn && (p.laneLoop >= 0 || p.layout == MatLayout::Packed)) return notExpanded(k, __LINE__);
  if (Kr.dyn && p.layout == MatLayout::RowsK) return notExpanded(k, __LINE__);
  if (Nr.dyn && p.layout != MatLayout::RowsK) return notExpanded(k, __LINE__);
  if (K % d || N % d) return notExpanded(k, __LINE__);
  const int64_t sb = lay.sbank, nt = N / d;
  Type accT = cast<MemRefType>(p.op.getDpsInits()[0].getType()).getElementType();
  if (!accT.isInteger(32) && !accT.isInteger(64)) return notExpanded(k, __LINE__);
  // which input is x (the other is the matrix), their maps; the output map
  auto maps = p.op.getIndexingMapsArray();
  auto a0 = km.operandOf(p.op, 0), a1 = km.operandOf(p.op, 1);
  if (!a0 || !a1) return notExpanded(k, __LINE__);
  const bool x0 = a0->coef == p.x.coef && a1->coef == p.m.coef, x1 = a1->coef == p.x.coef && a0->coef == p.m.coef;
  if (x0 == x1) return notExpanded(k, __LINE__);                     // (which is x must be unambiguous)
  const int ix = x0 ? 0 : 1;
  (void)ix;
  // each DDR operand's dimension -> loop map from its strides and the
  // operand's element offset per loop (the view may be permuted against the
  // contraction's operand: a transposing extension in between)
  auto viewMap = [&](Value v, ArrayRef<int64_t> coef) -> std::optional<AffineMap> {
    auto mt = cast<MemRefType>(v.getType());
    SmallVector<int64_t> st;
    int64_t off;
    if (failed(mt.getStridesAndOffset(st, off))) return std::nullopt;
    SmallVector<AffineExpr> rs;
    for (int64_t r = 0; r < mt.getRank(); ++r) {
      if (mt.getDimSize(r) == 1) {                // (index 0 whatever its stride: one KV head's batch)
        rs.push_back(getAffineConstantExpr(0, k.getContext()));
        continue;
      }
      if (ShapedType::isDynamic(st[r])) return std::nullopt;
      int found = -1;
      for (int L = 0; L < int(coef.size()); ++L)
        if (coef[L] == st[r] && (mt.isDynamicDim(r) ? bool(ranges[L].dyn) : ranges[L].size == mt.getDimSize(r))) {
          if (found >= 0) return std::nullopt;
          found = L;
        }
      if (found < 0) return std::nullopt;
      rs.push_back(getAffineDimExpr(found, k.getContext()));
    }
    return AffineMap::get(int(coef.size()), 0, rs, k.getContext());
  };
  // (rows of x a dynamic stride apart: through the row cursor, no map)
  auto xm = xRows ? std::optional<AffineMap>(AffineMap()) : viewMap(p.x.view, p.x.coef), mm = viewMap(p.m.view, p.m.coef);
  if (!xm || !mm) return notExpanded(k, __LINE__);
  AffineMap xMap = *xm, mMap = *mm, outMap = maps[2];
  if (xRows && Expander::rowDim(cast<MemRefType>(p.x.view.getType())) < 0) return notExpanded(k, __LINE__);
  // results of a dynamic length: rows back to back
  if (Nr.dyn && Expander::rowDim(cast<MemRefType>(p.store.getDst().getType())) < 0) return notExpanded(k, __LINE__);
  if (int64_t(outMap.getNumResults()) != cast<MemRefType>(p.store.getDst().getType()).getRank()) return notExpanded(k, __LINE__);
  auto outPos = [&](int L) -> int {
    if (L < 0) return -1;
    for (unsigned i = 0; i < outMap.getNumResults(); ++i)
      if (auto de = dyn_cast<AffineDimExpr>(outMap.getResult(i)); de && int(de.getPosition()) == L) return int(i);
    return -1;
  };
  int eb = outPos(p.bLoop), eg = outPos(p.gLoop), en = outPos(p.nLoop), el = outPos(p.laneLoop);
  if ((eb >= 0 && eg >= 0 && eb > eg) || (eg >= 0 && eg > en) || (eb >= 0 && eb > en)) return notExpanded(k, __LINE__);
  // epilogue inputs (the lowering's classes): 1 per element, 2 per column, 3 per (batch, row), 4 scalar
  linalg::GenericOp epi = p.epi;
  struct EpiIn {
    int kind = 0;
    Value src;
    int64_t sb = 0, sg = 0;
    AffineMap map;
  };
  llvm::DenseMap<int, EpiIn> epiIn;
  if (epi) {
    if (!epi.getRegion().front().getOps<linalg::IndexOp>().empty()) return notExpanded(k, __LINE__);
    auto emaps = epi.getIndexingMapsArray();
    for (auto &[i, src] : p.epiLoads) {
      auto mt = cast<MemRefType>(epi.getDpsInputs()[i].getType());
      AffineMap em = emaps[i];
      EpiIn e;
      e.src = src;
      e.map = em;
      if (em.isIdentity()) {
        if (Nr.dyn || !(mt.getElementType().isF32() || mt.getElementType().isInteger(32))) return notExpanded(k, __LINE__);
        e.kind = 1;
        epiIn[i] = e;
        continue;
      }
      SmallVector<int64_t> st;
      int64_t off;
      if (!mt.hasStaticShape() || failed(mt.getStridesAndOffset(st, off))) return notExpanded(k, __LINE__);
      bool onlyRow = true, onlyCol = true;
      int64_t sn = 0, sl = 0;
      for (unsigned r = 0; r < em.getNumResults(); ++r) {
        if (mt.getDimSize(r) == 1) continue;      // (index 0: a unit loop, e.g. one KV head's batch)
        if (auto de = dyn_cast<AffineDimExpr>(em.getResult(r))) {
          int pos = int(de.getPosition());
          if (pos == eb) e.sb = st[r], onlyCol = false;
          else if (pos == eg) e.sg = st[r], onlyCol = false;
          else if (pos == en) sn = st[r], onlyRow = false;
          else if (pos == el && el >= 0) sl = st[r], onlyRow = false;
          else onlyRow = onlyCol = false;
        } else if (auto c = dyn_cast<AffineConstantExpr>(em.getResult(r)); !c || c.getValue() != 0) {
          onlyRow = onlyCol = false;
        }
      }
      if (!onlyRow && onlyCol && sn == (el >= 0 ? d : 1) && (el < 0 || sl == 1) && !Nr.dyn && mt.getElementType().isF32()) {
        e.kind = 2;
        epiIn[i] = e;
        continue;
      }
      if (!onlyRow) return notExpanded(k, __LINE__);
      e.kind = (e.sb || e.sg) ? 3 : 4;
      epiIn[i] = e;
    }
    for (int i = 0; i < epi.getNumDpsInputs(); ++i)
      if (epi.getDpsInputs()[i] != p.op.getDpsInits()[0] && !epiIn.count(i)) return notExpanded(k, __LINE__);
    Type oet = cast<MemRefType>(epi.getDpsInits()[0].getType()).getElementType();
    if (!oet.isF32() && !oet.isInteger(32) && !oet.isInteger(8)) return notExpanded(k, __LINE__);
    if (oet.isInteger(8) && Nr.dyn) return notExpanded(k, __LINE__);
  } else if (Nr.dyn) {
    return notExpanded(k, __LINE__);
  }
  // x's place; K blocks (the strip of a block in one bank, its B tile in one DMA row); chunks
  const bool xI32 = p.x.et.isInteger(32), xInRaw = !xI32 && p.layout != MatLayout::RowsK;
  if (!xI32 && p.layout == MatLayout::RowsK) return notExpanded(k, __LINE__);
  if (xInRaw && K / d > sb) return notExpanded(k, __LINE__);
  // x in ACC: where the lowering will allocate it (the K blocks depend on it):
  // the allocator replayed over what comes before x, from a fresh state (the
  // kernel the first operation that emits commands; otherwise the lowering)
  int64_t xAt = sb;
  if (!xInRaw) {
    if (!firstToEmit(k)) return notExpanded(k, __LINE__);
    LocalMemory mem(lay);
    auto words = [&](int64_t n) { return uint32_t(std::max<int64_t>((n + d - 1) / d, 1)); };
    for (auto &[i, src] : p.epiLoads) {
      const EpiIn &ei = epiIn[i];
      if (ei.kind < 3) continue;
      int64_t n = cast<MemRefType>(epi.getDpsInputs()[i].getType()).getNumElements();
      if (!mem.allocAcc(words(n))) return notExpanded(k, __LINE__);                     // the values
      if (ei.kind == 4) {
        if (!mem.allocAcc(1)) return notExpanded(k, __LINE__);                           // their broadcast word
      } else {
        int64_t w = (n + d - 1) / d;
        if (!mem.allocAcc(uint32_t(w * d)) || !mem.allocAcc(uint32_t(w * d))) return notExpanded(k, __LINE__);   // replicated, transposed
      }
    }
    auto xw = mem.allocAcc(words(K));
    if (!xw) return notExpanded(k, __LINE__);
    xAt = *xw;
  }
  const int64_t xDepth = xInRaw ? 2 * sb : 2 * int64_t(lay.cbank);
  int64_t kmax = std::min<int64_t>(sb, 65535 / d / d * d);
  kmax = std::min<int64_t>(kmax, (xDepth - xAt - K / d) / d * d);
  if (kmax < d) return notExpanded(k, __LINE__);
  const int64_t kc = K > kmax ? kmax : K, nk = (K + kc - 1) / kc;
  if (nk > 1 && (G != 1 || dyn)) return notExpanded(k, __LINE__);
  int64_t nc = nt;
  if (!Nr.dyn && !gemvDyn) {
    int64_t cap = std::max<int64_t>(sb / kc, 1);
    for (nc = std::min(cap, nt); nc > 1 && nt % nc; --nc) {
    }
  }
  // C6.5: q . K^T over the cache's T in blocks (nb tiles, a power of two: the
  // block count is T >> log2(nb D) + 1)
  int64_t nb = 0;
  if (kc * nc > sb && !gemvDyn) {               // (the GEMV unit streams the matrix: no SPAD_B)
    if (!(Nr.dyn && p.layout == MatLayout::RowsK && !xRows && nk == 1 && epi)) return notExpanded(k, __LINE__);
    nb = int64_t(llvm::bit_floor(uint64_t(sb / kc)));
    // blocks of 128 rows by default (§8.22 step a: the cycle model over T = 16..2048,
    // Qwen3 and Gemma, against 32..512: short T pays for a whole block)
    const int64_t blk = attnBlock > 0 ? attnBlock : 128;
    nb = std::min(nb, std::max<int64_t>(int64_t(llvm::bit_floor(uint64_t(std::max<int64_t>(blk / d, 1)))), 1));
    if (nb < 1 || (G > 1 ? G > d || 2 * K > sb : K > sb)) return notExpanded(k, __LINE__);   // (x: one A tile at word 0; step b)
  }
  int64_t rowPitch = 0;                            // (the tiled form: bytes between the matrix's rows of T)
  if (nb) {
    SmallVector<int64_t> mst;
    int64_t moff;
    if (failed(cast<MemRefType>(p.m.view.getType()).getStridesAndOffset(mst, moff))) return notExpanded(k, __LINE__);
    for (unsigned r = 0; r < mMap.getNumResults(); ++r)
      if (auto de = dyn_cast<AffineDimExpr>(mMap.getResult(r)); de && int(de.getPosition()) == p.nLoop) rowPitch = mst[r];
    if (rowPitch <= 0 || ShapedType::isDynamic(rowPitch)) return notExpanded(k, __LINE__);
    rowPitch *= int64_t(cast<MemRefType>(p.m.view.getType()).getElementTypeBitWidth() / 8);
  }
  if (xRows && nc != nt && G > 1 && H > 1 && H * G > 64) return notExpanded(k, __LINE__);

  Expander e{OpBuilder(k), k.getLoc(), d};
  MLIRContext *ctx = k.getContext();
  Type f32 = Float32Type::get(ctx), i8 = IntegerType::get(ctx, 8);
  sahl::ClaimSpadOp::create(e.b, e.loc, 2 * sb);
  // per-row and scalar inputs: loaded once, broadcast
  llvm::DenseMap<int, Value> bc;
  for (auto &[i, src] : p.epiLoads) {
    const EpiIn &ei = epiIn[i];
    if (ei.kind < 3) continue;
    auto mt = cast<MemRefType>(epi.getDpsInputs()[i].getType());
    Value v = e.alloc(mt.getShape(), f32, "acc", "packed");
    sahl::LoadOp::create(e.b, e.loc, src, v);
    Value w = e.alloc(mt.getShape(), f32, "acc", "bcast");
    sahl::BcastOp::create(e.b, e.loc, v, w);
    bc[i] = w;
  }
  auto placedA = [&](ArrayRef<int64_t> shape, int64_t word) {
    Value v = e.alloc(shape, i8, "spad_a", "");
    v.getDefiningOp()->setAttr("sa.word", e.b.getI64IntegerAttr(word));
    return v;
  };
  Value xl;
  if (xInRaw) {
    xl = placedA({K}, sb);
  } else {
    xl = e.alloc({K}, p.x.et, xI32 ? "acc" : "spad_a", "packed");
    sahl::ReserveOp::create(e.b, e.loc, xl, false);
  }
  const int64_t nOut = nc * d;
  Value y = p.store.getDst();
  Value Kv = Kr.dyn ? Kr.val : Value(), Nv = Nr.dyn ? Nr.val : Value();
  if (xRows || Nr.dyn) sahl::CursorResetOp::create(e.b, e.loc);
  auto loadX = [&](int64_t b, int64_t g) {
    if (xRows) {                                    // row (b, g) of x at the cursor, the cursor on
      auto l = sahl::LoadOp::create(e.b, e.loc, e.row0(p.x.view, Kv), e.prefix(xl, Kv));
      l->setAttr("sa.cursor", e.b.getUnitAttr());
      l->setAttr("sa.cursor_step", e.b.getUnitAttr());
      return;
    }
    llvm::DenseMap<int, std::pair<int64_t, int64_t>> at;
    if (p.bLoop >= 0) at[p.bLoop] = {b, 1};
    if (p.gLoop >= 0) at[p.gLoop] = {g, 1};
    sahl::LoadOp::create(e.b, e.loc, e.view(p.x.view, xMap, at, {p.bLoop, p.gLoop}), xl);
  };
  auto replicate = [&](int64_t k0, int64_t kl) {
    sahl::StripOp::create(e.b, e.loc, Kv ? e.prefix(xl, Kv) : e.sub(xl, {k0}, {kl}), placedA({d, kl}, 0));
  };
  // a placed buffer of dynamic dimensions (SPAD_A word / SPAD_B bank)
  auto placedDyn = [&](ArrayRef<int64_t> shape, ValueRange dv, StringRef mem, int64_t at) -> Value {
    auto a = memref::AllocOp::create(e.b, e.loc, MemRefType::get(shape, i8), dv);
    a->setAttr("sa.mem", e.b.getStringAttr(mem));
    a->setAttr(mem == "spad_b" ? "sa.bank" : "sa.word", e.b.getI64IntegerAttr(at));
    return Value(a);
  };
  Value nTiles;
  if (Nv)
    nTiles = arith::ShRUIOp::create(e.b, e.loc, Nv,
                                    arith::ConstantIndexOp::create(e.b, e.loc, int64_t(llvm::Log2_64(uint64_t(d)))));
  int64_t step = 0;
  auto loadB = [&](int64_t b, int64_t c0, int64_t k0, int64_t kl) -> Value {
    const int64_t bank = step++ & 1;
    // the tiles: [nc, K, D] (dynamic: N / D tiles, or K rows)
    Value tiles = Nv   ? placedDyn({ShapedType::kDynamic, kl, d}, ValueRange{nTiles}, "spad_b", bank)
                  : Kv ? placedDyn({nc, ShapedType::kDynamic, d}, ValueRange{Kv}, "spad_b", bank)
                       : placedDyn({nc, kl, d}, ValueRange{}, "spad_b", bank);
    llvm::DenseMap<int, std::pair<int64_t, int64_t>> at;
    if (p.bLoop >= 0) at[p.bLoop] = {b, 1};
    at[p.kLoop] = {k0, kl};
    if (p.layout == MatLayout::Packed) {
      at[p.nLoop] = {c0, nc};
      at[p.laneLoop] = {0, d};
      sahl::LoadOp::create(e.b, e.loc, e.view(p.m.view, mMap, at, {p.bLoop}), tiles);
    } else if (p.layout == MatLayout::RowsK) {
      at[p.nLoop] = {c0 * d, nc * d};
      Value raw = Nv ? placedDyn({ShapedType::kDynamic, kl}, ValueRange{Nv}, "spad_a", sb) : placedA({nc * d, kl}, sb);
      llvm::DenseMap<int, Value> dv;
      if (Nv) dv[p.nLoop] = Nv;
      auto l = sahl::LoadOp::create(e.b, e.loc, e.view(p.m.view, mMap, at, {p.bLoop}, dv), raw);
      l->setAttr("sa.rows", e.b.getUnitAttr());
      sahl::TransposeOp::create(e.b, e.loc, raw, tiles);
    } else {
      at[p.nLoop] = {c0 * d, nc * d};
      llvm::DenseMap<int, Value> dv;
      if (Kv) dv[p.kLoop] = Kv;
      auto l = sahl::LoadOp::create(e.b, e.loc, e.view(p.m.view, mMap, at, {p.bLoop}, dv), tiles);
      l->setAttr("sa.interleave", e.b.getUnitAttr());
    }
    return tiles;
  };
  // a (b, g, chunk) view of an output-shaped DDR value
  auto outView = [&](Value v, AffineMap m, int64_t b, int64_t g, int64_t c0) {
    llvm::DenseMap<int, std::pair<int64_t, int64_t>> at;
    if (p.bLoop >= 0) at[p.bLoop] = {b, 1};
    if (p.gLoop >= 0) at[p.gLoop] = {g, 1};
    if (p.laneLoop >= 0) at[p.nLoop] = {c0, nc}, at[p.laneLoop] = {0, d};
    else at[p.nLoop] = {c0 * d, nc * d};
    return e.view(v, m, at, {p.bLoop, p.gLoop});
  };
  // a (batch, row, chunk)'s result: the epilogue over its accumulator, the store
  auto finish = [&](Value acc, int64_t b, int64_t g, int64_t c0) -> bool {
    const bool lastRow = b + 1 == H && g + 1 == G;
    Value ys = Nv ? e.row0(y, Nv) : outView(y, outMap, b, g, c0);
    if (!epi) {
      sahl::StoreOp::create(e.b, e.loc, acc, ys);
      return true;
    }
    Type oet = cast<MemRefType>(epi.getDpsInits()[0].getType()).getElementType();
    Value out = e.alloc({nOut}, oet, oet.isInteger(8) ? "spad_a" : "acc", "packed");
    sahl::ReserveOp::create(e.b, e.loc, out, false);
    SmallVector<Value> ins;
    SmallVector<AffineMap> m1;
    AffineMap id1 = AffineMap::getMultiDimIdentityMap(1, ctx), none = AffineMap::get(1, 0, ctx);
    for (int i = 0; i < epi.getNumDpsInputs(); ++i) {
      if (epi.getDpsInputs()[i] == p.op.getDpsInits()[0]) {
        ins.push_back(Nv ? e.prefix(acc, Nv) : acc), m1.push_back(id1);
        continue;
      }
      const EpiIn &ei = epiIn[i];
      if (ei.kind == 1 || ei.kind == 2) {
        Type et = cast<MemRefType>(epi.getDpsInputs()[i].getType()).getElementType();
        Value l = e.alloc({nOut}, et, "acc", "packed");
        Value src;
        if (ei.kind == 1) {
          src = outView(ei.src, outMap, b, g, c0);
        } else {
          llvm::DenseMap<int, std::pair<int64_t, int64_t>> at;
          if (p.laneLoop >= 0) at[en] = {c0, nc}, at[el] = {0, d};
          else at[en] = {c0 * d, nc * d};
          src = e.view(ei.src, ei.map, at, {});
        }
        sahl::LoadOp::create(e.b, e.loc, src, l);
        ins.push_back(l), m1.push_back(id1);
      } else if (ei.kind == 4) {
        ins.push_back(bc[i]), m1.push_back(none);
      } else {
        // this (batch, row)'s word of the broadcast values
        auto mt = cast<MemRefType>(bc[i].getType());
        int64_t w = b * ei.sb + g * ei.sg;
        SmallVector<int64_t> idx;
        int64_t rem = w;
        SmallVector<int64_t> st;
        int64_t off0;
        if (failed(mt.getStridesAndOffset(st, off0))) return false;
        for (int64_t r = 0; r < mt.getRank(); ++r) idx.push_back(rem / st[r]), rem %= st[r];
        SmallVector<OpFoldResult> o, sz, one;
        for (int64_t r = 0; r < mt.getRank(); ++r)
          o.push_back(e.b.getIndexAttr(idx[r])), sz.push_back(e.b.getIndexAttr(1)), one.push_back(e.b.getIndexAttr(1));
        auto rt = memref::SubViewOp::inferRankReducedResultType({}, mt, o, sz, one);
        ins.push_back(memref::SubViewOp::create(e.b, e.loc, cast<MemRefType>(rt), bc[i], o, sz, one));
        m1.push_back(none);
      }
    }
    m1.push_back(id1);
    Value outv = Nv ? e.prefix(out, Nv) : out;
    auto g1 = linalg::GenericOp::create(e.b, e.loc, TypeRange{}, ins, ValueRange{outv}, m1,
                                        SmallVector<utils::IteratorType>{utils::IteratorType::parallel});
    IRMapping map;
    epi.getRegion().cloneInto(&g1.getRegion(), map);
    auto st = sahl::StoreOp::create(e.b, e.loc, outv, ys);
    if (Nv) {                                   // row (b, g) at the cursor, then the cursor on
      st->setAttr("sa.cursor", e.b.getUnitAttr());
      if (!lastRow) st->setAttr("sa.cursor_step", e.b.getUnitAttr());
    }
    return true;
  };
  for (int64_t b = 0; b < H; ++b) {
    if (gemvDyn) {
      // per row of x (its T int8 at the cursor into SPAD_A): GEMVs over the n
      // blocks of the batch's [n, T, D], at most 8 a command (Gemma's 16: two;
      // rows bytes T * D a dynamic field)
      llvm::DenseMap<int, Value> dv;
      dv[p.kLoop] = Kv;
      for (int64_t g = 0; g < G; ++g) {
        loadX(b, g);
        auto scope = sahl::ScopeOp::create(e.b, e.loc, /*keep=*/true, IntegerAttr());
        OpBuilder::InsertionGuard guard(e.b);
        e.b.setInsertionPointToEnd(&scope.getBody().emplaceBlock());
        Value acc = e.alloc({nOut}, accT, "acc", "packed");
        for (int64_t c0 = 0; c0 < nt; c0 += 8) {
          const int64_t ncg = std::min<int64_t>(8, nt - c0);
          llvm::DenseMap<int, std::pair<int64_t, int64_t>> at;
          if (p.bLoop >= 0) at[p.bLoop] = {b, 1};
          at[p.kLoop] = {0, K};
          at[p.nLoop] = {c0, ncg};
          at[p.laneLoop] = {0, d};
          sahl::GemvOp::create(e.b, e.loc, xl, e.view(p.m.view, mMap, at, {p.bLoop}, dv),
                               nt > 8 ? e.sub(acc, {c0 * d}, {ncg * d}) : acc, false);
        }
        if (!finish(acc, b, g, 0)) return false;
      }
      continue;
    }
    if (nb) {
      // C6.5: the strips of all rows of x (bank 0, K words each); the blocks of
      // T in a loop whose count is T >> log2(Tb) + 1; then per row its epilogue
      // (in a scope per batch: its accumulators released after it)
      auto bscope = sahl::ScopeOp::create(e.b, e.loc, /*keep=*/true, IntegerAttr());
      OpBuilder::InsertionGuard bguard(e.b);
      e.b.setInsertionPointToEnd(&bscope.getBody().emplaceBlock());
      // §8.22 step b: G > 1 rows of x as the rows of one A tile (one EX a block
      // for all of them, its output rows 0 .. G - 1): the G rows loaded as rows
      // (bank 1, int32 x through ACC; the tile's other rows whatever is there,
      // their outputs unused),
      // transposed into the [K, D] tile form (after them), then each D x D block
      // transposed into the A form at word 0
      const bool oneEx = G > 1;
      if (oneEx) {
        llvm::DenseMap<int, std::pair<int64_t, int64_t>> at;
        if (p.bLoop >= 0) at[p.bLoop] = {b, 1};
        at[p.gLoop] = {0, G};
        Value xv = e.view(p.x.view, xMap, at, {p.bLoop});
        if (xI32) {                                 // (int32 x: the rows into ACC, one VE copy to int8)
          Value xa = e.alloc({G, K}, p.x.et, "acc", "packed");
          sahl::LoadOp::create(e.b, e.loc, xv, xa);
          sahl::StripOp::create(e.b, e.loc, xa, placedA({G, K}, sb))->setAttr("sa.packed", e.b.getUnitAttr());
        } else {
          sahl::LoadOp::create(e.b, e.loc, xv, placedA({G, K}, sb))->setAttr("sa.rows", e.b.getUnitAttr());
        }
        Value cols = placedA({K, d}, sb + K);
        sahl::TransposeOp::create(e.b, e.loc, placedA({d, K}, sb), cols);
        sahl::TransposeOp::create(e.b, e.loc, cols, placedA({d, K}, 0));
      } else {
        for (int64_t g = 0; g < G; ++g) {
          loadX(b, g);
          sahl::StripOp::create(e.b, e.loc, e.sub(xl, {0}, {K}), placedA({d, K}, g * K));
        }
      }
      const int64_t tb = nb * d, blocks = N / tb + 1, rowElems = blocks * tb;
      auto accBuf = [&](int64_t n) {
        Value a = e.alloc({n}, accT, "acc", "packed");
        if (accT.isInteger(64)) a.getDefiningOp()->setAttr("sa.accumulator", e.b.getUnitAttr());
        sahl::ReserveOp::create(e.b, e.loc, a, false);
        return a;
      };
      Value scratch = accBuf(((oneEx ? 0 : G - 1) + d) * tb);
      SmallVector<Value> rowsAcc;
      for (int64_t g = 0; g < G; ++g) rowsAcc.push_back(accBuf(rowElems));
      // the block count: T is a multiple of D (the runtime pads it), so
      // ceil(T / Tb) = ((T * (M - 1)) >> log2(M Tb)) + 1 for a power of two
      // M >= max_dynamic / D (the setup entry's shift: at most 15 bits); else
      // (T >> log2 Tb) + 1, one block past T when T is a multiple of Tb
      const int64_t lgTb = int64_t(llvm::Log2_64(uint64_t(tb)));
      const int64_t M = int64_t(llvm::bit_ceil(uint64_t(std::max<int64_t>(maxDyn / d, 2))));
      Value tv = Nv;
      int64_t sh = lgTb;
      if (lgTb + int64_t(llvm::Log2_64(uint64_t(M))) <= 15) {
        auto mul = arith::MulIOp::create(e.b, e.loc, Nv, arith::ConstantIndexOp::create(e.b, e.loc, M - 1));
        mul->setAttr("sa.index", e.b.getUnitAttr());
        tv = mul;
        sh += int64_t(llvm::Log2_64(uint64_t(M)));
      }
      Value shv = arith::ShRUIOp::create(e.b, e.loc, tv, arith::ConstantIndexOp::create(e.b, e.loc, sh));
      auto cnt = arith::AddIOp::create(e.b, e.loc, shv, arith::ConstantIndexOp::create(e.b, e.loc, 1));
      cnt->setAttr("sa.index", e.b.getUnitAttr());
      auto lp = sahl::LoopOp::create(e.b, e.loc, blocks, ArrayRef<int64_t>{tb * rowPitch, nb}, cnt);
      {
        OpBuilder::InsertionGuard guard(e.b);
        e.b.setInsertionPointToEnd(&lp.getBody().emplaceBlock());
        Value raw = placedA({tb, K}, sb);
        Value tiles = placedDyn({nb, K, d}, ValueRange{}, "spad_b", 0);
        llvm::DenseMap<int, std::pair<int64_t, int64_t>> at;
        if (p.bLoop >= 0) at[p.bLoop] = {b, 1};
        at[p.kLoop] = {0, K};
        at[p.nLoop] = {0, tb};
        auto l = sahl::LoadOp::create(e.b, e.loc, e.view(p.m.view, mMap, at, {p.bLoop}), raw);
        l->setAttr("sa.rows", e.b.getUnitAttr());
        l->setAttr("sa.advance", e.b.getI64IntegerAttr(0));
        sahl::TransposeOp::create(e.b, e.loc, raw, tiles);
        if (oneEx)
          sahl::MmaOp::create(e.b, e.loc, placedA({d, K}, 0), tiles, e.sub(scratch, {0}, {tb}), false);
        else
          for (int64_t g = 0; g < G; ++g)
            sahl::MmaOp::create(e.b, e.loc, placedA({d, K}, g * K), tiles, e.sub(scratch, {g * tb}, {tb}), false);
        for (int64_t g = 0; g < G; ++g) {
          auto c = memref::CopyOp::create(e.b, e.loc, e.sub(scratch, {g * tb}, {tb}), e.sub(rowsAcc[g], {0}, {tb}));
          c->setAttr("sa.advance", e.b.getI64IntegerAttr(1));
        }
      }
      for (int64_t g = 0; g < G; ++g) {
        auto scope = sahl::ScopeOp::create(e.b, e.loc, /*keep=*/true, IntegerAttr());
        OpBuilder::InsertionGuard guard(e.b);
        e.b.setInsertionPointToEnd(&scope.getBody().emplaceBlock());
        if (!finish(rowsAcc[g], b, g, 0)) return false;
      }
      continue;
    }
    if (G == 1) {
      loadX(b, 0);
      if (nk == 1) replicate(0, K);
    }
    step = 0;
    for (int64_t c0 = 0; c0 < nt; c0 += nc) {
      if (xRows && G > 1 && c0 > 0) {               // the rows of x again: the cursor back to row (b, 0)
        auto r = sahl::CursorResetOp::create(e.b, e.loc);
        if (b) r->setAttr("sa.skip", e.b.getI64IntegerAttr(b * G));
      }
      Value bk = nk == 1 ? loadB(b, c0, 0, K) : Value();
      for (int64_t g = 0; g < G; ++g) {
        if (G > 1) {
          loadX(b, g);
          replicate(0, K);
        }
        auto scope = sahl::ScopeOp::create(e.b, e.loc, /*keep=*/true, IntegerAttr());
        if (epi && xInRaw) scope.setSpadFromAttr(e.b.getI64IntegerAttr(sb + (K + d - 1) / d));
        OpBuilder::InsertionGuard guard(e.b);
        e.b.setInsertionPointToEnd(&scope.getBody().emplaceBlock());
        Value acc = e.alloc({nOut}, accT, "acc", "packed");
        if (accT.isInteger(64)) acc.getDefiningOp()->setAttr("sa.accumulator", e.b.getUnitAttr());
        if (nk == 1) {
          sahl::MmaOp::create(e.b, e.loc, placedA({d, K}, 0), bk, acc, false);
        } else {
          for (int64_t kb = 0; kb < nk; ++kb) {
            int64_t k0 = kb * kc, kl = std::min<int64_t>(kc, K - k0);
            Value bt = loadB(b, c0, k0, kl);
            replicate(k0, kl);
            sahl::MmaOp::create(e.b, e.loc, placedA({d, kl}, 0), bt, acc, kb > 0);
          }
        }
        if (!finish(acc, b, g, c0)) return false;
      }
    }
  }
  eraseKernel(k);
  return true;
}

// Decode's attention micro-kernel (AttnPlan).
bool expandAttention(sahl::KernelOp k, AttnPlan p, const ::sa::Layout &lay, int64_t d, int64_t maxDyn) {
  const int64_t H = p.H, hs = p.hs, sb = lay.sbank, hw = hs / d, logd = int64_t(llvm::Log2_64(uint64_t(d)));
  if (maxDyn % d || hs % d || maxDyn * hw > sb) return false;
  auto sv = p.cache.getDefiningOp<memref::SubViewOp>();
  if (!sv || sv.getSizes().size() != 1) return false;
  linalg::GenericOp epi;
  Value acc = p.bmm.getDpsInits()[0];
  for (Operation *u : acc.getUsers())
    if (auto g = dyn_cast<linalg::GenericOp>(u)) epi = g;
  if (!epi) return false;
  Type accT = cast<MemRefType>(acc.getType()).getElementType();
  Expander e{OpBuilder(k), k.getLoc(), d};
  MLIRContext *ctx = k.getContext();
  Type f32 = Float32Type::get(ctx), i8 = IntegerType::get(ctx, 8);
  Value T = sv.getSizes()[0];
  auto subT = [&](Value v, ArrayRef<int64_t> offs, ArrayRef<OpFoldResult> sizes, ArrayRef<int64_t> shape) {
    auto mt = cast<MemRefType>(v.getType());
    SmallVector<OpFoldResult> o, st;
    for (int64_t x : offs) o.push_back(e.b.getIndexAttr(x)), st.push_back(e.b.getIndexAttr(1));
    auto rt = memref::SubViewOp::inferRankReducedResultType(shape, mt, o, sizes, st);
    return memref::SubViewOp::create(e.b, e.loc, cast<MemRefType>(rt), v, o, sizes, st);
  };
  auto I = [&](int64_t x) -> OpFoldResult { return e.b.getIndexAttr(x); };
  const int64_t DYN = ShapedType::kDynamic;
  auto placed = [&](ArrayRef<int64_t> shape, ValueRange dyn, StringRef mem, int64_t at) {
    auto a = memref::AllocOp::create(e.b, e.loc, MemRefType::get(shape, i8), dyn);
    a->setAttr("sa.mem", e.b.getStringAttr(mem));
    a->setAttr(mem == "spad_b" ? "sa.bank" : "sa.word", e.b.getI64IntegerAttr(at));
    return Value(a);
  };
  auto accAlloc = [&](int64_t n) {
    Value c = e.alloc({n}, accT, "acc", "packed");
    if (accT.isInteger(64)) c.getDefiningOp()->setAttr("sa.accumulator", e.b.getUnitAttr());
    sahl::ReserveOp::create(e.b, e.loc, c, false);
    return c;
  };
  // the epilogue on a head: its inputs (the accumulator view, a per-head word) -> out
  auto epilogue = [&](Value accView, Value headWord, Value out) {
    SmallVector<Value> ins;
    SmallVector<AffineMap> maps;
    AffineMap id1 = AffineMap::getMultiDimIdentityMap(1, ctx), none = AffineMap::get(1, 0, ctx);
    for (Value in : epi.getDpsInputs()) {
      if (in == acc) ins.push_back(accView), maps.push_back(id1);
      else if (headWord) ins.push_back(headWord), maps.push_back(none);
      else return false;
    }
    maps.push_back(id1);
    auto g = linalg::GenericOp::create(e.b, e.loc, TypeRange{}, ins, ValueRange{out}, maps,
                                       SmallVector<utils::IteratorType>{utils::IteratorType::parallel});
    IRMapping map;
    epi.getRegion().cloneInto(&g.getRegion(), map);
    return true;
  };
  sahl::ClaimSpadOp::create(e.b, e.loc, 2 * sb);
  auto shift = arith::ShRUIOp::create(e.b, e.loc, T, arith::ConstantIndexOp::create(e.b, e.loc, logd));
  Value tiles = shift.getResult();
  Value y = p.store.getDst();
  if (p.scores) {
    if (epi.getNumDpsInputs() != 2) return false;
    Value sqv = e.alloc({H}, f32, "acc", "packed");
    sahl::LoadOp::create(e.b, e.loc, p.sq, sqv);
    Value sqb = e.alloc({H}, f32, "acc", "bcast");
    sahl::BcastOp::create(e.b, e.loc, sqv, sqb);
    Value srcw = e.alloc({hs}, cast<MemRefType>(p.small.getType()).getElementType(), "acc", "packed");
    sahl::ReserveOp::create(e.b, e.loc, srcw, false);
    Value c = accAlloc(d * (maxDyn / d) * d);
    auto ra = memref::AllocOp::create(e.b, e.loc, MemRefType::get({H, DYN}, f32), ValueRange{T});
    ra->setAttr("sa.mem", e.b.getStringAttr("acc"));
    ra->setAttr("sa.layout", e.b.getStringAttr("rows"));
    Value res = ra;
    sahl::ReserveOp::create(e.b, e.loc, res, false);
    for (int64_t h = 0; h < H; ++h) {
      Value kraw = placed({DYN, hs}, ValueRange{T}, "spad_a", sb);
      auto l = sahl::LoadOp::create(e.b, e.loc, subT(p.cache, {0, h, 0}, {T, I(1), I(hs)}, {DYN, hs}), kraw);
      l->setAttr("sa.rows", e.b.getUnitAttr());
      Value kt = placed({DYN, hs, d}, ValueRange{tiles}, "spad_b", 0);
      sahl::TransposeOp::create(e.b, e.loc, kraw, kt);
      sahl::LoadOp::create(e.b, e.loc, subT(p.small, {h, 0, 0}, {I(1), I(hs), I(1)}, {hs}), srcw);
      Value strip = placed({d, hs}, ValueRange{}, "spad_a", 0);
      sahl::StripOp::create(e.b, e.loc, srcw, strip);
      sahl::MmaOp::create(e.b, e.loc, strip, kt, c, false);
      Value word = subT(sqb, {h}, {I(1)}, {});
      if (!epilogue(subT(c, {0}, {T}, {DYN}), word, subT(res, {h, 0}, {I(1), T}, {DYN}))) return false;
    }
    // the H rows of T scores, back to back
    auto yt = cast<MemRefType>(y.getType());
    Value yv = y;
    if (yt.getRank() == 3 && yt.getDimSize(2) == 1) {
      SmallVector<ReassociationIndices> re = {{0}, {1, 2}};
      yv = memref::CollapseShapeOp::create(e.b, e.loc, y, re);
    }
    sahl::StoreOp::create(e.b, e.loc, res, yv);
  } else {
    if (epi.getNumDpsInputs() != 1) return false;
    auto pt = cast<MemRefType>(p.small.getType());
    auto psub = ddrRoot(p.small).getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!psub || psub.getDynamicDims().size() != 1 || pt.getRank() != 3 || pt.getDimSize(1) != 1) return false;
    // the H rows of p (its own length), back to back
    auto pa = memref::AllocOp::create(e.b, e.loc, MemRefType::get({H, DYN}, pt.getElementType()),
                                      ValueRange{psub.getDynamicDims()[0]});
    pa->setAttr("sa.mem", e.b.getStringAttr("acc"));
    pa->setAttr("sa.layout", e.b.getStringAttr("rows"));
    Value pl = pa;
    sahl::LoadOp::create(e.b, e.loc, p.small, pl);
    Value c = accAlloc(d * hw * d);
    Value res = e.alloc({hs}, f32, "acc", "packed");
    sahl::ReserveOp::create(e.b, e.loc, res, false);
    for (int64_t h = 0; h < H; ++h) {
      Value strip = placed({d, maxDyn}, ValueRange{}, "spad_a", 0);
      sahl::StripOp::create(e.b, e.loc, subT(pl, {h, 0}, {I(1), T}, {DYN}), strip);
      Value vb = placed({hw, DYN, d}, ValueRange{T}, "spad_b", 1);
      auto l = sahl::LoadOp::create(e.b, e.loc, subT(p.cache, {0, h, 0}, {T, I(1), I(hs)}, {DYN, hs}), vb);
      l->setAttr("sa.interleave", e.b.getUnitAttr());
      sahl::MmaOp::create(e.b, e.loc, strip, vb, c, false);
      if (!epilogue(subT(c, {0}, {I(hs)}, {hs}), Value(), res)) return false;
      sahl::StoreOp::create(e.b, e.loc, res, subT(y, {h, 0, 0}, {I(1), I(1), I(hs)}, {hs}));
    }
  }
  eraseKernel(k);
  return true;
}

struct SahlExpandKernelsPass : public PassWrapper<SahlExpandKernelsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SahlExpandKernelsPass)
  SahlExpandKernelsPass() = default;
  explicit SahlExpandKernelsPass(const TargetConfig &c) : cfg(c), fromOptions(false) {}
  SahlExpandKernelsPass(const SahlExpandKernelsPass &o) : PassWrapper(o), cfg(o.cfg), fromOptions(o.fromOptions) {}
  StringRef getArgument() const override { return "iree-sahl-expand-kernels"; }
  StringRef getDescription() const override {
    return "Writes the micro-kernels' schedules out in sahl (strips, EX, banks, prefetch, scopes)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<sahl::SahlDialect, memref::MemRefDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    if (fromOptions) {
      cfg.d = optD;
      cfg.spadBytes = optSpadKB * 1024;
      cfg.accBytes = optAccKB * 1024;
      cfg.gemvPorts = optGemvPorts;
    }
    func::FuncOp f = getOperation();
    SmallVector<sahl::KernelOp> ks(f.getBody().front().getOps<sahl::KernelOp>());
    ::sa::Layout lay(uint32_t(cfg.d), uint32_t(cfg.spadBytes), uint32_t(cfg.accBytes));
    for (sahl::KernelOp k : ks) {
      if (k.getKind() == "contraction") {
        for (Operation &o : k.getBody().front()) {
          auto c = dyn_cast<linalg::LinalgOp>(&o);
          if (!c || !o.hasAttr("sahl.anchor")) continue;
          KernelMatcher km(cfg);
          if (km.matchContraction(c)) expandContraction(k, km.contracts[c.getOperation()], km, lay, cfg.d, cfg.attnBlock, cfg.maxDynamic, cfg.gemvPorts);
          else notExpanded(k, __LINE__);
          break;
        }
        continue;
      }
      if (k.getKind() == "attention") {
        for (Operation &o : k.getBody().front()) {
          auto bm = dyn_cast<linalg::BatchMatmulOp>(&o);
          if (!bm || !o.hasAttr("sahl.anchor")) continue;
          KernelMatcher km(cfg);
          if (km.matchAttention(bm)) expandAttention(k, km.attns[bm.getOperation()], lay, cfg.d, cfg.maxDynamic);
          break;
        }
        continue;
      }
      if (k.getKind() != "linear") continue;
      for (Operation &o : k.getBody().front()) {
        auto g = dyn_cast<linalg::GenericOp>(&o);
        if (!g || !o.hasAttr("sahl.anchor")) continue;
        KernelMatcher km(cfg);
        if (!km.matchLinear(g)) break;
        LinearPlan p = km.linears[g.getOperation()];
        if (p.rows) expandLinearRows(k, p, lay, cfg.d);
        else expandLinear(k, p, lay, cfg.d, cfg.gemvPorts);
        break;
      }
    }
    // SA_EXPAND_STATS: the kernels left to the old lowering (to drop it once none remain)
    if (getenv("SA_EXPAND_STATS"))
      f.walk([&](sahl::KernelOp k) {
        auto why = k->getAttrOfType<StringAttr>("sa.unexpanded");
        llvm::errs() << "sa-expand: unexpanded " << k.getKind() << " kernel in " << f.getName()
                     << (why ? " (" + why.getValue().str() + ")" : std::string()) << "\n";
      });
  }

  TargetConfig cfg;
  bool fromOptions = true;
  Option<int64_t> optD{*this, "d", llvm::cl::desc("array size D"), llvm::cl::init(8)};
  Option<int64_t> optSpadKB{*this, "spad-kb", llvm::cl::desc("SPAD size (KB, two banks)"), llvm::cl::init(128)};
  Option<int64_t> optAccKB{*this, "acc-kb", llvm::cl::desc("ACC size (KB, two banks)"), llvm::cl::init(256)};
  Option<int64_t> optGemvPorts{*this, "gemv-ports", llvm::cl::desc("read ports of the GEMV unit (0: none)"),
                               llvm::cl::init(0)};
};

}  // namespace

std::unique_ptr<Pass> createSahlExpandKernelsPass(const TargetConfig &config) {
  return std::make_unique<SahlExpandKernelsPass>(config);
}
void registerSahlExpandKernelsPass() { PassRegistration<SahlExpandKernelsPass>(); }

}  // namespace mlir::iree_compiler::sa

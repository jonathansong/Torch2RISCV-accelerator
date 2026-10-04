// sahl-to-sahw (docs/iree_compiler_plan.md §8.4 step 10, §8.14 C8): one
// sahw.template per dispatch function in sahl form. The decisions come from
// the sahl passes before it (sa-to-sahl: kernels, gathers, to_i8, scatters;
// sahl-tile: pieces; sahl-plan-memory: places and layouts; sahl-schedule:
// the linear kernels' chunks); this pass translates them into commands:
//   - local buffers: words of SPAD_A / ACC in order at first use
//     (SahlLocalMemory.h), released after a piece, a chunk or a row;
//   - DDR views: binding subspan + subview offsets -> a BASE register (binding
//     + the subspan's offset, from push constants: Lin) + a static offset;
//   - linalg.generic: every arith / math operation of the body becomes one
//     single-stage VE (sahw-fuse-ve merges them); indexing maps give the
//     operand index modes (SahlLowerGeneric.cpp);
//   - kernels: the linear, attention and contraction schedules (SahlLowerKernels.cpp).
// This file: the pass, the block walk, registers, local memory, DDR / DMA,
// the VE helpers, scatter.
#include "SahlLower.h"

namespace mlir::iree_compiler::sa {
namespace lower {

bool Lowerer::run() {
  // the kernels sa-to-sahl grouped: each one's plan from its anchor
  for (auto k : f.getBody().front().getOps<sahl::KernelOp>()) {
    Operation *a = nullptr;
    for (Operation &o : k.getBody().front())
      if (o.hasAttr("sahl.anchor")) a = &o;
    StringRef kind = k.getKind();
    bool ok = false;
    if (!a) ok = false;
    else if (kind == "linear") ok = isa<linalg::GenericOp>(a) && km.matchLinear(cast<linalg::GenericOp>(a));
    else if (kind == "attention") ok = isa<linalg::BatchMatmulOp>(a) && km.matchAttention(cast<linalg::BatchMatmulOp>(a));
    else if (kind == "contraction") ok = isa<linalg::LinalgOp>(a) && km.matchContraction(cast<linalg::LinalgOp>(a));
    if (!ok) return fail("sahl.kernel \"" + kind.str() + "\": no plan for its operations");
    anchorOf[k] = a;
  }
  return lowerBlock(f.getBody().front());
}

bool Lowerer::lowerBlock(Block &block) {
  for (Operation &opRef : block) {
    Operation *op = &opRef;
    loc = op->getLoc();
    if (auto sc = dyn_cast<sahl::ScopeOp>(op)) {   // a piece (sahl-tile): its local memory released after it
      auto mark = mem.mark();
      if (sc.getKeep()) {                          // an expanded kernel's step: the outer buffers stay
        if (auto sf = sc.getSpadFrom()) mem.takeSpad(uint32_t(*sf));
        auto savedLocals = locals;
        auto savedBcast = bcastOf;
        if (!lowerBlock(sc.getBody().front())) return false;
        mem.release(mark);
        locals = std::move(savedLocals);
        bcastOf = std::move(savedBcast);
        continue;
      }
      locals = reserved;
      bcastOf.clear();
      if (!lowerBlock(sc.getBody().front())) return false;
      mem.release(mark);
      continue;
    }
    if (auto cs = dyn_cast<sahl::ClaimSpadOp>(op)) {
      if (mem.spadUsed()) return fail("a kernel claiming SPAD_A after other SPAD_A buffers");
      mem.takeSpad(uint32_t(cs.getWords()));
      continue;
    }
    if (auto tr = dyn_cast<sahl::TransposeOp>(op)) {
      auto a = bufOf(tr.getSrc()), b = bufOf(tr.getDst());
      if (!a || !b) return false;
      auto st = cast<MemRefType>(tr.getSrc().getType());
      if (st.getRank() != 2 || st.getDimSize(1) % d) return fail("transpose source [R, C]");
      // a dynamic row count (the cache's T): the length at the bound, T x C dynamic
      int64_t rows = st.isDynamicDim(0) ? cfg.maxDynamic : st.getDimSize(0);
      Value dl;
      if (a->dyn) {
        Lin x = *a->dyn;
        x.mul *= st.getDimSize(1);
        x.add *= st.getDimSize(1);
        dl = paramFor(x);
      }
      auto t = sahw::TransposeOp::create(bb, loc, int64_t(a->la), int64_t(b->la), rows * st.getDimSize(1),
                                         int64_t(::sa::vtypes(::sa::VT_I8, ::sa::VT_I8)), st.getDimSize(1) / d,
                                         dl ? ValueRange{dl} : ValueRange{});
      if (dl) {
        t.setDynFields(ArrayRef<int32_t>{::sa::DYN_VE_LEN});
        t.setDynAdd(ArrayRef<bool>{false});
      }
      continue;
    }
    if (auto lp = dyn_cast<sahl::LoopOp>(op)) {
      if (!loop(lp)) return false;
      continue;
    }
    if (auto st = dyn_cast<sahl::StripOp>(op)) {
      if (!strip(st)) return false;
      continue;
    }
    if (auto mm = dyn_cast<sahl::MmaOp>(op)) {
      if (!mma(mm)) return false;
      continue;
    }
    if (auto bc = dyn_cast<sahl::BcastOp>(op)) {
      auto l = bufOf(bc.getSrc());
      if (!l) return false;
      auto b = cast<MemRefType>(bc.getSrc().getType()).getRank() == 0
                   ? scalarBcast(bc.getSrc(), *l)
                   : perElementBcast(bc.getSrc(), *l, cast<MemRefType>(bc.getSrc().getType()).getNumElements());
      if (!b) return false;
      locals[bc.getDst()] = *b;
      continue;
    }
    if (auto rs = dyn_cast<sahl::ReserveOp>(op)) {  // a buffer shared by the pieces: allocated now
      auto l = bufOf(rs.getBuffer(), rs.getBcast());
      if (!l) return false;
      reserved[rs.getBuffer()] = *l;
      continue;
    }
    if (auto k = dyn_cast<sahl::KernelOp>(op)) {
      Operation *a = anchorOf.lookup(k);
      if (auto it = km.linears.find(a); it != km.linears.end()) {
        curKernel = k;
        bool ok = linear(it->second);
        curKernel = {};
        if (!ok) return false;
      } else if (auto it = km.attns.find(a); it != km.attns.end()) {
        if (!attention(it->second)) return false;
      } else if (!contraction(km.contracts[a])) {
        return false;
      }
      continue;
    }
    if (isa<arith::ConstantOp, memref::AllocOp, memref::DeallocOp, memref::SubViewOp, memref::CastOp,
            memref::DimOp, memref::CollapseShapeOp, IREE::HAL::InterfaceBindingSubspanOp, IREE::HAL::InterfaceConstantLoadOp,
            IREE::TensorExt::DispatchWorkloadOrdinalOp, func::ReturnOp>(op))
      continue;
    if (op->getDialect() && op->getDialect()->getNamespace() == "util") continue;
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtUIOp, arith::ShLIOp, arith::OrIOp, arith::ShRUIOp,
            arith::DivUIOp>(op))
      continue;
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
      // an expanded kernel's epilogue: its temps next to its accumulator (that ACC bank first)
      auto pb = g->getAttrOfType<IntegerAttr>("sa.prefer_bank");
      if (pb) mem.preferBank = int(pb.getInt());
      bool ok = generic(g);
      mem.preferBank = -1;
      if (!ok) return false;
      continue;
    }
    if (auto sc = dyn_cast<sahl::ScatterOp>(op)) {
      if (!scatter(sc)) return false;
      continue;
    }
    return fail("unsupported operation " + op->getName().getStringRef().str());
  }
  return err.empty();
}

KernelSchedule Lowerer::scheduleOf(const LinearPlan &p) {
  KernelSchedule ks = linearSchedule(p, lay, d);
  if (curKernel) {
    if (auto c = curKernel->getAttrOfType<IntegerAttr>("chunk_tiles")) ks.chunkTiles = c.getInt();
    if (auto l = curKernel->getAttrOfType<BoolAttr>("loop")) ks.loop = l.getValue();
  }
  return ks;
}

bool Lowerer::fail(const std::string &m) {
  if (err.empty()) err = m;
  return false;
}

Value Lowerer::baseFor(int binding, const Lin &off) {
  Lin key = off.isConst() ? Lin{-1, 1, 0, 0} : off;
  auto it = bases.find({binding, key});
  if (it != bases.end()) return it->second;
  Value v = sahw::BaseOp::create(regB, loc, regB.getType<sahw::BaseType>(), int64_t(binding), int64_t(key.ord),
                                 key.mul, int64_t(key.shift), key.add, IntegerAttr());
  bases[{binding, key}] = v;
  return v;
}

Value Lowerer::paramFor(const Lin &v) {
  if (auto it = params.find(v); it != params.end()) return it->second;
  Value p = sahw::ParamOp::create(regB, loc, regB.getType<sahw::ParamType>(), int64_t(v.ord), v.mul,
                                  int64_t(v.shift), v.add, IntegerAttr());
  params[v] = p;
  return p;
}

LocalBuf Lowerer::newLocal(VType vt, int64_t n, bool bcast) {
  LocalBuf l;
  l.vt = vt;
  l.n = n;
  l.bcast = bcast;
  uint32_t words = std::max<uint32_t>(bcast ? uint32_t(n) : uint32_t((n + d - 1) / d), 1);
  if (vt == ::sa::VT_I8) {
    auto w = mem.allocSpad(words);
    if (!w) fail("SPAD_A full");
    l.la = ::sa::laddr(::sa::MEM_SPAD_A, w.value_or(0));
  } else {
    auto w = mem.allocAcc(words);
    if (!w) fail("ACC full");
    l.la = ::sa::acc(w.value_or(0));
  }
  return l;
}

// A placed buffer's elements (a dynamic dimension at max_dynamic; its Lin kept in l.dyn).
int64_t Lowerer::placedElems(memref::AllocOp alloc, LocalBuf &l) {
  auto mt = cast<MemRefType>(alloc.getType());
  int64_t n = 1;
  for (int64_t i = 0; i < mt.getRank(); ++i) n *= mt.isDynamicDim(i) ? cfg.maxDynamic : mt.getDimSize(i);
  if (!alloc.getDynamicSizes().empty()) l.dyn = dynLin(alloc.getDynamicSizes()[0]);
  return n;
}

std::optional<LocalBuf> Lowerer::bufOf(Value v, bool bcast) {
  if (auto it = locals.find(v); it != locals.end()) return it->second;
  auto mt = cast<MemRefType>(v.getType());
  if (auto sv = v.getDefiningOp<memref::SubViewOp>(); sv && !ddrRoot(v)) {
    // a view of a local buffer (an expanded kernel's step): its words
    auto base = bufOf(sv.getSource());
    if (!base) return std::nullopt;
    for (int64_t o : sv.getStaticOffsets())
      if (ShapedType::isDynamic(o)) return fail("dynamic local view offset"), std::nullopt;
    LocalBuf l = *base;
    if (base->rowStride) {
      // a row (or rows) of a buffer in the rows layout ([H, ?]: rowStride words a row)
      auto offs = sv.getStaticOffsets();
      if (offs.size() != 2 || offs[1] != 0) return fail("view of a rows-layout buffer"), std::nullopt;
      l.la += uint32_t(offs[0]) * base->rowStride;
      l.rows = 1;
    } else {
      SmallVector<int64_t> ss;
      int64_t so;
      if (failed(sv.getSourceType().getStridesAndOffset(ss, so))) return fail("local view layout"), std::nullopt;
      int64_t elem = 0;
      for (auto [o, str] : llvm::zip(sv.getStaticOffsets(), ss)) {
        if (o && ShapedType::isDynamic(str)) return fail("local view layout"), std::nullopt;
        if (o) elem += o * str;
      }
      // one word per element in the broadcast layout, D elements per word packed
      if (!base->bcast && elem % d) return fail("local view not at a word boundary"), std::nullopt;
      l.la += uint32_t(base->bcast ? elem : elem / d);
    }
    if (mt.hasStaticShape()) {
      l.n = mt.getNumElements();
      l.dyn.reset();
      l.rowStride = 0;
    } else {                                       // [?] of it: the length dynamic (e.g. the cache's T)
      if (mt.getRank() != 1 || sv.getSizes().size() != 1) return fail("dynamic local view"), std::nullopt;
      auto dl = dynLin(sv.getSizes()[0]);
      if (!dl) return fail("dynamic local view size"), std::nullopt;
      l.dyn = dl;
      l.n = cfg.maxDynamic;
      l.rowStride = 0;
    }
    return l;
  }
  auto alloc = v.getDefiningOp<memref::AllocOp>();
  if (!alloc) return fail("operand is not a local buffer"), std::nullopt;
  auto vt = vtOf(mt.getElementType());
  if (auto w = alloc->getAttrOfType<IntegerAttr>("sa.word")) {
    auto pm = alloc->getAttrOfType<StringAttr>("sa.mem");
    if (pm && pm.getValue() == "spad_a") {         // a fixed place in SPAD_A (after sahl.claim_spad)
      if (!mt.getElementType().isInteger(8)) return fail("placed SPAD_A buffer"), std::nullopt;
      LocalBuf l;
      l.vt = ::sa::VT_I8;
      l.n = placedElems(alloc, l);
      l.la = ::sa::laddr(::sa::MEM_SPAD_A, uint32_t(w.getInt()));
      locals[v] = l;
      return l;
    }
    // a fixed place in ACC (sahl-expand-kernels: decode's linear chunk buffers;
    // an i64 accumulator, as IREE types it, holds int32 words: its epilogue truncates)
    Type et = mt.getElementType();
    if (!mt.hasStaticShape() || !(et.isF32() || et.isInteger(32) || et.isInteger(64))) return fail("placed buffer"), std::nullopt;
    LocalBuf l;
    l.vt = et.isF32() ? ::sa::VT_F32 : ::sa::VT_I32;
    l.n = mt.getNumElements();
    l.la = ::sa::acc(uint32_t(w.getInt()));
    locals[v] = l;
    return l;
  }
  if (auto pm = alloc->getAttrOfType<StringAttr>("sa.mem"); pm && pm.getValue() == "spad_b") {
    // weight tiles in a SPAD_B bank (sahl-expand-kernels: the bank chosen)
    auto bank = alloc->getAttrOfType<IntegerAttr>("sa.bank");
    if (!bank || !mt.getElementType().isInteger(8)) return fail("SPAD_B buffer"), std::nullopt;
    LocalBuf l;
    l.vt = ::sa::VT_I8;
    l.n = placedElems(alloc, l);
    l.la = ::sa::laddr(::sa::MEM_SPAD_B, uint32_t(bank.getInt()) * lay.sbank);
    locals[v] = l;
    return l;
  }
  if (alloc->hasAttr("sa.accumulator") && mt.getElementType().isInteger(64) && mt.hasStaticShape()) {
    // an expanded kernel's accumulator typed i64 as IREE gives it: int32 words (its epilogue truncates)
    LocalBuf l = newLocal(::sa::VT_I32, mt.getNumElements());
    locals[v] = l;
    return l;
  }
  // the plan (sahl-plan-memory): the layout; the place checked
  if (auto pl = alloc->getAttrOfType<StringAttr>("sa.layout")) bcast |= pl.getValue() == "bcast";
  if (auto pm = alloc->getAttrOfType<StringAttr>("sa.mem");
      pm && (pm.getValue() == "spad_a") != mt.getElementType().isInteger(8))
    return fail("local buffer placed in " + pm.getValue().str() + " against its type"), std::nullopt;
  int64_t n = mt.hasStaticShape() ? mt.getNumElements() : 0;
  if (mt.getElementType().isInteger(64)) vt = ::sa::VT_I32, n *= 2;       // (low, high) in two lanes
  if (!vt) return fail("local buffer type"), std::nullopt;
  if (!mt.hasStaticShape()) {
    // [?] or [H, ?]: rows of max_dynamic elements
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

std::optional<Lowerer::Ddr> Lowerer::ddrOf(Value v, bool allowPitch) {
  auto mt = cast<MemRefType>(v.getType());
  std::optional<Lin> dynRows;
  if (!mt.hasStaticShape() && v.getDefiningOp<memref::SubViewOp>() && mt.isDynamicDim(0)) {
    // a view of rows whose count is dynamic (an expanded kernel's cache rows):
    // the fields at the bound (max_dynamic rows), the count kept for a dynamic field
    bool rest = true;
    for (int64_t i = 1; i < mt.getRank(); ++i) rest &= !mt.isDynamicDim(i);
    auto sv = v.getDefiningOp<memref::SubViewOp>();
    if (!rest || sv.getSizes().empty()) return fail("dynamic DDR view"), std::nullopt;
    dynRows = dynLin(sv.getSizes()[0]);
    if (!dynRows) return fail("dynamic DDR view size"), std::nullopt;
    SmallVector<int64_t> shape(mt.getShape());
    shape[0] = cfg.maxDynamic;
    mt = MemRefType::get(shape, mt.getElementType(), mt.getLayout(), mt.getMemorySpace());
  }
  if (!mt.hasStaticShape()) {
    // a binding of [?] or [H, ?] (the rows contiguous; or such a view of a
    // binding with trailing unit dimensions collapsed)
    auto sub = v.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (auto c = v.getDefiningOp<memref::CollapseShapeOp>(); c && !sub) {
      auto st = c.getSrcType();
      bool unitTail = true;
      for (int64_t i = mt.getRank(); i < st.getRank(); ++i) unitTail &= !st.isDynamicDim(i) && st.getDimSize(i) == 1;
      if (unitTail) sub = c.getSrc().getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    }
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
    if (auto c = dyn_cast<memref::CollapseShapeOp>(op)) {   // (a contiguous view flattened: sahl-tile's pieces)
      cur = c.getSrc();
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
    r.dynRows = dynRows;
    if (pitchElems) {
      if (!allowPitch) return fail("non-contiguous DDR view"), std::nullopt;
      r.rows = mt.getDimSize(0);
      r.pitch = pitchElems * es;
    }
    return r;
  }
}

bool Lowerer::dynamicDma(const Ddr &r, const LocalBuf &l, bool isLoad) {
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

void Lowerer::contiguousDma(bool isLoad, Value base, int64_t off, uint32_t la, uint32_t bytes) {
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

bool Lowerer::rowsDma(bool isLoad, const Ddr &r, uint32_t la) {
  int64_t rb = r.n / r.rows * esize(r.et);
  uint32_t wb = (la >> 28) == uint32_t(::sa::MEM_ACC) ? 4 * uint32_t(d) : uint32_t(d);
  if (r.off % 8 || rb % 8 || r.pitch % 8 || rb % wb || rb > 65535 || r.rows > 65535)
    return fail("column slice: rows not whole words / 8-byte aligned");
  if (isLoad) sahw::LdOp::create(bb, loc, r.base, r.off, int64_t(la), r.rows, rb, r.pitch, 0, ValueRange{});
  else sahw::StOp::create(bb, loc, r.base, r.off, int64_t(la), r.rows, rb, r.pitch, ValueRange{});
  return true;
}

bool Lowerer::scalarByParam(const Ddr &r, const LocalBuf &dst) {
  Value pv = privateParam();
  sahw::LdParamOp::create(bb, loc, r.base, pv, r.off, 1, 0, ValueRange{});
  LocalBuf ones = newLocal(::sa::VT_F32, 1, true);
  ve({ones.la, ::sa::VT_I32}, std::nullopt, ones.la, ::sa::VT_F32, d, ::sa::VOP_COPY, 0.0f, 1.0f);
  ve({ones.la}, std::nullopt, dst.la, ::sa::VT_F32, d, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE,
     ::sa::RED_NONE, 0, 0, {{::sa::DYN_VE_A, pv, false}});
  return true;
}

// The bytes between the rows of a view (its leading stride; `rb` for a 1-D view).
int64_t Lowerer::leadingPitch(Value v, int64_t rb) {
  auto mt = cast<MemRefType>(v.getType());
  SmallVector<int64_t> st;
  int64_t off;
  if (mt.getRank() < 2 || failed(mt.getStridesAndOffset(st, off)) || ShapedType::isDynamic(st[0])) return rb;
  return st[0] * esize(mt.getElementType());
}

bool Lowerer::load(sahl::LoadOp l) {
  auto r = ddrOf(l.getSrc(), /*allowPitch=*/true);
  if (!r) return false;
  if (r->et.isInteger(64)) return fail("i64 load");
  auto dst = bufOf(l.getDst());
  if (!dst) return false;
  Value adv = advanceOf(l);
  if ((dst->la >> 28) == uint32_t(::sa::MEM_SPAD_B)) {
    // weight tiles [nc, K, D] into a SPAD_B bank: one row per tile; with
    // sa.prefix in the template's prefix (it may run during an earlier dispatch)
    // (rows of a larger matrix: their pitch; sa.interleave: K-major rows, interleaved)
    auto st = cast<MemRefType>(l.getSrc().getType());
    int64_t rows = st.isDynamicDim(0) ? cfg.maxDynamic : st.getDimSize(0), rb = r->n / rows * esize(r->et),
            pitch = leadingPitch(l.getSrc(), rb);
    if (r->dyn || r->off % 8 || rb % 8 || rb > 65535 || pitch % 8) return fail("weight tiles load");
    const int64_t ilv = l->hasAttr("sa.interleave") ? 1 : 0;
    if (r->dynRows) {                              // a dynamic number of rows (the cache's T)
      if (adv || l->hasAttr("sa.prefix")) return fail("weight tiles load: dynamic rows and a loop / prefix");
      auto ld = sahw::LdOp::create(bb, loc, r->base, r->off, int64_t(dst->la), rows, rb, pitch, ilv,
                                   ValueRange{paramFor(*r->dynRows)});
      ld.setDynFields(ArrayRef<int32_t>{::sa::DYN_DMA_ROWS});
      ld.setDynAdd(ArrayRef<bool>{false});
      return true;
    }
    OpBuilder *wb = &bb;
    std::optional<OpBuilder> pb;
    if (l->hasAttr("sa.prefix")) {
      auto pre = sahw::PrefixOp::create(regB, loc);
      pb.emplace(OpBuilder::atBlockEnd(&pre.getRegion().emplaceBlock()));
      wb = &*pb;
    }
    advanced(sahw::LdOp::create(*wb, loc, r->base, r->off, int64_t(dst->la), rows, rb, pitch, ilv,
                                adv ? ValueRange{adv} : ValueRange{}), adv);
    return true;
  }
  if (l->hasAttr("sa.rows")) {                     // one DMA row per row of the view (rows to transpose)
    auto st = cast<MemRefType>(l.getSrc().getType());
    int64_t rows = st.isDynamicDim(0) ? cfg.maxDynamic : st.getDimSize(0), rb = r->n / rows * esize(r->et),
            pitch = leadingPitch(l.getSrc(), rb);
    if (r->dyn || r->off % 8 || rb > 65535) return fail("rows load");
    auto ld = sahw::LdOp::create(bb, loc, r->base, r->off, int64_t(dst->la), rows, rb, pitch, 0,
                                 r->dynRows ? ValueRange{paramFor(*r->dynRows)} : ValueRange{});
    if (r->dynRows) {
      ld.setDynFields(ArrayRef<int32_t>{::sa::DYN_DMA_ROWS});
      ld.setDynAdd(ArrayRef<bool>{false});
    }
    return true;
  }
  if (adv) {                                       // a contiguous row moving on with the loop
    uint32_t bytes = uint32_t((r->n * esize(r->et) + 7) / 8 * 8);
    if (r->pitch || r->dyn || r->off % 8 || bytes > 65535) return fail("advancing load");
    advanced(sahw::LdOp::create(bb, loc, r->base, r->off, int64_t(dst->la), 1, bytes, bytes, 0, ValueRange{adv}), adv);
    return true;
  }
  if (r->dyn) return dynamicDma(*r, *dst, true);
  if (r->pitch) return rowsDma(true, *r, dst->la);
  if (r->off % 8 && r->n == 1 && r->et.isF32()) return scalarByParam(*r, *dst);
  uint32_t bytes = uint32_t((r->n * esize(r->et) + 7) / 8 * 8);
  if (r->off % 8) return fail("load not 8-byte aligned");
  contiguousDma(true, r->base, r->off, dst->la, bytes);
  return true;
}

bool Lowerer::store(sahl::StoreOp s) {
  auto r = ddrOf(s.getDst(), /*allowPitch=*/true);
  if (!r) return false;
  auto it = locals.find(s.getSrc());
  if (it == locals.end()) return fail("stored buffer not computed");
  LocalBuf l = it->second;
  if (Value adv = advanceOf(s)) {                  // a contiguous row moving on with the loop
    uint32_t bytes = uint32_t((r->n * esize(r->et) + 7) / 8 * 8);
    if (r->pitch || r->dyn || r->off % 8 || bytes > 65535) return fail("advancing store");
    advanced(sahw::StOp::create(bb, loc, r->base, r->off, int64_t(l.la), 1, bytes, bytes, ValueRange{adv}), adv);
    return true;
  }
  if (r->dyn) return dynamicDma(*r, l, false);
  if (l.bcast && l.n > 1) l = packBcast(l);
  if (r->pitch) return rowsDma(false, *r, l.la);
  uint32_t bytes = uint32_t((r->n * esize(r->et) + 7) / 8 * 8);
  if (r->off % 8) return fail("store not 8-byte aligned");
  contiguousDma(false, r->base, r->off, l.la, bytes);
  return true;
}

bool Lowerer::strip(sahl::StripOp st) {
  auto dst = bufOf(st.getDst());
  if (!dst) return false;
  if (!ddrRoot(st.getSrc())) {
    // a local vector [K] -> every element over the D rows (a VE copy, DIV mode)
    auto x = bufOf(st.getSrc());
    if (!x) return false;
    SmallVector<DynF> dyn;
    if (x->dyn) {                                  // a dynamic length (x * D elements)
      Lin l = *x->dyn;
      l.mul *= d;
      l.add *= d;
      dyn.push_back({::sa::DYN_VE_LEN, paramFor(l), false});
    }
    ve({x->la, x->vt, ::sa::IDX_DIV, uint32_t(d)}, std::nullopt, dst->la, ::sa::VT_I8, x->n * d, ::sa::VOP_COPY,
       1.0f, NEG0, ::sa::FUNC_NONE, ::sa::RED_NONE, 0, 0, dyn);
    return true;
  }
  // D rows of int8 x [D, K] -> the A strip: one LD with INTERLEAVE
  auto r = ddrOf(st.getSrc(), /*allowPitch=*/true);
  if (!r) return false;
  auto mt = cast<MemRefType>(st.getSrc().getType());
  if (mt.getRank() != 2 || !mt.getElementType().isInteger(8) || r->dyn) return fail("strip source");
  int64_t rows = mt.getDimSize(0), k = mt.getDimSize(1), pitch = r->pitch ? r->pitch : k;
  if (r->off % 8 || k % 8) return fail("strip source alignment");
  sahw::LdOp::create(bb, loc, r->base, r->off, int64_t(dst->la), rows, k, pitch, /*INTERLEAVE=*/1, ValueRange{});
  return true;
}

Value Lowerer::advanceOf(Operation *op) {
  auto a = op->getAttrOfType<IntegerAttr>("sa.advance");
  if (!a) return Value();
  if (size_t(a.getInt()) >= loopParams.size()) return fail("sa.advance outside a sahl.loop"), Value();
  return loopParams[a.getInt()];
}

void Lowerer::advanced(Operation *op, Value adv) {
  if (!adv) return;
  auto mark = [](auto o) {
    o.setDynFields(ArrayRef<int32_t>{::sa::DYN_DMA_DDR});
    o.setDynAdd(ArrayRef<bool>{true});
  };
  if (auto l = dyn_cast<sahw::LdOp>(op)) mark(l);
  else if (auto st = dyn_cast<sahw::StOp>(op)) mark(st);
}

bool Lowerer::loop(sahl::LoopOp lp) {
  // LOOP_END: a private PARAM per stride (0 first), the body, the loop around it
  if (!loopParams.empty() || lp.getStrides().size() > 2 || lp.getStrides().empty()) return fail("sahl.loop form");
  for (size_t j = 0; j < lp.getStrides().size(); ++j) loopParams.push_back(privateParam());
  SmallVector<int64_t> zeros(loopParams.size(), 0);
  sahw::SetRegOp::create(bb, loc, ValueRange(loopParams), zeros, 0, ValueRange{});
  Block *blk = bb.getInsertionBlock();
  Operation *before = blk->empty() ? nullptr : &blk->back();
  if (!lowerBlock(lp.getBody().front())) return false;
  ArrayRef<int64_t> st = lp.getStrides();
  auto l = sahw::LoopOp::create(bb, loc, lp.getCount(), loopParams[0], st[0],
                                loopParams.size() > 1 ? loopParams[1] : Value(), st.size() > 1 ? st[1] : 0);
  Block *lb = &l.getRegion().emplaceBlock();
  SmallVector<Operation *> moved;
  for (Operation *o = before ? before->getNextNode() : &blk->front(); o && o != l.getOperation(); o = o->getNextNode())
    moved.push_back(o);
  for (Operation *o : moved) o->moveBefore(lb, lb->end());
  loopParams.clear();
  return true;
}

bool Lowerer::mma(sahl::MmaOp m) {
  // acc = A strip x B tiles: one EX (nc tiles of K / D words each)
  auto a = bufOf(m.getA()), b = bufOf(m.getB()), c = bufOf(m.getAcc());
  if (!a || !b || !c) return false;
  auto bt = cast<MemRefType>(m.getB().getType());
  if (bt.getRank() != 3 || bt.getDimSize(2) != d) return fail("mma: B tiles [nc, K, D]");
  // dynamic sizes (the cache's T): the fields at the bound; a dynamic K -> KT
  // (K / D) and the B step, a dynamic tile count -> the repeat
  auto alloc = m.getB().getDefiningOp<memref::AllocOp>();
  auto sizeLin = [&](int dim) -> std::optional<Lin> {
    if (!alloc) return std::nullopt;
    int j = 0;
    for (int i = 0; i < dim; ++i) j += bt.isDynamicDim(i);
    return dynLin(alloc.getDynamicSizes()[j]);
  };
  const int64_t logd = int64_t(llvm::Log2_64(uint64_t(d)));
  const int64_t nc = bt.isDynamicDim(0) ? cfg.maxDynamic / d : bt.getDimSize(0);
  const int64_t k = bt.isDynamicDim(1) ? cfg.maxDynamic : bt.getDimSize(1);
  SmallVector<Value> dv;
  SmallVector<int32_t> df;
  if (bt.isDynamicDim(1)) {
    auto kl = sizeLin(1);
    if (!kl) return fail("mma: dynamic K");
    Lin kt = *kl;
    kt.shift += int(logd);
    dv.append({paramFor(kt), paramFor(*kl)});
    df.append({::sa::DYN_EX_KT, ::sa::DYN_EX_BSTEP});
  }
  if (bt.isDynamicDim(0)) {
    auto nl = sizeLin(0);
    if (!nl) return fail("mma: dynamic tile count");
    dv.push_back(paramFor(*nl));
    df.push_back(::sa::DYN_EX_REPEAT);
  }
  auto ex = sahw::ExOp::create(bb, loc, int64_t(a->la & 0xFFFFFFF), int64_t(b->la & 0xFFFFFFF),
                               int64_t(c->la & 0xFFFFFFF), k / d, m.getAccumulate(), nc, k, 1, nc, dv);
  if (!df.empty()) {
    ex.setDynFields(df);
    ex.setDynAdd(SmallVector<bool>(df.size(), false));
  }
  return true;
}

uint32_t Lowerer::types(const Opd &s1, const std::optional<Opd> &s2, VType out) {
  uint32_t t = ::sa::vtypes(s1.vt, out);
  if (s2 && !s2->isImm && s2->vt != s1.vt) t |= uint32_t(s2->vt == ::sa::VT_I8 ? 1 : s2->vt) << 4;
  return t;
}

void Lowerer::ve(const Opd &s1, const std::optional<Opd> &s2, uint32_t dst, VType out, int64_t n, ::sa::VOp op,
        float A, float B, ::sa::VFunc fn, ::sa::VRed red,
        uint32_t rowlen, uint32_t valid, ArrayRef<DynF> dyn, bool swapneg) {
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

std::optional<LocalBuf> Lowerer::materialize(Value t) {
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

std::optional<LocalBuf> Lowerer::scalarBcast(Value v, const LocalBuf &l) {
  if (l.bcast) return l;
  if (auto it = bcastOf.find(v); it != bcastOf.end()) return it->second;
  if (l.vt != ::sa::VT_F32) return fail("broadcast of a non-fp32 scalar"), std::nullopt;
  LocalBuf b = newLocal(::sa::VT_F32, 1, true);
  ve({l.la}, std::nullopt, b.la, ::sa::VT_F32, d, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE, ::sa::RED_MAX, 0, 1);
  bcastOf[v] = b;
  return b;
}

std::optional<LocalBuf> Lowerer::perElementBcast(Value v, const LocalBuf &l, int64_t n) {
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

LocalBuf Lowerer::packBcast(const LocalBuf &b) {
  uint32_t w = uint32_t((b.n + d - 1) / d);
  LocalBuf p = newLocal(::sa::VT_F32, int64_t(w + d - 1) * d);
  p.n = b.n;
  for (uint32_t k = 0; k < w; ++k)
    sahw::TransposeOp::create(bb, loc, int64_t(b.la + k * d), int64_t(p.la + k), int64_t(d * d),
                              int64_t(::sa::vtypes(::sa::VT_F32, ::sa::VT_F32)), 1, ValueRange{});
  return p;
}

bool Lowerer::scatter(sahl::ScatterOp sc) {
  Value upd = sc.getUpdates(), idx = sc.getIndices(), orig = sc.getDest();
  auto uT = cast<MemRefType>(upd.getType());
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

}  // namespace lower

namespace {
using lower::Lowerer;

struct SahlToSahwPass : public PassWrapper<SahlToSahwPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SahlToSahwPass)
  SahlToSahwPass() = default;
  explicit SahlToSahwPass(const TargetConfig &c) : cfg(c), fromOptions(false) {}
  SahlToSahwPass(const SahlToSahwPass &o) : PassWrapper(o), cfg(o.cfg), fromOptions(o.fromOptions) {}
  StringRef getArgument() const override { return "iree-sahl-to-sahw"; }
  StringRef getDescription() const override {
    return "Lowers each sahl function (linalg on local buffers + sahl.load / store) to a sahw.template";
  }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<sahw::SahwDialect>(); }

  void runOnOperation() override {
    if (fromOptions) {                         // iree-opt: the target configuration from the pass options
      cfg.d = optD;
      cfg.spadBytes = optSpadKB * 1024;
      cfg.accBytes = optAccKB * 1024;
      cfg.ukernels = optUkernels;
    }
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
  bool fromOptions = true;
  Option<int64_t> optD{*this, "d", llvm::cl::desc("array size D"), llvm::cl::init(8)};
  Option<int64_t> optSpadKB{*this, "spad-kb", llvm::cl::desc("SPAD size (KB, two banks)"), llvm::cl::init(128)};
  Option<int64_t> optAccKB{*this, "acc-kb", llvm::cl::desc("ACC size (KB, two banks)"), llvm::cl::init(256)};
  Option<std::string> optUkernels{*this, "ukernels", llvm::cl::desc("micro-kernels: all, none or a list"),
                                  llvm::cl::init("all")};
};

}  // namespace

std::unique_ptr<Pass> createSahlToSahwPass(const TargetConfig &config) {
  return std::make_unique<SahlToSahwPass>(config);
}
void registerSahlToSahwPass() { PassRegistration<SahlToSahwPass>(); }

}  // namespace mlir::iree_compiler::sa

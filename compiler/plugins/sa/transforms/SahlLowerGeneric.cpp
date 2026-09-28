// linalg.generic in sahl-to-sahw (SahlLower.h): element-wise operations,
// reductions, gathers, i64 scalars.
#include "SahlLower.h"

namespace mlir::iree_compiler::sa::lower {

std::optional<Val> Lowerer::packedRowGather(linalg::GenericOp g, sahl::GatherOp ga, int64_t K, int scalarArgNo) {
  if (K * d > 65535) return fail("packed gather: a tile beyond one DMA row / LDPARAM factor"), std::nullopt;
  auto sym = ddrOf(g.getDpsInputs()[scalarArgNo]);
  auto src = ddrOf(ga.getSource());
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

bool Lowerer::intScalar(linalg::GenericOp g) {
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

std::optional<LocalBuf> Lowerer::scalarI64(Value v) {
  if (auto it = locals.find(v); it != locals.end()) return it->second;
  auto r = ddrOf(v);
  if (!r) return std::nullopt;
  if (r->off % 8) return fail("i64 scalar not 8-byte aligned"), std::nullopt;
  LocalBuf l = newLocal(::sa::VT_I32, 2);
  sahw::LdOp::create(bb, loc, r->base, r->off, int64_t(l.la), 1, 8, 8, 0, ValueRange{});
  locals[v] = l;
  return l;
}

bool Lowerer::generic(linalg::GenericOp g, const RowSel *sel, const Chunk *chunk) {
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
        ok = g.getRegion().front().getOps<sahl::GatherOp>().empty();
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
      auto savedTop = mem.mark();
      size_t nLocals = locals.size(), nBcast = bcastOf.size();
      RowSel rs{r, *dynInner};
      if (!generic(g, &rs)) return false;
      if (locals.size() == nLocals && bcastOf.size() == nBcast) {
        mem.releaseAcc(savedTop);
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

  // gathers (sahl.gather: sa-to-sahl chose the form, gatherForm; checked
  // here against the iteration space this lowering uses)
  auto domain = points(ranges);
  if (domain.size() > 65536) domain.clear();          // (prefill: M rows of a table)
  for (sahl::GatherOp ga : body.getOps<sahl::GatherOp>()) {
    Value t = ga.getSource();
    auto mt = cast<MemRefType>(t.getType());
    SmallVector<SmallVector<int64_t>> dom = nloops ? domain : SmallVector<SmallVector<int64_t>>{{}};
    GatherForm gf = gatherForm(ga, dom, nloops, scalarArgNo >= 0, d, g.getNumLoops());
    if (gf.kind.empty()) return fail(gf.why);
    if (gf.kind != ga.getKind()) return fail("gather: sa-to-sahl chose " + ga.getKind().str() + ", the lowering " + gf.kind);
    Type et = mt.getElementType();
    auto vt = vtOf(et);
    uint32_t es = esize(et);
    if (!vt) return fail("gather element type");
    Val e;
    if (gf.kind == "packed") {
      auto pe = packedRowGather(g, ga, int64_t(dom.size()), scalarArgNo);
      if (!pe) return false;
      vals[ga.getResult()] = *pe;
      skip.push_back(ga);
      continue;
    }
    if (gf.kind == "row" || gf.kind == "scalar") {
      const int64_t C = gf.rowElems;
      auto sym = ddrOf(g.getDpsInputs()[scalarArgNo]);
      auto src = ddrOf(t);
      if (!sym || !src) return fail("gather through buffers not in DDR");
      Value p = privateParam();
      if (C * es > 0xFFFF) return fail("gather row larger than 64 KB");
      sahw::LdParamOp::create(bb, loc, sym->base, p, sym->off, C * es, 0, ValueRange{});
      if (gf.kind == "scalar") {
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
        if (src->off % 8 || (C * es) % 8) return fail("gathered rows not 8-byte aligned");
        auto l = sahw::LdOp::create(bb, loc, src->base, src->off, int64_t(row.la), 1, int64_t(bytes),
                                    int64_t(bytes), 0, ValueRange{p});
        l.setDynFields(ArrayRef<int32_t>{::sa::DYN_DMA_DDR});
        l.setDynAdd(ArrayRef<bool>{true});
        e.kind = Val::Mem;
        e.o = {row.la, *vt, ::sa::IDX_LIN, 0};
        e.fresh = true;
      }
    } else {                                           // "swap": the pair swap of a small tensor
      auto l = materialize(t);
      if (!l) return false;
      e.kind = Val::SwapSrc;
      e.o = {l->la, l->vt, ::sa::IDX_LIN, 0};
    }
    vals[ga.getResult()] = e;
    skip.push_back(ga);
  }

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
    if (auto t = dyn_cast<sahl::ToI8Op>(op)) {        // the VE's int8 output conversion
      Val v;
      v.kind = Val::ToI8;
      v.inner = std::make_shared<Val>(valOf(t.getX()));
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
    // a later piece (sahl-tile: sahl.accumulate): its maximum into a temp word, then into the result
    uint32_t dst = out->la;
    if (g->hasAttr("sahl.accumulate")) {
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

}  // namespace mlir::iree_compiler::sa::lower

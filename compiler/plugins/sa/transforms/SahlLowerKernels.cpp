// The kernels in sahl-to-sahw (SahlLower.h): linear layers (decode, prefill's
// rows), attention, generic contractions.
#include "SahlLower.h"

namespace mlir::iree_compiler::sa::lower {

std::optional<std::pair<Value, int64_t>> Lowerer::ddrStart(Value v, int64_t es) {
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

bool Lowerer::attention(AttnPlan &p) {
  const int64_t H = p.H, hs = p.hs, T = cfg.maxDynamic;
  if (T % d || hs % d) return fail("attention: T and head size must be multiples of D");
  if (mem.spadUsed()) return fail("attention after other SPAD_A buffers");
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
  mem.takeSpad(2 * sb);
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

bool Lowerer::contraction(ContractPlan &p) {
  auto ranges = *km.loopRanges(p.op);
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
  bool xRows = (p.bLoop >= 0 && p.x.coef[p.bLoop] < 0) || (p.gLoop >= 0 && p.x.coef[p.gLoop] < 0);
  if (mem.spadUsed()) return fail("contraction after other SPAD_A buffers");
  mem.takeSpad(2 * sb);                                      // the A strip at 0, raw rows in bank 1
  const uint32_t strip = 0, raw = sb;
  auto xs = ddrStart(p.x.view, esize(p.x.et)), ms = ddrStart(p.m.view, 1),
       ys = ddrStart(p.store.getDst(), esize(cast<MemRefType>(p.store.getDst().getType()).getElementType()));
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
  int eb = outPos(p.bLoop), eg = outPos(p.gLoop), en = outPos(p.nLoop), el = outPos(p.laneLoop);
  if ((eb >= 0 && eg >= 0 && eb > eg) || (eg >= 0 && eg > en) || (eb >= 0 && eb > en))
    return fail("contraction: output not in the order (batch, row, n)");
  // epilogue inputs: per output element (loaded per chunk), per output column
  // (the same for every (batch, row): loaded per chunk), per (batch, row) value, or a scalar
  enum class EpiKind { PerElement, PerColumn, PerRow, Scalar };
  struct EpiIn {
    EpiKind kind;
    Value src;
    LocalBuf l;
    int64_t sb = 0, sg = 0;                              // PerRow: element stride of the batch / row index
  };
  llvm::DenseMap<int, EpiIn> epiIn;
  auto emaps = p.epi ? p.epi.getIndexingMapsArray() : SmallVector<AffineMap>{};
  for (auto &[i, src] : p.epiLoads) {
    Value in = p.epi.getDpsInputs()[i];
    auto mt = cast<MemRefType>(in.getType());
    AffineMap em = emaps[i];
    EpiIn e;
    e.src = src;
    if (em.isIdentity()) {
      // fp32, or int32 (another contraction's accumulator: SwiGLU's up projection)
      if (Nr.dyn || !(mt.getElementType().isF32() || mt.getElementType().isInteger(32)))
        return fail("contraction epilogue input per element");
      e.kind = EpiKind::PerElement;
      epiIn[i] = e;
      continue;
    }
    SmallVector<int64_t> st;
    int64_t off;
    if (!mt.hasStaticShape() || failed(mt.getStridesAndOffset(st, off))) return fail("contraction epilogue input layout");
    bool onlyRow = true, onlyCol = true;
    int64_t sn = 0, sl = 0;
    for (unsigned r = 0; r < em.getNumResults(); ++r) {
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
    // per column: element n of the output row at n (tiles x lanes: tile * D + lane)
    if (!onlyRow && onlyCol && sn == (el >= 0 ? d : 1) && (el < 0 || sl == 1) && !Nr.dyn &&
        mt.getElementType().isF32()) {
      e.kind = EpiKind::PerColumn;
      epiIn[i] = e;
      continue;
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
  LocalBuf xl;
  if (!xI32 && p.layout != MatLayout::RowsK) {
    // int8 x: in SPAD_A bank 1 (only the RowsK layout uses it, for the raw rows)
    if (uint32_t(K / d) > sb) return fail("contraction: int8 x larger than a SPAD bank");
    xl.la = ::sa::laddr(::sa::MEM_SPAD_A, raw);
    xl.vt = ::sa::VT_I8;
    xl.n = K;
  } else {
    xl = newLocal(xI32 ? ::sa::VT_I32 : ::sa::VT_I8, K);
  }
  // K blocks: the A strip of a block (kc words) in one SPAD_A bank, one B tile of it
  // in one SPAD_B bank and in one DMA row (16-bit length: kc * D bytes), and the
  // strip's DIV-mode source range (the VE checks kc words from the source start,
  // no divider) inside x's memory; a longer K accumulates the blocks in ACC
  uint32_t xDepth = (xl.la >> 28) == uint32_t(::sa::MEM_ACC) ? 2 * lay.cbank : 2 * sb;
  int64_t xRoom = int64_t(xDepth) - int64_t(xl.la & 0xFFFF) - K / d;
  uint32_t kmax = std::min<uint32_t>(sb, 65535 / uint32_t(d) / uint32_t(d) * uint32_t(d));
  kmax = uint32_t(std::min<int64_t>(kmax, xRoom / d * d));
  if (kmax < uint32_t(d)) return fail("contraction: x leaves no room for a K block");
  const uint32_t kc = uint32_t(K) > kmax ? kmax : uint32_t(K);
  const uint32_t nk = uint32_t((K + kc - 1) / kc);
  if (nk > 1 && (Kr.dyn || Nr.dyn || G != 1 || xRows))
    return fail("contraction: K blocks with a dynamic K / N or several rows of x");
  // chunks of output tiles: the B tiles of a chunk (of a K block) in one SPAD_B bank
  // (a dynamic N: all tiles, one chunk)
  uint32_t nc = nt;
  if (!Nr.dyn) {
    uint32_t cap = std::max<uint32_t>(sb / kc, 1);
    for (nc = std::min(cap, nt); nc > 1 && nt % nc; --nc) {
    }
  }
  if (kc * nc > sb) return fail("contraction: one chunk of B tiles does not fit a SPAD bank");
  if (xRows && nc != nt) return fail("contraction: x rows of a dynamic length and several chunks");
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
  auto loadX = [&](int64_t b, int64_t g) -> bool {
    int64_t xoff = xs->second;
    if (!xRows) xoff += ((p.bLoop >= 0 ? b * p.x.coef[p.bLoop] : 0) + (p.gLoop >= 0 ? g * p.x.coef[p.gLoop] : 0)) *
                        int64_t(xes);
    if (xoff % 8) return fail("contraction: unaligned x");
    if (!xRows && !Kr.dyn) {                             // static: rows of at most 64 KB
      contiguousDma(true, xs->first, xoff, xl.la, uint32_t(K * xes));
      return true;
    }
    auto lx = sahw::LdOp::create(bb, loc, xs->first, xoff, int64_t(xl.la), 1, K * xes, K * xes, 0, ValueRange{});
    if (xRows) {
      Value xb = parOf(*Kr.dyn, xes, 0);
      dyn(lx, {{::sa::DYN_DMA_DDR, rowAcc, true}, {::sa::DYN_DMA_ROW_BYTES, xb, false}});
      addRow(xb);
    } else if (Kr.dyn) {
      dyn(lx, {{::sa::DYN_DMA_ROW_BYTES, parOf(*Kr.dyn, xes, 0), false}});
    }
    return true;
  };
  // x[k0, k0 + kl) -> the A strip (each element over the D rows)
  auto replicate = [&](int64_t k0, int64_t kl) {
    ve({xl.la + uint32_t(k0 / d), xl.vt, ::sa::IDX_DIV, uint32_t(d)}, std::nullopt,
       ::sa::laddr(::sa::MEM_SPAD_A, strip), ::sa::VT_I8, kl * d, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE,
       ::sa::RED_NONE, 0, 0,
       Kr.dyn ? SmallVector<DynF>{{::sa::DYN_VE_LEN, parOf(*Kr.dyn, d, 0), false}} : SmallVector<DynF>{});
  };
  auto loadStrip = [&](int64_t b, int64_t g) -> bool {
    if (!loadX(b, g)) return false;
    replicate(0, K);
    return true;
  };
  for (int64_t b = 0; b < H; ++b) {
    int64_t moff = ms->second + (p.bLoop >= 0 ? b * p.m.coef[p.bLoop] : 0);
    if (moff % 8) return fail("contraction: unaligned matrix");
    if (G == 1 && !(nk == 1 ? loadStrip(b, 0) : loadX(b, 0))) return false;
    uint32_t step = 0;                                   // B loads so far: the SPAD_B bank alternates
    // the B tiles of chunk c0, K range [k0, k0 + kl) -> a SPAD_B bank; returns its word address
    auto loadB = [&](uint32_t c0, int64_t k0, int64_t kl) -> uint32_t {
      uint32_t bw = (step++ & 1) * sb;
      uint32_t la = ::sa::laddr(::sa::MEM_SPAD_B, bw);
      if (p.layout == MatLayout::Packed) {
        sahw::LdOp::create(bb, loc, ms->first, moff + int64_t(c0) * K * d + k0 * d, int64_t(la), nc, kl * d,
                           K * d, 0, ValueRange{});
      } else if (p.layout == MatLayout::RowsK) {
        int64_t sN = p.m.coef[p.nLoop];
        auto l = sahw::LdOp::create(bb, loc, ms->first, moff + int64_t(c0) * d * sN + k0,
                                    int64_t(::sa::laddr(::sa::MEM_SPAD_A, raw)), int64_t(nc) * d, kl, sN, 0,
                                    ValueRange{});
        auto t = sahw::TransposeOp::create(bb, loc, int64_t(::sa::laddr(::sa::MEM_SPAD_A, raw)), int64_t(la),
                                           int64_t(nc) * d * kl, int64_t(::sa::vtypes(::sa::VT_I8, ::sa::VT_I8)),
                                           kl / d, ValueRange{});
        if (Nr.dyn) {
          dyn(l, {{::sa::DYN_DMA_ROWS, pN, false}});
          dyn(t, {{::sa::DYN_VE_LEN, parOf(*Nr.dyn, K, 0), false}});
        }
      } else {
        int64_t sK = p.m.coef[p.kLoop];
        auto l = sahw::LdOp::create(bb, loc, ms->first, moff + int64_t(c0) * d + k0 * sK, int64_t(la), kl,
                                    int64_t(nc) * d, sK, /*INTERLEAVE=*/1, ValueRange{});
        if (Kr.dyn) dyn(l, {{::sa::DYN_DMA_ROWS, pK, false}});
      }
      return bw;
    };
    auto exOp = [&](uint32_t bw, uint32_t cw, int64_t kl, bool accumulate) {
      auto ex = sahw::ExOp::create(bb, loc, int64_t(strip), int64_t(bw), int64_t(cw), kl / d, accumulate,
                                   int64_t(nc), kl, 1, int64_t(nc), ValueRange{});
      if (Kr.dyn) dyn(ex, {{::sa::DYN_EX_KT, parOf(*Kr.dyn, 1, logd), false}, {::sa::DYN_EX_BSTEP, pK, false}});
      if (Nr.dyn) dyn(ex, {{::sa::DYN_EX_REPEAT, parOf(*Nr.dyn, 1, logd), false}});
    };
    for (uint32_t c0 = 0; c0 < nt; c0 += nc) {
      // one K block: the B tiles of this chunk, shared by the rows of x
      uint32_t bw = nk == 1 ? loadB(c0, 0, K) : 0;
      for (int64_t g = 0; g < G; ++g) {
        if (G > 1 && !loadStrip(b, g)) return false;
        // the chunk's accumulator and epilogue buffers: released after its store
        auto savedTop = mem.mark();
        LocalBuf acc = newLocal(::sa::VT_I32, int64_t(nc) * d);
        uint32_t cw = acc.la & 0xFFFFFFF;
        if (nk == 1) {
          exOp(bw, cw, K, false);
        } else {
          for (uint32_t kb = 0; kb < nk; ++kb) {
            int64_t k0 = int64_t(kb) * kc, kl = std::min<int64_t>(kc, K - k0);
            uint32_t bwk = loadB(c0, k0, kl);
            replicate(k0, kl);
            exOp(bwk, cw, kl, kb > 0);
          }
        }
        int64_t row = b * G + g;
        if (!p.epi) {                                    // the accumulator stored as it is (int32)
          if (Nr.dyn) return fail("contraction without an epilogue: dynamic N");
          int64_t yoff = ys->second + (row * N + int64_t(c0) * d) * 4;
          if (yoff % 8) return fail("contraction: unaligned result");
          sahw::StOp::create(bb, loc, ys->first, yoff, int64_t(::sa::acc(cw)), 1, int64_t(nc) * d * 4,
                             int64_t(nc) * d * 4, ValueRange{});
          mem.releaseAcc(savedTop);
          continue;
        }
        // the epilogue on this chunk: n elements at (b, g, c0)
        Chunk ch;
        ch.n = int64_t(nc) * d;
        // the epilogue's result: fp32, int32 (an epilogue that only truncates
        // the accumulator) or int8 (one that ends in to_i8, e.g. a KV row)
        Type oet = cast<MemRefType>(p.epi.getDpsInits()[0].getType()).getElementType();
        if (!oet.isF32() && !oet.isInteger(32) && !oet.isInteger(8))
          return fail("contraction epilogue result other than fp32 / int32 / int8");
        ch.outVt = oet.isF32() ? ::sa::VT_F32 : oet.isInteger(32) ? ::sa::VT_I32 : ::sa::VT_I8;
        const int64_t es = oet.isInteger(8) ? 1 : 4;     // bytes per result element
        if (es == 1 && Nr.dyn) return fail("contraction epilogue: an int8 result of dynamic N");
        // the epilogue's SPAD temps (an int8 result): SPAD_A bank 1 after x
        // (the strip is in bank 0; released with the chunk)
        if (!xI32 && p.layout != MatLayout::RowsK) mem.takeSpad(raw + uint32_t((K + d - 1) / d));
        LocalBuf out = newLocal(ch.outVt, ch.n);
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
          if (e.kind == EpiKind::PerElement || e.kind == EpiKind::PerColumn) {
            auto r = ddrStart(e.src, 4);
            if (!r) return fail("contraction epilogue input not in DDR");
            bool i32 = cast<MemRefType>(p.epi.getDpsInputs()[i].getType()).getElementType().isInteger(32);
            VType vt = i32 ? ::sa::VT_I32 : ::sa::VT_F32;
            LocalBuf l = newLocal(vt, ch.n);
            int64_t off = r->second + ((e.kind == EpiKind::PerElement ? row * N : 0) + int64_t(c0) * d) * 4;
            if (off % 8) return fail("contraction epilogue input not 8-byte aligned");
            sahw::LdOp::create(bb, loc, r->first, off, int64_t(l.la), 1, ch.n * 4, ch.n * 4, 0, ValueRange{});
            v.o = {l.la, vt, ::sa::IDX_LIN, 0};
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
          int64_t yoff = ys->second + (row * N + int64_t(c0) * d) * es;
          if (yoff % 8) return fail("contraction: unaligned result");
          sahw::StOp::create(bb, loc, ys->first, yoff, int64_t(out.la), 1, ch.n * es, ch.n * es, ValueRange{});
        }
        mem.release(savedTop);                           // (SPAD_A: back to the two banks taken)
      }
    }
  }
  return err.empty();
}

// A per-element epilogue input's type in local memory: fp32, or int32 (the
// epilogue converts it as it converts the accumulator).
static VType chunkedVt(const LinearPlan &p, size_t m) {
  linalg::GenericOp epi = p.epi;
  Value in = epi.getDpsInputs()[p.chunked[m].first];
  return cast<MemRefType>(in.getType()).getElementType().isInteger(32) ? ::sa::VT_I32 : ::sa::VT_F32;
}

bool Lowerer::linearRows(LinearPlan &p) {
  auto wt = cast<MemRefType>(p.w.getType());
  const int64_t k = wt.getDimSize(1), nt = wt.getDimSize(0), N = nt * d, M = p.rows, nb = M / d;
  const uint32_t sb = lay.sbank;
  auto xd = ddrOf(p.x), wd = ddrOf(p.w), yd = ddrOf(p.store.getDst());
  if (!xd || !wd || !yd) return false;
  if (xd->off % 8 || wd->off % 8 || yd->off % 8 || k % 8) return fail("linear rows: unaligned operand");
  SmallVector<uint32_t> strips;
  for (int64_t rb = 0; rb < nb; ++rb) {
    LocalBuf st = newLocal(::sa::VT_I8, k * d);
    sahw::LdOp::create(bb, loc, xd->base, xd->off + rb * d * k, int64_t(st.la), d, k, k, /*INTERLEAVE=*/1,
                       ValueRange{});
    strips.push_back(st.la);
  }
  std::map<int, LocalBuf> rowB;                       // per-row inputs: a broadcast word per row
  for (auto &[i, src] : p.rowv) {
    Value in = p.epi.getDpsInputs()[i];
    auto l = materialize(src);
    if (!l) return false;
    locals[in] = *l;
    auto b = perElementBcast(in, *l, M);
    if (!b) return false;
    rowB[i] = *b;
  }
  std::map<int, LocalBuf> scal;                       // other inputs: scalars
  for (auto &[i, src] : p.whole) {
    Value in = p.epi.getDpsInputs()[i];
    if (cast<MemRefType>(in.getType()).getRank() != 0) return fail("linear rows: epilogue input layout");
    auto l = materialize(src);
    if (!l) return false;
    locals[in] = *l;
    auto b = scalarBcast(in, *l);
    if (!b) return false;
    scal[i] = *b;
  }
  SmallVector<Ddr> cd, ed;
  for (auto &[i, src] : p.cols) {
    auto r = ddrOf(src);
    if (!r || r->off % 8) return fail("linear rows: per-column input");
    cd.push_back(*r);
  }
  for (auto &[i, src] : p.chunked) {
    auto r = ddrOf(src);
    if (!r || r->off % 8) return fail("linear rows: per-element input");
    ed.push_back(*r);
  }
  // chunks (sahl-schedule): the B tiles in one SPAD_B bank, D x nc words per ACC buffer
  const int64_t nc = scheduleOf(p).chunkTiles;
  if (nc <= 0 || nt % nc || nc * k > int64_t(sb)) return fail("linear rows: one weight tile does not fit a SPAD_B bank");
  // chunk ci's B tiles in SPAD_B bank ci & 1, its per-column inputs in set ci & 1
  SmallVector<SmallVector<LocalBuf>> colSets(2);
  for (int set = 0; set < 2; ++set)
    for (size_t m = 0; m < cd.size(); ++m) colSets[set].push_back(newLocal(::sa::VT_F32, nc * d));
  const int64_t nch = nt / nc;
  auto loadChunk = [&](int64_t ci) {
    int64_t c0 = ci * nc;
    sahw::LdOp::create(bb, loc, wd->base, wd->off + c0 * k * d,
                       int64_t(::sa::laddr(::sa::MEM_SPAD_B, uint32_t(ci & 1) * sb)), nc, k * d, k * d, 0,
                       ValueRange{});
    for (size_t m = 0; m < cd.size(); ++m)
      sahw::LdOp::create(bb, loc, cd[m].base, cd[m].off + c0 * d * 4, int64_t(colSets[ci & 1][m].la), 1,
                         nc * d * 4, nc * d * 4, 0, ValueRange{});
  };
  loadChunk(0);
  for (int64_t c0 = 0, ci = 0; c0 < nt; c0 += nc, ++ci) {
    auto savedTop = mem.mark();
    uint32_t bw = uint32_t(ci & 1) * sb;
    const SmallVector<LocalBuf> &cl = colSets[ci & 1];
    for (int64_t rb = 0; rb < nb; ++rb) {
      auto savedRb = mem.mark();
      const int64_t n = d * nc * d;
      LocalBuf acc = newLocal(::sa::VT_I32, n);
      sahw::ExOp::create(bb, loc, int64_t(strips[rb] & 0xFFFFFFF), int64_t(bw), int64_t(acc.la & 0xFFFFFFF),
                         k / d, false, nc, k, 1, nc, ValueRange{});
      if (rb == 0 && ci + 1 < nch) loadChunk(ci + 1);          // prefetch into the other bank
      if (!p.epi) {                                 // the accumulator itself
        sahw::StOp::create(bb, loc, yd->base, yd->off + (rb * d * N + c0 * d) * 4, int64_t(acc.la), d, nc * d * 4,
                           N * 4, ValueRange{});
        mem.releaseAcc(savedRb);
        continue;
      }
      Chunk ch;
      ch.n = n;
      LocalBuf out = newLocal(::sa::VT_F32, n);
      ch.outLa = out.la;
      const int64_t off = (rb * d * N + c0 * d) * 4;
      for (int in = 0; in < p.epi.getNumDpsInputs(); ++in) {
        Val v;
        v.kind = Val::Mem;
        if (p.epi.getDpsInputs()[in] == p.acc) {
          v.o = {acc.la, ::sa::VT_I32, ::sa::IDX_LIN, 0};
          ch.inputs[in] = v;
        }
      }
      for (size_t m = 0; m < ed.size(); ++m) {
        VType vt = chunkedVt(p, m);
        LocalBuf l = newLocal(vt, n);
        sahw::LdOp::create(bb, loc, ed[m].base, ed[m].off + off, int64_t(l.la), d, nc * d * 4, N * 4, 0,
                           ValueRange{});
        Val v;
        v.kind = Val::Mem;
        v.o = {l.la, vt, ::sa::IDX_LIN, 0};
        ch.inputs[p.chunked[m].first] = v;
      }
      for (size_t m = 0; m < cl.size(); ++m) {
        Val v;
        v.kind = Val::Mem;
        v.o = {cl[m].la, ::sa::VT_F32, ::sa::IDX_MOD, uint32_t(nc)};
        ch.inputs[p.cols[m].first] = v;
      }
      for (auto &[i, b] : rowB) {
        Val v;
        v.kind = Val::Mem;
        v.o = {b.la + uint32_t(rb * d), ::sa::VT_F32, ::sa::IDX_DIV, uint32_t(nc)};
        ch.inputs[i] = v;
      }
      for (auto &[i, b] : scal) {
        Val v;
        v.kind = Val::Mem;
        v.o = {b.la, ::sa::VT_F32, ::sa::IDX_DIV, 0xFFFF};
        v.uni = 1;
        ch.inputs[i] = v;
      }
      if (!generic(p.epi, nullptr, &ch)) return false;
      sahw::StOp::create(bb, loc, yd->base, yd->off + off, int64_t(out.la), d, nc * d * 4, N * 4, ValueRange{});
      mem.releaseAcc(savedRb);
    }
    mem.releaseAcc(savedTop);
  }
  return err.empty();
}

bool Lowerer::linear(LinearPlan &p) {
  if (p.rows) return linearRows(p);
  auto wt = cast<MemRefType>(p.w.getType());
  const uint32_t k = uint32_t(wt.getDimSize(1)), nt = uint32_t(wt.getDimSize(0));
  const uint32_t sb = lay.sbank, cb = lay.cbank;
  if (k % d) return fail("linear: K not a multiple of D");
  // the schedule (sahl-schedule): chunks of nc output tiles, a LOOP_END loop over many
  KernelSchedule ks = scheduleOf(p);
  const uint32_t nc = uint32_t(ks.chunkTiles);
  if (nc == 0 || nt % nc || nc * k > sb) return fail("linear: one weight tile does not fit a SPAD_B bank");
  if (k + k / d > 2 * sb || lay.acc0 + k / d + 2 > cb) return fail("linear: K too large for the local memories");
  const uint32_t nch = nt / nc, wbytes = nc * k * d, fbytes = 4 * nc * d, oSw = d * nc, oY = d * nc + nc;
  const bool useLoop = ks.loop;
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
    if (!p.epi) {                                 // the accumulator itself (int32: the same bytes as fp32)
      dynAdd(sahw::StOp::create(bb, loc, yd->base, yd->off + int64_t(i) * fbytes, int64_t(::sa::acc(c)), 1, fbytes,
                                fbytes, dyn ? ValueRange{pf} : ValueRange{}),
             dyn);
      return true;
    }
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
      a.o = {::sa::acc(c + slot(m)), chunkedVt(p, m), ::sa::IDX_LIN, 0};
      ch.inputs[p.chunked[m].first] = a;
    }
    mem.preferBank = int(bank);
    bool ok = generic(p.epi, nullptr, &ch);
    mem.preferBank = -1;
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

}  // namespace mlir::iree_compiler::sa::lower

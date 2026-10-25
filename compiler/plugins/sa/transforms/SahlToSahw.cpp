// sahl-to-sahw (docs/iree_compiler_plan.md §8.4 step 10, §8.14 C8): one
// sahw.template per dispatch function in sahl form. The decisions come from
// the sahl passes before it (sa-to-sahl: kernels, gathers, to_i8, scatters;
// sahl-tile: pieces; sahl-plan-memory: places and layouts; sahl-schedule:
// the linear kernels' chunks; sahl-expand-kernels: the micro-kernels as
// strips, tiles, EX, loops, scopes); this pass translates them into commands:
//   - local buffers: words of SPAD_A / ACC in order at first use
//     (SahlLocalMemory.h), released after a piece, a chunk or a row;
//   - DDR views: binding subspan + subview offsets -> a BASE register (binding
//     + the subspan's offset, from push constants: Lin) + a static offset;
//   - linalg.generic: every arith / math operation of the body becomes one
//     single-stage VE (sahw-fuse-ve merges them); indexing maps give the
//     operand index modes (SahlLowerGeneric.cpp);
//   - the micro-kernel ops (sahl.strip / mma / bcast / transpose / loop /
//     scope / claim_spad / cursor_reset) and the placed buffers (sa.word, sa.bank);
//     a sahl.kernel left over is an error (the dispatch runs on the host).
// This file: the pass, the block walk, registers, local memory, DDR / DMA,
// the VE helpers, scatter.
#include "SahlLower.h"

namespace mlir::iree_compiler::sa {
namespace lower {

bool Lowerer::run() {
  // the micro-kernels are written out by sahl-expand-kernels; one left has a
  // form it does not cover (the dispatch then runs on the host)
  for (auto k : f.getBody().front().getOps<sahl::KernelOp>()) {
    auto why = k->getAttrOfType<StringAttr>("sa.unexpanded");
    return fail("sahl.kernel \"" + k.getKind().str() + "\" not expanded by sahl-expand-kernels" +
                (why ? " (" + why.getValue().str() + ")" : std::string()));
  }
  return lowerBlock(f.getBody().front()) && applyExpertBases();
}

// An id read as an index (an MoE expert slice's offset): index_cast(memref.load
// M[c..]) of an i32 / i64 DDR buffer at constant indices.
static memref::LoadOp expertIdLoad(Value index) {
  auto c = index.getDefiningOp<arith::IndexCastOp>();
  auto ld = c ? c.getIn().getDefiningOp<memref::LoadOp>() : memref::LoadOp();
  if (!ld || !(ld.getType().isInteger(32) || ld.getType().isInteger(64))) return {};
  for (Value i : ld.getIndices())
    if (!getConstantIntValue(i)) return {};
  return ld;
}

std::optional<std::pair<Value, int64_t>> Lowerer::expertId(Value index) {
  auto ld = expertIdLoad(index);
  if (!ld) return std::nullopt;
  auto mt = cast<MemRefType>(ld.getMemref().getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(mt.getStridesAndOffset(strides, offset))) return std::nullopt;
  int64_t elem = 0;
  for (auto [i, st] : llvm::zip(ld.getIndices(), strides)) elem += *getConstantIntValue(i) * st;
  auto r = ddrOf(ld.getMemref());
  if (!r || r->dyn || r->dynRows || r->dynMid || r->pitch || r->base.getDefiningOp<sahw::BaseOp>() == nullptr ||
      expertBases.count(r->base))
    return std::nullopt;
  int64_t off = r->off + elem * int64_t(esize(r->et));
  if (off % 4) return std::nullopt;
  return std::make_pair(r->base, off);
}

// P = id * stride for every tagged BASE, at the start of the body (after a
// FENCE: the ids are the previous dispatch's, and no leading LD goes to
// sahw.head ahead of it); then each DMA on a tagged BASE reads the real one
// and adds P to its DDR address (a dynamic slot, add mode). A stride within
// LDPARAM's 16-bit factor: one LDPARAM; beyond: id * stride on the VE (exact
// in fp32, checked), through an output-only binding's first 8 bytes (written
// by the dispatch later) into a PARAM.
bool Lowerer::applyExpertBases() {
  if (expertBases.empty()) return true;
  Block &blk = *bb.getBlock();
  if (blk.empty()) return fail("expert slice in an empty body");
  Operation *first = &blk.front();
  OpBuilder pb(first);
  sahw::FenceOp::create(pb, loc, int64_t(0));
  // the scratch: an output-only binding (every use a sahl.store into it or a view of it)
  std::optional<Ddr> scratch;
  auto outputOnly = [&](Value v) {
    SmallVector<Value> work = {v};
    bool stored = false;
    for (unsigned i = 0; i < work.size(); ++i)
      for (Operation *u : work[i].getUsers()) {
        if (isa<memref::SubViewOp, memref::CastOp, memref::CollapseShapeOp, memref::ExpandShapeOp>(u)) {
          work.push_back(u->getResult(0));
          continue;
        }
        auto st = dyn_cast<sahl::StoreOp>(u);
        if (!st || st.getDst() != work[i]) return false;
        stored = true;
      }
    return stored;
  };
  for (auto sub : f.getBody().front().getOps<IREE::HAL::InterfaceBindingSubspanOp>()) {
    if (!outputOnly(sub.getResult())) continue;
    auto r = ddrOf(sub.getResult());
    if (r && !r->dyn && !r->dynRows && r->off % 8 == 0 && r->n * int64_t(esize(r->et)) >= 8) {
      scratch = r;
      break;
    }
  }
  err.clear();                                     // (a binding ddrOf could not read is just not a scratch)
  std::map<std::tuple<const void *, int64_t, int64_t>, Value> pOf;
  for (auto &[tagged, eb] : expertBases) {
    auto key = std::make_tuple(eb.idBase.getAsOpaquePointer(), eb.idOff, eb.stride);
    Value &p = pOf[key];
    if (!p) {
      if (eb.stride <= 0 || eb.maxId * eb.stride >= (int64_t(1) << 31))
        return fail("expert slice: the table's byte offsets beyond 32 bits");
      p = privateParam();
      if (eb.stride <= 0xFFFF) {
        sahw::LdParamOp::create(pb, loc, eb.idBase, p, eb.idOff, eb.stride, 0, ValueRange{});
      } else {
        for (int64_t e = 1; e <= eb.maxId; ++e)
          if (double(float(e)) * double(float(eb.stride)) != double(e * eb.stride) ||
              double(float(double(e) * double(eb.stride))) != double(e * eb.stride))
            return fail("expert slice: id * stride not exact in fp32");
        if (!scratch) return fail("expert slice: no output-only binding for the id's byte offset");
        uint32_t w0 = ::sa::laddr(::sa::MEM_ACC, 0), w1 = ::sa::laddr(::sa::MEM_ACC, 1);
        int64_t off8 = eb.idOff & ~int64_t(7);
        sahw::LdOp::create(pb, loc, eb.idBase, off8, int64_t(w0), 1, 8, 8, 0, ValueRange{});
        {
          OpBuilder::InsertionGuard g(bb);
          StaticLen fixedLen(*this);               // (a full word, never row mode's dynamic LEN)
          Operation *savedVe = lastVe;
          bb.setInsertionPoint(first);
          ve({w0, ::sa::VT_I32}, std::nullopt, w1, ::sa::VT_I32, d, ::sa::VOP_COPY, float(eb.stride), 0.0f);
          lastVe = savedVe;
        }
        sahw::StOp::create(pb, loc, scratch->base, scratch->off, int64_t(w1), 1, 8, 8, ValueRange{});
        sahw::FenceOp::create(pb, loc, int64_t(0));
        sahw::LdParamOp::create(pb, loc, scratch->base, p, scratch->off + (eb.idOff - off8), 1, 0, ValueRange{});
      }
    }
    for (OpOperand &use : llvm::make_early_inc_range(tagged.getUses())) {
      Operation *op = use.getOwner();
      use.set(eb.real);                            // (first: appending the slot's PARAM reallocates the operands)
      auto addSlot = [&](auto o, int32_t field) -> bool {
        SmallVector<int32_t> fields(o.getDynFields());
        SmallVector<bool> adds(o.getDynAdd());
        if (fields.size() >= 2) return fail("expert slice: a DMA with no dynamic slot left");
        for (auto [fl, ad] : llvm::zip(fields, adds))
          if (fl == field && !ad) return fail("expert slice: a DMA whose address a PARAM replaces");
        fields.push_back(field);
        adds.push_back(true);
        o.getDynMutable().append(p);
        o.setDynFields(fields);
        o.setDynAdd(adds);
        return true;
      };
      bool ok;
      if (auto o = dyn_cast<sahw::LdOp>(op)) ok = addSlot(o, ::sa::DYN_DMA_DDR);
      else if (auto o = dyn_cast<sahw::StOp>(op)) ok = addSlot(o, ::sa::DYN_DMA_DDR);
      else if (auto o = dyn_cast<sahw::LdParamOp>(op)) ok = addSlot(o, ::sa::DYN_LDPARAM_ADDR);
      else ok = fail("expert slice read by " + op->getName().getStringRef().str());
      if (!ok) return false;
    }
    tagged.getDefiningOp()->erase();
  }
  return true;
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
        uint32_t savedBase = spadBase;
        // (sa.fresh_params: the VALID counts it loads are its own, their PARAMs free after
        // it: a chunk kernel's rows, each at its position)
        const bool fresh = sc->hasAttr("sa.fresh_params");
        auto savedValid = validParams;
        if (!lowerBlock(sc.getBody().front())) return false;
        if (fresh) validParams = std::move(savedValid);
        mem.release(mark);
        locals = std::move(savedLocals);
        bcastOf = std::move(savedBcast);
        spadBase = savedBase;
        continue;
      }
      locals = reserved;
      bcastOf.clear();
      if (!lowerBlock(sc.getBody().front())) return false;
      mem.release(mark);
      continue;
    }
    if (isa<sahl::CursorResetOp>(op)) {
      if (!rowAcc) rowAcc = privateParam();
      sahw::SetRegOp::create(bb, loc, ValueRange{rowAcc}, ArrayRef<int64_t>{0}, 0, ValueRange{});
      if (auto skip = op->getAttrOfType<IntegerAttr>("sa.skip")) {   // then on by `skip` rows (of the last cursor DMA's bytes)
        if (!cursorBytes) return fail("cursor reset with a skip before any cursor DMA");
        for (int64_t i = 0; i < skip.getInt(); ++i) {
          auto s = sahw::SetRegOp::create(bb, loc, ValueRange{rowAcc}, ArrayRef<int64_t>{0}, 1, ValueRange{cursorBytes});
          s.setDynFields(ArrayRef<int32_t>{::sa::DYN_SETREG_V0});
          s.setDynAdd(ArrayRef<bool>{false});
        }
      }
      continue;
    }
    if (auto cs = dyn_cast<sahl::ClaimSpadOp>(op)) {
      if (cs->hasAttr("sa.relative")) {
        // (a fused dispatch's kernel, in a scope: its placed words after the buffers
        // already in SPAD_A, which it may read; the words it claims after them)
        // (words of SPAD_A bank 0 after the base; the claim's words beyond bank 0 are
        // placed absolutely in bank 1)
        spadBase = mem.spadTop();
        const uint32_t words = uint32_t(cs.getWords());
        if (words > lay.sbank ? spadBase >= lay.sbank || words > 2 * lay.sbank : spadBase + words > 2 * lay.sbank)
          return fail("a relative SPAD_A claim beyond SPAD_A");
        mem.takeSpad(words > lay.sbank ? words : spadBase + words);
        continue;
      }
      if (mem.spadUsed()) return fail("a kernel claiming SPAD_A after other SPAD_A buffers");
      mem.takeSpad(uint32_t(cs.getWords()));
      continue;
    }
    if (auto io = dyn_cast<sahl::IotaOp>(op)) {
      // lane l's index: 15 - (the lanes < k, k = 1..15, counted: prefixes by VALID), then + D g in word g
      auto dst = bufOf(io.getDst());
      if (!dst || dst->vt != ::sa::VT_F32 || dst->bcast || dst->n % d) return fail("iota: a packed fp32 vector");
      StaticLen fixed(*this);
      LocalBuf ones = newLocal(::sa::VT_F32, d), cnt = newLocal(::sa::VT_F32, d), pre = newLocal(::sa::VT_F32, d);
      ve({ones.la, ::sa::VT_I32}, std::nullopt, ones.la, ::sa::VT_F32, d, ::sa::VOP_COPY, 0.0f, 1.0f);
      ve({ones.la}, std::nullopt, cnt.la, ::sa::VT_F32, d, ::sa::VOP_COPY, 0.0f, 0.0f);
      for (int64_t k = 1; k < d; ++k) {
        ve({ones.la}, std::nullopt, pre.la, ::sa::VT_F32, d, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE,
           ::sa::RED_NONE, 0, uint32_t(k));
        ve({cnt.la}, Opd{pre.la}, cnt.la, ::sa::VT_F32, d, ::sa::VOP_ADD);
      }
      for (int64_t g = 0; g < dst->n / d; ++g)
        ve({cnt.la}, std::nullopt, dst->la + uint32_t(g), ::sa::VT_F32, d, ::sa::VOP_COPY, -1.0f,
           float(d - 1 + d * g));
      continue;
    }
    if (auto sp = dyn_cast<sahl::SplatOp>(op)) {
      // (x * 0 + value over its own words, read as int32: finite whatever they hold)
      auto dst = bufOf(sp.getDst());
      if (!dst || dst->dyn) return fail("splat of a static buffer");
      StaticLen fixed(*this);
      const uint32_t memc = (dst->la >> 28) & 0xF;
      const bool spad = memc != uint32_t(::sa::MEM_ACC);
      ve({dst->la, spad ? ::sa::VT_I8 : ::sa::VT_I32}, std::nullopt, dst->la, dst->vt, dst->bcast ? dst->n * d : dst->n,
         ::sa::VOP_COPY, 0.0f, sp.getValue().convertToFloat());
      continue;
    }
    if (auto sf = dyn_cast<sahl::ScalarF32Op>(op)) {
      // (decode's i64 scalars: an LD of its 8 bytes, lane 0 through a REDUCE with VALID = 1)
      auto src = scalarI64(sf.getSrc());
      auto dst = bufOf(sf.getDst());
      if (!src || !dst || dst->vt != ::sa::VT_F32) return fail("scalar_f32 form");
      StaticLen fixed(*this);
      ve({src->la, ::sa::VT_I32}, std::nullopt, dst->la, ::sa::VT_F32, d, ::sa::VOP_COPY, 1.0f, NEG0, ::sa::FUNC_NONE,
         ::sa::RED_MAX, 0, 1);
      locals.erase(sf.getSrc());                     // (the next use loads it again: its word is a temp)
      continue;
    }
    if (auto tr = dyn_cast<sahl::TransposeOp>(op)) {
      auto a = bufOf(tr.getSrc()), b = bufOf(tr.getDst());
      if (!a || !b) return false;
      auto st = cast<MemRefType>(tr.getSrc().getType());
      if (st.getElementType().isF32()) {             // ACC -> ACC (a chunk kernel's per-unit s_q)
        if (st.getRank() != 2 || st.getDimSize(1) % d || !st.hasStaticShape()) return fail("fp32 transpose [R, C]");
        sahw::TransposeOp::create(bb, loc, int64_t(a->la), int64_t(b->la), st.getNumElements(),
                                  int64_t(::sa::vtypes(::sa::VT_F32, ::sa::VT_F32)), st.getDimSize(1) / d, ValueRange{});
        continue;
      }
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
    if (auto gv = dyn_cast<sahl::GemvOp>(op)) {
      if (!gemv(gv)) return false;
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
      // (sa.prefer_bank: in that ACC bank if it has room, as an epilogue's temps)
      auto pb = rs->getAttrOfType<IntegerAttr>("sa.prefer_bank");
      if (pb) mem.preferBank = int(pb.getInt());
      auto l = bufOf(rs.getBuffer(), rs.getBcast());
      mem.preferBank = -1;
      if (!l) return false;
      reserved[rs.getBuffer()] = *l;
      continue;
    }
    if (isa<arith::ConstantOp, memref::AllocOp, memref::DeallocOp, memref::SubViewOp, memref::CastOp,
            memref::DimOp, memref::CollapseShapeOp, memref::ExpandShapeOp, IREE::HAL::InterfaceBindingSubspanOp, IREE::HAL::InterfaceConstantLoadOp,
            IREE::TensorExt::DispatchWorkloadOrdinalOp, func::ReturnOp>(op))
      continue;
    if (op->getDialect() && op->getDialect()->getNamespace() == "util") continue;
    // an MoE expert id read as a DDR view's offset (ddrOf: an expert slice)
    if (auto ld = dyn_cast<memref::LoadOp>(op);
        ld && llvm::all_of(ld->getUsers(), [&](Operation *u) {
          return isa<arith::IndexCastOp>(u) && expertIdLoad(u->getResult(0)) == ld &&
                 llvm::all_of(u->getUsers(), [](Operation *v) { return isa<memref::SubViewOp>(v); });
        }))
      continue;
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtUIOp, arith::ShLIOp, arith::OrIOp, arith::ShRUIOp,
            arith::DivUIOp>(op))
      continue;
    // an f32 push constant (a generalized scale): read by the element-wise op that uses it
    if (auto bc = dyn_cast<arith::BitcastOp>(op); bc && bc.getType().isF32() && linOf(bc.getIn())) continue;
    if (isa<arith::AddIOp, arith::MulIOp>(op) && op->hasAttr("sa.index")) continue;   // (an expanded kernel's count: its Lin used)
    if (auto l = dyn_cast<sahl::LoadOp>(op)) {
      bool rawTopkInput = false;
      for (auto topk : block.getOps<sahl::TopkOp>())
        rawTopkInput |= topk.getScores() == l.getDst();
      if (rawTopkInput) continue;
      if (!load(l)) return false;
      continue;
    }
    if (auto s = dyn_cast<sahl::StoreOp>(op)) {
      bool rawTopkOutput = false;
      for (auto topk : block.getOps<sahl::TopkOp>())
        rawTopkOutput |= topk.getValues() == s.getSrc() || topk.getIndices() == s.getSrc();
      if (rawTopkOutput) continue;
      if (!store(s)) return false;
      continue;
    }
    if (auto fl = dyn_cast<linalg::FillOp>(op)) {
      // The RISC-V-only raw Top-k kernel starts from these exact sentinels in
      // registers. The tensor fills were fused into its dispatch; emitting a
      // separate descriptor fill would be redundant (and i32 fills are not
      // supported by the generic descriptor path).
      bool topkSentinel = false;
      for (auto topk : block.getOps<sahl::TopkOp>()) {
        if (fl.getDpsInits()[0] == topk.getValues()) {
          auto c = fl.getDpsInputs()[0].getDefiningOp<arith::ConstantOp>();
          auto v = c ? dyn_cast<FloatAttr>(c.getValue()) : FloatAttr();
          topkSentinel = v && v.getValue().isInfinity() && v.getValue().isNegative();
        } else if (fl.getDpsInits()[0] == topk.getIndices()) {
          auto c = fl.getDpsInputs()[0].getDefiningOp<arith::ConstantOp>();
          auto v = c ? dyn_cast<IntegerAttr>(c.getValue()) : IntegerAttr();
          topkSentinel = v && v.getValue().getSExtValue() == 0x7fffffff;
        }
        if (topkSentinel) break;
      }
      if (topkSentinel) continue;
      auto c = constF32(fl.getDpsInputs()[0]);
      if (!c) return fail("fill with a non-constant or non-fp32 value");
      fills[fl.getDpsInits()[0]] = *c;
      continue;
    }
    if (auto g = dyn_cast<linalg::GenericOp>(op)) {
      // an expanded kernel's epilogue: its temps next to its accumulator (that ACC bank first)
      auto pb = g->getAttrOfType<IntegerAttr>("sa.prefer_bank");
      // sa.release_temps (a fused dispatch's generics before its kernel): the results
      // allocated first, the temps released after it (unless a result ended up among them)
      const bool scoped = g->hasAttr("sa.release_temps");
      LocalMemory::Mark mark{};
      SmallVector<std::pair<Value, uint32_t>> outs;
      decltype(locals) savedLocals;
      decltype(bcastOf) savedBcast;
      if (scoped) {
        for (Value init : g.getDpsInits()) {
          auto l = bufOf(init, /*bcast=*/g.getNumReductionLoops() > 0);
          if (!l) return false;
          outs.push_back({init, l->la});
        }
        mark = mem.mark();
        savedLocals = locals;
        savedBcast = bcastOf;
      }
      if (pb) mem.preferBank = int(pb.getInt());
      bool ok = generic(g);
      mem.preferBank = -1;
      if (!ok) return false;
      // (a result may be a view of a local buffer: its words recomputed from the buffer)
      const bool isRed = g.getNumReductionLoops() > 0;
      if (scoped && llvm::all_of(outs, [&](auto &o) {
            auto l = bufOf(o.first, isRed);
            return l && l->la == o.second;
          })) {
        mem.release(mark);
        decltype(locals) kept = std::move(savedLocals);
        for (auto &o : outs)
          if (auto it = locals.find(o.first); it != locals.end()) kept[o.first] = it->second;
        locals = std::move(kept);
        bcastOf = std::move(savedBcast);
      }
      continue;
    }
    if (auto sc = dyn_cast<sahl::ScatterOp>(op)) {
      if (!scatter(sc)) return false;
      continue;
    }
    if (auto cp = dyn_cast<memref::CopyOp>(op)) {
      // local -> local (an expanded kernel's step): one VE copy, exact (the
      // integer VE for int32; sa.advance: the destination moving on with the loop, ACC words)
      auto s = bufOf(cp.getSource()), t = bufOf(cp.getTarget());
      if (!s || !t) return false;
      if (s->vt != t->vt || s->n != t->n || s->dyn || t->dyn || s->bcast || t->bcast) return fail("local copy form");
      SmallVector<Value> dv;
      SmallVector<int32_t> df;
      SmallVector<bool> da;
      if (Value adv = advanceOf(cp)) dv.push_back(adv), df.push_back(::sa::DYN_VE_DST), da.push_back(true);
      uint32_t len = uint32_t((s->n + d - 1) / d * d);
      bool fp = s->vt == ::sa::VT_F32;
      auto v = sahw::VeOp::create(bb, loc, int64_t(s->la), 0, int64_t(t->la), int64_t(len), int64_t(::sa::VOP_COPY),
                                  int64_t(::sa::vtypes(s->vt, t->vt)), 0, fp, 0, 0, 0, 0, false, llvm::APFloat(0.0f),
                                  llvm::APFloat(fp ? 1.0f : 0.0f), llvm::APFloat(fp ? NEG0 : 0.0f), 0, 0, 0, 1, 0, 0,
                                  int64_t(INT32_MIN), int64_t(INT32_MAX), dv);
      if (!df.empty()) {
        v.setDynFields(df);
        v.setDynAdd(da);
      }
      continue;
    }
    if (auto topk = dyn_cast<sahl::TopkOp>(op)) {
      SmallVector<int64_t> addresses;
      Value scoreDdr = topk.getScores(), valuesDdr = topk.getValues(), indicesDdr = topk.getIndices();
      unsigned scoreLoads = 0, valueStores = 0, indexStores = 0;
      for (auto l : block.getOps<sahl::LoadOp>()) {
        if (l.getDst() == topk.getScores()) {
          scoreDdr = l.getSrc();
          ++scoreLoads;
        }
      }
      for (auto s : block.getOps<sahl::StoreOp>()) {
        if (s.getSrc() == topk.getValues()) {
          valuesDdr = s.getDst();
          ++valueStores;
        }
        if (s.getSrc() == topk.getIndices()) {
          indicesDdr = s.getDst();
          ++indexStores;
        }
      }
      if (scoreLoads > 1 || valueStores > 1 || indexStores > 1)
        return fail("Top-k requires one input load and at most one store per output");
      for (Value v : {scoreDdr, valuesDdr, indicesDdr}) {
        auto r = ddrOf(v);
        if (!r || r->dyn || r->dynRows || r->dynMid || r->pitch) return fail("Top-k requires contiguous DDR buffers");
        auto base = r->base.getDefiningOp<sahw::BaseOp>();
        if (!base) return fail("Top-k requires a binding base");
        addresses.append({int64_t(base.getBinding()), int64_t(base.getConstant()), int64_t(base.getMul()),
                          int64_t(base.getShift()), int64_t(base.getAdd()), r->off});
      }
      templ->setAttr("sa.moe.raw_topk", bb.getI64IntegerAttr(topk.getExperts()));
      templ->setAttr("sa.moe.top_k", bb.getI64IntegerAttr(topk.getK()));
      templ->setAttr("sa.moe.addresses", bb.getDenseI64ArrayAttr(addresses));
      templ->setAttr("sa.riscv_only", bb.getUnitAttr());
      continue;
    }
    return fail("unsupported operation " + op->getName().getStringRef().str());
  }
  return err.empty();
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
    if (!w && getenv("SA_ATTN_DEBUG"))
      llvm::errs() << "ACC full: " << words << " words, banks at " << mem.mark().acc[0] << " / " << mem.mark().acc[1] << "\n";
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
  if (auto es = v.getDefiningOp<memref::ExpandShapeOp>(); es && !ddrRoot(v)) {
    // a packed local buffer seen in another static shape (a fused dispatch's
    // intermediate, after sa-to-sahl's expandLocalViews): the same words
    auto base = bufOf(es.getSrc());
    if (!base) return std::nullopt;
    if (base->bcast || base->rowStride || base->dyn || !mt.hasStaticShape()) return fail("expanded local buffer"), std::nullopt;
    LocalBuf l = *base;
    l.n = mt.getNumElements();
    return l;
  }
  if (auto cs = v.getDefiningOp<memref::CollapseShapeOp>(); cs && !ddrRoot(v)) {
    // a packed local buffer seen flat (an expanded kernel's step): the same words
    auto base = bufOf(cs.getSrc());
    if (!base) return std::nullopt;
    if (base->bcast || base->rowStride || base->dyn || !mt.hasStaticShape()) return fail("collapsed local buffer"), std::nullopt;
    LocalBuf l = *base;
    l.n = mt.getNumElements();
    return l;
  }
  auto alloc = v.getDefiningOp<memref::AllocOp>();
  if (!alloc) {
    if (getenv("SA_ATTN_DEBUG")) llvm::errs() << "sahl-to-sahw: not a local buffer: " << v << "\n";
    return fail("operand is not a local buffer"), std::nullopt;
  }
  auto vt = vtOf(mt.getElementType());
  if (auto w = alloc->getAttrOfType<IntegerAttr>("sa.word")) {
    auto pm = alloc->getAttrOfType<StringAttr>("sa.mem");
    if (pm && pm.getValue() == "spad_a") {         // a fixed place in SPAD_A (after sahl.claim_spad)
      if (!mt.getElementType().isInteger(8)) return fail("placed SPAD_A buffer"), std::nullopt;
      LocalBuf l;
      l.vt = ::sa::VT_I8;
      l.n = placedElems(alloc, l);
      // (a relative claim: bank 0 words after the base, bank 1 words as they are)
      uint32_t at = uint32_t(w.getInt());
      if (spadBase && at < lay.sbank) {
        at += spadBase;
        if (at + uint32_t((l.n + d - 1) / d) > lay.sbank) return fail("a placed SPAD_A buffer past bank 0"), std::nullopt;
      }
      l.la = ::sa::laddr(::sa::MEM_SPAD_A, at);
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
  if (auto sv = v.getDefiningOp<memref::SubViewOp>();
      sv && mt.getRank() == 2 && !mt.isDynamicDim(0) && mt.isDynamicDim(1)) {
    // rows [r0, r0 + n) of a [H, ?] binding, whole rows (sahl-tile's row pieces, C6.5b)
    auto st = cast<MemRefType>(sv.getSource().getType());
    auto o = sv.getStaticOffsets(), str = sv.getStaticStrides();
    auto dim = sv.getSizes().empty() ? memref::DimOp() : sv.getSizes()[0].getDefiningOp<memref::DimOp>();
    if (st.getRank() != 2 || !st.isDynamicDim(1) || ShapedType::isDynamic(o[0]) || o[1] != 0 || str[0] != 1 ||
        str[1] != 1 || !dim || dim.getSource() != sv.getSource() || dim.getConstantIndex() != 1)
      return fail("dynamic DDR view other than whole rows of a [H, ?] binding"), std::nullopt;
    auto r = ddrOf(sv.getSource(), allowPitch);
    if (!r) return std::nullopt;
    if (!r->dyn || r->rowSkip) return fail("rows of a dynamic DDR view"), std::nullopt;
    r->rowSkip = o[0];
    r->rows = mt.getDimSize(0);
    r->n = cfg.maxDynamic * r->rows;
    return r;
  }
  std::optional<Lin> dynMid;
  if (auto sv = v.getDefiningOp<memref::SubViewOp>();
      sv && mt.getRank() == 3 && !mt.isDynamicDim(0) && mt.isDynamicDim(1) && !mt.isDynamicDim(2)) {
    // [R, ?, C]: R blocks of a dynamic number of contiguous rows of C, a static
    // pitch apart (the packed V cache's 16-channel groups, p . V on the GEMV
    // unit, plan §8.22): the fields at the bound, the length kept for a dynamic field
    dynMid = dynLin(sv.getSizes()[0]);
    if (!dynMid) return fail("dynamic DDR view size"), std::nullopt;
    SmallVector<int64_t> shape(mt.getShape());
    shape[1] = cfg.maxDynamic;
    mt = MemRefType::get(shape, mt.getElementType(), mt.getLayout(), mt.getMemorySpace());
  }
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
  // a subview offset that is a push constant's Lin (in elements), e.g. a layer's
  // offset into a KV cache once iree-flow-generalize-executables made it an operand
  std::optional<Lin> dynElems;
  // a row of a table at an id in DDR (an MoE expert slice): (the id, the stride and rows of its dimension)
  std::optional<std::pair<Value, int64_t>> expert;
  int64_t expertStride = 0, expertRows = 0;
  Value cur = v;
  while (true) {
    Operation *op = cur.getDefiningOp();
    if (!op) return fail("DDR view without a subspan"), std::nullopt;
    if (auto s = dyn_cast<memref::SubViewOp>(op)) {
      auto st = s.getSourceType();
      SmallVector<int64_t> ss;
      int64_t so;
      if (failed(st.getStridesAndOffset(ss, so))) return fail("subview source strides"), std::nullopt;
      unsigned di = 0;
      int64_t dim = -1;
      for (auto [o, str] : llvm::zip(s.getStaticOffsets(), ss)) {
        ++dim;
        if (o == 0) continue;                      // (row 0 of rows a dynamic stride apart: a cursor's view)
        if (ShapedType::isDynamic(str)) return fail("dynamic subview"), std::nullopt;
        if (ShapedType::isDynamic(o)) {
          if (auto id = expertId(s.getOffsets()[di])) {
            if (expert) return fail("DDR view at two expert ids"), std::nullopt;
            expert = id;
            expertStride = str;
            expertRows = st.getDimSize(dim);
            ++di;
            continue;
          }
          auto l = linOf(s.getOffsets()[di++]);
          if (!l || dynElems || l->shift) return fail("dynamic subview"), std::nullopt;
          if (l->isConst()) {
            elem += l->add * str;
          } else {
            dynElems = Lin{l->ord, l->mul * str, 0, l->add * str};
          }
          continue;
        }
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
    if (auto x = dyn_cast<memref::ExpandShapeOp>(op)) {     // (the same elements: a chunk's q as rows of heads)
      cur = x.getSrc();
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
    r.et = mt.getElementType();
    uint32_t es = esize(r.et);
    if (!es) return fail("DDR element type"), std::nullopt;
    if (expert && dynElems) return fail("DDR view at an expert id and a push constant"), std::nullopt;
    if (dynElems) {                                // the base: the binding + the push constant's bytes
      if (!off.isConst()) return fail("dynamic subview of a binding at a dynamic offset"), std::nullopt;
      Lin b{dynElems->ord, dynElems->mul * es, 0, dynElems->add * es + off.add};
      if (b.mul % 8 || b.add % 8) return fail("dynamic subview not 8-byte aligned"), std::nullopt;
      r.base = baseFor(int(sub.getBinding().getZExtValue()), b);
      r.off = elem * es;
    } else {
      r.base = baseFor(int(sub.getBinding().getZExtValue()), off);
      r.off = (off.isConst() ? off.add : 0) + elem * es;
    }
    if (expert) {                                  // its own BASE, tagged (applyExpertBases)
      if (ShapedType::isDynamic(expertRows) || expertRows < 1) return fail("expert table rows"), std::nullopt;
      auto key = std::make_tuple(r.base.getAsOpaquePointer(), expert->second, expertStride * int64_t(es), expertRows,
                                 expert->first.getAsOpaquePointer());
      Value &tagged = expertTagged[key];
      if (!tagged) {
        auto b = r.base.getDefiningOp<sahw::BaseOp>();
        tagged = sahw::BaseOp::create(regB, loc, regB.getType<sahw::BaseType>(), b.getBinding(), b.getConstant(),
                                      b.getMul(), b.getShift(), b.getAdd(), IntegerAttr());
        expertBases[tagged] = {r.base, expert->first, expert->second, expertStride * int64_t(es), expertRows - 1};
      }
      r.base = tagged;
    }
    r.n = mt.getNumElements();
    r.dynRows = dynRows;
    r.dynMid = dynMid;
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
  if (r.rows == 1 && !r.rowSkip) {
    one(0, Value());
    return true;
  }
  rowLoop(r.rows, bp, [&](int64_t row) { one(row, rowAcc); }, r.rowSkip);
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

bool Lowerer::cursorDma(Operation *op, bool isLoad, const Ddr &r, uint32_t la) {
  if (!r.dynRows || r.rows != 1 || r.pitch || r.off % 8) return fail("cursor DMA other than a dynamic-length row");
  uint32_t es = esize(r.et);
  Lin b = *r.dynRows;
  b.mul *= es;
  b.add *= es;
  Value bp = paramFor(b);
  if (!rowAcc) rowAcc = privateParam();
  int64_t maxBytes = cfg.maxDynamic * es;
  cursorBytes = bp;
  if (isLoad) dynDma(sahw::LdOp::create(bb, loc, r.base, r.off, int64_t(la), 1, maxBytes, maxBytes, 0, ValueRange{rowAcc, bp}), rowAcc, bp);
  else dynDma(sahw::StOp::create(bb, loc, r.base, r.off, int64_t(la), 1, maxBytes, maxBytes, ValueRange{rowAcc, bp}), rowAcc, bp);
  if (op->hasAttr("sa.cursor_step")) {
    auto s = sahw::SetRegOp::create(bb, loc, ValueRange{rowAcc}, ArrayRef<int64_t>{0}, 1, ValueRange{bp});
    s.setDynFields(ArrayRef<int32_t>{::sa::DYN_SETREG_V0});
    s.setDynAdd(ArrayRef<bool>{false});
  }
  return true;
}

bool Lowerer::load(sahl::LoadOp l) {
  auto r = ddrOf(l.getSrc(), /*allowPitch=*/true);
  if (!r) return false;
  if (r->et.isInteger(64)) return fail("i64 load");
  auto dst = bufOf(l.getDst());
  if (!dst) return false;
  if (l->hasAttr("sa.cursor")) return cursorDma(l, true, *r, dst->la);
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
    if (r->dynRows && adv) return fail("rows load: dynamic rows in a loop");
    auto ld = sahw::LdOp::create(bb, loc, r->base, r->off, int64_t(dst->la), rows, rb, pitch, 0,
                                 r->dynRows ? ValueRange{paramFor(*r->dynRows)} : adv ? ValueRange{adv} : ValueRange{});
    if (r->dynRows) {
      ld.setDynFields(ArrayRef<int32_t>{::sa::DYN_DMA_ROWS});
      ld.setDynAdd(ArrayRef<bool>{false});
    }
    advanced(ld, adv);                             // (a block of the cache's rows moving on with the loop)
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
  std::optional<LocalBuf> sl;
  if (it != locals.end()) sl = it->second;
  else if (s.getSrc().getDefiningOp<memref::SubViewOp>()) sl = bufOf(s.getSrc());   // (a view of a computed buffer)
  if (!sl) return fail("stored buffer not computed");
  LocalBuf l = *sl;
  if (s->hasAttr("sa.cursor")) return cursorDma(s, false, *r, l.la);
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
    if (st->hasAttr("sa.packed")) {                // K2b GEMV: one row, packed (no copies over the D rows)
      if (x->dyn) return fail("packed strip of a dynamic length");
      ve({x->la, x->vt, ::sa::IDX_LIN, 0}, std::nullopt, dst->la, ::sa::VT_I8, x->n, ::sa::VOP_COPY, 1.0f, NEG0,
         ::sa::FUNC_NONE, ::sa::RED_NONE, 0, 0, {});
      return true;
    }
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
  if (auto f = op->getAttrOfType<IntegerAttr>("sa.advance_final")) {
    // after a sahl.loop: its PARAM as the loop left it (LOOP_END adds the
    // strides on every pass, the last one too: count x stride)
    if (!loopParams.empty() || size_t(f.getInt()) >= lastLoopParams.size())
      return fail("sa.advance_final not right after a sahl.loop"), Value();
    return lastLoopParams[f.getInt()];
  }
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
  // (loops one after another, not nested: their PARAMs are shared, e.g. C6.5's loop per KV head)
  while (loopPool.size() < lp.getStrides().size()) loopPool.push_back(privateParam());
  for (size_t j = 0; j < lp.getStrides().size(); ++j) loopParams.push_back(loopPool[j]);
  SmallVector<int64_t> zeros(loopParams.size(), 0);
  sahw::SetRegOp::create(bb, loc, ValueRange(loopParams), zeros, 0, ValueRange{});
  Block *blk = bb.getInsertionBlock();
  Operation *before = blk->empty() ? nullptr : &blk->back();
  if (!lowerBlock(lp.getBody().front())) return false;
  ArrayRef<int64_t> st = lp.getStrides();
  SmallVector<Value> dv;
  if (Value dc = lp.getDynCount()) {               // the count from a PARAM (patched when first reached)
    auto cl = linOf(dc);
    if (!cl) return fail("sahl.loop: dynamic count not from a push constant");
    dv.push_back(paramFor(*cl));
  }
  auto l = sahw::LoopOp::create(bb, loc, lp.getCount(), loopParams[0], st[0],
                                loopParams.size() > 1 ? loopParams[1] : Value(), st.size() > 1 ? st[1] : 0, dv);
  if (!dv.empty()) {
    l.setDynFields(ArrayRef<int32_t>{::sa::DYN_LOOP_COUNT});
    l.setDynAdd(ArrayRef<bool>{false});
  }
  Block *lb = &l.getRegion().emplaceBlock();
  SmallVector<Operation *> moved;
  for (Operation *o = before ? before->getNextNode() : &blk->front(); o && o != l.getOperation(); o = o->getNextNode())
    moved.push_back(o);
  for (Operation *o : moved) o->moveBefore(lb, lb->end());
  lastLoopParams = loopParams;
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
  SmallVector<bool> da;
  if (Value adv = advanceOf(m)) {                  // the accumulator moving on with the loop (ACC words)
    dv.push_back(adv);
    df.push_back(::sa::DYN_EX_C);
    da.push_back(true);
  }
  if (bt.isDynamicDim(1)) {
    auto kl = sizeLin(1);
    if (!kl) return fail("mma: dynamic K");
    Lin kt = *kl;
    kt.shift += int(logd);
    dv.append({paramFor(kt), paramFor(*kl)});
    df.append({::sa::DYN_EX_KT, ::sa::DYN_EX_BSTEP});
    da.append({false, false});
  }
  if (bt.isDynamicDim(0)) {
    auto nl = sizeLin(0);
    if (!nl) return fail("mma: dynamic tile count");
    dv.push_back(paramFor(*nl));
    df.push_back(::sa::DYN_EX_REPEAT);
    da.push_back(false);
  }
  auto ex = sahw::ExOp::create(bb, loc, int64_t(a->la & 0xFFFFFFF), int64_t(b->la & 0xFFFFFFF),
                               int64_t(c->la & 0xFFFFFFF), k / d, m.getAccumulate(), nc, k, 1, nc, dv);
  if (df.size() > 2) return fail("mma: more than two dynamic fields");
  if (!df.empty()) {
    ex.setDynFields(df);
    ex.setDynAdd(da);
  }
  return true;
}

bool Lowerer::gemv(sahl::GemvOp g) {
  // acc = x . weight strips: one LD in mode GEMV (docs/k2b_gemv_design.md §4)
  if (!cfg.gemvPorts) return fail("gemv: the target has no GEMV unit");
  auto x = bufOf(g.getX()), c = bufOf(g.getAcc());
  if (!x || !c) return false;
  if ((x->la >> 28) != uint32_t(::sa::MEM_SPAD_A) || x->vt != ::sa::VT_I8 || x->dyn) return fail("gemv: x");
  if ((c->la >> 28) != uint32_t(::sa::MEM_ACC)) return fail("gemv: accumulator");
  auto wt = cast<MemRefType>(g.getW().getType());
  // (a dynamic K: the packed V cache's [nc, T, D] in p . V, rows bytes T * D a dynamic field)
  if (wt.getRank() != 3 || wt.isDynamicDim(0) || wt.isDynamicDim(2) || wt.getDimSize(2) != d)
    return fail("gemv: weights [nc, K, D]");
  auto r = ddrOf(g.getW(), /*allowPitch=*/true);
  if (!r) return false;
  if (wt.isDynamicDim(1) != bool(r->dynMid)) return fail("gemv: dynamic K");
  const int64_t nc = wt.getDimSize(0), k = r->dynMid ? cfg.maxDynamic : wt.getDimSize(1), rb = k * d;
  const int64_t pitch = r->dynMid ? r->pitch : leadingPitch(g.getW(), rb);
  if (r->dyn || r->dynRows || r->off % 16 || pitch % 16 || rb > 65535 || k % d || nc < 1 || nc > 8)
    return fail("gemv: weight strips");
  Value adv = advanceOf(g);
  if (r->dynMid && adv) return fail("gemv: dynamic K in a loop");
  Value dk;
  if (r->dynMid) {
    Lin l = *r->dynMid;
    l.mul *= d;
    l.add *= d;
    dk = paramFor(l);
  }
  auto ld = sahw::LdOp::create(bb, loc, r->base, r->off, int64_t(c->la), nc, rb, pitch, ::sa::LD_GEMV,
                               dk ? ValueRange{dk} : adv ? ValueRange{adv} : ValueRange{});
  if (dk) {
    ld.setDynFields(ArrayRef<int32_t>{::sa::DYN_DMA_ROW_BYTES});
    ld.setDynAdd(ArrayRef<bool>{false});
  }
  ld.setXword(int64_t(x->la & 0xFFFF));
  ld.setCstep(1);
  if (g.getAccumulate()) ld.setGemvAcc(true);
  advanced(ld, adv);
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
  StaticLen fixed(*this);
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
  StaticLen fixed(*this);
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

// the constant added to a scatter's index (sa.index_add: replaceDynStores' layer offset)
static int64_t indexAdd(sahl::ScatterOp sc) {
  auto a = sc->getAttrOfType<IntegerAttr>("sa.index_add");
  return a ? a.getInt() : 0;
}

bool Lowerer::scatter(sahl::ScatterOp sc) {
  Value upd = sc.getUpdates(), idx = sc.getIndices(), orig = sc.getDest();
  if (auto rstride = sc->getAttrOfType<IntegerAttr>("sa.row_stride")) {
    // R rows of a contiguous [N, C] table at index + r * stride (the packed V
    // cache's token row): one ST of R rows, a pitch of stride rows
    auto dT = cast<MemRefType>(orig.getType());
    auto uT = cast<MemRefType>(upd.getType());
    uint32_t es = esize(dT.getElementType());
    int64_t rows = uT.getDimSize(0), rb = dT.getDimSize(1) * es, pitch = rstride.getInt() * rb;
    if (dT.getRank() != 2 || (!dT.getLayout().isIdentity() && !isa<StridedLayoutAttr>(dT.getLayout())) || rb % 8 ||
        pitch % 8 || rb > 0xFFFF || rows > 0xFFFF)
      return fail("scatter of strided rows: layout");
    auto u = materialize(upd);
    auto ir = ddrOf(idx);
    auto orr = ddrOf(orig);
    if (!u || !ir || !orr) return false;
    if (orr->off % 8) return fail("scatter target not 8-byte aligned");
    Value p = privateParam();
    sahw::LdParamOp::create(bb, loc, ir->base, p, ir->off, rb, indexAdd(sc) * rb, ValueRange{});
    auto s = sahw::StOp::create(bb, loc, orr->base, orr->off, int64_t(u->la), rows, rb, pitch, ValueRange{p});
    s.setDynFields(ArrayRef<int32_t>{::sa::DYN_DMA_DDR});
    s.setDynAdd(ArrayRef<bool>{true});
    return err.empty();
  }
  if (auto dim = sc->getAttrOfType<IntegerAttr>("sa.dim")) {
    // along dimension 1 of [R, N, C] (the packed V cache: R blocks of N rows of
    // C bytes): R rows of C bytes, a pitch of N * C apart, at index * C
    auto dT = cast<MemRefType>(orig.getType());
    SmallVector<int64_t> st;
    int64_t off0;
    if (dim.getInt() != 1 || dT.getRank() != 3 || !dT.hasStaticShape() || failed(dT.getStridesAndOffset(st, off0)))
      return fail("scatter along a dimension: [R, N, C]");
    uint32_t es = esize(dT.getElementType());
    int64_t rows = dT.getDimSize(0), rb = dT.getDimSize(2) * es, pitch = st[0] * es;
    if (st[2] != 1 || st[1] != dT.getDimSize(2) || rb % 8 || pitch % 8 || rb > 0xFFFF ||
        cast<MemRefType>(upd.getType()).getNumElements() != rows * dT.getDimSize(2))
      return fail("scatter along a dimension: layout");
    auto u = materialize(upd);
    auto ir = ddrOf(idx);
    auto orr = ddrOf(orig);
    if (!u || !ir || !orr) return false;
    if (orr->off % 8) return fail("scatter target not 8-byte aligned");
    Value p = privateParam();
    sahw::LdParamOp::create(bb, loc, ir->base, p, ir->off, rb, indexAdd(sc) * rb, ValueRange{});
    auto s = sahw::StOp::create(bb, loc, orr->base, orr->off, int64_t(u->la), rows, rb, pitch, ValueRange{p});
    s.setDynFields(ArrayRef<int32_t>{::sa::DYN_DMA_DDR});
    s.setDynAdd(ArrayRef<bool>{true});
    return err.empty();
  }
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
  sahw::LdParamOp::create(bb, loc, ir->base, p, ir->off, rowBytes, indexAdd(sc) * int64_t(rowBytes), ValueRange{});
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
      cfg.gemvPorts = optGemvPorts;
      cfg.ukernels = optUkernels;
      cfg.maxDynamic = optMaxDynamic;
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
      if (t->hasAttr("sa.riscv_only") &&
          (llvm::range_size(f.getBody().front().getOps<sahl::TopkOp>()) != 1 ||
           !body.getRegion().front().empty())) {
        f.emitError("sahl-to-sahw: an MoE RISC-V-only dispatch must be one Top-k and nothing else");
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
  Option<int64_t> optGemvPorts{*this, "gemv-ports", llvm::cl::desc("read ports of the GEMV unit (0: none)"),
                               llvm::cl::init(0)};
  Option<int64_t> optMaxDynamic{*this, "max-dynamic", llvm::cl::desc("upper bound of a dynamic dimension"),
                                llvm::cl::init(256)};
};

}  // namespace

std::unique_ptr<Pass> createSahlToSahwPass(const TargetConfig &config) {
  return std::make_unique<SahlToSahwPass>(config);
}
void registerSahlToSahwPass() { PassRegistration<SahlToSahwPass>(); }

}  // namespace mlir::iree_compiler::sa

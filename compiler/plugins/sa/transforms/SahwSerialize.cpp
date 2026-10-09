// sahw.template -> the descriptors and export information of sa-desc v3
// (compiler/runtime/tools/sadesc.py; docs/iree_compiler_plan.md §4.4). The
// encoding is DescList's (target/DescList.cpp); the tag (w0[63:32]) is the
// descriptor's index as DescList numbers it: the prefix from 0, the head and
// body from 0 (the head's RET takes the next index without advancing it).
#include <cstring>

#include "../target/DescList.h"
#include "SahwPasses.h"

namespace mlir::iree_compiler::sa {
namespace {

using Row = std::array<uint64_t, 8>;
using ::sa::DescList;
constexpr uint64_t M32 = 0xFFFFFFFFull;
constexpr uint64_t RELOC = 1u << 8, FENCE_BEFORE = 1u << 11;

uint64_t f32bits(const APFloat &f) { return f.bitcastToAPInt().getZExtValue() & M32; }

struct Serializer {
  std::string err;
  uint32_t reads = 0, writes = 0, prefixReads = 0;
  int prefixReg = -1;
  bool spadB = false;
  bool inPrefix = false;

  bool fail(Operation *op, const std::string &m) {
    if (err.empty()) {
      err = m;
      op->emitError() << "sahw serialization: " << m;
    }
    return false;
  }

  static int regOf(Value v) {
    Operation *d = v.getDefiningOp();
    if (auto o = dyn_cast_or_null<sahw::BaseOp>(d)) return o.getReg() ? int(*o.getReg()) : -1;
    if (auto o = dyn_cast_or_null<sahw::ParamOp>(d)) return o.getReg() ? int(*o.getReg()) : -1;
    if (auto o = dyn_cast_or_null<sahw::PrivateOp>(d)) return o.getReg() ? int(*o.getReg()) : -1;
    return -1;
  }
  static int bindingOf(Value base) { return int(cast<sahw::BaseOp>(base.getDefiningOp()).getBinding()); }

  // the header flags of a command with an optional base and dynamic fields
  bool header(Operation *op, Value base, ValueRange dyn, ArrayRef<int32_t> fields, ArrayRef<bool> adds,
              bool fenceBefore, uint64_t &flags) {
    flags = fenceBefore ? FENCE_BEFORE : 0;
    if (base) {
      int r = inPrefix ? 15 : regOf(base);
      if (r < 0) return fail(op, "BASE register not assigned");
      if (inPrefix) {
        int orig = regOf(base);
        if (prefixReg >= 0 && prefixReg != orig) return fail(op, "prefix loads through two BASE registers");
        prefixReg = orig;
      }
      flags |= RELOC | uint64_t(r & 3) << 9 | uint64_t(r >> 2 & 3) << 13;
    }
    if (dyn.size() > 2 || fields.size() != dyn.size() || adds.size() != dyn.size())
      return fail(op, "more than two dynamic fields, or mismatched dynamic field lists");
    for (size_t i = 0; i < dyn.size(); ++i) {
      int p = regOf(dyn[i]);
      if (p < 0) return fail(op, "PARAM register not assigned");
      uint64_t f = uint64_t(fields[i]) | uint64_t(p & 7) << 4 | (adds[i] ? 0x80 : 0);
      flags |= f << (16 + 8 * i);
    }
    return true;
  }

  bool emit(Block &blk, std::vector<Row> &rows, uint32_t &tag) {
    for (Operation &opRef : blk) {
      Operation *op = &opRef;
      Row w{};
      uint64_t flags = 0;
      auto put = [&](uint32_t opc) {
        w[0] = uint64_t(tag++) << 32 | flags | opc;
        rows.push_back(w);
      };
      if (auto o = dyn_cast<sahw::LoopOp>(op)) {
        uint32_t start = tag;
        if (!emit(o.getRegion().front(), rows, tag)) return false;
        int k1 = regOf(o.getK1()), k2 = regOf(o.getK2());
        if (k1 < 0 || k2 < 0) return fail(op, "loop PARAM not assigned");
        w[1] = uint64_t(uint32_t(int32_t(start) - int32_t(tag)));
        w[2] = uint64_t(o.getCount()) | uint64_t(k1 & 7) << 16 | uint64_t(k2 & 7) << 19;
        w[3] = uint64_t(o.getS1()) & M32;
        w[4] = uint64_t(o.getS2()) & M32;
        put(DescList::LOOP_END);
        continue;
      }
      if (!inPrefix && !isa<sahw::FenceOp, sahw::SetRegOp>(op)) {
        spadB |= [&] {
          if (isa<sahw::ExOp>(op)) return true;
          auto sb = [](int64_t a) { return (uint64_t(a) & M32) >> 28 == 2; };
          if (auto l = dyn_cast<sahw::LdOp>(op)) return sb(l.getLaddr());
          if (auto l = dyn_cast<sahw::StOp>(op)) return sb(l.getLaddr());
          if (auto v = dyn_cast<sahw::VeOp>(op)) return sb(v.getSrc1()) || sb(v.getSrc2()) || sb(v.getDst());
          if (auto v = dyn_cast<sahw::TransposeOp>(op)) return sb(v.getSrc()) || sb(v.getDst());
          return false;
        }();
      }
      if (auto o = dyn_cast<sahw::LdOp>(op)) {
        if (!header(op, o.getBase(), o.getDyn(), o.getDynFields(), o.getDynAdd(), o.getFenceBefore(), flags))
          return false;
        if (o.getBase()) (inPrefix ? prefixReads : reads) |= 1u << bindingOf(o.getBase());
        w[1] = uint64_t(o.getDdr()) & M32;
        w[2] = (uint64_t(o.getLaddr()) & M32) | uint64_t(o.getRows()) << 32 | uint64_t(o.getRowBytes()) << 48;
        w[3] = (uint64_t(o.getPitch()) & M32) | uint64_t(o.getMode()) << 32 |
               uint64_t(o.getXword() & 0xFFFF) << 34 | uint64_t(o.getCstep() & 0xFF) << 50 |
               uint64_t(o.getGemvAcc()) << 58;
        put(DescList::LD);
      } else if (auto o = dyn_cast<sahw::StOp>(op)) {
        if (inPrefix) return fail(op, "a store in the prefix");
        if (!header(op, o.getBase(), o.getDyn(), o.getDynFields(), o.getDynAdd(), o.getFenceBefore(), flags))
          return false;
        if (o.getBase()) writes |= 1u << bindingOf(o.getBase());
        w[1] = uint64_t(o.getDdr()) & M32;
        w[2] = (uint64_t(o.getLaddr()) & M32) | uint64_t(o.getRows()) << 32 | uint64_t(o.getRowBytes()) << 48;
        w[3] = uint64_t(o.getPitch());
        put(DescList::ST);
      } else if (auto o = dyn_cast<sahw::ExOp>(op)) {
        if (!header(op, Value(), o.getDyn(), o.getDynFields(), o.getDynAdd(), o.getFenceBefore(), flags)) return false;
        w[1] = uint64_t(o.getA()) | uint64_t(o.getB()) << 16 | uint64_t(o.getC()) << 32 | uint64_t(o.getKt()) << 48 |
               uint64_t(o.getAccumulate()) << 60;
        w[2] = uint64_t(o.getRepeat()) | uint64_t(o.getBstep()) << 16 | uint64_t(o.getCstep()) << 32 |
               uint64_t(o.getCrow()) << 48;
        put(DescList::EX);
      } else if (auto o = dyn_cast<sahw::VeOp>(op)) {
        if (!header(op, Value(), o.getDyn(), o.getDynFields(), o.getDynAdd(), o.getFenceBefore(), flags)) return false;
        uint64_t f = uint64_t(o.getFp()) | uint64_t(o.getFunc()) << 1 | uint64_t(o.getM1()) << 4 |
                     uint64_t(o.getM2()) << 6 | uint64_t(o.getReduce()) << 8 | uint64_t(o.getSwapneg()) << 10;
        w[1] = (uint64_t(o.getSrc1()) & M32) | uint64_t(o.getSrc2()) << 32;
        w[2] = (uint64_t(o.getDst()) & M32) | uint64_t(o.getLength()) << 32;
        w[3] = uint64_t(o.getOp()) | uint64_t(o.getTypes()) << 8 | uint64_t(o.getPeriod()) << 16 |
               (uint64_t(o.getScale()) & 0xFFFF) << 32 | uint64_t(o.getShift()) << 48 | f << 53;
        w[4] = (uint64_t(o.getZp()) & M32) | (uint64_t(o.getLo()) & M32) << 32;
        w[5] = (uint64_t(o.getHi()) & M32) | f32bits(o.getImm()) << 32;
        w[6] = f32bits(o.getA()) | f32bits(o.getB()) << 32;
        w[7] = uint64_t(o.getRowlen()) | uint64_t(o.getValid()) << 16 | uint64_t(o.getP1()) << 32;
        put(DescList::VE);
      } else if (auto o = dyn_cast<sahw::TransposeOp>(op)) {
        if (!header(op, Value(), o.getDyn(), o.getDynFields(), o.getDynAdd(), o.getFenceBefore(), flags)) return false;
        w[1] = uint64_t(o.getSrc());
        w[2] = (uint64_t(o.getDst()) & M32) | uint64_t(o.getLength()) << 32;
        w[3] = 6 | uint64_t(o.getTypes()) << 8;
        w[7] = uint64_t(o.getStride()) << 48;
        put(DescList::VE);
      } else if (auto o = dyn_cast<sahw::LdParamOp>(op)) {
        if (!header(op, o.getBase(), o.getDyn(), o.getDynFields(), o.getDynAdd(), o.getFenceBefore(), flags))
          return false;
        if (o.getBase()) reads |= 1u << bindingOf(o.getBase());
        int p = regOf(o.getTarget());
        if (p < 0) return fail(op, "LDPARAM target not assigned");
        w[1] = uint64_t(o.getAddr()) & M32;
        w[2] = uint64_t(p & 7);
        w[3] = uint64_t(o.getMul()) & 0xFFFF;
        w[4] = uint64_t(o.getAdd()) & M32;
        put(DescList::LDPARAM);
      } else if (auto o = dyn_cast<sahw::FenceOp>(op)) {
        w[1] = uint64_t(o.getMask());
        put(DescList::FENCE);
      } else if (auto o = dyn_cast<sahw::SetRegOp>(op)) {
        if (!header(op, Value(), o.getDyn(), o.getDynFields(), o.getDynAdd(), o.getFenceBefore(), flags)) return false;
        auto vals = o.getValues();
        if (o.getTargets().empty() || o.getTargets().size() > 3 || vals.size() != o.getTargets().size())
          return fail(op, "SETREG takes 1..3 registers");
        for (size_t i = 0; i < vals.size(); ++i) {
          int p = regOf(o.getTargets()[i]);
          if (p < 0) return fail(op, "SETREG target not assigned");
          w[1] |= uint64_t(0x40 | (DescList::REG_PARAM + p)) << (8 * i);
          w[2 + i] = uint64_t(vals[i]) & M32;
        }
        w[5] = uint64_t(o.getAddMask()) & 7;
        put(DescList::SETREG);
      } else {
        return fail(op, "unexpected operation " + op->getName().getStringRef().str());
      }
    }
    return true;
  }
};

void append(std::string &out, const std::vector<Row> &rows) {
  for (const Row &r : rows)
    for (uint64_t w : r)
      for (int b = 0; b < 8; ++b) out.push_back(char(w >> (8 * b) & 0xFF));
}

Row ret(uint32_t tag) {
  Row w{};
  w[0] = uint64_t(tag) << 32 | DescList::RET;
  return w;
}

}  // namespace

LogicalResult serializeTemplate(sahw::TemplateOp t, SerializedExport &out) {
  out = SerializedExport();
  Serializer s;
  sahw::PrefixOp prefix;
  sahw::HeadOp head;
  sahw::BodyOp body;
  for (Operation &op : t.getRegion().front()) {
    if (auto o = dyn_cast<sahw::BaseOp>(op)) {
      if (!o.getReg()) return o.emitError("BASE register not assigned");
      out.setup.push_back({SetupEntry::BASE, uint8_t(*o.getReg()), uint8_t(o.getBinding()), int16_t(o.getConstant()),
                           int32_t(o.getMul()), uint8_t(o.getShift()), int32_t(o.getAdd())});
    } else if (auto o = dyn_cast<sahw::ParamOp>(op)) {
      if (!o.getReg()) return o.emitError("PARAM register not assigned");
      out.setup.push_back({SetupEntry::PARAM, uint8_t(*o.getReg()), 0, int16_t(o.getConstant()), int32_t(o.getMul()),
                           uint8_t(o.getShift()), int32_t(o.getAdd())});
    } else if (auto o = dyn_cast<sahw::PrefixOp>(op)) {
      prefix = o;
    } else if (auto o = dyn_cast<sahw::HeadOp>(op)) {
      head = o;
    } else if (auto o = dyn_cast<sahw::BodyOp>(op)) {
      body = o;
    }
  }
  if (!body) return t.emitError("sahw.template without a body");
  std::vector<Row> rows;
  if (prefix) {
    uint32_t tag = 0;
    s.inPrefix = true;
    if (!s.emit(prefix.getRegion().front(), rows, tag)) return failure();
    rows.push_back(ret(tag));
    out.prefix = uint32_t(rows.size());
    s.inPrefix = false;
  }
  uint32_t tag = 0;
  if (head) {
    if (!s.emit(head.getRegion().front(), rows, tag)) return failure();
    rows.push_back(ret(tag));
    out.head = uint32_t(rows.size()) - out.prefix;
  }
  if (!s.emit(body.getRegion().front(), rows, tag)) return failure();
  rows.push_back(ret(tag));
  append(out.templ, rows);
  out.writes = s.writes;
  out.prefixReads = s.prefixReads;
  out.reads = s.reads | s.prefixReads;
  out.prefixReg = s.prefixReg < 0 ? 0 : uint32_t(s.prefixReg);
  out.flags = s.spadB ? 1 : 0;
  out.cycles = uint32_t(t.getEstCycles());
  if (t.getUnsupported()) {
    out.cycles = 0xFFFFFFFFu;
    out.flags = 1;
    out.setup.clear();
  }
  return success();
}

}  // namespace mlir::iree_compiler::sa

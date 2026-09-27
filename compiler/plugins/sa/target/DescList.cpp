// Descriptor-list encoder (DescList.h): driver/pynq_matmul.py DescList in C++.
#include "DescList.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace sa {

static constexpr uint64_t M32 = 0xFFFFFFFFull;
static constexpr uint64_t RELOC = 1u << 8, FENCE_BEFORE = 1u << 11;

// Programming errors in a template (the LLVM build has no exceptions).
[[noreturn]] static void fail(const char *msg) {
  std::fprintf(stderr, "sa DescList: %s\n", msg);
  std::abort();
}

static uint64_t f32bits(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return u;
}

DescList &DescList::put(uint32_t op, uint64_t flags, std::initializer_list<uint64_t> words,
                        const std::vector<Dyn> &dyn) {
  if (dyn.size() > 2) fail("at most two dynamic fields per descriptor");
  for (size_t i = 0; i < dyn.size(); ++i) {
    uint64_t f = dyn[i].field | (dyn[i].param & 7) << 4 | (dyn[i].add ? 0x80 : 0);
    flags |= f << (16 + 8 * i);
  }
  Row w{};
  w[0] = (uint64_t(rows.size()) << 32) | flags | op;
  size_t i = 1;
  for (uint64_t x : words) w[i++] = x;
  rows.push_back(w);
  return *this;
}

uint64_t DescList::dmaFlags(int base, bool fenceBefore) {
  uint64_t f = fenceBefore ? FENCE_BEFORE : 0;
  if (base >= 0) f |= RELOC | uint64_t(base & 3) << 9 | uint64_t(base >> 2 & 3) << 13;
  return f;
}

DescList &DescList::ld(uint32_t ddr, uint32_t la, uint32_t nrows, uint32_t rowBytes, uint32_t pitch, int base,
                       uint32_t mode, bool fenceBefore, const std::vector<Dyn> &dyn) {
  return put(LD, dmaFlags(base, fenceBefore),
             {ddr & M32, la | uint64_t(nrows) << 32 | uint64_t(rowBytes) << 48, pitch | uint64_t(mode) << 32},
             dyn);
}

DescList &DescList::st(uint32_t ddr, uint32_t la, uint32_t nrows, uint32_t rowBytes, uint32_t pitch, int base,
                       bool fenceBefore, const std::vector<Dyn> &dyn) {
  return put(ST, dmaFlags(base, fenceBefore),
             {ddr & M32, la | uint64_t(nrows) << 32 | uint64_t(rowBytes) << 48, uint64_t(pitch)}, dyn);
}

DescList &DescList::ex(uint32_t a, uint32_t b, uint32_t c, uint32_t kt, bool accumulate, uint32_t repeat,
                       uint32_t bstep, uint32_t cstep, uint32_t crow, bool fenceBefore, const std::vector<Dyn> &dyn) {
  return put(EX, dmaFlags(-1, fenceBefore),
             {a | uint64_t(b) << 16 | uint64_t(c) << 32 | uint64_t(kt) << 48 | uint64_t(accumulate) << 60,
              repeat | uint64_t(bstep) << 16 | uint64_t(cstep) << 32 | uint64_t(crow) << 48},
             dyn);
}

DescList &DescList::transpose(uint32_t src, uint32_t dst, uint32_t length, uint32_t types, uint32_t stride,
                              bool fenceBefore, const std::vector<Dyn> &dyn) {
  constexpr uint64_t VOP_TRANSPOSE = 6;
  return put(VE, dmaFlags(-1, fenceBefore),
             {uint64_t(src), dst | uint64_t(length) << 32, VOP_TRANSPOSE | uint64_t(types) << 8, 0, 0, 0,
              uint64_t(stride) << 48},
             dyn);
}

DescList &DescList::ldparam(uint32_t addr, uint32_t param, uint32_t mul, uint32_t add, int base, bool fenceBefore,
                            const std::vector<Dyn> &dyn) {
  return put(LDPARAM, dmaFlags(base, fenceBefore), {addr & M32, uint64_t(param & 7), uint64_t(mul & 0xFFFF),
                                                   add & M32},
             dyn);
}

DescList &DescList::fence(uint32_t mask) { return put(FENCE, 0, {uint64_t(mask)}); }

DescList &DescList::veFp(uint32_t src1, uint32_t src2, uint32_t dst, uint32_t length, VOp op, uint32_t types,
                         uint32_t period, const VeFp &fp, bool fenceBefore, const std::vector<Dyn> &dyn) {
  if (fp.t2 >= 0) types |= uint32_t(fp.t2 == 0 ? 1 : fp.t2) << 4;
  const int64_t scale = 1, shift = 0, zp = 0, lo = INT32_MIN, hi = INT32_MAX;
  uint64_t w3 = op | uint64_t(types) << 8 | uint64_t(period) << 16 | uint64_t(scale & 0xFFFF) << 32 |
                uint64_t(shift) << 48;
  uint64_t flags = 1 | uint64_t(fp.func) << 1 | uint64_t(fp.m1) << 4 | uint64_t(fp.m2) << 6 |
                   uint64_t(fp.reduce) << 8 | uint64_t(fp.swapneg) << 10;
  w3 |= flags << 53;
  uint64_t w4 = (uint64_t(zp) & M32) | (uint64_t(lo) & M32) << 32;
  uint64_t w5 = (uint64_t(hi) & M32) | f32bits(fp.imm) << 32;
  uint64_t w6 = f32bits(fp.A) | f32bits(fp.B) << 32;
  uint64_t w7 = fp.rowlen | uint64_t(fp.valid) << 16 | uint64_t(fp.p1) << 32;
  return put(VE, dmaFlags(-1, fenceBefore), {src1 | uint64_t(src2) << 32, dst | uint64_t(length) << 32, w3, w4,
                                             w5, w6, w7},
             dyn);
}

DescList &DescList::setreg(const std::vector<std::pair<uint32_t, uint32_t>> &regs, const std::vector<Dyn> &dyn) {
  if (regs.empty() || regs.size() > 3) fail("SETREG takes 1..3 registers");
  uint64_t w1 = 0, vals[3] = {0, 0, 0};
  for (size_t i = 0; i < regs.size(); ++i) {
    w1 |= uint64_t(0x40 | regs[i].first) << (8 * i);
    vals[i] = regs[i].second & M32;
  }
  return put(SETREG, 0, {w1, vals[0], vals[1], vals[2], 0}, dyn);
}

DescList &DescList::loopEnd(int32_t offset, uint32_t count, uint32_t k1, uint32_t s1, uint32_t k2, uint32_t s2) {
  return put(LOOP_END, 0,
             {uint64_t(uint32_t(offset)), count | uint64_t(k1 & 7) << 16 | uint64_t(k2 & 7) << 19, s1 & M32,
              s2 & M32});
}

DescList &DescList::ret() { return put(RET, 0, {}); }

std::string DescList::bytes() const {
  std::string out;
  out.reserve(rows.size() * 64);
  for (const Row &r : rows)
    for (uint64_t w : r)
      for (int b = 0; b < 8; ++b) out.push_back(char(w >> (8 * b) & 0xFF));
  return out;
}

}  // namespace sa

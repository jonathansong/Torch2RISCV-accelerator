// Descriptor-list encoder of the sa backend: the C++ twin of
// driver/pynq_matmul.py DescList (docs/double_buffer_design.md §8.6, L1
// command extensions in docs/llm_inference_plan.md §5.3). Byte for byte the
// same encoding (compiler/tests/test_c2.py compares the templates).
#ifndef SA_TARGET_DESCLIST_H_
#define SA_TARGET_DESCLIST_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace sa {

// Local memories and addresses (pynq_matmul.py: MEM_*, laddr)
enum Mem : uint32_t { MEM_SPAD_A = 1, MEM_SPAD_B = 2, MEM_ACC = 3 };
inline uint32_t laddr(Mem mem, uint32_t word) { return (uint32_t(mem) << 28) | word; }
inline uint32_t acc(uint32_t word) { return laddr(MEM_ACC, word); }

// Vector engine: ops, element types (VT_*), fp32 pipeline fields
enum VOp : uint32_t { VOP_ADD = 0, VOP_SUB = 1, VOP_MUL = 2, VOP_MAX = 3, VOP_MIN = 4, VOP_COPY = 5 };
enum VType : uint32_t { VT_I8 = 0, VT_I16 = 1, VT_I32 = 2, VT_F32 = 3 };
enum VFunc : uint32_t { FUNC_NONE = 0, FUNC_EXP = 1, FUNC_RECIP = 2, FUNC_RSQRT = 3, FUNC_ABS = 4 };
enum VIdx : uint32_t { IDX_LIN = 0, IDX_MOD = 1, IDX_DIV = 2, IDX_IMM = 3 };
enum VRed : uint32_t { RED_NONE = 0, RED_SUM = 1, RED_MAX = 2 };

// types byte of a VE descriptor: input | output << 2 (| src2 type << 4, fp only)
inline uint32_t vtypes(VType in, VType out) { return uint32_t(in) | uint32_t(out) << 2; }

// Fields a descriptor may take from a PARAM register (DescList.DYN_FIELDS).
enum DynField : uint32_t {
  DYN_DMA_DDR = 1, DYN_DMA_LADDR = 2, DYN_DMA_ROWS = 3, DYN_DMA_ROW_BYTES = 4, DYN_DMA_PITCH = 5,
  DYN_EX_A = 1, DYN_EX_B = 2, DYN_EX_C = 3, DYN_EX_KT = 4, DYN_EX_REPEAT = 5, DYN_EX_BSTEP = 6, DYN_EX_CSTEP = 7,
  DYN_VE_SRC1 = 1, DYN_VE_SRC2 = 2, DYN_VE_DST = 3, DYN_VE_LEN = 4, DYN_VE_VALID = 5, DYN_VE_ROWLEN = 6,
  DYN_VE_P1 = 7, DYN_VE_PERIOD = 8, DYN_VE_A = 9, DYN_VE_B = 10, DYN_VE_IMM = 11,
  DYN_LDPARAM_ADDR = 1,
  DYN_SETREG_V0 = 1, DYN_SETREG_V1 = 2, DYN_SETREG_V2 = 3,
};
struct Dyn {
  uint32_t field;
  uint32_t param;
  bool add;
};

// The fp32 VE pipeline options of DescList.ve(fp=True, ...).
struct VeFp {
  VFunc func = FUNC_NONE;
  VIdx m1 = IDX_LIN, m2 = IDX_LIN;
  VRed reduce = RED_NONE;
  bool swapneg = false;
  float imm = 0.0f, A = 1.0f, B = 0.0f;
  uint32_t rowlen = 0, valid = 0, p1 = 0;
  int t2 = -1;  // type of src2 when it differs from src1's: 0 int8, 2 int32, 3 fp32
};

class DescList {
public:
  using Row = std::array<uint64_t, 8>;

  static constexpr uint32_t LD = 0x01, ST = 0x02, EX = 0x03, VE = 0x04, FENCE = 0x10, JUMP = 0x11,
                            END = 0x12, LOOP_END = 0x13, SETREG = 0x14, CALL = 0x15, RET = 0x16,
                            LDPARAM = 0x17;
  static constexpr uint32_t REG_BASE = 0, REG_PARAM = 16;

  // base < 0: absolute DDR address; else relative to BASE base (0..15).
  DescList &ld(uint32_t ddr, uint32_t la, uint32_t rows, uint32_t rowBytes, uint32_t pitch, int base = -1,
               uint32_t mode = 0, bool fenceBefore = false, const std::vector<Dyn> &dyn = {});
  DescList &st(uint32_t ddr, uint32_t la, uint32_t rows, uint32_t rowBytes, uint32_t pitch, int base = -1,
               bool fenceBefore = false, const std::vector<Dyn> &dyn = {});
  DescList &ex(uint32_t a, uint32_t b, uint32_t c, uint32_t kt, bool accumulate = false, uint32_t repeat = 1,
               uint32_t bstep = 0, uint32_t cstep = 0, uint32_t crow = 1, bool fenceBefore = false,
               const std::vector<Dyn> &dyn = {});
  // The fp32 VE pipeline (DescList.ve with fp=True; scale / shift / zp / lo /
  // hi at their defaults).
  DescList &veFp(uint32_t src1, uint32_t src2, uint32_t dst, uint32_t length, VOp op, uint32_t types,
                 uint32_t period, const VeFp &fp, bool fenceBefore = false, const std::vector<Dyn> &dyn = {});
  // D x D block transpose (DescList.transpose): length elements, stride words.
  DescList &transpose(uint32_t src, uint32_t dst, uint32_t length, uint32_t types, uint32_t stride,
                      bool fenceBefore = false, const std::vector<Dyn> &dyn = {});
  // PARAM[param] = mem32[addr] * mul + add (DescList.ldparam).
  DescList &ldparam(uint32_t addr, uint32_t param, uint32_t mul = 1, uint32_t add = 0, int base = -1,
                    bool fenceBefore = false, const std::vector<Dyn> &dyn = {});
  DescList &fence(uint32_t mask = 0);
  // regs: up to three (register, value); addMask bit i: register i += value
  // (instead of =). A dynamic field v0..v2 (DYN_SETREG_V0 + i) takes the value from a PARAM.
  DescList &setreg(const std::vector<std::pair<uint32_t, uint32_t>> &regs, const std::vector<Dyn> &dyn = {},
                   uint32_t addMask = 0);
  DescList &loopEnd(int32_t offset, uint32_t count, uint32_t k1, uint32_t s1, uint32_t k2, uint32_t s2);
  DescList &ret();
  // Appends an encoded descriptor as is.
  DescList &raw(const Row &w) {
    rows.push_back(w);
    return *this;
  }

  size_t size() const { return rows.size(); }
  const std::vector<Row> &array() const { return rows; }
  // Little-endian bytes of the list.
  std::string bytes() const;

private:
  DescList &put(uint32_t op, uint64_t flags, std::initializer_list<uint64_t> words,
                const std::vector<Dyn> &dyn = {});
  static uint64_t dmaFlags(int base, bool fenceBefore);

  std::vector<Row> rows;
};

}  // namespace sa

#endif  // SA_TARGET_DESCLIST_H_

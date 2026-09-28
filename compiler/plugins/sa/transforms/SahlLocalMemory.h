// Local memory of one template (docs/iree_compiler_plan.md §8.14, C8 R4):
// SPAD_A and the two ACC banks (above the scratch area), allocated in order
// (bump). A mark / release pair frees everything allocated since the mark: a
// piece (sahl.scope), a chunk of a contraction, a row of a dynamic length.
// Addresses are words; the lowering turns them into local addresses.
#ifndef SA_TRANSFORMS_SAHLLOCALMEMORY_H_
#define SA_TRANSFORMS_SAHLLOCALMEMORY_H_

#include <cstdint>
#include <optional>

#include "../target/Layout.h"

namespace mlir::iree_compiler::sa {

class LocalMemory {
 public:
  explicit LocalMemory(const ::sa::Layout &lay) : lay(lay) {
    acc[0] = lay.acc0;
    acc[1] = lay.acc1;
  }

  // ACC words (bank `preferBank` first when set: a linear chunk's temps next to its accumulator)
  std::optional<uint32_t> allocAcc(uint32_t words) {
    if (preferBank >= 0 && acc[preferBank] + words <= uint32_t(preferBank + 1) * lay.cbank) {
      uint32_t w = acc[preferBank];
      acc[preferBank] += words;
      return w;
    }
    for (int b = 0; b < 2; ++b) {
      if (acc[b] + words <= uint32_t(b + 1) * lay.cbank) {
        uint32_t w = acc[b];
        acc[b] += words;
        return w;
      }
    }
    return std::nullopt;
  }
  // SPAD_A words (both banks)
  std::optional<uint32_t> allocSpad(uint32_t words) {
    if (spad + words > 2 * lay.sbank) return std::nullopt;
    uint32_t w = spad;
    spad += words;
    return w;
  }
  bool spadUsed() const { return spad != 0; }
  // a kernel's own SPAD_A layout: the first `words` words are its
  void takeSpad(uint32_t words) { spad = words; }

  struct Mark {
    uint32_t acc[2];
    uint32_t spad;
  };
  Mark mark() const { return Mark{{acc[0], acc[1]}, spad}; }
  void releaseAcc(const Mark &m) {
    acc[0] = m.acc[0];
    acc[1] = m.acc[1];
  }
  void release(const Mark &m) {
    releaseAcc(m);
    spad = m.spad;
  }

  int preferBank = -1;

 private:
  ::sa::Layout lay;
  uint32_t acc[2];
  uint32_t spad = 0;
};

}  // namespace mlir::iree_compiler::sa

#endif  // SA_TRANSFORMS_SAHLLOCALMEMORY_H_

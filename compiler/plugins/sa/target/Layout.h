// Local memory plan of the accelerator for array size D (compile_layer.Layout):
// SPAD / ACC banks, the chunk scratch at the start of each ACC bank, and the
// chunk size of a linear layer.
#ifndef SA_TARGET_LAYOUT_H_
#define SA_TARGET_LAYOUT_H_

#include <cstdint>

namespace sa {

struct Layout {
  explicit Layout(uint32_t d, uint32_t spadBytes = 128 * 1024, uint32_t accBytes = 256 * 1024);
  // Output tiles per chunk (Layout.chunk_tiles); 0 if one tile does not fit.
  uint32_t chunkTiles(uint32_t k, uint32_t nt, uint32_t rows = 1) const;

  uint32_t d, sbank, cbank, scr, acc0, acc1;
};

}  // namespace sa

#endif  // SA_TARGET_LAYOUT_H_

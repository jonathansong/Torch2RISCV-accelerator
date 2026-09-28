// Local memory plan (Layout.h), from llm/compile_layer.py.
#include "Layout.h"

#include <algorithm>

namespace sa {

Layout::Layout(uint32_t d, uint32_t spadBytes, uint32_t accBytes) : d(d) {
  sbank = spadBytes / d / 2;             // words per SPAD bank (pynq_matmul._banks)
  cbank = accBytes / (4 * d) / 2;        // words per ACC bank
  scr = cbank / 4;                       // chunk scratch at the start of each ACC bank
  acc0 = scr;
  acc1 = cbank + scr;
}

uint32_t Layout::chunkTiles(uint32_t k, uint32_t nt, uint32_t rows) const {
  uint32_t cap = std::min(sbank / k, scr / (d + 1 + rows));
  for (uint32_t n = cap; n >= 1; --n)
    if (nt % n == 0) return n;
  return 0;
}

}  // namespace sa

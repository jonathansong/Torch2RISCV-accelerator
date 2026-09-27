// Descriptor templates (Templates.h). Ported step for step from
// llm/compile_layer.py (Layout, linear) and
// compiler/plugins/sa/templates/reference.py (qlinear).
#include "Templates.h"

#include <algorithm>

namespace sa {

static constexpr uint32_t SPAD_BYTES = 128 * 1024;  // per SPAD (pynq_matmul.SPAD_BYTES)
static constexpr uint32_t ACC_BYTES = 262144;
static constexpr uint32_t PRIVATE_PARAM_W = 6, PRIVATE_PARAM_F = 7;  // PARAM6 / 7: the template's own

Layout::Layout(uint32_t d) : d(d) {
  sbank = SPAD_BYTES / d / 2;            // words per SPAD bank (pynq_matmul._banks)
  cbank = ACC_BYTES / (4 * d) / 2;       // words per ACC bank
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

uint32_t emitLinear(DescList &dl, const Layout &lay, uint32_t k, uint32_t nOut, uint32_t sX,
                    const LinearPlace &place, uint32_t pw, uint32_t pf, uint32_t scratchRows,
                    LinearEpilogue epilogue, void *epilogueCtx) {
  const uint32_t d = lay.d, sb = lay.sbank, cb = lay.cbank;
  const uint32_t nt = nOut / d, nc = lay.chunkTiles(k, nt, scratchRows);
  if (nc == 0) return 0;
  const uint32_t nch = nt / nc;
  const uint32_t wbytes = nc * k * d, fbytes = 4 * nc * d;
  const uint32_t oSw = d * nc, oY = d * nc + nc;
  const bool useLoop = nch > 4;

  auto dynOf = [](bool dyn, uint32_t p) { return dyn ? std::vector<Dyn>{{DYN_DMA_DDR, p, true}} : std::vector<Dyn>{}; };
  auto load = [&](uint32_t i, bool dyn) {
    uint32_t p = i & 1, j = i;
    dl.ld(place.wOff + j * wbytes, laddr(MEM_SPAD_B, p * sb), nc, k * d, k * d, place.wBase, 0, false, dynOf(dyn, pw));
    dl.ld(place.sOff + j * fbytes, acc(p * cb + oSw), 1, fbytes, fbytes, place.sBase, 0, false, dynOf(dyn, pf));
  };
  auto chunk = [&](uint32_t i, bool dyn, bool prefetch) {
    uint32_t p = i & 1, j = i, c = p * cb;
    dl.ex(/*a_strip=*/0, p * sb, c, k / d, false, /*repeat=*/nc, /*bstep=*/k, /*cstep=*/1, /*crow=*/nc);
    if (prefetch) load(i + 1, dyn);
    uint32_t y = c + oY;
    VeFp sw;                                   // * s_w (I32 x F32 -> F32)
    sw.t2 = VT_F32;
    dl.veFp(acc(c), acc(c + oSw), acc(y), nc * d, VOP_MUL, vtypes(VT_I32, VT_F32), 0, sw);
    VeFp sx;                                   // * s_x (broadcast word, DIV)
    sx.m2 = IDX_DIV;
    dl.veFp(acc(y), acc(sX), acc(y), nc * d, VOP_MUL, vtypes(VT_F32, VT_F32), nc, sx);
    if (epilogue) epilogue(epilogueCtx, dl, y, y + nc, j * fbytes, dyn, pf, nc);
    dl.st(place.yOff + j * fbytes, acc(y), 1, fbytes, fbytes, place.yBase, false, dynOf(dyn, pf));
  };

  load(0, false);
  uint32_t j = 0;
  if (useLoop) {
    uint32_t pairs = (nch - 1) / 2;            // the prefetch of the last pair stays in range
    if (pairs) {
      dl.setreg({{DescList::REG_PARAM + pw, 0}, {DescList::REG_PARAM + pf, 0}});
      int32_t start = int32_t(dl.size());
      chunk(0, true, true);
      chunk(1, true, true);
      dl.loopEnd(start - int32_t(dl.size()), pairs, pw, 2 * wbytes, pf, 2 * fbytes);
      j = 2 * pairs;
    }
  }
  for (uint32_t jj = j; jj < nch; ++jj) chunk(jj, false, jj + 1 < nch);
  return nc;
}

std::string buildQLinear(uint32_t d, const QLinear &q, DescList &dl) {
  Layout lay(d);
  const uint32_t k = q.k, n = q.n;
  if (k == 0 || n == 0 || k % d || n % d) return "k and n must be positive multiples of D";
  if (lay.chunkTiles(k, n / d) == 0) return "one weight tile (k words) does not fit a SPAD_B bank";
  // the A strip (k words) and, for i8 x, the raw x (k / D words) in SPAD_A;
  // for i32 x, k / D ACC words from acc0; s_x two words after x
  if (k + k / d > 2 * lay.sbank || lay.acc0 + k / d + 2 > lay.cbank) return "k too large for the local memories";

  uint32_t sx;
  if (q.xI32) {
    uint32_t xa = lay.acc0;
    dl.ld(0, acc(xa), 1, 4 * k, 4 * k, q.bX);
    VeFp rep;                                  // I32 -> I8, each element over the D rows
    rep.m1 = IDX_DIV;
    rep.p1 = d;
    dl.veFp(acc(xa), 0, laddr(MEM_SPAD_A, 0), k * d, VOP_COPY, vtypes(VT_I32, VT_I8), 0, rep);
    sx = xa + k / d;
  } else {
    dl.ld(0, laddr(MEM_SPAD_A, k), 1, k, k, q.bX);
    VeFp rep;
    rep.m1 = IDX_DIV;
    rep.p1 = d;
    dl.veFp(laddr(MEM_SPAD_A, k), 0, laddr(MEM_SPAD_A, 0), k * d, VOP_COPY, vtypes(VT_I8, VT_I8), 0, rep);
    sx = lay.acc0;
  }
  // s_x: one fp32 (the DMA moves 8 bytes), broadcast over the lanes by a max
  // reduction over its first element
  dl.ld(0, acc(sx), 1, 8, 8, q.bSx);
  VeFp bc;
  bc.reduce = RED_MAX;
  bc.valid = 1;
  dl.veFp(acc(sx), 0, acc(sx + 1), d, VOP_COPY, vtypes(VT_F32, VT_F32), 0, bc);
  LinearPlace place{q.bW, q.bSw, q.bY};
  emitLinear(dl, lay, k, n, sx + 1, place, PRIVATE_PARAM_W, PRIVATE_PARAM_F);
  dl.ret();
  return "";
}

}  // namespace sa

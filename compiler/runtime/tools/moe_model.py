"""SA descriptor stages for a Mixtral decoder with a separately dispatched MoE.

Each layer stage executes embedding or the previous FFN residual, RMSNorm,
GQA attention, and router projection on SA. The top-2 route kernel and MoE
descriptor consume its DDR outputs. The host only submits the three stages.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "llm"))

import compile_model as CM  # noqa: E402
from compile_layer import acc  # noqa: E402
from pynq_matmul import DescList  # noqa: E402

IO_RESID = CM.IO_X
IO_FFN = 0x1100
IO_SCORES = 0x2200


class MoEModelCompiler(CM.ModelCompiler):
    def __init__(self, model, experts, **kwargs):
        self.experts = experts
        if experts <= 0 or experts > model.d:
            raise ValueError("router score count must fit one output tile")
        super().__init__(model, **kwargs)
        if "wrouter" not in model.entries:
            raise ValueError("the dense model does not contain router weights")

    def _nout(self, name):
        return self.d if name == "wrouter" else super()._nout(name)

    def _restore_residual(self, dl):
        c = self.cfg
        dl.ld(self.rio(IO_RESID), acc(self.X), 1, 4 * c.dim, 4 * c.dim, base=self.rio.base)
        dl.ld(self.rio(IO_FFN), acc(self.Y), 1, 4 * c.dim, 4 * c.dim, base=self.rio.base)
        self.residual(dl)

    def build_layer(self, layer, prologue=None):
        if not 0 <= layer < self.cfg.layers:
            raise ValueError("layer index out of range")
        dl = DescList()
        if prologue is not None:
            p = prologue
            dl.setreg(*[(DescList.REG_PARAM + i, p[i]) for i in range(3)])
            dl.setreg(*[(DescList.REG_PARAM + i, p[i]) for i in range(3, 6)])
        self._seq = [(name, layer, self.cfg.dim) for name in ("wqkv", "wo", "wrouter")]
        self._li, self._pre_bank = 0, None
        if self.prefetch:
            self._prefetch(dl, self._seq[0], 0)
        if layer == 0:
            self.embed(dl)
        else:
            self._restore_residual(dl)
        c = self.cfg
        self.rmsnorm(dl, self.X, "rms_att", layer, self.XN)
        self.linear(dl, self.XN, c.dim, "wqkv", layer, out=self.QKV)
        self.rope(dl)
        self.kv_append(dl, layer)
        self.attention(dl, layer)
        self.linear(dl, self.ATT, c.dim, "wo", layer, out=self.Y)
        self.residual(dl)
        self.rmsnorm(dl, self.X, "rms_ffn", layer, self.XN)
        dl.st(self.rio(IO_RESID), acc(self.X), 1, 4 * c.dim, 4 * c.dim, base=self.rio.base)
        dl.st(self.rio(IO_FFN), acc(self.XN), 1, 4 * c.dim, 4 * c.dim, base=self.rio.base)
        self.linear(dl, self.XN, c.dim, "wrouter", layer, out_ddr=self.rio(IO_SCORES))
        return dl.end(0x14)

    def build_final(self):
        dl = DescList()
        self._seq = [("wcls", None, self.cfg.dim)]
        self._li, self._pre_bank = 0, None
        if self.prefetch:
            self._prefetch(dl, self._seq[0], 0)
        self._restore_residual(dl)
        self.final(dl)
        return dl.end(0x14)

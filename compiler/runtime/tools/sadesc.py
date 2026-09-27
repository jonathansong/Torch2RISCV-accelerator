"""The sa-desc-v1 executable format (docs/iree_compiler_plan.md §4): writer, reader, checks.

An executable holds one or more exports (entry points). Each export is a
position-independent descriptor-list template:
- only relative JUMP / CALL (w2[0] = REL) and LOOP_END; ends with RET, and
  the command buffer CALLs it;
- every DDR address is relocated by a BASE register: binding i -> BASE i;
- push constants arrive in PARAM0..5; PARAM6 / 7 are the template's own
  (loop counters, LDPARAM results).

File layout (little-endian):
  header, 64 bytes:
    0  magic "SADESC1\\0"      8  u32 version = 1     12 u32 D
    16 u32 required CAPS bits  20 u32 export count    24 u32 export table offset
    28 u32 templates offset (64-aligned)  32 u32 templates bytes
    36 u32 strings offset      40 u32 strings bytes   44..63 reserved (0)
  export table, 32 bytes per export:
    0  u32 name offset (in strings)   4  u32 name length
    8  u32 template offset (bytes, from the templates start, 64-aligned)
    12 u32 descriptor count
    16 u16 binding count  18 u16 constant count  20 u32 flags (0)
    24 u32 estimated cycles (0 = unknown)        28 u32 reserved
  templates: 64-byte descriptors, the exports back to back
  strings: export names (not NUL-terminated)

The C loader (compiler/runtime/sa/sa_loader.c) reads the same layout.
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "driver"))
from pynq_matmul import DescList  # noqa: E402

MAGIC = b"SADESC1\0"
VERSION = 1
HDR = struct.Struct("<8s9I20x")         # 64 bytes
EXP = struct.Struct("<4I2H3I")
MAX_CONSTANTS = 6                    # PARAM0..5; PARAM6 / 7 belong to the template
MAX_BINDINGS = 16                    # BASE0..15
CAPS_DESC, CAPS_NOTIFY, CAPS_FPVE, CAPS_CMDX = 1 << 21, 1 << 22, 1 << 23, 1 << 24
OP_JUMP, OP_CALL, OP_RET = DescList.JUMP, DescList.CALL, DescList.RET


def check_template(rows, bindings, constants):
    """Position independence and the calling convention; raises ValueError."""
    if rows.shape[0] == 0 or (int(rows[-1, 0]) & 0xFF) != OP_RET:
        raise ValueError("a template must end with RET")
    if bindings > MAX_BINDINGS or constants > MAX_CONSTANTS:
        raise ValueError(f"at most {MAX_BINDINGS} bindings and {MAX_CONSTANTS} constants")
    for i, w in enumerate(rows):
        op = int(w[0]) & 0xFF
        if op in (OP_JUMP, OP_CALL) and not int(w[2]) & 1:
            raise ValueError(f"descriptor {i}: absolute JUMP / CALL in a template")
        if op == DescList.END:
            raise ValueError(f"descriptor {i}: END inside a template (the command buffer ends the list)")
        if op in (DescList.LD, DescList.ST, DescList.LDPARAM) and not int(w[0]) & DescList.RELOC:
            raise ValueError(f"descriptor {i}: DDR address without BASE relocation in a template")
        if int(w[0]) & DescList.RELOC:
            base = ((int(w[0]) >> 9) & 3) | (((int(w[0]) >> 13) & 3) << 2)
            if base >= bindings:
                raise ValueError(f"descriptor {i}: BASE{base} but only {bindings} bindings")


class Executable:
    def __init__(self, d, caps=CAPS_DESC | CAPS_CMDX):
        self.d, self.caps = d, caps
        self.exports = []

    def add(self, name, dl, bindings, constants=0, est_cycles=0):
        """dl: a DescList ending with RET (or not: RET is appended)."""
        rows = dl.array()
        if rows.shape[0] == 0 or (int(rows[-1, 0]) & 0xFF) != OP_RET:
            dl.ret()
            rows = dl.array()
        check_template(rows, bindings, constants)
        self.exports.append((name, rows, bindings, constants, est_cycles))
        return len(self.exports) - 1

    def to_bytes(self):
        n = len(self.exports)
        exp_off = HDR.size
        tmpl_off = -(-(exp_off + EXP.size * n) // 64) * 64
        tmpl = b"".join(rows.tobytes() for _, rows, _, _, _ in self.exports)
        str_off = tmpl_off + len(tmpl)
        strings, table, t, s = b"", b"", 0, 0
        for name, rows, b, c, cyc in self.exports:
            nb = name.encode()
            table += EXP.pack(s, len(nb), t, rows.shape[0], b, c, 0, cyc, 0)
            strings += nb
            t += rows.nbytes
            s += len(nb)
        hdr = HDR.pack(MAGIC, VERSION, self.d, self.caps, n, exp_off, tmpl_off, len(tmpl), str_off, len(strings))
        blob = hdr + table
        blob += b"\0" * (tmpl_off - len(blob))
        return blob + tmpl + strings

    def save(self, path):
        with open(path, "wb") as f:
            f.write(self.to_bytes())


def read(data):
    """bytes -> (D, caps, [(name, rows uint64 (n, 8), bindings, constants, est_cycles)])."""
    f = HDR.unpack_from(data, 0)
    if f[0] != MAGIC or f[1] != VERSION:
        raise ValueError("not an sa-desc-v1 executable")
    d, caps, n, exp_off, tmpl_off, tmpl_bytes, str_off, str_bytes = f[2:10]
    out = []
    for i in range(n):
        so, sl, to, cnt, b, c, _, cyc, _ = EXP.unpack_from(data, exp_off + EXP.size * i)
        name = data[str_off + so:str_off + so + sl].decode()
        rows = np.frombuffer(data, "<u8", cnt * 8, tmpl_off + to).reshape(cnt, 8)
        out.append((name, rows, b, c, cyc))
    return d, caps, out


def dispatch_list(entry_phys, binding_phys, constants):
    """The list the sa HAL driver submits for one dispatch (sa_loader.c builds the
    same): SETREG BASE0.. = the bindings' physical addresses, SETREG PARAM0.. =
    the push constants, CALL the export's template, END."""
    regs = [(i, int(a)) for i, a in enumerate(binding_phys)]
    regs += [(DescList.REG_PARAM + i, int(v)) for i, v in enumerate(constants)]
    dl = DescList()
    for i in range(0, len(regs), 3):
        dl.setreg(*regs[i:i + 3])
    dl.call(entry_phys)
    return dl.end(0x5A)

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
    36 u32 strings offset      40 u32 strings bytes
    44 u32 export extensions offset (version 3)       48..63 reserved (0)
  export table, 32 bytes per export:
    0  u32 name offset (in strings)   4  u32 name length
    8  u32 template offset (bytes, from the templates start, 64-aligned)
    12 u32 descriptor count
    16 u16 binding count  18 u16 constant count
    20 u32 register setup entry count (version 2; 0 = the default setup)
    24 u32 estimated cycles (0 = unknown)
    28 u32 register setup table offset (version 2, bytes from the file start)
  templates: 64-byte descriptors, the exports back to back
  strings: export names (not NUL-terminated)
  register setup tables (version 2), 16 bytes per entry:
    0 u8 kind (0 BASE, 1 PARAM)  1 u8 register  2 u8 binding  3 u8 0
    4 i16 push constant ordinal (-1: none)  6 u16 0
    8 i32 mul  12 i32 add   (the divisor is a power of two: kind | log2(div) << 4)
  export extensions (version 3), 32 bytes per export, u32 each:
    0 prefix descriptors (its RET included; 0: none)   4 bindings read by the prefix (mask)
    8 bindings written (mask, ST)   12 bindings read (mask, LD / LDPARAM, the prefix included)
    16 head descriptors (the body's leading loads and a RET; 0: none)
    20 flags (bit 0: the export touches SPAD_B)   24 the prefix's register   28 0
  value = ((constant * mul) >> log2(div)) + add, and for BASE + the binding's
  physical address. The default setup (no table) is BASE i = binding i,
  PARAM j = push constant j. With a table, dynamic binding offsets (IREE
  passes them as push constants), lengths derived from push constants and
  constants beyond six are handled by the driver; the template only has
  static offsets.
  The prefix (version 3): a template may start with loads (ending with RET,
  through BASE15, which the driver sets to the value of the prefix's register)
  that the driver may run early: before the FENCE ordering the dispatch after
  the previous ones, or inside an earlier dispatch that does not touch SPAD_B,
  between its head and the rest, when the bindings they read are not written
  in the list. The body follows the prefix; with a head, the body is head +
  RET + the rest. Without extensions every binding counts as read and written.
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
VERSION = 2                          # written by Executable (no prefixes); read: 1..3
SETUP = struct.Struct("<4BhH2i")
SETUP_BASE, SETUP_PARAM = 0, 1
HDR = struct.Struct("<8s10I16x")         # 64 bytes
EXP = struct.Struct("<4I2H3I")
EXT = struct.Struct("<8I")
MAX_CONSTANTS = 6                    # PARAM0..5; PARAM6 / 7 belong to the template
MAX_BINDINGS = 16                    # BASE0..15
CAPS_DESC, CAPS_NOTIFY, CAPS_FPVE, CAPS_CMDX = 1 << 21, 1 << 22, 1 << 23, 1 << 24
OP_JUMP, OP_CALL, OP_RET = DescList.JUMP, DescList.CALL, DescList.RET


def check_template(rows, bindings, constants):  # bindings: BASE registers the template may use
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

    def add(self, name, dl, bindings, constants=0, est_cycles=0, setup=None):
        """dl: a DescList ending with RET (or not: RET is appended). setup:
        register setup entries (kind, reg, binding, const, mul, div, add), or
        None for the default (BASE i = binding i, PARAM j = constant j)."""
        rows = dl.array()
        if rows.shape[0] == 0 or (int(rows[-1, 0]) & 0xFF) != OP_RET:
            dl.ret()
            rows = dl.array()
        nbase = bindings
        if setup:
            nbase = 1 + max([e[1] for e in setup if e[0] == SETUP_BASE], default=-1)
        check_template(rows, nbase, constants if not setup else 0)
        self.exports.append((name, rows, bindings, constants, est_cycles, list(setup or [])))
        return len(self.exports) - 1

    def to_bytes(self):
        n = len(self.exports)
        exp_off = HDR.size
        tmpl_off = -(-(exp_off + EXP.size * n) // 64) * 64
        tmpl = b"".join(e[1].tobytes() for e in self.exports)
        str_off = tmpl_off + len(tmpl)
        strings = b"".join(e[0].encode() for e in self.exports)
        setup_off = -(-(str_off + len(strings)) // 16) * 16
        table, setups, t, s = b"", b"", 0, 0
        for name, rows, b, c, cyc, setup in self.exports:
            nb = name.encode()
            table += EXP.pack(s, len(nb), t, rows.shape[0], b, c, len(setup), cyc,
                              setup_off + len(setups) if setup else 0)
            for kind, reg, binding, const, mul, div, add in setup:
                if div & (div - 1):
                    raise ValueError("setup divisor must be a power of two")
                setups += SETUP.pack(kind | (div.bit_length() - 1) << 4, reg, binding, 0, const, 0, mul, add)
            t += rows.nbytes
            s += len(nb)
        hdr = HDR.pack(MAGIC, VERSION, self.d, self.caps, n, exp_off, tmpl_off, len(tmpl), str_off, len(strings), 0)
        blob = hdr + table
        blob += b"\0" * (tmpl_off - len(blob))
        blob += tmpl + strings
        blob += b"\0" * (setup_off - len(blob)) if setups else b""
        return blob + setups

    def save(self, path):
        with open(path, "wb") as f:
            f.write(self.to_bytes())


def read(data):
    """bytes -> (D, caps, [(name, rows uint64 (n, 8), bindings, constants, est_cycles, setup)]);
    setup: [(kind, reg, binding, const, mul, div, add)] (empty: the default)."""
    f = HDR.unpack_from(data, 0)
    if f[0] != MAGIC or f[1] not in (1, 2, 3):
        raise ValueError("not an sa-desc executable (version 1 to 3)")
    d, caps, n, exp_off, tmpl_off, tmpl_bytes, str_off, str_bytes = f[2:10]
    out = []
    for i in range(n):
        so, sl, to, cnt, b, c, nsetup, cyc, setup_off = EXP.unpack_from(data, exp_off + EXP.size * i)
        name = data[str_off + so:str_off + so + sl].decode()
        rows = np.frombuffer(data, "<u8", cnt * 8, tmpl_off + to).reshape(cnt, 8)
        setup = []
        for j in range(nsetup if f[1] >= 2 else 0):
            kd, reg, binding, _, const, _, mul, add = SETUP.unpack_from(data, setup_off + SETUP.size * j)
            setup.append((kd & 15, reg, binding, const, mul, 1 << (kd >> 4), add))
        out.append((name, rows, b, c, cyc, setup))
    return d, caps, out


def read_ext(data):
    """bytes -> [dict(prefix, prefix_reads, writes, reads, head, spad_b, prefix_reg)]
    per export (version 3; before: none, every binding read and written)."""
    f = HDR.unpack_from(data, 0)
    n = f[4]
    if f[1] < 3:
        return [dict(prefix=0, prefix_reads=0, writes=0xFFFF, reads=0xFFFF, head=0, spad_b=True, prefix_reg=0)] * n
    out = []
    for i in range(n):
        p, pr, w, r, h, fl, reg, _ = EXT.unpack_from(data, f[10] + EXT.size * i)
        out.append(dict(prefix=p, prefix_reads=pr, writes=w, reads=r, head=h, spad_b=bool(fl & 1), prefix_reg=reg))
    return out


def setup_values(setup, binding_phys, constants):
    """The register values of one dispatch: [(register number for SETREG, value)]."""
    if not setup:
        regs = [(i, int(a)) for i, a in enumerate(binding_phys)]
        return regs + [(DescList.REG_PARAM + i, int(v)) for i, v in enumerate(constants)]
    regs = []
    for kind, reg, binding, const, mul, div, add in setup:
        v = ((int(constants[const]) * mul) >> (div.bit_length() - 1) if const >= 0 else 0) + add
        if kind == SETUP_BASE:
            regs.append((reg, (int(binding_phys[binding]) + v) & 0xFFFFFFFF))
        else:
            regs.append((DescList.REG_PARAM + reg, v & 0xFFFFFFFF))
    return regs


def dispatch_list(entry_phys, binding_phys, constants, setup=None, ext=None):
    """The list the sa HAL driver submits for one dispatch (sa_context.c builds the
    same): SETREG of the register values (setup_values; BASE15 for a prefix),
    CALL the export's prefix, head and rest, END."""
    regs = setup_values(setup, binding_phys, constants)
    ext = ext or {}
    prefix, head = ext.get("prefix", 0), ext.get("head", 0)
    if prefix:
        regs.append((15, dict(regs)[ext["prefix_reg"]]))
    dl = DescList()
    for i in range(0, len(regs), 3):
        dl.setreg(*regs[i:i + 3])
    if prefix:
        dl.call(entry_phys)
    if head:
        dl.call(entry_phys + 64 * prefix)
    dl.call(entry_phys + 64 * (prefix + head))
    return dl.end(0x5A)

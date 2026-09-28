#!/usr/bin/env python3
"""Disassembles sa-desc executables (sadesc.py): one line per descriptor.

    python3 compiler/runtime/tools/sadis.py <file.sadesc> [...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sadesc  # noqa: E402

OPS = {1: "LD", 2: "ST", 3: "EX", 4: "VE", 0x10: "FENCE", 0x12: "END", 0x13: "LOOP_END", 0x14: "SETREG",
       0x15: "CALL", 0x16: "RET", 0x17: "LDPARAM"}


def line(i, w):
    w = [int(x) for x in w]
    op = w[0] & 0xFF
    head = f"{i:4} {OPS.get(op, hex(op)):8}"
    base = ((w[0] >> 9) & 3) | (((w[0] >> 13) & 3) << 2) if (w[0] >> 8) & 1 else None
    dyn = [(w[0] >> (16 + 8 * k)) & 0xFF for k in range(2)]
    extra = (f" base={base}" if base is not None else "") + (" fb" if (w[0] >> 11) & 1 else "")
    extra += "".join(f" dyn(f{x & 15},p{(x >> 4) & 7}{'+' if x & 128 else ''})" for x in dyn if x)
    M = 0xFFFFFFFF
    if op in (1, 2):
        body = f"ddr={w[1] & M:#x} la={w[2] & M:#x} rows={(w[2] >> 32) & 0xFFFF} rb={w[2] >> 48} pitch={w[3] & M}"
    elif op == 3:
        body = (f"a={w[1] & 0xFFFF:#x} b={(w[1] >> 16) & 0xFFFF:#x} c={(w[1] >> 32) & 0xFFFF:#x} kt={(w[1] >> 48) & 0xFFF}"
                f"{' acc' if (w[1] >> 60) & 1 else ''} rep={w[2] & 0xFFFF} bstep={(w[2] >> 16) & 0xFFFF} "
                f"cstep={(w[2] >> 32) & 0xFFFF} crow={w[2] >> 48}")
    elif op == 4:
        f = w[3] >> 53
        if (w[3] & 0xFF) == 6 and not f & 1:
            body = f"TRANSPOSE src={w[1]:#x} dst={w[2] & M:#x} len={w[2] >> 32} types={(w[3] >> 8) & 0xFF:#x} stride={w[7] >> 48}"
        else:
            fl = lambda b: float(np.array([b & M], np.uint32).view(np.float32)[0])
            body = (f"op={w[3] & 0xFF} s1={w[1] & M:#x} s2={w[1] >> 32:#x} dst={w[2] & M:#x} len={w[2] >> 32} "
                    f"types={(w[3] >> 8) & 0xFF:#x} func={(f >> 1) & 7} m1={(f >> 4) & 3} m2={(f >> 6) & 3} "
                    f"red={(f >> 8) & 3}{' swapneg' if (f >> 10) & 1 else ''} A={fl(w[6])} B={fl(w[6] >> 32)} "
                    f"imm={fl(w[5] >> 32)} period={(w[3] >> 16) & 0xFFFF} rowlen={w[7] & 0xFFFF} "
                    f"valid={(w[7] >> 16) & 0xFFFF} p1={w[7] >> 32}")
    elif op == 0x17:
        body = f"addr={w[1] & M:#x} param={w[2] & 7} mul={w[3] & 0xFFFF} add={w[4] & M}"
    elif op == 0x14:
        regs = [(w[1] >> (8 * k)) & 0xFF for k in range(3)]
        body = " ".join(f"r{r & 0x3F}={w[2 + k] & M}" for k, r in enumerate(regs) if r) + f" add={w[5] & 7}"
    elif op == 0x13:
        body = f"offset={np.int32(np.uint32(w[1] & M))} count={w[2] & 0xFFFF} k1={(w[2] >> 16) & 7} s1={w[3] & M} k2={(w[2] >> 19) & 7} s2={w[4] & M}"
    elif op == 0x10:
        body = f"mask={w[1] & 0xF}"
    else:
        body = ""
    return f"{head} {body}{extra}"


def main():
    for path in sys.argv[1:]:
        data = open(path, "rb").read()
        _, _, exps = sadesc.read(data)
        exts = sadesc.read_ext(data)
        for (name, rows, nb, nc, cyc, setup), ext in zip(exps, exts):
            print(f"{name}: {rows.shape[0]} descriptors, {nb} bindings, {nc} constants; prefix {ext['prefix']}, "
                  f"head {ext['head']}, reads {ext['reads']:#x}, writes {ext['writes']:#x}, SPAD_B {ext['spad_b']}")
            for s in setup:
                print(f"     setup {'BASE' if s[0] == 0 else 'PARAM'}{s[1]} = binding {s[2]} + ((c{s[3]} * {s[4]}) / {s[5]}) + {s[6]}"
                      if s[0] == 0 else f"     setup PARAM{s[1]} = ((c{s[3]} * {s[4]}) / {s[5]}) + {s[6]}")
            for i, w in enumerate(rows):
                print(line(i, w))


if __name__ == "__main__":
    main()

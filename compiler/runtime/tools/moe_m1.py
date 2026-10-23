#!/usr/bin/env python3
"""Builds the M1 host-assisted MoE dynamic expert GEMV executable.

The host writes a 32-byte routing record. The template reads byte offset 12
(`expert_offset[0]`) through LDPARAM and adds it to the weight GEMV's DDR
field, so one shared descriptor template addresses every expert.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "compiler", "frontend"))
sys.path.insert(0, os.path.join(REPO, "driver"))

import moe  # noqa: E402
import sadesc  # noqa: E402
from pynq_matmul import DescList, MEM_ACC, MEM_SPAD_A, laddr  # noqa: E402


def expert_projection_template(dim, hidden, d=16, offset_param=6):
    """One selected expert's W8 gate projection, producing raw int32 sums."""
    if dim <= 0 or hidden <= 0 or dim % d or hidden % d:
        raise ValueError("dim and hidden must be positive multiples of D")
    strips = hidden // d
    if strips > 8:
        raise ValueError("the M1 template supports at most 8 GEMV strips")
    row_bytes = dim * d
    return (DescList()
            .ld(0, laddr(MEM_SPAD_A, 0), 1, dim, dim, base=0)
            .ldparam(moe.ROUTE_OFFSETS_OFFSET, offset_param, base=1)
            .gemv(0, laddr(MEM_ACC, 0), strips, row_bytes, row_bytes, 0,
                  base=2, dyn=[("ddr", offset_param, True)])
            .st(0, laddr(MEM_ACC, 0), 1, hidden * 4, hidden * 4, base=3)
            .ret())


def make_executable(dim, hidden, d=16):
    executable = sadesc.Executable(d, caps=sadesc.CAPS_DESC | sadesc.CAPS_CMDX | sadesc.CAPS_GEMV)
    executable.add("moe_gate", expert_projection_template(dim, hidden, d), bindings=4)
    return executable.to_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128,
                        help="M1 supports hidden/D <= 8; larger projections are chunked in M2")
    parser.add_argument("--d", type=int, default=16)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(make_executable(args.dim, args.hidden, args.d))
    print(f"{args.out}: dynamic expert gate GEMV, H={args.dim}, I={args.hidden}, D={args.d}")


if __name__ == "__main__":
    main()

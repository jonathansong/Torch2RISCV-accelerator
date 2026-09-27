#!/usr/bin/env python3
"""One-screen summary of dumped dispatch sources (loads / stores with their
bindings and shapes, and every compute op with its maps and body).

    python3 compiler/tests/summarize.py build/c3/stories15M/sa_sources [numbers...]
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oracle import load_dispatch  # noqa: E402


def short(t):
    s = str(t)
    s = s.replace("!iree_tensor_ext.dispatch.tensor<", "dt<").replace("tensor<", "<")
    return s


def summarize(path):
    keep = load_dispatch(path)
    f = keep[2]
    name = f.attributes["sym_name"]
    out = [str(name).strip('"').replace("main$async_dispatch_", "#")]
    subspans = {}
    for op in f.regions[0].blocks[0].operations:
        o = op.operation
        if o.name == "hal.interface.binding.subspan":
            subspans[o.results[0]] = int(str(o.attributes["binding"]).split(":")[0])
        elif o.name == "iree_tensor_ext.dispatch.tensor.load":
            b = subspans.get(o.operands[0], "?")
            offs = re.sub(r"array<i64: ?", "", str(o.attributes["static_offsets"])).rstrip(">")
            out.append(f"  load  b{b} [{offs}] {short(o.results[0].type)}")
        elif o.name == "iree_tensor_ext.dispatch.tensor.store":
            b = subspans.get(o.operands[1], "?")
            out.append(f"  store b{b} {short(o.operands[0].type)}")
        elif o.name in ("linalg.generic", "linalg.batch_matmul", "linalg.fill", "iree_linalg_ext.scatter"):
            desc = o.name.split(".")[-1]
            if o.name == "linalg.generic":
                maps = [str(m).replace("affine_map<", "").rstrip(">") for m in o.attributes["indexing_maps"]]
                iters = "".join("r" if "reduction" in str(i) else "p" for i in o.attributes["iterator_types"])
                body = []
                for bo in o.regions[0].blocks[0].operations:
                    n = bo.operation.name
                    if n == "arith.constant":
                        body.append(str(bo.operation.attributes["value"]).split(":")[0].strip())
                    elif n != "linalg.yield":
                        body.append(n.split(".")[-1])
                desc += f" [{iters}] {' '.join(maps)} :: {' '.join(body)}"
            out.append(f"  {desc}  -> {short(o.results[0].type)}")
    return "\n".join(out)


def main():
    d = sys.argv[1]
    want = set(sys.argv[2:])
    files = sorted(glob.glob(os.path.join(d, "*.mlir")), key=lambda p: int(re.findall(r"_(\d+)\.mlir$", p)[0]))
    for p in files:
        num = re.findall(r"_(\d+)\.mlir$", p)[0]
        if want and num not in want:
            continue
        print(summarize(p))


if __name__ == "__main__":
    main()

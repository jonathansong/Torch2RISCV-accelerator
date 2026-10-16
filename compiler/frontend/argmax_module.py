#!/usr/bin/env python3
"""The block maxima of the logits for the decode loop on the RISC-V core
(docs/riscv_backend_design.md §7.1, plan B; sa-llm-run --rvsched): module
`sa_argmax`, function `blockmax(f32[vocab]) -> f32[vocab / block]`, one f32
max reduction per block of `block` logits, compiled for the sa device. The
core then scans the block maxima and the winning block for the argmax (the
first position holding the largest value, as the host's argmax): vocab / block
+ block uncached reads instead of vocab.

    compiler/frontend/argmax_module.py --vocab 151936 --out <dir>/argmax.vmfb [--block 0]
block 0: the powers of two dividing vocab (16-512), cheapest first (vocab / block + block reads), the
first that compiles for the accelerator.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
COMPILE = os.path.join(os.environ.get("IREE_BUILD", os.path.join(REPO, "build", "iree", "build-compiler")),
                       "tools", "iree-compile")


def blocks(vocab):
    """the powers of two from 16 to 512 dividing vocab, cheapest first (vocab / block + block reads)"""
    bs = [b for b in (16, 32, 64, 128, 256, 512) if vocab % b == 0]
    if not bs:
        raise SystemExit(f"vocab {vocab}: not a multiple of 16")
    return sorted(bs, key=lambda b: vocab // b + b)


def mlir(vocab, block):
    n = vocab // block
    return f"""module @sa_argmax {{
  func.func @blockmax(%x: tensor<{vocab}xf32>) -> tensor<{n}xf32> {{
    %e = tensor.expand_shape %x [[0, 1]] output_shape [{n}, {block}] : tensor<{vocab}xf32> into tensor<{n}x{block}xf32>
    %ninf = arith.constant 0xFF800000 : f32
    %i = tensor.empty() : tensor<{n}xf32>
    %f = linalg.fill ins(%ninf : f32) outs(%i : tensor<{n}xf32>) -> tensor<{n}xf32>
    %r = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}}
        ins(%e : tensor<{n}x{block}xf32>) outs(%f : tensor<{n}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      %m = arith.maximumf %a, %b : f32
      linalg.yield %m : f32
    }} -> tensor<{n}xf32>
    return %r : tensor<{n}xf32>
  }}
}}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cands = [args.block] if args.block else blocks(args.vocab)
    if args.vocab % cands[0]:
        raise SystemExit(f"block {cands[0]} does not divide vocab {args.vocab}")
    src = os.path.splitext(args.out)[0] + ".mlir"
    for block in cands:     # (a shape that does not compile: the next cheapest)
        with open(src, "w") as f:
            f.write(mlir(args.vocab, block))
        r = subprocess.run([COMPILE, src, "--iree-hal-target-device=sa", "--iree-hal-link-executables=false",
                            f"--iree-sa-d={os.environ.get('SA_D', '16')}",
                            f"--iree-sa-gemv-ports={os.environ.get('SA_GEMV_PORTS', '4')}",
                            "--iree-sa-codegen-report", "-o", args.out], capture_output=True, text=True)
        report = [line for line in r.stderr.splitlines() if line.startswith("sa codegen:")]
        if not r.returncode and report and all(line.endswith(": ok") for line in report):
            print(f"{args.out}: blockmax of {args.vocab} logits in {args.vocab // block} blocks of {block} ({report[0]})")
            return 0
        print(f"block {block}: {report or r.stderr.strip().splitlines()[-1:]}", file=sys.stderr)
    raise SystemExit("argmax module: no block size compiles for the accelerator")


if __name__ == "__main__":
    sys.exit(main())

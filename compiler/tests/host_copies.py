#!/usr/bin/env python3
"""Host copies per step of an export (iree_compiler_plan.md §8.22): the module
compiled to the stream level with compile_sa.sh's flags, then per public async
function the stream.cmd.copy ops, their bytes and how many touch a global
(`!stream.resource<variable>`: the KV caches). The sa driver runs these copies
on the CPU over the uncached device window, between the dispatches, where
neither the sim's checks nor the cycle model see them (on the board a packed V
cache written by a strided slice cost Qwen3 640 ms per decode step).

    compiler/tests/host_copies.py build/hfgen/<export> [--max-variable-bytes 0]
exit 1 when a function copies more bytes of a global than allowed (default 0).
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
COMPILE = os.path.join(os.environ.get("IREE_BUILD", os.path.join(REPO, "build", "iree", "build-compiler")),
                       "tools", "iree-compile")


def stream_ir(d):
    """the export compiled to the stream level, compile_sa.sh's flags"""
    flags = ["--iree-hal-target-device=sa", "--iree-opt-const-expr-max-size-increase-threshold=0",
             f"--iree-parameter-import=model={d}/qllama.irpa", "--iree-parameter-import-maximum-size=4294967295",
             "--iree-dispatch-creation-enable-aggressive-fusion", "--iree-opt-strip-assertions",
             "--iree-sa-host-fallback", "--iree-hal-link-executables=false",
             f"--iree-sa-d={os.environ.get('SA_D', '16')}", f"--iree-sa-gemv-ports={os.environ.get('SA_GEMV_PORTS', '4')}"]
    if os.environ.get("SA_GENERALIZE", "1") == "1":
        flags.append("--iree-flow-enable-executable-generalization")
    if os.environ.get("SA_FUSE_FFN", "1") == "1":
        flags.append("--iree-sa-fuse-ffn")
    if os.environ.get("SA_FUSE_ATTN", "1") == "1":
        flags.append("--iree-sa-fuse-attn")
        if os.environ.get("SA_FUSE_ATTN_CHUNK", "0") == "1":     # (compile_sa.sh's: the prefill chunk kernel)
            flags.append("--iree-sa-fuse-attn-chunk")
    cf = os.path.join(d, "compile_flags.txt")
    if os.path.exists(cf):
        flags += open(cf).read().split()
    flags += os.environ.get("SA_COMPILE_FLAGS", "").split()
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "s.mlir")
        r = subprocess.run([COMPILE, os.path.join(d, "qllama.mlir"), *flags, "--compile-to=stream",
                            "--mlir-elide-elementsattrs-if-larger=8", "-o", out], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("\n".join(r.stderr.strip().splitlines()[-5:]))
        return open(out).read()


def copies(text):
    """{function: (copies, bytes, copies touching a global, their bytes)}"""
    consts = {m.group(1): int(m.group(2)) for m in re.finditer(r"(%c\d+) = arith\.constant (\d+) : index", text)}
    starts = [(m.start(), m.group(1)) for m in re.finditer(r"util\.func public @([\w$]+)\(", text)]
    res = {}
    for i, (p, name) in enumerate(starts):
        if not name.endswith("$async"):
            continue
        body = text[p:starts[i + 1][0] if i + 1 < len(starts) else len(text)]
        n = nb = nv = vb = 0
        for line in body.split("\n"):
            m = re.search(r"stream\.cmd\.copy [^,]+, [^,]+, (%\w+) : (.*)", line)
            if not m:
                continue
            size = consts.get(m.group(1), int(m.group(1)[2:]) if re.fullmatch(r"%c\d+", m.group(1)) else 0)
            n, nb = n + 1, nb + size
            if "resource<variable>" in m.group(2):
                nv, vb = nv + 1, vb + size
        res[name[:-len("$async")]] = (n, nb, nv, vb)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export")
    ap.add_argument("--max-variable-bytes", type=int, default=0)
    args = ap.parse_args()
    bad = 0
    for fn, (n, nb, nv, vb) in copies(stream_ir(args.export)).items():
        flag = vb > args.max_variable_bytes
        bad += flag
        print(f"{fn:12s} {n:4d} host copies, {nb:10d} bytes; of a global (the KV caches): {nv} copies, {vb} bytes"
              + ("  <- TOO MANY" if flag else ""))
    print(f"host copies {'FAIL' if bad else 'PASS'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

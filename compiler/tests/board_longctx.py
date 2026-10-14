#!/usr/bin/env python3
"""Long context on the board (docs/iree_compiler_plan.md §8.22, C6.5 / C6.5b):
a bundle compiled at --max-len beyond 256 (sa_args.txt --valid_len) and a
prompt of P tokens (default: the first window of tf_tokens.npy, P = valid_len
- G), run twice on the device:

  1. prefill (sa_args.txt's --prefill=M) + G decode steps: the logits of the
     prompt's last position and of the generated steps; the speeds at long T
  2. decode only, the prompt token by token: every step's logits streamed
     (board_teacher.run_streamed, 1.2 GB for Qwen3 at 2048); the rows from
     P - 1 on must be bit-exact with run 1 (the sim checks the same equality:
     run_tests.sh longctx*, the 1104-token end to end of §8.22), the others are
     teacher forcing over the prompt: -log p of the next token by position
     range, against the fp32 reference (tf_ref.npz) when the prompt is its window

The full model has no sim reference at these lengths (the 2-layer sim run of
1104 tokens takes hours), so the check is the device against itself through
two different kernel sets (prefill's [M, T] attention, decode's T loop) plus
the quality against fp32. Writes longctx_board.npz.

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh; source /etc/profile.d/xrt_setup.sh 2>/dev/null; \\
        cd <bundle> && python3 board_longctx.py [--len P] [--generate G]'
    compiler/tests/board_longctx.py --sim --bundle <dir> --len 48 --generate 4     (on the host: the sim)
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from board_teacher import log_softmax_rows, run_streamed  # noqa: E402

RANGES = (0, 256, 512, 1024, 2048, 4096)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--len", type=int, help="prompt tokens (default: valid_len - generate)")
    ap.add_argument("--generate", type=int, default=16)
    ap.add_argument("--bundle", default=HERE)
    ap.add_argument("--sim", action="store_true", help="the functional simulator (x86 sa-llm-run) instead of the board")
    args = ap.parse_args()
    b, G = os.path.abspath(args.bundle), args.generate
    extra = open(os.path.join(b, "sa_args.txt")).read().split()
    opt = lambda k, d: int(next((a.split("=")[1] for a in extra if a.startswith(f"--{k}=")), d))  # noqa: E731
    vlen, M = opt("valid_len", 256), opt("prefill", 0)
    if not M:
        print("the bundle has no prefill (sa_args.txt --prefill=M)")
        return 1
    V = np.load(os.path.join(b, "expected_logits.npy"), mmap_mode="r").shape[1]
    win = np.load(os.path.join(b, "tf_tokens.npy"))[0]
    P = args.len or min(len(win), vlen) - G
    if P + G > vlen or P > len(win) or P < M:
        print(f"prompt {P} + {G} generated: needs M = {M} <= P <= {len(win)} (tf_tokens) and P + G <= {vlen} (valid_len)")
        return 1
    prompt = win[:P]
    mb = int(open(os.path.join(b, "board.txt")).read().split()[-1])
    if args.sim:
        repo = os.path.abspath(os.path.join(HERE, "..", ".."))
        exe = os.path.join(repo, "build/iree/build-sa-host/runtime/plugins/hal/drivers/sa/sa-llm-run")
    else:
        import board_launcher as BL
        exe = os.path.join(b, "sa-llm-run")
    base = [exe, "--device=sa", f"--module={b}/sa.vmfb", f"--parameters=model={b}/sa_packed.irpa",
            "--tokens=" + ",".join(str(int(t)) for t in prompt), f"--generate={G}"]

    def device(cmd, flag, on_rows):
        """one sa-llm-run on the board (or a fresh sim server)"""
        if not args.sim:
            return run_streamed(cmd, flag, V, on_rows, env=dict(env, SA_STATS="1"))
        sock = f"/tmp/sa_lc_{os.getpid()}.sock"
        srv = subprocess.Popen([sys.executable, os.path.join(HERE, "..", "sim", "sa_sim_server.py"), "--d",
                                os.environ.get("SA_D", "16"), "--mb", str(mb), "--socket", sock, "--once"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        try:
            return run_streamed(cmd, flag, V, on_rows, env=dict(os.environ, SA_STATS="1", SA_SIM_SOCKET=sock))
        finally:
            srv.wait(timeout=120)

    pre_rows, dec_rows = [], []
    nll = np.zeros(P - 1, np.float32)
    argmax = np.zeros(P, np.int32)

    def keep(rows):
        return lambda i, lg: rows.append(lg.copy())

    def teacher(i, lg):
        for r in range(len(lg)):
            j = i + r
            if j < P:
                argmax[j] = lg[r].argmax()
                if j + 1 < P:
                    nll[j] = -log_softmax_rows(lg[r:r + 1])[0, int(prompt[j + 1])]
            if j >= P - 1:
                dec_rows.append(lg[r].copy())

    if not args.sim:
        mm, buf, env = BL.start(os.path.join(b, "picorv32.bit"), os.path.join(b, "rt_fw.bin"), mb=mb, ring=16)
    try:
        print(f"{P}-token prompt (tf_tokens.npy window 0), {G} generated, valid_len {vlen}, prefill M = {M}"
              f"{' (sim)' if args.sim else ''}", flush=True)
        t0 = time.time()
        r1 = device(base + extra, "--logits_out", keep(pre_rows))
        t1 = time.time()
        print(f"1. prefill + decode: rc {r1[0]}, {r1[3]} rows, {t1 - t0:.1f} s", flush=True)
        r2 = device(base + [a for a in extra if not a.startswith("--prefill")], "--logits_out", teacher)
        t2 = time.time()
        print(f"2. decode only: rc {r2[0]}, {r2[3]} rows, {t2 - t1:.1f} s", flush=True)
    finally:
        if not args.sim:
            BL.stop(mm, buf)
    for name, (rc, out, err, _) in (("prefill + decode", r1), ("decode only", r2)):
        if rc:
            print(f"{name}: sa-llm-run failed ({rc})\n{out[-2000:]}\n{err[-3000:]}")
            return 1
        lines = out.strip().splitlines()
        print(f"  {name}: " + "; ".join(lines[1:3]))
        for l in err.splitlines():
            if l.startswith("sa:"):
                print("  " + l)
    tok1, tok2 = (r[1].strip().splitlines()[0].split(":")[1].split() for r in (r1, r2))
    pre = np.concatenate(pre_rows) if pre_rows else np.zeros((0, V), np.float32)
    n = min(len(pre), len(dec_rows))
    exact = sum(pre[i].tobytes() == dec_rows[i].tobytes() for i in range(n))
    same = tok1 == tok2
    print(f"prefill vs decode only: logits bit-exact in {exact}/{G + 1} rows (from position {P - 1}), "
          f"generated tokens {'identical' if same else 'DIFFERENT'}")
    # teacher forcing over the prompt, by position range
    ref = None
    if os.path.exists(os.path.join(b, "tf_ref.npz")):
        ref = np.load(os.path.join(b, "tf_ref.npz"))
    print("teacher forcing over the prompt (-log p of the next token; positions):")
    for lo, hi in zip(RANGES, RANGES[1:]):
        if lo >= P - 1:
            break
        hi = min(hi, P - 1)
        s = f"  [{lo:4d}, {hi:4d}): PPL {np.exp(nll[lo:hi].mean()):8.3f}"
        if ref is not None:
            s += (f"  fp32 {np.exp(ref['nll'][0, lo:hi].mean()):8.3f}"
                  f"  top-1 vs fp32 {100 * (argmax[lo:hi] == ref['argmax'][0, lo:hi]).mean():5.1f}%")
        print(s)
    s = f"  all {P - 1}: PPL {np.exp(nll.mean()):.3f}"
    if ref is not None:
        s += f" (fp32 {np.exp(ref['nll'][0, :P - 1].mean()):.3f}), top-1 vs fp32 {100 * (argmax == ref['argmax'][0, :P]).mean():.1f}%"
    print(s)
    np.savez(os.path.join(b, "longctx_sim.npz" if args.sim else "longctx_board.npz"), prompt=prompt, nll=nll,
             argmax=argmax, tokens=np.array(tok1, np.int64), seconds=np.array([t1 - t0, t2 - t1]),
             stdout=np.array([r1[1], r2[1]]))
    ok = exact == n == G + 1 and same
    print("long context PASS" if ok else "long context FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

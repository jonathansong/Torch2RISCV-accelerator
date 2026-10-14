#!/usr/bin/env python3
"""Teacher forcing on the board (docs/generality_implementation_plan.md §3.2
layer 5, §13.1): the windows of tf_tokens.npy [W, L] through the module's
prefill_all (sa-llm-run --prefill_all_out, the logits of every position), and
per position what compiler/tests/board_quality.py scores against the fp32
reference: the argmax, the log-probability of the next token, and the
log-probabilities at the reference's top-K ids (tf_ref_topk.npy [W, L, K]:
the KL over the reference's top K). Writes tf_board.npz. Same files as
board_llm.py, plus those two (board_quality.py prep).

--decode: the windows token by token through decode (sa-llm-run without
--prefill, --logits_out: every step's logits), for a module without
prefill_all (Llama-3.2-1B: its 1.2 GB of parameters leave no room in the
window for it; chosen automatically when the module has no prefill_all). Decode's logits are bit-exact with prefill_all's (the
prefill_all checks of test_hf_generic.py), so the scores are the same; one
step per position (Llama: 256 x 0.25 s per window). Also the default when
sa_args.txt has no --prefill.

The logits go through a FIFO (run_streamed) and are scored row block by row
block as they arrive: a window's logits are never held whole (Qwen3 at a
2048-token window: 2048 x 151936 x 4 = 1.2 GB; iree_compiler_plan.md §8.22).

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh; source /etc/profile.d/xrt_setup.sh 2>/dev/null; \\
        cd <bundle> && python3 board_teacher.py [--decode]'
"""
import os
import select
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def log_softmax_rows(x):
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def run_args(extra, L, decode):
    """sa-llm-run's flags for one window (prefill_all, or decode step by step)
    and the flag that names the logits' file"""
    if not decode:
        return extra, "--prefill_all_out"
    args = [a for a in extra if not a.startswith("--prefill=")]
    if L > 256:                                       # (sa-llm-run's default valid_len: the steps it runs)
        args.append(f"--valid_len={L}")
    return args, "--logits_out"


def run_streamed(cmd, out_flag, V, on_rows, env=None):
    """Runs sa-llm-run with its logits (out_flag: --logits_out / --prefill_all_out)
    written to a FIFO, handing the complete rows to on_rows(first row, [n, V]) as
    they arrive. Returns (returncode, stdout, stderr, rows)."""
    d = tempfile.mkdtemp(dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    fifo = os.path.join(d, "logits")
    os.mkfifo(fifo)
    fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)   # (open before the writer: its fopen does not block)
    so, se = tempfile.TemporaryFile("w+"), tempfile.TemporaryFile("w+")
    p = subprocess.Popen(cmd + [f"{out_flag}={fifo}"], stdout=so, stderr=se, text=True, env=env)
    row, buf, n, done = 4 * V, bytearray(), 0, False
    try:
        while True:
            select.select([fd], [], [], 1.0)
            try:
                chunk = os.read(fd, 1 << 22)
            except BlockingIOError:                   # (a writer, nothing new)
                chunk = b""
            if chunk:
                buf += chunk
                k = len(buf) // row
                if k:
                    on_rows(n, np.frombuffer(bytes(buf[:k * row]), np.float32).reshape(k, V))
                    n += k
                    del buf[:k * row]
            elif done:                                # (empty after the writer exited: all read)
                break
            else:
                done = p.poll() is not None           # (one more read after the exit)
                if not done:
                    time.sleep(0.02)
    finally:
        os.close(fd)
        os.remove(fifo)
        os.rmdir(d)
        if p.poll() is None:
            p.kill()
        p.wait()
    so.seek(0)
    se.seek(0)
    return p.returncode, so.read(), se.read(), n


def score_rows(win, topk, argmax, nll, lp_k):
    """on_rows for a teacher-forcing window: per position the argmax, the
    next token's -log p and the log-probabilities at the reference's top-K ids."""
    L = len(win)

    def on_rows(i, lg):
        for b in range(0, len(lg), 32):              # (row blocks: the board's memory)
            x, j = lg[b:b + 32], i + b
            lp = log_softmax_rows(x)
            argmax[j:j + len(x)] = x.argmax(1)
            lp_k[j:j + len(x)] = np.take_along_axis(lp, topk[j:j + len(x)].astype(np.int64), 1)
            for r in range(len(x)):
                if j + r + 1 < L:
                    nll[j + r] = -lp[r, int(win[j + r + 1])]
    return on_rows


def main():
    import board_launcher as BL      # (here: the host imports run_args without the board's modules)
    wins = np.load(os.path.join(HERE, "tf_tokens.npy"))
    topk = np.load(os.path.join(HERE, "tf_ref_topk.npy"))
    W, L = wins.shape
    mb = int(open(os.path.join(HERE, "board.txt")).read().split()[-1])
    extra = open(os.path.join(HERE, "sa_args.txt")).read().split()
    decode = "--decode" in sys.argv[1:] or not any(a.startswith("--prefill=") for a in extra)
    V = np.load(os.path.join(HERE, "expected_logits.npy"), mmap_mode="r").shape[1]
    print(f"teacher forcing through {'decode, step by step' if decode else 'prefill_all'}", flush=True)
    argmax = np.zeros((W, L), np.int32)
    nll = np.zeros((W, L - 1), np.float32)
    lp_k = np.zeros(topk.shape, np.float32)
    secs = []
    mm, buf, env = BL.start(os.path.join(HERE, "picorv32.bit"), os.path.join(HERE, "rt_fw.bin"), mb=mb, ring=16)
    try:
        for w in range(W):
            base = [os.path.join(HERE, "sa-llm-run"), "--device=sa", f"--module={HERE}/sa.vmfb",
                    f"--parameters=model={HERE}/sa_packed.irpa", "--tokens=" + ",".join(str(int(t)) for t in wins[w]),
                    "--generate=0"]
            t0 = time.time()
            args, flag = run_args(extra, L, decode)
            rc, out, err, n = run_streamed(base + args, flag, V, score_rows(
                wins[w], topk[w], argmax[w], nll[w], lp_k[w]), env=dict(env, SA_STATS="1"))
            if rc and not decode and "no function `prefill_all`" in err:
                # a prefill bundle without prefill_all (Llama-3.2-1B): decode step by step
                print("the module has no prefill_all: teacher forcing through decode, step by step", flush=True)
                decode = True
                args, flag = run_args(extra, L, decode)
                t0 = time.time()
                rc, out, err, n = run_streamed(base + args, flag, V, score_rows(
                    wins[w], topk[w], argmax[w], nll[w], lp_k[w]), env=dict(env, SA_STATS="1"))
            if rc or n != L:
                print(f"window {w}: sa-llm-run failed ({rc}, {n}/{L} rows)\n{out[-2000:]}\n{err[-3000:]}")
                return 1
            pl = [l for l in out.splitlines() if l.startswith("prefill") or "per step" in l]
            secs.append(time.time() - t0)
            host = [l for l in err.splitlines() if l.startswith("sa:")]
            print(f"window {w + 1}/{W}: {pl[0] if pl else ''}; {host[-1] if host else ''}; "
                  f"PPL so far {np.exp(nll[:w + 1].mean()):.3f}", flush=True)
    finally:
        BL.stop(mm, buf)
    np.savez(os.path.join(HERE, "tf_board.npz"), argmax=argmax, nll=nll, lp_topk=lp_k, seconds=np.array(secs))
    print(f"teacher forcing: {W} windows of {L}, PPL {np.exp(nll.mean()):.3f} -> tf_board.npz "
          f"(board_quality.py score)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

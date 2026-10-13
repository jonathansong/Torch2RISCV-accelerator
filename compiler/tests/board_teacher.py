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
window for it). Decode's logits are bit-exact with prefill_all's (the
prefill_all checks of test_hf_generic.py), so the scores are the same; one
step per position (Llama: 256 x 0.25 s per window). Also the default when
sa_args.txt has no --prefill.

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh; source /etc/profile.d/xrt_setup.sh 2>/dev/null; \\
        cd <bundle> && python3 board_teacher.py [--decode]'
"""
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def log_softmax_rows(x):
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def run_args(extra, L, decode, out):
    """sa-llm-run's flags for one window: prefill_all, or decode step by step"""
    if not decode:
        return [f"--prefill_all_out={out}"] + extra
    args = [a for a in extra if not a.startswith("--prefill=")] + [f"--logits_out={out}"]
    if L > 256:                                       # (sa-llm-run's default valid_len: the steps it runs)
        args.append(f"--valid_len={L}")
    return args


def main():
    import board_launcher as BL      # (here: the host imports run_args without the board's modules)
    wins = np.load(os.path.join(HERE, "tf_tokens.npy"))
    topk = np.load(os.path.join(HERE, "tf_ref_topk.npy"))
    W, L = wins.shape
    mb = int(open(os.path.join(HERE, "board.txt")).read().split()[-1])
    extra = open(os.path.join(HERE, "sa_args.txt")).read().split()
    decode = "--decode" in sys.argv[1:] or not any(a.startswith("--prefill=") for a in extra)
    # the logits of a window (Llama: 256 x 128256 x 4 = 131 MB): in RAM when /dev/shm exists (the SD card is small)
    out = os.path.join("/dev/shm" if os.path.isdir("/dev/shm") else HERE, f"tf_logits_{os.getpid()}.f32")
    print(f"teacher forcing through {'decode, step by step' if decode else 'prefill_all'}", flush=True)
    argmax = np.zeros((W, L), np.int32)
    nll = np.zeros((W, L - 1), np.float32)
    lp_k = np.zeros(topk.shape, np.float32)
    secs = []
    mm, buf, env = BL.start(os.path.join(HERE, "picorv32.bit"), os.path.join(HERE, "rt_fw.bin"), mb=mb, ring=16)
    try:
        for w in range(W):
            cmd = [os.path.join(HERE, "sa-llm-run"), "--device=sa", f"--module={HERE}/sa.vmfb",
                   f"--parameters=model={HERE}/sa_packed.irpa", "--tokens=" + ",".join(str(int(t)) for t in wins[w]),
                   "--generate=0"] + run_args(extra, L, decode, out)
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, env=dict(env, SA_STATS="1"))
            if r.returncode:
                print(f"window {w}: sa-llm-run failed ({r.returncode})\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
                return 1
            pl = [l for l in r.stdout.splitlines() if l.startswith("prefill") or "per step" in l]
            secs.append(time.time() - t0)
            lg = np.memmap(out, np.float32, "r").reshape(L, -1)  # (read in row blocks: the board's memory)
            for i in range(0, L, 32):                     # (row blocks: the board's memory)
                lp = log_softmax_rows(lg[i:i + 32])
                argmax[w, i:i + 32] = lg[i:i + 32].argmax(1)
                lp_k[w, i:i + 32] = np.take_along_axis(lp, topk[w, i:i + 32].astype(np.int64), 1)
                for j in range(lp.shape[0]):
                    if i + j + 1 < L:
                        nll[w, i + j] = -lp[j, int(wins[w, i + j + 1])]
            host = [l for l in r.stderr.splitlines() if l.startswith("sa:")]
            print(f"window {w + 1}/{W}: {pl[0] if pl else ''}; {host[-1] if host else ''}; "
                  f"PPL so far {np.exp(nll[:w + 1].mean()):.3f}", flush=True)
            del lg
    finally:
        BL.stop(mm, buf)
        if os.path.exists(out):
            os.remove(out)
    np.savez(os.path.join(HERE, "tf_board.npz"), argmax=argmax, nll=nll, lp_topk=lp_k, seconds=np.array(secs))
    print(f"teacher forcing: {W} windows of {L}, PPL {np.exp(nll.mean()):.3f} -> tf_board.npz "
          f"(board_quality.py score)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

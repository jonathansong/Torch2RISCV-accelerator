#!/usr/bin/env python3
"""Quality on the device (docs/generality_implementation_plan.md §3.2 layer 5):
the host side of board_teacher.py.

  prep:  the WikiText-2 test windows of quality_eval.py (same --tokens /
         --window / --skip) as tf_tokens.npy, and the fp32 reference's
         per-position argmax, next-token log-probability and top-K ids /
         log-probabilities (tf_ref.npz; tf_ref_topk.npy goes to the bundle)
  score: tf_board.npz (from the board, or from the sim: sim) against
         tf_ref.npz: PPL, top-1 agreement with fp32, KL(fp32 || device) over
         the reference's top K (with the reference mass it covers)
  sim:   board_teacher.py's measurement on the functional simulator (the same
         sa-llm-run, x86), for a check before the board

    compiler/tests/board_quality.py prep --model build/llm_cache/Qwen3-0.6B --bundle <dir> [--tokens 2048 --window 256 --k 64]
    compiler/tests/board_quality.py sim --bundle <dir> [--windows N]      (on the host: the sim)
    compiler/tests/board_quality.py score --bundle <dir> [--json out.json]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def log_softmax_rows(x):                 # (board_teacher.py's; that one imports the board launcher)
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def prep(args):
    import torch
    sys.path.insert(0, os.path.join(HERE, "..", "frontend"))
    import hf_generic as H
    from quality_eval import wikitext_tokens
    torch.set_grad_enabled(False)
    ids = wikitext_tokens(args.model, args.tokens, args.skip)
    wins = np.array([ids[i:i + args.window] for i in range(0, len(ids) - args.window + 1, args.window)], np.int64)
    m = H.hf_model(args.model, args.layers)
    W, L = wins.shape
    argmax = np.zeros((W, L), np.int32)
    nll = np.zeros((W, L - 1), np.float32)
    top_ids = np.zeros((W, L, args.k), np.int32)
    top_lp = np.zeros((W, L, args.k), np.float32)
    for w in range(W):
        lg = m(torch.tensor(wins[w:w + 1])).logits[0].float()
        lp = torch.log_softmax(lg, -1)
        argmax[w] = lg.argmax(-1).numpy()
        nll[w] = (-lp[:-1].gather(1, torch.tensor(wins[w, 1:])[:, None]))[:, 0].numpy()
        v, i = lp.topk(args.k, -1)
        top_ids[w], top_lp[w] = i.numpy(), v.numpy()
    np.save(os.path.join(args.bundle, "tf_tokens.npy"), wins)
    np.save(os.path.join(args.bundle, "tf_ref_topk.npy"), top_ids)
    np.savez(os.path.join(args.bundle, "tf_ref.npz"), argmax=argmax, nll=nll, top_ids=top_ids, top_lp=top_lp,
             model=args.model, tokens=W * L, window=L, skip=args.skip)
    print(f"{W} windows of {L} tokens; fp32 PPL {math.exp(nll.mean()):.3f}; top-{args.k} covers "
          f"{100 * np.exp(top_lp).sum(-1).mean():.1f}% of the reference's mass -> {args.bundle}/tf_tokens.npy, "
          f"tf_ref_topk.npy, tf_ref.npz")
    return 0


def sim(args):
    """board_teacher.py's loop on the functional simulator (sa-llm-run, x86)."""
    sys.path.insert(0, os.path.join(HERE, "..", "frontend"))
    b = args.bundle
    wins = np.load(os.path.join(b, "tf_tokens.npy"))[:args.windows]
    topk = np.load(os.path.join(b, "tf_ref_topk.npy"))[:args.windows]
    W, L = wins.shape
    mb = int(open(os.path.join(b, "board.txt")).read().split()[-1])
    extra = open(os.path.join(b, "sa_args.txt")).read().split()
    d = os.environ.get("SA_D", "16")
    repo = os.path.abspath(os.path.join(HERE, "..", ".."))
    run = os.path.join(repo, "build/iree/build-sa-host/runtime/plugins/hal/drivers/sa/sa-llm-run")
    argmax = np.zeros((W, L), np.int32)
    nll = np.zeros((W, L - 1), np.float32)
    lp_k = np.zeros(topk.shape, np.float32)
    out = os.path.join(b, "logits_all_sim.f32")
    for w in range(W):
        sock = f"/tmp/sa_bq_{os.getpid()}.sock"
        srv = subprocess.Popen([sys.executable, os.path.join(HERE, "..", "sim", "sa_sim_server.py"), "--d", d,
                                "--mb", str(mb), "--socket", sock, "--once"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        time.sleep(3)
        r = subprocess.run([run, "--device=sa", f"--module={b}/sa.vmfb", f"--parameters=model={b}/sa_packed.irpa",
                            "--tokens=" + ",".join(map(str, wins[w])), "--generate=0", f"--prefill_all_out={out}"]
                           + extra, capture_output=True, text=True, env=dict(os.environ, SA_SIM_SOCKET=sock))
        srv.wait(timeout=120)
        if r.returncode:
            print(r.stderr[-3000:])
            return 1
        lg = np.fromfile(out, np.float32).reshape(L, -1)
        lp = log_softmax_rows(lg)
        argmax[w] = lg.argmax(1)
        lp_k[w] = np.take_along_axis(lp, topk[w].astype(np.int64), 1)
        nll[w] = -lp[np.arange(L - 1), wins[w, 1:]]
        print(f"window {w + 1}/{W} (sim)", flush=True)
    os.remove(out)
    np.savez(os.path.join(b, "tf_sim.npz"), argmax=argmax, nll=nll, lp_topk=lp_k)
    print(f"-> {b}/tf_sim.npz (score --which sim)")
    return 0


def score(args):
    ref = np.load(os.path.join(args.bundle, "tf_ref.npz"))
    dev = np.load(os.path.join(args.bundle, f"tf_{args.which}.npz"))
    W = dev["argmax"].shape[0]
    ra, rn, rlp = ref["argmax"][:W], ref["nll"][:W], ref["top_lp"][:W]
    p = np.exp(rlp)
    kl = float((p * (rlp - dev["lp_topk"])).sum(-1).mean())
    res = dict(windows=W, window=int(ref["window"]), positions=int(ra.size),
               ppl_fp32=math.exp(float(rn.mean())), ppl=math.exp(float(dev["nll"].mean())),
               top1=float((dev["argmax"] == ra).mean()), kl_topk=kl, topk_mass=float(p.sum(-1).mean()),
               k=int(rlp.shape[-1]))
    if "seconds" in dev:
        res["seconds_per_window"] = float(np.mean(dev["seconds"]))
    print(f"{args.which}: {W} windows x {res['window']}: PPL {res['ppl']:.3f} (fp32 {res['ppl_fp32']:.3f}), top-1 vs fp32 "
          f"{100 * res['top1']:.1f}%, KL(fp32 || device) over the top {res['k']} {kl:.4f} "
          f"({100 * res['topk_mass']:.1f}% of the mass)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["prep", "sim", "score"])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--model", help="prep: the original HF model (the fp32 reference)")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--layers", type=int, help="prep: a truncated reference (a --layers test model)")
    ap.add_argument("--windows", type=int, default=None, help="sim: only the first windows")
    ap.add_argument("--which", default="board", help="score: board / sim")
    ap.add_argument("--json")
    args = ap.parse_args()
    return {"prep": prep, "sim": sim, "score": score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

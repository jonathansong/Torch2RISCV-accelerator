#!/usr/bin/env python3
"""S6 of docs/moe_native_lowering_design.md: a W8A8 Mixtral bundle (exported by
test_hf_moe.py --model --export, compiled by compile_sa.sh with
--iree-sa-form-moe --iree-sa-emit-riscv=<bundle>/rv_ll, its host build by
rv_image.py into <bundle>/rv) end to end on the sim (sa-llm-run against
compiler/sim/sa_sim_server.py):

  A  decode of the prompt, then --generate steps (descriptor lists; the Top-2
     as RISC-V code: SA_RV_ONLY=rv_only)
  B  the same prompt through prefill (--prefill M): its last position and the
     generated steps byte-equal to A's
  C  plan A, every export as RISC-V code (SA_RV, no SA_RV_ONLY): byte-equal to A
  ref the host model (test_hf_moe.sa_decoder: SaDecoder with QMixtralMoe), its
     KV scales calibrated on the export's tokens, teacher-forced on A's tokens:
     argmax per step and correlation (rounding order differs, so not bit-exact)

    python3 compiler/tests/test_moe_sim.py --bundle DIR --model DIR [--prompt ...] [--generate 4]
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
RUN = os.path.join(REPO, "build", "iree", "build-sa-host", "runtime", "plugins", "hal", "drivers", "sa", "sa-llm-run")


def sim_run(bundle, tokens, out, env_rv, mb, extra=(), d=16, valid_len=64):
    """One sa-llm-run against a fresh sim server: the logits rows written."""
    sock = f"/tmp/sa_moe_{os.getpid()}_{os.path.basename(out)}.sock"
    srv = subprocess.Popen([sys.executable, os.path.join(REPO, "compiler", "sim", "sa_sim_server.py"), "--d", str(d),
                            "--mb", str(mb), "--socket", sock, "--once"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    env = dict(os.environ, SA_SIM_SOCKET=sock, SA_STATS="1", **env_rv)
    t0 = time.time()
    r = subprocess.run([RUN, "--device=sa", f"--module={bundle}/sa.vmfb", f"--parameters=model={bundle}/sa_packed.irpa",
                        "--abi=hfv", f"--pad={d}", f"--valid_len={valid_len}", "--tokens=" + ",".join(map(str, tokens)),
                        f"--logits_out={out}", *extra], capture_output=True, text=True, env=env)
    if r.returncode:
        srv.kill()
    srv.wait(timeout=120)
    if r.returncode:
        sys.exit(f"sa-llm-run failed:\n{r.stderr[-3000:]}")
    stats = [l for l in r.stderr.splitlines() if l.startswith("sa:")]
    print(f"  {os.path.basename(out)}: {time.time() - t0:.0f} s; {stats[-1] if stats else ''}")
    return np.fromfile(out, np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--model", required=True, help="the checkpoint the bundle was exported from")
    ap.add_argument("--calib", default="1,42,43", help="the export's KV calibration tokens (test_hf_moe --tokens)")
    ap.add_argument("--prompt", default="1,415,2936,9060,285,1142,28705,28740,28723,13,1014,3075,349,264,1179,1170,"
                    "302,11843,28713,304,1413,272,1080,5853")
    ap.add_argument("--generate", type=int, default=4)
    ap.add_argument("--prefill", type=int, default=16)
    ap.add_argument("--mb", type=int, default=320)
    ap.add_argument("--min-corr", type=float, default=0.99)
    args = ap.parse_args()
    prompt = [int(t) for t in args.prompt.split(",")]
    rv = os.path.join(args.bundle, "rv")
    out = os.path.join(args.bundle, "s6")
    os.makedirs(out, exist_ok=True)
    P, G = len(prompt), args.generate

    print("sim runs:")
    a = sim_run(args.bundle, prompt, f"{out}/A_decode.f32", {"SA_RV": rv, "SA_RV_ONLY": "rv_only"}, args.mb,
                [f"--generate={G}"])
    V = a.size // (P + G)
    a = a.reshape(P + G, V)
    b = sim_run(args.bundle, prompt, f"{out}/B_prefill.f32", {"SA_RV": rv, "SA_RV_ONLY": "rv_only"}, args.mb,
                [f"--generate={G}", f"--prefill={args.prefill}"]).reshape(-1, V)
    c = sim_run(args.bundle, prompt, f"{out}/C_planA.f32", {"SA_RV": rv}, args.mb,
                [f"--generate={G}"]).reshape(-1, V)
    toks = prompt + [int(x) for x in a[P - 1:P + G - 1].argmax(1)]

    print("host model:")
    import test_hf_moe as HM
    from transformers import AutoModelForCausalLM
    torch.set_grad_enabled(False)
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa").eval()
    w = HM.sa_decoder(m, 64, [int(t) for t in args.calib.split(",")])
    ref = np.stack([w(torch.tensor([[t]]), torch.tensor([[p]]))[0, -1].numpy() for p, t in enumerate(toks)])

    ok = True
    b_same = b.shape == (1 + G, V) and np.array_equal(b.view(np.uint32), a[P - 1:].view(np.uint32))
    c_same = c.shape == a.shape and np.array_equal(c.view(np.uint32), a.view(np.uint32))
    corr = [float(np.corrcoef(a[i], ref[i])[0, 1]) for i in range(P + G)]
    same = [int(a[i].argmax()) == int(ref[i].argmax()) for i in range(P + G)]
    print(f"B prefill {args.prefill} vs A decode, last prompt position + {G} steps: "
          f"{'byte-equal' if b_same else 'DIFFERENT'}")
    print(f"C plan A vs A descriptors, {P + G} steps: {'byte-equal' if c_same else 'DIFFERENT'}")
    print(f"A vs the host model, {P + G} steps: argmax equal in {sum(same)}/{len(same)}, corr min {min(corr):.5f} "
          f"mean {np.mean(corr):.5f}")
    for i, (s, cr) in enumerate(zip(same, corr)):
        if not s or cr < args.min_corr:
            print(f"  step {i}: device top {int(a[i].argmax())}, host {int(ref[i].argmax())}, corr {cr:.5f}")
    ok = b_same and c_same and all(same) and min(corr) > args.min_corr
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

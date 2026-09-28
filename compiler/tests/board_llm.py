#!/usr/bin/env python3
"""C3 / C5.5 on the PYNQ-Z1: a decoder compiled by IREE for the sa device, run by
the armv7 sa-llm-run (the IREE runtime with the sa HAL driver) through the C1
launcher; the logits of every step compared bit for bit with the host reference
(expected_logits.npy: C3 DeviceModel(SfuExact) = the hand-written L5 path, C5.5
the functional simulator), and the generated text printed.

Files in one directory on the board: this script, board_launcher.py,
pynq_matmul.py, picorv32.bit / .hwh (L2), rt_fw.bin, sa-llm-run (armv7),
sa.vmfb, sa_packed.irpa, expected_tokens.npy, expected_logits.npy, prompt.npy,
the tokenizer (tokenizer.bin + tokenizer.py, or tokenizer.json + hf_tokenizer.py),
optional board.txt (the reference's name, the window size in MB) and
sa_args.txt (extra sa-llm-run arguments).

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh && \\
        cd /home/xilinx/c3 && python3 board_llm.py'
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board_launcher as BL  # noqa: E402


def decoder(vocab):
    """ids -> text: a byte-level BPE tokenizer.json (hf_tokenizer.py), or llama2.c tokenizer.bin."""
    if os.path.exists(os.path.join(HERE, "tokenizer.json")):
        from hf_tokenizer import BpeTokenizer
        return BpeTokenizer(HERE).decode
    from tokenizer import Tokenizer
    return Tokenizer(os.path.join(HERE, "tokenizer.bin"), vocab).decode


def main():
    prompt = np.load(os.path.join(HERE, "prompt.npy"))
    want_tokens = np.load(os.path.join(HERE, "expected_tokens.npy"))
    want_logits = np.load(os.path.join(HERE, "expected_logits.npy"))
    n_gen = len(want_tokens) - len(prompt)
    logits_path = os.path.join(HERE, "logits_board.f32")
    ref_name, mb = "DeviceModel(SfuExact)", 64
    if os.path.exists(os.path.join(HERE, "board.txt")):
        ref_name, mb = open(os.path.join(HERE, "board.txt")).read().rsplit(None, 1)
        mb = int(mb)
    args_file = os.path.join(HERE, "sa_args.txt")
    extra = open(args_file).read().split() if os.path.exists(args_file) else []
    mm, buf, env = BL.start(os.path.join(HERE, "picorv32.bit"), os.path.join(HERE, "rt_fw.bin"), mb=mb, ring=16)
    try:
        cmd = [os.path.join(HERE, "sa-llm-run"), "--device=sa", f"--module={HERE}/sa.vmfb",
               f"--parameters=model={HERE}/sa_packed.irpa", "--tokens=" + ",".join(str(int(t)) for t in prompt),
               f"--generate={n_gen}", f"--logits_out={logits_path}"] + extra
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    finally:
        head, beat = BL.stop(mm, buf)
    if r.returncode:
        print(f"sa-llm-run failed ({r.returncode})\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
        return 1
    lines = r.stdout.strip().splitlines()
    tokens = [int(t) for t in lines[0].split(":")[1].split()]
    got = np.fromfile(logits_path, np.float32).reshape(-1, want_logits.shape[1])
    n = min(len(got), len(want_logits))
    exact = sum(got[i].tobytes() == want_logits[i].tobytes() for i in range(n))
    print(f"sa device (board): {lines[1] if len(lines) > 1 else ''}; rt_fw completed {head} lists")
    print(f"logits bit-exact with {ref_name} in {exact}/{n} steps")
    same = tokens[:len(want_tokens)] == list(want_tokens)
    print(f"generated tokens {'identical to' if same else 'DIFFERENT from'} the host reference")
    print("text:", repr(decoder(want_logits.shape[1])(tokens)))
    ok = exact == n == len(want_logits) and same
    print("board PASS" if ok else "board FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

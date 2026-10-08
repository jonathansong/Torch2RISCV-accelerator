#!/usr/bin/env python3
"""C4 measurement on the board: where the time of one token goes for the
IREE-compiled model (device cycles per export from rt_fw's completion records,
host time of the submissions, the rest of the IREE runtime). Same files as
board_llm.py. One list per dispatch (so each export is timed); --batch keeps the
lists of whole command buffers (as board_llm.py runs) and times those;
NAME=VALUE arguments set environment variables of the runner (A/B switches).

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh && \\
        cd /home/xilinx/c3 && python3 board_profile.py [tokens] [--prompt N] [--batch] [NAME=VALUE...]'

With a prefill bundle (sa_args.txt --prefill=M) and --prompt N (N / M chunks) the prefill
chunks after the first are profiled too.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board_launcher as BL  # noqa: E402


def main():
    prompt = np.load(os.path.join(HERE, "prompt.npy"))
    args = sys.argv[1:]
    if "--prompt" in args:                          # a longer prompt (prefill): prompt.npy repeated
        i = args.index("--prompt")
        p = int(args[i + 1])
        prompt = np.resize(prompt, p)
        del args[i:i + 2]
    argv = [a for a in args if a != "--batch" and "=" not in a]
    n = int(argv[0]) if argv else 40
    args_file = os.path.join(HERE, "sa_args.txt")
    extra = open(args_file).read().split() if os.path.exists(args_file) else []
    mb = int(open(os.path.join(HERE, "board.txt")).read().split()[-1]) if os.path.exists(os.path.join(HERE, "board.txt")) else 64
    mm, buf, env = BL.start(os.path.join(HERE, "picorv32.bit"), os.path.join(HERE, "rt_fw.bin"), mb=mb, ring=16)
    env["SA_PROFILE"] = "batch" if "--batch" in sys.argv else "1"
    for a in sys.argv[1:]:
        if "=" in a:
            env[a.split("=", 1)[0]] = a.split("=", 1)[1]
    try:
        cmd = [os.path.join(HERE, "sa-llm-run"), "--device=sa", f"--module={HERE}/sa.vmfb",
               f"--parameters=model={HERE}/sa_packed.irpa", "--tokens=" + ",".join(str(int(t)) for t in prompt),
               f"--generate={max(0, n - len(prompt))}"] + extra
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    finally:
        BL.stop(mm, buf)
    print(r.stdout[r.stdout.index("\n") + 1:] if r.returncode == 0 else r.stdout + r.stderr)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())

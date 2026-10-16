#!/usr/bin/env python3
"""Plan A of the RISC-V backend on the board (docs/riscv_backend_design.md
§9.4): the bundle's model with every dispatch run as generated RISC-V code.
The driver (SA_RV=rv/) loads the model's code image (rv/rvcode.bin, built by
compiler/tests/rv_image.py from iree-compile --iree-sa-emit-riscv) into the
device window and sends each dispatch as one rt_fw RT_CALL entry: the
PicoRV32 runs the export's function, which issues the commands over PCPI from
DDR; no descriptor lists. The prompt and the reference as board_llm.py's: the
logits of every step bit-exact with the functional simulator's, the tokens
identical.

Files: the hfgen bundle plus rv/rvcode.bin, rv/rvcode.map, rt_fw.bin version
4 and an sa-llm-run with SA_RV (compiler/scripts/deploy_rvcode.sh).

    python3 board_rvcode.py [--decode-only]    (or python3 board_run.py rvcode)
Writes rvcode_board.json.
"""
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board_launcher as BL  # noqa: E402


def main():
    prompt = np.load(os.path.join(HERE, "prompt.npy"))
    want_tokens = [int(t) for t in np.load(os.path.join(HERE, "expected_tokens.npy"))]
    want_logits = np.load(os.path.join(HERE, "expected_logits.npy"))
    n_gen = len(want_tokens) - len(prompt)
    mb = int(open(os.path.join(HERE, "board.txt")).read().split()[-1])
    extra = open(os.path.join(HERE, "sa_args.txt")).read().split()
    if "--decode-only" in sys.argv[1:]:
        extra = [a for a in extra if not a.startswith("--prefill")]
    M = max([int(a.split("=")[1]) for a in extra if a.startswith("--prefill=")] or [0])
    if M and len(prompt) >= M:
        want_logits = want_logits[len(prompt) - 1:]
    lp = os.path.join(HERE, "logits_rvcode.f32")
    mm, buf, env = BL.start(os.path.join(HERE, "picorv32.bit"), os.path.join(HERE, "rt_fw.bin"), mb=mb, ring=16)
    try:
        cmd = [os.path.join(HERE, "sa-llm-run"), "--device=sa", f"--module={HERE}/sa.vmfb",
               f"--parameters=model={HERE}/sa_packed.irpa", "--tokens=" + ",".join(str(int(t)) for t in prompt),
               f"--generate={n_gen}", f"--logits_out={lp}"] + extra
        r = subprocess.run(cmd, capture_output=True, text=True,
                           env=dict(env, SA_RV=os.path.join(HERE, "rv"), SA_STATS="1"))
    finally:
        BL.stop(mm, buf)
    if r.returncode:
        print(f"sa-llm-run (SA_RV) failed ({r.returncode})\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
        return 1
    lines = r.stdout.strip().splitlines()
    tokens = [int(t) for t in lines[0].split(":")[1].split()]
    got = np.fromfile(lp, np.float32).reshape(-1, want_logits.shape[1])
    n = min(len(got), len(want_logits))
    exact = sum(got[i].tobytes() == want_logits[i].tobytes() for i in range(n))
    same = tokens[:len(want_tokens)] == want_tokens
    stats = [l for l in r.stderr.splitlines() if l.startswith(("sa:", "sa rv:"))]
    for l in stats:
        print(l)
    print(f"sa device (board, every dispatch as RISC-V code): {lines[1] if len(lines) > 1 else ''}")
    print(f"logits bit-exact with the functional simulator in {exact}/{n} steps; generated tokens "
          f"{'identical to' if same else 'DIFFERENT from'} the reference")
    calls = [int(m.group(1)) for l in stats if (m := re.search(r"(\d+) dispatches as RISC-V code", l))]
    lists = [int(m.group(1)) for l in stats if (m := re.search(r"(\d+) descriptor lists", l))]
    ok = exact == n == len(want_logits) and same and calls and calls[0] > 0
    json.dump({"exact": exact, "rows": n, "tokens_identical": same, "rv_calls": calls[0] if calls else 0,
               "lists": lists[0] if lists else None, "timing": lines[1] if len(lines) > 1 else "", "stats": stats},
              open(os.path.join(HERE, "rvcode_board.json"), "w"), indent=1)
    print("rvcode PASS" if ok else "rvcode FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

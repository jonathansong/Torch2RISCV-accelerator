#!/usr/bin/env python3
"""The performance-counter breakdown of plan A against the descriptor lists on
the board (docs/riscv_backend_design.md §9.10): the unit's event counters
(rtl/sysarray/sa_perf.v: STARVE, ALL_IDLE, PCPI_QFULL, the engines' busy
cycles, ...) and the core's instructions retired (rt_fw version 7) for

  desc       every dispatch as its descriptor list (one list per dispatch)
  rv         every dispatch as the S4 code embedded in sa.vmfb (RT_CALL each)
  sched      plan B's decode loop on the core, descriptor lists (counters over the run)
  sched_rv   plan B's loop calling the S4 code (counters over the run)

each writing rvperf_<run>.csv (SA_PROFILE_PERF's format; desc and rv: per
export, prefill chunks and decode steps; sched*: one row for the run). Copy
them back and run compiler/tests/rv_perf.py on them.

    python3 board_rvperf.py [--gen 16]          (or python3 board_run.py rvperf)
Files: a bundle with S4 (deploy_rvembed.sh), plan B (deploy_rvsched.sh) and rt_fw.bin version 7.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board_launcher as BL  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gen", type=int, default=16, help="tokens generated per run (the counters are 32-bit: < 40 s)")
    args = ap.parse_args()
    prompt = np.load(os.path.join(HERE, "prompt.npy"))
    mb = int(open(os.path.join(HERE, "board.txt")).read().split()[-1])
    extra = open(os.path.join(HERE, "sa_args.txt")).read().split()
    toks = "--tokens=" + ",".join(str(int(t)) for t in prompt)
    base = [os.path.join(HERE, "sa-llm-run"), "--device=sa", f"--parameters=model={HERE}/sa_packed.irpa", toks,
            f"--generate={args.gen}"] + extra
    runs = {
        "desc": ([f"--module={HERE}/sa.vmfb"], {"SA_PROFILE": "1"}),
        "rv": ([f"--module={HERE}/sa.vmfb"], {"SA_PROFILE": "1", "SA_RV": "embedded"}),
        "sched": ([f"--module={HERE}/argmax.vmfb", f"--module={HERE}/sa.vmfb", "--rvsched"], {}),
        "sched_rv": ([f"--module={HERE}/argmax.vmfb", f"--module={HERE}/sa.vmfb", "--rvsched"], {"SA_RV": "embedded"}),
    }
    mm, buf, env = BL.start(os.path.join(HERE, "picorv32.bit"), os.path.join(HERE, "rt_fw.bin"), mb=mb, ring=16)
    failed = []
    try:
        for name, (mods, e) in runs.items():
            csv = os.path.join(HERE, f"rvperf_{name}.csv")
            if os.path.exists(csv):
                os.remove(csv)
            cmd = base[:2] + mods + base[2:]
            r = subprocess.run(cmd, capture_output=True, text=True, env=dict(env, SA_PROFILE_PERF=csv, **e))
            open(os.path.join(HERE, f"rvperf_{name}.log"), "w").write(r.stdout + "\n" + r.stderr)
            ok = r.returncode == 0 and os.path.exists(csv)
            last = [l for l in r.stdout.splitlines() if "ms per step" in l or "rvsched:" in l]
            print(f"{name:9} {'OK' if ok else 'FAILED'}  {last[-1] if last else ''}")
            if not ok:
                print(r.stderr[-1500:])
                failed.append(name)
    finally:
        BL.stop(mm, buf)
    print(f"rvperf {'PASS' if not failed else 'FAILED (' + ', '.join(failed) + ')'}: rvperf_*.csv, rvperf_*.log in {HERE}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

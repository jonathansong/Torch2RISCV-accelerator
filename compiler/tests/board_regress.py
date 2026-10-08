#!/usr/bin/env python3
"""The board regression against the PYNQ-Z1 freeze baselines (docs/kv260_upgrade_plan.md §7.2):
runs every test directory staged by compiler/scripts/deploy_z1_freeze.sh in
turn and collects the logs, the generated tokens and the profiles into
results/ (copied back to the host for compiler/tests/make_z1_baselines.py).

    sudo ./run_board.sh [test ...] [--no-profile]      (the PYNQ environment of either board)

Tests (a directory each; missing ones are skipped), in the order of the KV260
K1a verification (docs/kv260_upgrade_plan.md):
  ddr          ddr_test.py: the PicoRV32 reads / writes DDR
  bwtest       m2_bw_test.py: DMA bandwidth
  fwdemo       m1_gemm_demo.py, m3_vector_demo.py, m5_desc_demo.py: firmware vs NumPy
  c1           board_launcher.py -- ./sa_hal_test test_d<D> 20: the sa HAL driver (C1)
  c3           board_llm.py: stories15M decode, bit-exact with DeviceModel
  c55          board_llm.py: SmolLM2-135M decode (qhf), bit-exact with the sim
  c6p_stories  board_llm.py: stories15M prefill + decode; board_profile.py (+ event counters: *_perf.csv)
  c6p_smollm2  board_llm.py: SmolLM2-135M prefill + decode (qhf); board_profile.py
  hfgen        board_llm.py: SmolLM2-135M prefill + decode (generic HF); board_profile.py
"""
import glob
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# (test, run name, script, arguments, required: a failure fails the regression)
RUNS = [
    ("ddr", "ddr", "ddr_test.py", [], True),
    ("bwtest", "bwtest", "m2_bw_test.py", [], True),
    ("fwdemo", "gemm", "m1_gemm_demo.py", [], True),
    ("fwdemo", "vector", "m3_vector_demo.py", [], True),
    ("fwdemo", "desc", "m5_desc_demo.py", [], True),
    ("c1", "c1", "board_launcher.py", ["--", "./sa_hal_test", "test_d*", "20"], True),
    ("c3", "c3", "board_llm.py", [], True),
    ("c55", "c55", "board_llm.py", [], True),
    ("c6p_stories", "c6p_stories", "board_llm.py", [], True),
    ("c6p_stories", "c6p_stories_profile", "board_profile.py", [f"SA_PROFILE_PERF={RES}/c6p_stories_perf.csv"], False),
    ("c6p_smollm2", "c6p_smollm2", "board_llm.py", [], True),
    ("c6p_smollm2", "c6p_smollm2_profile", "board_profile.py", [f"SA_PROFILE_PERF={RES}/c6p_smollm2_perf.csv"], False),
    ("hfgen", "hfgen", "board_llm.py", [], True),
    ("hfgen", "hfgen_profile", "board_profile.py", [f"SA_PROFILE_PERF={RES}/hfgen_perf.csv"], False),
]


def run(test, name, script, args):
    d = os.path.join(HERE, test)
    # test_d*: the C1 executables made for the overlay's D (deploy_c1.sh)
    args = [next(iter(sorted(glob.glob(os.path.join(d, a)))), a).replace(d + os.sep, "") if "*" in a else a
            for a in args]
    t0 = time.time()
    r = subprocess.run([sys.executable, script, *args], cwd=d, capture_output=True, text=True)
    dt = time.time() - t0
    out = r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr.strip() else "")
    with open(os.path.join(RES, name + ".log"), "w") as f:
        f.write(out)
    tok = os.path.join(d, "tokens_board.txt")
    if script == "board_llm.py" and os.path.exists(tok):
        shutil.copy(tok, os.path.join(RES, name + "_tokens.txt"))
    last = [l for l in r.stdout.strip().splitlines() if l.strip()][-1:] or ["(no output)"]
    return r.returncode == 0, dt, last[0]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    profile = "--no-profile" not in sys.argv[1:]
    os.makedirs(RES, exist_ok=True)
    lines = [f"board {platform.node()} {platform.machine()} {time.strftime('%Y-%m-%d %H:%M:%S')}"]
    if os.path.exists(os.path.join(HERE, "VERSION")):
        lines += open(os.path.join(HERE, "VERSION")).read().strip().splitlines()
    ok_all = True
    for test, name, script, a, required in RUNS:
        if args and test not in args:
            continue
        if not os.path.isdir(os.path.join(HERE, test)):
            print(f"{name:22s} skipped (no {test}/)")
            continue
        if script == "board_profile.py" and not profile:
            continue
        print(f"{name:22s} ...", end="", flush=True)
        ok, dt, last = run(test, name, script, a)
        status = "PASS" if ok else ("FAIL" if required else "fail (not required)")
        ok_all &= ok or not required
        line = f"{name:22s} {status:5s} {dt:7.1f} s  {last}"
        print("\r" + line)
        lines.append(line)
    lines.append("REGRESSION PASS" if ok_all else "REGRESSION FAILED")
    print(lines[-1])
    with open(os.path.join(RES, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

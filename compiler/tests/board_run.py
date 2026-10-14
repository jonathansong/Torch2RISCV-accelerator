#!/usr/bin/env python3
"""Runs one board script of a staged bundle and files what it produced under
the uniform results directory (docs/generality_implementation_plan.md WP0):

    results/<commit>/<test>/<overlay configuration>/<mode>/

from the bundle's VERSION (compiler/scripts/board_env.sh stage_version): the
commit (+ "-dirty"), the test (the deploy directory's name), the overlay. Each
run directory holds run.log (stdout + stderr), VERSION, sa_args.txt / board.txt
(the run parameters), board_env.txt (kernel, device tree, u-dma-buf, PYNQ,
rt_fw.bin hash, time) and the script's own files (tokens_board.txt,
tf_board.npz, the profile's perf CSV). A later run of the same mode goes to
<mode>.2, <mode>.3, ... Copy the tree back with
    rsync -a ubuntu@<board>:<bundle>/results/ build/results/

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh; source /etc/profile.d/xrt_setup.sh 2>/dev/null; \\
        cd ~ubuntu/<bundle> && python3 board_run.py <mode> [script arguments]'

Modes: llm (board_llm.py: prefill + decode, bit-exact with the sim), teacher
(board_teacher.py), profile (board_profile.py; --batch: the batched lists, the
mode then is profile_batch), generate (board_generate.py), longctx
(board_longctx.py: a prompt of the context length, prefill vs decode only).
"""
import glob
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = {"llm": "board_llm.py", "teacher": "board_teacher.py", "profile": "board_profile.py",
           "generate": "board_generate.py", "longctx": "board_longctx.py"}
OUTPUTS = {"llm": ["tokens_board.txt"], "teacher": ["tf_board.npz"], "longctx": ["longctx_board.npz", "longctx_sweep_board.json"]}


def version():
    v = {}
    path = os.path.join(HERE, "VERSION")
    if os.path.exists(path):
        for line in open(path):
            k, _, rest = line.strip().partition(" ")
            v.setdefault(k, rest)
    return v


def run_dir(mode, v):
    commit = v.get("commit", "unknown")
    commit = commit.split()[0][:7] + ("-dirty" if "(dirty)" in commit else "")
    test = v.get("test", os.path.basename(HERE))
    overlay = v.get("overlay", "").split()[1:2]          # "overlay config <name> (...)"
    overlay = overlay[0] if overlay else "unknown"
    base = os.path.join(HERE, "results", commit, test, overlay, mode)
    d, n = base, 1
    while os.path.exists(d):
        n += 1
        d = f"{base}.{n}"
    os.makedirs(d)
    return d


def read(path):
    try:
        with open(path, "rb") as f:
            return f.read().rstrip(b"\0").decode(errors="replace").strip()
    except OSError:
        return "?"


def board_env():
    lines = [f"time {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
             f"host {platform.node()} {platform.machine()} {platform.release()}",
             f"device tree {read('/proc/device-tree/model')}"]
    for d in sorted(glob.glob("/sys/class/u-dma-buf/*")):
        lines.append(f"u-dma-buf {os.path.basename(d)} size {read(d + '/size')} phys {read(d + '/phys_addr')}")
    try:
        import pynq
        lines.append(f"pynq {pynq.__version__}")
    except Exception:                                     # noqa: BLE001 (not in the PYNQ environment)
        lines.append("pynq ?")
    fw = os.path.join(HERE, "rt_fw.bin")
    if os.path.exists(fw):
        lines.append(f"rt_fw.bin {hashlib.sha256(open(fw, 'rb').read()).hexdigest()[:16]}")
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SCRIPTS:
        print(__doc__)
        return 2
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "profile" and "--batch" in args:
        mode = "profile_batch"
    script = SCRIPTS[mode.split("_")[0]]
    d = run_dir(mode, version())
    if mode.startswith("profile") and not any(a.startswith("SA_PROFILE_PERF=") for a in args):
        args = args + [f"SA_PROFILE_PERF={d}/perf.csv"]   # (the event counters)
    t0 = time.time()
    with open(os.path.join(d, "run.log"), "w") as log:
        log.write(f"$ python3 {script} {' '.join(args)}\n")
        log.flush()
        p = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, script), *args], cwd=HERE,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:                             # (tee: progress on the terminal)
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        rc = p.wait()
        log.write(f"exit {rc}, {time.time() - t0:.1f} s\n")
    with open(os.path.join(d, "board_env.txt"), "w") as f:
        f.write(board_env())
    for name in ["VERSION", "sa_args.txt", "board.txt"] + OUTPUTS.get(mode, []):
        src = os.path.join(HERE, name)
        if os.path.exists(src) and (name not in OUTPUTS.get(mode, []) or os.path.getmtime(src) >= t0):
            shutil.copy(src, d)                           # (outputs: only this run's)
    print(f"-> {os.path.relpath(d, HERE)} (exit {rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())

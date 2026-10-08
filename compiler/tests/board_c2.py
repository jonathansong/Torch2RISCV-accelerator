#!/usr/bin/env python3
"""C2 on the board: the IREE-compiled int8 linear layers (compiler/tests/test_c2.py
--board-bundle) run by the aarch64 iree-run-module on the sa device, each
result compared bit for bit with DeviceModel.linear (y_ref.npy).

Files in one directory on the board: this script, board_launcher.py,
pynq_matmul.py, picorv32.bit / .hwh (the overlay), rt_fw.bin, iree-run-module (aarch64,
with the sa driver), and one directory per case (model.vmfb, packed.irpa,
x.npy, s_x.npy, y_ref.npy).

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh && \\
        cd /home/xilinx/c2 && python3 board_c2.py'
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board_launcher as BL  # noqa: E402


def main():
    cases = sorted(d for d in os.listdir(HERE) if os.path.isfile(os.path.join(HERE, d, "model.vmfb")))
    mm, buf, env = BL.start(os.path.join(HERE, "picorv32.bit"), os.path.join(HERE, "rt_fw.bin"), mb=32)
    fails = 0
    try:
        for c in cases:
            p = os.path.join(HERE, c)
            y = os.path.join(p, "y.npy")
            if os.path.exists(y):
                os.remove(y)
            cmd = [os.path.join(HERE, "iree-run-module"), "--device=sa", f"--module={p}/model.vmfb",
                   f"--parameters=model={p}/packed.irpa", "--function=main",
                   f"--input=@{p}/x.npy", f"--input=@{p}/s_x.npy", f"--output=@{y}"]
            r = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if r.returncode:
                print(f"{c}: iree-run-module failed\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
                fails += 1
                continue
            got, ref = np.load(y), np.load(os.path.join(p, "y_ref.npy"))
            same = got.shape == ref.shape and got.tobytes() == ref.tobytes()
            print(f"{c}: y {got.shape} " + ("bit-exact with DeviceModel.linear" if same else
                                              f"DIFFERENT (max diff {np.abs(got - ref).max():g})"))
            fails += not same
    finally:
        head, beat = BL.stop(mm, buf)
    print(f"rt_fw completed {head} entries")
    print("C2 board PASS" if fails == 0 else f"C2 board FAILED ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

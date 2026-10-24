#!/usr/bin/env python3
"""Execute the Torch-exported MoE layer with SA simulator and host fallback."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "build/moe/m5h/board"
RUNNER = REPO / "build/iree/build-sa-host/tools/iree-run-module"
SERVER = REPO / "compiler/sim/sa_sim_server.py"


class NativeMoeSimTest(unittest.TestCase):
    def test_layer(self):
        if not (BUNDLE / "moe_layer.vmfb").is_file():
            self.skipTest("stage compiler/tests/moe_m5h_board_bundle.py first")
        if not RUNNER.is_file():
            self.skipTest("build the host SA runtime with compiler/scripts/build_sa_runtime.sh host")
        with tempfile.TemporaryDirectory() as directory:
            socket = str(Path(directory) / "sa.sock")
            server = subprocess.Popen([sys.executable, str(SERVER), "--d", "16",
                                       "--socket", socket], stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE)
            try:
                for _ in range(100):
                    if Path(socket).exists():
                        break
                    if server.poll() is not None:
                        self.fail(server.stderr.read().decode(errors="replace")[-2000:])
                    time.sleep(0.1)
                else:
                    self.fail("SA simulator did not open its socket")
                env = dict(os.environ, SA_TRANSPORT="sim", SA_SIM_SOCKET=socket,
                           SA_RV="embedded")
                for name in ("ramp", "zero", "random"):
                    with self.subTest(case=name):
                        path = BUNDLE / name
                        outputs = [str(Path(directory) / f"{name}_{i}.npy") for i in range(4)]
                        command = [str(RUNNER), "--device=sa",
                                   f"--module={BUNDLE / 'moe_layer.vmfb'}",
                                   f"--parameters=model={BUNDLE / 'moe_layer.irpa'}",
                                   "--function=main", f"--input=@{path / 'x.npy'}",
                                   *(f"--output=@{item}" for item in outputs)]
                        result = subprocess.run(command, capture_output=True, text=True,
                                                env=env, timeout=120)
                        self.assertEqual(result.returncode, 0,
                                         result.stdout[-2000:] + result.stderr[-2000:])
                        got = [np.load(item) for item in outputs]
                        ref = [np.load(path / f"ref_{i}.npy") for i in range(4)]
                        np.testing.assert_array_equal(got[2], ref[2])
                        for i in (0, 1, 3):
                            np.testing.assert_allclose(got[i], ref[i], rtol=2e-5,
                                                       atol=2e-5)
            finally:
                server.terminate()
                server.wait(timeout=5)
                server.stderr.close()


if __name__ == "__main__":
    unittest.main()

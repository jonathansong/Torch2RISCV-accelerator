#!/usr/bin/env python3
"""Compile a tensor-level MoE layer and check the current SA boundary."""

from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "frontend"))

import moe  # noqa: E402
from moe_native import NativeMixtralMoe, export  # noqa: E402


class NativeMoeCompileTest(unittest.TestCase):
    def test_torch_layer_forms_sa_top2_export(self):
        compiler = ROOT / "build/iree/build-compiler/tools/iree-compile"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = moe.MoeConfig(32, 64, 4, 2)
            mlir, irpa = export(NativeMixtralMoe(moe.synthetic_weights(config, seed=94)), directory)
            source = Path(mlir).read_text()
            self.assertIn("torch.aten.sort", source)
            self.assertIn("torch.aten.index_select", source)
            binaries = path / "binaries"
            binaries.mkdir()
            result = subprocess.run([
                str(compiler), mlir, "--iree-hal-target-device=sa",
                f"--iree-parameter-import=model={irpa}", "--iree-sa-form-moe",
                "--iree-sa-riscv", "--iree-sa-host-fallback",
                "--iree-hal-link-executables=false",
                "--iree-opt-const-expr-max-size-increase-threshold=0",
                "--iree-sa-codegen-report",
                f"--iree-hal-dump-executable-binaries-to={binaries}",
                "-o", str(path / "layer.vmfb")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater((path / "layer.vmfb").stat().st_size, 0)
            self.assertIn("topk_1x4xf32: ok", result.stderr)
            # These are explicit fallback boundaries, not a device FFN claim.
            self.assertIn("unsupported operation linalg.softmax", result.stderr)
            self.assertIn("unsupported operation linalg.batch_matmul", result.stderr)
            device_exports = 0
            for binary in binaries.glob("*.sadesc"):
                blob = binary.read_bytes()
                extension = struct.unpack_from("<I", blob, 44)[0]
                flags = struct.unpack_from("<I", blob, extension + 20)[0]
                device_exports += bool(flags & 2)
            self.assertEqual(device_exports, 1)


if __name__ == "__main__":
    unittest.main()

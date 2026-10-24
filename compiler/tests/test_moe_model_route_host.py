#!/usr/bin/env python3
"""Execute a model-specialized compiler route kernel on the host."""

import ctypes
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[2]
COMPILE = REPO / "build/iree/build-compiler/tools/iree-compile"
BUNDLE = REPO / "build/moe/m5k/real-layer0"
CASES = REPO / "build/moe/m5k/board-route/cases.json"


class ModelRouteHostTest(unittest.TestCase):
    def test_real_mixtral_route_specialization(self):
        if not (BUNDLE / "route.mlir").is_file() or not CASES.is_file():
            self.skipTest("stage M5k real model and board-route bundles first")
        if platform.machine() != "x86_64" or not shutil.which("clang-18"):
            self.skipTest("requires x86-64 MAP_32BIT and clang-18")
        cases = json.loads(CASES.read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ir = root / "ir"
            ir.mkdir()
            result = subprocess.run([str(COMPILE), str(BUNDLE / "route.mlir"),
                "--compile-from=executable-configurations", "--iree-sa-riscv",
                f"--iree-sa-emit-riscv={ir}", "-o", str(root / "route.vmfb")],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            path = next(ir.glob("*.ll"))
            raw = path.read_text()
            library = root / "route_host.so"
            subprocess.run(["clang-18", "-O2", "-fPIC", "-shared", str(path),
                            "-o", str(library)], check=True, capture_output=True, text=True)
            symbol = re.search(r"define void @([A-Za-z0-9_]+)\(", raw)
            self.assertIsNotNone(symbol)
            fn = getattr(ctypes.CDLL(str(library)), symbol.group(1))
            fn.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
            libc = ctypes.CDLL(None)
            libc.mmap.restype = ctypes.c_void_p
            libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
            libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            base = libc.mmap(None, 4096, 3, 0x22 | 0x40, -1, 0)
            self.assertLess(base, 1 << 32)
            try:
                for name, case in cases["cases"].items():
                    with self.subTest(case=name):
                        values = np.asarray(case["values"], np.float32)
                        ids = np.asarray(case["ids"], np.uint32)
                        ctypes.memmove(base, values.ctypes.data, values.nbytes)
                        ctypes.memmove(base + 64, ids.ctypes.data, ids.nbytes)
                        ctypes.memset(base + 128, 0, 96)
                        # Match the board binding: the 32-byte record starts
                        # 64 bytes into its allocated output buffer.
                        fn((ctypes.c_uint32 * 3)(base, base + 64, base + 128),
                           (ctypes.c_uint32 * 1)(0))
                        record = np.ctypeslib.as_array(
                            (ctypes.c_uint32 * 8).from_address(base + 192)).copy()
                        prefix = np.ctypeslib.as_array(
                            (ctypes.c_uint32 * 8).from_address(base + 128)).copy()
                        np.testing.assert_array_equal(prefix, np.zeros(8, np.uint32))
                        self.assertEqual(record[0], 2)
                        np.testing.assert_array_equal(record[1:3], ids)
                        np.testing.assert_array_equal(record[3:5],
                            ids * np.uint32(cases["expert_stride"]))
                        self.assertLessEqual(float(np.max(np.abs(
                            record[5:7].view(np.float32) - case["weights"]))), 5e-5)
            finally:
                libc.munmap(base, 4096)


if __name__ == "__main__":
    unittest.main()

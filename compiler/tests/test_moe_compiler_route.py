#!/usr/bin/env python3
"""Compile and execute the integer-only native SA MoE route export."""

import ctypes
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest

import numpy as np

from test_moe_top2_raw import topk_reference

REPO = Path(__file__).resolve().parents[2]
COMPILE = REPO / "build/iree/build-compiler/tools/iree-compile"
SOURCE = Path(__file__).with_name("moe_route_top2_export.mlir")


class MoeCompilerRouteTest(unittest.TestCase):
    def test_embedded_code_and_host_ir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binaries, ir = root / "binaries", root / "ir"
            binaries.mkdir()
            ir.mkdir()
            base_command = [str(COMPILE), str(SOURCE), "--compile-from=executable-configurations"]
            result = subprocess.run(base_command + ["-o", str(root / "without_riscv.vmfb")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires --iree-sa-riscv", result.stderr)
            result = subprocess.run(base_command + ["--iree-sa-riscv", f"--iree-sa-emit-riscv={ir}",
                                                f"--iree-hal-dump-executable-binaries-to={binaries}",
                                                "-o", str(root / "route.vmfb")],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            binary = next(binaries.glob("*.sadesc")).read_bytes()
            extension, table = struct.unpack_from("<II", binary, 44)
            self.assertEqual(struct.unpack_from("<I", binary, extension + 20)[0] & 2, 2)
            code_offset, code_bytes = struct.unpack_from("<II", binary, table)
            self.assertEqual(code_offset % 64, 0)
            self.assertGreater(code_bytes, 0)
            self.assertLess(code_bytes, 4096)
            self.assertLessEqual(code_offset + code_bytes, len(binary))
            path = next(ir.glob("*.ll"))
            raw = path.read_text()
            self.assertIn("udiv i32", raw)
            self.assertNotIn("declare float", raw)
            self.check_host_ir(root, path, raw)

    def check_host_ir(self, root, path, raw):
        host = root / "route_host.so"
        subprocess.run(["clang-18", "-O2", "-fPIC", "-shared", str(path), "-o", str(host)],
                       check=True, capture_output=True, text=True)
        symbol = re.search(r"define void @([A-Za-z0-9_]+)\(", raw).group(1)
        fn = getattr(ctypes.CDLL(str(host)), symbol)
        fn.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
        libc = ctypes.CDLL(None)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_size_t]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        base = libc.mmap(None, 4096, 3, 0x22 | 0x40, -1, 0)
        self.assertLess(base, 1 << 32)
        try:
            rng = np.random.default_rng(203)
            cases = [np.array([7, 6, 0, -1], np.float32),
                     np.array([5, 5, 5, 5], np.float32),
                     np.array([54, 40, -8, -9], np.float32)]
            cases += [rng.uniform(-100, 100, n).astype(np.float32)
                      for n in (4, 8, 16) for _ in range(30)]
            max_error = 0.0
            for scores in cases:
                values, ids = topk_reference(scores)
                ctypes.memmove(base, values.ctypes.data, values.nbytes)
                ctypes.memmove(base + 64, ids.ctypes.data, ids.nbytes)
                bindings = (ctypes.c_uint32 * 3)(base, base + 64, base + 128 - 64)
                fn(bindings, (ctypes.c_uint32 * 1)(0))
                record = np.ctypeslib.as_array((ctypes.c_uint32 * 8).from_address(base + 128)).copy()
                self.assertEqual(record[0], 2)
                np.testing.assert_array_equal(record[1:3], ids)
                np.testing.assert_array_equal(record[3:5], ids.astype(np.uint32) * 6784)
                self.assertEqual(record[7], 0)
                exact = np.exp((values - values[0]).astype(np.float64))
                exact /= exact.sum()
                max_error = max(max_error, float(np.max(np.abs(record[5:7].view(np.float32) - exact))))
            self.assertLess(max_error, 5e-5)
            nan = np.array([np.nan, 0], np.float32)
            ids = np.array([0, 1], np.int32)
            ctypes.memmove(base, nan.ctypes.data, nan.nbytes)
            ctypes.memmove(base + 64, ids.ctypes.data, ids.nbytes)
            fn((ctypes.c_uint32 * 3)(base, base + 64, base + 64), (ctypes.c_uint32 * 1)(0))
            self.assertEqual(ctypes.c_uint32.from_address(base + 128).value, 0)
        finally:
            libc.munmap(base, 4096)


if __name__ == "__main__":
    unittest.main()

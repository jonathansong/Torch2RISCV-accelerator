#!/usr/bin/env python3
"""Compile a recognized MoE router into an embedded RISC-V-only SA export."""

from pathlib import Path
import ctypes
import platform
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

import numpy as np

from test_moe_top2_raw import topk_reference


REPO = Path(__file__).resolve().parents[2]
COMPILE = REPO / "build/iree/build-compiler/tools/iree-compile"
SOURCE = Path(__file__).with_name("moe_router_w8_top2.mlir")
BOARD_SOURCE = Path(__file__).with_name("moe_router_w8_top2_d16.mlir")


class MoeCompilerRiscvTest(unittest.TestCase):
    def test_kv260_d16_router_compiles(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "router.vmfb"
            result = subprocess.run([str(COMPILE), str(BOARD_SOURCE), "--iree-hal-target-device=sa",
                                     "--iree-sa-form-moe", "--iree-sa-riscv", "--iree-sa-d=16",
                                     "--iree-sa-gemv-ports=4", "--iree-hal-link-executables=false",
                                     "-o", str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater(out.stat().st_size, 0)

    def test_router_code_table_without_riscv_option(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binaries, ir = root / "binaries", root / "ir"
            binaries.mkdir()
            ir.mkdir()
            common = [str(COMPILE), str(SOURCE), "--iree-hal-target-device=sa",
                      "--iree-sa-form-moe", "--iree-hal-link-executables=false"]
            # without --iree-sa-riscv only the RISC-V-only export (the Top-2) gets
            # embedded code (docs/moe_native_lowering_design.md §16)
            result = subprocess.run(common + [f"--iree-sa-emit-riscv={ir}",
                                              f"--iree-hal-dump-executable-binaries-to={binaries}",
                                              "-o", str(root / "router.vmfb")],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater((root / "router.vmfb").stat().st_size, 0)

            marked = []
            for path in binaries.glob("*.sadesc"):
                blob = path.read_bytes()
                extension, table = struct.unpack_from("<II", blob, 44)
                flags = struct.unpack_from("<I", blob, extension + 20)[0]
                if flags & 2:
                    export_table = struct.unpack_from("<I", blob, 24)[0]
                    self.assertEqual(struct.unpack_from("<I", blob, export_table + 12)[0], 1)
                    code_offset, code_bytes = struct.unpack_from("<II", blob, table)
                    self.assertEqual(code_offset % 64, 0)
                    self.assertGreater(code_bytes, 0)
                    self.assertLessEqual(code_offset + code_bytes, len(blob))
                    marked.append(path)
                else:
                    self.assertEqual(table, 0, f"{path.name}: code for an export with a descriptor form")
            self.assertEqual(len(marked), 1)
            raw_ir = next(ir.glob("*topk*.ll")).read_text()
            # the words read and written through the runtime (the sim's host build can run it)
            self.assertEqual(raw_ir.count("call void @sa_rv_store32("), 4)
            self.assertIn("declare i32 @sa_rv_ldparam(i32)", raw_ir)
            if platform.system() == "Linux" and platform.machine() == "x86_64" and shutil.which("clang-18"):
                self.check_host_ir(root, next(ir.glob("*topk*.ll")), raw_ir)

    def check_host_ir(self, root, ir_path, raw_ir):
        """Run the emitted target-neutral IR with physical-style 32-bit addresses
        (the runtime's word load and store as plain accesses)."""
        host = root / "top2_ir_host.so"
        shim = root / "rt.c"
        shim.write_text("#include <stdint.h>\n"
                        "uint32_t sa_rv_ldparam(uint32_t a) { return *(volatile uint32_t *)(uintptr_t)a; }\n"
                        "void sa_rv_store32(uint32_t a, uint32_t v) { *(volatile uint32_t *)(uintptr_t)a = v; }\n")
        subprocess.run(["clang-18", "-O2", "-fPIC", "-shared", "-Wno-override-module", str(ir_path), str(shim),
                        "-o", str(host)], check=True, capture_output=True, text=True)
        symbol = re.search(r"define void @([A-Za-z0-9_]+)\(", raw_ir).group(1)
        fn = getattr(ctypes.CDLL(str(host)), symbol)
        fn.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
        libc = ctypes.CDLL(None)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_size_t]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        # MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT keeps addresses representable
        # by the RISC-V calling convention's u32 binding table.
        base = libc.mmap(None, 4096, 3, 0x22 | 0x40, -1, 0)
        self.assertLess(base, 1 << 32)
        try:
            cases = [np.array([np.nan, 2, 3, 4, 5, 6, 7, 8], np.float32),
                     np.array([-0.0, 0.0, -np.inf, np.inf, 1, 1, 1, 1], np.float32),
                     np.random.default_rng(9).standard_normal(8).astype(np.float32)]
            for scores in cases:
                with self.subTest(scores=scores):
                    ctypes.memmove(base, scores.ctypes.data, scores.nbytes)
                    # The compiled IREE dispatch's third binding has a 64-byte subspan offset.
                    bindings = (ctypes.c_uint32 * 3)(base, base + 128, base + 256 - 64)
                    fn(bindings, (ctypes.c_uint32 * 1)(0))
                    values = np.ctypeslib.as_array((ctypes.c_uint32 * 2).from_address(base + 128)).copy()
                    indices = np.ctypeslib.as_array((ctypes.c_uint32 * 2).from_address(base + 256)).copy()
                    expected_values, expected_ids = topk_reference(scores)
                    np.testing.assert_array_equal(values, expected_values.view(np.uint32))
                    np.testing.assert_array_equal(indices, expected_ids.astype(np.uint32))
        finally:
            libc.munmap(base, 4096)


if __name__ == "__main__":
    unittest.main()

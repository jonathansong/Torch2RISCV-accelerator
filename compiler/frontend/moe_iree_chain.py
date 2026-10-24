#!/usr/bin/env python3
"""Package the compiled MoE route and shared FFN as one IREE HAL program.

The caller supplies selected Top-2 values/IDs, the activation, packed
parameters, a 96-byte route buffer, and the output buffer. The VMFB owns the
route-to-FFN dispatch ordering and buffer handoff. Router projection and
Top-2 selection are still outside this program.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "compiler/runtime/tools"))
import sadesc  # noqa: E402

COMPILE = REPO / "build/iree/build-compiler/tools/iree-compile"


def binary_op(name, data):
    return (f'  hal.executable @{name} {{\n'
            f'    hal.executable.binary @sa_desc_v1 attributes '
            f'{{data = dense<"0x{data.hex().upper()}"> : vector<{len(data)}xi8>, '
            'format = "sa-desc-v1", mime_type = "application/x-sa-desc"}\n'
            '  }\n')


def source(route_binary, ffn_binary, *, dim, param_offset):
    for name, data, bindings in (("route_top2", route_binary, 3),
                                 ("moe_top2_ffn", ffn_binary, 4)):
        d, _, exports = sadesc.read(data)
        if d != 16 or len(exports) != 1 or exports[0][0] != name or exports[0][2] != bindings:
            raise ValueError(f"{name}: incompatible D=16 SA executable")
    if dim <= 0 or param_offset < 0 or param_offset % 64:
        raise ValueError("invalid dimension or packed expert-table offset")
    activation_bytes = dim * 4
    return ("module {\n" + binary_op("route", route_binary)
            + binary_op("ffn", ffn_binary) + f'''
  util.func public @main(%values_view: !hal.buffer_view,
                         %ids_view: !hal.buffer_view,
                         %x_view: !hal.buffer_view,
                         %params_view: !hal.buffer_view,
                         %route_view: !hal.buffer_view,
                         %out_view: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c96 = arith.constant 96 : index
    %cparam = arith.constant {param_offset} : index
    %cx = arith.constant {activation_bytes} : index
    %affinity = arith.constant -1 : i64
    %device = hal.devices.get %c0 : !hal.device
    %route_exe = hal.executable.create device(%device : !hal.device)
        affinity(%affinity) target(@route::@sa_desc_v1) : !hal.executable
    %ffn_exe = hal.executable.create device(%device : !hal.device)
        affinity(%affinity) target(@ffn::@sa_desc_v1) : !hal.executable
    %values = hal.buffer_view.buffer<%values_view : !hal.buffer_view> : !hal.buffer
    %ids = hal.buffer_view.buffer<%ids_view : !hal.buffer_view> : !hal.buffer
    %x = hal.buffer_view.buffer<%x_view : !hal.buffer_view> : !hal.buffer
    %params = hal.buffer_view.buffer<%params_view : !hal.buffer_view> : !hal.buffer
    %route_buffer = hal.buffer_view.buffer<%route_view : !hal.buffer_view> : !hal.buffer
    %out = hal.buffer_view.buffer<%out_view : !hal.buffer_view> : !hal.buffer
    %params_size = hal.buffer.length<%params : !hal.buffer> : index
    %expert_bytes = arith.subi %params_size, %cparam : index
    %cmd = hal.command_buffer.create device(%device : !hal.device)
        mode("OneShot") categories("Dispatch") affinity(%affinity) : !hal.command_buffer
    hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
        target(%route_exe : !hal.executable)[%c0]
        workgroups([%c1, %c1, %c1]) constants([])
        bindings([(%values : !hal.buffer)[%c0, %c8],
                  (%ids : !hal.buffer)[%c0, %c8],
                  (%route_buffer : !hal.buffer)[%c0, %c96]]) flags("None")
    hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer>
        source("CommandIssue") target("CommandProcess") flags("None")
    hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
        target(%ffn_exe : !hal.executable)[%c0]
        workgroups([%c1, %c1, %c1]) constants([])
        bindings([(%x : !hal.buffer)[%c0, %cx],
                  (%route_buffer : !hal.buffer)[%c64, %c32],
                  (%params : !hal.buffer)[%cparam, %expert_bytes],
                  (%out : !hal.buffer)[%c0, %cx]]) flags("None")
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    %no_fence = util.null : !hal.fence
    hal.device.queue.execute<%device : !hal.device>
        affinity(%affinity) wait(%no_fence) signal(%no_fence)
        commands(%cmd) flags("None")
    util.return %route_view, %out_view : !hal.buffer_view, !hal.buffer_view
  }}
}}
''')


def build(bundle, out):
    bundle, out = Path(bundle), Path(out)
    manifest = json.loads((bundle / "moe_manifest.json").read_text())
    if manifest["d"] != 16 or manifest["top_k"] != 2:
        raise ValueError("only D=16 Top-2 bundles are supported")
    binaries = list((bundle / "route_binaries").glob("*.sadesc"))
    if len(binaries) != 1:
        raise ValueError("expected one compiled route executable")
    route_binary = binaries[0].read_bytes()
    ffn_binary = (bundle / "moe_ffn.sadesc").read_bytes()
    out.mkdir(parents=True, exist_ok=True)
    mlir = out / "moe_chain.mlir"
    vmfb = out / "moe_chain.vmfb"
    mlir.write_text(source(route_binary, ffn_binary, dim=manifest["dim"],
                           param_offset=manifest["expert_table_offset"]))
    subprocess.run([str(COMPILE), str(mlir), "--compile-from=hal", "-o", str(vmfb)],
                   check=True)
    return vmfb


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.bundle, args.out))


if __name__ == "__main__":
    main()

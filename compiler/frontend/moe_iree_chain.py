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


def vmvx_binary_op(name, data):
    return (f'  hal.executable @{name} {{\n'
            f'    hal.executable.binary @vmvx_bytecode_fb attributes '
            f'{{data = dense<"0x{data.hex().upper()}"> : vector<{len(data)}xi8>, '
            'format = "vmvx-bytecode-fb"}\n'
            '  }\n')


def source(route_binary, ffn_binary, *, dim, param_offset, top2_binary=None,
           experts=None, router_binary=None, top2_abi=None):
    for name, data, bindings in (("route_top2", route_binary, 3),
                                 ("moe_top2_ffn", ffn_binary, 4)):
        d, _, exports = sadesc.read(data)
        if d != 16 or len(exports) != 1 or exports[0][0] != name or exports[0][2] != bindings:
            raise ValueError(f"{name}: incompatible D=16 SA executable")
    if dim <= 0 or param_offset < 0 or param_offset % 64:
        raise ValueError("invalid dimension or packed expert-table offset")
    if top2_binary is not None:
        d, _, exports = sadesc.read(top2_binary)
        if d != 16 or len(exports) != 1 or "_topk_" not in exports[0][0] or exports[0][2] != 3:
            raise ValueError("incompatible compiled Top-2 export")
        if experts is None or experts < 4 or experts > 16:
            raise ValueError("Top-2 requires 4..16 experts")
    if router_binary is not None and top2_binary is None:
        raise ValueError("router dispatch requires the Top-2 dispatch")
    activation_bytes = dim * 4
    top2_abi = top2_abi or {"scores_offset": 0, "values_offset": 0, "ids_offset": 64}
    if top2_abi["scores_offset"] != 0:
        raise ValueError("unsupported nonzero Top-2 score binding offset")
    values_offset, ids_offset = top2_abi["values_offset"], top2_abi["ids_offset"]
    values_storage, ids_storage = values_offset + 8, ids_offset + 8
    top2_binary_op = binary_op("top2", top2_binary) if top2_binary is not None else ""
    router_op = vmvx_binary_op("router", router_binary) if router_binary is not None else ""
    router_arg = ("                         %router_view: !hal.buffer_view,\n"
                  if router_binary is not None else "")
    scores_arg = "%scores_view: !hal.buffer_view,\n                         " if top2_binary is not None else ""
    scores_size = f"    %cscores = arith.constant {experts * 4} : index\n" if top2_binary is not None else ""
    router_size = (f"    %crouter = arith.constant {experts * activation_bytes} : index\n"
                   if router_binary is not None else "")
    ids_binding = ("[%cids_offset, %c8]" if top2_binary is not None else "[%c0, %c8]")
    values_binding = ("[%cvalues_offset, %c8]" if top2_binary is not None else "[%c0, %c8]")
    top2_exe = ("    %top2_exe = hal.executable.create device(%device : !hal.device)\n"
                "        affinity(%affinity) target(@top2::@sa_desc_v1) : !hal.executable\n"
                if top2_binary is not None else "")
    router_exe = ("    %router_exe = hal.executable.create device(%device : !hal.device)\n"
                  "        affinity(%affinity) target(@router::@vmvx_bytecode_fb) : !hal.executable\n"
                  if router_binary is not None else "")
    scores_buffer = ("    %scores = hal.buffer_view.buffer<%scores_view : !hal.buffer_view> : !hal.buffer\n"
                     if top2_binary is not None else "")
    router_buffer = ("    %router = hal.buffer_view.buffer<%router_view : !hal.buffer_view> : !hal.buffer\n"
                     if router_binary is not None else "")
    router_dispatch = ('''
    hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
        target(%router_exe : !hal.executable)[%c0]
        workgroups([%c1, %c1, %c1]) constants([])
        bindings([(%x : !hal.buffer)[%c0, %cx],
                  (%router : !hal.buffer)[%c0, %crouter],
                  (%scores : !hal.buffer)[%c0, %cscores]]) flags("None")
    hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer>
        source("CommandIssue") target("CommandProcess") flags("None")'''
                       if router_binary is not None else "")
    top2_dispatch = ('''
    hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
        target(%top2_exe : !hal.executable)[%c0]
        workgroups([%c1, %c1, %c1]) constants([])
        bindings([(%scores : !hal.buffer)[%c0, %cscores],
                  (%values : !hal.buffer)[%c0, %cvalues_storage],
                  (%ids : !hal.buffer)[%c0, %cids_storage]]) flags("None")
    hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer>
        source("CommandIssue") target("CommandProcess") flags("None")'''
                     if top2_binary is not None else "")
    returned = ("%scores_view, %values_view, %ids_view, %route_view, %out_view : "
                "!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view"
                if router_binary is not None else
                "%values_view, %ids_view, %route_view, %out_view : "
                "!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view"
                if top2_binary is not None else
                "%route_view, %out_view : !hal.buffer_view, !hal.buffer_view")
    return_types = ("(!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view)"
                    if router_binary is not None else
                    "(!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view)"
                    if top2_binary is not None else "(!hal.buffer_view, !hal.buffer_view)")
    return ("module {\n" + binary_op("route", route_binary)
            + binary_op("ffn", ffn_binary) + top2_binary_op + router_op + f'''
  util.func public @main({scores_arg}%values_view: !hal.buffer_view,
                         %ids_view: !hal.buffer_view,
                         %x_view: !hal.buffer_view,
{router_arg}                         %params_view: !hal.buffer_view,
                         %route_view: !hal.buffer_view,
                         %out_view: !hal.buffer_view) -> {return_types} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c72 = arith.constant 72 : index
    %c96 = arith.constant 96 : index
    %cvalues_storage = arith.constant {values_storage} : index
    %cids_storage = arith.constant {ids_storage} : index
    %cvalues_offset = arith.constant {values_offset} : index
    %cids_offset = arith.constant {ids_offset} : index
{scores_size}{router_size}    %cparam = arith.constant {param_offset} : index
    %cx = arith.constant {activation_bytes} : index
    %affinity = arith.constant -1 : i64
    %device = hal.devices.get %c0 : !hal.device
{top2_exe}{router_exe}    %route_exe = hal.executable.create device(%device : !hal.device)
        affinity(%affinity) target(@route::@sa_desc_v1) : !hal.executable
    %ffn_exe = hal.executable.create device(%device : !hal.device)
        affinity(%affinity) target(@ffn::@sa_desc_v1) : !hal.executable
{scores_buffer}{router_buffer}    %values = hal.buffer_view.buffer<%values_view : !hal.buffer_view> : !hal.buffer
    %ids = hal.buffer_view.buffer<%ids_view : !hal.buffer_view> : !hal.buffer
    %x = hal.buffer_view.buffer<%x_view : !hal.buffer_view> : !hal.buffer
    %params = hal.buffer_view.buffer<%params_view : !hal.buffer_view> : !hal.buffer
    %route_buffer = hal.buffer_view.buffer<%route_view : !hal.buffer_view> : !hal.buffer
    %out = hal.buffer_view.buffer<%out_view : !hal.buffer_view> : !hal.buffer
    %params_size = hal.buffer.length<%params : !hal.buffer> : index
    %expert_bytes = arith.subi %params_size, %cparam : index
    %cmd = hal.command_buffer.create device(%device : !hal.device)
        mode("OneShot") categories("Dispatch") affinity(%affinity) : !hal.command_buffer
{router_dispatch}
{top2_dispatch}
    hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
        target(%route_exe : !hal.executable)[%c0]
        workgroups([%c1, %c1, %c1]) constants([])
        bindings([(%values : !hal.buffer){values_binding},
                  (%ids : !hal.buffer){ids_binding},
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
    util.return {returned}
  }}
}}
''')


def build(bundle, out, *, include_top2=False, include_router=False):
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
    top2_binary = ((bundle / "moe_top2.sadesc").read_bytes() if include_top2 else None)
    router_binary = ((bundle / "moe_router.vmvx").read_bytes() if include_router else None)
    abi_path = bundle / "moe_top2_abi.json"
    top2_abi = json.loads(abi_path.read_text()) if abi_path.is_file() else None
    mlir.write_text(source(route_binary, ffn_binary, dim=manifest["dim"],
                           param_offset=manifest["expert_table_offset"],
                           top2_binary=top2_binary, experts=manifest["experts"],
                           router_binary=router_binary, top2_abi=top2_abi))
    subprocess.run([str(COMPILE), str(mlir), "--compile-from=hal", "-o", str(vmfb)],
                   check=True)
    return vmfb


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-top2", action="store_true")
    parser.add_argument("--include-router", action="store_true")
    args = parser.parse_args()
    print(build(args.bundle, args.out, include_top2=args.include_top2,
                include_router=args.include_router))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build model-derived IREE MoE artifacts and a single route-to-FFN VMFB.

Checks the recognized Mixtral MoE shape against the packed W8A8 ABI, then
assembles the extracted executables into one HAL program.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "compiler/runtime/tools"))

import moe_hf  # noqa: E402
from moe_native import NativeMixtralMoeOutput, export  # noqa: E402
import moe_m3b  # noqa: E402
import sadesc  # noqa: E402

COMPILE = REPO / "build/iree/build-compiler/tools/iree-compile"
ROUTE_FIXTURE = REPO / "compiler/tests/moe_route_top2_export.mlir"


def artifact_info(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def compile_iree(source, irpa, out, *, preprocessing=False, d=16,
                 codegen_report=False, binary_dir=None, native_bundle=None,
                 model_abi=False):
    command = [str(COMPILE), str(source), "--iree-hal-target-device=sa",
               f"--iree-parameter-import=model={irpa}", "--iree-sa-form-moe",
               "--iree-sa-riscv", "--iree-sa-host-fallback",
               "--iree-hal-link-executables=false", f"--iree-sa-d={d}",
               "--iree-opt-const-expr-max-size-increase-threshold=0"]
    if preprocessing:
        command.append("--compile-to=preprocessing")
    if codegen_report:
        command.append("--iree-sa-codegen-report")
    if binary_dir is not None:
        command.append(f"--iree-hal-dump-executable-binaries-to={binary_dir}")
    if native_bundle is not None:
        command.append(f"--iree-sa-moe-bundle={native_bundle}")
    if model_abi:
        command.append("--iree-sa-moe-model-abi")
    return subprocess.run(command + ["-o", str(out)], check=True,
                          capture_output=True, text=True).stderr


def dispatch_coverage(report):
    """Summarize final SA codegen decisions without counting failure diagnostics twice."""
    decisions = {}
    for line in report.splitlines():
        match = re.fullmatch(r"sa codegen: (.*): (ok|on the host(?: \(.*\))?)", line)
        if match:
            name, decision = match.groups()
            if name in decisions:
                raise ValueError(f"duplicate SA codegen decision for {name}")
            decisions[name] = "sa" if decision == "ok" else "host"
    if not decisions:
        raise ValueError("IREE emitted no SA codegen decisions")
    return {"sa": sorted(name for name, target in decisions.items() if target == "sa"),
            "host": sorted(name for name, target in decisions.items() if target == "host")}


def recognized_fields(path):
    source = path.read_text()
    functions = [line for line in source.splitlines()
                 if 'sa.moe.arch = "mixtral_top2_ffn"' in line]
    if len(functions) != 1:
        raise ValueError("IREE did not recognize exactly one Mixtral Top-2 FFN")
    declaration = functions[0]
    fields = {}
    for key in ("dim", "hidden", "experts", "expert_stride"):
        matches = re.findall(rf"sa\.moe\.{key} = (\d+) : i64", declaration)
        if len(matches) != 1:
            raise ValueError(f"IREE preprocessing did not provide one sa.moe.{key}")
        fields[key] = int(matches[0])
    return fields


def route_source(experts, stride):
    source = ROUTE_FIXTURE.read_text()
    old = "sahl.route_top2 %values, %indices, %route {experts = 16 : i64, stride = 6784 : i64}"
    if source.count(old) != 1:
        raise ValueError("SAHL route fixture changed; update model-derived route generation")
    return source.replace(old, f"sahl.route_top2 %values, %indices, %route "
                               f"{{experts = {experts} : i64, stride = {stride} : i64}}")


def build(model, layer, out, d=16):
    if d != 16:
        raise ValueError("the current route export and KV260 template require D=16")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config, weights, source = moe_hf.load_layer(str(model), layer, d=d)
    if source["adapter"] != "mixtral" or not source["normalize_topk"] or config.top_k != 2:
        raise ValueError("expected a normalized Top-2 Mixtral checkpoint")
    mlir, irpa = export(NativeMixtralMoeOutput(weights), str(out))
    preprocessing = out / "preprocessing.mlir"
    compile_iree(mlir, irpa, preprocessing, preprocessing=True, d=d)
    recognized = recognized_fields(preprocessing)
    packed_config, _, manifest = moe_hf.pack_layer(str(model), str(out), layer=layer, d=d)
    expected = {"dim": packed_config.dim, "hidden": packed_config.hidden,
                "experts": packed_config.experts,
                "expert_stride": manifest["expert_stride"]}
    if recognized != expected:
        raise ValueError(f"compiler/packer MoE ABI mismatch: {recognized} != {expected}")

    module = out / "moe_layer.vmfb"
    with tempfile.TemporaryDirectory() as binary_dir:
        coverage = dispatch_coverage(compile_iree(mlir, irpa, module, d=d,
                                                  codegen_report=True,
                                                  binary_dir=binary_dir))
        top2 = []
        for path in Path(binary_dir).glob("*.sadesc"):
            binary = path.read_bytes()
            descriptor_d, _, exports = sadesc.read(binary)
            if len(exports) == 1 and "_topk_" in exports[0][0]:
                if descriptor_d != d or exports[0][2] != 3:
                    raise RuntimeError("compiled Top-2 has an incompatible SA ABI")
                top2.append(binary)
        if len(top2) != 1:
            raise RuntimeError(f"expected one native Top-2 export, got {len(top2)}")
        (out / "moe_top2.sadesc").write_bytes(top2[0])
        router = [re.fullmatch(rf"main\$async_dispatch_(\d+)_matmul_1x{config.experts}x{config.dim}_f32", name)
                  for name in coverage["host"]]
        router = [match.group(1) for match in router if match]
        if len(router) != 1:
            raise RuntimeError(f"expected one host router matmul, got {len(router)}")
        router_binary = Path(binary_dir) / f"module_main$async_dispatch_{router[0]}_vmvx_bytecode_fb.vmfb"
        if not router_binary.is_file():
            raise RuntimeError("missing model-compiled VMVX router executable")
        (out / "moe_router.vmvx").write_bytes(router_binary.read_bytes())
    ffn = out / "moe_ffn.sadesc"
    ffn_bytes = moe_m3b.make_executable(config.dim, config.hidden, d)
    descriptor_d, _, exports = sadesc.read(ffn_bytes)
    if descriptor_d != d or len(exports) != 1 or exports[0][0] != "moe_top2_ffn" or exports[0][2] != 4:
        raise RuntimeError("FFN descriptor does not have one reusable four-binding export")
    ffn.write_bytes(ffn_bytes)
    route_mlir = out / "route.mlir"
    route_mlir.write_text(route_source(config.experts, manifest["expert_stride"]))
    binary_dir = out / "route_binaries"
    binary_dir.mkdir(exist_ok=True)
    route_vmfb = out / "route.vmfb"
    subprocess.run([str(COMPILE), str(route_mlir),
                    "--compile-from=executable-configurations", "--iree-sa-riscv",
                    f"--iree-hal-dump-executable-binaries-to={binary_dir}",
                    "-o", str(route_vmfb)], check=True, capture_output=True, text=True)
    binaries = list(binary_dir.glob("*.sadesc"))
    if len(binaries) != 1:
        raise RuntimeError(f"expected one route executable, got {len(binaries)}")
    binary = binaries[0].read_bytes()
    extension, code_table = struct.unpack_from("<II", binary, 44)
    flags = struct.unpack_from("<I", binary, extension + 20)[0]
    code_offset, code_length = struct.unpack_from("<II", binary, code_table)
    if not flags & 2 or not code_length or code_offset + code_length > len(binary):
        raise RuntimeError("missing compiler-generated embedded route kernel")
    (out / "moe_route.bin").write_bytes(binary[code_offset:code_offset + code_length])
    np.save(out / "moe_router.npy", np.asarray(weights.router, dtype=np.float32))
    import iree.runtime as rt
    archive = rt.ParameterIndex()
    archive.add_buffer("router", np.ascontiguousarray(weights.router).tobytes())
    archive.add_buffer("params", (out / "moe_params.bin").read_bytes())
    archive.create_archive_file(str(out / "moe_native.irpa"))
    compile_iree(mlir, irpa, out / "moe_chain.mlir", preprocessing=True,
                 d=d, native_bundle=out)
    compile_iree(mlir, irpa, out / "moe_chain.vmfb", d=d, native_bundle=out)
    compile_iree(mlir, irpa, out / "moe_model.mlir", preprocessing=True,
                 d=d, native_bundle=out, model_abi=True)
    compile_iree(mlir, irpa, out / "moe_model.vmfb", d=d,
                 native_bundle=out, model_abi=True)
    result = {"format": "sa-moe-compiler-bundle-v1", "source": source,
              "recognized": recognized, "d": d,
              "native_module_dispatches": coverage,
              "artifacts": {name: artifact_info(out / name)
                            for name in ("moe_layer.vmfb", "moe_layer.irpa", "moe_params.bin",
                                         "moe_router.npy", "moe_router.vmvx", "moe_top2.sadesc",
                                         "moe_ffn.sadesc", "route.vmfb", "moe_route.bin",
                                         "moe_chain.mlir", "moe_chain.vmfb",
                                         "moe_native.irpa", "moe_model.mlir",
                                         "moe_model.vmfb")}}
    (out / "compiler_bundle.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--d", type=int, default=16)
    args = parser.parse_args()
    result = build(args.model, args.layer, args.out, args.d)
    print(f"{args.out}: compiler recognized {result['recognized']}; "
          f"emitted {len(result['artifacts'])} model-derived artifacts")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""IREE 3.11 issue (docs/iree_compiler_plan.md §8.16 F4): two mutable int8
globals (buffers of a torch module, externalized as parameters) with equal
initial contents, each written at an input position in one call; compiled with
--iree-parameter-import, the program writes the second one's data into both
(VMVX, no sa involved). Distinct initial contents (VINIT=1) give the right
result; hf_generic.SaDecoder gives each KV buffer a value of its own.

    VINIT=0 python3 compiler/tests/iree_global_merge_repro.py /tmp/repro   # kc wrong
    VINIT=1 python3 compiler/tests/iree_global_merge_repro.py /tmp/repro   # right
"""
import os, sys, subprocess, numpy as np, torch
from iree.turbine import aot
from iree.turbine.aot import decompositions
import iree.runtime as rt
O = sys.argv[1]; os.makedirs(O, exist_ok=True)
def to_i8(x): return torch.clamp(torch.round(torch.nan_to_num(x, nan=0.0)), -127.0, 127.0).to(torch.int8)
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("kc", torch.zeros(8, 4, dtype=torch.int8))
        self.register_buffer("vc", torch.full((8, 4), int(os.environ.get("VINIT", "0")), dtype=torch.int8))
    def forward(self, k, v, pos):
        self.kc.index_copy_(0, pos[0], to_i8(k * 12.5))
        self.vc.index_copy_(0, pos[0], to_i8(v * 200.0))
        return self.kc.to(torch.int32), self.vc.to(torch.int32)
torch.set_grad_enabled(False)
m = M()
ex = (torch.ones(1, 4), torch.ones(1, 4), torch.tensor([[0]]))
ep = torch.export.export(m, ex, strict=False).run_decompositions(decompositions.current_aot_decompositions())
aot.externalize_module_parameters(m, external_scope="model")
aot.export(ep).save_mlir(f"{O}/m.mlir"); aot.save_module_parameters(f"{O}/m.irpa", m)
IB = os.environ['IREE_BUILD']
subprocess.run([f"{IB}/tools/iree-compile", f"{O}/m.mlir", "--iree-hal-target-device=local", "--iree-hal-local-target-device-backends=vmvx",
                f"--iree-parameter-import=model={O}/m.irpa", "--iree-parameter-import-maximum-size=4294967295", "-o", f"{O}/m.vmfb"], check=True)
cfg = rt.Config("local-sync")
vm = rt.load_vm_modules(rt.create_hal_module(cfg.vm_instance, cfg.device), rt.VmModule.mmap(cfg.vm_instance, f"{O}/m.vmfb"), config=cfg)
k = np.array([[-3.44, -0.57, 0.33, -0.85]], np.float32); v = np.array([[-0.022, 0.035, -0.0073, -0.0021]], np.float32)
r = M()(torch.tensor(k), torch.tensor(v), torch.tensor([[0]]))
g = vm[-1].main(k, v, np.array([[0]], np.int64))
print("ref kc[0]", r[0][0].tolist(), " iree", np.asarray(g[0].to_host())[0].tolist())
print("ref vc[0]", r[1][0].tolist(), " iree", np.asarray(g[1].to_host())[0].tolist())

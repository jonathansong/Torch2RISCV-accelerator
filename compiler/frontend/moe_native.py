#!/usr/bin/env python3
"""Tensor-level Mixtral MoE layer for the native IREE compiler path.

The layer has one expert axis, so selecting either expert uses the same graph
and parameter layout. It exports ordinary tensor operations; SA preprocessing
forms dispatches and identifies the router instead of accepting a hand-written
dispatch list. The initial frontend keeps weights and math in fp32 to establish
the model semantics before the W8A8 expert lowering is connected.
"""

import argparse
import os

import numpy as np
import torch

import moe
import moe_hf


class NativeMixtralMoe(torch.nn.Module):
    """One batch-1 top-2 gated expert block with a shared expert axis."""

    def __init__(self, weights):
        super().__init__()
        self.register_buffer("router", torch.from_numpy(np.array(weights.router, copy=True, order="C")))
        self.register_buffer("gate", torch.from_numpy(np.array(weights.gate, copy=True, order="C")))
        self.register_buffer("up", torch.from_numpy(np.array(weights.up, copy=True, order="C")))
        self.register_buffer("down", torch.from_numpy(np.array(weights.down, copy=True, order="C")))

    def forward(self, x):
        scores = x @ self.router.T
        # The supported Torch sort preserves ascending IDs for tied scores in
        # this static batch-1 path. Its two-element prefix forms device Top-2.
        sorted_values, sorted_ids = torch.sort(scores, dim=-1, descending=True)
        values, ids = sorted_values[:, :2], sorted_ids[:, :2]
        routing = torch.softmax(values, dim=-1)
        chosen = ids.reshape(2)
        gate = torch.index_select(self.gate, 0, chosen)
        up = torch.index_select(self.up, 0, chosen)
        down = torch.index_select(self.down, 0, chosen)
        column = x.expand(2, -1).unsqueeze(-1)
        hidden = torch.nn.functional.silu(torch.bmm(gate, column).squeeze(-1))
        hidden = hidden * torch.bmm(up, column).squeeze(-1)
        outputs = torch.bmm(down, hidden.unsqueeze(-1)).squeeze(-1)
        result = (outputs * routing.reshape(2, 1)).sum(dim=0, keepdim=True)
        return result, scores, ids, routing


class NativeMixtralMoeOutput(NativeMixtralMoe):
    """The model-facing MoE function: one activation input, one output tensor."""

    def forward(self, x):
        return super().forward(x)[0]


def export(model, out):
    from iree.turbine import aot

    os.makedirs(out, exist_ok=True)
    dim = model.router.shape[1]
    example = torch.linspace(-0.625, 0.75, dim, dtype=torch.float32)[None]
    program = torch.export.export(model, (example,), strict=False)
    aot.externalize_module_parameters(model, external_scope="model")
    module = aot.export(program)
    mlir = os.path.join(out, "moe_layer.mlir")
    irpa = os.path.join(out, "moe_layer.irpa")
    module.save_mlir(mlir)
    aot.save_module_parameters(irpa, model)
    return mlir, irpa


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", help="Mixtral Hugging Face checkpoint directory")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true", help="Use a small 4-expert test layer")
    args = parser.parse_args()
    if args.synthetic == bool(args.model):
        parser.error("select exactly one of --synthetic and --model")
    if args.synthetic:
        config = moe.MoeConfig(32, 64, 4, 2)
        weights = moe.synthetic_weights(config, seed=94)
    else:
        config, weights, source = moe_hf.load_layer(args.model, args.layer)
        if source["adapter"] != "mixtral" or not source["normalize_topk"] or config.top_k != 2:
            raise ValueError("native frontend currently supports normalized top-2 Mixtral layers only")
    mlir, irpa = export(NativeMixtralMoe(weights), args.out)
    print(f"exported {config.experts}-expert layer: {mlir}, {irpa}")


if __name__ == "__main__":
    main()

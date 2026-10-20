# Local patches to IREE

Applied in order to the pinned IREE revision `e4a3b0405d` by
`compiler/scripts/fetch_iree_sources.sh` (each once: an applied patch is skipped,
one that does not apply stops the script). Each is guarded by a lit test of the
sa plugin (`compiler/plugins/sa/test/`).

| Patch | What it fixes or adds | Found by | Upstream | Guard |
|---|---|---|---|---|
| `0001-util-merge-index-switch-same-selector` | `MergeIndexSwitchPattern` merged adjacent `scf.index_switch` ops by their case values only, not their selectors: with the host fallback a host dispatch took the sa variant's workgroup count (prefill wrong, `iree_compiler_plan.md` §8.17) | prefill against decode | iree-org/iree#25037 (open) | `merge_index_switch.mlir` |
| `0002-util-merge-index-switch-yield-outer-values` | the same pattern looked the second switch's yields up with `lookupOrNull`: a value from above both switches became a null operand (crash) | Qwen3's generic path | iree-org/iree#25038 (open) | `merge_index_switch.mlir` |
| `0003-flow-generalize-executables` | a pass merging executables that differ only in constants (a layer's KV cache offset, a scale) into one with those values as push constants (`--iree-flow-enable-executable-generalization`) | the RISC-V backend's code size (`riscv_backend_design.md` §9.11) | iree-org/iree#25059 (open; the local-tree version of `compiler/patches/iree-pr3`) | the golden corpus, `rv_codesize.py` |
| `0004-pluginapi-dispatch-region-formation-hook` | a plugin hook after dispatch region formation (`PipelineExtensions::extendDispatchRegionFormationPassPipeline`), where the sa plugin merges the attention regions | the attention fusion (`fusion_design.md` §6.4.1) | not submitted (a plugin API addition) | `sa_form_attn_regions.mlir` |
| `0005-tensorext-fold-insert-slice-store-keep-dest` | see below | the prefill chunk kernel (`fusion_design.md` §6.7) | already fixed upstream: iree-org/iree#24266, #24276 (2026-04-27, after the pin) | `fold_insert_slice_store.mlir` |

## 0005: an insert_slice chain folded into its store kept only the last insert

`FoldInsertSliceWithTensorStoreOp` (`compiler/src/iree/compiler/Dialect/TensorExt/Transforms/Folders.cpp`,
run by `iree-dispatch-creation-convert-tensor-to-flow` on each dispatch's body)
rewrote

```mlir
%r = tensor.insert_slice %src into %dest[...]
iree_tensor_ext.dispatch.tensor.store %r, %target, offsets = [...]
```

as a store of `%src` alone at the slice, dropping `%dest`. That is correct only when
`%dest` is undefined (`tensor.empty`). A dispatch putting several results
together into one output writes a chain

```mlir
%e  = tensor.empty() : tensor<1x16x16x128xf32>
%r0 = tensor.insert_slice %row0 into %e[0, 0, 0, 0] ...
%r1 = tensor.insert_slice %row1 into %r0[0, 1, 0, 0] ...
...
%r15 = tensor.insert_slice %row15 into %r14[0, 15, 0, 0] ...
iree_tensor_ext.dispatch.tensor.store %r15, %out, ...
```

and the fold kept only `%row15`'s store: the other 15 rows' computations became
dead code, the output's other rows were never written, and nothing failed. It
showed as a fused prefill dispatch with one `batch_matmul` instead of 16
(found by counting the ops after each pass with `--mlir-print-ir-after-all`).

The local fix stores `%dest` first (whole, at the store's offsets) unless it is a
`tensor.empty`, then `%src` at the slice; applied again, the chain becomes one
store per slice. Upstream fixed the same defect differently (#24266: fold only when
`%dest` is a `tensor.empty` or a `dispatch.tensor.load` of the same target at the
same offsets; #24276: the `tensor.empty` case), so a chain is left unfolded there.
**When the pinned IREE moves past `349078834` (#24266), drop 0005** and check the
chunk kernel's dispatches again (`fusion_design.md` §6.7, the 2-layer Qwen3 / Gemma
runs); the guard test then shows upstream's form (no fold).

The golden corpus did not change with the patch: no earlier dispatch had such a
chain.

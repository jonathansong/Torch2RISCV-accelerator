#!/usr/bin/env bash
# Compiles an exported model (qllama.mlir + qllama.irpa) for the sa device:
# parameters imported, linear weights packed and evaluated at compile time,
# the packed parameters exported (docs/iree_compiler_plan.md §6.3).
#   compiler/scripts/compile_sa.sh <dir with qllama.mlir / .irpa> [extra iree-compile flags]
# -> <dir>/sa.vmfb, <dir>/sa_packed.irpa, <dir>/sa_sources/ (dispatch sources)
# Aggressive dispatch fusion is on (C4: 242 -> 200 dispatches per token).
# Assertions stripped: a dynamic attention length (hf_generic --dyn-len) otherwise
# leaves a "mismatching contracting dimension" check per layer, each a split of
# the command buffer (SmolLM2 2 layers: 3 lists a step instead of 1).
# SA_COMPILE_FLAGS: extra iree-compile flags. SA_D: the array size (--iree-sa-d; default 8).
# SA_GEMV_PORTS: the overlay's GEMV unit (--iree-sa-gemv-ports; default 0 = none).
# <dir>/compile_flags.txt (the export's, hf_generic.export): flags the model needs, e.g.
# --iree-sa-max-dynamic=<the KV cache's length> beyond the default 256.
# SA_GENERALIZE=0: one executable per layer as before (default 1: executables
# that differ only in constants, e.g. a layer's KV cache offset or scale, merged:
# --iree-flow-enable-executable-generalization, IREE patch 0003,
# docs/riscv_backend_design.md §9.11).
# SA_FUSE_FFN=0: decode's gated FFN in separate dispatches as before (default 1:
# the two projections, gating, amax and int8 output in one dispatch,
# --iree-sa-fuse-ffn, docs/fusion_design.md §4.9).
# SA_FUSE_ATTN=1: decode's attention (q path, scores, probabilities, p . V, int8
# output) in one dispatch per layer, prefill rows unfused (--iree-sa-fuse-attn, default 0;
# docs/fusion_design.md §6.4.1, §6.4.5; on the accelerator, bit-exact, not yet on the board).
# SA_HOST_FALLBACK=0: no host fallback (default 1: every dispatch also gets a
# VMVX variant; one the sa backend cannot compile runs on the ARM host, plan
# §8.15; IREE's executable linking is off, it would merge the VMVX variants of
# executables that keep their sa ones).
set -euo pipefail
source "$(dirname "$0")/../env.sh"
dir=$1; shift
mkdir -p "$dir/sa_sources"
rm -f "$dir"/sa_sources/*.mlir      # (a source left from an earlier compile has no binary)
"$IREE_BUILD/tools/iree-compile" "$dir/qllama.mlir" --iree-hal-target-device=sa \
  --iree-opt-const-expr-max-size-increase-threshold=0 \
  --iree-parameter-import=model="$dir/qllama.irpa" --iree-parameter-import-maximum-size=4294967295 \
  --iree-parameter-export=model="$dir/sa_packed.irpa" --iree-parameter-export-minimum-size=256 \
  --iree-dispatch-creation-enable-aggressive-fusion \
  --iree-opt-strip-assertions \
  $( [ "${SA_GENERALIZE:-1}" = 1 ] && echo --iree-flow-enable-executable-generalization ) \
  $( [ "${SA_FUSE_FFN:-1}" = 1 ] && echo --iree-sa-fuse-ffn ) \
  $( [ "${SA_FUSE_ATTN:-0}" = 1 ] && echo --iree-sa-fuse-attn ) \
  $( [ "${SA_HOST_FALLBACK:-1}" = 1 ] && echo --iree-sa-host-fallback --iree-hal-link-executables=false ) \
  $( [ -n "${SA_D:-}" ] && [ "$SA_D" != 8 ] && echo --iree-sa-d="$SA_D" ) \
  $( [ "${SA_GEMV_PORTS:-0}" != 0 ] && echo --iree-sa-gemv-ports="$SA_GEMV_PORTS" ) \
  --iree-hal-dump-executable-sources-to="$dir/sa_sources" $( [ -f "$dir/compile_flags.txt" ] && cat "$dir/compile_flags.txt" ) \
  ${SA_COMPILE_FLAGS:-} "$@" -o "$dir/sa.vmfb"

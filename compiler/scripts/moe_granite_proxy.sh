#!/bin/bash
# compiler/scripts/moe_granite_proxy.sh: a Granite-3.1-1B-A400M-shaped Mixtral proxy (random weights, 2 layers, E=16 Top-2: what the
# backend supports today) -> export, compile, RISC-V image, sim trace for the perf model
# (docs/moe_native_lowering_design.md §22). Run under systemd-run (MemoryMax=10G), IREE_BUILD set;
# writes build/moe/granite_proxy/{ckpt,gp} and build/perfmodel/moe_gp.trace.
set -e
R=$(cd "$(dirname "$0")/../.." && pwd); cd $R
P=build/iree/venv/bin/python
G=build/moe/granite_proxy; C=$G/ckpt; T=$G/gp
$P - <<PY
import torch
from transformers import MixtralConfig, MixtralForCausalLM
torch.manual_seed(0)
cfg = MixtralConfig(hidden_size=1024, intermediate_size=512, num_hidden_layers=2, num_attention_heads=16,
                    num_key_value_heads=8, head_dim=64, num_local_experts=16, num_experts_per_tok=2,
                    vocab_size=49152, tie_word_embeddings=True, rms_norm_eps=1e-6, rope_theta=1.5e6)
m = MixtralForCausalLM(cfg).eval()
with torch.no_grad():
    for layer in m.model.layers:
        layer.mlp.gate.weight.normal_(0.0, 0.5)
m.save_pretrained("$C")
print("ckpt saved")
PY
rm -rf $T; mkdir -p $T
$P compiler/tests/test_hf_moe.py --model $C --export $T 2>&1 | grep -v 'Warning\|warn\|Loading' | tail -3
mkdir -p $T/sa_binaries $T/rv_ll
SA_D=16 SA_GEMV_PORTS=4 SA_COMPILE_FLAGS="--iree-sa-form-moe --iree-sa-codegen-report --iree-sa-emit-riscv=$T/rv_ll --iree-hal-dump-executable-binaries-to=$T/sa_binaries" \
  compiler/scripts/compile_sa.sh $T > $T/compile.log 2>&1
echo "codegen ok: $(grep -c 'sa codegen.*: ok' $T/compile.log), host: $(grep 'sa codegen' $T/compile.log | grep -c 'on the host')"
$P compiler/tests/rv_image.py $T/rv_ll --out $T/rv | tail -1
$P -c "import numpy as np; np.save('$T/prompt.npy', np.array([1,415,2936,9060,285,1142,28705,28740,28723,13,1014,3075,349,264,1179,1170], dtype=np.int64))"
echo "--abi=hfv --pad=16 --valid_len=64 --prefill=16" > $T/sa_args.txt
SA_RV=$T/rv SA_RV_ONLY=rv_only DEPLOY=$G MODEL=gp MB=400 OUT=moe_gp compiler/scripts/perf_trace.sh
echo DONE

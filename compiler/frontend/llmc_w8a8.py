#!/usr/bin/env python3
"""A compressed-tensors W8A8 checkpoint of a HuggingFace model made with
llm-compressor (docs/generality_implementation_plan.md WP3): SmoothQuant, then
GPTQ with the W8A8 scheme (int8 weights per output channel, symmetric, static;
int8 activations per token, dynamic, symmetric), lm_head left in floating
point. The format ct_import.py reads (and vLLM serves).

Runs in its own environment (llm-compressor pins torch / transformers; the
export environment, build/iree/venv, stays as it is):
    ~/miniconda3/envs/py311/bin/python -m venv build/llmc_venv
    build/llmc_venv/bin/pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
    build/llmc_venv/bin/pip install --no-cache-dir llmcompressor==0.14.0

    build/llmc_venv/bin/python compiler/frontend/llmc_w8a8.py --model build/llm_cache/SmolLM2-135M \\
        --out build/llm_cache/SmolLM2-135M-W8A8 [--samples 256 --seq-len 512 --seed 0 --smoothing 0.8]

Calibration: WikiText-2 (raw) train, paragraphs over 200 characters, --samples
of them drawn with numpy's default_rng(seed) (their row numbers recorded). <out>/sa_manifest.json records the recipe, the
calibration rows, the versions and the source's file hashes (the evaluation
uses the test split only).
"""
import argparse
import hashlib
import json
import os
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoothing", type=float, default=0.8, help="SmoothQuant strength (0: no SmoothQuant)")
    args = ap.parse_args()

    import torch
    import transformers
    import llmcompressor
    import compressed_tensors
    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    rows = [i for i, t in enumerate(raw["text"]) if len(t.strip()) > 200]
    import numpy as np
    calib_rows = [rows[i] for i in np.random.default_rng(args.seed).permutation(len(rows))[:args.samples]]
    ds = raw.select(calib_rows)
    ds = ds.map(lambda e: tok(e["text"], max_length=args.seq_len, truncation=True, add_special_tokens=False),
                remove_columns=ds.column_names)
    recipe = ([SmoothQuantModifier(smoothing_strength=args.smoothing)] if args.smoothing else []) + \
             [GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])]
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=args.seq_len,
            num_calibration_samples=args.samples)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, save_compressed=True)
    tok.save_pretrained(args.out)
    src = {f: sha256(os.path.join(args.model, f)) for f in sorted(os.listdir(args.model))
           if f.endswith((".safetensors", ".json"))}
    manifest = {
        "source": os.path.abspath(args.model), "source_sha256": src,
        "recipe": [str(m) for m in recipe],
        "calibration": {"dataset": "Salesforce/wikitext/wikitext-2-raw-v1", "split": "train", "filter": "len(text.strip()) > 200",
                        "seed": args.seed, "samples": args.samples, "seq_len": args.seq_len, "rows": calib_rows},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "llmcompressor": llmcompressor.__version__, "compressed_tensors": compressed_tensors.__version__},
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "sa_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{args.out}: W8A8 in {manifest['seconds']} s ({args.samples} samples x {args.seq_len} tokens)")


if __name__ == "__main__":
    main()

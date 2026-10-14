#!/usr/bin/env python3
"""model.json for board_generate.py (the interactive generation of the Z1
bundles, iree_compiler_plan.md §8.7) in a generic-frontend bundle
(test_hf_generic.py --board-bundle, deploy_hfgen.sh): the tokenizer, the BOS
the HF tokenizer prepends, the end tokens (generation_config's and the
tokenizer's), the context (sa_args.txt --valid_len, else 256), and for a model
with a chat template the token ids around one user message and, from a
two-turn rendering, those around an earlier turn's answer (board_generate.py
--chat, a conversation; the board has no jinja). A checkpoint without a template (the local
Qwen3-0.6B ones) gets its family's from CHAT.

    compiler/tests/stage_model_json.py --model build/llm_cache/Qwen3-0.6B-W8A8 --bundle build/deploy_hfgen_qwen3_qs
"""
import argparse
import json
import os
import sys

# chat formats of model families whose local checkpoints lack the template: the
# text the official template renders before the first user message (no system
# prompt), after the last one (the generation prompt), and around an earlier
# turn's answer (between the user message and it, after it); Qwen3 non-thinking
# (enable_thinking=False: an empty think block; the earlier answers without one,
# as its template renders them)
CHAT = {"qwen3": ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                  "<|im_end|>\n<|im_start|>assistant\n", "<|im_end|>\n<|im_start|>user\n")}


def model_json(model_dir, bundle):
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    tok = AutoTokenizer.from_pretrained(model_dir)
    with_sp, without = tok("a").input_ids, tok("a", add_special_tokens=False).input_ids
    bos = with_sp[0] if len(with_sp) > len(without) and with_sp[1:] == without else None
    stops = []
    try:
        e = GenerationConfig.from_pretrained(model_dir).eos_token_id
        stops = list(e) if isinstance(e, (list, tuple)) else ([e] if e is not None else [])
    except OSError:                      # (no generation_config.json)
        pass
    if tok.eos_token_id is not None and tok.eos_token_id not in stops:
        stops.append(tok.eos_token_id)
    extra = open(os.path.join(bundle, "sa_args.txt")).read().split()
    ctx = int(next((a.split("=")[1] for a in extra if a.startswith("--valid_len=")), 256))
    cfg = {"tokenizer": "hf", "bos": bos, "stop": stops[0] if stops else None, "stops": stops, "context": ctx}
    if tok.chat_template:
        m1, a1, m2 = "\x01M1\x01", "\x01A1\x01", "\x01M2\x01"
        kw = {"enable_thinking": False}  # (Qwen3: an empty think block; the other templates ignore it)
        render = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)  # noqa: E731
        pre, suf = render([{"role": "user", "content": m1}]).split(m1)
        two = render([{"role": "user", "content": m1}, {"role": "assistant", "content": a1},
                      {"role": "user", "content": m2}])
        p2, rest = two.split(m1)
        hist, rest = rest.split(a1)
        nxt, suf2 = rest.split(m2)
        if p2 != pre or suf2 != suf:     # (a template whose turns are not a plain concatenation)
            hist = nxt = None
    else:
        mc = AutoConfig.from_pretrained(model_dir)
        pre, suf, hist, nxt = CHAT.get(getattr(mc, "text_config", mc).model_type, (None,) * 4)
    ids = lambda t: tok(t, add_special_tokens=False).input_ids  # noqa: E731
    if pre is not None:
        cfg["chat"] = [ids(pre), ids(suf)]
    if hist is not None:
        cfg["chat_turn"] = [ids(hist), ids(nxt)]
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="the HF model directory (its tokenizer and generation config)")
    ap.add_argument("--bundle", required=True)
    args = ap.parse_args()
    cfg = model_json(args.model, args.bundle)
    with open(os.path.join(args.bundle, "model.json"), "w") as f:
        json.dump(cfg, f)
    print(f"{args.bundle}/model.json: {json.dumps({k: v for k, v in cfg.items() if not k.startswith('chat')})}"
          f"{', chat template' if 'chat' in cfg else ''}{' (multi-turn)' if 'chat_turn' in cfg else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

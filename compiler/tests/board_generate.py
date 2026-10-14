#!/usr/bin/env python3
"""Interactive text generation on the board: a decoder compiled by IREE for the
sa device (the C3 / C5.5 board bundles), run by the aarch64 sa-llm-run.

The overlay and rt_fw are started once; then each prompt is tokenized on the
ARM, fed to sa-llm-run (a fresh KV cache per prompt), and the generated text is
printed as the tokens arrive (greedy, until the model's end token, --max-new or
the context length).

Files (the board bundle): this script, board_launcher.py, pynq_matmul.py,
picorv32.bit / .hwh, rt_fw.bin, sa-llm-run, sa.vmfb, sa_packed.irpa,
model.json ({"tokenizer": "hf" | "llama2", "bos": id or null, "stop": id or
null, "context": n, optional "stops": [ids] (every end token) and "chat":
[ids before, ids after] one user message; the generic-frontend bundles':
stage_model_json.py}), the tokenizer (tokenizer.json + hf_tokenizer.py, or
tokenizer.bin + tokenizer.py), optional board.txt (window MB, last field) and
sa_args.txt (extra sa-llm-run arguments).

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh && \\
        cd /home/xilinx/c55 && python3 board_generate.py [--max-new 64] [--prompt "Once upon a time"]'
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board_launcher as BL  # noqa: E402


def load_tokenizer(cfg):
    """(encode(text) -> ids, decode(ids) -> text)"""
    if cfg["tokenizer"] == "hf":
        from hf_tokenizer import BpeTokenizer
        tok = BpeTokenizer(HERE)
        bos = [cfg["bos"]] if cfg.get("bos") is not None else []
        return (lambda text: bos + tok.encode(text)), tok.decode
    from tokenizer import Tokenizer
    tok = Tokenizer(os.path.join(HERE, "tokenizer.bin"), cfg.get("vocab", 32000))
    return (lambda text: tok.encode(text, bos=cfg.get("bos") is not None)), tok.decode


def generate(prompt_ids, max_new, cfg, decode, env, extra):
    """Runs sa-llm-run, printing the text as it comes; returns its stats line.
    sa-llm-run stops at "stop"; another of "stops" (e.g. Qwen3's <|endoftext|>
    besides <|im_end|>) ends it here."""
    stops = set(cfg.get("stops") or []) | ({cfg["stop"]} if cfg.get("stop") is not None else set())
    cmd = [os.path.join(HERE, "sa-llm-run"), "--device=sa", f"--module={HERE}/sa.vmfb",
           f"--parameters=model={HERE}/sa_packed.irpa", "--tokens=" + ",".join(map(str, prompt_ids)),
           f"--generate={max_new}"] + extra
    if cfg.get("stop") is not None:
        cmd.append(f"--stop_token={cfg['stop']}")
    if cfg.get("prefill") and not any(a.startswith("--prefill") for a in cmd):
        cmd.append(f"--prefill={cfg['prefill']}")   # prompts of >= M tokens in chunks, shorter ones as decode
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    out, cur, ids = bytearray(), b"", []
    shown = decode(prompt_ids)
    t0 = time.time()
    first = None
    ended = False
    state = 0                                          # 0: before "tokens:", 1: the token list, 2: after it
    while True:
        c = p.stdout.read(1)
        if not c:
            break
        out += c
        if state == 0:
            state = 1 if out.endswith(b"tokens:") else 0
            continue
        if state == 2:
            continue
        if c not in b" \n":
            cur += c
            continue
        if c == b"\n":
            state = 2
        if not cur:
            continue
        ids.append(int(cur))
        cur = b""
        gen = ids[len(prompt_ids):]
        if gen and gen[-1] in stops:
            if gen[-1] != cfg.get("stop"):           # (not sa-llm-run's own: it would go on)
                p.terminate()
                ended = True
            gen = gen[:-1]
        if not gen:
            if ended:
                break
            continue
        first = first or time.time()
        text = decode(prompt_ids + gen)
        if not text.endswith("\ufffd") and text.startswith(shown):   # else: an incomplete UTF-8 sequence
            sys.stdout.write(text[len(shown):])
            sys.stdout.flush()
            shown = text
        if ended:
            break
    err = p.stderr.read().decode(errors="replace")
    p.wait()
    print()
    if ended:
        n = len(ids) - len(prompt_ids) - 1          # (the end token not counted)
        return f"{n} tokens in {time.time() - t0:.1f} s, stopped at an end token"
    if p.returncode:
        print(f"sa-llm-run failed ({p.returncode}): {err[-2000:]}")
        return None
    lines = out.decode(errors="replace").strip().splitlines()
    wait = f"; first token after {first - t0:.1f} s (loading the module and parameters)" if first else ""
    return (lines[1] if len(lines) > 1 else "") + wait


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt", help="one prompt, then exit (default: read prompts interactively)")
    ap.add_argument("--max-new", type=int, default=64, help="tokens to generate at most")
    ap.add_argument("--chat", action="store_true", help="each prompt as one user message in the model's chat "
                    "template (model.json \"chat\"; default: plain text continued)")
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(HERE, "model.json")))
    if args.chat and not cfg.get("chat"):
        print("model.json has no chat template (a base model): plain text")
        args.chat = False
    mb = 64
    if os.path.exists(os.path.join(HERE, "board.txt")):
        mb = int(open(os.path.join(HERE, "board.txt")).read().split()[-1])
    args_file = os.path.join(HERE, "sa_args.txt")
    extra = open(args_file).read().split() if os.path.exists(args_file) else []
    ctx = cfg.get("context", 256)
    # the window first: a large contiguous CMA block needs the free memory the
    # tokenizer's Python objects would otherwise take (CMA pages lent to the
    # kernel must migrate out)
    mm, buf, env = BL.start(os.path.join(HERE, "picorv32.bit"), os.path.join(HERE, "rt_fw.bin"), mb=mb, ring=16)
    try:
        encode, decode = load_tokenizer(cfg)
        while True:
            if args.prompt is not None:
                text = args.prompt
            else:
                try:
                    text = input("\nprompt> ")
                except EOFError:
                    break
                if text.strip() in ("", "q", "quit", "exit"):
                    if text.strip():
                        break
                    continue
            if args.chat:                              # (the template's ids carry the BOS)
                ids = cfg["chat"][0] + encode(text)[1 if cfg.get("bos") is not None else 0:] + cfg["chat"][1]
            else:
                ids = encode(text)
            if len(ids) >= ctx:
                print(f"(the prompt has {len(ids)} tokens: keeping the last {ctx // 2})")
                ids = ids[-(ctx // 2):]
            n = max(0, min(args.max_new, ctx - len(ids)))
            if not args.chat:                          # (the text continued: the prompt first)
                sys.stdout.write(decode(ids))
                sys.stdout.flush()
            stats = generate(ids, n, cfg, decode, env, extra)
            if stats:
                print(f"[{len(ids)} prompt tokens; {stats}]")
            if args.prompt is not None:
                break
    except KeyboardInterrupt:
        print()
    finally:
        BL.stop(mm, buf)
    return 0


if __name__ == "__main__":
    sys.exit(main())

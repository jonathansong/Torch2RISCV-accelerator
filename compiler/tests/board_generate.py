#!/usr/bin/env python3
"""Interactive text generation on the PYNQ-Z1: a decoder compiled by IREE for the
sa device (the C3 / C5.5 board bundles), run by the armv7 sa-llm-run.

The overlay and rt_fw are started once; then each prompt is tokenized on the
ARM, fed to sa-llm-run (a fresh KV cache per prompt), and the generated text is
printed as the tokens arrive (greedy, until the model's end token, --max-new or
the context length).

Files (the board bundle): this script, board_launcher.py, pynq_matmul.py,
picorv32.bit / .hwh, rt_fw.bin, sa-llm-run, sa.vmfb, sa_packed.irpa,
model.json ({"tokenizer": "hf" | "llama2", "bos": id or null, "stop": id or
null, "context": n}), the tokenizer (tokenizer.json + hf_tokenizer.py, or
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
    """Runs sa-llm-run, printing the text as it comes; returns its stats line."""
    cmd = [os.path.join(HERE, "sa-llm-run"), "--device=sa", f"--module={HERE}/sa.vmfb",
           f"--parameters=model={HERE}/sa_packed.irpa", "--tokens=" + ",".join(map(str, prompt_ids)),
           f"--generate={max_new}"] + extra
    if cfg.get("stop") is not None:
        cmd.append(f"--stop_token={cfg['stop']}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    out, cur, ids = bytearray(), b"", []
    shown = decode(prompt_ids)
    t0 = time.time()
    first = None
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
        if gen and cfg.get("stop") is not None and gen[-1] == cfg["stop"]:
            gen = gen[:-1]
        if not gen:
            continue
        first = first or time.time()
        text = decode(prompt_ids + gen)
        if not text.endswith("\ufffd") and text.startswith(shown):   # else: an incomplete UTF-8 sequence
            sys.stdout.write(text[len(shown):])
            sys.stdout.flush()
            shown = text
    err = p.stderr.read().decode(errors="replace")
    p.wait()
    print()
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
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(HERE, "model.json")))
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
            ids = encode(text)
            if len(ids) >= ctx:
                print(f"(the prompt has {len(ids)} tokens: keeping the last {ctx // 2})")
                ids = ids[-(ctx // 2):]
            n = max(0, min(args.max_new, ctx - len(ids)))
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

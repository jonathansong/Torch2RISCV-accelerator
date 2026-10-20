#!/usr/bin/env python3
"""Interactive text generation on the board: a decoder compiled by IREE for the
sa device (the C3 / C5.5 board bundles), run by the aarch64 sa-llm-run.

The overlay and rt_fw are started once; then each prompt is tokenized on the
ARM, fed to sa-llm-run (a fresh KV cache per prompt), and the generated text is
printed as the tokens arrive (greedy, until the model's end token, --max-new or
the context length). --chat: a conversation in the model's chat template, the
earlier turns (their generated ids) resent with each message (a fresh KV cache
per message, so the whole conversation is prefilled again); the oldest turns
are dropped when the context is full; "/new" starts over.

Files (the board bundle): this script, board_launcher.py, pynq_matmul.py,
picorv32.bit / .hwh, rt_fw.bin, sa-llm-run, sa.vmfb, sa_packed.irpa,
model.json ({"tokenizer": "hf" | "llama2", "bos": id or null, "stop": id or
null, "context": n, optional "stops": [ids] (every end token), "chat":
[ids before, ids after] one user message and "chat_turn": [ids between a user
message and the assistant's answer, ids between the answer and the next user
message] for the earlier turns; the generic-frontend bundles':
stage_model_json.py}), the tokenizer (tokenizer.json + hf_tokenizer.py, or
tokenizer.bin + tokenizer.py), optional board.txt (window MB, last field) and
sa_args.txt (extra sa-llm-run arguments).

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && \\
        cd /home/ubuntu/<bundle> && python3 board_generate.py [--chat] [--max-new 64] [--prompt "Once upon a time"]'
(the KV260 has no xrt_setup.sh; under sudo bash -c, ~ is root's home: an absolute path)
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
    """Runs sa-llm-run, printing the text as it comes; returns (its stats line,
    the generated ids without the end token).
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
    gen = [t for t in ids[len(prompt_ids):] if t not in stops]
    if ended:
        return f"{len(gen)} tokens in {time.time() - t0:.1f} s, stopped at an end token", gen
    if p.returncode:
        print(f"sa-llm-run failed ({p.returncode}): {err[-2000:]}")
        return None, gen
    lines = out.decode(errors="replace").strip().splitlines()
    wait = f"; first token after {first - t0:.1f} s (loading the module and parameters)" if first else ""
    return (lines[1] if len(lines) > 1 else "") + wait, gen


def conversation(cfg, turns, msg):
    """The chat's ids: the earlier turns [(user ids, answer ids)] and the new
    message; without "chat_turn" (a template that gave no multi-turn form) the
    message alone."""
    pre, gen = cfg["chat"]
    hist, nxt = cfg.get("chat_turn") or (None, None)
    ids = list(pre)
    for u, a in (turns if hist is not None else []):
        ids += u + hist + a + nxt
    return ids + msg + gen


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt", help="one prompt, then exit (default: read prompts interactively)")
    ap.add_argument("--max-new", type=int, default=64, help="tokens to generate at most")
    ap.add_argument("--chat", action="store_true", help="a conversation in the model's chat template (model.json "
                    "\"chat\"; default: plain text continued); /new starts over")
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(HERE, "model.json")))
    if args.chat and not cfg.get("chat"):
        print("model.json has no chat template (a base model): plain text")
        args.chat = False
    if args.chat:   # the end of an assistant turn (Llama <|eot_id|>, Qwen3 <|im_end|>) as sa-llm-run's own stop
        end = (cfg.get("chat_turn") or [None, cfg["chat"][1]])[1][:1]
        if end and end[0] in (cfg.get("stops") or []):
            cfg = dict(cfg, stop=end[0])
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
        turns = []                                     # (--chat: [(user ids, answer ids)])
        if args.chat:
            print("chat: " + ("each message with the earlier turns, /new starts over" if cfg.get("chat_turn")
                              else "one message at a time (no multi-turn form in model.json)"))
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
                if args.chat and text.strip() == "/new":
                    turns = []
                    print("(a new conversation)")
                    continue
            if args.chat:                              # (the template's ids carry the BOS)
                msg = encode(text)[1 if cfg.get("bos") is not None else 0:]
                ids = conversation(cfg, turns, msg)
                dropped = 0
                while turns and len(ids) + min(args.max_new, 64) > ctx:   # (room for an answer)
                    turns = turns[1:]
                    dropped += 1
                    ids = conversation(cfg, turns, msg)
                if dropped:
                    print(f"(the context of {ctx} is full: the oldest {dropped} turn{'s' * (dropped > 1)} dropped)")
            else:
                ids = encode(text)
            if len(ids) >= ctx:
                print(f"(the prompt has {len(ids)} tokens: keeping the last {ctx // 2})")
                ids = ids[-(ctx // 2):]
            n = max(0, min(args.max_new, ctx - len(ids)))
            if not args.chat:                          # (the text continued: the prompt first)
                sys.stdout.write(decode(ids))
                sys.stdout.flush()
            stats, answer = generate(ids, n, cfg, decode, env, extra)
            if stats:
                print(f"[{len(ids)} prompt tokens{f' ({len(turns)} earlier turns)' if args.chat and turns else ''}; {stats}]")
            if args.chat and stats:
                turns.append((msg, answer))
            if args.prompt is not None:
                break
    except KeyboardInterrupt:
        print()
    finally:
        BL.stop(mm, buf)
    return 0


if __name__ == "__main__":
    sys.exit(main())

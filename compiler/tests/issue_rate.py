#!/usr/bin/env python3
"""Command issue rate of a build's decode executables (docs/riscv_backend_design.md
§2): per dispatch kind the commands and the cycle model's cycles (four-port
overlay, T = 64), so cycles per command: the rate a core issuing the commands
must sustain.

    compiler/tests/issue_rate.py build/hfgen/qwen3_l2_e2e_qf
"""
import sys, os, re, tempfile, numpy as np, collections
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "llm"))
import attn_cycles as AC, dispatch_check as DC, perf_model as PM, golden as G
a = AC.arch(os.path.join(REPO, "build/perfmodel/qwen3_dyn_cal.json")); tmp = tempfile.mkdtemp(prefix="ir_")
def run(src, blob):
    cap = []; orig = DC.SaFuncSim.run_list
    def rl(self, addr):
        self.trace = []
        try: return orig(self, addr)
        finally: cap.append(self.trace); self.trace = None
    DC.SaFuncSim.run_list = rl
    try: DC.check(src, blob, 16, 64)
    finally: DC.SaFuncSim.run_list = orig
    return [PM.decode(a, np.asarray(r, np.uint64)) for tr in cap for r in tr]
build = sys.argv[1]
tot = collections.Counter(); rows = []
for f in sorted(os.listdir(f"{build}/sa_sources")):
    if not f.startswith("module_main$"): continue
    t = open(f"{build}/sa_sources/{f}").read()
    kind = re.sub(r"\d+", "N", re.search(r"export public @main\$async_dispatch_\d+_(\w+)", t).group(1))
    s2 = f"{tmp}/{f}"; open(s2, "w").write(G.sa_only(t))
    blob, err = G.compile_one(s2, [], s2 + ".sadesc")
    if blob is None: continue
    cmds = run(s2, blob)
    cyc, busy = PM.run_list(a, cmds, gemv=True)
    n = sum(c.op != PM.OP_FENCE for c in cmds)
    rows.append((kind, n, cyc))
g = collections.defaultdict(lambda: [0, 0])
for k, n, c in rows:
    key = "GEMV" if k.startswith("matvec") else "attention" if k.startswith(("batch_matmul", "matmul_like")) else "reduction" if k.startswith("reduction") else "other"
    g[key][0] += n; g[key][1] += c
for k, (n, c) in g.items(): print(f"{k:10s} commands {n:6d} cycles {c:9.0f}  cycles per command {c / max(n, 1):7.1f}")
n = sum(r[1] for r in rows); c = sum(r[2] for r in rows)
print(f"all       commands {n:6d} cycles {c:9.0f}  cycles per command {c / n:7.1f}  (unique executables, T = 64)")
print("min cycles/command per executable:", sorted((round(r[2] / max(r[1], 1)), r[0][:30]) for r in rows)[:5])

# PYNQ-Z1 baselines (frozen at `v1.0-pynq-z1`)

The reference the KV260 port is checked against without the Z1 board
(docs/kv260_upgrade_plan.md §7.2). Made on 2026-10-04 (bundles from commit 9e1f5cf's compiler, the runtime of ce3cf11):
`compiler/scripts/deploy_z1_freeze.sh` staged every board test,
`compiler/tests/board_regress.py` ran them on the PYNQ-Z1 (L2 bitstream,
D = 8, 50 MHz; all bit-exact, `REGRESSION PASS`), and
`compiler/tests/make_z1_baselines.py` wrote these files.

| File | Content |
|---|---|
| `golden_manifest.txt` | The golden corpus (`compiler/tests/golden.py`) re-recorded with this compiler: per case and dispatch source its key and the sha256 of its descriptors. 4002 unique dispatches in 12 cases; the 26 marked `-` (case `hf_smollm2_q`, the generic frontend before its rewrites) do not compile for sa by design and run on the host |
| `dispatch_check.json` | `dispatch_check.py --dirty` (oracle vs functional simulator, every call site) for the five model builds: all OK except two `UNSUPPORTED` in hfgen (prefill's Q RoPE, the two dispatches on the host) |
| `stories15m_tokens.txt` | The board's tokens: stories15M decode (c3, also bit-exact with the hand-written L5 path) and prefill + decode (c6p_stories) |
| `smollm2_tokens.txt` | The board's tokens: SmolLM2-135M decode (c55) and prefill + decode, hand-written qhf (c6p_smollm2) and unmodified HF (hfgen) |
| `perf.md` | Speeds, DMA bandwidth, per-export profiles (top 15), resources and timing of the bitstream |
| `z1_profile.md` | The decode / prefill time breakdown per dispatch type from the unit's event counters (`SA_PROFILE_PERF`, `compiler/tests/z1_profile.py`): loads and their floor, EX, VE / SFU, ST, fixed overhead, overlap |

Large outputs (logits, full logs) are not in git; they go with the release.
On the KV260 (K1a: the same accelerator), the token files must match exactly;
the descriptors of the D = 8 / default-memory target must match
`golden_manifest.txt` while the compiler is unchanged.
`compiler/tests/compare_z1_baselines.py <results>` checks a board run.
K1a (50 MHz, 2026-10-08) and K1b (100 MHz, 2026-10-08): all five runs
identical (KV260/README.md).

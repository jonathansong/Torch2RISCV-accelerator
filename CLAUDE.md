# Torch2RISCV-accelerator: working rules and context

PyTorch / HuggingFace LLMs compiled by IREE with this repository's `sa` backend
plugin for a systolic-array accelerator on the AMD Kria KV260 (ARM host,
PicoRV32 control core). Start with `README.md`, then
[`docs/iree_compiler_plan.md`](docs/iree_compiler_plan.md) (the compiler) and
[`docs/fusion_design.md`](docs/fusion_design.md) (the current work).

## How to work with the user

- **Reply in Chinese.** Code, comments, commit messages and committed docs in English.
- **Never launch Vivado** (synthesis, implementation, OOC runs). Give the user the
  command; they run it. xsim / iverilog are fine.
- **Long builds** (a full IREE / LLVM build): give the command; the user runs it.
  Incremental builds you may run yourself, under the memory cap below.
- **Memory**: the machine has 15 GB of RAM. Run any big job (iree-compile of a full
  model, the sim, golden, incremental builds) as
  `systemd-run --user --unit=<name> -p MemoryMax=10G -p MemorySwapMax=0 --working-directory=$PWD ...`,
  one big job at a time. systemd-run expands `$VAR` itself (put loops in a script
  file); `--wait` does not pipe stdout (redirect to a log in the script); pass
  environment with `-E NAME=value`. A unit name used before needs
  `systemctl --user reset-failed <name>`.
- **The board**: `ubuntu@10.42.0.216` (KV260), reachable by ssh key
  (`ssh -o BatchMode=yes ubuntu@10.42.0.216`): copy bundles with rsync, run the tests
  and copy the results back yourself (into `build/deploy_*/results`). sudo asks for a
  password the user gives; never write it into files. Tests as root need
  `source /etc/profile.d/pynq_venv.sh` (and `xrt_setup.sh`) first, as
  `run_board_tests.sh` does, and `/home/ubuntu` paths (`~` is `/root` under sudo).
  Ask before reboots or changes to the boot configuration.
- **Git**: commit after verification, never rewrite history, **push only when the
  user asks**. End commit messages with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Upstream PRs** (IREE: iree-org/iree #25037, #25038, #25059 from the fork
  `jonathansong/iree`, clone `/home/jon/Projects/iree-pr-work`): the DCO
  `Signed-off-by` is the user's own (`! git commit --amend -s`); you push after.
- **Docs**: `docs/*.md` in English are canonical and committed; the `*.zh.md`
  copies are local and not updated. `docs/results_v2.2.md` and
  `docs/results_v2.3.md` are frozen release records.
- **Leave alone**: `.vscode/`, the untracked `compiler/patches/iree-pr1/2/3`,
  `Hummingbird.pdf`; the sibling repo `../triton-kv260` (another session's project).
- **Before deleting or overwriting** anything under `build/`, check that no build
  or job is running there (`pgrep -af 'iree-compile|ninja|sa_sim_server|vivado'`),
  and use explicit absolute paths. Board results copied back live inside the
  bundle directories (`build/deploy_*/results`): do not delete them when
  rebuilding a bundle.

## Environment

- `source compiler/env.sh`; Python is `build/iree/venv/bin/python` (`$SA_PY`).
- The board's configuration: `SA_D=16 SA_GEMV_PORTS=4` (test scripts default to
  D = 8 otherwise).
- IREE sources `build/iree/src` (pinned `e4a3b0405d` plus
  `compiler/patches/iree/0001`–`0005`, applied by
  `compiler/scripts/fetch_iree_sources.sh`), build `build/iree/build-compiler`.
  Incremental build of the compiler after a plugin change:
  `cmake --build $IREE_BUILD -j 4 --target iree-compile iree-opt` (under systemd-run).
- Compile a model: `compiler/scripts/compile_sa.sh <dir>` (defaults: executable
  generalization, SwiGLU, decode-attention and prefill-chunk fusion on;
  `SA_FUSE_FFN=0`, `SA_FUSE_ATTN=0`, `SA_FUSE_ATTN_CHUNK=0`, `SA_GENERALIZE=0` turn
  them off). The chunk kernel needs an export from `hf_generic` of 2026-10-19 on
  (an earlier export's prefill rows stay unfused).

## Verification (before every commit that touches the compiler)

- Lit: `$IREE_BUILD/llvm-project/bin/llvm-lit compiler/plugins/sa/test/`.
- Golden guard: `compiler/scripts/run_tests.sh golden` (lit, then
  `compiler/tests/golden.py`: the recorded dispatch corpus must compile to the same
  bytes). A deliberate codegen change: check which cases change, verify them, then
  `golden.py --record -j 4`.
- Per dispatch: `compiler/tests/dispatch_check.py <sources> <binaries> --t T [--dirty] [--pos 0]`
  (the oracle against the functional simulator; test T at block boundaries and
  `--pos 0`, rows with one key unmasked).
- End to end on the sim: `sa-llm-run` against `compiler/sim/sa_sim_server.py`;
  logits compared byte for byte with the unfused build (the fusions are bit-exact).
- Speed: `compiler/tests/perf_model.py` on traces (`--trace` of the sim server),
  calibrations in `build/perfmodel/*_cal.json`.
- Board: `run_board_tests.sh <names>` on the board (copy the current
  `compiler/scripts/run_board_tests.sh` there when it changed); rerunning a bundle
  name that passed before needs `sudo REDO=1 bash run_board_tests.sh ...`.

## Where the work stands (2026-10-20)

- Release `v2.2.1-kv260` (`docs/results_v2.2.md`): four models (SmolLM2-135M,
  Gemma-3-270M, Qwen3-0.6B, Llama-3.2-1B) on the board, bit-exact.
- Release `v2.3-kv260` (`docs/results_v2.3.md`, branch `kv260-v2.3`): the same with
  the three fusions on by default; bundles `build/deploy_hfgen_*c_pcpi`.
- Since then, cross-operator fusion (`docs/fusion_design.md`): SwiGLU (§4) and
  decode attention (§6, kernel `attn_layer`, region pass
  `FormAttnRegions.cpp` after IREE's region formation via patch 0004), both on by
  default and accepted on the board (§4.9, §6.4.8).
- Prefill chunk kernel (§6.7, 2026-10-19/20): `--iree-sa-fuse-attn-chunk`, kernel
  `attn_chunk`, IREE patch 0005; accepted on the board on all four models (bundles
  `*c`, §6.7.1): prefill −9% to −25% (40 tokens), −20% to −45% (2032 tokens),
  decode unchanged. On by default since 2026-10-20 (`SA_FUSE_ATTN_CHUNK=0` off).
- Fusion closed (2026-10-20, fusion_design.md §7): the chains left (residual +
  RMSNorm + projection, prefill SwiGLU, the k / v path, q_proj / o_proj) are each
  worth a few percent at most and deferred; decode is bound by the GEMV's weight
  streaming. Next levers: W4A8 (weights), the hardware side (LD / GEMV width,
  command issue), the results for the talk (2026-11-15).
- Lifetime-based local memory planning (3a / 3b): designed, deferred
  (`docs/local_memory_planning.md`).
- MoE (2026-10-21..25): `docs/moe_implementation_plan.md` M0–M5 (Mixtral layers on the
  board; M5's compiler-owned HAL chain removed 2026-10-25, the M0–M4 code too, at the tag `moe-m0-m4-final`); then the native
  lowering, `docs/moe_native_lowering_design.md` S0–S7: `hf_generic.QMixtralMoe` (W8A8),
  expert tables packed at compile time and selected by a dynamic slice whose DDR offset the
  device computes (`PackLinearWeights`, `SahlToSahw` `applyExpertBases`), the Top-2 as
  embedded RISC-V code (`rv_only`; runtime runs it whatever `SA_RV` says, the sim through
  `SA_RV=<dir with rvcode_host.so>` `SA_RV_ONLY=rv_only`). tiny-mixtral (random weights)
  accepted on the board 2026-10-25 (`deploy_moe.sh`, `run_board_tests.sh moe`): bit-exact
  with the sim, decode 16.0 ms per step. Checks: `test_hf_moe.py`, `test_moe_sim.py`,
  `dispatch_check.py --ids .. --rv <bundle>/rv`, golden cases `moe_mixtral`, `moe_granite`. Since
  2026-10-25 a trained checkpoint, Granite-3.1-1B-A400M (Top-8 of 32): host quality (§21), decode
  estimate (§22), Top-k of up to 64 experts in the backend (§23); **accepted on the board 2026-10-25
  (§24)**: 24 layers decode-only, bit-exact with the sim, 104.4 ms per step (9.58 tok/s), pool now
  1536 MiB; quality on the board (§25, K per channel, 2048 cache): PPL 6.528 vs fp32 6.386, top-1
  92.1%, KL 0.030. Open: the k-expert kernel (v2, up to ~18%), the Top-8 calls (8.8%), prefill (M6).

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
  copies are local and not updated. `docs/results_v2.2.md`,
  `docs/results_v2.3.md` and `docs/results_v2.4.md` are frozen release records.
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

## Where the work stands

- Releases (frozen records): `v2.2.1-kv260` (`docs/results_v2.2.md`), `v2.3-kv260`
  (`docs/results_v2.3.md`, fusions on), `v2.4-kv260` (`docs/results_v2.4.md`, Granite MoE);
  branches `kv260-v2.x`. Board bundles of record: `build/deploy_hfgen_*c_pcpi`,
  `build/deploy_moe_granite24{,q}`; the board's u-dma-buf pool is 1536 MiB.
- Design records: `docs/fusion_design.md` (fusion, closed), `docs/moe_native_lowering_design.md`
  (MoE), `docs/local_memory_planning.md` (deferred), `docs/fp16_gemv_design.md` (FP16 GEMV, next).
- MoE on the sim needs `SA_RV=<dir with rvcode_host.so> SA_RV_ONLY=rv_only` (the Top-k is
  embedded RISC-V code, run whatever `SA_RV` says on the board).
- Next: `docs/future_work.md` (W4A8, MoE k-expert kernel and prefill, FP16 GEMV, CI;
  gains, effort and the proposed order).

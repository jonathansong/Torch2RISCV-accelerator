# compiler/: the MLIR / IREE compiler for the PYNQ-Z1 accelerator

This directory implements the end-to-end compiler of
[`docs/iree_compiler_plan.md`](../docs/iree_compiler_plan.md): PyTorch → iree-turbine → IREE
(with our `sa` target backend) → descriptor-list executables, run on the board by
an `sa` HAL driver through the command ring.

## Layout

| Path | What | Stage |
|---|---|---|
| `env.sh` | Paths and settings; `source compiler/env.sh` | setup |
| `scripts/fetch_iree_sources.sh` | Adds the compiler's submodules (llvm-project, torch-mlir) to the IREE source tree | setup |
| `scripts/build_iree_compiler.sh` | Builds iree-compile from source with our plugins | setup |
| `scripts/build_sa_runtime.sh` | Builds the IREE runtime with the `sa` HAL driver, `host` or `armv7` (about a minute) | C1 |
| `scripts/deploy_c1.sh` | Stages the C1 board test in `build/deploy_c1` | C1 |
| `scripts/deploy_c2.sh` | Runs the C2 host test and stages the board test in `build/deploy_c2` | C2 |
| `scripts/compile_sa.sh` | Compiles an exported model for the sa device (parameters imported, weights packed, packed parameters exported, dispatch sources dumped) | C3 |
| `scripts/deploy_c3.sh` | Runs the C3 host test and stages the board test in `build/deploy_c3` | C3 |
| `scripts/deploy_c55.sh` | Runs the C5.5 host test on SmolLM2-135M and stages the board test in `build/deploy_c55` | C5.5 |
| `plugins/sa/` | The `sa` HAL target plugin (C++). `dialect/`: the `sahl` (tile) and `sahw` (command) dialects. `transforms/`: the C5 pipeline (sa-to-sahl, sahl-to-sahw with the micro-kernels, sahw-fuse-ve, head split, register assignment), the sahw serializer, the target configuration. `target/`: the `sa` device and backend, the preprocessing passes (weight packing, cheap-producer cloning), `DescList`, `Layout`. `test/`: lit tests (iree-opt + FileCheck). `templates/reference.py`: the Python reference of the linear-layer schedule (test_c2 compares the commands) | C2–C5 |
| `frontend/` | `qllama.py` (the quantized llama, step for step `DeviceModel`), `export.py` (iree-turbine export, compile, compare, dispatch inventory, board bundle), `inventory/` (the dispatch inventories); `qhf.py` (HuggingFace Llama / Qwen3 decoders: config, safetensors, BPE tokenizer, fp32 and quantized models), `export_hf.py` (their export), `synth_hf.py` (random-weight HF models of any shape), `hf_tokenizer.py` (the BPE tokenizer, standard library only: also on the board) | C0, C5.5 |
| `runtime/` | `sa/`: the `sa` HAL driver (C; IREE external HAL driver) with `board` and `sim` transports; its own device and command buffer (C4) record a command buffer's dispatches and build one descriptor list per command buffer. `tools/`: sa-desc writer / reader (`sadesc.py`, format versions 1–3), the C1 test executable (`make_test_exec.py`). `test/`: `sa_hal_test.c` (HAL API test), `board_launcher.py` (PYNQ side on the board) | C1 |
| `sim/` | `sa_sim_server.py` (the `sim` transport's device: `llm/sa_funcsim.py` on a shared-memory DDR), `sa_board_emu.py` (rt_fw's ring emulated on the host, for the `board` transport) | C1 |
| `tests/` | `test_c2.py` / `test_c3.py` / `test_c5.py` / `test_c55.py` (host), `board_c2.py` / `board_llm.py` / `board_generate.py` (interactive) / `board_profile.py` (board); `oracle.py` (reference interpreter of dispatch IR with the device's numerics), `dispatch_check.py` (per-dispatch differential test), `summarize.py` | C2– |

## Environment

The IREE revision is fixed: **e4a3b0405d (IREE 3.11.0)**, the same as the pip
compiler and the board runtime of level L0 (`iree-sa/l0`). By default,
everything reuses what L0 set up under `build/iree/` (git-ignored):

| What | Where (default) | Used for |
|---|---|---|
| IREE source | `build/iree/src` | the compiler build (L0 has the runtime submodules; `fetch_iree_sources.sh` adds llvm-project and torch-mlir) |
| Python venv | `build/iree/venv`: torch 2.14 (CPU), iree-turbine 3.9, iree-base-compiler / runtime 3.11 | the frontend (C0); CPU-only compiles, including armv7 for the board |
| Compiler build | `build/iree/build-compiler` | `iree-compile` with the `sa` plugin |

Host tools: cmake ≥ 3.28, ninja, clang / clang++ / lld 18, ccache, git.

### Steps

```sh
source compiler/env.sh
compiler/scripts/fetch_iree_sources.sh                 # ~2-3 GB, a few minutes
compiler/scripts/build_iree_compiler.sh --configure    # fast: checks the CMake setup and the plugin
compiler/scripts/build_iree_compiler.sh                # long (1-2 h at 4 jobs): run it in your own terminal
```

At the end the script prints `iree-compile --iree-hal-list-target-backends`,
and the list must contain `sa`.

**What the build contains, and why it is kept small.** This machine has
15 GB of RAM and little free disk, so the build is cut down:
- Release build without debug information or assertions;
- input: torch only;
- target backends: `sa` plus `vmvx`. Compile-time constant evaluation runs on
  the `local` device with its host backend, which is then `vmvx`;
- without `llvm-cpu`, neither clang nor lld is built, and LLVM builds only
  the X86 code generator;
- `-j 4`, and one link at a time.

For mixed CPU and accelerator execution, set `SA_WITH_LLVM_CPU=1`. This
enables llvm-cpu for X86 and ARM and makes the build much larger. CPU-only
compiles for the board keep using the pip `iree-compile`.

**Disk space.** The build script refuses to start with less than 20 GB free
(set `SA_FORCE=1` to override). The sources take about 3 GB and the build
directory about 10–15 GB. ccache is capped at 8 GB.

## Status

- Setup: the frontend Python environment is checked. A turbine export of a
  small torch model (matmul + softmax), compiled with the pip `iree-compile`,
  matches torch.
- **iree-compile built from source** (2026-09-27): 5,261 build steps, 2 h
  20 min at 4 jobs (533 CPU-min). The build directory is 1.8 GB; the
  object files mostly live in ccache. `--iree-hal-list-target-backends`
  shows `sa`, `vmvx`, `vmvx-inline`.
  - With `--iree-hal-target-device=sa`, the C0 module runs the whole
    pipeline and every dispatch stops in the sa backend's
    `serializeExecutable` with the intended "not implemented yet (stage C2)"
    error. So registration, the device-to-backend wiring and serialization
    all work.
- The `sa` plugin registers `#hal.device.target<"sa">` and
  `#hal.executable.target<"sa", "sa-desc-v1", {d = 8}>` (option
  `--iree-sa-d`). Since C3 it compiles every dispatch of stories15M
  (`--iree-sa-allow-unsupported` turns missing templates into warnings, for
  development).
  Rebuilding iree-compile after plugin changes: `cmake --build
  $IREE_BUILD -j 4 --target iree-compile` (a few minutes with ccache).
- **C0 done** (docs/iree_compiler_plan.md §3.5):
  - stories15M exports with a dynamic attention length and the KV cache as
    mutable globals;
  - compiled with llvm-cpu, 12 steps match eager to 4e-7 and `DeviceModel`
    argmax 12/12;
  - the armv7 build passes on the board (two single-step cases);
  - the dispatch inventory is in `frontend/inventory/`: 241 calls and 65
    executables per token.

  ```sh
  python3 compiler/frontend/export.py --out build/c0/stories15M --armv7   # then on the board: bash run_c0.sh
  ```
- **C1 done** (docs/iree_compiler_plan.md §5.7): the `sa` HAL driver runs
  hand-built sa-desc-v1 executables through the plain IREE HAL API,
  bit-exact on sim, on the host ring emulation and on the board; device errors
  come back as failed submissions with the decoded status.

  ```sh
  compiler/scripts/build_sa_runtime.sh host
  python3 compiler/runtime/tools/make_test_exec.py --d 8 --out build/c1/d8
  python3 compiler/sim/sa_sim_server.py --d 8 --once &            # sim transport
  build/iree/build-sa-host/runtime/plugins/hal/drivers/sa/sa_hal_test build/c1/d8
  python3 compiler/sim/sa_board_emu.py --d 8 -- build/iree/build-sa-host/runtime/plugins/hal/drivers/sa/sa_hal_test build/c1/d8
  compiler/scripts/deploy_c1.sh     # board: python3 board_launcher.py -- ./sa_hal_test test_d8 1000
  ```
- **C2 done** (docs/iree_compiler_plan.md §6.7): torch int8 linear layers
  compile with the sa plugin (weights packed at compile time by
  `iree-sa-pack-linear-weights` + const-eval, exported to a new parameter
  archive; templates byte-identical to the Python reference) and run with
  `iree-run-module --device=sa`, bit-exact with `DeviceModel.linear` on sim
  (D = 8, 16) and on the board.

  ```sh
  python3 compiler/tests/test_c2.py [--d 16]       # host: export, compile, check, run on sim
  compiler/scripts/deploy_c2.sh                     # board: python3 board_c2.py
  ```
- **C3 done** (docs/iree_compiler_plan.md §6.9): the whole stories15M compiled
  by IREE for the sa device; every dispatch bit-exact with its IR (oracle) and
  the logits of every step bit-exact with DeviceModel(SfuExact) on sim and on
  the board, where the generated text equals the hand-written L5 path
  (12.6 tok/s).

  ```sh
  python3 compiler/tests/test_c3.py [--generate 8]    # export, compile, per-dispatch check, sim run
  compiler/scripts/deploy_c3.sh                       # board: python3 board_llm.py
  ```
- **C4 partly done, on hold** (docs/iree_compiler_plan.md §7.1): dynamic
  attention length, aggressive fusion (200 dispatches per token), uniform
  values computed once, one descriptor list per command buffer, and a list
  scheduler in the driver (chunk-0 weight prefetch inside earlier dispatches,
  FENCE masks from DDR read/write conflicts; sa-desc v3). Board: 2.591M
  cycles per token, 17.7 tok/s, bit-exact; the hand path is 2.294M (target
  within 10%: the SiLU / quantization fusion is deferred until after C5).

  ```sh
  python3 compiler/tests/test_c3.py --skip-export --out build/c4/fuse
  compiler/scripts/deploy_c3.sh --skip-export --out build/c4/fuse
  # board: python3 board_llm.py; python3 board_profile.py 40 [--batch] [NAME=VALUE...]
  ```
  Environment: `SA_PROFILE=1` (one list per dispatch, per-export cycles),
  `SA_PROFILE=batch` (per list), `SA_NO_BATCH=1` (one list per dispatch, to
  find a failing dispatch).
- **C5 done** (docs/iree_compiler_plan.md §8): the code generator is an
  MLIR pipeline (`plugins/sa/transforms/`): IREE bufferization, `iree-sa-to-sahl`
  (explicit DDR traffic, the `sahl` dialect), `iree-sahl-to-sahw` (local
  memory, one single-stage VE per operation, reductions, gathers, masks,
  dynamic lengths, contractions), `iree-sahw-fuse-ve`, `iree-sahw-split-head`,
  `iree-sahw-assign-registers`, the `sahw` serializer. The linear layer and
  attention are micro-kernels (`--iree-sa-ukernels=all|none|linear,attention`);
  without them a generic contraction lowering compiles everything.
  **C5.0–C5.4 done**: stories15M compiles with and without micro-kernels,
  bit-exact on sim and on the board (2.589M cycles / token with, 4.396M
  without); the hardware parameters come from the executable target
  configuration.
  **C5.5**: HuggingFace decoders with no model-specific code. SmolLM2-135M
  (30 layers, GQA 9/3) compiles to 230 dispatches, all bit-exact against the
  oracle, and runs end to end on sim (argmax equal to the eager quantized
  model with the device's EXP / RECIP / RSQRT); Qwen3-0.6B truncated to 2
  layers (QK-norm, head_dim 128, GQA 16/8) likewise on sim. SmolLM2 on the
  board: bit-exact with the sim, 1.90 tok/s. The device window
  is configurable (`--mb`; SmolLM2 needs 160.5 MB: parameter loads read
  straight into device memory, `sa-llm-run` prints the peak).
- **C6 in progress** (docs/iree_compiler_plan.md §8.12: models of the
  Qwen3-0.6B / Llama-3.2-1B class, for larger boards; sim here). Done: the
  target configuration cross-check (`--iree-sa-spad-kb`, `--iree-sa-acc-kb`;
  the functional simulator and `dispatch_check.py` take D and the memory
  sizes from each dispatch's target; stories15M bit-exact for D = 16 and
  larger SPAD / ACC) and K blocks (a contraction whose K exceeds one bank, one
  DMA row or the strip's source range accumulates K blocks in ACC); also
  DMAs beyond 64 KB and element-wise dispatches too large for ACC (lowered in
  pieces). A random-weight model with K = 16384 is bit-exact per dispatch and
  end to end. Next: C6.0 (the shared embedding stored once), C6.1 (Qwen3-0.6B,
  all 28 layers).

  ```sh
  python3 compiler/tests/test_c5.py                  # lit, both configurations, dispatch_check at T = 16, 80, 256
  python3 compiler/tests/test_c5.py --configs        # + other hardware (D = 16, larger SPAD / ACC: --iree-sa-d / -spad-kb / -acc-kb)
  python3 compiler/frontend/export_hf.py --model build/llm_cache/SmolLM2-135M --out build/c55/smollm2
  python3 compiler/tests/test_c55.py --model build/llm_cache/SmolLM2-135M --out build/c55/smollm2 --check
  compiler/scripts/deploy_c55.sh                     # board: python3 board_llm.py (window 168 MB; the board needs cma=320M in uEnv.txt, see the plan §8.7)
  # interactive generation on the board (either bundle: deploy_c3 = stories15M, deploy_c55 = SmolLM2-135M):
  #   python3 board_generate.py [--max-new 64] [--prompt "..."]   (a prompt per line; tokens printed as they come)
  python3 compiler/frontend/export_hf.py --model build/llm_cache/Qwen3-0.6B --out build/c55/qwen3_l2 --layers 2
  python3 compiler/tests/test_c55.py --model build/llm_cache/Qwen3-0.6B --out build/c55/qwen3_l2 --check --mb 384
  # a synthetic model beyond this board's shapes (K = 16384: K blocks; 64 KB vectors: pieces)
  python3 compiler/frontend/synth_hf.py --out build/llm_cache/synth-k16k --tokenizer build/llm_cache/SmolLM2-135M
  python3 compiler/frontend/export_hf.py --model build/llm_cache/synth-k16k --out build/c6/k16k
  python3 compiler/tests/test_c55.py --model build/llm_cache/synth-k16k --out build/c6/k16k --check --mb 64
  python3 compiler/runtime/tools/sadis.py <file.sadesc>   # disassembler
  $IREE_BUILD/llvm-project/bin/llvm-lit -v compiler/plugins/sa/test
  ```

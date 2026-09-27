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
| `plugins/sa/` | The `sa` HAL target plugin (C++): the `sa` device and backend, executable format `sa-desc-v1`. Today a skeleton that only registers them | C2–C5 |
| `frontend/` | `qllama.py` (the quantized llama, step for step `DeviceModel`), `export.py` (iree-turbine export, compile, compare, dispatch inventory, board bundle), `inventory/` (the dispatch inventories) | C0 |
| `runtime/` | The `sa` HAL driver (C), with `board` and `sim` transports (planned) | C1 |
| `sim/` | The simulator service on top of `llm/sa_funcsim.py` (planned) | C1 |
| `tests/` | End-to-end tests (planned) | all |

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
- The `sa` plugin is a skeleton. It registers `#hal.device.target<"sa">` and
  `#hal.executable.target<"sa", "sa-desc-v1", {d = 8}>` (option
  `--iree-sa-d`). Compiling a dispatch for `sa` stops with "descriptor
  generation is not implemented yet"; stage C2 fills this in.
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

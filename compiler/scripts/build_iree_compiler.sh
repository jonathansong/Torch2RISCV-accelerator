#!/usr/bin/env bash
# Configures and builds iree-compile from source with our plugins
# (IREE_CMAKE_PLUGIN_PATHS). Long: 1-2 hours at 4 jobs; run it in your own
# terminal. Re-running continues an interrupted build.
#
#   compiler/scripts/build_iree_compiler.sh              configure (first time) + build
#   compiler/scripts/build_iree_compiler.sh --configure  configure only (fast; checks the plugin CMake)
#
# Kept small for this machine (disk, RAM): Release without debug info, no
# assertions, input torch only, target backends sa (ours) + vmvx (compile-time
# constant evaluation runs on the "local" device, whose host backend is then
# vmvx), no tests / samples / Python bindings, clang + lld + ccache. Without
# llvm-cpu the build skips clang, lld and all LLVM code generators but X86.
# SA_WITH_LLVM_CPU=1 adds llvm-cpu (X86 + ARM) for mixed CPU / accelerator
# execution (much larger build). CPU-only compiles for the board's ARM keep
# using the pip iree-compile of L0 (the same IREE revision).
set -euo pipefail
source "$(dirname "$0")/../env.sh"
configure_only=0
[ "${1:-}" = "--configure" ] && configure_only=1

for d in llvm-project torch-mlir; do
  if [ ! -f "$IREE_SRC/third_party/$d/.git" ] && [ ! -d "$IREE_SRC/third_party/$d/.git" ]; then
    echo "error: $IREE_SRC/third_party/$d missing; run compiler/scripts/fetch_iree_sources.sh" >&2
    exit 1
  fi
done
avail=$(df -BG --output=avail "$SA_REPO" | tail -1 | tr -dc 0-9)
echo "free disk ${avail} GB (the build needs about 15-20 GB)"
if [ "$configure_only" = 0 ] && [ "$avail" -lt 20 ]; then
  echo "error: less than 20 GB free; free disk space first (or set SA_FORCE=1)" >&2
  [ "${SA_FORCE:-0}" = 1 ] || exit 1
fi

if [ "${SA_WITH_LLVM_CPU:-0}" = 1 ]; then
  llvm_cpu='-DIREE_TARGET_BACKEND_LLVM_CPU=ON -DIREE_DEFAULT_CPU_LLVM_TARGETS=X86;ARM'
else
  llvm_cpu='-DIREE_TARGET_BACKEND_LLVM_CPU=OFF -DLLVM_TARGETS_TO_BUILD=X86'
fi
if [ ! -f "$IREE_BUILD/build.ninja" ] || [ "$configure_only" = 1 ]; then
  cmake -G Ninja -S "$IREE_SRC" -B "$IREE_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_LLD=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_INSTALL_PREFIX="$IREE_INSTALL" \
    -DPython3_EXECUTABLE="$SA_PY" \
    -DIREE_ENABLE_ASSERTIONS=OFF \
    -DIREE_BUILD_COMPILER=ON \
    -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_BUILD_PYTHON_BINDINGS=OFF -DIREE_BUILD_BINDINGS_TFLITE=OFF \
    -DIREE_BUILD_DOCS=OFF \
    -DIREE_INPUT_TORCH=ON -DIREE_INPUT_STABLEHLO=OFF -DIREE_INPUT_TOSA=OFF \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF -DIREE_TARGET_BACKEND_VMVX=ON \
    $llvm_cpu \
    -DIREE_TARGET_BACKEND_LLVM_CPU_WASM=OFF \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DIREE_CMAKE_PLUGIN_PATHS="$SA_PLUGIN_PATHS" \
    -DCMAKE_JOB_POOLS="link=$SA_LINK_JOBS" -DCMAKE_JOB_POOL_LINK=link
fi
[ "$configure_only" = 1 ] && { echo "configured: $IREE_BUILD"; exit 0; }

ccache --max-size=8G >/dev/null
time cmake --build "$IREE_BUILD" -j "$SA_JOBS" --target iree-compile iree-opt iree-run-module
"$IREE_BUILD/tools/iree-compile" --version
echo "== target devices / backends (must include sa)"
"$IREE_BUILD/tools/iree-compile" --iree-hal-list-target-backends | sed -n 1,40p
du -sh "$IREE_BUILD"

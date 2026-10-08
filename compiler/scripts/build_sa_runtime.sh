#!/usr/bin/env bash
# Builds the IREE runtime with the sa HAL driver (compiler/runtime, stage C1):
# a runtime-only IREE build (no compiler; minutes) with our runtime plugin
# (IREE_CMAKE_PLUGIN_PATHS -> compiler/runtime/iree_runtime_plugin.cmake) and
# -DIREE_EXTERNAL_HAL_DRIVERS=sa, so iree-run-module & co. know the "sa" driver.
#
#   compiler/scripts/build_sa_runtime.sh host     -> build/iree/build-sa-host
#   compiler/scripts/build_sa_runtime.sh aarch64  -> build/iree/build-sa-aarch64 (KV260 A53, static)
#   (the PYNQ-Z1's armv7 build: pynq-z1 branch)
#
# Targets: sa_hal_test (the C1 test), iree-run-module.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
which=${1:-host}
common=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF
        -DIREE_BUILD_SAMPLES=OFF -DIREE_BUILD_PYTHON_BINDINGS=OFF
        -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON
        -DIREE_CMAKE_PLUGIN_PATHS="$SA_COMPILER/runtime" -DIREE_EXTERNAL_HAL_DRIVERS=sa)
case "$which" in
  host)
    B=$SA_REPO/build/iree/build-sa-host
    cmake -S "$IREE_SRC" -B "$B" "${common[@]}" -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 >/dev/null
    ;;
  aarch64)
    B=$SA_REPO/build/iree/build-sa-aarch64
    cmake -S "$IREE_SRC" -B "$B" "${common[@]}" \
      -DCMAKE_TOOLCHAIN_FILE="$SA_COMPILER/runtime/toolchains/aarch64-linux-gnu.cmake" \
      -DIREE_HOST_BIN_DIR="$SA_REPO/build/iree/install-host/bin" \
      -DIREE_HAL_EXECUTABLE_LOADER_DEFAULTS=OFF -DIREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF=ON \
      -DIREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE=ON -DIREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS=OFF \
      -DIREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF=ON >/dev/null
    ;;
  *) echo "usage: $0 host|aarch64" >&2; exit 2 ;;
esac
cmake --build "$B" -j "$SA_JOBS" --target sa_hal_test sa-llm-run iree-run-module 2>&1 | grep -v dlopen | tail -n 30 || true
echo "built: $(find "$B" -name sa_hal_test -type f) $B/tools/iree-run-module"

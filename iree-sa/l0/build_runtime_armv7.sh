#!/usr/bin/env bash
# L0 (docs/llm_inference_plan.md §4.4): build the IREE runtime tools for the
# PYNQ-Z1 ARM (armv7-a hard float, static) at the revision of the pip IREE
# compiler, and stage a board bundle with the test model.
#
#   iree-sa/l0/build_runtime_armv7.sh          (from the repository root)
#   scp -r build/deploy_iree_l0 xilinx@<board>:/home/xilinx/iree_l0
#
# Needs: clang-18, cmake, ninja, the Ubuntu armhf cross sysroot (see
# toolchain-armv7hf.cmake), and build/iree/venv with iree-base-compiler,
# iree-base-runtime, iree-turbine and torch (CPU).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
W=$ROOT/build/iree
PY=$W/venv/bin/python
REV=$($PY -c "import iree.compiler.version as v; print(v.REVISIONS['IREE'])")

if [ ! -d "$W/src/.git" ]; then
    git init -q "$W/src"
    git -C "$W/src" remote add origin https://github.com/iree-org/iree
fi
if [ "$(git -C "$W/src" rev-parse HEAD 2>/dev/null)" != "$REV" ]; then
    git -C "$W/src" fetch -q --depth 1 origin "$REV"
    git -C "$W/src" checkout -q FETCH_HEAD
    (cd "$W/src" && bash build_tools/scripts/git/update_runtime_submodules.sh)
fi

common=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF
        -DIREE_BUILD_SAMPLES=OFF -DIREE_BUILD_PYTHON_BINDINGS=OFF)
cmake -S "$W/src" -B "$W/build-host" "${common[@]}" -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_INSTALL_PREFIX="$W/install-host" >/dev/null
cmake --build "$W/build-host" --target install -j 8 >/dev/null
cmake -S "$W/src" -B "$W/build-armv7" "${common[@]}" -DCMAKE_TOOLCHAIN_FILE="$ROOT/iree-sa/l0/toolchain-armv7hf.cmake" \
      -DIREE_HOST_BIN_DIR="$W/install-host/bin" \
      -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
      -DIREE_HAL_EXECUTABLE_LOADER_DEFAULTS=OFF -DIREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF=ON \
      -DIREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE=ON -DIREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS=OFF \
      -DIREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF=ON >/dev/null
cmake --build "$W/build-armv7" --target iree-run-module iree-benchmark-module iree-cpuinfo -j 8 2>&1 | grep -v dlopen || true

"$PY" "$ROOT/iree-sa/l0/export_and_compile.py" --out "$W/l0" | grep -v "^ \|^[{}]"
D=$ROOT/build/deploy_iree_l0
rm -rf "$D" && mkdir -p "$D"
for t in iree-run-module iree-benchmark-module iree-cpuinfo; do
    llvm-strip-18 -o "$D/$t" "$W/build-armv7/tools/$t" 2>/dev/null || cp "$W/build-armv7/tools/$t" "$D/"
done
cp "$W/l0/model_armv7.vmfb" "$W/l0/input.npy" "$W/l0/expected.npy" "$ROOT/iree-sa/l0/run_iree_l0.sh" "$D/"
"$PY" "$ROOT/iree-sa/l0/op_tests.py" --out "$D/ops" | grep "op tests"
echo "IREE $REV -> $D"; ls -la "$D"

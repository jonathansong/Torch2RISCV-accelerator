# Cross-compile the IREE runtime with the sa HAL driver for the Kria KV260
# (K26 SOM: Cortex-A53, aarch64; docs/kv260_upgrade_plan.md K1a) with the
# host's clang and the Ubuntu arm64 cross sysroot (libc6-dev-arm64-cross,
# binutils-aarch64-linux-gnu, gcc-*-aarch64-linux-gnu for crt / libgcc).
# Linked statically, as for the PYNQ-Z1 (toolchain-armv7hf.cmake): the host
# sysroot's glibc may be newer than the board image's.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER clang-18)
set(CMAKE_CXX_COMPILER clang++-18)
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)
set(AARCH64_FLAGS "-mcpu=cortex-a53")
set(CMAKE_C_FLAGS_INIT "${AARCH64_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${AARCH64_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${AARCH64_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static")
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

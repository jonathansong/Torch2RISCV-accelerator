# Cross-compile for the PYNQ-Z1 ARM (Cortex-A9, armv7-a + NEON, hard float)
# with the host's clang and the Ubuntu armhf cross sysroot
# (libc6-dev-armhf-cross, libstdc++-*-dev-armhf-cross, binutils-arm-linux-gnueabihf).
# Binaries are linked statically: the host sysroot's glibc (2.39) is newer
# than the board's (Ubuntu 22.04, 2.35).
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7-a)
set(CMAKE_C_COMPILER clang-18)
set(CMAKE_CXX_COMPILER clang++-18)
set(CMAKE_C_COMPILER_TARGET arm-linux-gnueabihf)
set(CMAKE_CXX_COMPILER_TARGET arm-linux-gnueabihf)
set(ARMV7_FLAGS "-march=armv7-a -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard")
set(CMAKE_C_FLAGS_INIT "${ARMV7_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${ARMV7_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${ARMV7_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static")
set(CMAKE_FIND_ROOT_PATH /usr/arm-linux-gnueabihf)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

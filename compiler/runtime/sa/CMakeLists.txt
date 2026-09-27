# sa HAL driver (docs/iree_compiler_plan.md §5), built as an IREE external HAL
# driver: iree_runtime_plugin.cmake registers it; enabled with
# -DIREE_EXTERNAL_HAL_DRIVERS=sa (compiler/scripts/build_sa_runtime.sh).

# Libraries below this directory are named sa_runtime_* and go into the
# runtime's export set (the drivers library that registers them is exported).
iree_setup_c_src_root(
  PACKAGE_ROOT_PREFIX sa_runtime
  DEFAULT_EXPORT_SET Runtime
  DEFAULT_INSTALL_COMPONENT IREEDevLibraries-Runtime
)

iree_cc_library(
  NAME
    hal_driver
  HDRS
    "sa_context.h"
    "sa_loader.h"
  SRCS
    "sa_context.c"
    "sa_driver_module.c"
    "sa_loader.c"
    "sa_transport_board.c"
    "sa_transport_sim.c"
  DEPS
    iree::base
    iree::hal
    iree::hal::local
    iree::hal::local::executable_loader
    iree::hal::drivers::local_sync::sync_driver
  PUBLIC
)

# C1 test program (compiler/runtime/test/sa_hal_test.c)
add_executable(sa_hal_test ${CMAKE_CURRENT_SOURCE_DIR}/../test/sa_hal_test.c)
target_include_directories(sa_hal_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sa_hal_test PRIVATE sa_runtime::hal_driver iree::hal)

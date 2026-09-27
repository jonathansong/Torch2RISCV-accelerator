# The sahw dialect (command level of the sa backend, docs/iree_compiler_plan.md §8.3).
iree_setup_c_src_root(PACKAGE_ROOT_PREFIX sa)

iree_tablegen_library(
  NAME
    SahwOpsGen
  TD_FILE
    "SahwOps.td"
  OUTS
    --gen-dialect-decls SahwDialect.h.inc
    --gen-dialect-defs SahwDialect.cpp.inc
    --gen-typedef-decls SahwTypes.h.inc
    --gen-typedef-defs SahwTypes.cpp.inc
    --gen-op-decls SahwOps.h.inc
    --gen-op-defs SahwOps.cpp.inc
)

iree_cc_library(
  NAME
    SahwDialect
  HDRS
    "SahwDialect.h"
  SRCS
    "SahwDialect.cpp"
  INCLUDES
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
  DEPS
    ::SahwOpsGen
    LLVMSupport
    MLIRIR
    MLIRSupport
  PUBLIC
)

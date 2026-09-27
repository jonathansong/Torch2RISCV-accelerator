# sa HAL target: the "sa" device and the "sa" target backend (executable
# format sa-desc-v1: descriptor-list templates for the PYNQ-Z1 accelerator).

# C++ package names below this directory start with "sa" (sa::SATarget).
iree_setup_c_src_root(PACKAGE_ROOT_PREFIX sa)

iree_cc_library(
  NAME
    SATarget
  HDRS
    "DescList.h"
    "Match.h"
    "SAPasses.h"
    "Templates.h"
  SRCS
    "DescList.cpp"
    "LowerWorkgroupCount.cpp"
    "Match.cpp"
    "PackLinearWeights.cpp"
    "SATarget.cpp"
    "Templates.cpp"
  DEPS
    LLVMSupport
    MLIRArithDialect
    MLIRDialectUtils
    MLIRIR
    MLIRLinalgDialect
    MLIRPass
    MLIRSupport
    MLIRTensorDialect
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::HAL::Target
    iree::compiler::Dialect::TensorExt::IR
    iree::compiler::Dialect::Util::IR
    iree::compiler::PluginAPI
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    hal_target_sa
  TARGET
    ::SATarget
)

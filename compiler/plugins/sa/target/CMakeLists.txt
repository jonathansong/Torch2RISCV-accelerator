# sa HAL target: the "sa" device and the "sa" target backend (executable
# format sa-desc-v1: descriptor-list templates for the PYNQ-Z1 accelerator).

# C++ package names below this directory start with "sa" (sa::SATarget).
iree_setup_c_src_root(PACKAGE_ROOT_PREFIX sa)

# C7 S4 (riscv_backend_design.md §9.3): in-process rv32 code when LLVM's RISCV target is built
# (build_iree_compiler.sh SA_WITH_RISCV=1), as the llvm-cpu plugin keys off it
set(_SA_RISCV_DEPS)
set(_SA_RISCV_DEFINES)
if(TARGET LLVMRISCVCodeGen)
  set(_SA_RISCV_DEPS LLVMRISCVAsmParser LLVMRISCVCodeGen LLVMRISCVDesc LLVMRISCVInfo LLVMPasses LLVMTarget
                     LLVMObject LLVMMC)
  iree_redirect_llvm_dylib_deps(_SA_RISCV_DEPS)
  set(_SA_RISCV_DEFINES SA_HAVE_RISCV)
endif()

iree_cc_library(
  NAME
    SATarget
  HDRS
    "DescList.h"
    "SAPasses.h"
    "Layout.h"
    "../transforms/SahwPasses.h"
  SRCS
    "CloneCheapProducers.cpp"
    "DescList.cpp"
    "FormAttnRegions.cpp"
    "FormFfnRegions.cpp"
    "FormMoeRegions.cpp"
    "LowerWorkgroupCount.cpp"
    "PackLinearWeights.cpp"
    "SATarget.cpp"
    "Layout.cpp"
    "../transforms/SaLin.cpp"
    "../transforms/SahwPasses.cpp"
    "../transforms/SaCodegen.cpp"
    "../transforms/SahwSerialize.cpp"
    "../transforms/SahwToRiscv.cpp"
    "../transforms/SaToSahl.cpp"
    "../transforms/SahlKernels.cpp"
    "../transforms/SahlTile.cpp"
    "../transforms/SahlPlanMemory.cpp"
    "../transforms/SahlSchedule.cpp"
    "../transforms/SahlExpandKernels.cpp"
    "../transforms/SahlToSahw.cpp"
    "../transforms/SahlLowerGeneric.cpp"
    "../transforms/SahlPipeline.cpp"
  DEFINES
    ${_SA_RISCV_DEFINES}
  DEPS
    ${_SA_RISCV_DEPS}
    sa::SahwDialect
    LLVMSupport
    LLVMCore
    MLIRArithDialect
    MLIRDialectUtils
    MLIRFuncDialect
    MLIRMathDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRMemRefDialect
    iree::compiler::Codegen::Common
    MLIRPass
    MLIRSupport
    MLIRTensorDialect
    iree::compiler::DispatchCreation
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::Flow::Transforms
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::HAL::Target
    iree::compiler::Dialect::LinalgExt::IR
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

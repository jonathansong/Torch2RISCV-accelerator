# Included by IREE's compiler/CMakeLists.txt through IREE_CMAKE_PLUGIN_PATHS
# (compiler/scripts/build_iree_compiler.sh). Registers the sa HAL target
# (docs/iree_compiler_plan.md §6) with iree-compile.
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/target sa/target)

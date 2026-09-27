# IREE runtime plugin: the sa HAL driver (compiler/runtime/sa).
iree_register_external_hal_driver(
  NAME sa
  SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/sa"
  BINARY_DIR sa
  DRIVER_TARGET sa_runtime::hal_driver
  REGISTER_FN iree_hal_sa_driver_module_register
)

// sa HAL driver: registration of the "sa" driver (docs/iree_compiler_plan.md §5).
//
// The device (sa_device.c, from IREE's synchronous local device) runs the
// dispatches of a command buffer as one descriptor list, with the sa pieces: a heap allocator whose buffer
// memory comes from the device-visible arena, and the sa-desc-v1 loader.
// Transport selection: SA_TRANSPORT=sim|board (sa_context.c).
// Host fallback (docs/iree_compiler_plan.md §8.15): a second loader, VMVX, for
// the dispatches the sa backend left to the host; the command buffer runs them
// on the ARM (sa_command_buffer.c).
#include "iree/hal/local/loaders/vmvx_module_loader.h"
#include "sa_context.h"
#include "sa_driver.h"
#include "sa_loader.h"

static iree_status_t sa_driver_factory_enumerate(void* self, iree_host_size_t* out_count,
                                                 const iree_hal_driver_info_t** out_infos) {
  static const iree_hal_driver_info_t info = {
      .driver_name = IREE_SVL("sa"),
      .full_name = IREE_SVL("PYNQ-Z1 systolic-array accelerator (descriptor lists; sim or board transport)"),
  };
  *out_count = 1;
  *out_infos = &info;
  return iree_ok_status();
}

static iree_status_t sa_driver_factory_try_create(void* self, iree_string_view_t driver_name,
                                                  iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  if (!iree_string_view_equal(driver_name, IREE_SV("sa"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }
  sa_context_t* context = NULL;
  IREE_RETURN_IF_ERROR(sa_context_get(&context));

  sa_device_params_t params;
  sa_device_params_initialize(&params);

  iree_hal_executable_loader_t* loaders[2] = {NULL, NULL};
  iree_status_t status = sa_loader_create(context, host_allocator, &loaders[0]);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vmvx_module_loader_create_isolated(0, NULL, host_allocator, &loaders[1]);
  }

  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(IREE_SV("sa"), sa_context_data_allocator(context), host_allocator,
                                            &device_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = sa_driver_create(driver_name, &params, 2, loaders, device_allocator, host_allocator, out_driver);
  }
  iree_hal_allocator_release(device_allocator);
  for (int i = 0; i < 2; ++i)
    if (loaders[i]) iree_hal_executable_loader_release(loaders[i]);
  return status;
}

iree_status_t iree_hal_sa_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = sa_driver_factory_enumerate,
      .try_create = sa_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}

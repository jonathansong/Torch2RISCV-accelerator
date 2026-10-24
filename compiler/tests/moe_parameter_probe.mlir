// Host-only check that the model ABI's "moe" archive can be loaded by IREE.
module {
  util.func public @main() -> !hal.buffer_view {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %offset = arith.constant 0 : i64
    %affinity = arith.constant -1 : i64
    %device = hal.devices.get %c0 : !hal.device
    %no_fence = util.null : !hal.fence
    %loaded = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
    %memory_type = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
    %usage = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|SharingImmutable"> : i32
    %scope = util.buffer.constant : !util.buffer = "moe"
    %key = util.buffer.constant : !util.buffer = "router"
    %buffer = io_parameters.load<%device : !hal.device>
        affinity(%affinity) wait(%no_fence) signal(%loaded)
        type(%memory_type) usage(%usage) {
      %scope::%key[%offset] : !hal.buffer{%c32}
    }
    %timeout = arith.constant -1 : i32
    %status = hal.fence.await until([%loaded]) timeout_millis(%timeout) flags("None") : i32
    %f32 = hal.element_type<f32> : i32
    %encoding = hal.encoding_type<dense_row_major> : i32
    %view = hal.buffer_view.create buffer(%buffer : !hal.buffer)[%c0, %c32]
        shape([%c8]) type(%f32) encoding(%encoding) : !hal.buffer_view
    util.return %view : !hal.buffer_view
  }
}

R"SA_MOE(
  util.func public @main(%x_view: !hal.buffer_view) -> !hal.buffer_view {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c96 = arith.constant 96 : index
    %cvalues_storage = arith.constant @VALUES_STORAGE_BYTES@ : index
    %cids_storage = arith.constant @IDS_STORAGE_BYTES@ : index
    %cvalues_offset = arith.constant @VALUES_OFFSET@ : index
    %cids_offset = arith.constant @IDS_OFFSET@ : index
    %cx = arith.constant @ACTIVATION_BYTES@ : index
    %cscores = arith.constant @SCORES_BYTES@ : index
    %crouter = arith.constant @ROUTER_BYTES@ : index
    %cparams = arith.constant @PARAM_BYTES@ : index
    %cexperts = arith.constant @EXPERTS@ : index
    %cdim = arith.constant @DIM@ : index
    %coffset = arith.constant 0 : i64
    %affinity = arith.constant -1 : i64
    %timeout = arith.constant -1 : i32
    %zero = arith.constant 0 : i32
    %neg_inf = arith.constant -8388608 : i32
    %max_id = arith.constant 2147483647 : i32
    %device = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
    %memory_type = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
    %usage = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|SharingImmutable"> : i32
    %scope = util.buffer.constant : !util.buffer = "moe"
    %router_key = util.buffer.constant : !util.buffer = "router"
    %params_key = util.buffer.constant : !util.buffer = "params"
    %no_fence = util.null : !hal.fence
    %loaded = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
    %weights:2 = io_parameters.load<%device : !hal.device>
        affinity(%affinity) wait(%no_fence) signal(%loaded)
        type(%memory_type) usage(%usage) {
      %scope::%router_key[%coffset] : !hal.buffer{%crouter},
      %scope::%params_key[%coffset] : !hal.buffer{%cparams}
    }
    %scores = hal.allocator.allocate<%allocator : !hal.allocator>
        affinity(%affinity) type(%memory_type) usage(%usage) : !hal.buffer{%cscores}
    %values = hal.allocator.allocate<%allocator : !hal.allocator>
        affinity(%affinity) type(%memory_type) usage(%usage) : !hal.buffer{%cvalues_storage}
    %ids = hal.allocator.allocate<%allocator : !hal.allocator>
        affinity(%affinity) type(%memory_type) usage(%usage) : !hal.buffer{%cids_storage}
    %route = hal.allocator.allocate<%allocator : !hal.allocator>
        affinity(%affinity) type(%memory_type) usage(%usage) : !hal.buffer{%c96}
    %out = hal.allocator.allocate<%allocator : !hal.allocator>
        affinity(%affinity) type(%memory_type) usage(%usage) : !hal.buffer{%cx}
    %init = hal.command_buffer.create device(%device : !hal.device)
        mode("OneShot") categories("Transfer") affinity(%affinity) : !hal.command_buffer
    hal.command_buffer.fill_buffer<%init : !hal.command_buffer>
        target(%scores : !hal.buffer)[%c0, %cscores] pattern(%zero : i32) flags("None")
    hal.command_buffer.fill_buffer<%init : !hal.command_buffer>
        target(%values : !hal.buffer)[%c0, %cvalues_storage] pattern(%neg_inf : i32) flags("None")
    hal.command_buffer.fill_buffer<%init : !hal.command_buffer>
        target(%ids : !hal.buffer)[%c0, %cids_storage] pattern(%zero : i32) flags("None")
    hal.command_buffer.fill_buffer<%init : !hal.command_buffer>
        target(%ids : !hal.buffer)[%cids_offset, %c8] pattern(%max_id : i32) flags("None")
    hal.command_buffer.fill_buffer<%init : !hal.command_buffer>
        target(%route : !hal.buffer)[%c0, %c96] pattern(%zero : i32) flags("None")
    hal.command_buffer.fill_buffer<%init : !hal.command_buffer>
        target(%out : !hal.buffer)[%c0, %cx] pattern(%zero : i32) flags("None")
    hal.command_buffer.finalize<%init : !hal.command_buffer>
    %ready = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
    hal.device.queue.execute<%device : !hal.device>
        affinity(%affinity) wait(%loaded) signal(%ready)
        commands(%init) flags("None")
    %status = hal.fence.await until([%ready]) timeout_millis(%timeout) flags("None") : i32
    %f32 = hal.element_type<f32> : i32
    %i8 = hal.element_type<i8> : i32
    %encoding = hal.encoding_type<dense_row_major> : i32
    %scores_view = hal.buffer_view.create buffer(%scores : !hal.buffer)[%c0, %cscores]
        shape([%c1, %cexperts]) type(%f32) encoding(%encoding) : !hal.buffer_view
    %values_view = hal.buffer_view.create buffer(%values : !hal.buffer)[%c0, %cvalues_storage]
        shape([%cvalues_storage]) type(%i8) encoding(%encoding) : !hal.buffer_view
    %ids_view = hal.buffer_view.create buffer(%ids : !hal.buffer)[%c0, %cids_storage]
        shape([%cids_storage]) type(%i8) encoding(%encoding) : !hal.buffer_view
    %router_view = hal.buffer_view.create buffer(%weights#0 : !hal.buffer)[%c0, %crouter]
        shape([%cexperts, %cdim]) type(%f32) encoding(%encoding) : !hal.buffer_view
    %params_view = hal.buffer_view.create buffer(%weights#1 : !hal.buffer)[%c0, %cparams]
        shape([%cparams]) type(%i8) encoding(%encoding) : !hal.buffer_view
    %route_view = hal.buffer_view.create buffer(%route : !hal.buffer)[%c0, %c96]
        shape([%c96]) type(%i8) encoding(%encoding) : !hal.buffer_view
    %out_view = hal.buffer_view.create buffer(%out : !hal.buffer)[%c0, %cx]
        shape([%c1, %cdim]) type(%f32) encoding(%encoding) : !hal.buffer_view
    %result:5 = util.call @device(%scores_view, %values_view, %ids_view, %x_view,
        %router_view, %params_view, %route_view, %out_view) :
        (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view,
         !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) ->
        (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view)
    util.return %result#4 : !hal.buffer_view
  }
)SA_MOE"

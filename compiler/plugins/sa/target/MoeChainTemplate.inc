R"SA_MOE(
  util.func public @main(%scores_view: !hal.buffer_view,
                         %values_view: !hal.buffer_view,
                         %ids_view: !hal.buffer_view,
                         %x_view: !hal.buffer_view,
                         %router_view: !hal.buffer_view,
                         %params_view: !hal.buffer_view,
                         %route_view: !hal.buffer_view,
                         %out_view: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c72 = arith.constant 72 : index
    %c96 = arith.constant 96 : index
    %cscores = arith.constant @SCORES_BYTES@ : index
    %crouter = arith.constant @ROUTER_BYTES@ : index
    %cparam = arith.constant @PARAM_OFFSET@ : index
    %cx = arith.constant @ACTIVATION_BYTES@ : index
    %affinity = arith.constant -1 : i64
    %device = hal.devices.get %c0 : !hal.device
    %top2_exe = hal.executable.create device(%device : !hal.device)
        affinity(%affinity) target(@top2::@sa_desc_v1) : !hal.executable
    %router_exe = hal.executable.create device(%device : !hal.device)
        affinity(%affinity) target(@router::@vmvx_bytecode_fb) : !hal.executable
    %route_exe = hal.executable.create device(%device : !hal.device)
        affinity(%affinity) target(@route::@sa_desc_v1) : !hal.executable
    %ffn_exe = hal.executable.create device(%device : !hal.device)
        affinity(%affinity) target(@ffn::@sa_desc_v1) : !hal.executable
    %scores = hal.buffer_view.buffer<%scores_view : !hal.buffer_view> : !hal.buffer
    %router = hal.buffer_view.buffer<%router_view : !hal.buffer_view> : !hal.buffer
    %values = hal.buffer_view.buffer<%values_view : !hal.buffer_view> : !hal.buffer
    %ids = hal.buffer_view.buffer<%ids_view : !hal.buffer_view> : !hal.buffer
    %x = hal.buffer_view.buffer<%x_view : !hal.buffer_view> : !hal.buffer
    %params = hal.buffer_view.buffer<%params_view : !hal.buffer_view> : !hal.buffer
    %route_buffer = hal.buffer_view.buffer<%route_view : !hal.buffer_view> : !hal.buffer
    %out = hal.buffer_view.buffer<%out_view : !hal.buffer_view> : !hal.buffer
    %params_size = hal.buffer.length<%params : !hal.buffer> : index
    %expert_bytes = arith.subi %params_size, %cparam : index
    %cmd = hal.command_buffer.create device(%device : !hal.device)
        mode("OneShot") categories("Dispatch") affinity(%affinity) : !hal.command_buffer
    hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
        target(%router_exe : !hal.executable)[%c0]
        workgroups([%c1, %c1, %c1]) constants([])
        bindings([(%x : !hal.buffer)[%c0, %cx],
                  (%router : !hal.buffer)[%c0, %crouter],
                  (%scores : !hal.buffer)[%c0, %cscores]]) flags("None")
    hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer>
        source("CommandIssue") target("CommandProcess") flags("None")
    hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
        target(%top2_exe : !hal.executable)[%c0]
        workgroups([%c1, %c1, %c1]) constants([])
        bindings([(%scores : !hal.buffer)[%c0, %cscores],
                  (%values : !hal.buffer)[%c0, %c8],
                  (%ids : !hal.buffer)[%c0, %c72]]) flags("None")
    hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer>
        source("CommandIssue") target("CommandProcess") flags("None")
    hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
        target(%route_exe : !hal.executable)[%c0]
        workgroups([%c1, %c1, %c1]) constants([])
        bindings([(%values : !hal.buffer)[%c0, %c8],
                  (%ids : !hal.buffer)[%c64, %c8],
                  (%route_buffer : !hal.buffer)[%c0, %c96]]) flags("None")
    hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer>
        source("CommandIssue") target("CommandProcess") flags("None")
    hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
        target(%ffn_exe : !hal.executable)[%c0]
        workgroups([%c1, %c1, %c1]) constants([])
        bindings([(%x : !hal.buffer)[%c0, %cx],
                  (%route_buffer : !hal.buffer)[%c64, %c32],
                  (%params : !hal.buffer)[%cparam, %expert_bytes],
                  (%out : !hal.buffer)[%c0, %cx]]) flags("None")
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    %no_fence = util.null : !hal.fence
    hal.device.queue.execute<%device : !hal.device>
        affinity(%affinity) wait(%no_fence) signal(%no_fence)
        commands(%cmd) flags("None")
    util.return %scores_view, %values_view, %ids_view, %route_view, %out_view : !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view
  }
)SA_MOE"

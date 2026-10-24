// Device-only route export fixture. The production frontend must form this
// operation from a recognized MoE layer; this file checks the SA code path.
hal.executable public @moe_route_top2 {
  hal.executable.variant public @sa_desc_v1 target(<"sa", "sa-desc-v1", {acc_bytes = 262144 : i64, bases = 16 : i64, d = 16 : i64, dyn_fields = 2 : i64, max_dynamic = 256 : i64, params = 8 : i64, spad_bytes = 131072 : i64}>) {
    hal.executable.export public @route_top2 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @route_top2() {
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %values = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<1x2xf32, #hal.descriptor_type<storage_buffer>>
        %indices = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<1x2xi32, #hal.descriptor_type<storage_buffer>>
        %route = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>) binding(2) alignment(64) offset(%c64) : memref<8xi32, #hal.descriptor_type<storage_buffer>>
        sahl.route_top2 %values, %indices, %route {experts = 16 : i64, stride = 6784 : i64} : memref<1x2xf32, #hal.descriptor_type<storage_buffer>>, memref<1x2xi32, #hal.descriptor_type<storage_buffer>>, memref<8xi32, #hal.descriptor_type<storage_buffer>>
        return
      }
    }
  }
}

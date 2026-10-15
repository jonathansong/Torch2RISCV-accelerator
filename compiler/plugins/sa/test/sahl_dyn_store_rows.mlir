// sa-to-sahl's replaceDynStores along dimension 0 (generality plan §13.4): the
// K cache's row written at layer offset + position (SmolLM2 2 layers, layer 1:
// pos + 256 in a [512, 192] cache, torch's slice normalization around it;
// hf_generic's slice at the position as an integer, dispatch main 49 before
// sa-to-sahl) becomes sahl.scatter with sa.index_add = 256 (the chain
// evaluated at positions in range), lowered as LDPARAM (index * 192 + 256 *
// 192) and one ST of the row at that DDR offset.
// RUN: iree-opt --iree-sa-to-sahl %s | FileCheck %s
// RUN: iree-opt --iree-sa-to-sahl --iree-sahl-to-sahw="d=16" %s | FileCheck %s --check-prefix=LOW
// CHECK-LABEL: func.func @main$async_dispatch_49_slow_memcpy
// CHECK-NOT: memref.load
// CHECK-NOT: arith.select
// CHECK: sahl.scatter %{{.+}}, %{{.+}}, %{{.+}} {sa.index_add = 256 : i64} : memref<192xi8>, memref<i64
// LOW: sahw.ldparam %{{.+}}, %{{.+}} {add = 49152 : i64, addr = {{[0-9]+}} : i64, mul = 192 : i64}
// LOW-NEXT: sahw.st %{{.+}} dyn(%{{.+}}) {{.*}}dyn_fields = array<i32: 1>{{.*}}row_bytes = 192 : i64
func.func @main$async_dispatch_49_slow_memcpy() {
  %c512_i64 = arith.constant 512 : i64
  %c0_i64 = arith.constant 0 : i64
  %c256_i64 = arith.constant 256 : i64
  %c768_i64 = arith.constant 768 : i64
  %c0 = arith.constant 0 : index
  %c2624 = arith.constant 2624 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<i64, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer>], flags = Indirect>) binding(1) alignment(64) offset(%c2624) flags("ReadOnly|Indirect") : memref<192xi8, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags("None") : memref<512x192xi8, #hal.descriptor_type<storage_buffer>>
  %3 = memref.load %0[] : memref<i64, #hal.descriptor_type<storage_buffer>>
  %4 = arith.addi %3, %c256_i64 : i64
  %5 = arith.cmpi sge, %4, %c0_i64 : i64
  %6 = arith.addi %3, %c768_i64 : i64
  %7 = arith.select %5, %4, %6 : i64
  %8 = arith.cmpi slt, %7, %c0_i64 : i64
  %9 = arith.select %8, %c0_i64, %7 : i64
  %10 = arith.cmpi sgt, %9, %c512_i64 : i64
  %11 = arith.select %10, %c512_i64, %9 : i64
  %12 = arith.index_cast %11 : i64 to index
  %subview = memref.subview %2[%12, 0] [1, 192] [1, 1] : memref<512x192xi8, #hal.descriptor_type<storage_buffer>> to memref<192xi8, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1 : memref<192xi8, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%subview : memref<192xi8, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>) {
  ^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
  }
  return
}


// sahl-to-sahw (§8.4 step 10) and sahw-fuse-ve (step 11), the target
// configuration from the pass options (d, spad-kb, acc-kb, ukernels).
// RUN: iree-opt --split-input-file --iree-sahl-to-sahw %s | FileCheck %s --check-prefix=ONE
// RUN: iree-opt --split-input-file --iree-sahl-to-sahw --iree-sahw-fuse-ve %s | FileCheck %s --check-prefix=FUSED
// RUN: iree-opt --split-input-file --iree-sahl-to-sahw="d=16 acc-kb=512" --iree-sahw-fuse-ve %s | FileCheck %s --check-prefix=D16
// RUN: iree-opt --split-input-file --iree-sahl-plan-memory %s | FileCheck %s --check-prefix=PLAN

// exp(-x) over a column slice: a pitched LD of 8 rows; one single-stage VE per
// operation, fused into one VE (A = -1, FUNC EXP); one ST.
// ONE-LABEL: sahw.template "exp_cols"
// ONE: sahw.ld %{{.+}} {ddr = 0 : i64, laddr = [[X:[0-9]+]] : i64, pitch = 6144 : i64, row_bytes = 3072 : i64, rows = 8 : i64}
// ONE-NEXT: sahw.ve {a = -1.000000e+00 : f32, {{.*}}length = 6144 : i64, op = 5 : i64, src1 = [[X]] : i64
// ONE-NEXT: sahw.ve {{{.*}}func = 1 : i64, {{.*}}length = 6144
// ONE-NEXT: sahw.ve
// ONE-NEXT: sahw.st
// FUSED-LABEL: sahw.template "exp_cols" attributes {bindings = 0 : i64, constants = 0 : i64, sa.d = 8 : i64}
// FUSED: sahw.ld
// FUSED-NEXT: sahw.ve {a = -1.000000e+00 : f32, b = -0.000000e+00 : f32, dst = [[Y:[0-9]+]] : i64, fp = true, func = 1 : i64, {{.*}}length = 6144 : i64
// FUSED-NEXT: sahw.st %{{.+}} {ddr = 0 : i64, laddr = [[Y]] : i64, pitch = 24576 : i64, row_bytes = 24576 : i64, rows = 1 : i64}
// FUSED-NEXT: }
// D16-LABEL: sahw.template "exp_cols" attributes {bindings = 0 : i64, constants = 0 : i64, sa.d = 16 : i64}
// D16: sahw.ve {a = -1.000000e+00 : f32, {{.*}}func = 1 : i64, {{.*}}length = 6144 : i64
#map = affine_map<(d0, d1) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>
func.func @exp_cols() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant -1.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x1536xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<8x768xf32, #hal.descriptor_type<storage_buffer>>
  %alloc = memref.alloc() : memref<8x768xf32>
  %subview = memref.subview %0[0, 0] [8, 768] [1, 1] : memref<8x1536xf32, #hal.descriptor_type<storage_buffer>> to memref<8x768xf32, strided<[1536, 1]>, #hal.descriptor_type<storage_buffer>>
  %alloc_0 = memref.alloc() : memref<8x768xf32>
  sahl.load %subview, %alloc_0 : memref<8x768xf32, strided<[1536, 1]>, #hal.descriptor_type<storage_buffer>>, memref<8x768xf32>
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_0 : memref<8x768xf32>) outs(%alloc : memref<8x768xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.mulf %in, %cst : f32
    %3 = math.exp %2 : f32
    linalg.yield %3 : f32
  }
  sahl.store %alloc, %1 : memref<8x768xf32>, memref<8x768xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----

// Row abs-max, scale = max / 127, int8 quantization by row (the quant chain):
// REDUCE MAX with FUNC ABS (one broadcast word per row), TRANSPOSE to a packed
// vector, the scale, DIV-D replication, RECIP, then the I8 output with the
// per-row operand in DIV mode (period = the row's words). sahl.to_i8 is the
// whole quantization chain (sa-to-sahl makes it from arith; sa_to_sahl.mlir).
// sahl-plan-memory: the int8 result in SPAD_A, the row maxima as broadcast words.
// PLAN-LABEL: func.func @quant_rows
// PLAN: memref.alloc() {sa.layout = "packed", sa.mem = "acc"} : memref<8x64xf32>
// PLAN: memref.alloc() {sa.layout = "bcast", sa.mem = "acc"} : memref<8xf32>
// PLAN: memref.alloc() {sa.layout = "packed", sa.mem = "spad_a"} : memref<8x64xi8>
// FUSED-LABEL: sahw.template "quant_rows"
// FUSED: sahw.ld %{{.+}} {ddr = 0 : i64, {{.*}}row_bytes = 2048 : i64, rows = 1 : i64}
// FUSED-NEXT: sahw.ve {{{.*}}func = 4 : i64, {{.*}}length = 512 : i64, op = 5 : i64, reduce = 2 : i64, rowlen = 8 : i64
// FUSED-NEXT: sahw.transpose {{{.*}}length = 64
// FUSED-NEXT: sahw.ve {a = 0.00787401571 : f32, {{.*}}length = 8 : i64
// FUSED-NEXT: sahw.ve {{{.*}}m1 = 2 : i64, op = 5 : i64, p1 = 8 : i64
// FUSED-NEXT: sahw.transpose
// FUSED-NEXT: sahw.ve {{{.*}}func = 2 : i64, {{.*}}length = 64 : i64
// FUSED-NEXT: sahw.ve {{{.*}}length = 512 : i64, m2 = 2 : i64, op = 2 : i64, period = 8 : i64, {{.*}}types = 3 : i64}
// FUSED-NEXT: sahw.st %{{.+}} {{{.*}}row_bytes = 32 : i64
// FUSED-NEXT: sahw.st %{{.+}} {{{.*}}row_bytes = 512 : i64
#pl = #hal.pipeline.layout<constants = 0, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#v = affine_map<(d0) -> (d0)>
func.func @quant_rows() {
  %c0 = arith.constant 0 : index
  %c127 = arith.constant 1.270000e+02 : f32
  %cm127 = arith.constant -1.270000e+02 : f32
  %fmin = arith.constant -3.40282347E+38 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %fmax = arith.constant 3.40282347E+38 : f32
  %pinf = arith.constant 0x7F800000 : f32
  %inv127 = arith.constant 0.00787401571 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32
  %x = hal.interface.binding.subspan layout(#pl) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x64xf32, #hal.descriptor_type<storage_buffer>>
  %s = hal.interface.binding.subspan layout(#pl) binding(1) alignment(64) offset(%c0) : memref<8xf32, #hal.descriptor_type<storage_buffer>>
  %q = hal.interface.binding.subspan layout(#pl) binding(2) alignment(64) offset(%c0) : memref<8x64xi8, #hal.descriptor_type<storage_buffer>>
  %xl = memref.alloc() : memref<8x64xf32>
  sahl.load %x, %xl : memref<8x64xf32, #hal.descriptor_type<storage_buffer>>, memref<8x64xf32>
  %m = memref.alloc() : memref<8xf32>
  linalg.fill ins(%ninf : f32) outs(%m : memref<8xf32>)
  linalg.generic {indexing_maps = [#id, #row], iterator_types = ["parallel", "reduction"]} ins(%xl : memref<8x64xf32>) outs(%m : memref<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = math.absf %in : f32
    %b = arith.maximumf %a, %out : f32
    linalg.yield %b : f32
  }
  linalg.generic {indexing_maps = [#v, #v], iterator_types = ["parallel"]} ins(%m : memref<8xf32>) outs(%m : memref<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.mulf %in, %inv127 : f32
    linalg.yield %a : f32
  }
  %ql = memref.alloc() : memref<8x64xi8>
  linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]} ins(%xl, %m : memref<8x64xf32>, memref<8xf32>) outs(%ql : memref<8x64xi8>) {
  ^bb0(%in: f32, %sc: f32, %out: i8):
    %r = arith.divf %one, %sc : f32
    %y = arith.mulf %in, %r : f32
    %i = sahl.to_i8 %y
    linalg.yield %i : i8
  }
  sahl.store %m, %s : memref<8xf32>, memref<8xf32, #hal.descriptor_type<storage_buffer>>
  sahl.store %ql, %q : memref<8x64xi8>, memref<8x64xi8, #hal.descriptor_type<storage_buffer>>
  return
}

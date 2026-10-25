// A recognized, bufferized MoE router Top-k becomes an explicit SAHL boundary.
// RUN: iree-opt --split-input-file --iree-sa-to-sahl %s | FileCheck %s

// CHECK-LABEL: func.func @moe_top2
// CHECK-NOT: iree_linalg_ext.topk
// CHECK: sahl.topk %{{.+}}, %{{.+}}, %{{.+}} {experts = 8 : i64, k = 2 : i64} : memref<1x8xf32>, memref<1x2xf32>, memref<1x2xi32>
func.func @moe_top2(%scores: memref<1x8xf32>, %values: memref<1x2xf32>, %indices: memref<1x2xi32>) {
  iree_linalg_ext.topk {sa.moe.experts = 8 : i64, sa.moe.top_k = 2 : i64}
      dimension(1) ins(%scores : memref<1x8xf32>)
      outs(%values, %indices : memref<1x2xf32>, memref<1x2xi32>) {
  ^bb0(%a: f32, %b: f32):
    %gt = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %gt : i1
  }
  return
}

// -----

// CHECK-LABEL: func.func @moe_top8
// CHECK: sahl.topk %{{.+}}, %{{.+}}, %{{.+}} {experts = 32 : i64, k = 8 : i64} : memref<1x32xf32>, memref<1x8xf32>, memref<1x8xi32>
func.func @moe_top8(%scores: memref<1x32xf32>, %values: memref<1x8xf32>, %indices: memref<1x8xi32>) {
  iree_linalg_ext.topk {sa.moe.experts = 32 : i64, sa.moe.top_k = 8 : i64}
      dimension(1) ins(%scores : memref<1x32xf32>)
      outs(%values, %indices : memref<1x8xf32>, memref<1x8xi32>) {
  ^bb0(%a: f32, %b: f32):
    %gt = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %gt : i1
  }
  return
}

// -----

// An ordinary Top-K remains on IREE's existing path.
// CHECK-LABEL: func.func @ordinary_top2
// CHECK: iree_linalg_ext.topk
// CHECK-NOT: sahl.topk
func.func @ordinary_top2(%scores: memref<1x8xf32>, %values: memref<1x2xf32>, %indices: memref<1x2xi32>) {
  iree_linalg_ext.topk dimension(1) ins(%scores : memref<1x8xf32>)
      outs(%values, %indices : memref<1x2xf32>, memref<1x2xi32>) {
  ^bb0(%a: f32, %b: f32):
    %gt = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %gt : i1
  }
  return
}

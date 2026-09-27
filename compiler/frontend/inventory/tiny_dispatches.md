# Dispatch inventory (stage C0)

41 dispatches (llvm-cpu, host), grouped by the hand-written piece they correspond to:

| piece | dispatches |
|---|---|
| element-wise chain (VE expression compiler) | 20 |
| reduction: RMSNorm sum, amax, softmax max / sum | 7 |
| int8 linear: GEMV + dequant (compile_layer.linear) | 6 |
| gather: embedding / RoPE row (compile_model.embed, rope) | 3 |
| attention scores Q·K^T (compile_model.attention) | 2 |
| attention P·V (compile_model.attention) | 2 |
| KV cache row write (compile_model.kv_append) | 1 |

| # | dispatch | linalg ops | types | tensor shapes | piece |
|---|---|---|---|---|---|
| 0 | `main$async_dispatch_0_elementwise_broadcast_32_i64xi8` | generic | i8 | 32xi8; 64x32xi8; i64 | gather: embedding / RoPE row (compile_model.embed, rope) |
| 1 | `main$async_dispatch_1_elementwise` | generic | f32 | 64xf32; f32; i64 | element-wise chain (VE expression compiler) |
| 2 | `main$async_dispatch_2_reduction_32_f32` | generic×2, fill | f32 i8 | 32xi8; f32 | reduction: RMSNorm sum, amax, softmax max / sum |
| 3 | `main$async_dispatch_3_elementwise_32_f32` | generic×2 | f32 i8 | 2x32xf32; 32xf32; 32xi8; f32 | element-wise chain (VE expression compiler) |
| 4 | `main$async_dispatch_4_reduction_32_f32` | fill, generic | f32 | 32xf32; f32 | reduction: RMSNorm sum, amax, softmax max / sum |
| 5 | `main$async_dispatch_5_elementwise` | generic |  | f32 | element-wise chain (VE expression compiler) |
| 6 | `main$async_dispatch_6_elementwise_32_f32xf32xi32` | generic | f32 i32 | 32xf32; 32xi32; f32 | element-wise chain (VE expression compiler) |
| 7 | `main$async_dispatch_7_vecmat_96x32_i32xi32xi64` | generic×2, fill, vecmat | f32 i32 i64 i8 | 32xi32; 96x32xi8; 96xf32; f32 | int8 linear: GEMV + dequant (compile_layer.linear) |
| 8 | `main$async_dispatch_8_elementwise_broadcast_32_f32` | generic | f32 | 16x2x1xf32; 32xf32 | gather: embedding / RoPE row (compile_model.embed, rope) |
| 9 | `main$async_dispatch_9_elementwise_32_f32xi64xf32xf32xf32xf32` | generic | f32 | 16x32xf32; 32xf32; i64 | gather: embedding / RoPE row (compile_model.embed, rope) |
| 10 | `main$async_dispatch_11_elementwise_32_f32xf32xf32xf32xi8` | generic | f32 i8 | 32xf32; 32xi8 | element-wise chain (VE expression compiler) |
| 11 | `main$async_dispatch_12_scatter_32x2x16xi8_dispatch_tensor_store` | scatter | i64 i8 | 1x2x16xi8; 1xi64; 32x2x16xi8 | KV cache row write (compile_model.kv_append) |
| 12 | `main$async_dispatch_13_elementwise_32_f32xi8` | generic | f32 i8 | 32xf32; 32xi8 | element-wise chain (VE expression compiler) |
| 13 | `main$async_dispatch_15_reduction_2x16_f32` | fill, generic | f32 | 2x16xf32; 2xf32 | reduction: RMSNorm sum, amax, softmax max / sum |
| 14 | `main$async_dispatch_16_elementwise_2_f32` | generic | f32 | 2xf32 | element-wise chain (VE expression compiler) |
| 15 | `main$async_dispatch_17_elementwise_2x16_f32xf32xi32` | generic | f32 i32 | 2x16xf32; 2x16xi32; 2xf32 | element-wise chain (VE expression compiler) |
| 16 | `main$async_dispatch_18_batch_matmul_2xDx1x16_i32xi32xi64` | fill, generic×2, batch_matmul | f32 i32 i64 i8 | 2x16x1xi32; 2x?x1xf32; 2xf32; 32x2x16xi8 | attention scores Q·K^T (compile_model.attention) |
| 17 | `main$async_dispatch_19_reduction_2xD_f32` | fill, generic | f32 | 2x?xf32; 2xf32; ?xf32 | reduction: RMSNorm sum, amax, softmax max / sum |
| 18 | `main$async_dispatch_20_elementwise_2xD_f32` | generic | f32 | 2x?xf32; 2xf32; ?xf32 | element-wise chain (VE expression compiler) |
| 19 | `main$async_dispatch_21_reduction_2xD_f32` | fill, generic | f32 | 2x?xf32; 2xf32 | reduction: RMSNorm sum, amax, softmax max / sum |
| 20 | `main$async_dispatch_22_elementwise_2xD_f32xf32xi32` | generic | f32 i32 | 2x?xf32; 2x?xi32; 2xf32 | element-wise chain (VE expression compiler) |
| 21 | `main$async_dispatch_23_batch_matmul_2x1x16xD_i32xi32xi64` | fill, generic×2, batch_matmul | f32 i32 i64 i8 | 2x1x16xf32; 2x1x?xi32; 32x2x16xi8 | attention P·V (compile_model.attention) |
| 22 | `main$async_dispatch_27_vecmat_32x32_i32xi32xi64` | generic×3, fill, vecmat | f32 i32 i64 i8 | 32x32xi8; 32xf32; 32xi32; 32xi8 … | int8 linear: GEMV + dequant (compile_layer.linear) |
| 23 | `main$async_dispatch_28_reduction_32_f32` | fill, generic | f32 | 32xf32; f32 | reduction: RMSNorm sum, amax, softmax max / sum |
| 24 | `main$async_dispatch_29_elementwise_32_f32` | generic | f32 | 2x32xf32; 32xf32; f32 | element-wise chain (VE expression compiler) |
| 25 | `main$async_dispatch_33_vecmat_128x32_i32xi32xi64` | generic×2, fill, vecmat | f32 i32 i64 i8 | 128x32xi8; 128xf32; 32xi32; f32 | int8 linear: GEMV + dequant (compile_layer.linear) |
| 26 | `main$async_dispatch_34_elementwise_64_f32` | generic | f32 | 128xf32; 64xf32 | element-wise chain (VE expression compiler) |
| 27 | `main$async_dispatch_35_elementwise_64_f32` | generic | f32 | 128xf32; 64xf32 | element-wise chain (VE expression compiler) |
| 28 | `main$async_dispatch_36_reduction_64_f32` | fill, generic | f32 | 64xf32; f32 | reduction: RMSNorm sum, amax, softmax max / sum |
| 29 | `main$async_dispatch_38_elementwise_64_f32xf32xi32` | generic | f32 i32 | 64xf32; 64xi32; f32 | element-wise chain (VE expression compiler) |
| 30 | `main$async_dispatch_39_vecmat_32x64_i32xi32xi64` | generic×2, fill, vecmat | f32 i32 i64 i8 | 32x64xi8; 32xf32; 64xi32; f32 | int8 linear: GEMV + dequant (compile_layer.linear) |
| 31 | `main$async_dispatch_41_elementwise_32_f32` | generic | f32 | 2x32xf32; 32xf32; f32 | element-wise chain (VE expression compiler) |
| 32 | `main$async_dispatch_47_elementwise_32_f32` | generic | f32 | 32xf32 | element-wise chain (VE expression compiler) |
| 33 | `main$async_dispatch_49_elementwise` | generic |  | i64 | element-wise chain (VE expression compiler) |
| 34 | `main$async_dispatch_50_elementwise_32_f32xf32xf32xf32xi8` | generic | f32 i8 | 32xf32; 32xi8 | element-wise chain (VE expression compiler) |
| 35 | `main$async_dispatch_52_elementwise_32_f32xi8` | generic | f32 i8 | 32xf32; 32xi8 | element-wise chain (VE expression compiler) |
| 36 | `main$async_dispatch_57_batch_matmul_2xDx1x16_i32xi32xi64` | fill, generic×2, batch_matmul | f32 i32 i64 i8 | 2x16x1xi32; 2x?x1xf32; 2xf32; 32x2x16xi8 | attention scores Q·K^T (compile_model.attention) |
| 37 | `main$async_dispatch_62_batch_matmul_2x1x16xD_i32xi32xi64` | fill, generic×2, batch_matmul | f32 i32 i64 i8 | 2x1x16xf32; 2x1x?xi32; 32x2x16xi8 | attention P·V (compile_model.attention) |
| 38 | `main$async_dispatch_66_vecmat_32x32_i32xi32xi64` | generic×2, fill, vecmat | f32 i32 i64 i8 | 32x32xi8; 32xf32; 32xi32; f32 | int8 linear: GEMV + dequant (compile_layer.linear) |
| 39 | `main$async_dispatch_80_elementwise_32_f32` | generic | f32 | 32xf32; f32 | element-wise chain (VE expression compiler) |
| 40 | `main$async_dispatch_84_vecmat_64x32_i32xi32xi64` | generic×2, fill, vecmat | f32 i32 i64 i8 | 32xi32; 64x32xi8; 64xf32; f32 | int8 linear: GEMV + dequant (compile_layer.linear) |

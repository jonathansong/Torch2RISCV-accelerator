# Dispatch inventory (stage C0)

65 dispatches (llvm-cpu, host), grouped by the hand-written piece they correspond to:

| piece | dispatches |
|---|---|
| element-wise chain (VE expression compiler) | 36 |
| reduction: RMSNorm sum, amax, softmax max / sum | 7 |
| attention scores Q·K^T (compile_model.attention) | 6 |
| attention P·V (compile_model.attention) | 6 |
| int8 linear: GEMV + dequant (compile_layer.linear) | 5 |
| gather: embedding / RoPE row (compile_model.embed, rope) | 3 |
| KV cache row write (compile_model.kv_append) | 1 |
| classifier: int8 GEMV + dequant (compile_layer.linear, looped) | 1 |

| # | dispatch | linalg ops | types | tensor shapes | piece |
|---|---|---|---|---|---|
| 0 | `main$async_dispatch_0_elementwise_broadcast_288_i64xi8` | generic | i8 | 288xi8; 32000x288xi8; i64 | gather: embedding / RoPE row (compile_model.embed, rope) |
| 1 | `main$async_dispatch_1_elementwise` | generic | f32 | 32000xf32; f32; i64 | element-wise chain (VE expression compiler) |
| 2 | `main$async_dispatch_2_reduction_288_f32` | generic×2, fill | f32 i8 | 288xi8; f32 | reduction: RMSNorm sum, amax, softmax max / sum |
| 3 | `main$async_dispatch_3_elementwise_288_f32` | generic×2 | f32 i8 | 288xf32; 288xi8; 6x288xf32; f32 | element-wise chain (VE expression compiler) |
| 4 | `main$async_dispatch_4_reduction_288_f32` | fill, generic | f32 | 288xf32; f32 | reduction: RMSNorm sum, amax, softmax max / sum |
| 5 | `main$async_dispatch_5_elementwise` | generic |  | f32 | element-wise chain (VE expression compiler) |
| 6 | `main$async_dispatch_6_elementwise_288_f32xf32xi32` | generic | f32 i32 | 288xf32; 288xi32; f32 | element-wise chain (VE expression compiler) |
| 7 | `main$async_dispatch_7_vecmat_864x288_i32xi32xi64` | generic×2, fill, vecmat | f32 i32 i64 i8 | 288xi32; 864x288xi8; 864xf32; f32 | int8 linear: GEMV + dequant (compile_layer.linear) |
| 8 | `main$async_dispatch_8_elementwise_broadcast_288_f32` | generic | f32 | 144x2x1xf32; 288xf32 | gather: embedding / RoPE row (compile_model.embed, rope) |
| 9 | `main$async_dispatch_9_elementwise_288_f32xi64xf32xf32xf32xf32` | generic | f32 | 256x288xf32; 288xf32; i64 | gather: embedding / RoPE row (compile_model.embed, rope) |
| 10 | `main$async_dispatch_11_elementwise_288_f32xf32xf32xf32xi8` | generic | f32 i8 | 288xf32; 288xi8 | element-wise chain (VE expression compiler) |
| 11 | `main$async_dispatch_12_scatter_1536x6x48xi8_dispatch_tensor_store` | scatter | i64 i8 | 1536x6x48xi8; 1x6x48xi8; 1xi64 | KV cache row write (compile_model.kv_append) |
| 12 | `main$async_dispatch_13_elementwise_288_f32xi8` | generic | f32 i8 | 288xf32; 288xi8 | element-wise chain (VE expression compiler) |
| 13 | `main$async_dispatch_15_reduction_6x48_f32` | fill, generic | f32 | 6x48xf32; 6xf32 | reduction: RMSNorm sum, amax, softmax max / sum |
| 14 | `main$async_dispatch_16_elementwise_6_f32` | generic | f32 | 6xf32 | element-wise chain (VE expression compiler) |
| 15 | `main$async_dispatch_17_elementwise_6x48_f32xf32xi32` | generic | f32 i32 | 6x48xf32; 6x48xi32; 6xf32 | element-wise chain (VE expression compiler) |
| 16 | `main$async_dispatch_18_batch_matmul_6xDx1x48_i32xi32xi64` | fill, generic×2, batch_matmul | f32 i32 i64 i8 | 1536x6x48xi8; 6x48x1xi32; 6x?x1xf32; 6xf32 | attention scores Q·K^T (compile_model.attention) |
| 17 | `main$async_dispatch_19_reduction_6xD_f32` | fill, generic | f32 | 6x?xf32; 6xf32; ?xf32 | reduction: RMSNorm sum, amax, softmax max / sum |
| 18 | `main$async_dispatch_20_elementwise_6xD_f32` | generic | f32 | 6x?xf32; 6xf32; ?xf32 | element-wise chain (VE expression compiler) |
| 19 | `main$async_dispatch_21_reduction_6xD_f32` | fill, generic | f32 | 6x?xf32; 6xf32 | reduction: RMSNorm sum, amax, softmax max / sum |
| 20 | `main$async_dispatch_22_elementwise_6xD_f32xf32xi32` | generic | f32 i32 | 6x?xf32; 6x?xi32; 6xf32 | element-wise chain (VE expression compiler) |
| 21 | `main$async_dispatch_23_batch_matmul_6x1x48xD_i32xi32xi64` | fill, generic×2, batch_matmul | f32 i32 i64 i8 | 1536x6x48xi8; 6x1x48xf32; 6x1x?xi32 | attention P·V (compile_model.attention) |
| 22 | `main$async_dispatch_27_vecmat_288x288_i32xi32xi64` | generic×3, fill, vecmat | f32 i32 i64 i8 | 288x288xi8; 288xf32; 288xi32; 288xi8 … | int8 linear: GEMV + dequant (compile_layer.linear) |
| 23 | `main$async_dispatch_28_reduction_288_f32` | fill, generic | f32 | 288xf32; f32 | reduction: RMSNorm sum, amax, softmax max / sum |
| 24 | `main$async_dispatch_29_elementwise_288_f32` | generic | f32 | 288xf32; 6x288xf32; f32 | element-wise chain (VE expression compiler) |
| 25 | `main$async_dispatch_33_vecmat_1536x288_i32xi32xi64` | generic×2, fill, vecmat | f32 i32 i64 i8 | 1536x288xi8; 1536xf32; 288xi32; f32 | int8 linear: GEMV + dequant (compile_layer.linear) |
| 26 | `main$async_dispatch_34_elementwise_768_f32` | generic | f32 | 1536xf32; 768xf32 | element-wise chain (VE expression compiler) |
| 27 | `main$async_dispatch_35_elementwise_768_f32` | generic | f32 | 1536xf32; 768xf32 | element-wise chain (VE expression compiler) |
| 28 | `main$async_dispatch_36_reduction_768_f32` | fill, generic | f32 | 768xf32; f32 | reduction: RMSNorm sum, amax, softmax max / sum |
| 29 | `main$async_dispatch_38_elementwise_768_f32xf32xi32` | generic | f32 i32 | 768xf32; 768xi32; f32 | element-wise chain (VE expression compiler) |
| 30 | `main$async_dispatch_39_vecmat_288x768_i32xi32xi64` | generic×2, fill, vecmat | f32 i32 i64 i8 | 288x768xi8; 288xf32; 768xi32; f32 | int8 linear: GEMV + dequant (compile_layer.linear) |
| 31 | `main$async_dispatch_41_elementwise_288_f32` | generic | f32 | 288xf32; 6x288xf32; f32 | element-wise chain (VE expression compiler) |
| 32 | `main$async_dispatch_47_elementwise_288_f32` | generic | f32 | 288xf32 | element-wise chain (VE expression compiler) |
| 33 | `main$async_dispatch_49_elementwise` | generic |  | i64 | element-wise chain (VE expression compiler) |
| 34 | `main$async_dispatch_50_elementwise_288_f32xf32xf32xf32xi8` | generic | f32 i8 | 288xf32; 288xi8 | element-wise chain (VE expression compiler) |
| 35 | `main$async_dispatch_52_elementwise_288_f32xi8` | generic | f32 i8 | 288xf32; 288xi8 | element-wise chain (VE expression compiler) |
| 36 | `main$async_dispatch_57_batch_matmul_6xDx1x48_i32xi32xi64` | fill, generic×2, batch_matmul | f32 i32 i64 i8 | 1536x6x48xi8; 6x48x1xi32; 6x?x1xf32; 6xf32 | attention scores Q·K^T (compile_model.attention) |
| 37 | `main$async_dispatch_62_batch_matmul_6x1x48xD_i32xi32xi64` | fill, generic×2, batch_matmul | f32 i32 i64 i8 | 1536x6x48xi8; 6x1x48xf32; 6x1x?xi32 | attention P·V (compile_model.attention) |
| 38 | `main$async_dispatch_66_vecmat_288x288_i32xi32xi64` | generic×2, fill, vecmat | f32 i32 i64 i8 | 288x288xi8; 288xf32; 288xi32; f32 | int8 linear: GEMV + dequant (compile_layer.linear) |
| 39 | `main$async_dispatch_80_elementwise_288_f32` | generic | f32 | 288xf32; 6x288xf32; f32 | element-wise chain (VE expression compiler) |
| 40 | `main$async_dispatch_88_elementwise` | generic |  | i64 | element-wise chain (VE expression compiler) |
| 41 | `main$async_dispatch_89_elementwise_288_f32xf32xf32xf32xi8` | generic | f32 i8 | 288xf32; 288xi8 | element-wise chain (VE expression compiler) |
| 42 | `main$async_dispatch_91_elementwise_288_f32xi8` | generic | f32 i8 | 288xf32; 288xi8 | element-wise chain (VE expression compiler) |
| 43 | `main$async_dispatch_96_batch_matmul_6xDx1x48_i32xi32xi64` | fill, generic×2, batch_matmul | f32 i32 i64 i8 | 1536x6x48xi8; 6x48x1xi32; 6x?x1xf32; 6xf32 | attention scores Q·K^T (compile_model.attention) |
| 44 | `main$async_dispatch_101_batch_matmul_6x1x48xD_i32xi32xi64` | fill, generic×2, batch_matmul | f32 i32 i64 i8 | 1536x6x48xi8; 6x1x48xf32; 6x1x?xi32 | attention P·V (compile_model.attention) |
| 45 | `main$async_dispatch_119_elementwise_288_f32` | generic | f32 | 288xf32; 6x288xf32; f32 | element-wise chain (VE expression compiler) |
| 46 | `main$async_dispatch_127_elementwise` | generic |  | i64 | element-wise chain (VE expression compiler) |
| 47 | `main$async_dispatch_128_elementwise_288_f32xf32xf32xf32xi8` | generic | f32 i8 | 288xf32; 288xi8 | element-wise chain (VE expression compiler) |
| 48 | `main$async_dispatch_130_elementwise_288_f32xi8` | generic | f32 i8 | 288xf32; 288xi8 | element-wise chain (VE expression compiler) |
| 49 | `main$async_dispatch_135_batch_matmul_6xDx1x48_i32xi32xi64` | fill, generic×2, batch_matmul | f32 i32 i64 i8 | 1536x6x48xi8; 6x48x1xi32; 6x?x1xf32; 6xf32 | attention scores Q·K^T (compile_model.attention) |
| 50 | `main$async_dispatch_140_batch_matmul_6x1x48xD_i32xi32xi64` | fill, generic×2, batch_matmul | f32 i32 i64 i8 | 1536x6x48xi8; 6x1x48xf32; 6x1x?xi32 | attention P·V (compile_model.attention) |
| 51 | `main$async_dispatch_158_elementwise_288_f32` | generic | f32 | 288xf32; 6x288xf32; f32 | element-wise chain (VE expression compiler) |
| 52 | `main$async_dispatch_166_elementwise` | generic |  | i64 | element-wise chain (VE expression compiler) |
| 53 | `main$async_dispatch_167_elementwise_288_f32xf32xf32xf32xi8` | generic | f32 i8 | 288xf32; 288xi8 | element-wise chain (VE expression compiler) |
| 54 | `main$async_dispatch_169_elementwise_288_f32xi8` | generic | f32 i8 | 288xf32; 288xi8 | element-wise chain (VE expression compiler) |
| 55 | `main$async_dispatch_174_batch_matmul_6xDx1x48_i32xi32xi64` | fill, generic×2, batch_matmul | f32 i32 i64 i8 | 1536x6x48xi8; 6x48x1xi32; 6x?x1xf32; 6xf32 | attention scores Q·K^T (compile_model.attention) |
| 56 | `main$async_dispatch_179_batch_matmul_6x1x48xD_i32xi32xi64` | fill, generic×2, batch_matmul | f32 i32 i64 i8 | 1536x6x48xi8; 6x1x48xf32; 6x1x?xi32 | attention P·V (compile_model.attention) |
| 57 | `main$async_dispatch_197_elementwise_288_f32` | generic | f32 | 288xf32; 6x288xf32; f32 | element-wise chain (VE expression compiler) |
| 58 | `main$async_dispatch_205_elementwise` | generic |  | i64 | element-wise chain (VE expression compiler) |
| 59 | `main$async_dispatch_206_elementwise_288_f32xf32xf32xf32xi8` | generic | f32 i8 | 288xf32; 288xi8 | element-wise chain (VE expression compiler) |
| 60 | `main$async_dispatch_208_elementwise_288_f32xi8` | generic | f32 i8 | 288xf32; 288xi8 | element-wise chain (VE expression compiler) |
| 61 | `main$async_dispatch_213_batch_matmul_6xDx1x48_i32xi32xi64` | fill, generic×2, batch_matmul | f32 i32 i64 i8 | 1536x6x48xi8; 6x48x1xi32; 6x?x1xf32; 6xf32 | attention scores Q·K^T (compile_model.attention) |
| 62 | `main$async_dispatch_218_batch_matmul_6x1x48xD_i32xi32xi64` | fill, generic×2, batch_matmul | f32 i32 i64 i8 | 1536x6x48xi8; 6x1x48xf32; 6x1x?xi32 | attention P·V (compile_model.attention) |
| 63 | `main$async_dispatch_236_elementwise_288_f32` | generic | f32 | 288xf32; f32 | element-wise chain (VE expression compiler) |
| 64 | `main$async_dispatch_240_vecmat_32000x288_i32xi32xi64` | generic×2, fill, vecmat | f32 i32 i64 i8 | 288xi32; 32000x288xi8; 32000xf32; f32 | classifier: int8 GEMV + dequant (compile_layer.linear, looped) |

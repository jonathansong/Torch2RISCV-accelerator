# PYNQ-Z1 baseline: performance (689965a)

## Board regression

```
board pynq armv7l 2026-10-04 11:43:06
commit 689965a5cafcf17ddb83357bcd56a8ab0b949e74 (staged at 9e1f5cf with these test scripts uncommitted; compiler, runtime, frontend identical)
staged 2026-10-04T13:39:22+02:00
bitstream 93ec80eaca077626 (RISCV-on-PYNQ-Z1/bitstreams/l2)
bwtest                 PASS     17.9 s  PASS
c3                     PASS     25.4 s  board PASS
c55                    PASS     51.0 s  board PASS
c6p_stories            PASS     28.5 s  board PASS
c6p_stories_profile    PASS     20.0 s  total                                                         200.0      2709473             59039.4   (per step)
c6p_smollm2            PASS     46.8 s  board PASS
c6p_smollm2_profile    PASS     46.8 s  total                                                         910.0     22220734            466471.2   (per step)
hfgen                  PASS     50.3 s  board PASS
hfgen_profile          PASS     55.0 s  total                                                        1000.0     26671271            557703.9   (per step)
REGRESSION PASS
```

## Speed (board_llm.py: sa-llm-run's timing line)

| test | timing |
|---|---|
| c3 | sa device (board): 77 steps, 56.8 ms per step after the first (17.60 tokens/s); rt_fw completed 539 lists |
| c55 | sa device (board): 38 steps, 450.0 ms per step after the first (2.22 tokens/s); rt_fw completed 1178 lists |
| c6p_stories | sa device (board): prefill: 16 prompt tokens in 2 chunks of 8, 211.7 ms (75.57 tokens/s; the first chunk includes loading); 60 decode steps, 57.5 ms per step (17.39 tokens/s); rt_fw completed 526 lists |
| c6p_smollm2 | sa device (board): prefill: 20 prompt tokens in 3 chunks of 8, 2803.9 ms (7.13 tokens/s; the first chunk includes loading); 24 decode steps, 459.9 ms per step (2.17 tokens/s); rt_fw completed 1473 lists |
| hfgen | sa device (board): prefill: 20 prompt tokens in 3 chunks of 8, 6294.5 ms (3.18 tokens/s; the first chunk includes loading); 16 decode steps, 535.5 ms per step (1.87 tokens/s); rt_fw completed 211 lists |

## DMA bandwidth (m2_bw_test.py)

```
test                                           cycles  B/cycle     MB/s % of 8 B/c
LD  DDR -> SPAD_A, 64 KB contiguous              8250     7.94    397.2      99.3%
ST  SPAD_A -> DDR, 64 KB                         8262     7.93    396.6      99.2%
LD  DDR -> ACC, 64 KB contiguous                 8249     7.94    397.2      99.3%
ST  ACC -> DDR, 64 KB                            8283     7.91    395.6      98.9%
LD  8-byte rows, pitch 16 (1-beat bursts)       17199     3.81    190.5      47.6%
LD SPAD_B + ST SPAD_A concurrently               8318    15.76    787.9     197.0%
stored data matches source
PASS
```

## Profile: c6p_stories (board_profile.py; the full log in the release's results)

```
launcher: overlay D = 8, rt_fw ready, window 64 MB at 0xd800000, ring 16
export                                                        calls       cycles    cyc %    host us
main$async_dispatch_199_matvec_like_4000x8x288_i32xi8xi64       1.0      1259555    23.3%    25215.3
prefill$async_dispatch_96_reduction_8x768_f32                   6.0       737893    13.6%    14903.7
prefill$async_dispatch_95_elementwise_8x768_f32                 6.0       693307    12.8%    14014.2
prefill$async_dispatch_94_matmul_like_8x192x8x288_i8xi8xi3      6.0       499811     9.2%    10140.1
prefill$async_dispatch_25_matmul_like_8x108x8x288_i8xi8xi3      6.0       305857     5.7%     6262.1
prefill$async_dispatch_97_matmul_like_8x36x8x768_i8xi8xi32      5.0       210236     3.9%     4328.3
prefill$async_dispatch_90_reduction_8x288_f32                  11.0       144450     2.7%     3155.4
prefill$async_dispatch_89_matmul_like_8x36x8x288_i8xi8xi32      6.0       133610     2.5%     2816.6
prefill$async_dispatch_92_reduction_8x288_f32                  11.0       106390     2.0%     2396.9
main$async_dispatch_17_reduction_6xD_f32                       48.0       101210     1.9%     3173.4
prefill$async_dispatch_88_reduction_8x288_f32                   6.0        70883     1.3%     1568.4
prefill$async_dispatch_28_elementwise_2304_f32                  6.0        53925     1.0%     1222.5
prefill$async_dispatch_93_elementwise_8x288_f32xf32xi8         11.0        40221     0.7%     1067.6
prefill$async_dispatch_522_matmul_like_8x36x8x768_i8xi8xi3      1.0        37522     0.7%      775.1
main$async_dispatch_47_batch_matmul_6xDx1x48_i32xi32xi64        8.0        33759     0.6%      869.9
... (188 more exports)
total                                                         527.0      5412803            121044.6   (per step)
total                                                         200.0      2709473             59039.4   (per step)
```

## Profile: c6p_smollm2 (board_profile.py; the full log in the release's results)

```
launcher: 144 MB window: BO allocation failed: std::bad_alloc; dropping caches, compacting memory, retrying
launcher: overlay D = 8, rt_fw ready, window 144 MB at 0x9900000, ring 16
export                                                        calls       cycles    cyc %    host us
prefill$async_dispatch_96_matmul_like_8x384x8x576_i8xi8xi3     30.0      8142709    19.9%   163585.3
prefill$async_dispatch_98_reduction_8x1536_f32                 30.0      7329040    17.9%   147310.0
prefill$async_dispatch_97_elementwise_8x1536_f32               30.0      6922480    16.9%   139185.1
prefill$async_dispatch_99_matmul_like_8x72x8x1536_i8xi8xi3     29.0      3848200     9.4%    77670.5
prefill$async_dispatch_25_matmul_like_8x120x8x576_i8xi8xi3     30.0      2707332     6.6%    54870.5
prefill$async_dispatch_91_matmul_like_8x72x8x576_i8xi8xi32     30.0      1850908     4.5%    37748.7
prefill$async_dispatch_92_reduction_8x576_f32                  59.0      1522596     3.7%    31880.9
main$async_dispatch_15_reduction_3x3x64_f32                   240.0      1382562     3.4%    33532.6
prefill$async_dispatch_94_reduction_8x576_f32                  59.0      1114398     2.7%    23714.6
prefill$async_dispatch_90_reduction_8x576_f32                  30.0       665427     1.6%    14045.1
prefill$async_dispatch_95_elementwise_8x576_f32xf32xi8         59.0       351712     0.9%     8454.5
main$async_dispatch_11_elementwise_broadcast_576_f32          240.0       250590     0.6%    10854.3
main$async_dispatch_13_scatter_7680x3x64xi8_dispatch_tenso    232.0       106509     0.3%     7807.3
prefill$async_dispatch_24_reduction_8x576_f32                   1.0        51438     0.1%     1053.2
main$async_dispatch_316_batch_matmul_3x3xDx64_i32xi32xi64       8.0        48800     0.1%     1172.6
... (733 more exports)
total                                                        2264.0     40890400            872962.5   (per step)
total                                                         910.0     22220734            466471.2   (per step)
```

## Profile: hfgen (board_profile.py; the full log in the release's results)

```
launcher: overlay D = 8, rt_fw ready, window 168 MB at 0x9900000, ring 16
export                                                        calls       cycles    cyc %    host us
prefill$async_dispatch_89_matmul_like_8x192x8x576_i8xi8xi3     30.0     16560104    25.5%   331942.8
prefill$async_dispatch_92_matmul_like_8x72x8x1536_i8xi8xi3     30.0      3980840     6.1%    80343.0
prefill$async_dispatch_88_matmul_like_8x192x8x576_i8xi8xi3     30.0      3843937     5.9%    77605.4
prefill$async_dispatch_83_matmul_like_8x72x8x576_i8xi8xi32     30.0      1850878     2.8%    37747.0
prefill$async_dispatch_25_matmul_like_8x72x8x576_i8xi8xi32     30.0      1588216     2.4%    32488.0
prefill$async_dispatch_84_reduction_8x576_f32                  60.0      1548344     2.4%    32435.7
prefill$async_dispatch_90_reduction_8x1536_f32                 30.0      1488241     2.3%    30503.1
prefill$async_dispatch_27_matmul_like_8x24x8x576_i8xi8xi32     30.0       685406     1.1%    14427.9
prefill$async_dispatch_85_elementwise_8x576_f32                59.0       596781     0.9%    13375.9
prefill$async_dispatch_86_reduction_8x576_f32                  30.0       566659     0.9%    12062.7
prefill$async_dispatch_81_reduction_8x576_f32                  30.0       566634     0.9%    12075.7
prefill$async_dispatch_95_reduction_8x576_f32                  29.0       547702     0.8%    11656.1
prefill$async_dispatch_82_elementwise_8x576_f32xf32xi8         89.0       530606     0.8%    12759.5
prefill$async_dispatch_91_elementwise_8x1536_f32xf32xi8        30.0       409254     0.6%     8911.9
main$async_dispatch_19_batch_matmul_3x3x256x64_i32xi32xi64      8.0       382527     0.6%     7843.4
... (706 more exports)
total                                                        2119.0     65029140           1352267.4   (per step)
total                                                        1000.0     26671271            557703.9   (per step)
```

## Bitstream (RISCV-on-PYNQ-Z1/bitstreams/l2)

44,005 LUT (82.7 %), 33,622 FF, 116 DSP, 130 BRAM36, setup WNS +1.836 ns

Reports: `utilization.rpt`, `timing_summary.rpt` in the same directory.

# PYNQ-Z1 baseline: performance (ce3cf11)

## Board regression

```
board pynq armv7l 2026-10-04 12:20:29
commit ce3cf1125882080168b1fe253db2708ed7043102 (bundles staged at 9e1f5cf; sa-llm-run and board_regress.py refreshed at this commit; compiler and frontend identical)
staged 2026-10-04T13:39:22+02:00, refreshed 2026-10-04T14:15:42+02:00
bitstream 93ec80eaca077626 (RISCV-on-PYNQ-Z1/bitstreams/l2)
bwtest                 PASS     22.0 s  PASS
c3                     PASS     26.7 s  board PASS
c55                    PASS     50.9 s  board PASS
c6p_stories            PASS     29.0 s  board PASS
c6p_stories_profile    PASS     15.9 s  total                                                         200.0      2709281             75955.5   (per step)
c6p_smollm2            PASS     46.5 s  board PASS
c6p_smollm2_profile    PASS     46.4 s  total                                                         910.0     22220320            543599.5   (per step)
hfgen                  PASS     54.8 s  board PASS
hfgen_profile          PASS     58.0 s  total                                                        1000.0     26671269            642239.1   (per step)
REGRESSION PASS
```

## Speed (board_llm.py: sa-llm-run's timing line)

| test | timing |
|---|---|
| c3 | sa device (board): 77 steps, 56.8 ms per step after the first (17.62 tokens/s); rt_fw completed 539 lists |
| c55 | sa device (board): 38 steps, 449.6 ms per step after the first (2.22 tokens/s); rt_fw completed 1178 lists |
| c6p_stories | sa device (board): prefill: 16 prompt tokens in 2 chunks of 8, 211.2 ms (75.74 tokens/s; the first chunk includes loading); 60 decode steps, 57.4 ms per step (17.41 tokens/s); rt_fw completed 526 lists |
| c6p_smollm2 | sa device (board): prefill: 20 prompt tokens in 3 chunks of 8, 2805.0 ms (7.13 tokens/s; the first chunk includes loading); 24 decode steps, 460.1 ms per step (2.17 tokens/s); rt_fw completed 1473 lists |
| hfgen | sa device (board): prefill: 20 prompt tokens in 3 chunks of 8, 6298.4 ms (3.18 tokens/s; the first chunk includes loading); 16 decode steps, 535.8 ms per step (1.87 tokens/s); rt_fw completed 211 lists |

## DMA bandwidth (m2_bw_test.py)

```
test                                           cycles  B/cycle     MB/s % of 8 B/c
LD  DDR -> SPAD_A, 64 KB contiguous              8249     7.94    397.2      99.3%
ST  SPAD_A -> DDR, 64 KB                         8262     7.93    396.6      99.2%
LD  DDR -> ACC, 64 KB contiguous                 8249     7.94    397.2      99.3%
ST  ACC -> DDR, 64 KB                            8283     7.91    395.6      98.9%
LD  8-byte rows, pitch 16 (1-beat bursts)       17195     3.81    190.6      47.6%
LD SPAD_B + ST SPAD_A concurrently               8318    15.76    787.9     197.0%
stored data matches source
PASS
```

## Profile: c6p_stories (board_profile.py; the full log in the release's results)

```
launcher: overlay D = 8, rt_fw ready, window 64 MB at 0x9900000, ring 16
export                                                        calls       cycles    cyc %    host us
main$async_dispatch_199_matvec_like_4000x8x288_i32xi8xi64       1.0      1259554    23.3%    25300.4
prefill$async_dispatch_96_reduction_8x768_f32                   6.0       737907    13.6%    15408.9
prefill$async_dispatch_95_elementwise_8x768_f32                 6.0       693314    12.8%    14520.7
prefill$async_dispatch_94_matmul_like_8x192x8x288_i8xi8xi3      6.0       499844     9.2%    10648.5
prefill$async_dispatch_25_matmul_like_8x108x8x288_i8xi8xi3      6.0       305867     5.7%     6767.1
prefill$async_dispatch_97_matmul_like_8x36x8x768_i8xi8xi32      5.0       210264     3.9%     4750.4
prefill$async_dispatch_90_reduction_8x288_f32                  11.0       144486     2.7%     4084.2
prefill$async_dispatch_89_matmul_like_8x36x8x288_i8xi8xi32      6.0       133614     2.5%     3326.7
prefill$async_dispatch_92_reduction_8x288_f32                  11.0       106390     2.0%     3326.2
main$async_dispatch_17_reduction_6xD_f32                       48.0       101276     1.9%     7244.4
prefill$async_dispatch_88_reduction_8x288_f32                   6.0        70874     1.3%     2071.8
prefill$async_dispatch_28_elementwise_2304_f32                  6.0        53933     1.0%     1733.0
prefill$async_dispatch_93_elementwise_8x288_f32xf32xi8         11.0        40211     0.7%     1999.5
prefill$async_dispatch_522_matmul_like_8x36x8x768_i8xi8xi3      1.0        37508     0.7%      857.8
main$async_dispatch_175_batch_matmul_6xDx1x48_i32xi32xi64       8.0        33737     0.6%     1549.4
... (188 more exports)
total                                                         527.0      5413072            165658.0   (per step)
total                                                         200.0      2709281             75955.5   (per step)
```

## Profile: c6p_smollm2 (board_profile.py; the full log in the release's results)

```
launcher: 144 MB window: BO allocation failed: std::bad_alloc; dropping caches, compacting memory, retrying
launcher: overlay D = 8, rt_fw ready, window 144 MB at 0x9900000, ring 16
export                                                        calls       cycles    cyc %    host us
prefill$async_dispatch_96_matmul_like_8x384x8x576_i8xi8xi3     30.0      8142710    19.9%   166124.9
prefill$async_dispatch_98_reduction_8x1536_f32                 30.0      7329057    17.9%   149865.4
prefill$async_dispatch_97_elementwise_8x1536_f32               30.0      6922489    16.9%   141722.6
prefill$async_dispatch_99_matmul_like_8x72x8x1536_i8xi8xi3     29.0      3848211     9.4%    80129.0
prefill$async_dispatch_25_matmul_like_8x120x8x576_i8xi8xi3     30.0      2707327     6.6%    57412.9
prefill$async_dispatch_91_matmul_like_8x72x8x576_i8xi8xi32     30.0      1850884     4.5%    40306.1
prefill$async_dispatch_92_reduction_8x576_f32                  59.0      1522499     3.7%    36875.6
main$async_dispatch_15_reduction_3x3x64_f32                   240.0      1382642     3.4%    53847.7
prefill$async_dispatch_94_reduction_8x576_f32                  59.0      1114388     2.7%    28720.3
prefill$async_dispatch_90_reduction_8x576_f32                  30.0       665394     1.6%    16580.1
prefill$async_dispatch_95_elementwise_8x576_f32xf32xi8         59.0       351699     0.9%    13469.5
main$async_dispatch_11_elementwise_broadcast_576_f32          240.0       250604     0.6%    31200.3
main$async_dispatch_13_scatter_7680x3x64xi8_dispatch_tenso    232.0       106538     0.3%    27447.6
prefill$async_dispatch_24_reduction_8x576_f32                   1.0        51444     0.1%     1138.1
main$async_dispatch_346_batch_matmul_3x3xDx64_i32xi32xi64       8.0        48782     0.1%     1845.2
... (733 more exports)
total                                                        2264.0     40889257           1064760.6   (per step)
total                                                         910.0     22220320            543599.5   (per step)
```

## Profile: hfgen (board_profile.py; the full log in the release's results)

```
launcher: 168 MB window: BO allocation failed: std::bad_alloc; dropping caches, compacting memory, retrying
launcher: 168 MB window: BO allocation failed: std::bad_alloc; dropping caches, compacting memory, retrying
launcher: 168 MB window: BO allocation failed: std::bad_alloc; dropping caches, compacting memory, retrying
launcher: overlay D = 8, rt_fw ready, window 168 MB at 0x9900000, ring 16
export                                                        calls       cycles    cyc %    host us
prefill$async_dispatch_89_matmul_like_8x192x8x576_i8xi8xi3     30.0     16560126    25.5%   334476.3
prefill$async_dispatch_92_matmul_like_8x72x8x1536_i8xi8xi3     30.0      3980902     6.1%    82885.6
prefill$async_dispatch_88_matmul_like_8x192x8x576_i8xi8xi3     30.0      3843934     5.9%    80140.7
prefill$async_dispatch_83_matmul_like_8x72x8x576_i8xi8xi32     30.0      1850844     2.8%    40283.4
prefill$async_dispatch_25_matmul_like_8x72x8x576_i8xi8xi32     30.0      1588134     2.4%    35021.6
prefill$async_dispatch_84_reduction_8x576_f32                  60.0      1548370     2.4%    37492.7
prefill$async_dispatch_90_reduction_8x1536_f32                 30.0      1488234     2.3%    33038.0
prefill$async_dispatch_27_matmul_like_8x24x8x576_i8xi8xi32     30.0       685416     1.1%    16968.2
prefill$async_dispatch_85_elementwise_8x576_f32                59.0       596740     0.9%    18355.8
prefill$async_dispatch_81_reduction_8x576_f32                  30.0       566650     0.9%    14601.3
prefill$async_dispatch_86_reduction_8x576_f32                  30.0       566644     0.9%    14591.5
prefill$async_dispatch_95_reduction_8x576_f32                  29.0       547722     0.8%    14111.8
prefill$async_dispatch_82_elementwise_8x576_f32xf32xi8         89.0       530414     0.8%    20298.4
prefill$async_dispatch_91_elementwise_8x1536_f32xf32xi8        30.0       409192     0.6%    11450.4
main$async_dispatch_415_batch_matmul_3x3x256x64_i32xi32xi6      8.0       382525     0.6%     8521.3
... (706 more exports)
total                                                        2119.0     65028787           1531367.8   (per step)
total                                                        1000.0     26671269            642239.1   (per step)
```

## Bitstream (RISCV-on-PYNQ-Z1/bitstreams/l2)

44,005 LUT (82.7 %), 33,622 FF, 116 DSP, 130 BRAM36, setup WNS +1.836 ns

Reports: `utilization.rpt`, `timing_summary.rpt` in the same directory.

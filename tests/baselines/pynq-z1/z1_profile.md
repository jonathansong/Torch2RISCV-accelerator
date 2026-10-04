# PYNQ-Z1 time breakdown (decode)

Per decode step on the board (L2 bitstream, D = 8, 50 MHz), from the unit's event counters over each
dispatch's list (`board_profile.py` with `SA_PROFILE_PERF`; profiling runs one list per dispatch, so
dispatches do not overlap each other here). Columns: see `compiler/tests/z1_profile.py`. The busy
columns overlap and do not add up; *floor* is the bytes loaded at bwtest's 7.94 B/cycle (the
load-bound minimum), *idle* the cycles with every engine idle (fixed overhead).

## stories15M, hand-written path (export.py)

- device cycles per step: **2.71M** (54 ms at 50 MHz)
- bytes loaded: 15.83 MB; at bwtest's 7.94 B/cycle that is 1.99M (74%) — the load floor; LD busy 2.02M (75%)
- EX 2.00M (74%) (useful 1.91M (71%)), VE / SFU 0.37M (14%), ST 0.05M (2%)
- fixed overhead (all engines idle, plus rt_fw outside the counter window): 0.06M (2%)
- overlap: the busy columns add up to 4.45M against 2.66M non-idle cycles, so 1.79M engine cycles ran alongside another engine
- not hidden behind the load floor: 0.72M (26%); scoreboard waits (HAZ_*) 2.48M

| dispatch type | per step | cycles | % | LD busy | LD bytes | floor | EX | EX useful | VE | ST | idle | overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matvec_like_NxNxN_iNxiNxiN | 25 | 2,267k | 83.7 | 1,957k | 15.44 MB | 1,945k | 1,986k | 1,898k | 75k | 29k | 7k | 1,785k |
| reduction_N_fN | 38 | 123k | 4.6 | 14k | 0.09 MB | 11k | 0k | 0k | 98k | 3k | 12k | -1k |
| elementwise_N_fN | 7 | 89k | 3.3 | 3k | 0.02 MB | 3k | 0k | 0k | 82k | 3k | 2k | -0k |
| batch_matmul_NxDxNxN_iNxiNxiN | 6 | 42k | 1.6 | 10k | 0.06 MB | 8k | 9k | 7k | 22k | 1k | 2k | 2k |
| elementwise_N_fNxfNxiN | 25 | 29k | 1.1 | 7k | 0.04 MB | 5k | 0k | 0k | 11k | 5k | 7k | -0k |
| reduction_NxD_iNxfNxfN | 6 | 29k | 1.1 | 1k | 0.00 MB | 1k | 0k | 0k | 25k | 1k | 2k | -0k |
| batch_matmul_NxNxNxD_iNxiNxiN | 6 | 24k | 0.9 | 10k | 0.06 MB | 8k | 10k | 7k | 4k | 1k | 2k | 3k |
| reduction_NxN_fN | 6 | 21k | 0.8 | 4k | 0.02 MB | 3k | 0k | 0k | 14k | 1k | 2k | -0k |
| reduction_NxD_fN | 6 | 17k | 0.6 | 2k | 0.00 MB | 1k | 0k | 0k | 12k | 1k | 2k | -0k |
| elementwise_broadcast_N_iNxiN | 1 | 16k | 0.6 | 0k | 0.00 MB | 0k | 0k | 0k | 15k | 0k | 0k | 0k |
| elementwise_NxN_fN | 12 | 15k | 0.5 | 5k | 0.03 MB | 3k | 0k | 0k | 5k | 2k | 3k | -0k |
| elementwise | 31 | 12k | 0.4 | 1k | 0.00 MB | 0k | 0k | 0k | 2k | 0k | 11k | -0k |
| (5 other types) | 31 | 25k | 0.9 | 9k | 0.05 MB | 7k | 0k | 0k | 5k | 3k | 10k | -0k |
| **total** | 200 | 2,709k | 100.0 | 2,024k | 15.83 MB | 1,994k | 2,005k | 1,912k | 371k | 52k | 63k | 1,787k |

Prefill, per chunk of 8 tokens (after the first chunk):

- device cycles per chunk: **5.41M** (108 ms at 50 MHz)
- bytes loaded: 18.04 MB; at bwtest's 7.94 B/cycle that is 2.27M (42%) — the load floor; LD busy 2.36M (44%)
- EX 2.07M (38%) (useful 1.95M (36%)), VE / SFU 2.43M (45%), ST 0.22M (4%)
- fixed overhead (all engines idle, plus rt_fw outside the counter window): 0.17M (3%)
- overlap: the busy columns add up to 7.08M against 5.29M non-idle cycles, so 1.79M engine cycles ran alongside another engine
- not hidden behind the load floor: 3.14M (58%); scoreboard waits (HAZ_*) 4.14M

| dispatch type | per step | cycles | % | LD busy | LD bytes | floor | EX | EX useful | VE | ST | idle | overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matvec_like_NxNxN_iNxiNxiN | 1 | 1,260k | 23.3 | 1,182k | 9.35 MB | 1,177k | 1,208k | 1,152k | 33k | 18k | 0k | 1,182k |
| reduction_NxN_fN | 83 | 1,220k | 22.5 | 112k | 0.82 MB | 103k | 0k | 0k | 1,075k | 16k | 25k | 1k |
| matmul_like_NxNxNxN_iNxiNxiN | 24 | 1,187k | 21.9 | 788k | 6.22 MB | 784k | 778k | 746k | 126k | 73k | 7k | 583k |
| elementwise_NxN_fN | 6 | 693k | 12.8 | 19k | 0.15 MB | 19k | 0k | 0k | 654k | 19k | 2k | -0k |
| batch_matmul_NxDxNxN_iNxiNxiN | 48 | 202k | 3.7 | 54k | 0.28 MB | 35k | 36k | 28k | 111k | 7k | 14k | 15k |
| elementwise_broadcast_N_iNxfN | 24 | 139k | 2.6 | 6k | 0.04 MB | 5k | 0k | 0k | 122k | 4k | 9k | -0k |
| (14 other types) | 341 | 711k | 13.1 | 197k | 1.19 MB | 150k | 52k | 28k | 308k | 79k | 116k | 10k |
| **total** | 527 | 5,413k | 100.0 | 2,358k | 18.04 MB | 2,272k | 2,073k | 1,954k | 2,431k | 217k | 173k | 1,791k |

## SmolLM2-135M, hand-written qhf path (export_hf.py)

- device cycles per step: **22.22M** (444 ms at 50 MHz)
- bytes loaded: 138.44 MB; at bwtest's 7.94 B/cycle that is 17.44M (78%) — the load floor; LD busy 17.64M (79%)
- EX 17.36M (78%) (useful 16.96M (76%)), VE / SFU 3.03M (14%), ST 0.32M (1%)
- fixed overhead (all engines idle, plus rt_fw outside the counter window): 0.28M (1%)
- overlap: the busy columns add up to 38.36M against 22.03M non-idle cycles, so 16.33M engine cycles ran alongside another engine
- not hidden behind the load floor: 4.78M (22%); scoreboard waits (HAZ_*) 20.48M

| dispatch type | per step | cycles | % | LD busy | LD bytes | floor | EX | EX useful | VE | ST | idle | overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matvec_like_NxNxN_iNxiNxiN | 121 | 18,705k | 84.2 | 17,218k | 135.83 MB | 17,107k | 17,168k | 16,810k | 506k | 145k | 35k | 16,356k |
| reduction_N_fN | 182 | 1,127k | 5.1 | 133k | 0.90 MB | 114k | 0k | 0k | 931k | 26k | 50k | -3k |
| elementwise_N_fN | 31 | 876k | 3.9 | 28k | 0.19 MB | 24k | 0k | 0k | 819k | 24k | 8k | -0k |
| reduction_NxNxD_fN | 30 | 357k | 1.6 | 13k | 0.04 MB | 5k | 0k | 0k | 325k | 9k | 9k | -3k |
| batch_matmul_NxNxDxN_iNxiNxiN | 30 | 256k | 1.2 | 43k | 0.26 MB | 33k | 88k | 73k | 107k | 9k | 9k | -2k |
| elementwise_N_fNxfNxiN | 121 | 224k | 1.0 | 64k | 0.39 MB | 50k | 0k | 0k | 84k | 51k | 33k | -2k |
| batch_matmul_NxNxNxD_iNxiNxiN | 30 | 205k | 0.9 | 41k | 0.23 MB | 29k | 103k | 73k | 35k | 13k | 9k | -8k |
| reduction_NxNxN_fN | 30 | 173k | 0.8 | 26k | 0.15 MB | 19k | 0k | 0k | 130k | 10k | 8k | -1k |
| elementwise_NxN_fN | 60 | 116k | 0.5 | 46k | 0.28 MB | 35k | 0k | 0k | 40k | 18k | 17k | -1k |
| elementwise | 151 | 58k | 0.3 | 3k | 0.00 MB | 0k | 0k | 0k | 9k | 2k | 56k | -2k |
| elementwise_broadcast_N_fN | 30 | 31k | 0.1 | 11k | 0.07 MB | 9k | 0k | 0k | 5k | 9k | 8k | -0k |
| elementwise_broadcast_N_iNxiN | 1 | 31k | 0.1 | 1k | 0.00 MB | 1k | 0k | 0k | 30k | 0k | 0k | -0k |
| (4 other types) | 93 | 61k | 0.3 | 17k | 0.09 MB | 12k | 0k | 0k | 13k | 4k | 34k | -1k |
| **total** | 910 | 22,220k | 100.0 | 17,644k | 138.44 MB | 17,436k | 17,359k | 16,955k | 3,034k | 321k | 277k | 16,332k |

Prefill, per chunk of 8 tokens (after the first chunk):

- device cycles per chunk: **40.89M** (818 ms at 50 MHz)
- bytes loaded: 122.93 MB; at bwtest's 7.94 B/cycle that is 15.48M (38%) — the load floor; LD busy 15.86M (39%)
- EX 14.20M (35%) (useful 13.69M (33%)), VE / SFU 21.20M (52%), ST 1.40M (3%)
- fixed overhead (all engines idle, plus rt_fw outside the counter window): 0.72M (2%)
- overlap: the busy columns add up to 52.66M against 40.38M non-idle cycles, so 12.28M engine cycles ran alongside another engine
- not hidden behind the load floor: 25.41M (62%); scoreboard waits (HAZ_*) 29.93M

| dispatch type | per step | cycles | % | LD busy | LD bytes | floor | EX | EX useful | VE | ST | idle | overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matmul_like_NxNxNxN_iNxiNxiN | 119 | 16,549k | 40.5 | 13,604k | 107.77 MB | 13,573k | 13,432k | 13,160k | 1,183k | 653k | 34k | 12,345k |
| reduction_NxN_fN | 179 | 10,683k | 26.1 | 936k | 7.17 MB | 904k | 0k | 0k | 9,677k | 71k | 49k | 35k |
| elementwise_NxN_fN | 30 | 6,922k | 16.9 | 188k | 1.47 MB | 186k | 0k | 0k | 6,543k | 185k | 8k | -0k |
| reduction_NxNxD_fN | 180 | 1,502k | 3.7 | 65k | 0.14 MB | 17k | 0k | 0k | 1,337k | 42k | 54k | -21k |
| reduction_NxNxN_fN | 240 | 1,383k | 3.4 | 210k | 1.23 MB | 155k | 0k | 0k | 1,042k | 78k | 68k | -6k |
| batch_matmul_NxNxDxN_iNxiNxiN | 184 | 1,121k | 2.7 | 202k | 1.13 MB | 142k | 323k | 265k | 507k | 43k | 55k | -8k |
| (8 other types) | 1332 | 2,728k | 6.7 | 650k | 4.02 MB | 507k | 450k | 265k | 912k | 324k | 451k | -62k |
| **total** | 2264 | 40,889k | 100.0 | 15,855k | 122.93 MB | 15,483k | 14,205k | 13,690k | 21,202k | 1,396k | 719k | 12,282k |

## SmolLM2-135M, generic path (unmodified HF)

- device cycles per step: **26.67M** (533 ms at 50 MHz)
- bytes loaded: 140.93 MB; at bwtest's 7.94 B/cycle that is 17.75M (67%) — the load floor; LD busy 17.97M (67%)
- EX 18.43M (69%) (useful 17.92M (67%)), VE / SFU 5.28M (20%), ST 0.30M (1%)
- fixed overhead (all engines idle, plus rt_fw outside the counter window): 0.30M (1%)
- overlap: the busy columns add up to 41.98M against 26.46M non-idle cycles, so 15.52M engine cycles ran alongside another engine
- not hidden behind the load floor: 8.92M (33%); scoreboard waits (HAZ_*) 23.63M

| dispatch type | per step | cycles | % | LD busy | LD bytes | floor | EX | EX useful | VE | ST | idle | overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matvec_like_NxNxN_iNxiNxiN | 211 | 21,109k | 79.1 | 17,214k | 135.70 MB | 17,091k | 17,168k | 16,810k | 2,089k | 142k | 61k | 15,546k |
| batch_matmul_NxNxNxN_iNxiNxiN | 60 | 2,430k | 9.1 | 434k | 3.29 MB | 415k | 1,257k | 1,106k | 657k | 52k | 16k | -18k |
| reduction_NxNxN_fN | 60 | 2,226k | 8.3 | 69k | 0.43 MB | 54k | 0k | 0k | 2,098k | 45k | 17k | -2k |
| reduction_N_fN | 182 | 396k | 1.5 | 86k | 0.53 MB | 67k | 0k | 0k | 271k | 3k | 50k | -2k |
| elementwise_N_fNxfNxiN | 121 | 159k | 0.6 | 64k | 0.39 MB | 50k | 0k | 0k | 57k | 14k | 33k | -2k |
| elementwise_N_fN | 61 | 118k | 0.4 | 47k | 0.28 MB | 35k | 0k | 0k | 40k | 19k | 17k | -1k |
| elementwise | 151 | 58k | 0.2 | 3k | 0.00 MB | 0k | 0k | 0k | 9k | 2k | 56k | -2k |
| elementwise_N_iNxfNxfNxfN | 30 | 54k | 0.2 | 23k | 0.14 MB | 17k | 0k | 0k | 16k | 9k | 8k | -0k |
| elementwise_broadcast_N_fN | 30 | 31k | 0.1 | 11k | 0.07 MB | 9k | 0k | 0k | 5k | 9k | 8k | -0k |
| elementwise_broadcast_N_iNxiN | 1 | 31k | 0.1 | 1k | 0.00 MB | 1k | 0k | 0k | 30k | 0k | 0k | -0k |
| elementwise_NxN_fNxfNxfNxiN | 30 | 29k | 0.1 | 11k | 0.06 MB | 8k | 0k | 0k | 11k | 1k | 8k | -1k |
| scatter_NxNxNxNxNxNxiN_dispatch_tensor_store | 30 | 14k | 0.1 | 2k | 0.01 MB | 1k | 0k | 0k | 0k | 1k | 14k | -0k |
| (3 other types) | 33 | 15k | 0.1 | 2k | 0.01 MB | 1k | 0k | 0k | 0k | 2k | 14k | -0k |
| **total** | 1000 | 26,671k | 100.0 | 17,965k | 140.93 MB | 17,749k | 18,425k | 17,916k | 5,284k | 300k | 303k | 15,516k |

Prefill, per chunk of 8 tokens (after the first chunk):

- device cycles per chunk: **65.03M** (1301 ms at 50 MHz)
- bytes loaded: 142.20 MB; at bwtest's 7.94 B/cycle that is 17.91M (28%) — the load floor; LD busy 18.25M (28%)
- EX 21.17M (33%) (useful 19.97M (31%)), VE / SFU 35.16M (54%), ST 1.62M (2%)
- fixed overhead (all engines idle, plus rt_fw outside the counter window): 0.69M (1%)
- overlap: the busy columns add up to 76.19M against 64.54M non-idle cycles, so 11.65M engine cycles ran alongside another engine
- not hidden behind the load floor: 47.12M (72%); scoreboard waits (HAZ_*) 47.03M

| dispatch type | per step | cycles | % | LD busy | LD bytes | floor | EX | EX useful | VE | ST | idle | overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matmul_like_NxNxNxN_iNxiNxiN | 204 | 29,077k | 44.7 | 13,873k | 109.78 MB | 13,827k | 13,458k | 13,188k | 12,857k | 637k | 57k | 11,787k |
| batch_matmul_NxNxNxN_iNxiNxiN | 368 | 14,904k | 22.9 | 2,665k | 20.21 MB | 2,545k | 7,710k | 6,783k | 4,030k | 321k | 101k | -110k |
| reduction_NxNxN_fN | 424 | 13,673k | 21.0 | 352k | 2.26 MB | 284k | 0k | 0k | 12,936k | 293k | 116k | -15k |
| reduction_NxN_fN | 180 | 4,774k | 7.3 | 559k | 4.24 MB | 535k | 0k | 0k | 4,176k | 5k | 49k | -1k |
| elementwise_NxN_fNxfNxiN | 119 | 940k | 1.4 | 412k | 3.12 MB | 393k | 0k | 0k | 405k | 99k | 33k | -2k |
| elementwise_NxN_fN | 59 | 597k | 0.9 | 167k | 1.23 MB | 154k | 0k | 0k | 281k | 137k | 17k | -1k |
| (5 other types) | 765 | 1,065k | 1.6 | 219k | 1.36 MB | 171k | 0k | 0k | 471k | 126k | 314k | -4k |
| **total** | 2119 | 65,029k | 100.0 | 18,245k | 142.20 MB | 17,909k | 21,169k | 19,971k | 35,156k | 1,619k | 686k | 11,653k |

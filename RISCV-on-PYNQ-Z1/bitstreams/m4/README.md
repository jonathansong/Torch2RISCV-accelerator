# M4 overlay (prebuilt)

Double-buffered accelerator (`rtl/sysarray`) at **D = 16**: 16×16 array with
8 DSP columns + 8 LUT columns, vector engine VL = 16, one DMA port (HP2).
Built with Vivado 2024.1 (`scripts/build_bitstream.sh -jobs 4 -sa_d 16`).
Setup WNS +1.319 ns at 50 MHz; 47618 LUT (89.5 %), 34183 FF, 184 DSP,
130 BRAM36. The firmware reads D from CAPS and the driver from the .hwh, so
the same binaries run on the M3 (D = 8) overlay.

Verified on the board (`notebooks/m4_demo.py`), all bit-exact vs NumPy.

int32 GEMM (peak 256 MAC/cycle):

| GEMM (M×N×K) | bias | result | cycles | MAC/cycle | % peak | M3, D = 8 |
|---|---|---|---|---|---|---|
| 16×16×16 | – | PASS | 1310 | 3.1 | 1.2 % | – |
| 48×32×80 | yes | PASS | 3452 | 35.6 | 13.9 % | – |
| 64×64×64 | – | PASS | 3738 | 70.1 | 27.4 % | 37.5 |
| 128×128×128 | – | PASS | 15880 | 132.1 | 51.6 % | 49.7 |
| 32×256×64 | yes | PASS | 11859 | 44.2 | 17.3 % | – |
| 256×256×256 | – | PASS | 92986 | 180.4 | 70.5 % | 56.5 |
| 512×256×256 | – | PASS | 175556 | 191.1 | 74.7 % | – |
| 256×128×1024 | – | PASS | 186174 | 180.2 | 70.4 % | – |

Firmware schedule tuning on this bitstream (`notebooks/m4_sched_tune.py`,
A prefetch + B split for B >= 16 KB, now the `gemm_fw` default): 128³ 156.7,
256³ 202.6 (79 % of peak), 512×256×256 213.9, 256×128×1024 222.2 (87 %),
int8 256³ 181.9 MAC/cycle; details in docs/double_buffer_design.md §10.3.

Fused int8 GEMM (bias + RELU + requant on the VE): 64³ 57.5, 128³ 120.5,
256³ 167.6 MAC/cycle, all PASS.

Standalone vector ops: all 10 cases PASS; int32 add 16384 elements at
7.88 B/cycle (the HP port limit), int8 add 1.19 elements/cycle.

DMA (one HP port): contiguous LD/ST 7.91–7.94 B/cycle, LD + ST together
15.76 B/cycle, 1-beat bursts 3.81 B/cycle (same as M2/M3).

Legacy firmware (8×8×8 jobs zero-padded to 16×16): `matmul_insn_fw` 1024/1024
(387.0 cycles/job), `matmul_fw` 1024/1024 (533.4 cycles/job).

Needs `firmware/{gemm,vector,bwtest,matmul,matmul_insn}/*.bin` and `driver/pynq_matmul.py`.

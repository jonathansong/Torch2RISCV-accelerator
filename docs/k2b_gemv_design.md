# K2b / H4：2 口读入 + 流式 GEMV 单元的微结构

（2026-10-09。依据：`kv260_upgrade_plan.md` K2b 第 1 步的性能模型，`compiler/tests/perf_model.py`。）

## 1. 目标

decode 每步的时间几乎全在线性层的 matvec 上：Qwen3 每步读 607.5 MB 权重，LD 忙碌 39.5M 周期、EX 38.5M（每步 43.4M），两边都是每周期 16 字节，**同时饱和**。性能模型给出：

| 方案（100 MHz） | Qwen3 decode | SmolLM2 decode |
|---|---|---|
| 只加读入（32 B / 周期） | ×1.02 | ×1.03 |
| 只加 GEMV 单元（32 B / 周期） | ×1.02 | ×1.05 |
| 读入 32 + GEMV 32 | **×1.85**（2.31 → 4.26 token/s，设备时间） | **×1.76** |
| 读入 64 + GEMV 64（4 口） | ×3.15 | ×2.74 |

第一阶段的目标是 **P = 2 个读口、每周期 32 字节的 decode matvec**，频率不变（100 MHz），与功能仿真器逐位一致；结构上能扩展到 P = 4。

## 2. 关键观察：decode 的权重只用一次，不必进 SPAD

现在 decode 的线性层（`expandLinear`，`SahlExpandKernels.cpp`）每个 chunk 做：

```
LD   weights (nc strips x K x 16 B) -> SPAD_B bank (ci & 1)      64 KB 一块（Qwen3）
EX   a = A strip (x 复制到 16 行), b = SPAD_B, c = ACC, kt = K/16, rep = nc, cstep 1, crow nc
VE   epilogue: 读 ACC 的第 0 行（word c .. c+nc-1），写 fp32 结果
ST   结果
```

板上轨迹（Qwen3 `matvec_like_64x16x2048`）：`LD rows 2 rb 32768` → `EX kt 128 rep 2 bstep 2048 cstep 1 crow 2`，epilogue 只读 word c、c+1（第 0 行）。

- 权重先写进 SPAD_B，再由 EX 读出一次，就再也不用了。SPAD_B 的写口和读口都被占满，却没有复用。
- 阵列的 16 行里只有第 0 行有用（x 被复制到 16 行），乘累加的利用率是 1/16。

所以 H4 不经过 SPAD：**LD 引擎把权重直接流进 GEMV 单元**，每个 128 位 beat 就是一个 k 行的 16 个权重（与 SPAD_B 的字布局相同，DDR 里的打包格式不变），乘上 x[k] 累加到 16 个输出列。这样：

- 每个读口对应一组 16 个乘累加，**读口之间互不争用本地存储**：2 口就是 32 B / 周期，不需要 SPAD 子银行，也就没有子银行对齐、按子银行查冲突这些编译器约束（原 K2b 方案 A 的大部分工作推迟到需要 prefill 带宽时再做）；
- SPAD_B 在 decode 里空出来，EX 阵列空出来（decode 的注意力仍用阵列）；
- 整数乘累加与顺序无关，结果与 EX 的第 0 行逐位相同。

## 3. 总体结构

```
            HP1 (m0)            HP2 (m1)
               |  AR/R            |  AR/R
        +------v------------------v------+
        |  sa_ld   burst 队列 每口 QD = 8 |----> SPAD / ACC 写（每周期一个口，16 B，不变）
        |  GEMV 模式：两口同周期各收一拍  |
        +------+------------------+------+
               | beat + (s, k)    | beat + (s, k)
        +------v------+    +------v------+
        | lane grp 0  |    | lane grp 1  |    x 缓冲：每组一份（K_max = 4096 字节）
        | 16 x int8   |    | 16 x int8   |    x[k] 每周期每组读一个字节
        | MAC, acc[s] |    | MAC, acc[s] |    acc：每组 S_MAX x 16 x int32
        +------+------+    +------+------+
               +--------+---------+
                        v  写回：每个 strip 两组相加（可再加 ACC 原值）-> ACC 一个字（16 x int32）
                     ACC side B（LD 的请求口）
```

`sa_gemv.v` 是新模块，挂在 `sa_ld` 旁边，由 LD 引擎驱动；调度器把一条 GEMV 当作 LD 引擎的命令。

## 4. 命令：LD 的 GEMV 模式

不加新的引擎和操作码，用 LD 描述符的 `mode = 2`（现有 0 LINEAR、1 INTERLEAVE）：

| 字段 | LD 描述符位置 | GEMV 含义 |
|---|---|---|
| `ddr` | w1[31:0]（可重定位、动态字段 1） | strip 0 的权重起点（16 字节对齐） |
| `laddr` | w2[31:0] | mem = ACC，word = C（结果的第一个字） |
| `rows` | w2[47:32] | S：strip 数（1 .. S_MAX = 8） |
| `row_bytes` | w2[63:48] | 每个 strip 的字节数 = K × 16（K ≤ 4095，16 的倍数） |
| `pitch` | w3[31:0] | strip 之间的 DDR 间距（打包权重 = K × 16） |
| `mode` | w3[33:32] | 2 = GEMV |
| `xword` | w3[49:34]（新） | x 在 SPAD_A 的起始字（x 连续存放，K/16 个字） |
| `cstep` | w3[57:50]（新） | strip s 的结果写到 word C + s × cstep |
| `acc` | w3[58]（新） | 1：加到 ACC 原值上（K 分段时用） |

**语义**（功能仿真器按此实现）：对 s < S、j < 16

```
ACC[C + s*cstep][j] = (acc ? ACC[C + s*cstep][j] : 0) + Σ_{k<K} sx8(x[k]) * sx8(W_s[k][j])     (int32 回绕)
x[k]    = SPAD_A 字 xword + k/16 的第 k%16 字节
W_s[k][j] = DDR[ddr + s*pitch + k*16 + j]
```

只写每个 strip 的第 0 行（一个 ACC 字）。EX 版本还写了第 1..15 行（复制的 x 算出的相同值）；epilogue 只读第 0 行，所以对编译出的程序没有影响。

**调度器**（`sa_sched.v` d1 / d2 级）：GEMV 的读集 = SPAD_A [xword, xword + K/16)（acc = 1 时加上 ACC 的结果字），写集 = ACC [C, C + (S−1)·cstep]，引擎 = LD。现有的 bank 记分板直接适用：epilogue 的 VE 读 ACC → 等这条 GEMV 完成；下一条 GEMV 写另一个 ACC bank → 可以与 epilogue 并行（现在的双缓冲方式不变）。检查项：`row_bytes % 16 == 0`，`rows ≤ S_MAX`，范围不越界，`mem == ACC`。

## 5. 2 口读入（`sa_ld`）

现有的 `sa_ld` 已经有 `NPORTS` 参数：burst（≤ 16 拍、不跨 4 KB）轮流分给各口，每口 8 个在途，返回的 beat 按 burst 记下的本地位置写入。但 **R 侧每周期只接收一个口**（`r_sel`），本地写口也只有一个，所以多口不增加带宽。改动：

1. **AXI**：`NPORTS = 2`，m0 → `S_AXI_HP1_FPD`（不变，仍与描述符取指共用 AR / R，经 owner FIFO 区分），m1 → `S_AXI_HP2_FPD`（新增协议转换与地址映射；`build_bitstream.sh -nports 2`）。ST 也自动用两个口的写通道。
2. **burst 队列多记两个字段**：GEMV 模式下每个 burst 记下 (strip s, 起始 k)。beat 到达时 k 逐拍加 1，跨 strip 时 s 加 1、k 归零（burst 最长 256 字节 = 16 个 k 行，strip 长度是 16 字节的倍数，所以一个 beat 不会跨 strip）。
3. **R 侧**：GEMV 模式下每个口独立 `m_rready`，同一周期两个口各交一拍给自己的 lane 组；非 GEMV 模式保持现在的单口仲裁（写 SPAD / ACC 仍是 16 B / 周期，见 §8 第二阶段）。GEMV lane 组有 2–3 级流水，`rready` 直接为 1（没有反压，除非结果写回阶段）。
4. **完成**：所有 beat 都进了 MAC 流水、写回结束后 `done`；`err` 规则不变（任何 beat 的 SLVERR / DECERR）。

两个 HP 口到 DDR 控制器的路径不同，3.2 GB/s 远低于 KV260 DDR4 的峰值（约 19 GB/s），但两口并发的实际带宽要用 `bwtest` 新增的双口测试确认（并比较 HP1 + HP2 与 HP1 + HP3）。

## 6. GEMV 单元（`sa_gemv.v`）

每个 lane 组（每个读口一组）：

- **x 缓冲**：K_MAX = 4096 字节，每组一份（两个读口同周期各读一个 x[k]）。用 RAMB18（4 KB × 8 位）或 LUTRAM。命令开始时由 LD 引擎从 SPAD_A 读 K/16 个字（经 SPAD_A side A 的 LD 请求口，一周期一字，K = 3072 时 192 周期，与第一拍的 DDR 延迟部分重叠）。**驻留标记** (xword, K, valid)：一个 matvec 的所有 chunk 用同一个 x，第二条起标记命中就不再读；任何写 SPAD_A 的 LD / VE 命令清除标记（保守，硬件里只需一个 OR）。
- **乘累加**：一拍 = 16 个 int8 权重 × 同一个 x[k]。16 个乘法共享一个操作数，可以用 DSP48E2 的 int8 打包（两个权重共用 x，一个 DSP 两个乘法）：每组 8 个 DSP；或 LUT 乘法器。流水：beat → 读 x[k]（1 拍）→ 乘（1–2 拍）→ 累加。
- **累加器**：acc[s][j]，S_MAX × 16 × 32 位触发器（每组 4096 位）。同一个 burst 的 16 拍都属于同一个 strip，累加是 32 位加法 + S_MAX 选一，一拍内完成（100 MHz 没有问题）。
- **写回**：所有 beat 消化完以后，s = 0 .. S−1：两组的 acc[s] 相加（acc = 1 时再加 ACC 原值，经 ACC side B 的 LD 请求口读一拍），写 ACC 字 C + s·cstep（16 × int32，全部字节使能）。每个 strip 1–2 周期。

吞吐：每口每周期 16 字节 → P = 2 时 32 B / 周期。Qwen3 一个 64 KB 的 chunk：65（延迟）+ 2048（数据）+ 2（写回）≈ 2115 周期，现在 LD 与 EX 各约 4160 周期。

## 7. 编译器与运行时

- **目标配置**：`TargetConfig` 加 `gemv_ports`（0 = 没有 GEMV 单元；`--iree-sa-gemv-ports=N`，只支持 D = 16）；`.hwh` / 驱动报告 `NPORTS` 与是否有 GEMV。
- **SAHL**：新 op `sahl.gemv(x, w, acc)`（x：SPAD_A 中连续的 int8 向量；w：DDR 中 nc 个 strip 的子视图；acc：ACC 的 nc × 16 块）。`expandLinear` 在目标有 GEMV 时：不再分配 A strip，每个 chunk 的 `sahl.load` 权重 + `sahl.mma` 换成 `sahl.gemv`（nc > 8 时分成每条 ≤ 8 个 strip 的几条）；SPAD_B 的两个权重 bank 不再需要。x 是 int8 时直接用；x 是 int32（在 ACC 里，stories / SmolLM2 / Qwen3 的 decode 都是）时用一条 `sahl.strip {sa.packed}` 转成 SPAD_A 中连续的 int8（与原来的 strip 同样的 VE 转换，只是不复制到 16 行）。epilogue、ACC 布局（`sa.word`）、`sahl.loop` 的 advance（ddr 每次加两个 chunk）都不变。
- **约束**：K ≤ `gemv_kmax`（4095），nc ≤ S_MAX；不满足时退回现在的 LD + EX（正确但慢）。权重的打包格式与 irpa 不变。
- **SAHW / 序列化**：LD 描述符的新字段（mode 2、xword、cstep、acc）；`DescList` 编码；动态字段只用到 ddr（字段 1）。
- **功能仿真器 / oracle**：`sa_funcsim.ld` 实现 mode 2（§4 的语义，逐位）和 sched 的检查；`dispatch_check` 覆盖 decode 的所有 matvec；新目标配置建立自己的黄金语料基线（§5 第 3 条约束）。
- **性能模型**：`perf_model.py` 加 GEMV-LD 的代价（延迟 + x 读入（未命中时）+ 字节 / (16 × P) + 写回），用编译出的新轨迹重新预测。

## 8. 阶段

| 阶段 | 内容 | 验证 |
|---|---|---|
| **G1** | 功能仿真器 + 编译器（`sahl.gemv`、目标配置开关），不碰硬件 | Qwen3 / SmolLM2 / stories 的 decode 在 sim 上与现有路径逐位一致；`dispatch_check`；新基线 |
| **G2** | RTL：`sa_gemv` + `sa_ld` 的 GEMV 模式 + `sa_sched` 的检查与掩码，先 NPORTS = 1（功能正确，16 B / 周期） | `tb_sa_gemv`（随机 K / S / cstep / acc / 对齐，对照 funcsim）；`tb_sa_ld`；`tb_sa_unit`；固件系统仿真 `desc_run`；`make test / test16 / test128` |
| **G3** | NPORTS = 2：R 侧双口同时接收、BD 加 HP2、`bwtest` 双口测试 | `tb_sa_dma GEN="NP=2 D=16 DMA_W=128"` 加 GEMV 模式；用户构建比特流；板上 bwtest |
| **G4** | 板上验收 | REGRESSION PASS、逐位一致；decode 的 token/s 与模型预测（×1.76–1.85）对比 |

**第二阶段（按需）**：P = 4（HP0 现在是 PicoRV32 的口，要把它挪到 HPC0 或 LPD；×2.6–3.2）；SPAD 子银行（LD 写 SPAD / ACC 也到 32 B / 周期，prefill ×1.07）；LD 命令流水（隐藏每条命令约 65 周期的延迟，+2–3%）。

## 9. 资源与时序估计（P = 2）

| 部分 | LUT | FF | DSP | BRAM |
|---|---|---|---|---|
| `sa_gemv`：32 个 int8 乘（DSP 打包） | — | — | 16 | — |
| 累加器 2 × 8 × 16 × 32 位 + 32 个 32 位加法 + 选择 | ~2k | ~8.5k | — | — |
| 写回：16 个 32 位加法（两组 + ACC 原值） | ~0.6k | ~0.5k | — | — |
| x 缓冲 2 份 | — | — | — | 1 RAMB36（两个 RAMB18） |
| `sa_ld` R 侧双口 + burst 标签 | ~0.5k | ~0.3k | — | — |
| BD：第二个 HP 口的协议转换 | ~1–2k | ~2k | — | — |
| **合计** | **~4–5k** | ~11k | 16 | 1 |

现在 LUT 88.25k / 117k（75%）、FF 约 61k / 234k、DSP 347 / 1248、BRAM 130 / 144，都放得下（LUT 约 79%）。新路径都很短（一级 32 位加法、DSP 乘法），100 MHz 下不会成为关键路径。

## 10. 风险

- **两口并发的 DDR 带宽**：模型假设每口 16 B / 周期；用 G3 的 bwtest 双口测试确认，不足时先查 HP 口与 DDR 控制器端口的分配。
- **m0 与描述符取指共用**：取指在 m0 上与权重 burst 交错；轮流分配时 m0 多一点负担。可以让 GEMV 的 burst 优先走 m1，或者把取指挪到 m1。
- **x 驻留标记**：清除条件故意保守（任何写 SPAD_A 都清除）；功能仿真器同样实现标记并在 `dispatch_check` 中比较，标记错误会表现为结果不一致。
- **只写第 0 行**：依赖 epilogue 只读第 0 行。编译器只在 `expandLinear` 生成 GEMV，那里的 epilogue 布局固定；funcsim 的结果比较会抓到任何读了第 1..15 行的情况。

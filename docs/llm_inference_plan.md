# LLM 推理：架构方案与实施步骤

状态：**L0、L1 已完成并上板验证**。后续各级仍是方案。

**目标**：在 PYNQ-Z1 上做一个**架构与工业界主流 LLM 推理加速器一致**的完整系统，
端到端地在板上运行 llama2 结构的模型（TinyStories），生成文本。系统由四部分组成：

- **host**：ARM，运行时；
- **命令处理器**：PicoRV32，运行常驻固件；
- **加速器**：DMA、int8 矩阵引擎、fp32 向量引擎与特殊函数单元；
- **编译器**：以 **MLIR/IREE** 为目标。

**速度的定位**：架构对齐是第一目标。资源允许、成本又低的提速照做（§0）。

**时钟**：**50 MHz 是必须达到的目标**；75 MHz 作为端到端跑通后的附加目标，单独一级（L5.5）。

相关文档：

- [`double_buffer_design.md`](double_buffer_design.md)：当前加速器，§3 数据布局、§6 VE、§7 记分板、§8 指令与描述符；
- [`perf_counters_and_desc_dma_plan.md`](perf_counters_and_desc_dma_plan.md)：性能计数器与描述符 DMA（M5）；
- [`../notebooks/llm/README.md`](../notebooks/llm/README.md)：第一步测量。结论：单序列 decode 受带宽限制，D=16、50 MHz 下只计 GEMV 为 24.5 tok/s（stories15M），与 ARM 的 int8 版本 22.9 tok/s 基本持平。

---

## 目录

- [0. 原则](#0-原则)
- [1. 对标工业界架构](#1-对标工业界架构)
- [2. 总体架构与一个 token 的数据流](#2-总体架构与一个-token-的数据流)
- [3. 数值方案](#3-数值方案)
- [4. L0：基线、参考模型、功能模拟器与工具链验证](#4-l0基线参考模型功能模拟器与工具链验证)
- [5. L1：host 接口与命令处理扩展](#5-l1host-接口与命令处理扩展)
- [6. L2：fp32 向量引擎与特殊函数单元](#6-l2fp32-向量引擎与特殊函数单元)
- [7. L3：量化格式与 GEMM 后处理](#7-l3量化格式与-gemm-后处理)
- [8. L4：一个 decoder 层在设备上执行](#8-l4一个-decoder-层在设备上执行)
- [9. L5：ARM 运行时与端到端生成](#9-l5arm-运行时与端到端生成)
- [10. 编译器接口：MLIR / IREE](#10-编译器接口mlir--iree)
- [11. L5b、L5.5、L6：多序列并发、75 MHz 与后续扩展](#11-l5bl55l6多序列并发75-mhz-与后续扩展)
- [12. 资源与时序预算](#12-资源与时序预算)
- [13. 总体计划与验收](#13-总体计划与验收)
- [14. 风险与对策](#14-风险与对策)
- [附录 A：编码变化汇总](#附录-a编码变化汇总)
- [附录 B：新增与修改文件](#附录-b新增与修改文件)

---

## 0. 原则

1. **架构对齐优先，低成本的提速照做**。阵列退回 D=8，把 LUT 让给 fp32 向量单元。这对单序列 decode
   没有损失：decode 受带宽限制，D=8 的 EX 每周期正好消耗 8 个权重，与 LD 的约 7.9 B/cycle 相当。
   在资源允许的前提下：
   - VE 的逐元素运算做成流水结构（§6.7）；
   - 调度上让 LD、EX、VE 重叠；
   - 保留多序列并发和 prefill 的路径（§11）。
2. **一切都能逐位验证**。Python **功能模拟器**直接执行描述符列表。
   - 板上结果、RTL 联合仿真和功能模拟器三者逐位比对；
   - 精度只在“功能模拟器 vs fp32 参考模型”这一处评估。
3. **先测量再定方案**。量化方案先在主机上用参考模型测精度（L0），通过后再做硬件。
   IREE 运行时能否在 32 位 ARM 上运行，也在 L0 验证。
4. **面向编译器设计接口**。硬件提供 IREE 需要的抽象：binding、push constant、动态形状、命令缓冲、
   barrier、信号量（§10）。先手写列表生成器（L3–L5），之后由 IREE 目标后端替代，两者输出同一种描述符。
5. **时钟**：
   - 现有模块不做时序修改，在 50 MHz 下已有余量；
   - **新写的模块**（fp32 单元、SFU、取指单元的新译码）**都按 13.3 ns（75 MHz）设计流水**，
     只多用一些 FF（目前 FF 用了 34%），为 L5.5 保留余地。
6. **每一级都能单独上板**，有独立的板上脚本和验收标准，与 M1–M5 的做法相同。
7. **向后兼容**：
   - 现有指令、描述符和整数 VE 的语义不变；
   - 新功能通过新 opcode 或 key、以前的保留位和新 CAPS 位加入；
   - 现有测试全部保留。

---

## 1. 对标工业界架构

主流推理加速器（数据中心 NPU、手机 NPU、带 RISC-V 控制核的 AI 芯片）的共同结构，以及本项目的对应部分：

| 层次 | 工业界通行做法 | 本项目现状 | 本方案 |
|---|---|---|---|
| 编译器 | 框架 → MLIR 等 IR → 算子融合、分块、布局 → 设备代码（如 IREE、XLA、TVM） | 手写固件和驱动 | IREE：先用模板库，再做完整的代码生成（§10） |
| host 运行时 | 加载模型、分配 KV cache、调度 prefill/decode、分词、采样 | Python 驱动，每次调用一个算子 | L5 用 Python 调通，L6-IREE 换成 C 写的 IREE HAL 驱动 |
| host ↔ 设备接口 | 内存中的命令队列、门铃、完成中断或信号量 | mailbox 参数，每种算子一个固件，ARM 轮询 | L1：DDR 命令环、门铃、完成记录、PL→PS 中断 |
| 命令处理器 | 设备上的微控制器（很多芯片用 RISC-V）取命令、分派、同步 | PicoRV32，按算子换固件 | L1：常驻固件 `rt_fw` |
| 命令缓冲与可执行体 | 录一次、改参数就能重放（CUDA Graph、Vulkan 命令缓冲）；binding 与 push constant；位置无关的 kernel | 描述符列表 + 4 个 BASE（M5） | L1：16 个 BASE、8 个 PARAM、动态字段、SETREG、LOOP、CALL/RET、LDPARAM |
| 矩阵引擎 | int8/fp8/bf16 脉动阵列或张量核 | int8 16×16，int32 累加 | 保留 int8，改为 D=8 |
| 向量单元与特殊函数单元 | fp32/bf16 向量 ALU、归约，查表计算 exp/rsqrt 等 | 只有整数逐元素运算和 requant | L2：fp32 VPU 与 SFU，外加归约、转置和掩码 |
| 量化 | W8A8：权重按通道、激活按 token 动态量化；KV cache 用 int8/fp8 静态 scale | 只有一个整型 scale | L3：同样的方案 |
| 注意力 | KV cache 在显存里，decode 时整个注意力在设备上算 | 没有 | L4：q·Kᵀ、softmax、p·V 全在设备上 |
| 可观测性 | 硬件计数器、追踪 | 32 个计数器（M4p） | 保留，并新增 VPU 与取指事件 |

**不做的部分**：
- 多核和片上网络；
- HBM；
- 分页 KV cache（L6 可选）；
- 连续批处理；
- fp8/bf16 矩阵乘。

这些是规模上的差别，不是结构上的差别。

**EX 保持 int8**：
- W8A8 的 int32 累加是精确的，scale 在 VE 的后处理中乘一次，数学上与先反量化再做浮点乘法等价；
- 64 个 fp32 乘加单元约需 32k LUT、128 个 DSP（估计），放不下。

---

## 2. 总体架构与一个 token 的数据流

```
┌──────────────── ARM Cortex-A9 (host) ────────────────┐
│ 运行时：L5 用 Python LlamaDevice / L6 用 IREE + HAL 驱动   │
│  模型加载 → DDR │ KV cache │ 静态描述符列表（每个模型一张）  │
│  每个 token：写参数块（PARAM）+ 命令环条目 │ 分词、采样     │
└──────┬───────────────────────────────────▲───────────┘
       │ ① 写条目 + flush                   │ ⑤ IRQ (PL→PS)，读完成记录
       │ ② 门铃：mailbox RING_TAIL          │
┌──────▼───────────────────────────────────┴───────────┐
│ PicoRV32 常驻固件 rt_fw（命令处理器）                    │
│  ③ 取条目 → 设置 BASE0-3、PARAM0-7 → mat_submit → mat_fence │
│  ④ 写完成记录 → mat_notify                              │
└──────┬───────────────────────────────────────────────┘
       │ PCPI
┌──────▼─────────────── 加速器 sa_unit（D = 8）───────────┐
│ 描述符取指：BASE0-15 │ PARAM0-7 │ 动态字段 │ SETREG │     │
│             LOOP │ CALL/RET │ LDPARAM                  │
│ → 调度器（记分板）                                       │
│   LD / ST（DMA，HP2）│ EX（int8 阵列）│ VE（int + fp32 VPU/SFU）│
│   SPAD_A 128 KB │ SPAD_B 128 KB │ ACC 256 KB（int32 / fp32）│
└──────────────────────────────────────────────────────┘
DDR：权重（int8 + scale）、KV cache（int8）、RoPE 表、嵌入表、logits、列表、命令环、参数块
```

**一个 token 的执行流程**：

1. ARM 采样出上一个 token 后，写一个 32 字节的参数块：pos、pos+1、pos_pad、token×dim 等（§8.5）。
   **列表本身不改**；
2. 提交命令环条目，然后按门铃；
3. `rt_fw` 把参数块装进 PARAM，然后执行这个模型的静态列表。列表中每一层依次是：
   RMSNorm → QKV → RoPE → 追加 KV → 注意力 → Wo → 残差 → RMSNorm → SwiGLU FFN → 残差；
   层与层之间用 LOOP 串起来；最后是分类层；
4. logits（fp32）写回 DDR；
5. `rt_fw` 写完成记录，然后发中断；
6. ARM 采样。

---

## 3. 数值方案

### 3.1 格式

| 数据 | 格式 | 位置 |
|---|---|---|
| 线性层权重 W（out × in） | int8 对称量化，**每个输出通道一个 fp32 scale** `s_w[o] = max|W[o,:]| / 127`；DDR 中转置存放为 Wᵀ（in × out），作为 B 操作数 | DDR |
| 线性层输入激活 | int8 对称量化，**每个 token 动态计算** `s_x = max|x| / 127`，在设备上算 | SPAD |
| 矩阵乘累加 | int32 | ACC |
| 残差流、norm、RoPE、softmax、SiLU 的中间值 | **fp32** | ACC |
| KV cache | int8，**每层 K 和 V 各一个静态 scale**（L0 离线标定） | DDR |
| q（注意力） | int8，每个头动态 scale | SPAD |
| softmax 概率 p | int8，静态 scale 1/127 | SPAD |
| 嵌入表 | int8，每行一个 fp32 scale | DDR |
| RMSNorm 权重、RoPE 的 cos/sin 表 | fp32 | DDR |
| logits | fp32 | DDR |

这就是 W8A8 的 SmoothQuant 式做法，外加 int8 KV cache，是工业界 int8 推理的常见组合。
llama2.c 的 Q8_0（沿 K 方向每 32 个元素一组 scale）在脉动阵列上需要把 K 拆开分别累加，
所以本方案不采用；如果 L0 测出按通道量化的精度不够，再作为备选（§14）。

### 3.2 fp32 运算的约定

硬件、功能模拟器和编译器三方都照这份约定实现。它同时就是给 IREE 的数值约定：按 fast-math 声明（§10.4）。

- 加法和乘法按 IEEE-754 binary32 实现，**舍入到最近偶数（RNE）**。
- **非规格化数清零（FTZ）**：输入和输出的非规格化数都当作同号的 0。
- 溢出得到 ±inf。NaN 按 IEEE 规则传播，但 RTL 测试不覆盖 NaN。
- 没有融合乘加：`a·x + b` 是一次乘法、一次加法，舍入两次。
- 类型转换：
  - int32 → fp32 按 RNE；
  - fp32 → int8/int32 也按 RNE，再饱和；int8 的范围是 [−127, 127]；
  - （可选）bf16 ↔ fp32：截断，加最近偶数舍入。
- 归约的**加法顺序是固定的**（§6.4）。
- 特殊函数 EXP、RECIP、RSQRT 用固定的算法实现（§6.5），模型实现的是**同一套算法**，而不是 libm。
  所以除法 = RECIP 加乘法，**不是 IEEE 精确除法**。

在这些约定下，fp32 的每一步都可以用 NumPy float32 加 FTZ 处理精确复现。

---

## 4. L0：基线、参考模型、功能模拟器与工具链验证

L0 不改 RTL。

### 4.1 D=8 基线 bitstream 与时序摸底

- 构建 `scripts/build_bitstream.sh -jobs 4 -sa_d 8`，PERF 和 DESC 都保留，50 MHz。
- 记录资源，作为 L1/L2 的预算基准。M3 在 D=8 时是 21,720 LUT、96 DSP、130 BRAM36；
  加上计数器和取指单元后估计约 24k LUT。
- **时序摸底**：同一份 RTL 在 75 MHz 约束下做一次 OOC 综合，记录最差的 10 条路径，作为 L5.5 的输入。
  已知 m5（D=16）的最差路径是 `ex/drain_active` 驱动所有 PE 寄存器的使能（CE）：
  16.8 ns，其中走线占 15.8 ns，属于扇出问题。
- 板上回归：`m4_perf.py`、`m5_desc_demo.py`、`m4_demo.py` 都能按 D=8 运行。
- 归档为 `bitstreams/l0/`。

> **L0 实测**（`bitstreams/l0/`）：
> - **资源**：23,452 LUT（44.1%）、19,193 FF、96 DSP、130 BRAM36；
> - **时序**：50 MHz 下 WNS +1.555 ns；
> - **板上回归全部 PASS**：`m4_demo`（256³ 59.0 MAC/cycle，峰值的 92%）、`m4_perf` 69/69、`m5_desc_demo`。
>
> **75 MHz OOC 时序摸底**（`rtl/sysarray`，`make synth CLOCK_NS=13.333`，D=8）：WNS −0.717 ns，估计最高约 71.2 MHz。
> 最差的 20 条路径**全部在 `sa_ld.v`**，分两类：
> 1. **突发生成**：从 `off` 出发，先算地址，再取三个量的最小值决定突发长度（本行剩余、到 4 KB 边界的距离、16 拍），
>    然后乘以 `step`（经过 DSP）得到 `cur_word`。17 级逻辑，13.9 ns。
> 2. **读数据写回的地址**：从 `q_rp` 出发，计算 `lw_word = h_word + (h_sum >> lpwl) * step`（经过 DSP），
>    驱动 SPAD/ACC 的地址、数据和写使能。6 级逻辑，13.5 ns，其中走线占约 60%。
>
> 两类都是 `* step` 乘法串在组合路径上。改法：把乘法换成随拍递增的累加器；写回的地址、数据和使能多打一拍寄存器。
> 这两处就是 L5.5 的修复清单。整机构建还要加上 PicoRV32 和 AXI 互连，以 L5.5 的实际构建为准。

### 4.2 参考模型 `llm/ref_model.py`

读取 llama2.c 的 fp32 checkpoint，用 NumPy 写两种前向计算：

- **fp32 模式**：与 `run.c` 的计算顺序一致。
  - 验收：stories15M 贪心解码 256 个 token，文本与 `run.c` 完全相同。
- **设备数值模式**：用 §3.1 的量化方式模拟设备的计算。特殊函数先用 float64 近似，L2 之后换成功能模拟器。

精度评估（`llm/eval_quant.py`）：
- **teacher forcing**：两个模式都输入 fp32 模式生成的 token 序列，逐位置比较两者的 top-1 是否一致，
  同时算 logits 的 KL 散度。
- 模型：stories15M、42M、110M；
- 数据：约 8 条提示词，每条 256 个 token；
- 同时标定每层 KV 的静态 scale：取绝对值最大值或 99.99 分位数，两种都试。
- **验收门槛**：15M 的 top-1 一致率 ≥ 95%，贪心生成的文本通顺。
  - 未达标时按 §14 的顺序改方案；
  - 这个 95% 是初定值，L0 测完后写入本文档，作为之后各级的精度基准。

> **L0 实测**（主机，`llm/eval_quant.py`，详见 [`llm/README.md`](../llm/README.md)）：
> - 三个模型、两种 KV 标定都达到 95% 的门槛，设备模型的 argmax 100% 落在 fp32 的前 5 名内；
> - top-1 一致率：15M 为 95.6%（absmax）/ **96.3%**（99.99 分位数），42M 为 96.0% / 95.9%，110M 为 97.9% / 98.1%；
> - KL 散度均值：15M 约 0.011，42M 0.007–0.010，110M 约 0.004。
>
> **精度基准定为**：stories15M、99.99 分位数的 KV scale、top-1 96.3%。
> L2 把特殊函数换成硬件算法之后要重新评估，结果仍须 ≥ 95%。

### 4.3 功能模拟器 `llm/sa_funcsim.py`

用 Python 模拟描述符列表的执行：

- **状态**：DDR（字节数组）、SPAD_A、SPAD_B、ACC，按字存储，布局与设计文档 §3.4 一致；
  另外有 BASE 和 PARAM 寄存器。
- **LD / ST**：LINEAR、INTERLEAVE、ZERO_PAD。
- **EX**：包括 repeat、bstep、cstep、crow、accumulate。
- **VE**：现有的整数运算；L2 加入 fp32 的全部运算。
- **控制类描述符**：FENCE、JUMP、END、BASE 重定位；L1 的扩展（§5.3）也要实现。
- **周期估计模型**：给编译器用作代价模型。先按每种命令的理论周期估算，L1 之后用计数器校准。

**验收**：
- 用现有 `gen_desc_cases.py` 的 7 个 case 作输入，功能模拟器的输出与 RTL 联合仿真的期望字节逐位一致；
- 抽取 `m5_desc_demo.py` 里的几个列表，与板上结果比对。

之后每加一个硬件功能，都**先在功能模拟器里实现并给出测试向量**，再写 RTL。

### 4.4 IREE 工具链验证（最大的软件风险，放在最前面）

- **主机端**：安装或构建 IREE 编译器。用 iree-turbine 把一个小的 torch 模型（2 层 MLP，加一次 softmax）编译到 `llvm-cpu` 目标，
  target triple 是 `armv7-linux-gnueabihf`。
- **板上**：交叉编译 IREE 运行时（armhf，hard-float），用 `iree-run-module` 运行上面的模型，结果与 PyTorch 一致。
  IREE 主要测试的是 aarch64，32 位 ARM 没有保证，这一步就是要确认它能用。
- **还要确认**：
  - 自定义 HAL 驱动与目标后端的插件机制，在所选 IREE 版本上的接口；
  - C 运行时怎样访问 CMA 缓冲（u-dma-buf 或 PYNQ 的 CMA 接口）和中断（UIO）。
- **不通过时**：只用 IREE 编译器，运行时自己用 C 写一个小的执行器，读 IREE 输出的可执行体和调度信息。
  §10 的硬件接口不受影响。

> **L0 实测：通过**（IREE 3.11.0，详见 [`iree-sa/l0/README.md`](../iree-sa/l0/README.md)）。
> - **流程**：torch → iree-turbine → iree-compile（llvm-cpu，armv7）→ 板上的 IREE 运行时
>   （clang-18 交叉编译，静态链接）。
> - **结果**：测试模型用 local-sync 和 local-task 两种驱动都与 torch 一致，15 个单算子测试全部 PASS。
>
> 途中解决了 armv7 特有的两个问题，都通过链接器包装脚本加一个很小的 libm shim 解决：
> 1. **缺少 `fmaxf`/`fminf`**：Cortex-A9 没有 IEEE 语义的 max/min 指令，LLVM 会调用这两个函数，
>    而嵌入式 ELF 里没有 libm。
> 2. **IREE 自带 musl 里的 `fmaf` 是空实现（`unreachable`）**：Cortex-A9 没有 FMA，`exp` 的多项式近似会调用它，
>    结果 `exp` 总是返回常数 4。shim 提供了正确舍入的软件 `fmaf`，与 glibc 在 4000 万组输入上逐位一致。
>    这可能是 IREE 的 bug，值得向上游报告。
>
> 编译脚本会在主机上检查每个 armv7 可执行体，不允许有外部导入，也不允许 libm 函数是空实现。

---

## 5. L1：host 接口与命令处理扩展

L1 的硬件改动分两部分，一次构建：
- **命令环与完成中断**：面向 host 运行时；
- **命令处理扩展**：面向编译器，包括 IREE 的 binding、push constant、命令缓冲和动态形状。

> **L1 完成并上板验证**（`bitstreams/l1/`）。
> - **仿真**：
>   - 功能模拟器 272 项检查；
>   - `tb_sa_unit` 新增 15 项 L1 检查，8 个故意引入的错误都被发现；
>   - 两张 Python 生成的 L1 列表在 RTL 上联合仿真通过，D=8 和 16 都做了；
>   - `RING_TEST`：9 个条目提交到大小为 4 的命令环，3 个故意引入的固件错误都被发现；
>   - 所有旧测试不回归。
> - **板上**：`l1_ring_demo.py` 的 6 项测试全部通过；`m4_demo`、`m4_perf`、`m5_desc_demo` 不回归。
> - **与计划的差别**：
>   1. 完成中断改为边沿中断 `notify_irq`，因为 ARM 访问不到加速器的 CSR；
>   2. 驱动必须在读走完成记录之后才能复用命令环的槽。上板时发现，只按 `RING_HEAD` 判断环满，
>      会让固件覆盖还没读取的完成记录；已修正。
>   3. 取指单元实测 4.4k LUT，高于预估。

### 5.1 命令环

**提交命令环**：由 ARM 在 DDR 里分配，不开 cache，或者每次写完都 flush。条目个数是 2 的幂，默认 64 个；
每个条目 64 字节。

| 字 | 内容 |
|---|---|
| w0 | [7:0] 类型：0x01 RUN_LIST、0x02 NOP、0x03 RESET（`mat_reset` 后继续）；[8] 完成后中断；[9] 转存计数器；[63:32] `seq` |
| w1 | 列表地址（64 字节对齐） |
| w2 | [31:0] count（0 表示执行到 END） |
| w3–w6 | BASE0–BASE3 |
| w7 | [31:0] **参数块地址**：8 个 32 位 PARAM，按 32 字节对齐；0 表示不装载 |

BASE4–15 由列表内部的 SETREG 设置（§5.3）。

**完成记录环**：与提交环同样的索引，每条 32 字节。

| 字 | 内容 |
|---|---|
| c0 | [31:0] `seq`；[63:32] 状态：0 表示成功，否则是 `ext_status`，其中含错误引擎号和错误码 |
| c1 | [31:0] 周期数；[63:32] 执行过的描述符条数（`DL_EXEC`） |
| c2 | [31:0] END 的完成值（`DL_STATUS`）；[63:32] 保留 |
| c3 | 保留 |

**mailbox 新字段**（BRAM 0x1F00 起，0xB0–0xD0 目前空闲）：

| 偏移 | 名称 | 写方 | 说明 |
|---|---|---|---|
| 0xB0 | `RING_BASE` | ARM | 提交环物理地址 |
| 0xB4 | `RING_SIZE` | ARM | 条目数（2 的幂） |
| 0xB8 | `RING_TAIL` | ARM | **门铃**：新的尾指针 |
| 0xBC | `RING_HEAD` | 固件 | 已完成的条目数（单调递增） |
| 0xC0 | `CPL_BASE` | ARM | 完成记录环物理地址 |
| 0xC4 | `FW_STATE` | 固件 | 0x52554E00 `RUN\0` 表示就绪；出错时为 0xDEADxxxx |
| 0xC8 | `FW_VERSION` | 固件 | 接口版本 |
| 0xCC | `HEARTBEAT` | 固件 | 空闲循环计数，调试用 |

**门铃**：ARM 写 `RING_TAIL`，`rt_fw` 在空闲循环里轮询。这是一种简化：
真实芯片一般是写一个 MMIO 门铃寄存器，产生设备端中断。改成中断方式放在 L6。

**完成中断**（实现时修改为边沿中断，原因见下）：
- 新指令 `mat_notify`（funct7=1，funct3=7）：`sa_unit` 的新输出 **`notify_irq`** 产生一个脉冲
  （高 4 个周期，之后至少低 1 个周期；连续的 notify 会排队，保证每次都有独立的上升沿），`NOTIFY_COUNT` 加 1。
- BD：`irqConcat` 改为 `NUM_PORTS 2`，`In1` 接 `matmul_0/notify_irq`（接口声明为 `EDGE_RISING`），
  经已有的 `psInterruptController`（axi_intc）送到 `IRQ_F2P`。axi_intc 锁存边沿，ARM 在 intc 上应答。
- **为什么不用原计划的电平中断加 W1C**：当前 BD 中 `matmul_0` 的 CSR 只映射在 PicoRV32 的地址空间（0x80000000），
  ARM 访问不到，也就无法写 1 清除状态。边沿中断不需要清除加速器里的状态，性质上类似 PCIe 的 MSI 消息中断。
- ARM 端：Python 用 `pynq.Interrupt("matmul_0/notify_irq")`，C 运行时用 UIO；两者都保留轮询 `RING_HEAD` 的备用路径。
  多个边沿在 ARM 应答前可能合并成一次中断，所以 ARM 醒来后要按 `RING_HEAD` 处理所有已完成的条目。
- 为什么由固件发中断：固件必须先写完完成记录，ARM 被唤醒时才能读到一致的状态。
  END 描述符的 IRQ 位留给 ARM 直接提交的场景（L6）。

**新增 CSR**（AXI-Lite，只有 PicoRV32 可访问）：0xD4 `NOTIFY_COUNT`（只读）。
**CAPS bit 22** 表示有 `mat_notify` 和 `notify_irq`。

### 5.2 固件 `firmware/rt/rt_fw.c`

```c
init: 检查 CAPS（需要 bit 21 DESC、bit 22 NOTIFY、bit 24 CMDX）；FW_STATE = RUN；
loop:
  while (RING_HEAD == RING_TAIL) HEARTBEAT++;
  e = &ring[RING_HEAD % RING_SIZE];               // PicoRV32 经 HP0 读 DDR
  switch (e->type):
    RUN_LIST: mat_cfg(BASE0..3)；if (e->params) mat_cfg(PARAM0..7 ← 参数块)；
              t0 = rdcycle; mat_submit(e->list, e->count); mat_fence();
              st = ext_status;  if (error) mat_reset();
    NOP:      st = 0;
    RESET:    mat_reset(); st = 0;
  写完成记录；  [可选] 转存计数器；  RING_HEAD++；  if (e->flags & IRQ) mat_notify();
```

- 固件**常驻**：ARM 只加载一次，之后不再换固件。
- 程序必须放在 0x1E00 以下，这个循环很小，估计在 1 KB 以内。
- 现有的 `gemm_fw`、`vector_fw`、`desc_run_fw` 保留，供旧脚本使用。

### 5.3 命令处理扩展（面向编译器）

所有扩展都在 `sa_cmdfetch.v` 中，译码时生效。取指和译码本来就按顺序进行，所以后面的描述符自然看到新的寄存器值；
已经送出的命令包里的地址和长度都已算好，不受之后修改的影响。

| # | 扩展 | 编码 | 作用 |
|---|---|---|---|
| 1 | **BASE0–15** | BASESEL 加宽到 4 位：`{w0[14:13], w0[10:9]}`。旧列表的 w0[14:13] 为 0，含义不变 | IREE 一个 dispatch 常有 5–8 个 binding |
| 2 | **PARAM0–7**（32 位） | `mat_cfg` key 27–34；命令环参数块；SETREG；LDPARAM | 动态形状和位置，对应 IREE 的 push constant |
| 3 | **动态字段** | w0[31:16] 是两个 8 位动态槽（原为保留位）：`[3:0]` 字段号（0 = 不用），`[6:4]` 选哪个 PARAM，`[7]` 0 = 替换、1 = 相加 | 长度、次数、地址偏移可以取自 PARAM，于是一个模型只需一张静态列表 |
| 4 | **SETREG**（opcode 0x14） | w1：3 个 7 位项，位于 [6:0]、[14:8]、[22:16]，每项 {[6] 有效，[5:0] 寄存器号：0–15 = BASE，16–23 = PARAM}；w2–w4：对应的值；w5[2:0]：每项替换（0）还是相加（1）；按项的顺序依次写入 | 在列表内部绑定参数，相当于 Vulkan 的 bind descriptor set 和 push constants |
| 5 | **LOOP_END**（opcode 0x13） | w1：跳回的目标（相对偏移，单位是描述符条数，有符号）；w2[15:0]：次数（≥ 1，也可以用动态槽取自 PARAM）；w2[18:16] / w2[21:19]：每轮递增的 PARAM k1 / k2；w3 / w4：各自的步长（有符号 32 位） | 分块循环、按头循环、按层循环；对应 IREE 的 workgroup 数量 |
| 6 | **相对 JUMP、CALL、RET** | JUMP / CALL（0x15）：w2[0] = REL 时 w1 是有符号偏移（单位是描述符条数），否则是 64 字节对齐的绝对地址；CALL 把下一条的地址压入**深度 4** 的返回栈；RET（0x16）：弹出并跳转 | 位置无关的可执行体：命令缓冲 = SETREG … CALL 某个 kernel |
| 7 | **LDPARAM**（opcode 0x17） | w1：DDR 地址（可以重定位，4 字节对齐）；w2[2:0]：写入哪个 PARAM；w3[15:0]：乘数；w4：加数。结果 `PARAM = mem32 × mul + add` | 由数据决定的地址：嵌入表查找、按位置张量更新 KV、gather；如果数据刚由 ST 写入，前面必须有 FENCE |

**LOOP 的语义**：
- 循环体放在 LOOP_END 之前，执行 `count` 次，count 必须 ≥ 1；
- 计数器栈深度为 2，按 LOOP_END 自身的地址识别所属层：栈顶的地址与当前 LOOP_END 相同，就继续使用栈顶；
  否则压入一个新计数器；计数到 0 时弹出，继续往下执行；
- 每轮结束时做 `P[k1] += s1`、`P[k2] += s2`；循环结束后 PARAM 保持最终值，编译器需要时用 SETREG 复位；
- 跳回时丢弃已预取的描述符，重新取指，沿用 JUMP 的实现；每轮都会从 DDR 重新读取循环体，不做循环缓冲。

**动态字段号**：初稿如下，L1 设计评审时定稿。

| opcode | 字段号 |
|---|---|
| LD / ST | 1 DDR 地址，2 LADDR（字号），3 rows，4 row_bytes，5 pitch |
| EX | 1 A，2 B，3 C，4 Kt，5 repeat，6 bstep，7 cstep |
| VE | 1 src1，2 src2，3 dst，4 LEN，5 VALID，6 ROWLEN，7 P1，8 P2，9 A（fp32 位模式），10 B，11 IMM |
| LOOP_END | 1 次数 |
| LDPARAM | 1 地址 |

- 地址类字段的计算顺序：`BASE[sel] + 字段 + PARAM`；
- 替换或相加之后的值仍由调度器现有的范围检查把关，越界会报错。

**错误**：以下情况报错，错误引擎号为 FETCH（4），恢复方法仍是 `mat_reset`：
- 返回栈或计数器栈溢出、下溢；
- 次数为 0；
- 未定义的字段号；
- LDPARAM 读取时收到 AXI 错误。

**CAPS bit 24（CMDX）** 表示这些扩展都存在。

**`mat_cfg` 新 key**：15–26 设置 BASE4–15，27–34 设置 PARAM0–7。

### 5.4 驱动

- `driver/pynq_matmul.py` 新增 `class Device`：
  - `Device(bitfile)`：加载 `rt_fw`，分配两个环，等待 `FW_STATE == RUN`；
  - `submit(dl, bases, params=None, irq=True) -> seq`；
  - `wait(seq)` 和 `async wait_async(seq)`；
  - `run(dl, bases, params)`：同步调用的便捷接口。
- `DescList` 新增 `setreg`、`loop_end`、`call`、`ret`、`ldparam`，以及动态槽参数 `dyn=[(字段, param, add)]`。

### 5.5 验证

- **`tb_sa_unit`**：
  - `mat_notify`：每次一个 4 周期脉冲，连续两次产生两个独立的上升沿，`NOTIFY_COUNT` 计数，legacy `irq` 不受影响；
  - 负向测试原来用 funct3=7 作为非法编码，现在它已占用，改用 funct7=3；
  - **命令处理扩展的定向测试**：
    - 每个动态字段的替换和相加；
    - SETREG 对后续描述符生效，且不影响已送出的命令；
    - LOOP 的 1 次和 N 次、两层嵌套、PARAM 每轮递增；
    - CALL/RET 嵌套到深度 4；
    - LDPARAM 的乘加，以及“ST → FENCE → LDPARAM”的顺序；
    - 各类错误与恢复；
    - 与 FENCE、END、BASE0–15 混用；
  - **随机列表**：含动态槽、循环和调用，与功能模拟器逐位比对。
- **`tb_system.v` 新增 `RING_TEST`**：环中 N 条条目，覆盖回绕、NOP、一条故意出错的列表、RESET，以及带参数块的条目；
  核对每条完成记录和中断次数。
- **板上** `notebooks/llm/l1_ring_demo.py`：
  - 异步提交 200 条列表，结果全部正确，中断次数与完成数一致；
  - 用一张带 LOOP 和动态字段的静态列表，只改参数块就对不同长度的输入给出正确结果；
  - 测量提交到完成的延迟，只作记录。

---

## 6. L2：fp32 向量引擎与特殊函数单元

### 6.1 运算流水（fp32 模式）

VE 命令增加 **FP 位**。FP=0 时行为与现在完全相同；FP=1 时，每组（VL = D 个元素）依次经过：

```
src1 ─ 取数（下标模式 M1）─ 转 fp32 ─ [SWAPNEG] ─┐
                                               ├─ OP ─ ×A + B ─ FUNC ─ [RELU] ─┬─ 输出转换 → dst
src2 ─ 取数（M2，或立即数 IMM）─ 转 fp32 ────────┘                              └─ 归约 → dst
```

| 级 | 内容 |
|---|---|
| 输入 | src1 与 src2 的类型可以是 I8（SPAD）、I32 或 F32（ACC），读出后转成 fp32；src2 也可以是命令中的 fp32 立即数 `IMM` |
| SWAPNEG（src1，可选） | 每对相邻元素 (2i, 2i+1) 变为 (−x[2i+1], x[2i])，给 RoPE 用 |
| OP | ADD、SUB、MUL、MAX、MIN、COPY（与整数模式编码相同） |
| 仿射 | y = y × A + B，A、B 是 fp32 立即数，默认 1.0 和 0.0 |
| FUNC | NONE、EXP、RECIP、RSQRT、ABS |
| RELU | 与现有位相同 |
| 输出 | F32 写 ACC；I32 按 RNE 转换并饱和后写 ACC；I8 按 RNE 转换、饱和到 ±127 后写 SPAD；或者进入归约 |

新增类型编码 **VT_F32 = 3**，只能放在 ACC。（可选）VT_BF16 = 4，放在 SPAD，布局同 int16，只做存储格式和转换。

### 6.2 下标模式

src1 与 src2 统一。设输出组号为 g：

| 模式 | src 字 | 用途 |
|---|---|---|
| LINEAR | g | 普通逐元素运算 |
| MOD P | g mod P | 按列广播：每通道 scale、RMSNorm 权重、RoPE 表（现有的 period 模式） |
| DIV P | ⌊g / P⌋ | 按行广播：每行一个标量（max、sum、s_x）；**复制行**：P = D 时把一个向量展开成 A 条带的 D 行（§8.2） |
| IMM（仅 src2） | — | 命令中的 fp32 常数 |

两个源各有自己的 P：src1 用新字段 P1，src2 用现有的 period 字段 P2。

### 6.3 转置 TRANSPOSE（op=6）

按 D×D 块转置一个“R 行 × S 字”的矩阵，每行 S·D 个元素：
- 输出第 b 个块的第 i 个字 = 输入第 b 个块中每个字的第 i 个元素；
- 输入第 b 个块的第 j 个字位于 `src + (⌊b/S⌋·D + j)·S + (b mod S)`；
- 输出第 b 个块的第 i 个字位于 `dst + b·D + i`。

效果：一个行主序的 K（pos × hs）直接变成矩阵引擎的 B 条带布局（Kᵀ，每 D 个位置一段），见 §8.3。
- 支持 I8（SPAD → SPAD）和 F32/I32（ACC → ACC）；
- 硬件上用一个 D×D 的寄存器缓冲，吞吐为每周期一个字。

### 6.4 归约与尾部掩码

REDUCE ∈ {NONE, SUM, MAX}，按行进行：一行 = `ROWLEN` 个组。行内先按 lane 累加，再把 D 个 lane 合并。
**每行只输出一个 fp32 字，值广播到全部 D 个 lane**，这样后续用 DIV 模式就能直接按行取用。

- **VALID**：每行只有前 VALID 个元素有效，0 表示全部有效。
  - 无效元素在归约中取单位元：SUM 取 0，MAX 取 −inf；
  - 逐元素输出时，无效元素写 0。
- **加法顺序**（规范，模型必须照做）：
  1. 每个 lane 按组号顺序**依次**累加：`acc = ((v0 + v1) + v2) + …`；
  2. 然后对 D 个 lane 按 lane 号两两配对，做二叉树合并：`(l0+l1)+(l2+l3)…`。
- 归约走多拍路径（§6.7），每个组要等加法器的延迟，吞吐约为每 3–5 个周期一组，换来简单的硬件和确定的顺序。

### 6.5 特殊函数

| 函数 | 算法 | 目标精度（对照 float64） |
|---|---|---|
| EXP | t = x·log2(e)，拆成 n = ⌊t⌋ 和 f = t − n；2^f 用 64 段的表 T[i] 加斜率 S[i] 线性插值；结果再乘 2^n（直接加到指数上）；x < −87 输出 0，x > 88 输出 +inf | 相对误差 ≤ 3e-5 |
| RECIP | x = m·2^e；1/m 用 64 段的表线性插值得到初值；再做一次牛顿迭代 r = r·(2 − m·r)；最后把指数取负 | 相对误差 ≤ 2e-7 |
| RSQRT | 按指数奇偶把 m 规约到 [1, 4)；查表初值；再做一次牛顿迭代 r = r·(1.5 − 0.5·m·r²) | 相对误差 ≤ 5e-7 |
| ABS | 清掉符号位 | 精确 |

表格由 `llm/sfu_tables.py` 生成，RTL 和功能模拟器共用这一份。
其他激活函数都由这些基本函数组合，例如 sigmoid(x) = RECIP(1 + EXP(−x))，SiLU(x) = x·sigmoid(x)。

### 6.6 编码

**VE 描述符**：M5 格式的扩展。FP=0 时新增字段必须为 0，旧列表语义不变。

| 字 | 现有内容 | 新增 |
|---|---|---|
| w1 | src1、src2 LADDR | — |
| w2 | dst LADDR、LEN | — |
| w3 | [7:0] op（op 6 = TRANSPOSE），[13:8] types（[1:0] 输入，[3:2] 输出，[5:4] src2 类型，见 §6.9），[31:16] P2（period），[47:32] scale，[52:48] shift | [53] FP，[56:54] FUNC，[58:57] M1，[60:59] M2（3 = IMM），[62:61] REDUCE，[63] SWAPNEG |
| w4 | [31:0] zp，[63:32] lo | — |
| w5 | [31:0] hi | [63:32] IMM（fp32） |
| w6 | 保留 | [31:0] A（fp32），[63:32] B（fp32） |
| w7 | 保留 | [15:0] ROWLEN，[31:16] VALID，[47:32] P1，[63:48] S（转置的行宽，单位是字） |

- **PCPI `vec_cfg` 新 key**：10 FLAGS（对应 w3[63:53]），11 IMM，12 A，13 B，14 ROWLEN/VALID，15 P1/S。
- 调度器命令包 `SA_PKT_W` 从 261 位加宽；调度器 FIFO 用的是 LUTRAM，加宽不影响 BRAM。
- **CAPS bit 23** 表示有 fp32 VE。

### 6.7 实现：混合结构

每个 lane 的数据通路：

```
cvt(src1) ─ SWAPNEG ─┐
                      ├─ [M1 乘 | A1 加减 | 比较] ─ [M2 乘 A] ─ [A2 加 B] ─┬─ cvt/饱和 ─► 输出 FIFO
cvt(src2)/IMM ────────┘                                                   │
                                        FUNC ≠ NONE 或 REDUCE ≠ NONE：回到 M2/A2 做多拍微操作
                                        （查表、插值、牛顿迭代、累加）
```

- **流水部分**：输入转换 → OP → 仿射 → 输出转换。每个 lane 有 2 个乘法器（M1、M2）和 2 个加法器（A1、A2）。
  **逐元素运算和仿射每周期处理一组**，这是后处理、RoPE、残差、量化这些高频操作的路径。
- **多拍部分**：特殊函数和归约**复用 M2、A2**，由所有 lane 共用的一个微程序控制器按步执行。
  - EXP 约 8 步，RECIP/RSQRT 约 10 步，归约每组 1 步，外加加法器延迟；
  - 这段时间里读数单元由 `fp_ready` 暂停；
  - 特殊函数只出现在 softmax、norm、SiLU 中，元素数量不多。
- **为什么不全做成多拍**：
  - 单序列 decode 时，多拍结构的 VE 只多约 6% 的时间；
  - 但多序列并发和 prefill 时，VE 的工作量随行数成倍增加，而读权重的量不变。
    例如 8 条序列同时解码时，多拍结构的 VE 会占约 40% 的时间，流水结构约 5%（估计）。
  - 资源放得下（§12），所以选流水。
- **新文件**：
  - `sa_fp32_add.v`、`sa_fp32_mul.v`、`sa_fp32_cvt.v`：自己写，不用 Xilinx IP，这样 xsim 不需要 IP 仿真库；
    都**按 13.3 ns 设计流水**：加法器 5–6 级，乘法器 3–4 级，每个乘法器用 2 个 DSP；
  - `sa_sfu_seq.v`：微程序控制器与 LUTRAM 表格；
  - `sa_ve_fp.v`：D 个 lane，含下标生成、归约和转置缓冲。
- **与 `sa_ve.v` 的关系**：读写口、命令接口、FIFO 信用计数、写回级和记分板接口都共用，由 FP 位选择路径。
  - 整数路径不动，M3 的测试不受影响；
  - FIFO 信用数从 8 增加到 16，以覆盖更长的流水。
- **调度器**：新模式（DIV/MOD、IMM、TRANSPOSE、归约）下的**精确访问范围**：
  - DIV P 模式的最后一个字 = 起始 + ⌊(组数 − 1)/P⌋；
  - IMM 模式没有 src2；
  - TRANSPOSE 按块的公式计算。
  这些既用于范围检查，也用于记分板的 bank 掩码。记分板仍按 bank 跟踪；改成更细的按区跟踪是可选项，看数据再决定（§11）。
- **新增计数器事件**：`VE_FP_ACTIVE`、`VE_SFU_BUSY`、`VE_REDUCE`，编号在 L2 设计评审时确定。

### 6.8 验证

- **基本单元的 testbench**：`fp32_add`、`fp32_mul`、`fp32_cvt` 各 10⁶ 组随机输入，外加边界值：
  ±0、非规格化数、±inf、恰好在 .5 处的舍入、溢出。与 NumPy float32 加 FTZ 处理逐位比对。
- **`tb_sa_ve_fp`**：测试向量由功能模拟器生成，逐位比对。覆盖：
  - 每一种 OP、FUNC、下标模式和类型组合；
  - 归约（含 VALID）、转置、SWAPNEG；
  - fp32 → int8 的饱和与舍入。
- **SFU 精度测试**：扫描 10⁶ 个输入，最大相对误差必须满足 §6.5 的目标。
- **`tb_sa_unit`**：fp32 VE 与 EX、LD 并发时的记分板冲突测试，以及精确访问范围的测试。
- **系统级**：`gen_desc_cases.py` 增加 softmax、RMSNorm、RoPE、转置等列表，在 `tb_system` 上做联合仿真。
- **板上** `notebooks/llm/l2_vpu_demo.py`：同样这些算子，板上结果与功能模拟器逐位一致。
- 所有现有测试不回归：D=8 和 16，`PERF = 0`。

### 6.9 实现记录（与上文设计的差异）

**编码补充**
- **src2 类型**：types[5:4] 单独给出 src2 的类型：0 与 src1 相同，1 I8，2 I32，3 F32。
  例如反量化时 src1 是 I32 累加结果，src2 是 F32 的每通道 scale。只在 FP=1 时有效。
  驱动写作 `DescList.ve(..., t2=...)`，固件用 `SA_VT2(t)`。
- **PCPI 路径**：`sa_pcpi` 保存 key 10–15 的值，A 的复位值是 1.0。
  - 只有 FP 命令才把这些字段放进命令包；S 只随 TRANSPOSE 发送；
  - 所以整数 VE 命令不受之前 `vec_cfg` 设置的影响。

**范围规则**（调度器与功能模拟器相同，都是保守的规则）
- DIV 模式的源、以及 REDUCE 的输出，都按 G 个字做范围检查，而不是按 §6.7 的精确公式。
- 记分板仍按 bank 跟踪，这样做不影响正确性，只是少数合法命令会被拒绝，编译器必须遵守。
- TRANSPOSE 要求 (G/D) mod S = 0。功能模拟器检查这一条，RTL 不检查（结果未定义）。
- LEN 必须是 ROWLEN 的整数倍。功能模拟器报 SHAPE 错误，RTL 不检查。

**结构**（`rtl/sysarray/sa_vefp.v`）
- 整数的 `sa_ve.v` 保持原样，fp32 部分是并列的新模块 `sa_vefp`：
  - `sa_unit` 按 FP 位和 op=6 把命令分给两者之一；
  - 两者不会同时忙，存储口和事件按位或；
  - 计数器事件复用现有的 VE 事件，没有新增编号。
- **lane 折叠**：`sa_vefp` 只有 FL = D/2 个物理 fp lane（参数，也可以设成 FL = D）。
  - 一组 D 个元素分成两半，在相邻的两个周期依次进入流水，输出时再拼回一个字；
  - 原因：D 个 lane 的版本在 D=8 时 sa_unit 综合后约 51k LUT，加上 PicoRV32 装不进 xc7z020（布局失败）；
    折叠后 sa_vefp 从 28.9k 降到 17.6k LUT，sa_unit 约 40k；
  - 代价：二元运算本来每组就要读两次，所以速度不变；一元运算和多拍模式慢一半；
  - 指令集、数值和功能模拟器都不变，结果仍然逐位一致。
- **流水模式**：FUNC 为 NONE 或 ABS，且没有 REDUCE。
  - 每 lane 一条固定延迟的流水：i2f ×2 → SWAPNEG → A1/M1/比较 → M2（×A）→ A2（+B）→ RELU/VALID → f2i；
  - 每周期发出半组，输出 FIFO 32 项。
- **多拍模式**：EXP、RECIP、RSQRT 或 REDUCE。
  - 一次只处理一组；A2 之后由微程序控制器复用各 lane 的 M2、A2、i2f、f2i，按 `llm/sfu.py` 的步骤逐个半组执行；
  - 查表只有一个 ROM，按 lane 依次读取；
  - 之后做 RELU/VALID，再归约（A2 或比较器，最后是 lane 二叉树）或输出转换。
- **TRANSPOSE**：一个状态机，把 D 个字按步长 S 读进 D×D 寄存器缓冲，再写出 D 个转置后的字。
  支持 I8（SPAD）和 I32/F32（ACC）。
- **基本单元**：`sa_fp32_add.v` 4 级，`sa_fp32_mul.v` 4 级（DSP），`sa_fp32_cvt.v` 的 i2f 和 f2i 各 2 级。
- **SFU 表格**：`rtl/sysarray/gen_sfu_tables.py` 从 `llm/sfu.py` 生成 `sa_sfu_tables.vh`，一张 256 项的 ROM。
- **SFU 实测精度**（对照 float64）：EXP 1.84e-5，RECIP 1.18e-7，RSQRT 1.17e-7。

**验证**
- `tb_fp32`：56 万组以上向量，含 RNE 恰好在中点的情况；变异测试能检出错误。
- `tb_sa_vefp`：31 个用例，D=8 和 16，FL = D/2 和 FL = D 都测；折叠逻辑的变异测试能检出错误。
- `tb_sa_unit`：CAPS 检查，以及经 PCPI `vec_cfg` 的 fp32 与 TRANSPOSE 定向测试。
- 系统级 `desc_run` 联合仿真：
  - 4 个 L2 列表：softmax、GEMM + 反量化、RMSNorm + 量化、带 TRANSPOSE 的注意力分数；
  - 12 个随机 fp32 列表：输入含 NaN、inf、非规格化数；
  - 预期值由功能模拟器给出，与 RTL 逐位一致。
- 板上 `notebooks/llm/l2_vpu_demo.py`：
  - 同样这些列表，外加每次运行几百个随机 fp32 列表；
  - 功能模拟器拒绝的命令，设备也必须报错。
- **板上结果**（`bitstreams/l2`：44,005 LUT，82.7%；WNS +1.836 ns，50 MHz）：
  - 13 个描述符列表全部通过：softmax 4,235 周期，GEMM + 反量化 783，RMSNorm + 量化 997，注意力分数 524；
  - 300 个随机 fp32 列表与功能模拟器逐位一致；错误隔离通过；
  - L1、M5、M4 的回归测试全部通过。

---

## 7. L3：量化格式与 GEMM 后处理

### 7.1 模型导出 `llm/export_w8a8.py`

把 llama2.c 的 fp32 checkpoint 导出成设备用的 `*.w8a8` 文件：一个头部加若干段，每段 64 字节对齐。

| 段 | 内容 |
|---|---|
| header | 魔数、版本、dim、hidden、layers、heads、vocab、seq_len、D、各段偏移 |
| embed | 嵌入表 int8（vocab × dim），外加 s_emb fp32（vocab） |
| 每层（各层结构相同，按固定步长存放，方便 LOOP 按层递增地址） | rms_att、rms_ffn（fp32，dim）；**Wqkvᵀ**（dim × 3·dim，int8）加 s_w；Woᵀ 加 s_w；**W13ᵀ**（dim × 2·hidden，W1 与 W3 按输出通道拼接）加 s_w；W2ᵀ 加 s_w；K、V 的静态 scale（写入该层描述符的立即数） |
| final | rms_final（fp32）；Wclsᵀ（dim × vocab）加 s_w |
| rope | cos 和 sin 表，fp32，seq_len × hs/2，每个元素在一对位置上各存一次（与 SWAPNEG 的配对一致），并按 D 对齐 |

KV scale 在不同层取值不同，而 LOOP 按层重复的是同一段循环体。所以每层放一小段参数（fp32 位模式），
用 LDPARAM 读进 PARAM，再通过动态槽替换 VE 的 A 字段。

### 7.2 GEMM 与后处理的命令序列

y = x·Wᵀ：

1. **激活量化**（x 是 fp32 行向量）：
   - `amax = REDUCE MAX(ABS(x))`；
   - `inv = RECIP(amax × (1/127))`，得到 127/amax；
   - `s_x = amax × (1/127)`；
   - `xq = x × inv`（src2 用 DIV 模式），输出 I8，src1 用 **DIV D** 复制成 A 条带的 D 行（§8.2）。
2. **GEMM**：Wᵀ 按列分块放进 SPAD_B，两个 bank **交替使用**：计算第 i 块的同时加载第 i+1 块。
   A 是复制行后的 xq，EX 输出到 ACC（int32）。分块循环用 LOOP 表示，块的地址用 PARAM 按轮递增。
3. **后处理**（与下一块的加载重叠）：
   - `y = acc × s_w`：src2 用 MOD 模式，输入 I32 输出 F32；
   - `y = y × s_x`：src2 用 DIV 模式。

**L3 验收**：
- 板上 `l3_linear_demo.py` 用 stories15M 的真实权重，逐层比对 QKV、Wo、W13、W2 和分类层的输出，
  与功能模拟器逐位一致，与 fp32 参考的误差在 L0 标定的范围内；
- **计数器指标**：分类层 GEMV 的 LD 忙碌 ≥ 90%，说明 LD 与 EX 确实重叠了。

### 7.3 实现记录与结果

**文件**
- `llm/export_w8a8.py`：导出与读取 `.w8a8`。头部之后是张量表（名字、偏移、字节数、是否按层）。
  - 线性层权重按 B 条带**预先打包**：第 t 块是输出通道 [tD, (t+1)D) 的 (in × D) 行主序 int8；
  - 所以任意连续的几块在 DDR 中是连续的，一次 LINEAR LD 就能装进 SPAD_B，不需要跨步 DMA；
  - 打包与 D 有关，头部记录 D。stories15M 在 D=8 时是 24.2 MB。
- `llm/compile_layer.py`：手写的列表生成器。
  - `quant_act`：4 条 VE，依次算 amax、s_x、1/s_x，最后用 src1 DIV D 输出复制行的 int8 A 条带；
  - `linear`：每块 NC 个输出块。NC 取能整除输出块数的最大值，同时要放得下 SPAD_B 一个 bank 和 ACC 的暂存区；
  - DDR 地址都通过 BASE 重定位（BASE0 = 模型，BASE1 = 输入输出）；
  - 输出到 DDR 且块数多（分类层）时，用 LOOP_END 循环一对块，PARAM0/1 按步长递增 DDR 地址：
    列表只有 32 条描述符，实际执行 1046 条。
- `ref_model.SfuExact`：DeviceModel 用上与硬件逐位一致的 SFU。
- `llm/test_l3.py`：主机测试。用真实输入跑第 0、5 层和分类层，与 DeviceModel 逐位比对；
  `--save` 输出板上测试用的用例文件。
- `notebooks/llm/l3_linear_demo.py`：板上测试。L3 不改硬件，用的是 `bitstreams/l2`。

**验证**
- 主机上，D=8 和 16 时各层都与 DeviceModel 逐位一致，与 fp32 参考的相对误差为 0.37–3.27%（W2 最大）。
  循环形式与展开形式的结果完全相同。
- 变异测试：把反量化顺序改成先乘 s_x 再乘 s_w，每一层都被检出不一致。
- RTL 联合仿真：`desc_run` 增加一个合成的 L3 线性层（64 → 2880，强制使用循环形式），D=8 和 16 都通过。

**板上结果**（stories15M，D=8，50 MHz）

| 层 | in → out | 周期 | 权重 B/周期 | EX useful | LD 忙碌 |
|---|---|---|---|---|---|
| Wqkv | 288 → 864 | 42,121 | 5.91 | 73.7% | 75.9% |
| Wo | 288 → 288 | 16,997 | 4.88 | 60.7% | 63.5% |
| W13 | 288 → 1536 | 67,517 | 6.55 | 81.8% | 84.0% |
| W2 | 768 → 288 | 36,319 | 6.09 | 75.9% | 78.4% |
| 分类层 | 288 → 32000 | 1,258,768 | 7.32 | 91.5% | **93.9%** |

- 9 个线性层的板上结果与板上运行的功能模拟器、主机的 DeviceModel 都逐位一致。
- **分类层 LD 忙碌 93.9%，验收（≥ 90%）通过**。LD 每个忙碌周期 7.91 B，接近一个 HP 口的上限。
- 小的层 LD 忙碌只有 64–84%：第一块的加载和最后一块的后处理无法重叠，块数少时占比大。
  L4 把一层的多个线性层串成一个列表，可以把下一个线性层的第一块提前加载。
- 一层的 4 个线性层合计约 163k 周期（3.3 ms），6 层约 19.7 ms；加上分类层 25 ms，
  线性层部分约 45 ms/token（约 22 tok/s 的上限，不含注意力和其他 VE 运算）。

---

## 8. L4：一个 decoder 层在设备上执行

### 8.1 片上存储分配（D=8，示意，L4 设计时再细化）

| 区域 | 用途 |
|---|---|
| ACC bank 0 | 残差流 x（fp32，dim/D 个字）；xn；q、k、v；注意力输出；FFN 中间值 h（fp32）；标量（amax、s_x、sum 等，一个字一个） |
| ACC bank 1 | GEMM 输出（int32，复制行 D × N）；scores |
| SPAD_A | 复制行的 xq、q、p（int8）；RoPE 表的当前行；各种 scale 向量 |
| SPAD_B | Wᵀ 的列分块（两个 bank 交替使用）；注意力时放 Kᵀ 条带和 V |

手写路径（L4/L5）让残差流一直留在 ACC 中。IREE 路径中，dispatch 之间的数据经过 DDR（§10.3）。

### 8.2 复制行布局

A 条带的布局是：字 `w·D + g` = `A[g][w·D .. w·D+D−1]`。
如果**所有 D 行都放同一个向量 v**，那么字 `w·D + g` 就是 v 的第 w 段，这正好等于用 VE 的 src1 **DIV D** 模式
展开紧凑存放的 v。

于是 C 的 D 行完全相同，第 0 行按字连续存放（Cstep=1）就是结果向量。
- 单序列 decode 受带宽限制，D 倍的冗余计算不影响速度；
- 不需要改 EX；
- 多序列并发和 prefill 时，把这 D 行换成 D 条不同的序列或 D 个 token 即可（§11）。

### 8.3 注意力

KV cache 每层两块，都是行主序：位置 t 对应 `Smax × dim` 中的第 t 行（int8，所有头放在一起）。
- **追加一个 token**：k 和 v 各写一行，每行 dim 字节，是 8 的倍数，一条 ST 即可；
- **按头读取**：rows = pos_pad，row_bytes = hs，pitch = dim。

对每个头 h（pos_pad = ⌈(pos+1)/D⌉·D，由 PARAM 提供），这一段循环体用 LOOP 重复 heads 次，头的偏移用 PARAM 按轮递增：

| 步 | 引擎 | 操作 |
|---|---|---|
| 1 | LD | K_h（pos_pad × hs）按 LINEAR 读入 SPAD，每行 hs/D 个字。**FENCE_BEFORE**：本 token 的 K 刚由 ST 写入 DDR |
| 2 | VE | TRANSPOSE（S = hs/D）→ SPAD_B，得到 Kᵀ 条带 |
| 3 | VE | q_h：amax → s_q；量化成 I8，用 DIV D 复制成 A 条带（K 维 = hs） |
| 4 | EX | scores = Q_rep · Kᵀ：M = D，K = hs，N = pos_pad（repeat = pos_pad/D，bstep = hs） |
| 5 | VE | scores 第 0 行 × s_q（DIV 模式）；仿射 A = s_k/√hs；输出 F32 |
| 6 | VE | softmax，VALID = pos + 1：m = REDUCE MAX；e = EXP(s − m)；sum = REDUCE SUM(e)；r = RECIP(sum)；p = e × r |
| 7 | VE | pq = p × 127 输出 I8，用 DIV D 复制成 A 条带（K 维 = pos_pad） |
| 8 | LD | V_h（pos_pad × hs）按 INTERLEAVE 读入 SPAD_B，布局就是 B 条带 |
| 9 | EX | out_h = P_rep · V_h：M = D，K = pos_pad，N = hs |
| 10 | VE | × s_v/127（仿射），写入注意力输出的第 h 段（fp32） |

位置 pos 之后填充的 K、V 行由 ARM 在分配时清零，而且 p 在这些位置上为 0，所以结果不受影响。

### 8.4 一层的完整序列

| # | 内容 | 主要命令 |
|---|---|---|
| 0 | （第一层之前）x = 嵌入 | LD 嵌入行（地址 = BASE + PARAM[token×dim]）；LDPARAM 读 s_emb[token]；VE CVT × A（动态槽取 PARAM） |
| 1 | xn = RMSNorm(x) | VE：REDUCE SUM(x·x)；RSQRT(ss·(1/dim) + eps)；x × r（DIV）；× g（MOD） |
| 2 | q、k、v = GEMM(xn, Wqkv) | §7.2 |
| 3 | RoPE(q)、RoPE(k) | LD 读 cos/sin 表的第 pos 行（地址 + PARAM[pos×行字节]）；VE：t1 = x·cos，t2 = SWAPNEG(x)·sin，x = t1 + t2 |
| 4 | 追加 KV | VE：k × (1/s_k) 输出 I8；ST 写到 K cache 的第 pos 行（地址 = BASE + 层偏移 + PARAM[pos×dim]）；v 同理 |
| 5 | 注意力 | §8.3 |
| 6 | x += GEMM(att, Wo) | §7.2，外加 VE ADD |
| 7 | xn = RMSNorm(x) | 同第 1 步 |
| 8 | h13 = GEMM(xn, W13) | §7.2 |
| 9 | h = SiLU(h1) · h3 | VE：EXP(−h1)；RECIP(1 + ·)；× h1；× h3 |
| 10 | x += GEMM(h, W2) | §7.2，外加 VE ADD |

第 1–10 步是一段循环体，用 LOOP 重复 layers 次，每层的权重和 KV 地址用 PARAM 按层步长递增。
最后：RMSNorm → 分类层 GEMM（按列分块，LOOP）→ 后处理 → ST 把 logits 写回 DDR → END。

**规模估计**：有 LOOP 之后，一个模型的静态列表约 150–300 条描述符（展开后约 1.5–2k 条）。

### 8.5 PARAM 的分配（手写路径）

| PARAM | 内容 | 谁来算 |
|---|---|---|
| P0 | pos + 1（VALID） | ARM，每个 token |
| P1 | pos_pad（rows、LEN） | ARM |
| P2 | pos_pad / D（repeat、Kt） | ARM |
| P3 | pos × dim（KV 追加的偏移） | ARM |
| P4 | pos × RoPE 行字节数 | ARM |
| P5 | token × dim（嵌入行的偏移） | ARM；IREE 路径用 LDPARAM 在设备上计算 |
| P6、P7 | 循环变量（层偏移、头偏移、分块偏移） | LOOP 按轮递增；进入循环前用 SETREG 复位 |

### 8.6 验证

- **列表生成器 `llm/compile_layer.py`**：输入模型配置与存储分配，输出带 LOOP、PARAM 的静态列表。L6-IREE 会把它换成编译器。
- **小模型联合仿真**：随机权重，dim=32、hidden=64、2 个头、2 层（用来测试 LOOP）、vocab=64、seq=16。
  - 同一张列表在 `tb_system`（RTL）与功能模拟器上执行，logits 逐位一致；
  - 功能模拟器与 `ref_model.py` 的设备数值模式对比，误差在容限内；
  - 覆盖 pos = 0、pos = D−1、pos = D，**只改参数块，列表不变**。
- **板上** `l4_layer_demo.py`：stories15M 的第 0 层，多个 pos，与功能模拟器逐位一致。

---

## 9. L5：ARM 运行时与端到端生成

### 9.1 `llm/runtime.py`：`LlamaDevice`（Python，用于调通和作参考）

- **加载**：把 `*.w8a8` 读进一块 CMA 缓冲。stories15M 约 16 MB，110M 约 115 MB，需要确认 CMA 大小。
  另外分配 KV cache（清零）、RoPE 表、logits、scratch、两个命令环和参数块。
- **列表**：用 `compile_layer.py` 生成一张静态列表，只加载一次。
- **每个 token**：写参数块（§8.5）→ 提交条目 → 等中断 → 读回 logits → 采样（argmax 或 temperature/top-p，用 NumPy）。
- **提示词**：L5 逐个 token 走 decode 路径，只是不采样；真正的 prefill 放在 L6。
- **tokenizer**：`llm/tokenizer.py`，把 llama2.c 的 `tokenizer.bin`（BPE）编码和解码移植过来。

### 9.2 验收

板上 `notebooks/llm/l5_generate.py`：

1. **正确性**：stories15M，给定提示词，贪心生成 128 个 token。每个 token 的 logits 与功能模拟器逐位一致，文本完全相同。
2. **精度**：teacher-forced 条件下，与 fp32 参考的 top-1 一致率达到 L0 定的门槛。
3. **演示**：生成通顺的 TinyStories 文本，打印 tok/s 和每个 token 的周期分解，只作记录。
   估计 50 MHz 下约 25 tok/s（stories15M）。
4. **稳定性**：连续生成 10 段，每段 256 个 token，没有错误和超时，中断次数等于 token 数。

---

## 10. 编译器接口：MLIR / IREE

### 10.1 IREE 的结构与接入点

```
PyTorch ──(iree-turbine / torch-mlir)──► MLIR (linalg / tensor)
  IREE 编译器：Flow（算子融合，划分 dispatch）→ Stream（异步调度，分配缓冲）
             → HAL（每个 dispatch 编译成目标可执行体；host 端生成 VM 字节码）
IREE 运行时（ARM 上的 C 代码）：VM → HAL 驱动 → 命令缓冲 → 队列提交 → 信号量
```

接入一个新加速器要做两部分：
- **编译器端**：一个**目标后端**，把 dispatch 编译成我们的可执行体，也就是位置无关的描述符列表模板，附带元数据：
  binding 数、push constant 数、入口。
- **运行时端**：一个 **HAL 驱动**（C 代码），实现设备、内存分配、可执行体缓存、命令缓冲、队列和信号量。

### 10.2 概念对照与所需硬件

| IREE 概念 | 硬件与固件的对应 | 位置 |
|---|---|---|
| dispatch 的 binding | BASE0–15 + 重定位 | §5.3 #1 |
| push constants、动态维度 | PARAM0–7 + 动态字段 | §5.3 #2、#3 |
| workgroup 数量（可以是动态的） | LOOP_END，次数可以取自 PARAM | §5.3 #5 |
| 命令缓冲：记录多个“绑定 + dispatch” | SETREG + CALL 某个可执行体，组成一张列表 | §5.3 #4、#6 |
| 可执行体（只加载一次） | 位置无关的描述符模板（相对 JUMP/LOOP，CALL/RET） | §5.3 #6 |
| execution barrier | FENCE 描述符。IREE 要求在屏障处显式同步，与“DDR 依赖不由硬件跟踪”的规则一致 | 已有 |
| 队列提交与 timeline 信号量 | 命令环条目 + 完成记录的 seq + 中断 | §5.1 |
| 由数据决定的地址（gather、嵌入、按位置写） | LDPARAM | §5.3 #7 |
| 缓冲的 flush / invalidate | CMA 缓冲 + 缓存维护 | 软件 |
| fill、copy、update 缓冲命令 | 可执行体模板：DDR→SPAD→DDR 拷贝；VE 写立即数后 ST | 软件 |
| 数据分块（`tensor.pack`、mmt4d） | A、B 条带就是 [D, D] 内块的 pack 布局；LD INTERLEAVE 在搬运时完成 pack；权重在编译时预先 pack | 已有 |
| 多设备（部分 dispatch 在 CPU 上） | ARM 上的 `llvm-cpu` 目标；CMA 缓冲两边都能访问 | 软件 |

### 10.3 由 IREE 带来的约定（不改硬件）

1. **dispatch 之间的数据经过 DDR**。片上 SPAD/ACC 只相当于 GPU 的 shared memory，不跨 dispatch 保留。
   decode 时激活只有几 KB，影响很小；dispatch 划得多大由 IREE 的融合决定。
2. **数值约定**：§3.2 以 fast-math 的形式声明，与 GPU 的做法一致：
   - FTZ；
   - EXP/RECIP/RSQRT 是近似计算；
   - `arith.divf` 降级为 RECIP 加乘法；
   - `math.exp`、`math.rsqrt` 直接对应 FUNC。
3. **逐元素运算体**：一个 `linalg.generic` 如果超出 VE 的固定运算链，就拆成几条 VE 命令，中间值放在片上。
4. **布局类型**：编译器内部定义以下几种布局，并做布局传播：
   - 紧凑向量；
   - A 条带、B 条带；
   - 广播字（每行一个标量）；
   - Kᵀ 条带。
   转换算子有三种：TRANSPOSE、DIV-D 复制行、经 DDR 的 LD INTERLEAVE。
5. **代价模型**：功能模拟器的周期估计（§4.3），用计数器校准。

### 10.4 两种接入方式

- **A. 模板库**（先做，L6-IREE）：把 linalg 的具名算子匹配到预先写好的可执行体模板，
  算子包括 matmul 及其量化形式、softmax、RMSNorm 形式的归约、逐元素运算、transpose、gather。
  这些模板就是 `compile_layer.py` 中各段的通用化；不支持的 dispatch 留在 `llvm-cpu` 上运行。
- **B. 完整代码生成**（之后做）：定义一个 `sa` 方言。linalg 经过分块、按 SPAD 做打包和内存规划后，
  降级成 `sa.ld`、`sa.ex`、`sa.ve`、`sa.loop` 等操作，再序列化成描述符。

两种方式对硬件的要求**相同**，都由 §5.3 和 §6 满足。

### 10.5 L6-IREE 的交付与验收

- **`iree-sa/compiler`**：HAL 目标后端插件，使用模板库，可执行体格式为 `sa-desc-v1`。
- **`iree-sa/runtime`**：C 写的 HAL 驱动：
  - 访问 mailbox、命令环和参数块（通过 mmap）；
  - UIO 中断；
  - 用 CMA 分配缓冲；
  - 信号量对应完成记录的 seq。
- **验收**：
  1. IREE 编译的 MLP、softmax、单个 decoder 层，在“CPU + 加速器”混合执行下，结果与功能模拟器逐位一致；
  2. 用 IREE 编译 stories15M，在板上生成文本，与 L5 手写路径的文本一致（数值约定相同时应逐位一致）。

---

## 11. L5b、L5.5、L6：多序列并发、75 MHz 与后续扩展

### 11.1 L5b：多序列并发 decode（不改硬件）

复制行布局中的 D 行换成 D 条序列：
- 每条序列有自己的 KV cache 与 pos；
- A 条带由 D 条序列的激活组成：先经 DDR 用 LD INTERLEAVE 转成条带，或者 VE 按行写入；
- 注意力部分每条序列单独做，因为各自的 pos 不同；
- 同一份权重服务 D 行，吞吐接近单序列的 D 倍。

验收：8 条序列同时生成，每条的文本与单独生成时一致；记录总 tok/s。

### 11.2 L5.5：75 MHz（附加目标）

- `pico_bit.tcl` 中的 `subprocessorClk` 改为 75 MHz。PicoRV32、加速器和 HP 口都用这个时钟，HP 口本身支持到 150 MHz。
  驱动和运行时里的频率常数同步修改。
- 以 L0 的时序摸底结果为准，只修必要的路径。已知一条是 `ex/drain_active` 的扇出：
  复制寄存器，或者给使能加一级流水，按行分组驱动，数据同步延迟一拍。
- 新模块已经按 13.3 ns 设计，理论上不需要改。
- **验收**：
  - WNS ≥ 0；
  - L1–L5 的全部板上脚本在 75 MHz 下通过；
  - 性能按周期计算不变，按时间计算快 1.5 倍（估计 stories15M 约 37 tok/s）。
- **失败不影响前面的成果**：继续在 50 MHz 下使用。

### 11.3 L6：后续扩展（按需选做）

| 项 | 内容 |
|---|---|
| 真正的 prefill | D 个 token 一批做 GEMM：先 ST 到 DDR，再用 LD INTERLEAVE 读回，得到 A 条带；块内的因果掩码用按行变化的 VALID |
| 细粒度记分板 | 每块存储器从 2 个 bank 分成 8 个区跟踪（掩码从 6 位加宽到 24 位），减少虚假依赖。条件是 L4/L5 的计数器显示 `HAZ_*` 占比明显；约 400–600 LUT |
| bf16 | VE 的 bf16 存储格式与转换（如果 §6 没有做），用于 bf16 导出的 torch 模型 |
| int4 权重 | 在 LD 路径上把 int4 解包成 int8。**只有 EX 每周期能消耗更多权重时才有收益**，D=8 时没有收益 |
| ARM 直接提交 | 实现 END 的 IRQ 位，ARM 把列表直接交给取指单元 |
| 分页 KV cache | K、V 按块存放，ARM 维护块表，块地址通过 LDPARAM 从块表中读取 |
| 设备端中断门铃 | 用 MMIO 门铃寄存器触发 PicoRV32 中断，取代轮询 |
| 完整代码生成 | §10.4 的方式 B |

---

## 12. 资源与时序预算

xc7z020：53,200 LUT、106,400 FF、220 DSP、140 BRAM36。以下都是**估计值**，以综合结果为准：

| 部分 | LUT | DSP | BRAM36 | 依据 |
|---|---|---|---|---|
| D=8 基线（M3 实测 21.7k，加 PERF 和 DESC） | ~24k | 96 | 130 | M3 报告；D=16 时计数器与取指合计约 +1.8k |
| L1 命令环、中断、`mat_notify` | < 0.3k | 0 | 0 | 几个寄存器 |
| L1 命令处理扩展（16 个 BASE、PARAM、动态字段、SETREG、LOOP、CALL/RET、LDPARAM） | 1.5–2.5k（**实测约 3.6k**） | 1（实测 2） | 0 | **OOC 实测**（D=8，50 MHz）：取指单元 4,408 LUT、2,106 FF、2 DSP，整个 `sa_unit` 21,787 LUT，WNS +2.37 ns。多出的主要是寄存器组的多个读写口和字段改写逻辑 |
| L2 fp32 lane × 8（2 个乘法器、2 个加法器、转换、比较、表格） | 11–14k（**实测：8 lane 约 18.6k，放不下；折叠成 4 lane 后约 9.7k**） | 32（实测 16） | 0 | 每个 lane 实测约 2.3k LUT、2 DSP（i2f ×2、f2i 占了一半）；表格是一张 256 项的 ROM。折叠见 §6.9 |
| L2 微程序控制、转置缓冲、归约、下标生成、加宽的命令包、精确访问范围 | 2–3k（**实测约 7.5k + 调度器 / 取指 +1.2k**） | 0（实测 2） | 0 | **OOC 实测**（D=8，FL=4，50 MHz）：`sa_vefp` 17,197 LUT、11,696 FF、18 DSP；整个 `sa_unit` 39,062 LUT（73.4%）、28,232 FF、116 DSP、128 BRAM36，WNS +1.62 ns。整体估计约 44.4k（83%） |
| **合计** | **~39–44k（73–83%）** | **~129** | **130** | D=16 在 92% 时仍能满足 50 MHz |

- **BRAM** 是最紧张的资源，只剩 10 个。新增存储全部用 LUTRAM 或 FF。
- **FF** 预计约 50–55k（约 50%），流水寄存器的余量充足。
- **时序**：50 MHz 下，现有模块已有余量，新模块按 13.3 ns 设计。LUT 超过 80% 时布线拥挤会加重，但 50 MHz 仍有余量。
- **LUT 超出预算时，按以下顺序削减**：
  1. 去掉整数 VE 的 int16 路径；
  2. 去掉 CALL/RET，可执行体改由 ARM 在加载时修改绝对地址；
  3. fp32 lane 的 M1 与 M2 合并（MUL 与仿射乘法不能同时进行，改为两拍）；
  4. SFU 表格改成 32 段。

---

## 13. 总体计划与验收

| 级 | 内容 | 要改的地方 | 需要构建 bitstream | 验收 |
|---|---|---|---|---|
| **L0** | D=8 基线与 75 MHz 时序摸底；参考模型与量化精度；功能模拟器；IREE 在 armv7 上的验证 | Python；构建一次 D=8 | 是（不改 RTL） | D=8 回归全部通过；fp32 参考与 `run.c` 文本一致；top-1 一致率 ≥ 门槛；功能模拟器与联合仿真 case 逐位一致；IREE 的 `llvm-cpu` 小模型在板上运行正确，或确定退路 |
| **L1** | 命令环、`rt_fw`、中断；命令处理扩展 | sa_cmdfetch、sa_pcpi、sa_legacy、sa_unit、sa_defs、BD、固件、驱动、功能模拟器 | 是 | §5.5：定向测试与随机测试逐位一致；`RING_TEST`；板上 200 条异步提交全部正确；静态列表只改参数块就能正确运行 |
| **L2** | fp32 VPU（混合结构）、SFU、归约、转置、掩码 | sa_ve 与新增 fp32 模块、sa_sched、sa_defs、sa_pcpi、sa_cmdfetch、驱动、功能模拟器 | 是 | §6.8：逐位一致；SFU 精度达标；板上算子与功能模拟器逐位一致；不回归 |
| **L3** | W8A8 导出；GEMM 与后处理（B 双缓冲） | Python | 否 | 板上各线性层与功能模拟器逐位一致；分类层 GEMV 的 LD 忙碌 ≥ 90% |
| **L4** | 一个 decoder 层（含注意力）在设备上执行，静态列表加 PARAM | Python | 否 | 小模型 RTL、功能模拟器和参考模型三方比对；板上第 0 层逐位一致 |
| **L5** | Python 运行时；端到端生成文本 | Python | 否 | §9.2 的 4 项 |
| L5b | 多序列并发 decode | Python | 否 | §11.1 |
| L5.5 | 75 MHz | 时钟与少量时序修复 | 是 | §11.2 |
| **L6-IREE** | 目标后端（模板库）+ C 写的 HAL 驱动 | IREE 插件、C 运行时 | 否 | §10.5 |
| L6 | 其他扩展 | — | 视情况 | 每项单独定 |

- **建议顺序**：L0 → L1 → L2 → L3 → L4 → L5 → L6-IREE，L5b 和 L5.5 可以穿插进行。
  L1 与 L2 的 RTL 可以合并成一次 bitstream 构建，但分开验证。
- **构建约定**：Vivado 限 4 个 job，长时间的构建由用户在自己的终端里运行。
- **提交约定**：每一级上板通过后提交，bitstream 归档到 `bitstreams/l<n>/`。

---

## 14. 风险与对策

| 风险 | 表现 | 对策 |
|---|---|---|
| W8A8 按通道量化在 15M 这样的小模型上精度不够 | L0 的 top-1 一致率不达标 | 依次尝试：① 改用 42M 或 110M；② KV 改为动态 scale（每个 token 的 scale 存成广播字，读回时用 TRANSPOSE 整理）；③ 对异常值较多的层走 fp32 旁路（在 VE 上用逐元素乘加按行归约做 GEMV）；④ 按组量化：K 按组拆成多条 EX，结果放在不同的 ACC 区，在 VE 中乘各组的 scale 后再累加 |
| IREE 运行时不能在 armv7 上运行 | L0 §4.4 不通过 | 只用 IREE 编译器，运行时自己用 C 写一个执行器；硬件接口不受影响 |
| IREE 的插件接口随版本变化 | 升级后编译不过 | 固定一个 IREE 版本；插件代码与 IREE 本身隔离 |
| LUT 超出预算 | 超过 83%，或布线失败 | 按 §12 的顺序削减 |
| 动态字段与控制流描述符的逻辑出错 | 列表执行结果错误 | 每一类都有定向测试，外加与功能模拟器对照的随机列表测试；错误都是粘滞的，并记录出错描述符的序号 |
| PicoRV32 与 ARM 看到的 DDR 不一致 | 固件读到旧的条目或参数块，或 ARM 读到旧的完成记录 | 命令环、参数块和列表都用不开 cache 的缓冲，或者严格执行 flush / invalidate；完成记录先写数据，最后再更新 `RING_HEAD` |
| PYNQ 或 UIO 找不到中断 | 查不到 `matmul_0/irq` | 先离线用 pynqmetadata 检查 hwh；轮询 `RING_HEAD` 作为备用路径 |
| 75 MHz 无法满足时序 | L5.5 的 WNS < 0 | 继续使用 50 MHz，前面的成果都不受影响 |
| 功能模拟器与 RTL 不一致 | 逐位比对失败 | 这正是要发现的问题。以本文档的规范为准，两边都按规范修，并补上测试向量 |
| 程序 BRAM 不够 | `rt_fw` 超过 0x1E00 | 这个循环很小，估计在 1 KB 以内 |

---

## 附录 A：编码变化汇总

| 项 | 编码 | 级 |
|---|---|---|
| `mat_notify` | funct7=1，funct3=7：`notify_irq` 一个上升沿脉冲，`NOTIFY_COUNT` 加 1 | L1 |
| CSR | 0xD4 `NOTIFY_COUNT`（只读） | L1 |
| `sa_unit` 端口 | `notify_irq`（EDGE_RISING），BD 中接 `irqConcat/In1` | L1 |
| CAPS | bit 22 NOTIFY/IRQ，bit 23 FP32 VE，bit 24 CMDX（命令处理扩展） | L1 / L2 |
| mailbox | 0xB0–0xCC：命令环与固件状态（§5.1） | L1 |
| 命令环条目 | w7 = 参数块地址 | L1 |
| 描述符 w0 | BASESEL = `{w0[14:13], w0[10:9]}`；w0[31:16] 是两个动态槽 | L1 |
| 新 opcode | 0x13 LOOP_END，0x14 SETREG，0x15 CALL，0x16 RET，0x17 LDPARAM；JUMP 的 w2[0] = REL | L1 |
| `mat_cfg` key | 15–26 BASE4–15，27–34 PARAM0–7 | L1 |
| VE types | VT_F32 = 3（只能放在 ACC）；可选 VT_BF16 = 4（SPAD） | L2 |
| VE op | 6 = TRANSPOSE（7 保留） | L2 |
| VE 描述符 | w3[63:53]、w5[63:32]、w6、w7（§6.6） | L2 |
| `vec_cfg` key | 10 FLAGS，11 IMM，12 A，13 B，14 ROWLEN/VALID，15 P1/S | L2 |

## 附录 B：新增与修改文件

| 路径 | 级 | 内容 |
|---|---|---|
| `llm/ref_model.py`、`llm/eval_quant.py` | L0 | 参考模型、量化精度评估与 KV scale 标定 |
| `llm/sa_funcsim.py` | L0–L2 | 功能模拟器（逐位黄金参考）与周期估计 |
| `llm/sfu_tables.py`、`llm/gen_ve_fp_vectors.py` | L2 | SFU 表格，fp32 VE 测试向量 |
| `llm/export_w8a8.py` | L3 | 模型导出 |
| `llm/compile_layer.py`、`llm/test_l3.py` | L3、L4 | 手写列表生成器（LOOP、PARAM）及其主机测试 |
| `llm/runtime.py`、`llm/tokenizer.py` | L5 | Python 运行时 |
| `firmware/rt/` | L1 | 常驻固件 `rt_fw` |
| `firmware/include/mailbox.h`、`sysarray_intrinsics.h` | L1、L2 | 命令环字段、`mat_notify`、新的 `mat_cfg` / `vec_cfg` key |
| `rtl/sysarray/sa_cmdfetch.v` | L1 | 命令处理扩展 |
| `rtl/sysarray/sa_pcpi.v`、`sa_legacy.v`、`sa_unit.v`、`sa_defs.vh` | L1、L2 | `mat_notify`、中断、新 CSR 和 CAPS、新 key、加宽的命令包 |
| `rtl/sysarray/sa_ve.v`，新增 `sa_ve_fp.v`、`sa_fp32_add.v`、`sa_fp32_mul.v`、`sa_fp32_cvt.v`、`sa_sfu_seq.v` | L2 | fp32 VPU 与 SFU |
| `rtl/sysarray/sa_sched.v` | L2 | 新模式下的精确访问范围 |
| `rtl/sysarray/sim/tb_sa_ve_fp.v`，`tb_sa_unit.v`、`firmware/sim/tb_system.v` | L1、L2、L4 | 新测试 |
| `RISCV-on-PYNQ-Z1/scripts/pico_bit.tcl` | L1、L5.5 | `irqConcat` 改为 2 个端口；时钟改为 75 MHz |
| `driver/pynq_matmul.py` | L1、L2 | `Device`（命令环、参数块），`DescList` 的新描述符与新字段 |
| `iree-sa/compiler`、`iree-sa/runtime` | L6-IREE | IREE 目标后端插件与 HAL 驱动 |
| `notebooks/llm/l1_ring_demo.py` … `l5_generate.py` | L1–L5 | 板上脚本 |

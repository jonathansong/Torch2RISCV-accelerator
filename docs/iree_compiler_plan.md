# LLM 编译器：基于 MLIR / IREE 的端到端方案（PyTorch → 描述符列表）

状态：**C0–C3 完成**（§3.5、§5.7、§6.7、§6.9）；**C4 部分完成、暂缓**（§7.1：板上 2.591M 周期 / token，17.7 tok/s，与手写路径差 13%，目标 ≤ 10%；剩下的融合留到 C5 之后）；**C5 完成**（§8.7：两层方言 sahl / sahw 的完整代码生成，模板成为微内核；stories15M 板上 2.589M 周期 / token；SmolLM2-135M 板上 2.23 tok/s，Qwen3 结构 sim；两者都可在板上交互式生成）；**C6 进行中**（§8.12：目标配置交叉验证、K 分块、C6.0 嵌入只存一份、C6.1 Qwen3-0.6B 全 28 层在 sim 上逐位一致、C6.P prefill + decode 板上逐位一致已完成，下一步 C6.2 量化质量或 prefill 的优化）。这是 [`llm_inference_plan.md`](llm_inference_plan.md) 的 L6-IREE 一级的详细设计，
取代那里 §10.4、§10.5 的概要。

**目标**：从一个 PyTorch 写的 llama 类模型出发，用 MLIR / IREE 自动编译，得到在 PYNQ-Z1 上运行的完整程序：
- ARM 上跑 IREE 运行时；
- 计算由加速器执行，形式是描述符列表，经过命令环、`rt_fw`、取指单元进入各引擎；
- 最终用 IREE 编译的 stories15M 在板上生成文本，与 L5 手写路径的文本完全相同。

**不改硬件**：硬件接口在 L1、L2 已经按 IREE 的概念设计好了（§1.3），编译器与运行时全部是软件。

相关文档：
- [`llm_inference_plan.md`](llm_inference_plan.md)：§3 数值约定，§5 命令环与命令处理扩展，§6 fp32 VE，§7–§9 手写路径，§8.8 调度优化，§10 IREE 概要；
- [`double_buffer_design.md`](double_buffer_design.md)：加速器、记分板、指令与描述符格式；
- [`memory_model.md`](memory_model.md)：地址映射、缓存一致性、命令环的缓冲；
- [`../iree-sa/l0/README.md`](../iree-sa/l0/README.md)：L0 已验证 IREE 在板上 ARM（armv7）上可用。

---

## 目录

0. [原则](#0-原则)
1. [总体结构](#1-总体结构)
2. [继承已有的成果](#2-继承已有的成果)
3. [前端：量化模型与导出（C0）](#3-前端量化模型与导出c0)
4. [可执行体格式 sa-desc-v1](#4-可执行体格式-sa-desc-v1)
5. [运行时：sa HAL 驱动（C1）](#5-运行时sa-hal-驱动c1)
6. [编译器：sa 目标后端插件（C2、C3）](#6-编译器sa-目标后端插件c2c3)
7. [性能：跨 dispatch 的调度（C4）](#7-性能跨-dispatch-的调度c4)
8. [完整代码生成：sa 方言（C5）](#8-完整代码生成sa-方言c5)
9. [验证体系](#9-验证体系)
10. [分阶段计划与验收](#10-分阶段计划与验收)
11. [目录结构](#11-目录结构)
12. [风险与对策](#12-风险与对策)
13. [待定的决策](#13-待定的决策)

---

## 0. 原则

- **先打通一条最窄的纵向切片，再加宽**：第一个里程碑只支持一个量化 matmul，但从 torch 到板上结果全程打通。
  之后每次增加一类 dispatch，每一步都能验证。
- **数值由规格决定，不由实现决定**：`DeviceModel(sfu=SfuExact)` 就是数值规格（§3.2）。
  编译器的任何变换都不能改变运算顺序和舍入次数，否则“与手写路径逐位一致”这个最强的验收手段就没了。
- **沿用本项目的验证阶梯**：功能模拟器 → RTL 联合仿真 → 板上。编译器的开发大部分在主机上完成（§5.5 的模拟器传输层）。
- **先测量再优化**：先做对（C2、C3），再用计数器和 §8.8 那样的分段计时找瓶颈（C4）。
- **手写路径是参照物**：L3–L5 的 `compile_layer.py`、`compile_model.py` 已经把整个 decoder 编成了描述符列表，
  它们既是编译器模板的参考实现，也是性能的对照。

---

## 1. 总体结构

### 1.1 数据流

```
PyTorch 模型（显式量化的 llama，§3.1）
   │  iree-turbine：aot.export → torch dialect（权重外置为参数文件 .irpa）
   ▼
torch-mlir → linalg / tensor / arith / math
   │  预处理（§6.2）：保持 softmax 等高层算子，禁止拆分归约，KV cache 原地更新
   │  IREE DispatchCreation / Flow：融合，划分 dispatch
   │  数据分块（§6.3）：matmul 操作数带 encoding，解析成 [D, D] 内块的 pack
   │  IREE Stream：异步调度，分配缓冲（瞬时缓冲、常量）
   │  IREE HAL：每个 dispatch 交给目标后端；host 端的调度变成 VM 字节码
   ├──► sa 目标后端插件（C++，§6）：dispatch → sa-desc-v1 可执行体（位置无关的描述符模板 + 元数据）
   └──► VM 字节码 .vmfb（调用顺序、绑定、push constant、屏障、信号量）
ARM：IREE 运行时（C）
   └──► sa HAL 驱动（C，§5）
          命令缓冲 → 一张描述符列表（SETREG 绑定 + CALL 模板 + FENCE）
          → 命令环条目 → 门铃 → 等 notify 中断 → 完成记录 → 信号量
PicoRV32 rt_fw、sa_cmdfetch、sa_sched、各引擎：不变
```

### 1.2 两个要写的部分

| 部分 | 语言 | 运行在哪里 | 作用 |
|---|---|---|---|
| **sa 目标后端插件** | C++，MLIR | 主机上的 iree-compile | 把每个 dispatch 编成 sa-desc-v1 可执行体 |
| **sa HAL 驱动** | C | 板上 ARM 的 IREE 运行时 | 实现 IREE 的设备抽象：内存、可执行体、命令缓冲、队列、信号量 |

另外还有两个辅助部分：
- 前端的量化模型（Python，§3）；
- 模拟器服务（Python，复用 `llm/sa_funcsim.py`，§5.5），让驱动能在主机上对着功能模拟器运行。

### 1.3 IREE 概念与硬件的对应（沿用 `llm_inference_plan.md` §10.2）

| IREE 概念 | 硬件与固件 | 本方案中的用法 |
|---|---|---|
| dispatch 的 binding | BASE0–15 + 重定位 | binding i → BASE i（§4.3） |
| push constant、动态维度 | PARAM0–7 + 动态字段 | PARAM0–5 直接传，多出来的经 LDPARAM（§4.3） |
| workgroup 数量 | LOOP_END（次数可取自 PARAM） | 模板内部的循环 |
| 可执行体 | 位置无关的描述符模板 | 只用相对 JUMP / CALL / LOOP，加载时整体拷进 CMA |
| 命令缓冲 | SETREG + CALL，组成一张列表 | §5.2 |
| execution barrier | FENCE 描述符 | 屏障 → FENCE（DDR 依赖不由硬件跟踪） |
| 队列提交、timeline 信号量 | 命令环条目、完成记录的 seq、中断 | §5.4 |
| 由数据决定的地址 | LDPARAM | gather（嵌入表）、按位置写 KV |
| 数据分块（pack、mmt4d） | A、B 条带就是 [D, D] 内块 | 权重在编译期打包（§6.3） |

---

## 2. 继承已有的成果

| 已有成果 | 在编译器中的角色 |
|---|---|
| `llm/ref_model.py` 的 `DeviceModel(sfu=SfuExact)` | **数值规格**。前端模型、编译器变换、设备结果都以它为准（§3.2） |
| `llm/compile_layer.py`、`compile_model.py`、`compile_batch.py` | **模板的参考实现**：量化、分块双缓冲的线性层、RMSNorm、RoPE、注意力、SiLU、残差、嵌入 gather；§8.8 的预取、SiLU 融合就是 C4 要自动化的调度 |
| `llm/export_w8a8.py` | **权重打包规则**：B 条带布局、每通道 scale。IREE 的数据分块必须得到同样的布局 |
| `llm/sa_funcsim.py` | **金标准和代价模型**：逐位执行描述符列表；模拟器服务（§5.5）的核心 |
| `llm/runtime.py`（`LlamaDevice`） | **HAL 驱动的行为模型**：CMA 布局、参数块、命令环、中断、缓存维护 |
| `driver/pynq_matmul.py` 的 `DescList` | 描述符编码的参考；插件中的序列化器与它逐字节一致 |
| `firmware/rt`、命令环、取指单元、调度器 | 不变 |
| `iree-sa/l0` | armv7 交叉编译环境、libm shim、嵌入式 ELF 检查；IREE 3.11.0（e4a3b0405d） |
| 板上脚本、计数器、`l4_phase_cost.py` | C4 的测量工具 |

---

## 3. 前端：量化模型与导出（C0）

### 3.1 模型定义（`iree-sa/frontend/qllama.py`）

用 PyTorch 写一个**显式量化**的 llama，逐步对应 `DeviceModel.forward`。
不依赖编译器“自动量化”，因为那样前端和设备的数值对不上。

- **权重**：int8 张量加每通道 fp32 scale（`export_w8a8` 的量化结果），作为模块参数。
- **激活量化**：显式写成 `amax = max(|x|)`、`s = amax * (1/127)`、`inv = 1 / s`、
  `q = clamp(round_half_even(x * inv), -127, 127).to(int8)`。
- **int8 matmul**：`int32 = x_q @ W_qᵀ`（例如 `torch._int_mm`，或 int32 的 matmul），然后 `y = float(acc) * s_w * s_x`，两次舍入，顺序不变。
- **RMSNorm、RoPE、softmax、SiLU、残差**：与 `DeviceModel` 同样的算式与运算顺序：
  - RMSNorm 用 `ss * (1/dim) + 1e-5` 再 rsqrt；
  - RoPE 用 SWAPNEG 形式：`x*cos + swapneg(x)*sin`；
  - softmax 的 max、exp、sum、recip、乘法各一步；
  - SiLU 用 `h1 * recip(1 + exp(-h1))`。
- **KV cache**：int8，每层 K、V 各一个静态 scale；在导出的模块里是**可变的全局变量**（`util.global`），按位置写入一行。
- **decode 函数的签名**：`logits = decode(token: i32, pos: i32)`。token、pos 是标量输入，序列长度相关的量由 pos 算出。
  - 注意力只看前 pos+1 个位置：用 `pos_pad = ceil((pos+1)/D)·D` 的动态长度，加上掩码；
  - pos_pad 是动态维度，编译后成为 push constant。
- **嵌入**：`emb_q[token] * emb_s[token]`，这是一次 gather。

### 3.2 数值契约

编译器与设备都必须保持下列约定，这样 IREE 编译的结果才能与 `DeviceModel(sfu=SfuExact)` 逐位一致：

| 项 | 约定 |
|---|---|
| fp32 运算 | RNE、FTZ、canonical NaN（§3.2 of the LLM plan）；每个算子一次舍入 |
| 不允许的变换 | 重结合（reassociation）、把乘加收缩成 FMA、拆分归约（split-k / split reduction）、改变 max/min 对 NaN 和 ±0 的语义 |
| 归约顺序 | 先每个 lane 按组号依次累加，再对 D 个 lane 两两配对做树形合并（LLM plan §6.4）。编译器只能生成这种顺序，前端的参考（`DeviceModel`）也按这个顺序算 |
| 特殊函数 | `math.exp` → SFU EXP，`1/x` 与 `arith.divf` → RECIP（再乘），`math.rsqrt` → RSQRT；都是规定的近似算法（`llm/sfu.py`） |
| int8 量化 | RNE，饱和到 ±127，NaN → 0 |

- torch 本身不是逐位参考（它的 exp、归约顺序不同），torch 的结果只用来检查精度（误差范围与 L0 的 top-1 门槛）。
- 这些约定在编译器里以 fast-math 属性声明，插件在匹配时检查。不满足的 dispatch 报错，不做静默近似。
- **分块归约**（大模型需要，§8.9）：整数累加（EX 跨 K 块累加 int32）结果与顺序无关，可以随意分块。fp32 归约（softmax 的 max / sum、长向量的求和）如果要分块，必须由预处理在 IR 里**显式**改写成分块形式（例如在线 softmax：块内按上面的硬件顺序，块间按块号依次合并，缩放方式固定），oracle 按 IR 执行，所以仍然逐位一致。代码生成不得私自改变 fp32 归约的顺序；max / min 除外（与顺序无关，如 abs-max 折半树）。

### 3.3 导出

- `iree.turbine.aot.export` 导出 `decode`；权重用 `externalize_module_parameters` 外置成参数文件（.irpa），
  运行时加载进 CMA，不嵌在 .vmfb 里。
- KV cache 作为可变全局变量导出，确认生成的程序是**原地更新**，而不是每个 token 拷贝整块 cache（§12）。
- 先用已有的 `llvm-cpu`（armv7）后端编译，在主机和板上运行。

### 3.4 C0 的产出与验收

- `qllama.py` 在主机上与 `DeviceModel` 对比：logits 误差在 fp32 舍入范围内，argmax 一致率 ≥ 99%。
- `llvm-cpu` 版本在板上跑通，与主机结果一致（在 armv7 的数学库精度内）。
- **dispatch 清单**：用 `--iree-hal-dump-executable-sources-to` 导出每个 dispatch 的 linalg IR，列出：
  - 每类 dispatch 的形状、融合方式、数量；
  - 它们对应哪个手写片段（`compile_layer` / `compile_model` 的哪一段）。

  这份清单决定 C3 要写哪些模板，是整个计划的**第一个决策点**（§13）。

### 3.5 C0 结果（2026-09-27）

**实现**：`compiler/frontend/qllama.py`（量化模型）、`compiler/frontend/export.py`（导出、编译、对比、清单、板上包）。

**数值**
- **eager QLlama**：
  - 与 `DeviceModel(sfu=Sfu64)` 比，每个 token 的 logits 相对差约 1e-7，KV cache 完全一致，说明运算步骤一一对应；
  - 与 `SfuExact` 版比，EXP 的近似（误差约 1e-5）让 int8 量化偶尔翻转 ±1，差异约 2%，argmax 仍然一致。
    设备最终要逐位对上的是 `SfuExact` 版。
- **IREE 编译（llvm-cpu，主机）**：stories15M 连续 12 步，与 eager 版相差 ≤ 4e-7，argmax 12/12 与 `DeviceModel` 一致。
  KV cache 作为可变全局变量，在调用之间保留。
- **板上（armv7 Cortex-A9，L0 的 `iree-run-module`）**：pos 0、pos 5（T = 8）两个单步用例，与主机输出一致（阈值 1e-3），**C0 board PASS**。
  多步 decode 需要在一个进程里连续调用，等 C1 的运行时再测。

**导出与编译中确定的做法**（C2、C3 沿用）

| 问题 | 做法 |
|---|---|
| `emb_q[tok]` 这种用 0 维张量做下标的写法，被当成数据相关的标量，导出失败 | gather 一律写成 `index_select` |
| 按层写 KV cache（`kc[l].index_copy_`）导出成 index_put 加切片写回，IREE 的 CPU 后端拒绝（“write affecting operations on global resources are restricted to workgroup distributed contexts”） | KV cache 摊平成 (layers·seq_len, kv_dim)，每层一次 `index_copy_` 写第 l·S + pos 行 |
| 注意力的动态长度 | T 由输入 `valid[T]` 的维度带出，是有输入支撑的动态形状，导出稳定；编译后成为 push constant |
| 常量表达式提升把 int8 权重提前转置并扩展成 int32（占 4 倍内存，matmul 看到的是 i32×i32） | 编译加 `--iree-opt-const-expr-hoisting=false`，线性层 dispatch 直接读 int8 权重 |
| armv7 链接失败：`__aeabi_idiv` 未定义（Cortex-A9 没有硬件整数除法，动态形状的下标计算要用） | 在 `iree-sa/l0/armv7_libm_shim.c` 中补 `__aeabi_{u,}idiv{,mod}`，与 C 的 `/`、`%` 在 400 万组输入上一致 |
| 权重 | 外置到参数文件（.irpa，16.9 MB，嵌入表与分类层共用一份） |

**dispatch 清单**（`compiler/frontend/inventory/stories15M_dispatches.md`）：每个 token 调用 **241 次** dispatch，去重后 65 个可执行体。

| 类别 | 每个 token 的调用 | 对应的手写片段 | 模板 |
|---|---|---|---|
| 逐元素运算链 | 122 | RoPE、SiLU、残差、量化的缩放与取整、softmax 的 exp 等 | VE 表达式编译器 |
| 归约 | 56 | RMSNorm 的平方和、量化的 amax、softmax 的 max 与 sum | `reduce` |
| int8 线性层（GEMV + 反量化） | 24 + 分类层 1 | `compile_layer.linear` | `qlinear` |
| gather | 14 | 嵌入行、RoPE 行 | `gather` |
| KV cache 行写入（scatter） | 12 | `compile_model.kv_append` | `kv_write` |
| 注意力 Q·Kᵀ、P·V（按头的 batch_matmul，长度 T 动态） | 6 + 6 | `compile_model.attention` | `attn_scores`、`attn_pv` |

**对 C2、C3 的影响**
1. **线性层的形式正是模板要的**：dispatch 读取行主序的 int8 W (out × in)，在内部做符号扩展和转置，
   vecmat 用 i64 累加再截断成 i32（int8 × int8 在 k ≤ 768 时不会溢出，可以当 int32 累加处理），
   反量化尾部（`sitofp`、× s_w、× s_x）已经融在同一个 dispatch 里。
2. **权重布局**：参数文件里是行主序的 W，而设备要 B 条带的打包布局。两种办法：
   - 编译器端：数据分块，在编译期打包；
   - 驱动端：加载参数文件时重排一次。

   C2 先用驱动端的办法（简单，且与 `export_w8a8.pack_b` 相同），数据分块留到 C4。
3. **逐元素和归约占 178/241**，而且被切得很细（一次量化 = 一个归约 dispatch 加一个逐元素 dispatch）。
   VE 表达式编译器是 C3 的重点；C4 的融合与跨 dispatch 调度对速度影响最大。
4. **每层的常量**（a_k、a_v、1/s_k……）被写进了 dispatch 体，形状相同的层也不能共用可执行体（注意力每层一个）。
   改进办法：前端把它们作为参数张量传入，让各层的 dispatch 去重。
5. **dispatch 之间的数据经过 DDR**：241 次 dispatch，每次之间可能有 FENCE。
   C3 完成后，先用分段计时量化这部分开销，再决定 C4 的优先级。

---

## 4. 可执行体格式 sa-desc-v1

### 4.1 设计要点

- **一个可执行体 = 若干入口点**，每个入口点对应一个 dispatch 函数，内容是一段**位置无关**的描述符模板：
  - 只用相对 JUMP / CALL / LOOP；
  - 以 RET 结尾，由命令缓冲 CALL 进来。
- **加载一次**：驱动把整个可执行体拷进 CMA，记住每个入口点的物理地址。以后每次 dispatch 只是 CALL 这个地址。
- **所有 DDR 地址都通过 BASE 重定位**，模板里不出现绝对地址，所以同一个模板可以服务不同的缓冲。

### 4.2 文件布局（小端）

| 部分 | 内容 |
|---|---|
| 头部 | 魔数 `SADESC1\0`，版本，D，入口点个数，模板区的字节数，对硬件的要求（CAPS 位：CMDX、FPVE） |
| 入口点表 | 每项：名字、模板在模板区中的偏移和条数、binding 个数、push constant 个数、使用的 PARAM 与 LDPARAM 常量块大小、SPAD / ACC 用量、估计周期（来自代价模型） |
| 模板区 | 64 字节对齐的描述符，入口点首尾相接 |
| 调试段（可选） | 每条描述符对应的源 dispatch 与 linalg 位置，便于报错时定位 |

用 IREE 已有的 flatbuffer 工具生成也可以，关键是内容与上表一致。插件里的序列化器必须与 `DescList` 逐字节一致（有单元测试）。

### 4.3 调用约定

| 资源 | 约定 |
|---|---|
| BASE0–15 | binding i → BASE i。一个 dispatch 最多 16 个 binding（常量权重、scale、输入、输出、KV cache …）。BASE 中是 binding 的物理地址加偏移 |
| PARAM0–5 | 前 6 个 push constant（pos+1、pos_pad、pos_pad/D、偏移量……），由命令缓冲用 SETREG 写入 |
| PARAM6、PARAM7 | **模板内部使用**：循环变量、LDPARAM 的中间结果。调用方不能假定它们在 CALL 前后保持不变 |
| 更多常量 | 超过 6 个时，命令缓冲把常量写进一块 DDR（常量块，作为最后一个 binding），模板用 LDPARAM 按需读入 PARAM6/7 |
| 片上存储器 | 模板独占 SPAD、ACC；dispatch 之间不保留任何片上数据（数据经过 DDR，LLM plan §10.3） |
| 屏障 | 模板结束时不要求引擎空闲；dispatch 之间的依赖由命令缓冲的 FENCE 处理（§5.2） |

### 4.4 实际格式（C1–C4）

实现没有用 flatbuffer，而是自定义的小端二进制，定义与读写工具在 `compiler/runtime/tools/sadesc.py`（C 加载器 `runtime/sa/sa_loader.c` 读同样的布局）：
- **v1**（C1）：头部、入口点表（名字、模板偏移与条数、binding / push constant 个数、估计周期）、模板区、名字区。
- **v2**（C3）：每个入口点可带**装载表**（`BASE r = binding b + f(常量)`、`PARAM r = f(常量)`，`f(c) = ((c·mul) >> shift) + add`），由驱动求值。IREE 把 binding 偏移作为 push constant 传入，所以模板里只有静态偏移，同一个可执行体可服务多层。
- **v3**（C4）：每个入口点 32 字节的扩展：**前缀**（可提前执行的权重 LD，经 BASE15，以 RET 结尾）、**head**（主体开头的 LD，以 RET 结尾，驱动可在其后插入别的 dispatch 的前缀）、读 / 写的 binding 掩码、是否使用 SPAD_B、前缀原来的 BASE 寄存器。模板不能使用 BASE15。

---

## 5. 运行时：sa HAL 驱动（C1）

### 5.1 对象对照

实现的起点：IREE 运行时中新驱动的骨架（以所选版本为准，例如 null 驱动），参考 local-sync 驱动的结构。

| IREE HAL 对象 | sa 驱动中的实现 |
|---|---|
| driver / device | 打开 overlay（mailbox、命令环所在的程序 BRAM 通过 mmap），加载 `rt_fw.bin`，等 `FW_STATE = READY` |
| allocator / buffer | 从 CMA 分配物理连续的缓冲（u-dma-buf 或 PYNQ 使用的 CMA 接口）；每个缓冲记下物理地址 |
| 内存类型 | 设备本地、主机可见但**不一致**：`map` 时 invalidate，`flush` 时写回缓存（memory_model.md 的规则） |
| executable / executable cache | 解析 sa-desc-v1，模板区拷进 CMA，入口点 → 物理地址 |
| command buffer | 录制时直接生成一张描述符列表（§5.2） |
| queue_execute | 把列表作为一个命令环条目提交（门铃 = `RING_TAIL`） |
| semaphore | timeline 值 ↔ 命令环 seq；等待用 UIO 中断（`notify_irq`） |
| 参数文件（.irpa） | 权重加载进 CMA 缓冲，作为常量 binding |

### 5.2 命令缓冲的编码

录制每个操作时向列表追加描述符：

| HAL 操作 | 生成的描述符 |
|---|---|
| `dispatch(executable, entry, bindings, constants, workgroups)` | SETREG 若干条（每条 3 个寄存器）写 BASE0..n 和 PARAM0..5；常量超过 6 个时先写常量块（主机侧写入，作为 binding）；然后 `CALL` 入口点的绝对地址 |
| `execution_barrier` | `FENCE`（掩码 0 = 全部引擎）。IREE 只在需要的地方插屏障，与“DDR 依赖不由硬件跟踪”一致 |
| `fill_buffer` / `copy_buffer` / `update_buffer` | 驱动自带的内置模板：LD / ST 拷贝；VE 写立即数后 ST；小的 update 由主机直接写入 CMA 并 flush |
| 结束 | `END` |

- 一次 `queue_execute` 对应**一个**命令环条目，所以一个 token 的所有 dispatch 只经过 PicoRV32 一次（与 L4 的静态列表相同）。
- 命令缓冲可以重用（IREE 的可重用命令缓冲）：动态值通过 push constant 传入，列表本身不变，就像手写路径“每个 token 只改参数块”。
  如果 IREE 每次都重新录制，也只是 ARM 端的开销，功能相同。
- **实现（C4，§7.1）**：命令缓冲先记录 dispatch，flush 时由驱动统一排列表；屏障处的 FENCE 按 DDR 读写范围的冲突选择掩码，前缀放进更早的 dispatch 中。C1–C3 是一个 dispatch 一张列表。

### 5.3 内存与地址

- **物理地址**：设备只看物理地址。buffer 的物理地址在分配时确定；binding 的偏移在 SETREG 时加上。
- **对齐**：DMA 要求 8 字节对齐，描述符和模板要求 64 字节对齐；allocator 统一按 64 字节对齐分配。
- **瞬时缓冲**：IREE Stream 为 dispatch 之间的中间结果分配瞬时缓冲，驱动从一个 CMA 池里切分，避免频繁的系统调用。
- **大小**：stories15M 的权重约 24 MB，KV cache 0.9 MB，瞬时缓冲很小；110M 约 115 MB，需要确认 CMA 大小（LLM plan §9.1）。

### 5.4 同步

- 命令环按顺序完成，所以同一个队列上的 wait 信号量天然满足；驱动只需要处理主机等待。
- 每个条目带 IRQ 标志：完成时 `rt_fw` 发 `mat_notify`，驱动在 UIO 上等中断，然后读完成记录、推进信号量的值。
- 出错：完成记录的状态非零时，把信号量置为失败状态，报告出错的描述符序号（完成记录里有），映射回 dispatch 名（调试段）。

### 5.5 传输层：板子或模拟器

驱动底层分成可替换的“传输层”：

| 传输层 | 实现 | 用途 |
|---|---|---|
| `board` | mmap 程序 BRAM（mailbox、门铃）、CMA、UIO | 板上运行 |
| `sim` | 通过本地 socket 连接一个 Python 模拟器服务：服务持有一块模拟的 DDR（`SaFuncSim`），执行命令环条目，返回完成记录 | **主机上开发编译器**：整个 IREE 程序在主机上运行，设备部分由功能模拟器逐位执行 |

- `sim` 下的物理地址是模拟 DDR 窗口里的地址，allocator 在窗口里分配。
- 同一个 .vmfb 在两种传输层上的结果必须逐位相同。这也是驱动本身的测试。

### 5.6 C1 的验收

- 不接编译器：用 `compile_layer.py` 生成的模板手工封装成 sa-desc-v1，写一个小的 IREE 程序
  （或 C 测试程序直接调用 HAL API）执行它：
  - 一个量化线性层在 `sim` 和板上都与 `DeviceModel.linear` 逐位一致；
  - 连续提交 1000 次，没有错误和超时，信号量与中断次数一致。

### 5.7 C1 实现与结果（2026-09-27）

代码在 `compiler/runtime/`（驱动、测试）、`compiler/sim/`（模拟器服务、命令环模拟）、`compiler/runtime/tools/`（sa-desc-v1 读写、测试可执行体）。

**结构**：sa 设备 = IREE 的 local-sync 设备 + 三个自己的部件，不重写 HAL 对象：

| 部件 | 文件 | 作用 |
|---|---|---|
| context 与 arena | `sa/sa_context.c` | 设备可见窗口（板上 CMA / sim 的共享内存 DDR）上的首次适配分配器，作为 heap allocator 的 `data_allocator`，所以每个 buffer 都有物理地址；拼装并提交 dispatch 列表 |
| loader | `sa/sa_loader.c` | 解析 sa-desc-v1，模板拷进 arena；`issue_call` 只在 workgroup (0,0,0) 提交：`SETREG`（BASE i = binding i 的物理地址，PARAM j = push constant j）+ `CALL` 模板 + `END 0x5A` |
| 传输层 | `sa/sa_transport_sim.c`、`sa/sa_transport_board.c` | `sim`：Unix socket + 共享内存文件，服务端是 `compiler/sim/sa_sim_server.py`（`SaFuncSim`）。`board`：rt_fw 的命令环（环在窗口开头，完成记录在 +0x2000），轮询 `RING_HEAD` |
| 注册 | `sa/sa_driver_module.c`、`iree_runtime_plugin.cmake` | IREE 外部 HAL 驱动 `sa`（`-DIREE_EXTERNAL_HAL_DRIVERS=sa`），`iree-run-module --list_drivers` 可见 |

**与 §5.1–5.4 的差别**（C1 的简化，功能相同）：
- **一个 dispatch 一张列表、一个命令环条目**，不是一个命令缓冲一张列表（§5.2）。barrier 自然满足（逐个同步执行）。合并成一张列表留到 C4。
- **轮询 `RING_HEAD`**，没有用 `notify_irq`（§5.4）。
- **板上内存**：静态 armv7 程序不能用 PYNQ 的 CMA 接口，所以由 PYNQ launcher（`runtime/test/board_launcher.py`）分配窗口、设置命令环、启动 rt_fw；程序通过 `/dev/mem`（`O_SYNC`，ARM 上对 RAM 是不经缓存的 write-combine 映射）映射窗口和 mailbox，**不需要缓存维护**。之后可以换成 `sa_board_attach()`（由调用者传入已映射的内存）。
- **错误**：完成记录的扩展状态解码成 “引擎 + 错误类型”（`sa_sched.v`：[11:8] code，[15:12] engine，[24] fetch busy），附带已解码的描述符数；sim 与 emu 按同样的格式报告。
- 遇到一个 IREE 的问题：heap buffer 在 data/host 分配器不同（split 模式）时用 `malloc_aligned` 分配，却用 `free` 释放对齐后的指针（`hal/buffer_heap.c`）。arena 按地址查找所在的块，所以任何块内指针都能释放。

**测试**（`runtime/test/sa_hal_test.c`，只用 IREE HAL API：registry → device → executable cache → allocator → 命令缓冲 → queue_execute → 信号量）：

| 项目 | sim（主机） | board 模拟（主机，`sim/sa_board_emu.py`） | 板上（L2 bitstream、rt_fw） |
|---|---|---|---|
| qlinear（`compile_layer` 的量化线性层，K=64、N=160）vs `DeviceModel.linear` | 逐位一致 | 逐位一致 | 逐位一致 |
| axpb（push constant → PARAM → 动态字段） | 逐位一致 | 逐位一致 | 逐位一致 |
| binding 带偏移（同一 buffer 的子区间） | 逐位一致 | 逐位一致 | 逐位一致 |
| fault（LD 越界）：提交失败并报告设备状态 | LD range error，0x1000203，3 个描述符 | 同左 | 同左 |
| 出错后再运行 qlinear | 逐位一致 | 逐位一致 | 逐位一致 |
| 压力：qlinear + barrier + axpb 连续提交，每次检查结果与信号量 | — | 200 次，0 错（环回绕 25 圈） | **1000 次，0 错**，信号量 1003 = 期望；每次提交 152 µs（2 个 dispatch） |
| D 不匹配的可执行体 | 加载时报 INCOMPATIBLE | — | — |

构建：`compiler/scripts/build_sa_runtime.sh host|armv7`（只构建运行时，约 1 分钟）；板上部署 `compiler/scripts/deploy_c1.sh` → `build/deploy_c1`。

---

## 6. 编译器：sa 目标后端插件（C2、C3）

### 6.1 插件结构

- iree-compile 的目标后端是编译期插件，**必须从源码构建 iree-compile**：pip 包不能加载自定义后端。
  - 版本与 L0 的运行时相同（IREE 3.11.0，e4a3b0405d），用 `-DIREE_CMAKE_PLUGIN_PATHS=<iree-sa/compiler>` 把插件编进去；
  - 构建时间和内存都很大，限 4 个 job（与 Vivado 的约定相同），由用户在自己的终端里运行。
- 插件实现 IREE HAL 的 `TargetDevice` 与 `TargetBackend`（以所选版本的接口为准）：

| 接口 | 作用 |
|---|---|
| 默认的 executable target | 声明 `sa` 设备、`sa-desc-v1` 格式，配置里带 D、CAPS、SPAD / ACC 大小 |
| `buildConfigurationPassPipeline` | 给每个 dispatch 选模板，检查数值约定（§3.2）；选不到就报错并打印 dispatch 的 IR |
| `buildTranslationPassPipeline` | 把 dispatch 的 linalg IR 变成描述符模板（C3：模板匹配加参数化；C5：完整代码生成） |
| `serializeExecutable` | 输出 sa-desc-v1（§4） |

### 6.2 dispatch 的形成与预处理

IREE 怎样切 dispatch，决定了模板要处理什么形状。插件在 IREE 的流水线前后加入预处理：
- **保持高层算子**：softmax 等在分解之前先匹配，保留成一个 dispatch（或加自定义的 dispatch 区域）。
- **禁止**会改变数值的变换：拆分归约（split-k）、重结合；这些在插件的设备配置里关闭。
- **控制融合**：量化线性层的形状是“量化 → int8 matmul → 反量化”，希望融成一两个 dispatch；
  逐元素运算链融进前后的 dispatch 是可以接受的，由 VE 表达式编译器处理（§6.4）。
- **KV cache 原地更新**：确认 `tensor.insert_slice` 到全局变量被实现为在原缓冲上写一行。

### 6.3 数据分块与权重打包

- ~~matmul 的操作数带 encoding，由插件为 sa 设备解析成 pack~~。C2 的实验表明这条路现在走不通：IREE 的 data tiling 需要 i8 × i8 → i32 的 matmul 形式（前端写的是扩展成 i32 后的 matmul）和插件自定义的 encoding resolver；在 torch 图里手写打包又会被 IREE 的布局规范化撤销（变回运行时转置）。
  **实际做法（C2）**：插件的预处理 pass `iree-sa-pack-linear-weights`（`plugins/sa/target/PackLinearWeights.cpp`），只在目标是 sa 设备时运行：
  `vecmat(x, transpose(extsi(W)))`（W 是常量或不可变 global 的 i8 权重）→ `Wp = linalg.pack W inner_tiles=[D]`（i8 [N/D, K, D]，即 `pack_b`）+ 直接读 Wp 的 contraction + `collapse_shape`，算术与原 vecmat 相同。
- 常量权重在编译期求值（const-eval），pack 在编译时完成，运行时不做重排：编译时导入参数（`--iree-parameter-import=model=<irpa> --iree-parameter-import-maximum-size=...`），打包后的权重导出到新的归档（`--iree-parameter-export=model=<packed.irpa>`）。结果与 `export_w8a8.pack_b` 逐字节相同（`compiler/tests/test_c2.py` 检查）。
  权重先被打包成 i8，所以常量提升可以重新打开（阈值 `--iree-opt-const-expr-max-size-increase-threshold=0`，不再出现 C0 那样把权重提升成 i32 的情况）：stories15M 整个模型 4.5 秒编完，5 种线性层全部变成 pack 布局，打包后的参数 16.1 MB。
- 激活侧：复制行的 A 条带由模板内部生成（VE 的 DIV-D 复制，或 L5b 的经 DDR + LD INTERLEAVE），不经过 IREE 的 pack。

### 6.4 模板库与匹配（C3，方式 A）

匹配以 dispatch 函数体的**结构**为准：哪些 linalg 算子、iterator 类型、indexing map、元素类型、运算体里的 arith / math 操作。
每个模板是参数化的描述符生成器（C++ 实现，内容移植自 `compile_layer.py` / `compile_model.py`）：

| dispatch 形式 | 模板 | 参考实现 |
|---|---|---|
| 激活量化（max(abs) 归约 + 乘以倒数 + 转 int8） | `quant_act` | `compile_layer.quant_act` |
| int8 × int8 → int32 matmul，带或不带反量化尾部（乘 scale 向量、乘 s_x） | `qlinear`：分块、双缓冲、反量化 | `compile_layer.linear` |
| 逐元素运算链（`linalg.generic`，全是 parallel iterator） | **VE 表达式编译器**：运算体中每个 arith / math 操作 → 一条 VE（OP、×A+B、FUNC），indexing map → 下标模式（LIN、MOD、DIV、IMM）；中间值放在片上 | RoPE、SiLU、残差 |
| 按行归约（sum / max，reduction iterator） | `reduce`：REDUCE + ROWLEN | RMSNorm 的平方和、amax |
| softmax（带掩码） | `softmax`：max、exp、sum、recip、乘（VALID = 动态长度） | `compile_model.attention` 的 softmax 部分 |
| 注意力分数 / P·V（小的 int8 matmul，动态长度） | `attn_scores`、`attn_pv`（TRANSPOSE、INTERLEAVE、动态 repeat / Kt） | `compile_model.attention` |
| transpose | `transpose` | TRANSPOSE |
| gather（嵌入行） | `gather`：LDPARAM 算地址 + LD + 读 scale | `compile_model.embed` |

- VE 表达式编译器是 C3 中最通用的部分：它其实已经是一个小的代码生成器，只是限于逐元素运算。
- 模板的片上内存规划与分块大小，沿用 `compile_layer.Layout` 的规则（§6.5）。

### 6.5 dispatch 内的内存规划

- SPAD_A：A 条带；SPAD_B：权重块（两个 bank 交替）；ACC：暂存区与 fp32 中间值。
- 分块大小：能整除输出块数、放得下一个 SPAD_B bank 和 ACC 暂存区的最大值（`Layout.chunk_tiles`）。
- 片上空间不够时（大模型的长向量），模板报错，由预处理把 dispatch 切小。

### 6.6 不支持的 dispatch

- **默认**：报错，打印 dispatch 的 IR 和最接近的模板。这样覆盖范围永远是明确的。
- **可选**：CPU 与加速器混合执行，不支持的 dispatch 放在 `llvm-cpu` 上，用 IREE 的多设备与亲和性（affinity）。
  IREE 这部分还在演进，作为后备方案，不作为主路径（§12）。
- llama decode 用到的 dispatch 种类有限（手写路径已经全部覆盖），所以目标是**全部在加速器上**。

### 6.7 C2 结果（2026-09-27）

第一条纵向切片：一个 torch int8 线性层 → iree-turbine → 带 sa 插件的 iree-compile → .vmfb + 打包后的 .irpa → `iree-run-module --device=sa`（C1 驱动）。

| 部件 | 文件（`compiler/plugins/sa/`） | 作用 |
|---|---|---|
| 预处理 | `target/PackLinearWeights.cpp` | 线性层权重 → B 条带布局（§6.3） |
| 翻译 | `target/LowerWorkgroupCount.cpp` | 每个 export 的 workgroup 数为 (1, 1, 1)：一个 dispatch = 一个模板，只执行一次 |
| 匹配 | `target/Match.cpp` | 严格匹配 dispatch 的结构：x（i8 或 i32）的 load、`acc[t, j] += ext(x[k]) * ext(Wp[t, k, j])`（fill 0）、尾部 `(f32(i32(acc)) * s_w) * s_x`（运算顺序、无 fast-math）、整块的 binding；得到 k、n 和每个操作数的 binding |
| 模板 | `target/Templates.cpp`、`target/DescList.cpp` | `compile_layer.linear` 与 `DescList` 的 C++ 移植；Python 参考 `templates/reference.py` |
| 序列化 | `target/SATarget.cpp` | sa-desc-v1（与 `runtime/tools/sadesc.py` 相同的布局）；没有模板的 dispatch 报错并打印整个 dispatch |

qlinear 模板（参考 `templates/reference.py`）：
- A 条带：x 为 i8 时 LD 进 SPAD_A，再用 DIV-D 的 VE COPY 复制成 D 行；x 为 i32 时 LD 进 ACC，再做 I32 → I8 的 COPY。
- s_x：LD 一个 fp32（DMA 以 8 字节为单位），用只对第一个元素的 max 归约（`valid = 1`）把它广播到整个字。VE 的下标模式以字为单位，不能在字内广播。
- 之后与 `compile_layer.linear` 相同（分块、双缓冲、反量化；超过 4 块时用 LOOP_END，偏移放在模板私有的 PARAM6/7）。

测试（`compiler/tests/test_c2.py`，板上 `compiler/tests/board_c2.py`）：

| 用例 | 形状 | 模板 vs Python 参考 | 打包权重 vs `pack_b` | sim（D=8、16） | 板上（D=8） |
|---|---|---|---|---|---|
| qkv_i8 | k=288，n=864，x i8 | 逐字节相同 | 相同 | 逐位一致 | 逐位一致 |
| qkv_i32 | k=288，n=864，x i32 | 逐字节相同 | 相同 | 逐位一致 | 逐位一致 |
| w2_i32 | k=768，n=288 | 逐字节相同 | 相同 | 逐位一致 | 逐位一致 |
| long_i8 | k=288，n=2048（LOOP_END 形式） | 逐字节相同 | 相同 | 逐位一致 | 逐位一致 |

另外：C++ 模板生成器单独编译后，与 Python 参考逐字节比较了 56 个配置（D=8/16，含 n=32000 的分类器形状、binding 顺序打乱）；非 sa 目标（vmvx）不受预处理 pass 影响。

### 6.8 C3 实施方案（2026-09-27，基于 C2 流程下 stories15M 的实际 dispatch）

用 C2 的编译流程（权重打包 + 常量提升 + 导入参数）编译 stories15M，得到 64 个 dispatch（`build/c3/sources`）。与 C0 的清单相比，有几点决定了 C3 的做法：

| 现象 | 例子 | 对策 |
|---|---|---|
| 同一个 dispatch 被多层复用，binding 的偏移来自 push constant（多个常量打包在一个资源里），同一 binding 上有多个偏移不同的张量 | 线性层（6 层共用）、量化 | **sa-desc 的寄存器装载表**（下面 A） |
| push constant 超过 6 个；64 位的值由两个常量拼成（`extui`/`shli`/`ori`）；动态长度 T 以 workload ordinal 出现 | 注意力的 7 个常量 | 装载表只把设备需要的常量放进 PARAM |
| 按数据的掩码 `select(valid[t] > 0.5, x, -inf)` | softmax 的 max / exp | 前端改成**按长度的掩码** `arange(T) <= pos`（主流做法：注意力内核接收序列长度），编译器识别“下标 ≤ 标量”的前缀掩码 → VE 的 VALID 字段（LDPARAM 读 pos） |
| torch 的 exp、1/x、rsqrt | SiLU、softmax、RMSNorm、量化 | 映射到硬件 SFU（exp、recip、rsqrt）：结果按 DeviceModel(SfuExact) 的定义，与 L5 逐位一致 |
| `to_i8` 展开成 NaN→0、±inf 截断、roundeven、±127 截断、fptosi | 量化、KV 写入 | 识别整个模式 → VE 的 I8 输出转换（本来就是这个语义） |
| gather：`tensor.extract` 的行号来自设备上的 i64 标量 | 嵌入、RoPE 表 | LDPARAM 读行号算地址 → 动态地址的 LD |
| scatter：KV 行写入缓存 | 每层 K、V | LDPARAM → 动态地址的 ST |
| 设备上的 i64 标量运算 | `pos + l·S` | VE 上的 I32 运算（值 < 2^24，fp32 精确）；高位字清零 |
| 注意力的两个小 int8 matmul（动态 T） | Q·Kᵀ、P·V | 移植 `compile_model.attention` 的模板 |

实现分成几部分（C++，`plugins/sa/target/`；每一部分都有 Python 参考或 funcsim 测试）：

- **A. sa-desc 寄存器装载表**：每个 export 带一张表，`BASE r = binding b 的物理地址 + push constant j`（或 + 0），`PARAM i = push constant j`。驱动在 SETREG 时计算。模板里只有静态偏移。这相当于 Vulkan 的 dynamic offset，设备端没有额外开销。格式版本升到 2（`sadesc.py`、驱动的 loader 同步）。
- **B. dispatch 分析**：把 dispatch 函数解析成“张量 + 运算”的程序：每个 load / store 的（binding、偏移表达式、静态切片、形状、类型），push constant 的用途（偏移、长度、其他）。
- **C. 片上存储模型**：张量按行主序展平，元素 e 在字 e/D 的第 e%D 个 lane；fp32 / i32 放 ACC，i8 放 SPAD；标量和“每行一个”的值存成广播字（所有 lane 相同）。动态维度按上界分配（插件选项，默认 seq_len）。
- **D. VE 表达式编译器**（C3 的核心）：`linalg.generic` 的运算体（逐元素或按行归约）→ VE 指令序列。
  - 操作数的访问方式：原样、标量广播、按行广播、成对交换取负（SWAPNEG）、gather 的行；
  - 常量折进 A / B / imm；exp / recip / rsqrt / abs 用 SFU；
  - 归约用 REDUCE（sum / max，ROWLEN、VALID）；to_i8 用 I8 输出转换；前缀掩码用 VALID。
- **E. contraction 模板**：qlinear 推广到带偏移的 binding 和融合的尾部（例如残差加法，尾部其余的运算交给 D）；注意力的 Q·Kᵀ、P·V。
- **F. gather / scatter / 整数标量**：见上表。
- **G. 运行器**：一个 C 程序（IREE runtime API）逐 token 调用模型，写出 logits，相当于 L5 的 `LlamaDevice`（KV cache 是模块里的全局变量，要在同一进程里跨 token 保持）。
- **H. 验证**：
  - 每个模板有 Python 参考或 funcsim 测试；
  - 端到端：IREE 编译的 stories15M 在 sim 上逐 token 的 logits 与 DeviceModel(SfuExact) 逐位一致；
  - 板上生成的文本与 L5 完全相同。

顺序：H 的框架 + 前端改动 → A → B/C/D（逐元素、归约）→ E（qlinear 推广）→ F → E（注意力）→ G → 端到端 → 板上。

### 6.9 C3 结果（2026-09-27）

**IREE 编译的 stories15M 在板上生成的文本与 L5 手写路径完全相同，每一步的 logits 与 DeviceModel(SfuExact) 逐位一致。**

| 项目 | 结果 |
|---|---|
| 编译 | `compile_sa.sh`：4.4 s；64 个 dispatch 全部由 sa 后端生成模板（没有 CPU 回退）；sa.vmfb 10.3 MB + sa_packed.irpa 16.1 MB |
| 逐 dispatch 差分测试（`tests/dispatch_check.py`：oracle 执行 IR 语义 vs 模板在 funcsim 中运行，每个调用点） | 64/64 逐位一致 |
| 端到端，sim（`tests/test_c3.py`，`sa-llm-run` + `sa_sim_server.py`） | 20 步（12 个提示 + 8 个贪心）逐位一致；约 1.4 s / token（Python funcsim） |
| 端到端，板上（`tests/board_llm.py`，L2 bitstream、rt_fw） | 76/76 步逐位一致，生成的 token 与主机参考相同；**12.6 tok/s**（L5 手写：15.1 tok/s）；每 token 242 次 dispatch |

实现（`plugins/sa/target/`）：

| 文件 | 内容 |
|---|---|
| `Codegen.cpp` | dispatch 程序分析：push constant → `Lin`（常数或 f(一个 push constant)，含 64 位拼接）、binding 区域与 BASE / PARAM 分配（装载表）、片上张量（packed / 广播两种布局，互转用复制 + TRANSPOSE）、VE 表达式编译器、qlinear（带融合尾部）、注意力、scatter、i64 标量 |
| `CloneCheapProducers.cpp` | 预处理：只依赖下标和标量的逐元素值（掩码）复制进每个消费者，保留“下标 ≤ pos”的结构 |
| `Templates.cpp` | `emitLinear`：带偏移和逐块尾部回调的 `compile_layer.linear`（C2 的字节比对不变） |
| `DescList.cpp` | 增加动态字段、TRANSPOSE、LDPARAM、FENCE |

表达式编译器的要点：每条 VE 是 `FUNC(OP(s1', s2)·A + B)`，IR 的运算按顺序折进这几个阶段（常数 → A / B / IMM，`1/x` / exp / rsqrt / abs → SFU，negf → A = −1）；恒等的 B 用 −0（`y + (−0) = y`，保持零的符号）；标量广播用“只对第一个元素的 max 归约”；前缀掩码 → VALID（LDPARAM 读 pos）；`to_i8` 整条链 → I8 输出转换；gather 的下标通过在迭代空间上逐点求值来判定（行 gather、成对交换）。

与计划相比的决定：
- **注意力长度静态**（`export.py --static-len`：valid 固定为 seq_len，按 pos 屏蔽）。动态 T 的带掩码 softmax 需要 LEN、ROWLEN、VALID、按行周期 4 个动态字段，而一条描述符最多 2 个。静态长度下结果仍逐位一致（被屏蔽的位置 exp = 0，求和多加的 +0 不改变部分和，P = 0 的行对 P·V 贡献 0），代价是注意力总按 256 个位置计算。按长度分桶编译几个版本是 C4 的选项。
- **前端常数写成精确的 fp32 值**（`qllama.f32c`）：`1.0 / 288` 经导出后变成比 `F(1/288)` 小一个 ulp 的常数，是端到端唯一的差异来源（逐 dispatch 测试全过，用 `QLlama(n_layers, tap)` 逐层、逐步二分定位）。
- 同一个可执行体被多层复用：偏移经装载表在驱动中计算，所以 64 个可执行体服务 242 次 dispatch。

---

## 7. 性能：跨 dispatch 的调度（C4）

C3 完成以后，IREE 版本的速度会比手写路径（21.8 tok/s）低，因为：
- dispatch 之间的数据经过 DDR，而且每个 dispatch 之间有 FENCE；
- 手写路径 §8.8 的跨层优化（预取下一层的第 0 块、SiLU 融进 W13 的分块循环）不在单个 dispatch 的范围内。

C4 的做法，按收益排序：
1. **只在需要的地方 FENCE**：只有 DDR 读写真正相关时才加 FENCE；片上的依赖交给记分板。
2. **跨 dispatch 预取**：命令缓冲知道下一个 dispatch 是什么，驱动（或插件在链接阶段）可以把下一个 dispatch 第一块权重的 LD 提前。
   需要模板声明“第 0 块由调用方预取”这种接口（与 `compile_layer.linear(preloaded=…)` 相同）。
3. **更大的 dispatch**：在预处理里把一层内能融合的都融合（例如 W13 + SiLU），模板内部再按 §8.8 调度。
4. **代价模型**：功能模拟器扩展成周期估计，用板上计数器校准；插件用它选分块大小和融合方式。

**验收**：IREE 版本每个 token 的周期与手写路径的差距 ≤ 10%，用 `l4_phase_cost.py` 式的分段计时说明剩下的差距在哪里。

### 7.1 C4 结果（2026-09-27，暂缓）

**板上：每个 token 2,591,234 周期（C3 起点 3.63M），墙钟 17.7 tok/s（C3：12.6；L5 手写路径：15.1），76/76 步逐位一致，文本与 L5 相同。** 手写路径（预取 + SiLU 融合）是 2,293,970 周期，差距 13%，没有达到 ≤ 10%。按用户的决定，剩下的优化暂缓，先做 C5：大的融合在通用代码生成里做更自然，C4 已有的运行时机制在 C5 中照样使用。

| 步骤（提交） | 内容 | 周期 / token | tok/s（墙钟） |
|---|---|---|---|
| C3 起点（05d3c66） | 静态注意力长度，242 个 dispatch，一个 dispatch 一张列表 | 3.63M | 12.4 |
| 1 动态注意力（a377e75） | 注意力按实际长度（`--pad=8`）；[H, T] 的值按行处理，使每条描述符不超过 2 个动态字段 | 2.93M | 14.4 |
| 2 激进融合（173457e） | `--iree-dispatch-creation-enable-aggressive-fusion`：242 → 200 个 dispatch | 2.91M | 14.9 |
| 3 只算一次（322c271） | 标量 / 按行的值只计算一次；区域内相同的 LD 共用 | 2.69M | 15.9 |
| 4 批量提交（2a24101） | 一个命令缓冲一张列表（驱动自带的 sa device / command buffer，仿 IREE local-sync）；每 token 7 张列表；主机开销约 9 ms → 3.4 ms | 2.66M | 17.4 |
| 5 列表调度（7b47707） | sa-desc v3；驱动在 flush 时排列表：线性层第 0 块权重的 LD 放进更早的、不用 SPAD_B 的 dispatch 的 head 之后，与其 VE 计算重叠；FENCE 按 DDR 冲突选掩码（写后读只等 ST，读后写只等 LD） | 2.610M | 17.9 |
| 6 abs-max 折半树（e0f094e） | \|x\| 走流水模式，逐元素 MAX 折半，最后对少数几组 REDUCE（REDUCE 一次处理一组）；值 ≥ +0，顺序不影响结果 | 2.591M | 17.7 |

测量工具：`SA_PROFILE=1`（一个 dispatch 一张列表，按入口点统计设备周期与主机时间）、`SA_PROFILE=batch`（按列表统计，外加屏障 / FENCE / 前缀的计数）、`board_profile.py [--batch] [NAME=VALUE]`。

试过但放弃的：
- **把前缀放在本 dispatch 的 FENCE 之前**：只省 6k 周期。调度器按顺序派发，前缀排在上一个 dispatch 的 ST 之后，而 ST 要等 VE 算完才能派发，所以没有真正提前。结论：预取必须在列表中排在要重叠的 VE 工作之前（于是有了步骤 5）。
- **第 1 块权重也做成前缀**：变慢约 8 万周期。模板的双缓冲本来就让第 1 块的 LD 与第 0 块的 EX 并行；提前只会占用唯一的 LD 引擎，推迟宿主 dispatch 的加载。
- IREE 的 `fuse-multi-use`、`element-wise-fuse-multi-reduction` 不改变 dispatch 数；`enable-aggressive-reshape-movement` 把 200 个降到 173 个（61 个可执行体），尚未验证。

决定：**跨 dispatch 调度放在驱动里**（§13 第 5 条原本倾向编译器）。驱动在 flush 时能看到真实的 binding 地址，可以精确判断读写冲突；编译器通过 sa-desc v3 提供所需的信息（前缀、head、读写掩码、SPAD_B）。

剩下的差距（逐 dispatch 计时，每个 token）与暂缓的条目：
- **SiLU**：3 个 dispatch（exp；1 + 倒数、两次乘法、abs-max；量化），每层约 3 万周期，没有重叠。手写路径把 SiLU 融进 W13 的分块循环，与 EX 并行，省约 10.6 万周期。需要预处理自行组成 dispatch region（W13 与 SiLU 放在一起），模板内交错。
- **量化链**：每个线性层前有 abs-max → 标量 → 量化 3 个 dispatch，每条边界一次 DDR 往返 + FENCE；168 个屏障全部是真实依赖。
- **代价模型**（§7 第 4 条）：未做。第 1 块预取的教训说明，把哪些加载放进哪个宿主，需要周期估计才能判断。
- 分类层（1.26M，与手写路径相同）和每层的权重流是下限：权重搬运约 1.92M 周期。

---

## 8. 完整代码生成：sa 方言（C5）

状态：**方案**（2026-09-27）。C4 暂缓（§7.1），先做 C5；C4 剩下的融合在 C5 的调度器上做。

### 8.1 目标、范围与验收

C3/C4 的代码生成器是“模板库 + VE 表达式编译器”：线性层、注意力、scatter 各有一段专用的 C++，只认得 stories15M 里出现的那几种 dispatch 形式。C5 把它换成一条通用的 MLIR 流水线：dispatch 的 linalg IR → 分块 → 布局 → 缓冲化 → `sahl`（tile 级）→ `sahw`（命令级）→ 内存分配、流水、调度 → 描述符（§8.3）。新的模型结构不必再写模板。

模板**不删除**，而是改写成 `sahl` 层的**微内核**（§8.8）：通用代码生成保证什么都能编译，手写的微内核负责最关键、值得手工调的算子（与 IREE 的 ukernel、cuBLAS 与 Triton 并存同理）。

**不变的部分**：前端与导出（C0）、预处理（权重打包、`CloneCheapProducers`）、sa-desc v3 格式、驱动与列表调度（C1、C4）、数值约定（§3.2）、全部测试工具。硬件不变。

**验收**：
1. **正确**：旧的直接生成描述符的代码生成器（`Codegen.cpp` 的模板路径与 `Templates.cpp` 的 DescList 版本）退役，只用新流水线编译 stories15M：58 个 dispatch 全部通过 `dispatch_check.py`；sim 与板上 76/76 步逐位一致，文本与 L5 相同。默认配置（启用微内核）和 `--iree-sa-ukernels=none`（只用通用代码生成）**两种都要满足**。
2. **不慢**：默认配置下，板上每个 token ≤ 2,591,234 周期（C4 的结果，`board_profile.py --batch`）；逐 dispatch 的周期（`SA_PROFILE=1`）与 C4 相比不超过 +2%。`--iree-sa-ukernels=none` 的周期也要记录，作为通用路径的质量指标。
3. **通用**：stories15M 之外、没有为它写过任何专用代码的模型（§8.7 C5.5：SmolLM2-135M 与 Qwen3 结构的小配置），用 `--iree-sa-ukernels=none` 编译通过；逐 dispatch 与 oracle 逐位一致，端到端与 torch fp32 的误差在范围内（这些模型没有手写的 DeviceModel）。编译器不写死任何板子或模型的数字（§8.9）。

### 8.2 现有代码生成器的各部分在 C5 中的去向

| C3/C4 的部分（`plugins/sa/target/`） | C5 |
|---|---|
| push constant 分析 `Lin`（含 64 位拼接、`util.assume.int`）、装载表 | 保留为分析，由 `sahw-legalize-dynamic` 与 `sahw-assign-registers` 使用 |
| binding 区域、BASE / PARAM 分配（PARAM 从 0 往上、私有的从 7 往下、BASE15 留给前缀） | `sahw-assign-registers` |
| 片上张量的两种布局（packed、广播字）与行步长（动态 T） | 布局属性 `#sa.layout` + `sa-layout` 传播 + `sahl-plan-memory` / `sahw-allocate` |
| VE 表达式编译器（每条 VE = `FUNC(OP(s1', s2)·A + B)`，to_i8 链、前缀掩码 VALID、gather 下标判定、SWAPNEG） | 拆成两步：`sahl.elementwise` 的运算体逐个降级成单级 `sahw.ve`（`sahl-to-sahw`），再由 `sahw-fuse-ve` 合并成多级；规则不变 |
| `emitLinear`（分块、双缓冲、逐块尾部） | 分块（`sa-tile`）+ A / B 条带布局 + `sahl.matmul` + 软件流水（`sahl-pipeline`）；另外改写成 `linear` 微内核（§8.8） |
| 注意力模板（Kᵀ 条带、按行的动态长度） | `batch_matmul` 走同一条 contraction 路径，Kᵀ 由布局传播插入 TRANSPOSE；原模板改写成微内核（§8.8） |
| scatter、gather、i64 标量 | `sahl.gather / scatter / scalar`，降级到 `sahw`（LDPARAM、动态 DDR 地址、SWAPNEG 技巧） |
| 按行处理（每条描述符最多 2 个动态字段） | `sahw-legalize-dynamic`：超过就切成行循环 |
| 只算一次的标量 / 按行值、相同 LD 共用（C4 步骤 3） | `sahl-uniform`（带内存效果的 CSE） |
| 前缀、head（C4 步骤 5） | `sahl-schedule` 选出，`sahw.template` 表示 |
| abs-max 折半树（C4 步骤 6） | `sahl-to-sahw` 里 max 归约的一种降级方式，由代价模型选择 |
| `DescList`、sa-desc v3 序列化 | 保留，改为从 `sahw` 序列化 |

### 8.3 两层方言：`sahl` 与 `sahw`

```
linalg / scf（IREE 的 dispatch）
  → sahl：tile 级，按硬件语义，与命令编码无关
  → sahw：命令级，一个操作 = 加速器的一条命令
      ├→ 描述符序列化（sa-desc v3；C5）
      └→ LLVM dialect → LLVM IR → riscv32 目标文件（PicoRV32 用 PCPI 指令发命令；C7，§8.10）
```

分两层的理由：高层的决定（布局、分块后的 tile 运算、内存规划、流水、微内核）与命令的编码细节（VE 的级、每条最多 2 个动态字段、寄存器、LOOP_END）分开。硬件换代（D、VE 的级数、动态字段数）主要改 `sahw` 和目标配置；微内核写在 `sahl` 上，不绑定编码；同一组命令既可以编成描述符，也可以由 RISC-V 代码逐条发出。描述符路径不经过 LLVM IR，LLVM 只用于在处理器上运行的代码。

**共用的属性**：
- `#sa.mem<spad_a | spad_b | acc>`：片上内存空间（`sahl` 上带 bank 规划，`sahw` 上带字偏移）。
- `#sa.layout<packed | bcast | a_strip | b_strip | kt_strip>`：LLM plan §10.3 的布局类型（紧凑向量、广播字、A / B 条带、Kᵀ 条带），是片上 memref 类型的一部分。
- 目标配置（§8.9 第 1 项）：两层的 verifier 与 pass 都从这里取容量和能力。

**`sahl`（tile 级）**：操作数是 memref（DDR 一侧是 binding 的 subspan，片上一侧带 `#sa.mem` 与 `#sa.layout`），循环仍用 scf。
- `sahl.load`、`sahl.store`：DDR 与片上 tile 之间搬运（任意形状与步长，布局由目标 memref 决定）；
- `sahl.matmul`：int8 tile 矩阵乘，int32 累加（可选累加到已有结果）；
- `sahl.elementwise`：带 region 的逐元素运算，region 里是原来的 arith / math，indexing map 保留；
- `sahl.reduce`：sum / max 归约（顺序按 §3.2）；
- `sahl.relayout`：布局转换（转置、DIV-D 复制、打包成广播字）；
- `sahl.gather`、`sahl.scatter`：由数据决定地址的读写（嵌入行、KV 写入）；
- `sahl.scalar`：i64 标量运算（位置、行号）。

**`sahw`（命令级）**：一个操作对应一条命令，外加结构操作。
- `sahw.ld`、`sahw.st`：rows、row bytes、pitch、mode（LINEAR / INTERLEAVE）；
- `sahw.ex`：kt、accumulate、repeat、B / C 步长、C 行数；
- `sahw.ve`（级：`op`、`func`、`m1 / m2 / p1 / period`、`A`、`B`、`imm`、`swapneg`、`reduce`、`rowlen`、`valid`、输入 / 输出类型）、`sahw.transpose`；
- `sahw.ldparam`、`sahw.fence`、`sahw.setreg`；
- `sahw.loop`（带区域，降级成 LOOP_END 与 PARAM 步长；RISC-V 后端则是真正的循环）；
- `sahw.template { sahw.prefix {…} sahw.head {…} sahw.body {…} }`：一个 export 的结构，对应 sa-desc v3 的前缀、head、主体。
- **动态值**：大小和偏移是 `index` 类型的 SSA 值，来自 push constant。描述符后端要求每个操作最多 2 个动态字段（`sahw-legalize-dynamic` 之后由 verifier 检查）；RISC-V 后端没有这个限制。

**接口**（两层都实现）：内存效果（读写哪个片上 bank 的哪段、哪个 binding），供 CSE 与调度使用；代价（估计周期，§8.5）。

### 8.4 流水线（`buildTranslationPassPipeline`）

序列化器只把 `sahw.template` 翻译成字节，所有决定都在 pass 里完成，每个 pass 都可以用 `iree-opt` 单独运行和测试。

| # | 层 | pass | 要点 |
|---|---|---|---|
| 1 | linalg | `sa-normalize` | 具名算子泛化（batch_matmul、fill）；extsi / trunci 折进 contraction；识别 i8 × i8 → i32 的 contraction |
| 2 | linalg | `sa-tile` | TilingInterface（`scf::tileConsumerAndFuseProducersUsingSCF`）：contraction 按 N 分块，块大小由 SPAD_B bank 与 ACC 暂存区决定（同 `Layout.chunk_tiles`），尾部融进每块；K 放不进一个 B 条带时按 K 分块，EX 用累加标志跨块累加（int32，结果与顺序无关）；过长的向量切段；所有容量来自目标配置（§8.9），块大小可由代价模型选 |
| 3 | linalg | `sa-layout` | contraction 的 A → A 条带（由 packed x 做 DIV-D 复制），B → B 条带（打包后的权重直接 LD），Kᵀ → 转置，标量 → 广播字，其余 packed；布局不一致处插入转换 |
| 4 | linalg | 缓冲化 | tensor → memref。先试 IREE 的 one-shot bufferize（片上缓冲 = 带 `#sa.mem` 的 alloc）；不顺利就用自己的缓冲化（C3 已有同样的分析），见 §8.11 |
| 5 | linalg → sahl | `sa-to-sahl` | copy → `sahl.load / store`；contraction → `sahl.matmul`；逐元素 / 归约 → `sahl.elementwise / reduce`（运算体原样保留）；scatter / gather / i64 标量；**微内核在这里选择**（§8.8） |
| 6 | sahl | `sahl-uniform` | 标量 / 按行的值只算一次，相同的 load 共用（带内存效果的 CSE） |
| 7 | sahl | `sahl-plan-memory` | 按 bank 规划片上缓冲（记分板按 bank 判断依赖），双缓冲的块放在不同 bank，检查容量；放不下时报错并给出建议的块大小 |
| 8 | sahl | `sahl-pipeline` | 两级软件流水：第 i 块 matmul 之后 load 第 i+1 块到另一个 bank（即模板的双缓冲） |
| 9 | sahl | `sahl-schedule` | 粗粒度调度：提前独立的 load；选出 prefix（可提前的权重 load）与 head |
| 10 | sahl → sahw | `sahl-to-sahw` | 运算体的每个 arith / math → 单级 `sahw.ve`（indexing map → LIN / MOD / DIV / IMM）；matmul → `sahw.ex`；relayout → TRANSPOSE / DIV-D 复制；gather / scatter → LDPARAM + 动态地址；max 归约可选折半树（代价模型选择） |
| 11 | sahw | `sahw-fuse-ve` | C3 的折叠规则：常数乘 → A，常数加 → B，SFU 函数 → FUNC，取负 → A = −1，恒等的 B 用 −0；to_i8 链 → I8 输出；前缀掩码 → VALID。只做不改变舍入的合并 |
| 12 | sahw | `sahw-legalize-dynamic` | 动态值 → PARAM（装载表或私有 PARAM 的 `SETREG +=`）；超过 2 个动态字段就切成行循环（描述符后端） |
| 13 | sahw | `sahw-allocate` | 片上最终字偏移（含 VE 的中间值），遵守 `sahl-plan-memory` 的 bank 规划 |
| 14 | sahw | `sahw-schedule` | 细粒度表调度：按序派发、引擎队列、bank 冲突；把 VE 插进 EX 之间；次数多的循环 → `sahw.loop`，少的展开 |
| 15 | sahw | `sahw-assign-registers` | BASE / PARAM 分配与装载表（§4.4），BASE15 留给前缀 |
| 16 | sahw → 字节 | 序列化 | `sahw.template` → sa-desc v3，经 `DescList`（RISC-V 后端改为 `sahw-to-llvm`，§8.10） |

**实现状态（2026-09-29，C8 之后）**：分块（逐元素的分片：`sahl-tile`）、内存的位置与布局（`sahl-plan-memory`）、线性层调度（`sahl-schedule`）、内核与 gather 的识别（`sa-to-sahl`）已是独立的 pass；第 1 项的归一化、第 12 项、第 14 项的细粒度调度，以及 contraction 的分块与双缓冲仍在翻译里。见 §8.14。

### 8.5 代价模型

分块、调度，以及 C4 暂缓的融合都需要它。块 1 预取的教训（§7.1）说明，不能凭经验放置加载。

- **每个操作的估计**：
  - LD / ST：字节数 / 7.9 加上延迟；
  - EX：kt × repeat；
  - VE：流水模式每组约 2 周期（lane 折叠），多拍模式每组的固定代价（EXP、RECIP、RSQRT、REDUCE 各不相同）；
  - 每条命令的启动延迟。
- **dispatch 的估计**：一个按序派发、带引擎队列和 bank 记分板的小模拟器，是 C++ 实现，在 `sahl` 与 `sahw` 上运行（`sahl` 用于分块、流水与微内核的选择，`sahw` 用于细粒度调度）。
- **校准**：C4 已有 58 个 export 的逐 dispatch 板上周期（`SA_PROFILE=1`）。再加一组微基准列表，放进 `board_profile` 的同一套流程测。目标是每个 dispatch 误差 ±10%。
- 估计值写进 sa-desc 入口点表的“估计周期”字段，驱动以后可以用来选宿主 dispatch。

### 8.6 验证

| 层次 | 做法 |
|---|---|
| pass 单元测试 | `compiler/plugins/sa/test/`：`iree-opt` 运行单个 pass + lit / FileCheck（需要构建 FileCheck 目标） |
| 逐 dispatch | `dispatch_check.py` 不变（oracle 执行 IR 语义 vs funcsim 运行生成的列表，每个调用点）：这是主要的正确性保障 |
| 新旧对照 | 开关 `--iree-sa-codegen=templates|dialect`，两条路径并存到 C5.4：同一个 dispatch 的结果都要逐位一致，估计周期可以互相对照 |
| 微内核对照 | 同一个 dispatch 用微内核与 `--iree-sa-ukernels=none` 各编一次：结果逐位一致，周期对照（微内核的价值要有数据） |
| 端到端 | `test_c3.py`（sim）、`board_llm.py`、`board_profile.py`；新增 `compare_profiles.py`：两份逐 dispatch 周期表逐项对比 |
| 变异测试 | 故意改错 VE 折叠规则、下标模式、循环次数，确认测试能检出 |

### 8.7 分步计划（纵向切片，每步 sim → 板上 → 提交）

| 步骤 | 内容 | 验收 | 工作量 |
|---|---|---|---|
| **C5.0 基础设施** | `sahw` 方言（ODS / TableGen）、verifier、序列化器、`--iree-sa-codegen` 开关、流水线骨架、FileCheck；现有代码生成器改为输出 `sahw` 操作，然后由新的 `sahw-assign-registers` 和序列化器完成（prefix / head 暂由现有代码给出）；**目标配置**（§8.9 第 1 项）：`#hal.executable.target` 的 config 带全部硬件参数，新代码只从这里读 | 58 个 dispatch 的 sa-desc 与 C4 **逐字节相同**（先验证后半条流水线） | 中 |
| **C5.1 逐元素与归约** | `sahl` 方言；流水线：normalize、layout（packed / 广播字）、缓冲化、`sa-to-sahl`、`sahl-uniform`、`sahl-plan-memory`、`sahl-to-sahw`、`sahw-fuse-ve`、`sahw-legalize-dynamic`、`sahw-allocate`、`sahw-schedule`；funcsim 与 `dispatch_check` 支持目标配置的参数（D、SPAD / ACC 大小） | 约 40 个非 contraction 的 dispatch 用新路径全部通过 `dispatch_check`（D = 8，并用 D = 16 等配置交叉验证）；混合路径端到端 sim 逐位一致；逐 dispatch 周期不高于 C4 | 大 |
| **C5.2 线性层** | contraction 分块（N，以及 K 放不下时按 K 分块、跨块累加）、A / B 条带、尾部按块融合、软件流水（`sahl-pipeline`）、`sahw.loop`、prefix / head（`sahl-schedule`） | 5 种线性层通过；另加 K = 8192 等单独的线性层测试（覆盖 K 分块）；板上逐 dispatch 周期与 C4 相差 ≤ 1%（模板的调度就是参照） | 大 |
| **C5.3 注意力、scatter、gather、i64 标量** | batch_matmul（Kᵀ 条带、TRANSPOSE）、动态 T 的行循环、带掩码的 softmax、KV 写入（LDPARAM）、嵌入 gather；GQA 的下标映射（KV 头 = Q 头 / 组大小）按设计实现 | 58 个 dispatch 全部走新路径；GQA 用单独的注意力测试覆盖 | 大 |
| **C5.4 模板改成微内核** | 线性层、注意力的模板改写成输出 `sahl` 操作的微内核（§8.8），加 `--iree-sa-ukernels`；旧的直接生成描述符的路径与 `--iree-sa-codegen` 开关退役；板上验收 | §8.1 的 1、2（两种配置） | 中 |
| **C5.5 通用性** | 下面两个模型；前端（HuggingFace 导入、通用量化、QK-norm 等结构）、运行时（设备窗口可配置、分配器到 GB 级）、按 torch 误差验收的端到端测试 | §8.1 的 3；SmolLM2-135M 上板 | 中 |

前端（§8.9 第 2 项）是独立的 Python 工作，与代码生成无关，可以在 C5.1–C5.3 期间穿插先做：先用现有的模板路径试编 SmolLM2 / Qwen3，看会暴露哪些不支持的 dispatch，反过来检查 C5 的覆盖面。

C5.5 的模型（§8.9 的第一步）：
- **SmolLM2-135M**：Llama 结构，GQA（9 个 Q 头、3 个 KV 头）、dim 576、hidden 1536、30 层、词表 49152、嵌入与输出层共享。int8 约 135 MB，放得进 PYNQ-Z1。检验 HuggingFace 导入、通用量化、GQA，以及不同的形状；目标是板上运行。
- **Qwen3 结构的小配置**：QK-norm、GQA、head_dim 128，维度取小（随机或截断的权重）。只在 sim 上验收，检验 Qwen3 特有的算子。
- 验收（§8.1 第 3 条）：`--iree-sa-ukernels=none` 编译通过；逐 dispatch 与 oracle 逐位一致；端到端与 torch fp32 在误差范围内，生成质量正常（这些模型没有手写的 DeviceModel）。

**C5.0 结果（2026-09-27）**：完成。
- `sahw` 方言（`plugins/sa/dialect/SahwOps.td`）：`sahw.template` 与 prefix / head / body 三段、寄存器（`sahw.base`、`sahw.param`、`sahw.private`，`!sahw.base` / `!sahw.param` 类型）、全部命令（ld、st、ex、ve、transpose、ldparam、fence、setreg、loop）。
- 做法与方案的差别：没有改写现有代码生成器，而是把它的输出（描述符行 + 装载表）逐字段**提升**成 `sahw`（`transforms/SahwRaise.cpp`，pass `iree-sahw-legacy-codegen`；LOOP_END → `sahw.loop` 区域，寄存器号 → 值）。旧生成器在 C5.1–C5.3 中被逐步替换，提升只是过渡。
- 新的 pass：`iree-sahw-split-head`、`iree-sahw-assign-registers`；序列化器 `transforms/SahwSerialize.cpp`（sahw → sa-desc v3，含读写掩码、SPAD_B、前缀寄存器）。代码生成移到翻译流水线里，`serializeExecutable` 只序列化。
- 目标配置：`#hal.executable.target` 的 config 带 `d`、`spad_bytes`、`acc_bytes`、`max_dynamic`、`dyn_fields`、`bases`、`params`（`transforms/SahwPasses.h` 的 `TargetConfig`）；`Layout` 的容量改为从配置读取。
- 开关 `--iree-sa-codegen=dialect|templates`（默认 dialect）。
- 验收（`tests/test_c50.py`）：lit 测试（`plugins/sa/test/`：方言往返、两个 pass）通过；stories15M 的 58 个可执行体两条路径**逐字节相同**；C2（D = 8、16）、C3（58 个 dispatch、12/12 步逐位一致）在新路径上通过。可执行体与板上验证过的 C4 相同，不需要重新上板。

**C5.1 结果（2026-09-28）**：逐元素与归约类 dispatch 全部走新流水线。
- 流水线：IREE 的 comprehensive bufferize（§13 第 6 条：直接可用）→ `iree-sa-to-sahl`（DDR 读写显式化：计算只读写片上缓冲，`sahl.load` / `sahl.store`；i64 标量留在 DDR）→ `iree-sahl-to-sahw` → `iree-sahw-fuse-ve`。
- `sahl-to-sahw`：片上缓冲分配；DDR 视图 → BASE + 偏移；动态长度 → PARAM（来自 push constant），[H, T] 缓冲按行排列、DMA 逐行（PARAM 累加行偏移）；每个 arith / math 运算一条单级 VE；只依赖标量 / 行的值只算一次；归约（含 abs-max 折半树）；int8 / int32 输出；读取（单元素、按位置的整行、带取负的成对交换 → SWAPNEG），读出的行可直接作为结果；i64 标量运算；前缀掩码 → VALID（LDPARAM 读位置）；动态 T 的 generic 逐行降级（LEN 为动态字段）。
- `sahw-fuse-ve`：按 C3 的规则合并单级 VE（A、B、FUNC、VALID、REDUCE、输出转换依次），动态 LEN 相同才合并。
- 计算仍用片上 memref 上的 `linalg.generic` 表示，`sahl` 目前只有 load / store 两个操作。
- 驱动：每个 dispatch 先试新流水线，失败则用 C3/C4 生成器（`--iree-sa-new-codegen=on|only|off`，`--iree-sa-codegen-report`）。
- 结果（`tests/test_c51.py`）：stories15M 的 58 个可执行体中 34 个走新流水线（逐元素与归约类全部），58 个全部通过 `dispatch_check`，端到端 sim 逐位一致；描述符 / VE 数与 C3/C4 相同，dispatch 5、30 各少一条 VE（REDUCE 合并进了折半的最后一步）。C5.1a 的 23 个在板上逐 dispatch 周期不高于 C4，整体 2,590,630 周期 / token。
- 其余 24 个：线性层 5 个（C5.2）、注意力的 batch_matmul 12 个与 KV 写入的 scatter 7 个（C5.3）。

**C5.2（2026-09-28）**：线性层走新流水线。
- 在 `sahl` 层识别“int8 收缩 + 逐元素尾部”：x（i8 或 i32 中的 int8 值）× 打包的 W（i8 [N/D, K, D]）→ int32 累加器 → 一个 [N/D, D] 的逐元素尾部 → 写回。收缩与尾部的全部缓冲只能由这个结构使用。
- 降级：分块（块大小按 SPAD_B bank 与 ACC 暂存区，容量来自目标配置）；A 条带由 x 做 DIV-D 复制；第 i 块 EX 之后预取第 i+1 块的权重和尾部的逐块输入（另一个 bank）；尾部按块走 C5.1 的逐元素降级（任意逐元素运算，不再限于模板的几种形式），临时缓冲放在本块的 bank；块数多时用 LOOP_END 循环一对块；第 0 块权重放进前缀。
- 结果：5 种线性层全部通过 `dispatch_check`；命令的种类、顺序与数量与模板相同（分类层的循环结构一致），差别只在尾部中间结果的位置、残差随下一块预取、寄存器编号。
- 调试工具：`compiler/runtime/tools/sadis.py`（sa-desc 反汇编）。
- 板上：逐 dispatch 周期与模板相同或略少（33：−1,071；22：−582，残差预取），整体 2,588,992 周期 / token。**C5.2 验收通过。**

**C5.3（2026-09-28）**：注意力与 KV 写入走新流水线，stories15M 的 58 个可执行体全部由 C5 流水线生成（`--iree-sa-new-codegen=only` 可以编译）。
- scatter（KV 写入）：LDPARAM（位置 × 行字节数）+ 带动态 DDR 地址的 ST；“把 cache 拷给自己”的写回在 `sa-to-sahl` 中删去；只有恒等映射的多层逐元素循环压平成一维。
- 注意力：`sa-to-sahl` 保留 batch_matmul 与缓存切片的扩展 generic；`sahl-to-sahw` 把“扩展 + batch_matmul + 尾部 + 写回”作为一个结构，按 C3 的方式逐头降级（分数：K 行 → TRANSPOSE → Kᵀ 块、q 复制成 A 条带、EX、`×s_q[h]×a_k`；P·V：p 复制成 A 条带、V 按 INTERLEAVE 加载、EX、`×a_v`；T 动态）。这实际上是把注意力模板移植到了新流水线，C5.4 中成为 `attention` 微内核；batch_matmul 的通用降级（Kᵀ 布局由布局传播插入）暂缓。
- 板上：58 个可执行体全部来自 C5 流水线，2,588,991 周期 / token，逐位一致。**C5.3 验收通过。**

**C5.4 结果（2026-09-28）**：C3/C4 生成器退役，模板改成微内核，没有微内核也能编译 stories15M。
- 退役：`Codegen.cpp`（C3/C4 的直接生成描述符的生成器）、`Templates`（参考实现的 DescList 移植）、C5.0 的提升层、`--iree-sa-codegen` / `--iree-sa-new-codegen`。代码生成只有 `iree-sa-codegen` 一条路径；编译不了的 dispatch 报错（`--iree-sa-allow-unsupported`：生成运行即报错的导出）。
- 微内核（`--iree-sa-ukernels=all|none|<列表>`）：`linear`（C5.2 的调过的线性层降级）、`attention`（C5.3 的注意力降级）。
- 通用收缩降级（没有微内核时）：每个操作数沿 `sahl.load` 与扩展 generic（含置换）追到 DDR，得到“元素偏移 = 各循环维 × 系数”，循环分成 batch / N / K；矩阵按布局变成 B 块（已打包：直接 LD；K 连续：逐行 LD + TRANSPOSE；N 连续：INTERLEAVE LD），x 复制成 A 条带；按 batch、按输出块：B 块、EX、尾部（逐元素降级，恒等映射的输入逐块加载）、写回；动态 N / K 用 PARAM；串行调度（不预取、不用 LOOP_END）。块用过的片上缓冲在块结束后收回，因此尾部的广播字须在分块循环之前生成（板上第一次运行在 T ≥ 80 时发现的错误）。
- 验收：`--iree-sa-ukernels=all` 与 `none` 都编译出 58 个可执行体，`tests/test_c5.py` 在 T = 16、80、256 下全部通过 `dispatch_check`，sim 76 步逐位一致；板上两种都与 L5 文本相同、逐位一致。
- 微内核的价值（板上，周期 / token）：带微内核 2,588,975；只用通用降级 4,395,723（10.3 tok/s）。差别几乎都在线性层（没有预取、没有循环）：分类层 1.26M → 2.43M，W13 0.41M → 0.70M，Wqkv 0.26M → 0.40M，W2 0.23M → 0.35M；注意力相差很小。
- 测试：`tests/test_c5.py`（lit、两种配置、T 扫描、微内核改变了哪些 dispatch）；`test_c2.py` 改为检查命令的种类与数量与参考调度相同（不再逐字节）；`test_c50.py`、`test_c51.py` 删除。

**C5.5 结果（2026-09-28）**：两个没有写过专用代码的模型通过；SmolLM2-135M 板上通过。
- 前端：`frontend/qhf.py`（HuggingFace 配置、不依赖第三方包的 safetensors 读取（BF16 → f32）、字节级 BPE 分词、fp32 参考模型、KV 标定、量化模型 `QModel`；rotate-half RoPE 改成交错排列的行置换；GQA 写成 batch = KV 头、行 = 组内 Q 头；Qwen3 的 QK-norm），`frontend/export_hf.py`（导出，`--layers` 截断层数）。
- 验收的参照：设备的 EXP / RECIP / RSQRT 是硬件近似，torch 精确函数作参照时第一步相关系数只有 0.9875。`qhf.device_sfu()` 让 eager `QModel` 用设备的近似（`llm/ref_model.py` 的 `SfuExact`）：第一步与 sim 只差求和顺序（相对误差 2e-5）；之后求和顺序偶尔翻转一个 int8 舍入并逐层放大，按 argmax 相同、相关系数 > 0.99 验收（`tests/test_c55.py`）。板上与 sim 逐位比较。
- 运行时：设备窗口可配置（sim `--mb`、板上 `board.txt`）；`sa_device_queue_read` 直接把文件读进设备缓冲（原来的流式路径先分配一个和传输一样大的暂存缓冲，最大的参数要放两份：SmolLM2 峰值 184.5 → 160.5 MB）；`sa-llm-run` 打印设备内存峰值，arena 内存不足的错误带当前用量。
- 编译器：动态 T 的逐行降级在每行之后收回该行的临时缓冲（Qwen3 的 softmax 有 16 行，原来 ACC 用满）。
- SmolLM2-135M（30 层，GQA 9/3，dim 576，词表 49152）：230 个 dispatch，T = 16 / 80 / 256 全部与 oracle 逐位一致；sim 端到端 16 步 argmax 16/16、最小相关 0.9922，生成 “Once upon a time, there was a little girl named Lily. She lived in a big house with her family, and she loved to play with her toys. …”；窗口 168 MB（峰值 160.6 MB）；sim 9 s / token。板上包：`scripts/deploy_c55.sh`（`tests/board_llm.py`：C3 与 C5.5 共用，按 sim 的结果逐位比较）。
- **板上**：38 步 logits 与 sim 逐位一致，生成的 token 相同，1.90 tok/s（526 ms / token）。PYNQ-Z1 的 CMA 默认 128 MB，放不下窗口：启动分区加 `uEnv.txt`（`bootargs=<原命令行> cma=320M`，`boot.scr` 会导入它）；256M 时加载 overlay 后 CMA 碎片化，找不到 168 MB 的连续块（内核没有 compaction）。窗口分配失败时启动器清页缓存后重试；运行前停掉 Jupyter。
- **重新打包与交互式生成（2026-09-28）**：用最新的编译器（K 分块、分片之后）重新打包，板上再验：stories15M 76/76 步与 DeviceModel 逐位一致，17.75 tok/s；SmolLM2 38/38 步与 sim 逐位一致，2.23 tok/s（包里加了 `--pad=8`，注意力长度随位置增长，原来按 256 算是 1.90）。`tests/board_generate.py`：overlay 启动一次，之后逐行读 prompt，在 ARM 上分词（`frontend/hf_tokenizer.py`，只用标准库），`sa-llm-run --stop_token` 生成，token 边出边打印；两个包都带 `model.json`（分词器、BOS、结束符、上下文长度）。每个 prompt 重新加载模块与参数：SmolLM2 第一个 token 前约 11 s，stories15M 0.4–1.7 s。窗口要在加载分词器之前分配：SmolLM2 的分词器在 Python 里占几十 MB，CMA 中借给内核的页迁不出去，168 MB 的连续块就分不到。
- Qwen3-0.6B 截断到 2 层（QK-norm、head_dim 128、q_dim 2048 ≠ dim 1024、GQA 16/8、θ = 1e6、词表 151936）：42 个 dispatch 全部逐位一致；sim 8 步 argmax 8/8、相关 1.00000；峰值 330.5 MB。
- 发现的问题（留给 C6）：共享的嵌入存了两份（分类层的打包权重在参数文件里，gather 用的原始表作为常量在 vmfb 里）：SmolLM2 多 28 MB，Qwen3 多 155 MB；可以让 gather 直接读打包布局，或把原始表也导出成参数。

### 8.8 扩展点与微内核

**微内核**（ukernel）：
- 一个微内核 = 一个匹配条件 + 一个 C++ 函数，把匹配到的 linalg 算子（带它的尾部）改写成一段 `sahl` 操作（需要时也可以直接给出 `sahw` 片段）。它**不**直接输出描述符，之后照常经过内存规划（或声明自己占用的片上区域）、调度、`sahl-to-sahw`、寄存器分配、序列化，所以同样受 verifier、`dispatch_check`、驱动的前缀预取与 FENCE 规则约束。
- 选择在 `sa-to-sahl`：对每个算子先查微内核表，匹配就用，否则走通用降级；两者都可用时由代价模型或开关决定。开关 `--iree-sa-ukernels=all|none|<名字,…>`。
- 初始的微内核：`linear`（`emitLinear`：分块、双缓冲、逐块尾部，C2 起的字节比对测试保留）、`attention`（scores / P·V）。以后值得手工调的算子（例如 C4 续的 W13 + SiLU 交错循环）只需加一个微内核。
- 每个微内核要有对照数据：与通用路径逐位一致，并记录周期差。

**扩展点**：代码生成是一条 MLIR pass 流水线，每个 pass 都在插件里注册，可以用 `iree-opt` 单独运行与测试。新的优化 pass 放在以下位置之一，用编译选项开关，便于 A/B 测量：

| 扩展点 | 作用的 IR | 适合的优化 | 已有的例子 |
|---|---|---|---|
| 预处理（全局，dispatch 形成之前） | 整个模型的 linalg | 控制融合、自行组 dispatch region、常量打包 | `iree-sa-pack-linear-weights`、`iree-sa-clone-cheap-producers` |
| linalg 层（`sa-tile` 前后） | 单个 dispatch 的 linalg / scf | 分块策略、算子改写、微内核选择 | — |
| `sahl` 层 | tile 级运算 | 内存规划、流水、粗粒度调度、预取、tile 级改写 | — |
| `sahw` 层 | 命令 | 细粒度调度、窥孔优化、VE 级合并 | abs-max 折半树（C4 步骤 6，C5 中移到 `sahl-to-sahw`） |
| 驱动（运行时） | 一个命令缓冲的整张列表 | 跨 dispatch 调度 | C4 步骤 5 的列表调度 |

- 实验时不必改流水线：`iree-opt --pass-pipeline=...` 可以在任一阶段导出的 IR 上直接试。
- 可选：接入 IREE 的 transform dialect 脚本，按算子指定分块、融合策略，不必重新编译编译器。

### 8.9 扩展性：更大的模型与开发板

**目标**：不要求在 PYNQ-Z1 上运行 Qwen3、Llama3，但以后换了资源更多的开发板，编译器应当不改结构就能编译这些模型。所以编译器不能把 PYNQ-Z1 或 stories15M 的任何数字写死。

**PYNQ-Z1 的限制**（硬件，不是编译器的问题）：DDR 512 MB，与 Linux 共用；权重带宽约 390 MB/s。

| 模型 | 参数 | int8 权重 | PYNQ-Z1 |
|---|---|---|---|
| stories15M / 110M | 15M / 110M | 16 / 115 MB | 能（110M 需要扩大 CMA 窗口，约 3 tok/s） |
| SmolLM2-135M（Llama 结构） | 135M | 约 135 MB | 能（约 2.5 tok/s） |
| Qwen3-0.6B | 约 0.6B | 约 600 MB | 不能：超过整块 DDR |
| Llama-3.2-1B | 1.24B | 约 1.2 GB | 不能 |

描述符的 DDR 地址与 BASE 寄存器是 32 位，所以即使换板子，一个模型的全部数据也要在 4 GB 物理地址之内；更大需要改硬件。

**1. 目标配置参数化**：硬件参数全部来自 `#hal.executable.target<"sa", …, {config}>`（已有 `d`），编译器里没有常数：
- D、SPAD / ACC 大小与 bank 数、每条描述符的动态字段数、BASE / PARAM 个数、CAPS（FPVE、以后的 int4 解包等）；
- 代价模型的参数（带宽、各引擎的延迟、多拍模式的代价）；
- 驱动从设备读 D 与 CAPS，与可执行体比对（已有 D 的检查）。
验证：同一个模型用不同的配置（D = 8 / 16、更大的 SPAD）编译，在按同样参数配置的 funcsim 上逐 dispatch 验证（funcsim 需要支持这些参数）。

**2. 前端**：
- 从 HuggingFace 模型导入（transformers → torch → iree-turbine），不再依赖手写的 `qllama.py`；`qllama.py` 保留为 stories 系列的参考。
- 通用的量化改写：把 `nn.Linear` 换成量化线性层（W8A8，按输出通道的权重 scale）；可选 SmoothQuant 一类的离群值处理（只是把 scale 折进前后的算子，不改硬件）。硬件的 EX 只有 int8 × int8，更低位的权重（int4）需要硬件支持。
- 结构特性：GQA、QK-norm（Qwen3）、RoPE 的变体（theta、Llama3 的频率缩放：预先算成表）、嵌入与输出层共享、SwiGLU / GELU、LayerNorm。
- 长上下文：`max-dynamic` 与 KV cache 大小由模型配置决定，不是固定的 256。

**3. 代码生成**（C5 的通用部分覆盖）：
- K 维分块并跨块累加（Llama3-1B 的 W2：K = 8192 正好是一个 SPAD_B bank）；
- 注意力按 T 分块（长序列的 K / V 放不进片上）：在线 softmax 由预处理显式写进 IR（§3.2），代码生成只按块执行；
- GQA：KV 头的下标 = Q 头 / 组大小，用 DIV 下标模式或循环表示；
- 大词表的分类层（128K–152K 行）：只是更多的块；
- 一个命令缓冲的 dispatch 很多时（28 层约 900 个），驱动已会拆成多张列表。

**4. 运行时**：设备窗口大小由启动配置决定（现在固定 64 MB）；arena 分配器支持几百 MB 到几 GB；参数文件按需加载进窗口。

**5. 验证方法随规模调整**：
- 大模型没有手写的 DeviceModel：逐 dispatch 仍与 oracle 逐位一致（oracle 按 IR 执行，与模型无关）；端到端与 torch fp32 比较误差，再看生成质量（top-1 一致率、困惑度）。
- Python funcsim 跑 stories15M 约 1.4 s / token，整个 0.6B 模型在 sim 上不现实：用截断的层数（像 `QLlama(n_layers)`）做端到端检查，逐 dispatch 检查照常覆盖全部 dispatch；需要时把 funcsim 移植到 C（§13 第 2 条）。

**6. prefill 与批处理**（可选）：批量 prefill 是 M > 1 的矩阵乘，能用满阵列；权重打包 pass 现在只认 vecmat，要推广到 matmul。

**各项的时间表**（原则：越晚改代价越大的，写 C5 的 pass 时就做进去）：

| # | 项目 | 何时做 | 理由 |
|---|---|---|---|
| 1 | 目标配置参数化 | C5.0 起贯穿 C5；funcsim 的参数化与交叉验证：**补做于 C5.5 之后**（见下） | 新 pass 从第一行起不写常数，几乎没有额外成本；事后再改要翻遍所有 pass |
| 3a | K 维分块、跨块累加 | 原排 C5.2，实际没做；**补做于 C5.5 之后**（见下） | 通用 contraction 分块的一部分；stories15M 用不到，用合成模型测试覆盖 |
| 3b | GQA 下标映射 | C5.3 实现，C5.5 首次用上 | 注意力降级时按组映射设计，SmolLM2 需要 |
| 2 | 前端：HuggingFace 导入、通用量化、QK-norm 等结构 | C5.5（可在 C5.1–C5.3 期间穿插先做） | 与代码生成无关；SmolLM2 与 Qwen3 小配置需要 |
| 4 | 运行时：设备窗口可配置、分配器到 GB 级 | C5.5 | SmolLM2-135M 的权重约 135 MB，现在的 64 MB 窗口放不下；改动小 |
| 5 | 验证：与 torch 比误差、截断层数 | C5.5 起；C funcsim 在 C6 视需要 | 新模型没有手写的 DeviceModel；Python sim 只在大模型上才太慢 |
| 3c | 注意力按 T 分块（在线 softmax） | C6 | 涉及数值约定（预处理显式改写 IR、oracle 相应支持）；seq ≤ 2048 时是否需要，C5.5 前核实片上容量 |
| 3d | 大词表、多张列表 | 已具备，C6 验证 | 不需要新代码 |
| 6 | 批量 prefill | 提前到 C6.1 之后：**已完成**（C6.P，§8.13） | 只影响速度，不影响能否编译 |

**补做（2026-09-28）**：C5.5 之后核对发现，第 1 项的交叉验证与第 3a 项其实没做，补上：
- **目标配置交叉验证**：funcsim 的 SPAD / ACC 大小可配置（`SaFuncSim(d, …, spad_bytes, acc_bytes)`，片上地址是 16 位字地址，每块最多 2^16 字）；编译器加 `--iree-sa-spad-kb` / `--iree-sa-acc-kb`，目标配置不合法时报错（`TargetConfig::invalid`）；`dispatch_check.py` 从 dispatch 的 target 读 D 与存储大小，sim 服务加 `--spad-kb` / `--acc-kb`。`tests/test_c5.py --configs`：stories15M 在 D = 16、SPAD 256 KB / ACC 512 KB、D = 16 + SPAD 512 KB / ACC 1 MB 三种配置下，带与不带微内核都编译出 58 个可执行体，T = 16 / 256 下全部与 oracle 逐位一致。（第一次运行时 dispatch_check 的正则没匹配上 target，结果全按默认配置检查，报了一片假错误；修正后全部通过。）
- **K 分块**：通用收缩降级在 K 超过容量时按 K 块执行（块内：该块的 B 块、x 对应段复制成 A 条带、EX；第一块写累加器，之后累加，尾部在全部块之后）。块长受三个限制：一个 SPAD bank；一条 DMA 的行长（16 位：块长 × D ≤ 65535，D = 8 时 8184）；A 条带复制用的 DIV 下标模式的范围检查（硬件没有除法器，按 G 字检查，所以 x 的位置 + K/D + 块长不能超出 x 所在存储）。线性层微内核在 K 超出它的容量时不匹配，交给通用降级。Llama-3.2-1B 的 W2（K = 8192）原来就会超出 DMA 行长，现在按 2 块执行。
- 同时发现并补上的两个容量问题：大于 64 KB 的连续 DMA 拆成“行 × 行长”加尾部（`contiguousDma`）；元素个数超出 ACC 的逐元素 dispatch 分片执行（piece mode：每片重跑整个 dispatch 体，DDR 偏移与本地缓冲按片，片后释放；对整个向量的 max 归约（含 abs-max）逐片合并，max 与顺序无关所以仍逐位一致；求和逐片合并会改变求和顺序，暂不支持）。
- 验证：`frontend/synth_hf.py` 生成随机权重的 HuggingFace 目录（dim 256、hidden 16384、1 层，SmolLM2 的分词器）：26 个 dispatch 在 T = 16 / 80 / 256 下全部逐位一致，sim 端到端第一步误差 0、argmax 8/8、相关 1.0。回归：C3（stories15M 与 DeviceModel 逐位一致）、C5（两种配置 + 三种硬件配置）、SmolLM2（230 个 dispatch）全部通过。

**阶段**：
- **C5**：第 1、3a、3b 项（§8.7 的 C5.0–C5.3）。
- **C5.5**（§8.7）：第 2、4、5 项；SmolLM2-135M（板上）与 Qwen3 结构的小配置（sim）。
- **C6 更大的模型**：第 3c 项，第 5 项的 C funcsim（视需要）；Qwen3-0.6B、Llama-3.2-1B 在 sim 上编译并逐 dispatch 验证（截断层数做端到端）；换开发板后在板上运行。之后可选第 6 项。分步计划见 §8.12。

### 8.10 C5 之后

- **C4 续**：在 C5 的调度器和代价模型上做暂缓的融合。W13 + SiLU 由预处理组成一个 dispatch region，模板内交错；量化链合并成一个 dispatch；`enable-aggressive-reshape-movement`。目标仍是与手写路径相差 ≤ 10%。
- 可选扩展（§10）：批处理 decode、prefill、更大的模型。
- **C7 RISC-V 后端**（可选，C5 之后任何时候，不影响 C5 / C6 主线）：
  - 同一组加速器命令有两种发出方式：ARM 构建描述符列表由取指单元执行，或 PicoRV32 用 PCPI 自定义指令（custom-0，funct7 = 1 矩阵 / DMA、funct7 = 2 向量）逐条发出。两者的命令字段基本相同，所以 `sahw` 可以接第二个后端：`sahw` + scf / arith → LLVM dialect（命令用 `.insn` 内联汇编）→ LLVM IR → riscv32 目标文件。
  - 价值是灵活性：完整的控制流和标量计算（依赖数据的循环与分支、设备上的 argmax / top-k 采样），没有“每条最多 2 个动态字段”的限制；也为以后配有更强 RISC-V 核的硬件做准备。
  - 加速器做不了、要在 RISC-V 上算的通用运算，不走这条路径，而是复用 IREE 的 llvm-cpu 后端（linalg → vector → LLVM，支持 riscv32 / riscv64）。
  - PYNQ-Z1 上的限制：PicoRV32 是 rv32imc，程序存储器只有 8 KB BRAM，只能放小内核（能否从 DDR 取指需要核实）；逐条发 PCPI 比描述符 DMA 慢（M5：小算子 1.2–4.2 倍），所以在这块板上不是提速手段。
  - 前提：现在的 iree-compile 没有带 llvm-cpu，LLVM 的 RISC-V 目标可能没编进去，需要核实，必要时重新构建编译器（约 2 小时）。
  - 验收：一个 dispatch 由 PicoRV32 代码发 PCPI 指令执行，结果与描述符路径逐位一致；另用 IREE llvm-cpu 编一个 riscv32 的 dispatch 作为演示。

### 8.11 风险与对策

| 风险 | 对策 |
|---|---|
| IREE 的缓冲化对自定义内存空间和布局不顺手 | 退回自己的缓冲化：dispatch 的结构简单（load → 片上 → store），C3 已有同样的分析 |
| 通用降级达不到模板的速度，尤其是线性层的块大小和流水 | C5.2 以逐 dispatch 周期对照模板验收；代价模型选块大小；模板的调度作为参照写进 FileCheck 测试 |
| 描述符约束（每条最多 2 个动态字段、下标模式按整字、VE 源与目的的部分重叠无定义） | verifier 检查，加上专门的合法化 pass；违反时报错，而不是生成错误的列表 |
| 逐位一致被破坏（折叠改变了舍入、归约顺序变了） | 折叠规则沿用 C3，只合并不改变舍入的运算；求和归约的顺序固定为硬件顺序（只有 max 可以重排，如折半树）；每一步都跑 `dispatch_check` |
| 工作量大（MLIR C++） | 纵向切片，两条路径并存到 C5.4，任何时候模型都能完整运行；插件只需增量构建（几分钟） |
| 微内核掩盖通用路径的缺陷 | 验收同时要求 `--iree-sa-ukernels=none`；每个微内核都要有周期对照数据 |


---

### 8.12 C6 详细方案：更大的模型

**目标**：换了资源更多的开发板以后，编译器不改结构就能编译 Qwen3-0.6B、Llama-3.2-1B 这一级的模型。PYNQ-Z1 的 512 MB DDR 放不下它们，所以在这块板上只做 sim 验收：全部 dispatch 与 oracle 逐位一致，端到端（必要时截断层数）按 argmax 与相关系数和 eager 模型（设备 SFU）比较，生成质量另用 fp32 参照衡量。

**C5.5 之后核对的现状**：
- 已具备：HuggingFace 导入（Llama、Qwen3 结构）、GQA、QK-norm、嵌入共享、设备窗口可配置、按 torch 误差的端到端测试、截断层数、合成模型（`frontend/synth_hf.py`）。
- 计划里写着已做、实际没做的两项已补上（§8.9“补做”）：目标配置交叉验证、K 分块；顺带补上了大于 64 KB 的 DMA 与超出 ACC 的逐元素 dispatch（分片）。
- 已知的容量限制：注意力的 K 行要放进一个 SPAD bank（D = 8、128 KB SPAD：head_dim 128 时 T ≤ 512，64 时 T ≤ 1024）；分片的逐元素 dispatch 不支持整个向量的求和（会改变求和顺序）。

| 步骤 | 内容 | 验收 | 状态 |
|---|---|---|---|
| **C6.0 清理** | 共享的嵌入存了两份（分类层的打包权重在参数文件里，gather 用的原始表作为常量在 vmfb 里）：gather 直接读打包布局，或把原始表也导出成参数 | SmolLM2 板上仍逐位一致，窗口变小（Qwen3 省 155 MB，SmolLM2 省 28 MB） | **完成**（见下） |
| **C6.1 Qwen3-0.6B 全 28 层** | 导出、编译、逐 dispatch 检查、sim 端到端几步（估计每 token 约 40 s） | 全部 dispatch 逐位一致；argmax 与 eager 模型（设备 SFU）相同、相关 > 0.98；生成的文本通顺 | **完成**（见下） |
| **C6.2 量化质量** | 用 fp32 参照测 top-1 一致率与困惑度；不够再加 SmoothQuant 一类的离群值处理（scale 折进相邻算子，硬件不变） | 与 fp32 的 top-1、困惑度在可接受范围内 | |
| **C6.3 K 分块** | 通用收缩按 K 块累加；线性层微内核放不下时交给通用降级 | 合成模型 K = 16384 逐 dispatch 与端到端逐位一致 | **已完成**（§8.9 补做） |
| **C6.4 Llama-3.2-1B** | 前端：Llama3 的 RoPE 频率缩放（预先算成表；现在遇到会报错），词表 128K，GQA 32/8；权重需要 Meta 授权（或用公开镜像） | 同 C6.1 | |
| **C6.5 注意力按 T 分块** | 在线 softmax 由预处理显式写进 IR（§3.2），oracle 相应支持；KV cache 长度与 `max-dynamic` 由模型配置决定 | T = 2048 时逐 dispatch 一致 | |
| **C6.6 目标配置交叉验证** | funcsim 按目标配置参数化；同一模型用不同的 D、SPAD / ACC 编译并验证 | `test_c5.py --configs` 全部逐位一致 | **已完成**（§8.9 补做） |
| C6.7（可选） | 批量 prefill：M > 1 的矩阵乘；权重打包从向量×矩阵推广到矩阵乘 | 与逐 token decode 结果一致 | **已完成**，作为 C6.P（§8.13） |

**C6.0 结果（2026-09-28）**：gather 读打包布局（反过来让分类层读原始布局，要每 token 多一次整表 TRANSPOSE，stories15M 约 +40% 周期）。
- 预处理（`target/PackLinearWeights.cpp`）：每个常量权重只打包一次（放在它的定义之后）；它的 gather（`tensor.extract W[r, c]`）改成读打包的副本 `Wp[r / D, c, r % D]`，原始表没有读者，不再进 vmfb。条件：K × D ≤ 65535（LDPARAM 的 16 位乘数、一条 DMA 行）。
- 降级（`sahl-to-sahw` 的 `packedRowGather`）：命令处理器不会除法，r / D 与 r % D 由 VE 算（fp32 精确：`round(r / D − (D − 1) / 2D)`），经本 dispatch 结果的 DDR（最后被结果覆盖）写出、FENCE、LDPARAM 读成 PARAM；按 PARAM 加载 tile（K 个字）；one-hot 字由 VALID 做（两字前缀之差 `prefix(D + l + 1) − prefix(D + l)`，因为 VALID = 0 表示全部有效）；tile 乘 one-hot 后每字 REDUCE（只有一个非零 lane，求和精确），`packBcast` 把 K 个标量装回 K / D 个字。stories15M 的嵌入 dispatch 56 条描述符。
- 结果：vmfb stories15M 10.3 → 1.1 MB、SmolLM2 30.6 → 2.2 MB、Qwen3 157.3 → 1.7 MB；设备内存峰值 SmolLM2 160.6 → 133.6 MB（窗口 168 → 144 MB）、Qwen3（2 层）330.5 → 182.1 MB。stories15M 58 个 dispatch 与 sim 12/12 步与 DeviceModel 逐位一致；SmolLM2 230 个、Qwen3 42 个 dispatch 在 T = 16 / 80 / 256 下逐位一致，端到端通过；配置交叉检查（两种微内核设置 × 4 种硬件配置）全部通过。
- 板上：SmolLM2 38/38 步与 sim 逐位一致，2.22 tok/s，窗口 144 MB（仍要清页缓存重试一两次：加载 overlay 后 CMA 碎片化）；stories15M 76/76 步与 DeviceModel 逐位一致，17.65 tok/s（原 17.75：新的嵌入 gather 每 token 多约 56 条描述符）。
- 长时间测试：`compiler/scripts/run_tests.sh [-j 作业数] [-t 每个测试的线程数] 测试…`（每个测试一个日志；从 Claude Code 会话启动的进程会随会话结束，所以在自己的终端里运行）。

**C6.1 结果（2026-09-28）**：Qwen3-0.6B 全 28 层（dim 1024、hidden 3072、16 / 8 头、head_dim 128、QK-norm、词表 151936），没有改编译器：
- 导出 45.6 s（`qllama.irpa` 613 MB）；主机上 IREE llvm-cpu 与 eager 模型 argmax 8/8、最小相关 0.989（28 层的 int8 舍入翻转更多，端到端相关的门槛由 0.99 改为 0.98，第一步仍须 1e-4 以内）；QModel 与 fp32 的 top-1 只有 5/8（量化质量，C6.2）。
- sa 编译 70.6 s：276 个 dispatch，参数 610.7 MB；T = 16 / 80 / 256 下 276 个全部与 oracle 逐位一致。
- sim 端到端：第一步相对误差 2.1e-6，argmax 8/8 与 eager 模型（设备 SFU）相同，最小相关 0.9937，生成 “Once upon a time, there was a man”；设备内存峰值 587.4 MB（窗口 768 MB），sim 24 s / token。
- 测试的内存：oracle 原来一次性在整个迭代空间上求值，分类层（155M 点）的检查要 5 GB 以上，桌面的 systemd-oomd 在会话内存压力超过 50% 时连终端一起杀掉（此前几次“终端退出”都是它）；oracle 改为按最外层并行循环分片求值（归约的顺序不变），降到 2 GB；`run_tests.sh -m` 给每个测试一个有内存上限的 systemd scope。

**顺序**：C6.0 → C6.1 → **C6.P（prefill，§8.13，提前做：先完整编译一个 LLM）** → C6.2 → C6.4 → C6.5。C6.0–C6.2 只需 Qwen3、不需要新的代码生成，最快看到完整大模型的结果；C6.5 是剩下唯一的编译器新功能。Python 的 funcsim 每 token 的耗时与模型大小成正比（Qwen3-0.6B 估计约 40 s，Llama-1B 约 80 s），端到端只跑几步还可以接受，C 版 funcsim 等确实太慢再做。

### 8.13 C6.P prefill：完整编译一个 LLM（prefill + decode）

**目标**：一个模块里有 prefill 与 decode 两个函数，共用参数与 KV cache；prompt 按块做 prefill（权重读一次服务 M 个 token，D×D 阵列用满），之后逐 token decode。原计划的可选项（§8.9 第 6 项、C6.7），提到 C6.2 之前做。

**设计**：
- **固定块长 M**（导出参数；原定默认 16，板上对比后默认 8，见 P5）：长度 P 的 prompt 分 ⌈P / M⌉ 块，最后一块用填充 token 补齐；填充行写的 KV 在位置 ≥ P，decode 在读之前会覆盖、注意力也按位置屏蔽，所以无害。这样 T 仍是唯一的动态维（不碰“每条描述符最多 2 个动态字段”）。
- `prefill(tokens[M], start, valid[T])`：第 i 行的因果掩码是位置 ≤ start + i；一次写 M 行 KV；只算最后一行的 logits（分类层是最大的一层）。两个函数用 iree-turbine 的 `CompiledModule` 导出，KV cache 是共享的可变全局量。
- **验收的主要依据**：设备上 prefill 最后一个位置的 logits 与只用 decode 的路径逐位一致（线性层每行的运算相同：int32 累加精确、反量化逐元素；注意力在 T 的填充相同时求和顺序相同）。

| 步骤 | 内容 | 验收 |
|---|---|---|
| **P0 前端** | QModel 加 `prefill`（逐行因果掩码、M 行 KV、只输出最后一行的 logits）；`prefill`、`decode` 两个函数导出成一个模块 | eager 的 prefill + decode 与只用 decode 生成的 token 相同；IREE llvm-cpu 与 eager 一致 |
| **P1 线性层** | 打包 pass 也认 matmul（打包的权重与 decode 共用，C6.0 的“只打包一次”）；A 条带由 X 的 M 行经 TRANSPOSE 得到；微内核与通用降级 | 逐 dispatch 逐位一致；每块 EX 的效率 |
| **P2 注意力与其余** | M 行的分数 / P·V；softmax 的掩码按行（第 i 行有效长度 start + i + 1，LDPARAM 的加数）；RoPE 取 M 行；KV cache 一次写 M 行 | 逐 dispatch 逐位一致，T 扫描 |
| **P3 运行时** | `sa-llm-run` 先分块 prefill 再 decode；`board_generate.py` 用它 | sim：与只用 decode 的 token 相同，最后位置的 logits 逐位一致 |
| **P4 板上** | stories15M、SmolLM2；M = 8 / 16 / 32 对比 | 逐位一致；prompt 的 tok/s（预计比逐 token 快 5–10 倍）。**已完成**（见下）：M = 8 为默认；M = 16 每 token 只快 1.7%；M = 32 不做 |

**P0 结果（2026-09-28）**：`prefill(tokens[M], positions[M], valid[T])`（positions = start .. start + M − 1 由调用方给出：命令处理器不能把寄存器写回 DDR、VE 没有 iota，设备上算 i64 向量很别扭）；最后一块与前一块重叠而不是填充（从 P − M 开始，重算的行写回相同的 KV），最后一行总是 prompt 的最后一个 token。eager 的 prefill 与 decode 逐位一致（logits、KV cache）；IREE llvm-cpu 上 stories15M 的 prefill 与只用 decode 逐位一致，SmolLM2 相关 1.000000，之后生成的 token 相同。导出：`FxProgramsBuilder`，`main`（decode）与 `prefill` 共用参数与 KV 全局量。

**P1 / P2 的任务**（stories15M 的 prefill 在现有编译器上：decode 的 58 个可执行体全部可编，prefill 失败 47 个，19 类）：
1. 打包 pass 认 `matmul(X, Wᵀ)`，X 的 extsi 拉进 matmul（X 在 DDR 里保持 int8）；i64 累加器 + trunci 按 i32 累加（模 2³² 相同）。
2. 多行线性层微内核：EX 的 A 条带是“每 D 个 K 字一块、第 i 字是第 i 行的 D 个元素”，正是 DMA INTERLEAVE 装 D 行 X（int8）的布局，一条 LD 得到 D 行的条带；输出按 `crow` 逐行写；尾部 s_w 按列、s_x 按行。M = 16 时两块行共用一次权重加载。
3. i64 索引向量：gather / scatter / 掩码按行读 `positions[m]`、`tokens[m]`（每行一条 LDPARAM）；`向量 + 常数` 并进 LDPARAM 的加数。
4. 三层循环（头 × 行 × T）的 generic，按行的 VALID。
5. 多行的注意力收缩（每头 M 行的 batch_matmul）。
6. 按行跨步的 DDR 视图（`h13[:, :hidden]` 等：DMA 的行距）。
7. 零碎：i32 的 fill、对齐。
另：两个函数的 dispatch 源文件各自编号（`module_main$…_N`、`module_prefill$…_N`），`dispatch_check` 要按函数名配二进制。

**P1–P3 结果（2026-09-28）**：
- **做法的调整**：注意力、gather（嵌入、scale、RoPE 表）、scatter（KV 行）和 HF 模型的 QK-norm / RoPE 在前端逐行做，每行正是 decode 一步的形式（标量下标 `positions[m:m+1]`、`tokens[m:m+1]`），复用验证过的 decode 降级；批量做的是线性层（收益所在）与逐行的逐元素、归一化。上面第 3–5 项因此不用做，逐行带来的 dispatch 数量以后可以再合并。
- **编译器**：
  - 打包 pass 认 `matmul(X, Wᵀ)`（第 1 项）；一个权重全局量只打包一次：`W$packed` 全局量由 initializer 设置，IREE 编译时求值，prefill 与 decode、同一权重的每次 load 都用它（原来两个函数各打包一份，SmolLM2 的参数成了 272 MB）；打包全局量的 gather 按全局量名改读打包的副本（C6.0 在两个函数的模块里也成立）。
  - 多行线性层微内核 `linearRows`（第 2 项）：一条 INTERLEAVE LD 装 D 行 X 的 A 条带，D×D 阵列一次算 D 行；尾部按列（MOD nc）、按行（DIV nc）的输入；没有尾部时直接存累加器。串行调度，还没有预取。
  - 两层循环的合并可以在任意位置分开（行 [0, s)、内层 [s, n)）；按行跨步的 DDR 视图（第 6 项，只在 load / store）；4 字节对齐的 fp32 标量经 LDPARAM；循环内的标量 gather；打包 gather 的结果是精确的整数值（sitofp 直接通过）；广播布局的缓冲被逐元素结果复用时换成紧凑的新缓冲；**按行分片**：逐行的 dispatch 放不下 ACC 时按行分组（每组偶数行）。
  - `sa-llm-run --prefill=M`；`dispatch_check` 按函数名配二进制；`test_c6p.py`（两种运行的 logits 逐位比较）；`deploy_c6p.sh stories|smollm2`，板上 `board_llm.py [--decode-only]`、`board_generate.py` 用 prefill。
- **结果**（sim，M = 8）：
  - stories15M：181 个可执行体（decode + prefill）全部与 oracle 逐位一致；16 个 token 的 prompt 分 2 块 prefill，prompt 最后位置与之后 4 步的 logits 与只用 decode 逐位一致，token 相同；参数 16.1 MB（一份）。
  - SmolLM2-135M：777 个可执行体在 T = 16 / 80 / 256 下全部逐位一致；14 个 token 的 prompt，prefill + decode 与只用 decode 逐位一致（5/5 行），token 相同；参数 137.4 MB（一份）。
  - 主机 llvm-cpu 上 prefill 与只用 decode 只按相关比较：llvm-cpu 对 prefill 的形状生成的代码不同（向量化的求和），int8 舍入翻转会让接近的 argmax 变；sa 设备上要求逐位一致。
- **P4 板上（2026-09-28，M = 8）**：两个模型 prefill 与只用 decode 都与 sim 逐位一致，生成的文本相同。
  - stories15M：16 个 token 的 prompt 分 2 块，**258.6 ms（含加载）**；只用 decode 时 prompt 约 16 × 57 ms ≈ 915 ms（另加加载）：**至少 3.5 倍**。生成 17.4 tok/s（不变）。
  - SmolLM2-135M：14 个 token 分 2 块，**2.43 s（含加载）**；只用 decode 约 14 × 456 ms ≈ 6.4 s（另加加载）：**至少 2.6 倍**。生成 2.18 tok/s（不变）。
  - prompt 还短（2 块，第二块重叠），加载也算在 prefill 里，所以这是下限。还可以做的：`linearRows` 的预取与循环（现在是串行调度）、M = 16 / 32、逐行做的注意力和 gather / scatter 合并成批量。
- **P5 优化（2026-09-28）**：
  - `linearRows` 预取：第一条 EX 之后把下一块权重装进另一个 SPAD_B 存储体；只装权重的循环按列组预取。
  - 分类层跳过：导出第三个函数 `prefill_kv`（只写 KV，不算分类层），除最后一块外都用它；stories 每块省约 1.1M 周期。
  - M = 16 的修正：平坦分片只用于连续视图（列切片按平坦偏移分片会读错：SmolLM2 M = 16 的 SiLU 输入）；gather 的定义域上限 65536；测试 prompt 至少 2M；`deploy_c6p.sh <model> [M]` 带 `--check`。
  - **板上 M = 8**（与 sim 逐位一致；每块周期不含加载）：
    - stories15M：每块 6.43M → **4.82M（−25%）**，16 个 token 的 prompt 258.6 → **211.6 ms**；
    - SmolLM2：每块 53.9M → **42.1M（−22%）**，14 个 token 2426 → **1871.5 ms**；
    - 线性层快 30–43%（SmolLM2 W13 14.2M → 8.14M，W2 6.83M → 3.85M）。
    - 每 token 约 12 ms（stories，decode 57 ms：4.7 倍）、约 105 ms（SmolLM2，decode 456 ms：4.3 倍）。
  - **M = 8 与 M = 16（stories15M，64 个 token 的 prompt）**：
    - 两者都与 sim 逐位一致；
    - 981.7 ms（8 块）与 964.6 ms（4 块），每 token 只差 1.7%（每块 4.82M 与 9.80M 周期）；
    - 线性层在 M = 8 已接近计算上限：W13 每层每 8 行 83k 周期（M = 16 时 79k），D×D 阵列的理想值是 55k（利用率约 66%）；权重加载已被预取藏住，更多的行只摊薄剩下的少量开销。
    - **结论**：默认 M = 8；M = 32 的 ACC 问题不再追。
  - **剩下的瓶颈**是 VE：SiLU 链（`reduction_Mx768` + `elementwise_Mx768`）在 stories 占 30%、SmolLM2 占 34%，每层每 token 约 3 万周期，与 decode 相同（逐元素的活，批量不省）。
    - 分析：生成的 VE 指令已经最少（exp 一条、recip 一条，各覆盖全部元素）；时间花在硬件的多拍 SFU 上：每元素约 19 周期（exp 115k 周期 / 6144 元素）。
    - 把 SiLU 融进 W13 的分块循环（§8.8 的做法，纯编译器）最多藏住 W13 的 EX 时间（每层 83k，SiLU 238k），约 **−10%**。

**P6（候选，暂缓）：SFU 多槽交错**（`rtl/sysarray/sa_vefp.v` 的微程序控制器）
- **现状**：
  - 多拍模式一次只处理一组；每个微步只给一个半组（FL = 4 lane）发一条运算，然后等单元延迟（M2 / A2 4 拍、i2f / f2i 2 拍，加发出与写回约 6 拍）才发下一步。
  - EXP 11 步、RECIP 8 步、RSQRT 10 步，查表按 lane 逐个读一个 ROM。
  - 一组 8 个元素：EXP 约 150 周期，RECIP 约 110 周期；单元大约 80% 的时间空闲。
- **做法**：
  - **收集**：流水线把最多 NB = 8 个半组（4 组）送进槽，而不是一组就停。
  - **交错发出**：每个微步在槽 0 … NB−1 上连续发出，每周期一个；最后一个槽发出时，槽 0 的结果已经回来，下一步没有空泡。每步每半组约 1 周期（现在约 6 周期）。
  - **每槽的寄存器**：RF[8]、TT / SS、输入与结果。用触发器是现在的 8 倍，外加 64:1 的操作数多路器，约 5k LUT，放不下；改用 LUTRAM（每 lane 64 × 32 位，两份作两个读口），约 500–800 LUT。
  - **其余不变**：
    - 查表先保持按 lane 串行（每半组 4 周期），成为瓶颈时再改成双口 BRAM，每周期读 2 个 lane；
    - 组的输出顺序、目的地址不变；
    - SUM 归约不交错：要保持累加顺序，逐位一致依赖于此。
  - **参数 NB**：1 = 现在的行为；2 / 4 / 8 用综合结果权衡 LUT 与速度。
- **数值**：每个元素的运算与顺序完全不变，功能模拟器不改，结果仍逐位一致，只改时序。
- **预期（NB = 8，查表串行）**：
  - EXP 约 150 → 33 周期/组（4.5 倍），RECIP 约 110 → 25；
  - prefill：stories 每块 4.82M → 约 3.7M（**−23%**）；SmolLM2 约 **−26%**；
  - decode 只约 **−5%**：decode 的时间主要是权重加载与线性层，SiLU 只占每步的 6–8%。
  - 所以主要缩短首 token 时间（prompt 越长收益越大，多轮对话每轮也受益），不提高生成速度。
- **验证**：
  - `tb_fp32`、`tb_sa_vefp`（31 个用例）、`tb_sa_unit`、系统级 `desc_run` 联合仿真；NB = 1 / 2 / 8，FL = D/2 与 D，D = 8 / 16，逐位一致；
  - 新增 EXP / RECIP / RSQRT 的周期数测试，以及槽下标的变异测试；
  - `sa_unit` 的 out-of-context 综合：LUT（现在 83%）与 13.3 ns 时序；
  - 用户跑比特流构建；板上 `board_llm.py` 两个模型逐位一致，`board_profile.py` 前后对比。
- **风险**：
  - LUT 余量：先做 OOC 综合；NB = 8 放不下时，NB = 4 仍约 3 倍。
  - 时序：LUTRAM 读加操作数多路器要在 13.3 ns 内；不满足时操作数加一级寄存器，延迟多 1 拍，8 个槽仍能藏住。
- **状态**：暂缓（2026-09-28）。生成速度的瓶颈在 decode，先对 SmolLM2 的 decode 做一次板上剖析再决定下一步。

### 8.14 C8 代码生成分层重构：把 `sahl-to-sahw` 拆成 pass

**起因（2026-09-29 的评估）**：§8.3 / §8.4 设计了 16 个 pass，每个决定在自己的 pass 里做，可以用 `iree-opt` 单独运行和测试。实际实现只有：

| 实际的 pass | 做的事 |
|---|---|
| IREE 的 comprehensive bufferize | tensor → memref |
| `iree-sa-to-sahl`（140 行） | 只把 copy 换成 `sahl.load / store`；`sahl` 方言也只有这两个操作 |
| **`iree-sahl-to-sahw`（3052 行，一个 `Lowerer` 类）** | 其余几乎全部：微内核的匹配与选择、contraction / 注意力 / 线性层的识别、分块与分片（平坦分片、按行分片、K 块、N 块）、片上内存分配（顺序分配）、预取与双缓冲、DDR 视图与动态 DMA、gather / scatter、逐元素运算体的翻译 |
| `iree-sahw-fuse-ve`、`iree-sahw-split-head`、`iree-sahw-assign-registers` | 与设计一致 |

`SahlToSahw.cpp` 里各部分的大小：逐元素运算（`generic`）约 730 行，通用 contraction 约 520 行，线性层（含 `linearRows`）约 430 行，注意力约 250 行，DDR 与 DMA 约 300 行，分片约 200 行。

**问题**：
- **特例互相影响**：同一个 `Lowerer` 的成员状态（`piece`、`rowPiece`、`locals`、`bcastOf`、`accTop`）被各条路径共用。例如 C6.P 的 M = 16 错误：平坦分片的偏移用到了列切片的视图上。
- **决定不可见**：分块、分片、内存位置、预取都只存在于 C++ 的控制流里，IR 上看不到，也不能单独测试（`test/` 下只有 `sahw` 的 lit 测试）。
- **扩展困难**：T 分块注意力（C6.5）、代价模型（§8.5）、编译期调度（C4 暂缓的融合）、主机退路、C7 都需要在 IR 上可见的分块与内存规划。

**目标**：按 §8.3 / §8.4 的分层把决定移进独立的 pass；`sahl-to-sahw` 最后只做一对一的翻译（目标 < 1000 行）。**每一步生成的描述符逐字节不变**：这是重构，不是优化，所以不需要重新上板。

**验收的护栏：黄金语料**（第一步先做）
- 现在所有测试用到的可执行体，保存各自的 `.sadesc`：
  - stories15M decode + prefill（M = 8，181 个）；
  - SmolLM2 prefill M = 8（777 个）；
  - Qwen3 两层配置；
  - 合成的 K = 16384 模型；
  - `--iree-sa-ukernels=none` 配置；
  - D = 16、更大 SPAD / ACC 的配置。
- `compiler/tests/golden.py`：`--record` 保存，默认模式重新编译并逐字节比较，列出不同的可执行体；加进 `run_tests.sh`。
- 某一步必须改变输出时（例如内存位置的顺序变了），单独说明理由；这些可执行体改用 `dispatch_check`（T = 16 / 80 / 256）与 sim 的端到端逐位一致来验收，并对比 `board_profile` 的周期。

**步骤**

| 步骤 | 内容 | 验收 |
|---|---|---|
| **R0 护栏** | 黄金语料与 `golden.py`；给现有的 pass 补 lit 测试的框架（`iree-opt` 能单独运行每个 pass） | 语料可重现（同一编译器两次记录相同） |
| **R1 `sahl` 操作** | 按 §8.3 补齐 `sahl.matmul`（int8 tile 乘、int32 累加、可累加）、`sahl.elementwise`（带 region，保留 indexing map）、`sahl.reduce`、`sahl.relayout`（转置、DIV-D 复制、打包成广播字）、`sahl.gather`、`sahl.scatter`、`sahl.scalar`；片上 memref 带 `#sa.mem` 与 `#sa.layout`。`sa-to-sahl` 负责识别：to_i8 链、contraction、线性层、注意力的匹配从 `Lowerer` 挪过来；`sahl-to-sahw` 改为读 `sahl` 操作，不再重新匹配 linalg | 黄金语料逐字节相同；每种 `sahl` 操作有 roundtrip 与 verifier 的 lit 测试 |
| **R2 微内核** | 线性层（含 `linearRows`）与注意力的微内核改成 `sahl` 层的展开（§8.8）：一个 pass 把匹配到的操作展开成 `sahl.load / matmul / elementwise` 的序列，由 `--iree-sa-ukernels` 选择 | 逐字节相同；微内核与 `ukernels=none` 两种配置都在语料里 |
| **R3 `sahl-tile`** | 分块与分片变成 IR：contraction 的 N 块、K 块，逐元素运算的平坦分片与按行分片，都改写成 scf 循环（或展开的 `sahl` 操作）作用在子视图上；取代 `Lowerer` 里用成员状态重跑 `lowerBody` 的做法 | 逐字节相同；lit：给定目标配置的容量，检查分块的结果 |
| **R4 `sahl-plan-memory`** | 片上缓冲的 bank 与偏移由一个 pass 写成属性（双缓冲的块放不同 bank，容量不够时报错并给出建议的块大小）；取代 `allocAcc`、`newLocal` 的顺序分配 | 逐字节相同（先照搬现在的分配顺序）；lit：容量检查、bank 冲突 |
| **R5 `sahl-pipeline`** | 双缓冲与预取变成 IR：第 i 块 EX 之后装第 i + 1 块到另一个 bank（`linearRows` 的预取、线性层的双缓冲） | 逐字节相同 |
| **R6 翻译与动态值** | `sahl-to-sahw` 只做一对一翻译；动态值（行循环、动态 DMA、PARAM）挪到 `sahw-legalize-dynamic` | 逐字节相同；`SahlToSahw.cpp` < 1000 行；每个 pass 都有 lit 测试 |
| R7（之后） | 代价模型（§8.5）接进 `sahl-tile` 与 `sahl-pipeline` 的选择；这时才允许输出改变（按周期验收） | 板上逐 dispatch 周期不变差 |

**R0 结果（2026-09-29）**：
- **黄金语料**：`compiler/tests/golden.py`，存在 `build/golden`（源文件去重后的副本，不受模型重新导出影响）。
  - 9 组：stories15M M = 8（带 / 不带微内核）、stories15M M = 16、SmolLM2 M = 8 / 16、Qwen3 全 28 层、K = 16384 合成模型、D = 16、更大的 SPAD / ACC；
  - 共 3112 个 dispatch，去重后 2959 个；
  - 单个源文件用 `--compile-mode=hal-executable` 编译，与整模型编译导出的描述符逐字节相同（有导出的 7 组全部核对过）；
  - 重新编译并比较只要约 20 秒（6 个并行）；两次记录结果相同；故意改坏一个期望文件能检出。
  - `run_tests.sh golden`：先跑 lit，再比较黄金语料。
- **pass 的单独运行**：
  - `sahl` 方言原来没有注册到 `iree-opt`，`sahl` 层的 IR 读不进来：已注册；
  - `iree-sahl-to-sahw` 加了目标配置的 pass 选项（`d`、`spad-kb`、`acc-kb`、`ukernels`），`iree-opt` 可以单独运行；
  - 新 lit 测试：`sa_to_sahl.mlir`，以及 `sahl_to_sahw.mlir`（单级 VE、融合后、D = 16 三种配置；exp 与按行量化两个函数）。
- **发现的旧缺口**：`--iree-sa-ukernels=none` 时，prefill 的 5 个多行线性层编不了（`contraction epilogue input layout`：通用 contraction 的尾部不支持按行的输入）。§8.7 说“不用微内核也能编译全部”只对 decode 成立。语料里把它们记为“预期失败”；R2 把微内核改成 `sahl` 层展开时一并解决。

**R1 的拆分与进度**：
- **R1a 内核分组（完成，2026-09-29）**：
  - 三个匹配器（线性层、注意力、通用 contraction）从 `Lowerer` 抽到 `SahlKernels.{h,cpp}` 的 `KernelMatcher`，两个 pass 共用；
  - 新操作 `sahl.kernel "linear" | "attention" | "contraction"`：`sa-to-sahl` 跑匹配器，把每个匹配的操作按原顺序移进一个 `sahl.kernel`，放在其 contraction（标 `sahl.anchor`）的位置；之后的操作用到的分配、dim、常数、视图提到内核之前（这些不生成命令）；
  - `sa-to-sahl` 也有目标配置的 pass 选项（与 `sahl-to-sahw` 相同）；
  - `sahl-to-sahw` 不再自己在整个函数里找内核，只从每个 `sahl.kernel` 的锚点得到计划；
  - 结果：用哪个微内核、哪些操作一起降级，在 IR 上可见（例如同一个 W13 默认是 `"linear"`，`ukernels=none` 时是 `"contraction"`）；黄金语料逐字节相同；新 lit 测试 `sahl_kernels.mlir`；`SahlToSahw.cpp` 3052 → 2579 行。
- **R1b 运算体里的模式（完成，2026-09-29）**：
  - 逐元素运算与归约继续用 `linalg.generic` 表示：它本身就是 §8.3 设想的“带 region 与 indexing map 的逐元素操作”，另造 `sahl.elementwise` 只是换名。真正需要变成显式的，是降级在运算体里做的两类模式判断：
  - **`sahl.to_i8`**：`qllama.to_i8` 的整条链（NaN、±inf、舍入、截断、fptosi，12 个操作）在 `sa-to-sahl` 里换成一个操作，降级直接当作 VE 的 int8 输出转换；
  - **`sahl.gather "row" | "scalar" | "packed" | "swap"`**：运算体里的 memref.load 换成带种类的 gather。种类由共享的 `gatherForm`（`SahlKernels.cpp`）在 `sa-to-sahl` 里按降级将用的迭代空间决定；降级按种类生成命令，并核对自己的迭代空间得到同样的种类，不一致就报错；
  - 语料中：gather 整行 70、单个标量 63、打包的行 60、成对交换 118；to_i8 1148；内核 linear 59、attention 60、contraction 179（按每个 dispatch 的函数计）；
  - 黄金语料逐字节相同；新 lit 测试 `sahl_gather.mlir`，`sa_to_sahl.mlir` 加了 to_i8。
  - **为保持字节不变留下的一处**：平坦分片的容量估计（`pieceWords`）原来按运算体的操作数计，`sahl.to_i8` 仍按它替换的 12 个操作计。按 1 个计会让 2 个 dispatch 不再分片（描述符变小，可能是改进），留给 R4 的内存规划一起处理。
  - **没有做成操作的**：形状的归一化（多层循环压平、按行合并、动态长度逐行）属于 R3 的分块；i64 标量的 `x + c` 仍是一个 generic（只有位置计算用到，价值小）。
- **R1c scatter（完成）**：一行的 `iree_linalg_ext.scatter`（KV 写入）在 `sa-to-sahl` 里变成 `sahl.scatter`；lit 测试 `sahl_scatter.mlir`。布局转换（转置、DIV-D 复制、打包成广播字）没有做成操作：它们由降级在用到时插入，属于内存布局，见 R4。

**R2–R6 结果（2026-09-29，每步黄金语料逐字节相同，除 R2 的新增功能外）**：

| 步骤 | 实际做的 | 与原计划的差别及原因 |
|---|---|---|
| **R2 微内核** | 通用 contraction 补齐 prefill 线性层的三种形式（按列的尾部输入、int8 的 x 放在 SPAD_A bank 1、没有尾部直接存 int32 累加器）：`--iree-sa-ukernels=none` 现在能编译所有模型的全部 dispatch（原来每个模型有 5 个 prefill dispatch 编不了）。新编译的 dispatch 在 T = 16 / 80 / 256 下与 oracle 逐位一致，stories15M 不用微内核的 prefill + decode 在 sim 上与只用 decode 逐位一致；语料加了 `stories_m16_none`、`smollm2_m8_none`（共 3962 个 dispatch） | 没有把微内核改写成 `sahl` 层的展开：线性层、注意力的调度本质上是命令级的（EX 条带、SPAD_B bank 交替、LOOP_END 成对循环、前缀预取），在 `sahl` 上表达就得把 `sahw` 的概念再造一遍。微内核留在降级里，按种类分文件（R6） |
| **R3 `sahl-tile`** | 逐元素 dispatch 放不下 ACC 时的平坦分片与按行分片变成 IR：每片一个 `sahl.scope`（片内分配在片结束时释放），DDR 视图换成片的子视图（平坦分片经 `collapse_shape`），局部缓冲换成片的大小；跨片共用的归约结果用 `sahl.reserve` 预先分配、只 fill 一次，后面几片的归约标 `sahl.accumulate`。容量估计从降级挪到这个 pass。语料中 17 个 dispatch 被切片（55 片）；降级器不再有分片状态 | contraction 的 N / K 块仍在内核降级里（K 块取决于 x 在降级时落在哪个地址）；动态长度的逐行降级也留在降级里（它是按“每条描述符最多 2 个动态字段”做的，属于命令编码） |
| **R4 内存** | `sahl-plan-memory`：每个局部缓冲的位置（`sa.mem`：`spad_a` / `acc`）与布局（`sa.layout`：`packed` / `bcast` / `rows`）写成属性，降级照此分配并核对；分配器抽成 `LocalMemory`（`SahlLocalMemory.h`：顺序分配，片、块、行结束时按标记释放） | 地址没有在 pass 里定：降级的临时值与缓冲在首次使用时交错分配，事先定地址必然改变输出；带生存期分析的分配会改变地址和记分板看到的 bank 冲突，要先上板测周期（放进 R7） |
| **R5 `sahl-schedule`** | 线性层微内核的调度（`chunk_tiles`：每块多少输出 tile；`loop`：decode 形式超过 4 块时用 LOOP_END 成对循环）由这个 pass 写在 `sahl.kernel` 上，公式共用（`linearSchedule`），降级照此生成 | 双缓冲与预取的位置仍由微内核的降级决定（命令级，见 R2）；通用 contraction 的块由降级决定（见 R3） |
| **R6 只做翻译** | 降级器的声明移到 `SahlLower.h`，定义按职责分文件：`SahlToSahw.cpp` 541 行（pass、逐块遍历、寄存器、局部内存、DDR / DMA、VE、scatter），`SahlLowerGeneric.cpp`（逐元素运算、归约、gather），`SahlLowerKernels.cpp`（线性层、注意力、通用 contraction）；补了 `sa-pack-linear-weights`、`sa-clone-cheap-producers` 的 lit 测试（lit 共 10 个） | 没有 `sahw-legalize-dynamic`：动态值（PARAM、行循环、动态 DMA）是在生成命令时按描述符的编码限制处理的，`sahw` 现在没有“符号化的动态字段”这种中间形式；等 C7（RISC-V 后端，没有 2 个动态字段的限制）需要时再分出来 |

**C8 的结果**：
- **pass 流水线**（每个都能用 `iree-opt` 单独运行，都有 lit 测试）：IREE bufferize → `iree-sa-to-sahl`（DDR 流量显式；内核分组、gather 种类、to_i8、scatter）→ `iree-sahl-tile`（分片）→ `iree-sahl-plan-memory`（位置与布局）→ `iree-sahl-schedule`（线性层调度）→ `iree-sahl-to-sahw`（翻译）→ `iree-sahw-fuse-ve` → `iree-sahw-split-head` → `iree-sahw-assign-registers` → 序列化。
- **决定在 IR 上可见**：用哪个内核、gather 的种类、分片、缓冲的位置与布局、线性层的分块。
- **代码**：原来 3052 行的 `SahlToSahw.cpp` 拆成：
  - 共享的匹配与分类 `SahlKernels.cpp` 约 720 行；
  - 各 pass：`SaToSahl.cpp` 约 290 行、`SahlTile.cpp` 约 350 行、`SahlPlanMemory.cpp`、`SahlSchedule.cpp`；
  - 翻译：`SahlToSahw.cpp` 541 行、`SahlLowerGeneric.cpp` 754 行、`SahlLowerKernels.cpp` 710 行，加 `SahlLower.h`、`SahlLocalMemory.h`。
- **护栏**：`run_tests.sh golden`（lit + 3962 个 dispatch 逐字节比较，约 30 秒）。重构过程中所有已有 dispatch 的输出都没有改变，不需要重新上板。
- **没做到、留给以后的**：
  - 微内核在 `sahl` 层的展开及其双缓冲 / 预取的 IR 表示；
  - 带生存期的内存规划（会改变输出，需要板上周期验收）；
  - `sahw-legalize-dynamic`；
  - R7 代价模型。
  - 为保持字节不变保留的一处估计（`sahl.to_i8` 在分片估计里按 12 个操作计），在做带生存期的内存规划时一起处理。

**顺序与依赖**：R0 → R1 → R2 → R3 → R4 → R5 → R6，每步单独提交。R1 是最大的一步（识别逻辑全部搬家）；R3 与 R4 是 C6.5（T 分块注意力）与代价模型的前提。

**风险**：
- **逐字节相同太严**：某些步骤里操作或分配的顺序很难保持原样。对策：先在新 pass 里照搬现在的顺序，确实做不到的按上面的例外规则验收。
- **bufferize 与内存空间**：片上 memref 带 `#sa.mem` 后，IREE 的 bufferize 与 memref 的 canonicalize 可能不认识；必要时 `#sa.mem` 只在 `sa-to-sahl` 之后出现。
- **工作量**：大（约与 C5.1–C5.3 相当）。这一阶段没有可见的功能，价值体现在后面的扩展（T 分块、代价模型、主机退路、C7、前端通用化）。

**之后建议的覆盖面工作**（本节之外，另行规划）：
1. **主机退路**：sa 编不了的 dispatch 交给 ARM 上的 llvm-cpu（IREE 的异构设备）；现在只能报错。
2. **前端通用化**：量化与形式改写（gather、KV 的写法、prefill 的逐行展开）从手写模型挪进编译器的预处理 pass，让 HuggingFace 的原始建模代码可以直接编译。
3. **非 Llama 结构**：例如 GPT-2（LayerNorm、GELU），验证泛化能力。

### 8.15 主机退路：sa 编不了的 dispatch 在 ARM 上运行

**目标**：任何模型都能跑：sa 后端编不了的 dispatch 交给 ARM 主机，其余照旧在加速器上，两者在同一个设备、同一张命令缓冲里按顺序执行。业界编译器的标准做法（加速器不支持的算子回退到 CPU）。

**做法**（2026-09-29 完成）：
- **主机代码用 VMVX**：IREE 的参考 CPU 后端，把 dispatch 编成 VM 字节码，运行时解释执行（带微内核）。编译器构建里已经有，ARM 版运行时也带了 VMVX 加载器，不需要 LLVM 工具链、不需要重新构建编译器。慢，但只用于退路；以后需要速度可以换成 llvm-cpu（编译器要打开 LLVM_CPU 后端并交叉编译 armv7）。
- **编译器**（`--iree-sa-host-fallback`，`compile_sa.sh` 默认打开，`SA_HOST_FALLBACK=0` 关掉）：
  - sa 设备的目标同时列出 `sa-desc-v1` 与 `vmvx-bytecode-fb` 两种执行体格式，每个 dispatch 都编出两个变体（sa 在前）；
  - sa 后端编不了某个 dispatch 时，不再报错，而是给它的 sa 变体加一个恒为假的 `hal.executable.condition`：
    - IREE 在翻译之后生成 dispatch 调用处的变体选择（格式可用且条件成立的第一个），所以这个 dispatch 选 VMVX 变体；
    - 执行体创建时的选择（`MaterializeResourceCaches`）也一样；
  - 关掉 IREE 的执行体链接（`--iree-hal-link-executables=false`）：VMVX 的链接会把所有 VMVX 变体并进一个执行体，并把原执行体的引用整体改指过去（IREE 源码里标注为有风险），与仍保留 sa 变体的执行体冲突（编译器崩溃）。sa 本来就不链接，VMVX 不链接只是每个 dispatch 一个小模块；
  - 测试用：`--iree-sa-host-dispatches=<名字片段,...>` 把指定的 dispatch 强制放到主机上。
- **运行时**：
  - sa 驱动多注册一个 IREE 的 VMVX 加载器，执行体缓存按格式选加载器；
  - 命令缓冲遇到非 sa 的执行体时，先提交已排队的描述符列表（它的输入可能来自加速器），再按 IREE inline 命令缓冲的方式在 ARM 上逐个 workgroup 执行；
  - 缓冲就是设备窗口（sim：共享文件；板上：CMA 窗口），两边看到同一份内存；
  - `SA_STATS=1`：结束时打印加速器上的描述符列表数与主机上的 dispatch 数。
- **本来想用、没用的做法**：
  - 在链接阶段删掉编不了的 sa 变体：dispatch 调用处的变体选择在链接之前已经生成，删掉会留下悬空引用；
  - 两个设备（sa + local）按 dispatch 指定 affinity：是否支持要到 sa 代码生成时才知道，而 affinity 要在 dispatch 划分之前定。

**验证**（`run_tests.sh fallback`，`test_fallback.py`；lit `sa_host_fallback.mlir`）：
- 小模型：`tanh`（sa 没有，主机）→ `exp`（加速器的 SFU）：与 numpy 的最大相对误差 1.6e-5（来自 SFU 的 exp 近似）；分工为加速器 1 张列表、主机 1 个 dispatch；
- stories15M 把 decode 的全部线性层（25 个 dispatch / token）强制放到主机：16 个 token 的 prompt 加 4 步生成，logits 与只用加速器的版本 20/20 行逐位一致（线性层是整数乘加加 fp32 乘法，两边同样的位）；
- 默认打开退路后，stories15M 的 prefill + decode 仍逐位一致，182 个 sa 可执行体与之前逐字节相同；黄金语料不受影响。
- **板上（2026-09-29）**：
  - 线性层放 ARM 的 stories15M（`deploy_fallback.sh`）：与 sim 9/9 步逐位一致，文本相同；分工为加速器 354 张列表、主机 201 个 dispatch（8 步 × 25，加 prefill 最后一块的分类层）。很慢：decode 每步 16.8 s，prefill 10.5 s（其中的分类层 288 × 32000 也在主机上）。按权重字节算约 0.7 µs / 字节：VMVX 在 ARM 上解释执行，操作数直接读不缓存的 CMA 窗口；
  - 默认打开退路、不强制任何 dispatch 的普通 stories15M：61/61 步逐位一致，0 个主机 dispatch，prefill 211.3 ms（之前 211.6）、64 个 token 981.0 ms（之前 981.7）、decode 57.5 ms / 步（之前 57）：**不用时没有代价**。
- **代价**：SmolLM2 M = 8 的编译从 140 秒到 261 秒，vmfb 从 6.6 MB 到 10.1 MB（每个 dispatch 都多编一个 VMVX 变体）；运行时没有代价（sa 编得了的 dispatch 照旧选 sa 变体）。
- **以后可以做**：
  - 只给编不了的 dispatch 编 VMVX 变体（需要在翻译之前判断能不能编）；
  - 主机 dispatch 与加速器并行（现在是先等加速器做完）；
  - llvm-cpu 代替 VMVX；
  - 主机 dispatch 的操作数先拷到缓存内存（或给 CPU 用缓存映射、在加速器列表前后刷新缓存）。

### 8.16 前端通用化：HuggingFace 的原始建模代码直接编译

**目标**：不再手写模型（qhf.py 按 QLlama 的形式重写了 Llama / Qwen3）。transformers 的建模代码原样导出，量化与那些为加速器准备的写法（gather、KV 的布局与 int8、RoPE 表、成对交换、注意力的形式）由编译器前端的通用图改写完成，不针对某个模型。

**F0 / F1：原样导出，端到端跑对（完成，2026-09-29）**（`compiler/frontend/hf_generic.py`，`run_tests.sh hfgen`）：
- **环境**：前端 venv 装了 transformers 5.17（torch 2.14，iree-turbine 3.9）。
- **导出**：
  - `HFDecoder`：`decode(input_ids[1,1], position_ids[1,1]) -> logits`，里面是 `AutoModelForCausalLM`；
  - transformers 5.x 的静态 KV cache 自己记位置（一个每步递增的标量缓冲），导入 turbine 时出错；`PositionedCache` 改成在调用方给的位置写，写进封装模块的缓冲（通过模块属性访问，torch.export 才认出是缓冲的修改，KV cache 成为可变的全局量）；
  - transformers 自带的导出封装（`TorchExportableModuleForDecoderOnlyLM`）5.17 的外层 forward 丢了 `cache_position`，没用；
  - 流程：torch.export（non-strict）→ `run_decompositions()`（函数式 ATen）→ 图改写 → 参数外置（必须在分解之后：先标记会让 aot_autograd 失败）→ iree-turbine。
- **图改写**（`GRAPH_PASSES`）：目前只有 `pow(x, 2) → x * x`（VMVX 没有 `fpowi`，61 处）。
- **运行**：`sa-llm-run --abi=hf`（每步 `main(input_ids, position_ids)`，没有 valid）。
- **结果**（SmolLM2-135M，sim）：
  - 32 个可执行体，**3 个在加速器、30 个在主机**；
  - 9 步与 transformers 的 fp32 相比：最小相关 1.000000，最大误差 2.6e-3，argmax 9/9 相同；
  - 很慢（每步约 3.8 秒）；fp32 权重转置后在编译期求值，嵌进了 vmfb（538 MB），sim 窗口要开到 1.1 GB。

**F2：线性层与嵌入的 W8A8（完成，2026-09-29）**：
- 按模块类型替换：导出前把所有 `nn.Linear`（SmolLM2 211 个）换成 `QLinear`、`nn.Embedding` 换成 `QEmbedding`，算式与 qhf 相同（权重按行 int8、激活按 token 动态 int8、int32 乘加，再依次乘两个 scale）；与分类层共享的嵌入只量化一次，int8 表共用。不依赖具体模型（与 torch.ao 的按模块量化相同）。
- 分解表改用 iree-turbine 的（`aot.export(module)` 用的那张）：torch 默认的表会把量化链（`nan_to_num`、`round`、`clamp`）分解成编译器认不出的形式。
- **因此发现并修了 sa 编译器的一个错误**：
  - HF 的线性层是 `batch_matmul(x, extsi(Wᵀ))`，权重没打包，由通用 contraction 降级；
  - 当尾部只做截断、输出 int32 时，分块尾部仍按 fp32 写结果，int32 被转成浮点，结果全错；
  - 现在 `Chunk` 带输出类型，contraction 按尾部的实际类型分配和写回；
  - 由逐 dispatch 检查找到，修复后原有语料逐字节不变。这个模型的 dispatch 已加进黄金语料（共 4002 个）。
- **结果**（SmolLM2，sim）：
  - 13 个可执行体在加速器、27 个在主机（F0 是 3 与 30）；
  - 与 torch 里的量化模型相比：9 步最小相关 1.000000，最大误差 1.9e-4，argmax 9/9 相同；
  - 参数 176 MB（int8）。

**F3：RoPE（完成，2026-09-29）**（`rope_rewrite`）：
- 旋转位置编码模块（类名以 `RotaryEmbedding` 结尾的）换成 `RopeTable`：用原模块把 0 … max_len−1 的位置算一遍，得到 `cos` / `sin` 表，之后按位置查表（任何 RoPE 变体都适用，频率缩放也在表里）；
- `rotate_half` 变成成对交换：q / k 投影每个头的维度把前后两半交错（q·k 不变：两边同样排列），表的列、q / k norm 的权重同样排列；凡定义了 `rotate_half` 的 transformers 模型模块，都把它换成 `swapneg`（交错排列下两者相同）；
- **fp32 模型的输出与原模型完全相同**（最大误差 0）；
- SmolLM2：17 个可执行体在加速器、20 个在主机；9 步 argmax 9/9 相同；sa 编出的 17 个全部与 oracle 一致。

**RMSNorm 与查表**（完成）：类名以 `RMSNorm` 结尾的模块换成 `SaRMSNorm`（`mean` 改成乘以常数 1/n，与 qhf 相同）；嵌入与 RoPE 表的查表写成 `index_select`（编译器的按行 gather）。SmolLM2：22 个在加速器、15 个在主机。

**F4：int8 KV cache 与注意力（进行中，2026-09-29）**：
- 做法：
  - transformers 的 `AttentionInterface` 注册 `"sa"` 注意力（`sa_attention`，qhf 的算式：q 按头量化、int32 的分数、按位置掩码的 softmax、概率 int8、int32 的 P·V）；
  - `SaCache`：一个 K、一个 V 的 int8 缓冲 `[层数 × T, kv_dim]`，按行写，每层静态 scale；
  - `calibrate_kv`：fp32 模型在 qhf 的校准文本上统计 max |K|、|V|（与 qhf 相同）。
- **因此修了 sa 编译器的第二个错误**：带前缀掩码的多行静态 generic（静态 T 的 softmax，9 行 × 256）被合成一条 VE，VALID 只截了整个向量的前 pos + 1 个元素，第 1 行起全错；现在逐行降级（`RowSel::fixed`）。原有语料逐字节不变。
- **发现 IREE 3.11 的一个问题**（`compiler/tests/iree_global_merge_repro.py`，20 行即可复现，与 sa 无关，VMVX 上也出现）：两个初始内容相同的可变 int8 全局量（参数导入编译）在同一次调用里各写一行，第二个的数据会写进两个里。绕过：每个缓冲的初始内容不同；并且像 qhf 一样所有层共用一个 K、一个 V 缓冲。
- **封装的因果掩码错误（已修）**：transformers 5.x 按 cache 自己记录的长度构造因果掩码，`PositionedCache` 不更新它，掩码一直当作位置 0，注意力只看第 0 个位置。F1–F3 的测试拿编译结果和同一个封装比，一起错，所以当时没发现；F4 的校准也因此偏了。现在每步把输入位置设为这个长度（属性赋值，不是可变状态）；测试先检查封装与原版 HF（整段一次算完）一致（最大误差 1e-4）。F1–F3 的结论据此重新验证：W8A8 + RoPE 的配置与 torch 量化模型相比相关 0.984、argmax 8/9（之前的“完全一致”是因为注意力实际上只看一个位置）。
- **结果**（SmolLM2，sim）：
  - 171 个可执行体在加速器（KV 写入与注意力全部上了加速器），33 个在主机（主要是尾部带 int8 输出的线性层：K / V 投影与写 cache 的量化融在一起）；
  - sa 编出的每个 dispatch 都与 oracle 逐位一致（T = 16 / 256）；
  - 逐步与 torch 量化模型比较不适合作判据：int8 舍入对运算顺序敏感（设备与 torch 的求和顺序不同），翻转随步数放大；两个都正确的程序（手写 qhf 与通用路径，同样的 KV scale）在设备上也会逐渐分开。
  - **质量**（固定文本逐位置喂入 40 个 token，top-1 与原版 fp32 HF 一致）：**通用路径 35/40（平均相关 0.9567），手写 qhf 36/40（0.9569）**，两者之间 39/40（这次通用路径用的是 qhf 的 KV scale）；用自己校准的 KV scale（与 qhf 只差最低位）是 **34/40（0.9566）**，`run_tests.sh hfgen` 通过。
  - 测试判据改为这个质量指标（≥ 85%，平均相关 > 0.95）。参照模型必须在任何改写之前算好：`rope_rewrite` 替换 `rotate_half` 对整个进程生效。
- **F4 剩下的**：尾部带 int8 输出的线性层上加速器（通用 contraction 的尾部支持 int8 输出）。**已完成**（§8.19：int8 尾部、SwiGLU 的 int32 逐元素输入；decode 全部在加速器上）。

**落到主机的原因（F0 的测量）与对应的通用改写**：

| 数量 | 原因 | 改写 | 步骤 |
|---|---|---|---|
| 6 | 线性层是 fp32（EX 只做 int8） | W8A8：权重按通道 int8、激活按 token 动态 int8（qhf 的 qlinear 的形式），作用于每个常量权重的 `aten.linear` | F2 |
| 6 | ACC 放不下（fp32 的大融合 dispatch） | F2 之后规模变小；剩下的看分块 | F2 |
| 2 | `sin` / `cos`：HF 每步现算 RoPE | 位置无关的部分常量折叠成表 `[max_len, dim/2]`，按位置 gather | F3 |
| 1 | `rotate_half`（拼接两半） | q / k 投影的权重按头重排成交错的对，RoPE 变成成对交换（llama2.c 的导出、qhf 的做法） | F3 |
| 3 + 2 + 3 | 注意力是 fp32 的 `batch_matmul`；KV 按 `[B, H, T, D]` 写；位置是向量 | KV 改成 `[T, H, D]` 的 int8（按层的静态 scale，用 fp32 模型校准），SDPA 改写成分数、带掩码的 softmax、P·V，与 qhf 的形式一致 | F4 |
| 其余 | 除法等 | 按需 | F5 |

**验收的主要依据**：原定与手写 qhf 路径逐位一致；实际上两个前端各自的算式细节（HF 用 fp32 算 RoPE 表、qhf 用 float64；注意力的求和长度）不同，int8 舍入又对运算顺序敏感，逐位一致不是合适的目标。改为：sa 的每个 dispatch 与 oracle 逐位一致（编译器的正确性），加上与原版 fp32 HF 比较的质量不低于手写路径（前端的正确性），并且能上板；再用 Qwen3 验证同一套改写不需要改代码。之后 F6：非 Llama 结构（GPT-2：LayerNorm、GELU、learned position embedding）。

### 8.18 可用的 Qwen3 需要多大的 FPGA（估计，2026-09-29）

单序列生成受内存带宽限制：每生成一个 token，全部权重（加上分类层）要从 DDR 读一遍，生成速度 ≈ 可用带宽 ÷ 每 token 读的字节数。以下按公开的 Qwen3 配置推算，没有实测；“可用”按交互生成 ≥ 10 token/s、上下文 2K。

| 型号 | 每 token 读的权重（int8 / int4） | KV cache（int8，2K） | 10 token/s 需要的带宽（int8 / int4） |
|---|---|---|---|
| Qwen3-0.6B（28 层，隐藏维 1024） | 约 0.6 GB / 0.3 GB | 约 117 MB | 约 6 / 3 GB/s |
| Qwen3-1.7B（28 层，2048） | 约 1.7 GB / 0.9 GB | 约 117 MB | 约 17 / 9 GB/s |
| Qwen3-4B（36 层，2560） | 约 4.0 GB / 2.0 GB | 约 150 MB | 约 40 / 20 GB/s |
| Qwen3-8B（36 层，4096） | 约 7.6 GB / 3.8 GB | 约 150 MB | 约 76 / 38 GB/s |

- 小模型里分类层（词表 15 万 × 隐藏维）占每 token 读取量的很大一部分（0.6B 约四分之一）；生成时还要读 KV cache（平均上下文 1K 时每 token 约 60–75 MB）。
- 算力要求不高：decode 每 token 约“参数量”次乘加（0.6B 在 10 token/s 时约 6 GMAC/s）；prefill 才吃算力（512 个 token 约 300 GMAC，1–2 秒内完成需要 150–300 GMAC/s）。
- **PYNQ-Z1 不行**：512 MB DDR 装不下 0.6B 的 int8 权重（int4 能装下）；实测读权重约 0.39 GB/s（一个 64 位 HP 口、50 MHz），int4 也只有约 1 token/s；8×8 阵列、50 MHz 峰值 3.2 GMAC/s；LUT 已用 83%。

| 级别 | 典型器件 / 板子 | 内存与带宽 | 能跑的 Qwen3 |
|---|---|---|---|
| 入门 Zynq UltraScale+ | Kria KV260（K26 SOM，约 25 万逻辑单元、1248 个 DSP，4 GB 64 位 DDR4，理论约 19 GB/s，PL 实际约 10 GB/s） | 4 GB | 0.6B int8 约 10–15 token/s；1.7B int4 约 8–10 token/s |
| 中档 Zynq UltraScale+ | ZCU104 / ZCU102（PS 与 PL 各有 DDR4） | 4–6 GB，合计约 20–30 GB/s | 1.7B int8 约 10 token/s；4B int4 约 5 token/s |
| 带 HBM | Alveo U55C / U280，Versal HBM | 8–16 GB，约 400–460 GB/s | 4B / 8B int8 几十 token/s |

**结论**：Qwen3-0.6B / 1.7B 要“可用”，需要 KV260（K26）这一级；4B / 8B 需要 HBM。加速器要跟着改：DMA 加宽到 2–4 个 128 位 HP 口、200–300 MHz（提速的主要来源）；LD 路径上解包 int4（带宽减半）；阵列 D = 16 或 32（prefill 的算力，M4 已验证 D = 16）；SPAD / ACC 用 URAM 加大；编译器的 T 分块注意力（C6.5）。编译器这边已具备：硬件参数来自目标配置，Qwen3-0.6B 全 28 层 sim 上逐 dispatch 验证过，通用前端可直接用 HF 的 Qwen3 代码。

### 8.17 下一步（2026-09-29 定）

按顺序：

| 顺序 | 工作 | 内容 | 验收 | 风险与依赖 |
|---|---|---|---|---|
| 1 | **通用路径的 prefill**（端到端 prefill + decode） | 导出 `prefill(tokens[M], positions[M])`，与 decode 在同一个模块里共用参数和 KV cache；`SaCache` 一次写 M 行；`"sa"` 注意力按行加因果掩码（qhf 的逐行形式）；`sa-llm-run --abi=hf` 支持分块 prefill；顺带 F4 剩下的 int8 尾部 | 与同一模型只用 decode 在设备上逐位一致（C6.P 的判据）；质量指标不下降；Qwen3 不改代码；上板 | 主要复用现有编译器（多行线性层、逐行注意力都已有）；可能再暴露编译器的边界情况。**sim 上完成，待上板**（§8.19） |
| 2 | **微内核在 `sahl` 层展开** | 线性层、注意力、通用 contraction 的调度（分块、SPAD_B bank 交替、预取、LOOP_END 循环）用 `sahl` 操作显式表达，由一个 pass 展开；`sahl-to-sahw` 只做一对一翻译；`sahl` 需要补上能表达 bank、条带、循环的操作 | 黄金语料逐字节相同，不用上板 | 工作量大；C8 时没做（§8.14 R2：等于把 `sahw` 的概念再造一遍）。现在的价值是让调度能被代价模型和内存规划看见、能被改；也是第 3 项的前提 |
| 3 | **带生存期分析的内存规划** | 所有局部缓冲和临时值都在 IR 上之后（第 2 项），按生存期分配 SPAD / ACC，按 bank 避开冲突，放不下时由规划决定分片（替代现在的估计，包括 `sahl.to_i8` 按 12 个操作计的那处） | 会改变输出：逐 dispatch 检查与 sim 逐位一致；板上逐 dispatch 周期不变差 | 地址与 bank 的变化影响记分板的并发，可能变快也可能变慢，要在板上实测 |

理由：第 1 项是功能上的完整性（通用前端追上手写路径）；第 2 项是纯重构，由黄金语料守着；第 3 项依赖第 2 项，并且是唯一需要上板测性能的一步。

### 8.19 通用路径的 prefill（§8.17 第 1 项的结果，2026-09-29）

**结果**（sim；SmolLM2-135M 与 Qwen3-0.6B 截断到 2 层，都用原版 HF 建模代码，W8A8，M = 8）：

| | SmolLM2-135M | Qwen3-0.6B（2 层，不改代码） |
|---|---|---|
| prefill + decode 与只用 decode | 逐位一致（5/5 行，token 相同） | 逐位一致（5/5 行，token 相同） |
| decode 每步 | 一个描述符列表，**0 个主机 dispatch** | 一个描述符列表，0 个主机 dispatch |
| 可执行（decode + prefill） | 794 个上加速器，2 个在主机（prefill 里 Q 的 RoPE） | — |
| 质量（teacher-forced top-1 与 fp32 HF） | 设备 34/40（0.9573），torch W8A8 34/40（0.9558） | 设备 22/40，torch W8A8 21/40 |
| 逐 dispatch 检查（`--dirty`） | 794 个全部一致 | — |

**做了什么**：
- 前端：`prefill(tokens[1, M], positions[1, M])` / `prefill_kv` 与 decode 一起导出（`aot.FxPrograms`，共用参数和 KV cache）；`SaCache` 逐行写 KV；`"sa"` 注意力逐行加前缀掩码；嵌入、RoPE 表逐行 gather；`QLinear` 在二维行上做乘法（三维 int32 matmul 会导出成 i64 累加的 `batch_matmul`，只能在主机上跑）；`sa_attention` 把 query 转回 `[M, H, hd]`（每行连续）。`sa-llm-run --abi=hf` 支持分块 prefill。
- 编译器：通用 contraction 的尾部支持 int8 输出（K / V 投影融合了 KV 量化）和 int32 的逐元素输入（MLP 的 `silu(gate) · up` 融合进 gate 的尾部）；`sa-to-sahl` 把 lowering 不收的三层循环（外层是小的静态行循环：M 行的 RoPE 与 KV 量化）按行展开成 decode 的两层形式，gather 的源按行切片。
- 测试：`test_hf_generic.py --prefill M`（prefill + decode 与只用 decode 逐位一致）、`--layers N`（截断模型）、`--board-bundle`；`dispatch_check.py` 加了不同值的 i64 向量、存储区域以外不得写（8 字节对齐的范围除外）、`--skew`、`--dirty`；`sa_sim_server.py --wlog`。

**找到的 IREE 缺陷**：prefill 一开始与 decode 完全对不上（相关 ≈ 0），而每个 dispatch 单独检查都一致。用 `--iree-sa-host-dispatches` 把 prefill 的 dispatch 分批放到主机上二分，定位到一个 sa 上的 dispatch（27）；它的输出逐位正确，但紧跟着的主机 dispatch（28）在两次运行里输入相同、输出不同——workgroup 数不同（1 与 3）。原因是 IREE 的 `MergeIndexSwitchPattern` 合并相邻的 `scf.index_switch` 时只比较 case 值，不比较选择值；主机退路下相邻两个 dispatch 的变体选择值不同（一个选 sa，一个选 VMVX），合并后主机上的 dispatch 用了 sa 变体的 workgroup 数，只算了三分之一。修正是一行（要求选择值相同），放在 `compiler/patches/iree/`，`fetch_iree_sources.sh` 自动打上，`merge_index_switch.mlir` 守着。教训：逐 dispatch 检查看不到 dispatch 之间和运行时的问题；二分 + 两次运行逐 dispatch 比较绑定内容的哈希是有效的办法。

**质量判据改了**：原来是“与 fp32 HF 的 top-1 一致 ≥ 85%”，这其实在评判量化方案，不是编译器：完整的 Qwen3-0.6B 上 torch 的 W8A8 模型本身只有 25/40（2 层 21/40）。现在的判据是设备保持量化模型的质量：一致数不低于 torch W8A8 模型的减 2，平均相关不低于它的减 0.01；两者都打印出来。**Qwen3 的 W8A8 质量是单独的问题**（C6.2 量化质量：逐 token 激活 + 逐行权重 + 每层静态 KV int8 + int8 概率在 Qwen3 上损失明显），不在第 1 项里。

**剩下**：上板（`compiler/scripts/deploy_hfgen.sh` → `build/deploy_hfgen`，板上 `board_llm.py` 与 sim 逐位比较）；prefill 里 Q 的 RoPE 两个主机 dispatch（IREE 把 swap 后的 q 物化成按头在外的布局，gather 带了转置；改前端 swap 的写法反而让 31 个落到主机，没采用）；主机退路改用原生 ARM 代码（`llvm-cpu`，替代 VMVX 解释器，需要重建编译器）。

## 9. 验证体系

| 层次 | 内容 | 工具 |
|---|---|---|
| 插件单元测试 | 匹配与降级：MLIR 片段 → 期望的模板参数或描述符（FileCheck） | lit |
| 模板测试 | 每个模板对随机形状在功能模拟器上执行，与 NumPy / `DeviceModel` 的对应函数逐位比对；序列化结果与 `DescList` 逐字节比对 | pytest + `sa_funcsim` |
| 端到端（主机） | 整个 .vmfb 在 `sim` 传输层上运行，logits 与 `DeviceModel(sfu=SfuExact)` 逐位一致 | IREE 运行时 + 模拟器服务 |
| RTL 联合仿真 | 抽取 IREE 生成的若干命令缓冲（列表 + DDR 映像）放进 `desc_run`，在 `tb_system` 上执行 | 现有 `gen_desc_cases.py` 流程 |
| 板上 | 生成文本与 L5 手写路径完全相同；稳定性（10 × 256 个 token）；速度与计数器 | 板上脚本 |

变异测试照旧：故意改错模板（下标模式、舍入次数、循环次数），确认各层测试能检出。

---

## 10. 分阶段计划与验收

| 阶段 | 内容 | 验收 | 工作量（相对 L1–L5） |
|---|---|---|---|
| **C0 前端** | 量化 torch 模型；iree-turbine 导出；`llvm-cpu` 在主机和板上运行；dispatch 清单 | §3.4；清单与手写片段的对照表。**已完成**（§3.5） | 小 |
| **C1 驱动** | sa HAL 驱动（board、sim 两种传输层）；模拟器服务；sa-desc-v1 读写 | §5.6：手工封装的模板在 sim 与板上与 `DeviceModel` 逐位一致。**已完成**（§5.7） | 中 |
| **C2 第一条纵向切片** | 从源码构建带插件的 iree-compile；插件只支持量化线性层一种 dispatch，其余报错；用一个“单线性层”的 torch 模型测试 | torch → iree-compile → .vmfb → sim 与板上，结果与 `DeviceModel.linear` 逐位一致。**已完成**（§6.7） | 中（构建环境是主要难点） |
| **C3 模板库** | §6.4 的全部模板；VE 表达式编译器；数据分块与权重打包；KV cache 原地更新 | IREE 编译的 stories15M 在板上生成的文本与 L5 手写路径完全相同，logits 逐位一致。**已完成**（§6.9，板上 12.6 tok/s） | 大 |
| **C4 性能** | 精简 FENCE、跨 dispatch 预取、更大的 dispatch、代价模型 | 每 token 周期与手写路径相差 ≤ 10%。**部分完成，暂缓**（§7.1：差 13%，板上 17.7 tok/s） | 中 |
| **C5 代码生成** | `sahl` / `sahw` 两层方言与完整流水线（§8），模板改成 `sahl` 层微内核，分步 C5.0–C5.5（§8.7） | 只用通用代码生成（`--iree-sa-ukernels=none`）也能编译 stories15M，结果与 C3 一致；默认配置每 token 周期不高于 C4；一个模型变体编译通过（§8.1）。**已完成**（§8.7：stories15M 带 / 不带微内核板上逐位一致；SmolLM2-135M 板上 2.23 tok/s，可交互式生成） | 大 |

| **C6 更大的模型** | HuggingFace 导入与通用量化、目标配置参数化、K / T 分块、GQA / QK-norm（§8.9） | Qwen3-0.6B、Llama-3.2-1B 在 sim 上编译，逐 dispatch 逐位一致，截断层数的端到端误差在范围内；有资源更多的板子后上板。**进行中**（§8.12：目标配置交叉验证、K 分块、C6.0 嵌入只存一份、C6.1 Qwen3-0.6B 全 28 层 sim 逐位一致已完成；C6.P prefill + decode 两个模型板上逐位一致，prompt 每 token 比 decode 快 4.3–4.7 倍，§8.13） | 大 |

| **前端通用化** | HuggingFace 的原始建模代码直接导出，量化与写法由通用图改写完成（§8.16） | sa 的 dispatch 与 oracle 逐位一致，质量（与原版 fp32 HF 的 top-1）不低于手写 qhf 路径；Qwen3 不改代码；上板；非 Llama 结构。**进行中**（F0–F4 完成；通用路径的 prefill + decode 在 sim 上与只用 decode 逐位一致，decode 全部在加速器上，Qwen3 不改代码通过，§8.19；待上板） | 大 |
| **主机退路** | sa 编不了的 dispatch 在 ARM 上用 VMVX 运行（§8.15） | 任何 dispatch 都能跑；混合执行与只用加速器逐位一致（线性层放主机的 stories15M）。**完成**（sim 与板上） | 中 |
| **C8 代码生成分层重构** | 把 `sahl-to-sahw` 拆成 §8.4 设计的 pass：`sahl` 操作、微内核展开、分块、内存规划、流水、动态值（§8.14） | 每步黄金语料的描述符逐字节相同；`SahlToSahw.cpp` < 1000 行；每个 pass 有 lit 测试。**完成**（2026-09-29，§8.14：R0–R6；微内核的 `sahl` 展开、带生存期的内存规划已排进下一步（§8.17）；`sahw-legalize-dynamic`、R7 代价模型留给以后） | 大 |
| **C7 RISC-V 后端**（可选） | `sahw` → LLVM → riscv32，PicoRV32 用 PCPI 指令发命令（§8.10） | 一个 dispatch 由 PicoRV32 代码执行，与描述符路径逐位一致 | 中 |

- **顺序**：C0 → C1 → C2 → C3 → C4 → C5。C1 和 C0 可以并行；C2 依赖 C1 的驱动与 sim。
- **每一步都在主机上先过**（sim 传输层），再上板；上板通过后提交，与现有约定一致。
- **可选扩展**：批处理 decode（L5b 的布局）、prefill（D 个 token 一批）、更大的模型（stories42M / 110M）、GQA。

---

## 11. 目录结构

所有编译器相关的代码放在仓库顶层的 `compiler/`（环境与构建方法见 [`../compiler/README.md`](../compiler/README.md)）：

```
compiler/
  env.sh                  路径与设置（IREE 版本、源码、构建目录、Python 环境）
  scripts/                拉取 IREE 编译器的子模块；从源码构建带插件的 iree-compile
  plugins/sa/             sa 目标后端插件（C++，经 IREE_CMAKE_PLUGIN_PATHS 编进 iree-compile）
    target/               TargetDevice / TargetBackend、预处理（权重打包）、匹配、C++ 模板与 DescList、sa-desc-v1 序列化（C2）
    templates/            模板的 Python 参考（reference.py，C++ 输出须与之逐字节相同）
    dialect/              sahl、sahw 两层方言：ODS、属性、操作、接口（C5，§8.3）
    transforms/           C5 的流水线（§8.4）与代价模型（§8.5）
    ukernels/             sahl 层微内核（linear、attention，§8.8）
    riscv/                sahw → LLVM 的 RISC-V 后端（C7，§8.10）
    test/                 lit 测试（iree-opt + FileCheck）
  frontend/               C0：qllama.py（量化模型）、export.py（iree-turbine 导出）、对比脚本
  runtime/                C1：sa HAL 驱动（C，sa/），传输层 board / sim；tools/（sa-desc-v1）；test/
  sim/                    模拟器服务 sa_sim_server.py、命令环模拟 sa_board_emu.py（基于 llm/sa_funcsim.py）
  tests/                  test_c2 / test_c3（主机）、board_c2 / board_llm（板上）；oracle.py（dispatch 参考解释器）、
                          dispatch_check.py（逐 dispatch 差分测试）、summarize.py
iree-sa/l0/               已有：armv7 工具链验证（L0）
build/iree/               不入库：IREE 源码、Python 环境、编译器构建目录
```

## 12. 风险与对策

| 风险 | 对策 |
|---|---|
| **从源码构建 iree-compile**：耗时长、占内存，接口随版本变化 | 固定在 L0 的版本（3.11.0，e4a3b0405d）；构建脚本化；插件接口的用法以该版本的内置后端为范例；只在大版本升级时迁移 |
| **dispatch 的切法与预期不同**（融合过多或过细） | C0 先拿到 dispatch 清单再设计模板；用预处理控制切分；VE 表达式编译器兜住任意的逐元素链 |
| **数值不一致**（重结合、FMA、拆分归约、库函数） | §3.2 的约定在插件里检查，违反就报错；端到端测试以 `DeviceModel` 逐位比对，任何偏差立即可见 |
| **KV cache 每个 token 被整体拷贝** | C0 检查生成的 IR 与运行时的拷贝操作；必要时 KV 更新做成自定义算子，或由驱动直接管理 KV 缓冲 |
| **dispatch 之间经过 DDR、FENCE 过多**，C3 明显慢于手写路径 | 预期之内；C4 专门处理；手写路径作为对照，差距可以量化 |
| **CMA 与物理地址**：IREE 默认的分配方式不适合 | 驱动自带 CMA 池；参数文件直接加载进 CMA |
| **多设备（CPU + 加速器）不成熟** | 主路径是全部 dispatch 都在加速器上；混合执行只作后备 |
| **armv7 运行时的问题**（L0 遇到过 libm） | 沿用 L0 的交叉编译配置和 shim；驱动是纯 C，不依赖浮点库 |
| **工作量**：C3、C5 需要大量 C++ / MLIR | 纵向切片优先；模板从已验证的 Python 实现移植，逐个用 sim 比对 |

---

## 13. 待定的决策

1. **C0 之后**：根据 dispatch 清单确定模板的清单与粒度（是否需要预处理来控制切分）。
2. **模拟器传输层的实现**：Python 服务（快，复用现有代码）还是把 `sa_funcsim` 移植到 C（不依赖 Python，速度更快）。先用 Python。
3. **可执行体的编码**：自定义二进制还是 flatbuffer。倾向 flatbuffer（IREE 惯例），内容按 §4.2。实际用的是自定义二进制（§4.4）。
4. **常量的存放**：参数文件（.irpa，推荐）还是嵌在 .vmfb 中。
5. **C4 的跨 dispatch 调度放在哪里**：驱动（录制命令缓冲时看到相邻的 dispatch）还是编译器（链接阶段）。原本倾向编译器；**已定为驱动**（§7.1），编译器经 sa-desc v3 提供信息。
6. **C5 的缓冲化**：IREE 的 one-shot bufferize（带 `#sa.mem` 内存空间），还是自己的缓冲化。先试前者（§8.4 第 4 步、§8.11）。
7. **C5.5 的模型变体**：**已定**：SmolLM2-135M（板上）与 Qwen3 结构的小配置（sim）（§8.7、§8.9）。
8. **C5 的方言分层**：**已定**：两层，`sahl`（tile 级）与 `sahw`（命令级）；`sahw` 接描述符与 RISC-V（C7）两个后端（§8.3、§8.10）。

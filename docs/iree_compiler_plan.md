# LLM 编译器：基于 MLIR / IREE 的端到端方案（PyTorch → 描述符列表）

状态：**C0–C3 完成**（§3.5、§5.7、§6.7、§6.9）；**C4 部分完成、暂缓**（§7.1：板上 2.591M 周期 / token，17.7 tok/s，与手写路径差 13%，目标 ≤ 10%；剩下的融合留到 C5 之后）；**C5 进行中**（§8；C5.0 完成）。这是 [`llm_inference_plan.md`](llm_inference_plan.md) 的 L6-IREE 一级的详细设计，
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
| 端到端，板上（`tests/board_c3.py`，L2 bitstream、rt_fw） | 76/76 步逐位一致，生成的 token 与主机参考相同；**12.6 tok/s**（L5 手写：15.1 tok/s）；每 token 242 次 dispatch |

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
| 端到端 | `test_c3.py`（sim）、`board_c3.py`、`board_profile.py`；新增 `compare_profiles.py`：两份逐 dispatch 周期表逐项对比 |
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
| 1 | 目标配置参数化 | C5.0 起贯穿 C5；funcsim 的参数化与交叉验证在 C5.1 | 新 pass 从第一行起不写常数，几乎没有额外成本；事后再改要翻遍所有 pass |
| 3a | K 维分块、跨块累加 | C5.2 | 通用 contraction 分块的一部分；stories15M 用不到，用单独的线性层测试覆盖 |
| 3b | GQA 下标映射 | C5.3 实现，C5.5 首次用上 | 注意力降级时按组映射设计，SmolLM2 需要 |
| 2 | 前端：HuggingFace 导入、通用量化、QK-norm 等结构 | C5.5（可在 C5.1–C5.3 期间穿插先做） | 与代码生成无关；SmolLM2 与 Qwen3 小配置需要 |
| 4 | 运行时：设备窗口可配置、分配器到 GB 级 | C5.5 | SmolLM2-135M 的权重约 135 MB，现在的 64 MB 窗口放不下；改动小 |
| 5 | 验证：与 torch 比误差、截断层数 | C5.5 起；C funcsim 在 C6 视需要 | 新模型没有手写的 DeviceModel；Python sim 只在大模型上才太慢 |
| 3c | 注意力按 T 分块（在线 softmax） | C6 | 涉及数值约定（预处理显式改写 IR、oracle 相应支持）；seq ≤ 2048 时是否需要，C5.5 前核实片上容量 |
| 3d | 大词表、多张列表 | 已具备，C6 验证 | 不需要新代码 |
| 6 | 批量 prefill | C6 之后（可选） | 只影响速度，不影响能否编译 |

**阶段**：
- **C5**：第 1、3a、3b 项（§8.7 的 C5.0–C5.3）。
- **C5.5**（§8.7）：第 2、4、5 项；SmolLM2-135M（板上）与 Qwen3 结构的小配置（sim）。
- **C6 更大的模型**：第 3c 项，第 5 项的 C funcsim（视需要）；Qwen3-0.6B、Llama-3.2-1B 在 sim 上编译并逐 dispatch 验证（截断层数做端到端）；换开发板后在板上运行。之后可选第 6 项。

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
| **C5 代码生成** | `sahl` / `sahw` 两层方言与完整流水线（§8），模板改成 `sahl` 层微内核，分步 C5.0–C5.5（§8.7） | 只用通用代码生成（`--iree-sa-ukernels=none`）也能编译 stories15M，结果与 C3 一致；默认配置每 token 周期不高于 C4；一个模型变体编译通过（§8.1） | 大 |

| **C6 更大的模型** | HuggingFace 导入与通用量化、目标配置参数化、K / T 分块、GQA / QK-norm（§8.9） | Qwen3-0.6B、Llama-3.2-1B 在 sim 上编译，逐 dispatch 逐位一致，截断层数的端到端误差在范围内；有资源更多的板子后上板 | 大 |

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
  tests/                  test_c2 / test_c3（主机）、board_c2 / board_c3（板上）；oracle.py（dispatch 参考解释器）、
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

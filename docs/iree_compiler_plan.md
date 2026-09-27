# LLM 编译器：基于 MLIR / IREE 的端到端方案（PyTorch → 描述符列表）

状态：**C0 完成**（主机和板上验证，§3.5）；环境已就绪（`compiler/`）。C1–C5 还是方案。这是 [`llm_inference_plan.md`](llm_inference_plan.md) 的 L6-IREE 一级的详细设计，
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

---

## 8. 完整代码生成：sa 方言（C5）

模板库只覆盖已知的算子。完整代码生成让编译器能处理新的模型结构：

- **`sa` 方言**：`sa.ld`、`sa.st`、`sa.ex`、`sa.ve`（带下标模式、FUNC、REDUCE）、`sa.transpose`、
  `sa.loop`（对应 LOOP_END）、`sa.fence`、`sa.setreg`、`sa.ldparam`；
  类型上区分 SPAD / ACC 的地址空间与 bank。
- **流水线**：
  1. linalg 分块：按 SPAD / ACC 大小与 D 的倍数；
  2. 打包：A、B 条带的布局传播（LLM plan §10.3 的布局类型：紧凑向量、A 条带、B 条带、广播字、Kᵀ 条带），插入转换算子（TRANSPOSE、DIV-D 复制、经 DDR 的 LD INTERLEAVE）；
  3. 片上内存分配与 bank 双缓冲；
  4. 降级成 `sa` 方言；
  5. 调度：把 LD 提前、把 VE 与 EX 交错（按记分板规则判断能否并行）；
  6. 序列化成描述符。
- **验收**：去掉模板库，只用代码生成编译 stories15M，结果与 C3 逐位一致，速度不低于 C4。

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
| **C3 模板库** | §6.4 的全部模板；VE 表达式编译器；数据分块与权重打包；KV cache 原地更新 | IREE 编译的 stories15M 在板上生成的文本与 L5 手写路径完全相同，logits 逐位一致 | 大 |
| **C4 性能** | 精简 FENCE、跨 dispatch 预取、更大的 dispatch、代价模型 | 每 token 周期与手写路径相差 ≤ 10% | 中 |
| **C5 代码生成** | sa 方言与完整流水线（§8） | 不用模板库编译 stories15M，结果与 C3 一致 | 大 |

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
    ve_expr/              VE 表达式编译器（C3）
    dialect/              sa 方言与流水线（C5）
    test/                 lit 测试
  frontend/               C0：qllama.py（量化模型）、export.py（iree-turbine 导出）、对比脚本
  runtime/                C1：sa HAL 驱动（C，sa/），传输层 board / sim；tools/（sa-desc-v1）；test/
  sim/                    模拟器服务 sa_sim_server.py、命令环模拟 sa_board_emu.py（基于 llm/sa_funcsim.py）
  tests/                  端到端测试：test_c2.py（主机 sim）、board_c2.py（板上）
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
3. **可执行体的编码**：自定义二进制还是 flatbuffer。倾向 flatbuffer（IREE 惯例），内容按 §4.2。
4. **常量的存放**：参数文件（.irpa，推荐）还是嵌在 .vmfb 中。
5. **C4 的跨 dispatch 调度放在哪里**：驱动（录制命令缓冲时看到相邻的 dispatch）还是编译器（链接阶段）。倾向编译器，驱动保持简单。

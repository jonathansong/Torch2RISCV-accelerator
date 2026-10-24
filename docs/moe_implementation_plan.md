# KV260 上的 IREE 原生 MoE 实现方案

> Status: M0–M4e host-assisted two-layer Mixtral execution passes on KV260.
> Full-device IREE/SA inference is pending.
> 目标平台：AMD KV260 + PicoRV32 + SA accelerator
> 编译路径：PyTorch/Hugging Face → IREE Flow → SAHL → SAHW → descriptor list 或 RISC-V 代码
> 第一目标：不修改 GEMV RTL，实现 batch-1、W8A8、top-1/top-2 稀疏 MoE decode，并在模拟器和 KV260 上端到端验证。

## 1. 文档目标

本文给出在 Torch2RISCV-accelerator 中增加 Mixture-of-Experts（MoE）支持的工程方案，包括模型导入、权重打包、IREE dispatch、SAHL/SAHW、动态 expert 寻址、RISC-V 路由、KV260 内存规划、验证和性能优化。

第一版不重新设计整个加速器，而是复用现有 `sahl.gemv`、`sahw.ld mode=GEMV`、LDPARAM、动态 descriptor 字段、VE、RISC-V 代码生成和 host fallback。实施顺序遵循“先证明真正的稀疏执行，再扩展到通用 grouped GEMM”。

## 1.1 M0–M4 实现与验证结果

截至 2026 年 10 月，M0–M4b 已完成。M3 先验证 Top-2 路由、两次动态 GEMV 和 VE 加权合并，M3b 补齐完整 W8A8 gated expert FFN；M4a 验证 Hugging Face checkpoint 适配，M4b 使用公开 tiny Mixtral 的真实第 0 层权重，在模拟器和 KV260 上执行完整 Top-2 FFN。M0–M3 的基础板端验收使用 4-expert synthetic MoE，M4b 实际模型为 8 experts、`D=16`、`dim=1024`、`hidden=3584`。

| 阶段 | 已实现内容 | 验证环境 | 结果 |
|---|---|---|---|
| M0 | fp32/W8A8 reference、确定性 top-k、routing record ABI、统一 expert stride、权重 pack/manifest | Python/NumPy 单元测试 | 9 项测试通过；4 个 experts 均可独立寻址；layout pack/unpack 和 routing record round-trip 正确 |
| M1 | host-assisted top-1、LDPARAM 读取完整 byte offset、单一共享 GEMV 模板、动态 expert DDR 地址 | SA functional simulator | 4 个 expert ID 全部通过；实际 GEMV 地址等于 `expert_table_base + expert_offset`；输出与 int32 NumPy GEMV 完全一致 |
| M2 | PicoRV32 设备端 top-1 kernel、route record 写回、动态 GEMV 串接 | host-compiled RISC-V 测试、SA simulator、KV260 | 4 个 experts 全部 PASS；设备端 ID/offset/weight 与 reference 一致；板端完整 top-1 route→GEMV 通过 |
| M3 | PicoRV32 top-2、归一化 routing weights、两个动态 expert projection、VE weighted combine | SA simulator、KV260 | 4 组 expert pair 全部 PASS；tie 按较小 expert ID 决定；双 GEMV 加权结果与 reference 一致 |
| M3b | 每个 selected expert 的完整 `gate + up + SiLU + down` W8A8 FFN、hidden 重新量化、Top-2 ACC 合并 | SA simulator、KV260 | 4 组 expert pair 全部 PASS；完整 FFN 为 2717–2735 cycles；最大误差 `6.85453e-07` |
| M4a | Mixtral/Qwen2-MoE HF tensor schema adapter、单层 safetensors 导入、W8 pack 和 manifest | 分片 checkpoint 测试、SA simulator、KV260 | M0–M4a 共 23 项回归通过；HF-style Mixtral layer pack 到 M3b FFN 模拟器正确；KV260 schema acceptance PASS |
| M4b | 公开 TitanML/tiny-mixtral 第 0 层真实权重导入、真实尺寸 chunked GEMV、fp32/W8A8 routing 与输出报告 | SA functional simulator、KV260 | 8/8 随机向量 Top-2 routing agreement；KV260 四组真实 expert pair 全 PASS，最大绝对误差 `5.66244e-07` |

### 1.1.1 M0：reference 与参数布局

M0 冻结了第一版数值和 ABI 合约：

- `MoeConfig` 和 synthetic 4-expert 权重生成；
- fp32 expert 与 device-numeric W8A8 expert reference；
- 确定性 top-k，相同 score 时优先选择较小 expert ID；
- 32-byte routing record，包含 count、expert IDs、完整 byte offsets 和 fp32 weight bits；
- `sa-moe-parameters-v1` manifest；
- 每个 expert 按 `gate weight/scale → up weight/scale → down weight/scale` 排列；
- 64-byte section alignment 和统一 `expert_stride`。

当前 `dim=32`、`hidden=64` 配置的 projection 相对偏移为：

```text
gate.weight = 0
gate.scale  = 2048
up.weight   = 2304
up.scale    = 4352
down.weight = 4608
down.scale  = 6656
expert_stride = 6784 bytes
```

测试还覆盖了 zero activation、非法 top-k/shape、固定 expert 的 fp32 combine、所有 expert 的 W8A8 执行，以及 self-contained synthetic bundle。对应提交为 `16b42dd`。

### 1.1.2 M1：host-assisted 动态 top-1

M1 由 host 计算 top-1 并写入 routing record，descriptor 模板通过 LDPARAM 读取完整的 32-bit expert byte offset，再把它动态加到共享 GEMV 的 DDR 地址。4 个 experts 不再生成 4 份命令流，而是复用同一个 GEMV 模板。

功能模拟器验证：

- 每个 expert 都能由 score 单独选中；
- 每次只发出一个 GEMV；
- trace 中的权重地址严格等于 `expert_table_base + expert_offset`；
- GEMV int32 accumulator 与 NumPy 结果逐元素完全一致；
- W8A8 router reference 选择出的 expert 与设备执行的 expert 一致。

对应提交为 `9a29e23`。

### 1.1.3 M2：设备端 top-1

M2 把 top-1 从 host 下沉到 PicoRV32 kernel。kernel 直接读取 scores，生成固定 routing ABI；后续 descriptor 无需 ARM 解码 expert ID，即可通过动态地址执行选中的 expert projection。

模拟器覆盖所有 4 个 expert、tie 和 NaN 语义，并确认 kernel 是位置无关代码且小于 2 KiB。KV260 实机对 expert 0–3 全部验证通过，最近一次板端记录如下：

```text
expert 0: PASS, route cycles 4693, GEMV cycles 412
expert 1: PASS, route cycles 4774, GEMV cycles 407
expert 2: PASS, route cycles 4815, GEMV cycles 358
expert 3: PASS, route cycles 4977, GEMV cycles 360
moe_top1 board PASS
```

这证明了 `PicoRV32 route → routing record → LDPARAM → dynamic expert GEMV` 的完整设备链路。对应核心提交为 `50ea691`，板端完整 GEMV 验收随后补充。

### 1.1.4 M3：设备端 top-2 与 weighted combine

M3 将设备路由扩展到 top-2：PicoRV32 同时维护最大和次大 score，对 selected scores 做 softmax normalization，并输出两个 ID、offset 和 weight。descriptor 顺序执行两个动态 expert GEMV，VE 读取 routing weights 并完成 fp32 weighted combine。

KV260 实测的 4 组 expert pair 全部通过：

```text
experts 0,1: PASS, weights [0.8807970, 0.1192029], route cycles 49223, GEMV+combine cycles 727
experts 1,3: PASS, weights [0.8807970, 0.1192029], route cycles 48899, GEMV+combine cycles 686
experts 2,0: PASS, weights [0.8807970, 0.1192029], route cycles 48884, GEMV+combine cycles 701
experts 3,2: PASS, weights [0.8807970, 0.1192029], route cycles 48327, GEMV+combine cycles 691
moe_top2 board PASS
```

对应提交为 `494979f`。

### 1.1.5 M3b：完整 W8A8 gated expert FFN

M3 的首个板端版本重点验证两次动态 projection 与加权合并。M3b 在相同 routing ABI 和 expert layout 上补齐完整 expert：

```text
x(fp32)
  → dynamic A8 quantization
  → gate W8A8 GEMV ─┐
                    ├→ SiLU(gate) × up
  → up W8A8 GEMV ───┘
  → hidden dynamic A8 requantization
  → down W8A8 GEMV
  → routing weight multiply
  → Top-2 fp32 accumulate
```

实现中为直连 GEMV 生成紧凑 int8 activation layout；这与供矩阵 EX 单元使用的 lane-replicated layout 不同。输入 `x_q` 和 hidden `h_q` 放在不重叠的 SPAD_A 区域，两个 experts 复用同一份输入量化结果，并用 routing record 中的完整 byte offset 动态选择 gate/up/down 三组权重和 scale。

KV260 实测结果：

```text
experts 0,1: PASS, route cycles 49277, full FFN cycles 2717, max error 1.34110e-07
experts 1,3: PASS, route cycles 48829, full FFN cycles 2735, max error 2.38419e-07
experts 2,0: PASS, route cycles 48696, full FFN cycles 2726, max error 3.42727e-07
experts 3,2: PASS, route cycles 48281, full FFN cycles 2720, max error 6.85453e-07
moe_top2_ffn board PASS
```

完整 FFN 平均约为 2725 cycles，4 组测试的最大绝对误差均低于 `7e-7`。这验证了当前硬件无需新增 MoE RTL，即可正确执行稀疏 Top-2 完整 expert。对应提交为 `780aae7`。

M3b 小尺寸测试中路由约 48K–49K cycles，高于约 2.7K cycles 的 FFN；M4b 真实尺寸结果中，FFN 约 409K cycles，路由约 54K cycles，主要成本转为 expert 权重读取与分块 GEMV。后续优化应按目标模型尺寸分别 profile。

### 1.1.6 M4a：Hugging Face checkpoint adapter

M4a 增加 `compiler/frontend/moe_hf.py`，支持 Mixtral 和 Qwen2-MoE 常用 tensor naming、单文件与分片 safetensors、指定 layer 的按需读取，并将 gate/up/down 权重映射到 M0 统一表示后做 per-output-channel W8 量化和参数打包。单元测试以小型分片 Mixtral schema fixture 覆盖导入和 descriptor 模拟；M4a KV260 验收使用由该 schema 生成的 synthetic 权重，证明 checkpoint 文件形态与设备接口可连接，但还不代表公开模型权重已上板。

对应提交：`9f0a00a Add Hugging Face MoE checkpoint adapter`。M0–M4a 的 23 项测试全部通过。

### 1.1.7 M4b：真实 tiny Mixtral 第 0 层

真实 checkpoint 使用公开的 `TitanML/tiny-mixtral`。其权重为随机初始化的测试权重，因此本阶段验证架构和张量兼容、量化、路由和计算数值，不评估语言生成质量。模型配置为 2 层、8 experts、top-2、`dim=1024`、expert hidden `3584`。目前导入第 0 层；host 打包后的 expert table 为 88,358,976 bytes（约 84.2 MiB），每 token top-2 active expert 权重为 22,085,632 bytes（约 21.1 MiB），占该层 expert table 的 25%。

gate/up/down 输出宽度超过单条 GEMV 的 8-strip 上限，因此 M4b 将投影切分为多条动态 GEMV descriptor。完整 list 为 542 条 descriptor、34,688 bytes。下载权重 SHA-256：`2f01e93a605d534f122502244170604fda49d8bc39529db157f24ff008fdd797`。

8 个随机输入的 host 数值分析：Top-2 routing agreement 为 8/8；W8A8 层输出 relative L2 误差平均 2.98%、最大 3.60%；固定 fp32 route 的输出误差平均 2.95%、最大 3.55%。这些样本中 fp32 和 W8A8 router 选择了相同 expert IDs，约 3% 的输出误差主要来自 expert W8A8 计算。

真实参数先在 SA functional simulator 上通过，再由 KV260 执行。板端结果：

```text
experts 0,1: PASS, route cycles 54553, full FFN cycles 408647, max error 2.38419e-07
experts 1,2: PASS, route cycles 53985, full FFN cycles 408696, max error 4.47035e-07
experts 2,3: PASS, route cycles 54147, full FFN cycles 408575, max error 4.84288e-07
experts 3,4: PASS, route cycles 54277, full FFN cycles 408740, max error 5.66244e-07
MixtralForCausalLM layer 0 board PASS
```

完整双 expert FFN 平均约 408,665 cycles，路由约 54,000 cycles。该结果覆盖真实权重的一层 routed FFN，不代表全 Mixtral 模型生成或端到端 token-level board 测试；M4 的完整模型质量和容量退出条件仍待完成。对应提交：`d0afcc4 Run real-size Hugging Face MoE experts`。

## 2. 范围、假设与非目标

### 2.1 第一阶段范围

- decoder-only Transformer；
- 每层有一个 router 和若干结构相同的 gated FFN experts；
- batch size 1、单 token decode；
- top-1，随后扩展 top-2；
- expert 权重 W8、激活动态 A8；
- router/combine 使用 fp32 或现有 VE 支持的混合精度；
- 所有 expert 权重预先放入设备可寻址 DDR；
- 模拟器、RISC-V 路径和 KV260 可逐级验证。

建议第一个测试模型采用 4-expert synthetic MoE：`H=256/512`、expert intermediate size `512/1024`、一层、top-1。验证后再扩展 top-2、多层和真实 Hugging Face 模型。

### 2.2 后续范围

- Mixtral、Qwen MoE、Granite MoE 等架构适配；
- prefill、批处理、token grouping 和 grouped GEMM；
- shared expert、热门 expert 缓存和权重分层加载；
- 可 upstream 的通用 IREE MoE 表示。

### 2.3 第一阶段非目标

- 不新增 MoE 专用 RTL；
- 不一次支持所有 Hugging Face MoE 变体；
- 不实现 continuous batching、expert parallelism 或多 FPGA；
- 不支持超过 KV260 设备内存窗口的完整模型；
- 不将“运行所有 experts 再 mask”作为最终路径，因为它没有稀疏执行收益；
- 不依赖尚未实现的通用 IREE indirect dispatch。

## 3. 当前系统基础

### 3.1 可复用能力

| 能力 | 当前实现 | MoE 用途 |
|---|---|---|
| W8A8 linear | `qlinear`、权重 pack、linear micro-kernel | router 和 expert projection |
| 流式 GEMV | `sahl.gemv`、`sa_gemv.v` | batch-1 expert FFN |
| Gated FFN 融合 | `FormFfnRegions.cpp` | expert gate/up/activation/down |
| 动态 DDR 地址 | SAHW `dyn_fields`、`DYN_DMA_DDR` | 根据 expert ID 改变权重地址 |
| DDR 运行时参数 | `sahw.ldparam` / LDPARAM | 读取 ID、offset、动态长度 |
| 动态循环 | `sahl.loop` / `sahw.loop` | 多 expert、多 chunk 执行 |
| VE/reduction | `sahw.ve` | router 后处理和输出合并 |
| RISC-V 后端 | `SahwToRiscv.cpp` | top-k、地址计算、数据依赖控制 |
| host fallback | SA + VMVX variant | 原型期执行未下沉的 top-k |
| simulator/oracle | `sa_funcsim.py`、dispatch oracle | 数值与 descriptor 验证 |

### 3.2 当前缺口

- `compiler/frontend/qhf.py` 只加载 dense Llama/Qwen FFN；
- FFN matcher 不理解 router/expert 轴；
- SA HAL 的 indirect dispatch arguments 尚未实现；
- BASE/PARAM 和每条 descriptor 的动态字段数量有限；
- LDPARAM 的乘数/字段存在编码宽度限制；
- top-k 没有正式设备 kernel；
- 完整 MoE 权重仍要占用 DDR，MoE 只减少每 token 的活跃读取量。

### 3.3 关键判断

batch-1 decode 的核心不是 grouped GEMM，而是：

```text
router GEMV + top-k + k 个 gated FFN GEMV + weighted sum
```

现有流式 GEMV 很适合这个路径。grouped GEMM 是 prefill/批处理优化，不应成为 decode MVP 的前置条件。

## 4. MoE 工作负载与编译挑战

典型 MoE 层：

```text
logits[E]       = x[H] × W_router[E, H]
(ids, weights)  = topk_and_normalize(logits, K)

for j in 0 .. K-1:
  e    = ids[j]
  gate = x × W_gate[e]
  up   = x × W_up[e]
  h    = activation(gate) * up
  y_e  = h × W_down[e]

y = Σ_j weights[j] * y_e[j]
```

主要挑战是 expert ID 由数据决定、权重地址动态变化、top-2 需要多次执行和合并，以及 prefill 时每个 expert 得到的 token 数不同。若为每个 expert 生成独立 executable，会造成 code size 和调度开销；若执行所有 experts，则失去稀疏 MoE 的意义。

因此 decode 采用“一个 expert FFN 模板 + 动态权重基址”，而不是“每 expert 一个 dispatch”。

## 5. 总体架构

```text
PyTorch/Hugging Face MoE
    ↓ torch.export / Turbine
IREE Flow：router、top-k、expert FFN、combine
    ↓ SA target preprocessing
SAHL：gemv、topk、dynamic expert view、weighted combine
    ↓
SAHW：GEMV + LDPARAM/dynamic DDR，或生成 RISC-V 控制流
    ↓
PicoRV32 + descriptor fetch + SA GEMV/VE
    ↓
KV260 DDR 中的 expert 权重、routing record 和结果
```

保留两条后端：

- **Descriptor 路径**：命令发射成本低，适合固定 top-k 和规则 expert layout。由 LDPARAM 读取 offset，并用 `DYN_DMA_DDR` 加到 GEMV 地址；
- **RISC-V 路径**：适合 top-k、32 位地址运算、循环和模型特有路由语义。RISC-V 读取 routing record、计算地址并发出 GEMV/VE 命令。

第一版可以用 descriptor 完成 top-1；top-2 和复杂地址优先走 RISC-V。两条路径必须共享 oracle 测试。

## 6. 模型前端与权重格式

### 6.1 配置和架构适配

在 `HFConfig` 或新 `MoeConfig` 中增加：

```python
num_experts: int
num_experts_per_tok: int
moe_intermediate_size: int
router_score_function: str       # softmax/sigmoid/identity
router_normalize_topk: bool
shared_expert_intermediate_size: int | None
```

建立架构适配层，把不同 checkpoint 的权重名称统一为：

```text
layers.L.router
layers.L.experts.E.gate
layers.L.experts.E.up
layers.L.experts.E.down
layers.L.shared_expert.*          # 可选
```

适配器负责 expert/projection 维度顺序、gate/up 合并权重、router bias、shared expert、safetensors 分片和模型特有的 normalization。

### 6.2 Packed layout

每层连续存放 router 和 expert table：

```text
layer L:
  router
  expert 0: gate, up, down, scales
  expert 1: gate, up, down, scales
  ...
```

每个 projection 沿用当前 GEMV packed layout：

```text
expert_stride = align(gate + gate_scales + up + up_scales +
                      down + down_scales, 64)
expert_base(e) = layer_expert_base + e * expert_stride
```

所有 section 至少 64 字节对齐，GEMV 权重起点至少 16 字节对齐。manifest 必须记录 offset、size、shape、dtype、scale layout 和 expert stride。

### 6.3 Reference model

增加 fp32、device-numeric W8A8 和固定 routing 三种模式。固定 routing 可把 router/top-k 错误与 expert FFN 错误分离：

```python
def moe_layer(x, router, experts, top_k):
    scores = router(x)
    ids, weights = route(scores, top_k)
    y = sum(w * experts[e](x) for e, w in zip(ids, weights))
    return y, ids, weights
```

## 7. IREE 与 Flow 层设计

### 7.1 MVP 表示

第一版由项目前端生成容易识别的 router、top-k、expert table、gated FFN 和 combine，不立即匹配任意框架展开出的 scatter/gather 图。必要时保留清晰函数边界或项目私有 op，再由 SA preprocessing 展开。

逻辑上形成：

```text
router → top-k → selected expert gate/up → activation → down → combine
```

实现初期可拆成 router、routing、expert FFN、combine 四个 dispatch；性能稳定后再融合，减少 DDR 中间值和 dispatch overhead。

### 7.2 不采用每 expert 一个 dispatch

SA command buffer 当前不支持通用 indirect dispatch arguments。即使补上，为每个 expert 创建 executable 也会使 executable 数量、code size 和参数 patch 开销随 E 增长。应让所有 experts 复用一个结构模板，仅动态改变权重基址。

### 7.3 成熟版 pattern

新增 `FormMoeRegions.cpp`，按语义匹配：

- router contraction 的输出维是 expert 数；
- top-k/select/sort 产生 IDs 和 weights；
- expert 权重具有 expert 轴；
- selected experts 执行结构相同的 gated FFN；
- 输出沿 K 维加权 reduction。

模式不完整时保留通用 IREE lowering或走受控 host fallback，并给出具体诊断。

## 8. SAHL 中间表示设计

### 8.1 MVP

先复用基础操作：

- `sahl.gemv`：router 和 expert projections；
- `sahl.load/store`：routing buffer；
- `sahl.loop`：多个 selected experts；
- `sahl.kernel "linear"`：projection；
- VE：activation、scale、multiply、sum。

给 expert weight view 添加动态寻址属性：

```mlir
sa.expert_stride = ...
sa.expert_id_source = ...
sa.projection_offset = ...
```

### 8.2 正式扩展

MVP 稳定后增加语义 op：

```mlir
%ids, %weights = sahl.topk %scores {k = 2, normalize = true}
%ys = sahl.expert_ffn %x, %ids, %expert_weights
    {experts = 8, expert_stride = ..., activation = "silu"}
%out = sahl.weighted_combine %ys, %weights
```

`sahl.topk` 不绑定具体执行单元；`sahl.expert_ffn` 表达同一模板与动态 expert base；所有新 op 都必须能展开回基础 GEMV/VE，方便验证与 fallback。

MVP verifier 要求 decode 输入除末维外均为 1，K 为 1 或 2，所有 expert shape 相同，GEMV K 维满足 D=16 对齐或显式 padding，expert ID 落在 `[0,E)`。

## 9. SAHW 与 descriptor 设计

### 9.1 动态寻址

```text
expert_offset = expert_id * expert_stride
weight_addr   = expert_table_base + expert_offset
              + projection_offset + chunk_offset
```

Descriptor 路径：

```text
LDPARAM expert_id/offset
SETREG  expert_offset
LD/GEMV DDR = static_offset + PARAM(expert_offset), add=true
```

若 stride 超出 LDPARAM 乘数宽度，不能截断。推荐让 top-k kernel 直接输出完整的 32 位 byte offset，或自动切换到 RISC-V 路径。

### 9.2 top-k 执行

固定 top-2 先展开两次调用：

```text
run expert(ids[0]); scale output by weights[0]
run expert(ids[1]); scale-add output by weights[1]
```

通用 K 再使用 `sahw.loop` 或 RISC-V loop。PARAM 应通过 liveness 复用；寄存器不足时诊断并切换 RISC-V，不能静默生成错误 descriptor。

### 9.3 同步

DDR 不受本地 scoreboard 跟踪，必须在以下位置显式 fence：

- router scores store → top-k load；
- routing record store → LDPARAM/load；
- 写回 DDR 的中间结果 → 后续读取。

expert FFN 的中间值应尽可能保留在 SPAD/ACC，避免 fence 和 DDR 往返。

## 10. RISC-V 路由与调度设计

### 10.1 Top-k kernel

从现有 argmax kernel 扩展 top-k。对 4–64 个 experts，一次顺序扫描即可；top-2 同时维护最大和次大值。必须明确相等 score 时的 expert ID 顺序、NaN/Inf 处理、softmax/sigmoid 语义以及是否对 selected weights 重新归一化。

实现递进：

1. host fallback，只用于 bring-up；
2. 独立 RISC-V top-k kernel；
3. 编译生成的 MoE RISC-V 函数，完成 top-k、地址计算、expert 命令和 combine。

### 10.2 Routing record ABI

```c
struct sa_moe_route {
  uint32_t count;
  uint32_t expert_id[MAX_K];
  uint32_t expert_offset[MAX_K];
  uint32_t weight_bits[MAX_K];  // fp32 bit pattern
};
```

ABI 版本写入 executable metadata。输出 byte offset 能避免 descriptor 端进行大整数乘法，并把模型 layout 知识留在编译器生成的路由代码中。

## 11. GEMV 硬件复用与 RTL 需求

### 11.1 第一版不改 RTL

每个 expert projection 都映射到现有 GEMV：

```text
gate = GEMV(x, W_gate[expert])
up   = GEMV(x, W_up[expert])
h    = SiLU(gate) * up
down = GEMV(quantize(h), W_down[expert])
```

四端口 GEMV、x residency、输出 strip 分块、ACC 和 VE epilogue 都可复用。动态 expert 只改变 DDR 地址，不改变 GEMV 算术。

### 11.2 可选 RTL 优化

仅在 profile 证明必要时考虑 expert 地址预取、base table lookup、top-k 单元、两个 expert 的 fused weighted accumulation 或 expert weight cache。第一阶段的主要成本仍是权重读取，top-k 计算量很小，专用 RTL 通常不是最高优先级。

## 12. 内存布局与容量规划

一个 gated expert 的 int8 权重近似为：

```text
bytes_per_expert ≈ 3 × hidden_size × expert_intermediate_size
bytes_per_layer  ≈ num_experts × bytes_per_expert + router_bytes
```

MoE 减少 active bytes，不减少总模型容量。编译前由 `mem_preflight.py` 分别报告 packed parameters、KV cache、mutable globals、executable/code、temporary arena、routing buffers 和安全余量。

建议 DDR 布局：

```text
model parameters:
  dense attention weights
  layer 0 expert table
  layer 1 expert table ...
mutable state:
  KV cache, hidden buffers, router scores, route records
runtime:
  descriptor/RISC-V code, ring, completion records
```

设备对象必须位于当前 32 位地址限制允许的低 2 GB 和配置的 u-dma-buf window。容量不足时依次考虑小模型/截断层、W4 expert、按层换入、热门 expert 常驻和冷 expert 换入；第一版不实现换入换出。

## 13. Decode 执行方案

### 13.1 Top-1

```text
1. x → router GEMV → scores[E]
2. top1(scores) → expert_id、expert_offset、weight
3. 生成一次 x_q，并在所有 selected experts 间复用
4. 动态地址 GEMV gate
5. 动态地址 GEMV up
6. VE: SiLU(gate) × up
7. 动态量化 h
8. 动态地址 GEMV down
9. VE: output × routing_weight
10. residual
```

若模型规定 top-1 weight 恒为 1，可消除第 9 步。

### 13.2 Top-2

```text
expert(ids[0]) → y0 × w0 ─┐
expert(ids[1]) → y1 × w1 ─┴→ add → y
```

顺序执行两个 experts，复用 x/x_q。`y0*w0` 写入最终 accumulator，`y1*w1` 直接 accumulate，不同时保存两个完整 fp32 hidden vector。执行顺序必须固定，以保持 fp32 加法可重复。

### 13.3 FFN fusion

现有 `FormFfnRegions.cpp` 已识别 dense decode gated FFN。MoE lowering 应把动态 expert base 作为 weight view 的偏移，而不改变 contraction 结构。可以先建立动态 view，再复用现有 fusion；若 IREE 图形态不稳定，再增加 MoE-specific matcher 直接构造相同的 SAHL kernel。

## 14. Prefill 与 grouped GEMM 演进方案

prefill 时输入为 `T×H`，各 token 路由不同。高效执行需要：

1. 计算所有 token 的 router scores 和 top-k；
2. 统计每个 expert 的 token 数；
3. prefix sum 得到每组 offset；
4. 按 expert gather/reorder tokens；
5. 对每个非空 expert 执行 `M_e×H` 的计算；
6. scatter 回原 token 顺序并加权合并。

`M_e` 在运行时决定，这才是完整 grouped matrix multiplication。建议演进：

- P0：prefill 暂时走 VMVX/LLVM CPU；
- P1：设备 router/top-k，逐 token GEMV 保证正确；
- P2：按 expert grouping 后循环 GEMV；
- P3：`M_e >= D` 时选择 systolic GEMM；
- P4：增加 `sahl.grouped_gemm`，自动选择 GEMV/GEMM；
- P5：grouping、expert compute 和 scatter 流水化。

## 15. Runtime 与主机接口

最终热路径不应让 ARM 逐层读取 router 结果。ARM 负责模型加载、buffer 分配、command buffer 提交和最终 logits；PicoRV32/accelerator 负责 router、top-k、expert 地址、FFN 和 combine。

Phase 1 可使用 host top-k，但必须写入固定 routing ABI、在 profile 中可见，并保证替换为 RISC-V kernel 时无需修改模型与权重接口。

Executable/manifest 至少记录：MoE ABI version、层数、expert 数、top-k、expert stride、projection/scale offsets、router 语义、所需 GEMV/RISC-V capabilities 和设备内存需求。

## 16. 数值语义与量化方案

### 16.1 Expert

- weights：int8 symmetric、per-output-channel fp32 scale；
- input：per-token dynamic int8 scale；
- accumulation：int32；
- activation、scale、combine、residual：fp32 VE；
- down projection 输入：重新动态量化；
- 多个 experts 复用同一份 x_q 和 `s_x`。

### 16.2 Router

Router 对 expert 选择非常敏感。实现顺序应为 fp32 reference、设备 fp32/混合精度、W8A8 router。若 W8A8 导致 routing agreement 明显下降，router 保留高精度而 experts 继续 W8A8。

不同模型可能采用全量 softmax 后 top-k、top-k 后重新 softmax、sigmoid、不归一化 selected weights、routing bias 或 shared expert。前端必须将实际语义编码进 IR/metadata，不能统一假设为 Mixtral 语义。

## 17. 验证与测试计划

### 17.1 建议测试文件

```text
compiler/plugins/sa/test/sa_form_moe_regions.mlir
compiler/plugins/sa/test/sahl_topk.mlir
compiler/plugins/sa/test/sahl_expert_gemv.mlir
compiler/plugins/sa/test/sahw_moe_dynamic_addr.mlir
compiler/tests/test_moe_frontend.py
compiler/tests/test_moe_dispatch.py
compiler/tests/test_moe_e2e.py
compiler/tests/board_moe.py
```

覆盖 expert 0/最后一个、top-1/top-2、score tie、NaN/Inf、offset 跨 4 KB、最大 stride、维度 padding、连续选择同一 expert、重复 ID、非法 ID 和 descriptor/RISC-V 等价。

### 17.2 分层 oracle

```text
PyTorch fp32
 → PyTorch device-numeric W8A8
 → exported IREE/MLIR
 → dispatch oracle
 → SA functional simulator
 → RTL simulation（如修改 RTL）
 → KV260
```

失败时保存 hidden、router scores、IDs、weights、projection accumulators、expert output 和 combined output。

提供 `--force-experts` 测试模式绕过 router，分别验证 expert 0、最后一个 expert、top-2 和随机动态地址。

### 17.3 回归要求

- MoE pass 仅在检测到 MoE 或显式开关启用时运行；
- dense golden descriptor corpus 保持不变，除非变更明确批准；
- 现有 SmolLM2、Gemma、Qwen dense、Llama 测试继续通过；
- 无 MoE 的 executable 不增加 metadata、PARAM 压力或运行时成本。

## 18. 性能建模与优化

### 18.1 Decode 性能下界

```text
active_weight_bytes/token
  ≈ layers × top_k × 3 × H × I
    + dense_attention_weight_bytes
    + router_weight_bytes

token/s upper bound
  ≈ effective_DDR_bandwidth / active_weight_bytes_per_token
```

报告中必须同时列出总参数和 active 参数。总模型很大不等于每 token 都读取全部参数，但所有权重仍占用内存容量。

### 18.2 必测指标

- router、top-k、gate/up/down、combine cycles；
- 实际 expert weight bytes；
- GEMV 端口利用率与 x-residency hit；
- descriptor fetch 或 PCPI 发射开销；
- fence 等待；
- 每层 expert 选择分布；
- device memory peak、tokens/s、wall time；
- routing agreement、token agreement 和 PPL。

### 18.3 优化顺序

1. 确保只读取 active experts；
2. 多个 experts 复用 x_q；
3. gate/up 相邻调度并保留本地中间值；
4. top-2 直接在 ACC 中加权累积；
5. 减少 router/top-k 的 DDR round trip；
6. expert 权重地址预取；
7. 热门 expert locality/cache；
8. prefill token grouping/grouped GEMM。

功能路径稳定前不引入 expert cache，以免同时增加替换、一致性和容量问题。

## 19. 分阶段实施计划

### M0：Synthetic reference 和布局冻结

交付 4-expert synthetic 模型、fp32/W8A8 reference、固定 routing、packed layout、manifest 和容量报告。退出条件：每个 expert 的 indexed 执行与独立 dense 执行一致。

### M1：Host-assisted top-1

交付 router scores、host top-1、routing ABI、动态 expert offset、现有 GEMV 和模拟器端到端执行。退出条件：所有 expert IDs 通过，只读取选中 expert 权重。

### M2：设备端 top-1

交付 RISC-V top-1 kernel、全设备 router→route→expert 路径、KV260 profile。退出条件：ARM 不读取 router scores，board 与 sim 一致，dense 回归不变。

### M3：Top-2 和 weighted combine

交付 top-2 kernel、两次动态 expert FFN、fp32 weighted accumulation 和 tie/NaN/normalization 测试。退出条件：IDs、weights、输出与 reference 一致，只读取两个 active experts。

### M4：真实 Hugging Face MoE

Status: complete for the two-layer `TitanML/tiny-mixtral` batch-1 decode
acceptance. Mixtral/Qwen2-MoE tensor adapters, capacity preflight, real-token
routed FFN acceptance, and hybrid propagation passed earlier. M4f runs the
complete two-layer dense/attention/router/expert computation on the SA
simulator and KV260. All three teacher-forced top tokens and all six expert
pairs match the Hugging Face reference; board logits agree with the SA
simulator. Native IREE MoE compilation belongs to M5.

#### M4f complete SA decoder (2026-10-24)

`compiler/frontend/moe_device_dense.py` packs the dense Mixtral checkpoint
to W8A8, including an 8-expert router padded to one D=16 output tile. It
permutes Hugging Face split-half RoPE query/key rows into adjacent pairs so
the existing SA RoPE primitive works without changing attention dot products.
`llm/compile_model.py` now handles grouped-query attention (GQA), and the
16-token D=8 regression is bit-exact against `DeviceModel` for both ordinary
attention and GQA, including the int8 KV cache. The real tiny Mixtral shape
(dim 1024, 32 query heads, 8 KV heads, head size 32) fits the D=16 local
memory plan with 2040 of 2048 ACC bank words used.

`compiler/runtime/tools/moe_model.py` emits three SA descriptor stages per
token: layer 0, layer 1, and the final norm/classifier. Each layer executes
embedding or prior FFN residual, RMSNorm, QKV projection, RoPE, GQA with KV
cache, output projection, residual, second RMSNorm, and router projection.
The already validated Top-2 route kernel and two-expert FFN run between
stages. `compiler/tests/moe_full_device_sim.py` uses the functional simulator
for all arithmetic; because it cannot run the RISC-V firmware, it constructs
the Top-2 route record on the host from SA-produced scores. On KV260,
`board_moe_full_device.py` calls that RISC-V kernel directly and binds the
FFN output to its input buffer after the descriptor has loaded the input.
ARM submits work and reads results; it does not calculate model activations,
attention, router scores, expert outputs, or logits.

The simulator ran token IDs `[1, 42, 43]` against the cached random-initialized
`TitanML/tiny-mixtral` Hugging Face model:

| Position | SA top token | HF top token | Logit relative L2 vs HF |
|---:|---:|---:|---:|
| 0 | 25591 | 25591 | 0.036509 |
| 1 | 13899 | 13899 | 0.036336 |
| 2 | 26962 | 26962 | 0.323339 |

The position-2 logit gap is material despite token agreement, consistent
with the earlier expert-only W8A8 hybrid result. No language quality claim
follows from this random checkpoint and short sequence. The packed dense file
is 71,240,512 bytes; one reusable packed expert layer is 88,358,976 bytes;
the simulator DDR image is 159,858,688 bytes. All six simulator expert pairs
agree with the saved Hugging Face routing trace.

The KV260 run reported `full-device Mixtral board PASS`. Its copied
`build/moe/m4f/board/board_full_device_report.json` contains the complete
per-layer cycles, routes, logit comparisons, and memory report:

| Position | Board / SA sim / HF top token | Board vs sim logit relative L2 | Max absolute error | Layer 0 / 1 experts |
|---:|---:|---:|---:|---|
| 0 | 25591 | 5.82037e-7 | 1.90735e-6 | `[7,1]` / `[6,2]` |
| 1 | 13899 | 3.69794e-7 | 1.19209e-6 | `[7,1]` / `[2,5]` |
| 2 | 26962 | 3.42988e-7 | 9.53674e-7 | `[7,0]` / `[2,6]` |

The six layer calls took 227,772–228,373 dense cycles, 51,276–53,264
RISC-V route cycles, and 408,527–408,643 routed expert FFN cycles each.
Actual allocated board buffers totaled 159,815,570 bytes: 71,240,512 dense
parameter bytes, one 88,358,976-byte reusable expert slot, 196,608 IO bytes,
16,384 KV bytes, and smaller route/kernel buffers. `/proc/meminfo` reported
`CmaTotal` 1,572,864 KiB and `CmaFree` 2,776 KiB before the runner's buffer
allocations versus 744 KiB after. That counter change is not the sum of the
allocated buffers, so it should not be interpreted as the model footprint;
the successful allocations and their recorded sizes are the direct evidence
that this configuration fits the board's available memory path.

Host reproduction:

```bash
build/iree/venv/bin/python llm/test_l4.py --d 8 --tiny-tokens 16
build/iree/venv/bin/python compiler/frontend/moe_device_dense.py
build/iree/venv/bin/python compiler/tests/moe_full_device_sim.py
build/iree/venv/bin/python compiler/tests/moe_stage_full_device_board.py
```

To repeat the passed board acceptance, transfer the self-contained bundle:

```bash
scp -r build/moe/m4f/board ubuntu@192.168.0.121:~/moe-m4f
# On the KV260:
cd ~/moe-m4f
chmod +x run_board.sh
./run_board.sh
```

The board runner requires all three board/simulator/HF top tokens and all six
board/simulator expert pairs to agree. Board/simulator logit relative L2 and
maximum absolute error must remain below 0.01. It writes
`board_full_device_report.json`.

#### M4c capacity preflight (2026-10-24)

`compiler/tests/moe_mem_preflight.py` now estimates the full checkpoint footprint
from safetensors headers, without loading tensor values. It separates experts,
router, dense projections, embeddings, scales, and other parameters. The estimate
assumes W8 linear weights, per-row fp32 scales, int8 embeddings and KV cache,
64-byte alignment, a 2.5% packing allowance, and a configurable runtime margin.
These are planned full-model representations, not a measured full-model binary.

For the cached `TitanML/tiny-mixtral` checkpoint (2 layers, 8 experts per layer,
top-2, dim 1024, expert hidden 3584, 8 KV heads of dimension 32):

| Context | Parameter estimate | KV cache | Runtime margin | Total | 1280 MiB pool |
|---|---:|---:|---:|---:|---|
| 2048 | 242.2 MiB | 2.0 MiB | 64 MiB | 308.2 MiB | Fits by estimate |
| 131072 | 242.2 MiB | 128.0 MiB | 64 MiB | 434.2 MiB | Fits by estimate |

Expert weights and scales occupy approximately 84.2 MiB per layer. Selecting two
of eight experts accesses approximately 42.1 MiB of expert weights/scales across
the two layers per decode token, assuming no weight cache. This is a byte estimate,
not measured bandwidth. The 131072-row result only checks memory; it does not
establish runtime support, descriptor limits, or acceptable latency at that context.

The board configuration in `memory_model.md` reserves a 1280 MiB u-dma-buf pool,
while ordinary `pynq.allocate` uses the separate 256 MiB CMA pool. The current
single-layer Python MoE runner uses CMA. A complete model therefore needs the
u-dma-buf allocation path (as in `board_launcher.UdmabufWindow`) or another explicit
loading strategy; increasing a size argument to `pynq.allocate` is insufficient.
Check the actual board pool size and free allocations before deployment.

Reproduce from the repository root:

```bash
build/iree/venv/bin/python compiler/tests/moe_mem_preflight.py \
  build/llm_cache/TitanML-tiny-mixtral \
  --kv-len 2048 --pool-mib 1280 --margin-mib 64
```

Both context estimates completed successfully. The 256 MiB pool check correctly
returned exit status 1 (52.2 MiB short at context 2048), and zero context returned
an argument error. JSON reports are local under `build/moe/m4b/` as
`tiny-mixtral-preflight-t2048.json` and `tiny-mixtral-preflight-maxctx.json`.

#### M4c real-token layer traces (2026-10-24)

`compiler/tests/moe_hf_token_trace.py` runs the complete fp32 Hugging Face
decoder for a token sequence, captures the post-attention MoE input, router
scores, expert IDs/weights, and FFN output for each selected layer, then compares
the imported fp32 and W8A8 MoE implementations. For token IDs `[1, 42, 43]`
from the random-initialized `TitanML/tiny-mixtral` checkpoint:

| Layer | HF/SA expert IDs | fp32 vs HF relative L2 | W8A8 vs HF relative L2 | SA simulator vs W8A8 max absolute error |
|---|---|---:|---:|---:|
| 0 | `[7, 0]` | `5.63e-7` | `0.03436` | `0.0006171` |
| 1 | `[2, 0]` | `5.42e-7` | `0.02780` | `6.56e-7` |

Both routed layer FFNs passed the SA functional simulator using actual decoder
activations and scores. The original M4b forced-score simulator acceptance also
still passes. The full fp32 Hugging Face model's final argmax for this sequence
is token ID `26962`; no SA model-level token result exists yet for comparison.
The layer 1 input is captured from the fp32 Hugging Face upstream layers, so its
standalone test does not measure propagation of W8A8 error from layer 0.

Host reproduction:

```bash
build/iree/venv/bin/python compiler/tests/moe_hf_token_trace.py \
  --model build/llm_cache/TitanML-tiny-mixtral \
  --out build/moe/m4c/tiny-mixtral --tokens 1,42,43
build/iree/venv/bin/python compiler/frontend/moe_hf.py \
  --model build/llm_cache/TitanML-tiny-mixtral \
  --out build/moe/m4c/tiny-mixtral/layer0 --layer 0
build/iree/venv/bin/python compiler/frontend/moe_hf.py \
  --model build/llm_cache/TitanML-tiny-mixtral \
  --out build/moe/m4c/tiny-mixtral/layer1 --layer 1
build/iree/venv/bin/python compiler/tests/moe_m4b_sim.py \
  build/moe/m4c/tiny-mixtral/layer0 \
  --reference build/moe/m4c/tiny-mixtral/layer0/moe_token_acceptance.npz
build/iree/venv/bin/python compiler/tests/moe_m4b_sim.py \
  build/moe/m4c/tiny-mixtral/layer1 \
  --reference build/moe/m4c/tiny-mixtral/layer1/moe_token_acceptance.npz
```

Two standalone KV260 bundles were prepared at `build/moe/m4c/board/layer0`
and `layer1`. Each reuses the M4b bitstream, firmware, and Top-2 descriptor,
with its own packed layer weights and real-token acceptance vectors. The bundle
includes the descriptor's local `compile_layer.py` dependency. The packaging
script now builds the descriptor from inside each bundle as a host
smoke check, so an omitted local Python module fails before board transfer.
Copy the parent directory and run each layer separately:

```bash
scp -r build/moe/m4c/board ubuntu@192.168.0.121:~/moe-m4c
# On the board:
cd ~/moe-m4c/layer0 && ./run_board.sh
cd ~/moe-m4c/layer1 && ./run_board.sh
```

The user ran both bundles on KV260 on 2026-10-24 and reported:

| Layer | Expert IDs | Route cycles | Full FFN cycles | Max absolute error vs W8A8 host golden | Result |
|---|---|---:|---:|---:|---|
| 0 | `[7, 0]` | 53,008 | 408,694 | `0.000617288` | PASS |
| 1 | `[2, 0]` | 55,287 | 408,591 | `7.15256e-7` | PASS |

These are real-token activations and router scores produced by the full fp32
Hugging Face forward pass, followed by standalone SA route and expert FFN
execution on each layer. They do not yet prove that a W8A8 layer 0 output can
feed layer 1 on the device while preserving the model's final token.

M4c remains in progress. Model-level SA execution still needs attention,
residual/KV state, sequential execution of both routed FFNs, and the output
head. Its teacher-forced logits and token IDs must then be compared with the
complete Hugging Face reference on the simulator and KV260. The current board
bundles check one FFN layer at a time, with HF-supplied input and router scores.

#### M4d hybrid decoder propagation (2026-10-24)

`compiler/tests/moe_hf_hybrid_sim.py` runs a complete two-layer Hugging Face
forward pass twice on token IDs `[1, 42, 43]`. The reference uses fp32 Hugging
Face experts. The hybrid run replaces every routed FFN output with the SA
functional simulator's W8A8 descriptor output. Its later layers therefore see
the earlier SA result. Hugging Face still supplies embeddings, attention,
residuals, normalization, and the output head; host code computes top-2 from
the Hugging Face router scores. Six SA FFN calls run in total.

| Position | HF top token | Hybrid top token | Logit relative L2 |
|---|---:|---:|---:|
| 0 | 25591 | 25591 | 0.01814 |
| 1 | 13899 | 13899 | 0.02252 |
| 2 | 26962 | 26962 | 0.32353 |

All three top tokens agree for this random-initialized test checkpoint, but the
last position's logits differ substantially (`0.90051` maximum absolute
error). This is an observation on one short sequence, not a quality or
full-device token agreement claim. The hybrid output is saved locally in
`build/moe/m4c/hybrid-sim/report.json` and `teacher_logits.npz`.

For every layer and position the script also calculates an independent W8A8
host golden. All six SA simulator FFNs passed against it (maximum absolute
error `0.0006171` in layer 0 and `1.3113e-6` in layer 1). The emitted
`layer0/moe_acceptance.npz` and `layer1/moe_acceptance.npz` contain all three
positions, including the layer 1 inputs after layer 0 SA outputs propagated
through the decoder. The corresponding KV260 bundles already contain these
vectors as `hybrid_acceptance.npz` and a `run_hybrid.sh` launcher.

To repeat the host run and simulator acceptance:

```bash
build/iree/venv/bin/python compiler/tests/moe_hf_hybrid_sim.py
build/iree/venv/bin/python compiler/tests/moe_m4b_sim.py \
  build/moe/m4c/tiny-mixtral/layer0 --cases 3 \
  --reference build/moe/m4c/hybrid-sim/layer0/moe_acceptance.npz
build/iree/venv/bin/python compiler/tests/moe_m4b_sim.py \
  build/moe/m4c/tiny-mixtral/layer1 --cases 3 \
  --reference build/moe/m4c/hybrid-sim/layer1/moe_acceptance.npz
```

Only the small files below need to be added to the M4c board directories:

```bash
scp build/moe/m4c/board/layer0/{hybrid_acceptance.npz,run_hybrid.sh} \
  ubuntu@192.168.0.121:~/moe-m4c/layer0/
scp build/moe/m4c/board/layer1/{hybrid_acceptance.npz,run_hybrid.sh} \
  ubuntu@192.168.0.121:~/moe-m4c/layer1/
# On KV260, run each layer separately:
cd ~/moe-m4c/layer0 && chmod +x run_hybrid.sh && ./run_hybrid.sh
cd ~/moe-m4c/layer1 && chmod +x run_hybrid.sh && ./run_hybrid.sh
```

The user ran all six propagated-vector cases on KV260 on 2026-10-24:

| Layer | Position | Experts | Route cycles | Full FFN cycles | Max absolute error vs W8A8 host golden | Result |
|---|---:|---|---:|---:|---:|---|
| 0 | 0 | `[7, 1]` | 52,353 | 408,646 | `0.000460662` | PASS |
| 0 | 1 | `[7, 1]` | 53,155 | 408,538 | `1.07288e-6` | PASS |
| 0 | 2 | `[7, 0]` | 52,938 | 408,680 | `0.000617288` | PASS |
| 1 | 0 | `[6, 2]` | 51,437 | 408,684 | `8.68738e-6` | PASS |
| 1 | 1 | `[2, 5]` | 52,755 | 408,543 | `4.41074e-6` | PASS |
| 1 | 2 | `[2, 6]` | 53,110 | 408,631 | `5.36442e-7` | PASS |

The layer 1 inputs incorporate layer 0 SA simulator outputs through the Hugging
Face attention and residual path. Each board call still runs one route and FFN
in isolation; the model's final logits and token are calculated on the host
hybrid path, not on KV260.

The remaining full-device M4 work is to place attention, residuals, KV cache,
both MoE layers, and the output head into one SA execution flow, then compare
its teacher-forced logits and token IDs with the Hugging Face reference.

#### M4e one-process two-layer KV260 model runner (2026-10-24)

`compiler/frontend/moe_mixtral_numpy.py` implements a portable batch-1
Mixtral host path: embedding, fp32 RMSNorm, causal grouped-query attention with
RoPE, residuals, router projection, and the output head. Its FFN callback can
use fp32 experts, the SA functional simulator, or the KV260 device. For token
IDs `[1, 42, 43]`, the pure fp32 NumPy path matches Hugging Face logits with
relative L2 below `1e-6` at every position. With SA simulator FFNs it matches
the M4d Hugging Face/SA hybrid logits with relative L2 below `8e-7` at every
position. Both comparisons produce the same three top tokens.

`compiler/tests/moe_pack_dense_checkpoint.py` extracts the 17 tensors needed
by this host path into a 283,203,440-byte (271 MiB) safetensors checkpoint.
No expert matrices are included. The reduced checkpoint also reproduces the
M4d hybrid logits when used with the two existing packed expert bundles.

`compiler/tests/board_moe_model.py` keeps one overlay and reuses one packed
parameter buffer: it loads layer 0 weights once for three routed FFN calls,
then layer 1 weights once for three calls. The NumPy host path connects the
layers and computes final logits in the same process. The route kernel and
FFN descriptor run on KV260; attention, residuals, KV-related computation,
embedding, and output projection remain on ARM. It compares board logits with
the saved M4d SA simulator hybrid logits and reports the Hugging Face top
tokens separately. This is an integrated board model-level check, not a
fully device-executed IREE model.

The staged bundle is `build/moe/m4e/board` and reuses the already copied
`~/moe-m4c/layer0` and `layer1` parameter/bitstream files. The new transfer
is approximately 271 MiB plus small scripts/reference data:

```bash
scp -r build/moe/m4e/board ubuntu@192.168.0.121:~/moe-m4e
# On KV260:
cd ~/moe-m4e && chmod +x run_board.sh && ./run_board.sh
# On the host after the board run:
scp ubuntu@192.168.0.121:~/moe-m4e/board_model_report.json build/moe/m4e/
```

The user ran the integrated bundle on KV260 on 2026-10-24. Both layers loaded
88,358,976 packed parameter bytes in turn and all six route/FFN calls passed:

| Layer | Position | Experts | Route cycles | FFN cycles |
|---|---:|---|---:|---:|
| 0 | 0 | `[7, 1]` | 52,431 | 408,611 |
| 0 | 1 | `[7, 1]` | 53,023 | 408,574 |
| 0 | 2 | `[7, 0]` | 52,850 | 408,608 |
| 1 | 0 | `[6, 2]` | 52,542 | 408,698 |
| 1 | 1 | `[2, 5]` | 52,375 | 408,539 |
| 1 | 2 | `[2, 6]` | 52,974 | 408,622 |

| Position | Board/SA-sim/HF top token | Board vs SA-sim logit relative L2 | Max absolute error |
|---|---:|---:|---:|
| 0 | 25591 | `3.65624e-6` | `9.62615e-6` |
| 1 | 13899 | `2.50264e-6` | `6.76513e-6` |
| 2 | 26962 | `9.18536e-7` | `2.68221e-6` |

The board and simulator hybrid logits agree closely; their top tokens also
match Hugging Face on this short sequence. The M4d comparison showed that the
hybrid's position 2 logits differ from fp32 Hugging Face by relative L2
`0.32353`, so token agreement here does not imply fp32 logit agreement or
language quality. The checkpoint is random-initialized. Next move attention,
KV/cache, and dense projections into SA, then execute the model through IREE.

### M5：原生编译器表示

交付 `FormMoeRegions`、稳定 SAHL 表示、descriptor/RISC-V 双路径和清晰 fallback。退出条件：不依赖模型专用手写 dispatch 即可识别所选 MoE 架构。

#### M5a router recognition (2026-10-24)

The first opt-in compiler slice is `FormMoeRegions.cpp`, registered as
`--iree-sa-form-moe-regions` in `iree-opt` and enabled in the SA preprocessing
pipeline by `iree-compile --iree-sa-form-moe`. It recognizes a static batch-1
router projection feeding descending `iree_linalg_ext.topk` with K=2 and 4–16
experts. The pass wraps the Top-2 operation in one `flow.dispatch.region` and
records `sa.moe.experts`, `sa.moe.top_k`, and `sa.moe.kind`. It leaves the
projection and Top-2 semantics intact; it does not yet lower the region to the
RISC-V route kernel or include expert FFNs. The option is off by default.

`sa_form_moe_regions.mlir` covers direct and W8-quantized router projections,
K=3 rejection, and Top-2 without a router projection. The rebuilt `iree-opt` and `iree-compile`
pass this test, the complete SA lit suite, and the available golden guard.
The missing `build/golden/manifest.json` was regenerated from the four source
directories currently present: 636 unique dispatches, of which 26 are
recorded as compilation failures. The other historical source directories
are absent locally, so this baseline has four-case coverage, not the full
historical corpus. The generated `build/golden` directory occupies 8.6 MiB
and is local build data.

#### M5b SAHL Top-2 boundary (2026-10-24)

The recognizer now follows elementwise score conversion from a quantized
contraction to the router Top-2. It also marks the inner Top-K operation so
the classification survives dispatch formation and bufferization. The
`iree-sa-to-sahl` pass replaces a marked, bufferized Top-K with `sahl.top2`:
one f32 score buffer, one f32 Top-2 value buffer, and one i32 expert ID
buffer. Unmarked Top-K operations remain on their existing IREE path.

`sahl_moe_top2.mlir` verifies both paths. A full `iree-compile` probe with a
W8 router confirmed that the mark reaches the SA executable. This M5b step
preceded the RISC-V-only device lowering described below. The existing M4 RISC-V route kernel
outputs a packed route record with normalized weights; `sahl.top2` represents
raw values and IDs, so they cannot yet be substituted for each other. This
step does not execute a native IREE MoE model. The available golden guard
still passes: 636/636 dispatches identical across four local cases.

#### M5c raw Top-2 device kernel (2026-10-24)

`firmware/moe_top2_raw/moe_top2_raw.c` adds a separate PicoRV32 kernel for
the compiler's raw Top-2 boundary. Its bindings are score input, value output,
and `i32` expert ID output; the first push constant is the expert count. It
writes the selected scores and IDs without normalization or expert parameter
lookup. The existing M4 `sa_moe_route` kernel remains responsible for the
packed normalized route record used by the hand-built FFN dispatches.

`compiler/tests/moe_top2_raw_kernel.py` builds a position-independent rv32imc
blob and a host twin. `test_moe_top2_raw.py` checks 4, 8, and 16 experts,
negative scores, ties, signed zero, infinities, NaNs, and deterministic random
scores against IREE Top-K's scalar insertion rules. Both output tensors must
be initialized: f32 values to negative infinity and i32 IDs to `INT32_MAX`.
`FormMoeRegions` now checks those producer fills before marking a dispatch;
an uninitialized Top-K remains on IREE's ordinary path.

#### M5d embedded RISC-V export (2026-10-24)

`sahl-to-sahw` recognizes one isolated `sahl.top2` dispatch, records the three
DDR binding addresses, and emits a RISC-V-only template. In-process LLVM
code generation emits a specialized rv32imc function for the expert count.
It uses integer operations to implement ordered f32 comparison, avoiding
soft-float calls and relocations in the embedded code. The SA descriptor
extension has a RISC-V-only bit; the runtime rejects the export unless its
own embedded code runs with `SA_RV=embedded` on the board. Compilation
requires `--iree-sa-riscv` and rejects a route dispatch mixed with other
device computations.

`moe_router_w8_top2.mlir` and `test_moe_compiler_riscv.py` exercise the full
IREE compile path: the W8 router reaches an SA executable, the Top-2 export
has flag 2 and a nonempty code-table entry, and omission of `--iree-sa-riscv`
fails with a targeted diagnostic. The test also executes the emitted
target-neutral LLVM IR on the x86-64 host using 32-bit binding addresses,
including a nonzero subspan offset, and compares selected values and IDs for
NaNs, infinities, signed zero, ties, and random scores. Host firmware tests
cover the standalone C kernel. KV260 execution of the generated export is
recorded below.

The D=16, four-port KV260 fixture is `moe_router_w8_top2_d16.mlir` (16
experts). `test_moe_compiler_riscv.py` verifies that it compiles. Run
`compiler/tests/moe_m5d_board_bundle.py` to stage the VMFB, overlay, firmware,
three input cases and reference outputs in `build/moe/m5d/board`. The bundle
generator copies `build/iree/build-sa-aarch64/tools/iree-run-module` when that
runtime exists. Build it with `compiler/scripts/build_sa_runtime.sh aarch64`
and restage the bundle after changes to the SA runtime.
On KV260, `run_board.sh` starts the overlay and calls `iree-run-module` with
`SA_RV=embedded`; `board_moe_compiler_top2.py` checks both outputs bit for bit
for ascending, tied, and random router scores and writes `board_report.json`.

#### M5d KV260 acceptance (2026-10-24)

The user ran `~/moe-m5d/run_board.sh` on KV260 with the D=16, 100 MHz,
four-port GEMV/PCPI overlay and the newly built aarch64 SA runtime. All three
cases matched the generated host references bit for bit:

| Case | Top-2 IDs | Raw values | Result |
|---|---|---|---|
| Ascending | `[15, 14]` | `[7.0, 6.0]` | PASS |
| Ties | `[0, 1]` | `[5.0, 5.0]` | PASS |
| Random | `[5, 1]` | `[54.0, 40.0]` | PASS |

`board_report.json` records `pass: true`, six completed rt_fw entries, and
heartbeat 16313. The local report copy is in `build/moe/m5d/board`; its VMFB
SHA-256 is `0faf071beb5c9b4880361a2af35ba22bc314bdba3f71425046d70f147c8026f0`.
This accepts the native IREE Top-2 router export on KV260. It does not yet
cover route normalization, dynamic expert parameter selection, expert FFN,
or an end-to-end Mixtral layer compiled from IREE.

#### M5e native Top-2 to M4 route ABI bridge (host verified, 2026-10-24)

`firmware/moe_route_from_top2/moe_route_from_top2.c` consumes the two raw f32
values and i32 IDs emitted by the native IREE Top-2 export. Given the expert
count and packed expert stride, it computes the same approximate two-way
softmax as the M4 router and writes the existing 32-byte `sa_moe_route` v1
record: count, IDs, byte offsets, and f32 weight bits. The bridge is a separate
position-independent rv32imc kernel with a host twin. Invalid IDs, non-finite
values, unordered values, zero stride, and overflowing parameter offsets
produce a zero-count record; the caller must check this before launching an
FFN. The supported path assumes finite router scores, which is the input
domain of the Mixtral layer fixtures.

`test_moe_route_from_top2.py` verifies that this record matches M4's original
score-to-route kernel byte for byte on ties, negative scores, a wide score
gap, and deterministic random 4-, 8-, and 16-expert cases. It also checks
invalid-input rejection and PIC relocation invariance. The added
`test_moe_m3b.py` case feeds the bridge record to the existing W8A8 Top-2
FFN simulator and checks the combined output against the host reference.
These are host tests; the bridge has not yet run on KV260 or been emitted by
the IREE compiler. The end-to-end IREE MoE path therefore remains incomplete.

Next connect Top-2 values/IDs to the route bridge in the compiler/runtime,
then extend the pattern through dynamic expert indexing and weighted combine. Lower the
expert computation to one reusable SA FFN template. Compile a single Mixtral
layer from IREE input and compare route IDs, weights, and output to M4.

### M6：Prefill/grouped execution

交付 token count、prefix sum、gather/scatter、grouped GEMV，以及大组切换 systolic GEMM。退出条件：空 expert 不执行，相比逐 token prefill 有可测加速。

## 20. 风险、取舍与回退策略

| 风险 | 影响 | 对策 |
|---|---|---|
| 权重放不进设备窗口 | 无法加载 | 小/截断模型先行；预检；后续 W4/换入 |
| router 量化改变 ID | 质量下降 | router 保持高精度；测 routing agreement |
| descriptor 字段不足 | 地址无法编码 | kernel 输出完整 byte offset；切 RISC-V |
| PARAM/BASE 不足 | lowering 失败 | liveness 复用；RISC-V fallback |
| host top-k 太慢 | latency 上升 | 只用于 M1，M2 下沉 PicoRV32 |
| top-2 本地内存不足 | DDR spill | 顺序执行，ACC 中边算边累积 |
| pattern 不稳定 | 模型不兼容 | 显式前端 MVP；逐架构 adapter/tests |
| RISC-V 发射开销 | 小 op 变慢 | 保留 descriptor，按 profile 混合选择 |
| prefill 路由不均衡 | GEMM 利用率低 | 跳过空组，按 `M_e` 选择 GEMV/GEMM |
| dense 模型回归 | 现有功能受损 | feature gate、golden corpus、差分测试 |

## 21. 建议的代码改动清单

### 前端

- `compiler/frontend/qhf.py`：MoE config、权重和 reference；
- `compiler/frontend/hf_generic.py`：识别/替换受支持 MoE module；
- `compiler/frontend/export_hf.py`：MoE metadata/导出参数；
- 新增 `compiler/frontend/moe.py`：adapter、routing semantics、synthetic model。

### 编译器

- 新增 `compiler/plugins/sa/target/FormMoeRegions.cpp`；
- 更新 `SAPasses.h`、`SATarget.cpp` 和 target `CMakeLists.txt`；
- `PackLinearWeights.cpp`：expert 轴和统一 stride；
- `SahlOps.td`：动态 expert 属性，成熟后增加 topk/expert_ffn/combine；
- `SahlToSahw.cpp`：动态 expert offset；
- `SahlExpandKernels.cpp`：expert FFN 和 K 循环；
- `SahwToRiscv.cpp`：routing record、地址运算和控制流；
- 必要时更新 `SahwSerialize.cpp`，但优先避免 descriptor ABI 变化。

### Runtime 和 firmware

- 新增 `firmware/moe_topk/`，或复用现有 R5 kernel infrastructure；
- `firmware/include/rvsched.h`：routing kernel/table entry；
- `compiler/runtime/sa/sa_rvsched.c`：MoE kernel metadata；
- `compiler/runtime/tools/sa_llm_run.c`：MoE capability/profile；
- `compiler/tests/mem_preflight.py`：expert 参数和 route buffers。

### 模拟器、测试和 RTL

- `llm/sa_funcsim.py`、`compiler/tests/oracle.py`：新 ABI/中间结果；
- 增加 frontend、dispatch、RISC-V equivalence、board 测试；
- profile CSV 增加 router/top-k/expert/combine 分类；
- M0–M5 预计无需 RTL 修改；若数据证明必要，再考虑 `sa_cmdfetch.v`、`sa_gemv.v` 或 `sa_perf.v`。

## 22. 验收标准

### 功能

- synthetic 4-expert top-1/top-2 端到端运行；
- 每个 expert 都能动态选择，未选 expert 权重不读取；
- router、IDs、weights、expert output、combined output 可逐级比对；
- sim 与 KV260 一致；
- 至少一个真实 MoE 或可容纳截断版本运行。

### 编译器

- 所有 experts 复用同一结构模板，不生成 E 份相同 executable；
- descriptor/RISC-V 有差分测试；
- 不支持的 routing 语义产生明确诊断或受控 fallback；
- dense golden corpus 不回归。

### 性能与质量

- active weight bytes 与 top-k 而非 expert 总数成正比；
- 设备端 top-k 时 ARM 不参与每层路由；
- top-2 时间接近两个 active FFN 加 combine，而不是全部 experts；
- 固定 routing 满足既有 bit-exact 规则或明确容差；
- 报告 routing agreement、teacher-forced token agreement 和 PPL。

## 23. 后续方向

- IREE 通用 grouped matmul 和潜在上游接口；
- expert-aware executable generalization，降低 code size；
- multi-token/multi-sequence expert grouping；
- expert 热度统计、cache 和跨层预取；
- W4A8/W4A16 expert 流式解码；
- shared expert 与稀疏 expert 流水并行；
- 多 FPGA expert partition；
- 与 `torch.compile(backend="sa")` 原型整合。

本项目实现 MoE 的最短路径不是从通用 grouped GEMM 开始，而是利用现有动态 descriptor、RISC-V 控制和流式 GEMV，先做通 batch-1 的真正稀疏 top-1/top-2 decode；之后再把相同语义扩展到 token grouping 和 grouped GEMM。

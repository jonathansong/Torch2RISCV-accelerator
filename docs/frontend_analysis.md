# 编译器前端分析:现状、业界对照与通用化缺口

状态:2026-10-03,基于对 `compiler/frontend/` 与 `docs/iree_compiler_plan.md` 的阅读。没有在其他模型上实际运行验证,标注"推断"的地方是读代码的结论。

相关文档:[`iree_compiler_plan.md`](iree_compiler_plan.md)(§3 前端、§8.9 HuggingFace 导入、§8.16 前端通用化)、[`../compiler/README.md`](../compiler/README.md)。

---

## 1. 前端的两条路径

| 路径 | 文件 | 做法 | 状态 |
|---|---|---|---|
| 手写量化模型 | `qllama.py`、`qhf.py`、`export_hf.py` | 自己用 PyTorch 重写解码器(`QLlama`、`QModel`),量化与数值步骤显式写出 | stories15M、SmolLM2-135M、Qwen3 |
| 通用路径 | `hf_generic.py` | transformers 的原始建模代码原样导出,量化与写法由模块改写和图改写完成 | F0–F4 完成,F6(非 Llama 结构)未做 |

两条路径共用 `export.py` 的导出和编译流程,之后进入同一个 `sa` 后端。

### 1.1 手写路径(`qhf.py`)

```
config.json + *.safetensors
  → HFConfig(GQA、head_dim、QK-norm、theta、eps、tie;rope_scaling 直接报错)
  → load_weights(wq / wk 与 q_norm / k_norm 按头从 rotate-half 重排成交错对)
  → Fp32Model 校准 KV scale(amax / 127)
  → QModel(int8 权重 + 每通道 scale,int8 KV cache)
  → aot.export → qllama.mlir + qllama.irpa
```

- **GQA**:kv 头是注意力 matmul 的 batch 维,组内 G = heads / kv_heads 个 Q 头是行维。
- **KV cache**:摊平成 `(layers * seq_len, kv_dim)` 的 int8 缓冲,每层一次 `index_copy_`。
- **prefill**:线性层按 M 行批量,gather / RoPE / KV 写 / 注意力逐行按 decode 的标量形式做。
- **验证**:没有手写设备模型可对照。每个 dispatch 对着自己的 IR 检查(`dispatch_check.py` + `oracle.py`);端到端与 fp32 在容差内比较(相关系数 > 0.98 等)。

### 1.2 通用路径(`hf_generic.py`)

| 机制 | 作用 |
|---|---|
| `HFDecoder` + `PositionedCache` | 包装 `AutoModelForCausalLM`;静态 KV cache 写进模块缓冲,`torch.export` 能看到缓冲修改 |
| `quantize()` | 按模块类型把 `nn.Linear` → `QLinear`、`nn.Embedding` → `QEmbedding`(W8A8);与分类层共享的嵌入只量化一次 |
| `rope_rewrite()` | `*RotaryEmbedding` → `RopeTable`(cos / sin 查表);q / k 投影按头交错重排;`rotate_half` → `swapneg` |
| `norm_rewrite()` | `*RMSNorm` → `SaRMSNorm`(`mean` 改为乘常数 1/n) |
| `SaCache` + `sa_attention` | 通过 `AttentionInterface` 注册 `"sa"`;int8 KV、按位置掩码的 softmax、GQA 形式 |
| `GRAPH_PASSES` | 目前只有 `pow(x, 2) → x * x` |
| `export()` | `torch.export`(非严格)→ `run_decompositions`(iree-turbine 的分解表)→ 图改写 → 参数外置 → iree-turbine |

**判据**:不追求与手写路径逐位一致(两条路径的算式细节不同,int8 舍入对运算顺序敏感)。改为:sa 的每个 dispatch 与 oracle 逐位一致,加上与原版 fp32 HF 比较的质量(top-1、相关)不低于手写路径。

---

## 2. 业界主流 LLM 编译器前端

### 2.1 图捕获

| 方式 | 代表 | 特点 |
|---|---|---|
| Tracing | `torch.jit.trace`、`torch.fx` | 丢失数据相关控制流,已基本被取代 |
| 字节码级捕获 | TorchDynamo(`torch.compile`) | 能吃原始 HF 代码;不支持的 Python 触发 graph break |
| 符号导出 | `torch.export` | 整图、无 graph break、带动态形状约束;对写法有要求 |
| 重搭模型 | TensorRT-LLM、vLLM、MLC-LLM、llama.cpp | 不捕获任意 Python,用自有算子库重搭结构;工业上最常见 |
| JAX tracing | JAX `jit`、PyTorch/XLA | 产出 StableHLO;形状偏静态,变长靠分桶 / padding |

本仓库:手写路径属于"重搭模型";通用路径用 `torch.export`(经 iree-turbine)。

### 2.2 降到规范算子集

- PyTorch:ATen → Core ATen → Prims;MLIR 系:torch-mlir → `linalg` / `tensor` / `arith` / `math`,或 StableHLO / TOSA。
- LLM 编译器通常**保留** attention、RMSNorm、softmax 等高层算子,因为分解后再融合很难恢复到 FlashAttention 级别。本仓库靠 `sa-to-sahl` 的模式匹配恢复(`sahl.kernel`)。

### 2.3 LLM 特有问题的常见做法

| 问题 | 做法 |
|---|---|
| 动态形状 | 符号形状(`SymInt`、`torch.export.Dim`);分桶 + padding;inflight batching |
| KV cache | 显式输入输出;可变全局状态(本仓库);PagedAttention(vLLM) |
| prefill / decode | 编成两个图,或按 token 数特化 |
| 生成循环、采样 | 放在图外由运行时驱动 |
| 自定义算子 | `torch.library`,编译器当黑盒或映射到 kernel |
| 并行 | DTensor / GSPMD / Shardy 描述 sharding,编译器插入通信 |

### 2.4 量化在前端的位置

- PT2E / `torch.ao`:在图上插入 Q/DQ 节点,后端融合。
- 权重量化(GPTQ / AWQ / FP8 / INT4):离线转换成打包格式,图里用专用 dequant-matmul。
- 本仓库把整个量化链显式写进 PyTorch,**数值规格由前端固定**,保证与硬件逐位一致;工业上更多是容忍小的数值差异换取灵活性。

### 2.5 系统对照

| 系统 | 前端思路 | 后端 |
|---|---|---|
| `torch.compile` | Dynamo + AOTAutograd + FX | Inductor:Triton / C++ |
| TensorRT-LLM | 自有 Python 模型定义 + 插件 | TensorRT engine、CUTLASS |
| vLLM | 自有模型实现 + PagedAttention | 自有 CUDA kernel |
| XLA / JAX | JAX tracing → StableHLO | XLA 融合与代码生成 |
| IREE | `torch.export` / iree-turbine → linalg | HAL 后端(含本仓库的 `sa`) |
| MLC-LLM / TVM | Relax + 模型 builder | TIR 调度 |
| llama.cpp | 手写图 + GGUF 量化 | 手写 kernel |

---

## 3. GQA 与 `attention` 微内核(后端衔接)

读 `SahlKernels.cpp` 的结论(推断):

- `matchAttention` 只认 MHA 形式:分数 `bmm(ext(Kᵀ), q)`,q 为 `[H, hs, 1]`;P·V 的 p 为 `[H, 1, T]`;缓存按 `(t,h,j) -> (h,t,j)` 的扩展 generic 读入。
- GQA(`QModel` 与 `sa_attention` 的形式):`q[Hk, G, hs] @ K[Hk, hs, T]`,batch 是 KV 头,行是 G。这不满足上面的条件。
- 因此 GQA 走 `matchContraction`,成为 `sahl.kernel "contraction"`:
  - `bLoop` = KV 头(x 与 M 都有),`gLoop` = 组内 Q 头(只在 x 与输出),`nLoop` = T 或 hs,`kLoop` = 归约维;
  - 降级由 `Lowerer::contraction()` 完成:每个 KV 头加载一次 B 块,每个 `(b, g)` 加载一行 x 并 DIV-D 复制成 A 条带;
  - 限制:不支持动态 batch / 行;K 需要分块时不允许 G ≠ 1;串行调度,不预取。
- 影响:`--iree-sa-ukernels=all` 对 GQA 注意力无额外效果,调优空间在 `contraction()`。

---

## 4. 通用化缺口

### 4.1 结构覆盖

文档中的 F6(GPT-2:LayerNorm、GELU、learned position)未完成。

| 特性 | 现状 |
|---|---|
| LayerNorm(均值、bias) | `norm_rewrite` 只按类名匹配 `RMSNorm` |
| GELU 等激活 | 无改写;SFU 只有 EXP、RECIP、RSQRT,`tanh` / `erf` 无对应 |
| learned / ALiBi 位置编码 | `rope_rewrite` 只处理 RoPE;`sa_attention` 没有分数偏置 |
| Sliding window | 掩码只是 `arange(T) <= pos` |
| MoE | 无路由、top-k、按 expert 的 gather |
| MLA / 低秩 KV | `SaCache` 假定 `[T, kv_heads * head_dim]` |
| Encoder-decoder / 多模态 | `HFDecoder` 签名只有 `input_ids, position_ids` |

### 4.2 改写中的写死假设

- `rope_rewrite`:假设 `head_dim` 偶数、投影按头排列、模块名 `q_proj` / `k_proj` / `q_norm` / `k_norm`;partial rotary 未处理。
- **`rotate_half` 替换对整个进程生效**(docstring 自己指出):之后加载同族的未改写模型会算错 RoPE。
- 按类名 / 属性名匹配(`endswith("RMSNorm")`、`hasattr(mod, "q_proj")`),融合 QKV(`c_attn`、`qkv_proj`)匹配不上。
- `quantize()` 对所有 `nn.Linear` 一刀切,无"跳过敏感层"的配置(router、LoRA 等)。
- `from_pretrained(..., dtype=torch.float32)` 先整体以 fp32 载入,大模型内存不可行。

### 4.3 量化质量与校准

- 只有 W8A8 per-channel 朴素量化,无 GPTQ / AWQ / SmoothQuant,无 outlier 处理。
- KV scale 静态,用一小段固定文本校准,换领域可能溢出或浪费精度。
- 质量判据粗:top-1 一致率 ≥ 85%、平均相关 > 0.95;无困惑度或下游任务回归。
- 通用路径没有逐位一致的端到端参照。

### 4.4 与 transformers 版本强耦合

依赖 5.x 内部接口:`StaticCache.layers`、`layer.update`、`layer.cumulative_length`、`early_initialization`、`AttentionInterface`。文档已记录一次因版本行为引发的错误(因果掩码按 cache 内部长度构造)。

### 4.5 导出与形状

- batch 固定为 1;decode 为 `[1,1]`,prefill 为固定 M 的块。
- `max_len` 静态;长上下文受片上容量限制(记录:D = 8、128 KB SPAD、head_dim 128 时 T ≤ 512)。
- 所有层共用一个 K、一个 V 缓冲,是为绕开 IREE 3.11 的 mutable global 合并问题(`compiler/tests/iree_global_merge_repro.py`),属于对上游缺陷的绕行。

### 4.6 后端覆盖

- 分片的逐元素 dispatch 不支持整个向量的求和(会改变求和顺序)。
- `--iree-sa-ukernels=none` 下 prefill 多行线性层曾编不了(§8.14 记录)。
- 主机退路用 VMVX 解释执行:SmolLM2 强制线性层放主机时 decode 每步 16.8 s。"能编译"不等于"能用"。

---

## 5. 优先级建议

| 优先级 | 事项 | 理由 |
|---|---|---|
| 高 | 完成 F6:LayerNorm、GELU、learned position,用 GPT-2 验证 | 文档已规划,检验泛化是否真实 |
| 高 | 去掉 `rotate_half` 的进程全局替换 | 影响正确性,批量测试易踩坑 |
| 中 | 融合 QKV、partial rotary、sliding window | 覆盖 Phi、Mistral、GPT-NeoX 等家族 |
| 中 | 量化配置化:跳过层、SmoothQuant、更丰富的校准集 | 大模型精度的关键 |
| 中 | transformers 版本适配层与锁版本 | 降低维护风险 |
| 低 | MoE、MLA、多模态 | 需要后端新增能力,工作量大 |

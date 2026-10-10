# 4 口 GEMV 之后：通用性与质量优先的下一阶段方案

状态：草案（2026-10-10，依据 main `2ddab2f`：K2b G4 双口板上验收、G5 4 口比特流与 bwtest 通过，G5-4 完整 staging 进行中）。本文决定 4 口 GEMV 完成之后的工作顺序：**先做通用性与量化质量，`kv260_upgrade_plan.md` 中剩下的性能项（K2a-2、W4、H2、子银行、K3）放到 2026-11-25 IREE 社区会议之后，按本文第 9 步的覆盖矩阵与耗时分解选择。** 性能数字除"实测"外均为估算。

---

## 0. 决策与理由

| 理由 | 说明 |
|---|---|
| 性能的边际收益下降 | 4 口 GEMV 后 decode 已接近端口上限（bwtest 57.95 B/周期，64 的 90.5%）。剩下的两个翻倍级手段都贵且有风险：200 MHz（LUT 87%、WNS +0.168 ns，VE / fp32 / SFU 要加流水，每次构建数小时，可能不收敛）、W4 权重（LD 路径解包 + 编译器 + int4 质量）。其余各约 +10% 或更少（H2 + LD 命令流水约 +10%，子银行 prefill ×1.07） |
| 通用性主要是软件 | importer、Llama、Gemma 基本不需要新比特流，不占已经紧张的 LUT；在 sim 上迭代，板子只做验收 |
| 数据决定下一个硬件特性 | LUT 只剩约 13%，下一个硬件特性（200 MHz、W4、SFU 加 GELU / tanh……）必须选最值的；先拿到三个模型家族的耗时分解和回退统计再定 |
| 质量是最大短板 | Qwen3-0.6B W8A8 与 fp32 的 top-1 只有 25/40。"能跑、逐位正确、但质量差"比"慢 10%"更伤可信度；现成的 W8A8 checkpoint 是最便宜的改进途径 |
| 演讲 | IREE 社区关心编译器覆盖哪些模型、能否直接用生态里的 checkpoint；"三个模型家族 + 覆盖矩阵"比 decode +10% 更有说服力；200 MHz 若 11 月中仍未收敛会挤掉演讲准备 |

**例外**（便宜、只改编译器、穿插进行）：GEMV 的 K 分块（第 6 步，Llama 需要）；prefill 的逐行执行（Qwen3 SwiGLU 段，占 prefill 30% 以上，时间允许就做）。

**会改变本决策的情况**：近期目标若是以绝对吞吐为主的论文或基准对比，200 MHz 应提前，通用性后移。

---

## 1. 执行顺序

```
1 四口 GEMV 收尾 → 2 importer → 3 SmolLM2 验证 → 4 Qwen3 复测
  → 5 Llama-3.2-1B → 6 GEMV K 分块 → 7 消融初版
  → 8 Gemma-3-270M → 9 按覆盖矩阵优化 → 10 最终表格
```

与最初的列表相比：K 分块移到 Llama 之后（先正确、再性能，前后对比本身就是一组消融数据）；增加 Qwen3 复测（第 4 步）与消融初版（第 7 步）。

### 第 1 步：4 口 GEMV 收尾（G5-4）

- 4 口编译（`--iree-sa-gemv-ports 4`）的完整 staging；板上 logits 与 sim 逐位一致，token 与 EX 路径逐字相同（同双口的 G4 判据）。
- decode token/s 与模型预测比较（Qwen3 约 6.5–7 token/s，`k2b_gemv_design.md` §8.1）；`board_profile.py` 的 decode profile 喂给 `perf_model.py` 校准。
- **保存消融基线**：EX 路径（`p6_red`）、GEMV 1 / 2 / 4 口的 token/s、设备周期、资源（第 7、10 步用）。

**验收**：REGRESSION PASS（含 Qwen3）；预测与实测之比记录在案。**估计** 1–2 天。

### 第 2 步：compressed-tensors W8A8 importer

`compiler/frontend/hf_generic.py` 增加读入 llm-compressor 导出的 W8A8 checkpoint 的路径（`QLinear(lin, qs=(q, s))` 已支持外部给定的 int8 权重与 scale）。

1. **接受的格式**（第一版）：`config.json` 的 `quantization_config` 为 compressed-tensors；线性层权重 int8、按输出通道、对称、静态；激活 int8、按 token、对称、动态。**其他格式明确报错**：按组 / 按块的权重 scale、静态激活 scale、非对称（有 zero point）、int4、FP8。
2. **直接读 int8 与 scale**：safetensors 中的 `*.weight`（int8）与 `*.weight_scale`（bf16 / fp16 → fp32，精确）。**不能先反量化再用 `_quantize_rows` 重新量化**：成品的 scale 不一定是 max|w| / 127，值可以是 −128，重新量化得到的整数会不同。
3. **HF 模型里放反量化的 fp32 权重**（q · s），供 `rope_rewrite`、KV 校准与 fp32 形式的检查使用；同时把 (q, s) 挂在线性层模块上，`rope_rewrite` 对 q / k 投影的行重排同样作用于 q 的行与 s（按行量化与行重排可交换），`quantize()` 遇到挂好的 (q, s) 时直接用。
4. **`ignore` 列表中的层**（通常是 `lm_head`）：按现有方式自己量化；嵌入与分类层共享时沿用 `QEmbedding` 与共享的 int8 表（C6.0 的打包 gather）。
5. **SmoothQuant 的缩放**已折进 RMSNorm 权重与线性层权重，照常读入。
6. **−128**：前端的 `_to_i8` / `_quantize_rows` 把权重钳在 ±127；导入的权重是常量，直接打包。用逐 dispatch 检查确认打包、funcsim、oracle 都没有假设对称的 ±127。
7. **KV 的 int8 scale**：成品不含，沿用 `calibrate_kv`。

**验收**：合成的 compressed-tensors checkpoint（`synth_hf.py` 生成，含 −128、bf16 scale、`ignore: [lm_head]`）上，导入后模型的线性层整数权重与文件逐字节相同；不支持的格式各有一个报错测试。**估计** 2–3 天。

### 第 3 步：小模型验证（SmolLM2-135M）

- 用 llm-compressor 的 W8A8 示例（SmoothQuant + GPTQ）量化 SmolLM2-135M（CPU 上几分钟），导入。
- 编译、逐 dispatch 与 oracle 逐位一致；sim 端到端；上板（双口或 4 口 GEMV）与 sim 逐位一致。
- **质量对比**：固定文本 40 个位置的 top-1（与原版 fp32 HF），自量化的通用路径（现在 34–35/40）对 llm-compressor 版本。

**验收**：逐 dispatch 全部一致，板上逐位一致；质量不低于自量化版本（记录差值）。**估计** 1–2 天。

### 第 4 步：Qwen3-0.6B 复测质量

- llm-compressor 量化 Qwen3-0.6B（或用现成的 W8A8 checkpoint，如果有），导入，测 top-1。
- 若明显好于 25/40：作为 Qwen3 的默认权重。
- 若提升不大：误差主要在 KV、注意力概率或激活，转做 `kv260_upgrade_plan.md` Q 第 1 项（逐项换回 fp32 定位），可能需要更细的 KV scale（按头）。

**验收**：top-1 与平均相关系数记录在案，并给出误差来源的结论。**估计** 1–2 天（不含 Q 第 1 项）。

### 第 5 步：Llama-3.2-1B-Instruct（RedHatAI `Llama-3.2-1B-Instruct-quantized.w8a8`）

结构（以 `config.json` 为准）：16 层、dim 2048、FFN 8192、GQA 32 / 8、head_dim 64、词表 128256、嵌入共享、RoPE 带 llama3 频率缩放。

| 项 | 处理 |
|---|---|
| 量化格式 | 拿到文件后先核对 `quantization_config` 是否符合第 2 步的接受格式 |
| RoPE 频率缩放 | 通用路径的 `rope_rewrite` 用模型自己的 `*RotaryEmbedding` 生成表，缩放自动包含（`iree_compiler_plan.md` C6.4 写的"遇到会报错"是手写 qhf 路径）；验证 fp32 模型改写前后输出一致 |
| 分词器 | Llama 3 的字节级 BPE 正则与聊天模板的特殊 token（`<\|begin_of_text\|>`、`<\|start_header_id\|>` 等）：`hf_tokenizer.py` 是近似实现，逐条与 transformers 的分词结果比较 |
| 嵌入 gather | K × D = 2048 × 16 = 32768 ≤ 65535，C6.0 的打包 gather 可用 |
| 内存 | int8 权重约 1.24 GB（1178 MiB），KV 每 token 16 KB；先看 sim 的设备内存峰值。1280 MiB 的池余量很小，必要时扩池（受全局 CMA `0x65C0_0000`、启动时 initrd `0x75DA_E000` 与低 2 GB 的限制，`kv260_upgrade_plan.md` K5-M） |
| down_proj | K = 8192，`K × d ≤ 65535` 不满足（`SahlExpandKernels.cpp` 的 `useGemv`），退回 EX 路径：正确，较慢（第 6 步解决） |
| 许可 | Llama 3.2 Community License：下载需在 HF 接受许可；发布结果注明 "Built with Llama"；权重不入库 |

**验收**：逐 dispatch 一致；板上与 sim 逐位一致；质量（与原版 bf16 模型的 top-1，参照必须是原版，不是反量化的模型）；decode / prefill token/s。**估计** 3–5 天。**估计速度**：4 口 GEMV、down_proj 走 EX 约 2.5 token/s。

### 第 6 步：GEMV 的 K 分块

- 编译器（`expandLinear`）：K × d > 65535（或 K 超过 GEMV 的 x 缓冲 4096 字节）时，把 K 切成 ≤ 4096 的块，块之间用 GEMV 命令的 `acc` 位在 ACC 上累加（硬件已支持，`tb_sa_dma` 测过"K 两半累加"）。
- 整数累加与顺序无关：结果与不分块、与 EX 路径逐位相同。
- 测试：`synth_hf.py` 生成 K > 4096 的合成模型；`dispatch_check`；黄金语料（现有模型不受影响，K 都 ≤ 4095）。

**验收**：合成模型与 Llama 的 down_proj 走 GEMV，逐位一致；Llama decode 前后对比。**估计** 2–3 天；预计 Llama decode 到约 3–3.5 token/s。

### 第 7 步：消融初版

第 1 步的基线加 Qwen3、Llama 的结果，按第 3 节的格式出第一版表（演讲用）。**估计** 1–2 天。

### 第 8 步：Gemma-3-270M

结构要点（以 `config.json` 为准）：

| 特性 | 处理 |
|---|---|
| 分词器 | SentencePiece 风格的 BPE（▁、byte fallback），不是 GPT-2 式的字节级 BPE：**最大的隐藏工作量**；扩展 `hf_tokenizer.py`，或主机端用 `tokenizers` 库 |
| 滑动窗口注意力（每 6 层 5 层局部，窗口 512） | **第一版上下文 ≤ 512**：局部层与全局层的计算完全相同；`"sa"` 注意力按位置掩码、不依赖 HF 的掩码，不需要回退。局部 / 全局层的 RoPE base 不同，`rope_rewrite` 按模块各自生成表，需验证 |
| GELU-tanh | 不回退：前端图改写 0.5·x·(1 + tanh(u)) = x / (1 + exp(−2u))，只用 SFU 已有的 exp 与倒数 |
| head_dim 256 | 一个头的 K 行占的 SPAD 是 Qwen3 的两倍，确认 512 的上下文放得下 |
| 每层 4 个 RMSNorm、(1 + w) 形式、嵌入乘 √d、QK-norm | 逐元素与归约，VE 能做；`norm_rewrite` 的类名匹配要覆盖 Gemma 的 RMSNorm |
| 词表约 262K | 嵌入约占参数的六成，decode 读权重以分类层为主；吞吐特征与其他模型不同，表中注明 |
| 量化 | 大概率没有现成 W8A8，自量化或用 llm-compressor；Gemma 的激活离群值可能让 W8A8 质量成为主要风险 |
| 许可 | Gemma 使用条款，HF 上需先接受 |

第一版目标：sim 上跑通，允许个别 dispatch 回退主机，`SA_STATS=1` 统计；上板为加分项。**估计** 1–2 周。

### 第 9 步：按覆盖矩阵优化

- **先补测量**：`SA_STATS` 现在只给数量；运行时按 dispatch 记录主机执行时间（主机 dispatch 的墙钟）与加速器列表的设备周期，输出每个模型的回退清单（dispatch 名、次数、耗时占比）。
- 按耗时占比选 1–2 类回退：编译器能解决的（图改写、新的 lowering）优先；需要硬件的（SFU 新函数等）与第 4 节的候选一起评估。

### 第 10 步：最终表格

第 3 节的完整版：Qwen、Llama、Gemma 的正确性、质量、覆盖率、吞吐与资源消融。

---

## 2. 时间表（到 2026-11-25）

| 时间 | 内容 |
|---|---|
| 10/11–10/14 | 第 1 步 |
| 10/15–10/24 | 第 2、3、4 步 |
| 10/25–11/3 | 第 5、6 步 |
| 11/4–11/12 | 第 7 步；第 8 步 sim 跑通，覆盖矩阵初版 |
| **11/15 前后** | **演讲内容冻结**，之后只做幻灯片与排练 |
| 会后 | 第 9、10 步完整版；第 4 节的性能候选 |

**必须完成**：1、2、3、4、5。**争取**：6、7。**作为"进行中"展示**：8、9。

---

## 3. 交付表格的格式

**主表**（每个模型一行）：

| 模型 | 来源 | 正确性 | 质量 | 覆盖率 | decode | prefill | 带宽利用率 |
|---|---|---|---|---|---|---|---|
| | 自量化 / llm-compressor / 成品 | 逐 dispatch 与 oracle；板上与 sim 逐位 | 与 fp32 / bf16 的 top-1、平均相关 | 加速器上的 dispatch 占比、时间占比 | token/s | token/s | 实测 / 端口上限 |

**消融表**：EX 路径 → GEMV 2 口 → 4 口 → K 分块 → SFU 批量（P6、REDUCE），逐项开关，列 token/s 与 LUT / BRAM / DSP / WNS。已有比特流：`p6_red`、`_gemv`、`_gemv_np2`、`_gemv_np4`（资源见 `boards/kv260/README.md`）。

---

## 4. 会后的性能候选（按覆盖矩阵与耗时分解选择）

| 候选 | 预期 | 代价 / 风险 |
|---|---|---|
| K2a-2：200 MHz | decode 最多约 ×2（4 口 × 16 B × 200 MHz = 12.8 GB/s，DDR4 峰值约 19 GB/s） | VE / fp32 / SFU 加流水；LUT 87%；复位扇出（当前最差路径，97% 布线）需复制寄存器或复位树 |
| W4 权重（W4A8） | decode 约 ×1.8（字节减半） | LD 路径解包、编译器、int4 质量 |
| H2 + LD 命令流水 | 约 +10%（`k2b_gemv_design.md` §8.1：×3.15 → ×3.46） | 中 |
| prefill 逐行执行 | Qwen3 prefill 的 30% 以上 | 只改编译器 |
| C6.5 T 分块注意力 + K3 URAM | 长上下文（T = 2048） | 编译器新功能 + URAM 语义 |
| SPAD 子银行 | prefill ×1.07 | 中 |

---

## 5. 风险

| 风险 | 对策 |
|---|---|
| 成品 checkpoint 的格式与假设不同（按组 scale、静态激活、zero point） | importer 只接受明确的格式，其余报错；拿到文件先核对 `quantization_config` |
| Qwen3 换成成品权重后质量仍差 | 第 4 步给出结论后转 Q 第 1 项；演讲中如实说明 |
| Llama 的内存峰值超过 1280 MiB | sim 先测峰值；扩池或缩短上下文 |
| 分词器与 transformers 不一致 | 逐条比较测试；不一致时主机端用 `tokenizers` 库 |
| Gemma 工作量超出预期 | 第一版只求 sim 跑通与覆盖矩阵，允许回退；作为"进行中"展示 |
| 板子时间冲突（第 1 步与后续的上板验收） | 编译器与前端工作都在 sim 上进行，板子只做验收 |

---

## 6. 待确认

- RedHatAI `Llama-3.2-1B-Instruct-quantized.w8a8` 的 `quantization_config`（权重粒度、激活方式、`ignore` 列表、scale 的类型）；是否有 Qwen3-0.6B 的 W8A8 成品。
- Gemma-3-270M 的 `config.json`：滑动窗口大小、局部 / 全局层的比例与 RoPE base、激活函数、词表大小、是否有 logit softcapping。
- Llama-3.2-1B 在 sim 上的设备内存峰值。
- llm-compressor 在本机（CPU）量化 SmolLM2 / Qwen3 的可行性与耗时。

---

## 7. 文档同步

`kv260_upgrade_plan.md` 中按原设想写的部分需要更新：状态行（仍是 10/08 的 K1）、§3 阶段表的 K2b（"同时 D = 16 → 32"、200–250 MHz、子银行）与 K3（"为 D = 32"）、§3.1 H0（"D = 32：K2b"）、K3 第 6 项、§5 的 K2b 行与缺少的 G4 / G5 行、§6 已有答案的待确认项、K4b 按 250 MHz 的外推。K2b 实际是 100 MHz 的流式 GEMV（2 / 4 口），D = 32 不再需要。根目录 README 顶部的数字仍是 K1 的。

# 4 口 GEMV 之后：通用性与量化质量实现方案

状态：实施版草案（2026-10-10 按代码与四口板上结果复核：WP0、WP1 基本完成，WP5 的 EX 基线已存在，WP6 只差编译器接入）  
日期：2026-10-10  
基线：`main@b42fec6`（四口验收 `fb01909`）  
目标节点：2026-11-15 演讲内容冻结；2026-11-25 IREE 社区会议  

## 1. 目标与边界

本阶段在 G5 四口 GEMV 完成后，优先解决两个问题：

1. **通用性**：让同一套 PyTorch / Hugging Face → IREE → SA descriptor 流程覆盖 Qwen3、SmolLM2、Llama 3.2 和 Gemma 3 四类模型结构。
2. **量化质量**：支持导入 compressed-tensors W8A8 成品权重，并把“checkpoint 本身的量化损失”和“本项目额外引入的数值损失”分离测量。

会前不主动展开 200 MHz、W4A8、SPAD 子银行、K3 URAM 等高风险硬件工作。唯一例外是已经完成硬件支持、只需编译器接入的 GEMV K 分块，以及时间允许时的 prefill 逐行执行。

### 1.1 会前成功标准

| 级别 | 交付目标 |
|---|---|
| 必须完成 | ~~四口 GEMV 完整验收~~（2026-10-10 完成）；compressed-tensors W8A8 importer；SmolLM2 验证；Qwen3 质量复测；Llama 3.2 正确性路径与内存结论 |
| 争取完成 | Llama down_proj 的 GEMV K 分块；Llama 板上端到端；第一版消融表 |
| 进行中展示 | Gemma 3 sim 跑通、覆盖矩阵、主要回退原因 |
| 会后决定 | 200 MHz、W4A8、H2/LD 流水、SPAD 子银行、T 分块与 URAM |

### 1.2 非目标

- 不在第一版 importer 中支持 int4、FP8、非对称量化、静态激活量化、权重按组/按块量化。
- 不要求 Gemma 第一版在 KV260 上完整跑通长上下文。
- 不以 40 个位置的 top-1 作为最终质量结论；它只作为快速回归指标。
- 不把主机墙钟时间与设备周期简单相加，除非运行时确认两者没有重叠。

## 2. 总体执行顺序

```text
G5-4 收尾（已完成，只剩 perf_model 校准）
  → compressed-tensors importer
  → 合成 checkpoint + SmolLM2 验证
  → Qwen3 质量定位
  → Llama 前端、内存预检与 EX 基线
  → GEMV K 分块
  → Llama 完整 sim / 板上验收
  → 消融初版
  → Gemma 短上下文 sim 与覆盖矩阵
  → 根据耗时分解决定下一项优化
```

关键原则：Llama 先建立 EX 路径的正确性基线，再启用 K 分块。这样既降低调试耦合，也自然形成一组编译器消融实验。

## 3. 统一验收口径

### 3.1 正确性分层

每个模型依次通过以下层级：

1. **Checkpoint reference**：compressed-tensors / Transformers 或 vLLM 可识别的原始 checkpoint 参考。动态激活量化的约定以 **vLLM 运行时语义**为准（按 token、`scale = amax / 127`，与设备和 `qllama` 相同）；llm-compressor 校准时的 fake-quant 可能用 `amax / 127.5` 并截到 [-128, 127]（待核实），不作为参考，否则这一层会混入与本项目无关的约定差异。
2. **Importer eager reference**：项目 importer 构造的 PyTorch eager 模型。
3. **SA oracle**：采用设备量化语义、近似函数和舍入规则的主机参考。
4. **funcsim / dispatch_check**：每个 dispatch 的输入输出与 SA oracle 逐位一致。
5. **板上执行**：logits 与 funcsim 逐位一致，生成 token 与 EX 路径一致。

“逐位一致”只用于采用同一整数语义的 3–5 层，不要求量化模型与 bf16/fp32 模型逐位一致。

### 3.2 质量指标

| 指标 | 用途 | 最低数据量 |
|---|---|---|
| 固定 40 位置 top-1 | 快速回归、每日测试 | 40 个固定位置 |
| top-1 agreement | 正式报告 | 至少数千个 token 位置 |
| 平均相关系数 / cosine | 判断 logits 整体漂移 | 与 top-1 使用相同位置 |
| KL divergence | 判断概率分布偏移 | 与 top-1 使用相同位置 |
| WikiText-2 perplexity 或等价 PPL | 端到端语言建模质量 | 固定、可复现的数据切分 |

**在哪里跑**：功能仿真太慢（Qwen3 sim 上 4 步 decode 约 819 s），数千位置的正式指标不在 funcsim 上跑。两条路：① 主机上的 SA oracle（eager，设备量化语义），用于第 3 层；② 板上 prefill 做 teacher forcing（Qwen3 prefill 约 16 tok/s，4000 个位置约 4 分钟），用于第 5 层。② 需要运行时导出 prefill 每个位置的 logits（现在只导出最后一个位置），列入 WP9。

校准集与质量测试集必须分离；各实现使用完全相同的 token ID、上下文长度、聊天模板和截断规则。

### 3.3 覆盖率指标

同时报告四类覆盖率，避免“dispatch 数量很高但热点仍在主机”的误导：

- 静态覆盖率：编译图中可下沉到加速器的 dispatch 类型占比。
- 动态覆盖率：实际运行的加速器 dispatch 次数 / 总 dispatch 次数。
- 时间覆盖率：加速器执行时间 / 端到端执行时间，以及主机回退墙钟占比。
- 工作量覆盖率：加速器承担的估算 MAC 数和权重/激活字节占比。

回退清单至少包含：dispatch 名称、算子类别、形状、调用次数、累计耗时、占比、回退原因。

## 4. 工作包 WP0：基线冻结与文档同步

**状态（2026-10-10）**：文档同步已完成并推送（`b42fec6`）：根目录 README、`compiler/README.md`、`kv260_upgrade_plan.md`（K2b 写明 D = 16 + 流式 GEMV 2 / 4 口，不再以 D = 32 为前提）、`k2b_gemv_design.md`、`iree_compiler_plan.md`。板上结果目录的 `summary.txt` 已记录 commit、bitstream 哈希、overlay 配置与 WNS。剩下的是测试 prompt / 结果目录命名的统一。

### 实现

- 记录 `main` commit、Vivado/Vitis 版本、IREE 版本、板卡固件、设备树和 u-dma-buf 布局。
- 固定测试 prompts、token IDs、上下文长度和生成长度。
- 为结果目录定义统一命名：`<commit>/<model>/<bitstream>/<mode>/`。
- 立即修正根目录 README 中仍为 K1 的数字。
- 将 `kv260_upgrade_plan.md` 的 K2b 更新为 100 MHz 流式 GEMV 2/4 口，移除 D=32 是 K2b 前提的旧描述；G4/G5 资源和状态按已验收结果填写。

### 验收

- 所有后续表格都能追溯到 commit、bitstream 和运行参数。
- 文档不存在同时把 K2b 描述为 D=32、200–250 MHz 和 100 MHz 四口 GEMV 的冲突。

## 5. 工作包 WP1：G5 四口 GEMV 收尾

**状态（2026-10-10，`fb01909`）**：除 perf_model 校准外已完成。板上 REGRESSION PASS（含 Qwen3），logits 与 funcsim 逐位一致，token 与 EX / 2 口相同；bwtest GEMV 57.44–57.95 B / 周期。decode（tok/s，EX → 2 口 → 4 口）：Qwen3 2.22 → 3.98 → **6.28**（预测 ×3.15，实测 ×2.83），SmolLM2 7.71 → 12.58 → **17.04**，hfgen 7.62 → 12.73 → 17.19，stories 61.9 → 100.8 → 131.65；prefill 不变。由 2 口 / 4 口两点解出每步 = 固定部分 F + GEMV 部分：**F 约 67 ms（Qwen3，占 4 口每步的 42%）、约 38 ms（SmolLM2，65%）**，GEMV 部分已近线性随口数缩短。结果在 `k2b_gemv_design.md` §8.1（G5-4）与 `build/deploy_kv260_*_gemv{,_np2,_np4}/results`。

### 实现

- 使用 `--iree-sa-gemv-ports=4` 完成 Qwen3、SmolLM2、generic 和 stories 的 staging。
- 保存 EX、GEMV 1/2/4 口的可执行文件、bitstream 标识和完整运行日志。
- 用 `board_profile.py` 收集 decode 的设备周期、DDR 读字节、命令数量和各阶段耗时。
- 将 profile 输入 `perf_model.py`，同时保存预测值、实测值和误差。

### 验收

- 回归全集通过，包含 Qwen3。
- 板上 logits 与 funcsim 逐位一致；token 序列与 EX 路径一致。
- 四口带宽测试维持约 57.95 B/cycle，并记录实际波动范围。
- Qwen3 decode 实测值与预测值同时入表；不把 6.5–7 token/s 估算写成实测。

### 输出

- 不另开 closeout 文档：结果已在 `k2b_gemv_design.md` §8.1；剩下的 perf_model 校准（预测 / 实测 / 误差，以及固定部分 F 的分类拆分）也写在那里。
- 硬件消融表的 EX、1 口、2 口、4 口四行（1 口的 decode 板上数据尚未取回，可选）

## 6. 工作包 WP2：compressed-tensors W8A8 importer

### 6.1 接受格式

第一版只接受：

- `format: int-quantized`，状态为 compressed。
- 权重：int8、静态、按输出通道、对称、无 zero point、`group_size = null`。
- 激活：int8、动态、按 token、对称。
- `ignore` 列表允许包含 `lm_head` 等未量化层。
- scale 可以为 bf16/fp16/fp32，载入后转换为 fp32，不改变 int8 权重。

其他格式必须给出包含层名、实际字段和支持范围的明确错误，不允许静默回退或重新量化。

### 6.2 内存安全的数据结构

禁止长期同时保存整模型的 int8 权重、scale 和 `q * s` 的 fp32 副本。实现外部量化线性层，例如：

```python
QLinearExternal(
    qweight: Tensor[int8],
    scale: Tensor[fp32],
    bias: Optional[Tensor],
    source_name: str,
)
```

实现要求：

- safetensors 使用 mmap / lazy load；以层为单位读取和释放。
- 编译归档直接消费 `(qweight, scale)`，不经过 `_quantize_rows`。
- eager reference 仅在当前层或当前算子需要时临时反量化，随后释放。
- `rope_rewrite` 对 Q/K 投影执行行置换时，同时置换 qweight 的行和对应 scale；不创建完整 fp32 权重副本。
- 对无法直接处理量化权重的校准逻辑提供分层临时反量化接口。

### 6.3 数值与打包

- 导入的 int8 常量保留完整 `[-128, 127]` 范围。
- 审核打包、funcsim、oracle 和 DMA 路径，移除任何“负值最小为 -127”的隐含假设。已知位置：前端 `hf_generic._to_i8` / `_quantize_rows`、`qllama` 的转换都截到 ±127（这是自量化的约定，导入路径不能经过它们）；硬件的 int8 乘法（`sx8`）与 funcsim 本身对 -128 没有问题。激活量化仍由设备做，保持 ±127。
- 区分 importer 语义与本项目自量化语义；导入路径不调用 `_to_i8` 或 `_quantize_rows` 修改已有整数。
- SmoothQuant 不需要 importer 做任何事：llm-compressor 已把平滑因子折进 norm 权重（norm→QKV、norm→gate/up）和 up_proj（→down_proj），checkpoint 里的权重就是最终值，原样读入即可。

### 6.4 模型打包与 ignore

- `ignore` 中的层按现有路径处理，但必须在 manifest 中记录“原始 bf16/fp16”还是“由本项目重新量化”。
- tied embedding / lm_head 只保留一份设备端 int8 表，并验证权重共享没有被 importer 解开。
- manifest 记录来源模型、revision、配置哈希、safetensors SHA256、许可证和转换参数。

### 6.5 测试

用 `synth_hf.py` 生成至少以下用例：

1. 正常的 channel-wise W8A8。
2. 权重包含 -128。
3. bf16 scale。
4. `ignore: [lm_head]` 且 embedding/lm_head tied。
5. 按组权重 scale，应报错。
6. 静态激活 scale，应报错。
7. asymmetric / zero point，应报错。
8. int4 和 FP8，应报错。
9. 权重张量或 scale 缺失、形状不匹配，应报错。

### 验收

- 导入后的每个受支持线性层，整数权重与 safetensors 逐字节相同。
- scale 转 fp32 后数值与原文件一致。
- 合成模型经过 archive、funcsim 和 dispatch_check 后逐位一致。
- importer 峰值主机内存有测量值，且不因构造全模型 fp32 副本出现数量级增长。

## 7. 工作包 WP3：SmolLM2-135M 验证

### 实现

- 使用固定 llm-compressor recipe 生成 W8A8 checkpoint，并冻结 recipe、校准样本 ID 和软件版本。
- 依次完成 checkpoint reference、importer eager、SA oracle、funcsim、板上验证。
- 与现有自量化路径使用相同 prompts/token IDs 做 A/B。

### 验收

- 每个 dispatch 与 SA oracle 逐位一致，板上与 funcsim 逐位一致。
- 40 位置 top-1 不低于现有自量化结果，作为快速门禁。
- 同时记录至少数千 token 的 top-1、相关系数/KL 和 PPL；若不优于自量化，不阻塞 importer 合入，但必须定位损失发生在哪一层参考之间。

## 8. 工作包 WP4：Qwen3-0.6B 质量复测与定位

### 输入

优先评估现成 compressed-tensors W8A8 checkpoint（先确认 Qwen3-0.6B 是否有公开的 W8A8 版本；没有则用 WP3 的固定 recipe 自己用 llm-compressor 生成）；同时保留当前自量化模型作为对照。下载时固定 revision 与文件哈希，权重不入仓库。

### 四层质量对比

| 层级 | 目的 |
|---|---|
| 原版 bf16/fp32 HF | 最终质量参考 |
| 外部 W8A8 checkpoint 的标准运行时 | 测量 checkpoint 自身的量化损失 |
| importer eager | 检查 importer、RoPE rewrite、权重共享和 tokenizer |
| SA 端到端 | 测量 KV、注意力概率、近似函数和设备舍入的额外损失 |

### 决策门禁

- 若外部 W8A8 checkpoint 明显优于当前 25/40，且 importer eager 与其接近，则将外部权重设为默认演示路径。
- 若标准运行时质量好、但 importer eager 明显下降，优先修 importer / tokenizer / rewrite。
- 若 importer eager 良好、SA 明显下降，逐项切换为高精度定位：KV cache、QK logits、softmax/exp/recip、激活量化、最终分类层。
- KV 被确认是主因后，再评估按头 scale；不得在定位前直接增加硬件或格式复杂度。

### 验收

- 输出四层质量表和误差归因结论。
- 40 位置 top-1、正式 top-1、相关系数/KL、PPL 的数据和脚本可复现。

## 9. 工作包 WP5：Llama-3.2-1B 前端与内存预检

### 9.1 模型约束

以实际 `config.json` 为准，当前预期结构为 16 层、hidden size 2048、FFN 8192、32/8 GQA、head_dim 64、vocab 128256、tied embedding，以及 Llama 3 RoPE scaling。公开的 W8A8 版本（RedHatAI）是 **Instruct** 模型：质量参考用 Instruct 的 bf16，按 chat template 构造输入；这类 checkpoint 通常 `ignore: [lm_head]`，嵌入 / 分类表由本项目量化（manifest 记录）。

### 9.2 前端实现

- importer 先严格校验 compressed-tensors 配置是否属于 WP2 支持范围。
- `rope_rewrite` 使用模型 RotaryEmbedding 生成频率表，并做改写前后 eager 对比。
- 对 Llama 3 tokenizer 的特殊 token 和 chat template，逐条与 Transformers 比较；演讲前若本地 tokenizer 仍不完整，允许主机端使用 `tokenizers` 生成 token IDs。
- embedding gather 做边界和打包测试；tied classifier/embedding 在设备归档中只保留一份。

### 9.3 字节级内存预检

在开始完整 sim 前，生成 memory manifest，逐项统计：

- 各层 int8 权重和 scale。
- embedding/classifier 是否共享、是否仍含 bf16 副本。
- 常量区、descriptor、工作区和对齐浪费。
- KV cache 随上下文长度的增长。
- host staging buffer 和设备临时 buffer 的峰值生命周期。

注意：原始 checkpoint 文件约 2 GB，不等于设备归档大小；约 1.24 GB / 1178 MiB 的估计只有在 tied embedding/classifier 被重新量化并去重后才成立。

**预估**（manifest 出来前的量级）：int8 权重 = 嵌入 2048 × 128256（262.7M）+ 16 层 × 60.8M（Q / O 2 × 4.19M、K / V 2 × 1.05M、gate / up / down 3 × 16.8M）≈ 1236 MB = 1178 MiB；KV cache 每 token 16 层 × 8 头 × 64 × 2 = 16 KB，T = 512 时 8 MiB；pool 1280 MiB，余量约 90 MiB（Qwen3 的峰值 595 MB / heap 640 MB，工作区与 scale 很小），**大概率放得下**。主机侧也要预检：pool 与全局 CMA 之外 Linux 约剩 2.4 GB，加载 1.2 GB 的 irpa 时的主机内存峰值计入门禁。decode 量级：GEMV 部分按字节从 Qwen3（4 口约 92 ms / 607.5 MB）放大约 190 ms，加上固定部分，约 **4 tok/s**（down_proj 走 EX 时更低）。

内存门禁：

- `archive_bytes + peak_workspace + kv_bytes(T) + safety_margin ≤ pool_bytes` 才允许上板。
- safety margin 建议至少 32–64 MiB，并在实测后固定。
- 不把“扩大 1280 MiB pool”作为默认解决方案；现有 CMA、initrd 和低 2 GB 地址空间必须先做无冲突证明。
- 若短上下文仍无法满足门禁，则将高 2 GB DDR（K5-M M2）提升为 Llama 板上验收的前置任务。

### 9.4 EX 正确性基线

**已具备**：K 放不进时按 K 分块走 EX 是 C6.3 已完成的工作（`iree_compiler_plan.md` §8.9；合成模型 K = 16384 逐位一致，Llama W2 按 2 块执行）；编译器只在 `K × 16 ≤ 65535` 时用 GEMV（`SahlExpandKernels.cpp` `expandLinear`），所以 down_proj 现在就自动走 EX。这一步只需跑通并记录数据。

- K=8192 的 down_proj 走 EX 路径（现状）。
- 完成逐 dispatch 和端到端 sim，记录 EX 基线周期和 token/s。
- 若内存门禁通过，再做板上正确性验收。

### 验收

- tokenizer/token ID 与参考实现一致，或明确采用主机标准 tokenizer。
- RoPE 改写前后 eager 输出在约定容差内一致。
- memory manifest 给出可上板/不可上板的明确结论。
- EX 路径逐 dispatch 正确，质量参考为原版 bf16 模型。

## 10. 工作包 WP6：GEMV K 分块

### 10.1 分块规则

GEMV 描述符的 `row_bytes`（w2[63:48]）是 16 位、等于 K × 16，所以 K ≤ 4095；K 还必须是 16 的倍数，x 缓冲每 lane 4096 字节。因此安全上限为：

```text
K_BLOCK_MAX = floor(4095 / 16) * 16 = 4080
```

不得使用 4096。K=8192 的示例切分为：

```text
4080 + 4080 + 32
```

最后一块按现有 GEMV 尾部语义处理；若硬件要求完整 strip，则由编译器显式 padding，且 padding 不得越过归档边界。

### 10.2 编译器实现

硬件与下层已就绪：`sahl.gemv` 带 accumulate 标志，`SahlToSahw` 的 lowering 已设置描述符的 `acc` 位（w3[58]），funcsim 实现了 acc = 1 的语义；工作量只在 `expandLinear`（现在 `useGemv = gemvPorts && K × d ≤ 65535`，超出则整层走 EX）。

- 在 `expandLinear` 中判断 K、x-buffer 容量、descriptor 字段和对齐要求。
- 为每个 K block 计算 x/weight 基址偏移、有效 K 和 strip 数。
- 第一块 `acc=0`，后续块 `acc=1`，在同一 ACC 行累加。
- 仅最后一个 block 执行 requantize、bias、激活和写回，避免逐块舍入。
- 添加开发开关 `--iree-sa-gemv-k-split=0|1`，用于正确性对比和消融。
- 编译日志输出切分方案，例如 `K=8192 -> [4080, 4080, 32]`。
- **x 驻留**：GEMV 单元的 x 缓冲按 (xword, K) 判断驻留。若 K 块在 strip 组的内层循环，每条命令换一次 x 段，每次重读约 255 个字（down_proj 每条命令约 8160 周期数据，约 3%）。可选：K 块放外层（同一 K 段跑完所有 strip 组，部分和留在 ACC，需要 N × 4 字节 = 8 KB 的 ACC），或接受这 3%；先实现简单的，用 profile 决定。

### 10.3 溢出与数值语义

累加是 int32 回绕、中途没有饱和或舍入（`k2b_gemv_design.md` §4 的语义），而 |Σ| ≤ K × 128 × 128 = 8192 × 2¹⁴ = 2²⁷ ≪ 2³¹，K ≤ 131071 时不可能溢出，所以加法顺序无关。requantize 在 epilogue（读 ACC 之后），每块结束不发生截断。测试只需：

- 一个全 -128 × -128 的最坏情况用例（确认 2²⁷ 量级的和正确）。
- K block 边界附近的 4080、4096、4112、8192（16 的倍数）。

### 验收

- 合成 K>4080 模型与 EX 路径逐位一致。
- Llama down_proj 明确显示为多条 GEMV descriptor，且逐 dispatch 一致。
- K 分块开/关的 Llama 输出一致，性能差异入编译器消融表。
- 现有 K≤4080 黄金语料不发生行为变化。

## 11. 工作包 WP7：Llama 完整验收

### 实现

- 启用 K 分块重新编译 Llama。
- 完成 sim 端到端；内存门禁通过后进行四口板上运行。
- 收集 down_proj 从 EX 改为 GEMV 前后的周期、DDR 字节、token/s 和 dispatch 分解。

### 验收

- 逐 dispatch 与 SA oracle 一致；板上与 sim 逐位一致。
- 报告 decode、prefill、质量、内存峰值和覆盖率。
- 性能数字区分估算、sim 和板上实测。

## 12. 工作包 WP8：Gemma-3-270M 短上下文 bring-up

### 12.1 第一阶段范围

- 以 sim 为主，允许少量 host fallback。
- tokenizer 初期使用标准 `tokenizers` / Transformers 预生成 token IDs，不让 SentencePiece/byte fallback 阻塞编译器覆盖研究。
- 从 T=16/32/64/128 逐级增加，不预设 T=512 可以放入当前 SPAD。

### 12.2 关键适配

- 支持约 5 层 sliding-window + 1 层 full attention 的层型模式；T≤窗口时语义相同，但仍分别验证局部/全局 RoPE base。
- 支持 4 个 RMSNorm、`1 + w` 形式、embedding 乘 `sqrt(d)` 和 QK norm。
- GELU-tanh 改写为设备可执行图，但正确性参考必须使用设备 EXP/RECIP 的近似语义。
- 对 `x / (1 + exp(-2u))` 增加大正/负输入、clamp、overflow 和舍入测试，不宣称与 PyTorch GELU 逐位一致。
- 在 planner 中对 attention 的 K/V、临时 logits 和工作区做显式 SPAD 容量预检。
- head_dim = 256 是 Qwen3（128）的两倍：除容量外，先检查 attention 微内核、QK-norm、RoPE 改写对 head_dim 的假设与限制。

### 12.3 T=512 风险

head_dim=256 时，单个 K 张量在 T=512 已约为 131072 bytes，可能超过当前单银行容量。因此：

- 第一版只承诺实际通过容量检查的上下文长度。
- 若 T=512 不通过，文档明确标记 C6.5 T 分块为长上下文前置任务。
- 不以“局部层和全局层在 T≤512 算法相同”推导“现有 SPAD 一定能容纳”。

### 验收

- 至少一个短上下文在 sim 端到端完成。
- 生成完整回退清单和四类覆盖率。
- 不能下沉的算子都有可操作的原因分类：frontend、rewrite、lowering、硬件 primitive、内存容量或 tokenizer/runtime。

## 13. 工作包 WP9：测量基础设施与覆盖矩阵

### 13.1 运行时统计

扩展 `SA_STATS=1`，为每个动态 dispatch 记录：

- 名称、算子类别和关键 shape。
- 执行位置：host / accelerator。
- 调用次数。
- host dispatch 墙钟时间。
- accelerator 设备周期、命令数、估算 DDR 字节和 MAC。
- 回退原因。

另外：prefill 导出每个位置的 logits（§3.2 的板上 teacher forcing 需要），以及 decode 每步固定部分（attention EX、VE / SFU、LD 命令延迟、dispatch / 主机开销）的分类计时，用于 §16 的决策。

若 host 与 accelerator 可能异步重叠，分别报告两侧时间，并额外使用端到端墙钟作为总时间；不直接相加。

### 13.2 最终主表

| 模型 | 权重来源 | 正确性 | 质量 | 动态/时间/MAC覆盖率 | decode | prefill | 带宽利用率 | 内存峰值 |
|---|---|---|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | 自量化 / 外部 W8A8 | | | | | | | |
| SmolLM2-135M | llm-compressor | | | | | | | |
| Llama-3.2-1B | RedHatAI W8A8 | | | | | | | |
| Gemma-3-270M | 自量化 / llm-compressor | | | | | | | |

### 13.3 两张消融表

不要把硬件资源和纯编译器功能混在同一张表。

**硬件消融**：EX → GEMV 1 口 → 2 口 → 4 口，并单列 P6/REDUCE 等 bitstream 特性；列 token/s、设备周期、DDR 带宽、LUT、FF、BRAM、DSP、WNS。

**编译器/模型消融**：自量化→外部 W8A8、K 分块关→开、prefill 逐行关→开、host fallback→accelerator；列质量、覆盖率、周期、token/s 和内存峰值。K 分块不重复填写相同硬件资源。

## 14. 时间表与停止条件

| 日期 | 目标 | 停止条件 / 降级动作 |
|---|---|---|
| ~~10/11–10/14~~ | WP0–WP1：四口收尾 | **10/10 已完成**（剩 perf_model 校准，可与 WP2 并行）；bitstream 冻结为 `d16_100mhz_w128_p6_gemv_np4` |
| 10/11–10/18 | WP2：importer schema 与合成测试 | 10/18 前格式解析不稳定，则缩小到单一已验证 checkpoint schema |
| 10/19–10/24 | WP3–WP4：SmolLM2、Qwen3 | 10/24 前 Qwen 质量无改善，也必须输出四层定位结论，不继续盲调 scale |
| 10/25–10/29 | WP5：Llama 前端、EX 基线、内存预检 | 内存门禁失败则暂停板上 staging，保留 sim 结果并启动 M2 评估 |
| 10/30–11/03 | WP6–WP7：K 分块与 Llama | K 分块不稳定则演讲使用 EX 正确性基线，把性能优化标为进行中 |
| 11/04–11/09 | WP9 初版、消融表 | 先完成可复现数据，不等待 Gemma |
| 11/04–11/12 | WP8：Gemma 短上下文 sim | 11/10 前未端到端，则展示导入覆盖、首个阻塞点和回退清单 |
| 11/13–11/15 | 表格冻结、演讲材料 | 11/15 后不引入新硬件或大范围 compiler refactor |
| 11/16–11/25 | 幻灯片、复现实验、排练 | 只修阻断性 bug；所有新增结果必须能完整复现 |

## 15. 提交拆分建议

每个提交应能独立测试和回退：

1. ~~`docs: sync KV260 G4/G5 status and metrics`~~（`fb01909`、`b42fec6`）
2. `frontend: parse compressed-tensors W8A8 config`
3. `frontend: add external int8 weight/scale representation`
4. `frontend: preserve imported int8 values including -128`
5. `tests: add synthetic compressed-tensors checkpoints`
6. `frontend: import SmolLM2 and Qwen3 W8A8 checkpoints`
7. `tools: add quantization quality comparison pipeline`
8. `frontend: add Llama 3.2 tokenizer and RoPE coverage`
9. `tools: add device memory manifest and preflight`
10. `compiler: split GEMV K at aligned 4080-element boundary`
11. `tests: cover GEMV K split boundaries and overflow`
12. `runtime: report per-dispatch host/device timing and fallback reason`
13. `frontend: add Gemma 3 graph rewrites and short-context bring-up`

避免在同一提交中同时修改 importer、lowering、RTL 和黄金输出；否则逐位不一致时难以定位责任层。

## 16. 会后性能项的选择规则

使用 Qwen、Llama、Gemma 三个模型的耗时分解做决策：

| 观测结果 | 优先项 |
|---|---|
| decode 每步的固定部分占比高（四口实测：Qwen3 约 42%、SmolLM2 约 65%） | H2 乱序发射 + LD 命令流水 + 减少 dispatch 数与主机开销；按 WP9 的分类计时先找最大项 |
| decode 大部分时间为权重流量，四口接近端口上限 | W4A8（只缩短 GEMV 部分）；先做质量可行性实验，再做硬件解包 |
| 多模型仍受 100 MHz 算术/控制周期限制，且时序修复路径明确 | K2a-2 200 MHz |
| LD 命令气泡在多个模型中稳定超过约 8–10% | H2 + LD 命令流水 |
| Qwen prefill 的 SwiGLU/逐元素段占比高 | prefill 逐行执行 |
| Gemma/长上下文 attention 因容量失败 | C6.5 T 分块，再评估 K3 URAM |
| prefill 明确受 SPAD bank 冲突限制 | SPAD 子银行 |

候选必须同时给出：受益模型、时间占比上限、预期收益、资源预算、时序风险、质量风险和验证成本。

## 17. 最终完成定义

本阶段完成时，应能用同一套证据回答以下问题：

1. 编译器能导入哪些公开 W8A8 checkpoint，拒绝哪些格式，为什么？
2. Qwen3 的质量损失有多少来自 checkpoint，本项目又额外引入多少？
3. Llama 3.2 的 K=8192 如何在不改硬件的情况下映射到 GEMV，并保持逐位一致？
4. 三个模型家族中，多少动态工作在加速器执行，剩余热点在哪里？
5. 四口 GEMV 的实测收益、带宽利用率和资源代价分别是多少？
6. 下一项硬件投资为什么是 200 MHz、W4A8、URAM 或其他候选中的某一个？

只要这些问题有可复现的数据回答，即使 Gemma 仍有部分 host fallback，本阶段也已经形成比单纯追求额外 10% 性能更强的编译器与系统贡献。

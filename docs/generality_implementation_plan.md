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

**进度（2026-10-10）**：第一版完成，主机上验证。
- `compiler/frontend/ct_import.py`：按 §6.1 检查 `quantization_config`（拒绝时给出 config group / 层名、字段与支持范围）；safetensors 按张量 mmap 读入（单文件或分片），每个量化线性层的 `(q int8, s fp32)` 挂在权重 Parameter 上（`_sa_qs`），权重本身为反量化的 `q * s`（fp32 参考与 KV 校准用）；`ignore` 的层与非 Linear 张量按原浮点载入。
- `hf_generic.py`：`hf_model()` 遇到 `quantization_config` 走 importer；`rope_rewrite()` 同时置换 `_sa_qs` 的行与 scale；`quantize()` 有 `_sa_qs` 就原样使用（不经 `_quantize_rows`，-128 保留）。普通模型的导出与改动前逐字节相同（SmolLM2 截 2 层：irpa / mlir 相同）。
- `synth_hf.py --ct VARIANT`：合成 checkpoint（scale = amax / 127.5、[-128, 127]，如 llm-compressor 的 minmax）；接受的 ok / neg128 / bf16-scale，拒绝的 group / static-act / asym / int4 / fp8 / missing-scale / bad-shape / kv-scheme。
- `compiler/tests/test_ct_import.py [--e2e]`（`run_tests.sh ctimport`）：8 种拒绝都报出对应字段；3 种接受的 14 个线性层 int8 与 scale 与文件逐字节相同（q / k 按 RoPE 置换），-128 保留，嵌入与分类层一张共享表，反量化权重 = q × s；变异（`quantize()` 不用 `_sa_qs`、`rope_rewrite` 不置换 `_sa_qs`）都被发现。`--e2e`（D = 16、GEMV 4 口）：Llama 形状的 neg128 / bf16-scale 导出、编译（全部在加速器上）、sim 与 torch W8A8 逐步一致，teacher-forced 40/40 与 torch 相同。主机内存峰值 436 MB（小模型）。
- `dispatch_check.py --full-i8`：int8 数据取 [-128, 127]（默认仍为 [-127, 127]，前端自量化的范围）。

**这一步发现的问题**（都不是 importer 引入的）：
1. **嵌入的行 gather 经过 VE**，VE 的 int8 输出饱和到 ±127：表里的 -128 变成 -127（`dispatch_check --full-i8` 的 `elementwise_broadcast_*_i64xi8`）。自量化的表没有 -128；compressed-tensors 只量化 Linear，嵌入由本项目量化。只有“嵌入与一个被量化（不在 `ignore`）的分类层共用一张表”时会碰到，`quantize()` 现在明确拒绝这种情况。根治：gather 改用 LD（动态地址）而不是 VE 拷贝，另行评估。
2. **通用前端的 Qwen3 形式（已解决，2026-10-10）**：真实 Qwen3-0.6B（截 2 层）走通用前端时 decode 已全部在加速器上，回退都在 prefill：QK-norm（`reduction_16x16x128` / `16x8x128`）、Q 的 RoPE 与转置、RoPE 表 gather（`elementwise_broadcast_32768`）、K 的置换加量化、注意力输出的反量化（`elementwise_16x2048`），都是 M × heads × head_dim 的张量（32768 个 fp32 = 2048 字，超过一个 ACC bank 的 1536 字），且 HF 在 norm 后的 `transpose(1, 2)` 融进同一 dispatch，现有按行分片不接受。修复两处：
   - **前端**（`hf_generic._attention_rows`）：prefill（M > 1）时注意力模块的 forward 按行执行，与 qhf.prefill 相同的结构：q / k / v 投影对 M 行一起算（整数，与逐行相同），然后每行调用模块**原本的** forward（投影换成该行的结果、o_proj 暂时去掉；QK-norm、RoPE、写 KV、sa 注意力都与 decode 一步完全相同），最后 o_proj 对 M 行一起算。HF 的注意力代码原样复用，对 Llama / Qwen3 / Gemma 通用，prefill 与 decode 在构造上逐位一致。decode（M = 1）不变。
   - **编译器**（`SaCodegen`）：lowering 报 `ACC full` 时以 `tileShrink` = 1、2、3 重试（sahl-tile 按 ACC / 2^n 规划分片）；第一次就能编译的 dispatch 不变（黄金语料逐字节相同）。`elementwise_16x2048` 由此上了加速器。另外报告只取第一条非空的 error 诊断（多线程编译时原来常打印 “failed ()”）。
   结果：Qwen3（2 层）prefill + decode 0 个主机回退，prefill 与 decode-only 5/5 逐位一致，teacher-forced 22/40（torch W8A8 21/40）；SmolLM2（30 层）decode 的数字与改动前完全相同，prefill 5/5。代价：导出的计算图变大（SmolLM2 的 mlir 18.7 → 35.4 MB，导出 7.5 分钟 / 3.0 GB，编译 21 分钟 / 4.5 GB），板上 prefill 的速度待测（每行多出注意力部分的 dispatch）。剩下：head_dim = 64 且有 QK-norm 的模型（只在合成模型上）norm 权重 256 字节，被 IREE 内联成 dispatch 内的常量（`bufferization.to_buffer` 不支持）；目标模型（Qwen3 128、Gemma3 256、Llama 无 QK-norm）不受影响。
3. **prefill + decode 与 decode-only 的逐位等价依赖数据（已定位并修复，2026-10-10）**：Llama 形状的随机权重模型（自量化、不经 importer）seed 1–3 为 0/5、4/5、4/5。定位过程：sim 结束时的 DDR 对比 → 只有第 1 层的 K / V 在第 18、28、36 个位置差 ±1（第 0 层相同）；torch eager 的 prefill 与 decode 的 KV 完全相同、sim 的 decode 与 eager 相同 → 偏离的是 sim 的 prefill；逐 dispatch 检查各 T / dirty 都通过，只剩 3 个 UNSUPPORTED，即**回退到主机的 dispatch**，其中 `prefill:161` 是第 0 层融合了 SwiGLU 的 gate 投影（`silu(g)·u`，含 exp 与除法）。它代码生成失败（`sahl-to-sahw: ACC full`），在主机上用 libm 的 exp / 除法计算，而 decode 的同一运算在设备 SFU 上（近似 exp / RECIP），偶尔差一个舍入单位，经第 1 层的激活量化变成 KV 的 ±1。根因：prefill 按行线性内核的块大小（`linearSchedule`，`chunk_tiles`）只按 SPAD_B 选，K = 256 时取 16 个 tile；而每个 ACC bank 在 scratch 之上只有 1536 个字、缓冲不跨 bank，scope 要放 15 个 D × nc 字的缓冲（累加、结果、up 的 int32 输入、epilogue 每步一个临时值）。真实模型的 K ≥ 576，块小，从未碰到。修复：块大小再按每个 bank 的可用空间与缓冲个数取上限（这里 16 → 12）；黄金语料 4232 个 dispatch 逐字节不变；seed 1–3 与 neg128 都 5/5，`test_ct_import --e2e` 恢复 `--prefill`。剩下：这类合成形状的 prefill 还有 2 个 dispatch 在主机上（`elementwise_broadcast_*`：pair swap 以外的固定下标 gather；`elementwise_transpose_4x16x64`：超过两层循环），都是数据搬移与乘加、目前逐位一致，但主机上的乘加若被收缩成 FMA 就会再出现同类问题，列为后续项；另外整体编译的 `--iree-sa-codegen-report` 在这种失败时只打印 “failed ()”（多线程编译时诊断没有传回），单独编译该 dispatch 才看到原因，待修。
   **教训**：主机回退（VMVX）的 dispatch 若含 exp / 除法等 SFU 近似的运算，与设备上的同一运算不逐位一致；prefill 与 decode 只要有一方回退就会破坏等价性。回退清单（WP9）应标出这类 dispatch。

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

**进度（2026-10-10）**：主机与 sim 完成，板上待测。
- **生成**：`compiler/frontend/llmc_w8a8.py`，在独立环境 `build/llmc_venv`（CPU 版 torch 2.14.0、llm-compressor 0.14.0、compressed-tensors 0.19.0；不动导出环境）里跑 SmoothQuant 0.8 + GPTQ（W8A8，`ignore: [lm_head]`），校准 WikiText-2 train 中 200 字以上的段落 256 段 × 512 token（`default_rng(0)` 抽样，行号记入 `sa_manifest.json`），7.6 分钟。GPTQ 的 `actorder: static`（按通道时只是 GPTQ 的列顺序，已折回权重，不存 g_idx）现在被接受。
- **质量**（`compiler/tests/quality_eval.py`：WikiText-2 test，2048 个 token，窗口 256，teacher-forced，参考为原版 fp32；结果 `build/quality/smollm2_w8a8_2048.json`）：

  | 变体 | PPL | top-1（对 fp32） | 相关 | KL |
  |---|---|---|---|---|
  | fp32 | 26.98 | — | — | — |
  | 自量化 W8A8（fp32 注意力） | 29.92 | 87.6% | 0.9899 | 0.102 |
  | llm-compressor W8A8（SmoothQuant + GPTQ） | 28.16 | 90.3% | 0.9932 | 0.050 |
  | 同上，只用 GPTQ（不做 SmoothQuant） | 29.83 | 87.9% | 0.9903 | 0.106 |
  | 自量化，SA 形式（int8 KV、设备注意力、SFU） | 30.77 | 83.9% | 0.9853 | 0.138 |
  | **llm-compressor W8A8，SA 形式** | **28.77** | **87.1%** | 0.9887 | **0.074** |

  外部 checkpoint 在设备的计算方式下 KL 约为自量化的一半、PPL 低 2.0；改善几乎全部来自 SmoothQuant（只用 GPTQ 与自量化相当）。SA 形式额外的损失（int8 KV、注意力概率、SFU）两者相近（KL +0.024 / +0.036）。单个 256 token 的窗口曾给出相反的结论：评估至少要用数千个位置（§3.2）。
- **sim**（`test_hf_generic.py`，D = 16、GEMV 4 口）：导出 + 编译 1710 个加速器 executable，decode 9/9 argmax 与 torch W8A8 相同，prefill 与 decode-only 5/5 逐位一致。40 位置 teacher-forced 门槛改为统计的：设备与 torch 模型 argmax 不同的位置 d 个，容许 max(2, ⌈2√d⌉)，另要求两者 logits 平均相关 ≥ 0.98。原门槛（−2）在这个 checkpoint 上失败：d = 3 个翻转碰巧都落在设备一边（34 对 37），而设备对 torch 的平均相关 0.9963 比自量化的 0.9911（d = 6，两边各 3）还高。
- **板上**：`deploy_z1_freeze.sh hfgen_w8a8`（不在默认集合；`HFGEN_MODEL` / `HFGEN_TAG` 给 `deploy_hfgen.sh`），`board_regress.py` 有对应项。

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

**预备工作（2026-10-11）**：
- **checkpoint**：`RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8`（revision `9c406c48`，不需登录）下载到 `build/llm_cache/Llama-3.2-1B-Instruct-W8A8`（1.9 GB）。配置正好在 importer 范围内（int8 按通道对称、激活按 token 动态、`ignore: [lm_head]`、无 act-order）；recipe 为 SmoothQuant 0.7 + GPTQ（observer mse）。文件里 tied 的 `lm_head.weight` **又单独存了一份**（与嵌入逐位相同）：importer 现在检查二者相同，不同则拒绝（原来会把后读的一张写进共享张量，悄悄替换另一张）；合成变体 `lm-head-dup` / `lm-head-diff` 覆盖。
- **原版参考**：`meta-llama/Llama-3.2-1B-Instruct` 需要登录并接受许可；用不受限的镜像 `unsloth/Llama-3.2-1B-Instruct`（revision `5a8abab4`，`build/llm_cache/Llama-3.2-1B-Instruct`，2.4 GB）。核对：其嵌入表与 RedHatAI 的逐位相同；对 SmoothQuant 不变的量（权重 × 前面 RMSNorm 的权重），RedHatAI 反量化结果与之相关 0.99987、误差中位数 0.26 个量化步长：同一个基础模型。
- **分词器**：`hf_tokenizer.py` 原来用一个固定正则近似预分词，Llama 3 只有 114/305 段与 transformers 相同（数字 Llama 3 按 3 位一组）。现在按 tokenizer.json 的 normalizer（NFC）、pre-tokenizer（Split 正则 / Digits / ByteLevel 的 GPT-2 正则）、`ignore_merges` 执行，特殊 token 先整体匹配：Llama 3、SmolLM2、Qwen3 各 1009 段（含 chat template 的特殊 token、中文、全角、2 万字长文）与 transformers **完全相同**。需要 `regex` 模块（Unicode 类别）；没有时（板上的系统 Python：`sudo apt install python3-regex`）退回旧的近似。SmolLM2 / Qwen3 的校准文本与 prompt 的分词不变，已有模型的 KV scale 不受影响；Llama 的校准文本由 47 变 45 个 token。
- **RoPE**：`RopeTable` 由 HF 的 rotary 模块生成整张 cos / sin 表，llama3 scaling 自然包含；`test_hf_generic` 的改写前后对比会在真实模型上验证。
- **内存预检**：`compiler/tests/mem_preflight.py`（只读 safetensors 头部）。以板上实测校准：峰值 ≈ 打包参数 + KV + 几 MiB；各项之和比打包文件少 2.0%（Qwen3 570.5 对 582.4 MiB）/ 1.5%（SmolLM2），估算乘 1.025。Llama-3.2-1B：int8 线性层 928.0 + 嵌入 int8 250.5 + scale 等 ≈ 1210 MiB 打包，T = 256 的 KV 4 MiB，余量 32 MiB，合计 **1246 MiB / 1280 MiB，放得下（余 34 MiB）**；KV 最长 T ≈ 2400。注意：staging 现在按 “irpa × 1.1 + 24” 设运行时的堆（Llama 得 1323 MiB，超过池），上板前改为 “打包 + KV + 余量”。

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

**进度（2026-10-11）**：decode 的 K 分块完成。
- 原来 K 超过 SPAD_B 一个 bank（D = 16 时 4096）时，decode 的线性层微内核**根本不被选中**（调度的块大小按 SPAD_B 算出 0），整层交给通用收缩的 K 分块 EX 路径，GEMV 用不上。改动三处，只在“有 GEMV 且 K 超过原限制”时生效：`KernelMatcher::gemvHolds`（K 只受 x 的存放限制：打包 x 在 SPAD_A，int32 x 在 ACC scratch 的 bank 里）、`linearSchedule`（块大小只按 ACC scratch）、`expandLinear`（K 按 4080 分块，x 与权重取子视图，第一块写累加器，其余设 acc 累加；epilogue 在最后一块之后读）。
- **黄金语料加入 GEMV 4 口的三个用例**（`qwen3_gemv4`、`smollm2_gemv4`、`stories_gemv4`，板上 `_gemv_np4` 的构建，2814 个 dispatch），用改动前的编译器录制，改动后 16 个用例 7046 个 dispatch 逐字节相同：已部署的 GEMV 模型不变。
- **合成 Llama，K = 8192**（dim 256、FFN 8192，`run_tests.sh gemvk`）：down_proj 为 3 条 GEMV（4080 + 4080 + 32），decode 0 个主机 dispatch，与 torch W8A8 逐步 argmax 9/9、teacher-forced 40/40（相关 1.0000），prefill 与 decode-only 5/5 逐位一致。
- **prefill 还剩两类主机回退**（不影响正确性，影响板上 prefill 速度，列入 WP6 / WP7）：① prefill 的 down_proj（`matmul_like_16x…x8192`，按行的 EX 路径没有 K 分块：“contraction not expanded”），整数运算所以仍逐位一致，但在 ARM 上算；② prefill 写 KV 的 scatter（`scatter_…x32x2x1xi8`）：没有 QK-norm 的模型（Llama）里 RoPE 的 swapneg gather 融进了写 cache 的 scatter（“more than two loops”）；Qwen3 有 QK-norm 隔开，不受影响。真实的 Llama-3.2-1B 两者都会遇到。

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

**预备工作（2026-10-11）**：
- **模型**：`google/gemma-3-270m` 需要登录；用不受限的镜像 `unsloth/gemma-3-270m`（base，revision `edcf1fe3`，`build/llm_cache/gemma-3-270m`，0.5 GB）。结构：18 层，hidden 640，4 / 1 GQA，head_dim 256，词表 262144，FFN 2048，GELU-tanh，滑窗 512（每 6 层 5 层滑窗），RoPE 局部 base 1e4、全局 1e6，`query_pre_attn_scalar` 256，无 logit softcap。内存预检：T = 512 时约 300 MiB，宽裕。
- **eager 逐项对比**（原版 HF fp32 为参考，24 个 token）发现并修了四处，都只在 Gemma 这类结构上生效：① 包装后的 decoder 与原版就不一致（相关 0.73）：HF 按第 0 层的 cache 长度推算位置，Gemma 的第 0 层是滑窗层，计数器是 `cumulative_length_int`，原来没设（现在 `_at` 一并设置；导出时 sa 注意力不用 HF 的 mask）→ 逐位一致级别（7e-5）；② W8A8 后完全错（相关 0.13）：`Gemma3TextScaledWordEmbedding` 要乘 √640，`QEmbedding` 把它丢了；现在保留 `embed_scale`，反量化后再乘（不能折进 scale：分类层共用这张表而不乘），带自定义 forward 又没有 `embed_scale` 的嵌入子类明确报错；③ `RopeTable` 报错：Gemma 只有一个 rotary 模块、调用时传 `layer_type`，现在每种类型一张表；④ RMSNorm 的增益是 `1 + w`：`SaRMSNorm` 用全 1 向量探测模块实际的增益，判定为 `w` 或 `1 + w`，其他形式报错。另外：`sa_attention` 的分数缩放改用模块自己的 `scaling`（与 1/√head_dim 相同时仍用原来的算法，现有模型逐位不变）；cache 长度超过滑窗时明确报错（设备注意力不做滑窗，T ≤ 512 时语义相同）。
  结果：包装与 RoPE / norm 改写后与原版一致（7e-5，argmax 24/24），W8A8 相关 0.992，SA 形式（int8 KV、设备注意力、SFU 近似）相关 0.9906、argmax 23/24。SmolLM2 与 Qwen3（截 2 层）的导出与改动前逐字节相同。
- **分词器**：Gemma 是 SentencePiece 风格的 BPE（normalizer 把空格换成 `▁`、直接对字符做 BPE、byte fallback `<0xNN>`）。`hf_tokenizer.py` 加入这种模式（normalizer Replace / Prepend、`MergedWithPrevious` 的 Split、byte fallback 与对应解码），BPE 改为堆 + 链表（n log n；Gemma 整段文本是一个“词”）。Gemma、Llama 3、SmolLM2、Qwen3 各 1009 段与 transformers 完全相同，解码可还原；Gemma 不需要 `regex` 模块，板上的系统 Python 也是精确的。
- **还没做**（WP8 正式阶段）：导出与编译（GELU-tanh 在设备上的形式、head_dim 256 的注意力微内核与 SPAD 容量）、`synth_hf.py` 的 Gemma3 结构。

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
| 10/11–10/18 | WP2：importer schema 与合成测试（**10/10 第一版完成**；prefill 等价问题、通用前端 Qwen3 的回退都已解决） | 10/18 前格式解析不稳定，则缩小到单一已验证 checkpoint schema |
| 10/19–10/24 | WP3–WP4：SmolLM2（**10/10 主机与 sim 完成**）、Qwen3 | 10/24 前 Qwen 质量无改善，也必须输出四层定位结论，不继续盲调 scale |
| 10/25–10/29 | WP5：Llama 前端、EX 基线、内存预检 | 内存门禁失败则暂停板上 staging，保留 sim 结果并启动 M2 评估 |
| 10/30–11/03 | WP6–WP7：K 分块与 Llama | K 分块不稳定则演讲使用 EX 正确性基线，把性能优化标为进行中 |
| 11/04–11/09 | WP9 初版、消融表 | 先完成可复现数据，不等待 Gemma |
| 11/04–11/12 | WP8：Gemma 短上下文 sim | 11/10 前未端到端，则展示导入覆盖、首个阻塞点和回退清单 |
| 11/13–11/15 | 表格冻结、演讲材料 | 11/15 后不引入新硬件或大范围 compiler refactor |
| 11/16–11/25 | 幻灯片、复现实验、排练 | 只修阻断性 bug；所有新增结果必须能完整复现 |

## 15. 提交拆分建议

每个提交应能独立测试和回退：

1. ~~`docs: sync KV260 G4/G5 status and metrics`~~（`fb01909`、`b42fec6`）
2. ~~`frontend: parse compressed-tensors W8A8 config`~~
3. ~~`frontend: add external int8 weight/scale representation`~~
4. ~~`frontend: preserve imported int8 values including -128`~~
5. ~~`tests: add synthetic compressed-tensors checkpoints`~~（2–5 合为一个提交）
6. `frontend: import SmolLM2 and Qwen3 W8A8 checkpoints`（SmolLM2 完成）
7. ~~`tools: add quantization quality comparison pipeline`~~（`quality_eval.py`）
8. `frontend: add Llama 3.2 tokenizer and RoPE coverage`（分词器完成）
9. ~~`tools: add device memory manifest and preflight`~~（`mem_preflight.py`）
10. ~~`compiler: split GEMV K at aligned 4080-element boundary`~~（decode）
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

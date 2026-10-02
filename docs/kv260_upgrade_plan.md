# PYNQ-Z1 → Kria KV260 升级方案

状态：草案（2026-10-02）。本文给出从 PYNQ-Z1（Zynq-7020）迁移到 Kria KV260（K26 SOM，Zynq UltraScale+）的完整计划：分阶段目标、每个模块的改动、验收标准和风险。性能数字除"实测"外均为估算，需在板上验证。

相关文档：[`iree_compiler_plan.md`](iree_compiler_plan.md) §8.18（FPGA 规模估计）、[`double_buffer_design.md`](double_buffer_design.md)（加速器架构）、[`memory_model.md`](memory_model.md)（地址与一致性）。

---

## 0. 目标与原则

**目标**

1. 现有全部功能在 KV260 上逐位一致地运行（stories15M、SmolLM2-135M，prefill + decode）。
2. 把 decode 的瓶颈——权重读取带宽——从约 0.39 GB/s 提高到 **6–10 GB/s**。
3. 让 **Qwen3-0.6B（int8，全 28 层）** 在板上以可交互的速度运行（估计 10–15 token/s）。
4. 为后续的 int4、64 位地址、更大模型、RVV 核留出空间。

**原则**（沿用项目约定）

- **一次只改一类东西**：先换平台、参数不变，再提频率和带宽，再扩存储，最后上大模型。每一步出问题都能定位到这一步的改动。
- **每一步先 sim、再上板、逐位一致后提交**。逐位一致的参照：功能仿真器（`llm/sa_funcsim.py`）与 PYNQ-Z1 上已验证的结果。
- **编译器尽量不动**：硬件参数通过目标配置（`--iree-sa-d`、`--iree-sa-spad-kb`、`--iree-sa-acc-kb`）传入，黄金语料（4002 个 dispatch）守护代码生成不变。
- **保留 PYNQ-Z1 可构建**：新平台用新目录和参数，不破坏现有 bitstream 与测试。

---

## 1. 平台对比

| 项 | PYNQ-Z1（现状） | KV260（K26 SOM） |
|---|---|---|
| 器件 | Zynq-7020（xc7z020clg400-1） | Zynq UltraScale+（XCK26，与 ZU5EV 同级） |
| 处理器 | 2 × Cortex-A9（armv7，32 位），650 MHz | 4 × Cortex-A53（aarch64），约 1.3 GHz；另有 2 × R5F |
| 逻辑资源 | 约 53K LUT、220 DSP48E1、140 BRAM36；LUT 已用约 83% | 约 117K LUT、1248 DSP48E2、144 BRAM36、**64 URAM（约 2.25 MB）** |
| DDR | 512 MB DDR3 | **4 GB DDR4（64 位）**，理论约 19 GB/s |
| PS-PL 数据口 | HP0–HP3，各 **64 位**，AXI3 | HP0–HP3（及 HPC），各 **128 位**，AXI4 |
| 加速器时钟 | 50 MHz（实测时序约束） | 目标 200–250 MHz |
| 读权重带宽 | **实测约 0.39 GB/s**（1 个 HP 口，约 7.8 B/周期，利用率 99%） | 目标 6–10 GB/s（2–4 个 128 位 HP 口） |
| 软件 | PYNQ（armv7 Linux） | Kria-PYNQ 或 Ubuntu for Kria（aarch64） |

资源数字来自公开资料，按 AMD 数据手册核对。

---

## 2. 现有设计中与平台相关的部分

逐项盘点，按"不改 / 小改 / 重写"分类。

### 2.1 硬件（`rtl/`、`RISCV-on-PYNQ-Z1/`）

| 模块 | 分类 | 说明 |
|---|---|---|
| `rtl/sysarray/*.v`（阵列、VE、LD/ST、调度器、记分板） | 小改 | 通用 Verilog。`sa_unit` 已有参数 `D`、`DSP_COLS`、`NPORTS`、`SPAD_WORDS`、`ACC_WORDS`、`PERF`。DSP 推断从 DSP48E1 变为 DSP48E2（一般自动），`DSP_COLS` 的 DSP/LUT 列分配需重新评估；BRAM 推断模板（`sa_tdpram.v`、`sa_bankmem.v`）在 UltraScale+ 上需确认推断结果，SPAD/ACC 加大时改用 URAM |
| `rtl/sysarray/sa_ld.v`、`sa_st.v`（DMA） | 小改 → 中改 | 当前每个口 64 位数据（`m_rdata[NPORTS*64-1:0]`），本地写入每周期一个 64 位 lane。K1 用 AXI 位宽转换器接 128 位 HP 口；K2 再把数据通路加宽到 128 位 |
| `picorv32.v`、PCPI（`sa_pcpi.v`） | 不改 | 纯 RTL |
| `scripts/pico_bit.tcl`（block design） | **重写** | 器件写死为 `xc7z020clg400-1`；PS 是 `processing_system7`，要换成 `zynq_ultra_ps_e`；GP0 → HPM0_FPD/LPD；HP0/HP2（64 位，加 AXI4→AXI3 转换器 `matmulHpConverter`）→ HP0–HP3（128 位 AXI4，不再需要 AXI3 转换） |
| `scripts/build_bitstream.tcl/.sh` | 小改 | 增加 `-board kv260`；沿用 `-sa_d`、`-jobs` |
| `constrs/PYNQ-Z1.xdc` | 重写 | K26 只需时钟等少量约束（SOM 的 PL 引脚基本不用） |
| RISC-V 复位（EMIO GPIO[0]） | 小改 | ZynqMP 也有 EMIO GPIO，编号和 Linux 驱动不同；或改用 AXI GPIO |
| 完成中断（`mat_notify`） | 小改 | 接 `pl_ps_irq0`，设备树 / UIO 配置随之改 |

### 2.2 固件（`firmware/`）

| 模块 | 分类 | 说明 |
|---|---|---|
| `rt/rt_fw.c`（常驻命令处理器）、`include/sysarray_intrinsics.h` | 不改 | RISC-V 侧地址（BRAM `0xC0000000`、CSR `0x80000000`）不随 ARM 变化 |
| `common/link.ld`、`include/mailbox.h` | 不改（待确认） | mailbox 在 BRAM `0x1F00`，RISC-V 侧不变；ARM 侧基地址在驱动里 |
| 其他里程碑固件（`gemm/`、`bwtest/` …） | 不改 | K1 用它们做回归和带宽测试 |

### 2.3 主机软件

| 模块 | 分类 | 说明 |
|---|---|---|
| `driver/pynq_matmul.py`、`notebooks/` | 小改 | `BRAM_ARM_BASE = 0x40010000` 等基地址改为 ZynqMP 的 HPM 地址窗口（如 `0xA000_0000` 起，以 block design 为准）；overlay 名、CMA 分配 |
| IREE 运行时交叉编译（`iree-sa/l0/toolchain-armv7hf.cmake`、`build_runtime_armv7.sh`、`compiler/scripts/build_sa_runtime.sh armv7`） | 小改 | 新增 aarch64 工具链文件和 `build_sa_runtime.sh aarch64`；`armv7_libm_shim.c` 在 aarch64 上可能不再需要（待确认） |
| `compiler/runtime/sa/sa_transport_board.c` | 小改 | 通过 `/dev/mem` 映射设备窗口（`SA_BOARD_MEM=<phys>:<bytes>`、`SA_BOARD_MBOX=<phys>`），物理地址是 `uint32_t`。K1 保持 32 位，窗口必须在低 2 GB（见 2.4） |
| `compiler/tests/board_*.py`、`scripts/deploy_*.sh` | 小改 | 路径、架构、窗口大小、mailbox 地址 |

### 2.4 关键约束：32 位 DDR 地址

- 描述符的 DDR 地址字段是 32 位（`DescList`：`ddr & M32`），PicoRV32 也是 32 位核；运行时传物理地址用 `uint32_t`。
- KV260 的 4 GB DDR 在 ZynqMP 地址空间里分两段：**低 2 GB 在 `0x0000_0000` 起，高 2 GB 在 `0x8_0000_0000` 起**。
- **K1–K4 的做法：所有给加速器的缓冲（权重、KV cache、激活、描述符列表、环形队列）都放在低 2 GB。** 用设备树的 reserved-memory 或 CMA 区域固定在低地址，并在运行时检查分配到的物理地址 < 2 GB，否则报错。
- 容量够用：Qwen3-0.6B int8 权重约 0.6 GB，加 KV（2K 上下文约 117 MB）和激活，远小于可用空间。
- 64 位地址放到 K5（可选），只有上 1.7B int8 或 3B 以上模型才需要。

### 2.5 编译器（`compiler/plugins/sa/`）

| 模块 | 分类 | 说明 |
|---|---|---|
| sa 后端插件、`sahl`/`sahw` 流水线 | 不改 | 硬件参数来自目标配置 |
| `TargetConfig::invalid()` | 视情况 | 目前要求 D ∈ {8, 16}、片上存储不超过 2^16 字（16 位本地字地址）。SPAD/ACC 超过这个范围或 D = 32 时，需要同时改描述符编码、RTL 和这里的检查（见 K3） |
| 主机退路（VMVX） | 不改 | 与 CPU 架构无关 |
| 功能仿真器、`dispatch_check.py` | 不改 | 已支持任意 D 与存储大小 |

---

## 3. 分阶段计划

| 阶段 | 内容 | 主要风险 | 工作量 |
|---|---|---|---|
| **K0** | 准备：工具链、板卡上电、系统镜像 | 低 | 小 |
| **K1** | 平台移植，加速器参数不变（D = 16、1 个 HP 口、100 MHz） | block design、地址映射、aarch64 运行时 | 中 |
| **K2** | 带宽：200 MHz 以上、128 位 DMA、2–4 个 HP 口 | 时序收敛、DMA 改动 | 中–大 |
| **K3** | 片上存储：URAM 加大 SPAD/ACC；可选 D = 32 | 地址位宽、编码改动 | 中 |
| **K4** | 大模型：Qwen3-0.6B 全模型上板，性能调优 | 量化质量、注意力 T 分块 | 中 |
| **K5** | 可选扩展：64 位地址、int4、RVV 核、MoE | 各自独立 | 大 |

### K0 准备

1. **工具**：安装支持 K26 的 Vivado / Vitis 版本，下载 KV260 板级文件（board files）。确认所用 Vivado 版本的免费授权覆盖 XCK26。
2. **系统镜像**：Kria-PYNQ（推荐，Python 驱动和 notebook 基本可沿用）或 Ubuntu for Kria，写入 microSD，上电确认 Linux、网络、`/dev/mem` 访问权限。
3. **基线**：在板上跑 ARM 的 CPU 基线（对应 `notebooks/llm/arm_baseline.sh`），记录 A53 上 llama 推理的 token/s，作为加速器的对比对象。
4. **仓库结构**：新建 `KV260/`（与 `RISCV-on-PYNQ-Z1/` 并列），放 block design 脚本、约束、构建脚本和 bitstream 结果目录。

**验收**：板子能启动，能用 PYNQ（或 Python + `/dev/mem`）加载一个空 overlay。

### K1 平台移植（参数不变）

**硬件**

1. 新 block design（`KV260/scripts/kv260_bd.tcl`）：
   - `zynq_ultra_ps_e`，打开 HPM0_FPD（ARM → BRAM、CSR）、S_AXI_HP0_FPD（加速器 DMA）、`pl_clk0`（**100 MHz**）、`pl_ps_irq0`、EMIO GPIO（RISC-V 复位）；
   - PicoRV32（`picorv32_axi`）、程序 BRAM（8 KB，ARM 侧和 RISC-V 侧双口）、`sa_unit`（D = 16，`NPORTS = 1`）；
   - 加速器 64 位 AXI master → **AXI 位宽转换器** → 128 位 HP0（先不改 DMA RTL）；
   - PicoRV32 的 DDR 口（原 HP0，用于读环形队列）接 HP1 或经 SmartConnect 共用。
2. 地址映射：记录 ARM 侧的 BRAM、CSR、mailbox 新地址，写进 `docs/memory_model.md` 的 KV260 一节。
3. `synth_ooc.tcl` 加 K26 器件选项，先做脱离上下文的综合，确认资源与 100 MHz 时序。

**软件**

4. aarch64 交叉编译：`toolchain-aarch64.cmake`、`build_sa_runtime.sh aarch64`，构建 `sa-llm-run`、`sa_hal_test`、L0 的 `iree-run-module`。
5. 驱动与部署：`pynq_matmul.py` 和 `deploy_*.sh` 增加 `--board kv260`，读入新的基地址；`sa_transport_board.c` 的窗口参数改为低 2 GB 内的保留区域，并检查物理地址 < 2 GB。
6. 编译器：生成 D = 16、默认 SPAD/ACC 的模块，与 PYNQ-Z1 的 D = 16 bitstream（M4 起）相同。

**验证顺序**（每项都与 sim 逐位一致）

| 步骤 | 测试 | 预期 |
|---|---|---|
| 1 | `tests/ddr_access`：RISC-V 读写 DDR | 通过 |
| 2 | `bwtest`：DMA 带宽 | 100 MHz、64 位：约 0.8 GB/s（Z1 的 2 倍，来自频率） |
| 3 | `gemm`、`vector`、`desc_run` 固件 | 与 NumPy 逐位一致 |
| 4 | C1：`sa_hal_test`（手工模板） | 逐位一致 |
| 5 | C3：stories15M（`board_llm.py`） | 76/76 步与 DeviceModel 逐位一致 |
| 6 | SmolLM2-135M prefill + decode | 与 sim 逐位一致；记录 token/s |

**验收**：第 5、6 步通过。速度预计约为 Z1 的 2 倍（频率 50 → 100 MHz）。

### K2 带宽

decode 受权重读取带宽限制，这一阶段是性能提升的主要来源。

1. **提频率**：`pl_clk0` 提到 200 MHz，再试 250 MHz。关键路径大概率在 VE 的 fp32 单元（`sa_fp32_*.v`、`sa_vefp.v`）和 SFU 查表，按需加流水级；用 `synth_ooc.tcl` 迭代。
2. **加宽 DMA**：`sa_ld`/`sa_st` 的数据通路改为 128 位（或参数化 `DMA_W`），本地写入按 128 位一次两个 64 位 lane；去掉位宽转换器。
3. **多口**：`NPORTS` = 2，再试 4，分别接 HP0–HP3；`sa_ld` 已按口轮转发起读（`ar_rr`），需确认多口的乱序返回与写回顺序。
4. **测量**：`bwtest` 目标 **6–10 GB/s**（DDR4 理论约 19 GB/s，PL 侧通常能拿到一半左右）。
5. **编译器与运行时**：不需要改。带宽变大以后，每个 dispatch 的固定开销占比上升，用性能计数器（`board_profile.py`）看是否需要更大的 chunk 或更多的跨 dispatch 预取。

| 配置 | 估计读带宽 |
|---|---|
| 1 口 × 128 位 × 200 MHz | 约 3.2 GB/s |
| 2 口 × 128 位 × 200 MHz | 约 6.4 GB/s |
| 3–4 口 × 128 位 × 250 MHz | 受 DDR 控制器限制，约 8–10 GB/s |

**验收**：`bwtest` 实测达到目标；stories15M、SmolLM2 逐位一致；SmolLM2 decode 的 token/s 随带宽近似线性提升。

### K3 片上存储（与可选的 D = 32）

1. **SPAD/ACC 用 URAM 加大**：例如 SPAD 256 KB × 2、ACC 512 KB。更大的 chunk 减少每层的固定开销，也是长上下文注意力的前提。
   - 现有本地地址是 **16 位字地址**（`TargetConfig::invalid()` 要求每块存储不超过 2^16 字）。D = 16 时 SPAD 一字 16 字节，2^16 字 = 1 MB，ACC 一字 64 字节，上限更大，所以上述配置不需要改地址位宽。超过时需同步改描述符编码、RTL 和编译器检查。
   - 编译器：`--iree-sa-spad-kb`、`--iree-sa-acc-kb` 传入新大小；C6 已在 sim 上验证过 SPAD 256 KB / ACC 512 KB 和更大的配置（D = 16）。
2. **D = 32（可选）**：主要提升 prefill 的算力（decode 用不上）。需要 RTL（阵列、VE 宽度、存储字宽）、描述符编码、`TargetConfig`（目前只允许 8、16）、功能仿真器同时支持。建议放在 K4 之后、确认 prefill 是瓶颈再做。

**验收**：新配置下黄金语料按新目标重新记录，逐 dispatch 检查全部一致；上板逐位一致。

### K4 大模型：Qwen3-0.6B

1. **全模型上板**：28 层，int8 权重约 0.6 GB。C6.1 已在 sim 上逐 dispatch 逐位一致。
2. **量化质量（C6.2）**：torch W8A8 的 Qwen3 与 fp32 的 top-1 一致率只有 25/40。先按 `iree_compiler_plan.md` 的方法逐项定位误差来源（激活、KV、概率），再决定用 SmoothQuant（只改权重，硬件和编译器不变）还是更细的 KV scale。
3. **注意力按 T 分块（C6.5）**：head_dim = 128，上下文变长时一个头的 K 行放不进一个 SPAD bank，需要分块；这是剩下唯一的编译器新功能。
4. **性能调优**：用 `SA_PROFILE` 和性能计数器看每个 dispatch 与读权重下限的比值；通用 contraction 路径的线性层改走微内核（§8.19 的做法）。

**验收**：Qwen3-0.6B 在板上 prefill + decode 与 sim 逐位一致，生成文本正常；decode 达到 **≥ 10 token/s**（估计 10–15）。

### K5 可选扩展

| 项 | 内容 | 收益 | 依赖 |
|---|---|---|---|
| 64 位 DDR 地址 | 描述符地址字段、DMA 地址、PicoRV32 访存（或高位段寄存器）、运行时 `uint64_t` | 用满 4 GB：Qwen3-1.7B int8、3B–4B int4 | 改描述符格式（sa-desc 新版本） |
| int4 权重 | LD 通路上解包 int4；按组的 scale（每 64/128 个 K 一组），微内核按 K 组分段 EX、VE 缩放后累加 | 读取量减半，decode 约 2 倍；1.7B int4 约 8–10 token/s | 质量需配合 GPTQ/AWQ 一类方法 |
| RVV RISC-V 核 | 独立实验或替换 PicoRV32；评估 SiFive SKL、IREE 的 RISC-V 后端 | 不规则运算（采样、top-k、MoE 路由）、全 RISC-V 栈 | 资源、工具链 |
| MoE | 专家权重打包、动态权重基址的线性层微内核（`LDPARAM` + 动态 DMA 地址） | Granite-1B-A400M 这类小 MoE | K4、可能 64 位地址 |

---

## 4. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 缓冲分配到高 2 GB（≥ `0x8_0000_0000`） | 加速器读写错地址，结果错误且难查 | 保留区固定在低 2 GB；运行时检查物理地址并报错 |
| 200 MHz 以上时序不收敛 | 带宽目标达不到 | 先 200 MHz；VE fp32 和 SFU 加流水级；必要时 DMA 和计算分时钟域 |
| HP 口多口并发的实际带宽低于预期 | K2 收益打折 | `bwtest` 逐口测；对比 HP 与 HPC 口；调整突发长度和 outstanding 数 |
| PS 侧 cache 一致性 | ARM 写的数据加速器读不到最新值 | 沿用现有约定（非缓存映射 / 显式刷新），在 `memory_model.md` 补充 ZynqMP 的说明；不用 HPC 的一致性端口，除非测量证明有必要 |
| Kria-PYNQ 版本与 Vivado 版本不匹配 | overlay 加载失败 | K0 先确认版本组合；必要时用 Ubuntu + 自写的 FPGA manager 加载 |
| 编译器在新参数下出新的边界情况 | 某些 dispatch 错误 | 逐 dispatch 检查（`dispatch_check.py --dirty --skew`）、黄金语料、功能仿真器先行 |
| Qwen3 量化质量不够 | 输出文本质量差 | K4 第 2 步先定位误差来源，再选择修正方法 |

---

## 5. 各阶段指标汇总

| 阶段 | 加速器时钟 | 读带宽 | SmolLM2-135M decode | Qwen3-0.6B decode |
|---|---|---|---|---|
| PYNQ-Z1（实测） | 50 MHz | 约 0.39 GB/s | 约 2.2 token/s | 放不下（512 MB） |
| K1（估计） | 100 MHz | 约 0.8 GB/s | 约 4 token/s | 能放下，约 1 token/s |
| K2（估计） | 200–250 MHz | 6–10 GB/s | 带宽不再是唯一瓶颈，几十 token/s | 约 10–15 token/s |
| K4（目标） | 同 K2 | 同 K2 | — | **≥ 10 token/s，逐位一致** |

估算方法：decode 每 token 读一遍全部权重，速度 ≈ 可用带宽 ÷ 每 token 读取的字节数；K2 之后，小模型的每 dispatch 固定开销和 VE 的 SFU 延迟会成为新的上限，需实测。

---

## 6. 待确认事项

- Vivado 版本与免费授权是否覆盖 XCK26；KV260 板级文件版本。
- Kria-PYNQ 的当前版本与对应的 Vivado 版本。
- ZynqMP 上 ARM 访问 PL 的 HPM 地址窗口（以 block design 生成的地址为准）。
- `armv7_libm_shim.c` 在 aarch64 构建中是否还需要。
- 固件和驱动中是否还有其他写死的 ARM 侧地址（逐个 grep `0x4001`、`0x4000` 等）。
- UltraScale+ 上 `sa_tdpram.v`、`sa_bankmem.v` 的 BRAM/URAM 推断结果。
- 多个 HP 口并发时 DDR 控制器的实际可用带宽。

# PYNQ-Z1 → Kria KV260 升级方案

状态：草案（2026-10-02，按三轮实现前评审修订：K1 拆为 50 / 100 MHz 两步，K2 改为 DMA 与片上存储接口重构，K4 拆为正确性与性能两步验收）。**2026-10-04：Z1 已冻结**（§7.2 完成：tag `v1.0-pynq-z1`、分支 `pynq-z1`、基线文件 `tests/baselines/pynq-z1/`，含实测的耗时分解 `z1_profile.md`）；之后的所有优化都在 KV260 上做，包括编译器的带生存期内存规划（K2b 的编译器工作项）。**2026-10-08：K1 验收完成**（K1a 50 MHz、K1b 100 MHz，板上 token 序列与 Z1 基线完全相同；100 MHz 下 stories15M 35.44 token/s、SmolLM2 4.43 token/s），`kv260` 分支合入 main，main 从此是 KV260 版本（§7.3）。本文给出从 PYNQ-Z1（Zynq-7020）迁移到 Kria KV260（K26 SOM，Zynq UltraScale+）的完整计划：分阶段目标、每个模块的改动、验收标准和风险。性能数字除"实测"外均为估算，需在板上验证。

相关文档：[`iree_compiler_plan.md`](iree_compiler_plan.md) §8.18（FPGA 规模估计）、[`double_buffer_design.md`](double_buffer_design.md)（加速器架构）、[`memory_model.md`](memory_model.md)（地址与一致性）。

---

## 0. 目标与原则

**目标**

1. 现有全部功能在 KV260 上逐位一致地运行（stories15M、SmolLM2-135M，prefill + decode）。
2. 把 decode 的瓶颈——权重读取带宽——从约 0.39 GB/s 提高到**实测写入 SPAD 的有效带宽 6–10 GB/s**。这需要重构 DMA 接收端与片上存储写入接口（K2），不只是加宽数据通路。
3. 让 **Qwen3-0.6B（int8，全 28 层）** 先在板上正确运行，再按实测耗时分解调优（10–15 token/s 是冲刺目标，6 GB/s 时仅权重读取的上限约 10 token/s）。
4. 为后续的 int4、64 位地址、更大模型、RVV 核留出空间。

**原则**（沿用项目约定）

- **一次只改一类东西**：先换平台（时钟、参数都不变），再提频率，再改带宽架构，再扩存储，最后上大模型。每一步出问题都能定位到这一步的改动。
- **正确性与性能分开验收**：平台移植和大模型上板先以逐位一致验收；性能数字在硬件稳定后按实测设目标。
- **每一步先 sim、再上板、逐位一致后提交**。逐位一致的参照：功能仿真器（`llm/sa_funcsim.py`）与 PYNQ-Z1 上已验证的结果。
- **K1 不动编译器**：硬件参数通过目标配置（`--iree-sa-d`、`--iree-sa-spad-kb`、`--iree-sa-acc-kb`）传入，黄金语料（4002 个 dispatch）守护代码生成不变。**K2b 和 K3 有明确的编译器工作**（对齐、子银行冲突、读端口分配、chunk 选择），分别列在这两个阶段里。
- **PYNQ-Z1 已冻结**（2026-10-04）：tag `v1.0-pynq-z1`、分支 `pynq-z1`，基线在 `tests/baselines/pynq-z1/`。K1 完成前 main 仍是 Z1 版本，新平台在 `kv260` 分支、用新目录和参数，不破坏现有 bitstream 与测试；K1 验收后 main 完全转向 KV260（见 §7；2026-10-08 已合入）。

---

## 1. 平台对比

| 项 | PYNQ-Z1（现状） | KV260（K26 SOM） |
|---|---|---|
| 器件 | Zynq-7020（xc7z020clg400-1） | Zynq UltraScale+（XCK26，与 ZU5EV 同级） |
| 处理器 | 2 × Cortex-A9（armv7，32 位），650 MHz | 4 × Cortex-A53（aarch64），约 1.3 GHz；另有 2 × R5F |
| 逻辑资源 | 约 53K LUT、220 DSP48E1、140 BRAM36；LUT 已用约 83% | 约 117K LUT、1248 DSP48E2、144 BRAM36、**64 URAM（约 2.25 MB）**。BRAM 数量与 Z1 接近，新增的存储能力主要来自 URAM |
| DDR | 512 MB DDR3 | **4 GB DDR4（64 位）**，理论约 19 GB/s |
| PS-PL 数据口 | HP0–HP3，各 **64 位**，AXI3 | HP0–HP3（及 HPC），各 **128 位**，AXI4 |
| 加速器时钟 | 50 MHz（实测时序约束） | K1a 50 MHz → K1b 100 MHz → K2 目标 200–250 MHz |
| 读权重带宽 | **实测约 0.39 GB/s**（1 个 HP 口，约 7.8 B/周期，利用率 99%；LD 入口 64 位 × 50 MHz 的上限是 0.4 GB/s） | 目标：写入 SPAD 有效带宽 6–10 GB/s（多口接收 + 存储分银行，见 K2） |
| 软件 | PYNQ（armv7 Linux） | Kria-PYNQ 或 Ubuntu for Kria（aarch64） |

资源数字来自公开资料，按 AMD 数据手册核对。

---

## 2. 现有设计中与平台相关的部分

逐项盘点，按"不改 / 小改 / 重写"分类。

### 2.1 硬件（`rtl/`、`RISCV-on-PYNQ-Z1/`）

| 模块 | 分类 | 说明 |
|---|---|---|
| `rtl/sysarray/*.v`（阵列、VE、LD/ST、调度器、记分板） | 小改 | 通用 Verilog。`sa_unit` 已有参数 `D`、`DSP_COLS`、`NPORTS`、`SPAD_WORDS`、`ACC_WORDS`、`PERF`。DSP 推断从 DSP48E1 变为 DSP48E2（一般自动），`DSP_COLS` 的 DSP/LUT 列分配需重新评估；BRAM 推断模板（`sa_tdpram.v`、`sa_bankmem.v`）在 UltraScale+ 上需确认推断结果。SPAD/ACC 加大时改用 URAM，但 `sa_tdpram.v` 是 read-first、1 周期读延迟、上电清零的 BRAM 模板，URAM 的端口行为、读延迟和初值都不同，需要先抽象存储接口（K3） |
| `rtl/sysarray/sa_ld.v`、`sa_st.v`（DMA）及 SPAD/ACC 写入接口 | K1 不改 → **K2 重构** | 当前每个口 64 位数据（`m_rdata[NPORTS*64-1:0]`），返回端每周期只选一个口（`r_sel`），本地写入每周期一个 64 位 lane；多口只能隐藏延迟，接收带宽不能相加。K1 把 HP 口直接配置为 64 位，不需要位宽转换器；K2a 入口加宽到 128 位，K2b 多口接收 + 存储分银行 |
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
| `compiler/runtime/sa/sa_transport_board.c` | 小改 | 通过 `/dev/mem` 映射设备窗口（`SA_BOARD_MEM=<phys>:<bytes>`、`SA_BOARD_MBOX=<phys>`），物理地址目前是 `uint32_t`（`sa_devmem_map(uint32_t phys, ...)`）。改为 64 位解析与检查，描述符中仍用 32 位，窗口必须在低 2 GB（见 2.4） |
| `compiler/tests/board_*.py`、`scripts/deploy_*.sh` | 小改 | 路径、架构、窗口大小、mailbox 地址 |

### 2.4 关键约束：32 位 DDR 地址

- 描述符的 DDR 地址字段是 32 位（`DescList`：`ddr & M32`），PicoRV32 也是 32 位核；运行时传物理地址用 `uint32_t`。
- KV260 的 4 GB DDR 在 ZynqMP 地址空间里分两段：**低 2 GB 在 `0x0000_0000` 起，高 2 GB 在 `0x8_0000_0000` 起**。
- **K1–K4 的做法：所有给加速器的缓冲（权重、KV cache、激活、描述符列表、环形队列）都放在低 2 GB。** 这样不需要立即改描述符的地址格式。
- **地址检查的顺序**：物理地址（来自环境变量、驱动或分配器）一律先用 64 位类型（`uint64_t`）解析；检查**整个范围** `[phys, phys + size)` 都在保留区内、且末端 ≤ 2 GB 之后，才转成 32 位写进描述符。不能先截断再检查，否则高 2 GB 的地址会被静默折回低地址。
- **分配与映射**：reserved-memory 只是把一段内存留出来，不等于现有驱动能直接分配和映射。需要明确：
  - 分配方式（设备树 reserved-memory + `/dev/mem`、CMA、或 udmabuf 一类的驱动）；
  - 映射属性（ARM 侧非缓存 / 写合并 / 可缓存加显式刷新）；
  - CPU ↔ 设备的同步边界（提交描述符前、读取结果前分别做什么）。
  这些写进 `docs/memory_model.md` 的 KV260 一节。
- **ABI 审计**：aarch64 上 `long` 与指针是 64 位。主机与固件共享的结构（mailbox、环形队列项、完成记录、描述符）**全部使用定宽字段**（`uint32_t` 等），不含指针、`long`、`size_t`；用 `static_assert` 检查大小与字段偏移，确认对齐与填充与固件一致。不能只把主机程序重新编译一遍就认为 ABI 不变。
- 容量够用：Qwen3-0.6B int8 权重约 0.6 GB，加 INT8 KV（2048 token 约 112 MiB，未计 scale 与填充）和激活，远小于可用空间。
- 64 位地址放到 K5（可选），只有上 1.7B int8 或 3B 以上模型才需要。

### 2.5 编译器（`compiler/plugins/sa/`）

| 模块 | 分类 | 说明 |
|---|---|---|
| sa 后端插件、`sahl`/`sahw` 流水线 | K1 不改；K2 起复查 | 硬件参数来自目标配置。K2 / K3 若改变银行组织、读延迟、DMA 对齐或并行度，目标配置、内存规划、预取与调度需要相应更新，才能用上新硬件 |
| `TargetConfig::invalid()` | 视情况 | 目前要求 D ∈ {8, 16}、片上存储不超过 2^16 字（16 位本地字地址）。SPAD/ACC 超过这个范围或 D = 32 时，需要同时改描述符编码、RTL 和这里的检查（见 K3） |
| 主机退路（VMVX） | 不改 | 与 CPU 架构无关 |
| 功能仿真器、`dispatch_check.py` | 不改 | 已支持任意 D 与存储大小 |

---

## 3. 分阶段计划

| 阶段 | 内容 | 主要风险 | 工作量 |
|---|---|---|---|
| **K0** | 准备：固定系统镜像、启动固件、Vivado / Kria-PYNQ 版本；保存 Z1 可复现基线 | 版本组合 | 小 |
| **K1a** | 平台移植，加速器**完全不变**：D = 8（Z1 的 L2 bitstream）、1 个 64 位 HP 口、**50 MHz**、原存储容量 | block design、地址映射、aarch64 运行时 | 中 |
| **K1b** | 只提频到 100 MHz，重新记录时序与性能 | 时序 | 小 |
| **K1c** | 只把 D 从 8 改为 16（Z1 因 LUT 只能放 D = 8 的 LLM 加速器） | 资源、时序；VE 的 fp 通道折叠（FL）随 D 变化 | 小 |
| **KC** | 计算侧（§3.1 的 H1、H2）：VE 定序模式多组交错（P6）、调度器乱序发射 / 细粒度冲突检查；K1c 之后，可与 K2 并行 | 与功能仿真器逐位一致；记分板的正确性 | 中 |
| **K2a** | 单口 128 位：LD 入口、DMA 与 SPAD 写入加宽；测 DMA → SPAD 的端到端有效带宽 | DMA 与存储写入接口改动 | 中 |
| **K2b** | 多口接收 + 存储分银行，逐步追求 6–10 GB/s；**同时 D = 16 → 32**（decode 时阵列每周期只消耗 D 字节权重，见 §3.1） | **DMA + 片上存储接口重构**、D = 32 的资源与时序、内存规划 | 大 |
| **K3** | URAM 扩展 SPAD/ACC 容量（K2b 之后 D = 32：SPAD 字 256 位、ACC 字 1024 位） | URAM 的端口语义、初值、读延迟 | 中 |
| **K4a** | Qwen3-0.6B 全模型板上正确运行 | 地址与容量、长上下文 | 中 |
| **K4b** | 长上下文分块（C6.5）与性能调优，按实测耗时分解定目标 | 注意力 T 分块、调度 | 中 |
| **K5** | 可选扩展：int4、64 位地址、RVV 核、MoE，各自单独评估 | 各自独立 | 大 |
| **Q（并行）** | Qwen3 量化质量（C6.2），在 CPU / sim 上进行，不等 K4 | 量化方法的数学等价与 scale 语义 | 中 |

### 3.1 硬件改进项（Z1 实测 + RTL 确认，2026-10-04）

依据：`tests/baselines/pynq-z1/z1_profile.md` 的事件计数器（SmolLM2，decode 每步 / prefill 每块，手写路径 / 通用路径），并在 RTL 中逐项确认。

| 编号 | 改进 | 证据（计数器 + RTL） | 代价 | 阶段 |
|---|---|---|---|---|
| **H0** | **D 随带宽加大**（D = 16 → 32） | decode 的 `EX_STEP` 几乎等于读入字节 ÷ 8（SmolLM2 手写 17.36M 对 17.30M，stories 2.00M 对 1.98M）。`sa_ex.v`：每步读一个 SPAD_B 字（`sb_addr = b_base + c`，D 字节 = 一个 k 行的 D 个输出列）；decode 的 A 条带是 x 复制到 D 行（VE 的 DIV 复制），D 行算同一个结果，阵列利用率 1/D。所以每周期消耗的权重 = D 字节：D = 8 / 16 / 32 在 250 MHz 时上限 2 / 4 / 8 GB/s。每个 tile 另有 2(D−1) 周期的填充 / 排空（K = 576 时 D = 8 约 2.4%，D = 32 约 11%） | 见 K3 第 6 项 | D = 16：K1c；D = 32：K2b |
| **H1** | **VE 定序模式多组交错**（P6） | `sa_vefp.v`：EXP / RECIP / RSQRT 与 REDUCE 走定序模式，**一次只有一个组在途**（`cap = seqmode ? 1 : OQ - 4`），微程序逐步等 M2 / A2 的流水延迟；流模式已允许 28 个组在途，不是瓶颈。计数器 24..27 中的第 26 位在 fp VE 上是 `sq_act`（定序器在工作，与整数 VE 的"输出 FIFO 满"相或，名为 `VE_CREDIT`）：decode 1.56M / 2.65M、prefill 12.2M / 19.4M，占 VE 忙碌的 **51–58%**。端口冲突 `VE_RDBLOCK` ≈ 0 | 小到中：定序器同时推进多个组（每个组的微程序状态），复用现有 fp 单元的空闲流水级。**RTL 完成（2026-10-09）**：批量模式（EXP / RECIP / RSQRT 不带 REDUCE）每组约 4.5 倍，+2.65k LUT（`iree_compiler_plan.md` P6），待板上验收 | KC |
| **H2** | **调度器：乱序发射 + 细粒度冲突检查** | `sa_sched.v`：严格按程序顺序派发，记分板只有 6 个 bank（SPAD_A、SPAD_B、ACC 各两半）的粒度，队首受阻则后面全部受阻。prefill 每块（手写）队首受阻：LD 13.2M、ST 11.3M，引擎队列满 9.3M——下一块的权重不能在 VE 尾部期间提前读入；通用路径 decode 的线性层尾部与 EX 串行（约 4.0M）也源于此。前端饥饿 `STARVE` 与全空闲很低，发射本身不是问题 | 中：按地址区间的冲突检查（代替 6 个 bank），允许不冲突的后续命令越过受阻的队首（各引擎独立的发射窗口）；正确性逐条命令与功能仿真器对比 | KC |
| **H3** | **VE 宽度**（物理 fp 通道 FL） | `sa_vefp.v`：`FL = D / 2`（只支持 D / 2 或 D），**随 D 增长**；L2（D = 8，FL = 4）的 `sa_vefp` 约 17.2k LUT，大致与 FL 成正比，D = 32 时 FL = 16 约 69k LUT（估计）。流模式的吞吐随 FL 增长，定序模式受 H1 限制 | D = 32 时若 LUT 不够，需支持 FL = D / 4（每组 4 个半拍，RTL 改动） | 与 H0 一起评估 |
| **H4** | **decode 专用的矩阵乘向量通路**（H0 中 D = 32 的替代或补充） | H0：decode 时阵列只有 1/D 的乘累加有用，D = 32 用 1024 个乘累加换每周期 32 字节；一个每周期 32 字节的 int8 点积单元只要几十个乘累加。或用批量 decode / 投机解码填满其余的行（软件） | 中：多一个单元和编译器路径；prefill 也要提速时 D = 32 更通用 | K2b 前比较 |
| **H5** | **int4 权重，在 LD 通路上解包** | decode 受读入限制：字节减半，上限加倍 | 中到大，并有量化质量问题（GPTQ / AWQ 一类） | K5 |
| **H6** | **片上保留中间结果**（URAM） | dispatch 之间的激活不经 DDR；固定开销只有 1–2%，收益中等 | 中 | K3 |

不需要改的：发射前端（`STARVE` 0.4M、全空闲 0.3M）、VE 存储端口（`VE_RDBLOCK` ≈ 0）、AXI 读反压（`LD_ARSTALL` ≈ 0，Z1 上）。

**顺序**：H1、H2 便宜且直接针对没被读入掩盖的部分，K1c 之后就做（KC 阶段，100 MHz 下就能测出收益）；H0 的 D = 32 与 K2b 的带宽一起做，H4 作为替代方案在那之前比较；H3 随 H0 评估；H5 在 K5（Qwen3 量化质量解决之后）。每一步重做耗时分解（`board_profile.py` + `SA_PROFILE_PERF`，`compiler/tests/z1_profile.py`）。

**计数器命名的注意**：`PC_VE_CREDIT`（第 26 位）在 fp VE 上统计的是定序器在工作（`sq_act`），不是输出 FIFO 满；`PC_VE_GROUPS`（第 27 位）在 fp VE 上统计的是写入的字（含 TRANSPOSE）。分析 LLM 负载时按这个解释。

### K0 准备

1. **工具**：安装支持 K26 的 Vivado / Vitis 版本，下载 KV260 板级文件（board files）。确认所用 Vivado 版本的免费授权覆盖 XCK26。
2. **系统镜像与启动固件**：Kria-PYNQ（推荐，Python 驱动和 notebook 基本可沿用）或 Ubuntu for Kria，写入 microSD；**记录并固定**镜像版本、启动固件（boot firmware）版本和对应的 Vivado 版本，写进 `boards/kv260/README.md`。上电确认 Linux、网络、`/dev/mem` 访问权限。
3. **ARM 基线**：在板上跑 ARM 的 CPU 基线（对应 `notebooks/llm/arm_baseline.sh`），记录 A53 上 llama 推理的 token/s，作为加速器的对比对象。
4. **仓库结构**：在 `kv260` 分支上新建 `KV260/`（与 `RISCV-on-PYNQ-Z1/` 并列），放 block design 脚本、约束、构建脚本和 bitstream 结果目录。K1 验收后再整理目录（§7.4）。
5. **Z1 基线**：开始 K1 之前，先完成 §7.2 的 Z1 冻结和基线文件。

**验收**：板子能启动，能用 PYNQ（或 Python + `/dev/mem`）加载一个空 overlay；版本组合已记录。

**进度**（2026-10-07）：板子到手。镜像 Ubuntu 22.04.4（Ubuntu for Kria，内核 5.15.0-1027-xilinx-zynqmp），启动固件 K26-BootFW-01.02 / U-Boot 2023.01（A / B 相同）；DDR 低段 `0x0000_0000`–`0x7FEF_FFFF`，CMA 1000 MiB 在 `0x3780_0000`（整段在低 2 GB）；开机加载 `k26-starter-kits`，加载 overlay 前要 `xmutil unloadapp`（驱动与 `ddr_test.py` 已自动处理）。版本组合记在 `KV260/README.md`。Vivado 2024.1 支持 XCK26（已确认）。待做：安装 Kria-PYNQ、A53 的 CPU 基线、加载 overlay。回归测试包已补上 K1a 验证第 1、3、4 步（`ddr`、`fwdemo`、`c1`）。

### K1a 平台移植（加速器完全不变）

目标是在 KV260 上**以与 Z1 相同的加速器**完成闭环，把平台问题和时序问题分开：这一步出错，只可能是平台（block design、地址、运行时）的问题。

**硬件**

1. 新 block design（`KV260/scripts/kv260_bd.tcl`）：
   - `zynq_ultra_ps_e`，打开 HPM0_FPD（ARM → BRAM、CSR）、S_AXI_HP0_FPD（加速器 DMA）、`pl_clk0`（**50 MHz**，与 Z1 相同）、`pl_ps_irq0`、EMIO GPIO（RISC-V 复位）；
   - PicoRV32（`picorv32_axi`）、程序 BRAM（8 KB，ARM 侧和 RISC-V 侧双口）、`sa_unit`（**D = 8**，`NPORTS = 1`，SPAD/ACC 与 Z1 的 L2 bitstream 相同；Z1 上跑 LLM 的 L2 是 D = 8，`board_llm.py` 的 launcher 打印 `overlay D = 8`）；
   - **HP0 直接配置为 64 位**（ZynqMP 的 HP 口支持 32 / 64 / 128 位），加速器的 64 位 AXI master 直接连接，不需要位宽转换器，也不需要 Z1 上的 AXI4→AXI3 转换；
   - PicoRV32 的 DDR 口（原 HP0，用于读环形队列）接 HP1 或经 SmartConnect 共用。
   - **时钟域**：沿用 Z1 的做法，PicoRV32 与加速器在**同一个时钟**（Z1 的 `pico_bit.tcl` 中 `riscv_clk` 与 `matmul_0/aclk` 都接 `subprocessorClk`），PS 侧 AXI-Lite 经互连跨时钟域。之后提频时两者一起提：PCPI 是紧耦合接口，拆成两个时钟域需要在 PCPI 与命令路径上都做跨时钟域处理。只有当 PicoRV32 或 PCPI（`sa_pcpi.v`）的路径成为完整实现中的关键路径时，才评估拆分。
   **已写好**（2026-10-04，`kv260` 分支）：`KV260/scripts/kv260_bd.tcl` + `build_bitstream.{tcl,sh}`（`-sa_d`、`-sa_mhz`、`-bd_only`），block design 在 Vivado 2024.1 中验证通过（`-bd_only`）；与 Z1 的差别：只有一个 PL 时钟 `pl_clk0`（不用 clk_wiz），加速器 DMA 直接接 `S_AXI_HP1_FPD`（AXI4，不要协议转换），没有 PL 引脚。**完整构建通过**（2026-10-04）：LUT 34.5%、DSP 115、BRAM36 130 / 144（90%）、URAM 0；50 MHz 下 setup WNS +9.05 ns、hold WHS +0.010 ns，`check_timing` 无未约束路径。最差路径从 DMA 口（HP1）的读数据经 DSP 乘法到描述符取指单元的 `fetch/param_reg`（数据路径 10.5 ns），按现状约 90–95 MHz，**K1b 的 100 MHz 需要在这条路径上加一级寄存器**。构建在一个 Vivado 进程里完成（这台 15 GB 的机器上，分 IP 单独综合的子进程会被 systemd-oomd 杀掉）。见 `KV260/README.md`。
2. 地址映射：记录 ARM 侧的 BRAM、CSR、mailbox 新地址，写进 `docs/memory_model.md` 的 KV260 一节（**已写**，含 `STRICT_DEVMEM` 与 u-dma-buf 窗口）。ARM 侧：BRAM `0xA001_0000`（mailbox `0xA001_1F00`，perf 区 `0xA001_1E00`）、中断控制器 `0xA002_0000`；RISC-V 与加速器侧与 Z1 相同，DDR 为低 2 GB。
3. `synth_ooc.tcl` 加 K26 器件选项，先做脱离上下文的综合，确认资源。

**软件**

4. aarch64 交叉编译：`toolchain-aarch64.cmake`、`build_sa_runtime.sh aarch64`，构建 `sa-llm-run`、`sa_hal_test`、L0 的 `iree-run-module`。**已完成**（2026-10-04，`kv260` 分支）：`compiler/runtime/toolchains/aarch64-linux-gnu.cmake`，三个程序静态链接；不用板子的检查 `compiler/scripts/test_aarch64_runtime.sh`（qemu-aarch64 用户态 + 环形队列仿真器 `sa_board_emu.py`）：C1 HAL 测试 PASS，stories15M decode 与 prefill + decode 的 logits 与参照逐位一致。
5. **地址与 ABI**（细节见 §2.4）：`sa_transport_board.c` 改为用 64 位类型解析物理地址，检查整个窗口在保留区内之后再转成 32 位；明确保留区的分配、映射属性和同步边界；审计主机与固件共享的结构体。运行时部分**已完成**：`SA_BOARD_*` 按 64 位解析，窗口整段检查（非空、末端 ≤ 2 GB）后才形成 32 位设备地址，mailbox 可在任意物理地址；与固件共享的只有环形队列项（64 位字）与完成记录（32 位字），按偏移写，不含结构体、指针或 `long`。保留区的分配方式与映射属性等板子到手再定。
6. 驱动与部署：`pynq_matmul.py` 和 `deploy_*.sh` 增加 `--board kv260`，读入新的基地址。**已完成**（2026-10-04，`kv260` 分支）：驱动从 `.hwh` 读板子、D、ARM 侧 BRAM 地址与加速器时钟（`overlay_info`），launcher、bwtest、`ddr_test.py`（从 overlay 的 `ip_dict` 取地址）不再写死 Z1 的地址；部署脚本用 `SA_BOARD=kv260`（`compiler/scripts/board_env.sh`：aarch64 运行时、`KV260/build/output/<配置>` 的 overlay，`SA_KV260_CONFIG` 选配置，默认 `d8_50mhz`，K1 验收后改为 `d8_100mhz`，`SA_BOARD` 默认也改为 `kv260`；构建输出按配置分目录，如 K1b 的 `d8_100mhz`），整套回归 `deploy_z1_freeze.sh --board kv260` → `build/deploy_kv260`；不用板子的检查：KV260 测试包里的 aarch64 `sa-llm-run` 在 qemu 下、按 KV260 的地址（mailbox `0xA001_1F00`，窗口在 2 GB 以下）运行，stories15M 22/22 行逐位一致。
7. 编译器：不改；生成与 Z1 基线相同的模块（D = 8、默认 SPAD/ACC），描述符应与 `tests/baselines/pynq-z1/golden_manifest.txt` 逐字节相同。

**验证顺序**（每项都与 sim 逐位一致，并与 §7.2 的 Z1 基线文件对比；测试包与板上脚本沿用 `compiler/scripts/deploy_z1_freeze.sh` 和 `compiler/tests/board_regress.py`，加 `--board kv260`）

| 步骤 | 测试 | 预期 |
|---|---|---|
| 1 | `tests/ddr_access`：RISC-V 读写 DDR，包括保留区的首尾地址 | 通过 |
| 2 | `bwtest`：DMA 带宽 | 记录；观测值应接近 Z1（同为 64 位 × 50 MHz，入口上限 0.4 GB/s） |
| 3 | `gemm`、`vector`、`desc_run` 固件 | 与 NumPy 逐位一致 |
| 4 | C1：`sa_hal_test`（手工模板） | 逐位一致 |
| 5 | C3：stories15M（`board_llm.py`） | 76/76 步与 DeviceModel 逐位一致，token 序列与 Z1 基线相同 |
| 6 | SmolLM2-135M prefill + decode | 与 sim 逐位一致，token 序列与 Z1 基线相同 |

**验收**：第 5、6 步通过。**只以正确性验收**；token/s 记录为观测值（ARM 主机、DDR 延迟、驱动开销都变了，不能预设与 Z1 的比例）。

**K1a 验收通过**（2026-10-08，板上，`d8_50mhz` overlay，`compiler/scripts/deploy_z1_freeze.sh --board kv260` + `run_board.sh`）：第 1–6 步全部 PASS（`ddr`、`bwtest`、gemm / vector / desc_run、C1、C3、SmolLM2 与两条 prefill + decode 路径），`compiler/tests/compare_z1_baselines.py`：5 个测试的 token 序列与 Z1 基线完全相同。设备周期与 Z1 相差不到 0.03%（decode 每步：stories15M 2,708,522 / Z1 2,709,281，SmolLM2 qhf 22,215,268 / 22,220,320，通用路径 26,663,452 / 26,671,269），DMA 397 MB/s（99.3%）；墙钟快 1–4%（A53 主机：stories15M 54.7 ms/步 = 18.27 token/s，Z1 56.8；SmolLM2 qhf 440.9 ms/步 = 2.27 token/s，Z1 449.6）。

上板时解决的平台问题：Kria-PYNQ 没有 `/etc/profile.d/xrt_setup.sh`（测试包带 `run_board.sh`，自动找环境脚本）；开机加载的 `k26-starter-kits` 占着 PL（驱动先 `xmutil unloadapp`）；PYNQ 把 BRAM 控制器列在 `mem_dict`（`ddr_test.py` 曾退回 Z1 的地址，写到了 CMA 里的 DDR）；**内核 `CONFIG_STRICT_DEVMEM=y`，`/dev/mem` 对 root 也拒绝映射内存**：运行时的窗口改由 u-dma-buf 模块提供（`/dev/udmabuf0`，`sync_mode` 2 = 写合并；`SA_BOARD_MEM_DEV`），PL 地址照旧经 `/dev/mem`。

### K1b 提频到 100 MHz

1. 只改 `pl_clk0` 为 100 MHz，其他不变；记录 WNS / TNS、关键路径。
2. 重跑 K1a 的验证顺序第 2–6 步。

K1a 的构建显示最差路径（DMA 读数据 → DSP 乘法 → 描述符取指单元的 `fetch/param_reg`，数据路径 10.5 ns）只能到约 90–95 MHz：100 MHz 前先在这条路径上加一级寄存器（`sa_cmdfetch.v`），以功能仿真与板上逐位一致验收。**已改**（2026-10-04，`kv260` 分支）：LDPARAM 在最后一拍只锁存取出的字，下一周期再做乘加（LDPARAM 多一个周期，语义不变）；`tb_sa_unit` 在 D = 8 / 16 下全部通过（含 LDPARAM 的检查）。

K1a 布线后的前 40 条路径（`post_route.dcp`，在 20 ns 约束下工具没有对它们用力）中，修掉这条后接下来是：VE TRANSPOSE 的地址（`tr_word0`，DSP）→ ACC BRAM 地址 / 写使能（数据路径 9.5–10.1 ns，11 级）；LD 的 `q_rp` → ACC BRAM 地址（约 10.1 ns，13 级）；RISC-V 复位 → EX `bsr_reg` 的同步复位（约 10.2 ns，4 级，高扇出、以布线为主）。10 ns 约束下工具会优化（复制复位网络、就近布局），多数应能收敛，以 100 MHz 的实际构建为准；**K2a 的 200–250 MHz 需要在这几处都加流水级**。

**100 MHz 构建通过**（2026-10-04，`-sa_mhz 100`）：setup WNS +1.171 ns、hold WHS +0.010 ns，`check_timing` 无未约束路径，资源与 K1a 相同（LUT 34.4%、BRAM36 130）；最差路径变为 LD 的 `q_lane` → DSP → ACC BRAM 写使能（7.8 ns，13 级），按现状约 113 MHz。K1b 剩下板上验证（第 2–6 步）。

**K1b 验收通过**（2026-10-08，板上，`d8_100mhz` overlay，`SA_KV260_CONFIG=d8_100mhz compiler/scripts/deploy_z1_freeze.sh --board kv260` + `run_board.sh`）：第 2–6 步全部 PASS，5 个测试的 token 序列与 Z1 基线完全相同。墙钟几乎减半：stories15M decode 28.2 ms/步 = 35.44 token/s（K1a 18.27，Z1 17.62），SmolLM2 qhf decode 225.5 ms/步 = 4.43 token/s（K1a 2.27），通用路径 268.3 ms/步 = 3.73 token/s（K1a 1.88）。DMA 连续读写 794 MB/s（8 B/周期的 99.2%），读写并发 1574 MB/s。DDR 延迟按纳秒固定，100 MHz 下折合的周期数加倍：单拍突发从 47.7% 降到 38.0%，LLM 的设备周期比 50 MHz 多 0.2–0.4%（stories15M 2,719,808 / 2,708,522，SmolLM2 qhf 22,272,216 / 22,215,268，通用路径 26,729,622 / 26,663,452）。主机开销小，提频的收益几乎全部落到墙钟上。**K1（K1a + K1b）验收完成**，按 7.3 节可以合入 main。

**验收**：时序收敛，第 5、6 步逐位一致。性能为观测项：`bwtest` 的入口上限变为 0.8 GB/s（64 位 × 100 MHz），decode 的 token/s 预计有提升，但主机开销与 DDR 延迟使它不一定是 Z1 的两倍。

### K1c D = 8 → 16

1. 只改 `sa_unit` 的 `D`（`build_bitstream.sh -sa_d 16`），其他不变；记录资源与时序（Z1 上 D = 8 的 L2 已用 83% LUT，`sa_vefp` 单独 17.2k LUT）。
2. 编译器：`--iree-sa-d=16`；黄金语料已有 D = 16 的用例（`cfg_d16`），功能仿真器与 `dispatch_check.py` 支持任意 D。D = 16 的模块与 Z1 基线的描述符不同，以 sim 逐位一致验收，并建立 D = 16 配置自己的基线（§5）。
3. 重跑 K1a 的验证顺序第 2–6 步，参照换成 D = 16 的 sim（fp32 归约按 D 个 lane 分组，结果可能与 D = 8 差最低位，所以不要求 token 序列与 Z1 基线相同；记录是否相同）。

**验收**：第 5、6 步逐位一致；prefill 的 token/s 记录为观测值（阵列 4 倍乘累加，decode 仍受带宽限制）。之后各阶段以 D = 16 为准（SPAD 一字 128 位，正好一个 128 位 beat，见 K2a）。

**构建通过**（2026-10-08，`-sa_d 16 -sa_mhz 100`，`4bd8389`）：setup WNS +0.630 ns、hold WHS +0.011 ns；LUT 74,951（64.0%）、FF 59,944、DSP 347、BRAM36 130 / 144。所有 PE 列都放在 DSP 上（`kv260_bd.tcl` 设 `DSP_COLS = D`；Z1 的 D = 16 构建因为只有 220 个 DSP，只有 8 列用 DSP）。最差路径与 K1b 同类：LD 的 `q_rp` → DSP → ACC BRAM 写使能（8.4 ns，12 级）。

**软件**：D 跟着 overlay 走。`board_env.sh` 从配置名取 `SA_D`（`d16_100mhz` → 16）；`compile_sa.sh` 传 `--iree-sa-d`；测试里 sim 的 D、参照模型（`DeviceModel(d = D)`、qhf 的 `valid` 长度）、`--pad` 都用 `SA_D`；C1 的测试可执行文件按 D 生成（`test_d16`）。prefill 的块大小 M = D（`c6p_*`；通用路径 `hfgen` 仍是 M = 8，它的导出里固定了 M）。D ≠ 8 的主机构建放在 `build/<test>/<model>_d16`，不覆盖黄金语料读的 D = 8 目录；导出与 D 无关（注意力长度是动态的），`seed_export` 复制 D = 8 的导出。主机端：C1 三项、C3（stories15M 12/12 步与 `DeviceModel(d = 16)` 逐位一致，58 个 dispatch 逐个检查 OK）通过。

**D = 16 端到端发现的编译器错误**（`a1fefb6`）：C5.5（SmolLM2）在 D = 16 的 sim 上第 0 步正确、第 1 步起完全偏离；逐 dispatch 检查（`dispatch_check`）全部 OK，`--dirty`（片上存储从垃圾值开始）下 30 个注意力 softmax（`reduction_3x3xD`，GQA 的 3 × 3 行）失败。原因：行模式（动态长度、每次一行）里，长度等于整行（`maxDynamic` = 256）的 VE 取动态长度 LEN；按行广播（`perElementBcast`）先复制出 w × D × D 的块再转置，D = 16、不超过 16 行时这个块正好 256 个元素，于是也被改成 LEN = T，只写了块的第一个字，广播出的行读到上一步留下的通道（第 0 步片上存储还是 0、且只用通道 0，所以碰巧正确）。D = 8 时块是 128，不会撞上。修正：固定大小的辅助 VE（按行广播的块、标量广播字）在行模式里保持静态长度。黄金语料不变，新增 `smollm2_d16` 一组（共 4232 个 dispatch）。教训：新的目标配置要做端到端 sim，并用 `dispatch_check --dirty`。

**板上结果**（2026-10-08，`d16_100mhz` overlay，`SA_KV260_CONFIG=d16_100mhz compiler/scripts/deploy_z1_freeze.sh` + `run_board.sh`，bundle 来自 `fcc75ee`）：第 5、6 步全部 PASS（`c1`、`c3`、`c55`、`c6p_stories`、`c6p_smollm2`、`hfgen` 与 D = 16 的 sim 逐位一致），`ddr`、`bwtest`（792–794 MB/s，与 K1b 相同）、`desc` PASS。与 Z1 基线相比，`c3` 与 `hfgen` 的 token 相同，`c55`、`c6p_stories`、`c6p_smollm2` 分别从第 18、16、22 个 token 起不同（fp32 归约分组不同，符合预期）。`gemm`、`vector` 两个 demo 失败在脚本：形状写死为 D = 8 的倍数（8×8×8、24×16×40），驱动在 D = 16 时报 ValueError；固件本身在 D = 16 的系统仿真中通过（19 个 GEMM、7 个向量运算）。`6658deb` 让 demo 跳过不是 D 倍数的形状和 8×8×8 的 Phase 4 旧路径，并加了 D = 16 也合法的非方形用例；板上重跑 `fwdemo` 全部 PASS（`gemm`、`vector`、`desc`）。GEMM 256×256×256 为 202.6 MAC/周期（峰值 256 的 79%；K1b 为 58.9，3.4 倍），128³ 为 156.3（K1b 53.2）。**K1c 验收通过**。

| 运行 | K1b（D = 8） | **K1c（D = 16）** | |
|---|---|---|---|
| stories15M decode（c3） | 35.44 token/s | **39.36** | +11% |
| stories15M prefill + decode 的 decode | 34.84 | **37.84** | +9% |
| SmolLM2 qhf decode（c55） | 4.43 | **4.77** | +8% |
| SmolLM2 qhf prefill + decode 的 decode | 4.30 | **4.50** | +5% |
| SmolLM2 qhf prefill（M = D） | 14.04 | **22.22** | +58%（每块 16 个 token） |
| SmolLM2 通用路径 decode（hfgen） | 3.73 | **4.39** | +18%（设备周期 26.73M → 22.68M） |
| SmolLM2 通用路径 prefill | 6.84（M = 8） | **10.04**（M = 16；M = 8 时 4.17） | +47% |

decode 的提升小于 2 倍：每周期消耗 D 字节权重，但读带宽仍是 8 B/周期（§3.1 H0），D = 16 要和 K2a 的 128 位读通道一起才能发挥。**通用路径的 prefill 变慢**：它的导出固定 M = 8，在 D = 16 的阵列上每块只用一半的行，而 prefill 的 matmul（`8x96x16x576` 等）设备周期反而约为 D = 8 时的 2 倍（30.3M 对 16.6M）。原因（板上计数器）：down 投影的 `LD_BEATS` 与 D = 8 相近（权重只读一次），`EX_USEFUL` 却是 26.5M（按 D = 16 应为 3.3M，8 倍），`VE_ACTIVE` 27.5M。prefill 的 linear 微内核要求行数是 D 的倍数（`SahlKernels.cpp` `matchLinear` 的 `p.rows % d`），M = 8 < 16 时回退到通用 contraction，它对 x 的每一行单独做 decode 式的 GEMV（x 复制到 D 行，阵列利用率 1/D），8 行就是 8 遍。修正（`564a0de`）：`deploy_hfgen.sh` 按 D 导出 M = D（`build/hfgen/smollm2_p<D>`，导出固定 M，不再复制 D = 8 的）；D = 16 的 sim PASS（prefill + decode 与只 decode 5/5 行逐位一致），`dispatch_check --dirty` 1306 个 OK（另 2 个是 host 上的）。板上重跑 `hfgen` PASS（与 sim 逐位一致）：prefill 40 个 token 分 3 块，10.04 token/s（M = 8 时 4.17，D = 8 时 6.84），decode 4.38 token/s；down 投影的 `EX_USEFUL` 回到 3.3M。剩下的大头是 gate / up 投影（`matmul_like_16x96x16x576`，30 次 35.0M 周期，占 prefill 的 30%）：`VE_ACTIVE` 26.4M、`VE_CREDIT` 20.6M，受融合在里面的 SwiGLU 尾部（fp VE / SFU）限制，与 D 无关（D = 8 时 33.1M）。qhf 路径同样位置的 matmul 不带这个尾部（`VE_ACTIVE` 1.2M），所以 prefill 快一倍多（22.2 token/s）。这属于 P6（SFU 多槽交错）/ KC 的范围。让微内核直接支持 M < D（只装入 / 存回 M 行）留作以后的编译器工作：只有固定 M 的导出会遇到。

### K2 带宽：DMA 与片上存储接口

decode 受权重读取带宽限制，这一阶段是性能提升的主要来源。**它不只是加宽 DMA，而是 DMA 与片上存储写入接口的一起重构。**

**现状（`rtl/sysarray/sa_ld.v`、`sa_unit.v`）**

- `sa_ld` 可以向 `NPORTS` 个口轮流发起读（`ar_rr`），但**返回端每周期只选一个口**（`r_sel`）：`lw_data = m_rdata[64*r_sel +: 64]` 只有 64 位，每周期一个 `lw_en`，`m_rready` 只对选中的口拉高。多口只能隐藏延迟，不能让接收带宽相加。
- 写入端：每个 `lw_en` 只写 SPAD 字中的一个 64 位 lane（`spad_we = 8'hFF << (8*lw_lane)`）。D = 16 时一个 SPAD 字是 128 位，需要两个周期才能写满一个字。
- `sa_bankmem` 的两个银行按字地址最高位划分，用于双缓冲，不是给多口并行写入的；side A 的两个请求者之一已被 ST 的本地读取（`lr_word`）占用。

**入口带宽上限**（零停顿，入口宽度 × 时钟；实测只会更低）

| 入口 | 50 MHz | 100 MHz | 200 MHz | 250 MHz |
|---|---|---|---|---|
| 64 位（现状） | 0.4 GB/s | 0.8 GB/s | 1.6 GB/s | 2.0 GB/s |
| 128 位（K2a） | — | 1.6 GB/s | 3.2 GB/s | 4.0 GB/s |
| 256 位（K2b，2 字/周期） | — | — | 6.4 GB/s | 8.0 GB/s |
| 384 位（K2b，3 字/周期） | — | — | 9.6 GB/s | 12 GB/s |

Z1 实测 0.39 GB/s，已经贴着 64 位 × 50 MHz 的入口上限。实测 6–10 GB/s 意味着 250 MHz 时每周期向片上存储**写入 24–40 字节，即 2–3 个 SPAD 字**（D = 16）。

**目标带宽与配置对应**（实测目标 ÷ 入口上限 = 所需利用率）

| 实测目标 | 配置 | 入口上限 | 所需利用率 | 判断 |
|---|---|---|---|---|
| 6 GB/s | 2 字/周期 × 200 MHz | 6.4 GB/s | 约 94% | 余量太小，不作为计划配置 |
| 6 GB/s | 2 字/周期 × 250 MHz | 8.0 GB/s | 75% | 可行 |
| 6 GB/s | 3 字/周期 × 200 MHz | 9.6 GB/s | 约 63% | 可行 |
| 10 GB/s | 3 字/周期 × 200 MHz | 9.6 GB/s | > 100% | **达不到** |
| 10 GB/s | 3 字/周期 × 250 MHz | 12 GB/s | 约 83% | 需要高利用率 |
| 10 GB/s | 4 字/周期 × 200 MHz | 12.8 GB/s | 约 78% | 可行，资源更多 |

K2b 的目标必须和配置一起写：例如"2 字/周期 × 250 MHz，实测 ≥ 6 GB/s"，不能只写"6–10 GB/s"。DDR 控制器本身的可用带宽也可能低于入口上限，以 `bwtest` 实测为准。

**带宽之后：读权重以外的部分**

带宽提高后，读权重以外的周期（VE 尾部、SFU、EX、每个 DMA 与 dispatch 的固定开销）不会跟着减少；DDR 延迟按纳秒计不变，提频后折算成周期反而更多。Z1 冻结时在板上用加速器的事件计数器做了耗时分解（`tests/baselines/pynq-z1/z1_profile.md`，`compiler/tests/z1_profile.py`；decode 每步，一个 dispatch 一个列表）：

| 路径 | Z1 每步周期 | 读入字节 | 读入下限（7.94 B/周期） | 没被读入掩盖的部分 | 全部引擎空闲（固定开销） | K2b（24 B/周期）读入下限 | K2b 后其余部分的占比 |
|---|---|---|---|---|---|---|---|
| stories15M 手写路径 | 2.71M | 15.8 MB | 1.99M | 0.72M（26%） | 0.06M（2%） | 0.66M | 约 52% |
| SmolLM2 手写 qhf 路径（2.22 token/s） | 22.22M | 138.4 MB | 17.44M | 4.78M（22%） | 0.28M（1%） | 5.77M | 约 45% |
| SmolLM2 编译器通用路径（1.87 token/s） | 26.67M | 140.9 MB | 17.75M | 8.92M（33%） | 0.30M（1%） | 5.87M | 约 60% |

（其余部分按周期数不变估计，实际可能更多。）实测说明：
- **固定开销很小**（全部引擎空闲 1–2%）：描述符列表与 rt_fw 的设计已经把发射开销压住了，K2 之后它也不是主要问题。
- **线性层贴着读入下限**：SmolLM2 的 `matvec` 类 dispatch 读入 135.8 MB，LD 忙 17.2M 周期，EX 17.2M 与 LD 几乎完全重叠（重叠 16.4M 周期）。
- **没被掩盖的部分**：手写路径 4.78M 里，线性层的尾部约 1.6M（`matvec` 总周期减去读入下限）、RMSNorm / softmax 的归约约 1.7M、逐元素 0.9M、注意力 0.46M；通用路径 8.92M 里，线性层的尾部约 4.0M（通用 contraction 的尾部与 EX 串行）、注意力 2.4M、归约 2.6M。这些都是 VE / SFU 的工作，K2 不会让它们变快。
- **prefill** 每块（8 个 token）VE / SFU 占 52–54%，读入下限只占 28–42%：prefill 受 VE / SFU 限制，不是带宽（K1c 的 D = 16 只加快 EX）。

所以 K2b 以后，**性能主要取决于 VE / SFU 侧与调度的效率**：尾部与 EX 的重叠、归约与 SFU 的吞吐（P6 SFU 多槽交错）、注意力，而不只是带宽。另外，**decode 时阵列每周期只消耗 D 字节权重**（§3.1 H0），D = 8 时无论 DMA 多宽都卡在每周期 8 字节；带宽升级必须与 D 一起做（K1c 的 D = 16 对应 K2a，D = 32 对应 K2b）。K 各阶段的性能预期以 `z1_profile.md` 为依据。Qwen3-0.6B 的粗略外推见 K4b。

#### K2a 单口 128 位

1. **频率**：`pl_clk0` 提到 200 MHz（再试 250 MHz）。关键路径大概率在 VE 的 fp32 单元（`sa_fp32_*.v`、`sa_vefp.v`）和 SFU 查表，按需加流水级；用 `synth_ooc.tcl` 定位关键路径；**验收以完整设计布局布线后的时序为准**（见 §5）。
2. **LD 入口加宽到 128 位**：HP0 配置为 128 位；`sa_ld` 的 `lw_data` 改为 128 位（参数化 `DMA_W`），一次写满一个 SPAD 字（D = 16）；ACC 的写入（64 字节一字）按 128 位 lane 写。`sa_st` 同步加宽。
   **保留 8 字节粒度的旧语义**：现有 LD / ST 的最小粒度是 8 字节（`row_bytes` 为 8 的倍数，DDR 地址 8 字节对齐），`sa_st` 没有 WSTRB 逻辑（每个 64 位 beat 都是满的）。128 位后会出现**只有半个 beat 有效**的情况，必须由硬件处理，而不是收紧对齐约束让原有 dispatch 失效：
   - LD：按 beat 内有效字节生成本地写 mask（行首、行尾的半个 beat）；
   - ST：生成正确的 `WSTRB`，半个 beat 只写有效的 8 字节；
   - 兼容性清单（每项都要有 sim 测试和逐位一致的检查）：

     | 项 | 需要覆盖的情况 |
     |---|---|
     | DDR 起始地址 | 16 字节对齐；只有 8 字节对齐（第一个 beat 半有效） |
     | 行长度 `row_bytes` | 16 的倍数；8 的奇数倍（最后一个 beat 半有效） |
     | `pitch` | 等于行长（会被合并成一行，`flat`）；大于行长；不是 16 的倍数 |
     | 4 KB 边界 | burst 在 4 KB 处切分，切分点落在半个 beat 上 |
     | ACC lane | 64 字节一字，128 位 lane 的编号与 mask |
     | LINEAR / INTERLEAVE | 两种模式都覆盖；INTERLEAVE 的 chunk 跨 beat |
     | D = 8 / 16 | **D = 8 时一个 SPAD 字只有 64 位，一个 128 位 beat 跨两个字**；INTERLEAVE 下这两半写到相距 `rows` 个字的两个位置，不是相邻的字 |
3. **加性能计数器**：除 DDR 侧的 AXI 计数外，新增"**写入 SPAD/ACC 的有效字节数**"计数器（`sa_perf.v`），`bwtest` 同时报告两者。
   **在途 burst 数**：`sa_ld` 每口的 burst 队列是 8 项（`QD = 8`，索引 `q_rp[2:0]`；文件头注释写的 "up to 4 outstanding per port" 与代码不一致，顺手改正）。250 MHz、DDR 延迟约 40 周期时，每口 16 B/周期需要约 640 B 在途：长 burst（16 beat × 16 B = 256 B）3 个就够，**短 burst**（窄行、4 KB 切分、INTERLEAVE 的短行，1–2 beat）时 8 项只有 128–256 B 在途，带宽受延迟限制。`bwtest` 按 burst 长度（1、2、4、8、16 beat）分别测量，并统计"burst 队列已满"的周期（新增事件；现有 `perf_ev[2]` 只统计 AR 被端口阻塞的周期）；需要加深 QD 时，索引位宽一起改。
4. **编译器**：目标是**不改变**现有的对齐与最小粒度（8 字节），由硬件处理半个 beat；如果某项确实无法在硬件中兼容，才在目标配置和内存规划中加约束，并在新配置的黄金语料基线中体现（见 §5 的验收约束）。

**验收**：`bwtest` 的 DMA → SPAD 有效带宽接近单口 128 位的入口上限（200 MHz 时 3.2 GB/s 以内的实测值，记录实际比例）；stories15M、SmolLM2 逐位一致。

**分两步**（同 K1a / K1b，把位宽与时序分开）：**K2a-1** 在 100 MHz 下把 DMA 加宽到 128 位（`d16_100mhz_w128`，入口上限 1.6 GB/s），**K2a-2** 再提频到 200 MHz（第 1 项）。

**K2a-1 RTL 完成**（2026-10-08，`119a524`）：`sa_ld` / `sa_st` / `sa_cmdfetch` / `sa_unit` 加参数 `DMA_W`（64 | 128，默认 64，旧配置不变），KV260 构建加 `-dma_w 128`（HP1 128 位，配置名后缀 `_w128`）。
- 8 字节粒度不变：burst 从地址向下取整到 16 字节开始，burst 队列记下"首拍的低 lane 不是本命令的""末拍的高 lane 是不是本命令的"；LD 按此丢弃，ST 按此生成 `WSTRB`。
- 一拍的两个 lane 落在同一个本地字、且起始 lane 为偶数时一个周期写完（`lw_two`：D = 16 的 SPAD、ACC，DDR 地址 16 字节对齐时）；否则一个周期写一个 lane（R 保持一拍），所以 D = 8 的 SPAD、只 8 字节对齐的地址、INTERLEAVE 跨字的块都正确，只是这部分不加速。ST 的本地读同理。
- 描述符取指每拍两个 64 位字（4 拍一个描述符），LDPARAM 按 beat 内偏移取字。
- `LD_BEATS` / `ST_BEATS` 仍按 8 字节计（`sa_perf` 新增 `ev2`：两 lane 的写 / 拍计两次），`z1_profile.py` 等软件不用改；计数器 19 现在就是"写入 SPAD/ACC 的字节数 ÷ 8"（第 3 项的有效字节计数）。`sa_ld` 文件头的在途 burst 数改正为 8。
- 验证：`tb_sa_dma` 的 AXI 模型加 128 位与 `WSTRB`，另加 40 个随机形状（8 字节对齐的地址、8 的奇数倍行长、跨 4 KB、INTERLEAVE），故意改错合并条件或 `WSTRB` 都能抓到；`make test` / `test16` / `test128`、`tb_sa_unit`（365 项计数器检查、描述符列表）、固件系统仿真（gemm、vector、desc_run、rt、bwtest；D = 16 的 64 / 128 位）全部 PASS。仿真中 bwtest 连续读写 3.62 → 7.26 B/周期（DDR 模型带随机停顿，不是板上数字），`tb_sa_unit` 的 64×64×128 GEMM 10107 → 6413 周期。`-bd_only` 验证通过。编译器与运行时不变；驱动从 `.hwh` 读 `DMA_W`，`m2_bw_test.py` 按位宽报告占比。
- 构建（2026-10-08，`094d535`）：setup WNS +0.311 ns（K1c +0.630），WHS +0.010；LUT 75503（64.5%，比 K1c 多 552）、FF 60270（+326）、BRAM 130（90.3%）、DSP 347，后两项不变。最差路径仍是 K1c 那条 EX `drain_active` → PE `sh_out`（1 级，布线 96%），不在 DMA 上。
- **板上通过**（2026-10-08，`build/deploy_kv260_d16_100mhz_w128`）：REGRESSION PASS，全部测试与 sim 逐位一致，生成的 token 与 K1c 相同（与 Z1 基线的差异同 K1c，来自 D = 16）。`bwtest` 连续 LD / ST 15.6–15.8 B/周期（1.57 GB/s，128 位上限的 98%，K1c 7.9），LD + ST 并发 31.0 B/周期；8 字节小行（1 拍 burst）3.03 B/周期不变（受 burst 开销限制，符合预期）。

| 测试（token/s） | K1c | K2a-1 | 倍数 |
|---|---|---|---|
| stories15M decode (c3) | 39.36 | **61.72** | 1.57 |
| stories15M decode (c6p) | 37.84 | **58.40** | 1.54 |
| stories15M prefill (c6p) | 191.4 | **227.6** | 1.19 |
| SmolLM2 qhf decode (c55) | 4.77 | **7.98** | 1.67 |
| SmolLM2 qhf decode (c6p) | 4.50 | **7.26** | 1.61 |
| SmolLM2 qhf prefill (M = D) | 22.22 | **27.17** | 1.22 |
| SmolLM2 通用 decode (hfgen) | 4.38 | **6.89** | 1.57 |
| SmolLM2 通用 prefill | 10.04 | **11.21** | 1.12 |
| GEMM 256³（MAC/周期） | 202.6 | **216.0** | 1.07 |

decode 受带宽限制，提升 1.5–1.7 倍（不到 2 倍：VE 尾部、注意力、调度开销与带宽无关）；prefill 与 GEMM 受计算 / VE 限制，提升小。下一步 K2a-2（200 MHz）。

#### K2b 多口接收 + 存储分银行

1. **先确定架构**。功能仿真器（`sa_funcsim`、`sa_sim_server.py`）**没有时序模型**（返回的周期数为 0），不能用来比较方案，改用两种工具：
   - **解析性能模型**（先做，用于筛选）：输入每周期写入字数、子银行数、端口分配、DDR 延迟、burst 长度分布、每个 HP 口的带宽，输出 LINEAR / INTERLEAVE 各种传输形状的有效带宽和子银行冲突率；用 Z1 与 K2a 的 `bwtest` 实测校准。
   - **RTL 测试平台**（再确认）：扩展 `rtl/sysarray/sim/tb_sa_dma.v`（已有 NP 口 AXI 存储模型、每口多个在途 burst、随机停顿、SLVERR 注入、协议检查，`make sim TB=tb_sa_dma GEN="NP=3 D=16"`）：存储模型加可配置的延迟与每口带宽，统计写入 SPAD 的有效带宽，在 xsim 中运行。

   两种候选：

   | 方案 | 做法 | 编译器影响 | 风险 |
   |---|---|---|---|
   | **A（首选）汇聚 + 按字地址低位分银行** | 每个 HP 口一个接收 FIFO（口内按序、口间不同步，见下）；汇聚成 256 / 384 / 512 位内部总线；SPAD 按字地址低位分为 2 / 4 个子银行（偶 / 奇字），LINEAR 加载每周期写 2–4 个相邻字 | 地址空间仍连续；DMA 块要求按 2 / 4 字对齐，进入目标配置与内存规划的对齐约束 | 汇聚逻辑与 FIFO 资源；EX / VE 读端也要适应子银行 |
   | B 每口独立银行 | 每个 HP 口固定对应一组存储银行 | 内存规划需按银行放置数据，地址到银行的映射由编译器与硬件共同维护 | 编译器改动大；放置不均时带宽打折 |

   同时明确：每个子银行的读写端口分配（LD 写、EX / VE 读、ST 读）、读延迟、读写碰撞语义、字节写使能，以及记分板（scoreboard）按子银行的冲突检查。
2. **第一版只保证 LINEAR 的多字/周期吞吐**：
   - LINEAR（权重加载，decode 的主要流量）：连续的字依次落在不同子银行，可以每周期写多个字。
   - INTERLEAVE（阵列的 A 条带）的本地地址是 `word = base + chunk × rows + row`，同一行相邻 chunk 相距 `rows` 个字。`rows` 是子银行数的整数倍时（例如 4 个子银行、`rows = 16`），同一行的所有 chunk 都落在同一个子银行，不能并行写入。相邻的行落在不同子银行，如果两行的 burst 恰好在不同的口同时返回，可以并行，但这取决于返回时机，不作为吞吐保证。
   - 所以第一版对 INTERLEAVE **只保证正确**，吞吐单独测量（`bwtest` 增加 INTERLEAVE 模式）；如果它成为瓶颈，再评估地址 swizzle（例如子银行号 = 字地址低位 XOR chunk 号）或中转转置缓冲。
3. **AXI ID 与返回顺序**：
   - 每个 HP 口固定一个 AXI ID（与现有设计一致：现在 `sa_ld` 不输出 ARID，每口一个 burst 队列，按 `q_rp` 顺序消费）。同一 ID 的读事务按请求顺序返回，所以**口内按序**；不同口之间返回不同步。
   - 只有将来一个口使用多个 ID 时，才需要处理口内乱序。
4. **接收 FIFO 的内容与写回规则**：
   - 每个条目不只有数据，还要有对应的**本地字地址、byte mask、目标存储和事务状态**（`rresp` 错误、是否为 burst 的最后一个 beat）。
   - 同一条命令内各 burst 的本地地址互不重叠，所以**按子银行可用性写回**，不恢复跨口的全局顺序；否则一个慢口会阻塞其他口。
   - **`done` 的条件**：所有返回数据都已写入本地存储，且所有接收 FIFO 已排空；不能只看最后一个 AXI beat 已收到。（现有设计中数据在接收当周期写入，两者等价；加了 FIFO 之后不再等价。）`err` 在所有 beat 写回后再报告。
5. **RTL**：`sa_ld` 返回端改为多口同时接收（每口独立 `m_rready`）；汇聚与写回；`sa_bankmem` 增加子银行维度。先做独立的存储与 DMA 测试平台，再接 EX / VE。
6. **多口**：`NPORTS` = 2，再试 3–4，接 HP0–HP3；逐口测量，并对比 HP 与 HPC 口。
7. **编译器工作项**（方案 A）：
   - **带生存期的内存规划**（`iree_compiler_plan.md` §8.17 第 3 项，Z1 冻结时决定放到这里）分两步。**3a 框架**：在 `sahl-expand-kernels` 展开后的 IR 上做生存期分析，由规划器给出 `sa.word` / `sa.bank` / `sahl.scope`；第一版完全复现现有分配（黄金语料不变），bank 数、对齐、容量、读端口都从 `TargetConfig` 读，约束做成可替换的一层；去掉 `expandContraction` 重放分配器的耦合。不改输出，可以在 K1 期间就在 `kv260` 分支上做。**3b 优化**：按生存期复用、避开子银行冲突、放不下时分片，下面三项约束就是它的规则；改变输出，K2b / K3 的银行组织定下来之后做，以逐 dispatch 检查和板上周期验收；
   - **对齐**：多字写入要求的 DMA 块对齐（2 / 4 字）进入 `TargetConfig` 与内存规划；不满足对齐的块退回单字写入（正确但慢），而不是拒绝编译；
   - **子银行冲突**：记分板按子银行检查冲突；内存规划尽量把同时访问的 LD 写入与 EX / VE 读放在不同子银行，调度在冲突时串行化；
   - **读端口分配**：EX / VE / ST 的读端口与子银行的对应关系作为目标配置的一部分，调度器据此避免同周期争用；
   - **验证**：`dispatch_check` 与功能仿真器支持新的目标参数，新配置建立自己的黄金语料基线（§5）。
8. **运行时**：带宽变大后每个 dispatch 的固定开销占比上升，用性能计数器（`board_profile.py`）判断是否需要更大的 chunk 或更多的跨 dispatch 预取。

**验收**：

- `bwtest` 同时报告 DDR 侧带宽与**写入 SPAD 的有效带宽**，分 LINEAR 与 INTERLEAVE 两种模式、分 burst 长度，并报告队列已满的周期；LINEAR 的有效带宽达到按上表选定的配置目标（例如 2 字/周期 × 250 MHz，实测 ≥ 6 GB/s），按"DDR 能给多少"和"片上存储能收多少"分别记录，找出实际瓶颈。INTERLEAVE 记录实测值。
- stories15M、SmolLM2 逐位一致；SmolLM2 decode 的 token/s 随有效带宽提升，记录实测比例（不预设线性）。

### K3 URAM 容量扩展

1. **存储接口先抽象**：`sa_tdpram.v` 是 UG901 的 BRAM 模板（read-first、1 周期读延迟、byte-write），不能只改 `ram_style` 就认为行为不变。存储接口改为：
   - 可配置读延迟（URAM 通常需要额外的流水寄存器才能跑到高频）；
   - 明确的读写碰撞语义（同地址同周期读写返回旧值还是新值，或禁止）；
   - 字节写使能；
   - 支持 K2 的写入宽度与银行布局。
2. **初值**：URAM 不支持任意的 INIT 内容，**配置后为零**。区分三种状态：
   - **上电 / 配置后**：BRAM（INIT 全零）和 URAM 都为零；
   - **重新加载 overlay**：等于重新配置，同样为零；
   - **运行中的逻辑复位**：BRAM 和 URAM 都**不会**被清零。
   `sa_tdpram.v` 的注释写明 legacy sequencer 依赖的是"上电内容为零"（D > 8 时保留的 tile 槽位），也就是第一种状态，现有 BRAM 设计在逻辑复位后同样不清零。所以换成 URAM 后，依赖范围不变即可。如果以后软件要求**每次逻辑复位后**保留槽位仍为零，再显式初始化这些槽位；不必每次清空整个 SPAD/ACC。
3. **独立存储测试**：URAM 版本的存储先单独仿真和上板测试（读延迟、碰撞、字节写、子银行），再接 EX / VE。资源与时序重新验证。
4. **容量**：例如 SPAD 256 KB × 2、ACC 512 KB。更大的 chunk 减少每层的固定开销，也是长上下文注意力的前提。
   - 现有本地地址是 **16 位字地址**（`TargetConfig::invalid()` 要求每块存储不超过 2^16 字）。D = 16 时 SPAD 一字 16 字节，2^16 字 = 1 MB；ACC 一字 64 字节，上限更大，所以上述配置不需要改地址位宽。超过时需同步改描述符编码、RTL 和编译器检查。
   - 编译器：`--iree-sa-spad-kb`、`--iree-sa-acc-kb` 传入新大小；C6 已在 sim 上验证过 SPAD 256 KB / ACC 512 KB 和更大的配置（D = 16）。
5. **编译器工作项**：SPAD/ACC 变大后重新确定 chunk 的选择（每个线性层、注意力的块大小）、跨 dispatch 预取可用的空间、双缓冲的划分；新配置建立自己的黄金语料基线。
6. **D = 32 与 K2b 一起做**（原计划放到 K4b 之后；Z1 实测与 RTL 确认后改，见 §3.1 H0）：decode 时阵列每周期只消耗 D 字节权重，K2b 每周期 24–40 字节的带宽只有 D = 32 才用得上。需要 RTL（阵列 1024 个 int8 乘累加：DSP48E2 每个装两个约 512 个 DSP，或部分用 LUT；SPAD 字 256 位、ACC 字 1024 位；VE 的 FL）、描述符编码、`TargetConfig`（目前只允许 8、16）、功能仿真器同时支持，并建立 D = 32 配置自己的黄金语料基线。

**验收**：独立存储测试通过；新配置先通过语义验证（逐 dispatch 检查与功能仿真器一致），再建立**该配置自己的**黄金语料描述符基线，不覆盖 Z1 和其他配置的基线（见 §5）；上板逐位一致。

### K4a Qwen3-0.6B 全模型正确运行

1. **全模型上板**：28 层，int8 权重约 0.6 GB。C6.1 已在 sim 上逐 dispatch 逐位一致。
2. **容量与地址**：权重、KV cache、激活、描述符列表全部在低 2 GB 保留区内（§2.4）。INT8 KV：28 层 × 8 个 KV head × head_dim 128 × 2（K、V），2048 token 约 **112 MiB**，未计 scale 与布局填充。
3. 在短上下文（不超过当前不分块注意力能处理的长度）下，prefill + decode 与 sim 逐位一致。

**验收**：Qwen3-0.6B 在板上 prefill + decode 与 sim 逐位一致，生成文本正常（文本质量取决于并行的 Q 工作）。**不设 token/s 门槛。**

**结果（2026-10-09，提前在 K2a-1 的 `d16_100mhz_w128` 上完成，通过）**：
- **编译器**（f2479cc）：D = 16、M = 16 时，Qwen3 有 29 个 prefill dispatch 因 ACC 放不下而退回主机（主机 fp 与器件不逐位一致，prefill 与 decode 对不上）。现在 sahl-tile 的按行分片支持秩 ≥ 2（三维的 RMSNorm 应用）；连 2 行都放不下时（`reduction_16x3072`，SwiGLU 后的量化），按 2 行分片并标记 `sahl.row_by_row`，lowering 逐行执行、每行释放临时缓冲。黄金语料不变。
- **主机**：逐 dispatch 对拍 T = 16 / 80 / 256 各 1262 个 OK；sim 中 prefill + decode 与只用 decode 的 logits 5/5 行逐位一致。
- **板上**（`c6p_qwen3`，opt-in；启动后立即 `insmod u-dma-buf udmabuf0=671088640`，CMA 还没被泄漏）：2049 个描述符列表全部在加速器上，主机 0 个 dispatch；logits 与 sim 25/25 步逐位一致，生成的 token 与主机参考相同；器件内存峰值 595.4 MB / 640 MB。
- **速度**（100 MHz、1.576 GB/s）：prefill 40 个 token 3500.9 ms（11.43 token/s，含首块加载）；**decode 468.5 ms/步（2.13 token/s）**。仅权重读取的下限约 0.6 GB ÷ 1.576 GB/s ≈ 380 ms，decode 已达到读权重上限的约 81%。
- **prefill 耗时分解**（`c6p_qwen3_profile`，每个 16 token 的块约 89.9M 周期）：SwiGLU 段的 fp 运算占比最大：`reduction_16x3072`（逐行，16.4%）、`elementwise_16x3072`（14.8%），`matmul_like 16x384x16x1024`（15.2%）。逐行执行是 prefill 的首要优化点（K4b / KC）。

### K4b 长上下文与性能

1. **注意力按 T 分块（C6.5）**：head_dim = 128，上下文变长时一个头的 K 行放不进一个 SPAD bank，需要分块。
2. **编译器随硬件复查**：如果 K2 / K3 改变了银行组织、读延迟或 DMA 并行度，目标配置、内存规划、预取与调度都要复查，确认能用上新硬件的带宽；这不只是 T 分块一项。
3. **耗时分解**：用 `SA_PROFILE` 和性能计数器，把每个 token 的耗时分成**权重读取、KV 读取、计算（EX / VE / SFU）、调度（描述符处理、dispatch 固定开销、主机）**四部分，分别在上下文长度 **128、512、2048** 下测量。各引擎（LD、EX、VE、ST）的 busy 周期互相重叠，**不能直接相加**：按时间轴标出重叠区间，报告每个 token 的墙钟时间中各部分独占的时间与重叠的时间，以及关键路径上的引擎。
4. **性能目标按分解设定**：仅权重读取的上限是 有效带宽 ÷ 每 token 读取的权重字节数。Qwen3-0.6B 的嵌入与分类层共享一份权重（151936 × 1024，约 155 MB），分类层每个 token 读一遍，每 token 读取的权重合计约 0.6 GB：6 GB/s 时上限约 10 token/s，10 GB/s 时约 16 token/s。实际还要加上 KV 读取、计算与调度中不能重叠的部分，所以 **10–15 token/s 是冲刺目标，不是推算结果**。
   **粗略外推**（只用于说明量级）：按 24 B/周期，读 0.6 GB 权重约 25M 周期；读权重以外的部分如果按层宽与 hidden 维度从 SmolLM2 的约 9.5M 周期（通用路径）放大约 2 倍，约 19M 周期，合计约 44M 周期，250 MHz 时约 5.7 token/s。要达到 10 token/s（25M 周期以内），读权重以外的部分必须基本被读权重掩盖。所以 K4b 的主要工作是**压缩和重叠读权重以外的部分**（线性层合并、SFU 多槽交错、固定开销、跨 dispatch 预取），以 `z1_profile.md`（Z1 实测：通用路径 decode 没被读入掩盖的 8.92M 周期里，线性层尾部约 4.0M、注意力 2.4M、归约 2.6M）和 K2b 后的分解为依据排序。
5. **调优**：通用 contraction 路径的线性层改走微内核（§8.19 的做法）；按分解结果决定优先优化哪一部分。

**验收**：T = 2048 时逐 dispatch 一致、板上逐位一致；在三种上下文长度下给出耗时分解；性能目标以分解结果为依据设定并达到。

### Q（并行）Qwen3 量化质量（C6.2）

不需要等 K4，现在就可以在 CPU / sim 上进行。

1. **定位误差来源**：torch W8A8 的 Qwen3 与 fp32 的 top-1 一致率只有 25/40。按 `iree_compiler_plan.md` 的方法逐项替换成 fp32（激活、KV、概率），找出主要误差来源。
2. **SmoothQuant 类方法**：按通道把激活除以 s、权重乘以 s，这是**激活与权重的成对变换**。能折叠进相邻参数的部分（RMSNorm 增益 → q/k/v、gate/up；`up_proj` 的输出行 → `down_proj`；`Wv` 的输出行 → `wo`）折叠后，硬件与编译器不变，但要满足：
   - **数学等价**：在 fp32 下变换前后输出一致（容差内）；
   - **scale 语义**：变换改变了激活与 KV 的数值范围，int8 的 per-token 激活 scale 与 KV scale 要按变换后的数值重新校准；
   - **GQA 约束**：`Wv` 只有 8 × 128 个输出通道，而 `wo` 的输入有 16 × 128 个通道；共享同一个 KV head 的 Q head 必须用同一组 s，`wo` 处的平滑效果因此受限；
   - **SwiGLU**：`down_proj` 的输入是 `silu(gate_proj·x) ⊙ (up_proj·x)`。把 `up_proj` 的输出行除以 s，乘积的对应通道也除以 s，由 `down_proj` 的对应输入列乘以 s 补偿；缩放**不能穿过 SiLU 的 gate 分支**（非线性）。（llama2.c 的命名：`w1` = gate_proj、`w2` = down_proj、`w3` = up_proj；文档与代码统一用 HF 的名字。）
   - 分类层与嵌入共享权重，不做变换。
3. 其他候选：更细的 KV scale、个别层保留更高精度（如需硬件或编译器改动，单独评估）。

**验收**：与 fp32 的 top-1 一致率与困惑度达到 C6.2 定义的可接受范围；所选方法在 sim 上逐位一致。

### K5 可选扩展（分别评估）

| 项 | 内容 | 收益 | 依赖 |
|---|---|---|---|
| int4 权重 | LD 通路上解包 int4；按组的 scale（每 64/128 个 K 一组），微内核按 K 组分段 EX、VE 缩放后累加 | 读取量减半，decode 约 2 倍上限 | 质量需配合 GPTQ / AWQ 一类方法；LD 解包位于 K2 的新入口上 |
| 64 位 DDR 地址 | DMA 的 DDR 地址扩展到 64 位（描述符地址字段加宽，或一个高位段寄存器）、运行时 `uint64_t`。描述符列表与环形队列**继续放在低 2 GB**，PicoRV32 照常访问；它的地址空间高半部分已被 CSR（`0x8000_0000`）与 BRAM（`0xC000_0000`）占用，只有当 RISC-V 自己要读写高 2 GB 时，才需要加一个地址窗口 / 高位段寄存器 | 用满 4 GB：Qwen3-1.7B int8、3B–4B int4 | 改描述符格式（sa-desc 新版本） |
| RVV RISC-V 核 | 独立实验或替换 PicoRV32；评估 SiFive SKL、IREE 的 RISC-V 后端 | 不规则运算（采样、top-k、MoE 路由）、全 RISC-V 栈 | 资源、工具链 |
| MoE | 专家权重打包、动态权重基址的线性层微内核（`LDPARAM` + 动态 DMA 地址） | Granite-1B-A400M 这类小 MoE | K4、可能 64 位地址 |

---

## 4. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| **多口接收与存储写入吞吐不足**（K2b） | 多加 HP 口带宽不增加 | 先在 sim 中确定汇聚 + 分银行架构；`bwtest` 分别测 DDR 侧与 SPAD 写入侧；子银行冲突纳入记分板检查 |
| 缓冲分配到高 2 GB（≥ `0x8_0000_0000`），或地址先截断再检查 | 加速器读写错地址，结果错误且难查 | 保留区固定在低 2 GB；64 位解析、整段范围检查后再转 32 位；运行时报错而非静默截断 |
| 主机与固件共享结构在 AArch64 上布局变化 | 字段错位，偶发错误 | 共享结构只用定宽字段；`static_assert` 大小与偏移；固件 ABI 不随主机重编译而变 |
| 200 MHz 以上时序不收敛 | 带宽目标达不到 | 先 200 MHz；VE fp32 和 SFU 加流水级；必要时 DMA 和计算分时钟域 |
| URAM 语义与 BRAM 不同（读延迟、碰撞、不支持任意 INIT） | K3 后结果错误或时序失败 | 存储接口抽象；明确零初值只依赖配置后的状态；独立存储测试先行 |
| 带宽提高后读权重以外的部分成为主要耗时 | token/s 远低于带宽推算 | Z1 冻结前做耗时分解基线；K4b 按分解结果优先压缩和重叠非读权重部分 |
| 短 burst 时在途数不够 | 带宽受 DDR 延迟限制 | `bwtest` 按 burst 长度测量并统计队列已满周期；必要时加深 QD |
| INTERLEAVE 在子银行上冲突 | A 条带加载吞吐达不到多字/周期 | 第一版只保证正确并单独测量；成为瓶颈再做地址 swizzle 或转置缓冲 |
| 128 位后半个 beat 处理错误 | 行首行尾、4 KB 边界、D = 8 的数据错位或越界写 | LD 写 mask 与 ST `WSTRB`；按兼容性清单逐项 sim 测试 |
| HP 口多口并发的实际带宽低于预期 | K2b 收益打折 | `bwtest` 逐口测；对比 HP 与 HPC 口；调整突发长度和 outstanding 数 |
| PS 侧 cache 一致性 | ARM 写的数据加速器读不到最新值 | 沿用现有约定（非缓存映射 / 显式刷新），在 `memory_model.md` 补充 ZynqMP 的说明；明确每个缓冲的映射属性与同步点；不用 HPC 的一致性端口，除非测量证明有必要 |
| Kria-PYNQ 版本与 Vivado 版本不匹配 | overlay 加载失败 | K0 固定版本组合；必要时用 Ubuntu + 自写的 FPGA manager 加载 |
| 编译器在新参数或新银行组织下出新的边界情况 | 某些 dispatch 错误 | 逐 dispatch 检查（`dispatch_check.py --dirty --skew`）、黄金语料、功能仿真器先行 |
| Qwen3 量化质量不够 | 输出文本质量差 | Q 工作并行进行，先定位误差来源，再选择修正方法 |

---

## 5. 各阶段指标汇总

| 阶段 | 加速器时钟 | LD 入口上限 | 读带宽（实测 / 目标） | 性能 |
|---|---|---|---|---|
| PYNQ-Z1（实测） | 50 MHz | 0.4 GB/s | 实测约 0.39 GB/s | SmolLM2-135M 约 2.2 token/s（手写路径；通用路径 1.87）；Qwen3-0.6B 放不下 |
| K1a | 50 MHz | 0.4 GB/s | 实测 0.397 GB/s | 通过；stories15M 18.27 token/s，SmolLM2 2.27（通用 1.88） |
| K1b | 100 MHz | 0.8 GB/s | 实测 0.794 GB/s | 通过；stories15M 35.44 token/s，SmolLM2 4.43（通用 3.73） |
| K1c | 100 MHz（D = 16） | 0.8 GB/s | 实测 0.794 GB/s | 通过；stories15M 39.36 token/s，SmolLM2 4.77（通用 4.39），SmolLM2 prefill 22.2 token/s；GEMM 256³ 202.6 MAC/周期 |
| K2a-1 | 100 MHz（D = 16，128 位） | 1.6 GB/s | 实测 1.576 GB/s | 通过；stories15M 61.72 token/s，SmolLM2 7.98（通用 6.89），SmolLM2 prefill 27.2 token/s；GEMM 256³ 216.0 MAC/周期 |
| K2a-2 | 200–250 MHz | 3.2–4.0 GB/s | DMA → SPAD 有效带宽接近上限 | 观测 |
| K2b | 200–250 MHz | 6.4–12.8 GB/s（按所选配置） | **LINEAR 写入 SPAD 有效带宽**达到所选配置的目标（例如 2 字/周期 × 250 MHz ≥ 6 GB/s）；INTERLEAVE 记录实测值 | SmolLM2 随有效带宽提升（实测比例） |
| K4a | 同 K2b（实际在 K2a-1 上完成） | 同 K2b | — | **通过**（2026-10-09）：板上逐位一致；decode 2.13 token/s（100 MHz、1.576 GB/s），prefill 11.4 token/s |
| K4b | 同 K2b | 同 K2b | — | 按耗时分解设目标；仅权重读取的上限：6 GB/s 约 10 token/s、10 GB/s 约 16 token/s；10–15 token/s 为冲刺目标 |

**所有阶段通用的验收约束**

1. **带宽目标与配置成对**：每个带宽目标都写明对应的配置（字/周期 × 频率）和所需利用率（§3 K2 的对应表）；所需利用率超过约 85% 的组合不作为计划目标。
2. **时序以完整实现为准**：`synth_ooc.tcl` 的 OOC 综合只用来定位关键路径。验收看完整设计布局布线后的报告：setup 与 hold 都收敛（WNS、WHS ≥ 0）、所有时钟都有约束（`report_clocks`、`check_timing` 无未约束路径）、跨时钟域检查（`report_cdc`）没有未处理的路径。
3. **黄金语料按目标配置分别保留**：每个目标配置（Z1 的 D = 8、KV260 的 D = 16 与 K2 / K3 配置……）各有一份描述符基线。新配置先通过语义验证（逐 dispatch 与功能仿真器一致、端到端与 sim 逐位一致），再建立自己的基线；**不覆盖** Z1 基线（§7.2）和其他配置的基线。
4. **耗时分解标注重叠**：各引擎的 busy 周期有重叠，不能直接相加；报告墙钟时间中的独占与重叠部分（K4b 第 3 步）。

估算方法：decode 每 token 读一遍全部权重，**仅权重读取的上限** ≈ 有效带宽 ÷ 每 token 读取的权重字节数。实际还包括 KV 读取、量化、SFU、描述符处理以及不能完全重叠的计算，需按 K4b 的耗时分解实测。

---

## 6. 待确认事项

- Vivado 版本与免费授权是否覆盖 XCK26；KV260 板级文件版本。
- Kria-PYNQ 的当前版本、启动固件版本与对应的 Vivado 版本。
- ZynqMP 上 ARM 访问 PL 的 HPM 地址窗口（以 block design 生成的地址为准）。
- 低 2 GB 保留区的分配方式（设备树 reserved-memory、CMA 或 udmabuf）、映射属性，以及现有驱动能否直接使用。
- `armv7_libm_shim.c` 在 aarch64 构建中是否还需要。
- 固件和驱动中是否还有其他写死的 ARM 侧地址（逐个 grep `0x4001`、`0x4000` 等）。
- UltraScale+ 上 `sa_tdpram.v`、`sa_bankmem.v` 的 BRAM 推断结果；URAM 的读延迟配置与碰撞行为。
- K2b 的子银行数与内部总线宽度（2 / 3 / 4 字每周期），按目标带宽与配置对应表和解析性能模型选定。
- D = 32 的资源与时序：DSP48E2 每个装两个 int8 乘法的可行性、FL = 16 的 LUT（或改为 FL = D / 4）、D 通道归约树在 250 MHz 下的时序；与 H4（专用矩阵乘向量通路）比较。
- KV260 上 DDR 读延迟（随负载变化）的实测值，用于解析性能模型和 QD 的选择。
- PicoRV32 与 PCPI 在 200–250 MHz 下的时序余量。
- K2b 架构 A 中，EX / VE 读端与子银行的端口分配是否会降低现有的计算吞吐。
- 多个 HP 口并发时 DDR 控制器的实际可用带宽。

---

## 7. 仓库策略：Z1 冻结，main 转向 KV260

**决定**：不新建仓库。PYNQ-Z1 版本冻结成一个 tag 和一个分支，作为遗留版本保留；main 在 K1 验收后完全转向 KV260。

### 7.1 为什么不新建仓库

- 仓库名 `Torch2RISCV-accelerator` 与板子无关，KV260 版本继续用这个名字。
- star、关注者和 LinkedIn 帖子里的链接都留在原仓库，不需要引导读者跳转。
- 完整的提交历史和 `git blame` 保留在同一个仓库里，NOTICE 列出的第三方来源可以追溯。
- 编译器、固件、功能仿真器和黄金语料在两块板子之间不变，放在同一个仓库里，K1 的逐位一致验收可以直接对比。

### 7.2 Z1 冻结（K1 开始之前）

1. **完善 Z1**：完成计划中 Z1 上剩下的工作，main 上所有测试通过（黄金语料、`dispatch_check.py`、C3 的 stories15M、SmolLM2）。
2. **保存基线文件**，放在 `tests/baselines/pynq-z1/`，作为 KV260 逐位一致验收的参照（不再需要 Z1 板子在场）：

   | 文件 | 内容 |
   |---|---|
   | `golden_manifest.txt` | 黄金语料 4002 个 dispatch 的列表与哈希 |
   | `dispatch_check.json` | `dispatch_check.py` 的逐 dispatch 结果 |
   | `stories15m_tokens.txt` | stories15M 76 步的 token 序列（固定 prompt 和种子） |
   | `smollm2_tokens.txt` | SmolLM2-135M prefill + decode 的 token 序列 |
   | `perf.md` | 实测 token/s、`bwtest` 带宽、资源占用和时序（每个 bitstream 一行） |
   | `z1_profile.md` | 耗时分解：stories15M 与 SmolLM2 的手写路径和通用路径，按 dispatch 类型给出读入（及下限）、EX、VE / SFU、ST、固定开销和重叠的周期（`board_profile.py` + `SA_PROFILE_PERF`：运行时读加速器的事件计数器）；K 各阶段性能预期的依据 |

   **已完成**（2026-10-04）：`compiler/scripts/deploy_z1_freeze.sh` 生成全部测试包（bwtest、C3、C5.5、两个 C6.P、通用路径），`compiler/tests/board_regress.py` 在板上全部逐位一致（`REGRESSION PASS`），`compiler/tests/make_z1_baselines.py` 写出上表的文件（另有 `README.md`）。

   大的二进制输出（logits、中间张量）不进 git，放到 Release 附件里。
3. **打 tag 和分支**：

   ```sh
   git checkout main && git pull
   git tag -a v1.0-pynq-z1 -m "PYNQ-Z1: final release (stories15M 17.7 tok/s, SmolLM2-135M, bit-exact)"
   git branch pynq-z1 v1.0-pynq-z1
   git push origin v1.0-pynq-z1 pynq-z1
   ```

4. **GitHub Release**：在 `v1.0-pynq-z1` 上建一个 Release，附上最终的 bitstream / `.hwh`、固件二进制和 7.2 第 2 步中的大文件。读者不用装 Vivado 就能在 Z1 上复现。

### 7.3 KV260 开发（K0–K1）

- 在 `kv260` 分支上开发，main 在这期间仍然是可用的 Z1 版本。
- K1（K1a 与 K1b）验收通过后，把 `kv260` 合入 main。从这一刻起，main 是 KV260 版本。
- 合入的同一个 PR 里更新 README（7.5）。
- **已完成**（2026-10-08）：K1a、K1b 板上验收通过后快进合入 main，README 顶部换成 7.5 的说明，部署脚本的默认板子改为 `kv260`（`SA_BOARD=pynq-z1` 仍可用）。7.4 的目录整理另行提交。

### 7.4 合入之后整理 main

1. **目录**：

   ```
   rtl/              # 共用 RTL，参数化（D、NPORTS、SPAD/ACC 用 BRAM 或 URAM）
   rtl/picorv32/     # 从 RISCV-on-PYNQ-Z1/picorv32/ 移过来，保留 COPYING（ISC）
   firmware/         # 不变
   compiler/         # TargetConfig 默认值改为 KV260
   boards/kv260/     # block design tcl、约束、构建脚本、系统镜像说明
   driver/           # 板级地址从 board 配置读入，不写死
   ```

2. **删除 Z1 专用部分**：`RISCV-on-PYNQ-Z1/` 下的 Z1 block design、约束、bitstream 结果目录、armv7 相关的构建选项。它们都保留在 `pynq-z1` 分支和 tag 里。
3. **许可证**：如果 main 上还保留从 `RISCV-on-PYNQ-Z1`（Jinzzj，MIT）改来的脚本，NOTICE 中的那一项继续保留并更新路径；完全删除后再从 NOTICE 移除。
4. **bitstream 不再进 git**：KV260 的 bitstream 和 `.xsa` 比 Z1 的大得多，统一放 GitHub Releases（或 Git LFS）。
5. **文档**：`memory_model.md`、`hardware_learning_path.md` 等文档中 Z1 专用的内容，改成"历史：PYNQ-Z1"小节，或注明"见 `pynq-z1` 分支"。

**已完成**（2026-10-08）：

- `KV260/` → `boards/kv260/`（block design、构建脚本、README；构建输出仍在 `boards/kv260/build/`，不进 git）；`RISCV-on-PYNQ-Z1/picorv32/` → `rtl/picorv32/`，`RISCV-on-PYNQ-Z1/ip/` → `rtl/ip/`（IP 打包里 `../../picorv32/picorv32.v` 的相对路径不变），`pico_processor.tcl` → `boards/kv260/scripts/`，`RISCV-on-PYNQ-Z1/tests/ddr_access/` → `tests/ddr_access/`。
- 删除：`RISCV-on-PYNQ-Z1/` 的其余部分（Z1 block design `pico_bit.tcl`、Z1 构建脚本、约束、`bitstreams/` 下 phase3–l2 的全部结果、原始计划）、`iree-sa/l0`（armv7 的 L0）、`build_sa_runtime.sh armv7`、`board_env.sh` 的 `SA_BOARD=pynq-z1`、`export.py --armv7`、`make_z1_baselines.py`。都在 `pynq-z1` 分支上。
- 许可证：`rtl/ip/` 和 `pico_processor.tcl` 仍源自 RISCV-on-PYNQ-Z2（Jinzzj，MIT），许可证移到 `LICENSES/RISCV-on-PYNQ-Z1-MIT.txt`，NOTICE 更新路径。
- 编译器的目标配置默认值不用改：K1 的加速器与 Z1 相同（D = 8、128 KB SPAD、256 KB ACC）；K1c 起按新配置改。
- bitstream：main 上不再有 bitstream；KV260 的 overlay 放 GitHub Release。
- 文档：记录 Z1 工作的文档开头注明 `RISCV-on-PYNQ-Z1/...`、`bitstreams/...` 指 `pynq-z1` 分支；README 的目录说明、`hardware_learning_path.md` 的构建步骤改为 KV260。
- 验证：`-bd_only` 用新路径通过（地址映射与 K1 相同），`tests/ddr_access` 和 `firmware/gemm` 的系统仿真 PASS，aarch64 运行时构建通过。

### 7.5 README 说明

放在 README 最上方：

```markdown
> **Platform update:** Development has moved to the AMD Kria KV260.
> The PYNQ-Z1 version is frozen at tag
> [`v1.0-pynq-z1`](../../tree/v1.0-pynq-z1) (branch `pynq-z1`):
> stories15M at 17.7 tok/s and SmolLM2-135M, bit-exact on the board.
> It still works and is the cheapest way to try the full stack, but it
> only receives critical fixes.
```

在 K1 合入前，可以先放一个预告版本："KV260 port in progress on branch `kv260`"。

### 7.6 `pynq-z1` 分支的维护

- 只修严重问题：结果错误、构建失败、文档中的错误步骤。不加新功能、不做性能优化。
- 修复先在 `pynq-z1` 上提交，打小版本 tag（`v1.0.1-pynq-z1` …）；如果 main 上也有同样的问题，再单独移植过去。
- Issue 模板中加一个"Board: PYNQ-Z1 / KV260"选项，方便区分。

### 7.7 时间线

| 时间点 | main | 其他 |
|---|---|---|
| 现在 → Z1 完善完成 | Z1 版本 | — |
| Z1 冻结（**2026-10-04 完成**） | Z1 版本 | tag `v1.0-pynq-z1`、分支 `pynq-z1`、Release、基线文件 |
| K0–K1 | Z1 版本（README 预告 KV260） | `kv260` 分支开发 |
| K1 验收（**2026-10-08 完成**） | **合入 KV260**，README 更新，目录整理（之后另行提交） | `pynq-z1` 只修严重问题 |
| K2 及以后 | KV260 版本 | — |

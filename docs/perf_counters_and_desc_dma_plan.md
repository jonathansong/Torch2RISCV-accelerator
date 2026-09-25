# 性能计数器与描述符 DMA：方案、实现与结果

状态：**两部分都已完成并上板验证**。下文先是各步骤的完成记录（与原计划的差别都写在其中），
然后是原始方案，保留作为设计依据。

- **第 1 部分（性能计数器，P1–P4）已完成并上板验证**（`RISCV-on-PYNQ-Z1/bitstreams/m4p/`）
- **D0 决策已完成**（见下）
- **第 2 部分（描述符 DMA）D1–D5 已完成并上板验证**（`RISCV-on-PYNQ-Z1/bitstreams/m5/`）；D6（ARM 直接提交）未做

设计细节以设计文档为准：§8.6（描述符格式与语义）、§10.4（计数器实测）、§10.5（列表与 PCPI 对比）。

> **P1 完成情况**（仿真阶段；上板见 P3）：`rtl/sysarray/sa_perf.v` + 各模块事件端口 + `mat_perf`
> （funct7 = 1，funct3 = 5）+ CSR 镜像 0x40–0xBC / `PERF_CTRL` 0x3C + CAPS bit 20。
> `tb_sa_unit` 新增 365 项计数器检查（精确不变量、定向 RAW 冲突、冻结 / 清零 / CSR 镜像），
> D = 8 / 16 与 `PERF = 0` 全部通过；5 个突变全部被发现；`make test`、`make test16` 和
> 4 个固件系统仿真不回归。OOC（D = 16，50 MHz）：42,819 LUT（+988，`sa_perf` 558）、
> 29,729 FF（+953）、BRAM / DSP 不变，WNS +1.631 ns（M4 为 +1.864 ns）。
>
> **P2 完成情况**（仿真）：intrinsics（`mat_perf_ctl` / `mat_perf_read`、`sa_perf_begin` /
> `sa_perf_end`，先查 CAPS bit 20，旧硬件上不执行 `mat_perf`）；程序 BRAM 0x1E00–0x1EFF 为
> 计数器区，`link.ld` 缩到 0x1E00，全部固件重新链接；`gemm_fw` / `vector_fw` 在计时窗口两侧
> 开关计数器并转存，个数写入 `MBOX_PERF_COUNT`（0x8C）；`tb_system.v` 读回并核对不变量
> （gemm 178 项、vector 49 项，D = 8 / 16 全部通过），打印周期分解；驱动 `stats["perf"]`、
> `PERF_NAMES`、`perf_breakdown()` / `format_breakdown()`；板上脚本 `notebooks/m4_perf.py`
> （假 overlay dry run 通过）。与 1.5 的差别：驱动总是读回计数器（`stats["perf"]`，
> 旧硬件上为 None），没有单独的 `perf=` 参数；`CYCLES` 与固件 `rdcycle` 窗口的容差为 +128
> 周期（两端各有几条指令落在计数窗口内）。
>
> **P3 完成情况**（上板）：bitstream（D = 16，PERF = 1）WNS +1.904 ns、48,267 LUT（90.7%）；
> `m4_perf.py` 7 个 GEMM + 4 个向量运算结果全部正确，**69/69 项计数器检查通过**；`m4_demo.py`
> 回归全部 PASS，性能与 M4 调优后完全相同。板上 `CYCLES` 比固件 `rdcycle` 窗口多一个**固定**
> 偏移（GEMM 131、向量 175 周期：两端几条指令经 AXI 取指，每条约 10 周期），`m4_perf.py`
> 因此检查 0 ≤ 偏移 ≤ 512，而不是仿真时的 +128。
>
> **P4 完成情况**：实测周期分解写入设计文档 §10.4（替代 §10.3 的估算），bitstream 归档到
> `bitstreams/m4p/`。
>
> **D0 结论：描述符 DMA 只对小运算值得做。**
> - 大 GEMM（256³ 及以上）：CPU 61–72% 的时间卡在满队列上，队首为空只有 2–4%，前端不是瓶颈，
>   描述符 DMA 无收益。阵列有效 79–87%，其余主要是结构性填充（每 tile 2(D−1) 步）和等 bank（5–11%）。
> - 小 GEMM 和短向量运算：队首为空或完全空闲占 60–94%（64³ 86.8%，int8 add 4096 93.9%），
>   瓶颈是 PicoRV32 发射命令。描述符 DMA 的收益集中在这里；64³ 估计从约 4.2k 降到约 2.5k 周期
>   （之后受 16 KB 的 C 写回和计算限制），以实测为准。
> - DMA 忙时 7.4–7.97 B/cycle，接近端口上限，再次确认不需要三端口。
>
> **D1–D3 完成情况**（仿真）：格式与语义写入设计文档 §8.6；`pkt_*` 公共函数（`sa_defs.vh`）；
> `sa_cmdfetch.v` + 端口 0 读仲裁（归属 FIFO）+ `mat_submit`（funct3 = 6）+ BASE0–3 + 状态 CSR 0xC0–0xD0 +
> `ext_status` bit 24 + CAPS bit 21 + 计数器 28–31。`tb_sa_unit`：列表 GEMM、列表随机命令流（与顺序参考模型比对）、
> 26 项定向检查（混合顺序、FENCE / 无 FENCE / FENCE_BEFORE、JUMP、END、count、RELOC、5 类错误与恢复、FETCH_DESC），
> 4 个突变全部被发现；`make test` / `test16` / `PERF = 0` 与全部固件系统仿真不回归。软件：`firmware/desc_run`、
> 驱动 `DescList` / `build_gemm_list` / `build_vector_list` / `run_list` / `gemm_list` / `vector_list`，
> **联合仿真**：驱动建的 7 个列表（GEMM、B 拆分、int8、向量）在 `tb_system` 上由 RTL 执行，D = 8 / 16 逐字节正确；
> 板上脚本 `notebooks/m5_desc_demo.py`。OOC（D = 16）：43,540 LUT（+721，`sa_cmdfetch` 816）、30,970 FF、
> BRAM / DSP 不变，WNS +2.108 ns。
>
> 与 2.x 计划的差别：描述符 FIFO 用 LUTRAM（32 × 64 位，4 条预取），不占 BRAM；只检查 header 的保留位和 opcode；
> END 的 IRQ 标志保留未实现；FENCE 用调度器的队列占用和引擎忙位判断；**列表由 ARM 建**（PicoRV32 写一条
> 描述符要约 160 周期，比直接发 PCPI 还慢），因此没有做 `gemm_fw` / `vector_fw` 的列表模式，改为 `desc_run`
> 固件 + 驱动建表。
>
> **D4 / D5 完成情况**（上板）：bitstream WNS +2.460 ns、48,928 LUT（92.0%）。`m5_desc_demo.py` 全部正确：
> GEMM 16³ **4.19×**、64³ 1.23×、int8 64³ 1.52×、128³ 及以上 1.00×；向量 int8 add 1.52×、广播 max **2.60×**、
> int32→int8 1.27×；同一张表经 BASE 重定位复用三次全部正确。`m4_perf.py` 69/69、`m4_demo.py` 不变。
> 结果写入设计文档 §10.5。64³ 的收益（1.23×）低于 D0 估计的约 1.6×：用列表时队首仍有 24% 为空，
> 取指单元每条描述符至少 9 个周期，且每条都要等自己的 8 拍读取。

1. **性能计数器**（第 1 部分）：让硬件自己报告周期花在哪里；
2. **描述符 DMA**（第 2 部分）：软件把命令写成一张表放进 DDR，由硬件自己取指执行。

两者有先后依赖：先做性能计数器，用板上数据量化命令发射开销，再决定描述符
DMA 的投入和优化重点。这延续了本项目“先测量再建”的做法，参见设计文档 §10.1 里
三端口 DMA 的取舍。

相关文档：[`double_buffer_design.md`](double_buffer_design.md)（架构，§7 记分板、§8 指令、
§10.3 M4 实测）、[`execution_walkthrough.md`](execution_walkthrough.md)（一次 GEMM 的逐层执行过程）。

---

## 目录

- [0. 背景](#0-背景)
- [1. 性能计数器](#1-性能计数器)
  - [1.1 要回答的问题](#11-要回答的问题)
  - [1.2 计数器定义](#12-计数器定义)
  - [1.3 推导指标](#13-推导指标)
  - [1.4 硬件设计](#14-硬件设计)
  - [1.5 软件接口](#15-软件接口)
  - [1.6 验证](#16-验证)
  - [1.7 资源与时序](#17-资源与时序)
  - [1.8 实施步骤](#18-实施步骤)
- [2. 描述符 DMA](#2-描述符-dma)
  - [2.1 动机](#21-动机)
  - [2.2 描述符格式](#22-描述符格式)
  - [2.3 执行语义](#23-执行语义)
  - [2.4 硬件设计](#24-硬件设计)
  - [2.5 指令、CSR 与状态](#25-指令csr-与状态)
  - [2.6 软件](#26-软件)
  - [2.7 验证](#27-验证)
  - [2.8 资源](#28-资源)
  - [2.9 实施步骤](#29-实施步骤)
- [3. 总体计划](#3-总体计划)
- [4. 风险与对策](#4-风险与对策)
- [附录 A：扩展后的 funct7 = 1 编码](#附录-a扩展后的-funct7--1-编码)
- [附录 B：修改文件清单](#附录-b修改文件清单)

---

## 0. 背景

M4（D = 16）板上实测 256³ GEMM 为 82,813 周期、202.6 MAC/cycle（峰值的 79%），
但周期分解（设计文档 §10.3）是**估算**的：阵列计算约 73.5k、B 加载约 4.1k，
其余约 5k 说不清。另外两处明显的低效：

- 64³ 只有峰值的 25%；
- int8 向量加法只有 1.19 元素/周期。

它们是受命令发射速度限制，还是受 DMA 或调度限制？现在没有数据可以回答。

当前的可观测手段只有固件里的 `rdcycle`（总周期）和 `ext_status`（空闲、错误）。

---

## 1. 性能计数器

### 1.1 要回答的问题

| # | 问题 | 用哪些计数器回答 |
|---|---|---|
| Q1 | 阵列有多少周期在做有效乘累加？流水填充、写回等待、空闲各占多少？ | `EX_USEFUL`、`EX_STEP`、`EX_SWAPWAIT`、`CYCLES` |
| Q2 | 记分板卡住了哪类命令，卡了多久？ | `HAZ_LD/ST/EX/VE` |
| Q3 | 前端（固件发射）是否跟得上？ | `STARVE`、`PCPI_QFULL`、`ALL_IDLE` |
| Q4 | DMA 的实际带宽、AXI 反压比例是多少？ | `LD/ST_BUSY`、`LD/ST_BEATS`、`LD_ARSTALL`、`ST_WSTALL` |
| Q5 | VE 的端口冲突、输出 FIFO 满的比例是多少？ | `VE_ACTIVE`、`VE_RDBLOCK`、`VE_CREDIT` |

### 1.2 计数器定义

所有计数器都是 32 位（50 MHz 下约 86 秒溢出），只在使能窗口内计数。事件条件
全部用**现有 RTL 信号**表达，不改变任何引擎的行为。

| 编号 | 名称 | 何时 +1 | 信号来源 |
|---|---|---|---|
| 0 | `CYCLES` | 使能窗口内每个周期 | — |
| 1 | `CMD_LD` | `dispatch && h_eng == ENG_LD` | `sa_sched.v` |
| 2 | `CMD_ST` | `dispatch && h_eng == ENG_ST` | `sa_sched.v` |
| 3 | `CMD_EX` | `dispatch && h_eng == ENG_EX` | `sa_sched.v` |
| 4 | `CMD_VE` | `dispatch && h_eng == ENG_VE` | `sa_sched.v` |
| 5 | `PCPI_QFULL` | `state == S_EXEC && q_valid && !q_ready`（CPU 被满队列卡住） | `sa_pcpi.v` |
| 6 | `PCPI_FENCE` | `state == S_EXEC && grp == 1 && op == 4 && !(fence_ok \|\| sched_err)` | `sa_pcpi.v` |
| 7 | `HAZ_LD` | `d2_v && !err && h_valid_cmd && hazard && h_eng == ENG_LD` | `sa_sched.v` |
| 8 | `HAZ_ST` | 同上，`h_eng == ENG_ST` | `sa_sched.v` |
| 9 | `HAZ_EX` | 同上，`h_eng == ENG_EX` | `sa_sched.v` |
| 10 | `HAZ_VE` | 同上，`h_eng == ENG_VE` | `sa_sched.v` |
| 11 | `DISP_FULL` | `d2_v && !err && h_valid_cmd && !hazard && (e_full[h_eng] \|\| m_full[h_eng])` | `sa_sched.v` |
| 12 | `STARVE` | `!d2_v && eng_busy != 0`（队首无命令，但有引擎在忙） | `sa_sched.v` |
| 13 | `ALL_IDLE` | `pipe_empty && eng_busy == 0`（完全空转，纯软件开销） | `sa_sched.v` |
| 14 | `EX_STEP` | `step`（阵列使能） | `sa_ex.v` |
| 15 | `EX_USEFUL` | `step && live_new`（读入新的 K 数据） | `sa_ex.v` |
| 16 | `EX_SWAPWAIT` | `streaming && c == t_end && drain_active`（等上一个 tile 写回） | `sa_ex.v` |
| 17 | `EX_TILES` | `swap_now` | `sa_ex.v` |
| 18 | `LD_BUSY` | `active` | `sa_ld.v` |
| 19 | `LD_BEATS` | `lw_en`（每个 8 字节数据拍） | `sa_ld.v` |
| 20 | `LD_ARSTALL` | `\|(m_arvalid & ~m_arready)` | `sa_ld.v` |
| 21 | `ST_BUSY` | `active` | `sa_st.v` |
| 22 | `ST_BEATS` | `\|(m_wvalid & m_wready)` | `sa_st.v` |
| 23 | `ST_WSTALL` | `\|(m_wvalid & ~m_wready)` | `sa_st.v` |
| 24 | `VE_ACTIVE` | `active` | `sa_ve.v` |
| 25 | `VE_RDBLOCK` | `rd_can && rd_block`（读被写抢了端口） | `sa_ve.v` |
| 26 | `VE_CREDIT` | `active && !reading && rg < ngroups && inflight == OQ` | `sa_ve.v` |
| 27 | `VE_GROUPS` | `wr_now && wr_last`（写完一组） | `sa_ve.v` |
| 28–31 | 预留 | 给描述符 DMA 用：`FETCH_DESC`、`FETCH_WAIT`、`FETCH_ARSTALL` 等（见 2.4） | `sa_cmdfetch.v` |

说明：

- `HAZ_*` 按**队首**命令的类型分类。顺序派发下，队首卡住，后面的命令全部被卡。
- `STARVE` 高说明固件发射太慢，这正是描述符 DMA 要解决的问题；`PCPI_QFULL` 高说明
  固件已经领先硬件，瓶颈不在发射。

### 1.3 推导指标

| 指标 | 公式 | 说明 |
|---|---|---|
| 阵列有效利用率 | `EX_USEFUL / CYCLES` | 256³ 预计约 70% |
| 流水填充开销 | `(EX_STEP − EX_USEFUL) / CYCLES` | 每个 tile 的 2(D−1) 周期错位 |
| 有效 MAC 数 | `EX_USEFUL × D²` | 应等于 M·N·K（可用来校验） |
| LD 带宽（忙时） | `LD_BEATS × 8 / LD_BUSY` B/cycle | 对照 bwtest 的 7.9 |
| ST 带宽（忙时） | `ST_BEATS × 8 / ST_BUSY` | |
| 前端饥饿比例 | `STARVE / CYCLES` | 大说明需要描述符 DMA |
| 冲突等待比例 | `Σ HAZ_* / CYCLES` | 按类型拆开看 |
| VE 吞吐 | `VE_GROUPS / VE_ACTIVE` 组/周期 | |

### 1.4 硬件设计

#### 1.4.1 新模块 `rtl/sysarray/sa_perf.v`

```verilog
module sa_perf #(
    parameter integer NCNT = 32,        // 计数器数量（28 个已定义 + 4 个预留）
    parameter integer PERF = 1          // 0: 整块综合掉，读出恒为 0
) (
    input  wire              clk, resetn,
    input  wire [NCNT-1:0]   ev,        // 事件，每位对应一个计数器（bit 0 = CYCLES，恒为 1）
    input  wire              en,        // 使能：1 = 计数
    input  wire              clear,     // 单周期脉冲：全部清零
    input  wire [4:0]        rsel,      // 读出选择
    output reg  [31:0]       rdata      // 注册输出，1 周期延迟
);
```

- **事件先打一拍**（`ev_q <= ev`），再进计数器，避免把各引擎的组合逻辑拉长成
  新的关键路径。计数晚一个周期，对统计没有影响。
- **每个计数器**：`if (clear) cnt <= 0; else if (en && ev_q[i]) cnt <= cnt + 1;`
- **读出**：32 选 1 多路器 + 输出寄存器。
- **`PERF = 0` 时**：用 `generate` 把计数器和多路器整体去掉，读出为 0，给资源紧张
  时留一个开关。

#### 1.4.2 事件引出（各模块只加输出端口，不改逻辑）

| 模块 | 新增输出端口 | 内容 |
|---|---|---|
| `sa_sched.v` | `perf_ev[12:0]` | `CMD_*`（4）、`HAZ_*`（4）、`DISP_FULL`、`STARVE`、`ALL_IDLE`，外加 `dispatch` 等 |
| `sa_pcpi.v` | `perf_ev[1:0]` | `PCPI_QFULL`、`PCPI_FENCE` |
| `sa_ex.v` | `perf_ev[3:0]` | `EX_STEP`、`EX_USEFUL`、`EX_SWAPWAIT`、`EX_TILES` |
| `sa_ld.v` | `perf_ev[2:0]` | `LD_BUSY`、`LD_BEATS`、`LD_ARSTALL` |
| `sa_st.v` | `perf_ev[2:0]` | `ST_BUSY`、`ST_BEATS`、`ST_WSTALL` |
| `sa_ve.v` | `perf_ev[3:0]` | `VE_ACTIVE`、`VE_RDBLOCK`、`VE_CREDIT`、`VE_GROUPS` |
| `sa_unit.v` | — | 例化 `sa_perf`，把上面的端口按 1.2 的编号拼成 `ev` |

#### 1.4.3 控制与读出通路

两条路径，读的是同一组计数器：

1. **PCPI 指令 `mat_perf`**（funct7 = 1，funct3 = 5）：固件用，几个周期一条，见 1.5.1。
2. **CSR 镜像**（AXI-Lite，`sa_legacy.v` 的读出 `case`）：调试和 legacy 固件用。
   - 地址 `0x40 + 4·i`（i = 0…31，即 0x40–0xBC）；
   - 当前 CSR 只用到 0x28（`R_EXT`），`addr[7:2]` 共 64 个槽，空间足够；
   - `0x3C` 作为控制寄存器 `PERF_CTRL`：写 bit0 = 清零，bit1 = 使能；读回使能状态。

**CAPS 寄存器**：新增 bit 20 = `PERF`（有计数器），bit 21 = `DESC`（有描述符 DMA，第 2 部分）。
软件据此判断硬件能力，旧 bitstream 上这两位为 0。

### 1.5 软件接口

#### 1.5.1 指令 `mat_perf`

| 形式 | rs1 | rs2 | rd |
|---|---|---|---|
| 读 | bit31 = 0，[4:0] = 计数器编号 | — | 计数值 |
| 控制 | bit31 = 1 | bit0 = CLEAR，bit1 = ENABLE | 计数器个数（NCNT；`PERF = 0` 时为 0） |

```c
/* firmware/include/sysarray_intrinsics.h（新增） */
#define SA_PERF_CLEAR   (1u << 0)
#define SA_PERF_ENABLE  (1u << 1)
enum { SA_PC_CYCLES = 0, SA_PC_CMD_LD, SA_PC_CMD_ST, SA_PC_CMD_EX, SA_PC_CMD_VE,
       SA_PC_PCPI_QFULL, SA_PC_PCPI_FENCE, SA_PC_HAZ_LD, SA_PC_HAZ_ST, SA_PC_HAZ_EX,
       SA_PC_HAZ_VE, SA_PC_DISP_FULL, SA_PC_STARVE, SA_PC_ALL_IDLE, SA_PC_EX_STEP,
       SA_PC_EX_USEFUL, SA_PC_EX_SWAPWAIT, SA_PC_EX_TILES, SA_PC_LD_BUSY, SA_PC_LD_BEATS,
       SA_PC_LD_ARSTALL, SA_PC_ST_BUSY, SA_PC_ST_BEATS, SA_PC_ST_WSTALL, SA_PC_VE_ACTIVE,
       SA_PC_VE_RDBLOCK, SA_PC_VE_CREDIT, SA_PC_VE_GROUPS, SA_PC_NUM = 32 };

static inline uint32_t mat_perf_ctl(uint32_t flags)     /* 返回计数器个数 */
{
    uint32_t rd;
    __asm__ volatile (".insn r 0x0B, 5, 1, %0, %1, %2" : "=r"(rd) : "r"(1u << 31), "r"(flags));
    return rd;
}
static inline uint32_t mat_perf_read(uint32_t idx)
{
    uint32_t rd;
    __asm__ volatile (".insn r 0x0B, 5, 1, %0, %1, x0" : "=r"(rd) : "r"(idx));
    return rd;
}
```

#### 1.5.2 固件

计数窗口与现有的 `rdcycle` 计时窗口对齐：

```c
uint32_t n = mat_perf_ctl(SA_PERF_CLEAR | SA_PERF_ENABLE);   /* t0 之前 */
uint32_t t0 = rdcycle();
... 发射命令 ...
uint32_t st = mat_fence(SA_ENG_ALL);
uint32_t t1 = rdcycle();
mat_perf_ctl(0);                                             /* 冻结 */
for (uint32_t i = 0; i < n && i < PERF_AREA_WORDS; i++)      /* 写进计数器区（片上 BRAM） */
    PERF_AREA[i] = mat_perf_read(i);
```

**计数器区：放在程序 BRAM 里、mailbox 下面**

```
程序 BRAM（8 KB，RISC-V 0xC0000000，ARM 0x40010000）
0x0000 ┌──────────────────────────┐
       │ 固件代码 + 数据 + 栈       │  最大 0x1E00（7.5 KB），栈顶 0x1E00
0x1E00 ├──────────────────────────┤
       │ 计数器区 PERF_AREA        │  256 B = 64 个字（新增）
0x1F00 ├──────────────────────────┤
       │ mailbox                  │  256 B，布局不变
0x2000 └──────────────────────────┘
```

- **为什么不放 DDR、也不塞进 mailbox**：mailbox 只剩 29 个空闲字（0x8C–0xFC），放不下
  32 个计数器，塞满了以后也没地方加参数。在 mailbox 下面另开 256 B 的计数器区：
  - 不需要 DDR 缓冲区，RISC-V 也不用经 HP0 写 DDR；
  - 现有 mailbox 的字段偏移**完全不变**；
  - 最多放 64 个计数器，描述符 DMA 的 4 个也放得下；
  - 固件卡住时，ARM 也能直接读计数器区，看它停在哪里。
- **代价**：固件可用空间从 0x1F00 降到 0x1E00（7.5 KB）。目前最大的固件
  `vector_fw.bin` 约 1.4 KB，余量充足。
- **所有固件都要重新链接**（`link.ld` 的 `LENGTH` 改为 0x1E00，栈顶随之下移），代码不用改。
  **不需要新的 bitstream**：BRAM 大小和 mailbox 偏移都没变，已归档的 m1–m4 bitstream 配新固件照常工作。
- **`mailbox.h` 新增**：`PERF_AREA_BASE 0xC0001E00`、`PERF_AREA_OFFSET 0x1E00`、`PERF_AREA_WORDS 64`，
  以及 `#define PERF_AREA ((volatile uint32_t *)PERF_AREA_BASE)`。
- **开销**：32 次 `mat_perf` 读 + 32 次片上 BRAM 写，约 100–200 周期，都在计时窗口之外。
- **适用范围**：`gemm_fw.c`、`vector_fw.c` 都加；`bwtest_fw.c` 可选。

#### 1.5.3 驱动与脚本

- `pynq_matmul.py`：
  - `load_firmware()` 的大小上限从 0x1F00 改为 0x1E00，并同时清零计数器区；
  - `gemm()` / `vector()` 增加参数 `perf=False`。为 True 时，运行结束后经 MMIO 从
    `PERF_AREA_OFFSET`（ARM 地址 0x40011E00）读回 32 个字，不需要 DDR 缓冲区，也不需要 `invalidate()`；
  - `stats["perf"]` 为 `{名称: 值}` 字典，外加 1.3 的推导指标；
  - 新增 `PERF_NAMES` 列表（与 C 枚举同序）和 `perf_breakdown(stats)` 格式化函数。
- 新板上脚本 `notebooks/m4_perf.py`：对 64³、128³、256³、512×256×256、
  256×128×1024、int8 256³ 和几种向量运算，打印周期分解，并检查 1.6 的不变量。
  输出示例（数字仅为格式示意）：
  ```
  256x256x256   82813 cycles
    EX  useful 69.6%  fill 9.8%  swap-wait 1.2%
    head blocked: EX 7.1%  ST 3.0%  LD 0.4%   queue full 0.0%
    frontend starve 2.3%   all idle 0.5%   CPU stalled on full queue 61%
    LD 7.8 B/cycle busy 38%   ST 7.9 B/cycle busy 49%
  ```

### 1.6 验证

**单元测试**（`rtl/sysarray/sim/tb_sa_unit.v`）：每个 GEMM 之后，用精确的**不变量**核对计数器。

| 不变量 | 期望值 |
|---|---|
| `EX_TILES` | 发出的 tile 总数：Σ 每条 EX 的 repeat |
| `EX_USEFUL` | Σ tile × K |
| `EX_STEP` | Σ tile × (K + 2(D−1)) |
| `EX_USEFUL × D²` | M·N·K |
| `LD_BEATS × 8` | 所有 LD 的字节数之和 |
| `ST_BEATS × 8` | 所有 ST 的字节数之和 |
| `CMD_LD + CMD_ST + CMD_EX + CMD_VE` | 发出的命令数（被 reject 的不算） |
| `VE_GROUPS` | Σ 每条 VE 的 groups |
| `CYCLES` | 使能窗口的周期数（在 testbench 里另外计时对照） |
| 冻结后 | 两次读回的值相同 |
| 清零后 | 全部为 0 |

**其他检查**：

- **读出通路**：随机命令流结束后，PCPI 读出和 CSR 镜像读出的值一致。
- **`PERF = 0` 构建**：全部读 0，其他测试照常通过。
- **负向测试**：现有“funct7 = 1、funct3 = 5 不应被回应”的检查改成 funct3 = 7。
- **系统仿真**（`firmware/gemm`、`firmware/vector` 的 `make sim`）：`tb_system.v` 读回 DDR 里
  的计数器，打印分解，并检查 `EX_USEFUL × D² == M·N·K`。
- **突变测试**：故意把一个事件接错（比如 `EX_USEFUL` 接成 `EX_STEP`），确认不变量检查失败。
- **`make test` / `make test16`**：全部通过。

### 1.7 资源与时序

| 项 | 估算 |
|---|---|
| 计数器 32 × 32 位 | ≈ 1,024 FF |
| 递增器 | ≈ 32 × 32 LUT（进位链）≈ 1,000 LUT |
| 事件打拍 | ≈ 32 FF |
| 读出 32 选 1 × 32 位 | ≈ 250 LUT |
| **合计** | **≈ 1.2–1.3k LUT、≈ 1.1k FF、0 BRAM、0 DSP** |

- M4 现状：47.6k / 53.2k LUT（89.5%），剩余约 5.6k，放得下，放完约 92%。
- 时序：每个计数器是一级 32 位加法器，事件已打拍，50 MHz 下余量充足。
- 如果布线困难，可以把 `DSP_COLS` 从 8 改为 10：释放约 5k LUT，DSP 从 184 增到 216 / 220。

### 1.8 实施步骤

| 步骤 | 任务 | 验收标准 | 需要上板 |
|---|---|---|---|
| **P1** RTL | ① `sa_perf.v`；② 各模块引出事件端口；③ `sa_unit.v` 例化和拼接；④ `sa_pcpi.v` 认领 funct3 = 5，实现 `mat_perf`；⑤ `sa_legacy.v` 加 CSR 镜像、`PERF_CTRL`、CAPS bit 20；⑥ `sa_defs.vh` 定义计数器编号 | `make test`、`make test16` 全部通过；1.6 的不变量在 D = 8 和 D = 16 都成立；突变测试能被发现 | 否 |
| **P2** 软件 | ① intrinsics；② `mailbox.h` 加 `PERF_AREA_*`，`link.ld` 的 `LENGTH` 改为 0x1E00，重新链接全部固件；③ `gemm_fw.c`、`vector_fw.c` 把计数器写进计数器区；④ `tb_system.v` 从 `bram[0x1E00/4 + i]` 读回和检查；⑤ 驱动 `perf=` 参数和格式化；⑥ `notebooks/m4_perf.py`（先用假 overlay 做 dry run） | 系统仿真在 `SIM_D = 8`、`16` 下通过，打印分解；dry run 通过 | 否 |
| **P3** 板上 | ① OOC 综合看资源和时序；② `build_bitstream.sh -jobs 4 -sa_d 16`（由你在终端运行）；③ 上板跑 `m4_perf.py`、`m4_demo.py`（回归） | 时序满足；回归全部 PASS；不变量在板上成立；得到各形状的实测周期分解 | 是 |
| **P4** 文档 | 用实测分解替换设计文档 §10.3 的估算；新增 §10.4 “性能计数器”；bitstream 归档到 `bitstreams/m5/`（或 `m4p/`）并附 README；根据 `STARVE` / `PCPI_QFULL` 数据决定描述符 DMA（第 2 部分）的范围 | 文档更新；**描述符 DMA 是否实施、先优化哪些场景，有数据支撑** | — |

可选的后续（P5）：**事件追踪缓冲**。每次派发和完成时，把（时间戳、引擎、命令编号、
用户标签）写进一个 BRAM FIFO（64 位 × 512 深，占 1 个 BRAM36），驱动导出为 Perfetto /
Chrome trace JSON，得到各引擎的时间线（甘特图），可以直接看到“EX 在等 LD B 右半”。
与描述符的用户标签字段（2.2）配合使用最好。

---

## 2. 描述符 DMA

### 2.1 动机

当前每个硬件命令都是 PicoRV32 逐条执行的一条 PCPI 指令，外加若干 `mat_cfg`。例如
GEMM 的每个条带：`mat_cfg ×4 → mat_exec → mat_cfg ×4 → mat_exec → mat_cfg ×4 →
mat_load → mat_store …`，十几条 PCPI 指令，再加上计算参数的普通指令。

| 问题 | 描述符 DMA 如何解决 |
|---|---|
| 发射开销：小矩阵和向量运算受限于发射速度（需要 P3 的 `STARVE` 数据证实） | 硬件自己取指，一条命令只需一次 64 字节的突发读 |
| 作业期间 CPU 一直被占用 | 一条 `mat_submit` 之后 CPU 就空出来了 |
| 同一个网络每次推理都要重新执行发射代码 | 列表建一次反复用，换缓冲区只改基址（重定位） |
| Phase 5 编译器要生成代码 | 改为生成**数据**（描述符表），更简单，也更容易检查 |

**预期收益（以实测为准）**：
- 大矩阵 GEMM 基本不变，现在固件领先硬件，`PCPI_QFULL` 应该很高；
- 64³ 这类小矩阵和 int8 向量运算应有明显提升；
- 最大的变化是 CPU 在作业期间空出来，以及编译器多了一个更简单的目标。

**做之前先看 P3 数据**：如果 `STARVE` 在所有关心的形状上都接近 0，第 2 部分的优先级就应该降低。

### 2.2 描述符格式

每条描述符 **64 字节**，由 8 个小端 64 位字 `w0…w7` 组成，必须按 64 字节对齐。选 64 字节的理由：

- 正好一个 8 拍 × 8 字节的 AXI 突发；
- 64 字节对齐后永远不会跨 4 KB 边界；
- 最宽的 VE 命令（261 位命令包的字段）放得下；
- 每条自带全部配置，不依赖 `mat_cfg` 的状态，可以任意重排、复用、单独调试。

**公共头 `w0`**

| 位 | 字段 | 说明 |
|---|---|---|
| [7:0] | `opcode` | 0x01 LD、0x02 ST、0x03 EX、0x04 VE、0x10 FENCE、0x11 JUMP、0x12 END；**0x00 非法**（全零内存会被当作错误，防止忘了写 END 时跑飞） |
| [8] | `RELOC` | DDR 地址字段 = 偏移 + `BASE[BASESEL]` |
| [10:9] | `BASESEL` | 选择 BASE0–BASE3 |
| [11] | `FENCE_BEFORE` | 执行本条之前先等所有引擎空闲（相当于前面插一条 FENCE） |
| [12] | `IRQ` | 仅对 END 有效：完成时拉中断 |
| [31:13] | 保留 | 必须为 0，否则报错 |
| [63:32] | `tag` | 用户标签：调试、错误定位、追踪缓冲（P5） |

**各 opcode 的字段**（未列出的位必须为 0）

| opcode | `w1` | `w2` | `w3` | `w4` | `w5` |
|---|---|---|---|---|---|
| LD / ST | [31:0] DDR 地址（或 RELOC 偏移） | [31:0] 本地地址 LADDR，[47:32] rows，[63:48] row_bytes | [31:0] pitch，[33:32] mode（仅 LD；ST 必须为 0） | — | — |
| EX | [15:0] A，[31:16] B，[47:32] C，[59:48] Kt，[60] accumulate | [11:0] repeat（0 视为 1），[31:16] bstep，[47:32] cstep，[63:48] crow（0 视为 1） | — | — | — |
| VE | [31:0] src1 LADDR，[63:32] src2 LADDR | [31:0] dst LADDR，[63:32] LEN（元素数，D 的倍数） | [7:0] op 字节（op / RELU / REQUANT，同 `vec_cfg`），[13:8] types，[31:16] period，[47:32] scale，[52:48] shift | [31:0] zp，[63:32] clamp lo | [31:0] clamp hi |
| FENCE | [3:0] 引擎掩码（0 = 全部） | — | — | — | — |
| JUMP | [31:0] 下一段列表地址（64 字节对齐） | — | — | — | — |
| END | [31:0] 完成值（写入 `DESC_STATUS`） | — | — | — | — |

VE 的 LEN 用元素数，而不是组数，与 `vec_cfg` 的 LEN 语义保持一致；译码器右移 log2(D) 得到组数。

### 2.3 执行语义

- **启动**：`mat_submit(list, count)` 从 `list` 开始取指，执行到 END，或执行满 `count` 条
  （`count = 0` 表示不限）。一次只运行一个列表；上一个列表没结束时，新的 `mat_submit` 会等待。
- **顺序**：列表内按顺序送入调度器，与 PCPI 逐条发射完全等价。命令之间的片上依赖
  仍由记分板处理。
- **与 PCPI 混用**：`mat_submit` 本身算一条命令。列表执行期间，PCPI 上新到的排队命令
  （`mat_load/store/exec`、`vec_run`）被 `pcpi_wait` 挡住，直到列表结束。`mat_cfg`、
  `mat_perf`、`mat_fence` 不受影响。这样混合程序的执行顺序与书写顺序一致。
- **`mat_fence`**：语义扩展为同时等取指单元结束（列表跑完）。
- **DDR 依赖**：硬件仍然不跟踪。同一段 DDR 先 ST 后 LD 时，列表里必须有 FENCE 描述符，
  或者给 LD 置 `FENCE_BEFORE`，与 PCPI 路径要求 `mat_fence` 是同一条规则。
- **FENCE**：取指单元停止送出命令，直到掩码内的引擎全部空闲（复用调度器的 `fence_ok`）。
- **JUMP**：改写取指地址，可以把多段列表串起来，例如固定的前导 + 每层的列表。
- **END**：写 `DESC_STATUS`，`DESC_DONE` 计数 +1，置 IRQ 时拉中断，取指单元回到空闲。
- **错误**：以下情况置粘滞错误，**引擎号 = 4（FETCH）**，`DESC_ERR_IDX` 记下出错描述符的
  序号，取指单元停止；恢复方法与现在相同，用 `mat_reset`：
  - 非法 opcode、保留位非零、地址没对齐；
  - 取指时收到 AXI 读错误（错误码 3）；
  - 译码后的命令被调度器判为非法：现有的 `reject` 路径，错误码 1 或 2，引擎号为命令本身的引擎。

### 2.4 硬件设计

```
                PCPI: mat_submit(list, count)
                       │
┌──────────────────────▼──────────────────────────────┐
│ sa_cmdfetch.v（新模块）                                │
│  ① 取指状态机：IDLE → FETCH → RUN → (FENCE_WAIT) → IDLE │
│  ② AXI 读：每条 8 拍，最多 2 个突发在途                  │──┐
│  ③ 描述符 FIFO：1 × BRAM36（64 位 × 512 = 64 条）        │  │  AR/R
│  ④ 译码：描述符 → 261 位命令包（公共函数）                │  │
└──────────────────────┬──────────────────────────────┘  │
                       │ 优先级：legacy > cmdfetch > PCPI    │
                       ▼                                    ▼
                 sa_sched（不改）                    m0 读通道仲裁器（新）
                                                   ├── sa_ld（现有）
                                                   └── sa_cmdfetch
```

#### 2.4.1 取指单元 `sa_cmdfetch.v`

- **状态**：
  - `IDLE`：等 `mat_submit`；
  - `FETCH/RUN`：取指和送命令并行；
  - `FENCE_WAIT`：等 `fence_ok`；
  - `ERR`：停止，等 `clear_error`。
- **取指**：地址递增 64，最多 2 个突发在途。FIFO 可用空间不足 16 个 64 位字时暂停取指；
  遇到 JUMP 时丢掉已预取的后续字，从新地址重新开始。
- **送命令**：FIFO 里凑满一条完整描述符（8 个字）后译码。控制类描述符（FENCE / JUMP /
  END）在取指单元内部处理，命令类描述符送往调度器的输入端口。
- **重定位**：`BASE0–3` 四个 32 位寄存器，由 `mat_cfg` 的新 key 设置（见 2.5）。
  `RELOC = 1` 时 DDR 地址 = `BASE[BASESEL] + w1[31:0]`。
- **性能计数事件**（接入 1.2 预留的编号 28–31）：
  - `FETCH_DESC`：每译码一条描述符 +1；
  - `FETCH_WAIT`：有完整描述符但调度器输入不接收；
  - `FETCH_EMPTY`：在 RUN 状态但 FIFO 空（取指跟不上）；
  - `FETCH_ARSTALL`：取指的 AR 请求被反压。

#### 2.4.2 译码与 PCPI 共用

把 `sa_pcpi.v` 里“字段 → 261 位命令包”的拼装抽成公共函数，放进 `sa_defs.vh`：

```verilog
function [`SA_PKT_W-1:0] pkt_ld (input [31:0] ddr, laddr, input [15:0] rows, rb,
                                 input [31:0] pitch, input [1:0] mode);
function [`SA_PKT_W-1:0] pkt_st (...);
function [`SA_PKT_W-1:0] pkt_ex (input [15:0] a, b, c, input [11:0] kt, input acc,
                                 input [11:0] rep_m1, input [15:0] bstep, cstep, crow);
function [`SA_PKT_W-1:0] pkt_ve (...);
```

PCPI 路径和描述符路径都调用这些函数，命令格式只维护一份。这一步单独作为 D1 的重构，
先保证现有测试全部通过。

#### 2.4.3 读端口仲裁（在 `sa_unit.v`）

- **AR 通道**：LD 和取指单元二选一，轮询；取指请求很少，基本不影响 LD。
- **R 通道**：没有使用 AXI ID，同一端口的读数据按请求顺序返回。用一个 **归属 FIFO**
  （深度 16，覆盖 LD 最多 8 个加取指最多 2 个在途突发）记录每个已发出突发的归属；
  R 数据拍按 FIFO 头部分发，`RLAST` 时弹出。
- **不用额外的 HP 口**：描述符流量很小，每条命令 64 字节，而一条 LD 通常搬几 KB。
  这样也不用改 block design。
- **备选**：用空闲的 `m1` 端口接 HP1/HP3。这样不需要仲裁，但要改 `pico_bit.tcl`、多占
  一个 HP 口，不推荐。

#### 2.4.4 与 PCPI 的交互（`sa_pcpi.v`）

- funct3 = 6 `mat_submit`：取指单元空闲时锁存（list, count），立即回应；忙时 `pcpi_wait`。
- `fetch_busy = 1` 时，排队类指令（funct3 = 1/2/3 和 `vec_run`）不发 `q_valid`，
  保持 `pcpi_wait`，直到列表结束。
- `mat_fence` 的完成条件改为 `(fence_ok && !fetch_busy) || sched_err`。

### 2.5 指令、CSR 与状态

**新指令**

| 指令 | 编码 | 语义 |
|---|---|---|
| `mat_submit` | funct7 = 1，funct3 = 6；rs1 = 列表地址（64 字节对齐），rs2 = 最大条数（0 = 到 END） | 启动列表，立即返回（上一个列表未结束时等待） |

**`mat_cfg` 新 key**（接在现有 key 0–10 后面）

| key | 名称 | 值 |
|---|---|---|
| 11–14 | `BASE0`–`BASE3` | 重定位基址（32 位物理地址） |

**CSR（AXI-Lite 镜像，只读，调试用）**

| 地址 | 名称 | 内容 |
|---|---|---|
| 0xC0 | `DESC_ADDR` | 当前取指地址 |
| 0xC4 | `DESC_DONE` | 已完成的列表数（END 计数） |
| 0xC8 | `DESC_STATUS` | 最近一个 END 的完成值 |
| 0xCC | `DESC_ERR_IDX` | 出错描述符在当前列表里的序号 |
| 0xD0 | `DESC_EXEC` | 当前列表已译码的描述符数 |

**`ext_status`**：
- bit 24 = `FETCH_BUSY`（列表执行中）；现在 [31:21] 空闲；
- 错误引擎号 4 = FETCH（字段 [15:12] 为 4 位，放得下）。

**CAPS**：bit 21 = `DESC`。

**中断**：`sa_unit` 现有的 `irq` 输出（legacy 用）与 END 的 IRQ 做“或”。如果 block design 里
已经接到 PS 中断控制器，ARM 就可以等中断而不必轮询；是否接入在 D4 时确认。

### 2.6 软件

#### 2.6.1 固件：`firmware/include/sa_desc.h`（新）

```c
typedef struct { uint64_t w[8]; } __attribute__((aligned(64))) sa_desc_t;

/* 建表：每个函数写一条描述符，返回下一条的指针 */
sa_desc_t *desc_load (sa_desc_t *d, uint32_t ddr, uint32_t laddr, uint32_t rows,
                      uint32_t row_bytes, uint32_t pitch, uint32_t mode, uint32_t flags);
sa_desc_t *desc_store(sa_desc_t *d, uint32_t ddr, uint32_t laddr, uint32_t rows,
                      uint32_t row_bytes, uint32_t pitch, uint32_t flags);
sa_desc_t *desc_exec (sa_desc_t *d, uint32_t a, uint32_t b, uint32_t c, uint32_t kt, uint32_t acc,
                      uint32_t repeat, uint32_t bstep, uint32_t cstep, uint32_t crow);
sa_desc_t *desc_vec  (sa_desc_t *d, uint32_t src1, uint32_t src2, uint32_t dst, uint32_t len,
                      uint32_t op, uint32_t types, uint32_t period, int32_t scale, uint32_t shift,
                      int32_t zp, int32_t lo, int32_t hi);
sa_desc_t *desc_fence(sa_desc_t *d, uint32_t mask);
sa_desc_t *desc_jump (sa_desc_t *d, const sa_desc_t *next);
sa_desc_t *desc_end  (sa_desc_t *d, uint32_t status, uint32_t irq);

static inline void mat_submit(const sa_desc_t *list, uint32_t count)
{
    __asm__ volatile (".insn r 0x0B, 6, 1, x0, %0, %1" :: "r"(list), "r"(count) : "memory");
}
```

- **列表放在 DDR**。PicoRV32 经 HP0 写 DDR、加速器经 HP2 读 DDR，两条路径都不经过 ARM 缓存，
  所以不需要维护缓存一致性；但写完列表后、`mat_submit` 之前要保证写已完成（PicoRV32 的
  AXI 写是阻塞的，满足这一点）。
- **`gemm_fw.c` 列表模式**：`GEMM_FLAGS` bit 3 = `USE_DESC`。建表的逻辑和现在的 PCPI 发射
  顺序一一对应（A 预取、拆分 B 都保留），便于对比。建表时间单独计时（`MBOX_BUILD_CYCLES`），
  和执行时间分开。
- **`vector_fw.c` 列表模式**：同上。

#### 2.6.2 驱动：Python 建表器

- `pynq_matmul.py` 增加 `DescList`：用 NumPy 结构化数组（`dtype = [("w", "<u8", 8)]`）逐条
  添加描述符，`allocate()` 到 DDR 后 `flush()`。
- ARM 直接建好整张表，固件只执行 `mat_submit`（新固件 `firmware/desc_run/`：读 mailbox 里的
  列表地址和 BASE0–3，提交、fence、写回状态）。这样“建表”和“执行”完全分离，
  对应工业界的命令缓冲区模型。
- 重定位示例：一个 GEMM 的列表建好后，换输入 A 只需把新地址写进 BASE0，再重新提交。

#### 2.6.3 Phase 5 编译器

MLIR 降级的最后一步可以生成描述符数组（数据），而不是 `.insn` 序列（代码）：

- 更容易检查：可以直接打印、比对；
- 可以离线建好、运行时只改 BASE；
- 与 IREE 等框架“命令缓冲区 + 运行时提交”的模型一致。

### 2.7 验证

| 类别 | 测试 | 期望 |
|---|---|---|
| 等价性 | `tb_sa_unit` 的 GEMM、int8 GEMM、随机命令流改写成描述符列表 | 最终存储器与**同一个顺序参考模型**一致，与 PCPI 路径逐字相同 |
| 混合顺序 | PCPI 命令与列表交替 | 执行顺序与书写顺序一致 |
| DDR 依赖 | 同一段 DDR 先 ST 后 LD：有 FENCE / `FENCE_BEFORE` | 正确 |
| 负向测试 | 同上但**去掉** FENCE | 在 testbench 的 DDR 模型里能观察到读到旧数据（证明测试有效） |
| 控制描述符 | JUMP 串联三段列表；END 的状态和计数；IRQ；`count` 截断 | 符合 2.3 |
| 重定位 | 同一列表、不同 BASE | 结果分别正确 |
| 错误 | 非法 opcode（含全零）、保留位非零、列表地址没对齐、取指 SLVERR、描述符地址越界 | 错误码、引擎号 4 或命令引擎号、`DESC_ERR_IDX` 正确；`mat_reset` 后能恢复 |
| 端口仲裁 | 带随机延迟的 AXI 模型下，LD 大量读与取指并发 | R 拍分发无误；`tb_sa_dma` 式协议检查（4 KB、长度、RLAST）全部通过 |
| D = 8 / 16 | 以上全部 | 两种尺寸都通过 |
| 系统仿真 | `gemm_fw` 列表模式、`desc_run` 固件 | 结果正确；打印建表和执行周期 |

### 2.8 资源

| 项 | 估算 |
|---|---|
| 取指状态机 + 地址 / 计数寄存器 | ≈ 300 LUT、≈ 200 FF |
| 译码（公共函数，主要是多路选择） | ≈ 400–600 LUT |
| 读端口仲裁 + 归属 FIFO | ≈ 150 LUT、≈ 50 FF |
| BASE0–3、状态 CSR | ≈ 100 LUT、≈ 250 FF |
| 描述符 FIFO | **1 × BRAM36**（M4 用了 130 / 140） |
| **合计** | **≈ 1.0–1.5k LUT、≈ 0.5k FF、1 BRAM36** |

与性能计数器合计约 2.5k LUT，M4 的 LUT 占用会到约 94%。布线变难时，把 `DSP_COLS`
改为 10（释放约 5k LUT，DSP 216 / 220）。

### 2.9 实施步骤

| 步骤 | 任务 | 验收标准 | 需要上板 |
|---|---|---|---|
| **D0** 决策 | 看 P3 / P4 的 `STARVE`、`PCPI_QFULL`、`ALL_IDLE` 数据，确定要优化的场景和预期收益 | 在设计文档中写明结论 | — |
| **D1** 规范 + 重构 | ① 2.2 / 2.3 写进设计文档 §8.6；② 在 `sa_defs.vh` 抽出 `pkt_*` 公共函数，`sa_pcpi.v` 改用它们 | 文档评审；`make test`、`make test16` 全部通过（行为不变） | 否 |
| **D2** RTL | ① `sa_cmdfetch.v`；② 读端口仲裁 + 归属 FIFO；③ `sa_pcpi.v` 加 `mat_submit`、挡住 PCPI 排队命令、扩展 fence；④ BASE key、CSR、`ext_status`、CAPS bit 21；⑤ 性能计数事件 28–31；⑥ 2.7 的单元测试 | 2.7 的单元测试部分在 D = 8 / 16 全部通过；原有测试不回归 | 否 |
| **D3** 软件 | ① `sa_desc.h`；② `gemm_fw` / `vector_fw` 列表模式；③ `desc_run` 固件；④ Python `DescList`；⑤ `tb_system.v` 新模式；⑥ 板上脚本 `notebooks/m5_desc_demo.py`（dry run） | 系统仿真通过；dry run 通过 | 否 |
| **D4** 板上 | ① OOC 资源和时序；② 构建 bitstream（你在终端运行）；③ 上板：PCPI 模式 vs 列表模式（64³、128³、256³、向量运算）；CPU 空闲比例；所有回归 | 时序满足；回归全部 PASS；得到对比数据；性能计数器确认 `STARVE` 下降 | 是 |
| **D5** 文档 | 设计文档 §10.5 写结果；bitstream 归档；README / 学习路线更新 | — | — |
| **D6**（可选） | **ARM 直接提交**：把一组“门铃”寄存器（列表地址、count、启动）经 AXI GP 映射给 ARM，完全绕过 PicoRV32；需要改 block design，并处理 ARM 与 RISC-V 同时提交的仲裁 | 架构层面的决定，单独评估 | 是 |

---

## 3. 总体计划

```
P1 ─► P2 ─► P3（上板）─► P4（文档 + 决策）─► D0 ─► D1 ─► D2 ─► D3 ─► D4（上板）─► D5
                                                  └────────── 可选：P5 追踪缓冲、D6 ARM 直接提交
```

| 阶段 | 主要产出 | 新增资源（估算） | 是否需要新的 bitstream |
|---|---|---|---|
| P1–P4 | 性能计数器、实测周期分解 | ≈ 1.3k LUT、1.1k FF | 是（一次） |
| D1 | 打包函数重构 | 0 | 否 |
| D2–D5 | 描述符 DMA | ≈ 1.0–1.5k LUT、0.5k FF、1 BRAM36 | 是（一次） |
| P5（可选） | 事件追踪 | ≈ 0.5k LUT、1 BRAM36 | 是 |

两次上板构建都由你在自己的终端运行 `./scripts/build_bitstream.sh -jobs 4 -sa_d 16`，
Claude 负责构建前的 OOC 检查和构建后的时序、资源核对。

---

## 4. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| LUT 用到 92–94%，布线困难或时序变差 | 构建失败 / WNS 为负 | 先做 OOC；事件打拍；`PERF = 0` 开关；必要时 `DSP_COLS = 10` |
| 计数器事件接错，数据误导结论 | 做出错误的优化决策 | 1.6 的精确不变量 + 突变测试 |
| PCPI 与列表混用时顺序出错 | 结果错误且难以复现 | 2.3 的“列表期间挡住 PCPI 排队命令”规则；混合顺序测试 |
| 读端口仲裁破坏 AXI 顺序或 4 KB 规则 | 数据错乱 | 归属 FIFO 按发出顺序分发；沿用 `tb_sa_dma` 的协议检查 |
| 忘记 FENCE 导致读到旧的 DDR 数据 | 偶发错误 | 软件规则写进 `sa_desc.h` 的注释和设计文档；负向测试；以后可考虑硬件 DDR 依赖跟踪 |
| 全零或损坏的列表跑飞 | 执行垃圾命令 | opcode 0x00 非法；保留位必须为 0；`count` 上限 |
| 描述符 DMA 的收益比预期小 | 投入不划算 | D0 先用 P3 数据决策，做不做由数据决定 |

---

## 附录 A：扩展后的 funct7 = 1 编码

| funct3 | 指令 | 状态 |
|---|---|---|
| 0 | `mat_cfg`（key 0–10 现有；11–14 = BASE0–3，新增） | 现有 + 扩展 |
| 1 | `mat_load` | 现有 |
| 2 | `mat_store` | 现有 |
| 3 | `mat_exec` | 现有 |
| 4 | `mat_fence`（完成条件加上“取指单元空闲”） | 现有 + 扩展 |
| 5 | `mat_perf` | 新（第 1 部分） |
| 6 | `mat_submit` | 新（第 2 部分） |
| 7 | 保留（负向测试改用这个编码） | — |

## 附录 B：修改文件清单

| 文件 | 性能计数器 | 描述符 DMA |
|---|---|---|
| `rtl/sysarray/sa_perf.v` | 新 | 接入事件 28–31 |
| `rtl/sysarray/sa_cmdfetch.v` | — | 新 |
| `rtl/sysarray/sa_unit.v` | 例化、拼接事件 | 例化、读端口仲裁、输入端口优先级 |
| `rtl/sysarray/sa_pcpi.v` | funct3 = 5 | funct3 = 6、挡住排队命令、fence 扩展、改用 `pkt_*` |
| `rtl/sysarray/sa_sched.v` / `sa_ex.v` / `sa_ld.v` / `sa_st.v` / `sa_ve.v` | 引出事件端口 | `sa_sched`：`ext_status` bit 24 |
| `rtl/sysarray/sa_legacy.v` | CSR 镜像、`PERF_CTRL`、CAPS bit 20 | 描述符 CSR、CAPS bit 21、IRQ 或 |
| `rtl/sysarray/sa_defs.vh` | 计数器编号 | `pkt_*` 函数、描述符 opcode / 位定义 |
| `rtl/sysarray/sim/tb_sa_unit.v` | 不变量检查、负向测试改 funct3 = 7 | 2.7 的全部测试 |
| `firmware/include/sysarray_intrinsics.h` | `mat_perf_*` | `mat_submit`、BASE key |
| `firmware/include/sa_desc.h` | — | 新 |
| `firmware/include/mailbox.h` | `PERF_AREA_BASE` / `OFFSET` / `WORDS` | 列表地址、`MBOX_BUILD_CYCLES` 等 |
| `firmware/common/link.ld` | `LENGTH` 0x1F00 → 0x1E00（让出计数器区，栈顶下移） | — |
| `firmware/gemm`、`firmware/vector` | 把计数器写进计数器区 | 列表模式 |
| 其余固件（`matmul`、`matmul_insn`、`bwtest`） | 只需重新链接 | — |
| `firmware/desc_run/` | — | 新 |
| `firmware/sim/tb_system.v` | 读回计数器 | 列表模式测试 |
| `driver/pynq_matmul.py` | `perf=`、`PERF_NAMES`、格式化 | `DescList`、提交接口 |
| `notebooks/m4_perf.py` | 新 | — |
| `notebooks/m5_desc_demo.py` | — | 新 |
| `docs/double_buffer_design.md` | §10.3 实测、§10.4 | §8.6 格式、§10.5 结果 |

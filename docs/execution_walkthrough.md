# 执行过程详解：从 `m4_demo.py` 到 RTL

本文以 `notebooks/m4_demo.py` 里的一次 **int32 GEMM `mm.gemm(a, b)`**
(256×256×256，M4 板卡构建，D = 16) 为例，逐层跟踪一次运行：Python
驱动 → PicoRV32 固件 → PCPI 自定义指令 → 调度器 → 各引擎 → BRAM / DDR，
再返回 Python。int8 GEMM 和向量运算走的是同一条路，只是多了 VE 命令。

配套阅读：[`double_buffer_design.md`](double_buffer_design.md)（设计，
§4 EX、§5 DMA、§7 记分板、§8 指令）、[`memory_model.md`](memory_model.md)
（地址映射）、[`hardware_learning_path.md`](hardware_learning_path.md)
（学习路线，Stage 5 逐模块阅读）。

---

## 目录

0. [总览](#0-总览)
1. [Python 层](#1-python-层m4_demopy--driverpynq_matmulpy)
2. [固件层](#2-固件层picorv32-上的-firmwaregemmgemm_fwc)
3. [PicoRV32 → PCPI → `sa_pcpi`](#3-picorv32--pcpi--rtlsysarraysa_pcpiv)
4. [调度器 `sa_sched`](#4-调度器rtlsysarraysa_schedv)
5. [引擎与存储器](#5-引擎与存储器)
6. [结束与返回](#6-结束与返回)
7. [各层之间的数据契约](#7-各层之间的数据契约)
8. [按层调试清单](#8-按层调试清单)
9. [亲眼看这个过程](#9-亲眼看这个过程)
10. [另一条路径：描述符列表](#10-另一条路径描述符列表)

---

## 0. 总览

```
m4_demo.py ──► driver/pynq_matmul.py ──► AXI GP0 ──► 程序 BRAM (固件 + mailbox)
                                     └─► EMIO GPIO[0] 释放 RISC-V 复位
PicoRV32 取指 0xC0000000 ──► gemm_fw.c ──► .insn custom-0 指令
   ──► PCPI ──► sa_pcpi (解码、打包命令) ──► sa_sched (FIFO → 译码 → 记分板 → 派发)
   ──► sa_ld / sa_ex / sa_st / sa_ve ──► SPAD/ACC BRAM；DMA 经 HP2 读写 DDR
   ──► 引擎 done → 记分板释放 → mat_fence 返回 ──► 固件写 mailbox DONE
   ──► Python 轮询到 DONE → invalidate → 与 NumPy 比对
```

时钟：PL 侧 PicoRV32 与加速器共用 50 MHz（`subprocessorClk`）。
ARM 与 PL 之间只有两条控制通道：AXI GP0（写程序 BRAM / mailbox）和
EMIO GPIO[0]（RISC-V 复位）。数据全部经 DDR 交换。

---

## 1. Python 层：`m4_demo.py` → `driver/pynq_matmul.py`

### 1.1 加载 overlay：`MatmulOverlay(bit, gemm_fw.bin)`

| 步骤 | 做什么 | 走哪条硬件路径 |
|---|---|---|
| `Overlay(bitfile)` | 把 `.bit` 烧进 PL，解析 `.hwh` 得到地址映射 | PYNQ → Linux FPGA manager |
| `overlay_params()` | 从 `.hwh` 里 `MODTYPE="sa_unit"` 的 `PARAMETER` 读出 D = 16、NPORTS = 1 → `mm.d` | 纯文件解析 |
| `GPIO(EMIO[0]).write(1)` | RISC-V **保持复位** | PS GPIO → EMIO → `riscvReset` |
| `MMIO(0x40010000, 0x2000)` | 映射 8 KB 程序 BRAM 的 ARM 窗口 | `/dev/mem` |
| `load_firmware()` | BRAM 清零，再把 `gemm_fw.bin` 按 32 位逐字写入 | **AXI GP0 → `psBramController` → `riscvBram` A 口** |

`riscvBram` 是一块双口 BRAM：A 口给 ARM，B 口就是 PicoRV32 在
0xC0000000 看到的程序存储器。最后 512 字节是保留区：0x1E00–0x1EFF 是性能
计数器区，0x1F00–0x1FFF 是 mailbox。固件不能超过 0x1E00 字节（`load_firmware` 会检查）。

### 1.2 `mm.gemm(a, b)`

1. **检查**：M、N、K 都是 D = 16 的倍数；K×N ≤ 128 KB（B 常驻 SPAD_B）；
   K ≤ 65536 / D（一个 A 条带放得进一个 SPAD bank）。
2. **分配 DDR 缓冲**：`allocate()` 从 CMA 分配**物理连续**的 A、B、C，
   拷入数据。
3. **`flush()`**：把 ARM 缓存里的脏行写回 DDR。加速器经 HP 口直接访问
   DDR，不经过 ARM 缓存，不 flush 就会读到旧数据。
4. **写参数**：`A_BASE`、`B_BASE`、`C_BASE`（物理地址）、`GEMM_M/N/K`、
   `BIAS_BASE`、`GEMM_FLAGS = 0` 写进 mailbox。
5. **`_run()`**：
   1. 复位置 1；
   2. 清零 mailbox 全部 64 个字（偏移 0x1F00 起）；
   3. 写参数；
   4. **复位置 0**，RISC-V 开始从 0xC0000000 取指；
   5. 轮询 `MBOX_STATUS` 直到 `0x600D600D`（带超时）；
   6. 复位置 1（停住 RISC-V），读 `EXT_STATUS`，bit 1 = 1 表示加速器报错
      （错误码在 [11:8]，出错引擎在 [15:12]）。
6. **取结果**：`cbuf.invalidate()` 丢弃 ARM 缓存里的旧行，拷出 C，释放缓冲。
7. **`m4_demo.py`**：`np.array_equal(c, golden(a, b))`，打印 PASS、周期数、
   MAC/cycle。

---

## 2. 固件层：PicoRV32 上的 `firmware/gemm/gemm_fw.c`

### 2.1 启动：`firmware/common/start.S` + `link.ld`

- 复位向量 `PROGADDR_RESET = 0xC0000000`（`pico_processor.tcl`）。
- 设置栈顶 `__stack_top = 0xC0001E00`，紧贴在性能计数器区下面。
- 清零 `.bss`，然后 `call main`。
- `main` 返回后执行 `ebreak`：PicoRV32 进入 trap 停机，直到 ARM 再次复位它。

### 2.2 `main()` 准备阶段

1. 向 mailbox 写 `STATUS_RUNNING`。
2. `sa_init()`：读 **CAPS 寄存器（0x80000024）**。这是一次 PicoRV32
   的 AXI-Lite 读，经 RISC-V 的外设互连到 `sa_legacy` 的 CSR，读出 [7:0] = D = 16。
   同一份固件因此可以跑在 D = 8 和 D = 16 的 overlay 上。
3. 读取 mailbox 参数，**在计时前**算好所有和 D 有关的除法：
   `nt = N/16 = 16`（每个条带的 C tile 数）、`kt = K/16 = 16`、各 bank 的大小。
   PicoRV32 的除法指令要几十个周期，放进循环会拖慢命令发射。
4. `mat_reset()` 清除可能残留的粘滞错误。
5. `sa_perf_begin()`：CAPS bit 20 为 1（硬件带性能计数器）时，用 `mat_perf` 清零并启动计数器；
   旧硬件上什么都不做（否则 `mat_perf` 会触发非法指令异常）。然后 `t0 = rdcycle`。

### 2.3 发射命令

固件**只把命令排进队列**，不等它们执行完：

```
B = 64 KB ≥ 16 KB，所以拆分 B（B split）：
  mat_cfg_load(K, 8*16, N, INTERLEAVE); mat_load(B 左半 → SPAD_B bank0)
  load_strip(A0 → SPAD_A bank0)
  mat_cfg_load(K, 8*16, N, INTERLEAVE); mat_load(B 右半 → SPAD_B bank1)
  mat_cfg_store(16, 4N, 4N)
对每个条带 i（16 行 A，SPAD_A / ACC 的 bank 交替使用）：
  mat_cfg_exec(8, K, 1, N/16); mat_exec(A_i, B 左半, C_i,     kt)    ← 前 8 个 C tile
  mat_cfg_exec(8, K, 1, N/16); mat_exec(A_i, B 右半, C_i + 8, kt)    ← 后 8 个 C tile
  load_strip(A_{i+1} → 另一个 bank)      ← A 预取：排在 ST(i) 之前
  mat_store(C_i 的 16×N 条带 → DDR)
mat_fence(全部引擎)                      ← 唯一真正等待的地方
```

为什么这样排（M4 调度优化，见设计文档 §10.3）：

- **A 预取**：调度器按程序顺序派发。如果 `LD A(i+1)` 排在 `ST(i)` 之后，
  它就要等 `ST(i)`，而 `ST(i)` 要等 `EX(i)` 算完，下一个条带的加载就无法
  与当前计算重叠。
- **拆分 B**：记分板以整个 bank 为单位跟踪。B 全在一个 bank 里时，第一条
  EX 要等整个 B 加载完；拆成两半放进两个 bank，第一条 EX 只等左半。
  只在 B ≥ 16 KB 时拆分（小矩阵上两条短 EX 的开销反而更大）。

### 2.4 收尾

`st = mat_fence(0)`（等全部命令执行完）→ `t1 = rdcycle` →
`sa_perf_end()`：冻结计数器，把 32 个计数值复制到程序 BRAM 的计数器区（0x1E00），个数写进
`MBOX_PERF_COUNT` → 写 `TOTAL_CYCLES`、`JOBS_DONE`、`EXT_STATUS`、`ERRORS`，最后写
`STATUS_DONE`。驱动读回计数器，放进 `stats["perf"]`。

### 2.5 指令长什么样

每条命令是 `firmware/include/sysarray_intrinsics.h` 里的一个内联函数：

```c
mat_load(ddr, laddr)   →  .insn r 0x0B, 1, 1, x0, a0, a1
//                           opcode = custom-0, funct3 = 1 (load), funct7 = 1 (矩阵/DMA 组)
//                           rs1 = DDR 物理地址, rs2 = 本地地址 (mem << 28 | word)
mat_exec(a, b, c, kt, acc)
//  rs1 = B << 16 | A,  rs2 = acc << 28 | Kt << 16 | C
```

`llvm-objdump -d gemm_fw.elf` 里这些指令显示为 `<unknown>`
（反汇编器不认识 custom-0），例如 `0b 00 65 03` = `0x0365000b`。

---

## 3. PicoRV32 → PCPI → `rtl/sysarray/sa_pcpi.v`

### 3.1 握手

PicoRV32 遇到不认识的指令（custom-0）时，交给协处理器接口 PCPI：

```
周期     PicoRV32                      sa_pcpi
t0       pcpi_valid=1, insn, rs1, rs2  ours=1 → pcpi_wait=1（组合逻辑，同周期）
                                       S_IDLE：锁存 funct7/funct3/rs1/rs2，打包命令 → S_EXEC
t1..tn   保持 valid，等待               S_EXEC：按指令组处理（见下表）
tn       ←                             pcpi_ready=1（mat_fence 同时 pcpi_wr=1, pcpi_rd=ext_status）
tn+1     valid=0，执行下一条            S_DONE → S_IDLE
```

- **不触发异常**：16 个周期内没有 `pcpi_ready` 也没有 `pcpi_wait`，PicoRV32
  会触发非法指令异常。`sa_pcpi` 认领指令的当周期就拉高 `pcpi_wait`，所以
  `mat_fence` 这样的长停顿不会触发。
- **不认领的编码**（比如 funct7 = 3）不回应，CPU 会正常报非法指令。

### 3.2 各指令在 `S_EXEC` 里做什么

| 指令 | 行为 | CPU 停多久 |
|---|---|---|
| `mat_cfg` / `vec_cfg` | 写内部配置寄存器，立即 `respond` | 约 2 个周期 |
| `mat_load` / `mat_store` / `mat_exec` / `vec_run` | `q_valid = 1`，把命令包交给顶层仲裁器（legacy 序列器优先）；调度器 `in_ready`（8 深 FIFO 未满）时接收并 `respond` | FIFO 有空位时几个周期；**满时一直停**，这就是固件的背压 |
| `mat_fence` | 等 `fence_ok`（输入流水线空，且掩码里的引擎都空闲）或粘滞错误 → `respond(rd = ext_status)` | 直到队列里的命令全部执行完 |
| funct7 = 0（legacy） | 交给 `sa_legacy` 序列器（一个 8×8×8 作业） | 视指令而定 |

### 3.3 命令包：发射时就固化配置

`S_IDLE` 在指令到达的那一刻，用**当前**的配置寄存器组成命令包
（`SA_PKT_W` = 261 位，布局见 `sa_defs.vh`）：

- `mat_load`：`{ld_mode, ld_pitch, ld_rb, ld_rows, rs2(本地地址), rs1(DDR), CMD_LD}`
- `mat_exec`：`{crow, cstep, bstep, repeat-1, acc, Kt, C, B, A, CMD_EX}`
- `vec_run`：`{hi, lo, zp, shift, scale, period, types, op, groups, dst, src2, src1, CMD_VE}`

所以固件发射完一条命令，可以**立刻**改配置去准备下一条，已排队的命令不受影响。
粘滞错误已置位时，新命令被直接丢弃并回应，CPU 不会卡在停摆的队列上。

---

## 4. 调度器：`rtl/sysarray/sa_sched.v`

命令按程序顺序通过四级，每级最多一条命令：

```
输入 FIFO (8 深) → d1：拆字段、算 span (LD/ST 字数，EX 为 Kt×D)
               → d2：合法性检查 + bank 掩码
               → 派发：冲突检查 → 引擎队列 (4 深) + 在途掩码 FIFO (8 深)
```

### 4.1 d2：合法性检查与 bank 掩码

- **检查**：DDR 地址、行字节数、pitch 8 字节对齐；INTERLEAVE 只允许 LD、
  不能写 ACC、行字节数必须是 D 的倍数；地址范围不越界；EX 的 Kt ≠ 0；
  VE 的类型必须放在对应的存储器里（int32 在 ACC，int8/int16 在 SPAD）。
  不合法 → `reject` → 置位**粘滞错误**（错误码、引擎号进 `ext_status`）。
- **bank 掩码**：6 位 = SPAD_A b0/b1、SPAD_B b0/b1、ACC b0/b1。一条命令
  覆盖的字范围落在哪些 bank，就置哪些位；分为读掩码 `c_r` 和写掩码 `c_w`。
  例：`EX(条带 0, B 左半)` 读 SPAD_A b0 + SPAD_B b0，写 ACC b0。

### 4.2 派发与冲突

- 和**其他引擎**的在途掩码比较：RAW（我读、它写）、WAR（我写、它读）、
  WAW（都写）。
- EX 和 VE 共用 SPAD 的 B 口和 ACC 的 A 口，所以它俩只要碰到同一个 bank
  就算冲突（哪怕都是读）。
- **同一引擎**的命令天然按顺序执行，不互相检查。
- 无冲突、引擎队列和掩码 FIFO 都没满 → 派发。
- 掩码在引擎发出 **`done`** 时才弹出，而不是在引擎接收命令时：“在途”
  一直持续到这条命令真正执行完。
- **DDR 不在跟踪范围内**：同一段 DDR 先 ST 后 LD 时，软件必须在中间加
  `mat_fence`。

**顺序派发是性能关键**：一条命令卡在冲突上，它后面的所有命令都得等。

### 4.3 本例中的冲突

| 命令 | 等待什么 |
|---|---|
| EX(0, B 左半) | LD B 左半（SPAD_B b0）、LD A0（SPAD_A b0），RAW。**不等 B 右半**，这就是拆分 B 的收益 |
| EX(0, B 右半) | LD B 右半（SPAD_B b1） |
| LD A1 → SPAD_A b1 | 与 EX(0) 的 bank 不冲突，在 EX(0) 计算时并行加载 |
| ST(0)，读 ACC b0 | 两条 EX(0) 写完 ACC b0（RAW） |
| EX(1)，写 ACC b1 | LD A1 完成；可以和 ST(0) 并行 |
| LD A2 → SPAD_A b0 | EX(0) 读完 SPAD_A b0（WAR） |
| EX(2)，写 ACC b0 | ST(0) 读完 ACC b0（WAR） |

### 4.4 错误

- 非法命令，或 LD/ST 收到 AXI 的 SLVERR/DECERR → 粘滞错误，**停止派发**。
- 已派发的命令继续执行完；队列里的命令留在原地，直到 `clear_error`
  （`mat_reset`）把它们清掉。
- `mat_fence` 在有粘滞错误时也会返回，固件和驱动据此报错。

---

## 5. 引擎与存储器

### 5.1 存储器与端口分配（`sa_unit.v`）

| 存储器 | 大小 / 字宽 (D = 16) | A 口 | B 口 |
|---|---|---|---|
| SPAD_A | 128 KB，8192 字 × 16 B，两个 bank | LD 写 / ST 读 | EX 读 / VE 读写 |
| SPAD_B | 128 KB，同上 | LD 写 / ST 读 | EX 读 / VE 读写 |
| ACC | 256 KB，4096 字 × 64 B（16 个 int32），两个 bank | EX / VE 读写 | LD 写 / ST 读 |

每块由 `sa_bankmem` 分成两个 bank，每个 bank 是 `sa_tdpram`（UG901
字节写真双口 RAM 模板，推断为 RAMB36），合计 128 个 BRAM36。同一口上的
多个使用者（比如 LD 和 ST）按使能复用，记分板保证它们不会同时访问同一个 bank。

本地地址（LADDR）= `mem << 28 | word`，mem：1 = SPAD_A、2 = SPAD_B、3 = ACC。

### 5.2 LD 引擎：`sa_ld.v`

1. 命令形状：rows 行、每行 row_bytes 字节、行间距 pitch；LINEAR 或 INTERLEAVE。
   连续的行（pitch = row_bytes 且整字对齐）会合并成一行，得到更长的突发。
2. 拆成 AXI INCR 突发：每拍 8 字节、最多 16 拍、**不跨 4 KB 边界**；
   每个端口最多 8 个突发在途。
3. AR 请求：`m0_axi` → `matmulHpConverter`（AXI4 → AXI3）→ **HP2** → DDR 控制器。
4. 每个突发记住它的起始本地字和 lane。返回的 64 位数据拍按地址写进
   SPAD / ACC 对应字的对应 8 字节（字节写使能）：SPAD 字 = D/8 个 lane，
   ACC 字 = D/2 个 lane。
5. INTERLEAVE：本地字 = 块号 × rows + 行号。这样摆出 EX 需要的
   A 布局（k-tile 优先：字 w·D + g = A[g][wD .. wD+D−1]）和按列条带排的 B
   （条带 j 在字 j·K 开始，字 k = B[k][Dj .. Dj+D−1]）。
6. 最后一拍写完后发 `done`；中途收到 SLVERR/DECERR 时同时报 `err`。

### 5.3 EX 引擎：`sa_ex.v` + `sa_array.v`

1. **K 流式输入**：每周期从 SPAD_A 读一个字、从 SPAD_B 读一个字（BRAM
   读延迟 1 周期，提前一拍发读）。`rowreg` 保存每行当前的 A 字，`bsr`
   是 B 字的移位寄存器，二者一起做出对角线错位：PE(i,j) 在同一步拿到
   A[i][k] 和 B[k][j]。
2. **乘累加**：A 向右流、B 向下流，PE(i,j) 累加 C[i][j]（输出驻留）。
   前 8 列 PE 用 DSP48E1（`sa_pe_dsp`），后 8 列用 LUT 乘法器（`sa_pe_lut`）。
3. **tile 切换**：每个 tile 流 K + 2(D−1) + 1 = 287 个周期后 `swap`：累加
   结果转入影子寄存器，阵列清零，立刻开始下一个 tile。
4. **写回 ACC**：影子寄存器逐行上移，把 16 行写进 ACC，行地址 =
   c + r·cstep + i·crow（本例 crow = N/16，排成行主序的 C 条带）。写回与
   下一个 tile 的计算重叠；accumulate 模式要先读旧值（每行两个周期）。
5. **repeat**：一条命令算 8 个 tile，第 r 个 tile 的 B 基址 = b + r·bstep
   （bstep = K）。最后一个 tile 写完才发 `done`。

### 5.4 ST 引擎：`sa_st.v`

从 SPAD / ACC 按 lane 读出，经 AW/W 通道以突发写回 DDR（同样经 HP2，
同样不跨 4 KB）。所有 B 响应都收到后发 `done`；收到错误响应时报 `err`。
本例一条 ST 写 16 × 256 × 4 = 16 KB，约 2k 周期，与下一个条带的计算重叠。

### 5.5 VE 引擎：`sa_ve.v`（int8 GEMM / 向量运算时出现）

读单元每周期发一个读（先 src1 的字，再 src2 的字；广播的 src2 只读一次）→
4 级计算流水（op → ReLU + requant 乘法 → 舍入移位 → zp + 钳位 + 饱和 + 打包）→
8 深输出 FIFO → 写回（与读共用端口时写优先）。int8 GEMM 里 VE 从 ACC
读 int32，加上 bias 向量（src2 周期 = N/D）、ReLU、requant 后写进 SPAD_A
的上半 bank，再由 ST 以 int8 写回 DDR。

---

## 6. 结束与返回

1. 每个引擎的 `done` 让调度器弹出对应的在途掩码，后面被冲突卡住的命令随之派发。
2. 全部命令执行完 → `fence_ok` → `mat_fence` 返回 `ext_status`（无错误时 bit 1 = 0）。
3. 固件记录周期数，写 `STATUS_DONE`，然后 `ebreak` 停机。
4. Python 轮询到 DONE → `invalidate()` → 读出 C → 与 NumPy 比对：
   **PASS，82,813 周期，202.6 MAC/cycle**（`m4_sched_tune.py` 板上实测）。

### 周期花在哪（性能计数器实测，`notebooks/m4_perf.py`）

| 部分 | 占计数窗口 | 含义 |
|---|---|---|
| EX 有效计算（`EX_USEFUL`） | **79.0%** | 阵列在读入新的 K 数据：16 × 16 tile × 256 步 = 65,536 周期 |
| EX 填充（`EX_STEP − EX_USEFUL`） | 9.3% | 每个 tile 的 2(D−1) = 30 步错位，结构性开销 |
| 队首 EX 被冲突卡住（`HAZ_EX`） | 8.8% | EX 在等 bank（B 右半加载、上一轮 ST 读完 ACC 等） |
| 前端饥饿 + 完全空闲 | 4.3% + 0.4% | 队首没有命令：固件发射跟不上的时间，很少 |
| CPU 卡在满队列上（`PCPI_QFULL`） | 61.3% | 固件远远领先硬件，前端不是瓶颈 |
| LD / ST 忙 | 20.1% / 39.8% | 忙时 7.85 / 7.93 B/cycle，接近端口上限 |

“队首被 ST 卡住”占 86% 是预取顺序下的正常状态：ST(i) 在队首等 EX(i) 算完，后面的命令已经在引擎里执行。
详细分析见设计文档 §10.4。

---

## 7. 各层之间的数据契约

| 边界 | 契约 | 定义位置 |
|---|---|---|
| ARM ↔ RISC-V | mailbox：程序 BRAM 偏移 0x1F00 起的 256 字节；输入参数、输出状态和周期数 | `firmware/include/mailbox.h`，`driver/pynq_matmul.py` 的 `MBOX_*` |
| ARM ↔ 加速器 | DDR 物理地址、8 字节对齐；ARM 侧 flush / invalidate | `docs/memory_model.md` |
| 固件 ↔ 硬件 | custom-0 指令编码，`mat_cfg` / `vec_cfg` 的 key | `sysarray_intrinsics.h`，设计文档 §8 |
| PCPI ↔ 调度器 | 261 位命令包 | `rtl/sysarray/sa_defs.vh` |
| 调度器 ↔ 引擎 | `cmd_valid/ready` + 命令字段；`done`（+ `err`） | `sa_unit.v` 的例化连线 |
| 引擎 ↔ 存储器 | 本地字地址、字节写使能；bank = 地址高半部分 | `sa_bankmem.v` |
| 引擎 ↔ DDR | AXI4 INCR 突发，≤ 16 拍，不跨 4 KB，经 AXI3 转换到 HP2 | `sa_ld.v` / `sa_st.v`，`pico_bit.tcl` |
| 固件 ↔ ARM（计数器） | 程序 BRAM 0x1E00–0x1EFF 的计数器区，个数在 `MBOX_PERF_COUNT`（0x8C） | `mailbox.h`，`sa_defs.vh` 的 `PC_*` |
| ARM ↔ 取指单元（列表） | 64 字节描述符，64 字节对齐，放在 DDR；地址、条数、BASE0–3 经 mailbox（0x90–0xA4）传给 `desc_run` 固件 | 设计文档 §8.6，驱动 `DescList` |

---

## 8. 按层调试清单

| 现象 | 先查哪一层 | 怎么查 |
|---|---|---|
| `TimeoutError: ... not started` | 固件没跑起来 | 固件是否已加载（`load_firmware`）；复位 GPIO；固件是否超过 0x1E00 |
| `TimeoutError: ... running` | 固件卡在某条指令 | 多半卡在 `mat_fence`：看是否有命令永远派发不了（冲突不会解除）或引擎没发 `done`；用 `make sim` 复现 |
| `RuntimeError: accelerator error` | 调度器 / DMA | 错误码 [11:8]：1 = 形状 / 对齐非法，2 = 本地地址越界或存储器编号错，3 = AXI 读响应错，4 = AXI 写响应错；引擎号 [15:12]：0 = LD，1 = ST，2 = EX，3 = VE |
| 结果部分错误 | 数据布局 / 缓存 / DDR 顺序 | 是否 flush / invalidate；同一段 DDR 先 ST 后 LD 是否加了 `mat_fence`；A / B 的 INTERLEAVE 布局 |
| 结果全错但仿真正确 | 构建配置 | `.hwh` 里 `sa_unit` 的 D、SPAD_WORDS、ACC_WORDS 是否一致；固件读到的 CAPS |
| 性能比预期低 | 固件命令顺序 | 顺序派发下，一条命令在等什么（见 §4.3）；`GEMM_FLAGS` 对比各调度方案 |
| CPU 报非法指令 | 指令编码 | funct7 / funct3 是否在 `sa_pcpi` 的 `ours` 范围内；`mat_perf` / `mat_submit` 只在 CAPS bit 20 / 21 为 1 的 overlay 上可用 |
| 不知道时间花在哪 | 性能计数器 | `stats["perf"]` + `perf_breakdown()`，或 `m4_perf.py`：阵列有效率、哪个引擎在等冲突、前端是否饥饿、DMA 忙时带宽 |
| 列表执行出错（引擎号 4） | 描述符 | 错误码 1 = opcode 非法 / header 保留位非零 / 地址没对齐，3 = 取指时读错误；CSR 0xCC 是出错描述符的序号，0xC0 / 0xD0 是最后译码的地址和条数 |
| 列表里的 LD 读到旧数据 | DDR 顺序 | 同一段 DDR 先 ST 后 LD，中间要有 FENCE 描述符或给 LD 置 FENCE_BEFORE（硬件不跟踪 DDR） |

---

## 9. 亲眼看这个过程

**波形**：`firmware/gemm` 里的系统仿真覆盖了从固件指令到 RTL 的整条路径（ARM 那一侧由 `tb_system.v` 扮演）。

```sh
cd firmware/gemm && make sim SIM_D=16          # 先跑一次，生成编译产物
cd build/sim16
source /home/jon/Projects/Vivado/Vivado/2024.1/settings64.sh
xelab -debug typical tb_system -s tb_dbg
xsim tb_dbg -gui
```

在波形里依次添加：

1. `pcpi_valid / pcpi_insn / pcpi_wait / pcpi_ready`（CPU ↔ `sa_pcpi`）
2. `mm/sched/in_valid`、`in_ready`，以及 `dispatch`、`hazard`（调度器）
3. `mm/ld/cmd_valid`、`done`，`mm/ex/...`，`mm/st/...`（各引擎）
4. `mm/m0_axi_ar*`、`r*`、`aw*`、`w*`、`b*`（DMA 总线；在 `tb_system` 顶层叫 `m_ar*`、`m_r*` 等）

`dispatch` 为 0 且 `hazard` 为 1 的区间，就是命令在等记分板。

**单独看一个引擎**：`rtl/sysarray` 里 `make sim TB=tb_sa_ex GEN=D=16`
（或 `tb_sa_dma`、`tb_sa_ve`），方法同上，只是在对应的 build 目录里 elaborate。

---

## 10. 另一条路径：描述符列表

同一个 GEMM 也可以不由 PicoRV32 逐条发射，而是由 ARM 建一张描述符列表，
PicoRV32 只执行一条 `mat_submit`（M5，设计文档 §8.6）。从调度器开始，后面的路径和第 4–6 节完全相同；
不同的是命令从哪里来：

1. **Python**：`mm.gemm_list(a, b)` 调用 `build_gemm_list()`，按和 `gemm_fw.c` 完全相同的调度
   （常驻 B、B 拆分、A 预取）生成 `DescList`：每条命令一个 64 字节描述符，自带全部配置，
   末尾是 END。列表写进 `allocate()` 的 DDR 缓冲区并 `flush()`。
2. **固件**：`firmware/desc_run/desc_run_fw.c` 从 mailbox 读列表地址、条数和 BASE0–3，
   用 `mat_cfg` 设好 BASE，然后 `mat_submit(list, count)`（立即返回）和 `mat_fence`。
3. **PCPI**：funct3 = 6 发出一个启动脉冲给取指单元；列表执行期间，PCPI 上新的排队命令
   会被 `pcpi_wait` 挡住，`mat_fence` 也要等列表结束，程序顺序因此保持不变。
4. **取指单元**（`rtl/sysarray/sa_cmdfetch.v`）：从列表地址开始，每条描述符一个 8 拍突发，
   最多 4 个在途，数据进 32 × 64 位的 LUTRAM FIFO；凑满 8 个字后译码：
   - LD / ST / EX / VE：用和 PCPI 相同的 `pkt_*` 函数打包成 261 位命令包，送进调度器输入
     （优先级：legacy > 取指单元 > PCPI）；
   - FENCE：等调度器队列空、掩码内的引擎空闲；
   - JUMP：换到新地址继续，先丢掉已经在路上的预取数据（DRAIN）；
   - END：记录状态值，列表结束。
5. **读端口**：取指单元和 LD 引擎共用端口 0 的读通道。AR 请求被授权后锁定到握手完成；
   一个按 AR 顺序记录“这个突发属于谁”的归属 FIFO 把返回的 R 数据拍分给 LD 或取指单元。

实测（`notebooks/m5_desc_demo.py`，设计文档 §10.5）：本来受前端限制的小运算明显变快
（16³ 4.19×、64³ 1.23×、int8 64³ 1.52×、广播向量运算 2.60×），256³ 这类大 GEMM 不变，
因为它们的前端本来就跑在硬件前面。

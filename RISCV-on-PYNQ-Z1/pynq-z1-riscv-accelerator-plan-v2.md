# PYNQ-Z1 异构加速系统实现计划 v2
## ARM(Host/APU) + PicoRV32(RISC-V 控制核/NPU 前端) + Matrix/Vector 加速单元 + 自定义指令 + MLIR 编译链路

> 本版本相较 v1 的核心变化:**RISC-V 核心从 VexRiscv 切换为 PicoRV32**,并将"自定义指令支持"从 Phase 6 的可选加分项提升为核心主线目标。切换理由、影响范围与具体实现路径见文末《变更说明》。

---

## 0. 项目目标与价值主张

在一块 PYNQ-Z1 上复现一个简化但完整的"自研 AI 硬件"技术栈,核心目标是打通:

```
PyTorch (ARM/Linux, APU/Host)
  → torch-mlir → linalg dialect
  → 自定义 pattern match pass → sysarray dialect
  → lowering pass → RISC-V 自定义指令(.insn 内联汇编)/ LLVM dialect
  → PicoRV32(RISC-V 控制核, 裸机固件, 相当于 NPU 的标量控制前端)
  → 自定义指令译码 → PCPI 协处理器接口触发
  → Matrix/Vector 加速单元(PL, systolic array,相当于 NPU 的计算阵列)
  → 结果写回 DDR → ARM 读取验证
```

覆盖编译器架构师岗位的核心能力点:

| 岗位要求 | 项目对应部分 |
|---|---|
| 打通 PyTorch → 自研硬件的编译与执行链路 | torch-mlir → 自定义 dialect → RISC-V 自定义指令固件 |
| op dispatch、kernel launch、tensor memory、runtime ABI | PyTorch backend dispatch → PYNQ driver → PCPI 指令 ABI / CSR ABI |
| **自定义指令 / ISA 扩展设计能力** | **custom opcode 编码 + PCPI 协处理器实现 + MLIR 指令级 lowering** |
| Triton/MLIR 长期技术路线 | 自定义 MLIR dialect + lowering pass(含内联汇编生成) |
| 与 RTL/Runtime/FPGA 团队定义 DMA、内存模型、CSR、kernel ABI、指令编码 | 一人分饰,亲手设计并验证接口 |
| 测试、性能分析、软硬件一致性验证 | golden model 对比、trace 采集、松耦合 vs 紧耦合对比 |
| 指导团队开发新算子和优化 Pass | tiling pass、pattern matching pass |

---

## 1. 核心选型决策:PicoRV32 替代 VexRiscv

### 1.1 决策结论

项目主目标是"端到端 APU+NPU+accelerator 编译过程,最终支持自定义指令"——这本质上是一个**接口是否打通**的问题(编译器能否生成一条自定义指令、硬件能否正确识别并触发加速单元),而不是"CPU 微架构流水线改造深度"的问题。因此:

- **主线核心:PicoRV32**,通过其原生的 **PCPI(Pico Co-Processor Interface)** 机制承载自定义指令,复杂度低、文档成熟、有现成开源集成参考(`JacoboJin/RISCV-on-PYNQ-Z2`,fork 自 `drichmond/RISC-V-On-PYNQ`)。
- VexRiscv 的 SpinalHDL plugin 机制虽然能实现更深度的流水线级指令扩展,但对本项目目标是"重型方案",且需要额外解决 CPU 原生总线到 AXI 的桥接问题(VexRiscv 默认总线非 AXI),该仓库已经提供了这层桥接的现成实现,可直接复用。

### 1.2 选型对比

| 维度 | VexRiscv(v1 方案) | PicoRV32(v2 方案) |
|---|---|---|
| RTL 生成方式 | SpinalHDL/Scala 生成,需要 JDK+sbt 环境 | 纯手写 Verilog 单文件(~3000 行),无需生成工具链 |
| PS-PL 桥接 | 需自行解决总线协议适配(非原生 AXI) | 现成仓库已提供 `picobridge`/`picobram_if`/`picorv32_axi` 等 IP |
| 自定义指令机制 | Plugin 嵌入流水线,需处理 hazard/stall/flush | PCPI 协处理器接口,固定握手协议(`pcpi_insn`/`pcpi_rs1`/`pcpi_rs2`/`pcpi_wr`/`pcpi_rd`/`pcpi_wait`/`pcpi_ready`) |
| 实现难度 | 较高,需要读懂流水线内部时序 | 较低,接口文档化程度高,有现成范例(`pcpi_v1_0`) |
| 与项目目标契合度 | 契合"深入改造微架构"类目标 | **契合"打通端到端自定义指令链路"类目标** |
| Phase 1 预估工作量 | 2 周 | 3-5 天 |

### 1.3 决策的代价(需要认知到)

选择 PicoRV32 意味着放弃"读懂并修改 CPU 流水线内部机制"这个技术展示点。如果面试叙事中这一点很重要,可以将其列为**后续可选扩展**(见第 9 节),而不影响主线交付。

---

## 2. 总体阶段划分(v2)

| 阶段 | 内容 | 预估周期 | 产出 |
|---|---|---|---|
| Phase 0 | 环境搭建与资源摸底 | 3-5 天 | 综合报告、base overlay 资源占用基线、Vivado 版本确认 |
| Phase 1 | PicoRV32 控制核集成 | 3-5 天 | PicoRV32 跑通 PYNQ 环境,裸机 hello world |
| Phase 2 | Matrix 加速单元(CSR/AXI 松耦合,作为过渡验证) | 2-3 周 | 8x8 systolic array + CSR/DMA 接口 |
| Phase 3 | 端到端闭环(手写固件,CSR 路径) | 1-2 周 | ARM→PicoRV32→矩阵单元→结果验证 全链路跑通 |
| **Phase 4** | **自定义指令(PCPI)设计与实现** | **2 周** | **custom opcode 编码方案 + PCPI 协处理器 RTL + `.insn` 固件验证** |
| Phase 5 | MLIR dialect 设计与 lowering(含指令级 lowering) | 3-4 周 | torch-mlir → sysarray dialect → 自定义指令固件自动生成 |
| Phase 6 | 松耦合 vs 紧耦合对比实验 | 1 周 | CSR 方案 vs PCPI 指令方案的量化对比报告 |
| Phase 7 | Vector 单元(如资源允许) | 2 周 | 向量单元 + dialect/指令扩展 |
| Phase 8 | 自动 tiling + 性能优化 | 2 周 | 支持任意 shape 矩阵乘,双缓冲流水线 |
| Phase 9 | 文档整理与项目总结 | 1 周 | 技术博客/README/面试素材 |

总周期约 15-19 周(较 v1 略有压缩,主要来自 Phase 1 提速)。

**关键变化**:v1 中 Phase 6(自定义指令)是"风险预案里可裁剪的加分项",v2 中提升为 **Phase 4,属于主线核心,不可裁剪**,因为它直接对应你的核心目标。

---

## 3. Phase 0:环境搭建与资源摸底

### 3.1 任务清单
- [ ] **确认 Vivado 版本兼容性**:`JacoboJin/RISCV-on-PYNQ-Z2` 仓库要求 Vivado 2021.1,需与你手头 PYNQ-Z1 镜像配套的工具链版本核对。若版本差异较大,IP 打包/block design 可能有兼容性问题,需评估是"直接用现成 `.tcl` 脚本"还是"仅借鉴 IP 设计思路、在自己版本里重新搭建"。
- [ ] 拉取 PYNQ 官方 base overlay 工程,跑一次综合 + 实现,记录 `utilization_report`
- [ ] 记录基线数据:LUT / FF / BRAM / DSP 占用;关键 IP(Zynq PS 配置、AXI Interconnect、中断控制器)
- [ ] 确认板级 XDC 约束文件、PL 时钟来源(由 PS 提供,注意可用频率选项)
- [ ] 明确 **HP 口 vs GP 口的数据通路**:GP 口(AXI-Lite,低带宽,给 CSR/寄存器控制用)与 HP 口(高带宽,PL 访问 DDR 大块数据用)是两条不同路径,需在 block design 中提前规划清楚,避免 Phase 3 返工。
- [ ] 最小 cache coherency 验证:写一个最简单的 AXI 读写 testbench(不涉及 PicoRV32),验证 `pynq.allocate` 分配的内存在 ARM 写入、PL 读取时的实际行为(cacheable/non-cacheable),不要假设。

### 3.2 产出物
- `docs/baseline_utilization.md`
- `docs/ps_pl_datapath.md`:画清楚 GP/HP 口连接关系图

---

## 4. Phase 1:PicoRV32 控制核集成

### 4.1 核心选型
- PicoRV32,RV32I(可选 RV32IMC),精简配置起步:关闭乘除法(`ENABLE_MUL=0`, `ENABLE_DIV=0`)、开启 PCPI(`ENABLE_PCPI=1`,为 Phase 4 做准备)、不使能 IRQ(先简化)。
- 直接复用/参考 `JacoboJin/RISCV-on-PYNQ-Z2` 仓库中的 `picorv32_bram`(CPU+BRAM 一体化 IP,用于起步)和 `picorv32_axi`(暴露 AXI 接口的版本,用于后续接矩阵单元)。

### 4.2 任务清单
- [ ] Clone 参考仓库,阅读 `gold_ip/picobram_if`、`gold_ip/picobridge` 源码,理解 CPU 原生总线到 AXI 的桥接逻辑(即便最终不完全照搬,这层设计模式有通用参考价值)
- [ ] 按仓库 `notebooks/tutorial/2-Creating-A-Bitstream.ipynb` 或对应 `.tcl` 脚本(`pico_processor.tcl` + `pico_bit.tcl`)跑通最小 bitstream 生成流程
- [ ] 工具链安装:参考仓库方案(板上 SSH 编译 riscv-gnu-toolchain)或改用交叉编译(开发机上编译,视个人环境效率取舍),目标 `--with-arch=rv32im`(暂不含 c 压缩指令,简化调试)
- [ ] 编译裸机 hello world(UART 打印),验证核心真正在跑
- [ ] 烧录验证:ARM(Jupyter/PYNQ Overlay)加载 bitstream,通过 UART 或 AXI-Lite 寄存器观察 PicoRV32 输出

### 4.3 验收标准
ARM(Jupyter)能够:①复位 PicoRV32 核 ②加载固件到指令存储器 ③启动核心 ④通过 UART/寄存器观察到执行结果。

### 4.4 资源记录
对比 Phase 0 基线,记录 PicoRV32 本身实测资源开销,同时可与仓库 README 中已有的 utilization/power 截图做交叉验证(注意:仓库基于 PYNQ-Z2,与 Z1 同为 7020 系列但外设布局有差异,仅作参考不能直接当基线)。

---

## 5. Phase 2:Matrix 加速单元(CSR/AXI 松耦合方案)

> 本 Phase 目标从 v1 的"独立最终方案"调整为**过渡验证手段**:先用最直接的方式(访存式 CSR)确认矩阵单元本身逻辑正确,再在 Phase 4 引入自定义指令这一层。

### 5.1 设计目标
8x8 systolic array,int8 输入、int32 累加输出,通过 AXI-Lite CSR + AXI DMA 与外部交互。

### 5.2 CSR 寄存器组(与 v1 一致)

| 偏移 | 名称 | 读写 | 功能 |
|---|---|---|---|
| 0x00 | CTRL | RW | bit0: start, bit1: reset, bit2: irq_enable |
| 0x04 | STATUS | RO | bit0: done, bit1: busy, bit2: error |
| 0x08 | SRC_A_ADDR | RW | 输入矩阵 A 的 DDR 基地址 |
| 0x0C | SRC_B_ADDR | RW | 输入矩阵 B 的 DDR 基地址 |
| 0x10 | DST_ADDR | RW | 输出矩阵的 DDR 基地址 |
| 0x14 | DIM_M_N_K | RW | 矩阵维度打包(本阶段先固定 8x8x8) |
| 0x18 | IRQ_STATUS | RW1C | 中断状态,写1清除 |

### 5.3 任务清单
- [ ] Verilog/SystemVerilog 实现 8x8 systolic array PE(MAC + 数据前递)
- [ ] 片上 staging buffer(BRAM,先单缓冲)
- [ ] AXI-Lite CSR 接口封装
- [ ] AXI DMA 接口(可先用 Xilinx 官方 AXI DMA IP)
- [ ] Verilator/Vivado behavioral simulation,纯 testbench 验证矩阵乘正确性(脱离 PicoRV32/ARM)
- [ ] 综合 + 资源记录

### 5.4 验收标准
Testbench 给定已知 8x8 矩阵 A、B,单元输出与 numpy golden model 完全一致。

---

## 6. Phase 3:端到端闭环(CSR 路径,手写固件)

### 6.1 任务清单
- [ ] PicoRV32 固件(裸机 C):初始化矩阵单元 CSR、触发计算、轮询完成
- [ ] ARM 端 Python driver(基于 `pynq.MMIO`/`allocate`):分配物理连续内存、写入测试矩阵、配置地址映射、触发 PicoRV32、等待完成、读取结果并与 numpy 对比
- [ ] 明确内存模型文档:PicoRV32 侧地址空间 vs ARM 侧地址空间映射、cache coherency 处理方式(沿用 Phase 0 验证结论,先用最保守的 uncached/手动 flush 方案)

### 6.2 验收标准
Jupyter notebook 一个 cell 跑通:分配矩阵 → 写入 → 触发 PicoRV32 → 等待 → 读取结果 → assert 与 numpy 一致。**这是第一个完整可演示 demo。**

### 6.3 回归测试
- [ ] 随机生成多组 8x8 矩阵,批量跑硬件路径 vs numpy,统计通过率
- [ ] 记录端到端延迟(作为 Phase 4/6 对比的 baseline)

---

## 7. Phase 4:自定义指令(PCPI)设计与实现 ★核心主线

> 这是 v2 相对 v1 最重要的结构调整:自定义指令支持从"可选加分项"提升为项目核心目标,必须在 MLIR dialect 设计之前先把底层指令机制打通,后续 Phase 5 的 lowering pass 才有明确的目标格式。

### 7.1 指令编码方案

使用 RISC-V 规范中专门预留、不会与标准指令冲突的 **custom opcode 空间**:

- `custom-0`:opcode `0001011`(二进制),对应机器码低 7 位 `0x0B`
- 用 `funct3` 区分不同操作语义,例如:
  - `funct3=0`:`mat_trigger`(触发矩阵单元,rs1=参数打包地址,rs2=目标地址)
  - `funct3=1`:`mat_status`(查询状态,rd=返回 status 值)
  - `funct3=2`:`mat_reset`(复位加速单元)
- 暂不使用 `funct7`,预留给未来扩展(如 Phase 7 向量指令)

### 7.2 PCPI 端硬件实现

- [ ] 阅读参考仓库 `gold_ip/pcpi_v1_0` 作为实现范例,理解 PCPI 标准接口信号:`pcpi_valid`、`pcpi_insn`、`pcpi_rs1`、`pcpi_rs2`、`pcpi_wr`、`pcpi_rd`、`pcpi_wait`、`pcpi_ready`
- [ ] 实现 PCPI 译码逻辑:解析 `pcpi_insn` 的 opcode/funct3 字段,分发到对应操作
- [ ] 将 PCPI 端接到 Phase 2 的矩阵单元 CSR 接口(`mat_trigger` 内部等价于原来 CSR 的 start 触发序列,但通过指令直接传参而非多次访存)
- [ ] **明确执行语义**:选择"指令触发后立即返回(fire-and-forget),完成状态另行 poll/中断"还是"指令语义即阻塞直到完成(通过 `pcpi_wait` 让核心硬等)"。前者更贴近真实 NPU 的异步 kernel launch 语义,建议采用;后者实现更简单但会让 CPU 空转浪费,仅作为该 Phase 早期的验证手段。

### 7.3 软件侧:`.insn` 伪指令 + 包装 intrinsic

**关键结论:不需要实现真正的 LLVM/Clang target intrinsic(即改 LLVM 后端源码注册 `__builtin_xxx`),用 `.insn` 伪指令 + 内联汇编包装即可,工作量小一个数量级,且完全满足"编译链路自动生成自定义指令"这一目标。**

原因:
- 真 target intrinsic 需要在 `llvm/lib/Target/RISCV/RISCVInstrInfo*.td` 定义指令 pattern,并在 Clang `CGBuiltin.cpp` 注册 builtin,这是改 LLVM/Clang 上游代码库的工作量,和项目主线(MLIR dialect + lowering)关系不大,产出主要是"易用性",不是"链路打通"。
- 本项目的自动化链路本来就是 MLIR pass 直接在 LLVM IR 层生成 `InlineAsm` 节点,效果上与调用 target intrinsic 完全等价,但不需要碰 LLVM/Clang 源码。

汇编层面使用 `.insn` 直接编码:
```asm
# .insn r opcode, funct3, funct7, rd, rs1, rs2
.insn r 0x0B, 0, 0, x0, a0, a1   # mat_trigger: rs1=参数地址, rs2=目标地址
```

C 层面(供 Phase 3 手写测试/调试固件使用,便于快速验证 PCPI 端逻辑是否正确,不进入自动化 pipeline):
```c
static inline void mat_trigger(void *param_addr, void *dst_addr) {
    register uint32_t a0 asm("a0") = (uint32_t)param_addr;
    register uint32_t a1 asm("a1") = (uint32_t)dst_addr;
    asm volatile (".insn r 0x0B, 0, 0, x0, %0, %1" :: "r"(a0), "r"(a1));
}

static inline uint32_t mat_status(void) {
    uint32_t rd;
    asm volatile (".insn r 0x0B, 1, 0, %0, x0, x0" : "=r"(rd));
    return rd;
}
```

### 7.4 任务清单
- [ ] 完成 7.1 指令编码文档(`docs/custom_isa_encoding.md`)
- [ ] 完成 7.2 PCPI 硬件实现与仿真验证
- [ ] 编写 `sysarray_intrinsics.h`(C 包装函数,供调试用)
- [ ] 手写 C 测试固件,用自定义指令触发矩阵单元,验证与 Phase 3 CSR 路径结果一致
- [ ] 记录该路径的端到端延迟,与 Phase 3 CSR 路径对比(为 Phase 6 做数据准备)

### 7.5 验收标准
手写 C 固件通过 `mat_trigger()`(底层为 `.insn` 自定义指令)触发矩阵计算,结果与 numpy golden model 一致,且不再依赖 Phase 2 的 CSR 地址读写方式访问控制寄存器。

---

## 8. Phase 5:MLIR Dialect 设计与指令级 Lowering

### 8.1 Dialect 设计(与 v1 基本一致,lowering 目标改变)

```tablegen
def SysArray_MatmulTileOp : SysArray_Op<"matmul_tile"> {
  let summary = "Execute a fixed-size tile matmul on the systolic array";
  let arguments = (ins
    I32:$param_addr,
    I32:$dst_addr
  );
}
```

（`csr_write`/`poll_status` 两个 op 保留用于 Phase 6 对比实验的 CSR 路径,主线路径不再需要。）

### 8.2 Pipeline 设计(v2:指令级 lowering)

```
torch-mlir 输出 (linalg.matmul, 固定 8x8x8 shape)
    │
    ▼  Pass 1: MatchAndRewrite
sysarray.matmul_tile(param_addr, dst_addr)
    │
    ▼  Pass 2: Lower to LLVM dialect,生成内联汇编节点
llvm.inline_asm(".insn r 0x0B, 0, 0, x0, $0, $1", param_addr, dst_addr)
    │
    ▼  mlir-translate --mlir-to-llvmir
LLVM IR(含 InlineAsm 节点)
    │
    ▼  llc -march=riscv32
RISC-V 汇编(自定义指令原样透传)/ 目标文件
    │
    ▼  riscv32-unknown-elf-gcc (link)
可执行固件
```

相较 v1 的"CSR 写入序列"方案,这一版直接体现"自定义指令"这个目标本质——CSR 方案生成的仍是标准 `llvm.store` 访存指令(只是地址映射到外设),而这里生成的是真正意义上新增的 RISC-V 指令。

### 8.3 任务清单
- [ ] 环境搭建:llvm-project + torch-mlir,`LLVM_TARGETS_TO_BUILD` 含 RISCV
- [ ] 编写 `sysarray` dialect TableGen 定义 + C++ Op 实现
- [ ] Pass 1:pattern matching,从 `linalg.matmul` 识别可卸载子图(先严格匹配固定 shape)
- [ ] Pass 2:lowering 到 LLVM `InlineAsm`,复用 7.3 中确定的 `.insn` 编码模板
- [ ] 打通命令行全流程:PyTorch 脚本(单次 8x8 matmul)→ torch-mlir → 自定义 pass → LLVM IR → RISC-V 固件 → 烧录验证

### 8.4 验收标准
不再手写固件,从 PyTorch 代码自动生成含自定义指令的 RISC-V 固件,跑在板子上得到正确结果。**这是整个项目故事线的高潮部分**,建议录屏或详细记录过程。

---

## 9. Phase 6:松耦合(CSR) vs 紧耦合(PCPI 指令)对比实验

### 9.1 目标
量化两种 kernel ABI 设计的实际差异,这份对比本身是很好的面试素材。

### 9.2 任务清单
- [ ] 用 Phase 3(CSR)和 Phase 4(PCPI 指令)两条路径分别跑相同的矩阵乘任务
- [ ] 记录并对比:端到端延迟、指令数、访存次数、代码复杂度(行数/圈复杂度)
- [ ] 分析差异来源:CSR 方案需要多次 AXI-Lite 访存(写 4-5 个寄存器 + 轮询),指令方案一条指令传完两个地址,理论上访存次数和延迟应显著降低

### 9.3 产出物
`docs/tight_vs_loose_coupling.md`

---

## 10. Phase 7:Vector 单元(资源允许时进行,可选)

前置条件:完成 Phase 5 后重新跑资源综合,确认 LUT/BRAM 余量(建议 ≥ 30%)。

- 4-8 lane,int16 element-wise 运算(add/mul/mac)
- 复用 Phase 4 的自定义指令编码模式(新增 funct3 值,或使用 `custom-1` opcode 空间区分)
- 扩展 `sysarray` dialect 新增 `sysarray.vector_op`,扩展 lowering pass

---

## 11. Phase 8:自动 Tiling + 性能优化(可选)

- [ ] MLIR 层 tiling pass:任意 shape `linalg.matmul` 自动切分为多个 8x8 tile 的指令序列
- [ ] 边界情况处理(shape 非 8 整数倍的 padding)
- [ ] 硬件层双缓冲(ping-pong buffer),DMA 搬运与计算重叠
- [ ] 单缓冲 vs 双缓冲吞吐对比

---

## 12. Phase 9:文档与素材整理

- [ ] GitHub 仓库整理:README(架构图、复现步骤)、清晰的 commit 历史
- [ ] 技术博客,按"问题—设计取舍—实现—验证结果"结构,重点突出:
  - Phase 4 的自定义指令编码与 PCPI 实现(核心亮点)
  - Phase 5 的 MLIR 指令级 lowering(区别于常规 CSR lowering 的技术深度)
  - Phase 6 的松耦合 vs 紧耦合量化对比
- [ ] 面试素材:完整"排查故事"(从 PyTorch 调用到硬件执行各层可能的问题与排查方法)
- [ ] 架构图、时序图、指令编码格式图

---

## 13. 风险与应对预案

| 风险 | 应对 |
|---|---|
| PL 资源不够塞下三个模块 | 优先保 PicoRV32 + Matrix,Vector 单元降级为可选(Phase 7),或缩小 Matrix 阵列规模(4x4) |
| 时序收敛失败 | 降频运行是可接受方案,项目重点是架构与编译链路而非峰值性能 |
| MLIR/torch-mlir 环境搭建耗时过长 | Phase 0 就并行验证 torch-mlir 编译是否顺利 |
| RISC-V 与 ARM 内存一致性踩坑 | Phase 0 提前用最简单 testbench 验证 cache coherency 行为,不要等到 Phase 3 才发现 |
| Vivado 版本与参考仓库(2021.1)不匹配 | Phase 0 优先确认;若不匹配,仅借鉴 IP 设计思路,自行在对应版本重新搭建关键 IP |
| PCPI 多周期等待策略设计不当,导致 CPU 长期阻塞 | Phase 4 提前明确 fire-and-forget vs 阻塞语义,推荐前者,更贴近真实 NPU 异步 launch 模式 |
| 真 intrinsic 需求被误判为必需 | 明确本项目不需要改 LLVM/Clang 后端注册真 target intrinsic,`.insn`+InlineAsm 完全满足目标;真 intrinsic 列为可选的"未来工作"(见第 14 节) |

---

## 14. 未来可选扩展(不在主线交付范围内)

以下内容明确**不是**当前目标的必要组成部分,仅在主线跑通、且有余力/明确需求时考虑:

1. **真正的 LLVM/Clang target intrinsic**:在 LLVM 后端 TableGen 中注册指令 pattern,Clang `CGBuiltin.cpp` 中注册 builtin,使得普通用户可以在 C/C++ 代码中直接写 `__builtin_sysarray_matmul(a, b)`。价值在于工具链易用性/产品化,与"打通端到端链路"这一核心目标关系不大,工作量与整个主线相当。
2. **VexRiscv plugin 流水线级扩展**:作为与 Phase 4(PCPI)的对比分支,如果希望在面试中额外展示"深入 CPU 微架构流水线"的能力,可单独用 VexRiscv 重做一遍 Phase 4,对比 PCPI 协处理器 vs 流水线插件两种自定义指令集成方式的实现难度与性能差异。
3. **RV32C 压缩指令支持**、**标准 RVV 向量扩展兼容**等 ISA 层面的进一步完整度工作。

---

## 15. 与 v1 版本的核心变更说明

| 变更点 | v1 | v2 | 变更原因 |
|---|---|---|---|
| RISC-V 核心 | VexRiscv(SpinalHDL 生成) | PicoRV32(纯 Verilog) | 更契合"打通端到端自定义指令链路"目标,降低 Phase 0-1 风险 |
| 自定义指令定位 | Phase 6,可选加分项,可裁剪 | **Phase 4,主线核心,不可裁剪** | 项目主目标明确为"支持自定义指令" |
| 自定义指令实现机制 | VexRiscv plugin(流水线嵌入) | PCPI 协处理器接口(固定握手协议) | PicoRV32 原生机制,复杂度更低,已有现成参考实现 |
| 工具链前端扩展 | 曾设想扩展 GCC/LLVM 汇编器识别新指令 | 明确采用 `.insn` 伪指令,不改工具链前端 | `.insn` 已能满足自动化生成需求,真 intrinsic 列为可选未来工作 |
| MLIR lowering 目标 | CSR 写入序列(`llvm.store`) | 内联汇编 `.insn`(`llvm.inline_asm`) | 更直接体现"自定义指令"这一核心目标,而非仍停留在访存语义 |
| CSR/AXI 方案角色 | 独立最终方案 | 降级为 Phase 2-3 的过渡验证手段,兼作 Phase 6 对比基线 | 用最简单方式先验证加速单元逻辑正确,再叠加指令层复杂度 |
| 参考实现 | 无现成开源参考 | `JacoboJin/RISCV-on-PYNQ-Z2`(fork 自 `drichmond/RISC-V-On-PYNQ`) | 提供 PS-PL 桥接 IP、Vivado 集成脚本、PCPI 范例,可直接复用/参考 |

---

## 16. 里程碑检查点

- [ ] M1:PicoRV32 在 PYNQ-Z1 上跑通 hello world(Phase 1)
- [ ] M2:矩阵单元仿真验证通过,资源综合成功(Phase 2)
- [ ] M3:CSR 路径手写固件端到端闭环,结果与 numpy 一致(Phase 3)——**第一个可演示的最小项目**
- [ ] M4:**自定义指令(PCPI)路径跑通,手写固件通过 `.insn` 指令触发矩阵单元并得到正确结果(Phase 4)——核心目标达成的第一个验证点**
- [ ] M5:PyTorch 代码自动生成含自定义指令的 RISC-V 固件并跑通(Phase 5)——**核心故事线完整**
- [ ] M6:松耦合 vs 紧耦合量化对比报告完成(Phase 6)
- [ ] M7(可选):向量单元集成(Phase 7)
- [ ] M8(可选):自动 tiling + 双缓冲优化(Phase 8)
- [ ] M9:文档与面试素材整理完毕(Phase 9)

**最低目标:完成到 M5。** 此时已完整覆盖"端到端 APU+NPU+accelerator 编译过程,支持自定义指令"这一核心目标,后续阶段视时间精力弹性推进。

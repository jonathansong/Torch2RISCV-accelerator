// sa_unit: double-buffered matrix accelerator, top level
// (docs/double_buffer_design.md). M1 configuration: D = 8, one DMA port.
//
//   s_axi       legacy CSRs (Phase 2 map + CAPS / EXT_STATUS)
//   m0..2_axi   DMA masters (NPORTS used, the rest tied off) -> S_AXI_HP1..3
//   pcpi        custom-0 instructions: funct7 = 0 legacy, funct7 = 1 new ISA
//
// Engines: LD (DDR -> SPAD/ACC), ST (SPAD/ACC -> DDR), EX (array).
// Memories: SPAD_A / SPAD_B (D-byte words) and ACC (D x int32 words), each
// double-banked; sa_sched orders commands with a bank scoreboard.
`timescale 1ns / 1ps
`include "sa_macros.vh"

module sa_unit #(
    parameter integer D          = 8,
    parameter integer DSP_COLS   = 8,
    parameter integer NPORTS     = 1,
    parameter integer SPAD_WORDS = 131072 / D,        // 128 KB per SPAD
    parameter integer ACC_WORDS  = 262144 / (4 * D)   // 256 KB
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi:m0_axi:m1_axi:m2_axi, ASSOCIATED_RESET aresetn" *)
    input  wire        aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        aresetn,

    // AXI4-Lite CSR slave (legacy map, docs/double_buffer_design.md §8.5)
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWADDR", X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 8" *)
    input  wire [7:0]   s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWVALID" *)
    input  wire         s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWREADY" *)
    output wire         s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WDATA" *)
    input  wire [31:0]  s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WSTRB" *)
    input  wire [3:0]   s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WVALID" *)
    input  wire         s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WREADY" *)
    output wire         s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BRESP" *)
    output wire [1:0]   s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BVALID" *)
    output wire         s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BREADY" *)
    input  wire         s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARADDR" *)
    input  wire [7:0]   s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARVALID" *)
    input  wire         s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARREADY" *)
    output wire         s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RDATA" *)
    output wire [31:0]  s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RRESP" *)
    output wire [1:0]   s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RVALID" *)
    output wire         s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RREADY" *)
    input  wire         s_axi_rready,

    // AXI4 master 0 (64-bit) -> S_AXI_HP1 via protocol converter; unused when NPORTS <= 0
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi ARADDR", X_INTERFACE_PARAMETER = "PROTOCOL AXI4, DATA_WIDTH 64, ADDR_WIDTH 32, ID_WIDTH 0, MAX_BURST_LENGTH 16, NUM_READ_OUTSTANDING 4, NUM_WRITE_OUTSTANDING 4" *)
    output wire [31:0]  m0_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi ARLEN" *)
    output wire [7:0]   m0_axi_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi ARSIZE" *)
    output wire [2:0]   m0_axi_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi ARBURST" *)
    output wire [1:0]   m0_axi_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi ARCACHE" *)
    output wire [3:0]   m0_axi_arcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi ARPROT" *)
    output wire [2:0]   m0_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi ARVALID" *)
    output wire         m0_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi ARREADY" *)
    input  wire         m0_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi RDATA" *)
    input  wire [63:0]  m0_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi RRESP" *)
    input  wire [1:0]   m0_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi RLAST" *)
    input  wire         m0_axi_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi RVALID" *)
    input  wire         m0_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi RREADY" *)
    output wire         m0_axi_rready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi AWADDR" *)
    output wire [31:0]  m0_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi AWLEN" *)
    output wire [7:0]   m0_axi_awlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi AWSIZE" *)
    output wire [2:0]   m0_axi_awsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi AWBURST" *)
    output wire [1:0]   m0_axi_awburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi AWCACHE" *)
    output wire [3:0]   m0_axi_awcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi AWPROT" *)
    output wire [2:0]   m0_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi AWVALID" *)
    output wire         m0_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi AWREADY" *)
    input  wire         m0_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi WDATA" *)
    output wire [63:0]  m0_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi WSTRB" *)
    output wire [7:0]   m0_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi WLAST" *)
    output wire         m0_axi_wlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi WVALID" *)
    output wire         m0_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi WREADY" *)
    input  wire         m0_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi BRESP" *)
    input  wire [1:0]   m0_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi BVALID" *)
    input  wire         m0_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m0_axi BREADY" *)
    output wire         m0_axi_bready,

    // AXI4 master 1 (64-bit) -> S_AXI_HP2 via protocol converter; unused when NPORTS <= 1
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi ARADDR", X_INTERFACE_PARAMETER = "PROTOCOL AXI4, DATA_WIDTH 64, ADDR_WIDTH 32, ID_WIDTH 0, MAX_BURST_LENGTH 16, NUM_READ_OUTSTANDING 4, NUM_WRITE_OUTSTANDING 4" *)
    output wire [31:0]  m1_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi ARLEN" *)
    output wire [7:0]   m1_axi_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi ARSIZE" *)
    output wire [2:0]   m1_axi_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi ARBURST" *)
    output wire [1:0]   m1_axi_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi ARCACHE" *)
    output wire [3:0]   m1_axi_arcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi ARPROT" *)
    output wire [2:0]   m1_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi ARVALID" *)
    output wire         m1_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi ARREADY" *)
    input  wire         m1_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi RDATA" *)
    input  wire [63:0]  m1_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi RRESP" *)
    input  wire [1:0]   m1_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi RLAST" *)
    input  wire         m1_axi_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi RVALID" *)
    input  wire         m1_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi RREADY" *)
    output wire         m1_axi_rready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi AWADDR" *)
    output wire [31:0]  m1_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi AWLEN" *)
    output wire [7:0]   m1_axi_awlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi AWSIZE" *)
    output wire [2:0]   m1_axi_awsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi AWBURST" *)
    output wire [1:0]   m1_axi_awburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi AWCACHE" *)
    output wire [3:0]   m1_axi_awcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi AWPROT" *)
    output wire [2:0]   m1_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi AWVALID" *)
    output wire         m1_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi AWREADY" *)
    input  wire         m1_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi WDATA" *)
    output wire [63:0]  m1_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi WSTRB" *)
    output wire [7:0]   m1_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi WLAST" *)
    output wire         m1_axi_wlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi WVALID" *)
    output wire         m1_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi WREADY" *)
    input  wire         m1_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi BRESP" *)
    input  wire [1:0]   m1_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi BVALID" *)
    input  wire         m1_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m1_axi BREADY" *)
    output wire         m1_axi_bready,

    // AXI4 master 2 (64-bit) -> S_AXI_HP3 via protocol converter; unused when NPORTS <= 2
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi ARADDR", X_INTERFACE_PARAMETER = "PROTOCOL AXI4, DATA_WIDTH 64, ADDR_WIDTH 32, ID_WIDTH 0, MAX_BURST_LENGTH 16, NUM_READ_OUTSTANDING 4, NUM_WRITE_OUTSTANDING 4" *)
    output wire [31:0]  m2_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi ARLEN" *)
    output wire [7:0]   m2_axi_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi ARSIZE" *)
    output wire [2:0]   m2_axi_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi ARBURST" *)
    output wire [1:0]   m2_axi_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi ARCACHE" *)
    output wire [3:0]   m2_axi_arcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi ARPROT" *)
    output wire [2:0]   m2_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi ARVALID" *)
    output wire         m2_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi ARREADY" *)
    input  wire         m2_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi RDATA" *)
    input  wire [63:0]  m2_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi RRESP" *)
    input  wire [1:0]   m2_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi RLAST" *)
    input  wire         m2_axi_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi RVALID" *)
    input  wire         m2_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi RREADY" *)
    output wire         m2_axi_rready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi AWADDR" *)
    output wire [31:0]  m2_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi AWLEN" *)
    output wire [7:0]   m2_axi_awlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi AWSIZE" *)
    output wire [2:0]   m2_axi_awsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi AWBURST" *)
    output wire [1:0]   m2_axi_awburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi AWCACHE" *)
    output wire [3:0]   m2_axi_awcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi AWPROT" *)
    output wire [2:0]   m2_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi AWVALID" *)
    output wire         m2_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi AWREADY" *)
    input  wire         m2_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi WDATA" *)
    output wire [63:0]  m2_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi WSTRB" *)
    output wire [7:0]   m2_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi WLAST" *)
    output wire         m2_axi_wlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi WVALID" *)
    output wire         m2_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi WREADY" *)
    input  wire         m2_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi BRESP" *)
    input  wire [1:0]   m2_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi BVALID" *)
    input  wire         m2_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m2_axi BREADY" *)
    output wire         m2_axi_bready,

    // PicoRV32 co-processor interface (bus definition in RISCV-on-PYNQ-Z1/ip/pcpi_v1_0)
    (* X_INTERFACE_INFO = "cliffordwolf:ip:pcpi:1.0 pcpi pcpi_valid", X_INTERFACE_MODE = "slave" *)
    input  wire         pcpi_valid,
    (* X_INTERFACE_INFO = "cliffordwolf:ip:pcpi:1.0 pcpi pcpi_insn" *)
    input  wire [31:0]  pcpi_insn,
    (* X_INTERFACE_INFO = "cliffordwolf:ip:pcpi:1.0 pcpi pcpi_rs1" *)
    input  wire [31:0]  pcpi_rs1,
    (* X_INTERFACE_INFO = "cliffordwolf:ip:pcpi:1.0 pcpi pcpi_rs2" *)
    input  wire [31:0]  pcpi_rs2,
    (* X_INTERFACE_INFO = "cliffordwolf:ip:pcpi:1.0 pcpi pcpi_wr" *)
    output wire         pcpi_wr,
    (* X_INTERFACE_INFO = "cliffordwolf:ip:pcpi:1.0 pcpi pcpi_rd" *)
    output wire [31:0]  pcpi_rd,
    (* X_INTERFACE_INFO = "cliffordwolf:ip:pcpi:1.0 pcpi pcpi_wait" *)
    output wire         pcpi_wait,
    (* X_INTERFACE_INFO = "cliffordwolf:ip:pcpi:1.0 pcpi pcpi_ready" *)
    output wire         pcpi_ready,

    (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 irq INTERRUPT", X_INTERFACE_PARAMETER = "SENSITIVITY LEVEL_HIGH" *)
    output wire         irq
);
    `include "sa_defs.vh"
    localparam integer SAW = $clog2(SPAD_WORDS);
    localparam integer CAW = $clog2(ACC_WORDS);
    wire resetn = aresetn;

    // --------------------------------------------------- command sources
    wire                 p_valid, p_ready, l_valid;
    wire [PKT_W-1:0]     p_pkt, l_pkt;
    wire                 in_valid, in_ready;
    wire [PKT_W-1:0]     in_pkt;
    // legacy sequencer has priority
    assign in_valid = l_valid || p_valid;
    assign in_pkt   = l_valid ? l_pkt : p_pkt;
    assign p_ready  = in_ready && !l_valid;

    wire        sched_idle, fence_ok, clear_error;
    wire [3:0]  fence_mask;
    wire [31:0] ext_status;

    wire        ld_v, ld_r, st_v, st_r, ex_v, ex_r;
    wire [PKT_W-1:0] ld_pkt, st_pkt, ex_pkt;
    wire        ld_done, ld_err, st_done, st_err, ex_done;

    sa_sched #(.D(D), .SPAD_WORDS(SPAD_WORDS), .ACC_WORDS(ACC_WORDS)) sched (
        .clk(aclk), .resetn(resetn),
        .in_valid(in_valid), .in_ready(in_ready), .in_pkt(in_pkt),
        .ld_valid(ld_v), .ld_ready(ld_r), .ld_pkt(ld_pkt),
        .st_valid(st_v), .st_ready(st_r), .st_pkt(st_pkt),
        .ex_valid(ex_v), .ex_ready(ex_r), .ex_pkt(ex_pkt),
        .ld_done(ld_done), .ld_err(ld_err), .st_done(st_done), .st_err(st_err), .ex_done(ex_done),
        .clear_error(clear_error), .fence_mask(fence_mask), .fence_ok(fence_ok),
        .idle(sched_idle), .ext_status(ext_status));

    // ------------------------------------------------------------ PCPI
    wire        leg_trigger, leg_accept, leg_busy, leg_reset, leg_reset_done;
    wire [31:0] leg_desc, leg_dst, leg_status, leg_cycles;

    sa_pcpi pcpi (
        .clk(aclk), .resetn(resetn),
        .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn), .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2),
        .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd), .pcpi_wait(pcpi_wait), .pcpi_ready(pcpi_ready),
        .q_valid(p_valid), .q_ready(p_ready), .q_pkt(p_pkt),
        .sched_err(ext_status[1]), .fence_mask(fence_mask), .fence_ok(fence_ok), .ext_status(ext_status),
        .leg_trigger(leg_trigger), .leg_desc(leg_desc), .leg_dst(leg_dst), .leg_accept(leg_accept),
        .leg_busy(leg_busy), .leg_status(leg_status), .leg_cycles(leg_cycles),
        .leg_reset(leg_reset), .leg_reset_done(leg_reset_done));

    // ---------------------------------------------------------- legacy
    wire        lw_en;
    wire [3:0]  lw_mem;
    wire [15:0] lw_word;
    wire [7:0]  lw_lane;
    wire [63:0] lw_data;

    sa_legacy #(.D(D), .NPORTS(NPORTS), .VL(0), .SPAD_WORDS(SPAD_WORDS), .ACC_WORDS(ACC_WORDS)) legacy (
        .clk(aclk), .resetn(resetn),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr), .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp), .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        .q_valid(l_valid), .q_ready(in_ready), .q_pkt(l_pkt),
        .sched_idle(sched_idle), .ext_status(ext_status), .clear_error(clear_error),
        .desc_we(lw_en && lw_mem == MEM_DESC), .desc_word(lw_word), .desc_data(lw_data),
        .leg_trigger(leg_trigger), .leg_desc(leg_desc), .leg_dst(leg_dst), .leg_accept(leg_accept),
        .leg_busy(leg_busy), .leg_status(leg_status), .leg_cycles(leg_cycles),
        .leg_reset(leg_reset), .leg_reset_done(leg_reset_done), .irq(irq));

    // ---------------------------------------------------------- engines
    wire [NPORTS*32-1:0] araddr, awaddr;
    wire [NPORTS*8-1:0]  arlen, awlen;
    wire [NPORTS-1:0]    arvalid, arready, rlast, rvalid, rready;
    wire [NPORTS-1:0]    awvalid, awready, wlast, wvalid, wready, bvalid, bready;
    wire [NPORTS*64-1:0] rdata, wdata;
    wire [NPORTS*2-1:0]  rresp, bresp;

    sa_ld #(.D(D), .NPORTS(NPORTS)) ld (
        .clk(aclk), .resetn(resetn), .cmd_valid(ld_v), .cmd_ready(ld_r),
        .cmd_ddr(ld_pkt[33:2]), .cmd_mem(ld_pkt[65:62]), .cmd_word(ld_pkt[49:34]),
        .cmd_rows(ld_pkt[81:66]), .cmd_row_bytes(ld_pkt[97:82]), .cmd_pitch(ld_pkt[129:98]),
        .cmd_mode(ld_pkt[131:130]), .done(ld_done), .err(ld_err), .busy(),
        .lw_en(lw_en), .lw_mem(lw_mem), .lw_word(lw_word), .lw_lane(lw_lane), .lw_data(lw_data),
        .m_araddr(araddr), .m_arlen(arlen), .m_arvalid(arvalid), .m_arready(arready),
        .m_rdata(rdata), .m_rresp(rresp), .m_rlast(rlast), .m_rvalid(rvalid), .m_rready(rready));

    wire        lr_en;
    wire [3:0]  lr_mem;
    wire [15:0] lr_word;
    wire [8*D-1:0]  lr_a, lr_b;
    wire [32*D-1:0] lr_c;

    sa_st #(.D(D), .NPORTS(NPORTS)) st (
        .clk(aclk), .resetn(resetn), .cmd_valid(st_v), .cmd_ready(st_r),
        .cmd_ddr(st_pkt[33:2]), .cmd_mem(st_pkt[65:62]), .cmd_word(st_pkt[49:34]),
        .cmd_rows(st_pkt[81:66]), .cmd_row_bytes(st_pkt[97:82]), .cmd_pitch(st_pkt[129:98]),
        .done(st_done), .err(st_err), .busy(),
        .lr_en(lr_en), .lr_mem(lr_mem), .lr_word(lr_word),
        .lr_spad_a(lr_a), .lr_spad_b(lr_b), .lr_acc(lr_c),
        .m_awaddr(awaddr), .m_awlen(awlen), .m_awvalid(awvalid), .m_awready(awready),
        .m_wdata(wdata), .m_wlast(wlast), .m_wvalid(wvalid), .m_wready(wready),
        .m_bresp(bresp), .m_bvalid(bvalid), .m_bready(bready));

    wire               sa_en, sb_en, acc_en;
    wire [SAW-1:0]     sa_addr, sb_addr;
    wire [CAW-1:0]     acc_addr;
    wire [8*D-1:0]     sa_dout, sb_dout;
    wire [4*D-1:0]     acc_we;
    wire [32*D-1:0]    acc_din, acc_dout;

    sa_ex #(.D(D), .DSP_COLS(DSP_COLS), .SPAD_AW(SAW), .ACC_AW(CAW)) ex (
        .clk(aclk), .resetn(resetn),
        .cmd_valid(ex_v), .cmd_ready(ex_r), .cmd_a(ex_pkt[17:2]), .cmd_b(ex_pkt[33:18]),
        .cmd_c(ex_pkt[49:34]), .cmd_kt(ex_pkt[61:50]), .cmd_acc(ex_pkt[62]),
        .cmd_rep(ex_pkt[74:63]), .cmd_bstep(ex_pkt[90:75]), .cmd_cstep(ex_pkt[106:91]),
        .cmd_crow(ex_pkt[122:107]),
        .done(ex_done), .busy(),
        .sa_en(sa_en), .sa_addr(sa_addr), .sa_dout(sa_dout),
        .sb_en(sb_en), .sb_addr(sb_addr), .sb_dout(sb_dout),
        .acc_en(acc_en), .acc_we(acc_we), .acc_addr(acc_addr), .acc_din(acc_din), .acc_dout(acc_dout));

    // ---------------------------------------------------------- memories
    // SPAD side A: [0] LD write, [1] ST read; side B: EX read.
    // ACC  side A: EX drain / accumulate;  side B: [0] LD write, [1] ST read.
    wire [D-1:0]   spad_we = {D{1'b0}} | ({8'hFF} << (8 * lw_lane));
    wire [4*D-1:0] acc_lwe = {4*D{1'b0}} | ({8'hFF} << (8 * lw_lane));
    wire ld_a = lw_en && lw_mem == MEM_SPAD_A, ld_b = lw_en && lw_mem == MEM_SPAD_B, ld_c = lw_en && lw_mem == MEM_ACC;
    wire st_a = lr_en && lr_mem == MEM_SPAD_A, st_b = lr_en && lr_mem == MEM_SPAD_B, st_c = lr_en && lr_mem == MEM_ACC;
    wire [8*D-1:0]  unused_a, unused_b;
    wire [32*D-1:0] unused_c;

    sa_bankmem #(.W(8*D), .DEPTH(SPAD_WORDS), .NA(2), .NB(1)) spad_a (
        .clk(aclk),
        .a_en({st_a, ld_a}), .a_we({{D{1'b0}}, spad_we}),
        .a_addr({lr_word[SAW-1:0], lw_word[SAW-1:0]}), .a_din({{8*D{1'b0}}, {D/8{lw_data}}}),
        .a_dout({lr_a, unused_a}),
        .b_en(sa_en), .b_we({D{1'b0}}), .b_addr(sa_addr), .b_din({8*D{1'b0}}), .b_dout(sa_dout));
    sa_bankmem #(.W(8*D), .DEPTH(SPAD_WORDS), .NA(2), .NB(1)) spad_b (
        .clk(aclk),
        .a_en({st_b, ld_b}), .a_we({{D{1'b0}}, spad_we}),
        .a_addr({lr_word[SAW-1:0], lw_word[SAW-1:0]}), .a_din({{8*D{1'b0}}, {D/8{lw_data}}}),
        .a_dout({lr_b, unused_b}),
        .b_en(sb_en), .b_we({D{1'b0}}), .b_addr(sb_addr), .b_din({8*D{1'b0}}), .b_dout(sb_dout));
    sa_bankmem #(.W(32*D), .DEPTH(ACC_WORDS), .NA(1), .NB(2)) accm (
        .clk(aclk),
        .a_en(acc_en), .a_we(acc_we), .a_addr(acc_addr), .a_din(acc_din), .a_dout(acc_dout),
        .b_en({st_c, ld_c}), .b_we({{4*D{1'b0}}, acc_lwe}),
        .b_addr({lr_word[CAW-1:0], lw_word[CAW-1:0]}), .b_din({{32*D{1'b0}}, {D/2{lw_data}}}),
        .b_dout({lr_c, unused_c}));

    // ------------------------------------------------------ AXI masters
    // Port p of the engines -> m<p>_axi; ports >= NPORTS are tied off.
    generate if (NPORTS > 0) begin : port0
        assign m0_axi_araddr  = araddr[32*0 +: 32];
        assign m0_axi_arlen   = arlen[8*0 +: 8];
        assign m0_axi_arvalid = arvalid[0];
        assign arready[0]     = m0_axi_arready;
        assign rdata[64*0 +: 64] = m0_axi_rdata;
        assign rresp[2*0 +: 2]   = m0_axi_rresp;
        assign rlast[0]       = m0_axi_rlast;
        assign rvalid[0]      = m0_axi_rvalid;
        assign m0_axi_rready  = rready[0];
        assign m0_axi_awaddr  = awaddr[32*0 +: 32];
        assign m0_axi_awlen   = awlen[8*0 +: 8];
        assign m0_axi_awvalid = awvalid[0];
        assign awready[0]     = m0_axi_awready;
        assign m0_axi_wdata   = wdata[64*0 +: 64];
        assign m0_axi_wlast   = wlast[0];
        assign m0_axi_wvalid  = wvalid[0];
        assign wready[0]      = m0_axi_wready;
        assign bresp[2*0 +: 2]   = m0_axi_bresp;
        assign bvalid[0]      = m0_axi_bvalid;
        assign m0_axi_bready  = bready[0];
    end else begin : tie0
        assign m0_axi_araddr = 0; assign m0_axi_arlen = 0; assign m0_axi_arvalid = 0;
        assign m0_axi_rready = 0; assign m0_axi_awaddr = 0; assign m0_axi_awlen = 0;
        assign m0_axi_awvalid = 0; assign m0_axi_wdata = 0; assign m0_axi_wlast = 0;
        assign m0_axi_wvalid = 0; assign m0_axi_bready = 0;
    end endgenerate
    assign m0_axi_arsize  = 3'd3;    assign m0_axi_awsize  = 3'd3;      // 8-byte beats
    assign m0_axi_arburst = 2'b01;   assign m0_axi_awburst = 2'b01;     // INCR
    assign m0_axi_arcache = 4'b0011; assign m0_axi_awcache = 4'b0011;
    assign m0_axi_arprot  = 3'b000;  assign m0_axi_awprot  = 3'b000;
    assign m0_axi_wstrb   = 8'hFF;

    generate if (NPORTS > 1) begin : port1
        assign m1_axi_araddr  = araddr[32*1 +: 32];
        assign m1_axi_arlen   = arlen[8*1 +: 8];
        assign m1_axi_arvalid = arvalid[1];
        assign arready[1]     = m1_axi_arready;
        assign rdata[64*1 +: 64] = m1_axi_rdata;
        assign rresp[2*1 +: 2]   = m1_axi_rresp;
        assign rlast[1]       = m1_axi_rlast;
        assign rvalid[1]      = m1_axi_rvalid;
        assign m1_axi_rready  = rready[1];
        assign m1_axi_awaddr  = awaddr[32*1 +: 32];
        assign m1_axi_awlen   = awlen[8*1 +: 8];
        assign m1_axi_awvalid = awvalid[1];
        assign awready[1]     = m1_axi_awready;
        assign m1_axi_wdata   = wdata[64*1 +: 64];
        assign m1_axi_wlast   = wlast[1];
        assign m1_axi_wvalid  = wvalid[1];
        assign wready[1]      = m1_axi_wready;
        assign bresp[2*1 +: 2]   = m1_axi_bresp;
        assign bvalid[1]      = m1_axi_bvalid;
        assign m1_axi_bready  = bready[1];
    end else begin : tie1
        assign m1_axi_araddr = 0; assign m1_axi_arlen = 0; assign m1_axi_arvalid = 0;
        assign m1_axi_rready = 0; assign m1_axi_awaddr = 0; assign m1_axi_awlen = 0;
        assign m1_axi_awvalid = 0; assign m1_axi_wdata = 0; assign m1_axi_wlast = 0;
        assign m1_axi_wvalid = 0; assign m1_axi_bready = 0;
    end endgenerate
    assign m1_axi_arsize  = 3'd3;    assign m1_axi_awsize  = 3'd3;      // 8-byte beats
    assign m1_axi_arburst = 2'b01;   assign m1_axi_awburst = 2'b01;     // INCR
    assign m1_axi_arcache = 4'b0011; assign m1_axi_awcache = 4'b0011;
    assign m1_axi_arprot  = 3'b000;  assign m1_axi_awprot  = 3'b000;
    assign m1_axi_wstrb   = 8'hFF;

    generate if (NPORTS > 2) begin : port2
        assign m2_axi_araddr  = araddr[32*2 +: 32];
        assign m2_axi_arlen   = arlen[8*2 +: 8];
        assign m2_axi_arvalid = arvalid[2];
        assign arready[2]     = m2_axi_arready;
        assign rdata[64*2 +: 64] = m2_axi_rdata;
        assign rresp[2*2 +: 2]   = m2_axi_rresp;
        assign rlast[2]       = m2_axi_rlast;
        assign rvalid[2]      = m2_axi_rvalid;
        assign m2_axi_rready  = rready[2];
        assign m2_axi_awaddr  = awaddr[32*2 +: 32];
        assign m2_axi_awlen   = awlen[8*2 +: 8];
        assign m2_axi_awvalid = awvalid[2];
        assign awready[2]     = m2_axi_awready;
        assign m2_axi_wdata   = wdata[64*2 +: 64];
        assign m2_axi_wlast   = wlast[2];
        assign m2_axi_wvalid  = wvalid[2];
        assign wready[2]      = m2_axi_wready;
        assign bresp[2*2 +: 2]   = m2_axi_bresp;
        assign bvalid[2]      = m2_axi_bvalid;
        assign m2_axi_bready  = bready[2];
    end else begin : tie2
        assign m2_axi_araddr = 0; assign m2_axi_arlen = 0; assign m2_axi_arvalid = 0;
        assign m2_axi_rready = 0; assign m2_axi_awaddr = 0; assign m2_axi_awlen = 0;
        assign m2_axi_awvalid = 0; assign m2_axi_wdata = 0; assign m2_axi_wlast = 0;
        assign m2_axi_wvalid = 0; assign m2_axi_bready = 0;
    end endgenerate
    assign m2_axi_arsize  = 3'd3;    assign m2_axi_awsize  = 3'd3;      // 8-byte beats
    assign m2_axi_arburst = 2'b01;   assign m2_axi_awburst = 2'b01;     // INCR
    assign m2_axi_arcache = 4'b0011; assign m2_axi_awcache = 4'b0011;
    assign m2_axi_arprot  = 3'b000;  assign m2_axi_awprot  = 3'b000;
    assign m2_axi_wstrb   = 8'hFF;
endmodule

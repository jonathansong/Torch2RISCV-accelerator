// matmul_unit: 8x8x8 int8 matrix multiply accelerator, CSR + DMA version
// (Phase 2, loosely coupled). C = A * B with A, B int8 and C int32.
//
//   s_axi_*   AXI4-Lite CSR slave (register map in README.md)
//   m_axi_*   AXI4 64-bit master: fetches A and B from DDR, writes C back
//   irq       IRQ_STATUS[0] & CTRL.irq_enable
//
// Memory layout (all row-major, little-endian):
//   A: 8 rows x 8 int8   = 64 bytes,  one 64-bit beat per row
//   B: 8 rows x 8 int8   = 64 bytes,  one 64-bit beat per row
//   C: 8 rows x 8 int32  = 256 bytes, two 16-beat bursts
// A/B/C addresses must be 8-byte aligned and must not cross a 4 KB
// boundary; bursts are at most 16 beats, so the master is AXI3-safe
// (PS7 S_AXI_HP*).
`timescale 1ns / 1ps

// Explicit interface tags: with <RISCV-on-PYNQ-Z1>/ip in the IP repo path,
// Vivado's name-based inference picks the custom PicoBram bus for s_axi_*.
module matmul_unit (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi:m_axi, ASSOCIATED_RESET aresetn" *)
    input  wire        aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        aresetn,

    // AXI4-Lite CSR slave
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWADDR", X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 8" *)
    input  wire [7:0]  s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWVALID" *)
    input  wire        s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWREADY" *)
    output reg         s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WDATA" *)
    input  wire [31:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WSTRB" *)
    input  wire [3:0]  s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WVALID" *)
    input  wire        s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WREADY" *)
    output reg         s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BRESP" *)
    output wire [1:0]  s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BVALID" *)
    output reg         s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BREADY" *)
    input  wire        s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARADDR" *)
    input  wire [7:0]  s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARVALID" *)
    input  wire        s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARREADY" *)
    output reg         s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RDATA" *)
    output reg  [31:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RRESP" *)
    output wire [1:0]  s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RVALID" *)
    output reg         s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RREADY" *)
    input  wire        s_axi_rready,

    // AXI4 master to DDR
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARADDR", X_INTERFACE_PARAMETER = "PROTOCOL AXI4, DATA_WIDTH 64, ADDR_WIDTH 32, ID_WIDTH 0, MAX_BURST_LENGTH 16, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 1" *)
    output reg  [31:0] m_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARLEN" *)
    output wire [7:0]  m_axi_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARSIZE" *)
    output wire [2:0]  m_axi_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARBURST" *)
    output wire [1:0]  m_axi_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARCACHE" *)
    output wire [3:0]  m_axi_arcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARPROT" *)
    output wire [2:0]  m_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARVALID" *)
    output reg         m_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARREADY" *)
    input  wire        m_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RDATA" *)
    input  wire [63:0] m_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RRESP" *)
    input  wire [1:0]  m_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RLAST" *)
    input  wire        m_axi_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RVALID" *)
    input  wire        m_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RREADY" *)
    output reg         m_axi_rready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWADDR" *)
    output reg  [31:0] m_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWLEN" *)
    output wire [7:0]  m_axi_awlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWSIZE" *)
    output wire [2:0]  m_axi_awsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWBURST" *)
    output wire [1:0]  m_axi_awburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWCACHE" *)
    output wire [3:0]  m_axi_awcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWPROT" *)
    output wire [2:0]  m_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWVALID" *)
    output reg         m_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWREADY" *)
    input  wire        m_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WDATA" *)
    output wire [63:0] m_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WSTRB" *)
    output wire [7:0]  m_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WLAST" *)
    output wire        m_axi_wlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WVALID" *)
    output reg         m_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WREADY" *)
    input  wire        m_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi BRESP" *)
    input  wire [1:0]  m_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi BVALID" *)
    input  wire        m_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi BREADY" *)
    output reg         m_axi_bready,

    (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 irq INTERRUPT", X_INTERFACE_PARAMETER = "SENSITIVITY LEVEL_HIGH" *)
    output wire        irq
);
    localparam integer N = 8;

    // ---------------------------------------------------------- CSR map
    localparam [5:0] R_CTRL       = 6'h00 >> 2;
    localparam [5:0] R_STATUS     = 6'h04 >> 2;
    localparam [5:0] R_SRC_A      = 6'h08 >> 2;
    localparam [5:0] R_SRC_B      = 6'h0C >> 2;
    localparam [5:0] R_DST        = 6'h10 >> 2;
    localparam [5:0] R_DIM        = 6'h14 >> 2;
    localparam [5:0] R_IRQ_STATUS = 6'h18 >> 2;
    localparam [5:0] R_CYCLES     = 6'h1C >> 2;
    localparam [5:0] R_ID         = 6'h20 >> 2;

    localparam [31:0] ID_VALUE = 32'h4D4D_3038;                 // "MM08"
    localparam [31:0] DIM_888  = {2'b0, 10'd8, 10'd8, 10'd8};   // K | N | M

    localparam [3:0] ERR_NONE  = 4'd0;
    localparam [3:0] ERR_DIM   = 4'd1;   // DIM_M_N_K is not 8x8x8
    localparam [3:0] ERR_ADDR  = 4'd2;   // misaligned or crosses 4 KB
    localparam [3:0] ERR_RRESP = 4'd3;   // SLVERR/DECERR reading A or B
    localparam [3:0] ERR_BRESP = 4'd4;   // SLVERR/DECERR writing C

    // -------------------------------------------------------------- FSM
    localparam [2:0] S_IDLE  = 3'd0;
    localparam [2:0] S_RD_AR = 3'd1;
    localparam [2:0] S_RD_R  = 3'd2;
    localparam [2:0] S_COMP  = 3'd3;
    localparam [2:0] S_WR_AW = 3'd4;
    localparam [2:0] S_WR_W  = 3'd5;
    localparam [2:0] S_WR_B  = 3'd6;

    // Skewed operands reach PE(N-1,N-1) at t = (N-1)+(N-1)+(K-1).
    localparam [4:0] T_LAST = 2 * (N - 1) + N - 1;

    reg  [2:0]  state;
    reg         irq_en, done, error, irq_status;
    reg  [3:0]  err_code;
    reg  [31:0] src_a, src_b, dst, dim, cycles;

    reg         rd_sel;          // 0: fetching A, 1: fetching B
    reg  [2:0]  rd_beat;
    reg         rd_err;
    reg  [4:0]  t;               // compute step
    reg         wr_burst;        // which half of C
    reg  [3:0]  wr_beat;
    reg         wr_err;
    reg         array_clear;

    reg  [64*N-1:0] abuf;        // abuf[64*i + 8*k +: 8] = A[i][k]
    reg  [64*N-1:0] bbuf;        // bbuf[64*k + 8*j +: 8] = B[k][j]
    wire [32*N*N-1:0] acc;       // acc[32*(i*N+j) +: 32] = C[i][j]

    wire busy = state != S_IDLE;

    // ---------------------------------------------- AXI4-Lite CSR slave
    reg        aw_got, w_got;
    reg [7:0]  wr_addr;
    reg [31:0] wr_data;
    wire       csr_we  = aw_got && w_got && !s_axi_bvalid;
    wire [5:0] csr_reg = wr_addr[7:2];

    assign s_axi_bresp = 2'b00;
    assign s_axi_rresp = 2'b00;

    always @(posedge aclk) begin
        if (!aresetn) begin
            s_axi_awready <= 0;
            s_axi_wready  <= 0;
            s_axi_bvalid  <= 0;
            s_axi_arready <= 0;
            s_axi_rvalid  <= 0;
            aw_got        <= 0;
            w_got         <= 0;
        end else begin
            s_axi_awready <= !aw_got && s_axi_awvalid && !s_axi_awready;
            s_axi_wready  <= !w_got  && s_axi_wvalid  && !s_axi_wready;
            if (s_axi_awvalid && s_axi_awready) begin
                aw_got  <= 1;
                wr_addr <= s_axi_awaddr;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                w_got   <= 1;
                wr_data <= s_axi_wdata;
            end
            if (csr_we) begin
                aw_got       <= 0;
                w_got        <= 0;
                s_axi_bvalid <= 1;
            end else if (s_axi_bready) begin
                s_axi_bvalid <= 0;
            end

            s_axi_arready <= s_axi_arvalid && !s_axi_arready && !s_axi_rvalid;
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1;
                case (s_axi_araddr[7:2])
                    R_CTRL:       s_axi_rdata <= {29'd0, irq_en, 2'b00};
                    R_STATUS:     s_axi_rdata <= {20'd0, err_code, 5'd0, error, busy, done};
                    R_SRC_A:      s_axi_rdata <= src_a;
                    R_SRC_B:      s_axi_rdata <= src_b;
                    R_DST:        s_axi_rdata <= dst;
                    R_DIM:        s_axi_rdata <= dim;
                    R_IRQ_STATUS: s_axi_rdata <= {31'd0, irq_status};
                    R_CYCLES:     s_axi_rdata <= cycles;
                    R_ID:         s_axi_rdata <= ID_VALUE;
                    default:      s_axi_rdata <= 32'd0;
                endcase
            end else if (s_axi_rready) begin
                s_axi_rvalid <= 0;
            end
        end
    end

    // ------------------------------------------------ control + AXI4 master
    assign m_axi_arlen   = N - 1;               // one beat per row
    assign m_axi_arsize  = 3'd3;                // 8 bytes
    assign m_axi_arburst = 2'b01;               // INCR
    assign m_axi_arcache = 4'b0011;
    assign m_axi_arprot  = 3'b000;
    assign m_axi_awlen   = 8'd15;               // C = 2 bursts x 16 beats
    assign m_axi_awsize  = 3'd3;
    assign m_axi_awburst = 2'b01;
    assign m_axi_awcache = 4'b0011;
    assign m_axi_awprot  = 3'b000;
    assign m_axi_wstrb   = 8'hFF;
    assign m_axi_wlast   = wr_beat == 4'd15;
    // beat g of C holds C elements 2g and 2g+1 (row-major)
    assign m_axi_wdata   = acc[64*{wr_burst, wr_beat} +: 64];

    assign irq = irq_status & irq_en;

    // 8-byte aligned and [addr, addr+len) inside one 4 KB page
    function addr_ok(input [31:0] addr, input [12:0] len);
        addr_ok = addr[2:0] == 3'd0 && {1'b0, addr[11:0]} + len <= 13'h1000;
    endfunction

    task finish(input [3:0] code);
        begin
            state      <= S_IDLE;
            done       <= 1;
            error      <= code != ERR_NONE;
            err_code   <= code;
            irq_status <= 1;
        end
    endtask

    always @(posedge aclk) begin
        if (!aresetn) begin
            state         <= S_IDLE;
            irq_en        <= 0;
            done          <= 0;
            error         <= 0;
            err_code      <= ERR_NONE;
            irq_status    <= 0;
            src_a         <= 0;
            src_b         <= 0;
            dst           <= 0;
            dim           <= DIM_888;
            cycles        <= 0;
            array_clear   <= 1;
            m_axi_arvalid <= 0;
            m_axi_rready  <= 0;
            m_axi_awvalid <= 0;
            m_axi_wvalid  <= 0;
            m_axi_bready  <= 0;
        end else begin
            array_clear <= 0;
            if (busy)
                cycles <= cycles + 1;

            // CSR writes. Config registers are ignored while busy.
            if (csr_we) begin
                case (csr_reg)
                    R_CTRL: begin
                        irq_en <= wr_data[2];
                        if (!busy && wr_data[1]) begin          // soft reset
                            done       <= 0;
                            error      <= 0;
                            err_code   <= ERR_NONE;
                            irq_status <= 0;
                        end
                        if (!busy && wr_data[0]) begin          // start
                            done        <= 0;
                            error       <= 0;
                            err_code    <= ERR_NONE;
                            cycles      <= 0;
                            array_clear <= 1;
                            if (dim != DIM_888)
                                finish(ERR_DIM);
                            else if (!addr_ok(src_a, 13'd64) || !addr_ok(src_b, 13'd64) ||
                                     !addr_ok(dst, 13'd256))
                                finish(ERR_ADDR);
                            else begin
                                state         <= S_RD_AR;
                                rd_sel        <= 0;
                                rd_err        <= 0;
                                wr_err        <= 0;
                                m_axi_araddr  <= src_a;
                                m_axi_arvalid <= 1;
                            end
                        end
                    end
                    R_SRC_A:      if (!busy) src_a <= wr_data;
                    R_SRC_B:      if (!busy) src_b <= wr_data;
                    R_DST:        if (!busy) dst   <= wr_data;
                    R_DIM:        if (!busy) dim   <= wr_data;
                    R_IRQ_STATUS: if (wr_data[0]) irq_status <= 0;   // W1C
                    default: ;
                endcase
            end

            case (state)
                S_RD_AR:
                    if (m_axi_arready) begin
                        m_axi_arvalid <= 0;
                        m_axi_rready  <= 1;
                        rd_beat       <= 0;
                        state         <= S_RD_R;
                    end
                S_RD_R:
                    if (m_axi_rvalid) begin
                        if (rd_sel) bbuf[64*rd_beat +: 64] <= m_axi_rdata;
                        else        abuf[64*rd_beat +: 64] <= m_axi_rdata;
                        rd_beat <= rd_beat + 1;
                        if (m_axi_rresp != 2'b00)
                            rd_err <= 1;
                        if (m_axi_rlast) begin
                            m_axi_rready <= 0;
                            if (rd_err || m_axi_rresp != 2'b00)
                                finish(ERR_RRESP);
                            else if (!rd_sel) begin
                                rd_sel        <= 1;
                                m_axi_araddr  <= src_b;
                                m_axi_arvalid <= 1;
                                state         <= S_RD_AR;
                            end else begin
                                t     <= 0;
                                state <= S_COMP;
                            end
                        end
                    end
                S_COMP: begin
                    t <= t + 1;
                    if (t == T_LAST) begin
                        wr_burst      <= 0;
                        m_axi_awaddr  <= dst;
                        m_axi_awvalid <= 1;
                        state         <= S_WR_AW;
                    end
                end
                S_WR_AW:
                    if (m_axi_awready) begin
                        m_axi_awvalid <= 0;
                        m_axi_wvalid  <= 1;
                        wr_beat       <= 0;
                        state         <= S_WR_W;
                    end
                S_WR_W:
                    if (m_axi_wready) begin
                        wr_beat <= wr_beat + 1;
                        if (m_axi_wlast) begin
                            m_axi_wvalid <= 0;
                            m_axi_bready <= 1;
                            state        <= S_WR_B;
                        end
                    end
                S_WR_B:
                    if (m_axi_bvalid) begin
                        m_axi_bready <= 0;
                        if (!wr_burst) begin
                            // an error on the first half is reported at the end
                            if (m_axi_bresp != 2'b00)
                                wr_err <= 1;
                            wr_burst      <= 1;
                            m_axi_awaddr  <= dst + 32'd128;
                            m_axi_awvalid <= 1;
                            state         <= S_WR_AW;
                        end else begin
                            finish(wr_err || m_axi_bresp != 2'b00 ? ERR_BRESP : ERR_NONE);
                        end
                    end
                default: ;
            endcase
        end
    end

    // ---------------------------------------------------- systolic array
    // Row i of A enters i cycles late and column j of B j cycles late, so
    // PE(i,j) multiplies A[i][k] * B[k][j] at step t = i + j + k.
    wire [8*N-1:0] a_feed, b_feed;
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : feed
            wire [4:0] ka = t - g;
            wire       live = state == S_COMP && t >= g && t < g + N;
            assign a_feed[8*g +: 8] = live ? abuf[64*g + 8*ka[2:0] +: 8] : 8'd0;
            assign b_feed[8*g +: 8] = live ? bbuf[64*ka[2:0] + 8*g +: 8] : 8'd0;
        end
    endgenerate

    systolic_array #(.N(N)) array (
        .clk   (aclk),
        .clear (array_clear),
        .en    (state == S_COMP),
        .a_in  (a_feed),
        .b_in  (b_feed),
        .acc   (acc)
    );
endmodule
